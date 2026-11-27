//! Exact GPU-side comparison of card-resident writable pages. The baseline
//! is the *CPU staging image*, not a copy of the last GPU resource contents.
//! This distinction preserves writes that occurred before a range was first
//! declared writable. No probabilistic hashes or page-wide title writes.
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;
use crate::host::{Host, transition};

pub(crate) const PAGE: usize = 65536;
pub(crate) const MAX_FLAGS: usize = 65536;

const SOURCE: &str = r#"
#define RS "SRV(t0), UAV(u0), UAV(u1), UAV(u2), RootConstants(num32BitConstants=2,b0)"
ByteAddressBuffer current : register(t0);
RWByteAddressBuffer baseline : register(u0);
RWByteAddressBuffer staging : register(u1);
RWByteAddressBuffer flags : register(u2);
cbuffer Params : register(b0) { uint offset; uint first_flag; };
groupshared uint changed;
[RootSignature(RS)]
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint tid : SV_GroupIndex) {
    if (tid == 0) changed = 0;
    GroupMemoryBarrierWithGroupSync();
    uint base = offset + group.x * 65536 + tid * 16;
    bool differs = false;
    [loop] for (uint i = 0; i < 16; ++i) {
        uint at = base + i * 4096;
        uint4 after = current.Load4(at);
        uint4 before = baseline.Load4(at);
        if (any(after != before)) {
            // Private staging only. The CPU still merges exact changed
            // title bytes, including concurrent CPU writes in the gaps.
            staging.Store4(at, after);
            baseline.Store4(at, after);
            differs = true;
        }
    }
    if (differs) InterlockedOr(changed, 1);
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) flags.Store((first_flag + group.x) * 4, changed);
}
"#;

pub(crate) struct Scan {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    flags: ID3D12Resource,
    mapped: usize,
}

impl Scan {
    pub(crate) fn new(host: &Host) -> Option<Self> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: The static source spans SOURCE.len() bytes; the compiler receives valid output slots and static C strings.
        let compiled = unsafe { D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("card-scan"), None, None,
            s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors)) };
        if let Err(e) = compiled {
            if let Some(blob) = errors {
                // SAFETY: The live compiler blob owns the reported byte range for this borrow.
                let bytes = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
                tracing::warn!(error=%e, compiler=%String::from_utf8_lossy(bytes), "exact card scan unavailable");
            }
            return None;
        }
        let code = code?;
        // SAFETY: The live compiled blob supplies its bytecode, and the root signature and pipeline output are created on the live device.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).ok()?;
            let pso = host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            }).ok()?;
            (root, pso)
        };
        let (flags, mapped) = host.shared_buffer((MAX_FLAGS * 4) as u64)?;
        Some(Self { root, pso, flags, mapped: mapped as usize })
    }

    pub(crate) fn begin(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: The caller supplies an open list; flags starts in COMMON and the root and PSO belong to its device.
        unsafe {
            list.ResourceBarrier(&[transition(&self.flags, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootUnorderedAccessView(3, self.flags.GetGPUVirtualAddress());
        }
    }

    /// Call once per shadow, with sorted disjoint complete pages. All three
    /// resource identities are distinct; the baseline has been initialized.
    pub(crate) fn record(&self, list: &ID3D12GraphicsCommandList,
        current: &ID3D12Resource, baseline: &ID3D12Resource, staging: &ID3D12Resource,
        spans: &[(usize, usize, usize)]) {
        let common = D3D12_RESOURCE_STATE_COMMON;
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        // SAFETY: The caller provides distinct live buffers in COMMON, initialized baseline and in-bounds page spans; root addresses and constants match the scan PSO.
        unsafe {
            list.ResourceBarrier(&[transition(current, common, read), transition(baseline, common, write), transition(staging, common, write)]);
            list.SetComputeRootShaderResourceView(0, current.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(1, baseline.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, staging.GetGPUVirtualAddress());
            for &(from, to, flag) in spans {
                assert!(from % 16 == 0 && from < to && (to-from) % PAGE == 0 && to as u64 <= 1u64<<32);
                let pages = (to-from)/PAGE;
                assert!(flag+pages <= MAX_FLAGS && pages <= 65535);
                let constants = [from as u32, flag as u32];
                list.SetComputeRoot32BitConstants(4, 2, constants.as_ptr().cast(), 0);
                list.Dispatch(pages as u32, 1, 1);
            }
            list.ResourceBarrier(&[transition(current, read, common), transition(baseline, write, common), transition(staging, write, common)]);
        }
    }

    pub(crate) fn end(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: The open list has previously transitioned flags to UAV in begin.
        unsafe { list.ResourceBarrier(&[transition(&self.flags, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)]); }
    }

    /// Only after the queue fence has signalled. Every dispatched group
    /// writes its flag unconditionally, so stale flags cannot be consumed.
    pub(crate) fn unchanged(&self, flag: usize) -> bool {
        assert!(flag < MAX_FLAGS);
        // SAFETY: The caller has waited for the fence; flag is bounded by the mapped allocation.
        unsafe { (self.mapped as *const u32).add(flag).read_volatile() == 0 }
    }
}
