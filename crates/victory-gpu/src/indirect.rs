//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped into a separate
//! compact scratch buffer on the GPU, just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0), SRV(t1), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint has_count; };
ByteAddressBuffer args : register(t0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer output_args : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint limit = records;
    [branch]
    if (has_count != 0) {
        limit = min(records, count_buf.Load(0));
    }
    if (id.x >= limit) return;
    uint at = id.x * stride;
    uint out_at = id.x * (kind == 0 ? 16 : (kind == 1 ? 20 : 12));
    uint cost;
    uint4 draw = 0;
    uint tail = 0;
    uint3 groups = 0;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        draw = args.Load4(at);
        if (kind == 1) tail = args.Load(at + 16);
        uint count = min(draw.x, 1u << 24);
        uint instances = min(draw.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw.xy = uint2(count, instances);
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        groups = g;
    }
    uint cur = budget.Load(0);
    bool keep = false;
    if (cost == 0) {
        keep = cur < BUDGET;
    } else {
        [allow_uav_condition]
        while (cur < BUDGET && cur + cost <= BUDGET) {
            uint prev;
            budget.InterlockedCompareExchange(0, cur, cur + cost, prev);
            if (prev == cur) {
                keep = true;
                break;
            }
            cur = prev;
        }
    }
    if (kind <= 1) {
        if (!keep) draw.x = 0;
        output_args.Store4(out_at, draw);
        if (kind == 1) output_args.Store(out_at + 16, tail);
    } else {
        if (!keep) groups.x = 0;
        output_args.Store3(out_at, groups);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer` (bounded by `count` when present), into separate scratch.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). None if the clamp is not available.
pub(crate) fn compact_stride(kind: u32) -> Option<u32> {
    match kind { 0 => Some(16), 1 => Some(20), 2 | 10 => Some(12), _ => None }
}

pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let c = clamp()?;
    let out_stride = compact_stride(kind)?;
    if records == 0
        || !gpu.is_multiple_of(4)
        || !stride.is_multiple_of(4)
        || stride < out_stride
        || records.div_ceil(64) > 65535
        || u64::from(stride) * u64::from(records) > u64::from(u32::MAX) + 1
        || count.is_some_and(|(_, g)| !g.is_multiple_of(4))
    {
        return None;
    }
    let budget = list.budget()?;
    let output = list.indirect_buffer(u64::from(records) * u64::from(out_stride))?;
    let count_buf = count
        .map(|(b, _)| b)
        .filter(|b| b.as_raw() != buffer.as_raw());
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(count.is_some())];
    list.ensure_buffer_state(buffer, 0x40);
    if let Some(cb) = count_buf {
        list.ensure_buffer_state(cb, 0x40);
    }
    list.ensure_buffer_state(&output, 0x8);
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootShaderResourceView(3, count.map_or(gpu, |(_, g)| g));
        l.SetComputeRootUnorderedAccessView(4, output.GetGPUVirtualAddress());
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[uav_barrier()]);
    }
    Some(output)
}

pub(crate) fn ready(list: &HostList, scratch: &ID3D12Resource, count: Option<&ID3D12Resource>) {
    list.ensure_buffer_state(scratch, 0x200);
    if let Some(cb) = count {
        list.ensure_buffer_state(cb, 0x200);
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn clamp_honours_count_buffer_and_does_not_wrap_budget() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let (args, args_ptr) = host.shared_buffer(4096).unwrap();
        let (count, count_ptr) = host.shared_buffer(256).unwrap();
        // safety: both buffers are mapped shared memory of the sizes above.
        let (args_gpu, count_gpu) = unsafe {
            let records = std::slice::from_raw_parts_mut(args_ptr.cast::<[u32; 4]>(), 80);
            records[0] = [100, 2, 0, 0];
            for slot in &mut records[1..] {
                *slot = [1 << 24, 16, 0, 0];
            }
            count_ptr.cast::<u32>().write(1);
            (args.GetGPUVirtualAddress(), count.GetGPUVirtualAddress())
        };
        let output = super::bound(
            &list,
            &args,
            args_gpu,
            Some((&count, count_gpu)),
            (0, 16, 80),
        ).unwrap();
        let (result, result_ptr) = host.shared_buffer(4096).unwrap();
        super::ready(&list, &output, Some(&count));
        list.ensure_buffer_state(&result, 0x400);
        // SAFETY: The open list has prepared the source and destination states; both buffers cover the 16-byte copy.
        unsafe { list.list.CopyBufferRegion(&result, 0, &output, 0, 16); }
        list.close();
        host.execute(&[&list]);
        // safety: `host.execute` waits for completion; `args` is mapped.
        let out = unsafe { std::slice::from_raw_parts(args_ptr.cast::<[u32; 4]>(), 80) };
        assert_eq!(out[0], [100, 2, 0, 0]);
        assert_eq!(out[1], [1 << 24, 16, 0, 0]);
        // SAFETY: The completed GPU copy initialized the first aligned 16 bytes of the mapped result.
        assert_eq!(unsafe { result_ptr.cast::<[u32; 4]>().read() }, [100, 2, 0, 0]);

        let list2 = host.new_list().unwrap();
        // safety: mapped shared memory.
        unsafe {
            let records = std::slice::from_raw_parts_mut(args_ptr.cast::<[u32; 4]>(), 80);
            for slot in records.iter_mut() {
                *slot = [1 << 24, 16, 0, 0];
            }
        }
        let output = super::bound(&list2, &args, args_gpu, None, (0, 16, 80)).unwrap();
        super::ready(&list2, &output, None);
        list2.ensure_buffer_state(&result, 0x400);
        // SAFETY: The prepared source and destination cover all 80 records and the list is open.
        unsafe { list2.list.CopyBufferRegion(&result, 0, &output, 0, 80 * 16); }
        list2.close();
        host.execute(&[&list2]);
        // safety: `host.execute` waits for completion.
        let out = unsafe { std::slice::from_raw_parts(result_ptr.cast::<[u32; 4]>(), 80) };
        assert_eq!(out.iter().filter(|r| r[0] > 0).count(), 2);
        // SAFETY: The mapped input allocation contains 80 aligned records initialized by the test; the GPU execution is complete.
        assert!(unsafe { std::slice::from_raw_parts(args_ptr.cast::<[u32; 4]>(), 80) }.iter().all(|r| *r == [1 << 24, 16, 0, 0]));
    }

    #[test]
    fn compact_clamp_preserves_padded_indexed_records_and_shared_count() {
        use windows::core::Interface;
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let (source, ptr) = host.shared_buffer(4096).unwrap();
        let (result, dst) = host.shared_buffer(4096).unwrap();
        let first = [5u32, 3, 17, (-12i32) as u32, 29, 0xfeed, 0xbeef, 0x1234];
        let second = [7u32, 2, 19, (-4i32) as u32, 31, 0xaa, 0xbb, 0xcc];
        // SAFETY: The mapped source allocation covers both aligned records and the count at byte 256.
        let gpu = unsafe {
            ptr.cast::<[u32; 8]>().write(first);
            ptr.add(32).cast::<[u32; 8]>().write(second);
            ptr.add(256).cast::<u32>().write(1);
            source.GetGPUVirtualAddress()
        };
        let a = super::bound(&list, &source, gpu, Some((&source, gpu + 256)), (1, 32, 2)).unwrap();
        super::ready(&list, &a, Some(&source));
        list.ensure_buffer_state(&result, 0x400);
        // SAFETY: The prepared scratch and result cover the 20-byte copy and the list is open.
        unsafe { list.list.CopyBufferRegion(&result, 0, &a, 0, 20); }
        let b = super::bound(&list, &source, gpu + 32, None, (1, 32, 1)).unwrap();
        assert_eq!(a.as_raw(), b.as_raw());
        super::ready(&list, &b, None);
        // SAFETY: The second prepared scratch record and the result's second 20-byte range are live and in copy states.
        unsafe { list.list.CopyBufferRegion(&result, 20, &b, 0, 20); }
        list.close();
        host.execute(&[&list]);
        // SAFETY: The completed copies initialized 40 bytes of the aligned mapped result.
        assert_eq!(unsafe { std::slice::from_raw_parts(dst.cast::<u32>(), 10) }, [&first[..5], &second[..5]].concat());
        // SAFETY: The first aligned record is within the live mapped source.
        assert_eq!(unsafe { ptr.cast::<[u32; 8]>().read() }, first);
        // SAFETY: The second aligned record at byte 32 is within the live mapped source.
        assert_eq!(unsafe { ptr.add(32).cast::<[u32; 8]>().read() }, second);
        // SAFETY: The aligned count word at byte 256 is within the live mapped source.
        assert_eq!(unsafe { ptr.add(256).cast::<u32>().read() }, 1);
    }
}
