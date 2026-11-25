//! Single-plane D16 depth/UAV coherence.
//!
//! Canonical depth is current at every closed list's boundary. The first
//! UAV use in each recording copies it to the writable twin. Subsequent
//! uses switch sides on demand, and close copies back any newer twin.
//! This needs no recording-order global authority: the existing texture
//! prelude orders transitions in actual submission order, including repeat
//! submissions and lists recorded before the twin existed.
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use crate::{host::{HostList, host, transition}, object::This, resource::HostTexture};

const UAV: u32 = 0x8;
const SRC: u32 = 0x800;
const DST: u32 = 0x400;

pub(crate) struct Twin {
    color: Arc<HostTexture>,
    scratch: ID3D12Resource,
    // Every slice has separately certified identical 16-bit copy footprints.
    // Scratch is reused serially, never across pending unordered operations.
    copies: Vec<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)>,
}

// The resources are free-threaded; their commands are ordered on the host queue.
unsafe impl Send for Twin {}
unsafe impl Sync for Twin {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side { Depth, Twin }

pub(crate) struct Use {
    depth: Arc<HostTexture>,
    side: Side,
}

#[derive(Default)]
pub(crate) struct Uses {
    active: HashMap<usize, Use>,
    // A single dispatch/draw cannot simultaneously use both independent
    // resources as an alias of the same subresource.
    operation: Option<HashMap<usize, Side>>,
    conflict: bool,
}

pub(crate) struct Operation<'a>(&'a HostList);

pub(crate) fn begin(list: &HostList) -> Operation<'_> {
    let mut uses = list.twins.lock().unwrap_or_else(|e| e.into_inner());
    debug_assert!(uses.operation.is_none());
    uses.operation = Some(HashMap::new());
    uses.conflict = false;
    Operation(list)
}

impl Operation<'_> {
    pub(crate) fn valid(&self) -> bool {
        !self.0.twins.lock().unwrap_or_else(|e| e.into_inner()).conflict
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        let mut uses = self.0.twins.lock().unwrap_or_else(|e| e.into_inner());
        uses.operation = None;
        uses.conflict = false;
    }
}

impl Uses {
    fn note(&mut self, key: usize, side: Side) -> bool {
        if let Some(operation) = &mut self.operation {
            if operation.get(&key).is_some_and(|&old| old != side) {
                self.conflict = true;
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!("simultaneous D16 depth and UAV alias use is unsupported"));
                return false;
            }
            operation.insert(key, side);
        }
        true
    }
}

fn eligible(d: &D3D12_RESOURCE_DESC) -> bool {
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.Format == DXGI_FORMAT_R16_TYPELESS
        && d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0
        && d.Width > 0 && d.Height > 0
        && d.DepthOrArraySize >= 1
        && d.DepthOrArraySize <= crate::resource::D16_TWIN_MAX_SLICES
        && d.MipLevels == 1
        && d.SampleDesc.Count == 1 && d.SampleDesc.Quality == 0
}

/// Exact single-plane view, no reinterpretation of the requested data type.
fn view_format(d: &D3D12_UNORDERED_ACCESS_VIEW_DESC, slices: u16) -> Option<DXGI_FORMAT> {
    if !matches!(d.Format, DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_UNORM)
        || slices == 0 { return None; }
    match d.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D if slices == 1 => {
            let sub = unsafe { d.Anonymous.Texture2D };
            (sub.MipSlice == 0 && sub.PlaneSlice == 0).then_some(d.Format)
        }
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
            let sub = unsafe { d.Anonymous.Texture2DArray };
            (sub.MipSlice == 0 && sub.PlaneSlice == 0
                && sub.ArraySize != 0 && sub.FirstArraySlice < u32::from(slices)
                && sub.ArraySize <= u32::from(slices) - sub.FirstArraySlice).then_some(d.Format)
        }
        _ => None,
    }
}

/// Reuse the existing descriptor::host::fits normalization for 16-bit packed
/// UAV views backed by R16_TYPELESS. The previous scratch stand-in always
/// exposed these as UINT16; the coherent twin must keep that exact value
/// interpretation instead of rejecting the legacy console format.
fn normalize_view(d: &D3D12_UNORDERED_ACCESS_VIEW_DESC,
    resource: &D3D12_RESOURCE_DESC) -> D3D12_UNORDERED_ACCESS_VIEW_DESC
{
    let mut view = *d;
    if resource.Format == DXGI_FORMAT_R16_TYPELESS
        && matches!(view.Format.0, 0 | 86 | 115)
    { view.Format = DXGI_FORMAT_R16_UINT; }
    view
}

fn supported(format: DXGI_FORMAT) -> bool {
    static UINT: OnceLock<bool> = OnceLock::new();
    static UNORM: OnceLock<bool> = OnceLock::new();
    let cache = match format {
        DXGI_FORMAT_R16_UINT => &UINT,
        DXGI_FORMAT_R16_UNORM => &UNORM,
        _ => return false,
    };
    *cache.get_or_init(|| {
        let Some(host) = host() else { return false };
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        let required = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0 | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0;
        unsafe { host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut support).cast(),
            size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32)
        }.is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
            && support.Support2.0 & required == required
    })
}

/// A small bounded set of distinct rejected layout signatures. A single
/// scene may bind thousands of shadow-map UAVs, so report a representative
/// resource for each unsupported shape rather than logging every dispatch.
/// This preserves the fallback path and never treats pointer equality as an
/// alias-coherence proof.
fn report_rejection(resource: This, desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC, reason: &'static str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static LOGGED: OnceLock<Mutex<HashSet<(u64, u32, u16, i32, i32, &'static str)>>> = OnceLock::new();
    let dd = crate::resource::host_copy(resource)
        .map(|depth| unsafe { depth.texture.GetDesc() });
    let width = dd.as_ref().map_or(0, |d| d.Width);
    let height = dd.as_ref().map_or(0, |d| d.Height);
    let slices = dd.as_ref().map_or(0, |d| d.DepthOrArraySize);
    let key = (width, height, slices, desc.Format.0, desc.ViewDimension.0, reason);
    let mut logged = LOGGED.get_or_init(|| Mutex::new(HashSet::new()))
        .lock().unwrap_or_else(|e| e.into_inner());
    if logged.contains(&key) || logged.len() >= 128 { return; }
    logged.insert(key);
    drop(logged);
    let (first_slice, view_slices, mip, plane) = if desc.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY {
        let a = unsafe { desc.Anonymous.Texture2DArray };
        (a.FirstArraySlice, a.ArraySize, a.MipSlice, a.PlaneSlice)
    } else if desc.ViewDimension == D3D12_UAV_DIMENSION_TEXTURE2D {
        let a = unsafe { desc.Anonymous.Texture2D };
        (0, 1, a.MipSlice, a.PlaneSlice)
    } else { (0, 0, 0, 0) };
    let original_data = crate::resource::texture_memory(resource)
        .map(|(address, _)| address as usize);
    tracing::warn!(reason, object=?resource, width, height, slices,
        original_data=?original_data,
        host_format=dd.as_ref().map(|d| d.Format.0).unwrap_or(0),
        host_flags=dd.as_ref().map(|d| d.Flags.0).unwrap_or(0),
        title_rejection=?crate::resource::d16_twin_rejection(resource),
        uav_format=desc.Format.0, uav_dimension=desc.ViewDimension.0,
        first_slice, view_slices, mip, plane,
        "D16 coherent twin unavailable; depth UAV still uses scratch fallback");
}

fn footprint(device: &ID3D12Device, d: &D3D12_RESOURCE_DESC, index: u32)
    -> (D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u32, u64, u64)
{
    let (mut fp, mut rows, mut row_bytes, mut total) =
        (D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(), 0, 0, 0);
    unsafe { device.GetCopyableFootprints(d, index, 1, 0, Some(&mut fp),
        Some(&mut rows), Some(&mut row_bytes), Some(&mut total)) };
    (fp, rows, row_bytes, total)
}

impl Twin {
    fn make(depth: &HostTexture) -> Option<Self> {
        let host = host()?;
        let dd = unsafe { depth.texture.GetDesc() };
        if !eligible(&dd) { return None; }
        let cd = D3D12_RESOURCE_DESC {
            Alignment: 0,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..dd
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1, VisibleNodeMask: 1,
            ..Default::default()
        };
        let mut copies = Vec::with_capacity(dd.DepthOrArraySize as usize);
        let mut scratch_bytes = 0;
        // Both footprints must describe exactly the same 16-bit texels.
        // Reject instead of copying padding, a second plane or a conversion.
        let exact = |f: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows, bytes, total| {
            f.Offset == 0 && u64::from(f.Footprint.Width) == dd.Width
                && f.Footprint.Height == dd.Height && f.Footprint.Depth == 1
                && rows == dd.Height && bytes == dd.Width * 2
                && u64::from(f.Footprint.RowPitch) >= bytes
                && total != 0 && total != u64::MAX
                && crate::resource::family(f.Footprint.Format.0 as u32) == 53
        };
        for index in 0..u32::from(dd.DepthOrArraySize) {
            let (df, dr, db, ds) = footprint(&host.device, &dd, index);
            let (cf, cr, cb, cs) = footprint(&host.device, &cd, index);
            if !exact(&df, dr, db, ds) || !exact(&cf, cr, cb, cs)
                || df.Footprint.RowPitch != cf.Footprint.RowPitch { return None; }
            scratch_bytes = scratch_bytes.max(ds).max(cs);
            copies.push((df, cf));
        }
        let make = |d: &D3D12_RESOURCE_DESC| -> Option<ID3D12Resource> {
            let mut out = None;
            unsafe { host.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                d, D3D12_RESOURCE_STATE_COMMON, None, &mut out) }.ok()?;
            out
        };
        let color = make(&cd)?;
        let scratch = make(&D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: scratch_bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        })?;
        crate::host::name(&color, || format!("D16 writable twin {}x{}", dd.Width, dd.Height));
        tracing::info!(width=dd.Width, height=dd.Height, slices=dd.DepthOrArraySize,
            "D16 coherent writable twin created");
        Some(Self { color: HostTexture::internal(color), scratch, copies })
    }

    /// Record a whole-subresource copy through exact buffer footprints. The
    /// private scratch returns to COMMON after each copy, on the same queue.
    fn copy(&self, list: &HostList, depth: &Arc<HostTexture>, to: Side) {
        let (src, dst) = if to == Side::Depth {
            (&self.color, depth)
        } else {
            (depth, &self.color)
        };
        let mut barriers = Vec::new();
        barriers.extend(list.move_texture(src, SRC));
        barriers.extend(list.move_texture(dst, DST));
        barriers.push(transition(&self.scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
        let sub = |r: &ID3D12Resource, index| D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
        };
        let place = |f| D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&self.scratch) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        unsafe {
            list.list.ResourceBarrier(&barriers);
            for (index, &(depth_fp, color_fp)) in self.copies.iter().enumerate() {
                let (sf, df) = if to == Side::Depth { (color_fp, depth_fp) }
                    else { (depth_fp, color_fp) };
                list.list.CopyTextureRegion(&place(sf), 0, 0, 0,
                    &sub(&src.texture, index as u32), None);
                list.list.ResourceBarrier(&[transition(&self.scratch,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.list.CopyTextureRegion(&sub(&dst.texture, index as u32),
                    0, 0, 0, &place(df), None);
                // Each slice is complete on the single host queue before
                // scratch is reused, including the last slice's transition.
                list.list.ResourceBarrier(&[transition(&self.scratch,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    if index + 1 == self.copies.len() { D3D12_RESOURCE_STATE_COMMON }
                    else { D3D12_RESOURCE_STATE_COPY_DEST })]);
            }
        }
    }
}

pub(crate) fn uav(resource: This, desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource>
{
    if let Some(reason) = crate::resource::d16_twin_rejection(resource) {
        report_rejection(resource, desc, reason);
        return None;
    }
    let Some(depth) = crate::resource::host_copy(resource) else {
        report_rejection(resource, desc, "host_texture_unavailable");
        return None;
    };
    let dd = unsafe { depth.texture.GetDesc() };
    let resolved = normalize_view(desc, &dd);
    let Some(format) = view_format(&resolved, dd.DepthOrArraySize) else {
        report_rejection(resource, desc, "unsupported_view_format_or_slice_range");
        return None;
    };
    if !supported(format) {
        report_rejection(resource, desc, "host_typed_uav_unavailable");
        return None;
    }
    let first_use = depth.twin.get().is_none();
    let Some(twin) = depth.twin.get_or_init(|| Twin::make(&depth)).as_ref() else {
        report_rejection(resource, desc, "exact_twin_creation_or_footprint_failed");
        return None;
    };
    if first_use && dd.DepthOrArraySize > 1 {
        tracing::info!(object=?resource,
            original_data=?crate::resource::texture_memory(resource).map(|(address, _)| address as usize),
            width=dd.Width, height=dd.Height, slices=dd.DepthOrArraySize,
            requested_format=desc.Format.0, resolved_format=resolved.Format.0,
            "original D16 array UAV using coherent twin, not scratch");
    }
    unsafe { host()?.device.CreateUnorderedAccessView(&twin.color.texture,
        None::<&ID3D12Resource>, Some(&resolved), at) };
    Some(twin.color.texture.clone())
}

fn route(list: &HostList, depth: Arc<HostTexture>, side: Side, wanted: u32) -> bool {
    let key = Arc::as_ptr(&depth) as usize;
    let mut uses = list.twins.lock().unwrap_or_else(|e| e.into_inner());
    if !uses.note(key, side) { return true; } // caller drops conflicting draw
    let Some(Some(twin)) = depth.twin.get() else { return false };
    let usage = uses.active.entry(key).or_insert_with(|| Use { depth: depth.clone(), side: Side::Depth });
    if usage.side != side {
        twin.copy(list, &depth, side);
        usage.side = side;
    }
    let copy = if side == Side::Depth { &depth } else { &twin.color };
    if let Some(barrier) = list.move_texture(copy, wanted) {
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    } else if side == Side::Twin {
        // Consecutive UAV uses may read or write the earlier results.
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 { UAV: std::mem::ManuallyDrop::new(
                D3D12_RESOURCE_UAV_BARRIER { pResource: unsafe { std::mem::transmute_copy(&copy.texture) } }) },
        };
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    true
}

pub(crate) fn depth_use(list: &HostList, resource: This, wanted: u32) -> bool {
    if wanted & UAV != 0 || !crate::resource::d16_twin_candidate(resource) { return false; }
    let Some(depth) = crate::resource::host_copy(resource) else { return false };
    route(list, depth, Side::Depth, wanted)
}

pub(crate) fn uav_use(list: &HostList, resource: This, desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC) -> bool {
    if !crate::resource::d16_twin_candidate(resource) { return false; }
    let Some(depth) = crate::resource::host_copy(resource) else { return false };
    let dd = unsafe { depth.texture.GetDesc() };
    let resolved = normalize_view(desc, &dd);
    if !view_format(&resolved, dd.DepthOrArraySize).is_some_and(supported) { return false; }
    if !depth.twin.get().is_some_and(Option::is_some) { return false; }
    route(list, depth, Side::Twin, UAV)
}

/// Each submitted list, regardless of recording order or later re-use,
/// leaves the original depth texture authoritative.
pub(crate) fn finish(list: &HostList) {
    let mut uses = list.twins.lock().unwrap_or_else(|e| e.into_inner());
    debug_assert!(uses.operation.is_none());
    for u in uses.active.values_mut() {
        if u.side == Side::Twin {
            let twin = u.depth.twin.get().and_then(Option::as_ref).expect("retained D16 twin");
            twin.copy(list, &u.depth, Side::Depth);
            u.side = Side::Depth;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
    use windows::core::s;

    fn compile(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let device = &host().unwrap().device;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        unsafe {
            let result = D3DCompile(source.as_ptr().cast(), source.len(),
                s!("twin-test"), None, None, s!("main"), s!("cs_5_1"), 0, 0,
                &mut code, Some(&mut errors));
            if let Err(e) = result {
                let message = errors.as_ref().map(|b| String::from_utf8_lossy(
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())).into_owned());
                panic!("D16 test compile: {e}: {message:?}");
            }
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            }).unwrap();
            (root, pso)
        }
    }

    #[test]
    fn d16_twin_rejects_nonidentical_subresources_and_formats() {
        let mut d = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, Format: DXGI_FORMAT_R16_TYPELESS,
            Width: 8, Height: 8, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, ..Default::default()
        };
        assert!(eligible(&d));
        d.MipLevels = 2;
        assert!(!eligible(&d));
        d.MipLevels = 1;
        d.DepthOrArraySize = 2;
        assert!(eligible(&d), "all array slices must be copied coherently");
        d.DepthOrArraySize = crate::resource::D16_TWIN_MAX_SLICES + 1;
        assert!(!eligible(&d), "large arrays are not certified for whole-resource copies");
        d.DepthOrArraySize = 1;
        d.SampleDesc.Count = 4;
        assert!(!eligible(&d));
        d.SampleDesc.Count = 1;
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        assert!(!eligible(&d));
        let mut u = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        assert_eq!(view_format(&u, 1), Some(DXGI_FORMAT_R16_UINT));
        assert!(view_format(&u, 2).is_none(), "array requires 2DARRAY descriptor");
        u.Format = DXGI_FORMAT_R16_UNORM;
        assert_eq!(view_format(&u, 1), Some(DXGI_FORMAT_R16_UNORM));
        u.Format = DXGI_FORMAT_R16_FLOAT;
        assert!(view_format(&u, 1).is_none());
        u.Format = DXGI_FORMAT_R16_UINT;
        u.Anonymous.Texture2D.MipSlice = 1;
        assert!(view_format(&u, 1).is_none());
        u.Anonymous.Texture2D.MipSlice = 0;
        u.Anonymous.Texture2D.PlaneSlice = 1;
        assert!(view_format(&u, 1).is_none());
        let mut a = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: 0, FirstArraySlice: 2, ArraySize: 4, PlaneSlice: 0,
                },
            },
        };
        assert_eq!(view_format(&a, 6), Some(DXGI_FORMAT_R16_UINT));
        assert!(view_format(&a, 5).is_none(), "array view may not exceed allocation");
        a.Anonymous.Texture2DArray.ArraySize = 0;
        assert!(view_format(&a, 6).is_none(), "empty UAV array view is invalid");
        a.Anonymous.Texture2DArray.ArraySize = 1;
        a.Anonymous.Texture2DArray.PlaneSlice = 1;
        assert!(view_format(&a, 6).is_none(), "second plane is not coherent R16");
        d.Format = DXGI_FORMAT_R16_TYPELESS;
        a.Anonymous.Texture2DArray.PlaneSlice = 0;
        for format in [0, 86, 115] {
            a.Format = DXGI_FORMAT(format);
            assert_eq!(normalize_view(&a, &d).Format, DXGI_FORMAT_R16_UINT,
                "legacy packed-view normalization must match the scratch path");
        }
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        assert_eq!(normalize_view(&a, &d).Format, a.Format,
            "normalization must not reinterpret another resource family");
    }

    /// Hardware: actual typed UINT16 UAV read-modify-write and UNORM depth
    /// SRV. The writer is recorded BEFORE its initial clear, submitted AFTER
    /// it, then submitted again. A later list alternates UAV and depth reads
    /// twice, proving both the within-list route and canonical close.
    #[test]
    fn d16_twin_preserves_writer_order_across_lists_and_reexecution() {
        let Some(host) = host() else { return };
        assert!(supported(DXGI_FORMAT_R16_UINT), "hardware must support R16_UINT typed UAV loads/stores for this test");
        let layout = std::alloc::Layout::from_size_align(65536, 65536).unwrap();
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let words = [3u32, 0, 0, 0, 8, 0, 8, 1 | (1 << 16), 53, 1, 0, 0, 6, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut resource = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(std::ptr::null(), memory as u64,
            words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut resource), 0);
        assert!(crate::resource::d16_twin_candidate(resource));
        let depth = crate::resource::host_copy(resource).unwrap();
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0,
            }).unwrap()
        };
        let (cpu, gpu, step) = unsafe { (heap.GetCPUDescriptorHandleForHeapStart(),
            heap.GetGPUDescriptorHandleForHeapStart(),
            host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)) };
        let u = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let _twin = uav(resource, &u, cpu).expect("D16 twin with matching 16-bit copy footprints");
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UNORM, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + step as usize };
        let srv_gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(step) };
        unsafe { host.device.CreateShaderResourceView(&depth.texture, Some(&srv), srv_cpu) };
        let dsv = host.dsv(&depth.texture, Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D16_UNORM, ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        }));
        let (write_root, write_pso) = compile(r#"
RWTexture2D<uint> t : register(u0);
[RootSignature("DescriptorTable(UAV(u0))")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    t[id.xy] += 101 + id.x + 8 * id.y;
}
"#);
        let (read_root, read_pso) = compile(r#"
Texture2D<float> t : register(t0);
RWByteAddressBuffer outbuf : register(u1);
[RootSignature("DescriptorTable(SRV(t0)), UAV(u1)")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    outbuf.Store((id.x + 8 * id.y)*4, asuint(t.Load(int3(id.xy, 0))));
}
"#);
        let (out, mapped) = host.shared_buffer(1024).unwrap();
        let bind_write = |list: &HostList| {
            let op = begin(list);
            assert!(uav_use(list, resource, &u));
            assert!(op.valid());
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(&write_root);
                list.list.SetPipelineState(&write_pso);
                list.list.SetComputeRootDescriptorTable(0, gpu);
                list.list.Dispatch(1, 1, 1);
            }
        };
        let bind_read = |list: &HostList, offset: u64| {
            let op = begin(list);
            crate::list::ensure_state(list, resource, 0x40 | 0x80);
            assert!(op.valid());
            list.ensure_buffer_state(&out, UAV);
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(&read_root);
                list.list.SetPipelineState(&read_pso);
                list.list.SetComputeRootDescriptorTable(0, srv_gpu);
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress() + offset);
                list.list.Dispatch(1, 1, 1);
            }
        };
        let check = |offset: usize, times: u32| {
            for i in 0..64 {
                let value = unsafe { mapped.add(offset + i * 4).cast::<f32>().read_unaligned() };
                let expected = (times * (101 + i as u32)) as f32 / 65535.0;
                assert!((value - expected).abs() < 1e-7, "pixel {i}, writes {times}: {value} != {expected}");
            }
        };
        // Writer recorded first, clear recorded second, submit in reverse.
        let writer = host.new_list().unwrap();
        bind_write(&writer);
        writer.close();
        let clear = host.new_list().unwrap();
        crate::list::ensure_state(&clear, resource, 0x10);
        unsafe { clear.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0, 0, None) };
        clear.close();
        let reader = host.new_list().unwrap();
        bind_read(&reader, 0);
        reader.close();
        host.execute(&[&clear, &writer, &reader]);
        check(0, 1);
        host.execute(&[&writer, &reader]);
        check(0, 2);
        let alternating = host.new_list().unwrap();
        bind_write(&alternating);
        bind_read(&alternating, 256);
        bind_write(&alternating);
        bind_read(&alternating, 512);
        alternating.close();
        host.execute(&[&alternating, &reader]);
        check(256, 3);
        check(512, 4);
        check(0, 4);
        // Clear again after writes; a new submit of the older writer must
        // copy the current depth, not the previous execution's twin.
        host.execute(&[&clear, &writer, &reader]);
        check(0, 1);
        // Simultaneous depth and UAV alias use is not claimed supported.
        let conflict = host.new_list().unwrap();
        {
            let op = begin(&conflict);
            assert!(uav_use(&conflict, resource, &u));
            assert!(depth_use(&conflict, resource, 0x40));
            assert!(!op.valid());
        }
        conflict.close();
        host.execute(&[&conflict, &reader]);
        check(0, 1);
        crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource);
        unsafe { std::alloc::dealloc(memory, layout) };
    }

    /// Original prison compute shaders bind writable R16 depth arrays. Each
    /// UAV use must preserve all other array slices, including when lists are
    /// recorded out of submission order and a prior writer is replayed.
    #[test]
    fn d16_array_twin_preserves_slices_and_submission_order() {
        let Some(host) = host() else { return };
        assert!(supported(DXGI_FORMAT_R16_UINT), "R16 UINT typed UAV support required");
        const SLICES: usize = 6;
        const PIXELS: usize = 64;
        let layout = std::alloc::Layout::from_size_align(65536, 65536).unwrap();
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let words = [3u32, 0, 0, 0, 8, 0, 8, SLICES as u32 | (1 << 16), 53, 1, 0, 0, 6, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut resource = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(std::ptr::null(), memory as u64,
            words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut resource), 0);
        assert!(crate::resource::d16_twin_candidate(resource));
        let depth = crate::resource::host_copy(resource).unwrap();
        assert_eq!(unsafe { depth.texture.GetDesc() }.DepthOrArraySize, SLICES as u16);
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0,
            }).unwrap()
        };
        let (cpu, gpu, step) = unsafe { (heap.GetCPUDescriptorHandleForHeapStart(),
            heap.GetGPUDescriptorHandleForHeapStart(),
            host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)) };
        let array = |first, size| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UINT, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: 0, FirstArraySlice: first, ArraySize: size, PlaneSlice: 0,
                },
            },
        };
        let entire = array(0, SLICES as u32);
        let partial = array(2, 1);
        let mut legacy = entire;
        legacy.Format = DXGI_FORMAT_UNKNOWN;
        assert!(uav(resource, &entire, cpu).is_some(), "six coherent slices with exact footprints");
        let partial_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + 2 * step as usize };
        assert!(uav(resource, &partial, partial_cpu).is_some(), "one-slice UAV on full coherent array");
        let legacy_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + 3 * step as usize };
        assert!(uav(resource, &legacy, legacy_cpu).is_some(),
            "legacy untyped console view must normalize to the existing R16 UINT representation");
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16_UNORM, ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0, MipLevels: 1, FirstArraySlice: 0,
                    ArraySize: SLICES as u32, PlaneSlice: 0, ResourceMinLODClamp: 0.0,
                },
            },
        };
        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + step as usize };
        let srv_gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + u64::from(step) };
        let partial_gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + 2 * u64::from(step) };
        let legacy_gpu = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + 3 * u64::from(step) };
        unsafe { host.device.CreateShaderResourceView(&depth.texture, Some(&srv), srv_cpu) };
        let dsv = host.dsv(&depth.texture, Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D16_UNORM,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: 0, FirstArraySlice: 0, ArraySize: SLICES as u32,
                },
            },
            ..Default::default()
        }));
        let (full_root, full_pso) = compile(r#"
RWTexture2DArray<uint> t : register(u0);
[RootSignature("DescriptorTable(UAV(u0))")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    t[id] += 101 + id.x + 8 * id.y + 201 * id.z;
}
"#);
        let (partial_root, partial_pso) = compile(r#"
RWTexture2DArray<uint> t : register(u0);
[RootSignature("DescriptorTable(UAV(u0))")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    t[id] += 17;
}
"#);
        let (read_root, read_pso) = compile(r#"
Texture2DArray<float> t : register(t0);
RWByteAddressBuffer outbuf : register(u1);
[RootSignature("DescriptorTable(SRV(t0)), UAV(u1)")]
[numthreads(8,8,1)]
void main(uint3 id : SV_DispatchThreadID) {
    outbuf.Store((id.z * 64 + id.y * 8 + id.x) * 4,
        asuint(t.Load(int4(id, 0))));
}
"#);
        let (out, mapped) = host.shared_buffer((SLICES * PIXELS * 4) as u64).unwrap();
        let bind_write = |list: &HostList, subset: bool| {
            let op = begin(list);
            assert!(uav_use(list, resource, if subset { &partial } else { &entire }));
            assert!(op.valid());
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(if subset { &partial_root } else { &full_root });
                list.list.SetPipelineState(if subset { &partial_pso } else { &full_pso });
                list.list.SetComputeRootDescriptorTable(0, if subset { partial_gpu } else { gpu });
                list.list.Dispatch(1, 1, if subset { 1 } else { SLICES as u32 });
            }
        };
        let bind_read = |list: &HostList| {
            let op = begin(list);
            crate::list::ensure_state(list, resource, 0x40 | 0x80);
            assert!(op.valid());
            list.ensure_buffer_state(&out, UAV);
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                list.list.SetComputeRootSignature(&read_root);
                list.list.SetPipelineState(&read_pso);
                list.list.SetComputeRootDescriptorTable(0, srv_gpu);
                list.list.SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, SLICES as u32);
            }
        };
        let check = |writes: u32, partial_writes: u32| {
            for slice in 0..SLICES {
                for pixel in 0..PIXELS {
                    let index = slice * PIXELS + pixel;
                    let actual = unsafe { mapped.add(index * 4).cast::<f32>().read_unaligned() };
                    let expected = (writes * (101 + pixel as u32 + 201 * slice as u32)
                        + if slice == 2 { partial_writes * 17 } else { 0 }) as f32 / 65535.0;
                    assert!((actual - expected).abs() < 1e-7,
                        "slice={slice} pixel={pixel} full={writes} partial={partial_writes}: {actual} != {expected}");
                }
            }
        };
        let writer = host.new_list().unwrap();
        bind_write(&writer, false);
        writer.close();
        let clear = host.new_list().unwrap();
        crate::list::ensure_state(&clear, resource, 0x10);
        unsafe { clear.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.0, 0, None) };
        clear.close();
        let reader = host.new_list().unwrap();
        bind_read(&reader);
        reader.close();
        // Submission order matters more than recording order.
        host.execute(&[&clear, &writer, &reader]);
        check(1, 0);
        host.execute(&[&writer, &reader]);
        check(2, 0);
        let subset = host.new_list().unwrap();
        bind_write(&subset, true);
        subset.close();
        host.execute(&[&subset, &reader]);
        check(2, 1);
        // A new clear supersedes all earlier writes; replay a previously
        // recorded partial writer without reviving stale data in other slices.
        host.execute(&[&clear, &subset, &reader]);
        check(0, 1);
        // Within one list the writable twin is copied back only at close.
        let alternating = host.new_list().unwrap();
        bind_write(&alternating, false);
        bind_read(&alternating);
        alternating.close();
        host.execute(&[&alternating, &reader]);
        check(1, 1);
        let conflict = host.new_list().unwrap();
        {
            let op = begin(&conflict);
            assert!(uav_use(&conflict, resource, &partial));
            assert!(depth_use(&conflict, resource, 0x40));
            assert!(!op.valid(), "same-dispatch depth+UAV alias must remain rejected");
        }
        conflict.close();
        host.execute(&[&conflict, &reader]);
        check(1, 1);
        // The legacy descriptor's UNKNOWN/packed 16-bit format must route
        // through the same coherent UINT view, never a separate scratch.
        let legacy_writer = host.new_list().unwrap();
        {
            let op = begin(&legacy_writer);
            assert!(uav_use(&legacy_writer, resource, &legacy));
            assert!(op.valid());
            unsafe {
                legacy_writer.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                legacy_writer.list.SetComputeRootSignature(&full_root);
                legacy_writer.list.SetPipelineState(&full_pso);
                legacy_writer.list.SetComputeRootDescriptorTable(0, legacy_gpu);
                legacy_writer.list.Dispatch(1, 1, SLICES as u32);
            }
        }
        legacy_writer.close();
        host.execute(&[&legacy_writer, &reader]);
        check(2, 1);
        crate::object::slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource);
        unsafe { std::alloc::dealloc(memory, layout) };
    }
}
