//! Optional packing of independent sampled textures into non-overlapping
//! default-heap ranges. Offsets are never recycled; resources keep their
//! backing heap alive, including when held by a descriptor or command list.
use std::sync::{Mutex, OnceLock};
use windows::{core::{GUID, Interface}, Win32::Graphics::Direct3D12::*};

const CHUNK: u64 = 64 * 1024 * 1024;
const MAX_TEXTURE: u64 = 8 * 1024 * 1024;
const OWNER: GUID = GUID::from_u128(0xb04171fe_66ea_4e08_b581_87e822384058);

struct Chunk {
    device: usize,
    heap: ID3D12Heap,
    end: u64,
}

static POOL: Mutex<Option<Chunk>> = Mutex::new(None);

fn reserve(end: u64, size: u64, alignment: u64) -> Option<(u64, u64)> {
    if !alignment.is_power_of_two() || size == 0 {
        return None;
    }
    let offset = end.checked_add(alignment - 1)? & !(alignment - 1);
    let next = offset.checked_add(size)?;
    (next <= CHUNK).then_some((offset, next))
}

pub(crate) fn allocate(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PACK_TEXTURE_HEAPS"))
        || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.Flags != D3D12_RESOURCE_FLAG_NONE
        || desc.SampleDesc.Count != 1
    {
        return None;
    }
    let mut packed = *desc;
    // Query the driver rather than assuming that a texture supports 4 KiB
    // placement. The standard alignment remains a safe fallback.
    packed.Alignment = 4096;
    // SAFETY: The device is live and the slice contains one initialized resource description.
    let mut info = unsafe { device.GetResourceAllocationInfo(0, &[packed]) };
    if info.SizeInBytes == u64::MAX || info.Alignment != 4096 {
        packed.Alignment = 0;
        // SAFETY: The device is live and the description has been reset to default alignment.
        info = unsafe { device.GetResourceAllocationInfo(0, &[packed]) };
    }
    if info.SizeInBytes > MAX_TEXTURE || info.Alignment > 65536 {
        return None;
    }
    let mut pool = POOL.lock().ok()?;
    let device_key = device.as_raw() as usize;
    if pool.as_ref().is_none_or(|c| {
        c.device != device_key || reserve(c.end, info.SizeInBytes, info.Alignment).is_none()
    }) {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: CHUNK,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            Alignment: 65536,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap = None;
        // SAFETY: The initialized heap description and output slot are valid for the live device.
        unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
        *pool = Some(Chunk { device: device_key, heap: heap?, end: 0 });
    }
    let chunk = pool.as_mut()?;
    let (offset, next) = reserve(chunk.end, info.SizeInBytes, info.Alignment)?;
    // Consume the range even if creation or attaching the owner fails.
    chunk.end = next;
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: The chunk owns the heap; offset is aligned and the allocation fits the reserved range.
    unsafe {
        device.CreatePlacedResource(&chunk.heap, offset, &packed, state, None, &mut resource)
    }.ok()?;
    let resource = resource?;
    // The heap must outlive every COM reference to this resource. Private
    // interface data holds a reference until the resource itself is destroyed.
    // SAFETY: The resource and heap are live COM objects and OWNER is a valid private-data key.
    unsafe { resource.SetPrivateDataInterface(&OWNER, &chunk.heap) }.ok()?;
    Some(resource)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_aligned_disjoint_and_bounded() {
        assert_eq!(reserve(0, 4096, 4096), Some((0, 4096)));
        assert_eq!(reserve(4096, 65536, 65536), Some((65536, 131072)));
        assert_eq!(reserve(CHUNK - 4096, 4096, 4096), Some((CHUNK - 4096, CHUNK)));
        assert_eq!(reserve(CHUNK, 4096, 4096), None);
        assert_eq!(reserve(u64::MAX, 4096, 4096), None);
        assert_eq!(reserve(0, 4096, 0), None);
        assert_eq!(reserve(0, 0, 4096), None);
    }
}
