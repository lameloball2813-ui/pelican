//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from.
    made: Mutex<Vec<u64>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Arc<Mutex<(u32, Vec<bool>)>>,
    /// Only weak entries: recorded lists, not the cache, own the slots.
    shared_tables: Mutex<std::collections::HashMap<SharedTableKey, std::sync::Weak<SharedTable>>>,
    /// Actual bytes per element by original title descriptor heap slot.
    /// Kept as an upload buffer until the heap (and its submitted lists) dies.
    strides: Option<(ID3D12Resource, u64, usize)>,
    buffer_slots: Mutex<std::collections::BTreeSet<u32>>,
    /// The exact buffer copy behind a materialized original descriptor.
    /// A second address lookup can find a different, overlapping allocation.
    buffer_bindings: Mutex<std::collections::BTreeMap<u32, BufferBinding>>,
    /// Raw title descriptor observations, never host bindings. Only the
    /// authoritative physical-page journal can allow a page to be reused.
    read_records: Mutex<ReadRecords>,
    /// Serialize original-slot mutations and plan validation. Host-only
    /// dynamic slots do not participate in the unknown SRV demand.
    read_revision: Mutex<HeapRevision>,
}

const SLOTS_PER_PAGE: u32 = (crate::watch::PAGE / 32) as u32;

struct HeapRevision {
    generation: u64,
    pages: Vec<u64>,
}

impl HeapRevision {
    fn new(count: u32) -> Self {
        Self { generation: 0, pages: vec![0; count.div_ceil(SLOTS_PER_PAGE) as usize] }
    }

    fn changed(&mut self, index: u32) {
        self.generation = self.generation.checked_add(1).expect("heap revision exhausted");
        self.pages[(index / SLOTS_PER_PAGE) as usize] = self.generation;
    }

    // A physical record page may straddle two index pages if the title heap
    // begins at a non-page-aligned address. Each mutation has a unique
    // increasing epoch, so the maximum changes if either page changes.
    fn range(&self, from: u32, to: u32) -> u64 {
        if from == to { return 0; }
        self.pages[(from / SLOTS_PER_PAGE) as usize..=((to - 1) / SLOTS_PER_PAGE) as usize]
            .iter().copied().max().unwrap_or(0)
    }
}

#[derive(Default)]
struct ReadRecords {
    start: u64,
    count: u32,
    epoch: u64,
    pages: Vec<Vec<(u32, [u32; 3])>>,
    plan: Option<(u64, u64, Arc<ReadPlan>)>,
    mixed: Vec<(DemandKey, u64, u64, Arc<ReadPlan>)>,
    /// Includes dead tags and not-yet-allocated indices, not just live
    /// buffer descriptors. Registry changes can make these tags live.
    dependencies: Option<std::collections::HashSet<u32>>,
    page_plans: Vec<Option<PagePlan>>,
    chunks: Vec<Option<PlanChunk>>,
}

// A bounded cache of consecutive, already validated immutable page plans.
// Retain the Arc identities themselves: a freed allocation's address must
// never accidentally match a different plan. No descriptor validation is
// bypassed by this cache.
const PLAN_CHUNK_PAGES: usize = 8;
struct PlanChunk {
    pages: Vec<Arc<ReadPlan>>,
    combined: Arc<ReadPlan>,
}

fn combine_chunks(cache: &mut Vec<Option<PlanChunk>>, pages: Vec<Arc<ReadPlan>>) -> (Arc<ReadPlan>, usize) {
    cache.resize_with(pages.len().div_ceil(PLAN_CHUNK_PAGES), || None);
    let mut reused = 0;
    let mut groups = Vec::with_capacity(cache.len());
    for (entry, group) in cache.iter_mut().zip(pages.chunks(PLAN_CHUNK_PAGES)) {
        if let Some(saved) = entry.as_ref()
            && saved.pages.len() == group.len() && saved.pages.iter().zip(group).all(|(a,b)| Arc::ptr_eq(a,b)) {
                groups.push(saved.combined.clone());
                reused += 1;
                continue;
        }
        let combined = ReadPlan::combine_pages(group.to_vec());
        groups.push(combined.clone());
        *entry = Some(PlanChunk { pages: group.to_vec(), combined });
    }
    // combine_pages is associative over consecutive runs: its union keeps
    // first-seen shadow order and never crosses an original write boundary.
    (ReadPlan::combine_pages(groups), reused)
}

struct PagePlan {
    key: DemandKey,
    views: u64,
    heap: u64,
    dependencies: std::collections::HashSet<u32>,
    plan: Arc<ReadPlan>,
}

impl PagePlan {
    fn get(&mut self, key: &DemandKey, views: &super::Views, heap: u64) -> Option<Arc<ReadPlan>> {
        if self.key != *key || self.heap != heap || !views.unchanged_since(self.views, &self.dependencies) {
            return None;
        }
        self.views = views.revision;
        Some(self.plan.clone())
    }
}

#[derive(Clone, PartialEq, Eq)]
struct DemandKey {
    fallback: u8,
    exact: std::collections::BTreeMap<u32, u8>,
}

impl DemandKey {
    fn new(fallback: u8, mut exact: std::collections::BTreeMap<u32, u8>, count: u32) -> Self {
        exact.retain(|&slot, class| {
            *class &= !fallback;
            slot < count && *class != 0
        });
        Self { fallback, exact }
    }

    fn read_only(&self) -> bool { self.fallback == 1 && self.exact.is_empty() }

    fn classes(&self, index: u32) -> u8 {
        self.fallback | self.exact.get(&index).copied().unwrap_or(0)
    }

    fn page(&self, from: u32, to: u32) -> Self {
        Self { fallback: self.fallback, exact: self.exact.range(from..to).map(|(&i,&c)| (i,c)).collect() }
    }
}

impl ReadRecords {
    #[cfg(test)]
    fn cached_plan(&self, view_revision: u64, heap_revision: u64) -> Option<Arc<ReadPlan>> {
        let (views, heap, plan) = self.plan.as_ref()?;
        (*views == view_revision && *heap == heap_revision).then(|| plan.clone())
    }

    fn demand_plan(&mut self, key: &DemandKey, views: &super::Views, heap: u64) -> Option<Arc<ReadPlan>> {
        let dependencies = self.dependencies.as_ref()?;
        let (revision, host, plan) = if key.read_only() {
            let (v, h, p) = self.plan.as_mut()?;
            (v, h, p)
        } else {
            let (_, v, h, p) = self.mixed.iter_mut().find(|(k, _, _, _)| k == key)?;
            (v, h, p)
        };
        if *host != heap || !views.unchanged_since(*revision, dependencies) { return None; }
        // Advance only after checking every relevant mutation under VIEWS.
        *revision = views.revision;
        Some(plan.clone())
    }

    fn publish(&mut self, key: DemandKey, views: u64, heap: u64, plan: Arc<ReadPlan>) {
        self.dependencies.get_or_insert_with(|| self.pages.iter().flatten()
            .map(|(_, record)| record[1]).collect());
        if key.read_only() {
            self.plan = Some((views, heap, plan));
        } else {
            self.mixed.retain(|(k, _, h, _)| k != &key && *h == heap);
            if self.mixed.len() >= 4 { self.mixed.remove(0); }
            self.mixed.push((key, views, heap, plan));
        }
    }

    fn invalidate(&mut self) {
        self.plan = None;
        self.mixed.clear();
        self.dependencies = None;
        self.page_plans.clear();
        self.chunks.clear();
    }

    /// Return the number of raw descriptor records actually examined.
    /// Unknown/dirty pages are scanned again; retain every tagged record,
    /// including ones whose generation is currently invalid.
    fn observe(&mut self, start: u64, count: u32, observed: Option<(u64, Vec<(usize, u64)>)>) -> u64 {
        let page = crate::watch::PAGE as u64;
        let first = start / page;
        let end = start + u64::from(count) * 32;
        let pages = (end.div_ceil(page) - first) as usize;
        let changed_heap = self.start != start || self.count != count || self.pages.len() != pages;
        let full = observed.is_none() || self.epoch == 0 || changed_heap;
        if changed_heap { self.invalidate(); }
        self.start = start;
        self.count = count;
        self.pages.resize_with(pages, Vec::new);
        self.page_plans.resize_with(pages, || None);
        let mut written = observed.as_ref().map(|(_, writes)| writes.as_slice()).unwrap_or(&[]).iter().peekable();
        let mut checked = 0u64;
        let mut scratch = Vec::new();
        for (i, records) in self.pages.iter_mut().enumerate() {
            let p = first + i as u64;
            while written.peek().is_some_and(|&&(n, _)| (n as u64) < p) { written.next(); }
            let dirty = written.peek().is_some_and(|&&(n, generation)| n as u64 == p && generation > self.epoch);
            if !full && !dirty { continue; }
            scratch.clear();
            // Heap records are 32-byte aligned on the cached path; for an
            // unaligned full-scan heap assign each record by its first byte.
            let from = ((p * page).max(start) - start).div_ceil(32).min(u64::from(count));
            let to = (((p + 1) * page).min(end) - start).div_ceil(32).min(u64::from(count));
            for index in from..to {
                // SAFETY: the caller checked this live heap's readable range.
                let record = unsafe { ((start + index * 32) as *const [u32; 3]).read_unaligned() };
                if record[0] == VIEW_TAG { scratch.push((index as u32, record)); }
            }
            // Exact comparison, including all currently dead tags. A dirty
            // page may have been rewritten with identical descriptor words.
            if *records != scratch {
                std::mem::swap(records, &mut scratch);
                self.plan = None;
                self.mixed.clear();
                self.dependencies = None;
                self.page_plans[i] = None;
                if let Some(chunk) = self.chunks.get_mut(i / PLAN_CHUNK_PAGES) { *chunk = None; }
            }
            checked += to - from;
        }
        // No journal coverage means the next observation must scan again.
        self.epoch = observed.map_or(0, |(epoch, _)| epoch);
        checked
    }
}

/// Immutable, exact SRV demand in the original first-seen shadow order.
/// Each consuming list replays state transitions and deferred refreshes.
pub(crate) type ShadowRanges = (Arc<crate::shadow::Shadow>, Vec<(u64, u64)>);
pub(crate) struct ReadPlan {
    pub(crate) ranges: Vec<ShadowRanges>,
    pub(crate) prepared: u64,
    pub(crate) ordered: Vec<PlanStep>,
}

pub(crate) enum PlanStep {
    Reads(Vec<ShadowRanges>),
    Write(BufferBinding),
}

impl ReadPlan {
    fn new(bindings: Vec<BufferBinding>) -> Arc<Self> {
        let mut by_shadow = std::collections::HashMap::new();
        let mut ranges: Vec<ShadowRanges> = Vec::new();
        let mut prepared = 0;
        for BufferBinding { shadow, address, len } in bindings {
            if len == 0 { continue; }
            prepared += 1;
            let Some(end) = address.checked_add(len) else { continue };
            let next = ranges.len();
            let index = *by_shadow.entry(Arc::as_ptr(&shadow) as usize).or_insert(next);
            if index == next { ranges.push((shadow, Vec::new())); }
            ranges[index].1.push((address, end));
        }
        for (_, intervals) in &mut ranges {
            intervals.sort_unstable();
            let mut n = 0;
            for i in 0..intervals.len() {
                let (start, end) = intervals[i];
                if n > 0 && start <= intervals[n - 1].1 {
                    intervals[n - 1].1 = intervals[n - 1].1.max(end);
                } else {
                    intervals[n] = (start, end);
                    n += 1;
                }
            }
            intervals.truncate(n);
        }
        Arc::new(Self { ranges, prepared, ordered: Vec::new() })
    }

    /// Union only consecutive read runs. An original UAV slot is an ordering
    /// boundary: keep its exact window and its position relative to all reads.
    fn from_slots(mut slots: Vec<(u32, bool, BufferBinding)>) -> Arc<Self> {
        slots.sort_unstable_by_key(|(slot, _, _)| *slot);
        if !slots.iter().any(|(_, write, b)| *write && b.len != 0) {
            return Self::new(slots.into_iter().map(|(_, _, b)| b).collect());
        }
        let mut ordered = Vec::new();
        let mut reads = Vec::new();
        let mut prepared = 0;
        let flush = |reads: &mut Vec<BufferBinding>, ordered: &mut Vec<PlanStep>| {
            if !reads.is_empty() {
                let plan = Self::new(std::mem::take(reads));
                ordered.push(PlanStep::Reads(Arc::try_unwrap(plan).ok().unwrap().ranges));
            }
        };
        for (_, write, binding) in slots {
            if binding.len == 0 { continue; }
            prepared += 1;
            if write {
                flush(&mut reads, &mut ordered);
                ordered.push(PlanStep::Write(binding));
            } else { reads.push(binding); }
        }
        flush(&mut reads, &mut ordered);
        Arc::new(Self { ranges: Vec::new(), prepared, ordered })
    }

    /// Combine already certified page plans without reconstructing hundreds
    /// of thousands of per-slot bindings. Reads may be unioned across pages
    /// only until the next original write slot.
    fn combine_pages(pages: Vec<Arc<Self>>) -> Arc<Self> {
        let prepared = pages.iter().map(|p| p.prepared).sum();
        let mut ordered = Vec::new();
        let mut reads = Vec::new();
        let mut by_shadow = std::collections::HashMap::new();
        let append = |ranges: &[ShadowRanges],
                      reads: &mut Vec<ShadowRanges>,
                      by_shadow: &mut std::collections::HashMap<usize,usize>| {
            for (shadow, intervals) in ranges {
                let next = reads.len();
                let index = *by_shadow.entry(Arc::as_ptr(shadow) as usize).or_insert(next);
                if index == next { reads.push((shadow.clone(), Vec::new())); }
                reads[index].1.extend_from_slice(intervals);
            }
        };
        let finish = |reads: &mut Vec<ShadowRanges>| {
            for (_, intervals) in reads {
                intervals.sort_unstable();
                let mut n = 0;
                for i in 0..intervals.len() {
                    let (a,b) = intervals[i];
                    if n > 0 && a <= intervals[n-1].1 {
                        intervals[n-1].1 = intervals[n-1].1.max(b);
                    } else { intervals[n] = (a,b); n += 1; }
                }
                intervals.truncate(n);
            }
        };
        for page in pages {
            append(&page.ranges, &mut reads, &mut by_shadow);
            for step in &page.ordered {
                match step {
                    PlanStep::Reads(ranges) => append(ranges, &mut reads, &mut by_shadow),
                    PlanStep::Write(binding) => {
                        if !reads.is_empty() {
                            finish(&mut reads);
                            ordered.push(PlanStep::Reads(std::mem::take(&mut reads)));
                            by_shadow.clear();
                        }
                        ordered.push(PlanStep::Write(binding.clone()));
                    }
                }
            }
        }
        finish(&mut reads);
        if ordered.is_empty() { return Arc::new(Self { ranges: reads, prepared, ordered }); }
        if !reads.is_empty() { ordered.push(PlanStep::Reads(reads)); }
        Arc::new(Self { ranges: Vec::new(), prepared, ordered })
    }
}

#[derive(Clone)]
pub(crate) struct BufferBinding {
    pub(crate) shadow: Arc<crate::shadow::Shadow>,
    pub(crate) address: u64,
    pub(crate) len: u64,
}

impl BufferBinding {
    fn window(shadow: Arc<crate::shadow::Shadow>, base: u64, first: u64, count: u32, element: u64) -> Option<Self> {
        let start = first.checked_mul(element)?;
        let offset = start.checked_sub(shadow.offset(base))?;
        let address = base.checked_add(offset)?;
        let len = u64::from(count).checked_mul(element)?.min(shadow.left(address));
        Some(Self { shadow, address, len })
    }
}

/// Cache identity includes the title's exact record, shader's descriptor
/// class, and the actual retained shadow selected for this materialization.
/// A new registered or loose allocation at the same title address must
/// never pick up the old host descriptor.
type SharedTableEntry = (u32, [u32; 3], victory_shader::ResourceType, u8, Option<(usize, u64, u64)>);
#[derive(PartialEq, Eq, Hash)]
struct SharedTableKey(Vec<SharedTableEntry>);
pub(crate) type SharedRecord = (u32, [u32; 3], Option<(victory_shader::ResourceType, u8)>);

enum PreparedShared {
    Srv(D3D12_SHADER_RESOURCE_VIEW_DESC, BufferBinding),
    Uav(D3D12_UNORDERED_ACCESS_VIEW_DESC, BufferBinding),
    Null,
}

impl PreparedShared {
    fn binding(&self) -> Option<&BufferBinding> {
        match self {
            Self::Srv(_, binding) | Self::Uav(_, binding) => Some(binding),
            Self::Null => None,
        }
    }
}

/// Strongly owned by every recorded list using this allocation. The cache
/// has no strong references, and this allocation is NOT in the old
/// reclaimable list lease. Retain the actual heap/allocator rather than
/// looking up a possibly recycled title heap address on drop.
pub(crate) struct SharedTable {
    pub(crate) base: u32,
    pub(crate) bindings: Vec<(bool, BufferBinding)>,
    offsets: Vec<u32>,
    count: u32,
    allocator: Arc<Mutex<(u32, Vec<bool>)>>,
    _heap: ID3D12DescriptorHeap,
}

impl Drop for SharedTable {
    fn drop(&mut self) {
        let mut allocator = self.allocator.lock().unwrap_or_else(|e| e.into_inner());
        for &offset in &self.offsets {
            allocator.1[(self.base - self.count + offset) as usize] = false;
        }
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!(
                "{:#x} past its end (slot {})",
                va - start - bytes,
                (va - start) / step
            )
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The host's limit for shader-visible sampler heaps across all resource
/// binding tiers (D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE).
const PORTABLE_SAMPLERS: u32 = 2_048;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            let actual = if count > PORTABLE_SAMPLERS {
                tracing::warn!(count, "host sampler heap capped at {PORTABLE_SAMPLERS}");
                PORTABLE_SAMPLERS
            } else {
                count
            };
            match make(actual) {
                Ok(heap) => (heap, actual, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count = actual, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (
                        make(PORTABLE_VIEWS).ok()?,
                        PORTABLE_VIEWS - MIN_DYNAMIC,
                        MIN_DYNAMIC,
                    )
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = if sampler {
            None
        } else {
            // Header dword 0 holds the actual count, dwords 1..count the
            // original per-slot strides, and count+1 a zeroed invalid-index
            // sentinel. A small descriptor heap should not allocate an
            // enormous maximum-sized upload table merely for bounds safety.
            let bytes = stride_table_bytes(count);
            match host.upload_buffer(bytes) {
                Some((buffer, data)) => {
                    // SAFETY: a persistently mapped upload resource of `bytes`.
                    let gpu = unsafe {
                        std::ptr::write_bytes(data, 0, bytes as usize);
                        (data as *mut u32).write_volatile(count);
                        buffer.GetGPUVirtualAddress()
                    };
                    crate::host::name(&buffer, || "view heap element strides".into());
                    tracing::info!(count, bytes, "right-sized original descriptor stride table");
                    Some((buffer, gpu, data as usize))
                }
                None => {
                    tracing::warn!("no GPU descriptor stride table");
                    None
                }
            }
        };
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(vec![0; count as usize]),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Arc::new(Mutex::new((0, vec![false; extra as usize]))),
            shared_tables: Mutex::new(std::collections::HashMap::new()),
            strides,
            buffer_slots: Mutex::new(std::collections::BTreeSet::new()),
            buffer_bindings: Mutex::new(std::collections::BTreeMap::new()),
            read_records: Mutex::new(ReadRecords::default()),
            read_revision: Mutex::new(HeapRevision::new(count)),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(
            ms = started.elapsed().as_millis(),
            "host descriptor heap made null"
        );
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: BLOCK,
                        ..Default::default()
                    })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device
                    .CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device
                    .CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// Header, `count` original slots, and one guaranteed zero sentinel.
/// Converted to bytes; never scaled by the maximum possible title heap.
fn stride_table_bytes(count: u32) -> u64 {
    (u64::from(count) + 2) * 4
}

/// First fitting base, searching from the cursor and then wrapping. Never
/// wrap a table itself or change any bits when a reservation fails.
fn reserve_sparse(taken: &mut [bool], next: usize, offsets: &[u32]) -> Option<usize> {
    let last = *offsets.last()? as usize;
    if offsets.windows(2).any(|w| w[0] >= w[1]) { return None; }
    let bases = taken.len().checked_sub(last)?;
    let start = next.min(bases);
    let base = (start..bases).chain(0..start)
        .find(|&base| offsets.iter().all(|&k| !taken[base + k as usize]))?;
    for &k in offsets { taken[base + k as usize] = true; }
    Some(base)
}

/// A free contiguous range may straddle the search cursor, but never the
/// end of the heap. The second search includes the n-1 slots after the
/// cursor so it cannot miss such a range.
fn reserve_contiguous(taken: &mut [bool], next: usize, n: usize) -> Option<usize> {
    if n == 0 || n > taken.len() { return None; }
    let next = next.min(taken.len());
    let find = |from, to| {
        let mut run = 0;
        for (at, &used) in taken.iter().enumerate().take(to).skip(from) {
            run = if used { 0 } else { run + 1 };
            if run == n { return Some(at + 1 - n); }
        }
        None
    };
    let first = find(next, taken.len())
        .or_else(|| find(0, next.saturating_add(n - 1).min(taken.len())))?;
    taken[first..first + n].fill(true);
    Some(first)
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        let Some(first) = reserve_contiguous(taken, *next as usize, n) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(n, "no free dynamic descriptor slots: a table is not made")
            });
            return None;
        };
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Reserve precisely the sorted offsets of a statically proven table.
    /// The base stays in the dynamic region, so the shader's original
    /// base+offset arithmetic is unchanged. Holes belong to no lease and
    /// may serve another table. The entire reservation is atomic.
    pub(crate) fn dynamic_sparse(&self, offsets: &[u32]) -> Option<u32> {
        if self.sampler { return None; }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let base = reserve_sparse(taken, *next as usize, offsets)?;
        *next = ((base + 1) % taken.len()) as u32;
        Some(self.count + base as u32)
    }

    /// Share a proven static SRV/UAV table across recorded lists. The
    /// caller supplies the exact title record snapshot and used layout.
    /// Only buffer views are admitted for now: their selected backing
    /// shadow and final descriptor are prepared together, never resolved
    /// a second time by address. Unsupported views take the old path.
    pub(crate) fn share_static_table(
        &self,
        records: &[SharedRecord],
    ) -> Option<Arc<SharedTable>> {
        use victory_shader::ResourceType;
        if self.sampler || records.is_empty() { return None; }
        let mut prepared = Vec::with_capacity(records.len());
        let mut key = Vec::with_capacity(records.len());
        for &(offset, record, kind) in records {
            let (kind, dwords) = kind?;
            if !matches!(kind, ResourceType::Srv | ResourceType::Uav) || dwords != 4 {
                return None;
            }
            let [tag, index, generation] = record;
            let view = (tag == VIEW_TAG).then(|| view_at(index, generation)).flatten();
            let descriptor = match view {
                Some(view) => prepare_shared_buffer(&view, kind)?,
                None => PreparedShared::Null,
            };
            let identity = descriptor.binding().map(|b| (Arc::as_ptr(&b.shadow) as usize, b.address, b.len));
            key.push((offset, record, kind, dwords, identity));
            prepared.push(descriptor);
        }
        let key = SharedTableKey(key);
        let mut cache = self.shared_tables.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(table) = cache.get(&key).and_then(std::sync::Weak::upgrade) {
            static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let hits = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if hits.is_power_of_two() {
                tracing::info!(hits, slots = records.len(), "shared immutable static descriptor table reused");
            }
            return Some(table);
        }
        // Weak entries cannot pin descriptor slots or shadow resources.
        // Bound the dead metadata even when the title streams unique tables.
        if cache.len() >= 1024 && cache.len().is_multiple_of(256) {
            cache.retain(|_, v| v.strong_count() != 0);
        }
        let offsets: Vec<_> = records.iter().map(|r| r.0).collect();
        let device = &host()?.device;
        let base = self.dynamic_sparse(&offsets)?;
        let mut table = SharedTable {
            base,
            bindings: Vec::with_capacity(records.len()),
            offsets,
            count: self.count,
            allocator: self.next.clone(),
            _heap: self.heap.clone(),
        };
        for ((offset, _, kind), descriptor) in records.iter().zip(prepared) {
            let index = base + *offset;
            let at = self.cpu(index);
            match descriptor {
                PreparedShared::Srv(desc, binding) => {
                    // SAFETY: the reserved slot and its exact retained buffer.
                    unsafe { device.CreateShaderResourceView(&binding.shadow.buffer, Some(&desc), at) };
                    self.keep(index, Some(binding.shadow.buffer.clone()));
                    table.bindings.push((false, binding));
                }
                PreparedShared::Uav(desc, binding) => {
                    // SAFETY: the reserved slot and its exact retained buffer.
                    unsafe { device.CreateUnorderedAccessView(&binding.shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at) };
                    self.keep(index, Some(binding.shadow.buffer.clone()));
                    table.bindings.push((true, binding));
                }
                PreparedShared::Null => {
                    let (kind, dwords) = kind.expect("validated layout");
                    if kind == ResourceType::Srv {
                        null_buffer_view(at);
                    } else {
                        debug_assert_eq!(dwords, 4);
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_UINT,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            ..Default::default()
                        };
                        // SAFETY: the reserved slot, a typed null buffer UAV.
                        unsafe { device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&desc), at) };
                    }
                    self.keep(index, None);
                }
            }
        }
        let table = Arc::new(table);
        cache.insert(key, Arc::downgrade(&table));
        Some(table)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// GPU address of the title descriptor's actual bytes-per-element table.
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Title slots currently holding a buffer view. Shader heap indices can
    /// be computed at runtime, so they need preparation beyond named tables.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        self.buffer_slots.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().collect()
    }

    fn note_buffer(&self, index: u32, view: Option<&View>, binding: Option<BufferBinding>) {
        // Dynamically allocated host slots are not part of the title's heap
        // and must not be re-mirrored using the title's descriptor address.
        // Their lifetime and bindings are managed by the table allocator.
        if index >= self.count {
            return;
        }
        let stride = view.and_then(view_stride);
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // SAFETY: this dword belongs to our persistently mapped table.
            unsafe { (*data as *mut u32).add(index as usize + 1).write_volatile(stride.unwrap_or(0)) };
        }
        let mut buffers = self.buffer_slots.lock().unwrap_or_else(|e| e.into_inner());
        if stride.is_some() { buffers.insert(index); } else { buffers.remove(&index); }
        let mut bindings = self.buffer_bindings.lock().unwrap_or_else(|e| e.into_inner());
        // Keep the actual materialized backing even if this descriptor's
        // stride cannot be represented in the bindless stride sidecar.
        // Static table users still need the exact resource and byte range.
        if let Some(binding) = binding {
            bindings.insert(index, binding);
        } else {
            bindings.remove(&index);
        }
    }

    /// A title buffer view discovered while constructing a table may not be
    /// materialized until draw time (`mirror(..., false)`). Keep the original
    /// heap slot discoverable even though its host raw view and stride are
    /// not published yet; `heap_buffers` will materialize and validate it
    /// before issuing a draw. This never advertises an invalid GPU stride.
    fn note_deferred_buffer(&self, index: u32, view: &View) {
        // View::is_buffer also deliberately classifies placed *textures*
        // as deferred views. They are not buffer descriptors and must not
        // enter the buffer-stride scan: doing so materializes the entire
        // streamed texture heap before any shader actually reads it.
        if index < self.count && view_stride(view).is_some() {
            self.buffer_slots.lock().unwrap_or_else(|e| e.into_inner()).insert(index);
        }
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) -> Option<BufferBinding> {
        let mut revision = (index < self.count).then(|| self.read_revision.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(revision) = revision.as_deref_mut() {
            revision.changed(index);
            self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] = 0;
        }
        let mut binding = None;
        let kept = make(view, self.cpu(index), &mut binding);
        // Publish stride metadata only for an actual host resource view.
        // A null/unsupported view must not advertise an accessible buffer.
        let binding = binding.filter(|_| kept.as_ref().is_some_and(|r| r.is_some()));
        self.note_buffer(index, kept.as_ref().and_then(|r| r.as_ref()).map(|_| view), binding.clone());
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        binding
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        let mut revision = (index < self.count).then(|| self.read_revision.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(revision) = revision.as_deref_mut() {
            revision.changed(index);
            self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] = 0;
        }
        self.make_null_inner(index);
    }

    // Caller owns the original-slot mutation lock, if applicable.
    fn make_null_inner(&self, index: u32) {
        self.note_buffer(index, None, None);
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Makes slot `index` a null descriptor matching the resource kind and
    /// buffer/texture class (`dwords == 4` -> buffer, `8` -> texture) that
    /// the translated shader declares for that slot. Avoids GPU device
    /// removal on NVIDIA when an unwritten slot is read as a UAV, CBV, or
    /// buffer SRV instead of a 2D texture SRV.
    pub(crate) fn make_null_for_used(
        &self,
        index: u32,
        used: Option<&victory_shader::UsedResource>,
    ) {
        use victory_shader::ResourceType;
        if self.sampler {
            self.make_null(index);
            return;
        }
        let Some(u) = used else {
            self.make_null(index);
            return;
        };
        let is_buffer = u.dwords == 4;
        // Sentinel keys have generation == 0 (real view keys start at generation 1).
        let sentinel: u64 = match (u.kind, is_buffer) {
            (ResourceType::Cbv, _) => 1,
            (ResourceType::Srv, true) => 2,
            (ResourceType::Srv, false) => 3,
            (ResourceType::Uav, true) => 4,
            (ResourceType::Uav, false) => 5,
            (ResourceType::Sampler | ResourceType::Other(_), _) => return,
        };
        let mut revision = (index < self.count).then(|| self.read_revision.lock().unwrap_or_else(|e| e.into_inner()));
        if (index as usize) < self.count as usize {
            let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            if made[index as usize] == sentinel {
                return;
            }
            made[index as usize] = sentinel;
            let revision = revision.as_deref_mut().expect("original slot revision");
            revision.changed(index);
        }
        self.note_buffer(index, None, None);
        self.keep(index, None);
        let at = self.cpu(index);
        let Some(host) = host() else { return };
        match (u.kind, is_buffer) {
            (ResourceType::Cbv, _) => {
                let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                // SAFETY: a slot of our heap; zero location makes a null CBV.
                unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
            }
            (ResourceType::Srv, true) => null_buffer_view(at),
            (ResourceType::Srv, false) => null_view(at),
            (ResourceType::Uav, true) | (ResourceType::Uav, false) => {
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_UINT,
                    ViewDimension: if is_buffer {
                        D3D12_UAV_DIMENSION_BUFFER
                    } else {
                        D3D12_UAV_DIMENSION_TEXTURE2D
                    },
                    ..Default::default()
                };
                // SAFETY: a slot of our heap; null resource is valid with a desc.
                unsafe {
                    host.device.CreateUnorderedAccessView(
                        None::<&ID3D12Resource>,
                        None::<&ID3D12Resource>,
                        Some(&desc),
                        at,
                    );
                }
            }
            (ResourceType::Sampler | ResourceType::Other(_), _) => {}
        }
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        let mut revision = (index < self.count).then(|| self.read_revision.lock().unwrap_or_else(|e| e.into_inner()));
        if let Some(revision) = revision.as_deref_mut() {
            revision.changed(index);
            self.made.lock().unwrap_or_else(|e| e.into_inner())[index as usize] = 0;
        }
        self.note_buffer(index, None, None);
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_inner(index, handle, for_draw, None).map(|(view, _)| view)
    }

    /// Named table entries use the same exact binding as bindless ones.
    /// Unlike mirror_buffer, also return textures and unmaterializable
    /// views, which have no buffer binding.
    pub(crate) fn mirror_table(&self, index: u32, handle: u64) -> Option<(View, Option<BufferBinding>)> {
        self.mirror_inner(index, handle, true, None)
    }

    /// Decode the same record snapshot used as a relocated table's key,
    /// including checking that its generation is still live.
    pub(crate) fn view_from_record([tag, i, generation]: [u32; 3]) -> Option<View> {
        (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten()
    }

    /// Validate the title record and select the exact retained buffer while
    /// the descriptor-generation lock is held. Do not resolve its address
    /// again: it may name a different registered or loose shadow by then.
    pub(crate) fn mirror_buffer(&self, index: u32, handle: u64, classes: u8) -> Option<(View, BufferBinding)> {
        let (view, binding) = self.mirror_inner(index, handle, true, Some(classes))?;
        Some((view, binding?))
    }

    /// The exact read-only buffer demand for an unconstrained heap index.
    /// Validate the title records and the generations under one set of locks,
    /// rather than cloning and looking up each View under three locks. Scan
    /// even unknown original slots: titles can copy descriptor records as
    /// bytes without calling the descriptor API. Never scan host-only slots.
    /// Changed/deferred descriptors still use the ordinary materializer.
    /// None means the full title range is not known readable; retain the
    /// ordinary per-slot path in that case.
    #[cfg(test)]
    pub(crate) fn read_buffer_batch(&self, start: u64) -> Option<(Arc<ReadPlan>, u64)> {
        self.buffer_batch(start, 1, Default::default())
    }

    pub(crate) fn buffer_batch(&self, start: u64, fallback: u8, exact: std::collections::BTreeMap<u32, u8>) -> Option<(Arc<ReadPlan>, u64)> {
        let demand = DemandKey::new(fallback, exact, self.count);
        if self.sampler || !readable_descriptor_heap(start, self.count) {
            // The next readable observation cannot assume continuity.
            let mut cache = self.read_records.lock().unwrap_or_else(|e| e.into_inner());
            cache.epoch = 0;
            cache.invalidate();
            return None;
        }
        enum Work {
            Ready(Arc<ReadPlan>),
            Slow(Vec<(u32, bool, BufferBinding)>, Vec<(u32, u8)>),
        }
        let mut work = Vec::new();
        let mut slow_count = 0;
        let mut reused_pages = 0;
        let checked;
        {
            let mut cache = self.read_records.lock().unwrap_or_else(|e| e.into_inner());
            let end = start.checked_add(u64::from(self.count) * 32)?;
            // Only the authoritative physical registry knows aliases and
            // mapping changes. The legacy guard for an unknown MEM_MAPPED
            // view cannot establish that another alias did not write it.
            // Private/unregistered allocations retain a full raw scan.
            let observed = if start.is_multiple_of(32)
                && crate::watch::known_physical(start as usize, end as usize)
            {
                crate::watch::writes(start as usize, end as usize, true)
            } else { None };
            if cache.start != start || cache.count != self.count {
                tracing::info!(start = format_args!("{start:#x}"), count = self.count,
                    physical_journal = observed.is_some(), "descriptor SRV read-plan observation");
            }
            checked = cache.observe(start, self.count, observed);
            let revision = self.read_revision.lock().unwrap_or_else(|e| e.into_inner());
            // Same lock order as mirror_inner -> view_at -> note_buffer.
            let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
            let views = super::VIEWS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(plan) = cache.demand_plan(&demand, &views, revision.generation) {
                static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let hits = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if hits.is_power_of_two() {
                    tracing::info!(hits, checked, prepared = plan.prepared, shadows = plan.ranges.len(),
                        ordered = plan.ordered.len(), fallback, "exact buffer demand plan reused");
                }
                return Some((plan, checked));
            }
            let slots = self.buffer_slots.lock().unwrap_or_else(|e| e.into_inner());
            let bindings = self.buffer_bindings.lock().unwrap_or_else(|e| e.into_inner());
            let page = crate::watch::PAGE as u64;
            let first = start / page;
            let ReadRecords { pages, page_plans, .. } = &mut *cache;
            for (p, (raw, saved)) in pages.iter().zip(page_plans.iter_mut()).enumerate() {
                let p = first + p as u64;
                let from = (((p * page).max(start) - start).div_ceil(32).min(u64::from(self.count))) as u32;
                let to = ((((p + 1) * page).min(end) - start).div_ceil(32).min(u64::from(self.count))) as u32;
                let page_key = demand.page(from, to);
                let epoch = revision.range(from, to);
                if let Some(plan) = saved.as_mut().and_then(|s| s.get(&page_key, &views, epoch)) {
                    reused_pages += 1;
                    work.push(Work::Ready(plan));
                    continue;
                }
                let mut ready = Vec::new();
                let mut slow = Vec::new();
                let mut known = slots.range(from..to).copied().peekable();
                let mut bound = bindings.range(from..to).peekable();
                let mut records = raw.iter().copied().peekable();
                // Include tracked slots whose raw tags disappeared. Never
                // replace a retained backing with an address-only lookup.
                while let Some(index) = [known.peek().copied(), bound.peek().map(|(i, _)| **i),
                    records.peek().map(|(i, _)| *i)].into_iter().flatten().min() {
                    let tracked = known.next_if_eq(&index).is_some();
                    let binding = bound.next_if(|(slot, _)| **slot == index).map(|(_, b)| b);
                    let [tag, i, generation] = records.next_if(|(slot, _)| *slot == index)
                        .map(|(_, record)| record).unwrap_or([0; 3]);
                    let key = u64::from(generation) << 32 | u64::from(i);
                    let matches = made[index as usize] == key;
                    if matches && !tracked && binding.is_none() { continue; }
                    let view = if tag == VIEW_TAG { views.get(i, generation) } else { None };
                    let classes = page_key.classes(index);
                    if classes == 0 { continue; }
                    match batch_action(view, matches, tracked || binding.is_some(), binding.is_some(), classes) {
                        ReadBatchAction::Cached => ready.push((index, view.is_some_and(view_writes), binding.expect("cached binding").clone())),
                        ReadBatchAction::Mirror => slow.push((index, classes)),
                        ReadBatchAction::Skip => {}
                    }
                }
                if slow.is_empty() {
                    let plan = ReadPlan::from_slots(ready);
                    *saved = Some(PagePlan { key: page_key, views: views.revision, heap: epoch,
                        dependencies: raw.iter().map(|(_, r)| r[1]).collect(), plan: plan.clone() });
                    work.push(Work::Ready(plan));
                } else {
                    *saved = None;
                    slow_count += slow.len();
                    work.push(Work::Slow(ready, slow));
                }
            }
            if slow_count == 0 {
                let (view_revision, heap_revision) = (views.revision, revision.generation);
                // ready retains every exact binding. Preserve its observed
                // versions, but release descriptor locks before sorting
                // large interval lists. A later mutation cannot hit them.
                drop(bindings);
                drop(slots);
                drop(views);
                drop(made);
                drop(revision);
                let page_count = work.len();
                let combine_started = std::time::Instant::now();
                let (plan, reused_chunks) = combine_chunks(&mut cache.chunks, work.into_iter().map(|w| match w {
                    Work::Ready(plan) => plan,
                    Work::Slow(..) => unreachable!(),
                }).collect());
                let combine_us = combine_started.elapsed().as_micros() as u64;
                cache.publish(demand, view_revision, heap_revision, plan.clone());
                static REBUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let builds = REBUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if builds.is_power_of_two() {
                    tracing::info!(builds, reused_pages, page_count, reused_chunks, combine_us, checked, prepared=plan.prepared,
                        "buffer demand page plans combined");
                }
                return Some((plan, checked));
            }
        }
        // Do not hold VIEWS or the made lock across descriptor creation.
        // Re-read changed records using the ordinary synchronized path.
        let plans = work.into_iter().map(|w| match w {
            Work::Ready(plan) => plan,
            Work::Slow(mut ready, slow) => {
                for (index, classes) in slow {
                    if let Some((view, binding)) = self.mirror_buffer(index, start + u64::from(index) * 32, classes) {
                        ready.push((index, view_writes(&view), binding));
                    }
                }
                ReadPlan::from_slots(ready)
            }
        }).collect();
        // This observation performed slow descriptor writes. Do not stamp
        // the pre-write observation with post-write versions. Restore exact
        // slot order across cached and newly materialized bindings.
        #[cfg(feature = "diagnostics")]
        {
            static SLOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let samples = SLOW.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if samples.is_power_of_two() {
                tracing::info!(samples, reused_pages, slow_count, checked, fallback,
                    "buffer demand slow page observation");
            }
        }
        Some((ReadPlan::combine_pages(plans), checked))
    }

    fn mirror_inner(&self, index: u32, handle: u64, for_draw: bool, classes: Option<u8>) -> Option<(View, Option<BufferBinding>)> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut revision = self.read_revision.lock().unwrap_or_else(|e| e.into_inner());
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Deferred slots have made == 0 but may still be tracked.
            let tracked = self.buffer_slots.lock().unwrap_or_else(|e| e.into_inner()).contains(&index);
            let bound = self.buffer_bindings.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&index);
            if tracked || bound || made[index as usize] != 0 {
                revision.changed(index);
            }
            self.note_buffer(index, None, None);
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made[index as usize] != 0 {
                made[index as usize] = 0;
                self.make_null_inner(index);
            }
            return None;
        };
        if let Some(classes) = classes {
            // A class-filtered query is not a descriptor write. Do not
            // erase the retained backing of the still materialized slot:
            // a later named table or raw record copy can use it again.
            view_stride(&view)?;
            let writes = matches!(view, View::Unordered(..) | View::Placed(true, ..));
            if classes & if writes { 2 } else { 1 } == 0 {
                return None;
            }
        }
        if made[index as usize] != key {
            revision.changed(index);
            let later = !for_draw && view.is_buffer();
            made[index as usize] = if later { 0 } else { key };
            let mut binding = None;
            let kept = if later {
                None
            } else {
                make(&view, self.cpu(index), &mut binding)
            };
            let binding = binding.filter(|_| kept.as_ref().is_some_and(|r| r.is_some()));
            self.note_buffer(index, kept.as_ref().and_then(|r| r.as_ref()).map(|_| &view), binding);
            if later {
                self.note_deferred_buffer(index, &view);
            }
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            self.keep(index, kept.flatten());
        }
        let binding = for_draw.then(|| self.buffer_bindings.lock().unwrap_or_else(|e| e.into_inner()).get(&index).cloned()).flatten();
        Some((view, binding))
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self
            .made
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(index as usize)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// Bytes per index in the original console buffer descriptor. Raw buffer
/// indices are byte offsets, unlike the four-byte elements of a host raw view.
/// Unknown formats return None rather than a guessed stride.
#[allow(dead_code)] // Consumed by the forthcoming GPU stride sidecar.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER && d[1] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: the descriptor dimension selects the buffer union member.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (
                b.StructureByteStride,
                b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0,
                d.Format,
            )
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (
                b.StructureByteStride,
                b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0,
                d.Format,
            )
        }
    };
    if stride > 0 {
        Some(stride)
    } else if raw {
        Some(1)
    } else {
        let (bytes, _) = crate::resource::element(format.0 as u32)?;
        u32::try_from(bytes).ok().filter(|&n| n != 0)
    }
}

/// No speculative reads of reserved/no-access/guarded pages in a placed heap.
/// The ordinary descriptor path still handles heaps without a fully readable
/// original range. The title must keep descriptor memory mapped during use.
fn readable_descriptor_heap(start: u64, count: u32) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE,
        PAGE_WRITECOPY, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, VirtualQuery,
    };
    let Ok(mut at) = usize::try_from(start) else { return false };
    let Some(end) = at.checked_add(count as usize * 32) else { return false };
    while at < end {
        // SAFETY: MEMORY_BASIC_INFORMATION is a plain Windows output structure valid when zero initialized.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: VirtualQuery only queries the address and writes into the supplied correctly sized structure.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
            || info.State != MEM_COMMIT
            || info.Protect & PAGE_GUARD != 0
            || !matches!(info.Protect & 0xff, PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY
                | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)
        {
            return false;
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else { return false };
        if next <= at { return false; }
        at = next;
    }
    true
}

#[derive(Debug, PartialEq, Eq)]
enum ReadBatchAction { Skip, Cached, Mirror }

fn view_writes(view: &View) -> bool {
    matches!(view, View::Unordered(..) | View::Placed(true, ..))
}

#[cfg(test)]
fn read_batch_action(view: Option<&View>, matches: bool, tracked: bool, binding: bool) -> ReadBatchAction {
    batch_action(view, matches, tracked, binding, 1)
}

fn batch_action(view: Option<&View>, matches: bool, tracked: bool, binding: bool, classes: u8) -> ReadBatchAction {
    use ReadBatchAction::*;
    let Some(view) = view else { return if tracked { Mirror } else { Skip } };
    // A live view outside the requested classes is not a candidate.
    if classes & if view_writes(view) { 2 } else { 1 } == 0 {
        return Skip;
    }
    if matches && binding {
        // A matching generation with a retained binding was already validated
        // as a buffer when it was materialized.
        return Cached;
    }
    if view_stride(view).is_none() {
        // mirror_inner's class-filtered path returns immediately for this
        // live non-buffer view, even if a previous buffer occupied the slot.
        // Scheduling it as slow work cannot change the binding and prevents
        // the entire page from ever becoming cacheable. Named-table/texture
        // mirroring still owns actual descriptor changes. Raw record and
        // registry dependencies invalidate this skip if the slot changes.
        return Skip;
    }
    // A materialized null/unsupported view has no binding. Keep the same
    // behavior as mirror_inner; deferred buffers have made == 0.
    if matches { Skip } else { Mirror }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
#[cfg(feature = "diagnostics")]
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
#[cfg(feature = "diagnostics")]
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Re-express a structured buffer slice as raw dwords without changing its
/// byte range. The host raw view requires 16-byte start alignment.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 { return None; }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count).checked_mul(u64::from(stride))?;
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

fn buffer_srv(
    address: u64,
    len: u64,
    desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    binding: &mut Option<BufferBinding>,
) -> Option<Kept> {
    let (shadow, desc, exact) = prepare_buffer_srv(address, len, desc)?;
    *binding = Some(exact?);
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe { host.device.CreateShaderResourceView(&shadow.buffer, Some(&desc), at) };
    Some(Some(shadow.buffer.clone()))
}

fn prepare_buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
) -> Option<(Arc<crate::shadow::Shadow>, D3D12_SHADER_RESOURCE_VIEW_DESC, Option<BufferBinding>)> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // A raw DXIL access into a structured NVIDIA view reads zeros. Make
    // the view raw for the exact same title bytes when alignment permits.
    // SAFETY: Callers supply buffer SRV descriptions, so the Buffer union member is initialized.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement,
            structured.NumElements, structured.StructureByteStride))
        .flatten();
    if desc.Format == DXGI_FORMAT_UNKNOWN && structured.StructureByteStride != 0
        && window.is_none()
    {
        // The translated buffer instructions require raw bytes. A structured
        // view would return incorrect data (or fault) on NVIDIA; refuse it.
        return None;
    }
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 || (window.is_none() && offset % element != 0) {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    if window.is_none() { buffer.FirstElement += offset / element; }
    // A view past its host copy's end is an invalid call (the debug layer
    // removes the device; drivers may fault): none past the end, the rest
    // cut to fit.
    // SAFETY: a live buffer.
    let capacity = unsafe { shadow.buffer.GetDesc() }.Width / element;
    if buffer.FirstElement >= capacity {
        let first = buffer.FirstElement;
        once!(first, capacity, "buffer view past the end of its host copy");
        return None;
    }
    if buffer.FirstElement + u64::from(buffer.NumElements) > capacity {
        buffer.NumElements = (capacity - buffer.FirstElement) as u32;
    }
    // A view of no elements is an invalid call too (the title's own record
    // count can be 0): null, which reads 0 and drops writes.
    if buffer.NumElements == 0 {
        let first = buffer.FirstElement;
        once!(first, "buffer view of no elements");
        return None;
    }
    let binding = BufferBinding::window(shadow.clone(), address, buffer.FirstElement, buffer.NumElements, element);
    Some((shadow, desc, binding))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    binding: &mut Option<BufferBinding>,
) -> Option<Kept> {
    let (shadow, desc, exact) = prepare_buffer_uav(address, len, desc)?;
    *binding = Some(exact?);
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device.CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

fn prepare_buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
) -> Option<(Arc<crate::shadow::Shadow>, D3D12_UNORDERED_ACCESS_VIEW_DESC, Option<BufferBinding>)> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement,
            buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if desc.Format == DXGI_FORMAT_UNKNOWN && buffer.StructureByteStride != 0
        && window.is_none()
    {
        return None;
    }
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    let element = if window.is_some() { 4 } else { element };
    if element == 0 || (window.is_none() && offset % element != 0) {
        once!(
            offset,
            element,
            "unordered buffer view not on an element of its host copy"
        );
        return None;
    }
    if window.is_none() { buffer.FirstElement += offset / element; }
    // A view past its host copy's end is an invalid call (the debug layer
    // removes the device; drivers may fault): none past the end, the rest
    // cut to fit.
    // SAFETY: a live buffer.
    let capacity = unsafe { shadow.buffer.GetDesc() }.Width / element;
    if buffer.FirstElement >= capacity {
        let first = buffer.FirstElement;
        once!(first, capacity, "buffer view past the end of its host copy");
        return None;
    }
    if buffer.FirstElement + u64::from(buffer.NumElements) > capacity {
        buffer.NumElements = (capacity - buffer.FirstElement) as u32;
    }
    // A view of no elements is an invalid call too (the title's own record
    // count can be 0): null, which reads 0 and drops writes.
    if buffer.NumElements == 0 {
        let first = buffer.FirstElement;
        once!(first, "buffer view of no elements");
        return None;
    }
    buffer.CounterOffsetInBytes = 0;
    let binding = BufferBinding::window(shadow.clone(), address, buffer.FirstElement, buffer.NumElements, element);
    Some((shadow, desc, binding))
}

/// Use the same normalization as ordinary descriptor materialization.
/// Refuse texture views, counters, and mismatched resource classes instead
/// of guessing a binding or using per-list constant scratch.
fn prepare_shared_buffer(view: &View, kind: victory_shader::ResourceType) -> Option<PreparedShared> {
    use victory_shader::ResourceType;
    view_stride(view)?;
    let (address, len, words): (u64, u64, &[u32; 10]) = match (view, kind) {
        (View::Shader(resource, Some(words)), ResourceType::Srv)
        | (View::Unordered(resource, 0, Some(words)), ResourceType::Uav) => {
            let (data, len) = crate::resource::buffer_bytes(*resource as crate::object::This)?;
            (data as u64, len, words)
        }
        (View::Placed(false, resource, desc), ResourceType::Srv)
        | (View::Placed(true, resource, desc), ResourceType::Uav) if resource[0] == 1 => {
            (u64::from(desc[10]) | u64::from(desc[11]) << 32,
                u64::from(resource[4]) | u64::from(resource[5]) << 32,
                desc[..10].try_into().ok()?)
        }
        _ => return None,
    };
    match kind {
        ResourceType::Srv => {
            let (_, desc, binding) = prepare_buffer_srv(address, len, srv_desc(words))?;
            Some(PreparedShared::Srv(desc, binding?))
        }
        ResourceType::Uav => {
            let (_, desc, binding) = prepare_buffer_uav(address, len, uav_desc(words))?;
            Some(PreparedShared::Uav(desc, binding?))
        }
        _ => None,
    }
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(d: &D3D12_RESOURCE_DESC, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    // Packed 16-bit color textures stored as R16_TYPELESS (53) on NVIDIA.
    if own == 53 && matches!(view, 0 | 86 | 115) {
        if let Some(format) = format {
            format.0 = if unordered { 57 } else { 56 }; // R16_UINT / R16_UNORM
            return true;
        }
        return false;
    }
    if view == 0 {
        return !typeless;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Stand-in UAV scratch texture for a depth-stencil texture (PC strips
/// `ALLOW_UNORDERED_ACCESS` from depth targets; NVIDIA removes the device
/// if the shader then accesses a null descriptor in that slot).
fn depth_scratch(
    device: &ID3D12Device,
    texture: &ID3D12Resource,
    desc: Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    use windows::core::Interface;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0 {
        return None;
    }
    let (res_fmt, uav_fmt) = match d.Format.0 as u32 {
        53 | 55 => (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UINT),
        _ => (DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT),
    };
    static SCRATCH: std::sync::OnceLock<Mutex<std::collections::HashMap<usize, ID3D12Resource>>> =
        std::sync::OnceLock::new();
    let key = texture.as_raw() as usize;
    let scratch = {
        let mut map = SCRATCH
            .get_or_init(|| Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = map.get(&key) {
            existing.clone()
        } else {
            let mut sd = d;
            sd.Format = res_fmt;
            sd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource description.
            let ok = unsafe {
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &sd,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                )
            };
            if let Err(e) = ok {
                once!(error = %e, format = d.Format.0, "depth UAV scratch creation failed");
                return None;
            }
            let res = made?;
            crate::host::name(&res, || {
                format!("depth UAV scratch {}x{} f{}", d.Width, d.Height, res_fmt.0)
            });
            once!(
                format = d.Format.0,
                width = d.Width,
                height = d.Height,
                "depth UAV: scratch stand-in"
            );
            map.insert(key, res.clone());
            res
        }
    };
    let mut uav = desc.unwrap_or_else(|| D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: uav_fmt,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    });
    uav.Format = uav_fmt;
    // Non-planar scratch formats (R16_TYPELESS / R32_TYPELESS) only allow PlaneSlice 0.
    match uav.ViewDimension {
        D3D12_UAV_DIMENSION_TEXTURE2D => uav.Anonymous.Texture2D.PlaneSlice = 0,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY => uav.Anonymous.Texture2DArray.PlaneSlice = 0,
        _ => {}
    }
    // SAFETY: a live scratch texture and a slot of our heap.
    unsafe {
        device.CreateUnorderedAccessView(&scratch, None::<&ID3D12Resource>, Some(&uav), at);
    }
    Some(Some(scratch))
}

/// Route exactly the same descriptor shape/type that materialized the
/// coherent D16 twin. Unsupported legacy scratch views are not routed.
pub(crate) fn depth_uav_use(list: &crate::host::HostList, resource: crate::object::This,
    words: Option<&[u32; 10]>) -> bool {
    words.is_some_and(|w| crate::twin::uav_use(list, resource, &uav_desc(w)))
}

/// A shader view of a typeless depth texture names the readable aspect.
/// For a 2D view D3D12 also requires the plane index to match that format.
fn texture_srv_desc(
    resource: &D3D12_RESOURCE_DESC,
    words: &[u32; 10],
) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut view = srv_desc(words);
    if !resource
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
    {
        return view;
    }
    let (dsv, depth, stencil) = match resource.Format {
        DXGI_FORMAT_R32G8X24_TYPELESS => (
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            Some(DXGI_FORMAT_X32_TYPELESS_G8X24_UINT),
        ),
        DXGI_FORMAT_R24G8_TYPELESS => (
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            Some(DXGI_FORMAT_X24_TYPELESS_G8_UINT),
        ),
        DXGI_FORMAT_R32_TYPELESS => (DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, None),
        DXGI_FORMAT_R16_TYPELESS => (DXGI_FORMAT_D16_UNORM, DXGI_FORMAT_R16_UNORM, None),
        _ => return view,
    };
    let plane = if Some(view.Format) == stencil {
        1
    } else if [DXGI_FORMAT_UNKNOWN, resource.Format, dsv, depth].contains(&view.Format) {
        view.Format = depth;
        0
    } else {
        return view;
    };
    match view.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => view.Anonymous.Texture2D.PlaneSlice = plane,
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => view.Anonymous.Texture2DArray.PlaneSlice = plane,
        _ => {}
    }
    view
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE, binding: &mut Option<BufferBinding>) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // SAFETY: a live host texture.
                let info = unsafe { texture.GetDesc() };
                let mut desc = desc.as_ref().map(|words| texture_srv_desc(&info, words));
                if !fits(&info, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, binding)
            } else {
                None
            }
        }
        View::Unordered(resource, counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, binding)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            // SAFETY: a live host texture.
            let info = unsafe { texture.GetDesc() };
            if !fits(&info, desc.as_mut().map(|d| &mut d.Format), true) {
                if *counter == 0
                    && let Some(twin) = desc.as_ref().and_then(|d| crate::twin::uav(resource, d, at))
                {
                    return Some(Some(twin));
                }
                return depth_scratch(device, texture, desc, at);
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at, binding);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, binding)
            } else {
                buffer_srv(address, len, srv_desc(words), at, binding)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_descriptor_ranges_fit_after_dense_capacity_would_run_out() {
        let mut taken = vec![false; 65536];
        let mut next = 0;
        let mut leases = Vec::new();
        // 258 contiguous slots per draw fit only 254 tables. The sparse
        // pattern holds the two named descriptors at their original offsets.
        for _ in 0..1024 {
            let base = reserve_sparse(&mut taken, next, &[0, 257]).unwrap();
            leases.push(base);
            next = base + 1;
        }
        assert_eq!(taken.iter().filter(|&&x| x).count(), 2048);
        for &base in &leases {
            assert!(taken[base] && taken[base + 257]);
        }
        // Release precisely those leases, never their overlapping holes.
        for base in leases {
            assert!(taken[base] && taken[base + 257]);
            taken[base] = false;
            taken[base + 257] = false;
        }
        assert!(taken.iter().all(|&x| !x));
    }

    #[test]
    fn sparse_and_contiguous_descriptor_leases_do_not_overlap() {
        let mut taken = [false; 8];
        assert_eq!(reserve_sparse(&mut taken, 0, &[0, 7]), Some(0));
        assert_eq!(reserve_contiguous(&mut taken, 4, 6), Some(1));
        assert!(taken.iter().all(|&x| x));
        taken[0] = false;
        taken[7] = false;
        let snapshot = taken;
        assert_eq!(reserve_contiguous(&mut taken, 7, 2), None);
        assert_eq!(reserve_sparse(&mut taken, 7, &[0, 1]), None);
        assert_eq!(reserve_sparse(&mut taken, 0, &[7, 0]), None);
        assert_eq!(reserve_sparse(&mut taken, 0, &[0, 8]), None);
        assert_eq!(taken, snapshot);
        assert_eq!(reserve_sparse(&mut taken, 7, &[0, 7]), Some(0));
    }

    #[test]
    fn stride_table_bounds_match_actual_heap_including_null_sentinel() {
        let max = victory_shader::emit::SLOT_STRIDES_COUNT;
        assert_eq!(stride_table_bytes(0), 8);
        assert_eq!(stride_table_bytes(1), 12);
        assert_eq!(stride_table_bytes(4096), 16392);
        assert_eq!(stride_table_bytes(max), (u64::from(max) + 2) * 4);
        assert!(stride_table_bytes(max) < ((1 << 21) + 1) * 4);
        for count in [0, 1, 3, 257, 4096, max] {
            let bytes = stride_table_bytes(count);
            // The shader indexes dword slot+1 for valid slots and count+1
            // for every invalid slot, without crossing this allocation.
            for slot in [0, count.saturating_sub(1), count, count.saturating_add(1), u32::MAX] {
                let valid = slot < count && slot < max;
                let selected = if valid { slot } else { count };
                let offset = (u64::from(selected) + 1) * 4;
                assert!(offset + 4 <= bytes, "count={count} slot={slot}");
                if !valid { assert_eq!(offset + 4, bytes); }
            }
        }
    }

    #[test]
    fn raw_window_preserves_exact_structured_byte_range() {
        assert_eq!(raw_window(0, 0, 8, 12), Some((0, 24)));
        assert_eq!(raw_window(16, 2, 3, 24), Some((16, 18)));
        assert_eq!(raw_window(0, 1, 1, 12), None); // 12-byte start misaligned
        assert_eq!(raw_window(2, 0, 4, 16), None); // unaligned offset
        assert_eq!(raw_window(0, 0, 3, 6), None); // incomplete dword
        assert_eq!(raw_window(0, 0, 1, 0), None);
        assert_eq!(raw_window(0, u64::MAX, 10, 16), None);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn depth_srv_uses_the_requested_aspect() {
        for (own, request, expected, plane) in [
            (19, 20, 21, 0),
            (19, 22, 22, 1),
            (44, 45, 46, 0),
            (44, 47, 47, 1),
            (44, 120, 47, 1),
            (44, 119, 46, 0),
            (39, 40, 41, 0),
            (53, 55, 56, 0),
            (19, 0, 21, 0),
            (44, 44, 46, 0),
        ] {
            let resource = D3D12_RESOURCE_DESC {
                Format: DXGI_FORMAT(own),
                Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                ..Default::default()
            };
            for dim in [
                D3D12_SRV_DIMENSION_TEXTURE2D,
                D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            ] {
                let mut words = [0; 10];
                words[0] = request;
                words[1] = dim.0 as u32;
                words[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                words[4] = 2; // MostDetailedMip
                words[5] = 3; // MipLevels
                if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    words[6] = 1; // FirstArraySlice
                    words[7] = 2; // ArraySize
                }
                let plane_word = if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                    8
                } else {
                    6
                };
                words[plane_word] = 1 - plane;
                let view = texture_srv_desc(&resource, &words);
                assert_eq!(view.Format.0, expected);
                // SAFETY: read the member selected by the input view dimension.
                unsafe {
                    if dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY {
                        let a = view.Anonymous.Texture2DArray;
                        assert_eq!(
                            (
                                a.MostDetailedMip,
                                a.MipLevels,
                                a.FirstArraySlice,
                                a.ArraySize,
                                a.PlaneSlice
                            ),
                            (2, 3, 1, 2, plane)
                        );
                    } else {
                        let a = view.Anonymous.Texture2D;
                        assert_eq!(
                            (a.MostDetailedMip, a.MipLevels, a.PlaneSlice),
                            (2, 3, plane)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn view_stride_uses_descriptor_metadata() {
        for unordered in [false, true] {
            let (stride_word, flags_word) = if unordered { (5, 8) } else { (7, 8) };
            let view = |words: [u32; 10]| {
                if unordered {
                    View::Unordered(0, 0, Some(words))
                } else {
                    View::Shader(0, Some(words))
                }
            };
            let mut words = [0u32; 10];
            words[1] = 1;
            for stride in [12, 48, 2044] {
                words[stride_word] = stride;
                assert_eq!(view_stride(&view(words)), Some(stride));
                let mut resource = [0; 14];
                resource[0] = 1;
                let mut desc = [0; 16];
                desc[..10].copy_from_slice(&words);
                assert_eq!(
                    view_stride(&View::Placed(unordered, resource, desc)),
                    Some(stride)
                );
            }
            words[stride_word] = 0;
            words[flags_word] = 1;
            assert_eq!(view_stride(&view(words)), Some(1));
            words[flags_word] = 0;
            assert_eq!(view_stride(&view(words)), None);
            words[1] = 4;
            assert_eq!(view_stride(&view(words)), None);
        }
        assert_eq!(view_stride(&View::Shader(0, None)), None);
        assert_eq!(view_stride(&View::Constant(0, 16)), None);
    }

    #[test]
    fn heap_buffer_binding_keeps_the_descriptor_shadow_and_exact_window() {
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let base = 0x7d00_0000_0000_u64;
        let outer = usize::MAX - 10;
        let inner = usize::MAX - 11;
        crate::shadow::register(outer, base, 0x10000);
        let original = crate::shadow::find(base, 0x10000).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut desc = [0u32; 16];
        desc[1] = 1;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[4] = 32; // FirstElement: 32 * 16 = 512
        desc[6] = 8;
        desc[7] = 16;
        desc[10] = base as u32;
        desc[11] = (base >> 32) as u32;
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        super::super::record(handle, View::Placed(false, resource, desc));
        let (_, first) = heap.mirror_buffer(0, handle, 1).unwrap();
        assert!(Arc::ptr_eq(&first.shadow, &original));
        assert_eq!((first.address, first.len), (base + 512, 128));
        // A later address-only lookup resolves to a newly registered nested
        // allocation, which is NOT the resource the host descriptor views.
        crate::shadow::register(inner, base + 512, 128);
        let nested = crate::shadow::find(base + 512, 128).unwrap();
        assert!(!Arc::ptr_eq(&original, &nested));
        let (_, cached) = heap.mirror_buffer(0, handle, 1).unwrap();
        assert!(Arc::ptr_eq(&cached.shadow, &original));
        // Named table preparation must use the exact same retained backing,
        // not look up the view's subrange in the now-overlapping registry.
        let (view, named) = heap.mirror_table(0, handle).unwrap();
        let named = named.unwrap();
        assert!(Arc::ptr_eq(&named.shadow, &original));
        assert_eq!((named.address, named.len), (base + 512, 128));
        // A relocated slot returns the binding selected when it was made,
        // even though it has no entry in the original heap's stride table.
        let dynamic = heap.dynamic(1).unwrap();
        let relocated = heap.make_at(dynamic, &view).unwrap();
        assert!(Arc::ptr_eq(&relocated.shadow, &original));
        assert_eq!((relocated.address, relocated.len), (base + 512, 128));
        assert!(heap.buffer_bindings.lock().unwrap().get(&dynamic).is_none());
        heap.release(dynamic, 1);
        assert!(heap.mirror_buffer(0, handle, 2).is_none());
        // The corresponding UAV must declare writes on the outer shadow
        // too, not on the independently resolved nested address.
        let mut uav = [0u32; 16];
        uav[1] = 1;
        uav[2] = 32;
        uav[4] = 8;
        uav[5] = 16;
        uav[10] = base as u32;
        uav[11] = (base >> 32) as u32;
        super::super::record(handle, View::Placed(true, resource, uav));
        let (view, named) = heap.mirror_table(0, handle).unwrap();
        assert!(matches!(view, View::Placed(true, ..)));
        let named = named.unwrap();
        assert!(Arc::ptr_eq(&named.shadow, &original));
        assert_eq!((named.address, named.len), (base + 512, 128));
        let dynamic = heap.dynamic(1).unwrap();
        let relocated = heap.make_at(dynamic, &view).unwrap();
        assert!(Arc::ptr_eq(&relocated.shadow, &original));
        assert_eq!((relocated.address, relocated.len), (base + 512, 128));
        heap.release(dynamic, 1);
        assert!(heap.mirror_buffer(0, handle, 1).is_none());
        // Overwriting the slot invalidates the retained exact binding.
        slot.fill(0);
        assert!(heap.mirror_buffer(0, handle, 1).is_none());
        assert!(heap.mirror_table(0, handle).is_none());
        assert!(heap.buffer_bindings.lock().unwrap().is_empty());
        assert!(heap.buffer_slots().is_empty());
        crate::shadow::unregister(inner, base + 512);
        crate::shadow::unregister(outer, base);
    }

    #[test]
    fn read_buffer_batch_matches_live_views_and_discovers_raw_copies() {
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let base = 0x7d01_0000_0000_u64;
        let owner = usize::MAX - 20;
        crate::shadow::register(owner, base, 0x10000);
        let original = crate::shadow::find(base, 0x10000).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut desc = [0u32; 16];
        desc[1] = 1;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[4] = 32;
        desc[6] = 8;
        desc[7] = 16;
        desc[10] = base as u32;
        desc[11] = (base >> 32) as u32;
        let mut slots = [[0u32; 8]; 4];
        let start = slots.as_mut_ptr() as u64;
        super::super::record(start, View::Placed(false, resource, desc));
        let (_, old) = heap.mirror_buffer(0, start, 1).unwrap();
        assert!(Arc::ptr_eq(&old.shadow, &original));
        // The title copies a descriptor directly, without a mirror call.
        slots[1] = slots[0];
        super::super::record(start + 64, View::Placed(true, resource, desc));
        let mut texture = resource;
        texture[0] = 3;
        super::super::record(start + 96, View::Placed(false, texture, desc));
        assert_eq!(heap.buffer_slots(), vec![0]);
        let (batch, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert_eq!(batch.prepared, 2);
        assert_eq!(batch.ranges.len(), 1);
        assert!(Arc::ptr_eq(&batch.ranges[0].0, &original));
        assert_eq!(batch.ranges[0].1, [(base + 512, base + 640)]);
        assert_eq!(heap.buffer_slots(), vec![0, 1]);
        // After slow materialization, the next observation can publish.
        let (stable, _) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(stable.prepared, 2);
        // Untracked private heap still scans every record. With unchanged
        // versions the exact same immutable union can be returned.
        let (again, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert_eq!(again.prepared, 2);
        assert_eq!(again.ranges[0].1, stable.ranges[0].1);
        // Other tests can legitimately mutate the global registry between
        // calls. Assert the pointer fast hit separately using an empty
        // cache with fixed revisions below.
        // Same i/generation, wrong tag: must not reuse the cached binding.
        slots[0][0] = 0;
        let (batch, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert_eq!(batch.prepared, 1);
        assert!(!Arc::ptr_eq(&batch, &again));
        assert!(Arc::ptr_eq(&batch.ranges[0].0, &original));
        assert_eq!(heap.buffer_slots(), vec![1]);
        // The record changes to another valid window. Its descriptor and
        // retained binding must be remade, not taken from the prior scan.
        desc[4] = 64;
        super::super::record(start + 32, View::Placed(false, resource, desc));
        let (batch, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert_eq!(batch.prepared, 1);
        assert_eq!(batch.ranges[0].1, [(base + 1024, base + 1152)]);
        let (stable, _) = heap.read_buffer_batch(start).unwrap();
        // Host mutation without a title record mutation: rematerialize
        // against the new overlapping registered allocation.
        crate::shadow::register(owner + 1, base, 0x10000);
        let replacement = crate::shadow::find(base, 0x10000).unwrap();
        assert!(!Arc::ptr_eq(&original, &replacement));
        heap.make_null(1);
        let (changed, _) = heap.read_buffer_batch(start).unwrap();
        assert!(!Arc::ptr_eq(&changed, &stable));
        assert!(Arc::ptr_eq(&changed.ranges[0].0, &replacement));
        assert!(Arc::ptr_eq(&stable.ranges[0].0, &original));
        // A raw copy into a previously unknown original slot changes the
        // plan even if the added range is already in its union.
        let (stable, _) = heap.read_buffer_batch(start).unwrap();
        slots[0] = slots[1];
        std::hint::black_box(&slots);
        let (copied, _) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(copied.prepared, 2);
        assert!(!Arc::ptr_eq(&stable, &copied));
        // Same host heap, different observed title heap address.
        let mut empty = [[0u32; 8]; 4];
        let (other, _) = heap.read_buffer_batch(empty.as_mut_ptr() as u64).unwrap();
        assert_eq!(other.prepared, 0);
        assert!(other.ranges.is_empty());
        crate::shadow::unregister(owner + 1, base);
        crate::shadow::unregister(owner, base);
    }

    #[test]
    fn mixed_buffer_batch_preserves_slot_order_and_raw_copies() {
        let heap = HostHeap::new(false, 8).expect("hardware heap");
        let base = 0x7d04_0000_0000_u64;
        let owner = usize::MAX - 84;
        crate::shadow::register(owner, base, 0x10000);
        let shadow = crate::shadow::find(base, 0x10000).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut srv = [0u32; 16];
        srv[1] = 1;
        srv[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv[4] = 32;
        srv[6] = 8;
        srv[7] = 16;
        srv[10] = base as u32;
        srv[11] = (base >> 32) as u32;
        let mut uav = [0u32; 16];
        uav[1] = 1;
        uav[2] = 64;
        uav[4] = 8;
        uav[5] = 16;
        uav[10] = base as u32;
        uav[11] = (base >> 32) as u32;
        let mut slots = [[0u32; 8]; 8];
        let start = slots.as_mut_ptr() as u64;
        super::super::record(start, View::Placed(false, resource, srv));
        super::super::record(start + 32, View::Placed(true, resource, uav));
        slots[2] = slots[0];
        slots[3] = slots[1];
        slots[4] = slots[0];
        // Materialize a late slot first to exercise slow/cached interleaving.
        heap.mirror_buffer(4, start + 128, 1).unwrap();
        let exact = std::collections::BTreeMap::from([(1, 2)]);
        let (plan, checked) = heap.buffer_batch(start, 1, exact.clone()).unwrap();
        assert_eq!(checked, 8);
        assert_eq!(plan.prepared, 4);
        let [PlanStep::Reads(first), PlanStep::Write(write), PlanStep::Reads(last)] = plan.ordered.as_slice() else {
            panic!("write boundary lost");
        };
        assert!(Arc::ptr_eq(&first[0].0, &shadow));
        assert_eq!(first[0].1, [(base + 512, base + 640)]);
        assert!(Arc::ptr_eq(&write.shadow, &shadow));
        assert_eq!((write.address, write.len), (base + 1024, 128));
        assert_eq!(last[0].1, first[0].1);
        let (both, _) = heap.buffer_batch(start, 3, Default::default()).unwrap();
        assert_eq!(both.prepared, 5);
        assert_eq!(both.ordered.len(), 5);
        let (stable, _) = heap.buffer_batch(start, 1, exact.clone()).unwrap();
        let (same, _) = heap.buffer_batch(start, 1, exact.clone()).unwrap();
        assert!(Arc::ptr_eq(&stable, &same));
        let (read, _) = heap.buffer_batch(start, 1, Default::default()).unwrap();
        assert_eq!(read.prepared, 3);
        assert!(read.ordered.is_empty());
        // A raw class change must invalidate both the union and ordered plan.
        slots[1] = slots[0];
        std::hint::black_box(&slots);
        let (changed, _) = heap.buffer_batch(start, 1, exact).unwrap();
        assert_eq!(changed.prepared, 4);
        assert!(changed.ordered.is_empty());
        assert!(!Arc::ptr_eq(&changed, &stable));
        crate::shadow::unregister(owner, base);
    }

    #[test]
    fn original_unknown_heap_demand_classes() {
        let Some(dir) = std::env::var_os("VICTORY_COMPUTE_DESCRIPTOR_DIR") else { return; };
        for hash in ["9209777e8face9dd", "42f717f1b4d18e94"] {
            let bytes = std::fs::read(std::path::PathBuf::from(&dir).join(format!("compute-{hash}.cso"))).unwrap();
            let shader = victory_shader::emit::translate(&bytes).unwrap();
            let unknown = shader.heap_demands.iter().filter(|d| d.code.is_empty())
                .fold(0, |mask, d| mask | d.class);
            assert_ne!(unknown & 1, 0);
            eprintln!("{hash}: classes={} unknown={} exact expressions={}", shader.heap_buffer_classes,
                unknown, shader.heap_demands.iter().filter(|d| !d.code.is_empty()).count());
        }
    }

    #[test]
    #[ignore = "large descriptor preparation benchmark"]
    fn mixed_buffer_batch_large_heap_microbenchmark() {
        const COUNT: u32 = 200_000;
        let heap = HostHeap::new(false, COUNT).expect("hardware heap");
        let base = 0x7d05_0000_0000_u64;
        let owner = usize::MAX - 200;
        let mut slots = vec![[0u32; 8]; COUNT as usize];
        let start = slots.as_mut_ptr() as u64;
        let mut templates = Vec::new();
        for i in 0..32u64 {
            let address = base + i * 0x10000;
            crate::shadow::register(owner + i as usize, address, 0x10000);
            let mut resource = [0u32; 14];
            resource[0] = 1;
            resource[4] = 0x10000;
            for window in 0..8u32 {
                let mut srv = [0u32; 16];
                srv[1] = 1;
                srv[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv[4] = window * 32;
                srv[6] = 8;
                srv[7] = 16;
                srv[10] = address as u32;
                srv[11] = (address >> 32) as u32;
                let mut uav = [0u32; 16];
                uav[1] = 1;
                uav[2] = window * 32;
                uav[4] = 8;
                uav[5] = 16;
                uav[10] = address as u32;
                uav[11] = (address >> 32) as u32;
                let mut pair = [[0u32; 8]; 2];
                super::super::record(pair[0].as_mut_ptr() as u64, View::Placed(false, resource, srv));
                super::super::record(pair[1].as_mut_ptr() as u64, View::Placed(true, resource, uav));
                templates.push(pair);
            }
        }
        for (i, slot) in slots.iter_mut().enumerate() {
            *slot = templates[(i / 8 + i % 7) % templates.len()][usize::from(i % 512 == 511)];
        }
        // First observation discovers and materializes all raw title records;
        // second publishes the immutable, generation-validated plan.
        let (first, _) = heap.buffer_batch(start, 3, Default::default()).unwrap();
        assert_eq!(first.prepared, u64::from(COUNT));
        let (stable, _) = heap.buffer_batch(start, 3, Default::default()).unwrap();
        let (hit, _) = heap.buffer_batch(start, 3, Default::default()).unwrap();
        assert!(Arc::ptr_eq(&stable, &hit));
        let fingerprint = |p: &ReadPlan| -> Vec<(bool, usize, u64, u64)> {
            let mut result = Vec::new();
            for (shadow, intervals) in &p.ranges {
                for &(a,b) in intervals { result.push((false, Arc::as_ptr(shadow) as usize, a,b)); }
            }
            for step in &p.ordered {
                match step {
                    PlanStep::Reads(ranges) => for (shadow, intervals) in ranges {
                        for &(a, b) in intervals { result.push((false, Arc::as_ptr(shadow) as usize, a, b)); }
                    },
                    PlanStep::Write(b) => result.push((true, Arc::as_ptr(&b.shadow) as usize, b.address, b.address + b.len)),
                }
            }
            result
        };
        let expected = fingerprint(&stable);
        assert_eq!(expected.iter().filter(|x| x.0).count(), COUNT as usize / 512);
        for _ in 0..2 {
            for cached in [false, true] {
                let mut elapsed = Vec::new();
                for _ in 0..7 {
                    let began = std::time::Instant::now();
                    let plan = if cached {
                        heap.buffer_batch(start, 3, Default::default()).unwrap().0
                    } else {
                        let mut bindings = Vec::new();
                        for slot in heap.buffer_slots() {
                            if let Some((view, binding)) = heap.mirror_buffer(slot, start + u64::from(slot) * 32, 3) {
                                bindings.push((slot, view_writes(&view), binding));
                            }
                        }
                        ReadPlan::from_slots(bindings)
                    };
                    elapsed.push(began.elapsed().as_secs_f64() * 1000.0);
                    assert_eq!(plan.prepared, u64::from(COUNT));
                    assert_eq!(fingerprint(&plan), expected);
                }
                elapsed.sort_by(f64::total_cmp);
                eprintln!("cached={cached} records={COUNT} median={:.3}ms min={:.3}ms max={:.3}ms",
                    elapsed[3], elapsed[0], elapsed[6]);
            }
        }
        // Actual world shaders have unknown SRV indices but a few exact
        // UAV slots. Change one raw descriptor in every fourth page and
        // alternate exact UAVs. Compare against rebuilding the entire
        // observed plan (same scanner, materializer, and final union).
        let exacts = [
            std::collections::BTreeMap::from([(511,2),(1023,2)]),
            std::collections::BTreeMap::from([(4095,2),(8191,2)]),
        ];
        let mut reference = Vec::new();
        for cached in [false,true] {
            let mut elapsed = Vec::new();
            for iteration in 0..10usize {
                for i in (0..COUNT as usize).step_by(4*SLOTS_PER_PAGE as usize) {
                    slots[i] = templates[(i/8 + 1 + iteration%2) % templates.len()][0];
                }
                if !cached {
                    let mut cache = heap.read_records.lock().unwrap();
                    cache.plan = None;
                    cache.mixed.clear();
                    for page in &mut cache.page_plans { *page = None; }
                }
                let began = std::time::Instant::now();
                let plan = heap.buffer_batch(start,1,exacts[iteration%2].clone()).unwrap().0;
                let ms = began.elapsed().as_secs_f64()*1000.0;
                let output = fingerprint(&plan);
                assert_eq!(plan.prepared,u64::from(COUNT)-(COUNT as u64/512)+2);
                if !cached { reference.push(output); }
                else { assert_eq!(output,reference[iteration]); }
                if iteration>=3 { elapsed.push(ms); }
            }
            elapsed.sort_by(f64::total_cmp);
            eprintln!("churn page_cache={cached} records={COUNT} median={:.3}ms min={:.3}ms max={:.3}ms",
                elapsed[3],elapsed[0],elapsed[6]);
        }
        for i in 0..32u64 { crate::shadow::unregister(owner + i as usize, base + i * 0x10000); }
    }

    fn chunk_fingerprint(p: &ReadPlan) -> Vec<(bool, usize, u64, u64)> {
        let mut out = Vec::new();
        let append = |out: &mut Vec<_>, ranges: &[ShadowRanges]| {
            for (shadow, intervals) in ranges {
                for &(a,b) in intervals { out.push((false, Arc::as_ptr(shadow) as usize, a,b)); }
            }
        };
        append(&mut out, &p.ranges);
        for step in &p.ordered {
            match step {
                PlanStep::Reads(ranges) => append(&mut out, ranges),
                PlanStep::Write(b) => out.push((true, Arc::as_ptr(&b.shadow) as usize, b.address, b.address+b.len)),
            }
        }
        out
    }

    #[test]
    fn page_chunk_cache_preserves_order_and_exact_ranges() {
        chunk_cache_exercise(19, 24, false);
    }

    #[test]
    #[ignore = "combine-only descriptor plan benchmark"]
    fn page_chunk_cache_combine_microbenchmark() {
        chunk_cache_exercise(457, 512, true);
    }

    fn chunk_cache_exercise(npages: usize, nslots: usize, bench: bool) {
        let base = 0x7d09_0000_0000u64;
        let owner = usize::MAX - 500;
        let heap = HostHeap::new(false, 2).expect("hardware heap");
        let mut bindings = Vec::new();
        for i in 0..2u64 {
            let address = base+i*0x10000;
            crate::shadow::register(owner+i as usize,address,0x10000);
            let mut resource = [0u32;14];
            resource[0]=1; resource[4]=0x10000;
            let mut srv=[0u32;16];
            srv[1]=1; srv[2]=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv[6]=8; srv[7]=16; srv[10]=address as u32; srv[11]=(address>>32) as u32;
            let mut record=[0u32;8];
            let at=record.as_mut_ptr() as u64;
            super::super::record(at, View::Placed(false,resource,srv));
            bindings.push(heap.mirror_buffer(i as u32,at,1).expect("binding").1);
        }
        let page = |p: usize, mutation: usize| {
            let mut slots = Vec::new();
            for i in 0..nslots {
                let mut b=bindings[(i+p+mutation)%2].clone();
                b.address += ((i*7+p*3+mutation)%256*32) as u64;
                b.len=16;
                // Writes on both sides of chunk boundaries, including
                // consecutive writes and zero-length bindings.
                let write = (p%8==7 && i+2>=nslots) || (p.is_multiple_of(8) && i==0) || (mutation!=0 && i==3);
                if i==5 { b.len=0; }
                slots.push((i as u32,write,b));
            }
            ReadPlan::from_slots(slots)
        };
        let mut pages: Vec<_>=(0..npages).map(|p|page(p,0)).collect();
        let mut cache=Vec::new();
        let mut flat_times=Vec::new();
        let mut cached_times=Vec::new();
        for iteration in 0..12 {
            if iteration>0 {
                let p=(iteration*13)%npages;
                pages[p]=page(p,iteration);
            }
            let start=std::time::Instant::now();
            let flat=ReadPlan::combine_pages(pages.clone());
            let flat_ms=start.elapsed().as_secs_f64()*1000.0;
            let start=std::time::Instant::now();
            let (cached,reused)=combine_chunks(&mut cache,pages.clone());
            let cached_ms=start.elapsed().as_secs_f64()*1000.0;
            assert_eq!(cached.prepared,flat.prepared);
            assert_eq!(chunk_fingerprint(&cached),chunk_fingerprint(&flat));
            assert_eq!(reused,if iteration==0 {0} else {npages.div_ceil(PLAN_CHUNK_PAGES)-1});
            if iteration>2 { flat_times.push(flat_ms); cached_times.push(cached_ms); }
        }
        let (empty,reused)=combine_chunks(&mut cache,Vec::new());
        assert_eq!(empty.prepared,0); assert_eq!(reused,0); assert!(cache.is_empty());
        if bench {
            flat_times.sort_by(f64::total_cmp); cached_times.sort_by(f64::total_cmp);
            eprintln!("combine only pages={npages} slots/page={nslots} flat median={:.3}ms chunk median={:.3}ms",flat_times[flat_times.len()/2],cached_times[cached_times.len()/2]);
        }
        for i in 0..2u64 { crate::shadow::unregister(owner+i as usize,base+i*0x10000); }
    }

    #[test]
    fn page_plans_reuse_unmodified_pages_across_mixed_demands() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        const P: u32 = SLOTS_PER_PAGE;
        let heap = HostHeap::new(false, 3 * P).expect("hardware heap");
        // SAFETY: A null requested address lets Windows allocate three committed writable pages; success is checked below.
        let start = unsafe { VirtualAlloc(std::ptr::null(), 3 * crate::watch::PAGE,
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_ne!(start, 0);
        let base = 0x7d06_0000_0000_u64;
        let owner = usize::MAX - 250;
        crate::shadow::register(owner, base, 0x10000);
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut srv = [0u32; 16];
        srv[1] = 1;
        srv[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv[4] = 32;
        srv[6] = 8;
        srv[7] = 16;
        srv[10] = base as u32;
        srv[11] = (base >> 32) as u32;
        let mut uav = [0u32; 16];
        uav[1] = 1;
        uav[2] = 64;
        uav[4] = 8;
        uav[5] = 16;
        uav[10] = base as u32;
        uav[11] = (base >> 32) as u32;
        let at = |i: u32| start + u64::from(i) * 32;
        // SAFETY: The test calls this closure with distinct record indices within the committed three-page allocation.
        let copy = |from: u32, to: u32| unsafe {
            std::ptr::copy_nonoverlapping(at(from) as *const u32, at(to) as *mut u32, 8);
        };
        super::super::record(at(1), View::Placed(false, resource, srv));
        super::super::record(at(P + 2), View::Placed(true, resource, uav));
        copy(1, P + 1);
        copy(1, 2 * P + 1);
        copy(P + 2, 2 * P + 2);
        let exact = std::collections::BTreeMap::from([(P + 2, 2)]);
        let first = heap.buffer_batch(start, 1, exact.clone()).unwrap().0;
        assert_eq!(first.prepared, 4);
        let stable = heap.buffer_batch(start, 1, exact.clone()).unwrap().0;
        let saved = || heap.read_records.lock().unwrap().page_plans.iter()
            .map(|p| p.as_ref().map(|p| p.plan.clone())).collect::<Vec<_>>();
        let before = saved();
        assert!(before.iter().all(Option::is_some));
        assert_eq!(stable.ordered.len(), 3);
        // A raw copy on one page must not discard the other page plans.
        copy(1, P + 3);
        let changed = heap.buffer_batch(start, 1, exact.clone()).unwrap().0;
        assert_eq!(changed.prepared, 5);
        let stable = heap.buffer_batch(start, 1, exact.clone()).unwrap().0;
        let after = saved();
        for i in [0, 2] {
            assert!(Arc::ptr_eq(before[i].as_ref().unwrap(), after[i].as_ref().unwrap()));
        }
        assert!(!Arc::ptr_eq(before[1].as_ref().unwrap(), after[1].as_ref().unwrap()));
        assert_eq!(stable.ordered.len(), 3);
        // Different exact UAV slot: only the affected page keys change.
        let exact2 = std::collections::BTreeMap::from([(2 * P + 2, 2)]);
        assert_eq!(heap.buffer_batch(start, 1, exact2.clone()).unwrap().0.prepared, 5);
        let other = heap.buffer_batch(start, 1, exact2).unwrap().0;
        assert_eq!(other.ordered.len(), 2);
        let moved = saved();
        assert!(Arc::ptr_eq(after[0].as_ref().unwrap(), moved[0].as_ref().unwrap()));
        assert!(!Arc::ptr_eq(after[1].as_ref().unwrap(), moved[1].as_ref().unwrap()));
        // Original slot rematerialization cannot use a stale page.
        heap.make_null(1);
        let exact2 = std::collections::BTreeMap::from([(2 * P + 2, 2)]);
        heap.buffer_batch(start, 1, exact2.clone()).unwrap();
        heap.buffer_batch(start, 1, exact2).unwrap();
        let repaired = saved();
        assert!(!Arc::ptr_eq(moved[0].as_ref().unwrap(), repaired[0].as_ref().unwrap()));
        assert!(Arc::ptr_eq(moved[2].as_ref().unwrap(), repaired[2].as_ref().unwrap()));
        crate::shadow::unregister(owner, base);
        // SAFETY: start is the base of the allocation made above and is no longer accessed.
        assert_ne!(unsafe { VirtualFree(start as *mut _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn demand_plan_survives_only_unrelated_registry_changes() {
        let mut registry = super::super::Views {
            revision: 0, list: Vec::new(), live: Vec::new(),
            changes: Default::default(), changed_at: Vec::new(), index: Default::default(),
            by_resource: Default::default(), free: Vec::new(),
        };
        let (i, generation) = registry.intern(View::Shader(123, None));
        let mut records = ReadRecords::default();
        records.pages.push(vec![(0, [VIEW_TAG, i, generation]),
            (1, [VIEW_TAG, i, generation + 1]), (2, [VIEW_TAG, i + 2, 1])]);
        let key = DemandKey::new(3, Default::default(), 3);
        let plan = ReadPlan::new(Vec::new());
        records.publish(key.clone(), registry.revision, 7, plan.clone());
        registry.intern(View::Shader(456, None));
        assert!(Arc::ptr_eq(&records.demand_plan(&key, &registry, 7).unwrap(), &plan));
        assert!(records.demand_plan(&key, &registry, 8).is_none());
        // A previously absent index can become live without raw heap writes.
        registry.intern(View::Shader(789, None));
        assert!(records.demand_plan(&key, &registry, 7).is_none());
        records.publish(key.clone(), registry.revision, 7, plan.clone());
        registry.forget(123);
        assert!(registry.get(i, generation).is_none());
        assert!(registry.get(i, generation + 1).is_none());
        assert!(records.demand_plan(&key, &registry, 7).is_none());
        records.publish(key.clone(), registry.revision, 7, plan.clone());
        assert_eq!(registry.intern(View::Shader(999, None)), (i, generation + 1));
        assert!(registry.get(i, generation + 1).is_some());
        assert!(records.demand_plan(&key, &registry, 7).is_none());
        records.publish(key.clone(), registry.revision, 7, plan.clone());
        for _ in 0..8193 { registry.changed(9999); }
        assert!(Arc::ptr_eq(&records.demand_plan(&key, &registry, 7).unwrap(), &plan));
        registry.changed(i);
        for _ in 0..8193 { registry.changed(9999); }
        assert!(records.demand_plan(&key, &registry, 7).is_none());
    }

    #[test]
    fn read_batch_rejects_stale_generations_and_non_read_classes() {
        use ReadBatchAction::*;
        let mut words = [0u32; 10];
        words[1] = 1;
        words[7] = 16;
        let srv = View::Shader(0, Some(words));
        let uav = View::Unordered(0, 0, Some(words));
        assert_eq!(read_batch_action(None, true, true, true), Mirror);
        assert_eq!(read_batch_action(None, false, false, false), Skip);
        assert_eq!(read_batch_action(Some(&srv), true, true, true), Cached);
        assert_eq!(read_batch_action(Some(&srv), false, false, false), Mirror);
        assert_eq!(read_batch_action(Some(&srv), false, true, true), Mirror);
        assert_eq!(read_batch_action(Some(&uav), true, true, true), Skip);
        assert_eq!(read_batch_action(Some(&View::Constant(0, 16)), false, true, true), Skip);
        assert_eq!(read_batch_action(Some(&View::Shader(123, None)), false, true, true), Skip);
        assert_eq!(read_batch_action(Some(&srv), false, true, true), Mirror);
    }

    #[test]
    fn read_plan_invalidates_when_dead_tag_becomes_live_without_raw_write() {
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let base = 0x7d03_0000_0000_u64;
        let owner = usize::MAX - 81;
        crate::shadow::register(owner, base, 0x10000);
        let shadow = crate::shadow::find(base, 0x10000).unwrap();
        // The placeholder is not a buffer view and is never materialized.
        // Recycle its registry entry under one lock to avoid races with
        // unrelated parallel tests using the same registry free list.
        let placeholder = usize::MAX - 82;
        let (index, old_generation) = super::super::VIEWS.lock().unwrap()
            .intern(View::Shader(placeholder, None));
        let next_generation = old_generation.wrapping_add(1).max(1);
        let mut slots = [[0u32; 8]; 4];
        slots[0][..3].copy_from_slice(&[VIEW_TAG, index, old_generation]);
        slots[1][..3].copy_from_slice(&[VIEW_TAG, index, next_generation]);
        let start = slots.as_mut_ptr() as u64;
        let (before, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert_eq!(before.prepared, 0);
        assert!(heap.read_records.lock().unwrap().plan.is_some());
        let words = slots;
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut desc = [0u32; 16];
        desc[1] = 1;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[4] = 32;
        desc[6] = 8;
        desc[7] = 16;
        desc[10] = base as u32;
        desc[11] = (base >> 32) as u32;
        {
            let mut views = super::super::VIEWS.lock().unwrap();
            let revision = views.revision;
            views.forget(placeholder);
            assert!(views.revision > revision);
            let revision = views.revision;
            let (recycled, generation) = views.intern(View::Placed(false, resource, desc));
            assert_eq!((recycled, generation), (index, next_generation));
            assert!(views.revision > revision);
        }
        // Registry meaning changed, title records did not. The previously
        // dead tag must now materialize a real SRV and the old tag is dead.
        assert_eq!(slots, words);
        let (after, checked) = heap.read_buffer_batch(start).unwrap();
        assert_eq!(checked, 4);
        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.prepared, 1);
        assert_eq!(after.ranges.len(), 1);
        assert!(Arc::ptr_eq(&after.ranges[0].0, &shadow));
        assert_eq!(after.ranges[0].1, [(base + 512, base + 640)]);
        assert!(heap.mirror_table(0, start).is_none());
        crate::shadow::unregister(owner, base);
    }

    #[test]
    fn read_records_dirty_pages_preserve_raw_copies_and_clean_pages() {
        use crate::watch::PAGE;
        // 64 KiB alignment permits real page/record boundary assertions.
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        // SAFETY: Windows chooses the address for the three committed writable pages; success is checked below.
        let start = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE,
            MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as u64;
        assert_ne!(start, 0);
        let count = (3 * PAGE / 32) as u32;
        let per_page = (PAGE / 32) as u32;
        let page = start as usize / PAGE;
        // SAFETY: All test indices are within the allocated three pages; each record has room for three u32s.
        let write = |index: u32, words: [u32; 3]| unsafe {
            ((start + u64::from(index) * 32) as *mut [u32; 3]).write_unaligned(words)
        };
        let mut records = ReadRecords::default();
        write(1, [VIEW_TAG, 123, 77]);
        write(per_page, [VIEW_TAG, 456, 99]);
        assert_eq!(records.observe(start, count, Some((10, vec![]))), u64::from(count));
        assert_eq!(records.pages[0], [(1, [VIEW_TAG, 123, 77])]);
        assert_eq!(records.pages[1], [(per_page, [VIEW_TAG, 456, 99])]);
        let plan = ReadPlan::new(Vec::new());
        records.plan = Some((3, 4, plan.clone()));
        assert!(Arc::ptr_eq(&records.cached_plan(3, 4).unwrap(), &plan));
        assert!(records.cached_plan(4, 4).is_none());
        assert!(records.cached_plan(3, 5).is_none());
        assert_eq!(records.observe(start, count, Some((11, vec![]))), 0);
        assert!(Arc::ptr_eq(&records.cached_plan(3, 4).unwrap(), &plan));
        // A dirty page with identical raw records preserves the plan.
        assert_eq!(records.observe(start, count, Some((12, vec![(page, 12)]))), u64::from(per_page));
        assert!(Arc::ptr_eq(&records.cached_plan(3, 4).unwrap(), &plan));
        // Raw copy to a previously untouched page, and removal of a tag
        // on another page. An older journal entry cannot rescan a clean page.
        write(1, [0; 3]);
        write(2 * per_page + 7, [VIEW_TAG, 456, 99]);
        assert_eq!(records.observe(start, count, Some((15, vec![(page, 13),
            (page + 1, 10), (page + 2, 14)]))), 2 * u64::from(per_page));
        assert!(records.cached_plan(3, 4).is_none());
        assert!(records.pages[0].is_empty());
        assert_eq!(records.pages[1], [(per_page, [VIEW_TAG, 456, 99])]);
        assert_eq!(records.pages[2], [(2 * per_page + 7, [VIEW_TAG, 456, 99])]);
        // No watch coverage forces a full observation and invalidates the
        // cache epoch; reacquiring coverage requires another full scan.
        write(per_page, [0; 3]);
        assert_eq!(records.observe(start, count, None), u64::from(count));
        assert!(records.pages[1].is_empty());
        assert_eq!(records.observe(start, count, Some((16, vec![]))), u64::from(count));
        // SAFETY: start is the original allocation base and the test is finished accessing it.
        unsafe { VirtualFree(start as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn read_records_observe_raw_alias_copy_via_physical_journal() {
        use crate::watch::{self, PAGE};
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile, PAGE_READWRITE};
        // SAFETY: The null optional pointers request an unnamed pagefile-backed section of two pages.
        let section = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(),
            PAGE_READWRITE, 0, (2 * PAGE) as u32, std::ptr::null()) };
        assert!(!section.is_null());
        // SAFETY: section is a valid handle and the view fits the writable two-page section.
        let a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE) };
        // SAFETY: section is still live and the second writable view fits its size.
        let b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE) };
        let (aa, bb) = (a.Value as usize, b.Value as usize);
        assert_ne!(aa, 0);
        assert_ne!(bb, 0);
        watch::title_physical_view(aa, 2 * PAGE, usize::MAX - 75, 0);
        watch::title_physical_view(bb, 2 * PAGE, usize::MAX - 75, 0);
        let count = (2 * PAGE / 32) as u32;
        let per_page = (PAGE / 32) as u32;
        let words = [VIEW_TAG, 912, 13];
        // SAFETY: aa is the checked non-null writable mapping and the record fits in its first page.
        unsafe { (aa as *mut [u32; 3]).write_unaligned(words) };
        // Observe both aliases, so either guard policy can protect them.
        watch::writes(bb, bb + 2 * PAGE, true).unwrap();
        let mut records = ReadRecords::default();
        assert_eq!(records.observe(aa as u64, count, watch::writes(aa, aa + 2 * PAGE, true)), u64::from(count));
        assert_eq!(records.pages[0], [(0, words)]);
        assert_eq!(records.observe(aa as u64, count, watch::writes(aa, aa + 2 * PAGE, true)), 0);
        // No descriptor API: the title copies bytes through another alias.
        // SAFETY: The two 12-byte ranges are within distinct pages of the live writable mapping; the watch handler handles guard faults.
        unsafe { std::ptr::copy_nonoverlapping(bb as *const u8, (bb + PAGE) as *mut u8, 12) };
        assert_eq!(records.observe(aa as u64, count, watch::writes(aa, aa + 2 * PAGE, true)), u64::from(per_page));
        assert_eq!(records.pages[0], [(0, words)]);
        assert_eq!(records.pages[1], [(per_page, words)]);
        // A registered temporary external writer must keep the page dirty
        // even when its other aliases remain read-only.
        {
            let _external = watch::physical_write(usize::MAX - 75, 1, 1);
            assert_eq!(records.observe(aa as u64, count, watch::writes(aa, aa + 2 * PAGE, true)), u64::from(per_page));
            assert_eq!(records.observe(aa as u64, count, watch::writes(aa, aa + 2 * PAGE, true)), u64::from(per_page));
        }
        // SAFETY: Both views and the section are live and no further accesses follow.
        unsafe { UnmapViewOfFile(a); UnmapViewOfFile(b); CloseHandle(section); }
        watch::title_memory_remapped(aa, 2 * PAGE);
        watch::title_memory_remapped(bb, 2 * PAGE);
    }

    #[test]
    fn shared_static_table_retains_exact_slots_and_shadow_until_last_owner() {
        use victory_shader::ResourceType;
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let base = 0x7d02_0000_0000_u64;
        let owner = usize::MAX - 40;
        crate::shadow::register(owner, base, 0x10000);
        let original = crate::shadow::find(base, 0x10000).unwrap();
        let mut resource = [0u32; 14];
        resource[0] = 1;
        resource[4] = 0x10000;
        let mut desc = [0u32; 16];
        desc[1] = 1;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[4] = 32;
        desc[6] = 8;
        desc[7] = 16;
        desc[10] = base as u32;
        desc[11] = (base >> 32) as u32;
        let mut record = [0u32; 8];
        super::super::record(record.as_mut_ptr() as u64, View::Placed(false, resource, desc));
        let words = [record[0], record[1], record[2]];
        let records = [(0, words, Some((ResourceType::Srv, 4))),
                       (257, words, Some((ResourceType::Srv, 4)))];
        let a = heap.share_static_table(&records).unwrap();
        let b = heap.share_static_table(&records).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.bindings.len(), 2);
        assert!(Arc::ptr_eq(&a.bindings[0].1.shadow, &original));
        assert_eq!((a.bindings[0].1.address, a.bindings[0].1.len), (base + 512, 128));
        assert_eq!(heap.next.lock().unwrap().1.iter().filter(|&&taken| taken).count(), 2);
        drop(a);
        assert_eq!(heap.next.lock().unwrap().1.iter().filter(|&&taken| taken).count(), 2);

        // Same title record, new backing registration: do not reuse the
        // descriptor belonging to the previous still-recorded list.
        crate::shadow::register(owner + 1, base, 0x10000);
        let newer = crate::shadow::find(base, 0x10000).unwrap();
        assert!(!Arc::ptr_eq(&newer, &original));
        let c = heap.share_static_table(&records).unwrap();
        assert!(!Arc::ptr_eq(&b, &c));
        assert!(Arc::ptr_eq(&c.bindings[0].1.shadow, &newer));
        assert_eq!(heap.next.lock().unwrap().1.iter().filter(|&&taken| taken).count(), 4);
        drop(b);
        assert_eq!(heap.next.lock().unwrap().1.iter().filter(|&&taken| taken).count(), 2);
        drop(c);
        assert_eq!(heap.next.lock().unwrap().1.iter().filter(|&&taken| taken).count(), 0);
        assert!(heap.share_static_table(&[(0, words, Some((ResourceType::Cbv, 4)))]).is_none());
        assert!(heap.share_static_table(&[(0, words, Some((ResourceType::Uav, 4)))]).is_none());
        assert!(heap.share_static_table(&[(0, words, Some((ResourceType::Srv, 8)))]).is_none());
        assert!(heap.share_static_table(&[(0, words, None)]).is_none());
        crate::shadow::unregister(owner + 1, base);
        crate::shadow::unregister(owner, base);
    }

    #[test]
    fn deferred_buffer_scan_excludes_placed_textures_and_constants() {
        let Some(heap) = HostHeap::new(false, 4) else { return };
        let mut resource = [0; 14];
        let mut desc = [0; 16];
        resource[0] = 3; // Texture2D, intentionally deferred for table use.
        desc[1] = 4; // SRV texture2D dimension.
        let texture = View::Placed(false, resource, desc);
        assert!(texture.is_buffer()); // historical name includes deferred textures
        heap.note_deferred_buffer(0, &texture);
        heap.note_deferred_buffer(1, &View::Constant(0x1000, 256));
        resource[0] = 1;
        desc[1] = 1;
        desc[7] = 48;
        heap.note_deferred_buffer(2, &View::Placed(false, resource, desc));
        assert_eq!(heap.buffer_slots(), vec![2]);
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// A sampler heap larger than 2,048 is capped to PORTABLE_SAMPLERS (2,048).
    #[test]
    fn sampler_heap_over_limit_is_capped_to_portable_samplers() {
        let Some(_host) = host() else {
            return; // no host GPU
        };
        // 4096 exceeds D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE (2048)
        let heap = HostHeap::new(true, 4096).expect("sampler heap capped at 2048 should succeed");
        assert_eq!(heap.count, PORTABLE_SAMPLERS);
        assert_eq!(heap.dynamic, 0);
        assert!(heap.sampler);

        let heap_normal = HostHeap::new(true, 16).expect("normal sampler heap should succeed");
        assert_eq!(heap_normal.count, 16);
        assert_eq!(heap_normal.dynamic, 0);
        assert!(heap_normal.sampler);
    }

    /// A buffer view of no elements, which the host refuses (removing the
    /// device), makes a null slot; one of an element is made as usual.
    #[test]
    fn buffer_views_of_no_elements_are_null() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // A title buffer the view's memory lies in (its copy is made on the
        // host; title memory is not read).
        let (address, len) = (0x7e00_0000_0000_u64, 0x1000_u64);
        crate::shadow::register(usize::MAX - 1, address, len);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = len as u32;
        let placed = |unordered: bool, elements: u32| {
            let mut desc = [0u32; 16];
            desc[1] = 1; // buffer dimension
            if unordered {
                // NumElements, StructureByteStride
                (desc[4], desc[5]) = (elements, 16);
            } else {
                desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                (desc[6], desc[7]) = (elements, 16);
            }
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(unordered, resource, desc)
        };
        let kept = |index: u32| heap.kept.lock().unwrap()[index as usize].is_some();
        for unordered in [false, true] {
            heap.make_at(0, &placed(unordered, 1));
            assert!(kept(0), "a view of an element (unordered: {unordered})");
            heap.make_at(1, &placed(unordered, 0));
            assert!(!kept(1), "a view of no elements (unordered: {unordered})");
            // SAFETY: a plain device query.
            let removed = unsafe { host.device.GetDeviceRemovedReason() };
            assert!(
                removed.is_ok(),
                "device removed: {removed:?} (unordered: {unordered})"
            );
        }
        crate::shadow::unregister(usize::MAX - 1, address);
    }
}
