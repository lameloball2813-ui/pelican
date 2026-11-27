//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

/// An exact table snapshot, including the slot layout and null-view class.
/// Equal record hashes alone are not enough to reuse a relocated table.
#[derive(PartialEq, Eq, Hash)]
struct FreshTableKey {
    heap: usize,
    table: u64,
    slots: Vec<crate::descriptor::host::SharedRecord>,
    /// A resource can be released without changing the title slot words.
    /// Never reuse a live descriptor snapshot for a now-dead generation.
    live: Vec<bool>,
}

struct FreshTable {
    first: u32,
    /// In slot order, exactly the bindings made in the relocated slots.
    /// Retain these even if the title registers another overlapping buffer.
    bindings: Vec<Option<crate::descriptor::host::BufferBinding>>,
}

/// Layout for the shader's raw CBV scratch root SRV. Each record has exact
/// byte bounds; invalid views point at the dedicated zero dword.
type StaticCbvLayout = (u32,u32,Vec<(u32,u32)>);
fn static_cbv_layout(entries: &[(u64,u32)]) -> Option<StaticCbvLayout> {
    let sentinel = u32::try_from(entries.len().checked_mul(8)?).ok()?;
    let mut total = sentinel.checked_add(4)?.checked_add(255)? & !255;
    let mut header = Vec::with_capacity(entries.len());
    for &(_,size) in entries {
        if size == 0 { header.push((sentinel,0)); continue; }
        if size > 65536 || size%16 != 0 { return None; }
        header.push((total,size));
        total = total.checked_add(size.checked_add(255)? & !255)?;
    }
    Some((sentinel,total,header))
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: exact snapshot → allocated
    /// slots and their retained backing resources.
    fresh_tables: std::collections::HashMap<FreshTableKey, FreshTable>,
    /// Exact ordered original CBV address/size tuples, packed into one
    /// list-owned raw SRV. Deferred fills replay on every submission.
    static_cbv_cache: std::collections::HashMap<(u64,Vec<(u64,u32)>),u64>,
    /// Shared immutable tables are owned independently of the reclaimable
    /// list lease. Even a completed list can be submitted again, so keep
    /// every shared allocation until this state is reset or released.
    shared_tables: std::collections::HashMap<usize, std::sync::Arc<crate::descriptor::host::SharedTable>>,
    /// The dynamic slots the list holds (title heap, first, count), given
    /// back when the list is reset or released, or earlier when a heap runs
    /// short after the list ran (see [`crate::descriptor::lease`]).
    lease: std::sync::Arc<crate::descriptor::lease::Lease>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for (heap, first, n) in self.lease.take() {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    if address == 0 {
        return None;
    }
    let shadow = crate::shadow::find(address, 1)?;
    let len = root_view_len(&shadow, address, srv);
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        let wanted = if srv { 0x40 | 0x80 } else { 0x1 };
        shadow.ensure_state(list, wanted);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

/// The bytes of title memory at `address` a root view binds (and shaders
/// may read: see [`State::root_view_bytes`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootViewKind {
    Cbv,
    Srv,
    Uav,
}

fn root_view_bound(available: u64, kind: RootViewKind) -> u64 {
    match kind {
        RootViewKind::Cbv => available.min(1 << 16),
        RootViewKind::Srv => available.min(1 << 20),
        RootViewKind::Uav => available,
    }
}

fn root_view_len(shadow: &crate::shadow::Shadow, address: u64, srv: bool) -> u64 {
    root_view_bound(shadow.left(address), if srv { RootViewKind::Srv } else { RootViewKind::Cbv })
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The list's slots, to start and finish its submissions.
    pub(crate) fn lease(&self) -> &std::sync::Arc<crate::descriptor::lease::Lease> {
        &self.lease
    }

    /// `n` consecutive dynamic slots of `host` (the host heap of the title
    /// heap at address `key`), held by this list. When the heap has no room,
    /// the slots of the oldest list that already ran are taken back first,
    /// one list at a time, instead of dropping the draw.
    fn take_dynamic(
        &mut self,
        host: &crate::descriptor::host::HostHeap,
        key: usize,
        n: u32,
    ) -> Option<u32> {
        let mut reclaimed = false;
        loop {
            if let Some(first) = host.dynamic(n) {
                self.lease.add(key, first, n);
                return Some(first);
            }
            let more = crate::descriptor::lease::reclaim(key, |(heap, first, n)| {
                if heap == key {
                    host.release(first, n);
                } else {
                    crate::descriptor::release_dynamic(heap, first, n);
                }
            });
            if !more {
                return None;
            }
            if !reclaimed {
                reclaimed = true;
                once!("dynamic descriptor slots of lists that already ran are taken back");
            }
        }
    }

    /// As above, but each named range is leased separately. A hole can
    /// belong to a different table without either shader accessing it.
    fn take_dynamic_sparse(
        &mut self,
        host: &crate::descriptor::host::HostHeap,
        key: usize,
        offsets: &[u32],
    ) -> Option<u32> {
        loop {
            if let Some(base) = host.dynamic_sparse(offsets) {
                let mut remaining = offsets;
                while let Some((&first, rest)) = remaining.split_first() {
                    let mut n = 1;
                    while rest.get(n - 1) == Some(&(first + n as u32)) {
                        n += 1;
                    }
                    self.lease.add(key, base + first, n as u32);
                    remaining = &remaining[n..];
                }
                return Some(base);
            }
            if !crate::descriptor::lease::reclaim(key, |(heap, first, n)| {
                if heap == key { host.release(first, n); }
                else { crate::descriptor::release_dynamic(heap, first, n); }
            }) {
                return None;
            }
        }
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bytes root view `slot` holds from its start. Shaders check their
    /// accesses against it: a root view has
    /// no bounds on the host, and an access past the memory it maps is a
    /// GPU page fault, which some hosts stall on until the driver resets
    /// the device. 0 for a root view the title did not set.
    fn root_view_bytes(&self, compute: bool, slot: u8, kind: RootViewKind) -> u32 {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            return 0;
        };
        let Some(shadow) = crate::shadow::find(address, 1) else {
            return 0;
        };
        // A CBV may use a UAV-capable backing buffer. The allocation flag
        // says nothing about this stage's descriptor kind; reporting the
        // entire tail would let a shader read past the 64 KiB CBV copy.
        let len = root_view_bound(shadow.left(address), kind);
        u32::try_from(len).unwrap_or(u32::MAX)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    fn static_cbv_scratch(&mut self, list: &HostList, entries: &[victory_shader::emit::StaticTableCbv], compute: bool) -> Option<u64> {
        let mut key = Vec::with_capacity(entries.len());
        for entry in entries {
            let value = (|| {
                let Root::Table(table) = *self.roots_ref(compute).get(usize::from(entry.slot))? else { return None; };
                let handle = table.checked_add(u64::from(entry.table_offset_dwords)*4)?;
                let heap = self.heap_of(table)?;
                if heap.samplers() || (table-heap.start)%32 != 0 || (handle-heap.start)%32 != 0 { return None; }
                heap.index(handle)?;
                let View::Constant(address,size) = crate::descriptor::view(handle)? else { return None; };
                (address != 0 && size != 0 && size <= 65536 && size%16 == 0 && address.checked_add(u64::from(size)).is_some()).then_some((address,size))
            })().unwrap_or((0,0));
            key.push(value);
        }
        let key = (list.title_write_epoch(), key);
        if let Some(&gpu) = self.static_cbv_cache.get(&key) { return Some(gpu); }
        let (sentinel,total,header) = static_cbv_layout(&key.1)?;
        let (gpu,to) = list.scratch(u64::from(total))?;
        // SAFETY: scratch is an independent mapped upload allocation of
        // at least `total` bytes and retained until this list is reset.
        unsafe {
            for (i,&(offset,size)) in header.iter().enumerate() {
                to.add(i*8).cast::<u32>().write_unaligned(offset);
                to.add(i*8+4).cast::<u32>().write_unaligned(size);
            }
            to.add(sentinel as usize).cast::<u32>().write_unaligned(0);
        }
        for (&(address,size),&(offset,_)) in key.1.iter().zip(&header) {
            if size != 0 {
                // SAFETY: The scratch layout reserves this range within the mapped allocation.
                let dest = unsafe { to.add(offset as usize) };
                list.defer_fill(dest, address as *const u8, size as usize);
            }
        }
        self.static_cbv_cache.insert(key,gpu);
        Some(gpu)
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        let twin_operation = crate::twin::begin(list);
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            const VERTEX_AND_CONSTANT_BUFFER: u32 = 0x1;
            shadow.ensure_state(list, VERTEX_AND_CONSTANT_BUFFER);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            if v.address == 0 || v.size == 0 {
                return false;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            const INDEX_BUFFER: u32 = 0x2;
            shadow.ensure_state(list, INDEX_BUFFER);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        twin_operation.valid()
    }

    /// Prepare buffer descriptors the shader chooses by a dynamic heap index,
    /// and return the GPU address of the original heap's stride metadata.
    /// Their index is not in a used-resource entry, so normal tables() cannot
    /// discover them; both their data and descriptor lifetimes are required.
    fn heap_buffers(&self, list: &HostList, shader: &victory_shader::emit::HostShader, values: &[u32], _hash: &str) -> Option<u64> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let &at = self.heaps.first()?;
        if at == 0 { return None; }
        let heap = crate::descriptor::heap(at as This);
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        list.keep(host.strides_buffer()?);
        let (fallback, exact) = heap_buffer_demand(shader.heap_buffer_classes, &shader.heap_demands, values);
        // Unknown GPU-dependent indices still admit every original slot of
        // their class. Cache exact provenance, including mixed classes and
        // any named slots outside the fallback, not the actual GPU index.
        #[cfg(feature = "diagnostics")]
        let batch_started = std::time::Instant::now();
        let batch = (fallback != 0).then(|| host.buffer_batch(heap.start, fallback, exact.clone())).flatten();
        #[cfg(feature = "diagnostics")]
        let batch_us = batch_started.elapsed().as_micros() as u64;
        if let Some((plan, _checked)) = batch {
            // The exact shadow bindings survive descriptor updates; retain
            // the range union and state for this list, not a prior list.
            #[cfg(feature = "diagnostics")]
            let state_us = std::cell::Cell::new(0u64);
            #[cfg(feature = "diagnostics")]
            let enqueue_us = std::cell::Cell::new(0u64);
            let replay_reads = |ranges: &[crate::descriptor::host::ShadowRanges]| {
                #[cfg(feature = "diagnostics")]
                let phase = std::time::Instant::now();
                list.ensure_shadow_read_states(ranges);
                #[cfg(feature = "diagnostics")]
                state_us.set(state_us.get() + phase.elapsed().as_micros() as u64);
                #[cfg(feature = "diagnostics")]
                let phase = std::time::Instant::now();
                list.defer_refresh_run(ranges);
                #[cfg(feature = "diagnostics")]
                enqueue_us.set(enqueue_us.get() + phase.elapsed().as_micros() as u64);
            };
            replay_reads(&plan.ranges);
            for step in &plan.ordered {
                match step {
                    crate::descriptor::host::PlanStep::Reads(ranges) => replay_reads(ranges),
                    crate::descriptor::host::PlanStep::Write(binding) => {
                        binding.shadow.refresh(list, binding.address, binding.len);
                        binding.shadow.ensure_state(list, 0x8);
                        binding.shadow.gpu_writes(list, binding.address, binding.len);
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::heap_scan_phases(_hash, _checked, plan.prepared,
                started.elapsed().as_micros() as u64,
                Some((batch_us, state_us.get(), enqueue_us.get())));
            return Some(gpu);
        }
        // Preserve the old full class scan for anything not proven by the
        // emitted index expression. Fresh table slots live beyond count and
        // have already been prepared by tables(); do not read title memory there.
        let mut slots: Vec<(u32, u8)> = if fallback != 0 {
            host.buffer_slots().into_iter().map(|slot| (slot, fallback)).collect()
        } else {
            Vec::new()
        };
        for (slot, class) in exact {
            if slot < host.count() {
                match slots.binary_search_by_key(&slot, |&(s, _)| s) {
                    Ok(i) => slots[i].1 |= class,
                    Err(i) => slots.insert(i, (slot, class)),
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        let (visited, mut prepared) = (slots.len() as u64, 0u64);
        let mut reads = HeapReadRanges::default();
        for (slot, classes) in slots {
            let handle = heap.start.checked_add(u64::from(slot) * 32)?;
            // An SRV handle cannot access a UAV descriptor, or vice versa.
            // The translator records all emitted buffer handle annotations;
            // avoid declaring every unrelated UAV in the heap writable for
            // a shader which has only SRV buffer handles.
            let Some((view, binding)) = host.mirror_buffer(slot, handle, classes) else { continue };
            let crate::descriptor::host::BufferBinding { shadow, address, len } = binding;
            if len == 0 { continue; }
            #[cfg(feature = "diagnostics")]
            { prepared += 1; }
            let writes = matches!(view, View::Unordered(..) | View::Placed(true, ..));
            if writes {
                reads.flush(list);
                shadow.refresh(list, address, len);
                shadow.ensure_state(list, 0x8);
                shadow.gpu_writes(list, address, len);
            } else {
                shadow.ensure_state(list, 0x40 | 0x80);
                reads.add(shadow, address, len);
            }
        }
        reads.flush(list);
        #[cfg(feature = "diagnostics")]
        crate::copystat::heap_scan(_hash, visited, prepared, started.elapsed().as_micros() as u64);
        Some(gpu)
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        // Graphics callers already hold the operation through DSV setup.
        let twin_operation = compute.then(|| crate::twin::begin(list));
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let filtered;
            let used = if t.host.static_table_cbvs.is_empty() { &t.used[..] } else {
                filtered = t.used.iter().filter(|u| !(u.root == RootType::Table && u.kind == ResourceType::Cbv
                    && t.host.static_table_cbvs.iter().any(|c| c.slot == u.slot && c.table_offset_dwords == u.table_offset_dwords))).cloned().collect::<Vec<_>>();
                &filtered[..]
            };
            let made = self.tables(list, used, &t.sparse_tables, compute);
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            }
            let mut values = Vec::new();
            if let Some((param, dwords)) = stage.user_data {
                values.resize(dwords as usize, 0u32);
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if u.root == RootType::Descriptor => {
                            let kind = if stage.root_uavs.iter().any(|r| r.0 == u.slot) {
                                RootViewKind::Uav
                            } else if stage.root_srvs.iter().any(|r| r.0 == u.slot) {
                                RootViewKind::Srv
                            } else {
                                RootViewKind::Cbv
                            };
                            self.root_view_bytes(compute, u.slot, kind)
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::STATIC_CBV_SCRATCH && !t.host.static_table_cbvs.is_empty() {
                        let Some(gpu) = self.static_cbv_scratch(list,&t.host.static_table_cbvs,compute) else { return false; };
                        // SAFETY: The scratch allocation supplies a live GPU address and param is the root SRV slot in the bound signature.
                        unsafe {
                            if compute { l.SetComputeRootShaderResourceView(param,gpu); }
                            else { l.SetGraphicsRootShaderResourceView(param,gpu); }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        let Some(gpu) = self.heap_buffers(list, &t.host, &values, &stage.shader.hash) else {
                            once!("a shader needs a bound descriptor stride table");
                            return false;
                        };
                        // SAFETY: persistent stride upload buffer retained
                        // by the command list and its descriptor heap.
                        unsafe {
                            if compute { l.SetComputeRootShaderResourceView(param, gpu); }
                            else { l.SetGraphicsRootShaderResourceView(param, gpu); }
                        }
                        continue;
                    }
                    let gpu = match self.roots_ref(compute).get(usize::from(slot)) {
                        Some(Root::View(address)) => root_view(list, *address, srv),
                        _ => {
                            once!(slot, "a shader reads a root view the title did not set");
                            None
                        }
                    };
                    let Some(gpu) = gpu.or_else(|| crate::host::host()?.null_root_view()) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                // Root UAV descriptors are GPU virtual addresses, not table
                // indices. Bind only a live, writable host shadow; otherwise
                // refuse the dispatch rather than silently lose game writes.
                let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
                    once!(slot, "root UAV not supplied by the title; dropping operation");
                    return false;
                };
                let Some(shadow) = crate::shadow::find(address, 1) else {
                    once!(slot, "root UAV address has no host shadow; dropping operation");
                    return false;
                };
                // A D3D12 raw root UAV must point at a dword-aligned GPU
                // byte address; do not silently round it to another title
                // location, where writes would corrupt unrelated data.
                if address & 3 != 0 || shadow.gpu(address) & 3 != 0 {
                    once!(slot, "unaligned raw root UAV address; dropping operation");
                    return false;
                }
                // SAFETY: shadow owns a live resource; GetDesc only queries it.
                if unsafe { shadow.buffer.GetDesc() }.Flags.0
                    & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
                    once!(slot, "root UAV shadow is not GPU-writable; dropping operation");
                    return false;
                }
                // The root view has no title-side byte length. Bound the
                // refresh and writeback to the actual mapped buffer end.
                // Never turn an untracked GPU write into a silent no-op.
                let len = shadow.left(address);
                if len == 0 { return false; }
                shadow.refresh(list, address, len);
                shadow.gpu_writes(list, address, len);
                shadow.ensure_state(list, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
                let gpu = shadow.gpu(address);
                // SAFETY: live writable shadow and open command list.
                unsafe {
                    if compute { l.SetComputeRootUnorderedAccessView(param, gpu); }
                    else { l.SetGraphicsRootUnorderedAccessView(param, gpu); }
                }
            }
        }
        twin_operation.as_ref().is_none_or(|op| op.valid())
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some((view, binding)) = host.mirror_table(index, handle) {
            readable(list, u, &view, binding.as_ref());
        } else {
            host.make_null_for_used(index, Some(u));
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], sparse_tables: &[u8], compute: bool) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            if (table - heap.start) % 32 != 0
                || table.checked_add(u64::from(n - 1) * 32)
                    .and_then(|at| heap.index(at)).is_none()
            {
                self.table_failed = true;
                continue;
            }
            let sparse = sparse_tables.contains(&slot) && !heap.samplers();
            let offsets: Vec<u32> = if sparse {
                let mut offsets: Vec<_> = entries.iter()
                    .map(|u| u.table_offset_dwords / 8).collect();
                offsets.sort_unstable();
                offsets.dedup();
                offsets
            } else {
                (0..n).collect()
            };
            let heap_key = std::ptr::from_ref(heap) as usize;
            // The exact words and the null descriptor class are part of
            // the cache key. Different layouts cannot reuse a sparse table
            // if they would need a slot the first table didn't reserve.
            let records = offsets.iter().map(|&k| {
                let at = table.checked_add(u64::from(k) * 32)?;
                heap.index(at)?;
                let kind = entries.iter().find(|u| u.table_offset_dwords / 8 == k)
                    .map(|u| (u.kind, u.dwords));
                // SAFETY: checked to be a descriptor in the bound heap.
                Some((k, unsafe { (at as *const [u32; 3]).read_unaligned() }, kind))
            }).collect::<Option<Vec<_>>>();
            let Some(records) = records else {
                self.table_failed = true;
                continue;
            };
            // A proven static table that reaches beyond the host's original
            // slots can be shared if every actual descriptor is a supported
            // SRV/UAV with an exact retained backing. No scratch CBV, no
            // dynamic index proof by assumption, and no lost-list reclaim.
            if sparse && let Some(shared) = host.share_static_table(&records) {
                for (writes, binding) in &shared.bindings {
                    if binding.len == 0 { continue; }
                    binding.shadow.refresh(list, binding.address, binding.len);
                    binding.shadow.ensure_state(list, if *writes { 0x8 } else { 0x40 | 0x80 });
                    if *writes {
                        binding.shadow.gpu_writes(list, binding.address, binding.len);
                    }
                }
                made.push((slot, shared.base * 32));
                self.shared_tables.entry(std::sync::Arc::as_ptr(&shared) as usize).or_insert(shared);
                continue;
            }
            let views: Vec<_> = records.iter()
                .map(|r| crate::descriptor::host::HostHeap::view_from_record(r.1)).collect();
            let live = views.iter().map(Option::is_some).collect();
            let key = FreshTableKey { heap: heap_key, table, slots: records, live };
            if let Some(fresh) = self.fresh_tables.get(&key) {
                for ((&k, view), binding) in offsets.iter().zip(&views).zip(&fresh.bindings) {
                    if let Some(u) = entries.iter().find(|u| u.table_offset_dwords / 8 == k)
                        && u.kind != ResourceType::Cbv
                        && let Some(view) = view
                    {
                        readable(list, u, view, binding.as_ref());
                    }
                }
                made.push((slot, fresh.first * 32));
                continue;
            }
            let first = if sparse {
                self.take_dynamic_sparse(host, heap_key, &offsets)
            } else {
                self.take_dynamic(host, heap_key, n)
            };
            let Some(first) = first else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                self.table_failed = true;
                static FAILURES: AtomicU64 = AtomicU64::new(0);
                let failures = FAILURES.fetch_add(1, Ordering::Relaxed) + 1;
                if failures.is_power_of_two() {
                    tracing::warn!(slot, n, sparse, occupied = offsets.len(), failures,
                        "fresh descriptor table allocation failed");
                }
                continue;
            };
            if sparse {
                static SAVED: AtomicU64 = AtomicU64::new(0);
                let before = SAVED.fetch_add(u64::from(n) - offsets.len() as u64, Ordering::Relaxed);
                let after = before + u64::from(n) - offsets.len() as u64;
                if before == 0 || before.ilog2() != after.ilog2() {
                    tracing::info!(slot, n, occupied = offsets.len(), saved = after,
                        "statically proven sparse descriptor table");
                }
            }
            // Without a static proof this remains the whole range: a
            // shader may index past the entries its used-resource list names.
            let mut bindings = Vec::with_capacity(offsets.len());
            for (k, view) in offsets.into_iter().zip(views) {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                let binding = match view {
                    Some(View::Constant(address, size)) => {
                        if let Some((gpu, to)) = list.scratch(u64::from(size)) {
                            list.defer_fill(to, address as *const u8, size as usize);
                            host.constant_at(index, gpu, size);
                        } else {
                            host.make_null(index);
                        }
                        None
                    }
                    Some(view) => {
                        let binding = host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view, binding.as_ref());
                        }
                        binding
                    }
                    None => {
                        host.make_null_for_used(index, used.copied());
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                        None
                    }
                };
                bindings.push(binding);
            }
            self.fresh_tables.insert(key, FreshTable { first, bindings });
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        let ran = self.prepare(list, false) && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers (their
            // count checked to fit by `rectangles`).
            let (count, from) = if rects {
                (expanded_vertices(vertices).unwrap_or(0), 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn completes_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        // The draw runs `count / 3 * 6` vertices, a count the host takes in
        // 32 bits.
        if expanded_vertices(count).is_none() {
            once!(count, "too many rectangle vertices");
            return false;
        }
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let source = rect_source(first, count, v.stride, v.size)
                .filter(|_| v.address != 0)
                .map(|offset| v.address + offset)
                .filter(|&from| crate::shadow::find(from, rects * 3 * stride).is_some());
            let Some(from) = source else {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            };
            // The expanded vertices' view takes its size in 32 bits.
            let Ok(span) = u32::try_from(rects * 6 * stride) else {
                once!(slot, "rectangle vertices too many to expand");
                return false;
            };
            let Some((gpu, to)) = list.scratch(u64::from(span)) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: span,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        if rects {
            once!("indexed rectangles not drawn");
        }
        let ran = !rects && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        if max == 0 {
            return true;
        }
        // Do not run the argument clamp or allocate/refresh its buffers for
        // a pass already known to be unavailable. This used to spend GPU
        // work (and mutate title arguments/budget) even for refused PSOs.
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        if matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES && !p.rectangles {
            why("rectangle topology unsupported");
            return false;
        }
        if p.compute != (kind == DISPATCH) || p.mesh != (kind == MESH) {
            why("pipeline kind mismatch");
            return false;
        }
        let Some(compact_stride) = crate::indirect::compact_stride(kind) else {
            why("no host signature");
            return false;
        };
        if stride < compact_stride || !stride.is_multiple_of(4) {
            why("invalid argument stride");
            return false;
        }
        let Some(len) = u64::from(stride).checked_mul(u64::from(max)) else {
            why("arguments overflow");
            return false;
        };
        if len == 0 {
            why("zero stride");
            return false;
        }
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = host_signature(kind, compact_stride) else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.gpu(*at)));
        // Bounded first: the clamp binds a compute pipeline of its own.
        let Some(bounded) = crate::indirect::bound(
            list,
            &args.buffer,
            args.gpu(arguments),
            count_arg,
            (kind, stride, max),
        ) else {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        };
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => (self.topology != RECTANGLES || p.rectangles) && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            #[cfg(feature = "diagnostics")]
            why("host binding or index preparation failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        crate::indirect::ready(list, &bounded, count_arg.map(|(b, _)| b));
        // SAFETY: The list is open; the signature is cached and the ready argument and count buffers are kept alive by the list.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded,
                0,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View, binding: Option<&crate::descriptor::host::BufferBinding>) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    // Use the retained binding returned by the actual descriptor write.
    // A second address lookup may select a different overlapping shadow.
    // A failed/null buffer view has no binding and must not prepare a
    // guessed backing allocation.
    if let Some(binding) = binding {
        let writes = matches!(view, View::Unordered(..) | View::Placed(true, ..));
        if binding.len != 0 {
            binding.shadow.refresh(list, binding.address, binding.len);
            binding.shadow.ensure_state(list, if writes { UNORDERED_ACCESS } else { SHADER_RESOURCE });
            if writes {
                binding.shadow.gpu_writes(list, binding.address, binding.len);
            }
        }
        return;
    }
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, counter, desc), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            let needs_transition = *counter != 0 || !crate::descriptor::host::depth_uav_use(list, *resource as This, desc.as_ref());
            if needs_transition {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// The host vertices `vertices` rectangle-list vertices become (six per
/// rectangle), if the count fits in 32 bits.
fn expanded_vertices(vertices: u32) -> Option<u32> {
    (vertices / 3).checked_mul(6)
}

/// Unknown, missing or malformed expressions retain the full scan for their
/// class. A successful expression is evaluated using the actual root values
/// bound to the stage (including freshly relocated descriptor tables).
/// Merge the exact SRV range union before enqueueing refreshes. No write
/// declaration may intervene; the submission Batch would merge precisely
/// these same ranges. Keep first-seen shadow order and Arc identity.
#[derive(Default)]
struct HeapReadRanges {
    by_shadow: std::collections::HashMap<usize, usize>,
    ranges: Vec<crate::descriptor::host::ShadowRanges>,
}

impl HeapReadRanges {
    fn add(&mut self, shadow: std::sync::Arc<crate::shadow::Shadow>, address: u64, len: u64) {
        let Some(end) = address.checked_add(len) else { return };
        let next = self.ranges.len();
        let index = *self.by_shadow.entry(std::sync::Arc::as_ptr(&shadow) as usize).or_insert(next);
        if index == next {
            self.ranges.push((shadow, Vec::new()));
        }
        self.ranges[index].1.push((address, end));
    }

    fn flush(&mut self, list: &HostList) {
        for (shadow, mut ranges) in self.ranges.drain(..) {
            union_heap_reads(&mut ranges);
            for (start, end) in ranges {
                shadow.refresh(list, start, end - start);
            }
        }
        self.by_shadow.clear();
    }
}

fn union_heap_reads(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable();
    let mut n = 0;
    for i in 0..ranges.len() {
        let (start, end) = ranges[i];
        if start >= end { continue; }
        if n > 0 && start <= ranges[n - 1].1 {
            ranges[n - 1].1 = ranges[n - 1].1.max(end);
        } else {
            ranges[n] = (start, end);
            n += 1;
        }
    }
    ranges.truncate(n);
}

fn heap_buffer_demand(
    classes: u8,
    demands: &[victory_shader::emit::HeapDemand],
    values: &[u32],
) -> (u8, std::collections::BTreeMap<u32, u8>) {
    let mut seen = 0;
    let mut fallback = 0;
    let mut slots = std::collections::BTreeMap::new();
    for demand in demands {
        if !matches!(demand.class, 1 | 2) || demand.class & classes == 0 {
            return (classes, Default::default());
        }
        seen |= demand.class;
        if let Some(slot) = demand.slot(values) {
            *slots.entry(slot).or_insert(0) |= demand.class;
        } else {
            fallback |= demand.class;
        }
    }
    fallback |= classes & !seen;
    (fallback, slots)
}

/// The byte offset of vertex `first` in a buffer of `size` bytes holding
/// `stride`-byte vertices, if vertices `first..first + count` all lie in
/// it. Computed wide: `first + count` and the byte offsets overflow 32
/// bits with the values a title may pass.
fn rect_source(first: u32, count: u32, stride: u32, size: u32) -> Option<u64> {
    let stride = u64::from(stride);
    let end = (u64::from(first) + u64::from(count)).checked_mul(stride)?;
    (stride != 0 && end <= u64::from(size)).then(|| u64::from(first) * stride)
}

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::{expanded_vertices, rect_source};

    #[test]
    fn packed_static_cbv_scratch_has_exact_bounds_and_null_sentinel() {
        let (sentinel,total,header) = super::static_cbv_layout(&[(1,16),(0,0),(2,272),(3,65536)]).unwrap();
        assert_eq!(sentinel,32);
        assert_eq!(header,[(256,16),(32,0),(512,272),(1024,65536)]);
        assert_eq!(total,66560);
        assert!(super::static_cbv_layout(&[(1,17)]).is_none());
        assert!(super::static_cbv_layout(&[(1,65552)]).is_none());
    }

    #[test]
    fn rectangle_vertices_inside_their_buffer() {
        assert_eq!(rect_source(0, 3, 16, 48), Some(0));
        assert_eq!(rect_source(3, 3, 16, 96), Some(48));
        assert_eq!(rect_source(3, 3, 16, 95), None);
        assert_eq!(rect_source(0, 3, 0, 48), None);
    }

    /// `first + count` past 32 bits wrapped to a small count before, which
    /// passed the bounds check and read far outside the buffer.
    #[test]
    fn rectangle_bounds_do_not_wrap() {
        assert_eq!(rect_source(u32::MAX, 3, 16, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX - 1, 3, 1, u32::MAX), None);
        assert_eq!(rect_source(u32::MAX, u32::MAX, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn expanded_counts_fit_32_bits() {
        assert_eq!(expanded_vertices(3), Some(6));
        assert_eq!(expanded_vertices(7), Some(12));
        assert_eq!(expanded_vertices(u32::MAX / 6 * 3), Some(u32::MAX / 6 * 6));
        assert_eq!(expanded_vertices(u32::MAX), None);
    }

    use super::*;
    #[test]
    fn root_view_bounds_follow_descriptor_kind_not_allocation_flags() {
        // A writable backing allocation can hold a CBV, SRV or UAV. A CBV
        // must keep its 64 KiB cap even when the resource permits UAV writes;
        // an SRV must keep its 1 MiB cap, while raw UAV bounds extend to the
        // mapped end. Test both large and truncated available ranges.
        for available in [0, 1, 15, 65535, 65536, 65537, 1 << 20, (1 << 20) + 17] {
            assert_eq!(root_view_bound(available, RootViewKind::Cbv), available.min(1 << 16));
            assert_eq!(root_view_bound(available, RootViewKind::Srv), available.min(1 << 20));
            assert_eq!(root_view_bound(available, RootViewKind::Uav), available);
        }
        assert_eq!(root_view_bound(2 << 20, RootViewKind::Cbv), 1 << 16);
        assert_eq!(root_view_bound(2 << 20, RootViewKind::Srv), 1 << 20);
        assert_eq!(root_view_bound(2 << 20, RootViewKind::Uav), 2 << 20);
    }
    #[test]
    fn heap_read_union_preserves_exact_bytes_and_gaps() {
        let mut ranges = vec![(100, 120), (5, 10), (10, 15), (8, 12), (105, 108), (15, 16), (30, 30), (80, 90)];
        union_heap_reads(&mut ranges);
        assert_eq!(ranges, [(5, 16), (80, 90), (100, 120)]);
        let mut seed = 17u32;
        for _ in 0..128 {
            let mut ranges = Vec::new();
            let mut expected = [false; 256];
            for _ in 0..64 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let a = (seed & 255) as usize;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let b = (a + (seed & 15) as usize).min(256);
                expected[a..b].fill(true);
                ranges.push((a as u64, b as u64));
            }
            union_heap_reads(&mut ranges);
            let mut actual = [false; 256];
            for (a,b) in ranges { actual[a as usize..b as usize].fill(true); }
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn exact_heap_demand_preserves_unknown_classes_and_wrapping_indices() {
        use victory_shader::emit::HeapDemand;
        let srv = HeapDemand { class: 1, code: vec![1, 0, 0, 5, 5, 0, 0, 5, 6, 0] };
        let uav = HeapDemand { class: 2, code: vec![0, 123] };
        let unknown = HeapDemand { class: 2, code: vec![] };
        assert_eq!(heap_buffer_demand(3, &[srv.clone(), uav.clone()], &[0x0800_0012]),
            (0, [(18, 1), (123, 2)].into_iter().collect()));
        assert_eq!(heap_buffer_demand(3, &[srv.clone(), unknown], &[18]),
            (2, [(18, 1)].into_iter().collect()));
        assert_eq!(heap_buffer_demand(3, std::slice::from_ref(&srv), &[]).0, 3);
        assert_eq!(heap_buffer_demand(3, &[srv, uav], &[1_000_000]).1.get(&0), Some(&1));
        assert_eq!(heap_buffer_demand(1, &[HeapDemand { class: 1, code: vec![5, 0] }], &[]).0, 1);
    }
    use crate::descriptor::host::HostHeap;

    /// The dynamic slots of an empty heap (and the heap left empty).
    fn capacity(heap: &HostHeap) -> u32 {
        let mut n = 0;
        while heap.dynamic(1).is_some() {
            n += 1;
        }
        heap.release(heap.count(), n);
        n
    }

    /// A heap with no room takes back the slots of a list that already ran,
    /// not of one that has not, and the list that lost them is refused.
    #[test]
    fn a_full_heap_takes_back_the_slots_of_a_list_that_ran() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let key = 0x2000_0001;
        let total = capacity(&heap);
        let chunk = 1000;
        let mut ran = State::default();
        let mut taken = 0;
        while ran.take_dynamic(&heap, key, chunk).is_some() {
            taken += chunk;
        }
        assert!(taken >= total - chunk, "the list filled the heap");
        // Nothing has run yet: nothing is given back and the draw is dropped.
        let mut other = State::default();
        assert_eq!(other.take_dynamic(&heap, key, chunk), None);
        assert!(!ran.lease().is_lost());
        // The list ran: its slots serve the next list, once.
        assert!(ran.lease().start());
        ran.lease().finish();
        assert!(other.take_dynamic(&heap, key, chunk).is_some());
        assert!(ran.lease().is_lost());
        assert!(
            !ran.lease().start(),
            "a list that lost its slots must not run"
        );
        // Everything of the first list is free again except what the other
        // list now holds, and nothing was given back twice.
        assert!(ran.lease().take().is_empty());
        assert_eq!(capacity_left(&heap), total - chunk);
        // Another request finds no finished list left and is dropped.
        let mut third = State::default();
        assert_eq!(third.take_dynamic(&heap, key, total), None);
    }

    /// Free slots of the heap now (taken and released again).
    fn capacity_left(heap: &HostHeap) -> u32 {
        capacity(heap)
    }

    /// A list that is running when another needs room keeps its slots; a
    /// list reset by the title gives back its own and is not taken twice.
    #[test]
    fn a_running_or_reset_list_is_not_taken_back() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let key = 0x2000_0002;
        let total = capacity(&heap);
        let mut running = State::default();
        let first = running.take_dynamic(&heap, key, total).unwrap();
        assert_eq!(first, heap.count());
        running.lease().finish();
        // Submitted again: it is running and cannot be taken back.
        assert!(running.lease().start());
        let mut other = State::default();
        assert_eq!(other.take_dynamic(&heap, key, 1), None);
        assert!(!running.lease().is_lost());
        // It ran: now it can, whole.
        running.lease().finish();
        assert_eq!(other.take_dynamic(&heap, key, 1), Some(heap.count()));
        assert!(running.lease().is_lost());
        // A lease reset by the title (its slots taken, as `Drop` does) is
        // skipped: nothing is held, so nothing is taken back.
        let mut reset = State::default();
        assert!(reset.take_dynamic(&heap, key, 1).is_some());
        reset.lease().finish();
        assert_eq!(reset.lease().take().len(), 1);
        let mut last = State::default();
        assert_eq!(last.take_dynamic(&heap, key, total), None);
        assert!(!reset.lease().is_lost());
    }
}
