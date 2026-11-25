//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    // The CPU mapping always belongs to staging, never the DEFAULT heap.
    // The GPU address and every descriptor name `buffer` for its lifetime.
    card: Option<CardBacking>,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<CopiedPages>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Exact disjoint captured ranges before the submission ran. No gap
    /// between declarations is copied, and an existing capture never moves
    /// just because another declaration extends the writable union.
    before: Mutex<Snapshot>,
    /// Last lookup of this copy; loose windows may be retired only when idle.
    used: Mutex<(Instant, u64)>,
    allocated: u64,
    is_loose: bool,
}

/// Only diagnostics: GPU-visible committed host-mirror memory retained by
/// Shadow objects (registered title buffers and unregistered loose windows).
static LIVE_MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_MIRROR_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_LOOSE_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

struct CardBacking {
    staging: ID3D12Resource,
    // Offsets of exact CPU-modified bytes not yet uploaded to the card.
    dirty: Mutex<Vec<(usize, usize)>>,
    baseline: Option<CardBaseline>,
    bytes: u64,
}

struct CardBaseline {
    buffer: ID3D12Resource,
    known: Mutex<Vec<(usize, usize)>>,
}

impl Drop for CardBacking {
    fn drop(&mut self) {
        LIVE_CARD_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

fn card_limits() -> (u64, u64, bool) {
    static LIMITS: std::sync::OnceLock<(u64, u64, bool)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |key, default: u64| std::env::var(key).ok().and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default).min(4096) << 20;
        (mb("VICTORY_CARD_BUFFERS_MIN_MB", 1).max(65536),
         mb("VICTORY_CARD_BUFFERS_MB", 0), crate::switch::on("VICTORY_CARD_LOOSE"))
    })
}

fn reserve_card(bytes: u64, budget: u64) -> bool {
    LIVE_CARD_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
        used.checked_add(bytes).filter(|&n| n <= budget)
    }).is_ok()
}

pub(crate) fn card_bytes() -> u64 { LIVE_CARD_BYTES.load(Ordering::Relaxed) }

impl Drop for Shadow {
    fn drop(&mut self) {
        LIVE_MIRROR_BYTES.fetch_sub(self.allocated, Ordering::Relaxed);
        if self.is_loose {
            LIVE_LOOSE_BYTES.fetch_sub(self.allocated, Ordering::Relaxed);
        }
    }
}

/// Bound idle snapshot memory, while retaining useful small allocations as
/// well as large ones. Keeping only eight largest buffers made a sparse
/// workload repeatedly allocate and fault in thousands of smaller pieces.
const SNAPSHOT_POOL_BYTES: usize = 256 << 20;
#[derive(Default)]
struct SnapshotPool {
    by_capacity: BTreeMap<usize, Vec<Vec<u8>>>,
    bytes: usize,
}

impl SnapshotPool {
    fn get(&mut self, n: usize) -> Vec<u8> {
        // Avoid pinning a huge buffer to a small snapshot. Retaining the
        // larger buffer for a later large request costs less commit.
        let fit = self.by_capacity.range(n..=n.saturating_mul(2)).next().map(|(&k, _)| k);
        let Some(capacity) = fit else { return Vec::with_capacity(n); };
        let bucket = self.by_capacity.get_mut(&capacity).expect("known bucket");
        let mut v = bucket.pop().expect("nonempty bucket");
        if bucket.is_empty() { self.by_capacity.remove(&capacity); }
        self.bytes -= capacity;
        v.clear();
        v
    }

    fn put(&mut self, mut v: Vec<u8>) {
        let capacity = v.capacity();
        if capacity == 0 || capacity > SNAPSHOT_POOL_BYTES { return; }
        v.clear();
        // Keep small allocations cheap but avoid retaining an unbounded
        // number of tiny blocks and their allocator metadata.
        if self.by_capacity.get(&capacity).is_some_and(|b| b.len() >= 64) { return; }
        while self.bytes + capacity > SNAPSHOT_POOL_BYTES {
            let Some((&largest, _)) = self.by_capacity.last_key_value() else { return; };
            let bucket = self.by_capacity.get_mut(&largest).expect("known bucket");
            bucket.pop();
            if bucket.is_empty() { self.by_capacity.remove(&largest); }
            self.bytes -= largest;
        }
        self.bytes += capacity;
        self.by_capacity.entry(capacity).or_default().push(v);
    }
}

static POOL: Mutex<SnapshotPool> = Mutex::new(SnapshotPool {
    by_capacity: BTreeMap::new(), bytes: 0,
});

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    POOL.lock().unwrap_or_else(|e| e.into_inner()).get(n)
}

/// Returns a snapshot buffer within the bounded idle pool.
fn give_back(v: Vec<u8>) {
    POOL.lock().unwrap_or_else(|e| e.into_inner()).put(v);
}

#[derive(Default)]
struct Snapshot {
    parts: BTreeMap<usize, Vec<u8>>,
    retained: usize,
}

const BASELINE_LIMIT: usize = 768 << 20;
static BASELINE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn persistent_baseline() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_PERSISTENT_BASELINE"))
}

impl Snapshot {
    fn capture(&mut self, at: usize, bytes: &[u8]) {
        if bytes.is_empty() { return; }
        debug_assert!(self.parts.range(..=at).next_back().is_none_or(|(&a, v)| a + v.len() <= at));
        debug_assert!(self.parts.range(at..).next().is_none_or(|(&a, _)| at + bytes.len() <= a));
        let mut saved = pooled(bytes.len());
        saved.extend_from_slice(bytes);
        self.parts.insert(at, saved);
    }

    /// Existing baseline bytes always equal the host mirror outside pending
    /// GPU writes. Capture only newly needed bytes, with exact coverage.
    fn missing(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut at = from;
        let mut missing = Vec::new();
        let start = self.parts.range(..=from).next_back().map_or(from, |(&a,_)| a);
        for (&a, bytes) in self.parts.range(start..to) {
            if a + bytes.len() <= at { continue; }
            if at < a { missing.push((at, a.min(to))); }
            at = at.max(a + bytes.len());
            if at >= to { break; }
        }
        if at < to { missing.push((at,to)); }
        missing
    }

    /// Maintain existing baseline coverage, without extending it. The
    /// supplied bytes come from the actual host mirror after a CPU refresh
    /// or completed GPU writeback, never from another sampling of title VA.
    fn refresh(&mut self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() || self.parts.is_empty() { return; }
        let to = from + bytes.len();
        let start = self.parts.range(..=from).next_back().map_or(from, |(&a,_)| a);
        for (&a, old) in self.parts.range_mut(start..to) {
            let lo = a.max(from);
            let hi = (a + old.len()).min(to);
            if lo < hi { old[lo-a..hi-a].copy_from_slice(&bytes[lo-from..hi-from]); }
        }
    }

    /// Preserve the original 64 KiB comparison granularity, even when
    /// several exact snapshot segments meet inside the piece.
    fn piece<'a>(&'a self, from: usize, to: usize, scratch: &'a mut Vec<u8>) -> Option<&'a [u8]> {
        let (&base, first) = self.parts.range(..=from).next_back()?;
        let end = base.checked_add(first.len())?;
        if from >= end || from >= to { return None; }
        if to <= end { return Some(&first[from - base..to - base]); }
        scratch.clear();
        scratch.extend_from_slice(&first[from - base..]);
        let mut at = end;
        while at < to {
            let next = self.parts.get(&at)?;
            let n = next.len().min(to - at);
            if n == 0 { return None; }
            scratch.extend_from_slice(&next[..n]);
            at += n;
        }
        Some(scratch.as_slice())
    }

    fn clear(&mut self) {
        BASELINE_BYTES.fetch_sub(std::mem::take(&mut self.retained), Ordering::Relaxed);
        for (_, bytes) in std::mem::take(&mut self.parts) { give_back(bytes); }
    }

    fn finish(&mut self, keep: bool) {
        if keep {
            let bytes: usize = self.parts.values().map(Vec::capacity).sum();
            let extra = bytes.saturating_sub(self.retained);
            if bytes <= BASELINE_LIMIT && BASELINE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
                |used| used.checked_add(extra).filter(|&sum| sum <= BASELINE_LIMIT)).is_ok() {
                self.retained = bytes;
                return;
            }
        }
        self.clear();
    }

    fn is_empty(&self) -> bool { self.parts.is_empty() }
}

impl Drop for Snapshot {
    fn drop(&mut self) { BASELINE_BYTES.fetch_sub(self.retained, Ordering::Relaxed); }
}

const NOT_WATCHED: u8 = 2;

/// Full-page observations stay compact. Partial descriptor views retain the
/// exact byte coverage at their observed write generation; previously every
/// partial page was perpetually stale even when its bytes were unchanged.
struct CopiedPages {
    whole: Vec<u64>,
    partial: BTreeMap<usize, (u64, Vec<(usize, usize)>)>,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        Self { whole: vec![0; pages], partial: BTreeMap::new() }
    }

    fn invalidate(&mut self, page: usize) {
        self.whole[page] = 0;
        self.partial.remove(&page);
    }

    /// Emit only uncopied bytes in the requested portion of one page. A
    /// journal write invalidates *all* partial observations of that page.
    /// The bounds are clipped to the actual shadow, not rounded to include
    /// adjacent pending GPU writes or unrelated bytes.
    fn observe(&mut self, page: usize, write: u64, now: u64,
               bounds: (usize, usize), request: (usize, usize),
               mut missing: impl FnMut(usize, usize)) {
        let full = self.whole[page];
        if full != 0 && full >= write { return; }
        self.whole[page] = 0;
        let (from, to) = request;
        if let Some((seen, _)) = self.partial.get(&page) {
            if *seen < write { self.partial.remove(&page); }
        }
        if request == bounds && !self.partial.contains_key(&page) {
            missing(from, to);
            self.whole[page] = now;
            return;
        }
        let (seen, intervals) = self.partial.entry(page).or_insert_with(|| (now, Vec::new()));
        let mut at = from;
        let first = intervals.partition_point(|&(_, end)| end <= from);
        for &(a, b) in &intervals[first..] {
            if a >= to { break; }
            if at < a { missing(at, a.min(to)); }
            at = at.max(b);
            if at >= to { break; }
        }
        if at < to { missing(at, to); }
        union_pending(intervals, from, to);
        *seen = now;
        if pending_covers(intervals, bounds.0, bounds.1) {
            self.whole[page] = now;
            self.partial.remove(&page);
        }
    }
}

/// Half-open span of bytes the GPU copy changed since its snapshot.
/// The caller compares only valid equal-length slices of one mapped piece.
fn difference_span(now: &[u8], before: &[u8]) -> Option<(usize, usize)> {
    let n = now.len().min(before.len());
    // Most declared UAV pages are unchanged. Compare the complete byte
    // range using the optimized slice equality path before locating the
    // boundaries of a changed piece. This is an exact comparison, not a
    // hash or an assumed dirty range, and preserves every changed byte.
    if now[..n] == before[..n] {
        return None;
    }
    let whole = n / 8 * 8;
    let first_word = now[..whole].chunks_exact(8)
        .zip(before[..whole].chunks_exact(8))
        .position(|(a, b)| a != b);
    let from = first_word.map_or(whole, |i| i * 8);
    let first = now[from..n].iter().zip(&before[from..n])
        .position(|(a, b)| a != b).map(|i| from + i)?;
    let last = if let Some(i) = now[whole..n].iter().zip(&before[whole..n])
        .rposition(|(a, b)| a != b)
    {
        whole + i
    } else {
        let word = now[..whole].chunks_exact(8)
            .zip(before[..whole].chunks_exact(8))
            .rposition(|(a, b)| a != b)? * 8;
        word + now[word..word + 8].iter().zip(&before[word..word + 8])
            .rposition(|(a, b)| a != b)?
    };
    Some((first, last + 1))
}

/// Merge an already-read title span with the GPU's bytes, returning only
/// contiguous ranges whose title bytes actually need to change. Never
/// rewrite unchanged gaps: title threads may modify them between our read
/// and the kernel write, and overwriting the enclosing span loses that work.
/// The GPU wins where its byte differs from the pre-submit snapshot.
#[derive(Debug, PartialEq, Eq)]
struct MergeResult {
    writes: Vec<(usize,usize)>,
    // Only CPU-originated changes to the downloaded GPU image need upload.
    // Between these endpoints the unchanged bytes are equal to the card.
    staging_change: Option<(usize,usize)>,
}

fn extend_span(span: &mut Option<(usize,usize)>, a: usize, b: usize) {
    *span = Some(span.map_or((a,b), |(lo,hi)| (lo.min(a),hi.max(b))));
}

fn merge_gpu_title(title: &mut [u8], host: &mut [u8], old: &[u8]) -> MergeResult {
    debug_assert_eq!(title.len(), host.len());
    debug_assert_eq!(title.len(), old.len());
    #[cfg(target_arch = "x86_64")]
    if title.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection, equal-length independent slices.
        return unsafe { merge_gpu_title_avx2(title, host, old) };
    }
    merge_gpu_title_scalar(title,host,old)
}

fn merge_gpu_title_scalar(title: &mut [u8], host: &mut [u8], old: &[u8]) -> MergeResult {
    let mut runs = Vec::new();
    let mut open = None;
    let mut staging_change = None;
    for (i, ((t, h), o)) in title.iter_mut().zip(host).zip(old).enumerate() {
        let merged = if *h != *o { *h } else { *t };
        let change = *t != merged;
        if change {
            *t = merged;
            open.get_or_insert(i);
        } else if let Some(first) = open.take() {
            runs.push((first, i));
        }
        if *h != merged {
            extend_span(&mut staging_change, i, i+1);
            *h = merged;
        }
    }
    if let Some(first) = open {
        runs.push((first, title.len()));
    }
    MergeResult { writes: runs, staging_change }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_gpu_title_avx2(title: &mut [u8], host: &mut [u8], old: &[u8]) -> MergeResult {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize,usize)> = Vec::new();
    let mut append = |a,b| {
        if let Some(last) = runs.last_mut() {
            if last.1 == a { last.1 = b; return; }
        }
        runs.push((a,b));
    };
    let full = title.len() / 32 * 32;
    let mut staging_change = None;
    for at in (0..full).step_by(32) {
        // Each vector represents precisely 32 original byte comparisons.
        // Same-value GPU stores leave newer title bytes untouched.
        let (mut mask, staging) = unsafe {
            let t = _mm256_loadu_si256(title.as_ptr().add(at).cast());
            let h = _mm256_loadu_si256(host.as_ptr().add(at).cast());
            let o = _mm256_loadu_si256(old.as_ptr().add(at).cast());
            let same = _mm256_cmpeq_epi8(h,o);
            let merged = _mm256_blendv_epi8(h,t,same);
            _mm256_storeu_si256(title.as_mut_ptr().add(at).cast(),merged);
            _mm256_storeu_si256(host.as_mut_ptr().add(at).cast(),merged);
            (!(_mm256_movemask_epi8(_mm256_cmpeq_epi8(t,merged)) as u32),
             !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(h,merged)) as u32))
        };
        if staging != 0 {
            extend_span(&mut staging_change, at + staging.trailing_zeros() as usize,
                at + 32 - staging.leading_zeros() as usize);
        }
        while mask != 0 {
            let first = mask.trailing_zeros();
            let count = (mask >> first).trailing_ones();
            let end = first + count;
            append(at+first as usize,at+end as usize);
            mask = if end == 32 { 0 } else { mask & (u32::MAX << end) };
        }
    }
    let tail = merge_gpu_title_scalar(&mut title[full..],&mut host[full..],&old[full..]);
    for (a,b) in tail.writes {
        append(full+a,full+b);
    }
    if let Some((a,b)) = tail.staging_change { extend_span(&mut staging_change,full+a,full+b); }
    MergeResult { writes: runs, staging_change }
}

/// Exact pieces of `[from, to)` not covered by earlier declared GPU writes.
/// A bounding snapshot may already hold bytes in the gaps between disjoint
/// declarations, but those gaps were not captured after their own refresh.
fn uncovered_ranges(from: usize, to: usize, earlier: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut covered: Vec<_> = earlier.iter().filter_map(|&(a, b)| {
        let a = a.max(from);
        let b = b.min(to);
        (a < b).then_some((a, b))
    }).collect();
    covered.sort_unstable();
    let mut at = from;
    let mut missing = Vec::new();
    for (a, b) in covered {
        if at < a { missing.push((at, a)); }
        at = at.max(b);
    }
    if at < to { missing.push((at, to)); }
    missing
}

/// Keep pending GPU-write declarations as their exact interval union, not
/// thousands of duplicate/overlapping views. Both refreshing around pending
/// ranges and copy-back retirement depend on covered bytes, not declaration
/// multiplicity. Input and output are half-open absolute title addresses.
fn union_pending(pending: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    if from >= to { return; }
    let first = pending.partition_point(|&(_, end)| end < from);
    let mut combined = (from, to);
    let mut last = first;
    while last < pending.len() && pending[last].0 <= combined.1 {
        combined.0 = combined.0.min(pending[last].0);
        combined.1 = combined.1.max(pending[last].1);
        last += 1;
    }
    pending.splice(first..last, [combined]);
}

/// The union is sorted, with no touching ranges; at most one interval can
/// cover the entire request. A repeated declaration needs no new snapshot.
fn pending_covers(pending: &[(usize, usize)], from: usize, to: usize) -> bool {
    let i = pending.partition_point(|&(_, end)| end <= from);
    pending.get(i).is_some_and(|&(a, b)| a <= from && to <= b)
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, is_loose: bool) -> Shadow {
        let allocated = len.next_multiple_of(1 << 16);
        let live = LIVE_MIRROR_BYTES.fetch_add(allocated, Ordering::Relaxed) + allocated;
        PEAK_MIRROR_BYTES.fetch_max(live, Ordering::Relaxed);
        if is_loose {
            LIVE_LOOSE_BYTES.fetch_add(allocated, Ordering::Relaxed);
        }
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            card: None,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages as usize)),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            used: Mutex::new((Instant::now(), crate::frame::current())),
            allocated,
            is_loose,
        }
    }

    /// Promote only at allocation: old descriptors cannot silently change
    /// their resource or GPU address. Failure leaves the ordinary mirror.
    fn try_card(self, budget: u64) -> Self {
        self.try_card_mode(budget, crate::switch::on("VICTORY_CARD_COMPARE"))
    }

    fn try_card_mode(mut self, budget: u64, compare: bool) -> Self {
        if !reserve_card(self.allocated, budget) { return self; }
        let Some(local) = host().and_then(|h| h.card_shadow_buffer(self.allocated)) else {
            LIVE_CARD_BYTES.fetch_sub(self.allocated, Ordering::Relaxed);
            return self;
        };
        self.gpu = unsafe { local.GetGPUVirtualAddress() };
        let staging = std::mem::replace(&mut self.buffer, local);
        // Count both DEFAULT allocations against the same budget. Failure
        // to allocate the optional comparison baseline uses full readback.
        let baseline = if compare && self.allocated <= (1u64 << 32) && reserve_card(self.allocated, budget) {
            match host().and_then(|h| h.card_shadow_buffer(self.allocated)) {
                Some(buffer) => Some(CardBaseline { buffer, known: Mutex::default() }),
                None => { LIVE_CARD_BYTES.fetch_sub(self.allocated, Ordering::Relaxed); None }
            }
        } else { None };
        let bytes = self.allocated * (1 + u64::from(baseline.is_some()));
        self.card = Some(CardBacking { staging, dirty: Mutex::default(), baseline, bytes });
        tracing::debug!(start=self.start, bytes=self.allocated, "GPU-local shadow allocated");
        self
    }

    fn maybe_card(self) -> Self {
        let (min, budget, loose) = card_limits();
        if self.len >= min && (!self.is_loose || loose) { self.try_card(budget) } else { self }
    }

    pub(crate) fn card_staging(&self) -> Option<&ID3D12Resource> {
        self.card.as_ref().map(|card| &card.staging)
    }

    pub(crate) fn card_baseline(&self) -> Option<&ID3D12Resource> {
        self.card.as_ref()?.baseline.as_ref().map(|b| &b.buffer)
    }

    /// Baseline must equal staging, including bytes first declared writable
    /// after earlier GPU work. Upload all CPU-modified intervals and any
    /// as-yet uninitialized part of a download, never overwrite the live
    /// card resource for a baseline-only initialization.
    pub(crate) fn baseline_uploads(&self, dirty: &[(usize,usize)], needed: &[(usize,usize)]) -> Vec<(usize,usize)> {
        let Some(baseline) = self.card.as_ref().and_then(|c| c.baseline.as_ref()) else { return Vec::new() };
        let mut known = baseline.known.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = Vec::new();
        for &(a,b) in dirty {
            union_pending(&mut result,a,b);
            union_pending(&mut known,a,b);
        }
        for &(a,b) in needed {
            for (lo,hi) in uncovered_ranges(a,b,&known) { union_pending(&mut result,lo,hi); }
            union_pending(&mut known,a,b);
        }
        result
    }

    pub(crate) fn card_start(&self) -> usize { self.start as usize }

    fn card_dirty(&self, from: usize, to: usize) {
        if let Some(card) = &self.card {
            union_pending(&mut card.dirty.lock().unwrap_or_else(|e| e.into_inner()), from, to);
        }
    }

    /// Taken only under the submission lock, after all CPU refreshes, before
    /// GPU execution. A failed submission is fatal, not an ignored upload.
    pub(crate) fn take_card_uploads(&self) -> Vec<(usize, usize)> {
        self.card.as_ref().map_or_else(Vec::new, |card| {
            std::mem::take(&mut *card.dirty.lock().unwrap_or_else(|e| e.into_inner()))
        })
    }

    pub(crate) fn card_offsets(&self, address: usize, len: usize) -> (usize, usize) {
        self.inside(address, len)
    }

    /// Transitions `buffer` to `wanted` state on `list` if needed (e.g. from
    /// UNORDERED_ACCESS to VERTEX_AND_CONSTANT_BUFFER).
    pub(crate) fn ensure_state(&self, list: &HostList, wanted: u32) {
        list.ensure_buffer_state(&self.buffer, wanted);
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        if from == to {
            return;
        }
        let earlier = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            // Submissions are serialized; an earlier declaration has
            // already captured every byte of a covered interval. Avoid
            // allocating and recopying its metadata on repeated UAV binds.
            if pending_covers(&pending, address, address.saturating_add(len)) {
                return;
            }
            let previous = pending.iter().map(|&(a, b)| self.inside(a, b.saturating_sub(a))).collect::<Vec<_>>();
            union_pending(&mut pending, address, address.saturating_add(len));
            previous
        };
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // Capture each newly declared byte once, after its own refresh.
        // Previously captured bytes must retain their original observation.
        for (a, b) in uncovered_ranges(from, to, &earlier) {
            for (a,b) in before.missing(a,b) {
                before.capture(a, host(a,b));
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::SnapshotBytes, (b - a) as u64);
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_filtered(address, len, |_,_| false)
    }

    /// Only the exact, fence-completed GPU comparison against staging's
    /// before-image may certify a complete unchanged page.
    pub(crate) fn copy_back_filtered(&self, address: usize, len: usize, unchanged: impl Fn(usize,usize)->bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if !snapshot.is_empty() && from < to {
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut old_piece = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                if unchanged(at,n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::CardUnchangedBytes,n as u64);
                    at += n;
                    continue;
                }
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let Some(old) = snapshot.piece(at, at + n, &mut old_piece) else {
                    // Every merged declaration must have an exact capture.
                    // Do not synthesize a before-image for an uncovered gap.
                    debug_assert!(false, "missing GPU write snapshot at {at:#x}+{n:#x}");
                    at += n;
                    continue;
                };
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::BackPieces, 1);
                }
                #[cfg(feature = "diagnostics")]
                let scan = std::time::Instant::now();
                // Usually a few GPU-produced words differ within a much
                // larger heap view. Only read/merge/write the span that may
                // actually have changed, not the enclosing 64 KiB page.
                let differing = difference_span(host, old);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackScanUs, scan.elapsed().as_micros() as u64);
                if let Some((first, end)) = differing {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::BackChanged, 1);
                    let last = end - 1;
                    let changed_at = at + first;
                    let changed_len = last + 1 - first;
                    let host = &mut host[first..=last];
                    let old = &old[first..=last];
                    let title = &mut title[..changed_len];
                    #[cfg(feature = "diagnostics")]
                    let reading = std::time::Instant::now();
                    let read = crate::host::read_into((start + changed_at) as *const u8, title);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::BackReadUs, reading.elapsed().as_micros() as u64);
                    if !read {
                        if persistent_baseline() { snapshot.refresh(changed_at, host); }
                        at += n;
                        continue;
                    }
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    #[cfg(feature = "diagnostics")]
                    let merging = std::time::Instant::now();
                    let merged = merge_gpu_title(title, host, old);
                    // The download and comparison baseline already hold
                    // the GPU output. Only actual CPU-originated staging
                    // differences need to be uploaded again. Every other
                    // byte within the span is identical to the live card at
                    // this submission boundary.
                    if let Some((a,b)) = merged.staging_change {
                        self.card_dirty(changed_at+a, changed_at+b);
                    }
                    let runs = merged.writes;
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::BackMergeUs, merging.elapsed().as_micros() as u64);
                    // Keep the exact changed runs even when numerous. A
                    // single guard scope amortizes the protection work, but
                    // the kernel writes must never overwrite unchanged gaps.
                    if let (Some(&(a, _)), Some(&(_, b))) = (runs.first(), runs.last()) {
                        let write_at = start + changed_at;
                        #[cfg(feature = "diagnostics")]
                        let writing = std::time::Instant::now();
                        let done = crate::host::write_mapped_runs(write_at as *mut u8, title, &runs);
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::BackWriteUs, writing.elapsed().as_micros() as u64);
                        written += done;
                        // Even a short or failed kernel write must not leave
                        // the mirror marked current. Invalidating the whole
                        // touched page is conservative for successful writes
                        // as well: next refresh rechecks title memory.
                        let base = start / PAGE;
                        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                        for page in (write_at + a) / PAGE..=(write_at + b - 1) / PAGE {
                            copied.invalidate(page - base);
                        }
                    }
                    if persistent_baseline() { snapshot.refresh(changed_at, host); }
                }
                at += n;
            }
            // A trimmed merge did not reconcile the rest of its page with
            // current title memory. Leave the page's write-watch generation
            // stale so the next refresh safely rechecks it rather than
            // marking an unsynchronized whole page current.
        }
        // A copy-back interval can be the union of several overlapping
        // declared UAV views. Retire every fully covered pending declaration
        // so their snapshot is released when the last merged interval ends.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&(from, to)| !(address <= from && to <= address + len));
        if pending.is_empty() { snapshot.finish(persistent_baseline()); }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let busy = {
            let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            // union_pending keeps the ranges sorted and disjoint. Copy
            // only overlapping entries, without sorting or scanning all
            // pending intervals. Release the lock before observing title memory.
            let first = pending.partition_point(|&(_, e)| e <= from);
            let last = pending.partition_point(|&(a, _)| a < to);
            pending[first..last].to_vec()
        };
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED
            || crate::watch::known_physical(from, to)
        {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run): (usize, Option<(usize, usize)>) = (0, None);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let bounds = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            let request = (bounds.0.max(from), bounds.1.min(to));
            copied.observe(page - base, write, now, bounds, request, |a, b| {
                if let Some((first, last)) = run {
                    if last == a { run = Some((first, b)); return; }
                    bytes += self.copy(first, last);
                }
                run = Some((a, b));
            });
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((first, end)) = run {
            bytes += self.copy(first, end);
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        self.card_dirty(from - start, to - start);
        if persistent_baseline() {
            // All refreshes are serialized and exclude pending writes.
            // Preserve the captured before-image of pending ranges.
            let bytes = unsafe { std::slice::from_raw_parts(at, to-from) };
            self.before.lock().unwrap_or_else(|e| e.into_inner()).refresh(from-start,bytes);
        }
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if address == 0 {
        return None;
    }
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len, false).maybe_card()));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn idle(now: (Instant, u64), used: (Instant, u64)) -> bool {
    now.0.saturating_duration_since(used.0) > Duration::from_secs(10)
        && now.1.saturating_sub(used.1) >= 4
}

/// The temporary AddRef/Release pair leaves the COM reference count unchanged.
/// Descriptor slots hold COM references independently of the Shadow Arc.
fn resource_references(resource: &ID3D12Resource) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = resource.as_raw();
    // SAFETY: a live COM object's first three vtable entries are IUnknown.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Reclaim only loose windows that neither a list nor a descriptor holds.
/// Call after the reset list's last submission and copy-back completed.
/// Other lists must retain their resources until their submissions finish.
/// Holding LOOSE through the check and removal prevents concurrent lookups
/// from acquiring a reference between them. Registered buffers are untouched.
pub(crate) fn let_go_idle() {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let now = (Instant::now(), crate::frame::current());
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_some_and(|at| now.0.saturating_duration_since(at) < Duration::from_secs(1)) {
            return;
        }
        *last = Some(now.0);
    }
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let mut removed = 0u64;
    let mut bytes = 0u64;
    map.retain(|_, s| {
        let mut used = s.used.lock().unwrap_or_else(|e| e.into_inner());
        let unreferenced = Arc::strong_count(s) == 1
            && s.pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
            // A retained baseline is idle when no GPU write is pending;
            // dropping the shadow also releases its baseline reservation.
            && resource_references(&s.buffer) == 1;
        // A retained descriptor or list may be used without a fresh lookup.
        // Require the full idle period after the last observation of a pin.
        if !unreferenced {
            *used = now;
            return true;
        }
        let reclaim = idle(now, *used);
        if reclaim {
            removed += 1;
            bytes += s.len;
        }
        !reclaim
    });
    #[cfg(feature = "diagnostics")]
    tracing::info!(
        removed,
        bytes,
        live_windows = map.len(),
        live_bytes = map.values().map(|s| s.len).sum::<u64>(),
        mirror_live_bytes = LIVE_MIRROR_BYTES.load(Ordering::Relaxed),
        mirror_peak_bytes = PEAK_MIRROR_BYTES.load(Ordering::Relaxed),
        mirror_loose_live_bytes = LIVE_LOOSE_BYTES.load(Ordering::Relaxed),
        "loose shadow window sweep"
    );
    #[cfg(not(feature = "diagnostics"))]
    if removed != 0 {
        tracing::debug!(removed, bytes, "loose shadow window sweep");
    }
}

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        *shadow.used.lock().unwrap_or_else(|e| e.into_inner()) =
            (Instant::now(), crate::frame::current());
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len, true).maybe_card());
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "matched GPU microbenchmark, no game process"]
    fn card_scan_sparse_writeback_microbenchmark() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        let host=host().expect("hardware host");
        const SIZE:usize=64<<20;
        const N:usize=12;
        for compare in [false,true,false,true] {
            let title=unsafe { VirtualAlloc(std::ptr::null(),SIZE,MEM_RESERVE|MEM_COMMIT|0x20_0000,PAGE_READWRITE) }.cast::<u8>();
            assert!(!title.is_null());
            struct Free(*mut u8);
            impl Drop for Free { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(),0,MEM_RELEASE); } } }
            let _free=Free(title);
            unsafe { title.write_bytes(0x19,SIZE); }
            let (buffer,data)=host.shared_buffer(SIZE as u64).unwrap();
            let gpu=unsafe { buffer.GetGPUVirtualAddress() };
            let shadow=Arc::new(Shadow::new(buffer,data,gpu,title as u64,SIZE as u64,false).try_card_mode(u64::MAX,compare));
            assert!(shadow.card_staging().is_some());
            assert_eq!(shadow.card_baseline().is_some(),compare);
            let (source,src)=host.shared_buffer(PAGE as u64).unwrap();
            let list=host.new_list().unwrap();
            shadow.refresh(&list,title as u64,SIZE as u64);
            shadow.gpu_writes(&list,title as u64,SIZE as u64);
            shadow.ensure_state(&list,0x400);
            list.ensure_buffer_state(&source,0x800);
            for page in (0..SIZE/PAGE).step_by(64) {
                unsafe { list.list.CopyBufferRegion(&shadow.buffer,(page*PAGE+16) as u64,&source,0,16); }
            }
            list.close();
            let mut samples=Vec::new();
            for i in 0..N+3 {
                let value=if i%2==0 {0xa7} else {0x31};
                unsafe { src.write_bytes(value,16); }
                let before=Instant::now();
                host.execute(&[&list]);
                let ms=before.elapsed().as_secs_f64()*1000.0;
                if i>=3 {samples.push(ms);}
                for page in (0..SIZE/PAGE).step_by(64) {
                    assert_eq!(unsafe { std::slice::from_raw_parts(title.add(page*PAGE+16),16) },&[value;16]);
                    assert_eq!(unsafe { title.add(page*PAGE+32).read_volatile() },0x19);
                }
                if compare { assert_eq!(host.card_scan_flag(0),Some(false)); assert_eq!(host.card_scan_flag(1),Some(true)); }
            }
            samples.sort_by(f64::total_cmp);
            println!("64 MiB sparse writeback, compare={compare}, median={:.3}ms min={:.3}ms max={:.3}ms",(samples[N/2-1]+samples[N/2])*0.5,samples[0],samples[N-1]);
        }
    }

    /// Real D3D12 comparison over two complete pages and two partial pages.
    /// Verify both comparison flags, actual staging bytes, title gaps, and
    /// the case where a previous GPU operation was outside a write declaration.
    #[test]
    fn card_scan_exact_pages_and_late_declarations() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        let host = host().expect("D3D12 host");
        let bytes = PAGE*4;
        let title = unsafe { VirtualAlloc(std::ptr::null(),bytes,MEM_RESERVE|MEM_COMMIT|0x20_0000,PAGE_READWRITE) }.cast::<u8>();
        assert!(!title.is_null());
        struct Free(*mut u8);
        impl Drop for Free { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(),0,MEM_RELEASE); } } }
        let _free=Free(title);
        unsafe { title.write_bytes(0x19,bytes); }
        let at=title as usize;
        let (buffer,data)=host.shared_buffer(bytes as u64).unwrap();
        let gpu=unsafe { buffer.GetGPUVirtualAddress() };
        let shadow=Arc::new(Shadow::new(buffer,data,gpu,at as u64,bytes as u64,false).try_card_mode(u64::MAX,true));
        assert!(shadow.card_baseline().is_some());
        let (source,ptr)=host.shared_buffer(PAGE as u64).unwrap();
        unsafe { ptr.write_bytes(0xa7,16); }
        let list=host.new_list().unwrap();
        shadow.refresh(&list,at as u64+1,(bytes-2) as u64);
        shadow.gpu_writes(&list,at as u64+1,(bytes-2) as u64);
        shadow.ensure_state(&list,0x400);
        list.ensure_buffer_state(&source,0x800);
        unsafe {
            list.list.CopyBufferRegion(&shadow.buffer,(2*PAGE+32) as u64,&source,0,16);
            // Partial-page fallback at the end of the range.
            list.list.CopyBufferRegion(&shadow.buffer,(3*PAGE+12) as u64,&source,0,4);
        }
        list.defer(move || unsafe { ((at+PAGE+7) as *mut u8).write_volatile(0x42); });
        list.close();
        host.execute(&[&list]);
        assert_eq!(host.card_scan_flag(0),Some(true));
        assert_eq!(host.card_scan_flag(1),Some(false));
        let actual=unsafe { std::slice::from_raw_parts(title,bytes) };
        assert_eq!(actual[PAGE+7],0x42);
        assert_eq!(&actual[2*PAGE+32..2*PAGE+48],&[0xa7;16]);
        assert_eq!(&actual[3*PAGE+12..3*PAGE+16],&[0xa7;4]);
        assert_eq!(actual[0],0x19); assert_eq!(actual[bytes-1],0x19);
        host.execute(&[&list]);
        assert_eq!(host.card_scan_flag(0),Some(true));
        assert_eq!(host.card_scan_flag(1),Some(true));
        // A GPU operation outside a prior write declaration must not be
        // missed by comparing against a copy of the already modified card.
        let late=host.new_list().unwrap();
        shadow.refresh(&late,(at+2*PAGE) as u64,PAGE as u64);
        shadow.ensure_state(&late,0x400);
        late.ensure_buffer_state(&source,0x800);
        unsafe { late.list.CopyBufferRegion(&shadow.buffer,(2*PAGE+64) as u64,&source,0,16); }
        late.close(); host.execute(&[&late]);
        assert_eq!(unsafe { title.add(2*PAGE+64).read_volatile() },0x19);
        let declare=host.new_list().unwrap();
        shadow.refresh(&declare,(at+2*PAGE) as u64,PAGE as u64);
        shadow.gpu_writes(&declare,(at+2*PAGE) as u64,PAGE as u64);
        declare.close(); host.execute(&[&declare]);
        assert_eq!(host.card_scan_flag(0),Some(false));
        assert_eq!(unsafe { std::slice::from_raw_parts(title.add(2*PAGE+64),16) },&[0xa7;16]);
    }

    /// Real GPU: two dependent compute lists name the same immutable card
    /// backing through UAV and SRV. The second must see the first's result,
    /// not the staging copy. A CPU write between the snapshot and execution
    /// is preserved in the untouched gap and uploaded for the next submit.
    #[test]
    fn card_shadow_dependent_dispatches_preserve_title_gaps() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        let host = host().expect("D3D12 host for card test");
        let title = unsafe { VirtualAlloc(std::ptr::null(), PAGE * 2, MEM_RESERVE | MEM_COMMIT | 0x20_0000, PAGE_READWRITE) }.cast::<u8>();
        assert!(!title.is_null());
        struct Free(*mut u8);
        impl Drop for Free { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE); } } }
        let _free = Free(title);
        let make = |at: usize| {
            let (buffer, data) = host.shared_buffer(PAGE as u64).unwrap();
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let shadow = Arc::new(Shadow::new(buffer, data, gpu, at as u64, PAGE as u64, false).try_card(u64::MAX));
            assert!(shadow.card_staging().is_some(), "DEFAULT heap allocation required");
            assert_ne!(shadow.buffer.as_raw(), shadow.card_staging().unwrap().as_raw());
            shadow
        };
        let aa = title as usize;
        let ba = aa + PAGE;
        let a = make(aa);
        let b = make(ba);
        let set = |at: usize, offset: usize, value: u32| unsafe { ((at+offset) as *mut u32).write_volatile(value) };
        let get = |at: usize, offset: usize| unsafe { ((at+offset) as *const u32).read_volatile() };
        set(aa, 0, 10); set(aa, 4, 20); set(aa, 12, 30); set(aa, 20, 12345);
        set(ba, 0, 40); set(ba, 4, 50); set(ba, 12, 60); set(ba, 20, 54321);
        let source = r#"
            #define RS "SRV(t0), UAV(u0)"
            ByteAddressBuffer input : register(t0);
            RWByteAddressBuffer output : register(u0);
            [RootSignature(RS)] [numthreads(1,1,1)]
            void main() { output.Store(0, input.Load(0)+11); output.Store(12, input.Load(4)+37); }
        "#;
        let mut code: Option<ID3DBlob> = None;
        unsafe { D3DCompile(source.as_ptr().cast(), source.len(), s!("card-test"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None).unwrap(); }
        let code = code.unwrap();
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso = host.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            }).unwrap();
            (root, pso)
        };
        let record = |input: &Arc<Shadow>, from: usize, output: &Arc<Shadow>, to: usize| {
            let list = host.new_list().unwrap();
            input.refresh(&list, from as u64, 16);
            output.refresh(&list, to as u64, 16);
            input.ensure_state(&list, 0x40 | 0x80);
            output.ensure_state(&list, 8);
            output.gpu_writes(&list, to as u64, 16);
            // Overlapping declaration must not change the captured baseline.
            output.gpu_writes(&list, to as u64 + 4, 8);
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.SetComputeRootShaderResourceView(0, input.gpu(from as u64));
                list.list.SetComputeRootUnorderedAccessView(1, output.gpu(to as u64));
                list.list.Dispatch(1,1,1);
            }
            list
        };
        // Record consumer first to exercise submission-order transitions.
        let consumer = record(&a, aa, &b, ba);
        consumer.close();
        let producer = record(&b, ba, &a, aa);
        producer.defer(move || unsafe { ((aa+4) as *mut u32).write_volatile(77); });
        producer.close();
        host.execute(&[&producer, &consumer]);
        assert_eq!([get(aa,0),get(aa,4),get(aa,12),get(aa,20)], [51,77,87,12345]);
        assert_eq!([get(ba,0),get(ba,4),get(ba,12),get(ba,20)], [62,50,57,54321]);
        // Resubmit closed lists: preserved CPU gap now reaches the card.
        host.execute(&[&producer, &consumer]);
        assert_eq!([get(aa,0),get(aa,4),get(aa,12),get(aa,20)], [73,77,87,12345]);
        assert_eq!([get(ba,0),get(ba,4),get(ba,12),get(ba,20)], [84,50,114,54321]);
    }

    #[test]
    fn partial_page_refresh_matches_byte_coverage_and_invalidates_on_write() {
        let mut pages = CopiedPages::new(3);
        let mut known = [[false; 64]; 3];
        let mut writes = [0; 3];
        let mut seed = 0x1234ab98u32;
        for now in 1..5000u64 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let page = (seed >> 8) as usize % 3;
            if seed % 13 == 0 { writes[page] = now; known[page].fill(false); }
            if seed % 19 == 0 { pages.invalidate(page); known[page].fill(false); }
            let a = (seed >> 16) as usize % 64;
            let b = (a + 1 + (seed >> 24) as usize % 64).min(64);
            let base = 1000 + page * 64;
            let mut actual = [false; 64];
            pages.observe(page, writes[page], now, (base, base + 64), (base + a, base + b), |from, to| {
                assert!(from >= base + a && to <= base + b && from < to);
                for i in from-base..to-base { assert!(!actual[i]); actual[i] = true; }
            });
            for i in 0..64 {
                assert_eq!(actual[i], a <= i && i < b && !known[page][i], "generation {now}, page {page}, byte {i}");
            }
            known[page][a..b].fill(true);
        }
    }

    #[test]
    fn partial_shadow_refresh_reads_real_title_writes() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        let Some(host) = host() else { return };
        let src = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT | 0x20_0000, PAGE_READWRITE) }.cast::<u8>();
        assert!(!src.is_null());
        struct Free(*mut u8);
        impl Drop for Free { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE); } } }
        let _free = Free(src);
        let (buffer, data) = host.shared_buffer(PAGE as u64).unwrap();
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let shadow = Shadow::new(buffer, data, gpu, src as u64, PAGE as u64, false);
        unsafe { src.write_bytes(0x35, PAGE); }
        let base = src as usize;
        assert_eq!(shadow.update(base + 16, 32), 32);
        assert_eq!(shadow.update(base + 24, 16), 0);
        assert_eq!(shadow.update(base + 32, 32), 16);
        assert_eq!(unsafe { std::slice::from_raw_parts(data.add(16), 48) }, &[0x35; 48]);
        unsafe { src.add(25).write_volatile(0xc9); }
        assert_eq!(shadow.update(base + 24, 16), 16);
        assert_eq!(unsafe { data.add(25).read() }, 0xc9);
        assert_eq!(shadow.update(base + 24, 16), 0);
        assert_eq!(shadow.update(base + 16, 48), 32);
    }

    #[test]
    fn gpu_writeback_changes_only_actual_bytes_preserving_concurrent_title_gaps() {
        let before = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut host = before;
        host[0] = 30;
        host[3] = 40;
        host[4] = 50;
        host[8] = 90;
        let mut title = before;
        title[1] = 100; // title wrote an unchanged GPU position
        title[6] = 101; // a second title write in the gap
        title[8] = 90; // already matches a GPU change; no write required
        let merged = merge_gpu_title(&mut title, &mut host, &before);
        assert_eq!(merged.writes, [(0, 1), (3, 5)]);
        assert_eq!(merged.staging_change, Some((1,7)));
        assert_eq!(title, [30, 100, 3, 40, 50, 6, 101, 8, 90]);
        assert_eq!(host, title);
        // No write is needed when the GPU did not change a byte; the title's
        // newer values are mirrored without being rewritten through NT.
        let unchanged = title.clone();
        let merged = merge_gpu_title(&mut title, &mut host, &unchanged);
        assert!(merged.writes.is_empty());
        assert_eq!(merged.staging_change,None);
    }

    #[test]
    fn gpu_writeback_run_boundaries_cover_no_extra_bytes() {
        for n in 1..256 {
            let old = vec![0u8; n];
            let mut gpu = old.clone();
            let mut title = old.clone();
            for (i, byte) in gpu.iter_mut().enumerate() {
                if i % 7 == 0 { *byte = 42; }
            }
            let changed: Vec<_> = gpu.iter().enumerate().filter_map(|(i, &b)| (b != 0).then_some(i)).collect();
            let merged = merge_gpu_title(&mut title, &mut gpu, &old);
            assert_eq!(merged.staging_change,None);
            let runs = merged.writes;
            assert_eq!(title.iter().enumerate().filter_map(|(i, &b)| (b != 0).then_some(i)).collect::<Vec<_>>(), changed);
            assert_eq!(runs.iter().flat_map(|&(a, b)| a..b).collect::<Vec<_>>(), changed);
        }
    }

    #[test]
    fn vector_writeback_matches_scalar_on_unaligned_mixed_changes() {
        let mut seed = 0x73bc9a12u32;
        let mut random = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); (seed >> 24) as u8 };
        for n in [0,1,31,32,33,63,64,65,511,65536] {
            for offset in 0..8 {
                let old: Vec<_> = (0..n+offset).map(|_| random()).collect();
                let mut title = old.clone();
                let mut gpu = old.clone();
                for i in offset..offset+n {
                    if random() % 7 == 0 { title[i] = random(); }
                    if random() % 5 == 0 { gpu[i] = random(); }
                }
                let (mut t,mut g) = (title.clone(),gpu.clone());
                let a = merge_gpu_title(&mut title[offset..], &mut gpu[offset..], &old[offset..]);
                let b = merge_gpu_title_scalar(&mut t[offset..],&mut g[offset..],&old[offset..]);
                assert_eq!(a,b); assert_eq!(t,title); assert_eq!(g,gpu);
            }
        }
    }

    #[test]
    fn gpu_writeback_span_never_drops_changed_bytes() {
        let baseline = [0u8; 256];
        assert_eq!(difference_span(&baseline, &baseline), None);
        for first in [0, 1, 33, 254, 255] {
            for last in [first, 255] {
                let mut changed = baseline;
                changed[first] = 7;
                changed[last] = 9;
                assert_eq!(difference_span(&changed, &baseline), Some((first, last + 1)));
            }
        }
        for n in 1..257 {
            for first in [0, n / 2, n - 1] {
                let last = n - 1;
                let before = vec![0u8; n];
                let mut after = before.clone();
                after[first] = 7;
                after[last] = 9;
                assert_eq!(difference_span(&after, &before), Some((first, last + 1)), "n={n}, first={first}");
            }
        }
    }

    #[test]
    fn writeback_span_matches_byte_reference_for_unaligned_slices() {
        let mut seed = 0x3254_789au32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as usize
        };
        for n in [0, 1, 7, 8, 9, 63, 64, 65, 4095, 65536] {
            for offset in 0..8 {
                let old: Vec<u8> = (0..n + offset + 1).map(|_| random() as u8).collect();
                let mut changed = old.clone();
                for count in [0, 1, 2, 9, 40] {
                    if n != 0 {
                        for _ in 0..count {
                            let at = offset + random() % n;
                            changed[at] ^= 1;
                        }
                    }
                    let a = &changed[offset..offset+n];
                    let b = &old[offset..offset+n];
                    let expected = a.iter().zip(b).position(|(x,y)| x != y)
                        .map(|first| (first, a.iter().zip(b).rposition(|(x,y)| x != y).unwrap() + 1));
                    assert_eq!(difference_span(a, b), expected);
                }
            }
        }
    }

    #[test]
    fn newly_declared_gpu_writes_capture_holes_but_not_prior_ranges() {
        assert_eq!(uncovered_ranges(0, 100, &[]), [(0, 100)]);
        assert_eq!(uncovered_ranges(0, 100, &[(0, 20), (80, 100)]), [(20, 80)]);
        assert_eq!(uncovered_ranges(0, 100, &[(80, 100), (0, 20), (5, 25), (22, 30)]), [(30, 80)]);
        assert_eq!(uncovered_ranges(0, 100, &[(0, 200)]), []);
        assert_eq!(uncovered_ranges(10, 30, &[(0, 20), (25, 40)]), [(20, 25)]);
    }

    #[test]
    fn pending_gpu_write_union_preserves_every_declared_byte() {
        let declarations = [(50, 100), (50, 100), (75, 150), (300, 350), (200, 300), (0, 25), (10, 90), (149, 201), (0, 0), (500, 550)];
        for ordered in [declarations.to_vec(), declarations.into_iter().rev().collect()] {
            let mut ranges = Vec::new();
            let mut covered = vec![false; 600];
            for (from, to) in ordered {
                covered[from..to].fill(true);
                union_pending(&mut ranges, from, to);
                assert!(ranges.windows(2).all(|pair| pair[0].1 < pair[1].0));
                for (i, &wanted) in covered.iter().enumerate() {
                    assert_eq!(ranges.iter().any(|&(a, b)| a <= i && i < b), wanted, "byte={i}");
                }
            }
            assert_eq!(ranges, [(0, 350), (500, 550)]);
        }
    }

    #[test]
    fn duplicate_gpu_write_declarations_do_not_accumulate() {
        let mut ranges = Vec::new();
        for _ in 0..10000 {
            union_pending(&mut ranges, 16, 65536);
            union_pending(&mut ranges, 256, 16384);
        }
        assert_eq!(ranges, [(16, 65536)]);
        assert_eq!(uncovered_ranges(0, 65536, &ranges), [(0, 16)]);
    }

    #[test]
    fn covered_gpu_write_declarations_need_no_new_snapshot() {
        let mut ranges = Vec::new();
        union_pending(&mut ranges, 10, 20);
        union_pending(&mut ranges, 30, 40);
        assert!(pending_covers(&ranges, 10, 20));
        assert!(pending_covers(&ranges, 12, 19));
        assert!(!pending_covers(&ranges, 20, 21));
        assert!(!pending_covers(&ranges, 19, 31));
        assert!(!pending_covers(&ranges, 30, 41));
        union_pending(&mut ranges, 20, 30);
        assert!(pending_covers(&ranges, 10, 40));
    }

    #[test]
    fn snapshot_pool_reuses_small_buffers_without_consuming_large_ones() {
        let mut pool = SnapshotPool::default();
        let mut small = Vec::with_capacity(64);
        small.extend_from_slice(&[3; 64]);
        let small_ptr = small.as_ptr();
        let large = Vec::with_capacity(65536);
        let large_ptr = large.as_ptr();
        pool.put(small);
        pool.put(large);
        assert_eq!(pool.bytes, 64 + 65536);
        let reused = pool.get(40);
        assert_eq!(reused.as_ptr(), small_ptr);
        assert!(reused.is_empty());
        assert_eq!(pool.bytes, 65536);
        assert!(pool.get(16).capacity() < 65536);
        let reused_large = pool.get(65536);
        assert_eq!(reused_large.as_ptr(), large_ptr);
        assert_eq!(pool.bytes, 0);
        for _ in 0..100 { pool.put(Vec::with_capacity(64)); }
        assert_eq!(pool.bytes, 64 * 64);
        assert!(pool.by_capacity.values().all(|v| !v.is_empty()));
    }

    #[test]
    fn sparse_snapshot_keeps_old_bytes_and_joins_only_captured_ranges() {
        let mut snapshot = Snapshot::default();
        snapshot.capture(20, &[2; 10]);
        snapshot.capture(0, &[1; 10]);
        let mut scratch = Vec::new();
        assert_eq!(snapshot.piece(3, 8, &mut scratch), Some(&[1; 5][..]));
        assert!(snapshot.piece(5, 25, &mut scratch).is_none());
        snapshot.capture(10, &[3; 10]);
        assert_eq!(snapshot.piece(5, 25, &mut scratch), Some(&[1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2][..]));
        assert!(snapshot.piece(5, 31, &mut scratch).is_none());
        snapshot.clear();
        assert!(snapshot.is_empty());
    }

    #[test]
    fn persistent_snapshot_preserves_pending_bytes_and_reuses_exact_coverage() {
        let mut snapshot = Snapshot::default();
        snapshot.capture(0, &[1; 8]);
        snapshot.capture(16, &[2; 8]);
        snapshot.finish(true);
        assert_eq!(snapshot.missing(0,32), [(8,16),(24,32)]);
        snapshot.refresh(4,&[3;16]);
        let mut scratch = Vec::new();
        assert_eq!(snapshot.piece(0,8,&mut scratch), Some(&[1,1,1,1,3,3,3,3][..]));
        assert_eq!(snapshot.piece(16,24,&mut scratch), Some(&[3,3,3,3,2,2,2,2][..]));
        // A later declaration captures only the uncovered gap. The
        // earlier pending bytes must keep their previous baseline.
        for (a,b) in snapshot.missing(4,20) { snapshot.capture(a, &vec![4;b-a]); }
        assert_eq!(snapshot.piece(4,20,&mut scratch), Some(&[3,3,3,3,4,4,4,4,4,4,4,4,3,3,3,3][..]));
        snapshot.refresh(5,&[9,8,7]);
        snapshot.finish(true);
        assert_eq!(snapshot.missing(0,24), []);
        assert_eq!(snapshot.piece(4,8,&mut scratch), Some(&[3,9,8,7][..]));
        snapshot.clear();
        assert!(snapshot.is_empty());
    }

    #[test]
    fn idle_requires_both_time_and_frames() {
        let t = Instant::now();
        let used = (t, 20);
        assert!(!idle((t + Duration::from_secs(10), 24), used));
        assert!(!idle((t + Duration::from_secs(11), 23), used));
        assert!(!idle((t + Duration::from_secs(11), 19), used));
        assert!(idle(
            (t + Duration::from_secs(10) + Duration::from_millis(1), 24),
            used
        ));
    }

    #[test]
    fn shadow_buffer_transitions_state_when_requested() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(shadow) = loose(0x10000000, 0x10001000) else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        shadow.ensure_state(&list, 0x8);
        shadow.ensure_state(&list, 0x1);
        shadow.ensure_state(&list, 0x1);
        list.close();
        host.execute(&[&list]);
    }
}
