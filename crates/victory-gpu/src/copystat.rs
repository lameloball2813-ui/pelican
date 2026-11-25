//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// GPU→title buffer writeback, split into its expensive CPU phases.
    BackPieces,
    BackChanged,
    BackScanUs,
    BackReadUs,
    BackMergeUs,
    BackWriteUs,
    BackGuardUs,
    BackKernelWriteUs,
    SnapshotUs,
    SnapshotBytes,
    RefreshUs,
    ReadFallbackUs,
    RegionQueries,
    RegionQueryUs,
    PhysicalReadBytes,
    PhysicalReadUs,
    PhysicalWriteBytes,
    PhysicalWriteUs,
    PhysicalWriteRuns,
    BackKernelWriteRuns,
    CardUploadBytes,
    CardDownloadBytes,
    CardUploadCopies,
    CardDownloadCopies,
    CardComparedBytes,
    CardUnchangedBytes,
}
const COUNTERS: usize = C::CardUnchangedBytes as usize + 1;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range: requests, requested bytes, actual refreshed bytes, time.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u64)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// Per shader, time and slots visited in conservative bindless buffer scans.
/// This separates command-recording work from submission refresh/copy costs.
static HEAP_SCANS: Mutex<Option<HashMap<String, (u64, u64, u64, u64)>>> = Mutex::new(None);

pub(crate) fn heap_scan(hash: &str, visited: u64, prepared: u64, us: u64) {
    let mut held = HEAP_SCANS.lock().unwrap_or_else(|e| e.into_inner());
    let map = held.get_or_insert_with(HashMap::new);
    if let Some(e) = map.get_mut(hash) {
        e.0 += 1;
        e.1 += visited;
        e.2 += prepared;
        e.3 += us;
    } else {
        map.insert(hash.to_owned(), (1, visited, prepared, us));
    }
}

/// A request for `len` bytes refreshed only `copied` bytes after tracking.
pub(crate) fn range(src: usize, len: usize, copied: usize, us: u64) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, 0));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += copied as u64;
    e.3 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    );
    tracing::info!(
        "GPU writeback per frame: {:.0} pieces, {:.0} changed; {:.1} ms scanning, {:.1} ms reading title memory, {:.1} ms merging, {:.1} ms writing title memory ({:.1} ms guards, {:.1} ms kernel writes)",
        c[C::BackPieces as usize],
        c[C::BackChanged as usize],
        ms(c[C::BackScanUs as usize]),
        ms(c[C::BackReadUs as usize]),
        ms(c[C::BackMergeUs as usize]),
        ms(c[C::BackWriteUs as usize]),
        ms(c[C::BackGuardUs as usize]),
        ms(c[C::BackKernelWriteUs as usize]),
    );
    tracing::info!(
        "buffer phases per frame: {:.1} ms refresh, {:.1} ms snapshots ({:.1} MB copied), {:.1} ms failed-read fallback, {:.0} region queries {:.1} ms",
        ms(c[C::RefreshUs as usize]), ms(c[C::SnapshotUs as usize]),
        mb(c[C::SnapshotBytes as usize]), ms(c[C::ReadFallbackUs as usize]),
        c[C::RegionQueries as usize], ms(c[C::RegionQueryUs as usize]),
    );
    tracing::info!("stable physical alias reads per frame: {:.1} MB, {:.1} ms", mb(c[C::PhysicalReadBytes as usize]), ms(c[C::PhysicalReadUs as usize]));
    tracing::info!("stable physical alias writes per frame: {:.1} MB, {:.1} ms, {:.0} runs; kernel fallback {:.0} runs", mb(c[C::PhysicalWriteBytes as usize]), ms(c[C::PhysicalWriteUs as usize]), c[C::PhysicalWriteRuns as usize], c[C::BackKernelWriteRuns as usize]);
    tracing::info!("VRAM shadow transfers per frame: {:.2} MB up ({:.1} copies), {:.2} MB down ({:.1} copies); {:.1} MB live card backing", mb(c[C::CardUploadBytes as usize]), c[C::CardUploadCopies as usize], mb(c[C::CardDownloadBytes as usize]), c[C::CardDownloadCopies as usize], mb(crate::shadow::card_bytes() as f64));
    tracing::info!("exact GPU comparisons per frame: {:.2} MB scanned on card, {:.2} MB unchanged (CPU scan omitted)", mb(c[C::CardComparedBytes as usize]), mb(c[C::CardUnchangedBytes as usize]));
    let scans = HEAP_SCANS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut scans: Vec<_> = scans.into_iter().flatten().collect();
    let totals = scans.iter().fold((0, 0, 0, 0), |a, (_, b)| (a.0+b.0, a.1+b.1, a.2+b.2, a.3+b.3));
    if totals.0 != 0 {
        tracing::info!("heap scans per frame: {:.1} calls, {:.0} visited, {:.0} prepared, {:.1} ms recording", totals.0 as f64 / frames, totals.1 as f64 / frames, totals.2 as f64 / frames, ms(totals.3 as f64 / frames));
        scans.sort_unstable_by_key(|(_, e)| std::cmp::Reverse(e.3));
        for (hash, (calls, slots, prepared, us)) in scans.into_iter().take(8) {
            tracing::info!("  heap scan shader {hash}: {:.1} calls/frame, {:.0} slots/frame, {:.0} prepared/frame, {:.1} ms/frame", calls as f64 / frames, slots as f64 / frames, prepared as f64 / frames, ms(us as f64 / frames));
        }
    }
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, _, copied, us))| std::cmp::Reverse((us, copied)));
    for ((src, len), (n, requested, copied, us)) in ranges.into_iter().take(12) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  refreshed {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB requested, {:.2} MB actual, {:.2} ms/frame",
            n as f64 / frames,
            requested as f64 / 1e6 / frames,
            copied as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
