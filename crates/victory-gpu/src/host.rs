//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    null_root: OnceLock<Option<(ID3D12Resource, u64)>>,
    card_upload: OnceLock<TransferList>,
    card_download: OnceLock<TransferList>,
    card_scan: OnceLock<Option<crate::card_scan::Scan>>,
    /// Match the D3D12 device's actual adapter, not the first DXGI adapter.
    card_adapter: Option<IDXGIAdapter3>,
    card_admission: Mutex<()>,
}

type TransferList = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);
type ShadowRanges = Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>;
type ScanRanges = Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize, usize)>)>;

/// Split only complete title pages, with a 16-byte aligned GPU address,
/// into GPU comparisons. Keep the exact head/tail and any excess in the
/// ordinary readback. No widening, no assumed initialized padding.
fn card_download_plan(downloads: &ShadowRanges, enabled: bool) -> (ShadowRanges, ScanRanges) {
    use crate::card_scan::{PAGE, MAX_FLAGS};
    let mut fallback = Vec::new();
    let mut scan = Vec::new();
    let mut flags = 0;
    for (shadow, ranges) in downloads {
        let mut plain = Vec::new();
        let mut compare = Vec::new();
        let base = shadow.card_start();
        for &(a,b) in ranges {
            if !enabled || shadow.card_baseline().is_none() || base % 16 != 0 || b as u64 > 1u64<<32 {
                plain.push((a,b)); continue;
            }
            let first = (base+a).next_multiple_of(PAGE)-base;
            let last = ((base+b)/PAGE*PAGE).saturating_sub(base);
            let pages = last.saturating_sub(first)/PAGE;
            let pages = pages.min(MAX_FLAGS-flags).min(65535);
            if pages == 0 { plain.push((a,b)); continue; }
            let end = first + pages*PAGE;
            if a < first { plain.push((a,first)); }
            compare.push((first,end,flags));
            flags += pages;
            if end < b { plain.push((end,b)); }
        }
        if !plain.is_empty() { fallback.push((shadow.clone(),plain)); }
        if !compare.is_empty() { scan.push((shadow.clone(),compare)); }
    }
    (fallback,scan)
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let card_adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            null_root: OnceLock::new(),
            card_upload: OnceLock::new(),
            card_download: OnceLock::new(),
            card_scan: OnceLock::new(),
            card_adapter,
            card_admission: Mutex::new(()),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_process_reads_match_win32_and_reject_inaccessible_pages() {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const PAGE: usize = 4096;
        let src: Vec<u8> = (0..PAGE * 4).map(|x| x.wrapping_mul(37) as u8).collect();
        let mut from_win = vec![0u8; src.len()];
        let mut from_nt = vec![0u8; src.len()];
        let mut done = 0;
        // SAFETY: readable source; mapped destination; kernel probes both.
        unsafe {
            assert_ne!(ReadProcessMemory(GetCurrentProcess(), src.as_ptr().cast(), from_win.as_mut_ptr().cast(), src.len(), &mut done), 0);
            assert_eq!(done, src.len());
            done = 0;
            let status = NtReadVirtualMemory(GetCurrentProcess(), src.as_ptr().cast(), from_nt.as_mut_ptr().cast(), src.len(), &mut done);
            assert!(status >= 0, "native read failed with NTSTATUS {status:#x}");
            assert_eq!(done, src.len());
            assert_eq!(from_nt, from_win);
            assert_eq!(from_nt, src);

            // Reserve three pages but commit only the outside two. A read
            // crossing the uncommitted middle cannot succeed even if NT
            // transferred part of the readable prefix.
            let reserved = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE, PAGE_NOACCESS);
            assert!(!reserved.is_null());
            let first = VirtualAlloc(reserved, PAGE, MEM_COMMIT, PAGE_READWRITE);
            let last = VirtualAlloc((reserved as usize + 2 * PAGE) as _, PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(first, reserved);
            assert_eq!(last as usize, reserved as usize + 2 * PAGE);
            (first as *mut u8).write_bytes(0x5A, PAGE);
            (last as *mut u8).write_bytes(0xA5, PAGE);
            done = 0;
            let status = NtReadVirtualMemory(GetCurrentProcess(), first.cast(), from_nt.as_mut_ptr().cast(), 3 * PAGE, &mut done);
            assert!(status < 0 || done < 3 * PAGE);
            assert!(done <= 3 * PAGE);
            assert_ne!(VirtualFree(reserved, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn sparse_writeback_keeps_all_unwritten_gaps() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        let p = unsafe { VirtualAlloc(std::ptr::null(), 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!p.is_null());
        struct Release(*mut u8);
        impl Drop for Release { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE); } } }
        let _release = Release(p);
        // A newer CPU value in every gap must survive the GPU writeback,
        // including cases with far more than the old eight-run cutoff.
        unsafe { p.write_bytes(0x73, 65536); }
        let data = vec![0xc5; 65536];
        let runs: Vec<_> = (0..128).map(|i| (i * 503 + 7, i * 503 + 10)).collect();
        assert_eq!(write_mapped_runs(p, &data, &runs), 384);
        let actual = unsafe { std::slice::from_raw_parts(p, 65536) };
        for (i, &value) in actual.iter().enumerate() {
            assert_eq!(value, if runs.iter().any(|&(a,b)| a <= i && i < b) { 0xc5 } else { 0x73 });
        }
    }

    #[test]
    fn page_scoped_writeback_uses_physical_writer_where_valid_and_kernel_for_private_pages() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RESERVE, MEM_COMMIT, MEM_RELEASE, PAGE_READWRITE};
        const PAGE: usize = 64 << 10;
        let dst = unsafe { VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
        assert!(!dst.is_null());
        struct Release(*mut u8);
        impl Drop for Release { fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE); } } }
        let _release = Release(dst);
        let source = vec![0xB6; 3 * PAGE];
        // The middle physical-page candidate is deliberately rejected. The
        // successful candidates write only their exact changed bytes.
        unsafe { dst.write_bytes(0x53, 3 * PAGE); }
        let runs = [(PAGE - 7, PAGE + 5), (PAGE + 64, PAGE + 68),
                    (2 * PAGE + 1, 2 * PAGE + 9), (3 * PAGE - 1, 3 * PAGE)];
        let mut attempted = Vec::new();
        let mut fast_bytes = 0;
        let mut slow_bytes = 0;
        writeback_pages(dst as usize, &runs, |group| {
            let first = group[0].0;
            let last = group.last().unwrap().1;
            assert_eq!(((dst as usize) + first) / PAGE, ((dst as usize) + last - 1) / PAGE);
            attempted.push(((dst as usize) + first) / PAGE);
            let fast = first < PAGE || first >= 2 * PAGE;
            if fast {
                for &(a,b) in group {
                    unsafe { std::ptr::copy_nonoverlapping(source.as_ptr().add(a), dst.add(a), b-a) };
                    fast_bytes += b-a;
                }
            } else {
                for &(a,b) in group {
                    slow_bytes += write_mapped_open(dst.wrapping_add(a), source[a..b].as_ptr(), b-a);
                }
            }
        });
        assert_eq!(attempted.len(), 3);
        assert!(fast_bytes > 0 && slow_bytes > 0);
        assert_eq!(fast_bytes + slow_bytes, runs.iter().map(|(a,b)| b-a).sum::<usize>());
        let actual = unsafe { std::slice::from_raw_parts(dst, 3 * PAGE) };
        for (i,&byte) in actual.iter().enumerate() {
            assert_eq!(byte, if runs.iter().any(|&(a,b)| a<=i && i<b) {0xB6} else {0x53}, "byte {i}");
        }
    }

    #[test]
    #[ignore = "benchmark: run separately from a game process"]
    fn native_vs_win32_memory_read_microbenchmark() {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let src = vec![0x5Au8; 1 << 22];
        let mut dst = vec![0u8; 64 << 10];
        const REPEATS: usize = 4096;
        let bench = |native: bool, dst: &mut Vec<u8>| {
            let started = std::time::Instant::now();
            for i in 0..REPEATS {
                let from = unsafe { src.as_ptr().add((i * 65537) % (src.len() - dst.len())) };
                let mut done = 0;
                // SAFETY: both buffers are mapped for the requested span.
                unsafe {
                    if native {
                        assert!(NtReadVirtualMemory(GetCurrentProcess(), from.cast(), dst.as_mut_ptr().cast(), dst.len(), &mut done) >= 0);
                    } else {
                        assert_ne!(ReadProcessMemory(GetCurrentProcess(), from.cast(), dst.as_mut_ptr().cast(), dst.len(), &mut done), 0);
                    }
                }
                assert_eq!(done, dst.len());
            }
            started.elapsed()
        };
        bench(false, &mut dst);
        bench(true, &mut dst);
        let win = bench(false, &mut dst);
        let nt = bench(true, &mut dst);
        eprintln!("{REPEATS} x 64KiB kernel reads: Win32 {win:?}, direct NT {nt:?}");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn null_root_view_is_aligned() {
        let Some(host) = super::host() else {
            return;
        };
        let gpu = host.null_root_view().unwrap();
        assert_ne!(gpu, 0);
        assert!(gpu.is_multiple_of(256));
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    pub(crate) twins: Mutex<crate::twin::Uses>,
    // Buffers decay to COMMON between ExecuteCommandLists calls. Record
    // their first/last states locally rather than mutating a global state
    // while other command lists are still being recorded.
    buffer_states: Mutex<BufferStates>,
    /// The list that runs those transitions just before this one.
    prelude: OnceLock<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Reusable bounded indirect arguments. Original title argument/count
    /// buffers are read-only to the clamp, not mutated by compatibility code.
    indirect_scratch: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

#[cfg(feature = "diagnostics")]
static LIVE_LISTS: AtomicU32 = AtomicU32::new(0);
#[cfg(feature = "diagnostics")]
static LIVE_PRELUDES: AtomicU32 = AtomicU32::new(0);

/// Snapshot counters without constructing a host device or recording work.
#[cfg(feature = "diagnostics")]
pub fn diagnostic_memory() {
    let [runtime_graphics_bytes, title_graphics_bytes] = crate::resource::memory_bytes();
    let (host_texture_bytes, host_texture_count) = crate::resource::host_texture_totals();
    tracing::info!(runtime_graphics_bytes, title_graphics_bytes,
        texture_staging_bytes = crate::texture::staging_bytes(),
        host_texture_bytes, host_texture_count,
        live_lists = LIVE_LISTS.load(Ordering::Relaxed),
        live_preludes = LIVE_PRELUDES.load(Ordering::Relaxed),
        "periodic GPU memory accounting (texture Arc ownership is a lower bound)");
}

#[cfg(feature = "diagnostics")]
#[track_caller]
fn trace_mapped_buffer(kind: &str, size: u64, data: *mut std::ffi::c_void) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_MEMORY_ACCOUNTING")) || size < (1 << 20) {
        return;
    }
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a successfully mapped buffer and writable query result.
    if unsafe { VirtualQuery(data, &mut info, std::mem::size_of_val(&info)) } != 0 {
        tracing::info!(kind, size, data = format_args!("{data:p}"),
            allocation_base = format_args!("{:p}", info.AllocationBase),
            region_bytes = info.RegionSize, protect = info.Protect,
            caller = %std::panic::Location::caller(), "mapped host buffer allocation");
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for HostList {
    fn drop(&mut self) {
        LIVE_LISTS.fetch_sub(1, Ordering::Relaxed);
        if self.prelude.get().is_some() {
            LIVE_PRELUDES.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;
type BufferStates = std::collections::HashMap<usize, (u32, u32, ID3D12Resource)>;
type BufferExecution = std::collections::HashMap<usize, u32>;

// A buffer may be read through several different view types in one
// operation. Use the compatible GENERIC_READ union, never combine UAV
// with read-only states.
fn buffer_state(wanted: u32) -> u32 {
    const READ: u32 = 0x1 | 0x2 | 0x40 | 0x80 | 0x200 | 0x800;
    if wanted != 0 && wanted & !READ == 0 { READ } else { wanted }
}

fn buffer_before(execution: &mut BufferExecution, key: usize, last: u32) -> u32 {
    execution.insert(key, last).unwrap_or(0)
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Exact union of refresh requests per retained shadow and title-write
    /// epoch. A fully covered subview needs no additional deferred work.
    refreshed: RangeCoverage,
    /// UAV ranges already declared by this list since its last operation
    /// that may change title memory. Repeated binds do not need another
    /// identical pre-submission snapshot or pending-write declaration.
    gpu_written: RangeCoverage,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Changes at every recorded CPU title-memory write. Scratch data
    /// sampled before such a write cannot be reused after it.
    title_write_epoch: u64,
}

impl Deferred {
    /// Shared scalar/bulk registration: retain exactly the same range
    /// coverage and Step order, independent of how many times the mutex is
    /// acquired by its caller.
    fn record_refresh(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if self.refreshed.insert(key, address, len) {
            self.copies.push(Step::Refresh(shadow.clone(), address, len));
        }
    }
}

/// Apply the scalar registration primitive in the exact original read order.
/// Caller controls the title-write boundary and lock lifetime.
fn record_refresh_run(
    deferred: &mut Deferred,
    ranges: &[(std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>)],
) {
    for (shadow, intervals) in ranges {
        for &(start, end) in intervals {
            let len = (end - start).min(shadow.left(start));
            deferred.record_refresh(shadow, start as usize, len as usize);
        }
    }
}

/// Recording-time coverage, not a write-watch cache. Never infer contents
/// from this map across a title write, list reset, or submission.
#[derive(Default)]
struct RangeCoverage(std::collections::HashMap<usize, Vec<(usize, usize)>>);

impl RangeCoverage {
    fn clear(&mut self) { self.0.clear(); }

    /// Return true only when this request contributes new bytes.
    /// Keep partially new requests intact, including their original position
    /// relative to snapshot declarations and other shadows.
    fn insert(&mut self, key: usize, from: usize, len: usize) -> bool {
        let Some(to) = from.checked_add(len) else { return false };
        if from == to { return false; }
        let ranges = self.0.entry(key).or_default();
        let i = ranges.partition_point(|&(_, end)| end <= from);
        if ranges.get(i).is_some_and(|&(a,b)| a <= from && to <= b) {
            return false;
        }
        Batch::add_interval(ranges, from, to);
        true
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The host GPU's supported wave-lane range `(WaveLaneCountMin,
    /// WaveLaneCountMax)` from `D3D12_FEATURE_D3D12_OPTIONS1` (defaults to
    /// `(32, 64)` when the query fails, so nothing is refused for it).
    /// Note: WARP (`VICTORY_D3D12_WARP=1`) reports `(4, 4)` even on 64-lane
    /// hardware, so shaders requiring wider waves are refused there too.
    pub(crate) fn wave_lanes(&self) -> (u32, u32) {
        static LANES: OnceLock<(u32, u32)> = OnceLock::new();
        *LANES.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let (min, max) = if ok
                && options.WaveLaneCountMin > 0
                && options.WaveLaneCountMax >= options.WaveLaneCountMin
            {
                (options.WaveLaneCountMin, options.WaveLaneCountMax)
            } else {
                (32, 64)
            };
            if max < 32 {
                tracing::info!(
                    min,
                    max,
                    "host wave lanes (software/WARP adapter reports narrow waves; wider-wave shaders will be refused)"
                );
            } else {
                tracing::info!(min, max, "host wave lanes");
            }
            (min, max)
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            // Many title command lists never need a state-transition prelude.
            // Allocate its independent command allocator only if a submitted
            // list actually needs a barrier. This preserves batch ordering.
            #[cfg(feature = "diagnostics")]
            LIVE_LISTS.fetch_add(1, Ordering::Relaxed);
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                buffer_states: Mutex::default(),
                prelude: OnceLock::new(),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                indirect_scratch: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        let mut buffer_execution = BufferExecution::new();
        for list in runs {
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            if timing() {
                // Timing mode submits each list and its prelude separately.
                buffer_execution.clear();
            }
            if let Some(prelude) = list.prelude(&mut buffer_execution) {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        // Collect before submitting: all GPU writes must be fetched after
        // the last dependent list, not before another dispatch reads them.
        let mut back: ShadowRanges = Vec::new();
        let mut back_index = std::collections::HashMap::new();
        let mut cards = Vec::new();
        let mut card_seen = std::collections::HashSet::new();
        for list in runs {
            for step in &list.deferred().copies {
                let shadow = match step {
                    Step::Refresh(shadow, ..) | Step::GpuWrite(shadow, ..) => shadow,
                    _ => continue,
                };
                if shadow.card_staging().is_some() && card_seen.insert(std::sync::Arc::as_ptr(shadow) as usize) {
                    cards.push(shadow.clone());
                }
                if let Step::GpuWrite(shadow, address, len) = step
                    && let Some(end) = address.checked_add(*len)
                {
                    let (at, new) = Batch::slot(&mut back_index, back.len(), shadow);
                    if new { back.push((shadow.clone(), Vec::with_capacity(1))); }
                    Batch::add_interval(&mut back[at].1, *address, end);
                }
            }
        }
        let uploads: ShadowRanges = cards.into_iter().filter_map(|shadow| {
            let dirty = shadow.take_card_uploads();
            (!dirty.is_empty()).then_some((shadow, dirty))
        }).collect();
        let downloads: ShadowRanges = back.iter().filter_map(|(shadow, ranges)| {
            shadow.card_staging()?;
            let offsets = ranges.iter().map(|&(from, to)| shadow.card_offsets(from, to - from))
                .filter(|&(a,b)| a < b).collect::<Vec<_>>();
            (!offsets.is_empty()).then_some((shadow.clone(), offsets))
        }).collect();
        // Baseline is a separate image of CPU staging, not of the live
        // card resource. Initializing a newly declared write interval must
        // not erase a prior GPU write outside an earlier declaration.
        let mut baseline_uploads: ShadowRanges = Vec::new();
        for (shadow, dirty) in &uploads {
            // This exact dirty interval will be copied to the baseline
            // from the freshly uploaded card bytes, not over PCIe twice.
            shadow.baseline_uploads(dirty,&[]);
        }
        for (shadow, needed) in &downloads {
            let ranges = shadow.baseline_uploads(&[],needed);
            if !ranges.is_empty() { baseline_uploads.push((shadow.clone(),ranges)); }
        }
        let scanner = if downloads.iter().any(|(shadow,_)| shadow.card_baseline().is_some()) {
            self.card_scan.get_or_init(|| crate::card_scan::Scan::new(self)).as_ref()
        } else { None };
        let (fallback, scans) = card_download_plan(&downloads,scanner.is_some());
        let upload = self.card_transfer(&uploads, true, &baseline_uploads, None);
        let download = self.card_transfer(&fallback, false, &Vec::new(), scanner.map(|s|(s,&scans)));
        let prepared = std::time::Instant::now();
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                // Separate ExecuteCommandLists calls: buffers decay to
                // COMMON between groups. No re-upload between title lists.
                if i == 0 && upload.is_some() {
                    self.queue.ExecuteCommandLists(&[upload.clone()]);
                }
                self.queue.ExecuteCommandLists(group);
                if i + 1 == groups.len() && download.is_some() {
                    self.queue.ExecuteCommandLists(&[download.clone()]);
                }
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        if WaitForSingleObject(self.event, INFINITE) != windows::Win32::Foundation::WAIT_OBJECT_0 {
                            fail(format_args!("host submission fence wait failed"));
                        }
                    }
                    Err(e) => fail(format_args!("host submission not waited for ({e})")),
                }
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        for (shadow, ranges) in &back {
            let scan_ranges = scans.iter().find(|(s,_)| std::sync::Arc::ptr_eq(s,shadow)).map(|(_,r)| r.as_slice());
            for &(from, to) in ranges {
                let written = if let (Some(scanner),Some(spans)) = (scanner,scan_ranges) {
                    shadow.copy_back_filtered(from,to-from,|at,n| {
                        if n != crate::card_scan::PAGE { return false; }
                        let i = spans.partition_point(|&(_,end,_)| end <= at);
                        spans.get(i).is_some_and(|&(a,b,flag)| a <= at && at+n <= b &&
                            (at-a)%n == 0 && scanner.unchanged(flag+(at-a)/n))
                    })
                } else { shadow.copy_back(from, to - from) };
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
    }

    /// The submission owns both private transfer lists until its fence.
    /// All ranges are offsets into one stable pair of resources. Explicit
    /// COMMON boundaries also cover buffers not used by the main lists.
    fn card_transfer(&self, ranges: &ShadowRanges, upload: bool, baseline_uploads: &ShadowRanges,
        scan: Option<(&crate::card_scan::Scan, &ScanRanges)>) -> Option<ID3D12CommandList> {
        if ranges.is_empty() && baseline_uploads.is_empty() && scan.is_none_or(|(_,s)| s.is_empty()) { return None; }
        let slot = if upload { &self.card_upload } else { &self.card_download };
        let (allocator, list) = slot.get_or_init(|| unsafe {
            (|| -> windows::core::Result<TransferList> {
                let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })().unwrap_or_else(|e| fail(format_args!("no card transfer list ({e})")))
        });
        let recorded = unsafe {
            allocator.Reset().and_then(|()| list.Reset(allocator, None))
        };
        if let Err(e) = recorded { fail(format_args!("card transfer reset failed ({e})")); }
        let mut bytes = 0u64;
        let mut copies = 0u64;
        for (shadow, intervals) in ranges {
            let staging = shadow.card_staging().expect("card staging retained");
            let (src, dst) = if upload { (staging, &shadow.buffer) } else { (&shadow.buffer, staging) };
            let common = D3D12_RESOURCE_STATE_COMMON;
            let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
            let write = D3D12_RESOURCE_STATE_COPY_DEST;
            unsafe {
                list.ResourceBarrier(&[transition(src, common, read), transition(dst, common, write)]);
                for &(a, b) in intervals {
                    if a >= b { continue; }
                    list.CopyBufferRegion(dst, a as u64, src, a as u64, (b-a) as u64);
                    bytes += (b-a) as u64;
                    copies += 1;
                }
                list.ResourceBarrier(&[transition(src, read, common), transition(dst, write, common)]);
            }
            // Keep the baseline equal to staging on upload and fallback
            // readback, including partial pages. Copy locally on the card.
            if let Some(baseline) = shadow.card_baseline() {
                let local = if upload { dst } else { src };
                unsafe {
                    list.ResourceBarrier(&[transition(local,common,read),transition(baseline,common,write)]);
                    for &(a,b) in intervals { if a<b { list.CopyBufferRegion(baseline,a as u64,local,a as u64,(b-a) as u64); } }
                    list.ResourceBarrier(&[transition(local,read,common),transition(baseline,write,common)]);
                }
            }
        }
        for (shadow,intervals) in baseline_uploads {
            let src=shadow.card_staging().expect("baseline staging");
            let dst=shadow.card_baseline().expect("baseline retained");
            let common=D3D12_RESOURCE_STATE_COMMON;
            let read=D3D12_RESOURCE_STATE_COPY_SOURCE;
            let write=D3D12_RESOURCE_STATE_COPY_DEST;
            unsafe {
                list.ResourceBarrier(&[transition(src,common,read),transition(dst,common,write)]);
                for &(a,b) in intervals {
                    if a<b {
                        list.CopyBufferRegion(dst,a as u64,src,a as u64,(b-a) as u64);
                        bytes+=(b-a) as u64; copies+=1;
                    }
                }
                list.ResourceBarrier(&[transition(src,read,common),transition(dst,write,common)]);
            }
        }
        if let Some((scanner,scans))=scan && !scans.is_empty() {
            scanner.begin(list);
            for (shadow,spans) in scans {
                scanner.record(list,&shadow.buffer,shadow.card_baseline().expect("scan baseline"),shadow.card_staging().expect("scan staging"),spans);
                #[cfg(feature="diagnostics")]
                crate::copystat::add(crate::copystat::C::CardComparedBytes,spans.iter().map(|&(a,b,_)|(b-a) as u64).sum());
            }
            scanner.end(list);
        }
        if let Err(e) = unsafe { list.Close() } { fail(format_args!("card transfer close failed ({e})")); }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(if upload { C::CardUploadBytes } else { C::CardDownloadBytes }, bytes);
            add(if upload { C::CardUploadCopies } else { C::CardDownloadCopies }, copies);
        }
        let _ = (bytes, copies);
        Some(list.cast().unwrap_or_else(|e| fail(format_args!("card transfer cast failed ({e})"))))
    }

    #[cfg(test)]
    pub(crate) fn card_scan_flag(&self,flag:usize)->Option<bool> {
        Some(self.card_scan.get()?.as_ref()?.unchanged(flag))
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            let res_desc = resource.GetDesc();
            let mut patched = desc.copied();
            if res_desc.Format == DXGI_FORMAT_R16_TYPELESS {
                let d = patched.get_or_insert_with(|| D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: DXGI_FORMAT_R16_UNORM,
                    ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                    ..Default::default()
                });
                if matches!(d.Format.0, 0 | 86 | 115) {
                    d.Format = DXGI_FORMAT_R16_UNORM;
                }
            }
            self.device
                .CreateRenderTargetView(resource, patched.as_ref().map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A 64 KiB zeroed buffer for unset or null root CBVs and SRVs, so a
    /// freshly bound root signature never leaves a root descriptor undefined.
    pub(crate) fn null_root_view(&self) -> Option<u64> {
        self.null_root
            .get_or_init(|| {
                let (buffer, data) = self.shared_buffer(1 << 16)?;
                name(&buffer, || "null root view".into());
                // safety: `shared_buffer(1 << 16)` returns a mapped 64 KiB buffer.
                unsafe {
                    data.write_bytes(0, 1 << 16);
                    Some((buffer.clone(), buffer.GetGPUVirtualAddress()))
                }
            })
            .as_ref()
            .map(|(_, gpu)| *gpu)
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        crate::twin::finish(self);
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        *self.deferred() = Deferred::default();
        // The last GPU submission was waited for before reset, and this
        // list's shader-written shadows are no longer retained. Only now
        // may unused loose host copies be checked for safe reclamation.
        crate::shadow::let_go_idle();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        *self.twins.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        self.buffer_states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let before = self.move_texture_state(&copy, wanted)?;
        Some((texture, before))
    }

    fn move_texture_state(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// Internal texture transitions use the same Arc-keyed state tracker
    /// and submission prelude as title textures.
    pub(crate) fn move_texture(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        self.move_texture_state(copy, wanted).map(|before| transition(
            &copy.texture, D3D12_RESOURCE_STATES(before as i32), D3D12_RESOURCE_STATES(wanted as i32)))
    }

    /// A buffer's first use is handled by the prelude. Subsequent uses
    /// transition from the state recorded in this list, not another thread's.
    pub(crate) fn ensure_buffer_state(&self, resource: &ID3D12Resource, wanted: u32) {
        let mut states = self.buffer_states.lock().unwrap_or_else(|e| e.into_inner());
        self.ensure_buffer_state_locked(&mut states, resource, wanted);
    }

    /// Register a certified SRV read segment under one state-map lock.
    /// Preserve every first-use and transition in original shadow order;
    /// callers must not batch across a heap PlanStep::Write.
    pub(crate) fn ensure_shadow_read_states(
        &self,
        ranges: &[(std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>)],
    ) {
        let mut states = self.buffer_states.lock().unwrap_or_else(|e| e.into_inner());
        for (shadow, _) in ranges {
            self.ensure_buffer_state_locked(&mut states, &shadow.buffer, 0x40 | 0x80);
        }
    }

    /// Both scalar and run-based registration use one exact state machine.
    fn ensure_buffer_state_locked(&self, states: &mut BufferStates, resource: &ID3D12Resource, wanted: u32) {
        let wanted = buffer_state(wanted);
        let key = resource.as_raw() as usize;
        match states.entry(key) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, resource.clone()));
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    let barrier = transition(resource, D3D12_RESOURCE_STATES(before as i32), D3D12_RESOURCE_STATES(wanted as i32));
                    // SAFETY: the caller owns the open list; the map keeps
                    // this resource alive through execution.
                    unsafe { self.list.ResourceBarrier(&[barrier]) };
                }
            }
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self, execution: &mut BufferExecution) -> Option<ID3D12CommandList> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        let buffers = self.buffer_states.lock().unwrap_or_else(|e| e.into_inner());
        barriers.extend(buffers.iter().filter_map(|(&key, (first, last, resource))| {
            let before = buffer_before(execution, key, *last);
            (before != *first).then(|| transition(resource, D3D12_RESOURCE_STATES(before as i32), D3D12_RESOURCE_STATES(*first as i32)))
        }));
        if barriers.is_empty() {
            return None;
        }
        let (allocator, list) = self.prelude.get_or_init(|| {
            let host = host().expect("host exists for a live list");
            // SAFETY: a private allocator and list, initially closed. The
            // submission lock prevents concurrent recording of preludes.
            let made = unsafe {
                (|| -> windows::core::Result<_> {
                    let allocator: ID3D12CommandAllocator = host.device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                        0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                    list.Close()?;
                    Ok((allocator, list))
                })()
            };
            let pair = made.unwrap_or_else(|e| fail(format_args!("no host state prelude ({e})")));
            #[cfg(feature = "diagnostics")]
            LIVE_PRELUDES.fetch_add(1, Ordering::Relaxed);
            pair
        });
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                list.ResourceBarrier(&barriers);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred.title_write_epoch += 1;
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    pub(crate) fn title_write_epoch(&self) -> u64 { self.deferred().title_write_epoch }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        deferred.record_refresh(shadow, address, len);
    }

    /// Record an ordered run of certified heap SRV reads under one lock.
    /// The caller has already ensured resource states. This deliberately does
    /// not coalesce requests across shadows, reorder intervals or cross a
    /// PlanStep::Write (which must remain outside the lock and between runs).
    pub(crate) fn defer_refresh_run(
        &self,
        ranges: &[(std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>)],
    ) {
        let mut deferred = self.deferred();
        record_refresh_run(&mut deferred, ranges);
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if deferred.gpu_written.insert(key, address, len) {
            deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
            name(&buffer, || "list scratch".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            s.chunks.push((buffer, data as usize, gpu));
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, (data + at as usize) as *mut u8))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// One scratch argument buffer per list, reused between indirect calls.
    /// When it grows, buffer-state tracking retains the old resource until
    /// every command naming it has completed and the list is reset.
    pub(crate) fn indirect_buffer(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut slot = self.indirect_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, capacity)) = &*slot {
            if *capacity >= bytes { return Some(buffer.clone()); }
        }
        let capacity = bytes.checked_add(65535)? & !65535;
        // Private scratch has no CPU consumer, no title alias, and is
        // completely produced by the clamp before ExecuteIndirect. Keep it
        // local to the GPU rather than fetching its records across PCIe.
        let host = host()?;
        let buffer = host.local_uav_buffer(capacity.max(65536))
            .or_else(|| host.shared_buffer(capacity.max(65536)).map(|(b, _)| b))?;
        name(&buffer, || format!("list indirect scratch {capacity:#x}"));
        *slot = Some((buffer.clone(), capacity.max(65536)));
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.title_write_epoch += 1;
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            let copied = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data.
                    // Only the ordered prefix through this shadow is needed
                    // for its snapshot. Later entries remain in first-seen
                    // order until their own snapshot or a title-write barrier.
                    batch.flush_through(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot_started = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    add(C::SnapshotUs, snapshot_started.elapsed().as_micros() as u64);
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        staging.refresh();
                    }
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into
/// disjoint overlapping/adjacent intervals. Previously the first and last
/// requests formed one enclosing range, which also scanned untouched gaps
/// between distant views of the same large title buffer.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: std::collections::VecDeque<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>,
    refresh_base: usize,
    /// Position of each shadow in `refreshes`. The dynamic resource heap can
    /// supply thousands of refreshes per list; a linear pointer search for
    /// every request became quadratic in the number of distinct buffers.
    /// Keep the original first-seen ordering for flush and all interval data.
    refresh_index: std::collections::HashMap<usize, usize>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

impl Batch {
    /// Return a stable entry index, recording the current append position for
    /// a newly observed Arc. Key by allocation identity, never title address:
    /// distinct shadow copies may refer to overlapping title memory.
    fn slot<T>(index: &mut std::collections::HashMap<usize, usize>,
        next: usize, value: &std::sync::Arc<T>) -> (usize, bool) {
        use std::collections::hash_map::Entry;
        match index.entry(std::sync::Arc::as_ptr(value) as usize) {
            Entry::Occupied(found) => (*found.get(), false),
            Entry::Vacant(empty) => {
                empty.insert(next);
                (next, true)
            }
        }
    }

    /// Insert `[from, to)` into sorted non-overlapping intervals without
    /// adding any bytes outside the union of the requested ranges.
    fn add_interval(intervals: &mut Vec<(usize, usize)>, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let i = intervals.partition_point(|&(_, end)| end < from);
        let mut merged = (from, to);
        let mut j = i;
        while j < intervals.len() && intervals[j].0 <= merged.1 {
            merged.0 = merged.0.min(intervals[j].0);
            merged.1 = merged.1.max(intervals[j].1);
            j += 1;
        }
        intervals.splice(i..j, [merged]);
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let Some(end) = address.checked_add(len) else {
            return;
        };
        if address == end { return; }
        let (at, new) = Self::slot(&mut self.refresh_index, self.refresh_base + self.refreshes.len(), shadow);
        if new {
            self.refreshes.push_back((shadow.clone(), Vec::with_capacity(1)));
        }
        Self::add_interval(&mut self.refreshes[at - self.refresh_base].1, address, end);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    fn flush(&mut self) {
        self.flush_prefix(self.refreshes.len());
    }

    /// No GPU work runs until the whole submission has been prepared. Keep
    /// later refreshes queued, but never overtake an earlier shadow (even
    /// if its title address appears disjoint). Explicit title writes still
    /// flush everything. This changes observation timing within a submission,
    /// not the set of required ranges or snapshot ordering on the target.
    fn flush_through(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        // Opt in for matched runtime comparison: asynchronous CPU writes
        // can observe the later refresh timing despite the preserved order.
        #[cfg(feature = "diagnostics")]
        let enabled = {
            static ENABLED: OnceLock<bool> = OnceLock::new();
            *ENABLED.get_or_init(|| crate::switch::on("VICTORY_ORDERED_REFRESH_PREFIX"))
        };
        #[cfg(not(feature = "diagnostics"))]
        let enabled = false;
        if !enabled {
            self.flush();
            return;
        }
        self.flush_through_target(shadow);
    }

    fn flush_through_target(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        if let Some(&at) = self.refresh_index.get(&(std::sync::Arc::as_ptr(shadow) as usize)) {
            self.flush_prefix(at - self.refresh_base + 1);
        }
    }

    fn flush_prefix(&mut self, count: usize) {
        for _ in 0..count {
            let (shadow, intervals) = self.refreshes.pop_front().expect("queued refresh");
            self.refresh_index.remove(&(std::sync::Arc::as_ptr(&shadow) as usize));
            self.refresh_base += 1;
            for (from, to) in intervals {
                let started = std::time::Instant::now();
                let copied = shadow.update(from, to - from);
                let us = started.elapsed().as_micros() as u64;
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::range(from, to - from, copied, us);
                    crate::copystat::add(crate::copystat::C::RefreshUs, us);
                }
                PREPARING[0].fetch_add(us, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            }
        }
        if self.refreshes.is_empty() {
            self.refresh_base = 0;
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 || title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    let _write = title_dst.then(|| crate::watch::begin_kernel_write(dst as usize, len, false));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = !title_dst && read_physical(dst, src, len) || read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let fallback_started = std::time::Instant::now();
    // Something is unmapped. A failed large read can otherwise produce
    // thousands of failing per-page system calls over a reserved hole.
    // Only the host destination path uses this experimental fallback.
    #[cfg(feature = "diagnostics")]
    if !title_dst && region_copy_fallback() {
        copy_failed_regions(dst, src, len);
        crate::copystat::add(crate::copystat::C::ReadFallbackUs, fallback_started.elapsed().as_micros() as u64);
        return;
    }
    copy_failed_pages(dst, src as usize, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ReadFallbackUs, fallback_started.elapsed().as_micros() as u64);
}

type PhysicalReader = unsafe fn(*mut u8, *const u8, usize) -> bool;
static PHYSICAL_READER: OnceLock<PhysicalReader> = OnceLock::new();
type PhysicalWriter = unsafe fn(*mut u8, &[u8], &[(usize, usize)]) -> bool;
static PHYSICAL_WRITER: OnceLock<PhysicalWriter> = OnceLock::new();

/// Install a physical scatter writer; false must mean no writes occurred.
pub fn use_physical_writer(writer: PhysicalWriter) {
    let _ = PHYSICAL_WRITER.set(writer);
}

/// Install the HLE's reader for verified, stable physical section aliases.
/// The callback must never read an unverified/remappable title address and
/// must return false for ranges it cannot completely copy.
pub fn use_physical_reader(reader: PhysicalReader) {
    let _ = PHYSICAL_READER.set(reader);
}

fn read_physical(dst: *mut u8, src: *const u8, len: usize) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| crate::switch::on("VICTORY_PHYSICAL_READ_VIEW")) { return false; }
    let Some(reader) = PHYSICAL_READER.get() else { return false; };
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // Only the independent host-destination path calls this helper.
    let copied = unsafe { reader(dst, src, len) };
    #[cfg(feature = "diagnostics")]
    if copied {
        crate::copystat::add(crate::copystat::C::PhysicalReadBytes, len as u64);
        crate::copystat::add(crate::copystat::C::PhysicalReadUs, started.elapsed().as_micros() as u64);
    }
    copied
}

fn copy_failed_pages(dst: *mut u8, src: usize, len: usize) {
    // Page by page (64 KiB, then 4 KiB in a piece that fails), zeros
    // for what cannot be read. Keep kernel validation of readable pages.
    for_pieces(src, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

#[cfg(feature = "diagnostics")]
fn region_copy_fallback() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::switch::on("VICTORY_REGION_COPY_FALLBACK"))
}

#[cfg(any(feature = "diagnostics", test))]
fn copy_failed_regions(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let start = src as usize;
    let end = start.checked_add(len).expect("valid source range");
    let mut at = start;
    while at < end {
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: querying memory metadata only. Never dereference title pages.
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let ok = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::RegionQueries, 1);
            crate::copystat::add(crate::copystat::C::RegionQueryUs, started.elapsed().as_micros() as u64);
        }
        let boundary = (info.BaseAddress as usize).checked_add(info.RegionSize).unwrap_or(at).min(end);
        if ok == 0 || boundary <= at || info.BaseAddress as usize > at {
            copy_failed_pages(dst.wrapping_add(at - start), at, end - at);
            return;
        }
        let to = dst.wrapping_add(at - start);
        let n = boundary - at;
        if matches!(info.State, MEM_RESERVE | MEM_FREE) {
            // No committed bytes existed at the observation. As with the
            // original failed-read fallback, concurrent remapping after
            // this observation is not synchronized. Only our host mapping
            // is written; never assume committed source pages are readable.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            unsafe { to.write_bytes(0, n) };
        } else if info.State != MEM_COMMIT || !read(at as *const u8, to, n) {
            copy_failed_pages(to, at, n);
        }
        at = boundary;
    }
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if len == 0 { return true; }
    let mut done = 0;
    if direct_reads() {
        // NT validates the entire title source through the kernel without
        // ReadProcessMemory's higher-level setup. Do not read it by direct
        // pointer dereference: pages may be unmapped concurrently.
        // SAFETY: `dst` is a writable buffer of `len` bytes; NT probes `src`.
        let status = unsafe {
            NtReadVirtualMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done)
        };
        return status >= 0 && done == len;
    }
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 && done == len }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// Read title memory through the kernel, reporting inaccessible pages
    /// instead of causing an access violation in the emulator process.
    fn NtReadVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *const std::ffi::c_void,
        buffer: *mut std::ffi::c_void,
        size: usize,
        read: *mut usize,
    ) -> i32;
    /// Kernel-validated write without WriteProcessMemory's preliminary
    /// protection queries. A negative NTSTATUS or a short transfer leaves
    /// the suffix to the caller's page-sized fallback.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

fn direct_reads() -> bool {
    static DIRECT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DIRECT.get_or_init(|| std::env::var_os("VICTORY_DIRECT_READS").is_some_and(|v| v == "1"))
}

fn direct_writes() -> bool {
    static DIRECT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DIRECT.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Partition exact changed-byte intervals at the title's physical-page
/// boundaries. Every byte appears once, in order, and no intervening gap is
/// introduced. A physical writer certifies at most one 64-KiB section page;
/// a larger writeback span must not force its otherwise eligible pages to
/// fall back to thousands of individual kernel writes.
fn writeback_pages(base: usize, runs: &[(usize, usize)], mut page: impl FnMut(&[(usize, usize)])) {
    const PAGE: usize = 64 << 10;
    let mut at = 0;
    let mut from = runs.first().map(|r| r.0).unwrap_or(0);
    while at < runs.len() {
        let Some(address) = base.checked_add(from) else { return; };
        let boundary = address / PAGE;
        let mut group = Vec::new();
        loop {
            let (_, end) = runs[at];
            // Use subtraction rather than rounding a VA upward: no overflow
            // near the highest representable address.
            let Some(address) = base.checked_add(from) else { return; };
            let available = PAGE - (address % PAGE);
            let to = from + available.min(end - from);
            group.push((from, to));
            if to < end {
                from = to;
                break;
            }
            at += 1;
            if at == runs.len() { break; }
            from = runs[at].0;
            if base.checked_add(from).is_none_or(|address| address / PAGE != boundary) { break; }
        }
        page(&group);
    }
}

/// Writes only the changed runs of an already merged title span. Pin the
/// enclosing guard block once, not once per run. No byte in a gap is written.
pub(crate) fn write_mapped_runs(dst: *mut u8, src: &[u8], runs: &[(usize, usize)]) -> usize {
    let (Some(&(first, _)), Some(&(_, last))) = (runs.first(), runs.last()) else { return 0; };
    assert!(first < last && last <= src.len());
    assert!(runs.iter().all(|&(a,b)| a < b && b <= src.len()) && runs.windows(2).all(|p| p[0].1 <= p[1].0));
    #[cfg(feature = "diagnostics")]
    let guard = std::time::Instant::now();
    let _write = crate::watch::begin_kernel_write((dst as usize) + first, last - first, false);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackGuardUs, guard.elapsed().as_micros() as u64);
    static ALIAS: OnceLock<bool> = OnceLock::new();
    let alias = *ALIAS.get_or_init(|| std::env::var_os("VICTORY_PHYSICAL_WRITE_VIEW").is_some_and(|v| v == "1"));
    let writer = alias.then(|| PHYSICAL_WRITER.get()).flatten();
    let mut written = 0;
    // Every page has independent all-or-nothing physical certification. If a
    // title remaps one page, only that page uses the validated kernel path;
    // successful preceding pages are never replayed.
    writeback_pages(dst as usize, runs, |group| {
        if let Some(writer) = writer {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            // SAFETY: independent source, one page, exact changed runs, and
            // the enclosing title-write guard is held. False writes nothing.
            if unsafe { writer(dst, src, group) } {
                let bytes: usize = group.iter().map(|&(a,b)| b-a).sum();
                written += bytes;
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::PhysicalWriteBytes, bytes as u64);
                    crate::copystat::add(crate::copystat::C::PhysicalWriteUs, started.elapsed().as_micros() as u64);
                    crate::copystat::add(crate::copystat::C::PhysicalWriteRuns, group.len() as u64);
                }
                return;
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackKernelWriteRuns, group.len() as u64);
        #[cfg(feature = "diagnostics")]
        let kernel = std::time::Instant::now();
        written += group.iter().map(|&(a,b)| write_mapped_open(dst.wrapping_add(a), src[a..b].as_ptr(), b-a)).sum::<usize>();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackKernelWriteUs, kernel.elapsed().as_micros() as u64);
    });
    written
}

/// Copies host memory into title memory with the caller's guard scope active,
/// leaving pages that are not mapped alone.
fn write_mapped_open(dst: *mut u8, src: *const u8, len: usize) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: title pages are checked by the kernel, and the caller
            // already opened any guarded blocks with watch::writing.
            unsafe { NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) };
            return done.min(n);
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) };
        done.min(n)
    };
    if len == 0 {
        return 0;
    }
    let first = write(dst as usize, len);
    if first == len {
        return first;
    }
    let mut written = first;
    // Never replay a successful prefix of a partial write. That prefix
    // could have been modified by title threads after the kernel wrote it.
    for_pieces(dst as usize + first, len - first, 64 << 10, |at, n| {
        let done = write(at, n);
        written += done;
        if done < n {
            for_pieces(at + done, n - done, 4 << 10, |at, n| {
                written += write(at, n);
            });
        }
    });
    written
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    #[track_caller]
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                #[cfg(feature = "diagnostics")]
                trace_mapped_buffer("upload", size, data);
                Some((buffer?, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                let [runtime_graphics_bytes, title_graphics_bytes] = crate::resource::memory_bytes();
                tracing::info!(runtime_graphics_bytes, title_graphics_bytes, texture_staging_bytes = crate::texture::staging_bytes(),
                    live_lists = LIVE_LISTS.load(Ordering::Relaxed), live_preludes = LIVE_PRELUDES.load(Ordering::Relaxed),
                    "graphics allocation accounting (title allocations may overlap physical pages)");
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// Admission is advisory: WDDM may reduce its budget later. Serialize
    /// our shadow allocations and retain headroom for original textures,
    /// pipelines and other allocations. Failure leaves the shared mirror.
    pub(crate) fn card_shadow_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let _guard = self.card_admission.lock().unwrap_or_else(|e| e.into_inner());
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: adapter matches this device; node zero is the node used
        // by the single-node committed resources below.
        unsafe { self.card_adapter.as_ref()?.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info).ok()?; }
        let reserve = (info.Budget / 8).max(512 << 20);
        if !card_budget_fits(info.Budget, info.CurrentUsage, reserve, size) {
            return None;
        }
        self.local_uav_buffer(size)
    }

    /// Private GPU-only buffer. No CPU mapping or title memory alias.
    pub(crate) fn local_uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: ordinary default-heap buffer creation. State tracker
        // supplies transitions on each execution, as for shared buffers.
        unsafe { self.device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COMMON, None, &mut buffer).ok()?; }
        buffer
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    #[track_caller]
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                #[cfg(feature = "diagnostics")]
                trace_mapped_buffer("shared", size, data);
                Some((buffer?, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

fn card_budget_fits(budget: u64, usage: u64, reserve: u64, size: u64) -> bool {
    usage.checked_add(reserve).and_then(|n| n.checked_add(size))
        .is_some_and(|total| total <= budget)
}

#[test]
fn card_budget_admission_preserves_headroom_and_rejects_overflow() {
    assert!(card_budget_fits(1000, 600, 100, 300));
    assert!(!card_budget_fits(1000, 600, 100, 301));
    assert!(!card_budget_fits(1000, 1100, 100, 0));
    assert!(!card_budget_fits(u64::MAX, u64::MAX, 1, 0));
    assert!(!card_budget_fits(u64::MAX, u64::MAX - 1, 1, 1));
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn recorded_ranges_elide_covered_subviews_without_crossing_write_epochs() {
        use super::RangeCoverage;
        let mut c = RangeCoverage::default();
        assert!(c.insert(1, 100, 100));
        assert!(c.insert(1, 300, 50));
        assert!(!c.insert(1, 120, 60));
        assert!(c.insert(1, 190, 120));
        assert!(!c.insert(1, 100, 250));
        assert!(c.insert(2, 120, 60));
        assert!(c.insert(1, 400, 1));
        assert!(c.insert(1, 350, 50));
        assert!(!c.insert(1, 100, 301));
        assert!(!c.insert(1, usize::MAX, 1));
        assert!(!c.insert(1, 100, 0));
        c.clear();
        assert!(c.insert(1, 120, 60));
    }

    #[test]
    fn deferred_refresh_run_matches_scalar_steps_across_writes_and_reset() {
        use super::Step;
        // Two real, independent shadows with overlapping original title
        // addresses are important: deduplication is by Arc, never title VA.
        let base = 0x1d1100000u64;
        crate::shadow::register(0x71101, base, 4096);
        struct Unregister(u64);
        impl Drop for Unregister {
            fn drop(&mut self) {
                crate::shadow::unregister(0x71101, self.0);
                crate::shadow::unregister(0x71102, self.0);
            }
        }
        let _unregister = Unregister(base);
        let a = crate::shadow::find(base, 4096).expect("first real shadow");
        // Make a new independent shadow on the same title VA only after
        // retaining the first: a title may release/recreate a buffer while
        // an older list still holds the original mirror.
        crate::shadow::unregister(0x71101, base);
        crate::shadow::register(0x71102, base, 4096);
        let b = crate::shadow::find(base, 4096).expect("second real shadow");
        assert!(!Arc::ptr_eq(&a, &b));

        let read_runs = [
            vec![
                (a.clone(), vec![(base + 10, base + 100), (base + 10, base + 100),
                    (base + 99, base + 170), (base + 170, base + 190),
                    (base + 4080, base + 4096)]),
                (b.clone(), vec![(base + 10, base + 100), (base + 50, base + 200)]),
            ],
            vec![
                (a.clone(), vec![(base + 11, base + 40), (base + 250, base + 280)]),
                (b.clone(), vec![(base + 100, base + 180)]),
            ],
        ];
        let mut scalar = super::Deferred::default();
        let mut bulk = super::Deferred::default();
        let snapshot = |deferred: &super::Deferred| {
            let steps = deferred.copies.iter().map(|step| match step {
                Step::Refresh(shadow, address, len) =>
                    (0u8, Arc::as_ptr(shadow) as usize, *address, *len),
                Step::Copy(dst, src, len) => (1, *dst, *src, *len),
                Step::Run(_) => (2, 0, 0, 0),
                Step::GpuWrite(shadow, address, len) =>
                    (3, Arc::as_ptr(shadow) as usize, *address, *len),
                _ => panic!("unexpected step in equivalence fixture"),
            }).collect::<Vec<_>>();
            (steps, deferred.refreshed.0.clone(), deferred.title_write_epoch)
        };
        for (i, ranges) in read_runs.iter().enumerate() {
            for (shadow, intervals) in ranges {
                for &(start, end) in intervals {
                    scalar.record_refresh(shadow, start as usize,
                        (end - start).min(shadow.left(start)) as usize);
                }
            }
            super::record_refresh_run(&mut bulk, ranges);
            assert_eq!(snapshot(&scalar), snapshot(&bulk));
            if i == 0 {
                for deferred in [&mut scalar, &mut bulk] {
                    deferred.title_write_epoch += 1;
                    deferred.copies.push(Step::Copy(0, 0, 0));
                    deferred.refreshed.clear();
                    deferred.copies.push(Step::Run(Box::new(|| {})));
                    deferred.refreshed.clear();
                }
                assert_eq!(snapshot(&scalar), snapshot(&bulk));
            }
        }
        scalar = super::Deferred::default();
        bulk = super::Deferred::default();
        assert_eq!(snapshot(&scalar), snapshot(&bulk));
        for (shadow, intervals) in &read_runs[0] {
            for &(start, end) in intervals {
                scalar.record_refresh(shadow, start as usize,
                    (end - start).min(shadow.left(start)) as usize);
            }
        }
        super::record_refresh_run(&mut bulk, &read_runs[0]);
        assert_eq!(snapshot(&scalar), snapshot(&bulk));
    }

    #[test]
    fn shadow_read_state_run_matches_scalar_with_uav_boundaries() {
        use windows::core::Interface;
        let Some(host) = super::host() else { return };
        let base = 0x1d1200000u64;
        crate::shadow::register(0x72101, base, 4096);
        struct Unregister(u64);
        impl Drop for Unregister {
            fn drop(&mut self) {
                crate::shadow::unregister(0x72101, self.0);
                crate::shadow::unregister(0x72102, self.0);
            }
        }
        let _unregister = Unregister(base);
        let a = crate::shadow::find(base, 4096).expect("first shadow");
        crate::shadow::unregister(0x72101, base);
        crate::shadow::register(0x72102, base, 4096);
        let b = crate::shadow::find(base, 4096).expect("independent reused title VA");
        assert!(!Arc::ptr_eq(&a, &b));

        let first = vec![
            (a.clone(), vec![(base, base + 1024)]),
            (b.clone(), vec![(base + 1024, base + 2048)]),
            (a.clone(), vec![(base + 2048, base + 3072)]),
        ];
        let after_write = vec![
            (b.clone(), vec![(base + 42, base + 52)]),
            (a.clone(), vec![(base + 128, base + 256)]),
        ];
        let scalar = host.new_list().expect("scalar list");
        let bulk = host.new_list().expect("bulk list");
        let states = |list: &super::HostList| {
            let map = list.buffer_states.lock().unwrap_or_else(|e| e.into_inner());
            let mut values = map.iter().map(|(&key, &(first, last, ref resource))|
                (key, first, last, resource.as_raw() as usize)).collect::<Vec<_>>();
            values.sort_unstable();
            values
        };
        for (shadow, _) in &first { shadow.ensure_state(&scalar, 0x40 | 0x80); }
        bulk.ensure_shadow_read_states(&first);
        assert_eq!(states(&scalar), states(&bulk));
        assert_eq!(states(&scalar).len(), 2, "original title VA must not merge distinct buffers");

        for list in [&scalar, &bulk] { a.ensure_state(list, 0x8); }
        assert_eq!(states(&scalar), states(&bulk));
        for (shadow, _) in &after_write { shadow.ensure_state(&scalar, 0x40 | 0x80); }
        bulk.ensure_shadow_read_states(&after_write);
        assert_eq!(states(&scalar), states(&bulk));
        let snapshot = states(&bulk);
        assert!(snapshot.iter().all(|&(_, first, last, _)|
            first == super::buffer_state(0x40 | 0x80) && last == first));
        // A fresh list must not inherit recording-time buffer states.
        let fresh = host.new_list().expect("fresh list");
        assert!(states(&fresh).is_empty());
    }

    /// Isolates command-recording locks, not runtime shader/GPU or title
    /// write-watch costs. Intentionally ignored in the ordinary suite.
    #[test]
    #[ignore = "CPU microbenchmark; run without a game process"]
    fn original_heap_read_lock_batch_microbenchmark() {
        let Some(host) = super::host() else { return };
        const SHADOWS: usize = 32;
        const INTERVALS: usize = 16;
        const REPEATS: usize = 128;
        const BASE: u64 = 0x1d1300000;
        struct Unregister;
        impl Drop for Unregister {
            fn drop(&mut self) {
                for i in 0..SHADOWS {
                    crate::shadow::unregister(0x73100 + i, BASE + i as u64 * 65536);
                }
            }
        }
        let _unregister = Unregister;
        let mut ranges = Vec::new();
        for i in 0..SHADOWS {
            let start = BASE + i as u64 * 65536;
            crate::shadow::register(0x73100 + i, start, 4096);
            let shadow = crate::shadow::find(start, 4096).expect("benchmark shadow");
            let intervals = (0..INTERVALS)
                .map(|j| (start + j as u64 * 128, start + j as u64 * 128 + 64))
                .collect();
            ranges.push((shadow, intervals));
        }
        let scalar = host.new_list().expect("scalar benchmark list");
        let bulk = host.new_list().expect("bulk benchmark list");
        let run_scalar = || {
            for (shadow, _) in &ranges { shadow.ensure_state(&scalar, 0x40 | 0x80); }
            for (shadow, intervals) in &ranges {
                for &(from, to) in intervals { shadow.refresh(&scalar, from, to - from); }
            }
        };
        let run_bulk = || {
            bulk.ensure_shadow_read_states(&ranges);
            bulk.defer_refresh_run(&ranges);
        };
        run_scalar(); run_bulk();
        let start = std::time::Instant::now();
        for _ in 0..REPEATS { run_scalar(); }
        let scalar_us = start.elapsed().as_micros();
        let start = std::time::Instant::now();
        for _ in 0..REPEATS { run_bulk(); }
        let bulk_us = start.elapsed().as_micros();
        println!("{SHADOWS} shadows x {INTERVALS} intervals x {REPEATS} read runs: scalar {scalar_us}us, batched {bulk_us}us; synthetic recording only");
        assert_eq!(scalar.deferred().refreshed.0, bulk.deferred().refreshed.0);
        assert_eq!(scalar.buffer_states.lock().unwrap().len(), bulk.buffer_states.lock().unwrap().len());
    }

    #[test]
    fn recorded_range_coverage_matches_byte_oracle() {
        use super::RangeCoverage;
        let mut coverage = RangeCoverage::default();
        let mut oracle = [[false; 256]; 4];
        let mut seed = 8u32;
        for i in 0..10_000 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let key = (seed >> 28) as usize % 4;
            let from = (seed >> 12) as usize % 256;
            let len = ((seed >> 3) as usize % 48).min(256 - from);
            let expected = oracle[key][from..from+len].iter().any(|&b| !b);
            assert_eq!(coverage.insert(key, from, len), expected);
            oracle[key][from..from+len].fill(true);
            if i % 37 == 0 {
                coverage.clear();
                oracle = [[false;256];4];
            }
        }
    }

    #[test]
    fn exact_writeback_page_partition_preserves_sparse_bytes_and_order() {
        const PAGE: usize = 64 << 10;
        let base = PAGE * 2 + 17;
        let runs = [(0, 5), (37, PAGE - 20), (PAGE - 19, PAGE + 40),
                    (2 * PAGE - 20, 2 * PAGE + 200), (4 * PAGE + 11, 4 * PAGE + 14)];
        let mut groups = Vec::<Vec<(usize,usize)>>::new();
        super::writeback_pages(base, &runs, |group| groups.push(group.to_vec()));
        assert!(groups.len() >= 4);
        let mut expected = vec![false; 4 * PAGE + 14];
        for &(from, to) in &runs { expected[from..to].fill(true); }
        let mut seen = vec![0u8; expected.len()];
        let mut previous_end = 0;
        for group in groups {
            let low = group[0].0;
            let high = group.last().unwrap().1;
            assert!(low >= previous_end, "must preserve ordered run positions");
            assert_eq!((base + low) / PAGE, (base + high - 1) / PAGE);
            previous_end = high;
            for (from, to) in group {
                for byte in &mut seen[from..to] { *byte += 1; }
            }
        }
        assert_eq!(seen.iter().map(|&v| v != 0).collect::<Vec<_>>(), expected);
        assert!(seen.iter().all(|&v| v <= 1), "each changed byte written exactly once");
    }

    #[test]
    fn shadow_refresh_index_is_stable_by_arc_identity_and_resets_at_flush() {
        let (a, b, c) = (Arc::new(0u8), Arc::new(0u8), Arc::new(0u8));
        let mut index = std::collections::HashMap::new();
        let mut next = 0;
        for (shadow, wanted_index, wanted_new) in [
            (&a, 0, true), (&b, 1, true), (&a, 0, false),
            (&c, 2, true), (&b, 1, false), (&c, 2, false),
        ] {
            let (at, new) = Batch::slot(&mut index, next, shadow);
            assert_eq!((at, new), (wanted_index, wanted_new));
            if new { next += 1; }
        }
        assert_eq!(index.len(), 3);
        // The actual `Batch::flush` clears the map alongside draining the
        // ordered refreshes; a new submission starts its indexes from zero.
        index.clear();
        assert_eq!(Batch::slot(&mut index, 0, &b), (0, true));
        assert_eq!(Batch::slot(&mut index, 1, &a), (1, true));
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn ordered_prefix_keeps_suffix_and_reinserts_after_it() {
        // No real title memory is accessed: empty interval sets exercise
        // queue identity/order independently of watch/protection state.
        let Some(a) = crate::shadow::find(0x110000000, 16) else { return };
        let Some(b) = crate::shadow::find(0x120000000, 16) else { return };
        let Some(c) = crate::shadow::find(0x130000000, 16) else { return };
        let mut batch = Batch::default();
        for (shadow, address) in [(&a, 0x110000000), (&b, 0x120000000), (&c, 0x130000000)] {
            batch.add(shadow, address, 16);
        }
        assert_eq!(batch.refreshes.len(), 3);
        for (_, intervals) in &mut batch.refreshes {
            intervals.clear();
        }
        batch.flush_through_target(&b);
        assert_eq!(batch.refresh_base, 2);
        assert_eq!(batch.refreshes.len(), 1);
        assert!(Arc::ptr_eq(&batch.refreshes[0].0, &c));
        batch.flush_through_target(&a); // no pending refresh for a
        assert_eq!(batch.refreshes.len(), 1);
        batch.add(&a, 0x110000000, 16);
        batch.add(&c, 0x130000000, 32);
        assert_eq!(batch.refreshes[0].1, [(0x130000000, 0x130000020)]);
        assert!(Arc::ptr_eq(&batch.refreshes[1].0, &a));
        for (_, intervals) in &mut batch.refreshes {
            intervals.clear();
        }
        batch.title_written();
        assert!(batch.refreshes.is_empty());
        assert!(batch.refresh_index.is_empty());
        assert_eq!(batch.refresh_base, 0);
    }

    #[test]
    #[ignore = "hardware capability probe; run with no game process"]
    fn section_backed_buffer_cpu_gpu_alias_probe() {
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::core::Interface;
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory as mem;
        const K64: usize = 65536;
        const SIZE: usize = 4 * K64;
        struct Section(windows_sys::Win32::Foundation::HANDLE);
        impl Drop for Section {
            fn drop(&mut self) { unsafe { CloseHandle(self.0); } }
        }
        struct View(mem::MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) { unsafe { mem::UnmapViewOfFile(self.0); } }
        }
        let host = super::host().expect("a host GPU for the hardware probe");
        let device = host.device.cast::<ID3D12Device3>().expect("ID3D12Device3");
        let section = |flags| {
            let h = unsafe { mem::CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), mem::PAGE_READWRITE | flags, 0, SIZE as u32, std::ptr::null()) };
            assert!(!h.is_null());
            Section(h)
        };
        let reserved = section(mem::SEC_RESERVE);
        let reserved_result: windows::core::Result<ID3D12Heap> = unsafe {
            device.OpenExistingHeapFromFileMapping(HANDLE(reserved.0))
        };
        println!("reserved section opening: {}", if reserved_result.is_ok() { "accepted" } else { "refused" });
        drop(reserved_result);
        // XMem uses SEC_RESERVE to avoid committing the title's multi-GiB
        // allocation prematurely. Check whether committing every page later
        // makes the existing section eligible, without changing its identity.
        let reserved_view = View(unsafe { mem::MapViewOfFile(reserved.0, mem::FILE_MAP_WRITE, 0, 0, SIZE) });
        assert!(!reserved_view.0.Value.is_null());
        let committed_at = unsafe { mem::VirtualAlloc(reserved_view.0.Value, SIZE, mem::MEM_COMMIT, mem::PAGE_READWRITE) };
        assert_eq!(committed_at, reserved_view.0.Value);
        let fully_committed: windows::core::Result<ID3D12Heap> = unsafe {
            device.OpenExistingHeapFromFileMapping(HANDLE(reserved.0))
        };
        println!("fully committed SEC_RESERVE section opening: {}", if fully_committed.is_ok() { "accepted" } else { "refused" });
        drop(fully_committed);
        drop(reserved_view);
        let committed = section(mem::SEC_COMMIT);
        let opened: windows::core::Result<ID3D12Heap> = unsafe {
            device.OpenExistingHeapFromFileMapping(HANDLE(committed.0))
        };
        let heap = match opened {
            Ok(h) => h,
            Err(e) => { println!("committed section unsupported by this device: {e}"); return; }
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: K64 as u64, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut placed: Option<ID3D12Resource> = None;
        if let Err(e) = unsafe { host.device.CreatePlacedResource(&heap, K64 as u64, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut placed) } {
            println!("section heap accepted, buffer placement refused: {e}");
            return;
        }
        let placed = placed.unwrap();
        let map = || {
            let view = unsafe { mem::MapViewOfFile(committed.0, mem::FILE_MAP_WRITE, 0, K64 as u32, K64) };
            assert!(!view.Value.is_null());
            View(view)
        };
        let view = map();
        let (other, ptr) = host.shared_buffer(K64 as u64).unwrap();
        unsafe { view.0.Value.cast::<u8>().write_bytes(0x39, K64); }
        let fetch = host.new_list().unwrap();
        fetch.ensure_buffer_state(&placed, 0x800);
        fetch.ensure_buffer_state(&other, 0x400);
        unsafe { fetch.list.CopyBufferRegion(&other, 0, &placed, 0, K64 as u64); }
        fetch.close();
        host.execute(&[&fetch]);
        assert!(unsafe { std::slice::from_raw_parts(ptr, K64) }.iter().all(|&b| b == 0x39));
        drop(view); // underlying section survives the CPU view.
        unsafe { ptr.write_bytes(0xa7, K64); }
        let store = host.new_list().unwrap();
        store.ensure_buffer_state(&other, 0x800);
        store.ensure_buffer_state(&placed, 0x400);
        unsafe { store.list.CopyBufferRegion(&placed, 0, &other, 0, K64 as u64); }
        store.close();
        host.execute(&[&store]);
        let second = map();
        assert!(unsafe { std::slice::from_raw_parts(second.0.Value.cast::<u8>(), K64) }.iter().all(|&b| b == 0xa7));
        println!("PASS: committed section CPU->GPU, GPU->CPU, and unmap/remap aliasing on host device");
    }

    #[test]
    fn buffer_execution_state_is_local_to_execution_group() {
        use super::{buffer_before, buffer_state, BufferExecution};
        let read = buffer_state(0x1);
        assert_eq!(read, buffer_state(0x200));
        assert_eq!(read, buffer_state(0x40 | 0x80));
        assert_eq!(buffer_state(8), 8);
        assert_eq!(buffer_state(0x400), 0x400);
        let mut group = BufferExecution::new();
        assert_eq!(buffer_before(&mut group, 1, read), 0);
        assert_eq!(buffer_before(&mut group, 2, 8), 0);
        assert_eq!(buffer_before(&mut group, 1, 8), read);
        group.clear();
        assert_eq!(buffer_before(&mut group, 1, read), 0);
        assert_eq!(buffer_before(&mut group, 2, read), 0);
    }

    #[test]
    fn buffer_lists_run_in_submission_not_recording_order() {
        let Some(host) = super::host() else { return };
        let (a, ap) = host.shared_buffer(65536).unwrap();
        let (b, _) = host.shared_buffer(65536).unwrap();
        let (c, cp) = host.shared_buffer(65536).unwrap();
        let read = host.new_list().unwrap();
        read.ensure_buffer_state(&b, 0x800);
        read.ensure_buffer_state(&c, 0x400);
        unsafe { read.list.CopyBufferRegion(&c, 0, &b, 0, 256) };
        read.close();
        let write = host.new_list().unwrap();
        write.ensure_buffer_state(&a, 0x800);
        write.ensure_buffer_state(&b, 0x400);
        unsafe { write.list.CopyBufferRegion(&b, 0, &a, 0, 256) };
        write.close();
        for value in [0x37u8, 0xcbu8] {
            unsafe { ap.write_bytes(value, 256); cp.write_bytes(0, 256); }
            host.execute(&[&write, &read]);
            assert_eq!(unsafe { std::slice::from_raw_parts(cp, 256) }, &[value; 256]);
        }
    }

    #[test]
    fn region_copy_matches_page_fallback_with_uncommitted_holes() {
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RESERVE, MEM_RELEASE, PAGE_READWRITE, PAGE_NOACCESS};
        const PAGE: usize = 4096;
        let base = unsafe { VirtualAlloc(std::ptr::null(), 64 * PAGE, MEM_RESERVE, PAGE_NOACCESS) }.cast::<u8>();
        assert!(!base.is_null());
        struct Release(*mut u8);
        impl Drop for Release {
            fn drop(&mut self) { unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE); } }
        }
        let _release = Release(base);
        for (first, count, value) in [(0, 3, 0x42u8), (17, 5, 0x9a), (61, 3, 0xff)] {
            let p = unsafe { VirtualAlloc(base.add(first * PAGE).cast(), count * PAGE, MEM_COMMIT, PAGE_READWRITE) }.cast::<u8>();
            assert!(!p.is_null());
            unsafe { p.write_bytes(value, count * PAGE) };
        }
        let offset = 173;
        let len = 64 * PAGE - offset - 39;
        let mut expected = vec![0x55; len];
        let mut actual = vec![0xaa; len];
        let src = unsafe { base.add(offset) };
        super::copy_failed_pages(expected.as_mut_ptr(), src as usize, len);
        super::copy_failed_regions(actual.as_mut_ptr(), src, len);
        assert_eq!(actual, expected);
        assert_eq!(actual[3 * PAGE - offset], 0);
        assert_eq!(actual[17 * PAGE - offset], 0x9a);
    }

    #[test]
    fn refresh_intervals_keep_gaps_and_merge_overlaps() {
        let mut intervals = Vec::new();
        for (from, to) in [
            (100, 200),
            (400, 500),
            (150, 250),
            (300, 350),
            (250, 300),
            (500, 550),
            (45, 45),
        ] {
            Batch::add_interval(&mut intervals, from, to);
        }
        assert_eq!(intervals, [(100, 350), (400, 550)]);
    }

    #[test]
    fn refresh_intervals_equal_the_union_in_any_order() {
        let ranges = [
            (5, 10),
            (7, 19),
            (0, 1),
            (40, 41),
            (25, 31),
            (1, 5),
            (31, 40),
            (10, 27),
        ];
        for order in [ranges.to_vec(), ranges.into_iter().rev().collect()] {
            let mut intervals = Vec::new();
            let mut wanted = [false; 48];
            for (from, to) in order {
                wanted[from..to].fill(true);
                Batch::add_interval(&mut intervals, from, to);
                assert!(intervals.windows(2).all(|w| w[0].1 < w[1].0));
                for (i, is_set) in wanted.iter().enumerate() {
                    assert_eq!(intervals.iter().any(|&(a, b)| a <= i && i < b), *is_set);
                }
            }
        }
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}
