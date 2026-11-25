//! Diagnostics: distinguish live Rust allocations from title virtual memory
//! and opaque driver allocations. Counts requested bytes, not heap commit.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

struct Metered;

#[global_allocator]
static ALLOCATOR: Metered = Metered;

static LIVE: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];
static COUNT: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

fn bucket(n: usize) -> usize {
    match n {
        0..=65535 => 0,
        65536..=1048575 => 1,
        1048576..=16777215 => 2,
        _ => 3,
    }
}

fn added(n: usize) {
    let i = bucket(n);
    LIVE[i].fetch_add(n, Relaxed);
    COUNT[i].fetch_add(1, Relaxed);
}

fn removed(n: usize) {
    let i = bucket(n);
    LIVE[i].fetch_sub(n, Relaxed);
    COUNT[i].fetch_sub(1, Relaxed);
}

// SAFETY: delegates every allocation and free to the same System allocator
// with the original layout; no allocation or logging inside its callbacks.
unsafe impl GlobalAlloc for Metered {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() { added(layout.size()); }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() { added(layout.size()); }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        removed(layout.size());
        unsafe { System.dealloc(p, layout) };
    }

    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(p, layout, size) };
        if !next.is_null() {
            removed(layout.size());
            added(size);
        }
        next
    }
}

pub(crate) fn start() {
    if std::env::var_os("VICTORY_MEMORY_ACCOUNTING").as_deref() != Some(std::ffi::OsStr::new("1")) {
        return;
    }
    let _ = std::thread::Builder::new().name("victory-memory-accounting".into()).spawn(|| {
        loop {
            let bytes = LIVE.each_ref().map(|n| n.load(Relaxed));
            let counts = COUNT.each_ref().map(|n| n.load(Relaxed));
            eprintln!("rust allocator requested_bytes={bytes:?} allocation_counts={counts:?} buckets=<64KiB,<1MiB,<16MiB,>=16MiB");
            victory_hle::modules::xmem::diagnostic_memory();
            victory_hle::modules::d3d12x::diagnostic_memory();
            std::thread::sleep(std::time::Duration::from_secs(3));
        }
    });
}
