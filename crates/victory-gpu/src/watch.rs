//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! The physical mapping registry also tracks aliases and temporary writes
//! through DirectStorage views. A known page is clean only while every
//! registered alias is protected and no external writer is active.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
});

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // Physical sections cannot be write-watched. Use the authoritative
    // mapping registry instead of making a failing kernel query on every
    // view. Unknown blocks in a mixed range remain dirty.
    if guarded && let Some(physical) = guard_physical(first, last) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        j.asked += 1;
        let now = j.asked;
        for page in physical { j.written.insert(page, now); }
        return Some((now, j.written.range(first..last).map(|(&p, &w)| (p,w)).collect()));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    // The kernel may report up to sixteen 4 KiB pages per console page.
    // Update the journal directly, with no temporary vector of page numbers.
    let Journal { found, written, .. } = j;
    for page in found[..count].iter().map(|&at| at / PAGE).chain(guarded) {
        written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Virtual block -> stable (section identity, page index).
    physical: BTreeMap<usize, (usize, usize)>,
    aliases: BTreeMap<(usize, usize), BTreeSet<usize>>,
    external: BTreeMap<(usize, usize), usize>,
    // Kernel writes must remain writable for the entire operation, not
    // just at the instant the caller announces it.
    kernel: BTreeMap<usize, (usize, usize)>,
    asynchronous: BTreeMap<usize, usize>,
    next_kernel: usize,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    physical: BTreeMap::new(),
    aliases: BTreeMap::new(),
    external: BTreeMap::new(),
    kernel: BTreeMap::new(),
    asynchronous: BTreeMap::new(),
    next_kernel: 0,
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Guards {
    fn kernel_active(&self, block: usize) -> bool {
        self.kernel.values().any(|&(a,b)| a <= block && block < b)
            || self.asynchronous.range(..=block).next_back().is_some_and(|(_, &b)| block < b)
    }
    fn dirty_physical(&mut self, key: (usize, usize)) {
        if let Some(aliases) = self.aliases.get(&key) {
            for block in aliases { self.clean.remove(block); }
        }
    }

    fn remove_mapping(&mut self, block: usize) {
        self.protected.remove(&block);
        self.clean.remove(&block);
        if let Some(key) = self.physical.remove(&block) {
            self.dirty_physical(key);
            if let Some(aliases) = self.aliases.get_mut(&key) {
                aliases.remove(&block);
                if aliases.is_empty() { self.aliases.remove(&key); }
            }
        }
    }

    fn cacheable(&self, block: usize) -> bool {
        let Some(key) = self.physical.get(&block) else { return false; };
        !self.external.contains_key(key) && self.aliases.get(key).is_some_and(|aliases| {
            aliases.iter().all(|alias| self.protected.contains(alias) && !self.kernel_active(*alias))
        })
    }
}

/// Whether a range intersects a physical view. Allows a formerly unmapped
/// shadow to retry the fast mapped-page tracking path after it is mapped.
pub(crate) fn known_physical(start: usize, end: usize) -> bool {
    if start >= end { return false; }
    guards().physical.range(start / BLOCK..end.div_ceil(BLOCK)).next().is_some()
}

fn guard_physical(first: usize, last: usize) -> Option<Vec<usize>> {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    guard_physical_impl(first, last, *ALL.get_or_init(|| crate::switch::on("VICTORY_GUARD_ALL_ALIASES")))
}

fn guard_physical_impl(first: usize, last: usize, all: bool) -> Option<Vec<usize>> {
    let mut g = guards();
    g.physical.range(first..last).next()?;
    install();
    let mut written = Vec::new();
    let mut run = None;
    for block in first..=last {
        let dirty = block < last
            && (!g.physical.contains_key(&block) || !g.clean.contains(&block));
        if dirty {
            written.push(block);
            if g.physical.contains_key(&block) && !g.protected.contains(&block) && !g.kernel_active(block) {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() { protect(&mut g, from, block); }
    }
    // All known aliases need write protection before a physical page can
    // become clean. Scoped kernel writers and untracked asynchronous
    // destinations remain writable and force conservative refreshes.
    if all {
        let mut extra = BTreeSet::new();
        for block in &written {
            if let Some(key) = g.physical.get(block)
                && !g.external.contains_key(key)
                && let Some(aliases) = g.aliases.get(key) {
                    extra.extend(aliases.iter().copied().filter(|a| !g.protected.contains(a) && !g.kernel_active(*a)));
            }
        }
        let mut run: Option<(usize, usize)> = None;
        for &block in &extra {
            if let Some((a,b)) = run {
                if b == block { run = Some((a,b+1)); continue; }
                protect(&mut g,a,b);
            }
            run = Some((block,block+1));
        }
        if let Some((a,b)) = run { protect(&mut g,a,b); }
        // Protection alone is not an observation of this alias's bytes.
        // Its own next query must still publish its write generation.
        for block in extra { g.clean.remove(&block); }
    }
    for &block in &written {
        if g.cacheable(block) { g.clean.insert(block); }
        else { g.clean.remove(&block); }
    }
    Some(written)
}

/// Registers an actual mapped run of physical section pages. Section ids
/// must not be reused. The caller reports the mapping only after success.
pub fn title_physical_view(at: usize, len: usize, section: usize, index: usize) {
    if !at.is_multiple_of(BLOCK) || !len.is_multiple_of(BLOCK) || at.checked_add(len).is_none() { return; }
    let mut g = guards();
    for (i, block) in (at / BLOCK..(at + len) / BLOCK).enumerate() {
        let key = (section, index + i);
        g.remove_mapping(block);
        g.dirty_physical(key);
        g.physical.insert(block, key);
        g.aliases.entry(key).or_default().insert(block);
    }
}

/// A write through an unregistered temporary mapping (DirectStorage).
/// Keep all of its pages dirty for the whole writable window's lifetime.
pub struct PhysicalWrite {
    section: usize,
    first: usize,
    count: usize,
}

pub fn physical_write(section: usize, first: usize, count: usize) -> PhysicalWrite {
    let mut g = guards();
    for page in first..first.saturating_add(count) {
        let key = (section, page);
        g.dirty_physical(key);
        *g.external.entry(key).or_insert(0) += 1;
    }
    PhysicalWrite { section, first, count }
}

impl Drop for PhysicalWrite {
    fn drop(&mut self) {
        let mut g = guards();
        for page in self.first..self.first.saturating_add(self.count) {
            let key = (self.section, page);
            g.dirty_physical(key);
            if let Some(n) = g.external.get_mut(&key) {
                *n -= 1;
                if *n == 0 { g.external.remove(&key); }
            }
        }
    }
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let active = block < last && g.kernel_active(block);
        let open = block < last && (!g.clean.contains(&block) || active);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if active {
                g.clean.remove(&block);
            } else if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let affected: Vec<_> = g.physical.range(start / BLOCK..end.div_ceil(BLOCK))
        .map(|(_, &key)| key).collect();
    for key in affected { g.dirty_physical(key); }
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

pub struct KernelWrite(Option<usize>);

/// Keep a synchronous kernel destination open until the operation returns.
/// An overlapped destination is conservatively excluded from protection
/// indefinitely because completion may occur through an arbitrary API.
pub fn begin_kernel_write(at: usize, len: usize, asynchronous: bool) -> KernelWrite {
    if len == 0 { return KernelWrite(None); }
    let end = at.saturating_add(len);
    let (a,b) = (at / BLOCK, end.div_ceil(BLOCK));
    let mut g = guards();
    open(&mut g, at, end);
    if asynchronous {
        let mut a = a;
        let mut b = b;
        if let Some((&start, &stop)) = g.asynchronous.range(..=a).next_back()
            && stop >= a { a = start; b = b.max(stop); g.asynchronous.remove(&start); }
        while let Some((&start, &stop)) = g.asynchronous.range(a..=b).next() {
            b = b.max(stop); g.asynchronous.remove(&start);
        }
        g.asynchronous.insert(a,b);
        KernelWrite(None)
    } else {
        g.next_kernel += 1;
        let id = g.next_kernel;
        g.kernel.insert(id,(a,b));
        KernelWrite(Some(id))
    }
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if let Some(id) = self.0 {
            let mut g = guards();
            if let Some((a,b)) = g.kernel.remove(&id) {
                open(&mut g, a.saturating_mul(BLOCK), b.saturating_mul(BLOCK));
            }
        }
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    let mut blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    blocks.extend(g.physical.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).map(|(&block, _)| block));
    blocks.sort_unstable();
    blocks.dedup();
    for block in blocks {
        g.remove_mapping(block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn physical_aliases_and_external_writes_invalidate_every_mirror() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: Null optional pointers request an unnamed pagefile-backed writable section of one page.
        let section = unsafe {
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, PAGE as u32, std::ptr::null())
        };
        assert!(!section.is_null());
        // SAFETY: The checked section handle is live and the requested writable view fits its size.
        let a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: The section remains live and the second writable view fits its size.
        let b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        let (a_at, b_at) = (a.Value as usize, b.Value as usize);
        assert!(a_at != 0 && b_at != 0 && a_at != b_at);
        let id = usize::MAX;
        title_physical_view(a_at, PAGE, id, 0);
        title_physical_view(b_at, PAGE, id, 0);
        let ap = a_at / PAGE;
        let bp = b_at / PAGE;
        // A's observation cannot be cached while B is still writable.
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        assert_eq!(guard_physical(bp, bp+1), Some(vec![bp]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![]));
        // CPU write via B is caught by the handler, dirties A as well.
        // SAFETY: b_at names a live writable mapped page; the installed handler handles guard faults.
        unsafe { (b_at as *mut u8).write_volatile(0x73); }
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        // SAFETY: a_at names the live alias of the same page, after the write.
        assert_eq!(unsafe { (a_at as *const u8).read_volatile() }, 0x73);
        assert_eq!(guard_physical(bp, bp+1), Some(vec![bp]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![]));
        {
            let _active = physical_write(id, 0, 1);
            // SAFETY: The live section covers the requested writable page.
            let temporary = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
            assert!(!temporary.Value.is_null());
            assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
            // SAFETY: The checked non-null temporary mapping is writable and the byte lies within it.
            unsafe { temporary.Value.cast::<u8>().write_volatile(0xc8); }
            assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
            // SAFETY: The temporary view is live and no further access follows.
            unsafe { UnmapViewOfFile(temporary); }
        }
        assert_eq!(guard_physical(ap, ap+1), Some(vec![ap]));
        assert_eq!(guard_physical(ap, ap+1), Some(vec![]));
        // SAFETY: a_at still names the live mapped page.
        assert_eq!(unsafe { (a_at as *const u8).read_volatile() }, 0xc8);
        // SAFETY: Both mappings and the section handle are live and no longer used for memory access.
        unsafe { UnmapViewOfFile(a); UnmapViewOfFile(b); CloseHandle(section); }
        title_memory_remapped(a_at, PAGE);
        title_memory_remapped(b_at, PAGE);
        assert!(!known_physical(a_at, a_at+PAGE));
        assert!(!known_physical(b_at, b_at+PAGE));
    }

    #[test]
    fn all_alias_guard_keeps_kernel_writer_open_and_each_alias_dirty() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: Null optional pointers request an unnamed pagefile-backed writable section of one page.
        let section = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, PAGE as u32, std::ptr::null()) };
        assert!(!section.is_null());
        // SAFETY: The checked section handle is live and the writable view fits the section.
        let a = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        // SAFETY: The section remains live and the second view fits its size.
        let b = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGE) };
        let (aa,bb) = (a.Value as usize,b.Value as usize);
        assert!(aa != 0 && bb != 0);
        title_physical_view(aa,PAGE,usize::MAX-1,0);
        title_physical_view(bb,PAGE,usize::MAX-1,0);
        let (ap,bp) = (aa/PAGE,bb/PAGE);
        assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![ap]));
        assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![]));
        assert_eq!(guard_physical_impl(bp,bp+1,true),Some(vec![bp]));
        {
            let _writer = begin_kernel_write(bb,1,false);
            assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![ap]));
            assert_eq!(guard_physical_impl(bp,bp+1,true),Some(vec![bp]));
            assert!(writable(bb));
            // SAFETY: bb names the checked writable mapping and the byte is in bounds.
            unsafe { (bb as *mut u8).write_volatile(99); }
            assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![ap]));
        }
        assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![ap]));
        assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![]));
        assert_eq!(guard_physical_impl(bp,bp+1,true),Some(vec![bp]));
        // SAFETY: bb remains a live mapped page; the installed handler handles any guard fault.
        unsafe { (bb as *mut u8).write_volatile(100); }
        assert_eq!(guard_physical_impl(ap,ap+1,true),Some(vec![ap]));
        assert_eq!(guard_physical_impl(bp,bp+1,true),Some(vec![bp]));
        // SAFETY: aa is the live alias of the written page.
        assert_eq!(unsafe { (aa as *const u8).read_volatile() },100);
        // SAFETY: The two views and handle are live and no further memory access follows.
        unsafe { UnmapViewOfFile(a); UnmapViewOfFile(b); CloseHandle(section); }
        title_memory_remapped(aa,PAGE); title_memory_remapped(bb,PAGE);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (initial, before) = writes(start, end, false).expect("watched");
        // The journal intentionally outlives a single allocation. Windows
        // may reuse an address freed by another test; historical entries
        // must not be confused with a write during this observation.
        assert!(before.iter().all(|&(_, generation)| generation < initial));
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let mut expected: BTreeMap<_, _> = before.into_iter().collect();
        expected.insert(p / PAGE + 1, now);
        assert_eq!(after, expected.into_iter().collect::<Vec<_>>());
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}
