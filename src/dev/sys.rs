//! Thin FFI over `getrusage(2)`, `sysconf(3)`, `sched_setaffinity(2)`, and
//! `/proc/self/statm` for the performance harness: CPU time consumed, resident
//! memory, and which core the measuring runs on. Linux-only, like the rest of the
//! app; std already links libc so the symbols resolve without a `#[link]`.
//! CPU-seconds per workload is the battery metric without a battery: lower CPU
//! time is longer battery, directly.

use core::ffi::{c_int, c_long, c_ulong};
use std::time::Duration;

/// `getrusage` target: this process (self), not its children.
const RUSAGE_SELF: c_int = 0;
/// `_SC_PAGESIZE` on Linux (arch-independent in the generic ABI).
const SC_PAGESIZE: c_int = 30;

#[repr(C)]
struct Timeval {
    tv_sec: c_long,
    tv_usec: c_long,
}

/// `struct rusage`, Linux x86_64/arm64 ABI: two timevals then sixteen longs. Only
/// the times and `ru_maxrss` (the first long) are read; the rest is padding kept
/// so the struct has the size the kernel writes.
#[repr(C)]
#[allow(dead_code)]
struct Rusage {
    ru_utime: Timeval,
    ru_stime: Timeval,
    ru_maxrss: c_long,
    rest: [c_long; 15],
}

/// `cpu_set_t` as glibc sizes it: a bitmask of 1024 CPUs, one bit per CPU. The
/// kernel accepts any size; this is the conventional one.
#[repr(C)]
struct CpuSet {
    words: [c_ulong; 16],
}

impl CpuSet {
    /// A mask naming `cpu` alone. `None` if the CPU is past what the mask holds.
    fn only(cpu: usize) -> Option<Self> {
        let bits = c_ulong::BITS as usize;
        let mut words = [0 as c_ulong; 16];
        *words.get_mut(cpu / bits)? |= 1 << (cpu % bits);
        Some(Self { words })
    }
}

unsafe extern "C" {
    fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
    fn sysconf(name: c_int) -> c_long;
    fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: *const CpuSet) -> c_int;
}

/// Pin this process to a single CPU and report which, so a measurement is not at
/// the mercy of where the scheduler puts it. `None` if the machine names no CPUs
/// or the kernel refuses the affinity call (a restricted cpuset), which costs
/// noise, not correctness.
///
/// A hybrid CPU is the reason this exists. Since Alder Lake, Intel parts mix
/// performance and efficiency cores, and the same single-threaded work runs far
/// slower on an efficiency core: the editor's full-relayout scenario takes 1.8x
/// as long there. Left to the scheduler a run lands on either kind, and the spread
/// between runs comes out wider than the ratio the gate calls a regression, so the
/// gate fires at random on an unchanged tree. Pinning collapses that spread from
/// around 14% to around 1%.
///
/// The performance cores are the CPUs the kernel lists under the `cpu_core` PMU.
/// This takes the highest-numbered one: any fixed choice makes runs comparable,
/// and the high end stays clear of CPU 0, which carries interrupt duty and
/// measures consistently slower. A machine that lists no `cpu_core` is either not
/// hybrid or too old to say so, and falls back to the highest CPU online, which is
/// still one fixed core to measure on.
pub fn pin_to_measuring_core() -> Option<usize> {
    let cpu = highest_cpu_in("/sys/devices/cpu_core/cpus")
        .or_else(|| highest_cpu_in("/sys/devices/system/cpu/online"))?;
    let mask = CpuSet::only(cpu)?;
    // SAFETY: pid 0 is this process; the mask is a live, fully initialized CpuSet
    // and the kernel only reads `cpusetsize` bytes of it.
    let rc = unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &mask) };
    (rc == 0).then_some(cpu)
}

/// The highest CPU number in a sysfs cpu list ("0-15", "0-7,16-23", "3"), or
/// `None` when the file is absent or names nothing.
fn highest_cpu_in(path: &str) -> Option<usize> {
    let list = std::fs::read_to_string(path).ok()?;
    list.trim()
        .split(',')
        .filter_map(|part| {
            // A range's end is its highest; a bare number is its own.
            let end = part.split('-').next_back()?;
            end.trim().parse::<usize>().ok()
        })
        .max()
}

fn rusage() -> Option<Rusage> {
    // SAFETY: `getrusage` fills a fully owned, correctly sized `Rusage`; a zeroed
    // start is valid for every field. A nonzero return means it wrote nothing.
    let mut u = std::mem::MaybeUninit::<Rusage>::zeroed();
    let rc = unsafe { getrusage(RUSAGE_SELF, u.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    Some(unsafe { u.assume_init() })
}

/// Total CPU time (user + system) this process has consumed so far. Take a delta
/// across a workload for its CPU cost; monotonic, so a delta is always >= 0.
pub fn cpu_time() -> Duration {
    match rusage() {
        Some(u) => dur(&u.ru_utime) + dur(&u.ru_stime),
        None => Duration::ZERO,
    }
}

fn dur(t: &Timeval) -> Duration {
    // tv_usec is microseconds in [0, 1e6); *1000 stays under a second in nanos.
    Duration::new(
        t.tv_sec.max(0) as u64,
        (t.tv_usec.max(0) as u32).wrapping_mul(1000),
    )
}

/// The process's peak resident set size in bytes (a lifetime high-water mark, so
/// it only rises across scenarios). `ru_maxrss` is kilobytes on Linux.
pub fn peak_rss_bytes() -> u64 {
    match rusage() {
        Some(u) => (u.ru_maxrss.max(0) as u64) * 1024,
        None => 0,
    }
}

/// The process's current resident set size in bytes, from `/proc/self/statm`
/// (unlike the peak, this falls when memory is released). Zero if unreadable.
pub fn current_rss_bytes() -> u64 {
    let Ok(statm) = std::fs::read_to_string("/proc/self/statm") else {
        return 0;
    };
    // Fields, in pages: size resident shared text lib data dt. The second is RSS.
    let resident = statm
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<u64>().ok());
    match resident {
        Some(pages) => pages * page_size(),
        None => 0,
    }
}

fn page_size() -> u64 {
    // SAFETY: sysconf with a valid name has no memory effects.
    let p = unsafe { sysconf(SC_PAGESIZE) };
    if p > 0 { p as u64 } else { 4096 }
}
