//! Thin FFI over `getrusage(2)`, `sysconf(3)`, and `/proc/self/statm` for the
//! performance lab: CPU time consumed and resident memory. Linux-only, like the
//! rest of the app; std already links libc so the symbols resolve without a
//! `#[link]`. CPU-seconds per workload is the battery metric without a battery:
//! lower CPU time is longer battery, directly.

use core::ffi::{c_int, c_long};
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

extern "C" {
    fn getrusage(who: c_int, usage: *mut Rusage) -> c_int;
    fn sysconf(name: c_int) -> c_long;
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
    if p > 0 {
        p as u64
    } else {
        4096
    }
}
