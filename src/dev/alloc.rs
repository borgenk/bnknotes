//! A process-wide allocation counter for the performance harness. It wraps the
//! system allocator and bumps two relaxed atomics, so a scenario can report how
//! many allocations (and how many bytes) an operation cost, deterministically,
//! beside its wall-clock time. Allocation count is a clean signal that does not
//! swing with inlining the way a stopwatch does, so it is the metric a
//! regression gate can trust.
//!
//! It is compiled in as the global allocator only under the `perf-alloc` feature
//! (`make perf` enables it), because a global allocator taxes every allocation
//! in the whole process: the shipping editor never pays for it. When the feature
//! is off the counters still exist but nothing writes them, so [`snapshot`]
//! reads zero and the harness prints the allocation columns as unmeasured
//! (that is what [`ENABLED`] distinguishes).
//!
//! Only Rust-side allocations are counted. FreeType and HarfBuzz allocate on the
//! C side, invisibly to Rust's global allocator, so a scenario dominated by FFI
//! rasterization shows few Rust allocations by design.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the counting allocator is actually installed (the `perf-alloc`
/// feature is on). The harness uses this to tell a measured zero from an
/// unmeasured one.
#[cfg(feature = "perf-alloc")]
pub const ENABLED: bool = true;
#[cfg(not(feature = "perf-alloc"))]
pub const ENABLED: bool = false;

/// The counter is add-only within a scenario (reset zeroes it, alloc adds,
/// dealloc is not tracked), so it never underflows and is exactly reproducible
/// for a given code path and input, independent of the machine.
static ALLOCS: AtomicU64 = AtomicU64::new(0);

/// The counts read out after a scenario: the number of allocation calls.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub allocs: u64,
}

/// Zero the counter before measuring a scenario.
pub fn reset() {
    ALLOCS.store(0, Relaxed);
}

/// Read the counter after a scenario.
pub fn snapshot() -> Stats {
    Stats {
        allocs: ALLOCS.load(Relaxed),
    }
}

/// The allocator: delegate every call to the system allocator, counting on the
/// way through. Installed as the global allocator only under `perf-alloc`.
#[cfg(feature = "perf-alloc")]
pub struct Counting;

#[cfg(feature = "perf-alloc")]
unsafe impl std::alloc::GlobalAlloc for Counting {
    // SAFETY: every method passes its caller's arguments to the system allocator
    // unchanged, so the caller's `GlobalAlloc` contract carries over.
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { std::alloc::System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCS.fetch_add(1, Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        let ptr = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOCS.fetch_add(1, Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { std::alloc::System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // A realloc that grows is a fresh allocation's worth of work; count it
            // so growing a Vec in a loop shows the churn it really is.
            ALLOCS.fetch_add(1, Relaxed);
        }
        new_ptr
    }
}

#[cfg(feature = "perf-alloc")]
#[global_allocator]
static GLOBAL: Counting = Counting;
