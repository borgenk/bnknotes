//! Development and benchmarking tooling, off the editor's runtime path.
//!
//! The performance harness (`perf`, its Tier 1 gate and Tier 2 lab), the stress
//! corpora it runs over (`corpus`), the counting global allocator (`alloc`, only
//! installed under the `perf-alloc` feature), and the OS timing/memory probes
//! (`sys`). Reached only through the `--perf*` / `--load-bench` / `--mem-report`
//! CLI flags; the shipping editor never calls into any of it.

pub(crate) mod alloc;
pub(crate) mod corpus;
pub(crate) mod perf;
pub(crate) mod sys;
