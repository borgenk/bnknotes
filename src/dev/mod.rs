//! Development and benchmarking tooling, off the editor's runtime path.
//!
//! The performance harness (`perf`, its Tier 1 gate and Tier 2 lab), the stress
//! corpora it runs over (`corpus`), the counting global allocator (`alloc`, only
//! installed under the `perf-alloc` feature), the OS timing/memory probes (`sys`),
//! and the Unicode table generator (`gen_tables`). Only the dev feature and tests
//! compile it.

pub(crate) mod alloc;
pub(crate) mod corpus;
pub(crate) mod gen_tables;
#[cfg(test)]
mod golden;
pub(crate) mod perf;
pub(crate) mod sys;

use crate::platform::error::Result;

pub(crate) fn run(args: &[String]) -> Option<Result<()>> {
    let has = |flag: &str| args.iter().any(|a| a == flag);
    let result = if has("--perf-lab") {
        perf::run_lab(args)
    } else if has("--load-bench") {
        perf::run_load_bench(args)
    } else if has("--mem-report") {
        perf::run_mem_report(args)
    } else if has("--perf") {
        perf::run(args)
    } else if has("--gen-tables") {
        gen_tables::run(args)
    } else {
        return None;
    };
    Some(result)
}
