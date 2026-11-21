//! bnknotes: a GPU-rendered markdown notes editor and a native Wayland client.
//!
//! Every module is private, so the compiler reports code nothing reaches as dead.

use std::process::ExitCode;

mod ai;
mod app;
mod config;
mod dev;
mod file;
mod find;
mod fold;
mod frontmatter;
mod icon;
mod keymap;
mod layout;
mod markdown;
mod menu;
mod paint;
mod palette;
mod pane;
mod platform;
mod regex;
mod render;
mod rope;
mod scale;
mod search;
mod store;
mod syntax;
mod table;
mod text;
mod theme;
mod tree;
mod watch;

pub fn main() -> ExitCode {
    // `--perf` runs the offline regression gate, `--perf-lab` the corpus-driven
    // performance lab, `--gpu-probe` reports what the compositor offers the GPU
    // presentation path, and `--gen-tables` regenerates the Unicode property
    // tables from the vendored UCD; all without opening a window. Everything else
    // launches the editor (with an optional file path).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.iter().any(|a| a == "--perf-lab") {
        dev::perf::run_lab(&args)
    } else if args.iter().any(|a| a == "--load-bench") {
        dev::perf::run_load_bench(&args)
    } else if args.iter().any(|a| a == "--mem-report") {
        dev::perf::run_mem_report(&args)
    } else if args.iter().any(|a| a == "--perf") {
        dev::perf::run(&args)
    } else if args.iter().any(|a| a == "--gen-tables") {
        dev::gen_tables::run(&args)
    } else if args.iter().any(|a| a == "--gpu-probe") {
        app::gpu_probe()
    } else {
        app::run()
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bnknotes: {e}");
            ExitCode::FAILURE
        }
    }
}
