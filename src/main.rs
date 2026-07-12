//! bnknotes: a minimal, GPU-rendered markdown notes editor and a native
//! Wayland client (no winit, no smithay, almost no crates).

mod app;
mod config;
mod dev;
mod file;
mod find;
mod keymap;
mod layout;
mod markdown;
mod paint;
mod pane;
// The platform layer is written as a library, so it carries surface the editor
// does not call, such as a raw key char. That is library surface, not dead
// code to strip.
#[allow(dead_code)]
mod platform;
mod regex;
mod render;
mod rope;
mod scale;
mod store;
mod syntax;
mod table;
mod text;
mod theme;
mod tree;

fn main() {
    // `--perf` runs the offline regression gate, `--perf-lab` the corpus-driven
    // performance lab, and `--gpu-probe` reports what the compositor offers the
    // GPU presentation path, all without opening a window; everything else
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
    } else if args.iter().any(|a| a == "--gpu-probe") {
        app::gpu_probe()
    } else {
        app::run()
    };
    if let Err(e) = result {
        eprintln!("bnknotes: {e}");
        std::process::exit(1);
    }
}
