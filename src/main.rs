//! bnknotes: a minimal, GPU-rendered markdown notes editor and a native
//! Wayland client (no winit, no smithay, almost no crates).

mod alloc;
mod app;
mod browser;
mod bytes;
mod conn;
mod corpus;
mod display;
mod dmabuf;
mod error;
mod ffi;
mod file;
mod find;
mod freetype;
mod gpu;
mod grapheme;
mod image;
mod layout;
mod markdown;
mod paint;
mod pane;
mod perf;
mod protocol;
mod regex;
mod rope;
mod scroll;
mod shape;
mod store;
mod sys;
mod text;
mod theme;
mod tree;
mod vulkan;
mod wire;
mod xkb;

fn main() {
    // `--perf` runs the offline regression gate, `--perf-lab` the corpus-driven
    // performance lab, and `--gpu-probe` reports what the compositor offers the
    // GPU presentation path, all without opening a window; everything else
    // launches the editor (with an optional file path).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.iter().any(|a| a == "--perf-lab") {
        perf::run_lab(&args)
    } else if args.iter().any(|a| a == "--load-bench") {
        perf::run_load_bench(&args)
    } else if args.iter().any(|a| a == "--perf") {
        perf::run(&args)
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
