//! bnknotes: a minimal, software-rendered text edit field built on a native
//! Wayland client (no winit, no smithay, no crates at all). The point is to
//! find out how far we get owning the whole stack.

mod app;
mod browser;
mod conn;
mod display;
mod error;
mod ffi;
mod file;
mod find;
mod freetype;
mod grapheme;
mod image;
mod layout;
mod markdown;
mod perf;
mod protocol;
mod regex;
mod render;
mod rope;
mod scroll;
mod shape;
mod shm;
mod text;
mod wire;
mod xkb;

fn main() {
    // `--perf` runs the offline performance harness instead of opening a window;
    // everything else launches the editor (with an optional file path).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.iter().any(|a| a == "--perf") {
        perf::run(&args)
    } else {
        app::run()
    };
    if let Err(e) = result {
        eprintln!("bnknotes: {e}");
        std::process::exit(1);
    }
}
