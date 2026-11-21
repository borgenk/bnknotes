//! bnknotes: a GPU-rendered markdown notes editor and a native Wayland client.
//!
//! Every module is private, so the compiler reports code nothing reaches as dead.

use std::process::ExitCode;

mod ai;
mod app;
mod config;
#[cfg(any(test, feature = "dev"))]
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(any(test, feature = "dev"))]
    {
        if let Some(result) = dev::run(&args) {
            return report(result);
        }
    }
    // `--gpu-probe` reports what the compositor offers the GPU presentation path,
    // without opening a window. Everything else launches the editor (with an
    // optional file path).
    let result = if args.iter().any(|a| a == "--gpu-probe") {
        app::gpu_probe()
    } else {
        app::run()
    };
    report(result)
}

fn report(result: platform::error::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bnknotes: {e}");
            ExitCode::FAILURE
        }
    }
}
