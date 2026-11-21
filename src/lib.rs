//! bnknotes: a GPU-rendered markdown notes editor and a native Wayland client.
//!
//! Every module is private, so the compiler reports code nothing reaches as dead.

use std::process::ExitCode;

mod ai;
mod app;
mod cli;
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
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    #[cfg(any(test, feature = "dev"))]
    {
        let dev_args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into()).collect();
        if let Some(result) = dev::run(&dev_args) {
            return report(result);
        }
    }
    let result = match cli::Mode::parse(args) {
        Ok(cli::Mode::Run { path, demo_diff }) => app::run(path, demo_diff),
        Ok(cli::Mode::GpuProbe) => app::gpu_probe(),
        Err(arg) => {
            eprintln!("bnknotes: unknown argument {arg:?}; {}", cli::USAGE);
            return ExitCode::from(2);
        }
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
