//! bnknotes: a minimal, software-rendered text edit field built on a native
//! Wayland client (no winit, no smithay, no crates at all). The point is to
//! find out how far we get owning the whole stack.

mod app;
mod conn;
mod error;
mod ffi;
mod freetype;
mod grapheme;
mod layout;
mod markdown;
mod protocol;
mod render;
mod scroll;
mod shm;
mod text;
mod wire;
mod xkb;

fn main() {
    if let Err(e) = app::run() {
        eprintln!("bnknotes: {e}");
        std::process::exit(1);
    }
}
