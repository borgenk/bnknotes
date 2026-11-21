//! bnknotes: a GPU-rendered markdown notes editor and a native Wayland client.
//!
//! Every module but app is private, so the compiler reports code nothing reaches as
//! dead.

mod ai;
pub mod app;
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
