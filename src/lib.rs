//! bnknotes: a GPU-rendered markdown notes editor and a native Wayland client.
//!
//! Every module but app is private, so the compiler reports code nothing reaches as
//! dead.

/// Assert a C struct mirror's layout against numbers measured from the C headers
/// with sizeof and offsetof: its size, unless it mirrors only the leading fields of
/// a struct the C library allocates, and every field's offset.
#[cfg(test)]
macro_rules! pin_c_layout {
    ($ty:ident, size $size:expr; $($field:ident @ $offset:expr),* $(,)?) => {{
        assert_eq!(core::mem::size_of::<$ty>(), $size, concat!(stringify!($ty), " size"));
        pin_c_layout!($ty, prefix; $($field @ $offset),*);
    }};
    ($ty:ident, prefix; $($field:ident @ $offset:expr),* $(,)?) => {{
        $(assert_eq!(
            core::mem::offset_of!($ty, $field),
            $offset,
            concat!(stringify!($ty), ".", stringify!($field))
        );)*
    }};
}

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
