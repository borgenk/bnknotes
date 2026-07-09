//! App configuration, gathered in one place so it can grow a loader without
//! touching call sites. For now it is defaults only (no on-disk config yet);
//! [`Config::default`] holds the built-in values, and a future file loader
//! replaces them at startup.
//!
//! This module also owns the [`Default`] for the portable [`FontConfig`] type:
//! the concrete font paths are this app's choice, so they live here rather than
//! in the portable `platform/freetype.rs`, which stays free of app-specific fonts.

use crate::platform::freetype::{FontConfig, FontFamily};

/// The editor's configurable settings, the seam a config-file loader fills in.
#[derive(Clone, Debug)]
pub struct Config {
    /// Which fonts the editor opens for prose and code.
    pub fonts: FontConfig,
    /// Mask gamma applied to glyph coverage in the fragment shader. Linear-light
    /// compositing renders light-on-dark text heavier than the gamma-space stacks
    /// beside it; raising coverage to this power thins the anti-aliased edges back
    /// to a matching weight. Higher is thinner; 1.0 disables it.
    pub glyph_coverage_gamma: f32,
    /// Line height as a multiple of the font size, the CSS model. Each line box is
    /// grown to at least this many times its size (never below the font's own line
    /// height), and the extra space falls below the text. The default is 1.5.
    pub line_height_scale: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fonts: FontConfig::default(),
            // Around 1.8 undoes the linear-light fattening on its own; a little
            // above that also matches the browser's edge lightening.
            glyph_coverage_gamma: 2.0,
            line_height_scale: 1.5,
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            families: vec![
                // Roboto first, the preferred text face; the first family installed wins.
                FontFamily::new(
                    "/usr/share/fonts/TTF/Roboto-Regular.ttf",
                    "/usr/share/fonts/TTF/Roboto-Bold.ttf",
                    "/usr/share/fonts/TTF/Roboto-Italic.ttf",
                    "/usr/share/fonts/TTF/Roboto-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
                    "/usr/share/fonts/noto/NotoSans-Bold.ttf",
                    "/usr/share/fonts/noto/NotoSans-Italic.ttf",
                    "/usr/share/fonts/noto/NotoSans-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/Hack-Regular.ttf",
                    "/usr/share/fonts/TTF/Hack-Bold.ttf",
                    "/usr/share/fonts/TTF/Hack-Italic.ttf",
                    "/usr/share/fonts/TTF/Hack-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
                    "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
                    // Noto Sans Mono ships no italic; these paths simply will not
                    // exist, and the lookup falls back to the regular face.
                    "/usr/share/fonts/noto/NotoSansMono-Italic.ttf",
                    "/usr/share/fonts/noto/NotoSansMono-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-Italic.ttf",
                    "/usr/share/fonts/Adwaita/AdwaitaMono-BoldItalic.ttf",
                ),
                FontFamily::new(
                    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
                    "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
                    // DejaVu calls its slanted styles Oblique rather than Italic.
                    "/usr/share/fonts/TTF/DejaVuSansMono-Oblique.ttf",
                    "/usr/share/fonts/TTF/DejaVuSansMono-BoldOblique.ttf",
                ),
            ],
            // Code fonts: a monospace deliberately distinct from the prose family,
            // so code blocks and inline code read in their own typeface. Code
            // never renders bold or italic, so only the regular face is listed.
            code: vec![
                // JetBrains Mono first, the preferred code face. Noto Sans Mono next as a
                // near-universal fallback when JetBrains Mono is not installed.
                "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansMono-Regular.ttf".into(),
                "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf".into(),
                "/usr/share/fonts/liberation/LiberationMono-Regular.ttf".into(),
                "/usr/share/fonts/TTF/DejaVuSansMono.ttf".into(),
                "/usr/share/fonts/gnu-free/FreeMono.otf".into(),
            ],
            // No icon fallback: the editor renders prose and code, not private-use
            // Nerd Font / Powerline icons, so a missing glyph stays the `.notdef` box.
            fallback: vec![],
        }
    }
}
