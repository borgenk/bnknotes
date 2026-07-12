//! App configuration, gathered in one place so it can grow a loader without
//! touching call sites. For now it is defaults only (no on-disk config yet);
//! [`Config::default`] holds the built-in values, and a future file loader
//! replaces them at startup.
//!
//! This module also owns the [`Default`] for the portable [`FontConfig`] type:
//! the concrete font paths are this app's choice, so they live here rather than
//! in the portable `platform/freetype.rs`, which stays free of app-specific fonts.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::platform::freetype::{FontConfig, FontFamily};

/// A notes vault: a root directory the editor roots its file tree and every note
/// reference (images, wikilink embeds) against. Its name is the last component of
/// the path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vault {
    pub path: PathBuf,
}

/// The display name for a vault root shown in the top bar: the final path
/// component, or the whole path when it has none (a filesystem root). Takes a
/// path so the app can name the active vault straight from the file tree's root.
pub fn vault_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|s| s.to_str())
        .or_else(|| path.to_str())
        .unwrap_or("vault")
}

/// Expand a leading `~/` (or a bare `~`) to the user's home directory; any other
/// path passes through unchanged. Keeps configured vault paths short and stays
/// ready for a future config-file loader that reads them as text.
pub fn expand_tilde(path: &str) -> PathBuf {
    expand_tilde_in(
        path,
        std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .as_deref(),
    )
}

/// The pure core of [`expand_tilde`], taking the home directory explicitly so it
/// can be tested with and without one, rather than mutating the process HOME.
fn expand_tilde_in(path: &str, home: Option<&OsStr>) -> PathBuf {
    match home {
        Some(h) if path == "~" => PathBuf::from(h),
        Some(h) => match path.strip_prefix("~/") {
            Some(rest) => Path::new(h).join(rest),
            None => PathBuf::from(path),
        },
        None => PathBuf::from(path),
    }
}

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
    /// The same mask gamma for fenced-code and inline-code glyphs. Code is a
    /// smaller face on the code panel, where its thin monospace stems are almost
    /// all anti-aliased edge, so the prose thinning leaves them faint; a gentler
    /// value keeps code as crisp as the prose beside it. Higher is thinner.
    pub code_glyph_coverage_gamma: f32,
    /// Line height as a multiple of the font size, the CSS model. Each line box is
    /// grown to at least this many times its size (never below the font's own line
    /// height), and the extra space falls below the text. The default is 1.5.
    pub line_height_scale: f32,
    /// Caret motion: whether the caret eases toward its target and how it feels.
    pub caret: CaretConfig,
    /// The configured vaults, in priority order. The first is the default active
    /// vault when the session has no valid remembered one; the active vault roots
    /// the file tree and every note reference.
    pub vaults: Vec<Vault>,
}

/// Caret motion settings: one master toggle plus the knobs that shape the motion.
/// When [`Self::animate`] is off the caret snaps cell to cell as before.
#[derive(Clone, Copy, Debug)]
pub struct CaretConfig {
    /// Whether keyboard-driven moves ease the caret toward its target and stretch it
    /// along the way. Covers typing, Enter, and navigation; mouse clicks and reflows
    /// always snap regardless of this flag.
    pub animate: bool,
    /// Exponential decay constant, 1/second, for the edge leading the caret's travel.
    /// Higher is snappier; this sets the overall speed.
    pub rate: f32,
    /// The trailing edge's share of `rate`, in 0..1. The trailing edge eases at
    /// `rate * spring`, so it lags the leading edge and the box stretches along the
    /// caret's path. Lower is a longer, more visible smear; 1.0 is a rigid bar with no
    /// stretch.
    pub spring: f32,
    /// Largest animated target distance in caret heights. A larger jump (jump to
    /// end, a click across the document) settles immediately rather than drawing a
    /// document-long streak.
    pub max_jump_lines: f32,
}

impl Default for CaretConfig {
    fn default() -> Self {
        Self {
            // On, to dogfood the smooth caret on real content.
            animate: true,
            // Fast enough to feel snappy while typing, still visibly a smear. Higher
            // shortens the motion; lower makes it lazier.
            rate: 72.0,
            // A pronounced tail so the smear stays visible at the higher rate; 1.0
            // removes the stretch, near 0 is a long comet.
            spring: 0.5,
            // Wide enough that an Enter, Home/End, or a click across a full-width
            // line still animates (a content column is tens of caret heights wide);
            // only a document-spanning jump is farther than this and snaps.
            max_jump_lines: 50.0,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fonts: FontConfig::default(),
            // Around 1.8 undoes the linear-light fattening on its own; a little
            // above that also matches the browser's edge lightening.
            glyph_coverage_gamma: 2.0,
            // Code renders at a smaller size where the prose thinning reads as
            // faint, so it takes almost none: 1.0 leaves the coverage untouched.
            code_glyph_coverage_gamma: 1.0,
            line_height_scale: 1.5,
            caret: CaretConfig::default(),
            // One vault to start, the notes folder in the user's home. A future
            // loader replaces this list from an on-disk config.
            vaults: vec![Vault {
                path: expand_tilde("~/notes"),
            }],
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
            // Fallback chain for scalars the prose and code faces lack: arrows,
            // maths and technical signs, and the many symbol blocks a text font
            // omits. The Symbols faces cover most of them; Noto Sans Mono rounds out
            // the odd Latin-adjacent glyph. A glyph absent from all of these still
            // renders its .notdef box, the pre-fallback behavior. Consulted per
            // missing glyph only, so the common all-Latin path never touches it.
            fallback: vec![
                "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf".into(),
                "/usr/share/fonts/noto/NotoSansMono-Regular.ttf".into(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_joins_the_home_dir() {
        let home = OsStr::new("/home/user");
        assert_eq!(
            expand_tilde_in("~/notes", Some(home)),
            PathBuf::from("/home/user/notes")
        );
        assert_eq!(
            expand_tilde_in("~/a/b/c", Some(home)),
            PathBuf::from("/home/user/a/b/c")
        );
    }

    #[test]
    fn expand_tilde_handles_a_bare_tilde() {
        assert_eq!(
            expand_tilde_in("~", Some(OsStr::new("/home/user"))),
            PathBuf::from("/home/user")
        );
    }

    #[test]
    fn expand_tilde_leaves_non_tilde_paths_alone() {
        let home = Some(OsStr::new("/home/user"));
        assert_eq!(
            expand_tilde_in("/etc/notes", home),
            PathBuf::from("/etc/notes")
        );
        assert_eq!(
            expand_tilde_in("relative/dir", home),
            PathBuf::from("relative/dir")
        );
        // A tilde mid-path is not a home marker, only a leading one is.
        assert_eq!(expand_tilde_in("a/~/b", home), PathBuf::from("a/~/b"));
    }

    #[test]
    fn expand_tilde_without_a_home_passes_through() {
        assert_eq!(expand_tilde_in("~/notes", None), PathBuf::from("~/notes"));
        assert_eq!(expand_tilde_in("~", None), PathBuf::from("~"));
    }

    #[test]
    fn vault_name_is_the_last_path_component() {
        assert_eq!(vault_name(Path::new("/home/user/notes")), "notes");
    }

    #[test]
    fn vault_name_ignores_a_trailing_slash() {
        // A trailing separator does not hide the final component.
        assert_eq!(vault_name(Path::new("/home/user/work/")), "work");
    }

    #[test]
    fn vault_name_falls_back_to_the_whole_path_at_a_root() {
        assert_eq!(vault_name(Path::new("/")), "/");
    }

    #[test]
    fn default_config_has_a_notes_vault() {
        let config = Config::default();
        assert_eq!(config.vaults.len(), 1);
        assert_eq!(vault_name(&config.vaults[0].path), "notes");
    }
}
