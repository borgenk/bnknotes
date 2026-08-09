//! App configuration, gathered in one place. [`Config::default`] holds the
//! built-in values and [`Config::load`] replaces what the config file names at
//! startup.
//!
//! The file is `$XDG_CONFIG_HOME/bnknotes/config` (else `~/.config/bnknotes/config`),
//! in a line-based format: one `field value` per line, `#` comments and blank
//! lines ignored, an unknown field skipped rather than refused so a config
//! written for a later version still loads, and a repeating field accumulating.
//! Today one field is read:
//!
//! ```text
//! # bnknotes config
//! vault ~/notes
//! vault ~/work/journal
//! ```
//!
//! Nothing here is fatal: a missing, oversized, or unreadable file leaves the
//! defaults standing, because a bad config must never stop the editor opening.
//!
//! [`add_vault`] is the one write, for the rail's "Save as vault". It appends a
//! line to the file as it is on disk rather than re-encoding the config, so the
//! comments, the ordering, and any field this version does not know all survive.
//!
//! This module also owns the [`Default`] for the portable [`FontConfig`] type:
//! the concrete font paths are this app's choice, so they live here rather than
//! in the portable `platform/freetype.rs`, which stays free of app-specific fonts.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::icon::VaultIcon;
use crate::platform::error::{Error, Result};
use crate::platform::freetype::{FontConfig, FontFamily};
use crate::theme::VAULT_COLORS;

const FILENAME: &str = "config";
/// Sibling of the config file, flock'd for the duration of an [`add_vault`] so two
/// bnknotes saving a vault at once serialize their read-append-write.
const LOCK_FILENAME: &str = "config.lock";

/// Largest config file read or appended to. The config is a handful of short
/// lines; anything this size is not one, and refusing to load it beats reading a
/// pasted note into memory line by line.
const CONFIG_MAX_BYTES: u64 = 64 * 1024;

/// Most vaults the config can name. The rail draws them in one column, and the
/// list is input-fed, so it is bounded on the way in rather than truncated after.
const MAX_VAULTS: usize = 64;

/// A notes vault: a root directory the editor roots its file tree and every note
/// reference (images, wikilink embeds) against. Its name is the last component of
/// the path.
///
/// The chip the vault rail shows for it is the rest: a color and the initials
/// drawn on it. [`Vault::new`] derives both, so a vault is a path plus where it
/// came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vault {
    pub path: PathBuf,
    /// The rail chip's fill (0x00RRGGBB).
    pub color: u32,
    /// What the chip says: the first two characters of the vault's name, upper
    /// case. Derived once here rather than per frame.
    pub initials: String,
    /// Whether the config file names this vault, which decides both how the rail
    /// draws its chip and whether the chip offers to save it.
    pub kind: VaultKind,
    /// The picture the chip wears, if it was given one, and the square of it shown.
    pub icon: Option<VaultIcon>,
    /// The keys that picture's baked bitmaps are drawn under, `None` until they are
    /// baked (see the app's icon syncing). Derived, like the initials, rather than
    /// worked out per frame: a chip is drawn far more often than its icon changes.
    pub icon_keys: Option<IconKeys>,
}

/// What a chip's picture is drawn from: the bitmap it wears, and the one it wears
/// under the pointer, which is the same square of the same picture taken closer in.
///
/// Two bitmaps rather than one sampled two ways, because a chip's rounded corners
/// are baked into its picture's own alpha: drawing a smaller part of one bitmap
/// would leave the corners cut flat where the zoom crossed them, so each is rounded
/// as itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IconKeys {
    pub rest: String,
    pub hover: String,
}

/// Where a vault in the rail came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaultKind {
    /// The config file names it: it is in the rail at every launch, and its chip
    /// is drawn solid.
    Saved,
    /// A directory opened from the command line (`bnknotes .`) that the config
    /// does not name. It rides in the rail for this session only, drawn as an
    /// outline, until its context menu saves it into the config.
    Temporary,
}

impl Vault {
    /// The vault at `path`, wearing `color`, with the initials its name gives and no
    /// picture yet.
    pub fn new(path: PathBuf, color: u32, kind: VaultKind) -> Self {
        Self {
            initials: vault_initials(&path),
            color,
            path,
            kind,
            icon: None,
            icon_keys: None,
        }
    }

    /// Give the chip `icon` (or take its picture away with `None`), dropping the
    /// baked bitmaps' keys so the next frame bakes the new one.
    pub fn set_icon(&mut self, icon: Option<VaultIcon>) {
        self.icon = icon;
        self.icon_keys = None;
    }
}

/// A color for a new chip: the first in [`VAULT_COLORS`] that no chip in the rail
/// wears yet, else (once every color is spoken for) the one position `i` lands on.
///
/// Handing out the unused colors first is the whole job of the color: a rail of
/// eight or fewer has no two chips that look alike, so a glance tells them apart.
/// Past eight the palette wraps and two vaults share a color, told apart by their
/// initials. A color is dealt once and then remembered (see [`crate::store`]),
/// because the rail can be dragged into any order and a color re-derived from a
/// chip's position would repaint every chip a drag passed.
pub fn deal_color(taken: &[u32], i: usize) -> u32 {
    VAULT_COLORS
        .iter()
        .copied()
        .find(|c| !taken.contains(c))
        .unwrap_or(VAULT_COLORS[i % VAULT_COLORS.len()])
}

/// The saved vaults for `paths`, in the order given, each dealt a color.
pub fn vaults(paths: impl IntoIterator<Item = PathBuf>) -> Vec<Vault> {
    let mut taken: Vec<u32> = Vec::new();
    paths
        .into_iter()
        .enumerate()
        .map(|(i, path)| {
            let color = deal_color(&taken, i);
            taken.push(color);
            Vault::new(path, color, VaultKind::Saved)
        })
        .collect()
}

/// The initials a vault's chip shows: the first two characters of its name, upper
/// case. A one-character name gives one letter. A character that upper-cases into
/// several (the German ß into SS) keeps its first, so two characters in are always
/// two out and the chip cannot overflow its square.
pub fn vault_initials(path: &Path) -> String {
    vault_name(path)
        .chars()
        .take(2)
        .filter_map(|c| c.to_uppercase().next())
        .collect()
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

/// The directory holding everything bnknotes keeps for the user: the config file
/// here, the state file beside it (see [`crate::store`]). `$XDG_CONFIG_HOME/bnknotes`
/// when that is set, else `~/.config/bnknotes`. `None` when neither variable
/// resolves, and then nothing is loaded or persisted.
pub fn config_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(xdg).join("bnknotes"));
    }
    let home = std::env::var_os("HOME").filter(|s| !s.is_empty())?;
    Some(PathBuf::from(home).join(".config").join("bnknotes"))
}

/// Path to the config file, `None` when the config directory does not resolve.
pub fn config_path() -> Option<PathBuf> {
    Some(config_dir()?.join(FILENAME))
}

/// Resolve a path from the config file or the command line to the absolute one the
/// vault list compares against, so a vault named two ways (a relative ".", a
/// symlink, a trailing slash) is one entry and not two. A path that does not
/// resolve, a vault or file that does not exist yet, is returned unchanged: naming
/// a vault before creating it, and opening a new file, both still work.
pub fn resolve_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

impl Config {
    /// The configuration for this run: the built-in defaults with whatever the
    /// config file names laid over them. Never fails, and says why when it falls
    /// back, since the editor has to open either way.
    pub fn load() -> Self {
        match config_path() {
            Some(path) => load_from(&path),
            None => Config::default(),
        }
    }
}

/// Read and parse the config at `path`. A file that is not there is the common
/// case and stays silent; one that cannot be read, or is too big to be a config,
/// logs and leaves the defaults standing.
fn load_from(path: &Path) -> Config {
    match read_config(path) {
        Ok(Some(text)) => parse(&text),
        Ok(None) => Config::default(),
        Err(e) => {
            eprintln!("bnknotes: ignoring config {}: {e}", path.display());
            Config::default()
        }
    }
}

/// The config file's text, `None` when there is no file yet. The size is checked
/// against [`CONFIG_MAX_BYTES`] before the read allocates, so a huge file (a note
/// saved over the config by accident) is refused rather than read in.
fn read_config(path: &Path) -> Result<Option<String>> {
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > CONFIG_MAX_BYTES => {
            return Err(Error::msg("config file is too large to be a config"))
        }
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Parse config text over the built-in defaults: a field the file does not name
/// keeps the value [`Config::default`] gave it, so a config naming one thing does
/// not blank the rest.
fn parse(text: &str) -> Config {
    let mut config = Config::default();
    let paths = vault_paths(text);
    if !paths.is_empty() {
        config.vaults = vaults(paths);
    }
    config.ai = ai_config(text);
    config
}

/// The AI settings a config names, laid over the defaults. Every `ai ...` line is
/// applied in file order, so a later line overrides an earlier one; a value that
/// names nothing recognized leaves the default standing.
fn ai_config(text: &str) -> AiConfig {
    let mut ai = AiConfig::default();
    for line in text.lines() {
        let line = line.trim();
        let Some((field, value)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        if field != "ai" {
            continue;
        }
        match value.trim() {
            "on" | "true" => ai.enabled = true,
            "off" | "false" => ai.enabled = false,
            "provider claude" => ai.provider = crate::ai::Provider::Claude,
            "provider codex" => ai.provider = crate::ai::Provider::Codex,
            _ => {}
        }
    }
    ai
}

/// The vault paths a config's text names, in file order: resolved, deduplicated,
/// and capped at [`MAX_VAULTS`]. Shared by [`parse`] and [`add_vault_to`], which
/// has to know whether a vault is already named before it appends one.
fn vault_paths(text: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // The first whitespace run splits the field name from its value, so a vault
        // path may hold spaces of its own.
        let Some((field, value)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        // Unknown fields are skipped, not refused: a config written for a later
        // version, or for a knob still living in the source, still loads.
        if field != "vault" {
            continue;
        }
        let path = resolve_path(&expand_tilde(value));
        if paths.len() < MAX_VAULTS && !paths.contains(&path) {
            paths.push(path);
        }
    }
    paths
}

/// Name `vault` in the config file, the rail's "Save as vault", creating the file
/// and its directory when this is the first one. `vault` is a resolved path (see
/// [`resolve_path`]), which is what makes "the config already names it" answerable;
/// a vault it does name is left alone rather than repeated.
///
/// The line is appended to the file as it is on disk, so the user's comments,
/// ordering, and any field this version does not read survive untouched; a save
/// that re-encoded the config would quietly eat all three.
pub fn add_vault(vault: &Path) -> Result<()> {
    let path =
        config_path().ok_or_else(|| Error::msg("no XDG_CONFIG_HOME or HOME to save into"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    add_vault_to(&path, vault)
}

/// The core of [`add_vault`], taking the config path so it can be tested against a
/// temp file. The read, append, and atomic write run under the lock, so a second
/// bnknotes saving a vault at the same moment does not lose one of the two.
fn add_vault_to(path: &Path, vault: &Path) -> Result<()> {
    let line = vault_line(vault)?;
    let _lock = crate::file::FileLock::acquire(&path.with_file_name(LOCK_FILENAME));
    let mut text = read_config(path)?.unwrap_or_default();
    if vault_paths(&text).iter().any(|p| p == vault) {
        return Ok(());
    }
    // A config whose last line has no break of its own would otherwise swallow the
    // new field onto the end of it.
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&line);
    crate::file::save_atomic(path, text.as_bytes())
}

/// The `vault <path>` line for `vault`, or an error when the path cannot survive a
/// round trip through the line format: the format has no escaping, so a path that
/// is not UTF-8, or that a newline or edge whitespace would break or trim, has no
/// line that reads back as the same path.
fn vault_line(vault: &Path) -> Result<String> {
    let path = vault
        .to_str()
        .ok_or_else(|| Error::msg("cannot save a vault whose path is not valid UTF-8"))?;
    if path.is_empty() || path.trim() != path || path.contains('\n') {
        return Err(Error::msg(
            "cannot save a vault whose path holds a line break or leading/trailing spaces",
        ));
    }
    Ok(format!("vault {path}\n"))
}

/// Expand a leading `~/` (or a bare `~`) to the user's home directory; any other
/// path passes through unchanged. Keeps the vault paths in the config file short.
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

/// The editor's configurable settings: the built-in defaults, with what the config
/// file names laid over them by [`Config::load`].
#[derive(Clone, Debug)]
pub struct Config {
    /// Which fonts the editor opens for prose and code.
    pub fonts: FontConfig,
    /// Mask gamma applied to glyph coverage in the fragment shader. Linear-light
    /// compositing renders light-on-dark text heavier than the gamma-space stacks
    /// beside it; raising coverage to this power thins the anti-aliased edges back
    /// to a matching weight. Higher is thinner; 1.0 disables it.
    pub glyph_coverage_gamma: f32,
    /// The same mask gamma for a fenced block's glyphs. A block renders a step
    /// smaller on the code panel, where its thin monospace stems are almost all
    /// anti-aliased edge, so the prose thinning leaves them faint; a gentler value
    /// keeps a block as crisp as the prose beside it. Higher is thinner. An inline
    /// code span is not covered: it renders at the prose size inside a line of
    /// prose, so it takes [`Self::glyph_coverage_gamma`] and reads at the weight of
    /// its own sentence.
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
    /// AI text actions: the settings for the subscription-backed transforms offered
    /// over a selection.
    pub ai: AiConfig,
}

/// AI text-action settings: whether the feature is offered and which backend runs it.
/// The action table itself lives in [`crate::ai`] as a built-in until there is a
/// reason to name it here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiConfig {
    /// Whether the context menu offers AI transforms over a selection. On by default;
    /// each action still runs only on an explicit click, and hands the selection to a
    /// local tool that sends it to the provider's cloud model, so a user who would
    /// rather it never leave the machine turns it off with `ai off` in the config.
    pub enabled: bool,
    /// The command-line tool that runs the transforms (`ai provider claude` or
    /// `ai provider codex`). Claude by default.
    pub provider: crate::ai::Provider,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: crate::ai::Provider::Claude,
        }
    }
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
            // Around 1.8 undoes the linear-light fattening on its own, and a
            // little above that also matches the browser's edge lightening. 1.5
            // stays under both, so prose keeps some of the fattening: bolder,
            // with edges a little heavier than the browser's.
            glyph_coverage_gamma: 1.5,
            // A fenced block renders at a smaller size where the prose thinning
            // reads as faint, so it takes almost none: 1.0 leaves the coverage
            // untouched.
            code_glyph_coverage_gamma: 1.0,
            line_height_scale: 1.5,
            caret: CaretConfig::default(),
            // One vault to start, the notes folder in the user's home, for a config
            // file that names none (or is not there yet). A config's `vault` lines
            // replace this list; the rail's "Save as vault" writes the first one.
            // Resolved like a configured line, so every path in the vault list has
            // been through the same door and comparing two of them means something.
            vaults: vaults([resolve_path(&expand_tilde("~/notes"))]),
            ai: AiConfig::default(),
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
        // The home notes folder, resolved the way a config line is (asserting the
        // name instead would depend on whether ~/notes is a symlink where the test runs).
        assert_eq!(
            config.vaults[0].path,
            resolve_path(&expand_tilde("~/notes"))
        );
        assert_eq!(config.vaults[0].kind, VaultKind::Saved);
    }

    #[test]
    fn vault_initials_are_the_first_two_letters_upper_cased() {
        assert_eq!(vault_initials(Path::new("/home/user/notes")), "NO");
        assert_eq!(vault_initials(Path::new("/home/user/Work")), "WO");
    }

    #[test]
    fn vault_initials_take_what_a_short_name_has() {
        assert_eq!(vault_initials(Path::new("/home/user/w")), "W");
    }

    #[test]
    fn vault_initials_never_run_past_two_characters() {
        // ß upper-cases into two letters on its own, which must not push the chip to
        // three or swallow the second character's own initial.
        assert_eq!(vault_initials(Path::new("/home/user/ßb")), "SB");
        assert_eq!(vault_initials(Path::new("/home/user/ßß")), "SS");
    }

    #[test]
    fn vault_initials_keep_a_non_latin_name() {
        // A name with no case of its own passes through, rather than emptying the
        // chip: two characters in, two out.
        assert_eq!(vault_initials(Path::new("/home/user/日本語")), "日本");
    }

    /// The vaults for a list of names under one parent.
    fn named(names: &[&str]) -> Vec<Vault> {
        vaults(names.iter().map(|n| PathBuf::from("/home/user").join(n)))
    }

    #[test]
    fn a_configured_vault_carries_its_chip() {
        let vs = named(&["notes"]);
        assert_eq!(vs[0].path, PathBuf::from("/home/user/notes"));
        assert_eq!(vs[0].initials, "NO");
        assert!(VAULT_COLORS.contains(&vs[0].color));
    }

    #[test]
    fn a_full_palette_of_vaults_is_all_different_colors() {
        // The point of the color: no two chips in a configured list look alike, so
        // the rail is glanceable rather than a column of near-identical squares.
        let names = [
            "notes", "work", "journal", "archive", "recipes", "code", "personal", "wiki",
        ];
        assert_eq!(names.len(), VAULT_COLORS.len());
        let colors: std::collections::BTreeSet<u32> =
            named(&names).iter().map(|v| v.color).collect();
        assert_eq!(colors.len(), VAULT_COLORS.len());
    }

    #[test]
    fn vaults_past_the_palette_wrap_rather_than_run_out() {
        let names: Vec<String> = (0..VAULT_COLORS.len() + 2)
            .map(|i| format!("vault{i}"))
            .collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let vs = named(&refs);
        assert_eq!(vs.len(), VAULT_COLORS.len() + 2);
        // The ninth vault comes back round to the first color, not off the end.
        assert_eq!(vs[VAULT_COLORS.len()].color, vs[0].color);
    }

    #[test]
    fn vaults_with_no_paths_configure_nothing() {
        assert!(vaults([]).is_empty());
    }

    #[test]
    fn a_configured_vault_is_saved_not_temporary() {
        assert_eq!(named(&["notes"])[0].kind, VaultKind::Saved);
    }

    /// The vault paths a parsed config names, which is what every parse test is
    /// really asking about.
    fn parsed(text: &str) -> Vec<PathBuf> {
        parse(text).vaults.into_iter().map(|v| v.path).collect()
    }

    /// The paths the built-in defaults name: what a config that names no vault of
    /// its own leaves standing.
    fn default_vault_paths() -> Vec<PathBuf> {
        Config::default()
            .vaults
            .into_iter()
            .map(|v| v.path)
            .collect()
    }

    #[test]
    fn parse_reads_the_vaults_in_file_order() {
        assert_eq!(
            parsed("vault /home/user/notes\nvault /home/user/work\n"),
            [
                PathBuf::from("/home/user/notes"),
                PathBuf::from("/home/user/work")
            ]
        );
    }

    #[test]
    fn parse_ignores_blank_and_comment_lines() {
        assert_eq!(
            parsed("\n# my vaults\n   \nvault /home/user/notes\n"),
            [PathBuf::from("/home/user/notes")]
        );
    }

    #[test]
    fn parse_tolerates_unknown_fields() {
        // A config written for a later version (or for a knob still living in the
        // source) loads rather than being refused.
        assert_eq!(
            parsed("future_option whatever\nvault /home/user/notes\nmystery 42\n"),
            [PathBuf::from("/home/user/notes")]
        );
    }

    #[test]
    fn parse_tolerates_extra_whitespace() {
        assert_eq!(
            parsed("   vault    /home/user/notes   \n"),
            [PathBuf::from("/home/user/notes")]
        );
    }

    #[test]
    fn parse_keeps_a_vault_path_with_spaces_in_it() {
        // Only the first whitespace run splits the field from its value, so a folder
        // named like prose survives.
        assert_eq!(
            parsed("vault /home/user/My Notes\n"),
            [PathBuf::from("/home/user/My Notes")]
        );
    }

    #[test]
    fn parse_skips_a_vault_field_with_no_path() {
        // Nothing named means nothing configured, so the defaults stand.
        assert_eq!(parsed("vault\nvault   \n"), default_vault_paths());
    }

    #[test]
    fn parse_drops_a_repeated_vault() {
        assert_eq!(
            parsed("vault /home/user/notes\nvault /home/user/notes\n"),
            [PathBuf::from("/home/user/notes")]
        );
    }

    #[test]
    fn parse_expands_a_tilde_path() {
        let Some(home) = std::env::var_os("HOME").filter(|s| !s.is_empty()) else {
            return;
        };
        assert_eq!(
            parsed("vault ~/notes\n"),
            [resolve_path(&PathBuf::from(home).join("notes"))]
        );
    }

    #[test]
    fn parse_caps_the_vault_list() {
        // The list is input-fed, so it is bounded on the way in: a config naming a
        // thousand vaults cannot grow the rail past the cap.
        let text: String = (0..MAX_VAULTS + 10)
            .map(|i| format!("vault /home/user/v{i}\n"))
            .collect();
        assert_eq!(parsed(&text).len(), MAX_VAULTS);
    }

    #[test]
    fn parse_with_no_vaults_keeps_the_default_ones() {
        // A field absent from the file keeps its default, so a config about something
        // else does not empty the rail.
        assert_eq!(parsed("# nothing here\n"), default_vault_paths());
        assert_eq!(parsed(""), default_vault_paths());
    }

    #[test]
    fn resolve_path_makes_an_existing_directory_absolute_and_named() {
        // A real directory canonicalizes to an absolute path that still has a final
        // component, so a vault rooted there gets a name (not "." or "").
        let abs = resolve_path(&std::env::temp_dir());
        assert!(abs.is_absolute());
        assert!(abs.file_name().is_some());
    }

    #[test]
    fn resolve_path_passes_through_an_unresolvable_path() {
        // A not-yet-created vault or file cannot be canonicalized, so it is left as
        // given and both still work.
        let p = Path::new("/no/such/path/here.md");
        assert_eq!(resolve_path(p), p);
    }

    /// A scratch directory under the system temp dir, unique per test name so the
    /// parallel test runner does not collide two file-backed configs.
    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-config-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    /// Write `text` as the config in a scratch dir, returning its path.
    fn config_with(name: &str, text: &str) -> PathBuf {
        let path = tmp_dir(name).join(FILENAME);
        std::fs::write(&path, text).expect("write config");
        path
    }

    #[test]
    fn add_vault_writes_the_first_one_into_a_config_that_is_not_there_yet() {
        let path = tmp_dir("first").join(FILENAME);
        let _ = std::fs::remove_file(&path);
        add_vault_to(&path, Path::new("/home/user/notes")).expect("add");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "vault /home/user/notes\n"
        );
    }

    #[test]
    fn add_vault_appends_and_leaves_the_rest_of_the_file_alone() {
        // The whole point of appending: a user's comments, their ordering, and a
        // field this version does not read all survive being written to.
        let before = "# my vaults\nvault /home/user/notes\n\nfuture_option whatever\n";
        let path = config_with("append", before);
        add_vault_to(&path, Path::new("/home/user/work")).expect("add");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            format!("{before}vault /home/user/work\n")
        );
    }

    #[test]
    fn add_vault_does_not_run_onto_a_last_line_without_a_break() {
        let path = config_with("nobreak", "vault /home/user/notes");
        add_vault_to(&path, Path::new("/home/user/work")).expect("add");
        assert_eq!(
            parsed(&std::fs::read_to_string(&path).expect("read")),
            [
                PathBuf::from("/home/user/notes"),
                PathBuf::from("/home/user/work")
            ]
        );
    }

    #[test]
    fn add_vault_leaves_a_vault_the_config_already_names() {
        let before = "vault /home/user/notes\n";
        let path = config_with("dup", before);
        add_vault_to(&path, Path::new("/home/user/notes")).expect("add");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), before);
    }

    #[test]
    fn add_vault_then_load_reads_it_back() {
        let path = tmp_dir("roundtrip").join(FILENAME);
        let _ = std::fs::remove_file(&path);
        add_vault_to(&path, Path::new("/home/user/notes")).expect("add");
        add_vault_to(&path, Path::new("/home/user/work")).expect("add");
        let vaults = load_from(&path).vaults;
        assert_eq!(
            vaults.iter().map(|v| v.path.clone()).collect::<Vec<_>>(),
            [
                PathBuf::from("/home/user/notes"),
                PathBuf::from("/home/user/work")
            ]
        );
        // Read back from the config, both are saved vaults, not temporary ones.
        assert!(vaults.iter().all(|v| v.kind == VaultKind::Saved));
    }

    #[test]
    fn add_vault_refuses_a_path_the_line_format_cannot_hold() {
        // No escaping in the format, so a path a parse could not read back is an
        // error rather than a line that silently names another directory.
        let path = tmp_dir("unwritable").join(FILENAME);
        let _ = std::fs::remove_file(&path);
        assert!(add_vault_to(&path, Path::new("/home/user/two\nlines")).is_err());
        assert!(add_vault_to(&path, Path::new("/home/user/padded ")).is_err());
        assert!(add_vault_to(&path, Path::new("")).is_err());
        // None of them left a config behind.
        assert!(!path.exists());
    }

    #[test]
    fn load_from_a_missing_config_is_the_defaults() {
        let path = tmp_dir("missing").join("not-written");
        let _ = std::fs::remove_file(&path);
        assert_eq!(load_from(&path).vaults, Config::default().vaults);
    }

    #[test]
    fn load_from_an_oversized_config_is_the_defaults() {
        // A note saved over the config by accident is refused, not read in.
        let path = config_with("huge", &"# padding\n".repeat(CONFIG_MAX_BYTES as usize));
        assert_eq!(load_from(&path).vaults, Config::default().vaults);
    }

    #[test]
    fn ai_is_on_by_default_and_the_config_can_turn_it_off() {
        // Silent by default, and the last ai line wins.
        assert!(parse("").ai.enabled);
        assert!(parse("vault ~/notes").ai.enabled);
        assert!(!parse("ai off").ai.enabled);
        assert!(!parse("ai true\nai off").ai.enabled);
        assert!(parse("ai off\nai on").ai.enabled);
        // A value that is neither on nor off leaves the default standing.
        assert!(parse("ai maybe").ai.enabled);
    }

    #[test]
    fn ai_provider_defaults_to_claude_and_the_config_can_pick_codex() {
        use crate::ai::Provider;
        assert_eq!(parse("").ai.provider, Provider::Claude);
        assert_eq!(parse("ai provider codex").ai.provider, Provider::Codex);
        // Independent of the on/off toggle, and the last one wins.
        let c = parse("ai off\nai provider codex");
        assert!(!c.ai.enabled);
        assert_eq!(c.ai.provider, Provider::Codex);
        assert_eq!(
            parse("ai provider codex\nai provider claude").ai.provider,
            Provider::Claude
        );
    }
}
