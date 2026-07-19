//! On-disk session state: a small binary blob at `~/.config/bnknotes/state.bin`
//! remembering the pane layout across restarts, so reopening bnknotes restores
//! the active vault and its expanded folders, the column widths, and each
//! content pane's open tabs with their cursors and scroll. Each vault's own
//! settings ride along, keyed by its root, so they follow the vault rather than
//! the window.
//!
//! The format is a 4-byte magic (`BNKN`), a version byte, little-endian
//! fixed-width fields, and length-prefixed UTF-8 strings, written atomically
//! through a temp file + rename. [`encode`] and [`decode`] below are its two
//! halves and must stay in step.
//!
//! A corrupt or wrong-version file is not fatal: losing the last layout must
//! never stop the editor opening, so [`load`] logs and falls back to
//! [`Session::default`].
//!
//! Because several bnknotes can share one state file, a save is a read-merge-write
//! under an exclusive lock, not a blind overwrite: it re-reads the latest state and
//! replaces only the window fields and its own active vault, so a sibling instance
//! editing another vault does not lose its tabs. See [`save_to`] and [`merged_session`].

use std::path::{Path, PathBuf};

use crate::icon::VaultIcon;
use crate::platform::error::{Error, Result};

const FILENAME: &str = "state.bin";
/// Sibling of the state file, flock'd for the duration of a save so concurrent
/// bnknotes instances serialize their read-merge-write instead of racing.
const LOCK_FILENAME: &str = "state.lock";
const MAGIC: &[u8; 4] = b"BNKN";
/// The version written. Older versions still readable are listed in
/// [`MIN_VERSION`]; a decode fills anything they predate with its default, so a
/// schema addition costs the user nothing.
const VERSION: u8 = 7;
/// The oldest version [`decode`] understands. v6 is v7 without the per-tab fold
/// list; anything older is refused and the session falls back to defaults.
const MIN_VERSION: u8 = 6;
/// Cap on folded headings persisted per tab, matching the in-memory cap so a
/// state file cannot ask for a set the editor would not build.
const MAX_FOLDS: usize = crate::fold::MAX_FOLDS;

/// One vault's remembered state, keyed by its root directory: everything that
/// belongs to a vault's contents rather than to the window. What was open where
/// travels with the vault, so reopening a vault restores its tabs and its file
/// tree, not the last window's.
#[derive(Debug, Clone, PartialEq)]
pub struct VaultSession {
    pub root: PathBuf,
    /// Whether the file tree lists folders holding no note anywhere beneath them.
    pub show_empty_dirs: bool,
    /// The folders the file tree had expanded, as absolute paths under `root`.
    pub expanded: Vec<PathBuf>,
    /// Each content pane's open tabs and active index while this vault was shown.
    pub panes: [PaneGroupSession; 2],
}

impl VaultSession {
    /// A vault seen for the first time: its tree lists every folder (nothing
    /// hidden until asked), nothing expanded, and no tabs open.
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            show_empty_dirs: true,
            expanded: Vec::new(),
            panes: [PaneGroupSession::default(), PaneGroupSession::default()],
        }
    }
}

/// The remembered state of every vault the user has touched, not just the active
/// one, so switching vaults (or coming back to one) restores what that vault was
/// showing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VaultPrefs {
    vaults: Vec<VaultSession>,
}

impl VaultPrefs {
    /// The stored state for `root`, or `None` if this vault has never been seen.
    pub fn get(&self, root: &Path) -> Option<&VaultSession> {
        self.vaults.iter().find(|v| v.root == root)
    }

    /// Replace `vault.root`'s entry with `vault`, or append it if that vault is new.
    fn insert(&mut self, vault: VaultSession) {
        match self.vaults.iter().position(|v| v.root == vault.root) {
            Some(i) => self.vaults[i] = vault,
            None => self.vaults.push(vault),
        }
    }

    /// The entry for `root`, inserting a first-run one if this vault is new.
    fn entry_mut(&mut self, root: &Path) -> &mut VaultSession {
        match self.vaults.iter().position(|v| v.root == root) {
            Some(i) => &mut self.vaults[i],
            None => {
                self.vaults.push(VaultSession::new(root.to_path_buf()));
                self.vaults.last_mut().expect("just pushed an entry")
            }
        }
    }

    /// Whether `root`'s file tree lists folders with no notes. A vault seen for the
    /// first time lists them: nothing is hidden until the user asks for it.
    pub fn show_empty_dirs(&self, root: &Path) -> bool {
        self.get(root).is_none_or(|v| v.show_empty_dirs)
    }

    /// Remember `show` for `root`, replacing that vault's entry or adding one.
    pub fn set_show_empty_dirs(&mut self, root: &Path, show: bool) {
        self.entry_mut(root).show_empty_dirs = show;
    }

    /// Fold the active vault's live layout back into its entry, adding one if this
    /// vault is new. The other vaults' entries are left untouched, so each vault
    /// keeps the tabs and expanded folders it last had.
    pub fn set_layout(
        &mut self,
        root: &Path,
        expanded: Vec<PathBuf>,
        panes: [PaneGroupSession; 2],
    ) {
        let entry = self.entry_mut(root);
        entry.expanded = expanded;
        entry.panes = panes;
    }
}

/// One open tab: its file, where the caret and viewport sat in it, and which
/// heading sections were folded.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaneSession {
    pub path: Option<PathBuf>,
    pub cursor: u64,
    pub scroll_y: i32,
    /// The logical line numbers of the folded headings, ascending. Like `cursor`,
    /// these are positions into the file as it was last seen: a line that no longer
    /// holds a heading is dropped when the document is re-indexed, so an external
    /// edit degrades to fewer folds rather than to nonsense.
    pub folds: Vec<u32>,
}

/// A content pane's remembered stack of open tabs and which one was active. An
/// empty tab list is a never-opened pane, restored as one blank tab.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaneGroupSession {
    pub tabs: Vec<PaneSession>,
    pub active: u32,
}

/// Which region last held keyboard focus, so it is restored on reopen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    PaneA,
    PaneB,
}

impl Focus {
    fn tag(self) -> u8 {
        match self {
            Focus::Tree => 0,
            Focus::PaneA => 1,
            Focus::PaneB => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Focus::Tree),
            1 => Ok(Focus::PaneA),
            2 => Ok(Focus::PaneB),
            other => Err(Error::msg(format!("invalid focus tag {other}"))),
        }
    }
}

/// The whole persisted layout. [`Session::default`] is the first-run layout
/// (tree plus both content panes visible), used when there is no saved file.
#[derive(Debug, Clone, PartialEq)]
pub struct Session {
    pub window: (u32, u32),
    pub tree_width: u32,
    pub split_ratio: f32,
    pub tree_visible: bool,
    pub second_open: bool,
    pub focus: Focus,
    /// The active vault's root directory: the file tree roots here and every note
    /// reference resolves against it. `None` before a first vault is chosen. The
    /// active vault's tabs and expanded folders live in its [`VaultSession`],
    /// keyed by this root, not on the window.
    pub active_vault: Option<PathBuf>,
    /// The per-vault state, the active vault's among them.
    pub vaults: VaultPrefs,
    /// How the rail is arranged: its chips in order, each with the color it wears.
    /// Empty until a rail has been shown once.
    ///
    /// This sits on the window rather than in the per-vault entries because it is a
    /// fact about the whole rail, not about any one vault: an instance parked in one
    /// vault can only write that vault's entry (see [`merged_session`]), and could
    /// never record where the other chips sit.
    pub rail: Vec<RailVault>,
}

/// A chip's arrangement: which vault it is, and the color it wears. Its place in
/// [`Session::rail`] is its place in the rail.
///
/// The color is remembered rather than re-derived because the rail can be dragged
/// into any order: a color dealt by position would repaint every chip the drag
/// passed, when a chip's color is the one thing that says which vault it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RailVault {
    pub root: PathBuf,
    pub color: u32,
    /// The picture the chip was given, and the square of it shown. Stored rather
    /// than re-rolled at each launch: the roll is the user's, made once from the
    /// rail's menu, and a chip that dealt itself a new picture every morning would
    /// stop being the chip they know.
    pub icon: Option<VaultIcon>,
}

/// First-run defaults: sidebar and both content panes shown, an even split, no
/// remembered files. Column widths are clamped by [`Session::sanitized`] on the
/// way in, so these are also the fallback bounds.
impl Default for Session {
    fn default() -> Self {
        Self {
            window: (1100, 720),
            tree_width: 260,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
            focus: Focus::PaneA,
            active_vault: None,
            vaults: VaultPrefs::default(),
            rail: Vec::new(),
        }
    }
}

impl Session {
    /// Clamp externally-sourced fields into sane ranges, so a hand-edited or
    /// truncated file can never yield a zero-width sidebar or an off-screen
    /// split. Applied after every decode.
    fn sanitized(mut self) -> Self {
        // A window smaller than this is unusable, and one past 16384 (the app's
        // surface-dimension cap) risks overflowing the buffer-size arithmetic, so
        // clamp a hand-edited or truncated size into a sane range.
        self.window = (
            self.window.0.clamp(400, 16384),
            self.window.1.clamp(300, 16384),
        );
        self.tree_width = self.tree_width.clamp(120, 800);
        self.split_ratio = if self.split_ratio.is_finite() {
            self.split_ratio.clamp(0.1, 0.9)
        } else {
            0.5
        };
        // A chip color is 0x00RRGGBB, so a hand-edited one with anything in the top
        // byte is masked back to a color the painter can mean.
        for chip in &mut self.rail {
            chip.color &= 0x00ff_ffff;
        }
        // One vault, one chip: a file naming a vault twice would otherwise arrange two
        // chips for it, and a rail cannot say which of them the vault is.
        let mut seen: Vec<PathBuf> = Vec::new();
        self.rail.retain(|chip| {
            let fresh = !seen.contains(&chip.root);
            if fresh {
                seen.push(chip.root.clone());
            }
            fresh
        });
        self
    }
}

/// Path to the state file, `None` when the config directory does not resolve
/// (then state simply is not persisted). The state file sits beside the config
/// file; [`crate::config::config_dir`] is where that directory is decided.
pub fn session_path() -> Option<PathBuf> {
    Some(crate::config::config_dir()?.join(FILENAME))
}

/// Load the saved layout, or the first-run default when there is none or it is
/// unreadable. Never fails: a missing, corrupt, or wrong-version file logs (once)
/// and falls back, because a bad layout file must not stop the editor opening.
pub fn load() -> Session {
    match session_path() {
        Some(path) => load_from(&path),
        None => Session::default(),
    }
}

/// Read and decode the state at `path`, falling back to the first-run default on a
/// missing, unreadable, corrupt, or wrong-version file. A plain absence is silent;
/// anything else logs once. Reads need no lock: an atomic rename means a concurrent
/// save is seen as either the whole old file or the whole new one, never a tear.
fn load_from(path: &Path) -> Session {
    match std::fs::read(path) {
        Ok(bytes) => decode(&bytes).map(Session::sanitized).unwrap_or_else(|e| {
            eprintln!(
                "bnknotes: ignoring unreadable state file {}: {e}",
                path.display()
            );
            Session::default()
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Session::default(),
        Err(e) => {
            eprintln!("bnknotes: cannot read state file {}: {e}", path.display());
            Session::default()
        }
    }
}

/// Persist the layout, creating the config directory if needed. Best-effort at the
/// call site: a failure to persist is reported, never fatal.
pub fn save(session: &Session) -> Result<()> {
    let path =
        session_path().ok_or_else(|| Error::msg("no XDG_CONFIG_HOME or HOME to save state"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    save_to(&path, session)
}

/// Persist `session` to `path`, but never by blind overwrite: with a second
/// bnknotes open on another vault, that vault's tabs live only in the other
/// process, so writing this instance's whole view would drop them. Instead the
/// read, merge, and atomic write run as one step under an exclusive lock, and only
/// the window fields and this instance's own active vault are taken from `session`;
/// every other vault keeps what is on disk. See [`merged_session`].
fn save_to(path: &Path, session: &Session) -> Result<()> {
    let _lock = crate::file::FileLock::acquire(&path.with_file_name(LOCK_FILENAME));
    let merged = merged_session(&load_from(path), session);
    crate::file::save_atomic(path, &encode(&merged)?)
}

/// Fold a saving instance's live state onto the current on-disk state. The window
/// geometry, focus, and active vault come from the instance (last writer wins,
/// since they describe its window). Of the vaults, only the instance's active one
/// is replaced, the single entry it can have changed; every other vault keeps the
/// tabs and folders it has on disk, so a concurrent instance's vault survives.
fn merged_session(disk: &Session, incoming: &Session) -> Session {
    let mut out = disk.clone();
    out.window = incoming.window;
    out.tree_width = incoming.tree_width;
    out.split_ratio = incoming.split_ratio;
    out.tree_visible = incoming.tree_visible;
    out.second_open = incoming.second_open;
    out.focus = incoming.focus;
    out.active_vault = incoming.active_vault.clone();
    // The rail describes the window's own chips, so the saver's arrangement wins
    // whole. Merging two arrangements has no meaning: they are orderings of the same
    // list, and half of each is neither.
    out.rail = incoming.rail.clone();
    if let Some(entry) = incoming
        .active_vault
        .as_deref()
        .and_then(|root| incoming.vaults.get(root))
    {
        out.vaults.insert(entry.clone());
    }
    out
}

fn encode(s: &Session) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&s.window.0.to_le_bytes());
    out.extend_from_slice(&s.window.1.to_le_bytes());
    out.extend_from_slice(&s.tree_width.to_le_bytes());
    out.extend_from_slice(&s.split_ratio.to_le_bytes());
    out.push(u8::from(s.tree_visible));
    out.push(u8::from(s.second_open));
    out.push(s.focus.tag());
    write_opt_path(&mut out, s.active_vault.as_deref())?;

    let vaults = u32::try_from(s.vaults.vaults.len())
        .map_err(|_| Error::msg("too many vaults to persist"))?;
    out.extend_from_slice(&vaults.to_le_bytes());
    for vault in &s.vaults.vaults {
        write_vault(&mut out, vault)?;
    }

    let rail = u32::try_from(s.rail.len()).map_err(|_| Error::msg("too many chips to persist"))?;
    out.extend_from_slice(&rail.to_le_bytes());
    for chip in &s.rail {
        write_path(&mut out, &chip.root)?;
        out.extend_from_slice(&chip.color.to_le_bytes());
        write_icon(&mut out, chip.icon.as_ref())?;
    }
    Ok(out)
}

/// A chip's picture: an optional path, then the square of it, written only when
/// there is one. Mirror of [`read_icon`].
fn write_icon(out: &mut Vec<u8>, icon: Option<&VaultIcon>) -> Result<()> {
    // The path carries the tag: a picture whose path cannot be written (a non-UTF-8
    // path) is stored as no picture at all, rather than as a square of nothing.
    let icon = icon.filter(|i| i.source.to_str().is_some());
    write_opt_path(out, icon.map(|i| i.source.as_path()))?;
    if let Some(icon) = icon {
        out.extend_from_slice(&icon.x.to_le_bytes());
        out.extend_from_slice(&icon.y.to_le_bytes());
        out.extend_from_slice(&icon.side.to_le_bytes());
    }
    Ok(())
}

/// Mirror of [`write_icon`]: the square follows the path, and only when there is one.
fn read_icon(r: &mut Reader) -> Result<Option<VaultIcon>> {
    let Some(source) = r.opt_path()? else {
        return Ok(None);
    };
    Ok(Some(VaultIcon {
        source,
        x: r.u32()?,
        y: r.u32()?,
        side: r.u32()?,
    }))
}

/// One vault's block: its root and filter, then the folders it had expanded and
/// each content pane's tabs. Mirror of [`read_vault`].
fn write_vault(out: &mut Vec<u8>, vault: &VaultSession) -> Result<()> {
    write_path(out, &vault.root)?;
    out.push(u8::from(vault.show_empty_dirs));

    let expanded = u32::try_from(vault.expanded.len())
        .map_err(|_| Error::msg("too many expanded folders to persist"))?;
    out.extend_from_slice(&expanded.to_le_bytes());
    for dir in &vault.expanded {
        write_path(out, dir)?;
    }
    for pane in &vault.panes {
        write_pane_group(out, pane)?;
    }
    Ok(())
}

/// One pane's tab list and active index. Mirror of [`read_pane_group`].
fn write_pane_group(out: &mut Vec<u8>, pane: &PaneGroupSession) -> Result<()> {
    let count =
        u32::try_from(pane.tabs.len()).map_err(|_| Error::msg("too many tabs to persist"))?;
    out.extend_from_slice(&count.to_le_bytes());
    for tab in &pane.tabs {
        write_opt_path(out, tab.path.as_deref())?;
        out.extend_from_slice(&tab.cursor.to_le_bytes());
        out.extend_from_slice(&tab.scroll_y.to_le_bytes());
        let folds = u32::try_from(tab.folds.len().min(MAX_FOLDS))
            .map_err(|_| Error::msg("too many folds to persist"))?;
        out.extend_from_slice(&folds.to_le_bytes());
        for line in tab.folds.iter().take(MAX_FOLDS) {
            out.extend_from_slice(&line.to_le_bytes());
        }
    }
    out.extend_from_slice(&pane.active.to_le_bytes());
    Ok(())
}

fn decode(data: &[u8]) -> Result<Session> {
    let mut r = Reader::new(data);
    if r.take(4)? != MAGIC {
        return Err(Error::msg("not a bnknotes state file (bad magic)"));
    }
    let version = r.u8()?;
    if !(MIN_VERSION..=VERSION).contains(&version) {
        return Err(Error::msg(format!(
            "unsupported state version {version} (expected {MIN_VERSION}..={VERSION})"
        )));
    }
    r.version = version;
    let window = (r.u32()?, r.u32()?);
    let tree_width = r.u32()?;
    let split_ratio = r.f32()?;
    let tree_visible = r.u8()? != 0;
    let second_open = r.u8()? != 0;
    let focus = Focus::from_tag(r.u8()?)?;
    let active_vault = r.opt_path()?;

    let vault_count = r.u32()? as usize;
    let mut vaults = Vec::with_capacity(vault_count.min(4096));
    for _ in 0..vault_count {
        vaults.push(read_vault(&mut r)?);
    }

    let rail_count = r.u32()? as usize;
    let mut rail = Vec::with_capacity(rail_count.min(4096));
    for _ in 0..rail_count {
        rail.push(RailVault {
            root: r.path()?,
            color: r.u32()?,
            icon: read_icon(&mut r)?,
        });
    }
    r.expect_end()?;
    Ok(Session {
        window,
        tree_width,
        split_ratio,
        tree_visible,
        second_open,
        focus,
        active_vault,
        vaults: VaultPrefs { vaults },
        rail,
    })
}

fn read_vault(r: &mut Reader) -> Result<VaultSession> {
    let root = r.path()?;
    let show_empty_dirs = r.u8()? != 0;
    let expanded_count = r.u32()? as usize;
    let mut expanded = Vec::with_capacity(expanded_count.min(4096));
    for _ in 0..expanded_count {
        expanded.push(r.path()?);
    }
    let panes = [read_pane_group(r)?, read_pane_group(r)?];
    Ok(VaultSession {
        root,
        show_empty_dirs,
        expanded,
        panes,
    })
}

fn read_pane_group(r: &mut Reader) -> Result<PaneGroupSession> {
    let count = r.u32()? as usize;
    let mut tabs = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        tabs.push(read_pane(r)?);
    }
    let active = r.u32()?;
    Ok(PaneGroupSession { tabs, active })
}

fn read_pane(r: &mut Reader) -> Result<PaneSession> {
    let path = r.opt_path()?;
    let cursor = r.u64()?;
    let scroll_y = r.i32()?;
    // v6 predates folds, so a tab restored from one simply opens unfolded.
    let mut folds = Vec::new();
    if r.version >= 7 {
        let count = r.u32()? as usize;
        folds.reserve(count.min(MAX_FOLDS));
        for _ in 0..count {
            let line = r.u32()?;
            if folds.len() < MAX_FOLDS {
                folds.push(line);
            }
        }
    }
    Ok(PaneSession {
        path,
        cursor,
        scroll_y,
        folds,
    })
}

/// Paths persist as UTF-8. A non-UTF-8 path (rare on Linux, where paths are
/// bytes) is stored as `None` rather than failing the whole save; the layout is
/// convenience state, not the document.
fn write_opt_path(out: &mut Vec<u8>, path: Option<&Path>) -> Result<()> {
    match path.and_then(Path::to_str) {
        Some(s) => {
            out.push(1);
            write_string(out, s)
        }
        None => {
            out.push(0);
            Ok(())
        }
    }
}

fn write_path(out: &mut Vec<u8>, path: &Path) -> Result<()> {
    write_string(out, path.to_str().unwrap_or(""))
}

fn write_string(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let len = u16::try_from(s.len()).map_err(|_| Error::msg("string too long for state codec"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

/// Sequential reader over the state blob. Wraps the shared bounds-checked
/// [`crate::platform::bytes::Cursor`], decoding every field little-endian (the on-disk
/// format is fixed little-endian, independent of the host).
struct Reader<'a> {
    cursor: crate::platform::bytes::Cursor<'a>,
    /// The version of the file being read, set once the header is parsed, so a
    /// nested reader knows which fields the format it is reading actually carries.
    version: u8,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cursor: crate::platform::bytes::Cursor::new(data),
            version: VERSION,
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        self.cursor.take(n).ok_or_else(|| {
            Error::msg(format!(
                "unexpected end of state at offset {} (need {n} more)",
                self.cursor.pos()
            ))
        })
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u16()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|_| Error::msg("invalid utf-8 in state string"))
    }

    fn path(&mut self) -> Result<PathBuf> {
        Ok(PathBuf::from(self.string()?))
    }

    fn opt_path(&mut self) -> Result<Option<PathBuf>> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.path()?)),
            tag => Err(Error::msg(format!("invalid optional tag {tag}"))),
        }
    }

    fn expect_end(&self) -> Result<()> {
        let left = self.cursor.remaining();
        if left != 0 {
            return Err(Error::msg(format!("trailing {left} bytes after state")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Session {
        Session {
            window: (1280, 800),
            tree_width: 300,
            split_ratio: 0.4,
            tree_visible: true,
            second_open: false,
            focus: Focus::PaneB,
            active_vault: Some(PathBuf::from("/home/user/notes")),
            vaults: VaultPrefs {
                vaults: vec![
                    VaultSession {
                        root: PathBuf::from("/home/user/notes"),
                        show_empty_dirs: false,
                        expanded: vec![
                            PathBuf::from("/home/user/notes"),
                            PathBuf::from("/home/user/notes/journal"),
                        ],
                        panes: [
                            PaneGroupSession {
                                tabs: vec![
                                    PaneSession {
                                        path: Some(PathBuf::from("/home/user/notes/todo.md")),
                                        cursor: 42,
                                        scroll_y: 128,
                                        folds: vec![3, 17, 40],
                                    },
                                    PaneSession {
                                        path: Some(PathBuf::from(
                                            "/home/user/notes/journal/day.md",
                                        )),
                                        cursor: 7,
                                        scroll_y: 0,
                                        folds: Vec::new(),
                                    },
                                ],
                                active: 1,
                            },
                            PaneGroupSession {
                                tabs: vec![PaneSession {
                                    path: None,
                                    cursor: 0,
                                    scroll_y: -5,
                                    folds: Vec::new(),
                                }],
                                active: 0,
                            },
                        ],
                    },
                    VaultSession {
                        root: PathBuf::from("/home/user/work"),
                        show_empty_dirs: true,
                        expanded: vec![PathBuf::from("/home/user/work")],
                        panes: [
                            PaneGroupSession {
                                tabs: vec![PaneSession {
                                    path: Some(PathBuf::from("/home/user/work/spec.md")),
                                    cursor: 3,
                                    scroll_y: 12,
                                    folds: vec![0],
                                }],
                                active: 0,
                            },
                            PaneGroupSession::default(),
                        ],
                    },
                ],
            },
            // Dragged into the reverse of the config's order, each chip wearing a
            // color of its own, and one of them a picture.
            rail: vec![
                RailVault {
                    root: PathBuf::from("/home/user/work"),
                    color: 0x0098_c379,
                    icon: Some(VaultIcon {
                        source: PathBuf::from("/home/user/work/@attachment/shot.png"),
                        x: 120,
                        y: 48,
                        side: 300,
                    }),
                },
                RailVault {
                    root: PathBuf::from("/home/user/notes"),
                    color: 0x005c_9cf6,
                    icon: None,
                },
            ],
        }
    }

    #[test]
    fn default_round_trips() {
        let bytes = encode(&Session::default()).expect("encode");
        assert_eq!(decode(&bytes).expect("decode"), Session::default());
    }

    #[test]
    fn full_session_round_trips() {
        let original = sample();
        let bytes = encode(&original).expect("encode");
        assert_eq!(decode(&bytes).expect("decode"), original);
    }

    #[test]
    fn encode_starts_with_magic_and_version() {
        let bytes = encode(&Session::default()).expect("encode");
        assert_eq!(&bytes[0..4], MAGIC);
        assert_eq!(bytes[4], VERSION);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = encode(&sample()).expect("encode");
        bytes[0] = b'X';
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut bytes = encode(&sample()).expect("encode");
        bytes[4] = 99;
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn decode_rejects_truncation() {
        let bytes = encode(&sample()).expect("encode");
        assert!(decode(&bytes[..bytes.len() - 3]).is_err());
    }

    #[test]
    fn decode_rejects_trailing_bytes() {
        let mut bytes = encode(&sample()).expect("encode");
        bytes.push(0xAB);
        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn unicode_paths_round_trip() {
        let root = PathBuf::from("/home/user/notità/日本語");
        let s = Session {
            active_vault: Some(root.clone()),
            vaults: VaultPrefs {
                vaults: vec![VaultSession {
                    root: root.clone(),
                    show_empty_dirs: true,
                    expanded: vec![PathBuf::from("/home/user/notità/日本語/café")],
                    panes: [
                        PaneGroupSession {
                            tabs: vec![PaneSession {
                                path: Some(PathBuf::from("/home/user/café.md")),
                                ..Default::default()
                            }],
                            active: 0,
                        },
                        PaneGroupSession::default(),
                    ],
                }],
            },
            ..Session::default()
        };
        let loaded = decode(&encode(&s).expect("encode")).expect("decode");
        assert_eq!(loaded.active_vault, s.active_vault);
        let vault = loaded.vaults.get(&root).expect("vault entry");
        assert_eq!(
            vault.panes[0].tabs[0].path,
            Some(PathBuf::from("/home/user/café.md"))
        );
        assert_eq!(
            vault.expanded,
            vec![PathBuf::from("/home/user/notità/日本語/café")]
        );
    }

    #[test]
    fn multi_tab_vault_round_trips() {
        // A vault whose active pane holds several tabs at a non-zero active index
        // survives the codec, tabs, expanded folders and all.
        let loaded = decode(&encode(&sample()).expect("encode")).expect("decode");
        let notes = loaded
            .vaults
            .get(Path::new("/home/user/notes"))
            .expect("vault");
        assert_eq!(notes.panes[0].tabs.len(), 2);
        assert_eq!(notes.panes[0].active, 1);
        assert_eq!(notes.panes[1].tabs.len(), 1);
        assert_eq!(notes.expanded.len(), 2);
        assert_eq!(loaded, sample());
    }

    #[test]
    fn vault_prefs_round_trip_per_vault() {
        let loaded = decode(&encode(&sample()).expect("encode")).expect("decode");
        // Each vault keeps its own state, the inactive one included: the filter,
        // the expanded folders, and the open tabs all travel with the root.
        assert!(!loaded.vaults.show_empty_dirs(Path::new("/home/user/notes")));
        assert!(loaded.vaults.show_empty_dirs(Path::new("/home/user/work")));
        let work = loaded
            .vaults
            .get(Path::new("/home/user/work"))
            .expect("vault");
        assert_eq!(
            work.panes[0].tabs[0].path,
            Some(PathBuf::from("/home/user/work/spec.md"))
        );
    }

    /// Encode a session the way v6 did: identical to [`encode`] except that a tab
    /// carries no fold list. Kept as a literal transcription of the old writer so
    /// the migration is tested against the real historical bytes, not against a
    /// guess at them, and so a later change to `encode` cannot quietly redefine
    /// what "a v6 file" means.
    fn encode_v6(s: &Session) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(6);
        out.extend_from_slice(&s.window.0.to_le_bytes());
        out.extend_from_slice(&s.window.1.to_le_bytes());
        out.extend_from_slice(&s.tree_width.to_le_bytes());
        out.extend_from_slice(&s.split_ratio.to_le_bytes());
        out.push(u8::from(s.tree_visible));
        out.push(u8::from(s.second_open));
        out.push(s.focus.tag());
        write_opt_path(&mut out, s.active_vault.as_deref()).expect("path");
        out.extend_from_slice(&(s.vaults.vaults.len() as u32).to_le_bytes());
        for vault in &s.vaults.vaults {
            write_path(&mut out, &vault.root).expect("path");
            out.push(u8::from(vault.show_empty_dirs));
            out.extend_from_slice(&(vault.expanded.len() as u32).to_le_bytes());
            for dir in &vault.expanded {
                write_path(&mut out, dir).expect("path");
            }
            for pane in &vault.panes {
                out.extend_from_slice(&(pane.tabs.len() as u32).to_le_bytes());
                for tab in &pane.tabs {
                    write_opt_path(&mut out, tab.path.as_deref()).expect("path");
                    out.extend_from_slice(&tab.cursor.to_le_bytes());
                    out.extend_from_slice(&tab.scroll_y.to_le_bytes());
                }
                out.extend_from_slice(&pane.active.to_le_bytes());
            }
        }
        out.extend_from_slice(&(s.rail.len() as u32).to_le_bytes());
        for chip in &s.rail {
            write_path(&mut out, &chip.root).expect("path");
            out.extend_from_slice(&chip.color.to_le_bytes());
            write_icon(&mut out, chip.icon.as_ref()).expect("icon");
        }
        out
    }

    #[test]
    fn a_v6_state_file_still_loads_and_opens_unfolded() {
        // Adding folds must not cost anyone their session. A v6 file decodes with
        // every field it did carry intact, and its tabs simply open unfolded.
        let mut want = sample();
        for vault in &mut want.vaults.vaults {
            for pane in &mut vault.panes {
                for tab in &mut pane.tabs {
                    tab.folds.clear();
                }
            }
        }
        let loaded = decode(&encode_v6(&sample())).expect("a v6 file must still decode");
        assert_eq!(loaded, want);
    }

    #[test]
    fn a_v6_file_rewrites_as_v7() {
        // Reading old and writing new: what a v6 file becomes after one save.
        let loaded = decode(&encode_v6(&sample())).expect("decode v6");
        let rewritten = encode(&loaded).expect("encode");
        assert_eq!(rewritten[4], VERSION, "saves write the current version");
        assert_eq!(decode(&rewritten).expect("decode v7"), loaded);
    }

    #[test]
    fn a_version_outside_the_supported_range_is_refused() {
        // Too old to understand and newer than this build: both fall back to
        // defaults rather than misreading the bytes.
        let mut old = encode(&sample()).expect("encode");
        old[4] = MIN_VERSION - 1;
        assert!(decode(&old).is_err());
        let mut future = encode(&sample()).expect("encode");
        future[4] = VERSION + 1;
        assert!(decode(&future).is_err());
    }

    #[test]
    fn a_tabs_folds_survive_the_codec() {
        let loaded = decode(&encode(&sample()).expect("encode")).expect("decode");
        let notes = loaded
            .vaults
            .get(Path::new("/home/user/notes"))
            .expect("vault");
        assert_eq!(notes.panes[0].tabs[0].folds, vec![3, 17, 40]);
        assert!(notes.panes[0].tabs[1].folds.is_empty());
    }

    #[test]
    fn a_persisted_fold_list_is_capped() {
        // A state file cannot make a tab carry an unbounded fold set.
        let mut s = sample();
        s.vaults.vaults[0].panes[0].tabs[0].folds = (0..MAX_FOLDS as u32 * 2).collect();
        let loaded = decode(&encode(&s).expect("encode")).expect("decode");
        let notes = loaded
            .vaults
            .get(Path::new("/home/user/notes"))
            .expect("vault");
        assert_eq!(notes.panes[0].tabs[0].folds.len(), MAX_FOLDS);
    }

    #[test]
    fn an_unknown_vault_lists_its_empty_folders() {
        // The first-run answer for a vault with no entry: hide nothing.
        let prefs = VaultPrefs::default();
        assert!(prefs.show_empty_dirs(Path::new("/home/user/fresh")));
    }

    #[test]
    fn setting_a_vault_pref_replaces_that_vaults_entry_only() {
        let mut prefs = sample().vaults;
        prefs.set_show_empty_dirs(Path::new("/home/user/notes"), true);
        prefs.set_show_empty_dirs(Path::new("/home/user/other"), false);
        // The existing entry is updated in place, the new vault appended, and the
        // untouched vault left alone.
        assert_eq!(prefs.vaults.len(), 3);
        assert!(prefs.show_empty_dirs(Path::new("/home/user/notes")));
        assert!(prefs.show_empty_dirs(Path::new("/home/user/work")));
        assert!(!prefs.show_empty_dirs(Path::new("/home/user/other")));
    }

    #[test]
    fn sanitize_clamps_a_degenerate_layout() {
        let s = Session {
            tree_width: 5,
            split_ratio: 9.0,
            window: (10, 999_999),
            ..Session::default()
        }
        .sanitized();
        assert_eq!(s.tree_width, 120);
        assert_eq!(s.split_ratio, 0.9);
        assert_eq!(s.window, (400, 16384));
    }

    #[test]
    fn sanitize_leaves_a_sane_window_alone() {
        let s = Session {
            window: (1280, 800),
            ..Session::default()
        }
        .sanitized();
        assert_eq!(s.window, (1280, 800));
    }

    /// A vault rooted at `root` whose active pane holds the single tab `tab`.
    fn vault_with(root: &str, tab: &str) -> VaultSession {
        VaultSession {
            root: PathBuf::from(root),
            show_empty_dirs: true,
            expanded: Vec::new(),
            panes: [
                PaneGroupSession {
                    tabs: vec![PaneSession {
                        path: Some(PathBuf::from(tab)),
                        ..Default::default()
                    }],
                    active: 0,
                },
                PaneGroupSession::default(),
            ],
        }
    }

    /// A session active on `root`, carrying the given vault entries.
    fn session_on(root: &str, vaults: Vec<VaultSession>) -> Session {
        Session {
            active_vault: Some(PathBuf::from(root)),
            vaults: VaultPrefs { vaults },
            ..Session::default()
        }
    }

    /// The first tab's path of `root`'s vault in `session`, as a string.
    fn tab_of(session: &Session, root: &str) -> String {
        session
            .vaults
            .get(Path::new(root))
            .and_then(|v| v.panes[0].tabs.first())
            .and_then(|t| t.path.as_ref())
            .map(|p| p.display().to_string())
            .unwrap_or_default()
    }

    #[test]
    fn merge_replaces_only_the_incoming_active_vault() {
        // Disk has two vaults; the saver is active on the first and holds a stale
        // copy of the second. The merge takes the saver's first vault and keeps the
        // disk's second, so the other vault's tabs are never overwritten.
        let disk = session_on(
            "/x",
            vec![
                vault_with("/x", "/x/old.md"),
                vault_with("/y", "/y/live.md"),
            ],
        );
        let incoming = session_on(
            "/x",
            vec![
                vault_with("/x", "/x/new.md"),
                vault_with("/y", "/y/stale.md"),
            ],
        );
        let merged = merged_session(&disk, &incoming);
        assert_eq!(tab_of(&merged, "/x"), "/x/new.md");
        assert_eq!(tab_of(&merged, "/y"), "/y/live.md");
    }

    #[test]
    fn merge_takes_window_fields_from_the_saver() {
        let disk = Session {
            window: (100, 100),
            ..session_on("/x", vec![vault_with("/x", "/x/a.md")])
        };
        let incoming = Session {
            window: (1600, 900),
            focus: Focus::Tree,
            ..session_on("/x", vec![vault_with("/x", "/x/a.md")])
        };
        let merged = merged_session(&disk, &incoming);
        assert_eq!(merged.window, (1600, 900));
        assert_eq!(merged.focus, Focus::Tree);
    }

    #[test]
    fn merge_adds_a_vault_absent_from_disk() {
        let disk = Session::default();
        let incoming = session_on("/fresh", vec![vault_with("/fresh", "/fresh/note.md")]);
        let merged = merged_session(&disk, &incoming);
        assert_eq!(merged.active_vault, Some(PathBuf::from("/fresh")));
        assert_eq!(tab_of(&merged, "/fresh"), "/fresh/note.md");
    }

    /// A rail arrangement of `roots`, each chip wearing a color of its own.
    fn rail_of(roots: &[&str]) -> Vec<RailVault> {
        roots
            .iter()
            .enumerate()
            .map(|(i, root)| RailVault {
                root: PathBuf::from(root),
                color: i as u32,
                icon: None,
            })
            .collect()
    }

    #[test]
    fn a_hand_edited_rail_is_sanitized() {
        // The two ways a hand-edited rail could confuse the painter: a color with
        // something in the byte the format does not use, and a vault named twice,
        // which would arrange two chips for one vault.
        let session = Session {
            rail: vec![
                RailVault {
                    root: PathBuf::from("/x"),
                    color: 0xffab_cdef,
                    icon: None,
                },
                RailVault {
                    root: PathBuf::from("/x"),
                    color: 0x0011_2233,
                    icon: None,
                },
                RailVault {
                    root: PathBuf::from("/y"),
                    color: 0x0044_5566,
                    icon: None,
                },
            ],
            ..Session::default()
        };
        let clean = session.sanitized();
        assert_eq!(clean.rail.len(), 2, "the repeated vault is dropped");
        assert_eq!(
            clean.rail[0].color, 0x00ab_cdef,
            "the top byte is masked off"
        );
        assert_eq!(
            clean.rail[0].root,
            PathBuf::from("/x"),
            "the first one stands"
        );
        assert_eq!(clean.rail[1].root, PathBuf::from("/y"));
    }

    #[test]
    fn merge_takes_the_savers_whole_rail() {
        // The rail describes the saver's own window, so it crosses whole: half of one
        // arrangement and half of another is not an arrangement.
        let disk = Session {
            rail: rail_of(&["/x", "/y", "/z"]),
            ..Session::default()
        };
        let mut incoming = session_on("/x", vec![]);
        incoming.rail = rail_of(&["/z", "/y", "/x"]);
        let merged = merged_session(&disk, &incoming);
        assert_eq!(merged.rail, incoming.rail);
    }

    #[test]
    fn a_rail_survives_a_save_from_an_instance_in_another_vault() {
        // The reason the rail sits on the window and not in the per-vault entries: an
        // instance parked in /y still writes the whole rail, so dragging a chip in one
        // window is not undone by a save from another.
        let dir = tmp_dir("rail-concurrent");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join(FILENAME);

        let mut dragged = session_on("/x", vec![vault_with("/x", "/x/x1.md")]);
        dragged.rail = rail_of(&["/y", "/x"]);
        save_to(&path, &dragged).expect("seed");

        let mut elsewhere = session_on("/y", vec![vault_with("/y", "/y/y1.md")]);
        elsewhere.rail = rail_of(&["/y", "/x"]);
        save_to(&path, &elsewhere).expect("save");

        assert_eq!(load_from(&path).rail, rail_of(&["/y", "/x"]));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A scratch directory under the system temp dir, unique per test name so the
    /// parallel test runner does not collide two file-backed saves.
    fn tmp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bnknotes-store-{}-{}", std::process::id(), name))
    }

    #[test]
    fn save_to_then_load_from_round_trips() {
        // A first save writes the active vault; reloading returns it verbatim. (Only
        // the active vault, since a fresh disk holds no others: an instance can only
        // be in one vault, so that is the only one it can have.)
        let dir = tmp_dir("roundtrip");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join(FILENAME);
        let s = session_on(
            "/home/user/notes",
            vec![vault_with("/home/user/notes", "/home/user/notes/todo.md")],
        );
        save_to(&path, &s).expect("save");
        assert_eq!(load_from(&path), s);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_instance_save_keeps_the_first_instances_vault() {
        // The reported bug: two instances on different vaults, the second save must
        // not reset the first's vault to stale state. Modeled sequentially on one
        // file, since each save re-reads the latest before writing.
        let dir = tmp_dir("concurrent");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join(FILENAME);

        // Both instances start from this shared state: /x -> x1, /y -> y1.
        save_to(
            &path,
            &session_on(
                "/x",
                vec![vault_with("/x", "/x/x1.md"), vault_with("/y", "/y/y1.md")],
            ),
        )
        .expect("seed");

        // Instance A (active on /x) advances /x to x2. Its /y copy is the stale y1.
        save_to(
            &path,
            &session_on(
                "/x",
                vec![vault_with("/x", "/x/x2.md"), vault_with("/y", "/y/y1.md")],
            ),
        )
        .expect("save A");

        // Instance B (active on /y) advances /y to y2. It never saw x2, so its /x
        // copy is the stale x1: a blind overwrite here would roll /x back to x1.
        save_to(
            &path,
            &session_on(
                "/y",
                vec![vault_with("/x", "/x/x1.md"), vault_with("/y", "/y/y2.md")],
            ),
        )
        .expect("save B");

        let loaded = load_from(&path);
        assert_eq!(
            tab_of(&loaded, "/x"),
            "/x/x2.md",
            "A's vault survived B's save"
        );
        assert_eq!(tab_of(&loaded, "/y"), "/y/y2.md", "B's vault was written");
        std::fs::remove_dir_all(&dir).ok();
    }
}
