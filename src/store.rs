//! On-disk session state: a small binary blob at `~/.config/bnknotes/state.bin`
//! remembering the pane layout across restarts, so reopening bnknotes restores
//! the file tree's root and expanded folders, the column widths, and each
//! content pane's open file with its cursor and scroll.
//!
//! The format is a 4-byte magic (`BNKN`), a version byte, little-endian
//! fixed-width fields, and length-prefixed UTF-8 strings, written atomically
//! through a temp file + rename.
//!
//! Format (little-endian, no padding):
//! ```text
//!   magic          4 bytes "BNKN"
//!   version        u8 (1)
//!   window width   u32
//!   window height  u32
//!   tree_width     u32   (sidebar column width, px)
//!   split_ratio    f32   (fraction of the content area given to pane A)
//!   tree_visible   u8
//!   second_open    u8    (is content pane B open)
//!   focus          u8    (0 = tree, 1 = pane A, 2 = pane B)
//!   tree_root      opt string (absolute path, u8 tag + u16 len + utf-8)
//!   expanded count u32; per: u16 len + utf-8 path
//!   per pane (2):  opt string path, u64 cursor byte offset, i32 scroll_y
//! ```
//! A corrupt or wrong-version file is not fatal: losing the last layout must
//! never stop the editor opening, so [`load`] logs and falls back to
//! [`Session::default`].

use std::path::{Path, PathBuf};

use crate::platform::error::{Error, Result};

const FILENAME: &str = "state.bin";
const MAGIC: &[u8; 4] = b"BNKN";
const VERSION: u8 = 1;

/// A content pane's remembered file and where the caret and viewport sat in it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PaneSession {
    pub path: Option<PathBuf>,
    pub cursor: u64,
    pub scroll_y: i32,
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
    pub tree_root: Option<PathBuf>,
    pub expanded: Vec<PathBuf>,
    pub panes: [PaneSession; 2],
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
            tree_root: None,
            expanded: Vec::new(),
            panes: [PaneSession::default(), PaneSession::default()],
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
        self
    }
}

/// Path to the state file, `None` when neither `XDG_CONFIG_HOME` nor `HOME`
/// resolves (then state simply is not persisted).
pub fn session_path() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(xdg).join("bnknotes").join(FILENAME));
    }
    let home = std::env::var_os("HOME").filter(|s| !s.is_empty())?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("bnknotes")
            .join(FILENAME),
    )
}

/// Load the saved layout, or the first-run default when there is none or it is
/// unreadable. Never fails: a missing, corrupt, or wrong-version file logs (once)
/// and falls back, because a bad layout file must not stop the editor opening.
pub fn load() -> Session {
    let Some(path) = session_path() else {
        return Session::default();
    };
    match std::fs::read(&path) {
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

/// Save the layout atomically (temp file + rename via [`crate::file::save_atomic`]),
/// creating the config directory if needed. Best-effort at the call site: a
/// failure to persist the layout is reported, never fatal.
pub fn save(session: &Session) -> Result<()> {
    let path =
        session_path().ok_or_else(|| Error::msg("no XDG_CONFIG_HOME or HOME to save state"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::file::save_atomic(&path, &encode(session)?)
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
    write_opt_path(&mut out, s.tree_root.as_deref())?;

    let expanded = u32::try_from(s.expanded.len())
        .map_err(|_| Error::msg("too many expanded folders to persist"))?;
    out.extend_from_slice(&expanded.to_le_bytes());
    for dir in &s.expanded {
        write_path(&mut out, dir)?;
    }
    for pane in &s.panes {
        write_opt_path(&mut out, pane.path.as_deref())?;
        out.extend_from_slice(&pane.cursor.to_le_bytes());
        out.extend_from_slice(&pane.scroll_y.to_le_bytes());
    }
    Ok(out)
}

fn decode(data: &[u8]) -> Result<Session> {
    let mut r = Reader::new(data);
    if r.take(4)? != MAGIC {
        return Err(Error::msg("not a bnknotes state file (bad magic)"));
    }
    let version = r.u8()?;
    if version != VERSION {
        return Err(Error::msg(format!(
            "unsupported state version {version} (expected {VERSION})"
        )));
    }
    let window = (r.u32()?, r.u32()?);
    let tree_width = r.u32()?;
    let split_ratio = r.f32()?;
    let tree_visible = r.u8()? != 0;
    let second_open = r.u8()? != 0;
    let focus = Focus::from_tag(r.u8()?)?;
    let tree_root = r.opt_path()?;

    let expanded_count = r.u32()? as usize;
    let mut expanded = Vec::with_capacity(expanded_count.min(4096));
    for _ in 0..expanded_count {
        expanded.push(r.path()?);
    }
    let panes = [read_pane(&mut r)?, read_pane(&mut r)?];
    r.expect_end()?;
    Ok(Session {
        window,
        tree_width,
        split_ratio,
        tree_visible,
        second_open,
        focus,
        tree_root,
        expanded,
        panes,
    })
}

fn read_pane(r: &mut Reader) -> Result<PaneSession> {
    Ok(PaneSession {
        path: r.opt_path()?,
        cursor: r.u64()?,
        scroll_y: r.i32()?,
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
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            cursor: crate::platform::bytes::Cursor::new(data),
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
            tree_root: Some(PathBuf::from("/home/user/notes")),
            expanded: vec![
                PathBuf::from("/home/user/notes"),
                PathBuf::from("/home/user/notes/journal"),
            ],
            panes: [
                PaneSession {
                    path: Some(PathBuf::from("/home/user/notes/todo.md")),
                    cursor: 42,
                    scroll_y: 128,
                },
                PaneSession {
                    path: None,
                    cursor: 0,
                    scroll_y: -5,
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
        let s = Session {
            tree_root: Some(PathBuf::from("/home/user/notità/日本語")),
            panes: [
                PaneSession {
                    path: Some(PathBuf::from("/home/user/café.md")),
                    ..Default::default()
                },
                PaneSession::default(),
            ],
            ..Session::default()
        };
        let loaded = decode(&encode(&s).expect("encode")).expect("decode");
        assert_eq!(loaded.tree_root, s.tree_root);
        assert_eq!(loaded.panes[0].path, s.panes[0].path);
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
}
