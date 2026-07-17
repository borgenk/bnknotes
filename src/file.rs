//! Reading and writing the buffer to disk.
//!
//! The editor's buffer is always plain UTF-8 with `'\n'` line breaks (see
//! [`crate::text`]); a file on disk may not be. This module is the boundary that
//! reconciles the two:
//!
//! - [`decode`] turns the raw bytes of a file into the buffer's text. It strips
//!   a leading UTF-8 BOM, rejects non-UTF-8 input rather than corrupting it,
//!   normalizes every `\r\n` and lone `\r` to `\n`, and records the file's
//!   original conventions in a [`FileFormat`] so a round-trip preserves them.
//! - [`encode`] is its inverse: it re-applies the recorded line ending and BOM.
//! - [`save_atomic`] writes bytes durably, by writing a sibling temp file,
//!   flushing it, and renaming it over the target. A crash mid-write leaves
//!   either the old file or the complete new one, never a half-written file.
//! - [`FileLock`] serializes a read-modify-write against another instance, for
//!   the files two bnknotes share (the state file, the config's vault list).
//!
//! `decode`/`encode` are pure functions of bytes, so they are exhaustively unit
//! tested without touching the filesystem; only `save_atomic` does real I/O.

use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use crate::platform::error::{Error, Result};

/// The UTF-8 byte-order mark. Some editors prepend it; we strip it on load and,
/// if it was present, restore it on save.
const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// The newline convention a file uses on disk. The buffer always holds `'\n'`
/// internally; this records what to emit on save so a round-trip keeps the
/// file's original style instead of silently rewriting every line.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LineEnding {
    /// Unix `'\n'`, the default for a new file.
    #[default]
    Lf,
    /// Windows `"\r\n"`.
    Crlf,
    /// Classic Mac lone `'\r'`.
    Cr,
}

impl LineEnding {
    /// The sequence a single line break maps to on disk.
    fn as_str(self) -> &'static str {
        match self {
            LineEnding::Lf => "\n",
            LineEnding::Crlf => "\r\n",
            LineEnding::Cr => "\r",
        }
    }
}

/// The on-disk conventions of a loaded file, carried alongside the text so a
/// later save reproduces them. A brand-new file uses the defaults (LF, no BOM).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct FileFormat {
    pub line_ending: LineEnding,
    /// Whether the file began with a UTF-8 BOM.
    pub bom: bool,
}

/// A file's on-disk identity at a moment in time, used to notice a change made
/// outside the editor. Two snapshots that compare unequal mean the file was
/// written, truncated, moved, or replaced between them. It holds the raw stat
/// fields editors rely on for this: the inode identity (device plus inode, which
/// an atomic-rename save changes) alongside the size and modification time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DiskSnapshot {
    dev: u64,
    ino: u64,
    size: u64,
    mtime: i64,
    mtime_nsec: i64,
}

impl DiskSnapshot {
    /// The file's current on-disk state, or `None` when it does not exist: a
    /// buffer whose file has not been saved yet, or one deleted under us. A
    /// `None` to `Some` transition (or the reverse) is itself a change.
    pub fn of(path: &Path) -> Option<Self> {
        let m = std::fs::metadata(path).ok()?;
        Some(Self {
            dev: m.dev(),
            ino: m.ino(),
            size: m.size(),
            mtime: m.mtime(),
            mtime_nsec: m.mtime_nsec(),
        })
    }
}

/// Decode a file's raw bytes into buffer text plus the [`FileFormat`] needed to
/// write it back. Strips a leading BOM, fails on non-UTF-8 input rather than
/// corrupting it, and normalizes all line endings to `'\n'`.
pub fn decode(bytes: &[u8]) -> Result<(String, FileFormat)> {
    let (bytes, bom) = match bytes.strip_prefix(&BOM[..]) {
        Some(rest) => (rest, true),
        None => (bytes, false),
    };
    let raw = std::str::from_utf8(bytes)
        .map_err(|e| Error::msg(format!("file is not valid UTF-8: {e}")))?;
    let line_ending = detect_line_ending(raw);
    let text = normalize_newlines(raw);
    Ok((text, FileFormat { line_ending, bom }))
}

/// Encode buffer text back to file bytes under `format`: re-apply the line
/// ending and, if the original had one, the leading BOM.
pub fn encode(text: &str, format: &FileFormat) -> Vec<u8> {
    let body = match format.line_ending {
        // The buffer already uses `'\n'`; only the others need rewriting.
        LineEnding::Lf => text.to_string(),
        ending => text.replace('\n', ending.as_str()),
    };
    let mut out = Vec::with_capacity(body.len() + if format.bom { BOM.len() } else { 0 });
    if format.bom {
        out.extend_from_slice(&BOM);
    }
    out.extend_from_slice(body.as_bytes());
    out
}

/// The line-ending convention of `s`, taken from the first line break it
/// contains. An empty or single-line file reports the default ([`LineEnding::Lf`]).
fn detect_line_ending(s: &str) -> LineEnding {
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\n' => return LineEnding::Lf,
            b'\r' => {
                return if bytes.get(i + 1) == Some(&b'\n') {
                    LineEnding::Crlf
                } else {
                    LineEnding::Cr
                };
            }
            _ => {}
        }
    }
    LineEnding::Lf
}

/// Rewrite every `"\r\n"` and lone `'\r'` as `'\n'`, so the buffer never holds a
/// carriage return regardless of the source's (possibly mixed) endings. Applied
/// both on file load and on paste, the two ways foreign text enters the buffer.
pub fn normalize_newlines(s: &str) -> String {
    // No carriage return means the text is already LF-normalized: one copy, not
    // the two full allocating passes `replace("\r\n", ..).replace('\r', ..)` pays
    // even when there is nothing to replace.
    let Some(first) = crate::platform::bytes::find_byte(s.as_bytes(), b'\r') else {
        return s.to_string();
    };
    // Rewrite in a single pass from the first '\r': a CRLF pair and a lone CR each
    // collapse to one '\n'. '\r'/'\n' are ASCII, so slicing on their byte offsets
    // always lands on a char boundary; the runs between them copy verbatim.
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    out.push_str(&s[..first]);
    let mut run = first;
    let mut i = first;
    while i < bytes.len() {
        if bytes[i] == b'\r' {
            out.push_str(&s[run..i]);
            out.push('\n');
            // A '\n' right after the '\r' is part of the same CRLF line break.
            if bytes.get(i + 1) == Some(&b'\n') {
                i += 1;
            }
            run = i + 1;
        }
        i += 1;
    }
    out.push_str(&s[run..]);
    out
}

/// Write `bytes` to `path` atomically: stage them in a sibling temp file in the
/// same directory, flush to disk, then rename over the target. The rename is
/// atomic on a single filesystem, so a reader or a crash never sees a
/// partially-written file. The temp file is removed on any failure.
pub fn save_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = path
        .file_name()
        .ok_or_else(|| Error::msg("cannot save: path has no file name"))?;
    // A dotfile sibling (".name.tmp.<pid>") in the target's directory, so the
    // rename stays on the same filesystem and the temp name will not collide
    // with the target or with another process's save.
    let mut tmp_name = OsString::from(".");
    tmp_name.push(file_name);
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = path.with_file_name(tmp_name);

    match write_then_rename(&tmp, path, bytes) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Best effort: don't leave the staging file behind on failure.
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// An exclusive advisory lock over `path`, held across one read-modify-write so
/// two instances rewriting the same file serialize rather than clobber each
/// other. It guards a sibling lock file rather than the file being written, whose
/// inode [`save_atomic`] replaces by rename; the kernel releases the lock when the
/// file closes, so a crashed holder never wedges the others. Best-effort: if the
/// lock cannot be taken (no parent directory, or a filesystem without flock), the
/// caller proceeds unlocked rather than not at all.
pub struct FileLock {
    _file: File,
}

impl FileLock {
    /// Take the lock at `path`, creating the lock file if it is not there.
    pub fn acquire(path: &Path) -> Option<Self> {
        let file = File::create(path).ok()?;
        file.lock().ok()?;
        Some(Self { _file: file })
    }
}

fn write_then_rename(tmp: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(tmp)?;
    file.write_all(bytes)?;
    // Carry the target's permission bits onto the temp before it replaces it.
    // `File::create` makes the temp 0644 (& umask), so without this a private
    // 0600 note would silently become world-readable after the rename. A
    // brand-new file (no existing target) keeps the default permissions.
    if let Ok(meta) = std::fs::metadata(path) {
        file.set_permissions(meta.permissions())?;
    }
    // Flush the contents to disk before the rename publishes them, so the
    // renamed file is never an empty or truncated placeholder.
    file.sync_all()?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    #[test]
    fn detects_each_line_ending() {
        assert_eq!(decode(b"a\nb").unwrap().1.line_ending, LineEnding::Lf);
        assert_eq!(decode(b"a\r\nb").unwrap().1.line_ending, LineEnding::Crlf);
        assert_eq!(decode(b"a\rb").unwrap().1.line_ending, LineEnding::Cr);
        // No line break at all falls back to the default.
        assert_eq!(
            decode(b"single line").unwrap().1.line_ending,
            LineEnding::Lf
        );
    }

    #[test]
    fn first_line_ending_wins_in_a_mixed_file() {
        // The convention is taken from the first break, but every break is
        // still normalized to '\n' in the text.
        let (text, format) = decode(b"a\r\nb\nc\rd").unwrap();
        assert_eq!(text, "a\nb\nc\nd");
        assert_eq!(format.line_ending, LineEnding::Crlf);
    }

    #[test]
    fn strips_and_remembers_a_bom() {
        let mut bytes = BOM.to_vec();
        bytes.extend_from_slice(b"hello");
        let (text, format) = decode(&bytes).unwrap();
        assert_eq!(text, "hello");
        assert!(format.bom);
        // No BOM means none is recorded.
        assert!(!decode(b"hello").unwrap().1.bom);
    }

    #[test]
    fn rejects_invalid_utf8_instead_of_corrupting() {
        // A lone 0xFF is not valid UTF-8; decoding must fail, not lossily munge.
        assert!(decode(&[0xFF, 0xFE, b'h', b'i']).is_err());
    }

    #[test]
    fn round_trips_every_convention() {
        let cases: [&[u8]; 6] = [
            b"plain lf\nsecond line\n",
            b"windows\r\nsecond line\r\n",
            b"classic mac\rsecond line\r",
            b"no trailing newline",
            b"",
            &[0xEF, 0xBB, 0xBF, b'b', b'o', b'm', b'\n'],
        ];
        for original in cases {
            let (text, format) = decode(original).unwrap();
            assert_eq!(
                encode(&text, &format),
                original,
                "round-trip changed {original:?}"
            );
        }
    }

    #[test]
    fn preserves_the_final_newline_either_way() {
        // Whether the file ends with a newline is part of its content, so it
        // survives a round-trip untouched (neither stripped nor added).
        assert_eq!(
            encode_roundtrip(b"ends with newline\n"),
            b"ends with newline\n"
        );
        assert_eq!(encode_roundtrip(b"no newline"), b"no newline");
    }

    fn encode_roundtrip(bytes: &[u8]) -> Vec<u8> {
        let (text, format) = decode(bytes).unwrap();
        encode(&text, &format)
    }

    #[test]
    fn save_atomic_writes_then_replaces_without_leftovers() {
        let dir = tmp_dir("save_replace");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.md");

        save_atomic(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");

        // Overwriting an existing file works and leaves no staging file behind.
        save_atomic(&path, b"second contents").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second contents");

        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|name| name != "note.md")
            .collect();
        assert!(
            leftovers.is_empty(),
            "temp files left behind: {leftovers:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn save_atomic_preserves_an_existing_files_permissions() {
        let dir = tmp_dir("save_perms");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.md");

        // A private 0600 note must stay 0600 after a save, not revert to the
        // temp file's default 0644.
        save_atomic(&path, b"secret").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        save_atomic(&path, b"still secret").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "permissions survive an atomic save");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disk_snapshot_tracks_existence_and_change() {
        let dir = tmp_dir("snapshot");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.md");

        // A file that does not exist yet snapshots to None.
        assert_eq!(DiskSnapshot::of(&path), None);

        save_atomic(&path, b"first").unwrap();
        let a = DiskSnapshot::of(&path).expect("exists after save");
        // A re-stat with nothing touched is equal to itself.
        assert_eq!(DiskSnapshot::of(&path), Some(a));

        // A different-length write changes the size, so the snapshot differs.
        save_atomic(&path, b"second, longer").unwrap();
        let b = DiskSnapshot::of(&path).expect("still exists");
        assert_ne!(a, b, "a rewrite must change the snapshot");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn disk_snapshot_notices_an_atomic_replace_with_identical_content() {
        // The subtle case: another editor replaces the file with byte-identical
        // content via rename. Size and (often) mtime match, but the inode does
        // not, so the snapshot still reports a change.
        let dir = tmp_dir("snapshot_replace");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.md");

        save_atomic(&path, b"same").unwrap();
        let before = DiskSnapshot::of(&path).unwrap();

        // A fresh inode carrying the same bytes, renamed over the target.
        save_atomic(&path, b"same").unwrap();
        let after = DiskSnapshot::of(&path).unwrap();
        assert_ne!(before, after, "an inode swap must register as a change");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A unique scratch directory under the system temp dir, distinct per test
    /// name so parallel tests in one process do not collide.
    fn tmp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bnknotes-{}-{}", std::process::id(), name))
    }
}
