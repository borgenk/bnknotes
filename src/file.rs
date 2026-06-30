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
//!
//! `decode`/`encode` are pure functions of bytes, so they are exhaustively unit
//! tested without touching the filesystem; only `save_atomic` does real I/O.

use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};

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
/// carriage return regardless of the source file's (possibly mixed) endings.
fn normalize_newlines(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
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

    /// A unique scratch directory under the system temp dir, distinct per test
    /// name so parallel tests in one process do not collide.
    fn tmp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bnknotes-{}-{}", std::process::id(), name))
    }
}
