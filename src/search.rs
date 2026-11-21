//! Search across a vault: the pure scan behind the palette's Ctrl+Shift+F.
//!
//! Walks a vault's notes, scans each for a literal query, and answers with one hit
//! per matching line. The palette turns the hits into rows; nothing here draws or
//! knows what a row is.
//!
//! The scan is synchronous, on the frame thread, because it is cheap enough to be:
//! a 176-note vault walks, reads, and scans in about 10ms, and one twenty times that
//! size in about 30ms. What keeps it there as a query grows is [`Narrow`]: a longer
//! query can only match inside the files its prefix already matched, so the second
//! keystroke onward rescans a handful of files rather than the vault.
//!
//! Matching folds ASCII letters and is byte-exact otherwise, the same asymmetry (and
//! the same reason) as [`crate::find`]: a full Unicode fold changes byte lengths, and
//! a hit's offset has to keep pointing at the text it matched.
//!
//! The walk skips what the sidebar hides (dot-entries, so the vault trash and
//! `.git` never turn up), and the results put heading matches first, on
//! the reasoning that a query which is a note's heading usually names the note the
//! user is reaching for.

use std::path::{Path, PathBuf};

/// Extension a vault's notes carry. Everything else in a vault (a picture, an
/// attachment) is not text to search.
const NOTE_EXT: &str = "md";

/// Cap on collected hits. A short query against a big vault would otherwise build a
/// row per matching line of a common word, which is neither navigable nor worth the
/// memory; collection stops here and the palette says the count is a floor.
pub const MAX_HITS: usize = 2000;

/// Shortest query that scans. A single character matches most of a vault, so the
/// rows would be noise and the scan pure waste; the palette shows nothing until the
/// query says something.
pub const MIN_QUERY: usize = 2;

/// One matching line: the note it is in, where its first match starts, and the line
/// itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub path: PathBuf,
    /// Byte offset of the line's first match, for the caret to land on.
    pub offset: usize,
    /// The line, counting from 1, as an editor numbers lines.
    pub line: usize,
    /// The heading level of the line the match is on (1 for `#`, 2 for `##`, up to
    /// 6), or `None` when it is not a heading. A heading match weighs more, since a
    /// query that is a note's heading is likely the note wanted; see [`search`].
    pub heading: Option<u8>,
    /// The line's text, whitespace-trimmed, for the row to show.
    pub text: String,
}

/// What a scan answered: the hits, whether the cap cut them off, and the notes they
/// came from (which is what narrows the next, longer query).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Results {
    pub hits: Vec<Hit>,
    /// The cap stopped collection, so there are matching lines the rows do not show.
    pub truncated: bool,
    /// The notes that held a hit, in walk order.
    pub files: Vec<PathBuf>,
    /// The query these answer, so the next one knows whether it can narrow.
    pub query: String,
}

/// The files a query may restrict itself to: the notes a shorter query already
/// matched. A query that grew can only match inside them, so the walk is skipped.
pub enum Narrow<'a> {
    /// Walk the vault.
    Vault,
    /// Scan only these notes.
    Files(&'a [PathBuf]),
}

/// The notes open in the editor, whose buffers may differ from their files. A path
/// listed here is read from the text given rather than from disk, so a hit's line
/// number means what is on screen.
pub struct OpenNotes<'a> {
    pub notes: &'a [(PathBuf, String)],
}

impl OpenNotes<'_> {
    /// The buffer text for `path`, if that note is open.
    fn text_of(&self, path: &Path) -> Option<&str> {
        self.notes
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, t)| t.as_str())
    }
}

/// Search `root` for `query`, from disk except where [`OpenNotes`] has a buffer.
///
/// An empty or too-short query answers nothing rather than everything: there is no
/// useful "all matches" for one letter. `narrow` restricts the scan when the caller
/// knows a shorter query's hits (see [`Narrow`]); it is the caller's job to only
/// narrow on a query that grew, which [`Results::can_narrow`] answers.
pub fn search(root: &Path, query: &str, open: &OpenNotes, narrow: Narrow) -> Results {
    let mut out = Results {
        query: query.to_string(),
        ..Results::default()
    };
    if query.chars().count() < MIN_QUERY {
        return out;
    }
    let mut files = Vec::new();
    let scanned: &[PathBuf] = match narrow {
        Narrow::Files(f) => f,
        Narrow::Vault => {
            walk(root, &mut files);
            &files
        }
    };
    for path in scanned {
        let matched = match open.text_of(path) {
            Some(text) => scan(path, text, query, &mut out),
            None => {
                // A note that cannot be read (a permission, a race with a delete) is
                // skipped: a search must not fail because one file moved under it.
                let Ok(text) = std::fs::read_to_string(path) else {
                    continue;
                };
                scan(path, &text, query, &mut out)
            }
        };
        if matched {
            out.files.push(path.clone());
        }
    }
    // Heading lines weigh more: a query that is a note's heading is likely the note
    // wanted, so a heading match ranks above a body match, and a shallower heading
    // above a deeper one (# above ##). The sort is stable, so walk order (note order,
    // then line order) still decides between lines of the same rank. The narrowing
    // file list is built above in walk order and untouched by this.
    out.hits.sort_by_key(|hit| hit.heading.unwrap_or(u8::MAX));
    out
}

impl Results {
    /// Whether `query` can be answered by rescanning [`Self::files`] alone: it grew
    /// from the query these hits answer, so anything matching it lies inside a note
    /// matching that. A truncated scan narrows too, since it names every note it
    /// matched even where it stopped listing their lines.
    pub fn can_narrow(&self, query: &str) -> bool {
        self.query.len() >= MIN_QUERY
            && query.len() > self.query.len()
            && query.starts_with(&self.query)
    }
}

/// Every note under `root`, depth first. A directory that cannot be read is skipped
/// rather than failing the walk.
fn walk(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        // What the sidebar hides, search skips: a dot-entry (.git, and the vault trash,
        // .trash / .Trash), so a search never turns up a note the tree will not show. A
        // name that is not UTF-8 is skipped too, as the tree does.
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_none_or(crate::tree::is_hidden)
        {
            continue;
        }
        if kind.is_dir() {
            dirs.push(path);
        } else if path.extension().is_some_and(|e| e == NOTE_EXT) {
            out.push(path);
        }
    }
    // Sorted so a vault's hits come back in the same order every time, whatever the
    // filesystem hands back.
    dirs.sort();
    out.sort();
    for dir in dirs {
        walk(&dir, out);
    }
}

/// Append `text`'s matching lines for `query` to `out`, and report whether it held
/// any.
///
/// One hit per line, not per occurrence: a line holding the query twice ("ckeditor"
/// and "rich-text-editor" both matching "editor") would otherwise be two rows reading
/// exactly alike, since a row shows the line. The hit points at the line's first
/// match, and the scan resumes on the next line.
///
/// Past the cap it stops recording hits but keeps answering whether a note matched,
/// because the notes a query matched are what narrow the next, longer one. A scan that
/// gave up at the cap would name only the notes it reached, and narrowing to those
/// would silently lose the rest. Knowing a note matched costs only the search for its
/// first match, so a capped scan gets cheaper per note rather than more expensive.
fn scan(path: &Path, text: &str, query: &str, out: &mut Results) -> bool {
    let (hay, needle) = (text.as_bytes(), query.as_bytes());
    if needle.is_empty() || hay.len() < needle.len() {
        return false;
    }
    let (mut at, mut matched) = (0, false);
    while at + needle.len() <= hay.len() {
        if !hay[at..at + needle.len()].eq_ignore_ascii_case(needle) {
            at += 1;
            continue;
        }
        matched = true;
        if out.hits.len() >= MAX_HITS {
            out.truncated = true;
            return matched;
        }
        let (line, start, end) = line_at(hay, at);
        // Classify the raw line the way the editor does, so a match on `# Title` reads
        // as a heading here exactly when it renders as one there (an indented `#` is
        // not a heading to either).
        let raw = String::from_utf8_lossy(&hay[start..end]);
        let heading = match crate::markdown::classify(&raw).0 {
            crate::markdown::LineStyle::Heading(level) => Some(level),
            _ => None,
        };
        out.hits.push(Hit {
            path: path.to_path_buf(),
            offset: at,
            line,
            heading,
            text: raw.trim().to_string(),
        });
        // On to the next line: this one has its row.
        at = end + 1;
    }
    matched
}

/// The 1-based line holding `at`, and that line's byte range without its newline.
fn line_at(hay: &[u8], at: usize) -> (usize, usize, usize) {
    let line = 1 + hay[..at].iter().filter(|&&b| b == b'\n').count();
    let start = hay[..at]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    let end = hay[at..]
        .iter()
        .position(|&b| b == b'\n')
        .map_or(hay.len(), |i| at + i);
    (line, start, end)
}

/// A hit's note relative to `root`, for a row that names the note rather than the
/// machine it is on. Falls back to the whole path when it is somehow outside.
pub fn relative<'a>(path: &'a Path, root: &Path) -> &'a Path {
    path.strip_prefix(root).unwrap_or(path)
}

/// Write `text` to `path`, creating parents. Used by the tests to build a vault, and
/// by nothing else.
#[cfg(test)]
fn write(path: &Path, text: &str) -> crate::platform::error::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| crate::platform::error::Error::msg(format!("mkdir: {e}")))?;
    }
    std::fs::write(path, text)
        .map_err(|e| crate::platform::error::Error::msg(format!("write: {e}")))
}

#[cfg(test)]
mod tests {
    use crate::search::*;

    /// A vault of `notes` (relative path, contents) in a directory of this test's
    /// own, removed and rebuilt each run so a rerun cannot see the last one's files.
    fn vault(name: &str, notes: &[(&str, &str)]) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-search-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (path, text) in notes {
            write(&dir.join(path), text).expect("write a note");
        }
        dir
    }

    /// No note is open, the common case.
    fn none() -> OpenNotes<'static> {
        OpenNotes { notes: &[] }
    }

    /// The hits as "relative/path:line" plus the line's text, which is what a row says.
    fn rows(r: &Results, root: &Path) -> Vec<String> {
        r.hits
            .iter()
            .map(|h| {
                format!(
                    "{}:{} {}",
                    relative(&h.path, root).display(),
                    h.line,
                    h.text
                )
            })
            .collect()
    }

    #[test]
    fn a_query_finds_its_matches_with_the_line_they_are_on() {
        let root = vault(
            "basic",
            &[("a.md", "alpha beta\ngamma delta\nbeta again\n")],
        );
        let r = search(&root, "beta", &none(), Narrow::Vault);
        assert_eq!(rows(&r, &root), ["a.md:1 alpha beta", "a.md:3 beta again"]);
        assert!(!r.truncated);
    }

    #[test]
    fn a_hits_offset_points_at_the_text_it_matched() {
        // The offset is what the caret lands on, so it has to index the match itself.
        let text = "alpha beta\ngamma\n";
        let root = vault("offset", &[("a.md", text)]);
        let r = search(&root, "gamma", &none(), Narrow::Vault);
        assert_eq!(r.hits.len(), 1);
        let at = r.hits[0].offset;
        assert_eq!(&text[at..at + 5], "gamma");
    }

    #[test]
    fn an_offset_is_a_byte_index_past_a_multibyte_character() {
        // A naive char count would place the caret short of the match.
        let text = "æøå hello\nworld\n";
        let root = vault("utf8", &[("a.md", text)]);
        let r = search(&root, "hello", &none(), Narrow::Vault);
        let at = r.hits[0].offset;
        assert_eq!(&text[at..at + 5], "hello");
    }

    #[test]
    fn the_search_is_case_insensitive_either_way() {
        // A query finds a match whatever the case on either side: a lower-case query
        // hits an upper-case note and the reverse, so the search never depends on
        // matching the case the note happens to use.
        let root = vault("case", &[("a.md", "Alpha BETA\n")]);
        assert_eq!(search(&root, "beta", &none(), Narrow::Vault).hits.len(), 1);
        assert_eq!(search(&root, "BETA", &none(), Narrow::Vault).hits.len(), 1);
        assert_eq!(search(&root, "alpha", &none(), Narrow::Vault).hits.len(), 1);
        assert_eq!(search(&root, "ALPHA", &none(), Narrow::Vault).hits.len(), 1);
    }

    #[test]
    fn a_dot_directory_is_never_searched() {
        // The trash and the config folder hold notes the sidebar hides; search must
        // hide them too, or a deleted note keeps turning up. Same rule as the tree.
        let root = vault(
            "hidden",
            &[
                ("keep.md", "target\n"),
                (".trash/gone.md", "target\n"),
                (".Trash/gone.md", "target\n"),
                (".config/plugin.md", "target\n"),
                ("sub/.trash/deep.md", "target\n"),
            ],
        );
        let r = search(&root, "target", &none(), Narrow::Vault);
        let files: Vec<String> = r
            .files
            .iter()
            .map(|f| relative(f, &root).display().to_string())
            .collect();
        assert_eq!(files, ["keep.md"]);
    }

    #[test]
    fn a_hidden_note_is_never_searched() {
        // A dot-named note is hidden the same way a dot-directory is.
        let root = vault(
            "hiddennote",
            &[("keep.md", "target\n"), (".secret.md", "target\n")],
        );
        let r = search(&root, "target", &none(), Narrow::Vault);
        assert_eq!(r.files.len(), 1);
        assert_eq!(relative(&r.files[0], &root), Path::new("keep.md"));
    }

    #[test]
    fn a_heading_match_ranks_above_a_body_match() {
        // A note whose heading is the query is likely the one wanted, so it comes
        // first even though its file sorts after the body match in the walk.
        let root = vault(
            "heading",
            &[
                ("a-body.md", "a line mentioning gamma in passing\n"),
                ("z-heading.md", "# Gamma\n\nsome body\n"),
            ],
        );
        let r = search(&root, "gamma", &none(), Narrow::Vault);
        assert_eq!(
            rows(&r, &root),
            [
                "z-heading.md:1 # Gamma",
                "a-body.md:1 a line mentioning gamma in passing"
            ]
        );
    }

    #[test]
    fn a_shallower_heading_ranks_above_a_deeper_one() {
        let root = vault(
            "depth",
            &[
                ("a.md", "## Sub gamma\n"),
                ("b.md", "# Top gamma\n"),
                ("c.md", "### Deep gamma\n"),
            ],
        );
        let r = search(&root, "gamma", &none(), Narrow::Vault);
        assert_eq!(
            rows(&r, &root),
            [
                "b.md:1 # Top gamma",
                "a.md:1 ## Sub gamma",
                "c.md:1 ### Deep gamma"
            ]
        );
    }

    #[test]
    fn walk_order_still_decides_between_lines_of_the_same_rank() {
        // Two body matches keep their walk order; the heading sort is stable, so it
        // only moves headings up and never scrambles the rest.
        let root = vault(
            "stable",
            &[
                ("a.md", "gamma one\ngamma two\n"),
                ("b.md", "gamma three\n"),
            ],
        );
        let r = search(&root, "gamma", &none(), Narrow::Vault);
        assert_eq!(
            rows(&r, &root),
            ["a.md:1 gamma one", "a.md:2 gamma two", "b.md:1 gamma three"]
        );
    }

    #[test]
    fn an_indented_hash_is_not_a_heading_match() {
        // The editor does not render an indented `#` as a heading, so search must not
        // weight it as one: it stays a body match, below the real heading.
        let root = vault(
            "indent",
            &[
                ("a.md", "  # not a heading, has gamma\n"),
                ("b.md", "# gamma heading\n"),
            ],
        );
        let r = search(&root, "gamma", &none(), Narrow::Vault);
        assert_eq!(
            rows(&r, &root),
            [
                "b.md:1 # gamma heading",
                "a.md:1 # not a heading, has gamma"
            ]
        );
    }

    #[test]
    fn a_non_ascii_letter_matches_only_itself() {
        // The fold is ASCII-only, so an accented letter is not its plain twin. Nothing
        // here may split a character or mis-place an offset.
        let root = vault("fold", &[("a.md", "Ærlig\n")]);
        assert_eq!(search(&root, "Ærlig", &none(), Narrow::Vault).hits.len(), 1);
        assert!(
            search(&root, "aerlig", &none(), Narrow::Vault)
                .hits
                .is_empty()
        );
    }

    #[test]
    fn a_line_matched_twice_is_one_row_pointing_at_its_first_match() {
        // A row shows the line, so a second row for the same line would read exactly
        // like the first. The offset takes the caret to the first match on it.
        let root = vault(
            "perline",
            &[("a.md", "ckeditor and rich-text-editor\nplain\n")],
        );
        let r = search(&root, "editor", &none(), Narrow::Vault);
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].line, 1);
        assert_eq!(r.hits[0].offset, 2, "the caret goes to the first match");
    }

    #[test]
    fn every_matching_line_of_a_note_gets_its_own_row() {
        let root = vault("lines", &[("a.md", "target one\nno\ntarget two\n")]);
        let r = search(&root, "target", &none(), Narrow::Vault);
        assert_eq!(rows(&r, &root), ["a.md:1 target one", "a.md:3 target two"]);
    }

    #[test]
    fn an_overlapping_match_does_not_double_count_a_line() {
        // "aa" in "aaaa" is one line, so one row, whatever the scan's stride.
        let root = vault("overlap", &[("a.md", "aaaa\n")]);
        let r = search(&root, "aa", &none(), Narrow::Vault);
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].offset, 0);
    }

    #[test]
    fn a_query_under_the_minimum_answers_nothing() {
        // One letter matches most of a vault: rows nobody can navigate, for a scan
        // nobody wanted.
        let root = vault("min", &[("a.md", "alpha\n")]);
        assert!(search(&root, "a", &none(), Narrow::Vault).hits.is_empty());
        assert!(search(&root, "", &none(), Narrow::Vault).hits.is_empty());
        assert_eq!(search(&root, "al", &none(), Narrow::Vault).hits.len(), 1);
    }

    #[test]
    fn the_walk_reaches_nested_notes_and_skips_what_is_not_one() {
        let root = vault(
            "walk",
            &[
                ("a.md", "target\n"),
                ("sub/b.md", "target\n"),
                ("sub/deep/c.md", "target\n"),
                ("d.txt", "target\n"),
                ("notes.md.bak", "target\n"),
            ],
        );
        let r = search(&root, "target", &none(), Narrow::Vault);
        let files: Vec<String> = r
            .files
            .iter()
            .map(|f| relative(f, &root).display().to_string())
            .collect();
        assert_eq!(files, ["a.md", "sub/b.md", "sub/deep/c.md"]);
    }

    #[test]
    fn an_open_notes_buffer_is_searched_instead_of_its_file() {
        // The buffer is what is on screen, so a hit has to mean the buffer's line, not
        // the file's. Otherwise Enter lands on the wrong text in the tab you were in.
        let root = vault("open", &[("a.md", "on disk only\n")]);
        let buffers = [(
            root.join("a.md"),
            "typed but not saved\nsecond line\n".into(),
        )];
        let open = OpenNotes { notes: &buffers };

        // What is in the buffer is found; what is only in the file is not.
        let r = search(&root, "typed", &open, Narrow::Vault);
        assert_eq!(rows(&r, &root), ["a.md:1 typed but not saved"]);
        assert!(
            search(&root, "on disk", &open, Narrow::Vault)
                .hits
                .is_empty()
        );
    }

    #[test]
    fn a_note_that_is_not_open_still_comes_from_disk() {
        let root = vault("mixed", &[("a.md", "target\n"), ("b.md", "target\n")]);
        let buffers = [(root.join("a.md"), "target in buffer\n".into())];
        let r = search(
            &root,
            "target",
            &OpenNotes { notes: &buffers },
            Narrow::Vault,
        );
        assert_eq!(
            rows(&r, &root),
            ["a.md:1 target in buffer", "b.md:1 target"]
        );
    }

    #[test]
    fn the_cap_stops_collection_and_says_so() {
        let line = "x aa\n".repeat(MAX_HITS + 50);
        let root = vault("cap", &[("a.md", &line)]);
        let r = search(&root, "aa", &none(), Narrow::Vault);
        assert_eq!(r.hits.len(), MAX_HITS);
        assert!(r.truncated, "the rows are not the whole answer, so say so");
    }

    #[test]
    fn a_query_that_grew_can_narrow_to_the_files_that_already_matched() {
        let root = vault("narrow", &[("a.md", "gamma\n"), ("b.md", "gap\n")]);
        let wide = search(&root, "ga", &none(), Narrow::Vault);
        assert_eq!(wide.files.len(), 2);
        assert!(wide.can_narrow("gam"));

        // Scanning only the files "ga" matched finds what a full walk would.
        let narrow = search(&root, "gamm", &none(), Narrow::Files(&wide.files));
        let full = search(&root, "gamm", &none(), Narrow::Vault);
        assert_eq!(narrow.hits, full.hits);
        assert_eq!(rows(&narrow, &root), ["a.md:1 gamma"]);
    }

    #[test]
    fn a_query_that_did_not_grow_cannot_narrow() {
        let root = vault("nonarrow", &[("a.md", "gamma\n")]);
        let wide = search(&root, "gam", &none(), Narrow::Vault);
        // Backspaced: the shorter query can match files the longer one never did.
        assert!(!wide.can_narrow("ga"));
        // Retyped the same: nothing to narrow, and no scan to save.
        assert!(!wide.can_narrow("gam"));
        // A different query of the same length, and one that only shares a prefix
        // partway, both have to walk again.
        assert!(!wide.can_narrow("xyz"));
        assert!(!wide.can_narrow("gaXX"));
    }

    #[test]
    fn a_truncated_scan_still_names_every_file_it_matched() {
        // The cap stops the rows, not the walk. Were it to stop the walk, the file list
        // would name only what was reached before it, and narrowing to that would lose
        // the rest: exactly where narrowing matters most, since typing toward a common
        // word truncates on every keystroke.
        let root = vault(
            "trunc",
            &[("a.md", &"aa\n".repeat(MAX_HITS + 10)), ("b.md", "aab\n")],
        );
        let wide = search(&root, "aa", &none(), Narrow::Vault);
        assert!(wide.truncated, "the cap should have bitten");
        // b.md matched, though the cap was long spent by the time the walk reached it.
        let files: Vec<String> = wide
            .files
            .iter()
            .map(|f| relative(f, &root).display().to_string())
            .collect();
        assert_eq!(files, ["a.md", "b.md"]);

        // So narrowing to them finds exactly what a full walk would.
        assert!(wide.can_narrow("aab"));
        let narrowed = search(&root, "aab", &none(), Narrow::Files(&wide.files));
        let full = search(&root, "aab", &none(), Narrow::Vault);
        assert_eq!(narrowed.hits, full.hits);
        assert_eq!(rows(&narrowed, &root), ["b.md:1 aab"]);
    }

    #[test]
    fn a_rows_text_is_the_matched_line_trimmed() {
        let root = vault("trim", &[("a.md", "    indented target here   \n")]);
        let r = search(&root, "target", &none(), Narrow::Vault);
        assert_eq!(r.hits[0].text, "indented target here");
    }

    #[test]
    fn a_match_on_the_last_line_without_a_newline_still_reads() {
        let root = vault("noeol", &[("a.md", "first\nlast target")]);
        let r = search(&root, "target", &none(), Narrow::Vault);
        assert_eq!(rows(&r, &root), ["a.md:2 last target"]);
    }

    #[test]
    fn an_empty_vault_answers_nothing_rather_than_failing() {
        let root = vault("empty", &[]);
        let r = search(&root, "target", &none(), Narrow::Vault);
        assert!(r.hits.is_empty());
        assert!(!r.truncated);
    }

    #[test]
    fn a_vault_that_does_not_exist_answers_nothing_rather_than_failing() {
        let root = std::env::temp_dir().join("bnknotes-search-nonexistent-xyz");
        let _ = std::fs::remove_dir_all(&root);
        assert!(
            search(&root, "target", &none(), Narrow::Vault)
                .hits
                .is_empty()
        );
    }
}
