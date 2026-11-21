//! The YAML frontmatter block a note may open with: where it ends, so the rest of
//! the editor can treat it as metadata rather than markdown.
//!
//! ```text
//! ---                        <- line 0, the opener
//! title: Work log
//! tags: [Notebooks/Work]
//! ---                        <- the closer; the block is lines 0..4
//! ```
//!
//! Frontmatter is not markdown and has no CommonMark rule. What marks it is
//! position: only a `---` on the document's very first line opens one, which is why
//! recognizing it cannot live in [`crate::markdown`], where a line is classified
//! with no idea where in the document it sits. Without that context the opener and
//! closer read as thematic breaks and the pairs between them read as prose, so
//! emphasis and link markup inside a value would style it.
//!
//! The extent is all this layer resolves. The lines inside stay raw text: the
//! editor draws them verbatim and saves them byte for byte, so a note round-trips
//! whatever YAML dialect wrote it. Parsing the pairs into typed properties is a
//! separate job, and one nothing reads yet.

use crate::rope::Rope;

/// How far into a document the closing delimiter is looked for. A block this long
/// is not metadata, and the cap keeps the scan off the rest of the file: the common
/// document has no frontmatter at all and costs one line test.
pub const MAX_LINES: usize = 128;

/// The number of logical lines the frontmatter block covers, so lines `0..end` are
/// metadata and `end` is the document's first markdown line. `None` when the
/// document does not open with one, which includes an opener whose closer is
/// missing or further down than [`MAX_LINES`]: an unclosed `---` was a thematic
/// break all along.
pub fn extent(rope: &Rope) -> Option<usize> {
    if !delimiter_at(rope, 0, b'-') {
        return None;
    }
    let n = rope.len_lines().min(MAX_LINES);
    (1..n)
        .find(|&line| {
            let start = rope.line_to_byte(line);
            delimiter_at(rope, start, b'-') || delimiter_at(rope, start, b'.')
        })
        .map(|closer| closer + 1)
}

/// Whether the line at byte `start` is exactly three `marker` bytes followed by
/// nothing but spaces or tabs. `---` both opens and closes a block; `...` also
/// closes one, per YAML. Four dashes are a thematic break, not a delimiter.
///
/// Reads the rope byte by byte so the test never forms a text slice, and so a
/// document whose first line is a single huge word costs four byte reads.
fn delimiter_at(rope: &Rope, start: usize, marker: u8) -> bool {
    if (0..3).any(|i| rope.byte(start.saturating_add(i)) != Some(marker)) {
        return false;
    }
    let mut i = start.saturating_add(3);
    loop {
        match rope.byte(i) {
            None | Some(b'\n') => return true,
            Some(b' ' | b'\t') => i = i.saturating_add(1),
            Some(_) => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::frontmatter::*;

    fn rope(s: &str) -> Rope {
        Rope::from_str(s)
    }

    #[test]
    fn a_closed_block_at_the_top_covers_its_delimiters() {
        let r = rope("---\ntitle: Work log\ntags: [Notebooks/Work]\n---\n\n# Monday\n");
        assert_eq!(extent(&r), Some(4));
    }

    #[test]
    fn a_block_may_close_with_yaml_end_of_document() {
        assert_eq!(extent(&rope("---\ntitle: x\n...\nbody\n")), Some(3));
    }

    #[test]
    fn an_empty_block_still_counts() {
        assert_eq!(extent(&rope("---\n---\nbody\n")), Some(2));
    }

    #[test]
    fn trailing_whitespace_on_a_delimiter_is_allowed() {
        assert_eq!(extent(&rope("---  \ntitle: x\n---\t\nbody\n")), Some(3));
    }

    #[test]
    fn a_block_closing_on_the_last_line_needs_no_newline() {
        assert_eq!(extent(&rope("---\ntitle: x\n---")), Some(3));
    }

    #[test]
    fn an_unclosed_opener_is_a_thematic_break() {
        assert_eq!(extent(&rope("---\ntitle: x\nbody\n")), None);
        assert_eq!(extent(&rope("---\n")), None);
        assert_eq!(extent(&rope("---")), None);
    }

    #[test]
    fn a_block_must_start_on_the_first_line() {
        assert_eq!(extent(&rope("\n---\ntitle: x\n---\n")), None);
        assert_eq!(extent(&rope("# Note\n---\ntitle: x\n---\n")), None);
    }

    #[test]
    fn only_three_markers_delimit() {
        assert_eq!(extent(&rope("----\ntitle: x\n----\n")), None);
        assert_eq!(extent(&rope("--\ntitle: x\n--\n")), None);
        // A four-dash line does not close a block a three-dash line opened.
        assert_eq!(extent(&rope("---\ntitle: x\n----\n---\n")), Some(4));
    }

    #[test]
    fn a_thematic_break_variant_does_not_open_a_block() {
        assert_eq!(extent(&rope("***\ntitle: x\n***\n")), None);
        assert_eq!(extent(&rope("- - -\ntitle: x\n---\n")), None);
    }

    /// A block with its closer on logical line `closer`, padded with pairs.
    fn block_closing_at(closer: usize) -> Rope {
        let mut doc = String::from("---\n");
        for i in 1..closer {
            doc.push_str(&format!("k{i}: v\n"));
        }
        doc.push_str("---\nbody\n");
        rope(&doc)
    }

    #[test]
    fn the_closer_search_stops_at_the_cap() {
        // The last line the scan reaches still closes the block.
        assert_eq!(
            extent(&block_closing_at(MAX_LINES - 1)),
            Some(MAX_LINES),
            "a closer on the last scanned line is found"
        );
        assert_eq!(
            extent(&block_closing_at(MAX_LINES)),
            None,
            "one line further is past the cap, so the opener was a rule"
        );
    }

    #[test]
    fn the_first_closer_wins() {
        assert_eq!(extent(&rope("---\na: 1\n---\nbody\n---\nmore\n")), Some(3));
    }

    #[test]
    fn an_empty_document_has_no_block() {
        assert_eq!(extent(&rope("")), None);
    }
}
