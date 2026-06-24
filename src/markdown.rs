//! The markdown layer: classifying a logical line so the editor can render
//! markup inline rather than in a side preview.
//!
//! Markdown is rendered as you type, in two passes, both per *logical* line
//! (the text between two `'\n'`s):
//!
//! - **Block style** ([`style_of`]): an ATX heading (`# ` .. `###### `) renders
//!   larger. [`crate::layout`] resolves the style to a pixel size and stamps it
//!   onto every visual line the logical line wraps into, sharing all the cursor
//!   and selection math.
//! - **Inline spans** ([`inline_spans`]): code spans and (later) emphasis style
//!   interior runs of a line, with their delimiters hidden once the cursor
//!   leaves the line.
//!
//! Scope for now: ATX headings and inline code. Indented headings (CommonMark
//! allows up to three leading spaces) are deliberately not recognized; notes are
//! written at the left margin.

/// The block style of one logical line. A closed set, extended as more markup
/// lands (lists, quotes, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineStyle {
    /// Ordinary paragraph text.
    Body,
    /// An ATX heading of the given level, 1..=6.
    Heading(u8),
}

/// The ATX heading level of `line`, or `None` if it is not a heading. A heading
/// is one to six leading `'#'` followed immediately by a space or the line's
/// end, per CommonMark: `#tag` is not a heading, but `#`, `# `, and `# Title`
/// are. `line` must be a single logical line (no embedded `'\n'`).
pub fn heading_level(line: &str) -> Option<u8> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    match line.as_bytes().get(hashes) {
        // Hashes run to the end of the line: an empty heading.
        None => Some(hashes as u8),
        // A space separates the marker from the heading text.
        Some(&b' ') => Some(hashes as u8),
        // Anything else (e.g. `#tag`) is not a heading.
        Some(_) => None,
    }
}

/// The block style of `line`.
pub fn style_of(line: &str) -> LineStyle {
    match heading_level(line) {
        Some(level) => LineStyle::Heading(level),
        None => LineStyle::Body,
    }
}

/// Byte length of a heading's marker prefix: the run of `'#'` plus the single
/// separating space if one is present. `0` for a non-heading line. Used to draw
/// the markers in a distinct color so the markup reads as syntax.
pub fn marker_len(line: &str) -> usize {
    if heading_level(line).is_none() {
        return 0;
    }
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    match line.as_bytes().get(hashes) {
        Some(&b' ') => hashes + 1,
        _ => hashes,
    }
}

/// The pixel size `style` renders at, scaled from the body `base`. Headings step
/// down from H1 toward body size; H5 and H6 match body (they read as bold body
/// text in most themes, and bold is not wired up yet). Scales are in tenths to
/// stay in integer pixels without a float dependency on rounding.
pub fn font_size(style: LineStyle, base: u32) -> u32 {
    let tenths = match style {
        LineStyle::Body => return base,
        LineStyle::Heading(1) => 16,
        LineStyle::Heading(2) => 14,
        LineStyle::Heading(3) => 12,
        LineStyle::Heading(4) => 11,
        // H5, H6, and any out-of-range level fall back to body size.
        _ => 10,
    };
    (base * tenths + 5) / 10
}

/// Every distinct pixel size the styles use at body size `base`, so the font set
/// can be opened up front. Body first, then each heading size not already seen.
pub fn all_sizes(base: u32) -> Vec<u32> {
    let mut sizes = vec![font_size(LineStyle::Body, base)];
    for level in 1..=6u8 {
        let size = font_size(LineStyle::Heading(level), base);
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    sizes
}

// --- Inline spans -----------------------------------------------------------
//
// Within a logical line's content (past any block marker), markup like a code
// span styles an interior run rather than the whole line. The scanner below
// splits a content string into a contiguous list of [`Span`]s covering every
// byte, so [`crate::layout`] can clip them to visual lines and hide the
// delimiters once the cursor leaves the line, the same live-preview collapse
// headings already use.
//
// Scope for now: code spans only (`` `code` ``). This is a deliberately
// simplified reading of CommonMark: a run of N backticks opens a span that the
// next run of exactly N backticks closes, and an unclosed run is literal text.
// Backslash escapes and the one-space stripping CommonMark does inside a code
// span are not handled; notes rarely need either.

/// Inline emphasis of a text run, selecting the face and color it renders with.
/// A closed set, extended as bold and italic land.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Emphasis {
    /// Ordinary body text.
    Regular,
    /// An inline code span, drawn monospace on a tinted background.
    Code,
}

/// What one inline span is: text to render with an [`Emphasis`], or a delimiter
/// marker (a code span's backticks) that live-preview hides off the active line.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpanKind {
    Text(Emphasis),
    Marker,
}

/// One inline span: a byte range within the scanned content, and what it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: SpanKind,
}

/// Split `content` into a contiguous list of inline spans covering `[0, len)`,
/// recognizing code spans. Text between and around code spans becomes
/// `Text(Regular)` spans, so the result has no gaps. Empty input yields no spans.
pub fn inline_spans(content: &str) -> Vec<Span> {
    let bytes = content.as_bytes();
    let mut spans = Vec::new();
    // Start of the pending run of regular text not yet emitted.
    let mut text_start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        // A run of `fence` backticks opens a code span only if a run of exactly
        // the same length closes it later on the line.
        let open = i;
        let after_open = backtick_run_end(bytes, open);
        let fence = after_open - open;
        let Some((close, after_close)) = find_code_closer(bytes, after_open, fence) else {
            // Unclosed: the backticks are literal text. Skip past them so a later
            // run of the same length is not mistaken for this run's closer.
            i = after_open;
            continue;
        };
        if open > text_start {
            spans.push(Span {
                start: text_start,
                end: open,
                kind: SpanKind::Text(Emphasis::Regular),
            });
        }
        spans.push(Span {
            start: open,
            end: after_open,
            kind: SpanKind::Marker,
        });
        if close > after_open {
            spans.push(Span {
                start: after_open,
                end: close,
                kind: SpanKind::Text(Emphasis::Code),
            });
        }
        spans.push(Span {
            start: close,
            end: after_close,
            kind: SpanKind::Marker,
        });
        text_start = after_close;
        i = after_close;
    }
    if text_start < bytes.len() {
        spans.push(Span {
            start: text_start,
            end: bytes.len(),
            kind: SpanKind::Text(Emphasis::Regular),
        });
    }
    spans
}

/// The byte index just past the run of backticks starting at `start` (which must
/// be a backtick).
fn backtick_run_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() && bytes[i] == b'`' {
        i += 1;
    }
    i
}

/// Find the closing backtick run of exactly `fence` backticks at or after `from`,
/// returning its `(start, end)`. Runs of a different length are not closers and
/// are skipped over.
fn find_code_closer(bytes: &[u8], from: usize, fence: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i < bytes.len() {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let start = i;
        i = backtick_run_end(bytes, start);
        if i - start == fence {
            return Some((start, i));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_each_heading_level() {
        for level in 1..=6u8 {
            let line = format!("{} heading", "#".repeat(level as usize));
            assert_eq!(heading_level(&line), Some(level));
            assert_eq!(style_of(&line), LineStyle::Heading(level));
        }
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        assert_eq!(heading_level("####### too deep"), None);
        assert_eq!(style_of("####### too deep"), LineStyle::Body);
    }

    #[test]
    fn hash_without_a_space_is_not_a_heading() {
        assert_eq!(heading_level("#tag"), None);
        assert_eq!(heading_level("##bold"), None);
    }

    #[test]
    fn bare_and_space_only_markers_are_empty_headings() {
        assert_eq!(heading_level("#"), Some(1));
        assert_eq!(heading_level("###"), Some(3));
        assert_eq!(heading_level("# "), Some(1));
    }

    #[test]
    fn leading_space_disqualifies_a_heading() {
        assert_eq!(heading_level(" # indented"), None);
    }

    #[test]
    fn plain_text_and_empty_lines_are_body() {
        assert_eq!(style_of(""), LineStyle::Body);
        assert_eq!(style_of("just text"), LineStyle::Body);
    }

    #[test]
    fn marker_len_covers_the_hashes_and_one_space() {
        assert_eq!(marker_len("# Title"), 2);
        assert_eq!(marker_len("### Deep"), 4);
        assert_eq!(marker_len("#"), 1);
        assert_eq!(marker_len("## "), 3);
        assert_eq!(marker_len("body"), 0);
    }

    #[test]
    fn font_size_steps_down_to_body() {
        let base = 32;
        assert_eq!(font_size(LineStyle::Body, base), base);
        let h1 = font_size(LineStyle::Heading(1), base);
        let h2 = font_size(LineStyle::Heading(2), base);
        let h3 = font_size(LineStyle::Heading(3), base);
        assert!(h1 > h2 && h2 > h3 && h3 > base);
        assert_eq!(font_size(LineStyle::Heading(6), base), base);
    }

    #[test]
    fn all_sizes_are_distinct_and_include_body() {
        let sizes = all_sizes(32);
        assert_eq!(sizes[0], 32, "body size first");
        let mut seen = sizes.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), sizes.len(), "no duplicate sizes");
    }

    // --- Inline spans -----------------------------------------------------

    /// `(start, end, kind)` triples, terse enough to assert a whole span list.
    fn spans(s: &str) -> Vec<(usize, usize, SpanKind)> {
        inline_spans(s)
            .into_iter()
            .map(|sp| (sp.start, sp.end, sp.kind))
            .collect()
    }

    #[test]
    fn plain_text_is_one_regular_span() {
        assert_eq!(
            spans("hello"),
            vec![(0, 5, SpanKind::Text(Emphasis::Regular))]
        );
    }

    #[test]
    fn empty_content_has_no_spans() {
        assert!(inline_spans("").is_empty());
    }

    #[test]
    fn a_code_span_splits_into_marker_code_marker() {
        // "a `b` c": text, marker, code, marker, text.
        assert_eq!(
            spans("a `b` c"),
            vec![
                (0, 2, SpanKind::Text(Emphasis::Regular)),
                (2, 3, SpanKind::Marker),
                (3, 4, SpanKind::Text(Emphasis::Code)),
                (4, 5, SpanKind::Marker),
                (5, 7, SpanKind::Text(Emphasis::Regular)),
            ]
        );
    }

    #[test]
    fn spans_tile_the_content_with_no_gaps() {
        for input in ["`x`", "a`x`b", "plain", "`x` `y`", "``a`b``"] {
            let sp = inline_spans(input);
            assert_eq!(
                sp.first().map(|s| s.start),
                Some(0),
                "{input:?} starts at 0"
            );
            assert_eq!(
                sp.last().map(|s| s.end),
                Some(input.len()),
                "{input:?} reaches the end"
            );
            for pair in sp.windows(2) {
                assert_eq!(pair[0].end, pair[1].start, "{input:?} has no gap");
            }
        }
    }

    #[test]
    fn an_unclosed_backtick_is_literal_text() {
        assert_eq!(
            spans("a `b c"),
            vec![(0, 6, SpanKind::Text(Emphasis::Regular))],
            "no closer means the backtick is ordinary text"
        );
    }

    #[test]
    fn two_code_spans_on_a_line() {
        let sp = spans("`a` `b`");
        let markers = sp.iter().filter(|s| s.2 == SpanKind::Marker).count();
        let code = sp
            .iter()
            .filter(|s| s.2 == SpanKind::Text(Emphasis::Code))
            .count();
        assert_eq!(markers, 4, "two pairs of backticks");
        assert_eq!(code, 2, "two code runs");
    }

    #[test]
    fn a_double_backtick_fence_holds_a_literal_backtick() {
        // ``a`b`` : the inner single backtick is code content, not a closer.
        assert_eq!(
            spans("``a`b``"),
            vec![
                (0, 2, SpanKind::Marker),
                (2, 5, SpanKind::Text(Emphasis::Code)),
                (5, 7, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn a_lone_backtick_run_is_literal() {
        // Adjacent backticks are one greedy fence, so a bare "``" has no closer
        // and stays literal text rather than becoming an empty code span.
        assert_eq!(spans("``"), vec![(0, 2, SpanKind::Text(Emphasis::Regular))]);
    }
}
