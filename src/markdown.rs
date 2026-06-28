//! The markdown layer: classifying a logical line so the editor can render
//! markup inline rather than in a side preview.
//!
//! Markdown is rendered as you type, in two passes, both per *logical* line
//! (the text between two `'\n'`s):
//!
//! - **Block style** ([`style_of`]): an ATX heading (`# ` .. `###### `) renders
//!   larger; a bullet (`- `), ordered (`1. `), or quote (`> `) line carries a
//!   leading marker. [`crate::layout`] resolves the style to a pixel size and a
//!   marker length and stamps them onto every visual line the logical line wraps
//!   into, sharing all the cursor and selection math. A heading hides its marker
//!   once the cursor leaves; a list or quote marker persists (see
//!   [`block_marker_persists`]).
//! - **Inline spans** ([`inline_spans`]): code spans and emphasis (`*`/`_`)
//!   style interior runs of a line, with their delimiters hidden once the cursor
//!   leaves the line.
//!
//! Scope for now: ATX headings, bullet/ordered/quote lines (which may be indented
//! for nesting, [`LIST_INDENT_SPACES`] per level), thematic breaks (horizontal
//! rules), inline code, and bold/italic emphasis. Indented *headings* and quotes
//! are still not recognized; only list items nest.

/// The block style of one logical line. A closed set, extended as more markup
/// lands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineStyle {
    /// Ordinary paragraph text.
    Body,
    /// An ATX heading of the given level, 1..=6.
    Heading(u8),
    /// A bullet list item (`-`, `+`, or `*` then a space).
    Bullet,
    /// An ordered list item (`1.` or `1)` then a space).
    Ordered,
    /// A blockquote (`>` then an optional space).
    Quote,
    /// A thematic break (horizontal rule): a line of `---`, `***`, `___`, or a
    /// spaced variant. Drawn as a horizontal line once the cursor leaves it.
    Rule,
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

/// The block style of `line`. Headings win first (a `#` line is never a list),
/// then quotes, bullets, and ordered items; everything else is body text.
pub fn style_of(line: &str) -> LineStyle {
    if let Some(level) = heading_level(line) {
        LineStyle::Heading(level)
    } else if is_thematic_break(line) {
        // Checked before bullets so a spaced `- - -` or `* * *` reads as a rule,
        // not a list item.
        LineStyle::Rule
    } else if quote_marker(line).is_some() {
        LineStyle::Quote
    } else if bullet_marker(line).is_some() {
        LineStyle::Bullet
    } else if ordered_marker(line).is_some() {
        LineStyle::Ordered
    } else {
        LineStyle::Body
    }
}

/// Whether `line` is a thematic break (horizontal rule): three or more of the
/// same marker character (`-`, `*`, or `_`), separated only by spaces or tabs and
/// nothing else, per CommonMark. So `---`, `***`, `___`, and `- - -` are rules,
/// but `--` (too few), `-*-` (mixed), and `- item` (other text) are not.
pub fn is_thematic_break(line: &str) -> bool {
    let mut marker = None;
    let mut count = 0u32;
    for &b in line.as_bytes() {
        match b {
            b'-' | b'*' | b'_' => {
                if *marker.get_or_insert(b) != b {
                    return false;
                }
                count += 1;
            }
            b' ' | b'\t' => {}
            _ => return false,
        }
    }
    count >= 3
}

/// Byte length of `line`'s block marker prefix (the syntax drawn in a distinct
/// color so it reads as markup): a heading's `#`s plus a space, a list bullet or
/// number plus a space, or a quote `>` plus an optional space. `0` for body text.
pub fn marker_len(line: &str) -> usize {
    match style_of(line) {
        LineStyle::Body => 0,
        LineStyle::Heading(_) => heading_marker_len(line),
        LineStyle::Bullet => bullet_marker(line).unwrap_or(0),
        LineStyle::Ordered => ordered_marker(line).unwrap_or(0),
        LineStyle::Quote => quote_marker(line).unwrap_or(0),
        // The whole rule is its marker: it collapses to the drawn line off-cursor.
        LineStyle::Rule => line.len(),
    }
}

/// Whether `style`'s block marker stays visible once the cursor leaves the line.
/// A heading hides its `#`s (the large text reads cleaner bare), but a list
/// bullet or quote `>` is the only thing marking the line as a list or quote, so
/// it persists; otherwise the line would read as an ordinary paragraph.
pub fn block_marker_persists(style: LineStyle) -> bool {
    matches!(
        style,
        LineStyle::Bullet | LineStyle::Ordered | LineStyle::Quote
    )
}

/// The marker that should open the next item when Enter is pressed on `line`,
/// or `None` if `line` is not a continuable block (body, heading). A bullet or
/// quote repeats its own marker verbatim; an ordered item advances the number,
/// keeping its `.`/`)` delimiter and trailing space (`1. ` -> `2. `, `3) ` ->
/// `4) `). The caller decides emptiness; this only builds the marker.
pub fn next_item_marker(line: &str) -> Option<String> {
    let marker = &line[..marker_len(line)];
    match style_of(line) {
        // A bullet or quote repeats its marker verbatim, keeping any indentation.
        LineStyle::Bullet | LineStyle::Quote => Some(marker.to_string()),
        LineStyle::Ordered => {
            // The marker is <indent><digits><'.'|')'><space>; bump the number,
            // keep the indentation and delimiter so a nested item stays nested.
            let indent = leading_indent(marker);
            let rest = &marker[indent..];
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            let next = rest[..digits].parse::<u64>().ok()?.wrapping_add(1);
            Some(format!("{}{next}{}", &marker[..indent], &rest[digits..]))
        }
        LineStyle::Body | LineStyle::Heading(_) | LineStyle::Rule => None,
    }
}

/// If `line` opens a fenced code block, the number of backticks in its fence
/// (at least three). The info string after the fence must contain no backtick,
/// per CommonMark, so a one-line `` ``` x ``` `` is not taken for an opener. The
/// fence's open/close pairing is stateful and lives in [`crate::layout`], which
/// tracks it across lines; this only recognizes a single line in isolation.
pub fn opens_code_fence(line: &str) -> Option<usize> {
    let ticks = line.bytes().take_while(|&b| b == b'`').count();
    if ticks < 3 || line[ticks..].contains('`') {
        return None;
    }
    Some(ticks)
}

/// Whether `line` closes a fence opened with `open` backticks: a run of at least
/// `open` backticks followed by only whitespace (a closing fence carries no info
/// string).
pub fn closes_code_fence(line: &str, open: usize) -> bool {
    let ticks = line.bytes().take_while(|&b| b == b'`').count();
    ticks >= open && line[ticks..].trim().is_empty()
}

/// Byte length of a heading's `#` run plus the single separating space if present.
fn heading_marker_len(line: &str) -> usize {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    match line.as_bytes().get(hashes) {
        Some(&b' ') => hashes + 1,
        _ => hashes,
    }
}

/// Spaces of leading indentation per list nesting level. A list item with this
/// many leading spaces is one level deep, twice as many is two levels, and so on
/// (see [`crate::layout`], which resolves a depth against its parent). Tab on a
/// list item adds one unit; Shift+Tab removes one.
pub const LIST_INDENT_SPACES: usize = 2;

/// The count of leading spaces on `line` (its list indentation).
pub fn leading_indent(line: &str) -> usize {
    line.bytes().take_while(|&b| b == b' ').count()
}

/// Marker length if `line` is a bullet item: optional leading indentation (for
/// nesting), then `-`, `+`, or `*` followed by a space. The returned length spans
/// the indentation and the marker. The space is required, so `*italic*` is not
/// mistaken for a bullet and a bare `-` stays body text.
fn bullet_marker(line: &str) -> Option<usize> {
    let indent = leading_indent(line);
    let rest = &line.as_bytes()[indent..];
    match rest.first() {
        Some(b'-' | b'+' | b'*') if rest.get(1) == Some(&b' ') => Some(indent + 2),
        _ => None,
    }
}

/// Marker length if `line` is an ordered item: optional leading indentation, one
/// to nine digits, then `.` or `)`, then a required space (so `1.foo` and a bare
/// `1.` stay body text). The length spans the indentation and the marker.
fn ordered_marker(line: &str) -> Option<usize> {
    let indent = leading_indent(line);
    let bytes = &line.as_bytes()[indent..];
    let digits = bytes.iter().take_while(|&&b| b.is_ascii_digit()).count();
    if digits == 0 || digits > 9 {
        return None;
    }
    match bytes.get(digits) {
        Some(b'.' | b')') if bytes.get(digits + 1) == Some(&b' ') => Some(indent + digits + 2),
        _ => None,
    }
}

/// Marker length if `line` is a blockquote: a leading `>`, plus a single space if
/// one follows. Single level only (`>> nested` is read as one quote).
fn quote_marker(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    if bytes.first() != Some(&b'>') {
        return None;
    }
    Some(if bytes.get(1) == Some(&b' ') { 2 } else { 1 })
}

/// The pixel size `style` renders at, scaled from the body `base`. Heading scales
/// step down from 1.7x at H1 to body size at H5, and H6 is slightly *smaller*
/// than body (0.9x): per-level color, not size, marks the lower headings. Lists
/// and quotes are body size (they style via their marker and indentation, not
/// size). Scales are in tenths to stay in integer pixels without a float
/// dependency on rounding.
pub fn font_size(style: LineStyle, base: u32) -> u32 {
    let tenths = match style {
        LineStyle::Body
        | LineStyle::Bullet
        | LineStyle::Ordered
        | LineStyle::Quote
        | LineStyle::Rule => {
            return base;
        }
        LineStyle::Heading(1) => 17,
        LineStyle::Heading(2) => 15,
        LineStyle::Heading(3) => 12,
        LineStyle::Heading(4) => 11,
        LineStyle::Heading(5) => 10,
        LineStyle::Heading(6) => 9,
        // Any out-of-range level falls back to body size.
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
// span or emphasis styles an interior run rather than the whole line. The
// scanner below splits a content string into a contiguous list of [`Span`]s
// covering every byte, so [`crate::layout`] can clip them to visual lines and
// hide the delimiters once the cursor leaves the line, the same live-preview
// collapse headings already use.
//
// Two layers, in CommonMark's precedence order:
//
//   1. Code spans (`` `code` ``): a run of N backticks opens a span the next run
//      of exactly N backticks closes; an unclosed run is literal. Delimiters
//      inside a code span (a `*`, say) are inert.
//   2. Emphasis (`*`/`_`, `**`/`__`, `***`/`___`): `*`/`_` runs outside code are
//      paired opener-to-closer with a stack (innermost first) to mark italic,
//      bold, or both. Whether a run can open or close follows a simplified
//      reading of CommonMark's flanking rules (including the `_` intraword
//      restriction, so `snake_case` stays literal).
//
// Deliberate simplifications: only single/double/triple delimiter runs emphasize
// (longer runs stay literal), an opener matches only a closer of the same length
// (no partial-run consumption), emphasis is single-line, and backslash escapes
// and the one-space stripping inside code spans are not handled. Notes rarely
// need the omitted cases.

/// Inline emphasis of a text run, selecting the face and color it renders with.
/// A closed set; [`crate::freetype::FontStyle`] maps it to an actual face.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Emphasis {
    /// Ordinary body text.
    Regular,
    /// An inline code span, drawn monospace on a tinted background.
    Code,
    /// `**bold**` / `__bold__`, drawn in the bold face.
    Bold,
    /// `*italic*` / `_italic_`, drawn in the italic face.
    Italic,
    /// `***both***`, drawn in the bold-italic face.
    BoldItalic,
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
/// recognizing code spans and `*`/`_` emphasis. Text between and around markup
/// becomes `Text(..)` spans, so the result has no gaps. Empty input yields no
/// spans.
pub fn inline_spans(content: &str) -> Vec<Span> {
    let bytes = content.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return Vec::new();
    }

    // Code spans first, since they bind tighter than emphasis. Flag each fence and
    // content byte; `in_code` marks every byte a `*`/`_` would be inert inside.
    let mut is_code_marker = vec![false; n];
    let mut is_code_content = vec![false; n];
    let mut in_code = vec![false; n];
    let mut i = 0;
    while i < n {
        if bytes[i] != b'`' {
            i += 1;
            continue;
        }
        let open = i;
        let after_open = backtick_run_end(bytes, open);
        let fence = after_open - open;
        let Some((close, after_close)) = find_code_closer(bytes, after_open, fence) else {
            // Unclosed: the backticks are literal text. Skip past them so a later
            // run of the same length is not mistaken for this run's closer.
            i = after_open;
            continue;
        };
        for b in open..after_open {
            is_code_marker[b] = true;
            in_code[b] = true;
        }
        for b in after_open..close {
            is_code_content[b] = true;
            in_code[b] = true;
        }
        for b in close..after_close {
            is_code_marker[b] = true;
            in_code[b] = true;
        }
        i = after_close;
    }

    // Then emphasis. Pair `*`/`_` delimiter runs outside code, then flag the
    // matched delimiters as markers and stamp bold/italic over the text between.
    let delims = emphasis_delims(content, &in_code);
    let mut is_em_marker = vec![false; n];
    let mut is_bold = vec![false; n];
    let mut is_italic = vec![false; n];
    for (open, close) in match_delims(&delims) {
        is_em_marker[delims[open].start..delims[open].end].fill(true);
        is_em_marker[delims[close].start..delims[close].end].fill(true);
        // A run of 1 is italic, 2 is bold, 3 is both.
        let bold = delims[open].len != 1;
        let italic = delims[open].len != 2;
        for b in delims[open].end..delims[close].start {
            is_bold[b] |= bold;
            is_italic[b] |= italic;
        }
    }

    // Resolve each byte to a single kind, then coalesce equal neighbors so the
    // spans tile the content with no gaps and no redundant splits.
    let kind_at = |b: usize| -> SpanKind {
        if is_code_marker[b] || is_em_marker[b] {
            SpanKind::Marker
        } else if is_code_content[b] {
            SpanKind::Text(Emphasis::Code)
        } else {
            SpanKind::Text(match (is_bold[b], is_italic[b]) {
                (true, true) => Emphasis::BoldItalic,
                (true, false) => Emphasis::Bold,
                (false, true) => Emphasis::Italic,
                (false, false) => Emphasis::Regular,
            })
        }
    };
    let mut spans = Vec::new();
    let mut start = 0;
    let mut cur = kind_at(0);
    for b in 1..n {
        let kind = kind_at(b);
        if kind != cur {
            spans.push(Span {
                start,
                end: b,
                kind: cur,
            });
            start = b;
            cur = kind;
        }
    }
    spans.push(Span {
        start,
        end: n,
        kind: cur,
    });
    spans
}

/// One maximal run of `*` or `_` lying outside any code span, with whether it can
/// open and/or close emphasis under the (simplified) flanking rules.
struct Delim {
    start: usize,
    end: usize,
    ch: u8,
    len: usize,
    can_open: bool,
    can_close: bool,
}

/// Collect the emphasis delimiter runs of `content`, skipping any byte flagged in
/// `in_code`. Only single/double/triple runs are kept; longer runs stay literal.
fn emphasis_delims(content: &str, in_code: &[bool]) -> Vec<Delim> {
    let bytes = content.as_bytes();
    let n = bytes.len();
    let mut delims = Vec::new();
    let mut i = 0;
    while i < n {
        let ch = bytes[i];
        if in_code[i] || (ch != b'*' && ch != b'_') {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end < n && bytes[end] == ch && !in_code[end] {
            end += 1;
        }
        let len = end - start;
        if (1..=3).contains(&len) {
            // Flanking is decided by the raw neighboring characters; `start` and
            // `end` sit on char boundaries (the run is ASCII `*`/`_`).
            let before = content[..start].chars().next_back();
            let after = content[end..].chars().next();
            let (can_open, can_close) = flanking(ch, before, after);
            delims.push(Delim {
                start,
                end,
                ch,
                len,
                can_open,
                can_close,
            });
        }
        i = end;
    }
    delims
}

/// Pair delimiter runs into `(opener, closer)` index tuples, matching the same
/// character and length innermost-first with a stack. Unmatched runs are dropped
/// (they stay literal text). A run that can both open and close prefers closing.
fn match_delims(delims: &[Delim]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut open_stack: Vec<usize> = Vec::new();
    for (idx, delim) in delims.iter().enumerate() {
        if delim.can_close {
            if let Some(pos) = open_stack
                .iter()
                .rposition(|&oi| delims[oi].ch == delim.ch && delims[oi].len == delim.len)
            {
                pairs.push((open_stack[pos], idx));
                // Drop the opener and any unmatched openers nested inside it.
                open_stack.truncate(pos);
                continue;
            }
        }
        if delim.can_open {
            open_stack.push(idx);
        }
    }
    pairs
}

/// Whether a delimiter run of `ch` can open and/or close emphasis, from the
/// characters immediately before and after it: a simplified reading of
/// CommonMark's left/right-flanking rules. A missing neighbor (the line edge)
/// counts as whitespace. The `_` form keeps the intraword restriction so an
/// underscore inside a word neither opens nor closes.
fn flanking(ch: u8, before: Option<char>, after: Option<char>) -> (bool, bool) {
    let before_ws = before.is_none_or(|c| c.is_whitespace());
    let after_ws = after.is_none_or(|c| c.is_whitespace());
    let before_punct = before.is_some_and(|c| c.is_ascii_punctuation());
    let after_punct = after.is_some_and(|c| c.is_ascii_punctuation());

    let left_flanking = !after_ws && (!after_punct || before_ws || before_punct);
    let right_flanking = !before_ws && (!before_punct || after_ws || after_punct);
    if ch == b'_' {
        (
            left_flanking && (!right_flanking || before_punct),
            right_flanking && (!left_flanking || after_punct),
        )
    } else {
        (left_flanking, right_flanking)
    }
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
    fn font_size_steps_down_with_a_subbody_h6() {
        let base = 32;
        assert_eq!(font_size(LineStyle::Body, base), base);
        let h1 = font_size(LineStyle::Heading(1), base);
        let h2 = font_size(LineStyle::Heading(2), base);
        let h3 = font_size(LineStyle::Heading(3), base);
        assert!(h1 > h2 && h2 > h3 && h3 > base);
        // H5 matches body; H6 is smaller (the theme marks low headings by color).
        assert_eq!(font_size(LineStyle::Heading(5), base), base);
        assert!(font_size(LineStyle::Heading(6), base) < base);
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

    // --- Lists & blockquotes ----------------------------------------------

    #[test]
    fn recognizes_bullets_with_each_marker() {
        for marker in ["- ", "+ ", "* "] {
            let line = format!("{marker}item");
            assert_eq!(style_of(&line), LineStyle::Bullet, "{line:?}");
            assert_eq!(marker_len(&line), 2, "{line:?}");
        }
    }

    #[test]
    fn a_bullet_needs_its_space_so_emphasis_and_words_are_safe() {
        // `*italic*` and a hyphenated word must not read as bullets.
        assert_eq!(style_of("*italic*"), LineStyle::Body);
        assert_eq!(style_of("-dash"), LineStyle::Body);
        assert_eq!(style_of("-"), LineStyle::Body);
        assert_eq!(marker_len("*italic*"), 0);
    }

    #[test]
    fn recognizes_ordered_items_with_dot_or_paren() {
        assert_eq!(style_of("1. first"), LineStyle::Ordered);
        assert_eq!(marker_len("1. first"), 3);
        assert_eq!(style_of("10) tenth"), LineStyle::Ordered);
        assert_eq!(marker_len("10) tenth"), 4);
        // No space, too many digits, or a bare number: body text.
        assert_eq!(style_of("1.foo"), LineStyle::Body);
        assert_eq!(style_of("1234567890. x"), LineStyle::Body);
        assert_eq!(style_of("1."), LineStyle::Body);
    }

    #[test]
    fn recognizes_quotes_with_optional_space() {
        assert_eq!(style_of("> quoted"), LineStyle::Quote);
        assert_eq!(marker_len("> quoted"), 2);
        assert_eq!(style_of(">tight"), LineStyle::Quote);
        assert_eq!(marker_len(">tight"), 1);
        assert_eq!(style_of(">"), LineStyle::Quote);
        assert_eq!(marker_len(">"), 1);
    }

    #[test]
    fn headings_win_over_other_blocks_and_lists_persist() {
        // A `#` line is a heading, not a list, and a heading hides its marker.
        assert_eq!(style_of("# h"), LineStyle::Heading(1));
        assert!(!block_marker_persists(LineStyle::Heading(1)));
        assert!(!block_marker_persists(LineStyle::Body));
        // Lists and quotes keep their markers on screen.
        for style in [LineStyle::Bullet, LineStyle::Ordered, LineStyle::Quote] {
            assert!(block_marker_persists(style), "{style:?} persists");
            assert_eq!(font_size(style, 32), 32, "{style:?} is body size");
        }
    }

    #[test]
    fn next_item_marker_repeats_a_bullet_or_quote_and_advances_a_number() {
        assert_eq!(next_item_marker("- x").as_deref(), Some("- "));
        assert_eq!(next_item_marker("+ x").as_deref(), Some("+ "));
        assert_eq!(next_item_marker("* x").as_deref(), Some("* "));
        assert_eq!(next_item_marker("1. a").as_deref(), Some("2. "));
        assert_eq!(
            next_item_marker("9) z").as_deref(),
            Some("10) "),
            "digit carry"
        );
        assert_eq!(next_item_marker("> q").as_deref(), Some("> "));
        assert_eq!(next_item_marker(">tight").as_deref(), Some(">"));
        // Headings and body never continue.
        assert_eq!(next_item_marker("# h"), None);
        assert_eq!(next_item_marker("plain"), None);
    }

    #[test]
    fn recognizes_indented_list_items_for_nesting() {
        // Leading spaces before a marker are nesting indentation; the marker
        // length spans them.
        assert_eq!(style_of("  - sub"), LineStyle::Bullet);
        assert_eq!(marker_len("  - sub"), 4);
        assert_eq!(leading_indent("  - sub"), 2);
        assert_eq!(style_of("    1. deep"), LineStyle::Ordered);
        assert_eq!(marker_len("    1. deep"), 7);
        // A flat item is unchanged, and a bare indented word is still body.
        assert_eq!(marker_len("- x"), 2);
        assert_eq!(leading_indent("- x"), 0);
        assert_eq!(style_of("    plain"), LineStyle::Body);
    }

    #[test]
    fn next_item_marker_preserves_indentation() {
        assert_eq!(next_item_marker("  - a").as_deref(), Some("  - "));
        assert_eq!(next_item_marker("  1. a").as_deref(), Some("  2. "));
        assert_eq!(
            next_item_marker("    9) z").as_deref(),
            Some("    10) "),
            "indented ordered carries its indent and carries the digit"
        );
    }

    #[test]
    fn recognizes_thematic_breaks() {
        for line in [
            "---",
            "***",
            "___",
            "- - -",
            "* * *",
            "_ _ _",
            "----------",
            "*** ",
        ] {
            assert!(is_thematic_break(line), "{line:?} is a rule");
            assert_eq!(style_of(line), LineStyle::Rule, "{line:?}");
        }
        // The whole line is the marker, so it collapses to the drawn rule off-line.
        assert_eq!(marker_len("---"), 3);
        assert_eq!(marker_len("- - -"), 5);
        // A rule never continues on Enter and is body size.
        assert_eq!(next_item_marker("---"), None);
        assert_eq!(font_size(LineStyle::Rule, 32), 32);
    }

    #[test]
    fn non_rules_are_not_thematic_breaks() {
        for line in [
            "--", "__", "**", "-x-", "-*-", "- item", "* list", "**bold**", "", "## h", "1. a",
        ] {
            assert!(!is_thematic_break(line), "{line:?} is not a rule");
        }
        // A spaced dash run is a rule, but a real bullet stays a bullet.
        assert_eq!(style_of("- - -"), LineStyle::Rule);
        assert_eq!(style_of("- item"), LineStyle::Bullet);
    }

    #[test]
    fn recognizes_code_fence_openers_and_closers() {
        // Three or more backticks open; an info string (language) is allowed.
        assert_eq!(opens_code_fence("```"), Some(3));
        assert_eq!(opens_code_fence("```rust"), Some(3));
        assert_eq!(opens_code_fence("````"), Some(4));
        // Fewer than three, or an info string holding a backtick, is not an opener
        // (so a one-line `` ``` x ``` `` is left to the inline pass).
        assert_eq!(opens_code_fence("``"), None);
        assert_eq!(opens_code_fence("``` x ```"), None);
        assert_eq!(opens_code_fence("not a fence"), None);
        // A closer is bare backticks (>= the opener), only whitespace after.
        assert!(closes_code_fence("```", 3));
        assert!(closes_code_fence("````", 3), "a longer fence still closes");
        assert!(closes_code_fence("```   ", 3), "trailing space is allowed");
        assert!(!closes_code_fence("``", 3), "too few backticks");
        assert!(
            !closes_code_fence("```rust", 3),
            "a closer carries no info string"
        );
        assert!(!closes_code_fence("```", 4), "must match the opener length");
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
        for input in [
            "`x`",
            "a`x`b",
            "plain",
            "`x` `y`",
            "``a`b``",
            "*i*",
            "**b**",
            "***x***",
            "_u_",
            "a*b*c",
            "**a *b* c**",
            "*a `b` c*",
            "foo_bar",
            "*unclosed",
        ] {
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

    // --- Emphasis ---------------------------------------------------------

    #[test]
    fn single_run_is_italic_double_is_bold_triple_is_both() {
        assert_eq!(
            spans("*i*"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 2, SpanKind::Text(Emphasis::Italic)),
                (2, 3, SpanKind::Marker),
            ]
        );
        assert_eq!(
            spans("**b**"),
            vec![
                (0, 2, SpanKind::Marker),
                (2, 3, SpanKind::Text(Emphasis::Bold)),
                (3, 5, SpanKind::Marker),
            ]
        );
        assert_eq!(
            spans("***x***"),
            vec![
                (0, 3, SpanKind::Marker),
                (3, 4, SpanKind::Text(Emphasis::BoldItalic)),
                (4, 7, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn underscores_emphasize_like_asterisks() {
        assert_eq!(
            spans("_u_"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 2, SpanKind::Text(Emphasis::Italic)),
                (2, 3, SpanKind::Marker),
            ]
        );
        assert_eq!(
            spans("__u__"),
            vec![
                (0, 2, SpanKind::Marker),
                (2, 3, SpanKind::Text(Emphasis::Bold)),
                (3, 5, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn intraword_underscores_stay_literal_but_asterisks_emphasize() {
        // `_` inside a word neither opens nor closes, so identifiers are safe.
        assert_eq!(
            spans("foo_bar_baz"),
            vec![(0, 11, SpanKind::Text(Emphasis::Regular))]
        );
        // `*` has no intraword restriction, so it still italicizes mid-word.
        assert_eq!(
            spans("foo*bar*baz"),
            vec![
                (0, 3, SpanKind::Text(Emphasis::Regular)),
                (3, 4, SpanKind::Marker),
                (4, 7, SpanKind::Text(Emphasis::Italic)),
                (7, 8, SpanKind::Marker),
                (8, 11, SpanKind::Text(Emphasis::Regular)),
            ]
        );
    }

    #[test]
    fn an_unmatched_or_mismatched_run_is_literal() {
        assert_eq!(
            spans("*unclosed"),
            vec![(0, 9, SpanKind::Text(Emphasis::Regular))]
        );
        // A `**` opener and a `*` closer differ in length, so neither matches.
        assert_eq!(
            spans("**a*"),
            vec![(0, 4, SpanKind::Text(Emphasis::Regular))]
        );
    }

    #[test]
    fn italic_nested_in_bold_is_bold_italic() {
        // The inner `*b*` lies within the outer `**..**`, so "b" gains both.
        assert_eq!(
            spans("**a *b* c**"),
            vec![
                (0, 2, SpanKind::Marker),
                (2, 4, SpanKind::Text(Emphasis::Bold)),
                (4, 5, SpanKind::Marker),
                (5, 6, SpanKind::Text(Emphasis::BoldItalic)),
                (6, 7, SpanKind::Marker),
                (7, 9, SpanKind::Text(Emphasis::Bold)),
                (9, 11, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn emphasis_wraps_an_inner_code_span() {
        // Code binds tighter, so its backticks stay inert inside the italic, and
        // the emphasis spans across it.
        assert_eq!(
            spans("*a `b` c*"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 3, SpanKind::Text(Emphasis::Italic)),
                (3, 4, SpanKind::Marker),
                (4, 5, SpanKind::Text(Emphasis::Code)),
                (5, 6, SpanKind::Marker),
                (6, 8, SpanKind::Text(Emphasis::Italic)),
                (8, 9, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn a_star_inside_a_code_span_is_inert() {
        // The `*`s are code content, not emphasis delimiters.
        assert_eq!(
            spans("`a*b*c`"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 6, SpanKind::Text(Emphasis::Code)),
                (6, 7, SpanKind::Marker),
            ]
        );
    }
}
