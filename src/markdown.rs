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
//!   into, sharing all the cursor and selection math. A heading with text hides
//!   its marker once the cursor leaves; an empty heading's marker, and a list or
//!   quote marker, persist (see [`block_marker_persists`]).
//! - **Inline spans** ([`inline_spans_append`]): code spans, inline links, emphasis
//!   (`*`/`_`), and strikethrough (`~~`) style interior runs of a line, with their
//!   delimiters hidden once the cursor leaves the line.
//!
//! Scope for now: ATX headings, bullet/ordered/quote lines (which may be indented
//! for nesting, [`LIST_INDENT_SPACES`] per level), task-list items (`- [ ]` /
//! `- [x]`), thematic breaks (horizontal rules), inline code, bold/italic
//! emphasis, strikethrough, and inline links. Indented *headings* and quotes are
//! still not recognized; only list items nest.

use std::ops::Range;

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
    /// A task-list item: a bullet whose content opens with a checkbox, `[ ]`
    /// (unchecked) or `[x]`/`[X]` (checked), then a space. The whole
    /// `- [ ] ` prefix is its block marker.
    Task { checked: bool },
    /// A blockquote (`>` then an optional space).
    Quote,
    /// A thematic break (horizontal rule): a line of `---`, `***`, `___`, or a
    /// spaced variant. Drawn as a horizontal line once the cursor leaves it.
    Rule,
}

/// An ATX heading's level and marker length (its `'#'` run plus the single
/// separating space, if present), or `None` if `line` is not a heading. A heading
/// is one to six leading `'#'` followed immediately by a space or the line's end,
/// per CommonMark: `#tag` is not a heading, but `#`, `# `, and `# Title` are.
/// `line` must be a single logical line (no embedded `'\n'`). Both pieces fall out
/// of one scan, so a caller needing both (see [`classify`]) does not scan twice.
fn heading(line: &str) -> Option<(u8, usize)> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    match line.as_bytes().get(hashes) {
        // Hashes run to the end of the line: an empty heading, no space.
        None => Some((hashes as u8, hashes)),
        // A space separates the marker from the heading text.
        Some(&b' ') => Some((hashes as u8, hashes + 1)),
        // Anything else (e.g. `#tag`) is not a heading.
        Some(_) => None,
    }
}

/// The block style of `line` together with its block-marker length, classified
/// in a single pass. Headings win first (a `#` line is never a list), then
/// quotes, bullets, and ordered items; everything else is body text. The marker
/// length falls straight out of the predicate that matched (each returns it), so
/// a caller needing both does not run the predicates twice.
pub fn classify(line: &str) -> (LineStyle, usize) {
    if let Some((level, marker)) = heading(line) {
        (LineStyle::Heading(level), marker)
    } else if is_thematic_break(line) {
        // Checked before bullets so a spaced `- - -` or `* * *` reads as a rule,
        // not a list item. The whole rule is its marker (collapses off-cursor).
        (LineStyle::Rule, line.len())
    } else if let Some(len) = quote_marker(line) {
        (LineStyle::Quote, len)
    } else if let Some(len) = bullet_marker(line) {
        // A bullet whose content opens with a checkbox is a task item, and the
        // checkbox belongs to the marker.
        match task_box(line, len) {
            Some(checked) => (LineStyle::Task { checked }, len + 4),
            None => (LineStyle::Bullet, len),
        }
    } else if let Some((_, len)) = ordered_parts(line) {
        (LineStyle::Ordered, len)
    } else {
        (LineStyle::Body, 0)
    }
}

/// The block style of `line`. A thin projection of [`classify`].
pub fn style_of(line: &str) -> LineStyle {
    classify(line).0
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

/// Whether the block marker of `line` stays visible once the cursor leaves it,
/// given the `style` and `marker` length [`classify`] found. A list bullet or
/// quote `>` is the only thing marking the line as a list or quote, so it
/// persists; otherwise the line would read as an ordinary paragraph. A heading
/// hides its `#`s (the large text reads cleaner bare) as long as it has text to
/// read: an empty heading keeps them, since hiding the only glyphs on the line
/// would leave a blank the writer cannot see or click into.
pub fn block_marker_persists(line: &str, style: LineStyle, marker: usize) -> bool {
    match style {
        LineStyle::Bullet | LineStyle::Ordered | LineStyle::Quote | LineStyle::Task { .. } => true,
        LineStyle::Heading(_) => line.get(marker..).unwrap_or("").trim().is_empty(),
        LineStyle::Body | LineStyle::Rule => false,
    }
}

/// The marker that should open the next item when Enter is pressed on `line`,
/// or `None` if `line` is not a continuable block (body, heading). A bullet or
/// quote repeats its own marker verbatim; a task item continues unchecked
/// (`- [x] ` -> `- [ ] `); an ordered item advances the number, keeping its
/// `.`/`)` delimiter and trailing space (`1. ` -> `2. `, `3) ` -> `4) `). The
/// caller decides emptiness; this only builds the marker.
pub fn next_item_marker(line: &str) -> Option<String> {
    let (style, marker_len) = classify(line);
    let marker = &line[..marker_len];
    match style {
        // A bullet or quote repeats its marker verbatim, keeping any indentation.
        LineStyle::Bullet | LineStyle::Quote => Some(marker.to_string()),
        // A task item continues with a fresh unchecked box, keeping its
        // indentation and bullet character; done-ness is not inherited.
        LineStyle::Task { .. } => {
            let indent = leading_indent(marker);
            Some(format!("{}[ ] ", &marker[..indent + 2]))
        }
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

/// The text a paste inserts on a quote line, so the pasted block stays inside the
/// quote. The marker is the line's own, and at_content_start says the caret sits
/// right after it. Every line the paste adds becomes a quote line: one already
/// quoted stays as it is, a blank one takes the bare marker, and any other takes
/// the marker in front. The last line always takes the full marker, since the caret
/// and the rest of the caret's line land on it. At the content start, a first line
/// that is already a quote drops its own marker, because the caret's line supplies
/// one. None when a single line has no marker to drop and goes in as it is.
pub fn quote_paste(marker: &str, at_content_start: bool, pasted: &str) -> Option<String> {
    let (first, rest) = match pasted.split_once('\n') {
        Some((first, rest)) => (first, Some(rest)),
        None => (pasted, None),
    };
    let first = match quote_marker(first) {
        Some(len) if at_content_start => &first[len..],
        _ => first,
    };
    let Some(rest) = rest else {
        return (first.len() < pasted.len()).then(|| first.to_string());
    };
    // Each added line grows by at most one marker, so this is the most the paste
    // can come to.
    let added = rest.bytes().filter(|&b| b == b'\n').count() + 1;
    let mut out = String::with_capacity(pasted.len() + added * marker.len());
    out.push_str(first);
    let mut lines = rest.split('\n').peekable();
    while let Some(line) = lines.next() {
        out.push('\n');
        let last = lines.peek().is_none();
        if quote_marker(line).is_some() {
            out.push_str(line);
        } else if !last && line.trim().is_empty() {
            out.push_str(marker.trim_end());
        } else {
            out.push_str(marker);
            out.push_str(line);
        }
    }
    Some(out)
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

/// The checked state of the `[ ]`, `[x]`, or `[X]` checkbox opening a bullet
/// item's content at byte `after` (just past the bullet's marker), making the
/// bullet a task item, or `None` when there is no such checkbox. The box needs
/// a following space, so `- [ ]x` and a plain `- [y] z` stay ordinary bullets.
/// The checkbox is 4 bytes, which the task item's marker length adds on top of
/// the bullet's.
fn task_box(line: &str, after: usize) -> Option<bool> {
    let rest = &line.as_bytes()[after..];
    if rest.first() != Some(&b'[') || rest.get(2) != Some(&b']') || rest.get(3) != Some(&b' ') {
        return None;
    }
    match rest.get(1)? {
        b' ' => Some(false),
        b'x' | b'X' => Some(true),
        _ => None,
    }
}

/// If `line` is an ordered item, the byte range of its digits and the full marker
/// length (both spanning any leading indentation). An ordered item is optional
/// indentation, one to nine digits, then `.` or `)`, then a required space (so
/// `1.foo` and a bare `1.` stay body text). One scan feeds both the classifier
/// (which wants the length) and renumbering (which wants the digits).
fn ordered_parts(line: &str) -> Option<((usize, usize), usize)> {
    let indent = leading_indent(line);
    let bytes = &line.as_bytes()[indent..];
    let digits = bytes.iter().take_while(|&&b| b.is_ascii_digit()).count();
    if digits == 0 || digits > 9 {
        return None;
    }
    match bytes.get(digits) {
        Some(b'.' | b')') if bytes.get(digits + 1) == Some(&b' ') => {
            Some(((indent, indent + digits), indent + digits + 2))
        }
        _ => None,
    }
}

/// The number of an ordered item together with the byte range of its digits
/// (which sit just past any indentation), or `None` when `line` is not an
/// ordered item: `"  12. x"` yields `(12, (2, 4))`. Drives the renumbering of
/// the items that follow when Enter inserts a new numbered item.
pub fn ordered_number(line: &str) -> Option<(u64, (usize, usize))> {
    let ((ds, de), _) = ordered_parts(line)?;
    let value = line[ds..de].parse().ok()?;
    Some((value, (ds, de)))
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
        | LineStyle::Task { .. }
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

/// The pixel size a fenced code block renders at, a step below the body `base`
/// (0.875x): monospace at the full body size reads noticeably larger than the
/// surrounding prose, so the block sits a touch smaller. Rounded to the nearest
/// pixel with the same integer-tenths style the heading scales use.
pub fn code_size(base: u32) -> u32 {
    (base * 7 + 4) / 8
}

/// Every distinct pixel size the styles use at body size `base`, so the font set
/// can be opened up front. Body first, then each heading size, then the code size,
/// each added only if not already seen.
pub fn all_sizes(base: u32) -> Vec<u32> {
    let mut sizes = vec![font_size(LineStyle::Body, base)];
    for level in 1..=6u8 {
        let size = font_size(LineStyle::Heading(level), base);
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    let code = code_size(base);
    if !sizes.contains(&code) {
        sizes.push(code);
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
// Three layers, in CommonMark's precedence order (links and code bind tighter
// than emphasis):
//
//   1. Code spans (`` `code` ``): a run of N backticks opens a span the next run
//      of exactly N backticks closes; an unclosed run is literal. Delimiters
//      inside a code span (a `*`, say) are inert.
//   2. Inline links (`[text](dest)`): the `[`, `]`, and `(dest)` are markers; the
//      text styles as a link. The whole link is inert for emphasis, so a `*` or
//      `_` inside the text or the URL stays literal.
//   2b. Bare URLs: a `https://…` typed straight into the prose is a link with no
//      markup, so it has no markers and nothing to collapse, but it styles and
//      follows like one. It is inert for emphasis too, which keeps an `_` in a path
//      literal. Detection is [`crate::platform::link`], shared with the browser so
//      only a URL we would actually open is ever decorated.
//   3. Emphasis and strikethrough (`*`/`_`, `**`/`__`, `***`/`___`, `~~`): runs
//      outside code and links are paired opener-to-closer with a stack (innermost
//      first) to mark italic, bold, both, or struck. Whether a run can open or
//      close follows a simplified reading of CommonMark's flanking rules
//      (including the `_` intraword restriction, so `snake_case` stays literal).
//
// Deliberate simplifications: only single/double/triple `*`/`_` runs emphasize
// and only a `~~` pair strikes (other lengths stay literal, so a lone `~` in
// prose is safe), an opener matches only a closer of the same length (no
// partial-run consumption), strikethrough does not compose with bold/italic,
// links do not nest brackets and reference links (`[text][id]`) are unhandled,
// everything is single-line, and backslash escapes and the one-space stripping
// inside code spans are not handled. Notes rarely need the omitted cases.

/// Inline emphasis of a text run, selecting the face and color it renders with.
/// A closed set; [`crate::platform::freetype::FontStyle`] maps it to an actual face.
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
    /// The text of an inline link `[text](dest)`, drawn in the accent color and
    /// underlined. The brackets and `(dest)` are [`SpanKind::Marker`]s, hidden
    /// off the active line; only this run shows when the link is collapsed.
    Link,
    /// `~~struck~~`, drawn body weight with a line through its middle.
    Strikethrough,
}

/// What one inline span is: text to render with an [`Emphasis`], a delimiter
/// marker (a code span's backticks) that live-preview hides off the active line,
/// or a syntax-highlighted token inside a fenced code block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpanKind {
    Text(Emphasis),
    Marker,
    /// A token inside a fenced code block, colored by its [`CodeToken`] class and
    /// drawn in the code face (see [`crate::syntax`]).
    Code(CodeToken),
}

/// The lexical class of a code-block token, mapped to a color by the painter. A
/// language the lexer does not know yields a single [`CodeToken::Plain`] span, so
/// the block still reads as flat code in the body foreground.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CodeToken {
    Plain,
    Keyword,
    Str,
    Comment,
    Number,
}

/// One inline span: a byte range within the scanned content, and what it is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub kind: SpanKind,
}

// Per-byte inline-markup flags, packed one `u16` per content byte so the inline
// parser needs a single line-length buffer instead of a dozen bool masks.
const F_CODE_MARKER: u16 = 1 << 0;
const F_CODE_CONTENT: u16 = 1 << 1;
const F_IN_CODE: u16 = 1 << 2;
const F_LINK_MARKER: u16 = 1 << 3;
const F_LINK_TEXT: u16 = 1 << 4;
const F_IN_LINK: u16 = 1 << 5;
const F_EM_MARKER: u16 = 1 << 6;
const F_BOLD: u16 = 1 << 7;
const F_ITALIC: u16 = 1 << 8;
const F_STRIKE: u16 = 1 << 9;
/// A byte inside a code span or a link is inert for emphasis.
const F_INERT: u16 = F_IN_CODE | F_IN_LINK;

/// Reusable scratch for [`inline_spans_append`], so parsing a line's inline
/// markup allocates nothing once the buffers have grown. `flags` is the packed
/// per-byte mask; the rest are the parser's small working lists. The layout pools
/// one of these across a build.
#[derive(Default)]
pub struct InlineScratch {
    flags: Vec<u16>,
    links: Vec<LinkSpan>,
    bare: Vec<Range<usize>>,
    delims: Vec<Delim>,
    pairs: Vec<(usize, usize)>,
    open_stack: Vec<usize>,
}

/// Split `content` into a contiguous list of inline spans covering `[0, len)`,
/// recognizing code spans, inline links, and `*`/`_`/`~~` emphasis. Text between
/// and around markup becomes `Text(..)` spans, so the result has no gaps. Empty
/// input yields no spans. The owning wrapper over [`inline_spans_append`]; the
/// layout reads spans through the appending form to pool the buffers, so this is
/// used only by tests.
#[cfg(test)]
pub fn inline_spans(content: &str) -> Vec<Span> {
    let mut scratch = InlineScratch::default();
    let mut out = Vec::new();
    inline_spans_append(content, &mut scratch, &mut out);
    out
}

/// Append `content`'s inline spans, content-relative and tiling `[0, len)`, to
/// `out` without clearing it, so a caller can reuse a pooled buffer across lines.
/// The owning wrapper, inline_spans, is for tests.
pub fn inline_spans_append(content: &str, scratch: &mut InlineScratch, out: &mut Vec<Span>) {
    let bytes = content.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return;
    }

    // Fast path: content with none of the inline-markup bytes has no code span,
    // emphasis, strikethrough, or link, so every byte resolves to `Regular` and the
    // whole thing coalesces to a single plain run. Pushing it directly skips the
    // dozen per-line mask allocations below, and this is the shape of ordinary
    // prose, most lines of most documents. Every scheme a bare URL can carry holds a
    // `:`, so that byte is what keeps one off this path.
    if !bytes
        .iter()
        .any(|&b| matches!(b, b'`' | b'*' | b'_' | b'~' | b'[' | b':'))
    {
        out.push(Span {
            start: 0,
            end: n,
            kind: SpanKind::Text(Emphasis::Regular),
        });
        return;
    }

    // Split the scratch so each layer borrows the buffers it needs independently;
    // all are cleared and refilled, so a warmed pool allocates nothing here.
    let InlineScratch {
        flags,
        links,
        bare,
        delims,
        pairs,
        open_stack,
    } = scratch;
    flags.clear();
    flags.resize(n, 0);

    // Code spans first: they bind tighter than emphasis, and their bytes are inert
    // for the `*`/`_`/link markup below.
    code_masks_into(content, flags);

    // Then inline links `[text](dest)`. The `[`, `]`, and `(dest)` are markers
    // (hidden off the active line like any other markup); the text styles as a
    // link. The whole link is inert for emphasis. An image `![..](..)` is left
    // alone (its `!` guards it) and reference links `[text][id]` are not handled.
    links.clear();
    link_spans_into(bytes, flags, links);
    for l in &*links {
        flags[l.open] |= F_LINK_MARKER; // `[`
        for f in &mut flags[l.open + 1..l.rb] {
            *f |= F_LINK_TEXT; // text
        }
        for f in &mut flags[l.rb..=l.rp] {
            *f |= F_LINK_MARKER; // `]` .. `)`
        }
        for f in &mut flags[l.open..=l.rp] {
            *f |= F_IN_LINK;
        }
    }

    // Then a URL typed straight into the prose. It carries no markup, so it has no
    // markers and nothing collapses: the whole span is link text, and it goes inert so
    // an `_` in a path stays literal. Runs after the links because that is what makes
    // a URL sitting in a link's own `(dest)` already inert here.
    bare_urls_into(content, flags, bare);
    for r in &*bare {
        for f in &mut flags[r.start..r.end] {
            *f |= F_LINK_TEXT | F_IN_LINK;
        }
    }

    // Then emphasis and strikethrough. Pair `*`/`_`/`~~` delimiter runs that
    // lie outside both code and links, then flag the matched delimiters as markers
    // and stamp bold/italic/strike over the text between.
    delims.clear();
    emphasis_delims_into(content, flags, delims);
    pairs.clear();
    match_delims_into(delims, pairs, open_stack);
    for &(open, close) in &*pairs {
        for f in &mut flags[delims[open].start..delims[open].end] {
            *f |= F_EM_MARKER;
        }
        for f in &mut flags[delims[close].start..delims[close].end] {
            *f |= F_EM_MARKER;
        }
        let (text_start, text_end) = (delims[open].end, delims[close].start);
        if delims[open].ch == b'~' {
            for f in &mut flags[text_start..text_end] {
                *f |= F_STRIKE;
            }
        } else {
            // A run of 1 is italic, 2 is bold, 3 is both.
            let bold = delims[open].len != 1;
            let italic = delims[open].len != 2;
            for f in &mut flags[text_start..text_end] {
                if bold {
                    *f |= F_BOLD;
                }
                if italic {
                    *f |= F_ITALIC;
                }
            }
        }
    }

    // Resolve each byte to a single kind, then coalesce equal neighbors so the
    // spans tile the content with no gaps and no redundant splits. Markers win,
    // then code, then a link's text, then strikethrough, then plain emphasis.
    let kind_at = |b: usize| -> SpanKind {
        let fl = flags[b];
        if fl & (F_CODE_MARKER | F_EM_MARKER | F_LINK_MARKER) != 0 {
            SpanKind::Marker
        } else if fl & F_CODE_CONTENT != 0 {
            SpanKind::Text(Emphasis::Code)
        } else if fl & F_LINK_TEXT != 0 {
            SpanKind::Text(Emphasis::Link)
        } else if fl & F_STRIKE != 0 {
            SpanKind::Text(Emphasis::Strikethrough)
        } else {
            SpanKind::Text(match (fl & F_BOLD != 0, fl & F_ITALIC != 0) {
                (true, true) => Emphasis::BoldItalic,
                (true, false) => Emphasis::Bold,
                (false, true) => Emphasis::Italic,
                (false, false) => Emphasis::Regular,
            })
        }
    };
    let mut start = 0;
    let mut cur = kind_at(0);
    for b in 1..n {
        let kind = kind_at(b);
        if kind != cur {
            out.push(Span {
                start,
                end: b,
                kind: cur,
            });
            start = b;
            cur = kind;
        }
    }
    out.push(Span {
        start,
        end: n,
        kind: cur,
    });
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

/// Collect the emphasis and strikethrough delimiter runs of `content` into `out`,
/// skipping any byte flagged `F_INERT` in `flags` (inside a code span or a link).
/// An `*`/`_` run of one to three is kept; a `~` run is kept only when it is
/// exactly `~~` (so a lone `~` in prose stays literal). Other lengths are dropped.
fn emphasis_delims_into(content: &str, flags: &[u16], out: &mut Vec<Delim>) {
    let bytes = content.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let ch = bytes[i];
        if flags[i] & F_INERT != 0 || !matches!(ch, b'*' | b'_' | b'~') {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        while end < n && bytes[end] == ch && flags[end] & F_INERT == 0 {
            end += 1;
        }
        let len = end - start;
        let keep = if ch == b'~' {
            len == 2
        } else {
            (1..=3).contains(&len)
        };
        if keep {
            // Flanking is decided by the raw neighboring characters; `start` and
            // `end` sit on char boundaries (the run is ASCII `*`/`_`/`~`).
            let before = content[..start].chars().next_back();
            let after = content[end..].chars().next();
            let (can_open, can_close) = flanking(ch, before, after);
            out.push(Delim {
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
}

/// Pair delimiter runs into `(opener, closer)` index tuples, matching the same
/// character and length innermost-first with a stack. Unmatched runs are dropped
/// (they stay literal text). A run that can both open and close prefers closing.
fn match_delims_into(
    delims: &[Delim],
    pairs: &mut Vec<(usize, usize)>,
    open_stack: &mut Vec<usize>,
) {
    open_stack.clear();
    for (idx, delim) in delims.iter().enumerate() {
        if delim.can_close
            && let Some(pos) = open_stack
                .iter()
                .rposition(|&oi| delims[oi].ch == delim.ch && delims[oi].len == delim.len)
        {
            pairs.push((open_stack[pos], idx));
            // Drop the opener and any unmatched openers nested inside it.
            open_stack.truncate(pos);
            continue;
        }
        if delim.can_open {
            open_stack.push(idx);
        }
    }
}

/// Whether a delimiter run of `ch` can open and/or close emphasis, from the
/// characters immediately before and after it: a simplified reading of
/// CommonMark's left/right-flanking rules. A missing neighbor (the line edge)
/// counts as whitespace. The `_` form keeps the intraword restriction so an
/// underscore inside a word neither opens nor closes; `*` and `~` (strikethrough)
/// share the unrestricted form.
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

/// The index of the first `target` byte at or after `from` that is not inside a
/// code span, or `None`. Used to find a link's `]` and `)` while skipping any that
/// sit in inline code.
fn find_byte(bytes: &[u8], from: usize, target: u8, flags: &[u16]) -> Option<usize> {
    (from..bytes.len()).find(|&i| bytes[i] == target && flags[i] & F_IN_CODE == 0)
}

/// The index of the `)` that closes a link destination starting at from, the way
/// CommonMark reads one: a `<...>` destination ends at its `>`, and a bare one may
/// hold balanced parentheses (`wiki/Cure_(album)`) and backslash-escaped ones. A
/// quoted title after whitespace may hold any parenthesis. None when the
/// parentheses never balance, which leaves the text literal.
fn dest_close(bytes: &[u8], from: usize, flags: &[u16]) -> Option<usize> {
    let mut i = from;
    while matches!(bytes.get(i), Some(b' ' | b'\t')) {
        i += 1;
    }
    if bytes.get(i) == Some(&b'<') {
        let gt = find_byte(bytes, i + 1, b'>', flags)?;
        return find_byte(bytes, gt + 1, b')', flags);
    }
    let mut depth = 0usize;
    let mut title = None;
    while let Some(&b) = bytes.get(i) {
        if flags[i] & F_IN_CODE != 0 {
            i += 1;
            continue;
        }
        if let Some(quote) = title {
            if b == quote {
                title = None;
            }
            i += 1;
            continue;
        }
        match b {
            b'\\' => i += 1,
            b'"' | b'\''
                if depth == 0 && matches!(bytes.get(i.wrapping_sub(1)), Some(b' ' | b'\t')) =>
            {
                title = Some(b);
            }
            b'(' => depth += 1,
            b')' if depth == 0 => return Some(i),
            b')' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    None
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

/// Flag the code-span bytes of `content` into `flags`: the fence backticks
/// (`F_CODE_MARKER`), the span interior (`F_CODE_CONTENT`), and every byte inside
/// a span (`F_IN_CODE`, which keeps other markup inert). The single source so
/// [`inline_spans_append`] and [`links`] agree on where code lies.
fn code_masks_into(content: &str, flags: &mut [u16]) {
    let bytes = content.as_bytes();
    let n = bytes.len();
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
        for f in &mut flags[open..after_open] {
            *f |= F_CODE_MARKER | F_IN_CODE;
        }
        for f in &mut flags[after_open..close] {
            *f |= F_CODE_CONTENT | F_IN_CODE;
        }
        for f in &mut flags[close..after_close] {
            *f |= F_CODE_MARKER | F_IN_CODE;
        }
        i = after_close;
    }
}

/// The bracket indices of one inline link, into the scanned content.
struct LinkSpan {
    /// Index of the opening `[`. The text is `open + 1 .. rb`.
    open: usize,
    /// Index of the `]`; the `(` is always `rb + 1` and the dest is `rb + 2 .. rp`.
    rb: usize,
    /// Index of the closing `)`.
    rp: usize,
}

/// Locate the inline links in `bytes`, skipping any markup inside a code span
/// (flagged in `in_code`). A link is `[text](dest)` with non-empty text and
/// destination; an image `![..](..)` is skipped (its `!` guards it) and reference
/// links `[text][id]` are ignored (no `(` follows the `]`). The first `]` outside
/// code closes the text, so nested brackets in it are not handled; the destination
/// ends where dest_close says.
fn link_spans_into(bytes: &[u8], flags: &[u16], out: &mut Vec<LinkSpan>) {
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        if bytes[i] != b'[' || flags[i] & F_IN_CODE != 0 || (i > 0 && bytes[i - 1] == b'!') {
            i += 1;
            continue;
        }
        let open = i;
        let Some(rb) = find_byte(bytes, open + 1, b']', flags) else {
            i = open + 1;
            continue;
        };
        // A `(` must immediately follow the `]`, and a `)` must close the dest.
        if bytes.get(rb + 1) != Some(&b'(') {
            i = rb + 1;
            continue;
        }
        let Some(rp) = dest_close(bytes, rb + 2, flags) else {
            i = rb + 2;
            continue;
        };
        // Empty text or empty destination is not a link; leave the bytes literal.
        if rb == open + 1 || rp == rb + 2 {
            i = rp + 1;
            continue;
        }
        out.push(LinkSpan { open, rb, rp });
        i = rp + 1;
    }
}

/// Collect the bare URLs of `content` into `out`: the ones written with no markup at
/// all, which style and follow as links anyway (see [`crate::platform::link`]).
///
/// `flags` must already carry the code and link masks, because a URL is only bare
/// when nothing else already claims it: the one in a link's own `(dest)`, the one a
/// link uses as its text, and the one inside a code span are all `F_INERT` by now and
/// drop out here.
fn bare_urls_into(content: &str, flags: &[u16], out: &mut Vec<Range<usize>>) {
    out.clear();
    out.extend(
        crate::platform::link::urls(content)
            .filter(|r| flags[r.start..r.end].iter().all(|f| f & F_INERT == 0)),
    );
}

/// One link located in a line, as byte ranges into that line. Either an inline
/// `[text](dest)` or a bare URL, which is simply the link whose text *is* its
/// destination, so a caller follows both without telling them apart. See [`links`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Link {
    /// Byte range of the link text: between `[` and `]`, or the URL itself.
    pub text: (usize, usize),
    /// Byte range of the destination URL, with any `<...>` wrapper and `"title"`
    /// suffix already removed, so it is ready to hand to a browser.
    pub dest: (usize, usize),
}

/// The links in `content`, each with its text range and a cleaned destination, as
/// byte ranges into `content`. The renderer reaches links through [`inline_spans_append`];
/// this is the structured view the editor uses to follow the link the pointer clicks.
/// A `[x](y)` inside a code span is skipped, matching the renderer.
///
/// Threads the caller's pooled [`InlineScratch`] so the masking that makes code and
/// markup inert runs through the same layers the renderer uses (layers 1-2b of
/// [`inline_spans_append`]) instead of a parallel per-call allocation, which is what
/// keeps the two views agreeing on what a link is. Only `flags`, `links`, and `bare`
/// are touched; the returned `Vec` is fresh.
pub fn links(content: &str, scratch: &mut InlineScratch) -> Vec<Link> {
    let bytes = content.as_bytes();
    let InlineScratch {
        flags, links, bare, ..
    } = scratch;
    flags.clear();
    flags.resize(bytes.len(), 0);
    code_masks_into(content, flags);
    links.clear();
    link_spans_into(bytes, flags, links);
    // Only `F_IN_LINK` matters here: it is what tells the bare-URL pass that a URL in a
    // link's `(dest)`, or standing as its text, is already spoken for.
    for l in &*links {
        for f in &mut flags[l.open..=l.rp] {
            *f |= F_IN_LINK;
        }
    }
    bare_urls_into(content, flags, bare);
    let inline = links.iter().map(|l| Link {
        text: (l.open + 1, l.rb),
        dest: path_of_dest(content, l.rb + 2, l.rp),
    });
    let plain = bare.iter().map(|r| Link {
        text: (r.start, r.end),
        dest: (r.start, r.end),
    });
    inline.chain(plain).collect()
}

// --- Block images -----------------------------------------------------------

/// One image link `![alt](dest)` parsed from a line, as byte ranges into that
/// line. See [`standalone_image`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageLink {
    /// Byte range of the alt text, between `![` and `]`.
    pub alt: (usize, usize),
    /// Byte range of the destination path: the bytes the editor resolves to a
    /// file, with any `<...>` wrapper and `"title"` suffix already removed.
    pub path: (usize, usize),
}

/// If `line` is, ignoring surrounding whitespace, a single image and nothing else,
/// the byte ranges of its alt text and destination path. Both the CommonMark form
/// `![alt](dest)` and the wikilink embed `![[target]]` (see [`wikilink_image`]) are
/// recognized. Such a line renders as its picture in place; an image mixed with
/// other text on the line (an inline image) is not matched and stays source for now.
///
/// `dest` may wrap its path in `<...>` (which lets the path hold spaces) and may
/// carry a `"title"`/`'title'` after the path; both are stripped to leave the
/// path. A path containing a literal `)` is not handled: the first `)` closes the
/// link.
pub fn standalone_image(line: &str) -> Option<ImageLink> {
    if let Some(embed) = wikilink_image(line) {
        return Some(embed);
    }
    let open = line.find("![")?;
    // Only whitespace may precede the image, so this is a block on its own line.
    if !line[..open].trim().is_empty() {
        return None;
    }
    let alt_start = open + 2;
    let alt_end = alt_start + line[alt_start..].find(']')?;
    // A `(` must immediately follow the `]`.
    if line.as_bytes().get(alt_end + 1) != Some(&b'(') {
        return None;
    }
    let dest_start = alt_end + 2;
    let dest_end = dest_start + line.get(dest_start..)?.find(')')?;
    // Only whitespace may follow the closing `)`.
    if !line[dest_end + 1..].trim().is_empty() {
        return None;
    }
    let (path_start, path_end) = path_of_dest(line, dest_start, dest_end);
    (path_start < path_end).then_some(ImageLink {
        alt: (alt_start, alt_end),
        path: (path_start, path_end),
    })
}

/// If `line` is a single wikilink embed `![[target]]` (optionally `![[target|label]]`),
/// ignoring surrounding whitespace, the byte ranges of its target and label. The
/// target names a vault file that [`crate::render::image::ImageStore`] resolves by
/// name against the note's directory, the vault root, and its attachment folders, so
/// a bare `![[Pasted image 20250101.png]]` finds its picture wherever the vault filed
/// it. A `|label` (a display width or caption) is kept as the alt range.
///
/// Recognition is blind to the target's kind: a note embed like `![[Some Note]]`
/// matches here too, then fails to decode as an image and falls back to its source
/// line, which is the right outcome until note transclusion exists.
fn wikilink_image(line: &str) -> Option<ImageLink> {
    let open = line.find("![[")?;
    // Only whitespace may precede the embed, so this is a block on its own line.
    if !line[..open].trim().is_empty() {
        return None;
    }
    let inner_start = open + 3;
    let inner_end = inner_start + line[inner_start..].find("]]")?;
    // Only whitespace may follow the closing `]]`.
    if !line[inner_end + 2..].trim().is_empty() {
        return None;
    }
    // An optional `|label` (a display size or caption) trails the target.
    let (path, alt) = match line[inner_start..inner_end].find('|') {
        Some(off) => {
            let bar = inner_start + off;
            (
                trim_range(line, inner_start, bar),
                trim_range(line, bar + 1, inner_end),
            )
        }
        None => (
            trim_range(line, inner_start, inner_end),
            (inner_end, inner_end),
        ),
    };
    (path.0 < path.1).then_some(ImageLink { alt, path })
}

/// The subrange `line[start..end]` shrunk past any leading and trailing ASCII
/// whitespace, as absolute byte offsets into `line`. An all-whitespace range
/// collapses to empty at its trimmed start, never inverts.
fn trim_range(line: &str, start: usize, end: usize) -> (usize, usize) {
    let seg = &line[start..end];
    let s = start + (seg.len() - seg.trim_start().len());
    (s, s + seg.trim().len())
}

/// The path's byte range within a link destination `line[start..end]`, dropping
/// surrounding whitespace, an optional `<...>` wrapper, and a trailing
/// `"title"`/`'title'`. A bare path keeps any interior spaces (more forgiving than
/// CommonMark, which would need the `<...>` form), so local filenames with spaces
/// just work.
fn path_of_dest(line: &str, start: usize, end: usize) -> (usize, usize) {
    let dest = &line[start..end];
    let s = start + (dest.len() - dest.trim_start().len());
    let e = s + line[s..end].trim_end().len();
    // A <...> wrapper: the path is whatever sits inside, spaces and all.
    if line.as_bytes().get(s) == Some(&b'<') && e > s + 1 && line.as_bytes()[e - 1] == b'>' {
        return (s + 1, e - 1);
    }
    // Otherwise a "title"/'title' may trail the path after whitespace; the path
    // ends at the whitespace that introduces the opening quote.
    let inner = &line.as_bytes()[s..e];
    for i in 0..inner.len() {
        if (inner[i] == b' ' || inner[i] == b'\t')
            && matches!(
                line[s + i..e].trim_start().bytes().next(),
                Some(b'"' | b'\'')
            )
        {
            return (s, s + line[s..s + i].trim_end().len());
        }
    }
    (s, e)
}

// --- Tables -----------------------------------------------------------------
//
// A GFM pipe table is a cross-line block, like a fenced code block: a header row,
// a delimiter row that both confirms the header and sets each column's alignment,
// then zero or more body rows. The delimiter row is the signal, so a table begins
// only where a header line sits directly above a valid delimiter line with the
// same column count. [`crate::layout`] gathers the block and lays it out as a
// grid; this layer supplies the pure recognition and cell splitting.
//
// Deliberate simplifications: a body row is any following line holding an
// unescaped pipe (a blank or pipeless line ends the table), cells split on
// unescaped pipes only (`\|` stays literal), and a bare `---` alignment reads as
// left (GitHub renders an un-colonned column left-aligned).

/// The horizontal alignment of a table column, read from its delimiter cell:
/// `:---` is left, `:-:` center, `---:` right, and a bare `---` left.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Align {
    Left,
    Center,
    Right,
}

/// Whether `bytes` hold a `|` that is not backslash-escaped, the mark of a table
/// row. A run of backslashes toggles the escape, so `\|` is escaped but `\\|` is
/// a live pipe. Takes an iterator so a caller with the line in hand and one
/// reading it out of a rope share the one state machine.
pub fn has_unescaped_pipe_bytes(bytes: impl Iterator<Item = u8>) -> bool {
    let mut escaped = false;
    for b in bytes {
        match b {
            b'\\' => escaped = !escaped,
            b'|' if !escaped => return true,
            _ => escaped = false,
        }
    }
    false
}

/// [`has_unescaped_pipe_bytes`] over a line already in memory.
fn has_unescaped_pipe(line: &str) -> bool {
    has_unescaped_pipe_bytes(line.bytes())
}

/// Whether `line` can continue a table body: it carries an unescaped pipe. A
/// blank line (no pipe) or a pipeless paragraph line ends the table.
pub fn is_table_row(line: &str) -> bool {
    has_unescaped_pipe(line)
}

/// Split table row `line` into the trimmed byte ranges of its cells, into `out`
/// (cleared first). Cells divide on unescaped `|`; a `\|` stays inside a cell.
/// Optional outer pipes contribute no empty end cells. Ranges index into `line`.
pub fn table_row_cells(line: &str, out: &mut Vec<(usize, usize)>) {
    out.clear();
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut start = 0;
    let mut escaped = false;
    let mut i = 0;
    while i < n {
        match bytes[i] {
            b'\\' => escaped = !escaped,
            b'|' if !escaped => {
                out.push(trim_span(line, start, i));
                start = i + 1;
            }
            _ => escaped = false,
        }
        i += 1;
    }
    out.push(trim_span(line, start, n));
    // Optional outer pipes leave an empty cell at each end; drop them.
    if out.first().is_some_and(|&(s, e)| s == e) {
        out.remove(0);
    }
    if out.last().is_some_and(|&(s, e)| s == e) {
        out.pop();
    }
}

/// The byte range `[start, end)` of `line` with surrounding ASCII whitespace
/// trimmed off, staying in the line's own coordinates so a cell maps back to its
/// source bytes.
fn trim_span(line: &str, start: usize, end: usize) -> (usize, usize) {
    let seg = &line[start..end];
    let s = start + (seg.len() - seg.trim_start().len());
    let e = s + line[s..end].trim_end().len();
    (s, e)
}

/// The alignment of one delimiter cell (already trimmed): a run of `-` optionally
/// flanked by `:`, or `None` if it holds anything else. `-` alone and `:-` are
/// left, `-:` right, `:-:` center.
fn delimiter_cell(cell: &str) -> Option<Align> {
    let c = cell.as_bytes();
    if c.is_empty() {
        return None;
    }
    let left = c[0] == b':';
    let right = c[c.len() - 1] == b':';
    let start = left as usize;
    let end = c.len() - right as usize;
    if start >= end || c[start..end].iter().any(|&b| b != b'-') {
        return None;
    }
    Some(match (left, right) {
        (true, true) => Align::Center,
        (false, true) => Align::Right,
        _ => Align::Left,
    })
}

/// The per-column alignments if `line` is a table delimiter row (`| --- | :-: |
/// ---: |`), else `None`. A delimiter row holds only `|`, `-`, `:`, and spaces,
/// carries at least one `|` (so a bare `---`, a thematic break, is never taken
/// for a one-column table) and one `-`, and every cell is a valid delimiter cell.
/// Outer pipes are optional.
pub fn table_delimiter(line: &str) -> Option<Vec<Align>> {
    let trimmed = line.trim();
    // Fast reject before allocating: prose fails the byte-set test at once.
    if !trimmed.contains('|')
        || !trimmed
            .bytes()
            .all(|b| matches!(b, b'|' | b'-' | b':' | b' ' | b'\t'))
        || !trimmed.bytes().any(|b| b == b'-')
    {
        return None;
    }
    let mut cells = Vec::new();
    table_row_cells(trimmed, &mut cells);
    if cells.is_empty() {
        return None;
    }
    let mut aligns = Vec::with_capacity(cells.len());
    for &(s, e) in &cells {
        aligns.push(delimiter_cell(&trimmed[s..e])?);
    }
    Some(aligns)
}

/// The column alignments if logical line `header` and the line directly below it,
/// `delim`, open a table: `delim` is a delimiter row, `header` carries an
/// unescaped pipe, and the two split into the same number of cells (GFM requires
/// the counts match). This is the signal that a table block begins at `header`.
pub fn table_head(header: &str, delim: &str) -> Option<Vec<Align>> {
    if !has_unescaped_pipe(header) {
        return None;
    }
    let aligns = table_delimiter(delim)?;
    let mut header_cells = Vec::new();
    table_row_cells(header, &mut header_cells);
    (header_cells.len() == aligns.len()).then_some(aligns)
}

#[cfg(test)]
mod tests {
    use crate::markdown::*;

    /// The marker-length half of [`classify`], kept terse for the assertions.
    fn marker_len(line: &str) -> usize {
        classify(line).1
    }

    /// Whether `line` keeps its block marker off the cursor's line, classified
    /// the way [`crate::layout`] does it.
    fn persists(line: &str) -> bool {
        let (style, marker) = classify(line);
        block_marker_persists(line, style, marker)
    }

    #[test]
    fn recognizes_each_heading_level() {
        for level in 1..=6u8 {
            let line = format!("{} heading", "#".repeat(level as usize));
            // Level plus marker length (the `#` run and its trailing space).
            assert_eq!(heading(&line), Some((level, level as usize + 1)));
            assert_eq!(style_of(&line), LineStyle::Heading(level));
        }
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        assert_eq!(heading("####### too deep"), None);
        assert_eq!(style_of("####### too deep"), LineStyle::Body);
    }

    #[test]
    fn hash_without_a_space_is_not_a_heading() {
        assert_eq!(heading("#tag"), None);
        assert_eq!(heading("##bold"), None);
    }

    #[test]
    fn bare_and_space_only_markers_are_empty_headings() {
        // No space after the hashes, so the marker is just the hashes.
        assert_eq!(heading("#"), Some((1, 1)));
        assert_eq!(heading("###"), Some((3, 3)));
        // A trailing space is part of the marker.
        assert_eq!(heading("# "), Some((1, 2)));
    }

    #[test]
    fn leading_space_disqualifies_a_heading() {
        assert_eq!(heading(" # indented"), None);
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

    #[test]
    fn code_size_is_a_step_below_body_and_is_opened() {
        // 0.875x, rounded to the nearest pixel: 14 -> 12, 16 -> 14, 32 -> 28.
        assert_eq!(code_size(14), 12);
        assert_eq!(code_size(16), 14);
        assert_eq!(code_size(32), 28);
        assert!(code_size(32) < 32, "code sits below body");
        assert!(
            all_sizes(14).contains(&code_size(14)),
            "the code face size is opened up front"
        );
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
        // A `#` line is a heading, not a list, and a heading with text hides its
        // marker.
        assert_eq!(style_of("# h"), LineStyle::Heading(1));
        assert!(!persists("# h"));
        assert!(!persists("plain"));
        // Lists and quotes keep their markers on screen.
        for line in ["- item", "1. item", "> quoted"] {
            assert!(persists(line), "{line:?} persists");
            assert_eq!(font_size(style_of(line), 32), 32, "{line:?} is body size");
        }
    }

    #[test]
    fn an_empty_headings_marker_persists() {
        // Nothing but the `#`s on the line: hiding them would render the line
        // blank, so they stay on screen like a bullet's disc does.
        for line in ["#", "##", "###### ", "##   "] {
            assert!(persists(line), "{line:?} keeps its marker");
        }
        // The first character of text is enough to earn the bare headline.
        for line in ["# a", "##  x  ", "#### -"] {
            assert!(!persists(line), "{line:?} hides its marker");
        }
    }

    #[test]
    fn a_thematic_break_does_not_persist_its_marker() {
        // A rule collapses to a drawn line, which is visible on its own; the
        // dashes must not linger next to it.
        assert!(!persists("---"));
        assert!(!persists("- - -"));
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

    // --- Task lists ---------------------------------------------------------

    #[test]
    fn recognizes_task_items_with_each_bullet_and_case() {
        for marker in ["- ", "+ ", "* "] {
            let unchecked = format!("{marker}[ ] todo");
            assert_eq!(
                style_of(&unchecked),
                LineStyle::Task { checked: false },
                "{unchecked:?}"
            );
            assert_eq!(marker_len(&unchecked), 6, "{unchecked:?}");
        }
        for check in ["x", "X"] {
            let done = format!("- [{check}] done");
            assert_eq!(
                style_of(&done),
                LineStyle::Task { checked: true },
                "{done:?}"
            );
            assert_eq!(marker_len(&done), 6, "{done:?}");
        }
    }

    #[test]
    fn recognizes_indented_task_items() {
        assert_eq!(
            style_of("  - [ ] sub"),
            LineStyle::Task { checked: false },
            "indentation nests a task like any list item"
        );
        assert_eq!(marker_len("  - [x] sub"), 8);
        assert_eq!(leading_indent("  - [x] sub"), 2);
    }

    #[test]
    fn near_miss_checkboxes_stay_bullets_or_body() {
        // No space after the box, an empty box pair, an unknown check char, or a
        // box past other content: a plain bullet whose text happens to hold
        // brackets. No bullet at all stays body.
        for line in ["- [ ]x", "- []", "- [y] z", "- a [ ] b", "- [x]"] {
            assert_eq!(style_of(line), LineStyle::Bullet, "{line:?}");
            assert_eq!(marker_len(line), 2, "{line:?}");
        }
        assert_eq!(style_of("[ ] x"), LineStyle::Body);
        assert_eq!(style_of("-[ ] x"), LineStyle::Body);
    }

    #[test]
    fn an_empty_task_item_is_still_a_task() {
        // The Enter continuation leaves `- [ ] ` (trailing space, no content);
        // it must classify as a task so a second Enter clears the marker.
        assert_eq!(style_of("- [ ] "), LineStyle::Task { checked: false });
        assert_eq!(marker_len("- [ ] "), 6);
    }

    #[test]
    fn a_task_is_body_sized_and_its_marker_persists() {
        for line in ["- [ ] todo", "- [x] done"] {
            assert!(persists(line), "{line:?} persists");
            assert_eq!(font_size(style_of(line), 32), 32, "{line:?} is body size");
        }
    }

    #[test]
    fn next_item_marker_continues_a_task_unchecked() {
        assert_eq!(next_item_marker("- [ ] a").as_deref(), Some("- [ ] "));
        assert_eq!(
            next_item_marker("- [x] done").as_deref(),
            Some("- [ ] "),
            "a finished item's successor starts fresh"
        );
        assert_eq!(
            next_item_marker("  + [X] b").as_deref(),
            Some("  + [ ] "),
            "indentation and the bullet character carry over"
        );
    }

    #[test]
    fn ordered_number_reports_the_value_and_digit_range() {
        assert_eq!(ordered_number("1. a"), Some((1, (0, 1))));
        assert_eq!(ordered_number("  12) b"), Some((12, (2, 4))));
        for line in ["- a", "- [ ] a", "plain", "1.a", "1."] {
            assert_eq!(ordered_number(line), None, "{line:?}");
        }
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
            "[a](b)",
            "see [x](y) end",
            "[*a*](b_c)",
            "~~x~~",
            "~~a~~ b",
            "~lone",
            "[a](b) and ~~c~~",
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

    // --- Inline links -----------------------------------------------------

    #[test]
    fn a_link_splits_into_markers_and_link_text() {
        // "[a](b)": '[' marker, "a" link text, "](b)" marker.
        assert_eq!(
            spans("[a](b)"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 2, SpanKind::Text(Emphasis::Link)),
                (2, 6, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn a_link_inside_prose_keeps_its_surroundings() {
        assert_eq!(
            spans("see [docs](u) now"),
            vec![
                (0, 4, SpanKind::Text(Emphasis::Regular)),
                (4, 5, SpanKind::Marker),
                (5, 9, SpanKind::Text(Emphasis::Link)),
                (9, 13, SpanKind::Marker),
                (13, 17, SpanKind::Text(Emphasis::Regular)),
            ]
        );
    }

    #[test]
    fn link_text_and_url_are_inert_for_emphasis() {
        // A `*` in the text or a `_` in the URL stays literal: the text is one
        // Link run and the destination one marker run, with no emphasis splits.
        assert_eq!(
            spans("[*a*](b_c)"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 4, SpanKind::Text(Emphasis::Link)),
                (4, 10, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn an_image_is_not_treated_as_a_link() {
        // The `!` guards `![alt](url)`; inline images stay literal source.
        assert_eq!(
            spans("![a](b)"),
            vec![(0, 7, SpanKind::Text(Emphasis::Regular))]
        );
    }

    #[test]
    fn malformed_links_stay_literal() {
        // `[id]` with no destination (a bare bracket pair or a reference link).
        assert_eq!(
            spans("[id]"),
            vec![(0, 4, SpanKind::Text(Emphasis::Regular))]
        );
        // A space between `]` and `(` breaks the link.
        assert_eq!(
            spans("[x] (y)"),
            vec![(0, 7, SpanKind::Text(Emphasis::Regular))]
        );
        // Empty text or empty destination is not a link.
        assert_eq!(
            spans("[](y)"),
            vec![(0, 5, SpanKind::Text(Emphasis::Regular))]
        );
        assert_eq!(
            spans("[x]()"),
            vec![(0, 5, SpanKind::Text(Emphasis::Regular))]
        );
    }

    /// `(text, cleaned dest)` pairs for every link found, sliced back out of `s`.
    fn link_list(s: &str) -> Vec<(&str, &str)> {
        let mut scratch = InlineScratch::default();
        links(s, &mut scratch)
            .into_iter()
            .map(|l| (&s[l.text.0..l.text.1], &s[l.dest.0..l.dest.1]))
            .collect()
    }

    #[test]
    fn links_returns_text_and_destination() {
        assert_eq!(link_list("[a](b)"), vec![("a", "b")]);
        assert_eq!(
            link_list("see [docs](http://x) and [more](http://y) now"),
            vec![("docs", "http://x"), ("more", "http://y")]
        );
    }

    #[test]
    fn links_clean_the_destination() {
        // A title and an angle-bracket wrapper are stripped, matching images.
        assert_eq!(
            link_list(r#"[a](http://x "title")"#),
            vec![("a", "http://x")]
        );
        assert_eq!(link_list("[a](<b c>)"), vec![("a", "b c")]);
    }

    #[test]
    fn a_destination_holds_balanced_parentheses() {
        // CommonMark lets a bare destination nest parentheses, as Wikipedia titles do.
        assert_eq!(
            link_list("[Cure](https://en.wikipedia.org/wiki/Cure_(album)) rocks"),
            vec![("Cure", "https://en.wikipedia.org/wiki/Cure_(album)")]
        );
        assert_eq!(link_list("[a](b(c(d)e)f)"), vec![("a", "b(c(d)e)f")]);
        // An escaped parenthesis does not count, a quoted title may hold one, and a
        // <...> destination ends at its >.
        assert_eq!(link_list(r"[a](b\)c)"), vec![("a", r"b\)c")]);
        assert_eq!(
            link_list(r#"[a](http://x "see (note)")"#),
            vec![("a", "http://x")]
        );
        assert_eq!(link_list("[a](<b)c>)"), vec![("a", "b)c")]);
        // Parentheses after the link are prose.
        assert_eq!(link_list("[a](b) (aside)"), vec![("a", "b")]);
        // A destination left unbalanced is not a link.
        assert!(link_list("[a](b(c)").is_empty());
    }

    #[test]
    fn links_skip_images_code_and_malformed() {
        assert!(link_list("![a](b)").is_empty(), "image, not a link");
        assert!(link_list("`[a](b)`").is_empty(), "inside a code span");
        assert!(link_list("[a]").is_empty(), "no destination");
        assert!(link_list("[a] (b)").is_empty(), "gap before the dest");
        assert!(link_list("[](b)").is_empty(), "empty text");
    }

    // --- Bare URLs --------------------------------------------------------

    #[test]
    fn a_bare_url_styles_as_link_text_with_no_markers() {
        // Nothing is markup, so nothing collapses: the URL is one Link run, and the
        // prose around it is untouched.
        assert_eq!(
            spans("see https://x.com/a now"),
            vec![
                (0, 4, SpanKind::Text(Emphasis::Regular)),
                (4, 19, SpanKind::Text(Emphasis::Link)),
                (19, 23, SpanKind::Text(Emphasis::Regular)),
            ]
        );
        assert_eq!(
            spans("https://x.com"),
            vec![(0, 13, SpanKind::Text(Emphasis::Link))]
        );
    }

    #[test]
    fn a_bare_url_is_a_link_whose_text_is_its_destination() {
        // The editor follows it through the same `Link` an inline one produces, so
        // the click path never has to tell the two apart.
        assert_eq!(
            link_list("see https://x.com/a now"),
            vec![("https://x.com/a", "https://x.com/a")]
        );
        assert_eq!(
            link_list("mail me at mailto:a@b.com"),
            vec![("mailto:a@b.com", "mailto:a@b.com")]
        );
    }

    #[test]
    fn a_bare_url_is_inert_for_emphasis() {
        // The `_`s in the path pair into italics if the URL is not inert, which would
        // both mis-style it and (worse) cut the link short of its real end.
        assert_eq!(
            spans("https://x.com/a_b_c"),
            vec![(0, 19, SpanKind::Text(Emphasis::Link))]
        );
        assert_eq!(
            link_list("https://x.com/a_b_c"),
            vec![("https://x.com/a_b_c", "https://x.com/a_b_c")]
        );
    }

    #[test]
    fn a_url_already_inside_markup_is_not_claimed_a_second_time() {
        // In an inline link the URL is the `(dest)`, already a marker; detecting it
        // again would style the collapsed markup as link text and report the link
        // twice. Same for a URL standing as the link's own text, and for one quoted
        // inside a code span.
        assert_eq!(
            spans("[docs](https://x.com)"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 5, SpanKind::Text(Emphasis::Link)),
                (5, 21, SpanKind::Marker),
            ]
        );
        assert_eq!(
            link_list("[docs](https://x.com)"),
            vec![("docs", "https://x.com")]
        );
        assert_eq!(
            link_list("[https://x.com](https://y.com)"),
            vec![("https://x.com", "https://y.com")]
        );
        assert_eq!(
            spans("`https://x.com`"),
            vec![
                (0, 1, SpanKind::Marker),
                (1, 14, SpanKind::Text(Emphasis::Code)),
                (14, 15, SpanKind::Marker),
            ]
        );
        assert!(link_list("`https://x.com`").is_empty(), "inside code");
    }

    #[test]
    fn prose_that_only_looks_like_a_url_stays_prose() {
        // The fast path is keyed on `:`, so a colon in ordinary prose reaches the full
        // scan and must still come out as one plain run. A scheme we refuse to open is
        // never decorated, and neither is a bare host.
        for line in [
            "Note: this is a note",
            "javascript:alert(1)",
            "ftp://example.com/a",
            "see example.com/a",
            "https:// alone",
        ] {
            assert_eq!(
                spans(line),
                vec![(0, line.len(), SpanKind::Text(Emphasis::Regular))],
                "{line:?}"
            );
            assert!(link_list(line).is_empty(), "{line:?}");
        }
    }

    #[test]
    fn a_url_gives_trailing_punctuation_back_to_the_sentence() {
        // The extent rules live in `platform::link`; this pins that a line actually
        // gets them, so the sentence's period never renders underlined.
        assert_eq!(
            spans("go to https://x.com/a."),
            vec![
                (0, 6, SpanKind::Text(Emphasis::Regular)),
                (6, 21, SpanKind::Text(Emphasis::Link)),
                (21, 22, SpanKind::Text(Emphasis::Regular)),
            ]
        );
    }

    #[test]
    fn several_bare_urls_and_an_inline_link_coexist() {
        assert_eq!(
            link_list("[a](http://x) then https://y.com and https://z.com"),
            vec![
                ("a", "http://x"),
                ("https://y.com", "https://y.com"),
                ("https://z.com", "https://z.com"),
            ]
        );
    }

    // --- Strikethrough ----------------------------------------------------

    #[test]
    fn a_double_tilde_pair_strikes_through() {
        assert_eq!(
            spans("~~x~~"),
            vec![
                (0, 2, SpanKind::Marker),
                (2, 3, SpanKind::Text(Emphasis::Strikethrough)),
                (3, 5, SpanKind::Marker),
            ]
        );
    }

    #[test]
    fn a_lone_or_unclosed_tilde_stays_literal() {
        // A single `~` never strikes, so "~5 min" and a file path are safe.
        assert_eq!(
            spans("~5 min"),
            vec![(0, 6, SpanKind::Text(Emphasis::Regular))]
        );
        // An unmatched `~~` is literal too.
        assert_eq!(
            spans("~~x"),
            vec![(0, 3, SpanKind::Text(Emphasis::Regular))]
        );
    }

    fn img(line: &str) -> Option<(&str, &str)> {
        standalone_image(line).map(|l| (&line[l.alt.0..l.alt.1], &line[l.path.0..l.path.1]))
    }

    #[test]
    fn recognizes_a_bare_standalone_image() {
        assert_eq!(img("![a cat](cat.png)"), Some(("a cat", "cat.png")));
        // Empty alt is fine.
        assert_eq!(img("![](cat.png)"), Some(("", "cat.png")));
    }

    #[test]
    fn allows_surrounding_whitespace_only() {
        assert_eq!(img("   ![x](a.png)  "), Some(("x", "a.png")));
    }

    #[test]
    fn an_image_mixed_with_text_is_not_a_block_image() {
        // Leading or trailing prose means it is an inline image, left as source.
        assert!(img("see ![x](a.png)").is_none());
        assert!(img("![x](a.png) here").is_none());
    }

    #[test]
    fn strips_a_title_keeping_the_path() {
        assert_eq!(img(r#"![x](a.png "my cat")"#), Some(("x", "a.png")));
        assert_eq!(img("![x](a.png 'my cat')"), Some(("x", "a.png")));
    }

    #[test]
    fn angle_brackets_wrap_a_path_with_spaces() {
        assert_eq!(img("![x](<my cat.png>)"), Some(("x", "my cat.png")));
    }

    #[test]
    fn a_bare_path_keeps_interior_spaces() {
        // More forgiving than CommonMark, which would require the <...> form.
        assert_eq!(img("![x](my cat.png)"), Some(("x", "my cat.png")));
    }

    #[test]
    fn rejects_malformed_or_empty_links() {
        assert!(img("![x]").is_none(), "no destination");
        assert!(img("![x] (a.png)").is_none(), "space between ] and (");
        assert!(img("![x]()").is_none(), "empty destination");
        assert!(img("plain text").is_none());
        assert!(img("[x](a.png)").is_none(), "a link, not an image");
    }

    #[test]
    fn recognizes_a_wikilink_embed() {
        // The target becomes the path; there is no separate alt.
        assert_eq!(
            img("![[Pasted image 20251219150959.png]]"),
            Some(("", "Pasted image 20251219150959.png"))
        );
    }

    #[test]
    fn a_wikilink_label_after_the_pipe_is_the_alt() {
        // A `|300` display-size suffix parses as the alt, path is the target.
        assert_eq!(img("![[cat.png|300]]"), Some(("300", "cat.png")));
        assert_eq!(
            img("![[ cat.png | a caption ]]"),
            Some(("a caption", "cat.png"))
        );
    }

    #[test]
    fn a_wikilink_embed_allows_surrounding_whitespace_only() {
        assert_eq!(img("   ![[cat.png]]  "), Some(("", "cat.png")));
        // Trailing prose makes it inline, left as source.
        assert!(img("![[cat.png]] and text").is_none());
        assert!(img("see ![[cat.png]]").is_none());
    }

    #[test]
    fn rejects_malformed_wikilink_embeds() {
        assert!(img("![[]]").is_none(), "empty target");
        assert!(img("![[   ]]").is_none(), "whitespace-only target");
        assert!(img("![[cat.png").is_none(), "unclosed");
        assert!(
            img("[[cat.png]]").is_none(),
            "a link embed marker needs the !"
        );
    }

    // --- Tables -------------------------------------------------------------

    /// The trimmed cell strings of a table row, terse enough to assert against.
    fn cells(line: &str) -> Vec<&str> {
        let mut out = Vec::new();
        table_row_cells(line, &mut out);
        out.iter().map(|&(s, e)| &line[s..e]).collect()
    }

    #[test]
    fn splits_rows_with_optional_outer_pipes() {
        assert_eq!(cells("| a | b | c |"), vec!["a", "b", "c"]);
        assert_eq!(cells("a | b | c"), vec!["a", "b", "c"]);
        assert_eq!(cells("| a | b"), vec!["a", "b"]);
        assert_eq!(cells("a | b |"), vec!["a", "b"]);
        // An empty interior cell survives; only the outer-pipe ends are dropped.
        assert_eq!(cells("| a |  | c |"), vec!["a", "", "c"]);
        assert_eq!(cells("| x |"), vec!["x"]);
    }

    #[test]
    fn an_escaped_pipe_stays_inside_a_cell() {
        // `\|` is literal, so "a|b" is one cell, not two.
        assert_eq!(cells(r"| a \| b | c |"), vec![r"a \| b", "c"]);
        assert!(!is_table_row(r"no pipes here"));
        assert!(!is_table_row(r"only escaped \| pipe"));
        assert!(is_table_row("a | b"));
    }

    #[test]
    fn recognizes_delimiter_rows_and_alignment() {
        assert_eq!(
            table_delimiter("| --- | :--- | :---: | ---: |"),
            Some(vec![Align::Left, Align::Left, Align::Center, Align::Right])
        );
        // Outer pipes optional, and a single dash per cell is enough.
        assert_eq!(
            table_delimiter("- | -"),
            Some(vec![Align::Left, Align::Left])
        );
        assert_eq!(table_delimiter("| :-: |"), Some(vec![Align::Center]));
    }

    #[test]
    fn a_thematic_break_is_not_a_delimiter_row() {
        // `---` carries no pipe, so it stays a thematic break, never a 1-col table.
        assert_eq!(table_delimiter("---"), None);
        assert_eq!(table_delimiter("- - -"), None);
        // A cell needs at least one dash; bare colons or other text disqualify it.
        assert_eq!(table_delimiter("| :: |"), None);
        assert_eq!(table_delimiter("| :- | x |"), None);
        assert_eq!(table_delimiter("| a | b |"), None);
    }

    #[test]
    fn table_head_needs_a_piped_header_and_matching_columns() {
        assert_eq!(
            table_head("| Name | Score |", "| --- | ---: |"),
            Some(vec![Align::Left, Align::Right])
        );
        // Header cell count must equal the delimiter's.
        assert_eq!(table_head("| a | b | c |", "| --- | --- |"), None);
        assert_eq!(table_head("| a |", "| --- | --- |"), None);
        // A header with no pipe never opens a table (guards stray prose above a
        // coincidental delimiter-looking line).
        assert_eq!(table_head("just prose", "| --- |"), None);
        // The second line must be a real delimiter row.
        assert_eq!(table_head("| a | b |", "| a | b |"), None);
    }
}
