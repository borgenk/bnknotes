//! Turning the flat text buffer into positioned visual lines, and mapping
//! between byte offsets and pixels.
//!
//! The buffer is one string with `'\n'` separators. This pass splits it on those
//! newlines into logical lines, then soft-wraps each to the available width with
//! a greedy word wrap (break after the last space that fits; an over-long word
//! breaks mid-word at a grapheme boundary). The result is a flat list of
//! [`VisualLine`]s, each with a baseline and a `hard` flag distinguishing a real
//! newline from a soft wrap. Everything geometric lives here so the editor model
//! stays free of font metrics:
//!
//! - [`Layout::caret`] maps a cursor offset to the caret bar's pixel rectangle.
//! - [`Layout::offset_at`] maps a click `(x, y)` back to a byte offset, snapping
//!   to grapheme cluster boundaries so the cursor never lands inside a cluster.
//! - [`Layout::offset_vertical`] moves up or down a visual line, keeping a goal x.
//! - [`Layout::selection_rects`] turns a byte range into the highlight rectangles
//!   covering it, one per visual line it spans.
//!
//! `x`/`y` are window-local pixels (the same space the pointer reports), so the
//! left text margin is baked into the math via `left`. A soft-wrap boundary is
//! one byte offset shared by two visual lines; the cursor there is given to the
//! later line (it shows at the start of the next row), while a cursor at a hard
//! newline stays at the end of its line.
//!
//! Lines are not all the same height: each logical line is classified by
//! [`crate::markdown`] and resolved to a pixel size, so headings render larger.
//! Every visual line therefore carries its own `size`, and the geometry methods
//! take a [`Fonts`] to look up the right face and metrics per line. Baselines
//! stack by each line's own height rather than a single global line height.
//!
//! Each visual line is also split into inline [`Run`]s: stretches of text with
//! one [`markdown::Emphasis`] (regular, code, bold, italic), each measured and
//! drawn in its own [`FontStyle`] face, and the markers that delimit them (a
//! heading's `#`s, a list/quote marker, a code span's backticks, emphasis
//! `*`/`_`). Markers live-preview: on a line the cursor has left they collapse
//! (`Run::visible` is false, zero width), drawing only the styled text; they
//! reappear when the cursor returns. The exception is a list or quote block
//! marker, which persists (it is the only sign the line is a list or quote). Wrapping
//! always measures the source text, so this collapse only shifts runs
//! horizontally within a line, never reflowing the document.
//!
//! A list or quote item also hangs its wrapped continuations under the text: each
//! visual line past the first is inset by the marker's width (`VisualLine::indent`)
//! so the column aligns. Once the cursor leaves the line the raw marker gives way
//! to a [`Gutter`] decoration, a bullet point `•` for a list or a bar for a quote,
//! always the same width as the source marker so the text never shifts; on the
//! cursor's own line the raw, editable marker shows instead.
//!
//! Fenced code blocks are the one construct whose classification crosses line
//! boundaries: a ``` line opens a block and a later ``` closes it, and every line
//! between is code, not markdown. So `build` carries a fence flag down the loop,
//! and a line inside a fence (`VisualLine::code_block`) skips the inline pass
//! entirely, rendering in the distinct code face on a full-width padded panel.
//! The ``` fence lines hide their backticks off the cursor's line, like a heading
//! hides its `#`, so an unfocused block reads without visible fences; the panel
//! still draws those rows, leaving blank top and bottom padding.

use crate::freetype::{Face, FontStyle, Fonts};
use crate::grapheme;
use crate::markdown;
use crate::render;
use crate::rope::Rope;

/// Extra highlight width drawn past a line's text when the selection includes
/// that line's trailing newline, hinting the break itself is selected.
const NEWLINE_PAD: i32 = 8;

/// Horizontal padding inside a fenced code block's panel: the code text (and the
/// ``` fences) is inset this many pixels from the panel's left edge, and wrapping
/// keeps the same gap on the right, so the tinted background frames the code
/// rather than butting against it.
const CODE_PAD_X: i32 = 10;

/// A small left indent for list items (bullet and ordered), so a list sits inset
/// from the body margin. It applies to every visual line of the item; wrapped
/// continuations add the marker's width on top for the hanging indent.
const LIST_INDENT: i32 = 14;

/// One on-screen line: the byte range of its text (excluding any trailing
/// `'\n'`), the y of its baseline, whether a hard newline follows it, the pixel
/// size it renders at, and its inline runs.
#[derive(Clone)]
pub struct VisualLine {
    pub start: usize,
    pub end: usize,
    pub baseline: i32,
    /// True if a `'\n'` immediately follows `end` (a hard break); false for a
    /// soft wrap or the final line of the buffer.
    pub hard: bool,
    /// The pixel size this line renders at; selects the face and metrics.
    pub size: u32,
    /// The line's inline runs in display order, tiling `[start, end)` (an empty
    /// line has none). Each is a styled stretch of text or a delimiter marker;
    /// a hidden marker (`!visible`) is collapsed to zero width.
    pub runs: Vec<Run>,
    /// Pixels the line's content (its runs, caret, and selection) is inset past
    /// `left`, for the hanging indent of a wrapped list or quote item: `0` on the
    /// marker-bearing first line (whose marker run fills the gutter inline) and on
    /// non-list lines; the marker's width on continuation lines, so wrapped text
    /// aligns under the first line's text rather than under the marker.
    pub indent: i32,
    /// How this line's left gutter is decorated beyond its runs once the cursor
    /// leaves it: a bullet glyph for a list, a bar for a quote. [`Gutter::None`]
    /// on ordinary lines and on the cursor's own line (which shows its raw marker).
    pub gutter: Gutter,
    /// True for a line inside a fenced code block (the ``` fences and the code
    /// between them). It is drawn on a full-width tinted panel and its text is
    /// not parsed as inline markdown.
    pub code_block: bool,
    /// The ATX heading level (1..=6) if this line is a heading, else `None`. The
    /// app colors heading text by level, and the pixel `size` already carries the
    /// heading's scale.
    pub heading: Option<u8>,
    /// True to draw a horizontal rule for this line: a thematic break (`---`)
    /// whose cursor has left, so its raw markers are collapsed. While the cursor is
    /// on the line this is false and the editable `---` shows instead.
    pub rule: bool,
}

/// The live-preview decoration of a visual line's left gutter, chosen once the
/// cursor leaves the line. On the cursor's own line the raw marker shows so it
/// stays editable, so this is [`Gutter::None`] there. A decoration always occupies
/// the same width as the source marker, so toggling it never shifts the text (the
/// no-reflow invariant the whole module preserves).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gutter {
    /// No decoration: body and heading lines, ordered items (their number stays
    /// put), and any list or quote line the cursor is currently on.
    None,
    /// Paint a bullet point `•` in place of a `-`/`+`/`*` marker's glyph.
    Bullet,
    /// Draw a blockquote bar down the gutter. `marker` is true on the first line,
    /// where the bar stands in for the `>` marker (whose glyphs are not drawn),
    /// and false on wrapped continuations, which have no marker run of their own.
    QuoteBar { marker: bool },
}

/// One inline run of a visual line: an absolute byte range, what it is, and
/// whether it is drawn. A marker run is hidden (`visible == false`) on a line
/// the cursor has left, the live-preview collapse; text runs are always visible.
#[derive(Clone, Copy)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    pub kind: markdown::SpanKind,
    pub visible: bool,
}

/// Map a span/run kind to the face style it renders in within the prose family.
/// Markers use the regular face; only bold/italic emphasis selects a variant.
/// Code is not handled here (it draws in the separate code family, see
/// [`run_face`]); a stray code kind falls back to regular.
fn font_style(kind: markdown::SpanKind) -> FontStyle {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Bold) => FontStyle::Bold,
        markdown::SpanKind::Text(markdown::Emphasis::Italic) => FontStyle::Italic,
        markdown::SpanKind::Text(markdown::Emphasis::BoldItalic) => FontStyle::BoldItalic,
        _ => FontStyle::Regular,
    }
}

/// The face a run or span of `kind` is measured and drawn in: the distinct code
/// family for an inline-code or fenced-code run, otherwise the prose family in
/// the kind's emphasis style. On a heading line the prose text is emboldened (a
/// heading renders bold), composing with any inline emphasis; the `#` marker and
/// inline code keep their own face. The single source of truth so layout's
/// measuring and the app's drawing pick the same face and never disagree on widths.
pub(crate) fn run_face(fonts: &Fonts, size: u32, kind: markdown::SpanKind, heading: bool) -> &Face {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Code) => fonts.code_face(size),
        markdown::SpanKind::Text(_) if heading => fonts.face(size, embolden(font_style(kind))),
        other => fonts.face(size, font_style(other)),
    }
}

/// Add weight to a prose style, so heading text renders bold and bold-italic
/// heading text keeps its slant. Already-bold styles are unchanged.
fn embolden(style: FontStyle) -> FontStyle {
    match style {
        FontStyle::Regular | FontStyle::Bold => FontStyle::Bold,
        FontStyle::Italic | FontStyle::BoldItalic => FontStyle::BoldItalic,
    }
}

/// The span kind covering byte `off`. `spans` tile `[0, len)` in order, so the
/// first span ending past `off` is the one holding it; a byte past the end
/// (never asked for in practice) falls back to regular text.
fn kind_at(spans: &[markdown::Span], off: usize) -> markdown::SpanKind {
    spans
        .iter()
        .find(|s| off < s.end)
        .map_or(markdown::SpanKind::Text(markdown::Emphasis::Regular), |s| {
            s.kind
        })
}

/// An axis-aligned rectangle in window pixels.
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Clone)]
pub struct Layout {
    lines: Vec<VisualLine>,
    left: i32,
    /// Total height of the text block in pixels (the sum of every visual line's
    /// line height), for the scroll extent.
    text_height: i32,
}

impl Layout {
    /// Lay `text` out within `wrap_width` pixels (<= 0 disables wrapping), with
    /// the text block starting at `top` and inset by `left`. Each logical line
    /// is classified by [`markdown`] and resolved to a pixel size from
    /// `body_size`, so headings render larger; baselines stack by each line's
    /// own metrics rather than one shared line height.
    ///
    /// `active` is the cursor/selection byte range (`(c, c)` for a bare cursor).
    /// A heading whose logical line the active range does not touch has its `#`
    /// markers hidden (the live-preview collapse); the markers reappear when the
    /// range reaches the line. Wrapping always uses the *source* text, so reveal
    /// state never changes line breaks or baselines, only the horizontal
    /// rendering of the heading's first segment.
    pub fn build(
        fonts: &Fonts,
        rope: &Rope,
        left: i32,
        top: i32,
        wrap_width: i32,
        body_size: u32,
        active: (usize, usize),
    ) -> Self {
        let mut lines = Vec::new();
        // `box_top` is the top of the current line's box; the baseline sits one
        // ascent below it, and the next box follows one line height down.
        let mut box_top = top;
        // `fence` holds the open fence's backtick count while inside a code block,
        // the one piece of state that carries across logical lines.
        let mut fence: Option<usize> = None;
        // `list_depth` is the nesting depth of the most recent valid list item (or
        // `None` outside a list), so an item is rendered as a list only when its
        // parent, one level shallower, is present above it.
        let mut list_depth: Option<usize> = None;
        let n_lines = rope.len_lines();
        for i in 0..n_lines {
            let ls = rope.line_to_byte(i);
            // Every logical line but the last ends in a '\n'; its content
            // excludes that byte. The rope materializes just this line.
            let has_newline = i + 1 < n_lines;
            let le = if has_newline {
                rope.line_to_byte(i + 1) - 1
            } else {
                rope.line_to_byte(i + 1)
            };
            let line = rope.byte_slice(ls..le);
            // Whether the cursor/selection touches this logical line: the same
            // reveal test headings use, applied to fence lines too.
            let on_line = active.0 <= le && active.1 >= ls;
            // A line inside a fenced code block is code, not markdown: it skips the
            // inline pass and renders as one code run. The ``` fence lines render
            // their backticks as a marker that hides off the cursor's line, like a
            // heading's `#`, so an unfocused block reads without fences (the panel
            // still draws, leaving a blank padded row). The code lines between
            // carry no markers, so their reveal flag is inert; keep it true. A code
            // line carries `Body` as an inert style (its gutter is unused).
            let (style, size, spans, block_marker, block_persists, revealed, code_block) =
                if let Some(open) = fence {
                    list_depth = None;
                    let delimiter = markdown::closes_code_fence(&line, open);
                    if delimiter {
                        fence = None;
                    }
                    let spans = code_block_spans(&line, delimiter);
                    // Only a fence delimiter line has backticks to hide.
                    let revealed = !delimiter || on_line;
                    (
                        markdown::LineStyle::Body,
                        body_size,
                        spans,
                        0,
                        false,
                        revealed,
                        true,
                    )
                } else if let Some(open) = markdown::opens_code_fence(&line) {
                    list_depth = None;
                    fence = Some(open);
                    let spans = code_block_spans(&line, true);
                    (
                        markdown::LineStyle::Body,
                        body_size,
                        spans,
                        0,
                        false,
                        on_line,
                        true,
                    )
                } else {
                    // Resolve list nesting across lines: an item deeper than one
                    // past the previous item has no parent and renders as body.
                    let (style, marker) = resolve_list(&line, &mut list_depth);
                    (
                        style,
                        markdown::font_size(style, body_size),
                        line_spans(&line, marker),
                        marker,
                        markdown::block_marker_persists(style),
                        on_line,
                        false,
                    )
                };
            let metrics = fonts.metrics(size);
            // Heading lines color their text by level (fence/code lines carry the
            // inert `Body` style, so they are never headings here).
            let heading = match style {
                markdown::LineStyle::Heading(level) => Some(level),
                _ => None,
            };
            // A thematic break draws as a horizontal rule once the cursor leaves
            // it; on the line itself the raw `---` shows (and `revealed` is true),
            // matching how a heading reveals its `#`.
            let rule = matches!(style, markdown::LineStyle::Rule) && !revealed;
            // A list or quote item hangs its wrapped continuations under the text:
            // the inset is the marker's width, measured on the source marker in the
            // regular face at this line's size, so it matches the width the marker
            // run itself advances on the first line and the columns line up.
            let block_indent = if block_persists && block_marker > 0 {
                render::text_advance(fonts.face(size, FontStyle::Regular), &line[..block_marker])
                    .round() as i32
            } else {
                0
            };
            // A left inset shared by every segment of the line, and how much it
            // narrows the wrap width: a code block pads both sides inside its
            // panel; a list item shifts right by a small indent (one side only).
            let is_list = matches!(
                style,
                markdown::LineStyle::Bullet | markdown::LineStyle::Ordered
            );
            let (seg_pad, wrap_reduce) = if code_block {
                (CODE_PAD_X, 2 * CODE_PAD_X)
            } else if is_list {
                (LIST_INDENT, LIST_INDENT)
            } else {
                (0, 0)
            };
            let line_wrap = if wrap_width > 0 {
                (wrap_width - wrap_reduce).max(1)
            } else {
                wrap_width
            };
            // `wrap_segments` returns ranges relative to the line; offset them
            // back to absolute buffer positions for the visual lines. It measures
            // the source text (every run in its own face), so clipping spans to
            // each segment never disagrees with the breaks, and reveal state,
            // which only hides markers, never reflows the document.
            let segments = wrap_segments(
                fonts,
                size,
                &line,
                &spans,
                line_wrap,
                block_indent,
                heading.is_some(),
            );
            let last = segments.len() - 1;
            for (k, (start, end)) in segments.into_iter().enumerate() {
                // Every segment carries the shared left inset (`seg_pad`); wrapped
                // continuations add the marker's width on top so their text hangs
                // under the first line's text.
                let indent = seg_pad + if k == 0 { 0 } else { block_indent };
                // A bullet shows its dot decoration even on the cursor's own line,
                // so a list stays a list while you edit it (the raw `- ` is the
                // editable source beneath the dot). A quote still reveals its raw
                // `> ` for editing, drawing the bar only once the cursor leaves.
                let gutter = match style {
                    markdown::LineStyle::Bullet if k == 0 => Gutter::Bullet,
                    markdown::LineStyle::Quote if !revealed => Gutter::QuoteBar { marker: k == 0 },
                    _ => Gutter::None,
                };
                lines.push(VisualLine {
                    start: ls + start,
                    end: ls + end,
                    baseline: box_top + metrics.ascent,
                    // Only the final segment of a logical line that has a
                    // trailing '\n' is a hard break.
                    hard: k == last && has_newline,
                    size,
                    runs: clip_runs(
                        &spans,
                        start,
                        end,
                        ls,
                        revealed,
                        block_marker,
                        block_persists,
                    ),
                    indent,
                    gutter,
                    code_block,
                    heading,
                    rule,
                });
                box_top += metrics.line_height;
            }
        }
        Self {
            lines,
            left,
            text_height: box_top - top,
        }
    }

    pub fn lines(&self) -> &[VisualLine] {
        &self.lines
    }

    /// Total height of the laid-out text block in pixels, for sizing the
    /// scrollable content.
    pub fn text_height(&self) -> i32 {
        self.text_height
    }

    /// The byte range `(start, end)` of the visual line containing `offset`,
    /// for Home/End motion.
    pub fn line_bounds(&self, offset: usize) -> (usize, usize) {
        let line = self.line_for(offset);
        (line.start, line.end)
    }

    /// The caret rectangle (its x, top y, and height) for the cursor at `offset`,
    /// sized to the metrics of the line it lands on. The caret only ever lands on
    /// a revealed line (the cursor's own line shows its markers), so its `offset`
    /// never falls inside a hidden run.
    pub fn caret(&self, fonts: &Fonts, rope: &Rope, offset: usize) -> (i32, i32, i32) {
        let line = self.line_for(offset);
        let m = fonts.metrics(line.size);
        let x = self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, offset);
        (
            x.round() as i32,
            line.baseline - m.ascent,
            m.ascent + m.descent,
        )
    }

    /// The byte offset nearest a click at window `(x, y)`.
    pub fn offset_at(&self, fonts: &Fonts, rope: &Rope, x: f32, y: f32) -> usize {
        let line = self.line_for_y(fonts, y);
        self.hit_line(fonts, rope, line, x)
    }

    /// If the visual line at content-space `y` is a rendered horizontal rule (a
    /// thematic break whose cursor has left), its content byte range
    /// `(start, end)`. A click there selects the whole rule rather than dropping a
    /// caret between its `---` markers; `None` for any other line.
    pub fn rule_at(&self, fonts: &Fonts, y: f32) -> Option<(usize, usize)> {
        let line = self.line_for_y(fonts, y);
        line.rule.then_some((line.start, line.end))
    }

    /// The byte offset reached by moving `dir` lines (-1 up, +1 down) from
    /// `offset`, aiming for window x `goal_x`. Clamps at the first and last line.
    pub fn offset_vertical(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        offset: usize,
        dir: i32,
        goal_x: f32,
    ) -> usize {
        let idx = self.line_index_for(offset) as i32;
        let target = idx + dir;
        if target < 0 || target as usize >= self.lines.len() {
            return offset;
        }
        self.hit_line(fonts, rope, &self.lines[target as usize], goal_x)
    }

    /// The highlight rectangles covering the byte range `[start, end)`, one per
    /// visual line it touches, each sized to that line's metrics.
    pub fn selection_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        start: usize,
        end: usize,
    ) -> Vec<Rect> {
        let mut rects = Vec::new();
        for line in &self.lines {
            let m = fonts.metrics(line.size);
            let sel_start = start.max(line.start);
            let sel_end = end.min(line.end);
            // A hard line's newline is selected when the range reaches past the
            // line's text into the break. Soft wraps have no newline to select.
            let newline_selected = line.hard && start <= line.end && end > line.end;
            if sel_start >= sel_end && !newline_selected {
                continue;
            }
            // Measure the run-walked advance to each end of the selection; bytes
            // in a hidden marker collapse to the marker's edge, so a selection
            // touching one (in practice a revealed line) measures sensibly.
            let x0 = self.left as f32
                + line.indent as f32
                + self.advance_to(fonts, rope, line, sel_start);
            let x1 =
                self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, sel_end);
            let mut w = (x1 - x0).round() as i32;
            if newline_selected {
                w += NEWLINE_PAD;
            }
            rects.push(Rect {
                x: x0.round() as i32,
                y: line.baseline - m.ascent,
                w: w.max(1),
                h: m.ascent + m.descent,
            });
        }
        rects
    }

    /// The tinted-panel rectangles behind every visible inline code span, one per
    /// run, each sized to its line's metrics. Fenced code blocks are excluded:
    /// they have their own full-width panel. Like [`Self::selection_rects`], these
    /// are background decorations the app paints beneath the selection and text, so
    /// a selection over inline code still shows through. Positions are run-walked
    /// the same way as the selection, so the two agree on a shared line.
    pub fn code_rects(&self, fonts: &Fonts, rope: &Rope) -> Vec<Rect> {
        let mut rects = Vec::new();
        for line in &self.lines {
            if line.code_block {
                continue;
            }
            let m = fonts.metrics(line.size);
            for run in &line.runs {
                if !run.visible
                    || !matches!(run.kind, markdown::SpanKind::Text(markdown::Emphasis::Code))
                {
                    continue;
                }
                let x0 = self.left as f32
                    + line.indent as f32
                    + self.advance_to(fonts, rope, line, run.start);
                let w = render::text_advance(
                    run_face(fonts, line.size, run.kind, line.heading.is_some()),
                    &rope.byte_slice(run.start..run.end),
                );
                rects.push(Rect {
                    x: x0.round() as i32,
                    y: line.baseline - m.ascent,
                    w: (w.round() as i32).max(1),
                    h: m.ascent + m.descent,
                });
            }
        }
        rects
    }

    /// The visual line containing `offset`. A cursor at a soft-wrap boundary is
    /// given to the later line; one at a hard newline stays on its line.
    fn line_for(&self, offset: usize) -> &VisualLine {
        self.lines
            .iter()
            .find(|l| offset < l.end || (offset == l.end && l.hard))
            .unwrap_or_else(|| self.last_line())
    }

    fn line_index_for(&self, offset: usize) -> usize {
        self.lines
            .iter()
            .position(|l| offset < l.end || (offset == l.end && l.hard))
            .unwrap_or(self.lines.len().saturating_sub(1))
    }

    /// The visual line whose vertical band contains `y`, clamped to the first or
    /// last line for clicks above or below the text. Each line's band reaches to
    /// its own descent below the baseline.
    fn line_for_y(&self, fonts: &Fonts, y: f32) -> &VisualLine {
        self.lines
            .iter()
            .find(|l| y < (l.baseline + fonts.metrics(l.size).descent) as f32)
            .unwrap_or_else(|| self.last_line())
    }

    /// The byte offset within `line` nearest window x `x`, snapped to a grapheme
    /// cluster boundary (a click in a cluster's left half lands before it). Hit
    /// testing walks only the visible runs, so a click on a line with hidden
    /// markers lands in the visible text, never inside a collapsed marker.
    fn hit_line(&self, fonts: &Fonts, rope: &Rope, line: &VisualLine, x: f32) -> usize {
        let mut acc = self.left as f32 + line.indent as f32;
        // Past every glyph: the end of the last visible run, or the line start
        // for an empty line.
        let mut last = line.start;
        for run in &line.runs {
            if !run.visible {
                continue;
            }
            let face = run_face(fonts, line.size, run.kind, line.heading.is_some());
            let slice = rope.byte_slice(run.start..run.end);
            for (off, cluster) in grapheme::graphemes(&slice) {
                let w = render::text_advance(face, cluster);
                if x < acc + w / 2.0 {
                    return run.start + off;
                }
                acc += w;
            }
            last = run.end;
        }
        last
    }

    /// Pen advance in pixels from the left margin to absolute byte `offset`,
    /// summing the line's visible runs; a hidden (collapsed) marker contributes
    /// nothing. An `offset` at or past the line's end measures the whole line.
    fn advance_to(&self, fonts: &Fonts, rope: &Rope, line: &VisualLine, offset: usize) -> f32 {
        let mut x = 0.0;
        for run in &line.runs {
            // Runs are ordered, so once a run starts at or past `offset` the rest
            // do too; hidden runs add nothing.
            if !run.visible || offset <= run.start {
                continue;
            }
            let end = offset.min(run.end);
            let face = run_face(fonts, line.size, run.kind, line.heading.is_some());
            x += render::text_advance(face, &rope.byte_slice(run.start..end));
        }
        x
    }

    fn last_line(&self) -> &VisualLine {
        // build always pushes at least one line, so this never panics.
        &self.lines[self.lines.len() - 1]
    }
}

/// Resolve a line's block style against the running list nesting `list_depth`
/// (the depth of the most recent valid list item, or `None` outside a list),
/// returning the effective style and its block-marker length. A list item nests
/// by leading indentation ([`markdown::LIST_INDENT_SPACES`] per level); it only
/// renders as a list item when its parent, one level shallower, is present above
/// it, otherwise it falls back to plain body text (marker length `0`). Blank
/// lines leave the nesting untouched; any other non-list line ends it.
fn resolve_list(line: &str, list_depth: &mut Option<usize>) -> (markdown::LineStyle, usize) {
    let style = markdown::style_of(line);
    match style {
        markdown::LineStyle::Bullet | markdown::LineStyle::Ordered => {
            let depth = markdown::leading_indent(line) / markdown::LIST_INDENT_SPACES;
            let allowed = list_depth.map_or(0, |d| d + 1);
            if depth <= allowed {
                *list_depth = Some(depth);
                (style, markdown::marker_len(line))
            } else {
                // Over-nested: no parent one level up, so render as plain text.
                *list_depth = None;
                (markdown::LineStyle::Body, 0)
            }
        }
        other => {
            if !line.trim().is_empty() {
                *list_depth = None;
            }
            (other, markdown::marker_len(line))
        }
    }
}

/// The full inline span list of a logical `line`, line-relative: a leading
/// block marker of `marker` bytes (a heading's `#`s, a list `- `, etc., if any)
/// followed by the inline spans of the content past it, offset back over the
/// marker. Tiles `[0, line.len())`. `marker` is passed in (not recomputed) so a
/// caller can suppress it, e.g. an over-nested list item that renders as plain
/// body text gets `marker == 0`.
fn line_spans(line: &str, marker: usize) -> Vec<markdown::Span> {
    let mut spans = Vec::new();
    if marker > 0 {
        spans.push(markdown::Span {
            start: 0,
            end: marker,
            kind: markdown::SpanKind::Marker,
        });
    }
    for mut span in markdown::inline_spans(&line[marker..]) {
        span.start += marker;
        span.end += marker;
        spans.push(span);
    }
    spans
}

/// The spans of a line inside a fenced code block: a whole-line code run, or, on
/// a fence delimiter line (the opening or closing ```), a whole-line marker run
/// so the fence and any info string read as dimmed syntax. No inline parsing
/// happens inside a code block. An empty line has no runs (the panel still draws).
fn code_block_spans(line: &str, delimiter: bool) -> Vec<markdown::Span> {
    if line.is_empty() {
        return Vec::new();
    }
    let kind = if delimiter {
        markdown::SpanKind::Marker
    } else {
        markdown::SpanKind::Text(markdown::Emphasis::Code)
    };
    vec![markdown::Span {
        start: 0,
        end: line.len(),
        kind,
    }]
}

/// Clip line-relative `spans` to the visual segment `[seg_start, seg_end)`,
/// offsetting to absolute by `ls`. Text runs always show. A marker shows only on
/// a revealed line, except the block marker (the leading `[0, block_marker)`
/// bytes) when `block_persists`: a list bullet or quote `>` stays on screen even
/// off the active line. Empty overlaps are dropped, so the runs tile the segment.
fn clip_runs(
    spans: &[markdown::Span],
    seg_start: usize,
    seg_end: usize,
    ls: usize,
    revealed: bool,
    block_marker: usize,
    block_persists: bool,
) -> Vec<Run> {
    let mut runs = Vec::new();
    for span in spans {
        let s = span.start.max(seg_start);
        let e = span.end.min(seg_end);
        if s >= e {
            continue;
        }
        let is_marker = matches!(span.kind, markdown::SpanKind::Marker);
        // The block marker occupies the line's leading `[0, block_marker)` bytes;
        // inline markers (code, emphasis) start past it.
        let is_block_marker = is_marker && span.start < block_marker;
        let visible = if is_block_marker {
            revealed || block_persists
        } else {
            !is_marker || revealed
        };
        runs.push(Run {
            start: ls + s,
            end: ls + e,
            kind: span.kind,
            visible,
        });
    }
    runs
}

/// Greedily word-wrap the logical `line` to `wrap_width` pixels, returning the
/// byte ranges of its visual segments. Breaks after the last space that fits; a
/// word wider than `wrap_width` breaks at a grapheme boundary so every segment
/// holds at least one cluster. `wrap_width <= 0` disables wrapping. Each cluster
/// is measured in its run's face (per `spans` and `size`), so bold/italic widths
/// wrap correctly; the source text is measured in full (markers included), so
/// hiding markers later never reflows the line.
///
/// `block_indent` is the hanging indent of a list or quote item: the first
/// segment uses the full `wrap_width` (its marker sits in that width), and every
/// continuation is wrapped `block_indent` pixels narrower, since it renders inset
/// by that much. So the text column is the same width on every line of the item.
fn wrap_segments(
    fonts: &Fonts,
    size: u32,
    line: &str,
    spans: &[markdown::Span],
    wrap_width: i32,
    block_indent: i32,
    heading: bool,
) -> Vec<(usize, usize)> {
    if wrap_width <= 0 || line.is_empty() {
        return vec![(0, line.len())];
    }
    // Measure each cluster once: its offset within the line, width in its run's
    // face, and whether it is a space (a wrap opportunity).
    let clusters: Vec<(usize, f32, bool)> = grapheme::graphemes(line)
        .into_iter()
        .map(|(off, cluster)| {
            let face = run_face(fonts, size, kind_at(spans, off), heading);
            (off, render::text_advance(face, cluster), cluster == " ")
        })
        .collect();
    if clusters.is_empty() {
        return vec![(0, line.len())];
    }

    let mut segments = Vec::new();
    let mut i = 0;
    while i < clusters.len() {
        // The first segment gets the full width; continuations are narrower by
        // the hanging indent, since they render inset by it.
        let limit = (wrap_width - if segments.is_empty() { 0 } else { block_indent }).max(1) as f32;
        let seg_start = clusters[i].0;
        let mut width = 0.0;
        // Cluster index that would start the next segment if we break at the
        // last fitting space; `None` until a space is seen in this segment.
        let mut last_break: Option<usize> = None;
        let mut j = i;
        while j < clusters.len() {
            let (_, w, is_space) = clusters[j];
            // Always keep at least one cluster (j > i) before allowing a break.
            if width + w > limit && j > i {
                break;
            }
            width += w;
            j += 1;
            if is_space {
                last_break = Some(j);
            }
        }
        if j >= clusters.len() {
            segments.push((seg_start, line.len()));
            break;
        }
        // Break after the last space if there was one, else mid-word at j.
        let next = match last_break {
            Some(b) if b > i => b,
            _ => j,
        };
        segments.push((seg_start, clusters[next].0));
        i = next;
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown::LineStyle;

    const BODY: u32 = 32;
    const LEFT: i32 = 40;
    const TOP: i32 = 10;

    fn fonts() -> Fonts {
        Fonts::new(&markdown::all_sizes(BODY)).expect("a default font")
    }

    fn h1_size() -> u32 {
        markdown::font_size(LineStyle::Heading(1), BODY)
    }

    /// Build with wrapping effectively disabled, at body size, with the whole
    /// buffer active so every heading is revealed.
    /// Reveal-specific tests build explicitly with a narrower active range.
    fn layout(f: &Fonts, r: &Rope) -> Layout {
        Layout::build(f, r, LEFT, TOP, 1_000_000, BODY, (0, r.len()))
    }

    /// A rope from a string literal, for the geometry tests.
    fn rope(s: &str) -> Rope {
        Rope::from_str(s)
    }

    /// A wrap width that fits "aaa " but not a following cluster, so "aaa bbb"
    /// splits deterministically into "aaa " and "bbb".
    fn tight_wrap(f: &Fonts) -> i32 {
        let face = f.face(BODY, FontStyle::Regular);
        (render::text_advance(face, "aaa ") + render::text_advance(face, "b") / 2.0) as i32
    }

    #[test]
    fn splits_logical_lines_with_stacked_baselines() {
        let f = fonts();
        let m = f.metrics(BODY);
        let l = layout(&f, &rope("ab\ncd\n"));
        let lines = l.lines();
        assert_eq!(lines.len(), 3, "trailing newline yields an empty last line");
        assert_eq!((lines[0].start, lines[0].end), (0, 2));
        assert_eq!((lines[1].start, lines[1].end), (3, 5));
        assert_eq!((lines[2].start, lines[2].end), (6, 6));
        // Uniform body text stacks one line height at a time.
        assert_eq!(lines[0].baseline, TOP + m.ascent);
        assert_eq!(lines[1].baseline, TOP + m.ascent + m.line_height);
        assert_eq!(lines[2].baseline, TOP + m.ascent + 2 * m.line_height);
        // The first two are hard breaks; the empty last line is not.
        assert!(lines[0].hard && lines[1].hard && !lines[2].hard);
    }

    #[test]
    fn empty_text_is_one_line() {
        let f = fonts();
        let l = layout(&f, &rope(""));
        assert_eq!(l.lines().len(), 1);
        assert_eq!((l.lines()[0].start, l.lines()[0].end), (0, 0));
        assert!(!l.lines()[0].hard);
    }

    #[test]
    fn caret_sits_at_the_left_margin_for_a_line_start() {
        let f = fonts();
        let m = f.metrics(BODY);
        let r = rope("hello\nworld");
        let l = layout(&f, &r);
        let (x0, top0, h) = l.caret(&f, &r, 0);
        assert_eq!(x0, LEFT, "first column is the left margin");
        assert_eq!(
            top0, TOP,
            "caret top is baseline - ascent, i.e. the box top"
        );
        assert_eq!(h, m.ascent + m.descent, "caret spans ascent + descent");
        let (x1, top1, _) = l.caret(&f, &r, 6);
        assert_eq!(x1, LEFT, "second line also starts at the margin");
        assert_eq!(top1, TOP + m.line_height, "and one line lower");
    }

    #[test]
    fn caret_advances_along_a_line() {
        let f = fonts();
        let r = rope("mm");
        let l = layout(&f, &r);
        let (x0, _, _) = l.caret(&f, &r, 0);
        let (x1, _, _) = l.caret(&f, &r, 1);
        let (x2, _, _) = l.caret(&f, &r, 2);
        assert!(x1 > x0 && x2 > x1, "caret x grows with the column");
    }

    #[test]
    fn click_maps_to_the_clicked_line_and_column() {
        let f = fonts();
        let text = "abc\ndefg";
        let r = rope(text);
        let l = layout(&f, &r);
        let (y0, y1) = (l.lines()[0].baseline as f32, l.lines()[1].baseline as f32);
        let off = l.offset_at(&f, &r, 10_000.0, y1);
        assert_eq!(off, text.len(), "past the last glyph of line two -> end");
        let off = l.offset_at(&f, &r, -100.0, y0);
        assert_eq!(off, 0);
        let off = l.offset_at(&f, &r, -100.0, 9_000.0);
        assert_eq!(off, 4, "start of the last line");
    }

    #[test]
    fn vertical_motion_changes_line_and_clamps() {
        let f = fonts();
        let r = rope("abc\ndef\nghi");
        let l = layout(&f, &r);
        let (x, _, _) = l.caret(&f, &r, 5);
        let up = l.offset_vertical(&f, &r, 5, -1, x as f32);
        assert_eq!(up, 1, "same column on the line above");
        let down = l.offset_vertical(&f, &r, 5, 1, x as f32);
        assert_eq!(down, 9, "same column on the line below");
        assert_eq!(l.offset_vertical(&f, &r, 1, -1, x as f32), 1);
    }

    #[test]
    fn selection_on_one_line_is_one_rect() {
        let f = fonts();
        let m = f.metrics(BODY);
        let l = layout(&f, &rope("hello"));
        let rects = l.selection_rects(&f, &rope("hello"), 1, 4);
        assert_eq!(rects.len(), 1);
        assert!(rects[0].w > 0);
        assert_eq!(rects[0].h, m.ascent + m.descent);
        assert_eq!(rects[0].y, TOP);
    }

    #[test]
    fn selection_across_hard_lines_is_one_rect_per_line() {
        let f = fonts();
        let l = layout(&f, &rope("ab\ncd\nef"));
        let rects = l.selection_rects(&f, &rope("ab\ncd\nef"), 1, 7);
        assert_eq!(rects.len(), 3, "three lines touched");
        assert!(rects[0].w >= NEWLINE_PAD, "hard newline adds the pad");
    }

    #[test]
    fn long_line_soft_wraps_at_spaces() {
        let f = fonts();
        let r = rope("aaa bbb");
        let l = Layout::build(&f, &r, 0, 0, tight_wrap(&f), BODY, (0, r.len()));
        let lines = l.lines();
        assert_eq!(lines.len(), 2, "wraps into two visual lines");
        assert_eq!(
            (lines[0].start, lines[0].end),
            (0, 4),
            "'aaa ' on the first"
        );
        assert_eq!(
            (lines[1].start, lines[1].end),
            (4, 7),
            "'bbb' on the second"
        );
        assert!(
            !lines[0].hard && !lines[1].hard,
            "a wrap is not a hard break"
        );
    }

    #[test]
    fn overlong_word_breaks_mid_word() {
        let f = fonts();
        let text = "mmmmmmmmmm"; // one word, no spaces
        let r = rope(text);
        let wrap = (render::text_advance(f.face(BODY, FontStyle::Regular), "m") * 3.0) as i32;
        let l = Layout::build(&f, &r, 0, 0, wrap, BODY, (0, r.len()));
        let lines = l.lines();
        assert!(lines.len() >= 2, "an unbreakable word still wraps");
        assert!(
            lines.iter().all(|ln| ln.end > ln.start),
            "every segment holds at least one cluster"
        );
        // Segments are contiguous and cover the whole word.
        assert_eq!(lines.first().unwrap().start, 0);
        assert_eq!(lines.last().unwrap().end, text.len());
    }

    #[test]
    fn caret_affinity_puts_a_soft_boundary_on_the_next_line() {
        let f = fonts();
        let m = f.metrics(BODY);
        let r = rope("aaa bbb");
        let l = Layout::build(&f, &r, 0, 0, tight_wrap(&f), BODY, (0, r.len()));
        // Offset 4 is the soft-wrap boundary; the caret shows at the start of
        // the second visual line, not the end of the first.
        let (x, top, _) = l.caret(&f, &r, 4);
        assert_eq!(x, 0, "left margin of the wrapped row");
        assert_eq!(top, m.line_height, "second box top: one line height down");
    }

    #[test]
    fn selection_across_a_soft_wrap_has_no_newline_pad() {
        let f = fonts();
        let r = rope("aaa bbb");
        let l = Layout::build(&f, &r, 0, 0, tight_wrap(&f), BODY, (0, r.len()));
        let rects = l.selection_rects(&f, &r, 0, 7);
        assert_eq!(rects.len(), 2, "one rect per visual line");
        let line0_text_w =
            render::text_advance(f.face(BODY, FontStyle::Regular), "aaa ").round() as i32;
        assert!(
            (rects[0].w - line0_text_w).abs() <= 1,
            "a soft wrap adds no newline pad"
        );
    }

    #[test]
    fn heading_line_renders_larger_and_pushes_the_body_down() {
        let f = fonts();
        let body = f.metrics(BODY);
        let h1 = f.metrics(h1_size());
        assert!(
            h1.line_height > body.line_height,
            "the heading box is taller"
        );
        let l = layout(&f, &rope("# Title\nbody"));
        let lines = l.lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0].size,
            h1_size(),
            "the heading line uses the H1 size"
        );
        assert_eq!(lines[1].size, BODY, "the body line uses the body size");
        // The body baseline clears the full heading box, not a body line height.
        assert_eq!(lines[0].baseline, TOP + h1.ascent);
        assert_eq!(lines[1].baseline, TOP + h1.line_height + body.ascent);
    }

    #[test]
    fn caret_on_a_heading_spans_the_taller_box() {
        let f = fonts();
        let body = f.metrics(BODY);
        let h1 = f.metrics(h1_size());
        let r = rope("# Title");
        let l = layout(&f, &r);
        let (_, _, h) = l.caret(&f, &r, 0);
        assert_eq!(h, h1.ascent + h1.descent);
        assert!(h > body.ascent + body.descent, "taller than a body caret");
    }

    #[test]
    fn heading_text_is_wider_per_character_than_body() {
        // The same characters measure wider at the heading size, which is what
        // lets clicks and the caret land correctly on a heading line.
        let f = fonts();
        let body_w = render::text_advance(f.face(BODY, FontStyle::Regular), "Title");
        let h1_w = render::text_advance(f.face(h1_size(), FontStyle::Regular), "Title");
        assert!(h1_w > body_w);
    }

    // --- marker-hiding live preview ---------------------------------------

    /// Build "# Title\nbody" with the active range placed by `active`.
    fn heading_and_body(f: &Fonts, active: (usize, usize)) -> Layout {
        Layout::build(
            f,
            &rope("# Title\nbody"),
            LEFT,
            TOP,
            1_000_000,
            BODY,
            active,
        )
    }

    /// Whether `line`'s leading marker run is collapsed. Asserts the line starts
    /// with a marker run, which every heading line in these tests does.
    fn marker_hidden(line: &VisualLine) -> bool {
        let run = line.runs.first().expect("a heading line has runs");
        assert!(
            matches!(run.kind, markdown::SpanKind::Marker),
            "the first run is the heading marker"
        );
        !run.visible
    }

    #[test]
    fn markers_hide_when_the_cursor_is_off_the_heading_line() {
        let f = fonts();
        let body_off = "# Title\nbody".find("body").unwrap();
        let l = heading_and_body(&f, (body_off, body_off));
        assert!(marker_hidden(&l.lines()[0]), "the '# ' marker collapses");
        assert!(
            l.lines()[1]
                .runs
                .iter()
                .all(|r| matches!(r.kind, markdown::SpanKind::Text(_))),
            "body has no marker to hide"
        );
    }

    #[test]
    fn markers_show_while_the_cursor_edits_the_heading() {
        let f = fonts();
        // Cursor inside "Title".
        let l = heading_and_body(&f, (4, 4));
        assert!(!marker_hidden(&l.lines()[0]), "markers stay while editing");
    }

    #[test]
    fn markers_show_at_the_very_end_of_the_heading_line() {
        let f = fonts();
        // Cursor at offset 7, right before the '\n': still on the heading line.
        let l = heading_and_body(&f, (7, 7));
        assert!(!marker_hidden(&l.lines()[0]));
    }

    #[test]
    fn a_selection_touching_a_heading_reveals_it() {
        let f = fonts();
        // Select the whole buffer: the heading is within it, so it reveals.
        let l = heading_and_body(&f, (0, "# Title\nbody".len()));
        assert!(
            !marker_hidden(&l.lines()[0]),
            "a selected heading shows markers"
        );
    }

    #[test]
    fn hidden_markers_shift_clicks_into_the_visible_text() {
        let f = fonts();
        let r = rope("# Title");
        // No cursor on the line, so the markers hide.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        assert!(marker_hidden(&l.lines()[0]));
        // A click at the far left lands at the start of the visible text
        // ("Title", offset 2), never inside the collapsed "# ".
        let off = l.offset_at(&f, &r, -100.0, l.lines()[0].baseline as f32);
        assert_eq!(off, 2, "click maps past the collapsed marker");
    }

    #[test]
    fn revealing_markers_shifts_the_heading_text_right_by_the_marker_width() {
        let f = fonts();
        let r = rope("# Title");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        // The first visible glyph is at byte offset 2 ("T") in both; collapsed it
        // sits at the margin, expanded it sits a marker-width to the right.
        let (x_hidden, _, _) = hidden.caret(&f, &r, 2);
        let (x_shown, _, _) = shown.caret(&f, &r, 2);
        assert_eq!(x_hidden, LEFT, "collapsed: visible text at the margin");
        let marker_w =
            render::text_advance(f.face(h1_size(), FontStyle::Regular), "# ").round() as i32;
        assert_eq!(x_shown - x_hidden, marker_w, "expanded by the marker width");
    }

    #[test]
    fn reveal_state_does_not_change_vertical_layout() {
        // Wrapping uses the source text, so collapsing markers never reflows the
        // document: line count, baselines, and height are reveal-independent.
        let f = fonts();
        let r = rope("# A longer heading that might wrap\nbody text below the heading");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (0, 0));
        assert_eq!(hidden.lines().len(), shown.lines().len());
        for (h, s) in hidden.lines().iter().zip(shown.lines()) {
            assert_eq!(h.baseline, s.baseline, "baselines are reveal-independent");
        }
        assert_eq!(hidden.text_height(), shown.text_height());
    }

    // --- inline code runs -------------------------------------------------

    #[test]
    fn a_code_span_becomes_marker_code_marker_runs() {
        let f = fonts();
        let r = rope("a `b` c");
        // Cursor on the line: every run is shown.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let kinds: Vec<_> = l.lines()[0].runs.iter().map(|run| run.kind).collect();
        assert_eq!(
            kinds,
            vec![
                markdown::SpanKind::Text(markdown::Emphasis::Regular),
                markdown::SpanKind::Marker,
                markdown::SpanKind::Text(markdown::Emphasis::Code),
                markdown::SpanKind::Marker,
                markdown::SpanKind::Text(markdown::Emphasis::Regular),
            ]
        );
        assert!(
            l.lines()[0].runs.iter().all(|run| run.visible),
            "all runs shown on the active line"
        );
    }

    #[test]
    fn code_markers_collapse_off_the_line() {
        let f = fonts();
        let src = "a `b` c\nx";
        let r = rope(src);
        // Cursor on the second line.
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let line = &l.lines()[0];
        assert!(
            line.runs
                .iter()
                .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
                .all(|run| !run.visible),
            "the backticks collapse off the line"
        );
        assert!(
            line.runs
                .iter()
                .filter(|run| matches!(
                    run.kind,
                    markdown::SpanKind::Text(markdown::Emphasis::Code)
                ))
                .all(|run| run.visible),
            "the code text stays shown"
        );
    }

    #[test]
    fn clicks_never_land_inside_hidden_code_markers() {
        let f = fonts();
        let r = rope("a `b` c");
        // Cursor off the line, so the backticks (offsets 2 and 4) collapse.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let y = l.lines()[0].baseline as f32;
        let mut seen = std::collections::BTreeSet::new();
        let mut x = (LEFT - 50) as f32;
        while x < (LEFT + 600) as f32 {
            seen.insert(l.offset_at(&f, &r, x, y));
            x += 1.0;
        }
        assert!(
            !seen.contains(&2) && !seen.contains(&4),
            "a click never resolves to a collapsed backtick"
        );
        assert!(
            seen.contains(&0) && seen.contains(&7),
            "the line's ends stay reachable"
        );
    }

    #[test]
    fn hiding_a_code_marker_shifts_the_following_text_left() {
        let f = fonts();
        let r = rope("`b`");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        // Offset 1 is "b", just past the opening backtick. Collapsed it sits at
        // the margin; shown it sits one backtick-width to the right.
        let (x_hidden, _, _) = hidden.caret(&f, &r, 1);
        let (x_shown, _, _) = shown.caret(&f, &r, 1);
        assert_eq!(x_hidden, LEFT, "collapsed: code text at the margin");
        let tick_w = render::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        assert_eq!(
            x_shown - x_hidden,
            tick_w,
            "shown: shifted by the backtick width"
        );
    }

    // --- emphasis runs ----------------------------------------------------

    #[test]
    fn emphasis_composes_with_a_block_marker() {
        // A heading whose text holds a bold span: the block `# ` marker and the
        // inline `**` markers coexist, and `line_spans` offsets the inline runs
        // past the block marker so the bold "b" lands at the right bytes.
        let f = fonts();
        let r = rope("# a **b** c");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let line = &l.lines()[0];
        assert_eq!(line.size, h1_size(), "still an H1 line");
        let runs: Vec<_> = line.runs.iter().map(|run| (run.start, run.kind)).collect();
        assert_eq!(
            runs,
            vec![
                (0, markdown::SpanKind::Marker),                            // "# "
                (2, markdown::SpanKind::Text(markdown::Emphasis::Regular)), // "a "
                (4, markdown::SpanKind::Marker),                            // "**"
                (6, markdown::SpanKind::Text(markdown::Emphasis::Bold)),    // "b"
                (7, markdown::SpanKind::Marker),                            // "**"
                (9, markdown::SpanKind::Text(markdown::Emphasis::Regular)), // " c"
            ]
        );
        assert!(line.runs.iter().all(|run| run.visible), "active: all shown");
    }

    #[test]
    fn emphasis_markers_collapse_but_the_bold_text_stays() {
        let f = fonts();
        let src = "# a **b** c\nx";
        let r = rope(src);
        // Cursor on the second line, so the heading's markers hide.
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let line = &l.lines()[0];
        assert!(
            line.runs
                .iter()
                .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
                .all(|run| !run.visible),
            "every marker (heading and emphasis) collapses off the line"
        );
        assert!(
            line.runs
                .iter()
                .filter(|run| matches!(
                    run.kind,
                    markdown::SpanKind::Text(markdown::Emphasis::Bold)
                ))
                .all(|run| run.visible),
            "the bold text stays shown"
        );
    }

    // --- lists & blockquotes ----------------------------------------------

    #[test]
    fn a_list_bullet_persists_when_the_cursor_leaves() {
        let f = fonts();
        let src = "- item\nx";
        let r = rope(src);
        // Cursor on the second line, so a heading would hide its marker.
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let bullet = l.lines()[0].runs.first().expect("a bullet line has runs");
        assert!(
            matches!(bullet.kind, markdown::SpanKind::Marker),
            "the line starts with the bullet marker"
        );
        assert!(
            bullet.visible,
            "the bullet stays on screen off the line (unlike a heading)"
        );
    }

    #[test]
    fn a_quote_marker_persists_while_its_inline_markers_still_hide() {
        let f = fonts();
        let src = "> a `b` c\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let runs = &l.lines()[0].runs;
        let quote = runs.first().expect("a quote line has runs");
        assert!(
            matches!(quote.kind, markdown::SpanKind::Marker) && quote.visible,
            "the quote '> ' persists"
        );
        // The code span's backticks (inline markers past the block marker) still
        // collapse off the line.
        let inline_markers: Vec<_> = runs
            .iter()
            .skip(1)
            .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
            .collect();
        assert!(
            !inline_markers.is_empty() && inline_markers.iter().all(|run| !run.visible),
            "inline markers hide even though the block marker persists"
        );
    }

    // --- hanging indent & gutter decoration -------------------------------

    #[test]
    fn body_lines_have_no_indent_or_gutter() {
        let f = fonts();
        let l = layout(&f, &rope("plain text\nmore"));
        for line in l.lines() {
            assert_eq!(line.indent, 0, "body text is flush to the margin");
            assert_eq!(line.gutter, Gutter::None, "body text has no gutter");
        }
    }

    /// A wrap width that fits `prefix` but not one more cluster, so a line whose
    /// text begins with `prefix` breaks right after it.
    fn wrap_after(f: &Fonts, prefix: &str) -> i32 {
        let face = f.face(BODY, FontStyle::Regular);
        (render::text_advance(face, prefix) + render::text_advance(face, "b") / 2.0) as i32
    }

    #[test]
    fn a_wrapped_list_item_hangs_its_continuation_under_the_text() {
        let f = fonts();
        let marker_w = render::text_advance(f.face(BODY, FontStyle::Regular), "- ").round() as i32;
        let src = "- aaaaa bbbbb";
        let r = rope(src);
        // The list indent narrows the wrap, so widen by it to keep the split.
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "- aaaaa ") + LIST_INDENT,
            BODY,
            (0, r.len()),
        );
        let lines = l.lines();
        assert_eq!(
            lines.len(),
            2,
            "the bullet item wraps into two visual lines"
        );
        assert_eq!(
            lines[0].indent, LIST_INDENT,
            "the marker line sits at the small list indent"
        );
        assert_eq!(
            lines[1].indent,
            LIST_INDENT + marker_w,
            "the continuation hangs under the text, past indent and marker"
        );
        // The caret at the start of the continuation sits at the indented column,
        // not back at the margin.
        let cont = src.find("bbbbb").unwrap();
        let (x, _, _) = l.caret(&f, &r, cont);
        assert_eq!(
            x,
            LEFT + LIST_INDENT + marker_w,
            "continuation text starts past the indent and marker"
        );
    }

    #[test]
    fn a_nested_item_with_a_parent_is_a_list_item() {
        let f = fonts();
        // Child at depth 1 (two spaces) under a depth-0 parent.
        let r = rope("- a\n  - b");
        let l = layout(&f, &r);
        let child = &l.lines()[1];
        assert_eq!(
            child.gutter,
            Gutter::Bullet,
            "a nested item with a parent renders as a bullet"
        );
        assert_eq!(child.indent, LIST_INDENT, "and carries the list indent");
        let marker = child.runs.first().expect("a marker run");
        assert!(matches!(marker.kind, markdown::SpanKind::Marker));
        assert_eq!(
            marker.end - marker.start,
            4,
            "the marker spans the indentation and the dash (`  - `)"
        );
    }

    #[test]
    fn an_over_nested_item_without_a_parent_renders_as_body() {
        let f = fonts();
        // Depth 2 (four spaces) right after a depth-0 item: no depth-1 parent.
        let r = rope("- a\n    - b");
        let l = layout(&f, &r);
        let over = &l.lines()[1];
        assert_eq!(over.gutter, Gutter::None, "no bullet without a parent");
        assert_eq!(over.indent, 0, "and not given the list indent");
        assert!(
            over.runs
                .iter()
                .all(|run| matches!(run.kind, markdown::SpanKind::Text(_))),
            "the whole line is plain body text, no marker"
        );
    }

    #[test]
    fn outdenting_a_nested_item_reparents_its_child() {
        // Three items nested one inside the next: a (depth 0) > b (1) > c (2).
        // Reverse-indent the middle item b. Its child c should come along, so c
        // renders as a list item at one indent (under b), not orphaned into plain
        // text by losing its depth-1 parent.
        let f = fonts();
        let mut buf = crate::text::TextBuffer::new();
        buf.insert_str("- a\n  - b\n    - c");
        buf.set_cursor(8, false); // on the middle item "  - b"
        buf.nest_list_item(true); // Shift+Tab: outdent b
        let l = Layout::build(&f, buf.rope(), LEFT, TOP, 1_000_000, BODY, (0, 0));
        let third = &l.lines()[2];
        assert_eq!(
            third.gutter,
            Gutter::Bullet,
            "the third item still renders as a list item"
        );
        assert_eq!(third.indent, LIST_INDENT, "and carries the list indent");
        let marker = third.runs.first().expect("a marker run");
        assert_eq!(
            marker.end - marker.start,
            4,
            "rendered at one indent (`  - `), reparented under b"
        );
    }

    #[test]
    fn a_bullet_shows_its_dot_on_and_off_its_line() {
        let f = fonts();
        let src = "- item\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::Bullet,
            "a bullet point stands in once the cursor leaves"
        );
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::Bullet,
            "and the dot stays while the cursor edits the line"
        );
    }

    #[test]
    fn a_quote_shows_a_bar_off_its_line_and_a_raw_marker_on_it() {
        let f = fonts();
        let src = "> quoted\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::QuoteBar { marker: true },
            "the bar stands in for the '>' off-line"
        );
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::None,
            "raw '> ' shows while editing"
        );
    }

    #[test]
    fn a_wrapped_quote_keeps_the_bar_without_a_second_marker() {
        let f = fonts();
        let marker_w = render::text_advance(f.face(BODY, FontStyle::Regular), "> ").round() as i32;
        let src = "> aaaaa bbbbb\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "> aaaaa "),
            BODY,
            (off, off),
        );
        let lines = l.lines();
        assert!(
            lines.len() >= 3,
            "the quote wraps in two, then a plain line"
        );
        assert_eq!(lines[0].gutter, Gutter::QuoteBar { marker: true });
        assert_eq!(
            lines[1].gutter,
            Gutter::QuoteBar { marker: false },
            "the continuation keeps the bar but has no marker run of its own"
        );
        assert_eq!(lines[1].indent, marker_w, "and hangs under the text");
    }

    #[test]
    fn an_ordered_item_keeps_a_plain_gutter_but_still_indents() {
        let f = fonts();
        let marker_w = render::text_advance(f.face(BODY, FontStyle::Regular), "1. ").round() as i32;
        let r = rope("1. aaaaa bbbbb");
        // Cursor off the line: the number must still show (never substituted).
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "1. aaaaa ") + LIST_INDENT,
            BODY,
            (99, 99),
        );
        let lines = l.lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0].gutter,
            Gutter::None,
            "an ordered number is never replaced"
        );
        assert_eq!(lines[1].gutter, Gutter::None);
        assert_eq!(
            lines[0].indent, LIST_INDENT,
            "the number sits at the small list indent"
        );
        assert_eq!(
            lines[1].indent,
            LIST_INDENT + marker_w,
            "and wrapped lines still hang-indent past it"
        );
    }

    // --- Fenced code blocks -----------------------------------------------

    /// The first run kind of each visual line, for terse fence assertions.
    fn first_kinds(l: &Layout) -> Vec<Option<markdown::SpanKind>> {
        l.lines()
            .iter()
            .map(|line| line.runs.first().map(|r| r.kind))
            .collect()
    }

    #[test]
    fn a_fenced_block_marks_its_lines_as_code() {
        let f = fonts();
        let r = rope("```rust\nfn x() {}\n```\nbody");
        let l = layout(&f, &r);
        let code: Vec<bool> = l.lines().iter().map(|line| line.code_block).collect();
        assert_eq!(
            code,
            vec![true, true, true, false],
            "the fence lines and the code between are a block; the body is not"
        );
        // The ``` fences read as markers; the code line is a code run.
        assert_eq!(
            first_kinds(&l),
            vec![
                Some(markdown::SpanKind::Marker),                         // ```rust
                Some(markdown::SpanKind::Text(markdown::Emphasis::Code)), // fn x() {}
                Some(markdown::SpanKind::Marker),                         // ```
                Some(markdown::SpanKind::Text(markdown::Emphasis::Regular)), // body
            ]
        );
    }

    #[test]
    fn markdown_inside_a_fence_is_literal() {
        // A '#' line that would be an H1 is plain code inside a fence: body size,
        // a single code run, no heading marker.
        let f = fonts();
        let r = rope("```\n# not a heading\n```");
        let l = layout(&f, &r);
        let inner = &l.lines()[1];
        assert!(inner.code_block);
        assert_eq!(inner.size, BODY, "no heading scaling inside a code block");
        assert_eq!(inner.runs.len(), 1);
        assert_eq!(
            inner.runs[0].kind,
            markdown::SpanKind::Text(markdown::Emphasis::Code),
            "the '#' is code, not a heading marker"
        );
    }

    #[test]
    fn an_unclosed_fence_runs_to_the_end() {
        let f = fonts();
        let r = rope("```\n# H");
        let l = layout(&f, &r);
        assert!(
            l.lines().iter().all(|line| line.code_block),
            "with no closing fence every following line stays code"
        );
    }

    #[test]
    fn text_resumes_after_a_closed_fence() {
        let f = fonts();
        let r = rope("```\nx\n```\n# H");
        let l = layout(&f, &r);
        let last = l.lines().last().expect("a last line");
        assert!(
            !last.code_block,
            "the heading after the fence is normal again"
        );
        assert_eq!(last.size, h1_size(), "and is scaled as an H1");
    }

    /// The first run of a line (its fence marker, for fence lines).
    fn first_run(line: &VisualLine) -> &Run {
        line.runs.first().expect("a fence line has a run")
    }

    #[test]
    fn fence_backticks_hide_off_their_line_and_show_on_it() {
        let f = fonts();
        let src = "```rust\nfn x() {}\n```\nbody";
        let r = rope(src);
        // Cursor on the trailing body line: both fence lines are off-cursor.
        let off = src.find("body").unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let (open, close) = (first_run(&away.lines()[0]), first_run(&away.lines()[2]));
        assert!(matches!(open.kind, markdown::SpanKind::Marker) && !open.visible);
        assert!(
            matches!(close.kind, markdown::SpanKind::Marker) && !close.visible,
            "both ``` fences collapse when the cursor is outside the block"
        );
        // The code line between always shows, fence or no fence.
        assert!(away.lines()[1].runs.iter().all(|run| run.visible));
        // Cursor on the opening fence line reveals only that fence (per-line, like
        // a heading): the closing fence on another line stays hidden.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert!(
            first_run(&on.lines()[0]).visible,
            "the ``` shows on its line"
        );
        assert!(
            !first_run(&on.lines()[2]).visible,
            "the closing fence is on another line, still hidden"
        );
    }

    #[test]
    fn code_lines_inset_by_the_panel_padding() {
        let f = fonts();
        let r = rope("```\ncode here\n```");
        let l = layout(&f, &r);
        assert!(
            l.lines().iter().all(|line| line.indent == CODE_PAD_X),
            "every code-block line (fences and code) insets by the panel padding"
        );
        // A plain line carries no such inset.
        assert_eq!(layout(&f, &rope("plain")).lines()[0].indent, 0);
    }

    #[test]
    fn caret_on_a_code_line_starts_past_the_padding() {
        let f = fonts();
        let r = rope("```\nx\n```");
        let code_start = "```\n".len();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (code_start, code_start));
        let (x, _, _) = l.caret(&f, &r, code_start);
        assert_eq!(
            x,
            LEFT + CODE_PAD_X,
            "code text begins past the left panel padding"
        );
    }

    #[test]
    fn a_selection_over_a_code_line_yields_a_padded_rect() {
        // The selection rect for a code line sits at the padded column and has
        // width, so the highlight is visible inside the panel (the app paints the
        // panel beneath it). Regression guard for the selection-in-code-block fix.
        let f = fonts();
        let src = "```\nxyz\n```";
        let r = rope(src);
        let start = src.find("xyz").unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (start, start + 3));
        let rects = l.selection_rects(&f, &r, start, start + 3);
        assert_eq!(rects.len(), 1, "one rect for the single code line");
        assert_eq!(
            rects[0].x,
            LEFT + CODE_PAD_X,
            "rect starts past the padding"
        );
        assert!(rects[0].w > 0, "and has positive width");
    }

    #[test]
    fn inline_code_has_a_background_rect_but_a_fenced_block_does_not() {
        let f = fonts();
        // An inline code span yields one background panel, sitting past the line
        // start (after "a " and the opening backtick) with real size.
        let r = rope("a `b` c");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let rects = l.code_rects(&f, &r);
        assert_eq!(rects.len(), 1, "one panel behind the inline code span");
        assert!(rects[0].x > LEFT && rects[0].w > 0 && rects[0].h > 0);
        // A fenced block has no inline rects: its full-width panel covers it.
        let r2 = rope("```\ncode\n```");
        let l2 = layout(&f, &r2);
        assert!(
            l2.code_rects(&f, &r2).is_empty(),
            "block code uses the full-width panel, not per-run inline rects"
        );
    }

    #[test]
    fn heading_lines_carry_their_level_others_do_not() {
        let f = fonts();
        let r = rope("### h\nbody\n```\ncode\n```");
        let l = layout(&f, &r);
        let lines = l.lines();
        assert_eq!(lines[0].heading, Some(3), "the ### line is an H3");
        assert_eq!(lines[1].heading, None, "body has no heading level");
        assert!(
            lines[2..].iter().all(|ln| ln.heading.is_none()),
            "fence and code lines are never headings"
        );
    }

    #[test]
    fn heading_text_routes_to_the_bold_face() {
        // Monospace bold and regular share advances, so the bolding can't be seen
        // in widths; assert the face routing directly. (Robust to a family that
        // lacks a bold variant: face() then falls back, and both sides agree.)
        let f = fonts();
        let size = h1_size();
        let text = |em| markdown::SpanKind::Text(em);
        // On a heading: prose is bold, italic prose is bold-italic, already-bold
        // stays bold, but the `#` marker is never emboldened.
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Regular), true),
            f.face(size, FontStyle::Bold),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Italic), true),
            f.face(size, FontStyle::BoldItalic),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Bold), true),
            f.face(size, FontStyle::Bold),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, markdown::SpanKind::Marker, true),
            f.face(size, FontStyle::Regular),
        ));
        // Off a heading, prose stays regular.
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Regular), false),
            f.face(size, FontStyle::Regular),
        ));
    }

    #[test]
    fn a_thematic_break_draws_a_rule_off_its_line_and_dashes_on_it() {
        let f = fonts();
        let src = "a\n---\nb";
        let r = rope(src);
        // Cursor on "a": the rule line (line 1) is off-cursor, so it draws a rule
        // and its raw `---` collapses.
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let line = &away.lines()[1];
        assert!(line.rule, "the rule draws when the cursor is elsewhere");
        assert!(
            line.runs.iter().all(|run| !run.visible),
            "the raw --- is collapsed off the line"
        );
        // Cursor on the --- line: it shows the editable dashes, no rule.
        let on_off = src.find("---").unwrap();
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (on_off, on_off));
        let line = &on.lines()[1];
        assert!(!line.rule, "no rule while the cursor edits the line");
        assert!(
            line.runs
                .iter()
                .any(|run| run.visible && matches!(run.kind, markdown::SpanKind::Marker)),
            "the raw --- shows for editing"
        );
    }

    #[test]
    fn rule_at_reports_a_rendered_rule_for_a_click() {
        let f = fonts();
        let src = "a\n---\nb";
        let r = rope(src);
        // Cursor on "a": the rule line (line 1, bytes 2..5) is rendered.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            l.rule_at(&f, l.lines()[1].baseline as f32),
            Some((2, 5)),
            "a click on the rendered rule yields its content range"
        );
        // A body line is never a click-selectable rule.
        assert_eq!(l.rule_at(&f, l.lines()[0].baseline as f32), None);
        // Once the cursor edits the rule it shows the raw ---, so a click there
        // edits rather than reselecting.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (2, 2));
        assert_eq!(on.rule_at(&f, on.lines()[1].baseline as f32), None);
    }
}
