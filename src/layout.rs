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
//! heading's `#`s, a code span's backticks, emphasis `*`/`_`). Markers live-preview: on a
//! line the cursor has left they collapse (`Run::visible` is false, zero width),
//! drawing only the styled text; they reappear when the cursor returns. Wrapping
//! always measures the source text, so this collapse only shifts runs
//! horizontally within a line, never reflowing the document.

use crate::freetype::{FontStyle, Fonts};
use crate::grapheme;
use crate::markdown;
use crate::render;
use crate::rope::Rope;

/// Extra highlight width drawn past a line's text when the selection includes
/// that line's trailing newline, hinting the break itself is selected.
const NEWLINE_PAD: i32 = 8;

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

impl Run {
    /// The face style this run is measured and drawn in.
    pub fn font_style(&self) -> FontStyle {
        font_style(self.kind)
    }
}

/// Map a span/run kind to the face style it renders in. Markers and inline code
/// use the regular face (code is monospace already); only bold/italic emphasis
/// selects a variant. The single source of truth for both measuring (layout) and
/// drawing (the app render loop).
fn font_style(kind: markdown::SpanKind) -> FontStyle {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Bold) => FontStyle::Bold,
        markdown::SpanKind::Text(markdown::Emphasis::Italic) => FontStyle::Italic,
        markdown::SpanKind::Text(markdown::Emphasis::BoldItalic) => FontStyle::BoldItalic,
        _ => FontStyle::Regular,
    }
}

/// The face style of the (line-relative) span covering byte `off`. `spans` tile
/// `[0, len)` in order, so the first span ending past `off` is the one holding
/// it; a byte past the end (never asked for in practice) falls back to regular.
fn style_at(spans: &[markdown::Span], off: usize) -> FontStyle {
    spans
        .iter()
        .find(|s| off < s.end)
        .map_or(FontStyle::Regular, |s| font_style(s.kind))
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
            let size = markdown::font_size(markdown::style_of(&line), body_size);
            let metrics = fonts.metrics(size);
            // The line's inline spans (line-relative), and whether its markers
            // are revealed: shown while the active range touches the line, hidden
            // once the cursor leaves it.
            let spans = line_spans(&line);
            let revealed = active.0 <= le && active.1 >= ls;
            // `wrap_segments` returns ranges relative to the line; offset them
            // back to absolute buffer positions for the visual lines. It measures
            // the source text (every run in its own face), so clipping spans to
            // each segment never disagrees with the breaks, and reveal state,
            // which only hides markers, never reflows the document.
            let segments = wrap_segments(fonts, size, &line, &spans, wrap_width);
            let last = segments.len() - 1;
            for (k, (start, end)) in segments.into_iter().enumerate() {
                lines.push(VisualLine {
                    start: ls + start,
                    end: ls + end,
                    baseline: box_top + metrics.ascent,
                    // Only the final segment of a logical line that has a
                    // trailing '\n' is a hard break.
                    hard: k == last && has_newline,
                    size,
                    runs: clip_runs(&spans, start, end, ls, revealed),
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
        let x = self.left as f32 + self.advance_to(fonts, rope, line, offset);
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
            let x0 = self.left as f32 + self.advance_to(fonts, rope, line, sel_start);
            let x1 = self.left as f32 + self.advance_to(fonts, rope, line, sel_end);
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
        let mut acc = self.left as f32;
        // Past every glyph: the end of the last visible run, or the line start
        // for an empty line.
        let mut last = line.start;
        for run in &line.runs {
            if !run.visible {
                continue;
            }
            let face = fonts.face(line.size, run.font_style());
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
            let face = fonts.face(line.size, run.font_style());
            x += render::text_advance(face, &rope.byte_slice(run.start..end));
        }
        x
    }

    fn last_line(&self) -> &VisualLine {
        // build always pushes at least one line, so this never panics.
        &self.lines[self.lines.len() - 1]
    }
}

/// The full inline span list of a logical `line`, line-relative: a leading
/// block marker (a heading's `#`s, if any) followed by the inline spans of the
/// content past it, offset back over the marker. Tiles `[0, line.len())`.
fn line_spans(line: &str) -> Vec<markdown::Span> {
    let marker = markdown::marker_len(line);
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

/// Clip line-relative `spans` to the visual segment `[seg_start, seg_end)`,
/// offsetting to absolute by `ls`. A marker run is hidden unless `revealed`; a
/// text run is always shown. Empty overlaps are dropped, so the runs tile the
/// segment.
fn clip_runs(
    spans: &[markdown::Span],
    seg_start: usize,
    seg_end: usize,
    ls: usize,
    revealed: bool,
) -> Vec<Run> {
    let mut runs = Vec::new();
    for span in spans {
        let s = span.start.max(seg_start);
        let e = span.end.min(seg_end);
        if s >= e {
            continue;
        }
        let is_marker = matches!(span.kind, markdown::SpanKind::Marker);
        runs.push(Run {
            start: ls + s,
            end: ls + e,
            kind: span.kind,
            // A marker shows only on a revealed line; text always shows.
            visible: !is_marker || revealed,
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
fn wrap_segments(
    fonts: &Fonts,
    size: u32,
    line: &str,
    spans: &[markdown::Span],
    wrap_width: i32,
) -> Vec<(usize, usize)> {
    if wrap_width <= 0 || line.is_empty() {
        return vec![(0, line.len())];
    }
    // Measure each cluster once: its offset within the line, width in its run's
    // face, and whether it is a space (a wrap opportunity).
    let clusters: Vec<(usize, f32, bool)> = grapheme::graphemes(line)
        .into_iter()
        .map(|(off, cluster)| {
            let face = fonts.face(size, style_at(spans, off));
            (off, render::text_advance(face, cluster), cluster == " ")
        })
        .collect();
    if clusters.is_empty() {
        return vec![(0, line.len())];
    }

    let limit = wrap_width as f32;
    let mut segments = Vec::new();
    let mut i = 0;
    while i < clusters.len() {
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
}
