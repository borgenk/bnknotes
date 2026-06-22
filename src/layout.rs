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

use crate::freetype::Face;
use crate::grapheme;
use crate::render;

/// Extra highlight width drawn past a line's text when the selection includes
/// that line's trailing newline, hinting the break itself is selected.
const NEWLINE_PAD: i32 = 8;

/// One on-screen line: the byte range of its text (excluding any trailing
/// `'\n'`), the y of its baseline, and whether a hard newline follows it.
pub struct VisualLine {
    pub start: usize,
    pub end: usize,
    pub baseline: i32,
    /// True if a `'\n'` immediately follows `end` (a hard break); false for a
    /// soft wrap or the final line of the buffer.
    pub hard: bool,
}

/// An axis-aligned rectangle in window pixels.
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

pub struct Layout {
    lines: Vec<VisualLine>,
    left: i32,
    ascent: i32,
    descent: i32,
}

impl Layout {
    /// Lay `text` out with the first baseline at `top + ascent` and successive
    /// baselines `line_height` apart. `left` is the text's left margin and
    /// `wrap_width` the pixel width to wrap within (<= 0 disables wrapping).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        face: &Face,
        text: &str,
        left: i32,
        top: i32,
        ascent: i32,
        descent: i32,
        line_height: i32,
        wrap_width: i32,
    ) -> Self {
        let mut lines = Vec::new();
        let mut baseline = top + ascent;
        let mut ls = 0usize;
        loop {
            let rel = text[ls..].find('\n');
            let le = rel.map_or(text.len(), |r| ls + r);
            let segments = wrap_segments(face, text, ls, le, wrap_width);
            let last = segments.len() - 1;
            for (k, (start, end)) in segments.into_iter().enumerate() {
                lines.push(VisualLine {
                    start,
                    end,
                    baseline,
                    // Only the final segment of a logical line that has a
                    // trailing '\n' is a hard break.
                    hard: k == last && rel.is_some(),
                });
                baseline += line_height;
            }
            match rel {
                Some(r) => ls = ls + r + 1,
                None => break,
            }
        }
        Self {
            lines,
            left,
            ascent,
            descent,
        }
    }

    pub fn lines(&self) -> &[VisualLine] {
        &self.lines
    }

    /// The byte range `(start, end)` of the visual line containing `offset`,
    /// for Home/End motion.
    pub fn line_bounds(&self, offset: usize) -> (usize, usize) {
        let line = self.line_for(offset);
        (line.start, line.end)
    }

    /// The caret rectangle (its x, top y, and height) for the cursor at `offset`.
    pub fn caret(&self, face: &Face, text: &str, offset: usize) -> (i32, i32, i32) {
        let line = self.line_for(offset);
        let x = self.left as f32 + render::text_advance(face, &text[line.start..offset]);
        (
            x.round() as i32,
            line.baseline - self.ascent,
            self.ascent + self.descent,
        )
    }

    /// The byte offset nearest a click at window `(x, y)`.
    pub fn offset_at(&self, face: &Face, text: &str, x: f32, y: f32) -> usize {
        let line = self.line_for_y(y);
        self.hit_line(face, text, line, x)
    }

    /// The byte offset reached by moving `dir` lines (-1 up, +1 down) from
    /// `offset`, aiming for window x `goal_x`. Clamps at the first and last line.
    pub fn offset_vertical(
        &self,
        face: &Face,
        text: &str,
        offset: usize,
        dir: i32,
        goal_x: f32,
    ) -> usize {
        let idx = self.line_index_for(offset) as i32;
        let target = idx + dir;
        if target < 0 || target as usize >= self.lines.len() {
            return offset;
        }
        self.hit_line(face, text, &self.lines[target as usize], goal_x)
    }

    /// The highlight rectangles covering the byte range `[start, end)`, one per
    /// visual line it touches.
    pub fn selection_rects(&self, face: &Face, text: &str, start: usize, end: usize) -> Vec<Rect> {
        let mut rects = Vec::new();
        for line in &self.lines {
            let sel_start = start.max(line.start);
            let sel_end = end.min(line.end);
            // A hard line's newline is selected when the range reaches past the
            // line's text into the break. Soft wraps have no newline to select.
            let newline_selected = line.hard && start <= line.end && end > line.end;
            if sel_start >= sel_end && !newline_selected {
                continue;
            }
            let x0 = self.left as f32 + render::text_advance(face, &text[line.start..sel_start]);
            let x1 = self.left as f32 + render::text_advance(face, &text[line.start..sel_end]);
            let mut w = (x1 - x0).round() as i32;
            if newline_selected {
                w += NEWLINE_PAD;
            }
            rects.push(Rect {
                x: x0.round() as i32,
                y: line.baseline - self.ascent,
                w: w.max(1),
                h: self.ascent + self.descent,
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
    /// last line for clicks above or below the text.
    fn line_for_y(&self, y: f32) -> &VisualLine {
        self.lines
            .iter()
            .find(|l| y < (l.baseline + self.descent) as f32)
            .unwrap_or_else(|| self.last_line())
    }

    /// The byte offset within `line` nearest window x `x`, snapped to a grapheme
    /// cluster boundary (a click in a cluster's left half lands before it).
    fn hit_line(&self, face: &Face, text: &str, line: &VisualLine, x: f32) -> usize {
        let slice = &text[line.start..line.end];
        let mut acc = self.left as f32;
        for (off, cluster) in grapheme::graphemes(slice) {
            let w = render::text_advance(face, cluster);
            if x < acc + w / 2.0 {
                return line.start + off;
            }
            acc += w;
        }
        line.end
    }

    fn last_line(&self) -> &VisualLine {
        // build always pushes at least one line, so this never panics.
        &self.lines[self.lines.len() - 1]
    }
}

/// Greedily word-wrap the logical line `text[ls..le]` to `wrap_width` pixels,
/// returning the byte ranges of its visual segments. Breaks after the last space
/// that fits; a word wider than `wrap_width` breaks at a grapheme boundary so
/// every segment holds at least one cluster. `wrap_width <= 0` disables wrapping.
fn wrap_segments(
    face: &Face,
    text: &str,
    ls: usize,
    le: usize,
    wrap_width: i32,
) -> Vec<(usize, usize)> {
    if wrap_width <= 0 || ls == le {
        return vec![(ls, le)];
    }
    // Measure each cluster once: its absolute start, width, and whether it is a
    // space (a wrap opportunity).
    let clusters: Vec<(usize, f32, bool)> = grapheme::graphemes(&text[ls..le])
        .into_iter()
        .map(|(off, cluster)| {
            (
                ls + off,
                render::text_advance(face, cluster),
                cluster == " ",
            )
        })
        .collect();
    if clusters.is_empty() {
        return vec![(ls, le)];
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
            segments.push((seg_start, le));
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

    fn face() -> Face {
        Face::open_default(32).expect("a default font")
    }

    /// Build with round metrics and wrapping effectively disabled.
    fn layout(text: &str) -> Layout {
        Layout::build(&face(), text, 40, 10, 24, 6, 30, 1_000_000)
    }

    /// A wrap width that fits "aaa " but not a following cluster, so "aaa bbb"
    /// splits deterministically into "aaa " and "bbb".
    fn tight_wrap(f: &Face) -> i32 {
        (render::text_advance(f, "aaa ") + render::text_advance(f, "b") / 2.0) as i32
    }

    #[test]
    fn splits_logical_lines_with_stacked_baselines() {
        let l = layout("ab\ncd\n");
        let lines = l.lines();
        assert_eq!(lines.len(), 3, "trailing newline yields an empty last line");
        assert_eq!((lines[0].start, lines[0].end), (0, 2));
        assert_eq!((lines[1].start, lines[1].end), (3, 5));
        assert_eq!((lines[2].start, lines[2].end), (6, 6));
        assert_eq!(lines[0].baseline, 34);
        assert_eq!(lines[1].baseline, 64);
        assert_eq!(lines[2].baseline, 94);
        // The first two are hard breaks; the empty last line is not.
        assert!(lines[0].hard && lines[1].hard && !lines[2].hard);
    }

    #[test]
    fn empty_text_is_one_line() {
        let l = layout("");
        assert_eq!(l.lines().len(), 1);
        assert_eq!((l.lines()[0].start, l.lines()[0].end), (0, 0));
        assert!(!l.lines()[0].hard);
    }

    #[test]
    fn caret_sits_at_the_left_margin_for_a_line_start() {
        let f = face();
        let l = layout("hello\nworld");
        let (x0, top0, h) = l.caret(&f, "hello\nworld", 0);
        assert_eq!(x0, 40, "first column is the left margin");
        assert_eq!(top0, 34 - 24, "caret top is baseline - ascent");
        assert_eq!(h, 30, "caret spans ascent + descent");
        let (x1, top1, _) = l.caret(&f, "hello\nworld", 6);
        assert_eq!(x1, 40, "second line also starts at the margin");
        assert_eq!(top1, 64 - 24, "and one line lower");
    }

    #[test]
    fn caret_advances_along_a_line() {
        let f = face();
        let text = "mm";
        let l = layout(text);
        let (x0, _, _) = l.caret(&f, text, 0);
        let (x1, _, _) = l.caret(&f, text, 1);
        let (x2, _, _) = l.caret(&f, text, 2);
        assert!(x1 > x0 && x2 > x1, "caret x grows with the column");
    }

    #[test]
    fn click_maps_to_the_clicked_line_and_column() {
        let f = face();
        let text = "abc\ndefg";
        let l = layout(text);
        let off = l.offset_at(&f, text, 10_000.0, 64.0);
        assert_eq!(off, text.len(), "past the last glyph of line two -> end");
        let off = l.offset_at(&f, text, -100.0, 34.0);
        assert_eq!(off, 0);
        let off = l.offset_at(&f, text, -100.0, 9_000.0);
        assert_eq!(off, 4, "start of the last line");
    }

    #[test]
    fn vertical_motion_changes_line_and_clamps() {
        let f = face();
        let text = "abc\ndef\nghi";
        let l = layout(text);
        let (x, _, _) = l.caret(&f, text, 5);
        let up = l.offset_vertical(&f, text, 5, -1, x as f32);
        assert_eq!(up, 1, "same column on the line above");
        let down = l.offset_vertical(&f, text, 5, 1, x as f32);
        assert_eq!(down, 9, "same column on the line below");
        assert_eq!(l.offset_vertical(&f, text, 1, -1, x as f32), 1);
    }

    #[test]
    fn selection_on_one_line_is_one_rect() {
        let f = face();
        let text = "hello";
        let l = layout(text);
        let rects = l.selection_rects(&f, text, 1, 4);
        assert_eq!(rects.len(), 1);
        assert!(rects[0].w > 0);
        assert_eq!(rects[0].h, 30);
        assert_eq!(rects[0].y, 34 - 24);
    }

    #[test]
    fn selection_across_hard_lines_is_one_rect_per_line() {
        let f = face();
        let text = "ab\ncd\nef";
        let l = layout(text);
        let rects = l.selection_rects(&f, text, 1, 7);
        assert_eq!(rects.len(), 3, "three lines touched");
        assert!(rects[0].w >= NEWLINE_PAD, "hard newline adds the pad");
    }

    #[test]
    fn long_line_soft_wraps_at_spaces() {
        let f = face();
        let text = "aaa bbb";
        let l = Layout::build(&f, text, 0, 0, 24, 6, 30, tight_wrap(&f));
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
        let f = face();
        let text = "mmmmmmmmmm"; // one word, no spaces
        let wrap = (render::text_advance(&f, "m") * 3.0) as i32;
        let l = Layout::build(&f, text, 0, 0, 24, 6, 30, wrap);
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
        let f = face();
        let text = "aaa bbb";
        let l = Layout::build(&f, text, 0, 0, 24, 6, 30, tight_wrap(&f));
        // Offset 4 is the soft-wrap boundary; the caret shows at the start of
        // the second visual line, not the end of the first.
        let (x, top, _) = l.caret(&f, text, 4);
        assert_eq!(x, 0, "left margin of the wrapped row");
        assert_eq!(top, (24 + 30) - 24, "second baseline minus ascent");
    }

    #[test]
    fn selection_across_a_soft_wrap_has_no_newline_pad() {
        let f = face();
        let text = "aaa bbb";
        let l = Layout::build(&f, text, 0, 0, 24, 6, 30, tight_wrap(&f));
        let rects = l.selection_rects(&f, text, 0, 7);
        assert_eq!(rects.len(), 2, "one rect per visual line");
        let line0_text_w = render::text_advance(&f, "aaa ").round() as i32;
        assert!(
            (rects[0].w - line0_text_w).abs() <= 1,
            "a soft wrap adds no newline pad"
        );
    }
}
