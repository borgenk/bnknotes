use crate::display::{self, DisplayList, DrawCmd};
use crate::find::{self, FindState};
use crate::freetype::{FaceKey, FontStyle, Fonts, Metrics};
use crate::grapheme;
use crate::layout::{face_key, image_button_rect, Gutter, Layout, Rect};
use crate::markdown;
use crate::pane::TreePane;
use crate::scroll;
use crate::shape;
use crate::theme::{
    BACKGROUND, BULLET, CARET_COLOR, CARET_WIDTH, CODE_BG, CODE_COLOR, CODE_RADIUS, FIND_BG,
    FIND_ERROR_COLOR, FIND_MATCH_COLOR, FIND_TICK_COLOR, FONT_SIZE, FOREGROUND, HEADING_COLORS,
    IMAGE_BORDER_COLOR, IMAGE_BORDER_THICKNESS, IMAGE_BUTTON_BG, IMAGE_BUTTON_RADIUS, LINK_COLOR,
    MARGIN, MARKER_COLOR, QUOTE_BAR_COLOR, QUOTE_BAR_WIDTH, RULE_COLOR, RULE_THICKNESS,
    SELECTION_COLOR, SPLITTER_COLOR, TICK_HEIGHT, TICK_WIDTH, TREE_BG, TREE_DIR_COLOR,
    TREE_HOVER_BG, TREE_INDENT, TREE_PAD,
};

/// The per-frame inputs [`build_display_list`] needs beyond the layout itself:
/// the surface size, the scroll offset, the live selection and cursor, and
/// whether the caret is on this blink. Bundled so the build signature stays
/// readable and so the perf harness can drive a build without a live surface.
pub(crate) struct PaintParams<'a> {
    pub width: usize,
    pub height: usize,
    pub scroll_y: i32,
    pub selection: Option<(usize, usize)>,
    pub cursor: usize,
    pub caret_visible: bool,
    /// The find bar's paint state while it is open; `None` draws no bar and no
    /// match highlights.
    pub find: Option<FindPaint<'a>>,
}

/// What the painter needs from an open find bar, borrowed from the app's
/// [`FindState`] for the frame being built.
pub(crate) struct FindPaint<'a> {
    /// The query as typed; the bar shows the tail that fits.
    pub query: &'a str,
    /// The replacement text when the replace row is shown.
    pub replace: Option<&'a str>,
    /// Whether the replace field (rather than the query) has the keyboard.
    pub focus_replace: bool,
    /// The focused field's whole content is selected (Ctrl+A or a reopened
    /// query); drawn on a selection fill since the next edit replaces it.
    pub selected: bool,
    /// Mode flags, drawn bright when on.
    pub word: bool,
    pub regex: bool,
    /// The query is an invalid regular expression.
    pub error: bool,
    /// Byte ranges of every match, in document order.
    pub matches: &'a [(usize, usize)],
    /// Index into `matches` of the current match. It is not highlighted here
    /// because the app mirrors it into the buffer selection.
    pub current: usize,
    /// Content-space top of each match's line, from the line index, so the tick
    /// strip places a match that sits outside the painted window correctly.
    pub tick_tops: &'a [i32],
    /// Total document content height, for the tick strip's proportional mapping.
    pub doc_height: i32,
}

/// A [`DrawCmd::Text`] for `text` from `x` along `baseline` in `face`/`color`,
/// where `end` is the pen x after the run (its drawn width). The damage bounds are
/// the run's box padded for glyph ink that overhangs the advance, so a diff or a
/// clip never shears a glyph's edge.
fn text_cmd(
    face: FaceKey,
    x: i32,
    baseline: i32,
    end: i32,
    color: u32,
    text: String,
    m: Metrics,
) -> DrawCmd {
    let size = face.size();
    let pad_x = (size as i32 / 2).max(2);
    let pad_y = (size as i32 / 4).max(1);
    let bounds = Rect {
        x: x - pad_x,
        y: baseline - m.ascent - pad_y,
        w: (end - x).max(0) + 2 * pad_x,
        h: m.ascent + m.descent + 2 * pad_y,
    };
    DrawCmd::Text {
        bounds,
        x,
        baseline,
        face,
        color,
        text,
    }
}

/// The painter's view of an open find bar, or `None` when it is closed.
/// `tick_tops` and `doc_height` come from the caller's line index, so the tick
/// strip does not depend on which lines the windowed layout currently holds.
pub(crate) fn build_find_paint<'a>(
    find: &'a FindState,
    tick_tops: &'a [i32],
    doc_height: i32,
) -> Option<FindPaint<'a>> {
    if !find.is_open() {
        return None;
    }
    Some(FindPaint {
        query: find.query(),
        replace: find.show_replace().then(|| find.replacement()),
        focus_replace: find.focus() == find::Field::Replace,
        selected: find.selected(),
        word: find.word(),
        regex: find.regex(),
        error: find.error(),
        matches: find.matches(),
        current: find.current(),
        tick_tops,
        doc_height,
    })
}

/// Draw a splitter grab bar: its gap cleared to the background with a thin
/// divider line down the center. Drawn last, over any content that overhangs.
pub(crate) fn push_splitter(list: &mut DisplayList, rect: Rect) {
    list.push(DrawCmd::Fill {
        rect,
        color: BACKGROUND,
    });
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: rect.x + rect.w / 2,
            y: rect.y,
            w: 1,
            h: rect.h,
        },
        color: SPLITTER_COLOR,
    });
}

/// Draw the file-tree sidebar into its column `rect`: a darker background, then
/// each visible row (a folder marker plus the name), indented by depth and
/// offset by the tree's own scroll. Updates `content_height` for scroll
/// clamping. Rows do not wrap; a long name that overhangs is covered by the
/// editor's own background, which is drawn after this.
pub(crate) fn push_tree(
    list: &mut DisplayList,
    fonts: &Fonts,
    tree: &mut TreePane,
    rect: Rect,
    focused: bool,
) {
    list.push(DrawCmd::Fill {
        rect,
        color: TREE_BG,
    });
    let m = fonts.metrics(FONT_SIZE);
    let row_h = m.line_height.max(1);
    let rows = tree.tree.rows();
    tree.content_height = rows.len() as i32 * row_h + 2 * TREE_PAD;
    tree.scroll_y = scroll::clamp(tree.scroll_y, tree.content_height, rect.h);
    let face = FaceKey::Prose {
        size: FONT_SIZE,
        style: FontStyle::Regular,
    };
    let text_right = rect.x + rect.w - TREE_PAD;
    for (i, row) in rows.iter().enumerate() {
        let top = rect.y + TREE_PAD + i as i32 * row_h - tree.scroll_y;
        if top + row_h <= rect.y || top >= rect.y + rect.h {
            continue;
        }
        // The keyboard-selected row (while the tree is focused) reads as a
        // selection; a mouse-hovered row gets a subtler tint.
        let highlight = if focused && tree.selected == i {
            Some(SELECTION_COLOR)
        } else if tree.hovered == Some(i) {
            Some(TREE_HOVER_BG)
        } else {
            None
        };
        if let Some(color) = highlight {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: rect.x,
                    y: top,
                    w: rect.w,
                    h: row_h,
                },
                color,
            });
        }
        let x = rect.x + TREE_PAD + row.depth as i32 * TREE_INDENT;
        let baseline = top + m.ascent;
        let (marker, color) = if row.is_dir {
            (if row.expanded { "▾ " } else { "▸ " }, TREE_DIR_COLOR)
        } else {
            ("", FOREGROUND)
        };
        let label = format!("{marker}{}", row.name);
        list.push(text_cmd(face, x, baseline, text_right, color, label, m));
    }
}

/// Build the frame as a display list in screen (post-scroll) pixels: background,
/// code-block panels, inline-code panels, the selection highlight, the text (run
/// by run, in each run's face and color, with markers collapsed or revealed as
/// the layout decided), and the caret, in that stacking order. Pure CPU work over
/// the layout and rope, with no Wayland or `State` dependency. Off-screen lines
/// and spans are skipped, so the list (and the cost) track the viewport, not the
/// document.
pub(crate) fn build_display_list(
    fonts: &Fonts,
    layout: &Layout,
    rope: &crate::rope::Rope,
    p: &PaintParams,
) -> DisplayList {
    let (width, scroll_y) = (p.width, p.scroll_y);
    let view_h = p.height as i32;
    // The content's left edge (the column's inner margin). Text, panels, and
    // images hang off this so a pane's content sits in its own column; the
    // background clears from the column's outer edge, `MARGIN` to its left.
    let left = layout.left();
    // Roughly a few commands per visual line (panel, runs, decorations) plus the
    // background and caret; one upfront allocation beats growing the list mid-build.
    let mut list: DisplayList = Vec::with_capacity(layout.lines().len() * 3 + 8);

    // The background, full surface, first so a clipped repaint clears its region
    // before the rest of the list draws over it.
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: left - MARGIN,
            y: 0,
            w: width as i32,
            h: view_h,
        },
        color: BACKGROUND,
    });

    // Block backgrounds paint first, beneath both the selection highlight and the
    // text: a fenced code block sits on a full-width tinted panel, one line box
    // tall so consecutive lines tile into a continuous block, and that panel must
    // sit behind the selection rather than cover it (otherwise a selection inside
    // the block would vanish). A collapsed fence row still paints, so an unfocused
    // block keeps blank top and bottom rows that read as vertical padding. The
    // block's first row rounds its top corners and its last row its bottom ones, so
    // the whole panel reads as one softly rounded block.
    let lines = layout.lines();
    let panel_w = (width as i32 - 2 * MARGIN).max(0);
    for (i, line) in lines.iter().enumerate() {
        if !line.code_block {
            continue;
        }
        let m = fonts.metrics(line.size);
        let top = line.baseline - scroll_y - m.ascent;
        let h = m.line_height;
        if top + h <= 0 || top >= view_h {
            continue;
        }
        let first = i == 0 || !lines[i - 1].code_block;
        let last = i + 1 >= lines.len() || !lines[i + 1].code_block;
        let corners = match (first, last) {
            (true, true) => display::RoundedCorners::Both,
            (true, false) => display::RoundedCorners::Top,
            (false, true) => display::RoundedCorners::Bottom,
            (false, false) => display::RoundedCorners::None,
        };
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: left,
                y: top,
                w: panel_w,
                h,
            },
            color: CODE_BG,
            radius: CODE_RADIUS,
            corners,
        });
    }

    // Inline code spans get their own tinted panel, also beneath the selection so a
    // selection over inline code shows through (the glyphs still draw on top in the
    // text loop). Block code is excluded here; its full-width panel above covers it.
    for rect in layout.code_rects(fonts, rope) {
        let y = rect.y - scroll_y;
        if y + rect.h <= 0 || y >= view_h {
            continue;
        }
        list.push(DrawCmd::Fill {
            rect: Rect { y, ..rect },
            color: CODE_BG,
        });
    }

    // Everything is laid out in content space; the viewport scroll shifts it up by
    // `scroll_y`. Off-screen rectangles are dropped from the list.
    if let Some((start, end)) = p.selection {
        for rect in layout.selection_rects(fonts, rope, start, end) {
            let y = rect.y - scroll_y;
            if y + rect.h <= 0 || y >= view_h {
                continue;
            }
            list.push(DrawCmd::Fill {
                rect: Rect { y, ..rect },
                color: SELECTION_COLOR,
            });
        }
    }

    // Find matches, filled like a dimmer selection beneath the text. The current
    // match is skipped: the app mirrors it into the buffer selection, which was
    // just painted. Only matches inside the visible byte range are measured, so a
    // short query with thousands of hits does not price every frame by them.
    if let Some(find) = &p.find {
        let visible = visible_byte_range(fonts, layout, scroll_y, view_h);
        let lo = find.matches.partition_point(|&(_, end)| end <= visible.0);
        let hi = find
            .matches
            .partition_point(|&(start, _)| start < visible.1);
        for (i, &(start, end)) in find.matches.iter().enumerate().take(hi).skip(lo) {
            if i == find.current {
                continue;
            }
            for rect in layout.selection_rects(fonts, rope, start, end) {
                let y = rect.y - scroll_y;
                if y + rect.h <= 0 || y >= view_h {
                    continue;
                }
                list.push(DrawCmd::Fill {
                    rect: Rect { y, ..rect },
                    color: FIND_MATCH_COLOR,
                });
            }
        }
    }

    // The text, one visual line at a time, each at its own font size, materialized
    // from the rope. A line is a sequence of inline runs: regular text, code spans
    // (drawn in the code face on a tinted panel), and the delimiter markers, which
    // are dimmed on the cursor's line and collapse to nothing (skipped) once it
    // leaves, the live-preview reveal that `layout` bakes into each run's `visible`.
    for line in lines {
        let m = fonts.metrics(line.size);
        let baseline = line.baseline - scroll_y;
        // An image line blits its decoded picture into its (taller-than-text) box,
        // centered with the layout's vertical padding, and nothing else draws on the
        // line. Cull on the full box height so a tall image straddling a viewport
        // edge still paints. A picture that has not decoded leaves the box blank.
        if let Some(img) = line.image {
            let box_top = baseline - m.ascent;
            if box_top < view_h && box_top + img.box_height() > 0 {
                let pic = Rect {
                    x: left + line.indent,
                    y: box_top + (img.box_height() - img.height) / 2,
                    w: img.width,
                    h: img.height,
                };
                // Blit the cached display-size bitmap (resampled once, not per
                // frame); a no-op at paint time if the image is not decoded.
                list.push(DrawCmd::Image {
                    rect: pic,
                    path: rope.byte_slice(img.path_start..img.path_end),
                });
                // A selected image gets an accent border (four thin edge fills) and
                // a corner toggle button: a rounded panel with a centered `</>`
                // "source" glyph, the affordance to reveal its markdown.
                if img.selected {
                    push_image_chrome(&mut list, fonts, &pic);
                }
            }
            continue;
        }
        // Skip lines fully above or below the viewport, so a frame only rasterizes
        // what is on screen.
        if baseline + m.descent <= 0 || baseline - m.ascent >= view_h {
            continue;
        }
        // A blockquote draws a bar down its gutter, on every wrapped line.
        if matches!(line.gutter, Gutter::QuoteBar { .. }) {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: left,
                    y: baseline - m.ascent,
                    w: QUOTE_BAR_WIDTH,
                    h: m.line_height,
                },
                color: QUOTE_BAR_COLOR,
            });
        }
        // A thematic break draws a horizontal line centered in its row; the raw `---`
        // is collapsed (its runs are hidden), so nothing else paints on the line.
        if line.rule {
            let mid = baseline - m.ascent + m.line_height / 2;
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: left,
                    y: mid - RULE_THICKNESS / 2,
                    w: (width as i32 - 2 * MARGIN).max(0),
                    h: RULE_THICKNESS,
                },
                color: RULE_COLOR,
            });
        }
        // Content is inset by the line's hanging indent (0 unless this is a wrapped
        // list/quote continuation).
        let heading = line.heading.is_some();
        // A completed task's text reads dim and struck through, over the whole
        // item (wrapped continuations included), so done items scan as done.
        let done = line.task == Some(true);
        let mut pen = left + line.indent;
        for (i, run) in line.runs.iter().enumerate() {
            if !run.visible {
                continue;
            }
            // Each run draws in its own face, so bold/italic emphasis is carried by
            // the glyphs, not color: only code and markers recolor, the rest
            // (regular and emphasized prose) is FG.
            let fk = face_key(line.size, run.kind, heading);
            let face = fonts.face_for(fk);
            let slice = rope.byte_slice(run.start..run.end);
            // The block marker (the first run on a list/quote line) gives way to its
            // gutter decoration: a bullet glyph for a list, nothing for a quote (the
            // bar above replaces it). Either keeps the source width, so text never
            // shifts.
            if i == 0 && matches!(run.kind, markdown::SpanKind::Marker) {
                let advance = shape::text_advance(face, &slice).round() as i32;
                match line.gutter {
                    Gutter::Bullet => {
                        // Place the marker's leading indentation first, so a nested
                        // item's dot sits at its depth, then the dot in place of the
                        // dash. The dot stands in for the `- ` marker, so it takes
                        // the muted marker color (MARKER_COLOR), not the body FG.
                        let indent = slice.len() - slice.trim_start().len();
                        let dot_x =
                            pen + shape::text_advance(face, &slice[..indent]).round() as i32;
                        let end = (dot_x as f32 + shape::text_advance(face, BULLET)).round() as i32;
                        list.push(text_cmd(
                            fk,
                            dot_x,
                            baseline,
                            end,
                            MARKER_COLOR,
                            BULLET.to_string(),
                            m,
                        ));
                        pen += advance;
                        continue;
                    }
                    // A task item's raw `- [ ]` gives way to a checkbox drawn
                    // with fills (the prose face carries no ☐/☑ glyphs), placed
                    // by the same geometry clicks hit-test against. The pen
                    // still advances by the full marker, so the text never
                    // shifts between the box and the revealed source.
                    Gutter::Checkbox { checked } => {
                        if let Some(rect) = layout.checkbox_rect(fonts, rope, line) {
                            let rect = Rect {
                                y: rect.y - scroll_y,
                                ..rect
                            };
                            push_checkbox(&mut list, rect, checked);
                        }
                        pen += advance;
                        continue;
                    }
                    Gutter::QuoteBar { marker: true } => {
                        pen += advance;
                        continue;
                    }
                    _ => {}
                }
            }
            let color = match run.kind {
                markdown::SpanKind::Marker => MARKER_COLOR,
                // Every text run of a done task dims to the marker color, so
                // the whole item (its code and links included) reads muted.
                markdown::SpanKind::Text(_) if done => MARKER_COLOR,
                markdown::SpanKind::Text(markdown::Emphasis::Code) => CODE_COLOR,
                markdown::SpanKind::Text(markdown::Emphasis::Link) => LINK_COLOR,
                // Heading prose takes the level's color (the marker and any inline
                // code on the line keep their own); body prose is FG.
                markdown::SpanKind::Text(_) => match line.heading {
                    Some(level) => HEADING_COLORS
                        .get(level.saturating_sub(1) as usize)
                        .copied()
                        .unwrap_or(FOREGROUND),
                    None => FOREGROUND,
                },
            };
            // Code backgrounds (inline panels and the full-width block panel) are
            // emitted in the pre-pass above, beneath the selection; here we only
            // draw the glyphs over them.
            let run_x = pen;
            let end = (run_x as f32 + shape::text_advance(face, &slice)).round() as i32;
            list.push(text_cmd(fk, run_x, baseline, end, color, slice, m));
            pen = end;
            // Inline decorations over the glyphs, each spanning the run's drawn
            // width: a link is underlined and struck text gets a line through its
            // middle, both in the run's own color. Thickness scales with the
            // line's size so it stays proportionate on larger headings. A done
            // task strikes through all of its text instead.
            let deco = match run.kind {
                markdown::SpanKind::Text(_) if done => Some(baseline - m.ascent / 3),
                markdown::SpanKind::Text(markdown::Emphasis::Link) => {
                    Some(baseline + m.descent / 3)
                }
                markdown::SpanKind::Text(markdown::Emphasis::Strikethrough) => {
                    Some(baseline - m.ascent / 3)
                }
                _ => None,
            };
            if let Some(y) = deco {
                let thickness = (line.size as i32 / 14).max(1);
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: run_x,
                        y,
                        w: end - run_x,
                        h: thickness,
                    },
                    color,
                });
            }
        }
    }

    // The caret, drawn only on the visible half of its blink cycle.
    if p.caret_visible {
        let (cx, cy, ch) = layout.caret(fonts, rope, p.cursor);
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: cx,
                y: cy - scroll_y,
                w: CARET_WIDTH,
                h: ch,
            },
            color: CARET_COLOR,
        });
    }

    // The find bar and its match ticks float over everything else, so they go
    // last.
    if let Some(find) = &p.find {
        push_find_ticks(&mut list, find, width as i32, view_h);
        push_find_bar(&mut list, fonts, find, width as i32);
    }

    list
}

/// Append the right-edge tick strip while find is open: one mark per match at
/// its document-proportional height, so the distribution of hits is visible at
/// a glance and survives scrolling. Ticks are deduplicated per two-pixel row,
/// which bounds the strip to O(viewport) commands however many matches exist;
/// the current match's tick is wider and bright.
fn push_find_ticks(list: &mut DisplayList, find: &FindPaint, width: i32, view_h: i32) {
    if find.matches.is_empty() || view_h <= 0 {
        return;
    }
    let content_h = find.doc_height.max(view_h) as i64;
    let mut occupied = vec![false; (view_h / 2 + 1) as usize];
    for i in 0..find.matches.len() {
        // Each match's content-space top comes from the line index, so an offscreen
        // match still maps to its true position rather than the window's edge.
        let top = find.tick_tops.get(i).copied().unwrap_or(0) as i64;
        let y = (top * (view_h - TICK_HEIGHT) as i64 / content_h) as i32;
        let bucket = (y / 2).clamp(0, view_h / 2) as usize;
        let current = i == find.current;
        if occupied[bucket] && !current {
            continue;
        }
        occupied[bucket] = true;
        let (w, color) = if current {
            (TICK_WIDTH + 3, CARET_COLOR)
        } else {
            (TICK_WIDTH, FIND_TICK_COLOR)
        };
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: width - w - 1,
                y,
                w,
                h: TICK_HEIGHT,
            },
            color,
        });
    }
}

/// The byte range spanned by the visual lines that touch the viewport, for
/// culling by byte offset before any geometry is measured. Conservative: a
/// line's visibility is judged by its text box, and the empty viewport yields
/// an empty range.
fn visible_byte_range(
    fonts: &Fonts,
    layout: &Layout,
    scroll_y: i32,
    view_h: i32,
) -> (usize, usize) {
    let mut range = (usize::MAX, 0);
    for line in layout.lines() {
        let m = fonts.metrics(line.size);
        let baseline = line.baseline - scroll_y;
        if baseline - m.ascent >= view_h || baseline + m.descent <= 0 {
            continue;
        }
        range.0 = range.0.min(line.start);
        range.1 = range.1.max(line.end);
    }
    if range.0 > range.1 {
        (0, 0)
    } else {
        range
    }
}

/// Append the find bar: a rounded panel floating at the top-right with the
/// query (its tail, when it overflows — the end is where editing happens), a
/// steady caret after it, and a match counter. Drawn from plain display-list
/// commands, so damage tracking and clipped repaints treat it like any other
/// content.
fn push_find_bar(list: &mut DisplayList, fonts: &Fonts, find: &FindPaint, width: i32) {
    let size = FONT_SIZE;
    let key = FaceKey::Prose {
        size,
        style: FontStyle::Regular,
    };
    let face = fonts.face(size, FontStyle::Regular);
    let m = fonts.metrics(size);
    let pad = size as i32 / 2;
    let row_h = m.ascent + m.descent;

    // The right-hand cluster on the query row: the mode flags (bright when
    // on) and the match counter (the alert color for a broken regex).
    let counter = if find.query.is_empty() {
        String::new()
    } else if find.error {
        "!".to_string()
    } else if find.matches.is_empty() {
        "0/0".to_string()
    } else {
        format!("{}/{}", find.current + 1, find.matches.len())
    };
    let counter_w = shape::text_advance(face, &counter).ceil() as i32;
    let flag_word_w = shape::text_advance(face, "W").ceil() as i32;
    let flag_regex_w = shape::text_advance(face, ".*").ceil() as i32;
    let gap = pad / 2;
    let tail_w = flag_word_w + gap + flag_regex_w + gap + counter_w;

    // Keep a field's display within a third of the window by dropping leading
    // clusters: the end is where editing happens. Queries are short, so the
    // remeasure loop is cheap.
    fn tail_that_fits<'t>(face: &crate::freetype::Face, text: &'t str, max: f32) -> &'t str {
        let mut shown = text;
        while shape::text_advance(face, shown) > max && !shown.is_empty() {
            shown = &shown[grapheme::next_boundary(shown, 0)..];
        }
        shown
    }
    let field_max = (width / 3).max(3 * size as i32) as f32;
    let query = tail_that_fits(face, find.query, field_max);
    let query_w = shape::text_advance(face, query).ceil() as i32;
    let replace = find.replace.map(|r| tail_that_fits(face, r, field_max));
    let replace_w = replace
        .map(|r| shape::text_advance(face, r).ceil() as i32)
        .unwrap_or(0);
    let arrow = "→ ";
    let arrow_w = shape::text_advance(face, arrow).ceil() as i32;

    let rows = 1 + replace.is_some() as i32;
    let replace_row_w = if replace.is_some() {
        arrow_w + replace_w
    } else {
        0
    };
    let field_w = query_w.max(replace_row_w) + CARET_WIDTH + 2;
    let w = (field_w + pad + tail_w + 2 * pad).max(6 * size as i32);
    let h = rows * row_h + (rows + 1) * (pad / 2);
    let x = width - w - MARGIN;
    let y = MARGIN / 2;
    list.push(DrawCmd::RoundRect {
        rect: Rect { x, y, w, h },
        color: FIND_BG,
        radius: (pad / 2).max(2),
        corners: display::RoundedCorners::Both,
    });

    // A steady caret (no blink) after the focused field's text: it marks where
    // typing goes while the bar holds the keyboard.
    let caret_after = |list: &mut DisplayList, pen: i32, baseline: i32| {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: pen + 1,
                y: baseline - m.ascent,
                w: CARET_WIDTH,
                h: row_h,
            },
            color: CARET_COLOR,
        });
    };

    // Row one: the query, a selected query on a selection-colored fill,
    // invalid regexes in the alert color.
    let baseline = y + pad / 2 + m.ascent;
    let pen = x + pad;
    if !query.is_empty() {
        if find.selected && !find.focus_replace {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: pen,
                    y: baseline - m.ascent,
                    w: query_w,
                    h: row_h,
                },
                color: SELECTION_COLOR,
            });
        }
        let color = if find.error {
            FIND_ERROR_COLOR
        } else {
            FOREGROUND
        };
        list.push(text_cmd(
            key,
            pen,
            baseline,
            pen + query_w,
            color,
            query.to_string(),
            m,
        ));
    }
    if !find.focus_replace {
        caret_after(list, pen + query_w, baseline);
    }

    // The flags and counter, right-aligned on the query row.
    let mut tx = x + w - pad - tail_w;
    list.push(text_cmd(
        key,
        tx,
        baseline,
        tx + flag_word_w,
        if find.word { FOREGROUND } else { MARKER_COLOR },
        "W".to_string(),
        m,
    ));
    tx += flag_word_w + gap;
    list.push(text_cmd(
        key,
        tx,
        baseline,
        tx + flag_regex_w,
        if find.regex { FOREGROUND } else { MARKER_COLOR },
        ".*".to_string(),
        m,
    ));
    tx += flag_regex_w + gap;
    if !counter.is_empty() {
        let color = if find.error {
            FIND_ERROR_COLOR
        } else {
            MARKER_COLOR
        };
        list.push(text_cmd(
            key,
            tx,
            baseline,
            tx + counter_w,
            color,
            counter,
            m,
        ));
    }

    // Row two: the replacement, prefixed with a dim arrow; selected content
    // sits on the same selection fill as a selected query.
    if let Some(replacement) = replace {
        let baseline = baseline + row_h + pad / 2;
        list.push(text_cmd(
            key,
            pen,
            baseline,
            pen + arrow_w,
            MARKER_COLOR,
            arrow.to_string(),
            m,
        ));
        if !replacement.is_empty() {
            if find.selected && find.focus_replace {
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: pen + arrow_w,
                        y: baseline - m.ascent,
                        w: replace_w,
                        h: row_h,
                    },
                    color: SELECTION_COLOR,
                });
            }
            list.push(text_cmd(
                key,
                pen + arrow_w,
                baseline,
                pen + arrow_w + replace_w,
                FOREGROUND,
                replacement.to_string(),
                m,
            ));
        }
        if find.focus_replace {
            caret_after(list, pen + arrow_w + replace_w, baseline);
        }
    }
}

/// Append a task item's checkbox at `rect` (screen space, from
/// [`Layout::checkbox_rect`]): unchecked is an outlined rounded square (an
/// inset background fill hollows the border), checked fills with the accent
/// and carries a check mark. Built entirely from fills, since the prose face
/// has no checkbox glyphs to lean on.
fn push_checkbox(list: &mut DisplayList, rect: Rect, checked: bool) {
    let radius = (rect.w / 4).max(2);
    list.push(DrawCmd::RoundRect {
        rect,
        color: if checked { LINK_COLOR } else { MARKER_COLOR },
        radius,
        corners: display::RoundedCorners::Both,
    });
    if checked {
        push_check_mark(list, rect);
    } else {
        let t = (rect.w / 8).max(1);
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: rect.x + t,
                y: rect.y + t,
                w: rect.w - 2 * t,
                h: rect.h - 2 * t,
            },
            color: BACKGROUND,
            radius: (radius - t).max(1),
            corners: display::RoundedCorners::Both,
        });
    }
}

/// Append a check mark inside the checked box `b`: a small tick built from
/// overlapping square fills stepping along its two arms (a short down-stroke
/// meeting a longer up-stroke), in the background color over the accent fill.
fn push_check_mark(list: &mut DisplayList, b: Rect) {
    let u = (b.w / 6).max(1);
    // The arms meet low and left of center, like a hand-drawn tick.
    let (mx, my) = (b.x + b.w * 2 / 5, b.y + b.h * 13 / 20);
    // Steps up-left (the short arm), the meeting point, then up-right (the
    // long arm); cells one pixel larger than the step so the diagonal connects.
    let cells = [
        (mx - u, my - u),
        (mx, my),
        (mx + u, my - u),
        (mx + 2 * u, my - 2 * u),
        (mx + 3 * u, my - 3 * u),
    ];
    for (x, y) in cells {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x,
                y,
                w: u + 1,
                h: u + 1,
            },
            color: BACKGROUND,
        });
    }
}

/// Append the selection chrome for a selected image at picture rect `pic` (screen
/// space): a four-edge accent border and the corner `</>` toggle button.
fn push_image_chrome(list: &mut DisplayList, fonts: &Fonts, pic: &Rect) {
    let t = IMAGE_BORDER_THICKNESS;
    for rect in [
        Rect {
            x: pic.x,
            y: pic.y,
            w: pic.w,
            h: t,
        },
        Rect {
            x: pic.x,
            y: pic.y + pic.h - t,
            w: pic.w,
            h: t,
        },
        Rect {
            x: pic.x,
            y: pic.y,
            w: t,
            h: pic.h,
        },
        Rect {
            x: pic.x + pic.w - t,
            y: pic.y,
            w: t,
            h: pic.h,
        },
    ] {
        list.push(DrawCmd::Fill {
            rect,
            color: IMAGE_BORDER_COLOR,
        });
    }

    let btn = image_button_rect(pic);
    list.push(DrawCmd::RoundRect {
        rect: btn,
        color: IMAGE_BUTTON_BG,
        radius: IMAGE_BUTTON_RADIUS,
        corners: display::RoundedCorners::Both,
    });
    // Center the glyph in both axes from the code-face metrics.
    let fk = FaceKey::Code { size: FONT_SIZE };
    let face = fonts.face_for(fk);
    let glyph = "</>";
    let advance = shape::text_advance(face, glyph).round() as i32;
    let m = fonts.metrics(FONT_SIZE);
    let gx = btn.x + (btn.w - advance) / 2;
    let gbaseline = btn.y + (btn.h - (m.ascent + m.descent)) / 2 + m.ascent;
    let gend = (gx as f32 + shape::text_advance(face, glyph)).round() as i32;
    list.push(text_cmd(
        fk,
        gx,
        gbaseline,
        gend,
        FOREGROUND,
        glyph.to_string(),
        m,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tick strip's per-match tops and document height, matching the values
    /// the app derives from its line index: each match's line baseline, and the
    /// content height including margins.
    fn tick_tops(layout: &Layout, matches: &[(usize, usize)]) -> (Vec<i32>, i32) {
        let tops = matches
            .iter()
            .map(|&(start, _)| {
                let idx = layout
                    .lines()
                    .partition_point(|l| l.start <= start)
                    .saturating_sub(1);
                layout.lines().get(idx).map_or(0, |l| l.baseline)
            })
            .collect();
        (tops, layout.text_height() + 2 * MARGIN)
    }
    use crate::rope::Rope;

    fn paint_params(width: usize, height: usize) -> PaintParams<'static> {
        PaintParams {
            width,
            height,
            scroll_y: 0,
            selection: None,
            cursor: 0,
            caret_visible: false,
            find: None,
        }
    }

    #[test]
    fn task_checkboxes_and_done_styling_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "- [ ] open\n- [x] done\nx";
        let rope = Rope::from_str(src);
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        // Cursor on the trailing "x": both task lines render their boxes.
        let off = src.len() - 1;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (off, off));
        let list = build_display_list(&fonts, &layout, &rope, &paint_params(w, h));

        // One outlined box (marker-color shell hollowed by a background inset)
        // and one accent-filled box.
        let round = |color: u32| {
            list.iter()
                .filter(|c| matches!(c, DrawCmd::RoundRect { color: c2, .. } if *c2 == color))
                .count()
        };
        assert_eq!(round(MARKER_COLOR), 1, "the unchecked shell");
        assert_eq!(round(BACKGROUND), 1, "its hollowing inset");
        assert_eq!(round(LINK_COLOR), 1, "the checked fill");
        // The check mark: small background-color cells inside the accent box.
        let accent = list.iter().find_map(|c| match c {
            DrawCmd::RoundRect { rect, color, .. } if *color == LINK_COLOR => Some(*rect),
            _ => None,
        });
        let accent = accent.expect("the checked box");
        let cells = list
            .iter()
            .filter(|c| {
                matches!(c, DrawCmd::Fill { rect, color }
                    if *color == BACKGROUND
                        && rect.x >= accent.x
                        && rect.x + rect.w <= accent.x + accent.w
                        && rect.y >= accent.y
                        && rect.y + rect.h <= accent.y + accent.h)
            })
            .count();
        assert!(cells >= 5, "a stair of check cells, got {cells}");
        // The raw markers are substituted, not drawn: no bracket glyphs.
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !texts.iter().any(|t| t.contains('[')),
            "no raw checkbox glyphs on rendered lines: {texts:?}"
        );
        // The done item's text dims to the marker color and gains a strike
        // fill spanning its drawn width; the open item's text stays foreground.
        let text_color = |needle: &str| {
            list.iter().find_map(|c| match c {
                DrawCmd::Text { text, color, .. } if text == needle => Some(*color),
                _ => None,
            })
        };
        assert_eq!(text_color("open"), Some(FOREGROUND));
        assert_eq!(text_color("done"), Some(MARKER_COLOR), "done text is dim");
        let done_run = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::Text {
                    text, x, baseline, ..
                } if text == "done" => Some((*x, *baseline)),
                _ => None,
            })
            .expect("the done run");
        let m = fonts.metrics(FONT_SIZE);
        assert!(
            list.iter().any(|c| {
                matches!(c, DrawCmd::Fill { rect, color }
                    if *color == MARKER_COLOR
                        && rect.x == done_run.0
                        && rect.y > done_run.1 - m.ascent
                        && rect.y < done_run.1)
            }),
            "the done text is struck through"
        );
    }

    #[test]
    fn find_bar_and_match_highlights_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("one two one two one\n");
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let matches = [(0usize, 3usize), (8, 11), (16, 19)];
        let (tops, doc_h) = tick_tops(&layout, &matches);
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            // The app mirrors the current match into the selection.
            selection: Some((8, 11)),
            cursor: 11,
            caret_visible: false,
            find: Some(FindPaint {
                query: "one",
                replace: None,
                focus_replace: false,
                selected: false,
                word: false,
                regex: false,
                error: false,
                matches: &matches,
                current: 1,
                tick_tops: &tops,
                doc_height: doc_h,
            }),
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);
        let match_fills = list
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == FIND_MATCH_COLOR))
            .count();
        assert_eq!(match_fills, 2, "the two non-current matches highlight");
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == SELECTION_COLOR)),
            "the current match is painted as the selection"
        );
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::RoundRect { color, .. } if *color == FIND_BG)),
            "the bar panel is drawn"
        );
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"one"), "the query is drawn in the bar");
        assert!(texts.contains(&"2/3"), "the counter shows current/total");
    }

    #[test]
    fn the_replace_row_flags_and_ticks_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("one two one\nfiller\none more\n");
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let matches = [(0usize, 3usize), (8, 11), (19, 22)];
        let (tops, doc_h) = tick_tops(&layout, &matches);
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            selection: Some((0, 3)),
            cursor: 3,
            caret_visible: false,
            find: Some(FindPaint {
                query: "one",
                replace: Some("two"),
                focus_replace: true,
                selected: false,
                word: true,
                regex: false,
                error: false,
                matches: &matches,
                current: 0,
                tick_tops: &tops,
                doc_height: doc_h,
            }),
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"→ "), "the replace row is prefixed");
        assert!(texts.contains(&"two"), "the replacement text is drawn");
        assert!(
            texts.contains(&"W") && texts.contains(&".*"),
            "both mode flags are drawn"
        );
        // The right-edge tick strip: one accent tick for the current match and
        // gold ticks for the rest (the bar caret sits well left of the edge).
        let edge = w as i32 - 10;
        let ticks = list
            .iter()
            .filter(|c| {
                matches!(c, DrawCmd::Fill { color, rect }
                    if (*color == FIND_TICK_COLOR || *color == CARET_COLOR) && rect.x >= edge)
            })
            .count();
        assert!(
            ticks >= 2,
            "expected match ticks at the right edge, got {ticks}"
        );
    }
}
