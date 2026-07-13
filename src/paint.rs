use crate::find::{self, FindState};
use crate::layout::{
    aligned_x, backtick_reveal_shift, cell_line_top, face_key, image_button_rect, inline_code_pad,
    is_code_marker, row_max_lines, CellLine, Gutter, Layout, TableRow, TABLE_BORDER,
    TABLE_CELL_PAD_Y,
};
use crate::markdown;
use crate::pane::TreePane;
use crate::platform::freetype::{Face, FaceKey, FontStyle, Fonts, Metrics};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::platform::pixel;
use crate::platform::scroll;
use crate::platform::shape;
use crate::render::display::{self, DisplayList, DrawCmd};
use crate::scale::Scale;
use crate::theme::{
    BACKGROUND, CARET_COLOR, CARET_WIDTH, CODE_BG, CODE_COLOR, CODE_COMMENT_COLOR,
    CODE_KEYWORD_COLOR, CODE_MARKER_COLOR, CODE_NUMBER_COLOR, CODE_PAD_X, CODE_PAD_Y, CODE_RADIUS,
    CODE_STRING_COLOR, DECOR_TEXT_GAP, FIND_BG, FIND_ERROR_COLOR, FIND_MATCH_COLOR,
    FIND_TICK_COLOR, FOREGROUND, HEADING_COLORS, IMAGE_BORDER_COLOR, IMAGE_BORDER_THICKNESS,
    IMAGE_BUTTON_BG, IMAGE_BUTTON_RADIUS, LINK_COLOR, MARGIN, MARKER_COLOR, QUOTE_BAR_COLOR,
    QUOTE_BAR_WIDTH, RULE_COLOR, RULE_THICKNESS, SCROLL_INK, SCROLL_INSET, SCROLL_MIN_THUMB,
    SCROLL_NEAR, SCROLL_THIN, SCROLL_THUMB_COVER, SCROLL_THUMB_HOVER_COVER, SCROLL_TROUGH_COVER,
    SCROLL_WIDE, SELECTION_COLOR, SPLITTER_COLOR, TABLE_BORDER_COLOR, TABLE_HEADER_BG,
    TAB_ACTIVE_BG, TAB_ACTIVE_TEXT, TAB_BUTTON_HOVER_BG, TAB_BUTTON_RADIUS, TAB_CLOSE_GAP,
    TAB_CLOSE_W, TAB_DIRTY_COLOR, TAB_HOVER_BG, TAB_INACTIVE_TEXT, TAB_MAX_W, TAB_MIN_W, TAB_NEW_W,
    TAB_PAD_X, TAB_RADIUS, TAB_VPAD, TICK_HEIGHT, TICK_WIDTH, TOP_BAR_BG, TREE_BG,
    TREE_CHEVRON_MAX, TREE_FOOTER_GAP, TREE_FOOTER_VPAD, TREE_HOVER_BG, TREE_INDENT,
    TREE_MARKER_COLOR, TREE_MARKER_W, TREE_PAD, TREE_ROW_INSET_Y, TREE_ROW_PAD, TREE_ROW_RADIUS,
    TREE_ROW_VPAD, TREE_TEXT_COLOR, VAULT_PAD_X,
};

/// The per-frame inputs [`build_display_list`] needs beyond the layout itself:
/// the pane's column, the scroll offset, the live selection and cursor, and
/// whether the caret is on this blink. Bundled so the build signature stays
/// readable and so the perf harness can drive a build without a live surface.
pub(crate) struct PaintParams<'a> {
    /// The pane's column in window pixels. The layout is built at absolute positions
    /// inside it, so the painter culls against this band and hangs its chrome (the
    /// scrollbar, the find bar and its ticks) off these edges rather than off a size.
    pub view: Rect,
    pub scroll_y: i32,
    /// The document's full height, which the scrollbar's thumb and the find tick
    /// strip both scale against. The layout only covers the visible window, so this
    /// comes from the pane's line index.
    pub content_height: i32,
    /// The pane's overlay scrollbar, as lit and as wide as its last tick left it.
    pub scrollbar: &'a scroll::Scrollbar,
    /// The display scale, so the painter's pixel constants (margin, caret, rule,
    /// tick strip) land at the right device size. Identity at [`Scale::ONE`].
    pub scale: Scale,
    /// Device-pixel body font size, for the find bar and other chrome sized off the
    /// body ([`crate::theme::FONT_SIZE`] at scale 1.0).
    pub body_size: u32,
    pub selection: Option<(usize, usize)>,
    /// The caret to draw this frame, already reduced to a content-space rectangle by
    /// the caller's caret animation; `None` when the caret is hidden (unfocused,
    /// blinked off, off-window, or an empty pane).
    pub caret: Option<CaretPaint>,
    /// The find bar's paint state while it is open; `None` draws no bar and no
    /// match highlights.
    pub find: Option<FindPaint<'a>>,
    /// The table add-row / add-column strip under the pointer, drawn as a hover
    /// affordance; `None` when the pointer is not over one.
    pub table_hover: Option<crate::layout::TableButton>,
}

/// The caret reduced to what the painter draws: its content-space (pre-scroll)
/// rectangle and a corner radius. A zero radius is the plain caret bar (drawn as a
/// [`DrawCmd::Fill`], the pre-animation appearance); a positive radius is a smear
/// mid-motion (a rounded rectangle). The animation lives on the pane, so the
/// painter holds no caret state.
pub(crate) struct CaretPaint {
    pub rect: Rect,
    pub radius: i32,
}

/// What the painter needs from an open find bar: the [`FindState`] itself,
/// borrowed for the frame being built, plus the tick-strip tops that come from the
/// caller's line index rather than the find state.
pub(crate) struct FindPaint<'a> {
    /// The open find bar's state (query, matches, current index, mode flags).
    pub state: &'a FindState,
    /// Content-space top of each match's line, from the line index, so the tick
    /// strip places a match that sits outside the painted window correctly.
    pub tick_tops: &'a [i32],
}

impl FindPaint<'_> {
    /// The replacement text to show, or `None` when the replace row is hidden.
    fn replace(&self) -> Option<&str> {
        self.state.show_replace().then(|| self.state.replacement())
    }

    /// Whether the replace field (rather than the query) holds the keyboard.
    fn focus_replace(&self) -> bool {
        self.state.focus() == find::Field::Replace
    }
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

/// The color for a code-block token's class (see [`crate::syntax`]); plain code
/// keeps the body [`CODE_COLOR`].
fn code_token_color(token: markdown::CodeToken) -> u32 {
    match token {
        markdown::CodeToken::Plain => CODE_COLOR,
        markdown::CodeToken::Keyword => CODE_KEYWORD_COLOR,
        markdown::CodeToken::Str => CODE_STRING_COLOR,
        markdown::CodeToken::Comment => CODE_COMMENT_COLOR,
        markdown::CodeToken::Number => CODE_NUMBER_COLOR,
    }
}

/// The painter's view of an open find bar, or `None` when it is closed.
/// `tick_tops` and `doc_height` come from the caller's line index, so the tick
/// strip does not depend on which lines the windowed layout currently holds.
pub(crate) fn build_find_paint<'a>(
    find: &'a FindState,
    tick_tops: &'a [i32],
) -> Option<FindPaint<'a>> {
    find.is_open().then_some(FindPaint {
        state: find,
        tick_tops,
    })
}

/// The three rectangles an overlay scrollbar owns in a scrollable column `view`: the
/// lane its thumb travels down, the band a press grabs it in, and the zone whose
/// pointer wakes it into a full slider. Shared by the painter and the pointer
/// hit-testing so the bar is grabbable exactly where it is drawn.
///
/// ```text
///                      view's right edge ─┐
///     │        zone            │  grab  │ │
///     │                        │ track  │ │
///     ├────────────────────────┼────────┤ │
///     │  ....................  │  ███   │ │  ← thumb, right-aligned in the track
/// ```
pub(crate) struct ScrollLane {
    /// The thumb's lane: the full scrollable height of the column, inset from its
    /// right edge.
    pub track: Rect,
    /// Where a press takes the bar: the lane plus the inset out to the column's edge,
    /// so a click at the very edge of a maximized window still lands on it.
    pub grab: Rect,
    /// Where the pointer expands the bar: the lane plus a margin of slack to its left.
    pub zone: Rect,
}

/// Split the right edge of a scrollable column `view` into the scrollbar's lane, its
/// grab band, and its proximity zone.
pub(crate) fn scroll_lane(view: Rect, scale: Scale) -> ScrollLane {
    let wide = scale.px(SCROLL_WIDE);
    let inset = scale.px(SCROLL_INSET);
    let right = view.x + view.w;
    let track = Rect {
        x: right - inset - wide,
        y: view.y,
        w: wide,
        h: view.h,
    };
    let from = |x: i32| Rect {
        x,
        y: view.y,
        w: (right - x).max(0),
        h: view.h,
    };
    ScrollLane {
        track,
        grab: from(track.x),
        zone: from(track.x - scale.px(SCROLL_NEAR)),
    }
}

/// Draw the overlay scrollbar for a column: a faint trough (only once the pointer has
/// grown the bar into a slider) and the thumb, both tinted over `behind`, the surface
/// they float on, by how lit the bar is. A bar that is out of sight, or a column whose
/// content fits, draws nothing.
pub(crate) fn push_scrollbar(
    list: &mut DisplayList,
    view: Rect,
    bar: &scroll::Scrollbar,
    content_height: i32,
    scroll_y: i32,
    behind: u32,
    scale: Scale,
) {
    let lit = bar.lit();
    if lit <= 0.0 {
        return;
    }
    let track = scroll_lane(view, scale).track;
    let (thin, wide) = (scale.px(SCROLL_THIN), scale.px(SCROLL_WIDE));
    let Some(full) = scroll::thumb(
        track,
        view.h,
        content_height,
        scroll_y,
        scale.px(SCROLL_MIN_THUMB),
    ) else {
        return;
    };
    // The bar grows out of the window edge: the thumb keeps its right edge and widens
    // leftward, and the trough appears under it as it goes.
    let grown = bar.wide();
    if grown > 0.0 {
        list.push(DrawCmd::Fill {
            rect: track,
            color: pixel::tint(behind, SCROLL_INK, SCROLL_TROUGH_COVER * grown * lit),
        });
    }
    let w = thin + (((wide - thin) as f32 * grown).round() as i32);
    let rect = Rect {
        x: full.x + full.w - w,
        w,
        ..full
    };
    // The thumb lifts under the pointer, so a bar that is merely reporting a scroll
    // reads quieter than one that is asking to be grabbed.
    let cover = SCROLL_THUMB_COVER + (SCROLL_THUMB_HOVER_COVER - SCROLL_THUMB_COVER) * grown;
    list.push(DrawCmd::RoundRect {
        rect,
        color: pixel::tint(behind, SCROLL_INK, cover * lit),
        radius: w / 2,
        corners: display::RoundedCorners::Both,
    });
}

/// Draw a splitter grab bar: its gap cleared to the background with a thin
/// divider line down the center. Drawn last, over any content that overhangs.
pub(crate) fn push_splitter(list: &mut DisplayList, rect: Rect, scale: Scale) {
    list.push(DrawCmd::Fill {
        rect,
        color: BACKGROUND,
    });
    let divider = scale.px(1).max(1);
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: rect.x + (rect.w - divider) / 2,
            y: rect.y,
            w: divider,
            h: rect.h,
        },
        color: SPLITTER_COLOR,
    });
}

/// The sidebar's text size. The sidebar is chrome, not content, so its names read
/// a step below the body, the way their muted color does: nine tenths of the body
/// size, rounded to the nearest pixel, so it tracks the body through every font
/// size and display scale.
pub(crate) fn tree_font_size(body_size: u32) -> u32 {
    (body_size * 9 + 5) / 10
}

/// The metrics the sidebar lays out in. Its rows, its footer, and the hit-testing
/// all measure against these, so a click lands on the row it looks like.
pub(crate) fn tree_metrics(fonts: &Fonts, body_size: u32) -> Metrics {
    fonts.metrics(tree_font_size(body_size))
}

/// Every pixel size the editor opens a face at: the markdown styles' sizes plus the
/// sidebar's. A size that was never opened falls back to the body face, so the
/// sidebar's has to be in the set handed to [`Fonts`].
pub(crate) fn font_sizes(body_size: u32) -> Vec<u32> {
    let mut sizes = markdown::all_sizes(body_size);
    let tree = tree_font_size(body_size);
    if !sizes.contains(&tree) {
        sizes.push(tree);
    }
    sizes
}

/// The pixel height of one tree row: the font's line height plus vertical
/// padding above and below, so items get breathing room and a taller hit area.
/// Shared by the painter and the pointer/scroll hit-testing so they agree.
pub(crate) fn tree_row_height(m: Metrics, scale: Scale) -> i32 {
    (m.line_height + 2 * scale.px(TREE_ROW_VPAD)).max(1)
}

/// The sidebar column split into the rows that scroll and the footer pinned to its
/// bottom, with the folder-filter toggle's hit box inside the footer. Shared by the
/// painter and the pointer hit-testing so a click lands where it looks.
pub(crate) struct TreeColumn {
    /// The rows' viewport: the column above the footer. Scroll is clamped to it.
    pub rows: Rect,
    pub footer: Rect,
    /// The filter toggle (its checkbox and label) within the footer.
    pub button: Rect,
}

/// Split the sidebar `column` into its rows viewport and its footer. The footer is
/// one row tall plus its padding, and never taller than the column itself.
pub(crate) fn tree_column(column: Rect, m: Metrics, scale: Scale) -> TreeColumn {
    let vpad = scale.px(TREE_FOOTER_VPAD);
    let h = (tree_row_height(m, scale) + 2 * vpad).min(column.h);
    let pad = scale.px(TREE_PAD);
    let footer = Rect {
        x: column.x,
        y: column.y + column.h - h,
        w: column.w,
        h,
    };
    TreeColumn {
        rows: Rect {
            h: (column.h - h).max(0),
            ..column
        },
        footer,
        button: Rect {
            x: footer.x + pad,
            y: footer.y + vpad,
            w: (footer.w - 2 * pad).max(0),
            h: (footer.h - 2 * vpad).max(0),
        },
    }
}

/// The sidebar's scrollable height: every row, plus the padding above the first and
/// below the last.
fn tree_content_height(rows: usize, m: Metrics, scale: Scale) -> i32 {
    rows as i32 * tree_row_height(m, scale) + 2 * scale.px(TREE_PAD)
}

/// Measure the sidebar into its column `rect`: how tall its rows stand, and its scroll
/// clamped to them. The clamp is against the rows' viewport (the column above the
/// pinned footer), not the whole column, so the last row cannot come to rest under the
/// footer. The app runs this before it paints, so the tree's scroll, its scrollbar, and
/// what [`push_tree`] draws all agree on how tall the tree is.
pub(crate) fn measure_tree(tree: &mut TreePane, rect: Rect, m: Metrics, scale: Scale) {
    let rows = tree.tree.rows().len();
    tree.content_height = tree_content_height(rows, m, scale);
    let view = tree_column(rect, m, scale).rows;
    tree.scroll_y = scroll::clamp(tree.scroll_y, tree.content_height, view.h);
}

/// Draw the file-tree sidebar into its column `rect`: a darker background, then
/// each visible row (a folder chevron plus the name), indented by depth and
/// offset by the tree's own scroll, then its scrollbar, and finally the footer over
/// the column's bottom. Rows do not wrap; a long name that overhangs is covered by
/// the editor's own background, which is drawn after this.
pub(crate) fn push_tree(
    list: &mut DisplayList,
    fonts: &Fonts,
    tree: &TreePane,
    rect: Rect,
    scale: Scale,
    body_size: u32,
) {
    list.push(DrawCmd::Fill {
        rect,
        color: TREE_BG,
    });
    let size = tree_font_size(body_size);
    let m = fonts.metrics(size);
    let col = tree_column(rect, m, scale);
    let view = col.rows;
    let row_h = tree_row_height(m, scale);
    // Baseline that centers the glyph ink (ascent + descent) in the row, so the
    // label sits in the middle of its pill rather than riding the line's top.
    let text_offset = (row_h - (m.ascent + m.descent)) / 2 + m.ascent;
    let tree_pad = scale.px(TREE_PAD);
    let row_pad = scale.px(TREE_ROW_PAD);
    let inset_y = scale.px(TREE_ROW_INSET_Y);
    let marker_w = scale.px(TREE_MARKER_W);
    let tree_indent = scale.px(TREE_INDENT);
    let rows = tree.tree.rows();
    let face = FaceKey::Prose {
        size,
        style: FontStyle::Regular,
    };
    let text_right = rect.x + rect.w - tree_pad - row_pad;
    for (i, row) in rows.iter().enumerate() {
        let top = view.y + tree_pad + i as i32 * row_h - tree.scroll_y;
        if top + row_h <= view.y || top >= view.y + view.h {
            continue;
        }
        // The active row (last clicked or opened) and the mouse-hovered row both
        // read as the same soft gray pill, inset from the column edges and within
        // the row so background frames it on every side.
        if tree.selected == Some(i) || tree.hovered == Some(i) {
            list.push(DrawCmd::RoundRect {
                rect: Rect {
                    x: rect.x + tree_pad,
                    y: top + inset_y,
                    w: (rect.w - 2 * tree_pad).max(0),
                    h: (row_h - 2 * inset_y).max(0),
                },
                color: TREE_HOVER_BG,
                radius: scale.px(TREE_ROW_RADIUS),
                corners: display::RoundedCorners::Both,
            });
        }
        // Names share one column, `marker_w` in from the row start, whether or not
        // a chevron is drawn, so files and folders align. A folder's chevron is a
        // vector shape (an anti-aliased distance field, like the task check mark),
        // left-aligned in that column and centered vertically on the label, and
        // dimmer than the name it points at.
        let content_x = rect.x + tree_pad + row_pad + row.depth as i32 * tree_indent;
        if row.is_dir {
            let chev = (row_h * 9 / 16).clamp(scale.px(8), scale.px(TREE_CHEVRON_MAX));
            list.push(DrawCmd::Chevron {
                rect: Rect {
                    x: content_x,
                    y: top + (row_h - chev) / 2,
                    w: chev,
                    h: chev,
                },
                color: TREE_MARKER_COLOR,
                down: row.expanded,
            });
        }
        list.push(text_cmd(
            face,
            content_x + marker_w,
            top + text_offset,
            text_right,
            TREE_TEXT_COLOR,
            row.name.clone(),
            m,
        ));
    }
    // The sidebar's own scrollbar, over the rows it scrolls (never over the footer,
    // which is pinned below them).
    push_scrollbar(
        list,
        view,
        &tree.scrollbar,
        tree.content_height,
        tree.scroll_y,
        TREE_BG,
        scale,
    );
    push_tree_footer(list, fonts, &col, tree, scale, size);
}

/// The sidebar footer's toggle: whether the tree lists folders that hold no note
/// anywhere beneath them. The label doubles as the button's only affordance, so it
/// says what the checkbox means rather than naming the filter.
const TREE_FILTER_LABEL: &str = "Show folders without notes";

/// Draw the sidebar footer: a divider across its top, then the folder-filter
/// toggle, a checkbox and label that brighten under the pointer. The strip is
/// filled first, so a row scrolled beneath it is covered rather than showing
/// through.
fn push_tree_footer(
    list: &mut DisplayList,
    fonts: &Fonts,
    col: &TreeColumn,
    tree: &TreePane,
    scale: Scale,
    size: u32,
) {
    let m = fonts.metrics(size);
    let hovered = tree.filter_hovered;
    list.push(DrawCmd::Fill {
        rect: col.footer,
        color: TREE_BG,
    });
    let divider = scale.px(1).max(1);
    list.push(DrawCmd::Fill {
        rect: Rect {
            h: divider,
            ..col.footer
        },
        color: SPLITTER_COLOR,
    });
    // The pill under the pointer, the same soft gray a hovered row gets; the
    // checkbox is hollowed against whichever of the two it sits on.
    let behind = if hovered { TREE_HOVER_BG } else { TREE_BG };
    if hovered {
        list.push(DrawCmd::RoundRect {
            rect: col.button,
            color: TREE_HOVER_BG,
            radius: scale.px(TREE_ROW_RADIUS),
            corners: display::RoundedCorners::Both,
        });
    }
    let side = (m.ascent * 13 / 12).max(scale.px(11));
    let box_ = Rect {
        x: col.button.x + scale.px(TREE_ROW_PAD),
        y: col.button.y + (col.button.h - side) / 2,
        w: side,
        h: side,
    };
    push_checkbox(list, box_, tree.tree.show_empty_dirs(), behind);
    let x = box_.x + box_.w + scale.px(TREE_FOOTER_GAP);
    let right = col.button.x + col.button.w - scale.px(TREE_ROW_PAD);
    let face = fonts.face(size, FontStyle::Regular);
    list.push(text_cmd(
        FaceKey::Prose {
            size,
            style: FontStyle::Regular,
        },
        x,
        col.button.y + (col.button.h - (m.ascent + m.descent)) / 2 + m.ascent,
        right,
        if hovered { FOREGROUND } else { TREE_TEXT_COLOR },
        fit_label(face, TREE_FILTER_LABEL, (right - x).max(0)),
        m,
    ));
}

/// The pixel height of the top bar: the body line height plus vertical padding
/// above and below, matching how a tree row is sized. Shared by the layout (which
/// reserves it) and the painter so they agree.
pub(crate) fn top_bar_height(m: Metrics, scale: Scale) -> i32 {
    (m.line_height + 2 * scale.px(TAB_VPAD)).max(1)
}

/// One tab's geometry within a strip: its full pill, the close-glyph box inside
/// it, and the label already truncated to fit. Shared by the painter and the
/// pointer hit-testing so a click lands where the eye expects.
pub(crate) struct TabHit {
    pub rect: Rect,
    pub close: Rect,
    pub label: String,
}

/// The paint inputs for one pane's tab strip: where it sits and its tabs' labels,
/// unsaved flags, active index, and what the pointer is over: a tab body
/// (`hovered`), a tab's close glyph (`close_hover`), or the new-tab button
/// (`new_hover`).
pub(crate) struct TabStrip<'a> {
    pub rect: Rect,
    pub labels: &'a [String],
    pub dirty: &'a [bool],
    pub active: usize,
    pub hovered: Option<usize>,
    pub close_hover: Option<usize>,
    pub new_hover: bool,
    /// The tab being dragged to reorder, if any: it lifts out of the run (leaving a
    /// gap) and follows the pointer.
    pub drag: Option<TabDragView>,
}

/// A tab lifted for a drag-reorder: which tab (its index in the current order) and
/// the left edge it floats at, in device pixels before clamping into the strip.
pub(crate) struct TabDragView {
    pub tab: usize,
    pub left: i32,
}

/// The side of a tab's square close-glyph box.
fn tab_close_side(scale: Scale) -> i32 {
    scale.px(TAB_CLOSE_W)
}

/// The new-tab (+) button's box: a square just past the last tab (or at the strip
/// start when the pane is empty). Shared by the painter and the hit-testing.
pub(crate) fn new_tab_button_rect(strip: Rect, tabs: &[TabHit], scale: Scale) -> Rect {
    let gap = scale.px(TAB_CLOSE_GAP);
    let side = scale.px(TAB_NEW_W);
    let x = tabs.last().map(|t| t.rect.x + t.rect.w).unwrap_or(strip.x) + gap;
    Rect {
        x,
        y: strip.y + (strip.h - side) / 2,
        w: side,
        h: side,
    }
}

/// Truncate `label` with a trailing ellipsis until it fits `max_w` pixels in
/// `face`; returns it unchanged when it already fits.
fn fit_label(face: &Face, label: &str, max_w: i32) -> String {
    if shape::text_advance(face, label) as i32 <= max_w {
        return label.to_string();
    }
    let ellipsis = '…';
    let budget = (max_w - shape::text_advance(face, "…") as i32).max(0);
    let mut out = String::new();
    for ch in label.chars() {
        out.push(ch);
        if shape::text_advance(face, &out) as i32 > budget {
            out.pop();
            break;
        }
    }
    out.push(ellipsis);
    out
}

/// The tabs of one strip, laid left to right, each sized to its label between a
/// min and max width. A tab that would spill past the strip's right edge stops the
/// run (overflow is clipped this way), but the first tab always draws even in a
/// narrow strip. Shared by the painter and the hit-testing.
pub(crate) fn tab_rects(
    strip: Rect,
    labels: &[String],
    fonts: &Fonts,
    body_size: u32,
    scale: Scale,
) -> Vec<TabHit> {
    let face = fonts.face(body_size, FontStyle::Regular);
    let pad = scale.px(TAB_PAD_X);
    let gap = scale.px(TAB_CLOSE_GAP);
    let min_w = scale.px(TAB_MIN_W);
    let max_w = scale.px(TAB_MAX_W);
    let close = tab_close_side(scale);
    let strip_right = strip.x + strip.w;
    let mut out = Vec::with_capacity(labels.len());
    let mut x = strip.x;
    for label in labels {
        if x >= strip_right {
            break;
        }
        let label_w = shape::text_advance(face, label).ceil() as i32;
        let tab_w = (pad + label_w + gap + close + pad).clamp(min_w, max_w);
        // Stop before a tab that would spill, but never drop the first one.
        if x + tab_w > strip_right && x > strip.x {
            break;
        }
        let rect = Rect {
            x,
            y: strip.y,
            w: tab_w.min(strip_right - x),
            h: strip.h,
        };
        let close_rect = Rect {
            x: rect.x + rect.w - pad - close,
            y: rect.y + (rect.h - close) / 2,
            w: close,
            h: close,
        };
        let budget = (close_rect.x - gap - (rect.x + pad)).max(0);
        out.push(TabHit {
            rect,
            close: close_rect,
            label: fit_label(face, label, budget),
        });
        x += tab_w;
    }
    out
}

/// Where a dragged tab should sit in this strip given pointer x `px`: the count of
/// tabs whose horizontal midpoint the pointer has passed. Comparing against the
/// neighbors' midpoints (not the dragged tab's own slot) keeps a reorder from
/// oscillating when tabs differ in width. `dragged` is the dragged tab's own index
/// when it lives in this strip (its slot is skipped, so the result indexes the run
/// with it removed, exactly where it reinserts); pass `None` when dropping a tab in
/// from the other pane, where every tab counts.
pub(crate) fn tab_drop_index(tabs: &[TabHit], dragged: Option<usize>, px: f32) -> usize {
    let mut target = 0;
    for (k, t) in tabs.iter().enumerate() {
        if dragged == Some(k) {
            continue;
        }
        let mid = t.rect.x as f32 + t.rect.w as f32 / 2.0;
        if px > mid {
            target += 1;
        }
    }
    target
}

/// The shared drawing context for the top bar's helpers: the fonts, display
/// scale, and body size everything in the bar is measured and drawn at. Bundled
/// so the helpers thread one context instead of three parallel arguments.
pub(crate) struct BarCtx<'a> {
    fonts: &'a Fonts,
    scale: Scale,
    body_size: u32,
}

impl<'a> BarCtx<'a> {
    pub(crate) fn new(fonts: &'a Fonts, scale: Scale, body_size: u32) -> Self {
        Self {
            fonts,
            scale,
            body_size,
        }
    }

    fn metrics(&self) -> Metrics {
        self.fonts.metrics(self.body_size)
    }

    fn regular(&self) -> FaceKey {
        FaceKey::Prose {
            size: self.body_size,
            style: FontStyle::Regular,
        }
    }
}

/// Draw the top bar: fill the strip and a bottom divider, continue the column
/// dividers up through it, the (optional) vault label above the sidebar, and each
/// pane's tab strip above its column. The active tab is filled with the pane
/// background and reaches the strip bottom, covering the divider so it reads as
/// joined to its pane; inactive tabs stay muted, with a soft highlight under the
/// hovered one.
pub(crate) fn push_top_bar(
    list: &mut DisplayList,
    ctx: &BarCtx,
    bar: Rect,
    vault: Option<(Rect, &str)>,
    strips: [Option<TabStrip>; 2],
    dividers: &[Rect],
) {
    let m = ctx.metrics();
    list.push(DrawCmd::Fill {
        rect: bar,
        color: TOP_BAR_BG,
    });
    let divider = ctx.scale.px(1).max(1);
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: bar.x,
            y: bar.y + bar.h - divider,
            w: bar.w,
            h: divider,
        },
        color: SPLITTER_COLOR,
    });
    // The column splitters' divider lines continue up through the bar, so the bar
    // reads as segmented into each pane's strip.
    for d in dividers {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: d.x + (d.w - divider) / 2,
                y: bar.y,
                w: divider,
                h: bar.h,
            },
            color: SPLITTER_COLOR,
        });
    }
    if let Some((vrect, name)) = vault {
        let x = vrect.x + ctx.scale.px(VAULT_PAD_X);
        let right = vrect.x + vrect.w - ctx.scale.px(VAULT_PAD_X);
        let baseline = vrect.y + (vrect.h - (m.ascent + m.descent)) / 2 + m.ascent;
        let fitted = fit_label(
            ctx.fonts.face(ctx.body_size, FontStyle::Bold),
            name,
            (right - x).max(0),
        );
        list.push(text_cmd(
            FaceKey::Prose {
                size: ctx.body_size,
                style: FontStyle::Bold,
            },
            x,
            baseline,
            right,
            FOREGROUND,
            fitted,
            m,
        ));
    }
    for strip in strips.into_iter().flatten() {
        push_tab_strip(list, ctx, &strip);
    }
}

/// A small rounded button that lights up under the pointer, with `glyph` centered
/// in it (the tab close × and the new-tab +). Drawn highlighted only when hovered.
fn push_glyph_button(list: &mut DisplayList, ctx: &BarCtx, box_: Rect, glyph: &str, hovered: bool) {
    if hovered {
        list.push(DrawCmd::RoundRect {
            rect: box_,
            color: TAB_BUTTON_HOVER_BG,
            radius: ctx.scale.px(TAB_BUTTON_RADIUS),
            corners: display::RoundedCorners::Both,
        });
    }
    let m = ctx.metrics();
    let gw = shape::text_advance(ctx.fonts.face(ctx.body_size, FontStyle::Regular), glyph) as i32;
    list.push(text_cmd(
        ctx.regular(),
        box_.x + (box_.w - gw) / 2,
        box_.y + (box_.h - (m.ascent + m.descent)) / 2 + m.ascent,
        box_.x + box_.w,
        if hovered {
            TAB_ACTIVE_TEXT
        } else {
            TAB_INACTIVE_TEXT
        },
        glyph.to_string(),
        m,
    ));
}

/// Draw one pane's tabs and its new-tab button (see [`push_top_bar`]).
fn push_tab_strip(list: &mut DisplayList, ctx: &BarCtx, strip: &TabStrip) {
    let scale = ctx.scale;
    let tabs = tab_rects(strip.rect, strip.labels, ctx.fonts, ctx.body_size, scale);
    let vpad = scale.px(TAB_VPAD);
    let divider = scale.px(1).max(1);
    let dragged = strip.drag.as_ref().map(|d| d.tab);
    let is_active = |i: usize| i == strip.active;
    for (i, tab) in tabs.iter().enumerate() {
        // The dragged tab lifts out of the run and is drawn floating below, leaving
        // a gap here.
        if dragged == Some(i) {
            continue;
        }
        let active = is_active(i);
        let hovered = strip.hovered == Some(i);
        // A hairline between two resting tabs, but never touching the active or
        // hovered tab, nor the gap a lifted tab left behind.
        if !active
            && !hovered
            && i + 1 < tabs.len()
            && dragged != Some(i + 1)
            && !is_active(i + 1)
            && strip.hovered != Some(i + 1)
        {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: tab.rect.x + tab.rect.w - divider,
                    y: tab.rect.y + vpad,
                    w: divider,
                    h: (tab.rect.h - 2 * vpad).max(1),
                },
                color: SPLITTER_COLOR,
            });
        }
        let dirty = strip.dirty.get(i).copied().unwrap_or(false);
        push_tab_body(
            list,
            ctx,
            tab,
            active,
            hovered,
            strip.close_hover == Some(i),
            dirty,
        );
    }
    // The new-tab (+) button at the end of the strip, drawn when it fits.
    let plus = new_tab_button_rect(strip.rect, &tabs, scale);
    if plus.x + plus.w <= strip.rect.x + strip.rect.w {
        push_glyph_button(list, ctx, plus, "+", strip.new_hover);
    }
    // The floating dragged tab, drawn last so it rides over its neighbors and the
    // + button. It follows the pointer, clamped to stay within the strip.
    if let Some(d) = &strip.drag {
        if let Some(tab) = tabs.get(d.tab) {
            let max_left = (strip.rect.x + strip.rect.w - tab.rect.w).max(strip.rect.x);
            let left = d.left.clamp(strip.rect.x, max_left);
            let dx = left - tab.rect.x;
            let floated = TabHit {
                rect: Rect {
                    x: tab.rect.x + dx,
                    ..tab.rect
                },
                close: Rect {
                    x: tab.close.x + dx,
                    ..tab.close
                },
                label: tab.label.clone(),
            };
            let dirty = strip.dirty.get(d.tab).copied().unwrap_or(false);
            push_tab_body(list, ctx, &floated, true, false, false, dirty);
        }
    }
}

/// Draw one tab's contents: its active or hover background, the label, and either
/// the close glyph (active or hovered tab) or the unsaved dot (a resting modified
/// tab). The inter-tab hairline is the strip's business, not a tab's, so it stays
/// out of here.
fn push_tab_body(
    list: &mut DisplayList,
    ctx: &BarCtx,
    tab: &TabHit,
    active: bool,
    hovered: bool,
    close_hover: bool,
    dirty: bool,
) {
    let scale = ctx.scale;
    let m = ctx.metrics();
    let radius = scale.px(TAB_RADIUS);
    let pad = scale.px(TAB_PAD_X);
    if active || hovered {
        list.push(DrawCmd::RoundRect {
            rect: tab.rect,
            color: if active { TAB_ACTIVE_BG } else { TAB_HOVER_BG },
            radius,
            corners: display::RoundedCorners::Top,
        });
    }
    let color = if active {
        TAB_ACTIVE_TEXT
    } else {
        TAB_INACTIVE_TEXT
    };
    let baseline = tab.rect.y + (tab.rect.h - (m.ascent + m.descent)) / 2 + m.ascent;
    list.push(text_cmd(
        ctx.regular(),
        tab.rect.x + pad,
        baseline,
        tab.close.x - scale.px(TAB_CLOSE_GAP),
        color,
        tab.label.clone(),
        m,
    ));
    if active || hovered {
        push_glyph_button(list, ctx, tab.close, "×", close_hover);
    } else if dirty {
        let dot = scale.px(7);
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: tab.close.x + (tab.close.w - dot) / 2,
                y: tab.close.y + (tab.close.h - dot) / 2,
                w: dot,
                h: dot,
            },
            color: TAB_DIRTY_COLOR,
            radius: dot / 2,
            corners: display::RoundedCorners::Both,
        });
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
    let mut painter = Painter::new(fonts, layout, rope, p);
    // Bottom-to-top stacking order. This sequence is also the damage-diff order,
    // so the passes must run in exactly this order; each is one method below.
    painter.background();
    painter.code_block_panels();
    painter.inline_code_panels();
    if let Some((start, end)) = p.selection {
        painter.selection(start, end);
    }
    if let Some(find) = &p.find {
        painter.find_matches(find);
    }
    painter.text();
    if let Some(button) = p.table_hover {
        painter.table_button(button);
    }
    if let Some(caret) = &p.caret {
        painter.caret(caret);
    }
    // The chrome floats over the content, so it goes last: the scrollbar, then the
    // find bar and the match ticks that share the bar's lane.
    painter.scrollbar(p.scrollbar, p.content_height);
    if let Some(find) = &p.find {
        painter.find_ticks(find, p.content_height);
        painter.find_bar(find);
    }
    painter.list
}

/// Accumulates one frame's display list from the layout and rope. The read-only
/// context (`fonts`/`layout`/`rope`) plus the viewport (`scroll_y` and the column
/// `view`) and the content's left edge are held once so each pass reads them
/// without re-plumbing; [`Self::screen`] and [`Self::push_text`] fold the
/// shift-and-cull and measure-then-emit patterns the passes repeat.
///
/// Everything here is in window pixels: the layout bakes the column's origin into
/// its positions (see `EditorPane::ensure_window`), so a pass culls against `view`'s
/// band and pins its chrome to `view`'s edges, and a pane that is not the top-left
/// one lands where it looks.
struct Painter<'a> {
    list: DisplayList,
    fonts: &'a Fonts,
    layout: &'a Layout,
    rope: &'a crate::rope::Rope,
    scroll_y: i32,
    /// The pane's column in window pixels.
    view: Rect,
    /// The content's left edge (the column's inner margin). Text, panels, and
    /// images hang off this so a pane's content sits in its own column.
    left: i32,
    /// The display scale, so each pixel constant scales to device pixels.
    scale: Scale,
    /// Device-pixel body font size, for chrome sized off the body.
    body_size: u32,
}

impl<'a> Painter<'a> {
    fn new(
        fonts: &'a Fonts,
        layout: &'a Layout,
        rope: &'a crate::rope::Rope,
        p: &PaintParams,
    ) -> Self {
        Self {
            // Roughly a few commands per visual line (panel, runs, decorations)
            // plus the background and caret; one upfront allocation beats growing
            // the list mid-build.
            list: Vec::with_capacity(layout.lines().len() * 3 + 8),
            fonts,
            layout,
            rope,
            scroll_y: p.scroll_y,
            view: p.view,
            left: layout.left(),
            scale: p.scale,
            body_size: p.body_size,
        }
    }

    /// The window y of the column's bottom edge, the cull line every pass measures
    /// against.
    fn bottom(&self) -> i32 {
        self.view.y + self.view.h
    }

    /// The width text wraps within: the column less its margins. What a full-width
    /// decoration (a code panel, a thematic break) spans.
    fn content_w(&self) -> i32 {
        (self.view.w - 2 * self.scale.px(MARGIN)).max(0)
    }

    /// Shift a content-space rect into screen space (up by the scroll) and cull
    /// it when it falls entirely above or below the column. `None` means the
    /// caller emits nothing for it.
    fn screen(&self, rect: Rect) -> Option<Rect> {
        let y = rect.y - self.scroll_y;
        if y + rect.h <= self.view.y || y >= self.bottom() {
            return None;
        }
        Some(Rect { y, ..rect })
    }

    /// Measure `text` in `face` and emit its run from `x` along `baseline` in
    /// `color`, returning the pen x after the run (its drawn right edge).
    fn push_text(
        &mut self,
        face: FaceKey,
        x: i32,
        baseline: i32,
        color: u32,
        text: String,
        m: Metrics,
    ) -> i32 {
        let end = (x as f32 + shape::text_advance(self.fonts.face_for(face), &text)).round() as i32;
        self.list
            .push(text_cmd(face, x, baseline, end, color, text, m));
        end
    }

    /// The background, the pane's whole column, first so a clipped repaint clears its
    /// region before the rest of the list draws over it.
    fn background(&mut self) {
        self.list.push(DrawCmd::Fill {
            rect: self.view,
            color: BACKGROUND,
        });
    }

    /// Block backgrounds paint first, beneath both the selection highlight and the
    /// text: a fenced code block sits on a full-width tinted panel, one line box
    /// tall so consecutive lines tile into a continuous block, and that panel must
    /// sit behind the selection rather than cover it (otherwise a selection inside
    /// the block would vanish). A collapsed fence row still paints, so an unfocused
    /// block keeps blank top and bottom rows that read as vertical padding. The
    /// block's first row rounds its top corners and its last row its bottom ones, so
    /// the whole panel reads as one softly rounded block.
    fn code_block_panels(&mut self) {
        let lines = self.layout.lines();
        let panel_w = self.content_w();
        for (i, line) in lines.iter().enumerate() {
            if !line.code_block {
                continue;
            }
            let m = self.fonts.metrics(line.size);
            let Some(rect) = self.screen(Rect {
                x: self.left,
                y: line.baseline - m.ascent,
                w: panel_w,
                h: m.line_height,
            }) else {
                continue;
            };
            let first = i == 0 || !lines[i - 1].code_block;
            let last = i + 1 >= lines.len() || !lines[i + 1].code_block;
            let corners = match (first, last) {
                (true, true) => display::RoundedCorners::Both,
                (true, false) => display::RoundedCorners::Top,
                (false, true) => display::RoundedCorners::Bottom,
                (false, false) => display::RoundedCorners::None,
            };
            self.list.push(DrawCmd::RoundRect {
                rect,
                color: CODE_BG,
                radius: self.scale.px(CODE_RADIUS),
                corners,
            });
        }
    }

    /// Inline code spans get their own tinted panel, also beneath the selection so a
    /// selection over inline code shows through (the glyphs still draw on top in the
    /// text loop). Block code is excluded here; its full-width panel above covers it.
    fn inline_code_panels(&mut self) {
        let pad_x = self.scale.px(CODE_PAD_X);
        let pad_y = self.scale.px(CODE_PAD_Y);
        for rect in self.layout.code_rects(self.fonts, self.rope) {
            let panel = Rect {
                x: rect.x - pad_x,
                y: rect.y - pad_y,
                w: rect.w + 2 * pad_x,
                h: rect.h + 2 * pad_y,
            };
            if let Some(rect) = self.screen(panel) {
                self.list.push(DrawCmd::RoundRect {
                    rect,
                    color: CODE_BG,
                    radius: self.scale.px(CODE_RADIUS),
                    corners: display::RoundedCorners::Both,
                });
            }
        }
    }

    /// The selection highlight. Everything is laid out in content space; the
    /// viewport scroll shifts it up by `scroll_y`, and off-screen rectangles are
    /// dropped from the list.
    fn selection(&mut self, start: usize, end: usize) {
        for rect in self
            .layout
            .selection_rects(self.fonts, self.rope, start, end)
        {
            if let Some(rect) = self.screen(rect) {
                self.list.push(DrawCmd::Fill {
                    rect,
                    color: SELECTION_COLOR,
                });
            }
        }
    }

    /// Find matches, filled like a dimmer selection beneath the text. The current
    /// match is skipped: the app mirrors it into the buffer selection, which was
    /// just painted. Only matches inside the visible byte range are measured, so a
    /// short query with thousands of hits does not price every frame by them.
    fn find_matches(&mut self, find: &FindPaint) {
        let visible = visible_byte_range(self.fonts, self.layout, self.scroll_y, self.view);
        let matches = find.state.matches();
        let lo = matches.partition_point(|&(_, end)| end <= visible.0);
        let hi = matches.partition_point(|&(start, _)| start < visible.1);
        for (i, &(start, end)) in matches.iter().enumerate().take(hi).skip(lo) {
            if i == find.state.current() {
                continue;
            }
            for rect in self
                .layout
                .selection_rects(self.fonts, self.rope, start, end)
            {
                if let Some(rect) = self.screen(rect) {
                    self.list.push(DrawCmd::Fill {
                        rect,
                        color: FIND_MATCH_COLOR,
                    });
                }
            }
        }
    }

    /// The text, one visual line at a time, each at its own font size, materialized
    /// from the rope. A line is a sequence of inline runs: regular text, code spans
    /// (drawn in the code face on a tinted panel), and the delimiter markers, which
    /// are dimmed on the cursor's line and collapse to nothing (skipped) once it
    /// leaves, the live-preview reveal that `layout` bakes into each run's `visible`.
    fn text(&mut self) {
        let (fonts, layout, rope) = (self.fonts, self.layout, self.rope);
        let (scroll_y, left) = (self.scroll_y, self.left);
        let (top, bottom) = (self.view.y, self.bottom());
        for line in layout.lines() {
            let m = fonts.metrics(line.size);
            let baseline = line.baseline - scroll_y;
            // An image line blits its decoded picture into its (taller-than-text)
            // box, centered with the layout's vertical padding, and nothing else
            // draws on the line. Cull on the full box height so a tall image
            // straddling a viewport edge still paints. A picture that has not
            // decoded leaves the box blank.
            if let Some(img) = line.image {
                let box_top = baseline - m.ascent;
                if box_top < bottom && box_top + img.box_height() > top {
                    let pic = Rect {
                        x: left + line.indent,
                        y: box_top + (img.box_height() - img.height) / 2,
                        w: img.width,
                        h: img.height,
                    };
                    // Blit the cached display-size bitmap (resampled once, not per
                    // frame); a no-op at paint time if the image is not decoded.
                    self.list.push(DrawCmd::Image {
                        rect: pic,
                        path: rope.byte_slice(img.path_start..img.path_end),
                    });
                    // A selected image gets an accent border (four thin edge fills)
                    // and a corner toggle button: a rounded panel with a centered
                    // `</>` "source" glyph, the affordance to reveal its markdown.
                    if img.selected {
                        push_image_chrome(&mut self.list, fonts, &pic, self.scale, self.body_size);
                    }
                }
                continue;
            }
            // A table row draws its grid box (header tint, borders, separators) and
            // its cells' wrapped, aligned text, and nothing else. Cull on the full
            // (taller) row box.
            if let Some(table) = &line.table {
                let box_top = baseline - m.ascent;
                if box_top < bottom && box_top + table.box_height() > top {
                    self.table_row(left + line.indent, box_top, line.size, table, m);
                }
                continue;
            }
            // Skip lines fully above or below the viewport, so a frame only
            // rasterizes what is on screen.
            if baseline + m.descent <= top || baseline - m.ascent >= bottom {
                continue;
            }
            // A blockquote draws a bar down its gutter, on every wrapped line.
            if matches!(line.gutter, Gutter::QuoteBar { .. }) {
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: left,
                        y: baseline - m.ascent,
                        w: self.scale.px(QUOTE_BAR_WIDTH),
                        h: m.line_height,
                    },
                    color: QUOTE_BAR_COLOR,
                });
            }
            // A thematic break draws a horizontal line centered in its row; the raw
            // `---` is collapsed (its runs are hidden), so nothing else paints on
            // the line.
            if line.rule {
                let mid = baseline - m.ascent + m.line_height / 2;
                let thickness = self.scale.px(RULE_THICKNESS);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: left,
                        y: mid - thickness / 2,
                        w: self.content_w(),
                        h: thickness,
                    },
                    color: RULE_COLOR,
                });
            }
            // Content is inset by the line's hanging indent (0 unless this is a
            // wrapped list/quote continuation).
            let heading = line.heading.is_some();
            // A completed task's text reads dim, over the whole item (wrapped
            // continuations included), so done items scan as done while the
            // checkbox carries the crossed-off signal.
            let done = line.task == Some(true);
            let mut pen = left + line.indent;
            let runs = layout.run_slice(line);
            for (i, run) in runs.iter().enumerate() {
                if !run.visible {
                    continue;
                }
                // Each run draws in its own face, so bold/italic emphasis is carried
                // by the glyphs, not color: only code and markers recolor, the
                // rest (regular and emphasized prose) is FG.
                let fk = face_key(line.size, run.kind, heading);
                let face = fonts.face_for(fk);
                let slice = rope.byte_slice(run.start..run.end);
                // The block marker (the first run on a list/quote line) gives way to
                // its gutter decoration: a filled disc for a list, nothing for a
                // quote (the bar above replaces it). Either keeps the source width,
                // so text never shifts.
                if i == 0 && matches!(run.kind, markdown::SpanKind::Marker) {
                    let advance = shape::text_advance(face, &slice).round() as i32;
                    match line.gutter {
                        Gutter::Bullet => {
                            // A filled disc stands in for the source marker
                            // ("- ", "* ", "+ "), in the muted marker color
                            // (MARKER_COLOR), not the body FG. Right-anchor it a
                            // gap short of where the text begins, the inset the
                            // checkbox also uses, so the disc hangs in the item's
                            // gutter with breathing room before the words and
                            // sits centered on the ascent band. The pen still
                            // advances by the full source marker, so revealing it
                            // never shifts the text.
                            let text_x = pen + advance;
                            let d = (m.ascent * 2 / 5).max(4);
                            let gap = (m.ascent as f32 * DECOR_TEXT_GAP).max(4.0);
                            let x = ((text_x as f32 - gap).round() as i32 - d).max(left);
                            let cy = baseline - m.ascent / 2;
                            self.list.push(DrawCmd::RoundRect {
                                rect: Rect {
                                    x,
                                    y: cy - d / 2,
                                    w: d,
                                    h: d,
                                },
                                color: MARKER_COLOR,
                                radius: d / 2,
                                corners: display::RoundedCorners::Both,
                            });
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
                                push_checkbox(&mut self.list, rect, checked, BACKGROUND);
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
                    markdown::SpanKind::Marker if is_code_marker(rope, run) => CODE_MARKER_COLOR,
                    markdown::SpanKind::Marker => MARKER_COLOR,
                    // A syntax-highlighted code-block token takes its class color.
                    markdown::SpanKind::Code(tok) => code_token_color(tok),
                    // Every text run of a done task dims to the marker color, so
                    // the whole item (its code and links included) reads muted.
                    markdown::SpanKind::Text(_) if done => MARKER_COLOR,
                    markdown::SpanKind::Text(markdown::Emphasis::Code) => CODE_COLOR,
                    markdown::SpanKind::Text(markdown::Emphasis::Link) => LINK_COLOR,
                    // Heading prose takes the level's color (the marker and any
                    // inline code on the line keep their own); body prose is FG.
                    markdown::SpanKind::Text(_) => match line.heading {
                        Some(level) => HEADING_COLORS
                            .get(level.saturating_sub(1) as usize)
                            .copied()
                            .unwrap_or(FOREGROUND),
                        None => FOREGROUND,
                    },
                };
                // Code backgrounds (inline panels and the full-width block panel)
                // are emitted in the pre-pass above, beneath the selection; here we
                // only draw the glyphs over them. An inline code run's glyphs sit its
                // pill pad in from the pen, and the pen clears the pad after them, so
                // the panel (drawn from the same geometry) frames the code without
                // painting over the spaces around it.
                let pad = inline_code_pad(run.kind, line.code_block, self.scale);
                // A revealed backtick delimiter is nudged in against the code so it
                // hugs the glyphs inside the pill; the nudge is draw-only, so the pen
                // still advances by the run's true width (subtract the shift back out).
                let shift = backtick_reveal_shift(rope, runs, i, self.scale);
                let run_x = pen + pad + shift;
                let run_end = self.push_text(fk, run_x, baseline, color, slice, m);
                pen = run_end + pad - shift;
                // Inline decorations over the glyphs, each spanning the run's drawn
                // width: a link is underlined and struck text gets a line through its
                // middle, both in the run's own color. Thickness scales with the
                // line's size so it stays proportionate on larger headings. A done
                // task only dims its text (the color above), so its label stays
                // legible rather than crossed out.
                let deco = match run.kind {
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
                    self.list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: run_x,
                            y,
                            w: run_end - run_x,
                            h: thickness,
                        },
                        color,
                    });
                }
            }
        }
    }

    /// Draw one table row: the header tint (header rows only), the grid borders
    /// (a vertical at every column edge full row height, a top border on the
    /// table's first row and a bottom border under every row so the rows read as a
    /// continuous grid), then each cell's wrapped text, aligned within its column.
    /// `box_top` is already in screen space.
    fn table_row(&mut self, row_x: i32, box_top: i32, size: u32, t: &TableRow, m: Metrics) {
        if t.header {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x: row_x,
                    y: box_top,
                    w: t.width,
                    h: t.height,
                },
                color: TABLE_HEADER_BG,
            });
        }
        let border = self.scale.px(TABLE_BORDER);
        for &bx in &t.borders {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x: row_x + bx,
                    y: box_top,
                    w: border,
                    h: t.height,
                },
                color: TABLE_BORDER_COLOR,
            });
        }
        // The first row draws the table's top edge; every row draws its own bottom
        // edge, which doubles as the separator to the row below.
        if t.first {
            self.list.push(DrawCmd::Fill {
                rect: Rect {
                    x: row_x,
                    y: box_top,
                    w: t.width,
                    h: border,
                },
                color: TABLE_BORDER_COLOR,
            });
        }
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: row_x,
                y: box_top + t.height - border,
                w: t.width,
                h: border,
            },
            color: TABLE_BORDER_COLOR,
        });
        let max_lines = row_max_lines(t.height, m.line_height, self.scale);
        let cell_pad_y = self.scale.px(TABLE_CELL_PAD_Y);
        for (col, cell) in t.columns.iter().zip(&t.cells) {
            let n_lines = cell.lines.len().max(1);
            for (k, cl) in cell.lines.iter().enumerate() {
                let baseline =
                    box_top + cell_pad_y + cell_line_top(k, n_lines, max_lines, m) + m.ascent;
                // Align the wrapped line within the column's content width.
                let tx = aligned_x(*col, cl.width);
                self.push_cell_line(row_x + tx, baseline, size, t.header, cl, m);
            }
        }
    }

    /// Draw one wrapped line of a table cell: its visible runs in their own faces
    /// and colors, header cells emboldened, with the same link/strike decorations
    /// the body text loop draws. Markers are already hidden in a cell's runs.
    fn push_cell_line(
        &mut self,
        x: i32,
        baseline: i32,
        size: u32,
        header: bool,
        cl: &CellLine,
        m: Metrics,
    ) {
        let mut pen = x;
        for (i, run) in cl.runs.iter().enumerate() {
            if !run.visible {
                continue;
            }
            let fk = face_key(size, run.kind, header);
            let color = match run.kind {
                markdown::SpanKind::Marker if is_code_marker(self.rope, run) => CODE_MARKER_COLOR,
                markdown::SpanKind::Marker => MARKER_COLOR,
                markdown::SpanKind::Code(tok) => code_token_color(tok),
                markdown::SpanKind::Text(markdown::Emphasis::Code) => CODE_COLOR,
                markdown::SpanKind::Text(markdown::Emphasis::Link) => LINK_COLOR,
                markdown::SpanKind::Text(_) => FOREGROUND,
            };
            // An inline code run's glyphs sit its pill pad in from the pen, and the
            // pen clears the pad after them, so the tint panel frames the code without
            // covering the text around it, the same as the body loop. A revealed
            // backtick is nudged in against the code (draw-only, advance unchanged).
            let pad = inline_code_pad(run.kind, false, self.scale);
            let shift = backtick_reveal_shift(self.rope, &cl.runs, i, self.scale);
            let run_x = pen + pad + shift;
            let run_end = self.push_text(
                fk,
                run_x,
                baseline,
                color,
                self.rope.byte_slice(run.start..run.end),
                m,
            );
            pen = run_end + pad - shift;
            let deco = match run.kind {
                markdown::SpanKind::Text(markdown::Emphasis::Link) => {
                    Some(baseline + m.descent / 3)
                }
                markdown::SpanKind::Text(markdown::Emphasis::Strikethrough) => {
                    Some(baseline - m.ascent / 3)
                }
                _ => None,
            };
            if let Some(y) = deco {
                let thickness = (size as i32 / 14).max(1);
                self.list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: run_x,
                        y,
                        w: run_end - run_x,
                        h: thickness,
                    },
                    color,
                });
            }
        }
    }

    /// Draw a table add-row / add-column hover strip: a subtle rounded fill with a
    /// centered "+", inviting a click to grow the table by a row or a column.
    fn table_button(&mut self, button: crate::layout::TableButton) {
        let Some(rect) = self.screen(button.rect) else {
            return;
        };
        self.list.push(DrawCmd::RoundRect {
            rect,
            color: TABLE_HEADER_BG,
            radius: self.scale.px(CODE_RADIUS),
            corners: display::RoundedCorners::Both,
        });
        let face = FaceKey::Prose {
            size: self.body_size,
            style: FontStyle::Regular,
        };
        let m = self.fonts.metrics(self.body_size);
        let plus_w = shape::text_advance(self.fonts.face_for(face), "+").round() as i32;
        let x = rect.x + (rect.w - plus_w) / 2;
        let baseline = rect.y + (rect.h + m.ascent - m.descent) / 2;
        self.push_text(face, x, baseline, MARKER_COLOR, "+".to_string(), m);
    }

    /// The caret, from the pre-scroll rectangle its animation produced. A zero
    /// radius is the plain bar (a `Fill`, the pre-animation appearance); a positive
    /// radius is a smear mid-motion (a rounded rectangle). Culled when it scrolls
    /// out of the viewport.
    fn caret(&mut self, caret: &CaretPaint) {
        let Some(rect) = self.screen(caret.rect) else {
            return;
        };
        if caret.radius > 0 {
            self.list.push(DrawCmd::RoundRect {
                rect,
                color: CARET_COLOR,
                radius: caret.radius,
                corners: display::RoundedCorners::Both,
            });
        } else {
            self.list.push(DrawCmd::Fill {
                rect,
                color: CARET_COLOR,
            });
        }
    }

    /// The scrollbar for this pane, floating in the right margin.
    fn scrollbar(&mut self, bar: &scroll::Scrollbar, content_height: i32) {
        push_scrollbar(
            &mut self.list,
            self.view,
            bar,
            content_height,
            self.scroll_y,
            BACKGROUND,
            self.scale,
        );
    }

    /// The match tick strip, riding the scrollbar's lane.
    fn find_ticks(&mut self, find: &FindPaint, content_height: i32) {
        push_find_ticks(&mut self.list, find, self.view, content_height, self.scale);
    }

    /// The find bar panel, floating top-right over everything else.
    fn find_bar(&mut self, find: &FindPaint) {
        push_find_bar(
            &mut self.list,
            self.fonts,
            find,
            self.view,
            self.scale,
            self.body_size,
        );
    }
}

/// Append the match tick strip while find is open: one mark per match at its
/// document-proportional height, so the distribution of hits is visible at a glance
/// and survives scrolling. Ticks are deduplicated per two-pixel row, which bounds the
/// strip to O(viewport) commands however many matches exist; the current match's tick
/// is wider and bright.
///
/// The strip shares the scrollbar's lane and is drawn after it, so the ticks read as
/// marks on the bar (and stay visible where the thumb passes over them): the two say
/// the same thing, where you are in the document and where the hits are.
fn push_find_ticks(
    list: &mut DisplayList,
    find: &FindPaint,
    view: Rect,
    content_height: i32,
    scale: Scale,
) {
    if find.state.matches().is_empty() || view.h <= 0 {
        return;
    }
    let tick_h = scale.px(TICK_HEIGHT);
    let tick_w = scale.px(TICK_WIDTH);
    let content_h = content_height.max(view.h) as i64;
    let lane = scroll_lane(view, scale).track;
    let right = lane.x + lane.w;
    let mut occupied = vec![false; (view.h / 2 + 1) as usize];
    for i in 0..find.state.matches().len() {
        // Each match's content-space top comes from the line index, so an offscreen
        // match still maps to its true position rather than the window's edge.
        let top = find.tick_tops.get(i).copied().unwrap_or(0) as i64;
        let offset = (top * (view.h - tick_h) as i64 / content_h) as i32;
        let bucket = (offset / 2).clamp(0, view.h / 2) as usize;
        let current = i == find.state.current();
        if occupied[bucket] && !current {
            continue;
        }
        occupied[bucket] = true;
        let (w, color) = if current {
            (tick_w + scale.px(3), CARET_COLOR)
        } else {
            (tick_w, FIND_TICK_COLOR)
        };
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: right - w,
                y: view.y + offset,
                w,
                h: tick_h,
            },
            color,
        });
    }
}

/// The byte range spanned by the visual lines that touch the viewport, for
/// culling by byte offset before any geometry is measured. Conservative: a
/// line's visibility is judged by its text box, and the empty viewport yields
/// an empty range.
fn visible_byte_range(fonts: &Fonts, layout: &Layout, scroll_y: i32, view: Rect) -> (usize, usize) {
    let mut range = (usize::MAX, 0);
    for line in layout.lines() {
        let m = fonts.metrics(line.size);
        let baseline = line.baseline - scroll_y;
        if baseline - m.ascent >= view.y + view.h || baseline + m.descent <= view.y {
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

/// Append the find bar: a rounded panel floating at the top-right of the pane with
/// the query (its tail, when it overflows, since the end is where editing happens), a
/// steady caret after it, and a match counter. Drawn from plain display-list
/// commands, so damage tracking and clipped repaints treat it like any other
/// content.
fn push_find_bar(
    list: &mut DisplayList,
    fonts: &Fonts,
    find: &FindPaint,
    view: Rect,
    scale: Scale,
    body_size: u32,
) {
    let size = body_size;
    let caret_w = scale.px(CARET_WIDTH);
    let margin = scale.px(MARGIN);
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
    let counter = if find.state.query().is_empty() {
        String::new()
    } else if find.state.error() {
        "!".to_string()
    } else if find.state.matches().is_empty() {
        "0/0".to_string()
    } else {
        format!(
            "{}/{}",
            find.state.current() + 1,
            find.state.matches().len()
        )
    };
    let counter_w = shape::text_advance(face, &counter).ceil() as i32;
    let flag_word_w = shape::text_advance(face, "W").ceil() as i32;
    let flag_regex_w = shape::text_advance(face, ".*").ceil() as i32;
    let gap = pad / 2;
    let tail_w = flag_word_w + gap + flag_regex_w + gap + counter_w;

    // Keep a field's display within a third of the window by dropping leading
    // clusters: the end is where editing happens. Queries are short, so the
    // remeasure loop is cheap.
    fn tail_that_fits<'t>(
        face: &crate::platform::freetype::Face,
        text: &'t str,
        max: f32,
    ) -> &'t str {
        let mut shown = text;
        while shape::text_advance(face, shown) > max && !shown.is_empty() {
            shown = &shown[grapheme::next_boundary(shown, 0)..];
        }
        shown
    }
    let field_max = (view.w / 3).max(3 * size as i32) as f32;
    let query = tail_that_fits(face, find.state.query(), field_max);
    let query_w = shape::text_advance(face, query).ceil() as i32;
    let replace = find.replace().map(|r| tail_that_fits(face, r, field_max));
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
    let field_w = query_w.max(replace_row_w) + caret_w + 2;
    let w = (field_w + pad + tail_w + 2 * pad).max(6 * size as i32);
    let h = rows * row_h + (rows + 1) * (pad / 2);
    // Tucked into the pane's top-right corner, clear of the scrollbar's lane.
    let x = view.x + view.w - w - margin;
    let y = view.y + margin / 2;
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
                w: caret_w,
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
        if find.state.selected() && !find.focus_replace() {
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
        let color = if find.state.error() {
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
    if !find.focus_replace() {
        caret_after(list, pen + query_w, baseline);
    }

    // The flags and counter, right-aligned on the query row. Each flag glyph is
    // lit when its mode is on, dim otherwise.
    let mut tx = x + w - pad - tail_w;
    for (glyph, on, glyph_w) in [
        ("W", find.state.word(), flag_word_w),
        (".*", find.state.regex(), flag_regex_w),
    ] {
        let color = if on { FOREGROUND } else { MARKER_COLOR };
        list.push(text_cmd(
            key,
            tx,
            baseline,
            tx + glyph_w,
            color,
            glyph.to_string(),
            m,
        ));
        tx += glyph_w + gap;
    }
    if !counter.is_empty() {
        let color = if find.state.error() {
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
            if find.state.selected() && find.focus_replace() {
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
        if find.focus_replace() {
            caret_after(list, pen + arrow_w + replace_w, baseline);
        }
    }
}

/// Append a checkbox at `rect` (screen space): unchecked is an outlined rounded
/// square, checked fills with the accent and carries a check mark drawn from a
/// distance field. The prose face has no checkbox glyphs to lean on, so the box is
/// built from fills, and `behind` is the color it sits on: it inks the check mark
/// and hollows the unchecked box's border. Drawn for a task item (over the editor,
/// from [`Layout::checkbox_rect`]) and for the sidebar footer's filter toggle (over
/// the tree, where the surface is darker).
fn push_checkbox(list: &mut DisplayList, rect: Rect, checked: bool, behind: u32) {
    let radius = (rect.w / 4).max(2);
    list.push(DrawCmd::RoundRect {
        rect,
        color: if checked { LINK_COLOR } else { MARKER_COLOR },
        radius,
        corners: display::RoundedCorners::Both,
    });
    if checked {
        list.push(DrawCmd::CheckMark {
            rect,
            color: behind,
        });
    } else {
        let t = (rect.w / 8).max(1);
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: rect.x + t,
                y: rect.y + t,
                w: rect.w - 2 * t,
                h: rect.h - 2 * t,
            },
            color: behind,
            radius: (radius - t).max(1),
            corners: display::RoundedCorners::Both,
        });
    }
}

/// Append the selection chrome for a selected image at picture rect `pic` (screen
/// space): a four-edge accent border and the corner `</>` toggle button.
fn push_image_chrome(
    list: &mut DisplayList,
    fonts: &Fonts,
    pic: &Rect,
    scale: Scale,
    body_size: u32,
) {
    let t = scale.px(IMAGE_BORDER_THICKNESS);
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

    let btn = image_button_rect(pic, scale);
    list.push(DrawCmd::RoundRect {
        rect: btn,
        color: IMAGE_BUTTON_BG,
        radius: scale.px(IMAGE_BUTTON_RADIUS),
        corners: display::RoundedCorners::Both,
    });
    // Center the glyph in both axes from the code-face metrics.
    let fk = FaceKey::Code { size: body_size };
    let face = fonts.face_for(fk);
    let glyph = "</>";
    let advance = shape::text_advance(face, glyph).round() as i32;
    let m = fonts.metrics(body_size);
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
    use std::time::{Duration, Instant};

    use super::*;
    use crate::theme::FONT_SIZE;

    /// The tick strip's per-match tops, matching what the app derives from its line
    /// index: each match's line baseline.
    fn tick_tops(layout: &Layout, matches: &[(usize, usize)]) -> Vec<i32> {
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
        tops
    }
    use crate::rope::Rope;

    /// A pane at the window origin, the shape the layout helpers below build into,
    /// with a document no taller than the column (so no scrollbar) and a bar at rest.
    const RESTING_BAR: scroll::Scrollbar = scroll::Scrollbar::hidden();

    fn view(w: i32, h: i32) -> Rect {
        Rect { x: 0, y: 0, w, h }
    }

    fn paint_params(w: i32, h: i32) -> PaintParams<'static> {
        PaintParams {
            view: view(w, h),
            scroll_y: 0,
            content_height: h,
            scrollbar: &RESTING_BAR,
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            selection: None,
            caret: None,
            find: None,
            table_hover: None,
        }
    }

    /// A tab occupying `[x, x + w)` on the strip, with the close box irrelevant to
    /// the drop math.
    fn tab_at(x: i32, w: i32) -> TabHit {
        TabHit {
            rect: Rect { x, y: 0, w, h: 30 },
            close: Rect {
                x: x + w,
                y: 0,
                w: 0,
                h: 0,
            },
            label: String::new(),
        }
    }

    #[test]
    fn tab_drop_index_tracks_the_neighbors_the_pointer_has_passed() {
        // Three tabs of unequal width: [0,100), [100,150), [150,300); midpoints at
        // 50, 125, 225.
        let tabs = vec![tab_at(0, 100), tab_at(100, 50), tab_at(150, 150)];

        // Dragging the first tab. It stays at 0 while left of tab 1's midpoint,
        // slides past tab 1 beyond 125, then past tab 2 beyond 225.
        assert_eq!(tab_drop_index(&tabs, Some(0), 40.0), 0);
        assert_eq!(tab_drop_index(&tabs, Some(0), 130.0), 1);
        assert_eq!(tab_drop_index(&tabs, Some(0), 260.0), 2);

        // Dragging the last tab back toward the front: the dragged tab's own slot is
        // skipped, so only tabs 0 and 1 count.
        assert_eq!(tab_drop_index(&tabs, Some(2), 260.0), 2);
        assert_eq!(tab_drop_index(&tabs, Some(2), 110.0), 1);
        assert_eq!(tab_drop_index(&tabs, Some(2), 30.0), 0);

        // Dropping a tab in from the other pane: none of these tabs is the dragged
        // one, so every midpoint the pointer passed counts toward the insert slot.
        assert_eq!(tab_drop_index(&tabs, None, 30.0), 0);
        assert_eq!(tab_drop_index(&tabs, None, 130.0), 2);
        assert_eq!(tab_drop_index(&tabs, None, 260.0), 3);
    }

    #[test]
    fn top_bar_draws_vault_label_and_tabs_with_the_active_one_highlighted() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let top = top_bar_height(fonts.metrics(FONT_SIZE), Scale::ONE);
        let bar = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: top,
        };
        let vrect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: top,
        };
        let strip_rect = Rect {
            x: 206,
            y: 0,
            w: 594,
            h: top,
        };
        // Two tabs, the second active, the first unsaved.
        let labels = vec!["intro".to_string(), "todo".to_string()];
        let dirty = vec![true, false];
        let strips = [
            Some(TabStrip {
                rect: strip_rect,
                labels: &labels,
                dirty: &dirty,
                active: 1,
                hovered: None,
                close_hover: None,
                new_hover: false,
                drag: None,
            }),
            None,
        ];
        // A splitter divider between the vault region and pane A.
        let dividers = [Rect {
            x: 200,
            y: top,
            w: 6,
            h: 100,
        }];
        let mut list: DisplayList = Vec::new();
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        push_top_bar(
            &mut list,
            &ctx,
            bar,
            Some((vrect, "Notes")),
            strips,
            &dividers,
        );

        let has_text = |s: &str| {
            list.iter()
                .any(|c| matches!(c, DrawCmd::Text { text, .. } if text == s))
        };
        let has_round = |color: u32| {
            list.iter()
                .any(|c| matches!(c, DrawCmd::RoundRect { color: c2, .. } if *c2 == color))
        };
        // The strip fill, the vault label, and both tab labels reach the list.
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == TOP_BAR_BG)),
            "top bar background"
        );
        assert!(has_text("Notes"), "vault label");
        assert!(has_text("intro"), "resting tab label");
        assert!(has_text("todo"), "active tab label");
        // The active tab (index 1) melts into the pane background and shows a close
        // glyph; the resting unsaved tab (index 0) shows a dirty dot instead.
        assert!(has_round(TAB_ACTIVE_BG), "active tab highlight");
        assert!(has_text("×"), "close glyph on the active tab");
        assert!(
            has_round(TAB_DIRTY_COLOR),
            "dirty dot on the resting unsaved tab"
        );
        // The new-tab button sits at the end of the strip.
        assert!(has_text("+"), "new-tab button");
    }

    #[test]
    fn task_checkboxes_and_done_styling_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "- [ ] open\n- [x] done\nx";
        let rope = Rope::from_str(src);
        let (w, h) = (600i32, 200i32);
        let wrap = w - 2 * MARGIN;
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
        assert_eq!(round(LINK_COLOR), 1, "the checked fill");
        let accent = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::RoundRect { rect, color, .. } if *color == LINK_COLOR => Some(*rect),
                _ => None,
            })
            .expect("the checked box");
        let inside = |rect: &Rect| {
            rect.x >= accent.x
                && rect.x + rect.w <= accent.x + accent.w
                && rect.y >= accent.y
                && rect.y + rect.h <= accent.y + accent.h
        };
        // The check mark: one distance-field command in the background color,
        // inked inside the checked box.
        let checks = list
            .iter()
            .filter(|c| {
                matches!(c, DrawCmd::CheckMark { rect, color } if *color == BACKGROUND && inside(rect))
            })
            .count();
        assert_eq!(checks, 1, "one check mark inside the checked box");
        // The unchecked box is hollowed by a background inset (outside the checked box).
        assert!(
            list.iter().any(|c| {
                matches!(c, DrawCmd::RoundRect { rect, color, .. } if *color == BACKGROUND && !inside(rect))
            }),
            "the unchecked shell is hollowed by a background inset"
        );
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
        // The done item's text dims to the marker color; the open item's text
        // stays foreground.
        let text_color = |needle: &str| {
            list.iter().find_map(|c| match c {
                DrawCmd::Text { text, color, .. } if text == needle => Some(*color),
                _ => None,
            })
        };
        assert_eq!(text_color("open"), Some(FOREGROUND));
        assert_eq!(text_color("done"), Some(MARKER_COLOR), "done text is dim");
        // A done task is only dimmed, never crossed out: no strike fill runs
        // through its label.
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
            !list.iter().any(|c| {
                matches!(c, DrawCmd::Fill { rect, color }
                    if *color == MARKER_COLOR
                        && rect.x == done_run.0
                        && rect.y > done_run.1 - m.ascent
                        && rect.y < done_run.1)
            }),
            "a done task is not struck through, only dimmed"
        );
    }

    #[test]
    fn a_list_bullet_renders_as_a_disc_in_the_marker_color() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "- item\nx";
        let rope = Rope::from_str(src);
        let (w, h) = (600i32, 200i32);
        let wrap = w - 2 * MARGIN;
        let off = src.len() - 1;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (off, off));
        let list = build_display_list(&fonts, &layout, &rope, &paint_params(w, h));

        // The dash gives way to a filled disc: a square rounded rect whose radius
        // is half its side, in the muted marker color, sitting left of the text.
        let disc = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::RoundRect {
                    rect,
                    color,
                    radius,
                    ..
                } if *color == MARKER_COLOR => Some((*rect, *radius)),
                _ => None,
            })
            .expect("a bullet line draws a disc");
        let (rect, radius) = disc;
        assert_eq!(rect.w, rect.h, "a square");
        assert_eq!(radius, rect.w / 2, "radius is half the side, so a disc");
        assert!(rect.w >= 4, "a visible dot");
        // The disc hangs a clear gap short of the item's text, not crowding it.
        let text_x = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::Text { text, x, .. } if text == "item" => Some(*x),
                _ => None,
            })
            .expect("the item text");
        assert!(
            rect.x + rect.w <= text_x - 3,
            "the disc sits a gap short of the text (disc ends at {}, text at {text_x})",
            rect.x + rect.w
        );
        // The raw dash is substituted, not drawn.
        assert!(
            !list
                .iter()
                .any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains('-'))),
            "no raw dash glyph on the bullet line"
        );
    }

    #[test]
    fn find_bar_and_match_highlights_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("one two one two one\n");
        let (w, h) = (600i32, 200i32);
        let wrap = w - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        // Drive a real find state to the same shape: query "one", three matches,
        // the current one at index 1 (byte 8) — the anchor aims `current` at the
        // first match at or after it.
        let buf = crate::text::TextBuffer::from_text("one two one two one\n".to_string());
        let mut bar = FindState::new();
        bar.open_at(8, &buf);
        bar.push_str("one", &buf);
        let tops = tick_tops(&layout, bar.matches());
        let params = PaintParams {
            // The app mirrors the current match into the selection.
            selection: Some((8, 11)),
            find: build_find_paint(&bar, &tops),
            ..paint_params(w, h)
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
        let (w, h) = (600i32, 200i32);
        let wrap = w - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        // Drive a real find state: whole-word query "one" (three matches, current
        // 0) with the replace row open, focused, and holding "two".
        let buf = crate::text::TextBuffer::from_text("one two one\nfiller\none more\n".to_string());
        let mut bar = FindState::new();
        bar.open_at(0, &buf);
        bar.push_str("one", &buf);
        bar.toggle_word(&buf);
        bar.open_replace_at(0, &buf);
        bar.push_str("two", &buf);
        let tops = tick_tops(&layout, bar.matches());
        let params = PaintParams {
            selection: Some((0, 3)),
            find: build_find_paint(&bar, &tops),
            ..paint_params(w, h)
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
        let edge = w - 10;
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

    #[test]
    fn a_table_reaches_the_display_list_as_grid_and_cells() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "| Name | Score |\n| --- | ---: |\n| Ada | 92 |\n";
        let rope = Rope::from_str(src);
        let (w, h) = (600i32, 300i32);
        let wrap = w - 2 * MARGIN;
        // Tables render as a grid regardless of the cursor, so `active` is moot.
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let list = build_display_list(&fonts, &layout, &rope, &paint_params(w, h));

        let fills = |color: u32| {
            list.iter()
                .filter(|c| matches!(c, DrawCmd::Fill { color: c2, .. } if *c2 == color))
                .count()
        };
        assert_eq!(fills(TABLE_HEADER_BG), 1, "the header row's tint");
        // Two columns: three verticals plus a top and bottom edge on the header,
        // and three verticals plus a bottom edge on the body row.
        assert!(
            fills(TABLE_BORDER_COLOR) >= 8,
            "grid border lines, got {}",
            fills(TABLE_BORDER_COLOR)
        );
        // Cell text is drawn from the source; the pipes are not.
        let texts: String = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        for cell in ["Name", "Score", "Ada", "92"] {
            assert!(texts.contains(cell), "cell {cell:?} drawn, got {texts:?}");
        }
        assert!(!texts.contains('|'), "the pipes are not drawn");
    }

    #[test]
    fn a_table_cell_reveals_its_backticks_when_the_cursor_is_on_the_row() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "| File |\n| --- |\n| `main.rs`, `error.rs` |\n";
        let rope = Rope::from_str(src);
        let (w, h) = (600i32, 300i32);
        let wrap = w - 2 * MARGIN;
        let body = rope.line_to_byte(2);

        // Cursor outside the table: the cell is cooked, so no backtick is drawn but
        // the two code spans still get their tint pills.
        let cooked = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let cooked_list = build_display_list(&fonts, &cooked, &rope, &paint_params(w, h));
        let backticks = |list: &[DrawCmd]| {
            list.iter()
                .filter(|c| {
                    matches!(c, DrawCmd::Text { text, color, .. }
                    if text.contains('`') && *color == CODE_MARKER_COLOR)
                })
                .count()
        };
        let pills = |list: &[DrawCmd]| {
            list.iter()
                .filter(|c| matches!(c, DrawCmd::RoundRect { color, .. } if *color == CODE_BG))
                .count()
        };
        assert_eq!(
            backticks(&cooked_list),
            0,
            "cooked cell hides its backticks"
        );
        assert_eq!(
            pills(&cooked_list),
            2,
            "two inline-code pills, one per span"
        );

        // Cursor in the body cell: the backticks reveal as CODE_MARKER_COLOR text.
        let revealed = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (body, body));
        let revealed_list = build_display_list(&fonts, &revealed, &rope, &paint_params(w, h));
        assert!(
            backticks(&revealed_list) > 0,
            "the active row draws its backticks so they can be edited"
        );
        assert_eq!(
            pills(&revealed_list),
            2,
            "still two pills, now over the backticks"
        );
    }

    #[test]
    fn a_hovered_table_button_reaches_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("| a | b |\n| - | - |\n| c | d |\n");
        let (w, h) = (600i32, 300i32);
        let wrap = w - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        // Point just below the last row: the add-row strip.
        let bot = layout
            .lines()
            .iter()
            .rev()
            .find(|v| v.table.is_some())
            .unwrap();
        let box_bottom = bot.baseline - fonts.metrics(bot.size).ascent
            + bot.table.as_ref().unwrap().box_height();
        let button = layout
            .table_button_at(&fonts, &rope, (MARGIN + 5) as f32, (box_bottom + 4) as f32)
            .expect("an add-row button");
        let mut params = paint_params(w, h);
        params.table_hover = Some(button);
        let list = build_display_list(&fonts, &layout, &rope, &params);
        // The strip's "+" is drawn for the affordance.
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "+")),
            "the add button's + reaches the list"
        );
    }

    /// A throwaway directory removed on drop, for driving the real [`push_tree`].
    struct TreeDir(std::path::PathBuf);

    impl TreeDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("bnknotes_paint_tree_{}_{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("folder")).expect("mkdir folder");
            std::fs::write(dir.join("note.md"), b"n").expect("write note");
            Self(dir)
        }
    }

    impl Drop for TreeDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A sidebar over `root` with nothing hovered and `selected` active, listing
    /// every folder (the first-run filter).
    fn tree_pane(root: &std::path::Path, selected: Option<usize>) -> TreePane {
        TreePane {
            tree: crate::tree::Tree::new(root.to_path_buf(), Default::default(), true),
            scroll_y: 0,
            content_height: 0,
            scrollbar: scroll::Scrollbar::hidden(),
            hovered: None,
            selected,
            filter_hovered: false,
        }
    }

    #[test]
    fn the_sidebar_reads_a_step_below_the_body_in_a_face_that_is_opened() {
        // A step down, and one that survives rounding at every size the body takes
        // across the DPI scales (14 -> 13, 28 at 2x -> 25).
        assert_eq!(tree_font_size(14), 13);
        assert_eq!(tree_font_size(28), 25);
        for body in 8..=64 {
            assert!(tree_font_size(body) < body, "body {body}");
            assert!(
                font_sizes(body).contains(&tree_font_size(body)),
                "the sidebar's face is opened, so it never falls back to the body's"
            );
        }

        // Every string the sidebar draws (its rows and its footer's label) is in that
        // size, not the body's.
        let fx = TreeDir::new("size");
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let mut list = DisplayList::new();
        let rect = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 400,
        };
        push_tree(
            &mut list,
            &fonts,
            &tree_pane(&fx.0, None),
            rect,
            Scale::ONE,
            FONT_SIZE,
        );
        let sizes: Vec<u32> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { face, .. } => Some(face.size()),
                _ => None,
            })
            .collect();
        assert!(sizes.len() >= 3, "two rows and the footer label");
        assert!(
            sizes.iter().all(|&s| s == tree_font_size(FONT_SIZE)),
            "the sidebar draws in its own size, got {sizes:?}"
        );
    }

    #[test]
    fn tree_rows_paint_chevron_column_names_and_inset_highlight() {
        let fx = TreeDir::new("rows");
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        // Row 0 (the folder) is active, so it draws its highlight.
        let tree = tree_pane(&fx.0, Some(0));
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 400,
        };
        let mut list = DisplayList::new();
        push_tree(&mut list, &fonts, &tree, rect, Scale::ONE, FONT_SIZE);

        // The active pill is a rounded rect in the hover gray, inset from both
        // column edges so background frames it left and right.
        let pill = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::RoundRect { rect, color, .. } if *color == TREE_HOVER_BG => Some(*rect),
                _ => None,
            })
            .expect("the active row's highlight");
        assert_eq!(pill.x, TREE_PAD, "highlight inset from the left edge");
        assert_eq!(
            pill.w,
            rect.w - 2 * TREE_PAD,
            "highlight inset from both edges"
        );

        // The folder draws one chevron (a vector shape, marker-colored, pointing
        // right while collapsed); the file draws none, so there is exactly one.
        let chevrons: Vec<(bool, i32)> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Chevron { rect, color, down } if *color == TREE_MARKER_COLOR => {
                    Some((*down, rect.x))
                }
                _ => None,
            })
            .collect();
        assert_eq!(chevrons.len(), 1, "one chevron, for the folder only");
        assert!(!chevrons[0].0, "a collapsed folder points right, not down");
        // Left-aligned at the row's content start, ahead of the name column.
        assert_eq!(
            chevrons[0].1,
            TREE_PAD + TREE_ROW_PAD,
            "the chevron sits at the row's content start"
        );

        // Both names read in the muted sidebar tone, dimmer than the body text and
        // brighter than the chevron beside them, aligned in the column past it. The
        // footer's label shares the tone, so only the rows' own band is scanned.
        let rows_bottom = tree_column(rect, tree_metrics(&fonts, FONT_SIZE), Scale::ONE)
            .rows
            .h;
        let names: Vec<(&str, i32)> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text {
                    text,
                    color,
                    x,
                    baseline,
                    ..
                } if *color == TREE_TEXT_COLOR && *baseline < rows_bottom => {
                    Some((text.as_str(), *x))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            [
                ("folder", TREE_PAD + TREE_ROW_PAD + TREE_MARKER_W),
                ("note", TREE_PAD + TREE_ROW_PAD + TREE_MARKER_W),
            ],
            "folder and file names share the muted color and one column"
        );
        assert_ne!(
            TREE_TEXT_COLOR, FOREGROUND,
            "the sidebar sits below the body text"
        );
        assert_ne!(
            TREE_MARKER_COLOR, TREE_TEXT_COLOR,
            "the chevron sits below the name it points at"
        );
    }

    #[test]
    fn tree_active_and_hover_share_the_gray_highlight() {
        let fx = TreeDir::new("hover");
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 400,
        };
        let gray_pill_tops = |selected, hovered| {
            let mut tree = tree_pane(&fx.0, selected);
            tree.hovered = hovered;
            let mut list = DisplayList::new();
            push_tree(&mut list, &fonts, &tree, rect, Scale::ONE, FONT_SIZE);
            list.iter()
                .filter_map(|c| match c {
                    DrawCmd::RoundRect { rect, color, .. } if *color == TREE_HOVER_BG => {
                        Some(rect.y)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };

        // Nothing active or hovered draws no highlight.
        assert!(gray_pill_tops(None, None).is_empty());
        // The active row alone draws one gray pill (regardless of focus).
        assert_eq!(gray_pill_tops(Some(0), None).len(), 1, "active only");
        // Hover alone draws one gray pill, on the hovered row.
        assert_eq!(gray_pill_tops(None, Some(1)).len(), 1, "hover only");
        // Active and hover on different rows draw two, both in the same gray.
        let two = gray_pill_tops(Some(0), Some(1));
        assert_eq!(two.len(), 2, "active and hover, distinct rows");
        assert_ne!(two[0], two[1], "the two pills sit on different rows");
    }

    #[test]
    fn the_footer_takes_its_row_off_the_bottom_of_the_column() {
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let m = tree_metrics(&fonts, FONT_SIZE);
        let column = Rect {
            x: 10,
            y: 30,
            w: 200,
            h: 400,
        };
        let col = tree_column(column, m, Scale::ONE);
        // The two parts tile the column, the footer pinned to its bottom.
        assert_eq!(col.rows.y, column.y);
        assert_eq!(col.rows.h + col.footer.h, column.h);
        assert_eq!(col.footer.y, col.rows.y + col.rows.h);
        assert_eq!(col.footer.y + col.footer.h, column.y + column.h);
        // Its button is inset from the strip on every side, like a row's pill.
        assert!(col.button.x > col.footer.x);
        assert!(col.button.y > col.footer.y);
        assert!(col.button.x + col.button.w < col.footer.x + col.footer.w);
        assert!(col.button.y + col.button.h < col.footer.y + col.footer.h);
        // A column too short for a footer gives it what there is and no rows, rather
        // than a footer hanging off the bottom.
        let squat = tree_column(Rect { h: 10, ..column }, m, Scale::ONE);
        assert_eq!(squat.footer.h, 10);
        assert_eq!(squat.rows.h, 0);
    }

    #[test]
    fn the_footer_filter_draws_its_state_and_lights_up_when_hovered() {
        let fx = TreeDir::new("footer");
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        // Wide enough for the whole label, so the assertions below name it in full; a
        // narrower sidebar ellipsizes it like any other row.
        let rect = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 400,
        };
        let paint = |show_empty: bool, hovered: bool| {
            let mut tree = tree_pane(&fx.0, None);
            tree.tree.set_show_empty_dirs(show_empty);
            tree.filter_hovered = hovered;
            let mut list = DisplayList::new();
            push_tree(&mut list, &fonts, &tree, rect, Scale::ONE, FONT_SIZE);
            list
        };
        let col = tree_column(rect, tree_metrics(&fonts, FONT_SIZE), Scale::ONE);
        let in_footer = |r: &Rect| r.y >= col.footer.y && r.y + r.h <= col.footer.y + col.footer.h;

        // Listing the empty folders is the checked state: the box fills with the
        // accent and carries a check mark.
        let on = paint(true, false);
        assert!(
            on.iter().any(
                |c| matches!(c, DrawCmd::RoundRect { rect, color, .. } if *color == LINK_COLOR && in_footer(rect))
            ),
            "the checked box fills with the accent"
        );
        assert!(
            on.iter()
                .any(|c| matches!(c, DrawCmd::CheckMark { rect, .. } if in_footer(rect))),
            "the checked box carries a check mark"
        );
        // Hiding them unchecks it: an outlined box, hollowed against the sidebar's own
        // background rather than the editor's.
        let off = paint(false, false);
        assert!(
            off.iter()
                .all(|c| !matches!(c, DrawCmd::CheckMark { rect, .. } if in_footer(rect))),
            "the unchecked box has no check mark"
        );
        assert!(
            off.iter().any(
                |c| matches!(c, DrawCmd::RoundRect { rect, color, .. } if *color == TREE_BG && in_footer(rect))
            ),
            "the unchecked box is hollowed against the sidebar background"
        );
        // The label names what the checkbox means, and both it and the pill under it
        // light up only while the pointer is on the toggle.
        let label = |list: &DisplayList| {
            list.iter().find_map(|c| match c {
                DrawCmd::Text { text, color, .. } if text == TREE_FILTER_LABEL => Some(*color),
                _ => None,
            })
        };
        assert_eq!(label(&on), Some(TREE_TEXT_COLOR), "muted at rest");
        let hot = paint(true, true);
        assert_eq!(
            label(&hot),
            Some(FOREGROUND),
            "brightened under the pointer"
        );
        assert!(
            hot.iter().any(
                |c| matches!(c, DrawCmd::RoundRect { rect, color, .. } if *color == TREE_HOVER_BG && *rect == col.button)
            ),
            "the toggle's pill fills its button box"
        );
        assert!(
            on.iter()
                .all(|c| !matches!(c, DrawCmd::RoundRect { color, .. } if *color == TREE_HOVER_BG)),
            "no pill without the pointer"
        );
    }

    #[test]
    fn tree_rows_scroll_within_the_viewport_above_the_footer() {
        let fx = TreeDir::new("scroll");
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let mut tree = tree_pane(&fx.0, None);
        // A column too short for both rows, so the scroll has somewhere to go.
        let m = tree_metrics(&fonts, FONT_SIZE);
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: tree_row_height(m, Scale::ONE) * 2
                + tree_column(
                    Rect {
                        x: 0,
                        y: 0,
                        w: 200,
                        h: 400,
                    },
                    m,
                    Scale::ONE,
                )
                .footer
                .h,
        };
        tree.scroll_y = 10_000;
        measure_tree(&mut tree, rect, m, Scale::ONE);
        // The scroll clamps against the rows' viewport, not the whole column, so the
        // last row cannot come to rest under the footer.
        let view = tree_column(rect, m, Scale::ONE).rows;
        assert_eq!(
            tree.scroll_y,
            (tree.content_height - view.h).max(0),
            "clamped to the rows viewport"
        );
    }

    /// The columns a pane paints into once the sidebar and the top bar are in front of
    /// it: not the window origin, which is the case every edge-anchored piece of chrome
    /// has to get right.
    const OFFSET_PANE: Rect = Rect {
        x: 246,
        y: 28,
        w: 800,
        h: 600,
    };

    /// A pane laid out into `view` the way the app lays one out: absolute positions,
    /// with the column's origin and margin baked into the layout.
    fn pane_layout(fonts: &Fonts, rope: &Rope, view: Rect) -> Layout {
        Layout::build(
            fonts,
            rope,
            view.x + MARGIN,
            view.y + MARGIN,
            view.w - 2 * MARGIN,
            FONT_SIZE,
            (0, 0),
        )
    }

    /// A bar as the pointer leaves it: fully lit and grown into the slider.
    fn slider() -> scroll::Scrollbar {
        let mut bar = scroll::Scrollbar::hidden();
        bar.set_near(true);
        let start = Instant::now();
        for f in 1..=20 {
            bar.tick(start + Duration::from_millis(f * 16));
        }
        assert_eq!((bar.lit(), bar.wide()), (1.0, 1.0), "the bar is fully open");
        bar
    }

    #[test]
    fn a_pane_away_from_the_origin_paints_into_its_own_column() {
        // The regression this guards: the painter used to be handed the pane's size but
        // read the layout's absolute positions, so it cleared the wrong band and culled
        // the bottom of the column (the strip the top bar's height pushed it down by).
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str(&"a line of prose\n".repeat(400));
        let view = OFFSET_PANE;
        let layout = pane_layout(&fonts, &rope, view);
        let params = PaintParams {
            view,
            scroll_y: 300,
            content_height: layout.text_height() + 2 * MARGIN,
            ..paint_params(view.w, view.h)
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);

        let bg = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::Fill { rect, color } if *color == BACKGROUND => Some(*rect),
                _ => None,
            })
            .expect("the pane's background");
        assert_eq!(bg, view, "the background clears exactly the column");
        // Text reaches the column's bottom edge: with a scrolled document there is a
        // line in every band of the column, including the last one.
        let lowest = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { bounds, .. } => Some(bounds.y + bounds.h),
                _ => None,
            })
            .max()
            .expect("text");
        let bottom = view.y + view.h;
        assert!(
            lowest > bottom - fonts.metrics(FONT_SIZE).line_height,
            "text is drawn into the column's last line ({lowest} vs bottom {bottom})"
        );
        assert!(
            list.iter().all(|c| {
                let b = match c {
                    DrawCmd::Text { bounds, .. } => *bounds,
                    DrawCmd::Fill { rect, .. } => *rect,
                    _ => return true,
                };
                b.x >= view.x && b.x + b.w <= view.x + view.w + MARGIN
            }),
            "nothing is painted outside the column"
        );
    }

    #[test]
    fn the_find_bar_and_its_ticks_ride_the_panes_right_edge() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "one two one\n".repeat(50);
        let rope = Rope::from_str(&src);
        let view = OFFSET_PANE;
        let layout = pane_layout(&fonts, &rope, view);
        let buf = crate::text::TextBuffer::from_text(src);
        let mut find = FindState::new();
        find.open_at(0, &buf);
        find.push_str("one", &buf);
        let tops = tick_tops(&layout, find.matches());
        let params = PaintParams {
            view,
            content_height: layout.text_height() + 2 * MARGIN,
            find: build_find_paint(&find, &tops),
            ..paint_params(view.w, view.h)
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);

        let panel = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::RoundRect { rect, color, .. } if *color == FIND_BG => Some(*rect),
                _ => None,
            })
            .expect("the find panel");
        assert!(
            panel.y >= view.y,
            "the panel hangs below the pane's top, not under the top bar"
        );
        assert_eq!(
            view.x + view.w - (panel.x + panel.w),
            MARGIN,
            "and one margin in from the pane's right edge"
        );
        // Every tick sits in the scrollbar's lane, right-aligned with it.
        let lane = scroll_lane(view, Scale::ONE).track;
        let ticks: Vec<Rect> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Fill { rect, color } if *color == FIND_TICK_COLOR => Some(*rect),
                _ => None,
            })
            .collect();
        assert!(!ticks.is_empty(), "the matches tick the strip");
        assert!(
            ticks.iter().all(|t| t.x + t.w == lane.x + lane.w),
            "ticks share the scrollbar's right edge: {ticks:?} vs lane {lane:?}"
        );
        assert!(
            ticks.iter().all(|t| t.y >= view.y && t.y < view.y + view.h),
            "and stay inside the pane's band"
        );
    }

    #[test]
    fn the_scrollbar_thumb_tracks_the_scroll_down_the_lane() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str(&"a line of prose\n".repeat(400));
        let view = OFFSET_PANE;
        let layout = pane_layout(&fonts, &rope, view);
        let content_height = layout.text_height() + 2 * MARGIN;
        let bar = slider();
        let thumb_at = |scroll_y: i32| {
            let params = PaintParams {
                view,
                scroll_y,
                content_height,
                scrollbar: &bar,
                ..paint_params(view.w, view.h)
            };
            build_display_list(&fonts, &layout, &rope, &params)
                .into_iter()
                .find_map(|c| match c {
                    DrawCmd::RoundRect { rect, radius, .. } if radius * 2 >= rect.w => Some(rect),
                    _ => None,
                })
                .expect("the thumb")
        };
        let lane = scroll_lane(view, Scale::ONE).track;
        let top = thumb_at(0);
        assert_eq!(top.w, SCROLL_WIDE, "grown to the slider's full width");
        assert_eq!(
            top.x + top.w,
            lane.x + lane.w,
            "right-aligned in the lane, so it grows leftward from the window edge"
        );
        assert_eq!(
            top.y, view.y,
            "at the top of the document, at the top of the lane"
        );
        assert!(top.h >= SCROLL_MIN_THUMB, "at least a grabbable length");

        let end = thumb_at(scroll::max_scroll(content_height, view.h));
        assert_eq!(
            end.y + end.h,
            view.y + view.h,
            "at the end of the document its bottom meets the lane's"
        );
        let middle = thumb_at(scroll::max_scroll(content_height, view.h) / 2);
        assert!(
            middle.y > top.y && middle.y < end.y,
            "and it travels in between"
        );
    }

    #[test]
    fn a_resting_bar_and_a_document_that_fits_paint_no_scrollbar() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let view = OFFSET_PANE;
        // A long document, but a bar at rest: the fade is the only thing that decides.
        let long = Rope::from_str(&"a line of prose\n".repeat(400));
        let layout = pane_layout(&fonts, &long, view);
        let resting = PaintParams {
            view,
            content_height: layout.text_height() + 2 * MARGIN,
            ..paint_params(view.w, view.h)
        };
        assert!(
            !has_scrollbar(&build_display_list(&fonts, &layout, &long, &resting)),
            "a hidden bar draws nothing, however long the document"
        );
        // And an open bar over a document that fits has nothing to point at.
        let short = Rope::from_str("one line\n");
        let layout = pane_layout(&fonts, &short, view);
        let bar = slider();
        let fits = PaintParams {
            view,
            content_height: layout.text_height() + 2 * MARGIN,
            scrollbar: &bar,
            ..paint_params(view.w, view.h)
        };
        assert!(
            !has_scrollbar(&build_display_list(&fonts, &layout, &short, &fits)),
            "no bar when the whole document is on screen"
        );
    }

    /// Whether the list holds a thumb: a pill (its radius is half its width) in the
    /// scrollbar's lane.
    fn has_scrollbar(list: &DisplayList) -> bool {
        let lane = scroll_lane(OFFSET_PANE, Scale::ONE).track;
        list.iter().any(|c| match c {
            DrawCmd::RoundRect { rect, radius, .. } => radius * 2 >= rect.w && rect.x >= lane.x,
            DrawCmd::Fill { rect, .. } => *rect == lane,
            _ => false,
        })
    }
}
