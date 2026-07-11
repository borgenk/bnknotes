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
    QUOTE_BAR_WIDTH, RULE_COLOR, RULE_THICKNESS, SELECTION_COLOR, SPLITTER_COLOR,
    TABLE_BORDER_COLOR, TABLE_HEADER_BG, TAB_ACTIVE_BG, TAB_ACTIVE_TEXT, TAB_BUTTON_HOVER_BG,
    TAB_BUTTON_RADIUS, TAB_CLOSE_GAP, TAB_CLOSE_W, TAB_DIRTY_COLOR, TAB_HOVER_BG,
    TAB_INACTIVE_TEXT, TAB_MAX_W, TAB_MIN_W, TAB_NEW_W, TAB_PAD_X, TAB_RADIUS, TAB_VPAD,
    TICK_HEIGHT, TICK_WIDTH, TOP_BAR_BG, TREE_BG, TREE_CHEVRON_MAX, TREE_HOVER_BG, TREE_INDENT,
    TREE_MARKER_COLOR, TREE_MARKER_W, TREE_PAD, TREE_ROW_INSET_Y, TREE_ROW_PAD, TREE_ROW_RADIUS,
    TREE_ROW_VPAD, VAULT_PAD_X,
};

/// The per-frame inputs [`build_display_list`] needs beyond the layout itself:
/// the surface size, the scroll offset, the live selection and cursor, and
/// whether the caret is on this blink. Bundled so the build signature stays
/// readable and so the perf harness can drive a build without a live surface.
pub(crate) struct PaintParams<'a> {
    pub width: usize,
    pub height: usize,
    pub scroll_y: i32,
    /// The display scale, so the painter's pixel constants (margin, caret, rule,
    /// tick strip) land at the right device size. Identity at [`Scale::ONE`].
    pub scale: Scale,
    /// Device-pixel body font size, for the find bar and other chrome sized off the
    /// body ([`crate::theme::FONT_SIZE`] at scale 1.0).
    pub body_size: u32,
    pub selection: Option<(usize, usize)>,
    pub cursor: usize,
    pub caret_visible: bool,
    /// The find bar's paint state while it is open; `None` draws no bar and no
    /// match highlights.
    pub find: Option<FindPaint<'a>>,
    /// The table add-row / add-column strip under the pointer, drawn as a hover
    /// affordance; `None` when the pointer is not over one.
    pub table_hover: Option<crate::layout::TableButton>,
}

/// What the painter needs from an open find bar: the [`FindState`] itself,
/// borrowed for the frame being built, plus the two tick-strip inputs that come
/// from the caller's line index rather than the find state.
pub(crate) struct FindPaint<'a> {
    /// The open find bar's state (query, matches, current index, mode flags).
    pub state: &'a FindState,
    /// Content-space top of each match's line, from the line index, so the tick
    /// strip places a match that sits outside the painted window correctly.
    pub tick_tops: &'a [i32],
    /// Total document content height, for the tick strip's proportional mapping.
    pub doc_height: i32,
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
    doc_height: i32,
) -> Option<FindPaint<'a>> {
    find.is_open().then_some(FindPaint {
        state: find,
        tick_tops,
        doc_height,
    })
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

/// The pixel height of one tree row: the font's line height plus vertical
/// padding above and below, so items get breathing room and a taller hit area.
/// Shared by the painter and the pointer/scroll hit-testing so they agree.
pub(crate) fn tree_row_height(m: Metrics, scale: Scale) -> i32 {
    (m.line_height + 2 * scale.px(TREE_ROW_VPAD)).max(1)
}

/// Draw the file-tree sidebar into its column `rect`: a darker background, then
/// each visible row (a folder chevron plus the name), indented by depth and
/// offset by the tree's own scroll. Updates `content_height` for scroll
/// clamping. Rows do not wrap; a long name that overhangs is covered by the
/// editor's own background, which is drawn after this.
pub(crate) fn push_tree(
    list: &mut DisplayList,
    fonts: &Fonts,
    tree: &mut TreePane,
    rect: Rect,
    scale: Scale,
    body_size: u32,
) {
    list.push(DrawCmd::Fill {
        rect,
        color: TREE_BG,
    });
    let m = fonts.metrics(body_size);
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
    tree.content_height = rows.len() as i32 * row_h + 2 * tree_pad;
    tree.scroll_y = scroll::clamp(tree.scroll_y, tree.content_height, rect.h);
    let face = FaceKey::Prose {
        size: body_size,
        style: FontStyle::Regular,
    };
    let text_right = rect.x + rect.w - tree_pad - row_pad;
    for (i, row) in rows.iter().enumerate() {
        let top = rect.y + tree_pad + i as i32 * row_h - tree.scroll_y;
        if top + row_h <= rect.y || top >= rect.y + rect.h {
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
        // left-aligned in that column and centered vertically on the label, tinted;
        // both folder and file names are drawn in the foreground.
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
            FOREGROUND,
            row.name.clone(),
            m,
        ));
    }
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
    let m = ctx.metrics();
    let tabs = tab_rects(strip.rect, strip.labels, ctx.fonts, ctx.body_size, scale);
    let radius = scale.px(TAB_RADIUS);
    let vpad = scale.px(TAB_VPAD);
    let divider = scale.px(1).max(1);
    let pad = scale.px(TAB_PAD_X);
    let is_active = |i: usize| i == strip.active;
    for (i, tab) in tabs.iter().enumerate() {
        let active = is_active(i);
        let hovered = strip.hovered == Some(i);
        if active {
            list.push(DrawCmd::RoundRect {
                rect: tab.rect,
                color: TAB_ACTIVE_BG,
                radius,
                corners: display::RoundedCorners::Top,
            });
        } else if hovered {
            list.push(DrawCmd::RoundRect {
                rect: tab.rect,
                color: TAB_HOVER_BG,
                radius,
                corners: display::RoundedCorners::Top,
            });
        } else if i + 1 < tabs.len() && !is_active(i + 1) && strip.hovered != Some(i + 1) {
            // A hairline between two resting tabs, but never touching the active or
            // hovered tab (those stand on their own fill).
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
        // The close glyph shows on the active or hovered tab (with a button
        // highlight when the pointer is on it); a resting unsaved tab shows a dot
        // in its place instead.
        let dirty = strip.dirty.get(i).copied().unwrap_or(false);
        if active || hovered {
            push_glyph_button(list, ctx, tab.close, "×", strip.close_hover == Some(i));
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
    // The new-tab (+) button at the end of the strip, drawn when it fits.
    let plus = new_tab_button_rect(strip.rect, &tabs, scale);
    if plus.x + plus.w <= strip.rect.x + strip.rect.w {
        push_glyph_button(list, ctx, plus, "+", strip.new_hover);
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
    if p.caret_visible {
        painter.caret(p.cursor);
    }
    // The find bar and its match ticks float over everything else, so they go last.
    if let Some(find) = &p.find {
        painter.find_ticks(find);
        painter.find_bar(find);
    }
    painter.list
}

/// Accumulates one frame's display list from the layout and rope. The read-only
/// context (`fonts`/`layout`/`rope`) plus the viewport (`scroll_y`/`view_h`/
/// `width`) and the content's left edge are held once so each pass reads them
/// without re-plumbing; [`Self::screen`] and [`Self::push_text`] fold the
/// shift-and-cull and measure-then-emit patterns the passes repeat.
struct Painter<'a> {
    list: DisplayList,
    fonts: &'a Fonts,
    layout: &'a Layout,
    rope: &'a crate::rope::Rope,
    scroll_y: i32,
    view_h: i32,
    width: i32,
    /// The content's left edge (the column's inner margin). Text, panels, and
    /// images hang off this so a pane's content sits in its own column; the
    /// background clears from the column's outer edge, `MARGIN` to its left.
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
            view_h: p.height as i32,
            width: p.width as i32,
            left: layout.left(),
            scale: p.scale,
            body_size: p.body_size,
        }
    }

    /// Shift a content-space rect into screen space (up by the scroll) and cull
    /// it when it falls entirely above or below the viewport. `None` means the
    /// caller emits nothing for it.
    fn screen(&self, rect: Rect) -> Option<Rect> {
        let y = rect.y - self.scroll_y;
        if y + rect.h <= 0 || y >= self.view_h {
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

    /// The background, full surface, first so a clipped repaint clears its region
    /// before the rest of the list draws over it.
    fn background(&mut self) {
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: self.left - self.scale.px(MARGIN),
                y: 0,
                w: self.width,
                h: self.view_h,
            },
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
        let panel_w = (self.width - 2 * self.scale.px(MARGIN)).max(0);
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
        let visible = visible_byte_range(self.fonts, self.layout, self.scroll_y, self.view_h);
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
        let (scroll_y, view_h, left) = (self.scroll_y, self.view_h, self.left);
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
                if box_top < view_h && box_top + img.box_height() > 0 {
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
                if box_top < view_h && box_top + table.box_height() > 0 {
                    self.table_row(left + line.indent, box_top, line.size, table, m);
                }
                continue;
            }
            // Skip lines fully above or below the viewport, so a frame only
            // rasterizes what is on screen.
            if baseline + m.descent <= 0 || baseline - m.ascent >= view_h {
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
                        w: (self.width - 2 * self.scale.px(MARGIN)).max(0),
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
                                push_checkbox(&mut self.list, rect, checked);
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

    /// The caret, drawn only on the visible half of its blink cycle.
    fn caret(&mut self, cursor: usize) {
        let (cx, cy, ch) = self.layout.caret(self.fonts, self.rope, cursor);
        self.list.push(DrawCmd::Fill {
            rect: Rect {
                x: cx,
                y: cy - self.scroll_y,
                w: self.scale.px(CARET_WIDTH),
                h: ch,
            },
            color: CARET_COLOR,
        });
    }

    /// The right-edge match tick strip, floating over the content.
    fn find_ticks(&mut self, find: &FindPaint) {
        push_find_ticks(&mut self.list, find, self.width, self.view_h, self.scale);
    }

    /// The find bar panel, floating top-right over everything else.
    fn find_bar(&mut self, find: &FindPaint) {
        push_find_bar(
            &mut self.list,
            self.fonts,
            find,
            self.width,
            self.scale,
            self.body_size,
        );
    }
}

/// Append the right-edge tick strip while find is open: one mark per match at
/// its document-proportional height, so the distribution of hits is visible at
/// a glance and survives scrolling. Ticks are deduplicated per two-pixel row,
/// which bounds the strip to O(viewport) commands however many matches exist;
/// the current match's tick is wider and bright.
fn push_find_ticks(
    list: &mut DisplayList,
    find: &FindPaint,
    width: i32,
    view_h: i32,
    scale: Scale,
) {
    if find.state.matches().is_empty() || view_h <= 0 {
        return;
    }
    let tick_h = scale.px(TICK_HEIGHT);
    let tick_w = scale.px(TICK_WIDTH);
    let content_h = find.doc_height.max(view_h) as i64;
    let mut occupied = vec![false; (view_h / 2 + 1) as usize];
    for i in 0..find.state.matches().len() {
        // Each match's content-space top comes from the line index, so an offscreen
        // match still maps to its true position rather than the window's edge.
        let top = find.tick_tops.get(i).copied().unwrap_or(0) as i64;
        let y = (top * (view_h - tick_h) as i64 / content_h) as i32;
        let bucket = (y / 2).clamp(0, view_h / 2) as usize;
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
                x: width - w - scale.px(1),
                y,
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
fn push_find_bar(
    list: &mut DisplayList,
    fonts: &Fonts,
    find: &FindPaint,
    width: i32,
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
    let field_max = (width / 3).max(3 * size as i32) as f32;
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
    let x = width - w - margin;
    let y = margin / 2;
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

/// Append a task item's checkbox at `rect` (screen space, from
/// [`Layout::checkbox_rect`]): unchecked is an outlined rounded square (an
/// inset background fill hollows the border), checked fills with the accent and
/// carries a check mark drawn from a distance field. The prose face has no
/// checkbox glyphs to lean on, so the box is built from fills.
fn push_checkbox(list: &mut DisplayList, rect: Rect, checked: bool) {
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
            color: BACKGROUND,
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
            color: BACKGROUND,
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
    use super::*;
    use crate::theme::FONT_SIZE;

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
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            selection: None,
            cursor: 0,
            caret_visible: false,
            find: None,
            table_hover: None,
        }
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
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
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
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        // Drive a real find state to the same shape: query "one", three matches,
        // the current one at index 1 (byte 8) — the anchor aims `current` at the
        // first match at or after it.
        let buf = crate::text::TextBuffer::from_text("one two one two one\n".to_string());
        let mut bar = FindState::new();
        bar.open_at(8, &buf);
        bar.push_str("one", &buf);
        let (tops, doc_h) = tick_tops(&layout, bar.matches());
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            // The app mirrors the current match into the selection.
            selection: Some((8, 11)),
            cursor: 11,
            caret_visible: false,
            find: build_find_paint(&bar, &tops, doc_h),
            table_hover: None,
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
        // Drive a real find state: whole-word query "one" (three matches, current
        // 0) with the replace row open, focused, and holding "two".
        let buf = crate::text::TextBuffer::from_text("one two one\nfiller\none more\n".to_string());
        let mut bar = FindState::new();
        bar.open_at(0, &buf);
        bar.push_str("one", &buf);
        bar.toggle_word(&buf);
        bar.open_replace_at(0, &buf);
        bar.push_str("two", &buf);
        let (tops, doc_h) = tick_tops(&layout, bar.matches());
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            selection: Some((0, 3)),
            cursor: 3,
            caret_visible: false,
            find: build_find_paint(&bar, &tops, doc_h),
            table_hover: None,
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

    #[test]
    fn a_table_reaches_the_display_list_as_grid_and_cells() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "| Name | Score |\n| --- | ---: |\n| Ada | 92 |\n";
        let rope = Rope::from_str(src);
        let (w, h) = (600usize, 300usize);
        let wrap = w as i32 - 2 * MARGIN;
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
        let (w, h) = (600usize, 300usize);
        let wrap = w as i32 - 2 * MARGIN;
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
        let (w, h) = (600usize, 300usize);
        let wrap = w as i32 - 2 * MARGIN;
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

    #[test]
    fn tree_rows_paint_chevron_column_names_and_inset_highlight() {
        let fx = TreeDir::new("rows");
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let mut tree = TreePane {
            tree: crate::tree::Tree::new(fx.0.clone(), std::collections::BTreeSet::new()),
            scroll_y: 0,
            content_height: 0,
            hovered: None,
            selected: Some(0),
        };
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 400,
        };
        let mut list = DisplayList::new();
        // Row 0 (the folder) is active, so it draws its highlight.
        push_tree(&mut list, &fonts, &mut tree, rect, Scale::ONE, FONT_SIZE);

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

        // Both names are foreground and aligned in the column past the chevron slot.
        let names: Vec<(&str, i32)> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, color, x, .. } if *color == FOREGROUND => {
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
            "folder and file names share the foreground color and one column"
        );
    }

    #[test]
    fn tree_active_and_hover_share_the_gray_highlight() {
        let fx = TreeDir::new("hover");
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 400,
        };
        let gray_pill_tops = |selected, hovered| {
            let mut tree = TreePane {
                tree: crate::tree::Tree::new(fx.0.clone(), std::collections::BTreeSet::new()),
                scroll_y: 0,
                content_height: 0,
                hovered,
                selected,
            };
            let mut list = DisplayList::new();
            push_tree(&mut list, &fonts, &mut tree, rect, Scale::ONE, FONT_SIZE);
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
}
