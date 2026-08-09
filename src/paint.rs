use std::borrow::Cow;

use crate::ai::{self, DiffKind, Preview};
use crate::config::{Vault, VaultKind};
use crate::find::{self, FindState};
use crate::layout::{
    aligned_x, cell_line_top, face_key, image_button_rect, is_code_marker, row_max_lines, CellLine,
    Gutter, Layout, TableRow, TABLE_BORDER, TABLE_CELL_PAD_Y,
};
use crate::markdown;
use crate::menu::{Menu, MenuLayout};
use crate::palette::{self, Palette, PaletteLayout};
use crate::pane::{DiffColumns, DiffView, DiskConflict, TabView, TreePane};
use crate::platform::freetype::{CodeInk, Face, FaceKey, FontStyle, Fonts, Metrics};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::platform::pixel;
use crate::platform::scroll;
use crate::platform::shape;
use crate::render::display::{self, DisplayList, DrawCmd};
use crate::scale::Scale;
use crate::theme::{
    AI_DELETE_BG, AI_ERROR_TEXT, AI_HINT, AI_INSERT_BG, AI_LINE_VPAD, AI_MAX_LINES, AI_PANEL_BG,
    AI_PANEL_BORDER, AI_PANEL_BORDER_W, AI_PANEL_PAD, AI_PANEL_RADIUS, AI_PANEL_TOP,
    AI_PANEL_WIDTH, AI_TEXT, BACKGROUND, CARET_COLOR, CARET_WIDTH, CODE_BG, CODE_COLOR,
    CODE_COMMENT_COLOR, CODE_KEYWORD_COLOR, CODE_MARKER_COLOR, CODE_NUMBER_COLOR, CODE_PAD_X,
    CODE_PAD_Y, CODE_RADIUS, CODE_STRING_COLOR, CONFLICT_BANNER_BG, CONFLICT_BANNER_TEXT,
    CONFLICT_BUTTON_BG, CONFLICT_BUTTON_HOVER_BG, DECOR_TEXT_GAP, DIFF_DELETE_ACCENT, DIFF_DIVIDER,
    DIFF_DIVIDER_W, DIFF_GUTTER_BORDER, DIFF_GUTTER_PAD, DIFF_GUTTER_TEXT, DIFF_HATCH,
    DIFF_HATCH_PERIOD, DIFF_HATCH_STROKE, DIFF_INSERT_ACCENT, DIFF_LINE_DELETE_BG,
    DIFF_LINE_INSERT_BG, DIFF_LINE_PAD, FIND_BG, FIND_ERROR_COLOR, FIND_MATCH_COLOR,
    FIND_TICK_COLOR, FOREGROUND, HEADING_COLORS, IMAGE_BORDER_COLOR, IMAGE_BORDER_THICKNESS,
    IMAGE_BUTTON_BG, IMAGE_BUTTON_RADIUS, IMAGE_VIEW_BG, IMAGE_VIEW_PAD, LINK_COLOR, MARGIN,
    MARKER_COLOR, MENU_BG, MENU_BORDER, MENU_BORDER_W, MENU_HOVER_BG, MENU_ITEM_PAD_X,
    MENU_ITEM_RADIUS, MENU_RADIUS, MENU_TEXT, PALETTE_BG, PALETTE_BORDER, PALETTE_BORDER_W,
    PALETTE_HINT, PALETTE_MATCH, PALETTE_RADIUS, PALETTE_ROW_PAD_X, PALETTE_ROW_RADIUS,
    PALETTE_SELECTED_BG, PALETTE_TEXT, QUOTE_BAR_COLOR, QUOTE_BAR_WIDTH, RAIL_ACTIVE_COLOR,
    RAIL_ACTIVE_SPAN, RAIL_ACTIVE_W, RAIL_BG, RAIL_GAP, RAIL_HOVER_INK, RAIL_HOVER_LIFT,
    RAIL_INK_DARK, RAIL_INK_LIGHT, RAIL_PAD, RAIL_RADIUS, RAIL_TEMP_RING, RULE_COLOR,
    RULE_THICKNESS, SCROLL_INK, SCROLL_INSET, SCROLL_MIN_THUMB, SCROLL_NEAR, SCROLL_THIN,
    SCROLL_THUMB_COVER, SCROLL_THUMB_HOVER_COVER, SCROLL_TROUGH_COVER, SCROLL_WIDE,
    SELECTION_COLOR, SPLITTER_COLOR, TABLE_BORDER_COLOR, TABLE_HEADER_BG, TAB_ACTIVE_BG,
    TAB_ACTIVE_TEXT, TAB_BUTTON_HOVER_BG, TAB_BUTTON_RADIUS, TAB_CLOSE_GAP, TAB_CLOSE_W,
    TAB_DIRTY_COLOR, TAB_HOVER_BG, TAB_INACTIVE_TEXT, TAB_MAX_W, TAB_MIN_W, TAB_NEW_W, TAB_PAD_X,
    TAB_PINNED_W, TAB_VPAD, TICK_HEIGHT, TICK_WIDTH, TOP_BAR_BG, TREE_BG, TREE_FOOTER_GAP,
    TREE_FOOTER_VPAD, TREE_HOVER_BG, TREE_INDENT, TREE_MARKER_COLOR, TREE_MARKER_W, TREE_PAD,
    TREE_ROW_INSET_Y, TREE_ROW_PAD, TREE_ROW_RADIUS, TREE_TEXT_COLOR, VAULT_PAD_X,
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
    /// The logical-line start of the heading whose fold chevron the pointer is
    /// over. An open section shows its chevron only here; a folded one shows it
    /// regardless, so this only ever adds a chevron.
    pub fold_hover: Option<usize>,
    /// The inline diff preview's tinted byte ranges, when this pane is rendering a
    /// Ready AI diff over its synthetic document; `None` for a normal pane.
    pub diff: Option<DiffTint<'a>>,
}

/// The byte ranges of an inline diff preview to tint over the synthetic preview
/// document: a green wash behind insertions, a red wash plus a strikethrough behind
/// deletions. The ranges are into the document the layout was built from.
pub(crate) struct DiffTint<'a> {
    pub inserts: &'a [(usize, usize)],
    pub deletes: &'a [(usize, usize)],
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
        clip_x: None,
    }
}

/// Scissor a text command to the window x-range `[left, right)`: each glyph is
/// clipped there at draw time, and the damage bounds tighten to the same range so
/// a repaint stays inside it. Used for the tab labels, which slide under the
/// strip's edges when the strip scrolls. A non-text command passes through.
fn clip_text_x(mut cmd: DrawCmd, left: i32, right: i32) -> DrawCmd {
    if let DrawCmd::Text { clip_x, bounds, .. } = &mut cmd {
        *clip_x = Some([left, right]);
        let x0 = bounds.x.max(left);
        let x1 = (bounds.x + bounds.w).min(right);
        bounds.x = x0;
        bounds.w = (x1 - x0).max(0);
    }
    cmd
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
    list.push(DrawCmd::Fill {
        rect: crate::pane::divider_line(rect, scale),
        color: SPLITTER_COLOR,
    });
}

/// A chip's side in device pixels: the square left between the rail's edges and its
/// padding. What the chips are laid out on, and what an icon is baked at.
pub(crate) fn rail_chip_side(chips: Rect, scale: Scale) -> i32 {
    (chips.w - 2 * scale.px(RAIL_PAD)).max(0)
}

/// The vault chips that fit in the rail's `chips` region, top to bottom, each paired
/// with its index in the configured vaults. A chip is the square left between the
/// region's edges and its padding; one that would fall past its bottom is dropped, so
/// the painter and the pointer hit-testing offer exactly the same vaults. Shared by
/// both, so a click lands on the chip it looks like.
pub(crate) fn rail_buttons(
    chips: Rect,
    count: usize,
    scale: Scale,
) -> impl Iterator<Item = (usize, Rect)> {
    let pad = scale.px(RAIL_PAD);
    let side = rail_chip_side(chips, scale);
    let step = side + scale.px(RAIL_GAP);
    let bottom = chips.y + chips.h;
    (0..count)
        .map(move |i| {
            (
                i,
                Rect {
                    x: chips.x + pad,
                    y: chips.y + pad + i as i32 * step,
                    w: side,
                    h: side,
                },
            )
        })
        .take_while(move |(_, chip)| chip.y + chip.h <= bottom)
}

/// A color's relative luminance, the light it puts out once the sRGB gamma is
/// undone. The channel weights are the eye's: green carries most of the
/// brightness, blue almost none.
fn relative_luminance(color: u32) -> f32 {
    let channel = |shift: u32| {
        let c = ((color >> shift) & 0xff) as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * channel(16) + 0.7152 * channel(8) + 0.0722 * channel(0)
}

/// The contrast between two colors, from 1.0 (the same color) to 21.0 (black on
/// white).
fn contrast(a: u32, b: u32) -> f32 {
    let (a, b) = (relative_luminance(a), relative_luminance(b));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

/// The ink a chip's initials take over `fill`: whichever of the two reads better on
/// it, so a vault color of any tone keeps its letters legible. Measuring beats a
/// brightness cutoff here, which lands a mid-tone fill on the wrong side often
/// enough to matter: every color the palette deals is one.
fn chip_ink(fill: u32) -> u32 {
    if contrast(fill, RAIL_INK_DARK) >= contrast(fill, RAIL_INK_LIGHT) {
        RAIL_INK_DARK
    } else {
        RAIL_INK_LIGHT
    }
}

/// A chip lifted out of the rail by a drag, following the pointer. `top` is where
/// its top edge wants to be, before the clamp that keeps it in the rail.
#[derive(Clone, Copy)]
pub(crate) struct RailDragView {
    pub vault: usize,
    pub top: i32,
}

/// What the rail draws this frame: its vaults, which is active, which the pointer is
/// over, and any chip in flight. Bundled because the rail's painter had run out of
/// room for another positional argument, the way the tab strip's did.
pub(crate) struct RailView<'a> {
    pub vaults: &'a [Vault],
    pub active: Option<usize>,
    pub hovered: Option<usize>,
    pub drag: Option<RailDragView>,
}

/// The place a chip dropped at pointer `py` lands, given the chips' resting slots:
/// the number of other chips whose midpoint the pointer has passed. `dragged` is the
/// chip in flight, skipped so its own slot never counts against itself, which is
/// what keeps a drag from oscillating between two indices. The rail's mirror of
/// [`tab_drop_index`], down the y axis.
pub(crate) fn rail_drop_index(
    chips: Rect,
    count: usize,
    scale: Scale,
    dragged: Option<usize>,
    py: f32,
) -> usize {
    let mut target = 0;
    for (k, chip) in rail_buttons(chips, count, scale) {
        if dragged == Some(k) {
            continue;
        }
        if py > chip.y as f32 + chip.h as f32 / 2.0 {
            target += 1;
        }
    }
    target
}

/// Draw the vault rail into its full-height column: a background a shade below the
/// sidebar's, a divider down its right edge, and one chip per vault in `chips`, the
/// rail below the top bar, so the first chip starts on the line the tree and the
/// pane text do. The chip carries the vault's color and initials, solid for a
/// saved vault and an outline for a temporary one. The pointer lifts a chip's fill;
/// the active vault is marked by a bar at the rail's left edge, so a chip's color
/// stays the vault's identity rather than doubling as its state.
///
/// A chip being dragged is left out of the stack and drawn last, under the pointer:
/// the gap it leaves is its own slot, which the reorder has already moved, so the
/// chips it passes settle into place as it goes rather than waiting for the drop.
pub(crate) fn push_rail(
    list: &mut DisplayList,
    ctx: &BarCtx,
    rail: Rect,
    chips: Rect,
    view: &RailView,
) {
    let (vaults, active, hovered) = (view.vaults, view.active, view.hovered);
    let scale = ctx.scale;
    list.push(DrawCmd::Fill {
        rect: rail,
        color: RAIL_BG,
    });
    let divider = scale.px(1).max(1);
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: rail.x + rail.w - divider,
            w: divider,
            ..rail
        },
        color: SPLITTER_COLOR,
    });
    let dragged = view.drag.map(|d| d.vault);
    let mut lifted = None;
    for (i, chip) in rail_buttons(chips, vaults.len(), scale) {
        // The chip in flight leaves its slot empty and is drawn after the rest, so it
        // passes over its neighbors rather than under them.
        if dragged == Some(i) {
            lifted = Some(chip);
            continue;
        }
        push_chip(
            list,
            ctx,
            rail,
            chip,
            &vaults[i],
            active == Some(i),
            hovered == Some(i),
        );
    }
    if let (Some(drag), Some(resting)) = (view.drag, lifted) {
        // Clamped to the chips' own region: a chip dragged past the ends of the rail
        // stops there rather than following the pointer out of the column.
        let top = drag
            .top
            .clamp(chips.y, (chips.y + chips.h - resting.h).max(chips.y));
        let chip = Rect { y: top, ..resting };
        if let Some(vault) = vaults.get(drag.vault) {
            // Drawn active-styled the way a lifted tab is: the chip under the pointer
            // is the one being acted on, whichever vault the window is showing.
            push_chip(list, ctx, rail, chip, vault, false, true);
        }
    }
}

/// Draw one vault chip at `chip`: its color, its initials, and, for the active
/// vault, the marker at the rail's left edge. Shared by the chips at rest and the
/// one in flight, so a dragged chip is the same chip, moved.
fn push_chip(
    list: &mut DisplayList,
    ctx: &BarCtx,
    rail: Rect,
    chip: Rect,
    vault: &Vault,
    active: bool,
    hovered: bool,
) {
    let scale = ctx.scale;
    let m = ctx.metrics();
    let face = ctx.fonts.face(ctx.body_size, FontStyle::Bold);
    let fill = if hovered {
        pixel::tint(vault.color, RAIL_HOVER_INK, RAIL_HOVER_LIFT)
    } else {
        vault.color
    };
    let radius = scale.px(RAIL_RADIUS);
    let temporary = vault.kind == VaultKind::Temporary;
    // The ring the chip's color shows around its middle says one thing only: this
    // vault is not written down. A saved chip is filled (or pictured) edge to edge.
    let ring = if temporary {
        scale.px(RAIL_TEMP_RING).max(1)
    } else {
        0
    };
    let inner = Rect {
        x: chip.x + ring,
        y: chip.y + ring,
        w: (chip.w - 2 * ring).max(0),
        h: (chip.h - 2 * ring).max(0),
    };
    list.push(DrawCmd::RoundRect {
        rect: chip,
        color: fill,
        radius,
        corners: display::RoundedCorners::Both,
    });
    // A chip wearing a picture is the picture: it goes where the fill would be, corners
    // already rounded into it, and nothing is drawn over it. A picture chosen for a
    // chip is what the chip is for, and a color washed through it or two letters sat
    // on it would be the editor talking over it. The color is still the vault's, and
    // still says so: on the marker beside it, on a temporary chip's ring, and on the
    // chip again the moment the picture is cleared.
    //
    // Under the pointer it swaps to the same picture taken closer in, which is the
    // thumbnail's answer to a hover: the picture pushes forward while the chip stays
    // exactly where it is, so a rail of them never wobbles under the pointer.
    //
    // A chip with no picture (or one whose bake has not landed, which lasts a frame)
    // keeps the flat fill and its initials, and a temporary one has the rail's own
    // background punched back out of it, leaving its color as the outline.
    match &vault.icon_keys {
        Some(keys) => list.push(DrawCmd::Image {
            rect: inner,
            path: if hovered {
                keys.hover.clone()
            } else {
                keys.rest.clone()
            },
        }),
        None if temporary => list.push(DrawCmd::RoundRect {
            rect: inner,
            color: RAIL_BG,
            radius: (radius - ring).max(0),
            corners: display::RoundedCorners::Both,
        }),
        None => {}
    }
    if vault.icon_keys.is_none() {
        // The initials read against the vault's color behind them. A hollow temporary
        // chip is the exception, since what is behind them there is the rail.
        let ink = if temporary {
            fill
        } else {
            // The ink follows the vault's own color, not this frame's fill, so the
            // hover lift brightens the chip without flipping its letters.
            chip_ink(vault.color)
        };
        let w = shape::text_advance(face, &vault.initials) as i32;
        let x = chip.x + (chip.w - w) / 2;
        list.push(text_cmd(
            FaceKey::Prose {
                size: ctx.body_size,
                style: FontStyle::Bold,
            },
            x,
            chip.y + (chip.h - (m.ascent + m.descent)) / 2 + m.ascent,
            x + w,
            ink,
            vault.initials.clone(),
            m,
        ));
    }
    if active {
        let h = (chip.h as f32 * RAIL_ACTIVE_SPAN) as i32;
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: rail.x,
                y: chip.y + (chip.h - h) / 2,
                w: scale.px(RAIL_ACTIVE_W),
                h,
            },
            color: RAIL_ACTIVE_COLOR,
            radius: scale.px(RAIL_ACTIVE_W) / 2,
            corners: display::RoundedCorners::Both,
        });
    }
}

/// Where a `w` by `h` picture sits when shown full-window: centered in `window`, with
/// [`IMAGE_VIEW_PAD`] of clear space around it, scaled down to fit but never scaled
/// up past its own pixels.
///
/// Never up: a picture blown past its size is a picture of its own pixels, and the
/// point of showing it big is to see what is in it. One that is already small enough
/// is simply shown, in the middle, at the size it is.
pub(crate) fn image_view_rect(window: (i32, i32), size: (u32, u32), scale: Scale) -> Rect {
    let pad = scale.px(IMAGE_VIEW_PAD);
    let (avail_w, avail_h) = ((window.0 - 2 * pad).max(1), (window.1 - 2 * pad).max(1));
    let (w, h) = (size.0.max(1) as i64, size.1.max(1) as i64);
    // The picture's size in window pixels, shrunk by whichever axis runs out first.
    // Compared as fractions cross-multiplied, so the fit is exact rather than a float
    // that rounds a pixel off one edge.
    let (fit_w, fit_h) = if w <= avail_w as i64 && h <= avail_h as i64 {
        (w, h)
    } else if w * avail_h as i64 > h * avail_w as i64 {
        (avail_w as i64, (h * avail_w as i64 / w).max(1))
    } else {
        ((w * avail_h as i64 / h).max(1), avail_h as i64)
    };
    let (fit_w, fit_h) = (fit_w as i32, fit_h as i32);
    Rect {
        x: (window.0 - fit_w) / 2,
        y: (window.1 - fit_h) / 2,
        w: fit_w,
        h: fit_h,
    }
}

/// Draw a picture over the whole window: a backdrop that covers what it is over,
/// then the picture itself, centered. Pushed after the rest of the frame, so nothing
/// the editor was showing shows through.
pub(crate) fn push_image_view(
    list: &mut DisplayList,
    window: (i32, i32),
    size: (u32, u32),
    raw: &str,
    scale: Scale,
) {
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: 0,
            y: 0,
            w: window.0,
            h: window.1,
        },
        color: IMAGE_VIEW_BG,
    });
    list.push(DrawCmd::Image {
        rect: image_view_rect(window, size, scale),
        path: raw.to_string(),
    });
}

/// Draw the open palette: a bordered panel holding the query and the rows it matched,
/// with the row Enter would take filled. Pushed after the rest of the frame, so the
/// panel covers what it floats over.
///
/// The border is the panel drawn twice, the inner one inset by its thickness, the way
/// the context menu's is.
pub(crate) fn push_palette(
    list: &mut DisplayList,
    ctx: &BarCtx,
    palette: &Palette,
    layout: &PaletteLayout,
) {
    let scale = ctx.scale;
    let radius = scale.px(PALETTE_RADIUS);
    let border = scale.px(PALETTE_BORDER_W).max(1);
    list.push(DrawCmd::RoundRect {
        rect: layout.panel,
        color: PALETTE_BORDER,
        radius,
        corners: display::RoundedCorners::Both,
    });
    list.push(DrawCmd::RoundRect {
        rect: Rect {
            x: layout.panel.x + border,
            y: layout.panel.y + border,
            w: (layout.panel.w - 2 * border).max(0),
            h: (layout.panel.h - 2 * border).max(0),
        },
        color: PALETTE_BG,
        radius: (radius - border).max(0),
        corners: display::RoundedCorners::Both,
    });

    let m = ctx.metrics();
    let face = ctx.fonts.face(ctx.body_size, FontStyle::Regular);
    let pad_x = scale.px(PALETTE_ROW_PAD_X);
    let text_h = m.ascent + m.descent;

    // The query and, after it, a steady caret (no blink): it marks where typing goes
    // while the palette holds the keyboard, the way the find bar's does. An empty
    // query puts the caret at the field's start and the prompt beyond it, so the
    // prompt reads as the space the text will fill rather than as text already there.
    let caret_w = scale.px(CARET_WIDTH);
    let query_w = shape::text_advance(face, palette.query()).ceil() as i32;
    let pen = layout.field.x + pad_x;
    // The note (a search's match count) sits at the field's right, so the query and
    // its caret run up to that edge, or to the field's inner edge when there is none.
    let note_w = if palette.note().is_empty() {
        0
    } else {
        shape::text_advance(face, palette.note()).ceil() as i32 + pad_x
    };
    let right = layout.field.x + layout.field.w - pad_x - note_w;
    // The field takes text one character at a time, always at the end, so a query
    // longer than the field scrolls left to keep the caret in view rather than running
    // the text (and the caret) off past the panel's edge. The text is clipped to the
    // field so what scrolls off the left does not spill into the padding.
    let caret_x = pen + query_w + 1;
    let shift = (caret_x + caret_w - right).max(0);
    if !palette.query().is_empty() {
        list.push(clip_text_x(
            text_cmd(
                ctx.regular(),
                pen - shift,
                layout.field_baseline,
                pen - shift + query_w,
                PALETTE_TEXT,
                palette.query().to_string(),
                m,
            ),
            pen,
            right,
        ));
    }
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: caret_x - shift,
            y: layout.field_baseline - m.ascent,
            w: caret_w,
            h: text_h,
        },
        color: CARET_COLOR,
    });
    if palette.query().is_empty() {
        let x = pen + caret_w + 1 + pad_x / 2;
        let w = shape::text_advance(face, palette.prompt()).ceil() as i32;
        list.push(text_cmd(
            ctx.regular(),
            x,
            layout.field_baseline,
            x + w,
            PALETTE_HINT,
            palette.prompt().to_string(),
            m,
        ));
    }

    // What the rows cost, right-aligned in the field the way the find bar's counter
    // is: a capped search shows a fraction of what it found, and the panel is the only
    // place that can say so.
    if !palette.note().is_empty() {
        let w = shape::text_advance(face, palette.note()).ceil() as i32;
        let x = layout.field.x + layout.field.w - pad_x - w;
        list.push(text_cmd(
            ctx.regular(),
            x,
            layout.field_baseline,
            x + w,
            PALETTE_HINT,
            palette.note().to_string(),
            m,
        ));
    }

    if layout.rule.h > 0 {
        list.push(DrawCmd::Fill {
            rect: layout.rule,
            color: PALETTE_BORDER,
        });
    }

    // One row is lit, the selected one, whether the keyboard or the pointer put the
    // selection there: what Enter takes and what a click takes are the same row, so
    // they are the same fill.
    for (vis, (pos, row)) in palette.visible().enumerate() {
        let rect = layout.row_rect(vis);
        if palette.selected() == pos {
            list.push(DrawCmd::RoundRect {
                rect,
                color: PALETTE_SELECTED_BG,
                radius: scale.px(PALETTE_ROW_RADIUS),
                corners: display::RoundedCorners::Both,
            });
        }
        // A row's label is whatever it names: a search hit carries a path and the line
        // it found, which readily outruns the panel. Clipped to the row rather than
        // shortened, so what is there reads as running on rather than as all there is.
        let x = rect.x + pad_x;
        let right = rect.x + rect.w - pad_x;
        // Behind the label, a gold fill under each place the query occurs, so a row
        // shows what it matched. Same mark the find bar puts on a match; the label
        // draws once on top, so clipping and measurement stay simple. An empty query
        // (a vault palette before anything is typed) yields no spans and no fills.
        let top = layout.baseline(vis) - m.ascent;
        let text_h = m.ascent + m.descent;
        for (start, end) in palette::match_spans(&row.label, palette.query()) {
            let mx = x + shape::text_advance(face, &row.label[..start]).round() as i32;
            let mw = shape::text_advance(face, &row.label[start..end]).round() as i32;
            let x0 = mx.max(x);
            let x1 = (mx + mw).min(right);
            if x1 > x0 {
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: x0,
                        y: top,
                        w: x1 - x0,
                        h: text_h,
                    },
                    color: PALETTE_MATCH,
                });
            }
        }
        let w = shape::text_advance(face, &row.label) as i32;
        list.push(clip_text_x(
            text_cmd(
                ctx.regular(),
                x,
                layout.baseline(vis),
                x + w,
                PALETTE_TEXT,
                row.label.clone(),
                m,
            ),
            x,
            right,
        ));
    }
}

/// Draw an open context menu: a bordered panel with one row per item, the row under
/// the pointer filled. Pushed after the rest of the frame, so the panel covers the
/// chrome it floats over.
///
/// The border is the panel drawn twice, the inner one inset by its thickness, which
/// is what gives a rounded panel a rounded outline (there is no stroked-rect
/// command, and a hairline needs none).
pub(crate) fn push_menu(list: &mut DisplayList, ctx: &BarCtx, menu: &Menu, layout: &MenuLayout) {
    let scale = ctx.scale;
    let radius = scale.px(MENU_RADIUS);
    let border = scale.px(MENU_BORDER_W).max(1);
    list.push(DrawCmd::RoundRect {
        rect: layout.panel,
        color: MENU_BORDER,
        radius,
        corners: display::RoundedCorners::Both,
    });
    list.push(DrawCmd::RoundRect {
        rect: Rect {
            x: layout.panel.x + border,
            y: layout.panel.y + border,
            w: (layout.panel.w - 2 * border).max(0),
            h: (layout.panel.h - 2 * border).max(0),
        },
        color: MENU_BG,
        radius: (radius - border).max(0),
        corners: display::RoundedCorners::Both,
    });

    let m = ctx.metrics();
    let face = ctx.fonts.face(ctx.body_size, FontStyle::Regular);
    for (i, item) in menu.items().iter().enumerate() {
        let row = layout.item_rect(i);
        if menu.hovered() == Some(i) {
            list.push(DrawCmd::RoundRect {
                rect: row,
                color: MENU_HOVER_BG,
                radius: scale.px(MENU_ITEM_RADIUS),
                corners: display::RoundedCorners::Both,
            });
        }
        let x = row.x + scale.px(MENU_ITEM_PAD_X);
        let w = shape::text_advance(face, &item.label) as i32;
        list.push(text_cmd(
            FaceKey::Prose {
                size: ctx.body_size,
                style: FontStyle::Regular,
            },
            x,
            layout.baseline(i),
            x + w,
            MENU_TEXT,
            item.label.clone(),
            m,
        ));
    }
}

/// One word placed on a wrapped diff line: its x within the panel's inner box, its
/// advance, the text, and how it is drawn.
struct DiffWord {
    x: i32,
    w: i32,
    text: String,
    color: u32,
    bg: Option<u32>,
}

/// Greedily wrap `words` (each a word or a whitespace run, tagged with its color)
/// into lines no wider than `max_w`, breaking at the whitespace and forcing a break
/// on a newline. A word wider than the box is left to overflow; the caller clips it
/// to the panel.
fn wrap_diff(
    words: &[(String, bool, u32, Option<u32>)],
    face: &Face,
    max_w: i32,
) -> Vec<Vec<DiffWord>> {
    let mut lines: Vec<Vec<DiffWord>> = Vec::new();
    let mut cur: Vec<DiffWord> = Vec::new();
    let mut pen = 0i32;
    let mut pending_space = 0i32;
    for (text, space, color, bg) in words {
        if *space {
            if text.contains('\n') {
                lines.push(std::mem::take(&mut cur));
                pen = 0;
                pending_space = 0;
            } else {
                pending_space += shape::text_advance(face, text).ceil() as i32;
            }
            continue;
        }
        let w = shape::text_advance(face, text).ceil() as i32;
        let lead = if cur.is_empty() { 0 } else { pending_space };
        if !cur.is_empty() && pen + lead + w > max_w {
            lines.push(std::mem::take(&mut cur));
            pen = 0;
            pending_space = 0;
        }
        let lead = if cur.is_empty() { 0 } else { pending_space };
        pen += lead;
        cur.push(DiffWord {
            x: pen,
            w,
            text: text.clone(),
            color: *color,
            bg: *bg,
        });
        pen += w;
        pending_space = 0;
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Draw the AI text-action modal: a floating panel, like the palette, holding the
/// action's name, its body (a running note or a failure message), and the dismiss
/// hint. Pushed after the rest of the frame, so it covers what it floats over. A
/// successful answer is not shown here; it opens a diff tab. The body wraps to the
/// panel and, past [`AI_MAX_LINES`], notes how many lines are hidden rather than
/// growing without bound.
pub(crate) fn push_ai_preview(
    list: &mut DisplayList,
    ctx: &BarCtx,
    preview: &Preview,
    window: (i32, i32),
) {
    let scale = ctx.scale;
    let m = ctx.metrics();
    let face = ctx.fonts.face(ctx.body_size, FontStyle::Regular);
    let pad = scale.px(AI_PANEL_PAD);
    let vpad = scale.px(AI_LINE_VPAD);
    let text_h = m.ascent + m.descent;
    let line_h = text_h + 2 * vpad;
    let rule_h = scale.px(AI_PANEL_BORDER_W).max(1);

    let w = scale.px(AI_PANEL_WIDTH).min(window.0 - 2 * pad).max(0);
    let inner_w = (w - 2 * pad).max(0);

    let (title, base_hint) = match preview {
        Preview::Thinking { label, .. } => (*label, "Esc to cancel"),
        Preview::Failed { label, .. } => (*label, "Esc to dismiss"),
    };

    // The body's words: a running note, or the failure in red. Owned, so nothing
    // borrows the preview across the wrap.
    let mut words: Vec<(String, bool, u32, Option<u32>)> = Vec::new();
    match preview {
        Preview::Thinking { .. } => {
            words.push(("Working…".to_string(), false, AI_HINT, None));
        }
        Preview::Failed { message, .. } => {
            let span = [ai::DiffSpan {
                kind: DiffKind::Equal,
                text: message.clone(),
            }];
            for p in ai::pieces(&span) {
                words.push((p.text.to_string(), p.space, AI_ERROR_TEXT, None));
            }
        }
    }

    let mut lines = wrap_diff(&words, face, inner_w);
    let hidden = lines.len().saturating_sub(AI_MAX_LINES);
    lines.truncate(AI_MAX_LINES);
    let hint = if hidden > 0 {
        format!("{base_hint}  ·  +{hidden} more changed lines")
    } else {
        base_hint.to_string()
    };

    // Title row, a rule, the body lines, then the hint row, inside the padding.
    let body_h = lines.len() as i32 * line_h;
    let h = 2 * pad + line_h + rule_h + body_h + line_h;
    let x = ((window.0 - w) / 2).max(0);
    let y = scale.px(AI_PANEL_TOP).min(window.1 - h).max(0);

    let radius = scale.px(AI_PANEL_RADIUS);
    let border = scale.px(AI_PANEL_BORDER_W).max(1);
    list.push(DrawCmd::RoundRect {
        rect: Rect { x, y, w, h },
        color: AI_PANEL_BORDER,
        radius,
        corners: display::RoundedCorners::Both,
    });
    list.push(DrawCmd::RoundRect {
        rect: Rect {
            x: x + border,
            y: y + border,
            w: (w - 2 * border).max(0),
            h: (h - 2 * border).max(0),
        },
        color: AI_PANEL_BG,
        radius: (radius - border).max(0),
        corners: display::RoundedCorners::Both,
    });

    let inner_x = x + pad;
    let inner_right = inner_x + inner_w;

    // Title.
    let title_base = y + pad + vpad + m.ascent;
    let title_w = shape::text_advance(face, title).ceil() as i32;
    list.push(clip_text_x(
        text_cmd(
            ctx.regular(),
            inner_x,
            title_base,
            inner_x + title_w,
            AI_TEXT,
            title.to_string(),
            m,
        ),
        inner_x,
        inner_right,
    ));

    // The rule under the title.
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: inner_x,
            y: y + pad + line_h,
            w: inner_w,
            h: rule_h,
        },
        color: AI_PANEL_BORDER,
    });

    // Body lines.
    let body_top = y + pad + line_h + rule_h;
    for (li, line) in lines.iter().enumerate() {
        let top = body_top + li as i32 * line_h;
        let baseline = top + vpad + m.ascent;
        for word in line {
            let wx = inner_x + word.x;
            if let Some(bg) = word.bg {
                let x0 = wx.max(inner_x);
                let x1 = (wx + word.w).min(inner_right);
                if x1 > x0 {
                    list.push(DrawCmd::Fill {
                        rect: Rect {
                            x: x0,
                            y: baseline - m.ascent,
                            w: x1 - x0,
                            h: text_h,
                        },
                        color: bg,
                    });
                }
            }
            list.push(clip_text_x(
                text_cmd(
                    ctx.regular(),
                    wx,
                    baseline,
                    wx + word.w,
                    word.color,
                    word.text.clone(),
                    m,
                ),
                inner_x,
                inner_right,
            ));
        }
    }

    // Hint row, along the bottom padding.
    let hint_top = y + h - pad - line_h;
    let hint_base = hint_top + vpad + m.ascent;
    let hint_w = shape::text_advance(face, &hint).ceil() as i32;
    list.push(clip_text_x(
        text_cmd(
            ctx.regular(),
            inner_x,
            hint_base,
            inner_x + hint_w,
            AI_HINT,
            hint,
            m,
        ),
        inner_x,
        inner_right,
    ));
}

/// Draw the diff view's accept/reject hint: a small pill along the bottom of pane
/// `rect`, reminding the user that Enter keeps the change and Escape discards it,
/// kept out of the way at the pane's foot.
fn push_diff_hint(list: &mut DisplayList, ctx: &BarCtx, rect: Rect) {
    let scale = ctx.scale;
    let m = ctx.metrics();
    let face = ctx.fonts.face(ctx.body_size, FontStyle::Regular);
    let text = "Enter to accept   ·   Esc to discard";
    let text_w = shape::text_advance(face, text).ceil() as i32;
    let text_h = m.ascent + m.descent;
    let pad_x = scale.px(AI_PANEL_PAD);
    let pad_y = scale.px(AI_LINE_VPAD) * 2;
    let w = text_w + 2 * pad_x;
    let h = text_h + 2 * pad_y;
    let gap = scale.px(AI_PANEL_PAD);
    let x = rect.x + (rect.w - w) / 2;
    let y = rect.y + rect.h - h - gap;

    let radius = scale.px(AI_PANEL_RADIUS);
    let border = scale.px(AI_PANEL_BORDER_W).max(1);
    list.push(DrawCmd::RoundRect {
        rect: Rect { x, y, w, h },
        color: AI_PANEL_BORDER,
        radius,
        corners: display::RoundedCorners::Both,
    });
    list.push(DrawCmd::RoundRect {
        rect: Rect {
            x: x + border,
            y: y + border,
            w: (w - 2 * border).max(0),
            h: (h - 2 * border).max(0),
        },
        color: AI_PANEL_BG,
        radius: (radius - border).max(0),
        corners: display::RoundedCorners::Both,
    });
    let tx = x + pad_x;
    let base = y + pad_y + m.ascent;
    list.push(clip_text_x(
        text_cmd(
            ctx.regular(),
            tx,
            base,
            tx + text_w,
            AI_HINT,
            text.to_string(),
            m,
        ),
        tx,
        tx + text_w,
    ));
}

/// The full geometry of a diff view laid into pane `view`: the two full side columns
/// (gutter plus text), the two text areas the layout builds into (what [`DiffColumns`]
/// carries, so the layout lands where the render draws it), the gutter width, and the
/// divider between the columns.
struct DiffGeom {
    cols: DiffColumns,
    old_side: Rect,
    new_side: Rect,
    divider: Rect,
}

/// The width of a diff view's line-number gutter: room for three digits (a diff of an AI
/// edit numbers its own fragment's lines, rarely into the hundreds; a longer one clips
/// the number, not the diff) plus its padding each side.
fn diff_gutter_w(scale: Scale, body_size: u32, fonts: &Fonts) -> i32 {
    let digit = shape::text_advance(fonts.face(body_size, FontStyle::Regular), "0").ceil() as i32;
    digit * 3 + 2 * scale.px(DIFF_GUTTER_PAD)
}

/// Split pane `view` into the diff's two side-by-side columns: the old on the left, the
/// new on the right, with a divider in the seam. Each column reserves a fixed
/// line-number gutter on its left; the text area is what is left.
fn diff_geom(view: Rect, scale: Scale, body_size: u32, fonts: &Fonts) -> DiffGeom {
    let divider_w = scale.px(DIFF_DIVIDER_W).max(1);
    let gutter_w = diff_gutter_w(scale, body_size, fonts);
    let half_w = (view.w - divider_w) / 2;
    let old_side = Rect { w: half_w, ..view };
    let new_side = Rect {
        x: view.x + half_w + divider_w,
        w: (view.w - half_w - divider_w).max(0),
        ..view
    };
    let divider = Rect {
        x: view.x + half_w,
        w: divider_w,
        ..view
    };
    // The text area within a side, past its gutter.
    let text_of = |side: Rect| Rect {
        x: side.x + gutter_w,
        w: (side.w - gutter_w).max(0),
        ..side
    };
    DiffGeom {
        cols: DiffColumns {
            old_text: text_of(old_side),
            new_text: text_of(new_side),
        },
        old_side,
        new_side,
        divider,
    }
}

/// The two text columns a diff view lays out into, for [`DiffView::ensure_layouts`].
/// The render derives the same rectangles from the same inputs, so the tinted words land
/// exactly where the text is drawn.
pub(crate) fn diff_columns(view: Rect, scale: Scale, body_size: u32, fonts: &Fonts) -> DiffColumns {
    diff_geom(view, scale, body_size, fonts).cols
}

/// Draw a side-by-side diff view into pane `view`: the old text on the left with its
/// removed lines washed red, the new text on the right with its added lines washed
/// green, aligned line for line so a change reads across the divider. A changed line
/// carries a faint full-line wash and, over it, the stronger wash on just its edited
/// words; the empty half of a one-sided change is hatched; a line-number gutter runs
/// down each column, and one scrollbar serves the pane. The two sides share
/// `diff.scroll_y`, so they move together.
pub(crate) fn push_diff_view(
    list: &mut DisplayList,
    fonts: &Fonts,
    view: Rect,
    diff: &DiffView,
    scale: Scale,
    body_size: u32,
    scrollbar: &scroll::Scrollbar,
) {
    let g = diff_geom(view, scale, body_size, fonts);
    // The pane background covers both columns and the divider seam; the row bands, the
    // text, and the chrome all draw over it.
    list.push(DrawCmd::Fill {
        rect: view,
        color: BACKGROUND,
    });
    // The aligned row bands, beneath the text: a faint full-line wash and accent bar on
    // a changed side, and diagonal hatching on the empty half of a one-sided change.
    push_diff_bands(list, diff, &g, view, scale);
    // Each side's text draws with the stronger word washes over it (the DiffTint pass),
    // but without its own background fill: the pane background and the row bands beneath
    // it must show through, so the opening background command is dropped.
    let mut draw_side = |layout: &Layout,
                         rope: &crate::rope::Rope,
                         ranges: &[(usize, usize)],
                         text_rect: Rect,
                         deletes: bool| {
        let tint = if deletes {
            DiffTint {
                inserts: &[],
                deletes: ranges,
            }
        } else {
            DiffTint {
                inserts: ranges,
                deletes: &[],
            }
        };
        let params = PaintParams {
            view: text_rect,
            scale,
            body_size,
            scroll_y: diff.scroll_y,
            content_height: 0,
            scrollbar: &scroll::Scrollbar::default(),
            selection: None,
            caret: None,
            find: None,
            table_hover: None,
            fold_hover: None,
            diff: Some(tint),
        };
        let mut side_list = build_display_list(fonts, layout, rope, &params);
        if matches!(side_list.first(), Some(DrawCmd::Fill { rect, color }) if *rect == text_rect && *color == BACKGROUND)
        {
            side_list.remove(0);
        }
        list.append(&mut side_list);
    };
    if let Some(old) = diff.old_side() {
        draw_side(old.layout, old.rope, old.ranges, g.cols.old_text, true);
    }
    if let Some(new) = diff.new_side() {
        draw_side(new.layout, new.rope, new.ranges, g.cols.new_text, false);
    }
    list.push(DrawCmd::Fill {
        rect: g.divider,
        color: DIFF_DIVIDER,
    });
    push_diff_gutters(list, fonts, diff, &g, scale, body_size);
    push_scrollbar(
        list,
        view,
        scrollbar,
        diff.content_height(),
        diff.scroll_y,
        BACKGROUND,
        scale,
    );
    // The keys are not obvious for a tab that is not a document, so a small pill at the
    // foot names them.
    push_diff_hint(list, &BarCtx::new(fonts, scale, body_size), view);
}

/// Draw the aligned row bands beneath the diff's text. A changed line takes a faint
/// full-row wash (its whole column, gutter and text, so the tint reads at the number
/// too), inset only on the right so it clears the divider. The empty half of a one-sided
/// change is split: its gutter is a flat solid and its text area is diagonal hatching, so
/// the number column reads as a clean edge and only the text carries the "nothing here"
/// texture. Scroll-shifted and culled to the pane.
fn push_diff_bands(
    list: &mut DisplayList,
    diff: &DiffView,
    g: &DiffGeom,
    view: Rect,
    scale: Scale,
) {
    let line_pad = scale.px(DIFF_LINE_PAD);
    let period = scale.px(DIFF_HATCH_PERIOD).max(2);
    let stroke = scale.px(DIFF_HATCH_STROKE).max(1);
    let (top, bottom) = (view.y, view.y + view.h);
    // `full` is the whole side column; `text` is its text area past the gutter.
    let side_band = |list: &mut DisplayList,
                     full: Rect,
                     text: Rect,
                     cell: &crate::pane::DiffCell,
                     band_y: i32,
                     band_h: i32,
                     line_bg: u32| {
        let y = band_y - diff.scroll_y;
        if !cell.present {
            // The empty half of a one-sided change: a solid gutter and a hatched text
            // area, so the number column stays clean and the text reads as absent.
            if y + band_h > top && y < bottom {
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        y,
                        w: (text.x - full.x).max(0),
                        h: band_h,
                        ..full
                    },
                    color: DIFF_HATCH,
                });
                list.push(DrawCmd::Hatch {
                    rect: Rect {
                        y,
                        h: band_h,
                        ..text
                    },
                    color: DIFF_HATCH,
                    period,
                    stroke,
                });
            }
            return;
        }
        if !cell.changed {
            return;
        }
        // A changed line: the faint wash over the whole column (gutter included), inset on
        // the right so it clears the divider. The text keeps its own margin inside, so the
        // words and their washes stay well within the band.
        if y + cell.height > top && y < bottom {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    y,
                    w: (full.w - line_pad).max(0),
                    h: cell.height,
                    ..full
                },
                color: line_bg,
            });
        }
    };
    for band in diff.bands() {
        side_band(
            list,
            g.old_side,
            g.cols.old_text,
            &band.old,
            band.y,
            band.height,
            DIFF_LINE_DELETE_BG,
        );
        side_band(
            list,
            g.new_side,
            g.cols.new_text,
            &band.new,
            band.y,
            band.height,
            DIFF_LINE_INSERT_BG,
        );
    }
}

/// Draw both columns' line-number gutters: a subtle hairline parting each gutter from its
/// text, then each present line's real (1-based) number, right-aligned in its gutter on
/// the first row of the line, scroll-shifted and culled to the pane. A changed line's
/// number takes the accent color (green added, red removed); an unchanged line's is dim.
/// An absent side (the empty half of a one-sided change) has no number there.
fn push_diff_gutters(
    list: &mut DisplayList,
    fonts: &Fonts,
    diff: &DiffView,
    g: &DiffGeom,
    scale: Scale,
    body_size: u32,
) {
    // The numbers read a step below the body, the way the sidebar's chrome text does
    // (and that size is already opened, so it renders crisp rather than falling back).
    let num_size = tree_font_size(body_size);
    let face_key = FaceKey::Prose {
        size: num_size,
        style: FontStyle::Regular,
    };
    let face = fonts.face(num_size, FontStyle::Regular);
    let num_m = fonts.metrics(num_size);
    let pad = scale.px(DIFF_GUTTER_PAD);
    let (top, bottom) = (g.old_side.y, g.old_side.y + g.old_side.h);
    // A hairline down each gutter's inner edge (just left of its text), full column
    // height, so the numbers read as their own column rather than floating loose.
    let border_w = scale.px(1).max(1);
    for text in [g.cols.old_text, g.cols.new_text] {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: text.x - border_w,
                y: top,
                w: border_w,
                h: (bottom - top).max(0),
            },
            color: DIFF_GUTTER_BORDER,
        });
    }
    let number = |list: &mut DisplayList,
                  cell: &crate::pane::DiffCell,
                  gutter_right: i32,
                  left: i32,
                  accent: u32| {
        if !cell.present {
            return;
        }
        let baseline = cell.baseline - diff.scroll_y;
        if baseline + num_m.descent <= top || baseline - num_m.ascent >= bottom {
            return;
        }
        let num = cell.number.to_string();
        let w = shape::text_advance(face, &num).ceil() as i32;
        // Right-aligned, but never spilling past the column's left edge: a rare four-digit
        // number (summarizing a very long note) crowds the border rather than the rail.
        let x = (gutter_right - w).max(left);
        let color = if cell.changed {
            accent
        } else {
            DIFF_GUTTER_TEXT
        };
        list.push(text_cmd(face_key, x, baseline, x + w, color, num, num_m));
    };
    // The numbers sit a little left of the gutter's inner border, with more breathing
    // room on their right than the tight pad, so they do not crowd the text.
    let right_gap = border_w + 2 * pad;
    let old_right = g.cols.old_text.x - right_gap;
    let new_right = g.cols.new_text.x - right_gap;
    for band in diff.bands() {
        number(list, &band.old, old_right, g.old_side.x, DIFF_DELETE_ACCENT);
        number(list, &band.new, new_right, g.new_side.x, DIFF_INSERT_ACCENT);
    }
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
    crate::theme::row_height(m.line_height, scale)
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
/// the column's bottom. Rows do not wrap; a long name is scissored at the column's
/// edge, so nothing the sidebar draws lands outside it.
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
            let chev = crate::theme::chevron_side(row_h, scale);
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
        // A name longer than the column is cut at its edge rather than run on into
        // whatever is beside it. The pane to the right does paint over an overhang,
        // but only when it has something to paint: an empty one has no background of
        // its own, and a name is not the sidebar's to put there either way.
        list.push(clip_text_x(
            text_cmd(
                face,
                content_x + marker_w,
                top + text_offset,
                text_right,
                TREE_TEXT_COLOR,
                row.name.clone(),
                m,
            ),
            rect.x,
            text_right,
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
        fit_label(face, TREE_FILTER_LABEL, (right - x).max(0)).into_owned(),
        m,
    ));
}

/// The pixel height of the top bar: the body line height plus vertical padding
/// above and below, matching how a tree row is sized. Shared by the layout (which
/// reserves it) and the painter so they agree.
pub(crate) fn top_bar_height(m: Metrics, scale: Scale) -> i32 {
    (m.line_height + 2 * scale.px(TAB_VPAD)).max(1)
}

/// One tab's geometry within a strip: its full pill and the close-glyph box inside
/// it. Index-aligned with the strip's tabs, so the label is read from those; the
/// painter scissors it to the space before the close box (a long name hard-cuts, no
/// ellipsis). Shared by the painter and the pointer hit-testing so a click lands
/// where the eye expects.
#[derive(Clone, Copy)]
pub(crate) struct TabHit {
    pub rect: Rect,
    pub close: Rect,
}

/// The paint inputs for one pane's tab strip: where it sits, its tabs (label,
/// unsaved flag, pinned flag), the active index, and what the pointer is over: a tab
/// body (`hovered`), a tab's close glyph (`close_hover`), or the new-tab button
/// (`new_hover`).
pub(crate) struct TabStrip<'a> {
    pub rect: Rect,
    pub tabs: &'a [TabView<'a>],
    pub active: usize,
    pub hovered: Option<usize>,
    pub close_hover: Option<usize>,
    pub new_hover: bool,
    /// How far the tab run is scrolled left, in device pixels, when the tabs
    /// overflow the strip. Zero while they all fit.
    pub scroll_x: i32,
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

/// One strip's tab geometry, derived from the strip width and which tabs are
/// pinned. A pinned tab takes the fixed compact `pinned_w`; the unpinned tabs share
/// `unpinned_w`, which shrinks from `TAB_MAX_W` toward `TAB_MIN_W` over whatever
/// space the pins leave, so freeing a tab to a pin widens the rest. Below the floor
/// the run overflows and the strip scrolls, so `content` (the tabs' total width)
/// exceeds `avail` (the strip minus the reserved new-tab lane). The scroll clamp
/// turns that difference into the overflow (see [`crate::platform::scroll::clamp`]).
#[derive(Clone, Copy)]
pub(crate) struct TabLayout {
    /// The fixed compact width of a pinned tab.
    pub pinned_w: i32,
    /// The shared width of an unpinned tab.
    pub unpinned_w: i32,
    /// The tabs' total width, pinned and unpinned summed.
    pub content: i32,
    /// The visible tab lane: the strip minus the reserved new-tab (+) button.
    pub avail: i32,
}

impl TabLayout {
    /// The right edge of the visible tab lane, where the pinned new-tab button
    /// begins; tabs are culled and their labels scissored to `[strip.x, this)`.
    pub(crate) fn viewport_right(&self, strip: Rect) -> i32 {
        strip.x + self.avail
    }

    /// The width tab `i` takes: the compact pinned width for a pinned tab, else the
    /// shared unpinned width.
    fn width_of(&self, tabs: &[TabView], i: usize) -> i32 {
        if tabs.get(i).is_some_and(|t| t.pinned) {
            self.pinned_w
        } else {
            self.unpinned_w
        }
    }

    /// Tab `i`'s left offset from the strip's left (before scroll) and its width,
    /// summing the widths of the tabs before it. For the reveal that scrolls a tab
    /// into view, where the run is variable-width so a tab's slot is not `i * width`.
    pub(crate) fn tab_span(&self, tabs: &[TabView], i: usize) -> (i32, i32) {
        let start = (0..i).map(|k| self.width_of(tabs, k)).sum();
        (start, self.width_of(tabs, i))
    }
}

/// Lay the tabs of `strip` out, reserving a lane at the right for the
/// always-reachable new-tab button. Each pinned tab takes the fixed compact width;
/// the unpinned tabs share what the pins leave, shrinking between the ceiling and
/// the floor. Shared by the painter, the hit-testing, and the scroll clamp so all
/// three agree on the widths and the scroll range.
pub(crate) fn tab_layout(strip: Rect, tabs: &[TabView], scale: Scale) -> TabLayout {
    let new_lane = scale.px(TAB_CLOSE_GAP) + scale.px(TAB_NEW_W);
    let avail = (strip.w - new_lane).max(0);
    let pinned_w = scale.px(TAB_PINNED_W);
    let pinned_count = tabs.iter().filter(|t| t.pinned).count() as i32;
    let unpinned_count = tabs.len() as i32 - pinned_count;
    let pinned_total = pinned_w * pinned_count;
    // The unpinned tabs share the space the pins leave; a strip that is all pins
    // has none to share, and its width goes unused.
    let unpinned_w = if unpinned_count == 0 {
        0
    } else {
        ((avail - pinned_total).max(0) / unpinned_count)
            .clamp(scale.px(TAB_MIN_W), scale.px(TAB_MAX_W))
    };
    TabLayout {
        pinned_w,
        unpinned_w,
        content: pinned_total + unpinned_w * unpinned_count,
        avail,
    }
}

/// The new-tab (+) button's box: just past the last tab while the tabs fit, or
/// pinned at the right of the strip once they overflow and scroll under it, so it
/// stays reachable. Shared by the painter and the hit-testing.
pub(crate) fn new_tab_button_rect(strip: Rect, layout: TabLayout, scale: Scale) -> Rect {
    let gap = scale.px(TAB_CLOSE_GAP);
    let side = scale.px(TAB_NEW_W);
    let x = strip.x + layout.content.min(layout.avail) + gap;
    Rect {
        x,
        y: strip.y + (strip.h - side) / 2,
        w: side,
        h: side,
    }
}

/// Truncate `label` with a trailing ellipsis until it fits `max_w` pixels in
/// `face`; a label that already fits is borrowed through unchanged.
fn fit_label<'a>(face: &Face, label: &'a str, max_w: i32) -> Cow<'a, str> {
    if shape::text_advance(face, label) as i32 <= max_w {
        return Cow::Borrowed(label);
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
    Cow::Owned(out)
}

/// The tabs of one strip (see [`tab_layout`]), laid left to right from `strip.x`
/// and shifted left by `scroll_x`. A pinned tab takes the compact width and the
/// rest share theirs, so the run is variable-width and each tab starts where the
/// one before it ends. The result stays index-aligned with `tabs`: a tab scrolled
/// out of view keeps its slot with an off-strip rect, so the painter and the
/// hit-testing cull it by position against the viewport. Shared by both so a click
/// lands where the eye expects.
pub(crate) fn tab_rects(strip: Rect, tabs: &[TabView], scale: Scale, scroll_x: i32) -> Vec<TabHit> {
    let pad = scale.px(TAB_PAD_X);
    let close = tab_close_side(scale);
    let layout = tab_layout(strip, tabs, scale);
    let mut out = Vec::with_capacity(tabs.len());
    let mut x = strip.x - scroll_x;
    for i in 0..tabs.len() {
        let w = layout.width_of(tabs, i);
        let rect = Rect {
            x,
            y: strip.y,
            w,
            h: strip.h,
        };
        let close_rect = Rect {
            x: rect.x + w - pad - close,
            y: rect.y + (rect.h - close) / 2,
            w: close,
            h: close,
        };
        out.push(TabHit {
            rect,
            close: close_rect,
        });
        x += w;
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

/// The shared drawing context for the chrome's helpers, the top bar and the vault
/// rail: the fonts, display scale, and body size everything in them is measured and
/// drawn at. Bundled so the helpers thread one context instead of three parallel
/// arguments.
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
        let line = crate::pane::divider_line(*d, ctx.scale);
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: line.x,
                y: bar.y,
                w: line.w,
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
            fitted.into_owned(),
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

/// The pin badge a pinned tab wears in its close slot: the pin marker inked from
/// its distance field, over the same rounded highlight the close and new-tab
/// buttons light up with under the pointer (`hovered`). It stands where the close
/// cross would, so the click that lands there unpins the tab. `bright` inks it in
/// the active-tab text color (on the active or hovered tab) and the subdued
/// inactive color otherwise, matching how the close cross reads.
fn push_pin_badge(list: &mut DisplayList, ctx: &BarCtx, box_: Rect, bright: bool, hovered: bool) {
    if hovered {
        list.push(DrawCmd::RoundRect {
            rect: box_,
            color: TAB_BUTTON_HOVER_BG,
            radius: ctx.scale.px(TAB_BUTTON_RADIUS),
            corners: display::RoundedCorners::Both,
        });
    }
    let inset = ctx.scale.px(3);
    let side = (box_.w.min(box_.h) - 2 * inset).max(1);
    list.push(DrawCmd::Pin {
        rect: Rect {
            x: box_.x + (box_.w - side) / 2,
            y: box_.y + (box_.h - side) / 2,
            w: side,
            h: side,
        },
        color: if bright {
            TAB_ACTIVE_TEXT
        } else {
            TAB_INACTIVE_TEXT
        },
    });
}

/// Draw one pane's tabs and its new-tab button (see [`push_top_bar`]). The tabs
/// are scissored to the viewport `[strip.x, viewport_right)`, so a scrolled run
/// slides under the strip's edges and the pinned + button; a tab fully outside it
/// is skipped.
fn push_tab_strip(list: &mut DisplayList, ctx: &BarCtx, strip: &TabStrip) {
    let scale = ctx.scale;
    let layout = tab_layout(strip.rect, strip.tabs, scale);
    let tabs = tab_rects(strip.rect, strip.tabs, scale, strip.scroll_x);
    let vpad = scale.px(TAB_VPAD);
    let divider = scale.px(1).max(1);
    let clip = [strip.rect.x, layout.viewport_right(strip.rect)];
    let visible = |t: &TabHit| t.rect.x < clip[1] && t.rect.x + t.rect.w > clip[0];
    let dragged = strip.drag.as_ref().map(|d| d.tab);
    let is_active = |i: usize| i == strip.active;
    for (i, tab) in tabs.iter().enumerate() {
        // The dragged tab lifts out of the run and is drawn floating below, leaving
        // a gap here; a tab scrolled out of the viewport draws nothing.
        if dragged == Some(i) || !visible(tab) {
            continue;
        }
        let active = is_active(i);
        let hovered = strip.hovered == Some(i);
        // A hairline between two resting tabs, but never touching the active or
        // hovered tab, nor the gap a lifted tab left behind, and clipped to the
        // viewport so it does not draw under the + button.
        if !active
            && !hovered
            && i + 1 < tabs.len()
            && dragged != Some(i + 1)
            && !is_active(i + 1)
            && strip.hovered != Some(i + 1)
        {
            let x = tab.rect.x + tab.rect.w - divider;
            if x >= clip[0] && x + divider <= clip[1] {
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x,
                        y: tab.rect.y + vpad,
                        w: divider,
                        h: (tab.rect.h - 2 * vpad).max(1),
                    },
                    color: SPLITTER_COLOR,
                });
            }
        }
        let Some(view) = strip.tabs.get(i) else {
            continue;
        };
        push_tab_body(
            list,
            ctx,
            tab,
            view,
            active,
            hovered,
            strip.close_hover == Some(i),
            clip,
        );
    }
    // The new-tab (+) button: after the last tab, or pinned at the strip's right
    // once the tabs overflow.
    let plus = new_tab_button_rect(strip.rect, layout, scale);
    if plus.x + plus.w <= strip.rect.x + strip.rect.w {
        push_glyph_button(list, ctx, plus, "+", strip.new_hover);
    }
    // The floating dragged tab, drawn last so it rides over its neighbors and the
    // + button. It follows the pointer, clamped to stay within the visible lane.
    if let Some(d) = &strip.drag {
        if let (Some(tab), Some(view)) = (tabs.get(d.tab), strip.tabs.get(d.tab)) {
            let max_left = (clip[1] - tab.rect.w).max(strip.rect.x);
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
            };
            // The float rides above the run, clipped to the whole strip so it is not
            // cut at the + lane it may overlap.
            push_tab_body(
                list,
                ctx,
                &floated,
                view,
                true,
                false,
                false,
                [strip.rect.x, strip.rect.x + strip.rect.w],
            );
        }
    }
}

/// Draw one tab's contents: its active or hover background, the label, and the one
/// glyph in its close slot: a pin badge when the tab is pinned, else the close
/// cross (active or hovered tab) or the unsaved dot (a resting modified tab). A pin
/// owns the slot outright, since a pinned tab shows a pin badge in place of its
/// close cross and cannot be closed by the click there. The inter-tab hairline is
/// the strip's business, not a tab's, so it stays out of here. Everything is
/// scissored to `clip` (a window x-range): the fill clamps to it, the label glyphs
/// slide under its edges, and the close box, pin, or dot draws only while it sits
/// wholly inside, so a scrolled tab never spills past the viewport.
#[allow(clippy::too_many_arguments)]
fn push_tab_body(
    list: &mut DisplayList,
    ctx: &BarCtx,
    tab: &TabHit,
    view: &TabView,
    active: bool,
    hovered: bool,
    close_hover: bool,
    clip: [i32; 2],
) {
    let scale = ctx.scale;
    let m = ctx.metrics();
    let pad = scale.px(TAB_PAD_X);
    if active || hovered {
        let x0 = tab.rect.x.max(clip[0]);
        let x1 = (tab.rect.x + tab.rect.w).min(clip[1]);
        if x1 > x0 {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: x0,
                    w: x1 - x0,
                    ..tab.rect
                },
                color: if active { TAB_ACTIVE_BG } else { TAB_HOVER_BG },
            });
        }
    }
    let color = if active {
        TAB_ACTIVE_TEXT
    } else {
        TAB_INACTIVE_TEXT
    };
    let baseline = tab.rect.y + (tab.rect.h - (m.ascent + m.descent)) / 2 + m.ascent;
    // The label hard-cuts at the gap before the close box (no ellipsis), and never
    // past the viewport, so a long name fills the space and slides under both edges.
    let label_right = (tab.close.x - scale.px(TAB_CLOSE_GAP)).min(clip[1]);
    list.push(clip_text_x(
        text_cmd(
            ctx.regular(),
            tab.rect.x + pad,
            baseline,
            label_right,
            color,
            view.label.to_string(),
            m,
        ),
        clip[0],
        label_right,
    ));
    // The close box, pin, and dot are small targets, so they show only while wholly
    // inside the viewport rather than as a sheared sliver. A pin badge owns the slot
    // whenever the tab is pinned, standing where the close cross would so the click
    // there unpins rather than closes.
    let close_in = tab.close.x >= clip[0] && tab.close.x + tab.close.w <= clip[1];
    if view.pinned {
        if close_in {
            push_pin_badge(list, ctx, tab.close, active || close_hover, close_hover);
        }
    } else if active || hovered {
        if close_in {
            push_glyph_button(list, ctx, tab.close, "×", close_hover);
        }
    } else if view.dirty && close_in {
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
    let mut list = DisplayList::new();
    build_display_list_into(&mut list, fonts, layout, rope, p);
    list
}

/// [`build_display_list`] appending into `list`, so the window's frame is composed
/// in one buffer instead of a Vec per pane that is then copied in.
pub(crate) fn build_display_list_into(
    list: &mut DisplayList,
    fonts: &Fonts,
    layout: &Layout,
    rope: &crate::rope::Rope,
    p: &PaintParams,
) {
    let mut painter = Painter::new(list, fonts, layout, rope, p);
    // Bottom-to-top stacking order. This sequence is also the damage-diff order,
    // so the passes must run in exactly this order; each is one method below.
    painter.background();
    painter.code_block_panels();
    painter.inline_code_panels();
    if let Some((start, end)) = p.selection {
        painter.selection(start, end);
    }
    if let Some(diff) = &p.diff {
        painter.diff(diff);
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
    list: &'a mut DisplayList,
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
    /// The heading whose fold chevron the pointer is over, by logical-line start.
    fold_hover: Option<usize>,
}

impl<'a> Painter<'a> {
    fn new(
        list: &'a mut DisplayList,
        fonts: &'a Fonts,
        layout: &'a Layout,
        rope: &'a crate::rope::Rope,
        p: &PaintParams,
    ) -> Self {
        // Roughly a few commands per visual line (panel, runs, decorations) plus the
        // background and caret; reserving up front beats growing the list mid-build.
        list.reserve(layout.lines().len() * 3 + 8);
        Self {
            list,
            fonts,
            layout,
            rope,
            scroll_y: p.scroll_y,
            view: p.view,
            left: layout.left(),
            scale: p.scale,
            body_size: p.body_size,
            fold_hover: p.fold_hover,
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

    /// The diff view's word washes, beneath the text like the selection: a stronger
    /// green fill behind each added word and a stronger red behind each removed one, over
    /// the faint full-line wash already drawn, so the edited words stand out within the
    /// changed line. The ranges are into the side's own text.
    fn diff(&mut self, diff: &DiffTint) {
        let mut wash = |ranges: &[(usize, usize)], color: u32| {
            for &(start, end) in ranges {
                for rect in self
                    .layout
                    .diff_wash_rects(self.fonts, self.rope, start, end)
                {
                    if let Some(rect) = self.screen(rect) {
                        self.list.push(DrawCmd::Fill { rect, color });
                    }
                }
            }
        };
        wash(diff.inserts, AI_INSERT_BG);
        wash(diff.deletes, AI_DELETE_BG);
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
                        push_image_chrome(self.list, fonts, &pic, self.scale, self.body_size);
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
            // A heading that owns a section draws a fold chevron in the margin, the
            // same shape and color the file tree marks a folder with: down while
            // the section is open, right once it is folded. An open one appears only
            // under the pointer, so a document at rest is not lined with chevrons; a
            // folded one always shows, since it is the only sign that content is
            // hidden there. The chevron sits left of the text column, so it never
            // shifts a glyph.
            if let Some(folded) = line.fold {
                if folded || self.fold_hover == Some(line.start) {
                    if let Some(rect) = self.layout.fold_rect(fonts, line, self.body_size) {
                        self.list.push(DrawCmd::Chevron {
                            rect: Rect {
                                y: rect.y - scroll_y,
                                ..rect
                            },
                            color: MARKER_COLOR,
                            down: !folded,
                        });
                    }
                }
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
                // Borrowed while the run sits inside one rope leaf, which is the
                // normal case, so a marker that is only measured and then replaced by
                // its gutter decoration costs nothing.
                let slice = rope.slice(run.start..run.end).as_cow();
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
                                push_checkbox(self.list, rect, checked, BACKGROUND);
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
                // only draw the glyphs over them. A run's glyphs sit its leading pill
                // pad in from the pen, and the pen clears the trailing one after them,
                // so the panel (drawn from the same geometry) frames the code without
                // painting over the spaces around it.
                let (lead, trail) = run.pads.px(self.scale);
                let run_x = pen + lead;
                let run_end = self.push_text(fk, run_x, baseline, color, slice.into_owned(), m);
                pen = run_end + trail;
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
        for run in &cl.runs {
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
            // A run's glyphs sit its leading pill pad in from the pen, and the pen
            // clears the trailing one after them, so the tint panel frames the code
            // without covering the text around it, the same as the body loop.
            let (lead, trail) = run.pads.px(self.scale);
            let run_x = pen + lead;
            let run_end = self.push_text(
                fk,
                run_x,
                baseline,
                color,
                self.rope.byte_slice(run.start..run.end),
                m,
            );
            pen = run_end + trail;
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
            self.list,
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
        push_find_ticks(self.list, find, self.view, content_height, self.scale);
    }

    /// The find bar panel, floating top-right over everything else.
    fn find_bar(&mut self, find: &FindPaint) {
        push_find_bar(
            self.list,
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
    let fk = FaceKey::Code {
        size: body_size,
        ink: CodeInk::Inline,
    };
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

/// Which button of a conflict banner a click lands on. The primary action takes
/// the disk's side (reload, or recreate a deleted file); the secondary keeps the
/// buffer's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BannerButton {
    Primary,
    Secondary,
}

/// The rectangles a conflict banner occupies: the full strip and its two buttons.
/// Shared by the painter and the pointer hit-testing so a click lands on the
/// button it looks like.
#[derive(Clone, Copy)]
pub(crate) struct BannerLayout {
    pub bar: Rect,
    pub primary: Rect,
    pub secondary: Rect,
}

impl BannerLayout {
    /// The button at window point `(x, y)`, if the point is on one.
    pub(crate) fn button_at(&self, x: f32, y: f32) -> Option<BannerButton> {
        if rect_contains(self.primary, x, y) {
            Some(BannerButton::Primary)
        } else if rect_contains(self.secondary, x, y) {
            Some(BannerButton::Secondary)
        } else {
            None
        }
    }

    /// Whether the point falls anywhere on the banner strip (so a click there is
    /// the banner's, not the text beneath it).
    pub(crate) fn contains(&self, x: f32, y: f32) -> bool {
        rect_contains(self.bar, x, y)
    }
}

fn rect_contains(r: Rect, x: f32, y: f32) -> bool {
    x >= r.x as f32 && x < (r.x + r.w) as f32 && y >= r.y as f32 && y < (r.y + r.h) as f32
}

/// The message and the two button labels a conflict of `kind` presents.
fn banner_labels(kind: DiskConflict) -> (&'static str, &'static str, &'static str) {
    match kind {
        DiskConflict::Changed => ("This file changed on disk.", "Reload", "Keep mine"),
        DiskConflict::Deleted => ("This file was deleted on disk.", "Save", "Dismiss"),
    }
}

/// Lay out the conflict banner across the top of pane `view`: a full-width strip
/// with the primary and secondary buttons right-aligned. Pure geometry, so the
/// painter and the click hit-test agree on where the buttons are.
pub(crate) fn conflict_banner_layout(
    view: Rect,
    kind: DiskConflict,
    body_size: u32,
    fonts: &Fonts,
) -> BannerLayout {
    let face = fonts.face(body_size, FontStyle::Regular);
    let m = fonts.metrics(body_size);
    let pad = (body_size as i32 / 2).max(4);
    let row_h = m.ascent + m.descent;
    let h = row_h + 2 * pad;
    let bar = Rect {
        x: view.x,
        y: view.y,
        w: view.w,
        h,
    };

    let (_, primary_label, secondary_label) = banner_labels(kind);
    let button_w = |label: &str| shape::text_advance(face, label).ceil() as i32 + 2 * pad;
    let btn_h = row_h + pad;
    let btn_y = bar.y + (h - btn_h) / 2;
    let sec_w = button_w(secondary_label);
    let pri_w = button_w(primary_label);
    let secondary = Rect {
        x: bar.x + bar.w - pad - sec_w,
        y: btn_y,
        w: sec_w,
        h: btn_h,
    };
    let primary = Rect {
        x: secondary.x - pad - pri_w,
        y: btn_y,
        w: pri_w,
        h: btn_h,
    };
    BannerLayout {
        bar,
        primary,
        secondary,
    }
}

/// Draw the conflict banner into `list` for a document in conflict of `kind`,
/// using the already-computed `layout`; `hover` names the button under the
/// pointer so it can light up. The message sits on the left, the two buttons on
/// the right.
pub(crate) fn push_conflict_banner(
    list: &mut DisplayList,
    fonts: &Fonts,
    layout: &BannerLayout,
    kind: DiskConflict,
    hover: Option<BannerButton>,
    body_size: u32,
) {
    let face = fonts.face(body_size, FontStyle::Regular);
    let m = fonts.metrics(body_size);
    let key = FaceKey::Prose {
        size: body_size,
        style: FontStyle::Regular,
    };
    let pad = (body_size as i32 / 2).max(4);
    let (message, primary_label, secondary_label) = banner_labels(kind);

    list.push(DrawCmd::Fill {
        rect: layout.bar,
        color: CONFLICT_BANNER_BG,
    });

    // The message, left-aligned and vertically centered in the strip.
    let baseline = layout.bar.y + (layout.bar.h - (m.ascent + m.descent)) / 2 + m.ascent;
    let msg_x = layout.bar.x + pad;
    let msg_w = shape::text_advance(face, message).ceil() as i32;
    list.push(text_cmd(
        key,
        msg_x,
        baseline,
        msg_x + msg_w,
        CONFLICT_BANNER_TEXT,
        message.to_string(),
        m,
    ));

    let mut button = |rect: Rect, label: &str, hot: bool| {
        list.push(DrawCmd::Fill {
            rect,
            color: if hot {
                CONFLICT_BUTTON_HOVER_BG
            } else {
                CONFLICT_BUTTON_BG
            },
        });
        let label_w = shape::text_advance(face, label).ceil() as i32;
        let lx = rect.x + (rect.w - label_w) / 2;
        let lb = rect.y + (rect.h - (m.ascent + m.descent)) / 2 + m.ascent;
        list.push(text_cmd(
            key,
            lx,
            lb,
            lx + label_w,
            FOREGROUND,
            label.to_string(),
            m,
        ));
    };
    button(
        layout.primary,
        primary_label,
        hover == Some(BannerButton::Primary),
    );
    button(
        layout.secondary,
        secondary_label,
        hover == Some(BannerButton::Secondary),
    );
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::theme::{FONT_SIZE, RAIL_W, VAULT_COLORS};

    /// The app's real font stack at the body size, so a panel is measured against the
    /// same text the painter draws.
    fn test_fonts() -> Fonts {
        Fonts::with_config(
            &crate::platform::freetype::FontConfig::default(),
            &[FONT_SIZE],
            1.5,
        )
        .expect("open the default fonts")
    }

    #[test]
    fn the_diff_view_tints_each_side_and_numbers_its_lines() {
        use crate::pane::{DiffView, DocId};
        use crate::theme::{
            AI_DELETE_BG, AI_INSERT_BG, DIFF_DELETE_ACCENT, DIFF_DIVIDER, DIFF_INSERT_ACCENT,
        };

        let fonts = test_fonts();
        let view = Rect {
            x: 0,
            y: 0,
            w: 900,
            h: 400,
        };
        let mut diff = DiffView::new(
            DocId::from_raw(0),
            (0, 7),
            0,
            "teh cat".into(),
            "the cat".into(),
            "Fix spelling & grammar",
        );
        let cols = diff_columns(view, Scale::ONE, FONT_SIZE, &fonts);
        diff.ensure_layouts(&fonts, &cols, Scale::ONE, FONT_SIZE);
        let mut list: DisplayList = Vec::new();
        push_diff_view(
            &mut list,
            &fonts,
            view,
            &diff,
            Scale::ONE,
            FONT_SIZE,
            &scroll::Scrollbar::default(),
        );

        // The removed word is washed red on the left; the added word is washed green on
        // the right. Neither is struck through.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == AI_DELETE_BG)));
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == AI_INSERT_BG)));
        // Both sides draw their text.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("teh"))));
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Text { text, .. } if text.contains("the cat"))));
        // A line-number gutter numbers the first line on each side; this line changed, so
        // its number takes the accent color (red on the left, green on the right).
        assert!(list.iter().any(|c| matches!(
            c,
            DrawCmd::Text { text, color, .. } if text == "1" && *color == DIFF_DELETE_ACCENT
        )));
        assert!(list.iter().any(|c| matches!(
            c,
            DrawCmd::Text { text, color, .. } if text == "1" && *color == DIFF_INSERT_ACCENT
        )));
        // The two columns are parted by a divider.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == DIFF_DIVIDER)));
    }

    #[test]
    fn a_one_sided_change_hatches_the_empty_half_and_bands_the_added_line() {
        use crate::pane::{DiffView, DocId};
        use crate::theme::{
            DIFF_GUTTER_BORDER, DIFF_GUTTER_TEXT, DIFF_HATCH, DIFF_INSERT_ACCENT,
            DIFF_LINE_INSERT_BG,
        };

        let fonts = test_fonts();
        // A non-zero top (as a real pane sits below the top bar), so the test also guards
        // that the aligned content is placed in the pane rather than above it.
        let view = Rect {
            x: 0,
            y: 40,
            w: 900,
            h: 400,
        };
        // A line inserted between two kept lines: "a\nc" -> "a\nb\nc". The new middle
        // line has no counterpart on the left, so that half is hatched and the added
        // line is banded and accented on the right.
        let mut diff = DiffView::new(
            DocId::from_raw(0),
            (0, 3),
            0,
            "a\nc".into(),
            "a\nb\nc".into(),
            "Fix spelling & grammar",
        );
        let cols = diff_columns(view, Scale::ONE, FONT_SIZE, &fonts);
        diff.ensure_layouts(&fonts, &cols, Scale::ONE, FONT_SIZE);
        let mut list: DisplayList = Vec::new();
        push_diff_view(
            &mut list,
            &fonts,
            view,
            &diff,
            Scale::ONE,
            FONT_SIZE,
            &scroll::Scrollbar::default(),
        );

        // The empty left half of the added line has a hatched text area and a solid
        // gutter of the same tone.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Hatch { color, .. } if *color == DIFF_HATCH)));
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == DIFF_HATCH)));
        // The added line carries a faint full-line wash (extending into its gutter) and
        // its line number is drawn in the green accent.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == DIFF_LINE_INSERT_BG)));
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Text { color, .. } if *color == DIFF_INSERT_ACCENT)));
        // The right side numbers three lines, the left only two: the gutter numbers are
        // the real line numbers, aligned across the change.
        let numbers: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } if text.chars().all(|ch| ch.is_ascii_digit()) => {
                    Some(text.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(numbers.iter().filter(|n| **n == "3").count(), 1);
        assert_eq!(numbers.iter().filter(|n| **n == "2").count(), 2);
        // The unchanged first line's number is drawn in the dim gutter tone (only changed
        // lines take an accent color).
        assert!(list.iter().any(|c| matches!(
            c,
            DrawCmd::Text { text, color, .. } if text == "1" && *color == DIFF_GUTTER_TEXT
        )));
        // A subtle hairline parts each gutter from its text.
        assert!(list
            .iter()
            .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == DIFF_GUTTER_BORDER)));
        // Nothing is drawn above the pane top: the aligned bands are placed in the pane's
        // window space, not at the window origin.
        let min_text_y = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { baseline, .. } => Some(*baseline),
                _ => None,
            })
            .min()
            .expect("some text");
        assert!(min_text_y >= view.y, "text is drawn above the pane top");
    }

    #[test]
    fn a_wholly_added_list_item_washes_its_bullet_too() {
        use crate::pane::{DiffView, DocId};
        use crate::theme::{AI_INSERT_BG, MARKER_COLOR};

        let fonts = test_fonts();
        let view = Rect {
            x: 0,
            y: 0,
            w: 900,
            h: 400,
        };
        // A bullet item added at the end: its whole line changed, so the wash must reach
        // its bullet decoration rather than starting at the text past the collapsed marker.
        let mut diff = DiffView::new(
            DocId::from_raw(0),
            (0, 0),
            0,
            "- a\n- b".into(),
            "- a\n- b\n- c".into(),
            "x",
        );
        let cols = diff_columns(view, Scale::ONE, FONT_SIZE, &fonts);
        diff.ensure_layouts(&fonts, &cols, Scale::ONE, FONT_SIZE);
        let mut list: DisplayList = Vec::new();
        push_diff_view(
            &mut list,
            &fonts,
            view,
            &diff,
            Scale::ONE,
            FONT_SIZE,
            &scroll::Scrollbar::default(),
        );

        // The bullet disc of the added line, and the strong wash on that same row.
        let disc = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::RoundRect { rect, color, .. } if *color == MARKER_COLOR => Some(*rect),
                _ => None,
            })
            .max_by_key(|r| r.y)
            .expect("a bullet disc for the added item");
        let wash = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Fill { rect, color } if *color == AI_INSERT_BG => Some(*rect),
                _ => None,
            })
            .find(|r| r.y <= disc.y && r.y + r.h >= disc.y + disc.h)
            .expect("a strong wash on the added item's row");
        // The wash's left edge reaches the disc, so the bullet sits inside the highlight.
        assert!(
            wash.x <= disc.x,
            "the wash (x={}) does not reach the bullet (x={})",
            wash.x,
            disc.x
        );
    }

    #[test]
    fn a_long_palette_query_keeps_the_caret_inside_the_field() {
        use crate::palette::Palette;

        let fonts = test_fonts();
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        // A query far longer than the field, typed into the custom-prompt input.
        let mut p = Palette::query_input("Describe the change");
        for _ in 0..200 {
            p.push('x');
        }
        let layout = p.layout(&fonts, FONT_SIZE, Scale::ONE, (600, 800));
        let mut list: DisplayList = Vec::new();
        push_palette(&mut list, &ctx, &p, &layout);

        // The steady caret is the one CARET_COLOR fill; it must stay within the field
        // rather than running off past the panel's edge with the text.
        let caret = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::Fill { rect, color } if *color == CARET_COLOR => Some(*rect),
                _ => None,
            })
            .expect("a caret fill");
        assert!(caret.x >= layout.field.x, "the caret is left of the field");
        assert!(
            caret.x + caret.w <= layout.field.x + layout.field.w,
            "the caret runs past the field's right edge"
        );
        // The query text is drawn clipped to the field, so what scrolls off does not
        // spill into the panel's padding.
        assert!(list.iter().any(
            |c| matches!(c, DrawCmd::Text { clip_x: Some(_), color, .. } if *color == PALETTE_TEXT)
        ));
    }

    /// A rail column the width the theme asks for, at scale 1.0.
    fn rail(h: i32) -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: RAIL_W,
            h,
        }
    }

    #[test]
    fn rail_chips_are_squares_stacked_from_the_top() {
        let chips: Vec<Rect> = rail_buttons(rail(700), 3, Scale::ONE)
            .map(|(_, c)| c)
            .collect();
        assert_eq!(chips.len(), 3);
        let side = RAIL_W - 2 * RAIL_PAD;
        for (i, chip) in chips.iter().enumerate() {
            // Square, inset from both edges, so the chip is centered in the column.
            assert_eq!((chip.w, chip.h), (side, side));
            assert_eq!(chip.x, RAIL_PAD);
            assert_eq!(RAIL_W - (chip.x + chip.w), RAIL_PAD);
            assert_eq!(chip.y, RAIL_PAD + i as i32 * (side + RAIL_GAP));
        }
        // Stacked with a gap between, never overlapping.
        assert_eq!(chips[1].y - (chips[0].y + chips[0].h), RAIL_GAP);
    }

    #[test]
    fn rail_chips_stack_from_their_regions_top_not_the_windows() {
        // The chip region starts below the top bar, and the chips follow it down: the
        // padding is measured from the region, and one that would hang off the rail's
        // bottom is still dropped.
        let region = Rect { y: 30, ..rail(670) };
        let chips: Vec<Rect> = rail_buttons(region, 2, Scale::ONE)
            .map(|(_, c)| c)
            .collect();
        assert_eq!(chips[0].y, 30 + RAIL_PAD);
        assert_eq!(chips[1].y, chips[0].y + chips[0].h + RAIL_GAP);
        assert!(chips.iter().all(|c| c.y + c.h <= region.y + region.h));
    }

    #[test]
    fn rail_chips_are_indexed_by_their_vault() {
        let indices: Vec<usize> = rail_buttons(rail(700), 4, Scale::ONE)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn rail_drops_the_chips_that_do_not_fit() {
        // A column two chips tall offers two vaults, not the four configured: a chip
        // hanging off the bottom is neither drawn nor clickable.
        let side = RAIL_W - 2 * RAIL_PAD;
        let two = RAIL_PAD + 2 * side + RAIL_GAP;
        let chips: Vec<Rect> = rail_buttons(rail(two), 4, Scale::ONE)
            .map(|(_, c)| c)
            .collect();
        assert_eq!(chips.len(), 2);
        assert!(chips.iter().all(|c| c.y + c.h <= two));
        // One pixel short of the second chip's bottom edge drops it.
        assert_eq!(rail_buttons(rail(two - 1), 4, Scale::ONE).count(), 1);
    }

    #[test]
    fn rail_chips_scale_with_the_display() {
        let scale = Scale::from_120(240);
        let column = Rect {
            w: scale.px(RAIL_W),
            ..rail(1400)
        };
        let (_, chip) = rail_buttons(column, 1, scale).next().expect("one chip");
        assert_eq!(chip.w, scale.px(RAIL_W) - 2 * scale.px(RAIL_PAD));
        assert_eq!(chip.x, scale.px(RAIL_PAD));
    }

    #[test]
    fn rail_offers_no_chips_without_vaults() {
        assert_eq!(rail_buttons(rail(700), 0, Scale::ONE).count(), 0);
    }

    #[test]
    fn chip_ink_takes_the_side_that_reads_on_the_fill() {
        // The extremes are unambiguous either way.
        assert_eq!(chip_ink(0x00ff_ffff), RAIL_INK_DARK);
        assert_eq!(chip_ink(0x0000_0000), RAIL_INK_LIGHT);
        // Green carries most of the luminance and blue almost none, so a pure blue
        // takes light ink while a pure green, no brighter in raw bytes, takes dark.
        assert_eq!(chip_ink(0x0000_00ff), RAIL_INK_LIGHT);
        assert_eq!(chip_ink(0x0000_ff00), RAIL_INK_DARK);
    }

    #[test]
    fn every_palette_color_carries_legible_initials() {
        // WCAG's AA floor for text. The palette is tuned to clear it; a color added
        // to it that cannot is a color the rail should not deal.
        for color in VAULT_COLORS {
            let ratio = contrast(color, chip_ink(color));
            assert!(ratio >= 4.5, "{color:#08x} inks at only {ratio:.2}:1",);
        }
    }

    #[test]
    fn contrast_runs_from_one_to_twenty_one() {
        assert!((contrast(0x0012_3456, 0x0012_3456) - 1.0).abs() < 0.001);
        assert!((contrast(0x0000_0000, 0x00ff_ffff) - 21.0).abs() < 0.001);
        // Symmetric: it is a ratio between two colors, not of one against another.
        assert_eq!(
            contrast(0x0000_0000, 0x00ff_ffff),
            contrast(0x00ff_ffff, 0x0000_0000)
        );
    }

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
            fold_hover: None,
            diff: None,
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
        }
    }

    /// Tab views from labels plus their unsaved and pinned flags, both padded with
    /// false, for the strip painting and geometry tests.
    fn views<'a>(labels: &[&'a str], dirty: &[bool], pinned: &[bool]) -> Vec<TabView<'a>> {
        labels
            .iter()
            .enumerate()
            .map(|(i, label)| TabView {
                label,
                dirty: dirty.get(i).copied().unwrap_or(false),
                pinned: pinned.get(i).copied().unwrap_or(false),
            })
            .collect()
    }

    /// `n` unlabeled, unpinned tabs, for the tests that only measure widths.
    fn plain_views(n: usize) -> Vec<TabView<'static>> {
        views(&vec![""; n], &[], &[])
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

    /// Whether `r` sits inside a `window`-sized window, padding included.
    fn within(r: Rect, window: (i32, i32)) -> bool {
        let pad = IMAGE_VIEW_PAD;
        r.x >= pad && r.y >= pad && r.x + r.w <= window.0 - pad && r.y + r.h <= window.1 - pad
    }

    /// How far `r`'s aspect is from `w`/`h`'s, as a fraction.
    fn aspect_error(r: Rect, w: u32, h: u32) -> f32 {
        let want = w as f32 / h as f32;
        let got = r.w as f32 / r.h as f32;
        (got - want).abs() / want
    }

    #[test]
    fn a_picture_too_big_for_the_window_is_shrunk_to_fit() {
        let window = (1000, 800);
        let r = image_view_rect(window, (4000, 2000), Scale::ONE);
        assert!(within(r, window), "{r:?}");
        assert!(aspect_error(r, 4000, 2000) < 0.01, "{r:?}");
        // Wider than it is tall, so the width is what runs out: it uses all of it.
        assert_eq!(r.w, window.0 - 2 * IMAGE_VIEW_PAD);
    }

    #[test]
    fn a_tall_picture_is_fitted_by_its_height() {
        let window = (1000, 800);
        let r = image_view_rect(window, (500, 4000), Scale::ONE);
        assert!(within(r, window), "{r:?}");
        assert!(aspect_error(r, 500, 4000) < 0.01, "{r:?}");
        assert_eq!(r.h, window.1 - 2 * IMAGE_VIEW_PAD);
    }

    #[test]
    fn a_small_picture_is_shown_at_its_own_size() {
        // Never blown up: a picture past its own pixels is a picture of its pixels,
        // and the point of opening it is to see what is in it.
        let window = (1000, 800);
        let r = image_view_rect(window, (64, 48), Scale::ONE);
        assert_eq!((r.w, r.h), (64, 48));
        // And centered in the window, not in the padded box.
        assert_eq!(r.x, (1000 - 64) / 2);
        assert_eq!(r.y, (800 - 48) / 2);
    }

    #[test]
    fn a_picture_is_centered_whichever_way_it_fits() {
        let window = (1000, 800);
        for size in [(4000u32, 2000u32), (500, 4000), (64, 48), (900, 700)] {
            let r = image_view_rect(window, size, Scale::ONE);
            assert_eq!(r.x, (window.0 - r.w) / 2, "{size:?} across");
            assert_eq!(r.y, (window.1 - r.h) / 2, "{size:?} down");
        }
    }

    #[test]
    fn a_picture_exactly_the_padded_size_is_left_alone() {
        let window = (1000, 800);
        let (w, h) = (
            (window.0 - 2 * IMAGE_VIEW_PAD) as u32,
            (window.1 - 2 * IMAGE_VIEW_PAD) as u32,
        );
        let r = image_view_rect(window, (w, h), Scale::ONE);
        assert_eq!((r.w as u32, r.h as u32), (w, h), "no shrink, no growth");
    }

    #[test]
    fn a_picture_in_a_window_smaller_than_its_padding_still_has_a_size() {
        // A window shrunk to nothing must not give a rectangle of nothing (or of a
        // negative something the renderer would refuse).
        let r = image_view_rect((10, 10), (4000, 2000), Scale::ONE);
        assert!(r.w >= 1 && r.h >= 1, "{r:?}");
    }

    #[test]
    fn a_picture_with_no_size_still_has_a_rectangle() {
        let r = image_view_rect((1000, 800), (0, 0), Scale::ONE);
        assert!(r.w >= 1 && r.h >= 1, "{r:?}");
    }

    #[test]
    fn the_view_covers_the_window_and_then_draws_the_picture() {
        // The backdrop is what makes it a view rather than a picture floating over the
        // text: it covers everything, and the picture goes on top of it.
        let mut list = Vec::new();
        push_image_view(&mut list, (1000, 800), (4000, 2000), "shot.png", Scale::ONE);
        assert_eq!(list.len(), 2);
        match &list[0] {
            DrawCmd::Fill { rect, .. } => {
                assert_eq!((rect.x, rect.y, rect.w, rect.h), (0, 0, 1000, 800));
            }
            other => panic!("the backdrop is a fill, not {other:?}"),
        }
        match &list[1] {
            DrawCmd::Image { rect, path } => {
                assert_eq!(path, "shot.png");
                assert_eq!(
                    *rect,
                    image_view_rect((1000, 800), (4000, 2000), Scale::ONE)
                );
            }
            other => panic!("the picture is an image, not {other:?}"),
        }
    }

    #[test]
    fn rail_drop_index_tracks_the_chips_the_pointer_has_passed() {
        let chips = rail(700);
        let stack: Vec<Rect> = rail_buttons(chips, 3, Scale::ONE).map(|(_, r)| r).collect();
        let mid = |i: usize| stack[i].y as f32 + stack[i].h as f32 / 2.0;

        // Dragging the first chip down: it holds its slot until it passes the chip
        // below's midpoint, then takes that slot, and so on.
        assert_eq!(rail_drop_index(chips, 3, Scale::ONE, Some(0), mid(0)), 0);
        assert_eq!(
            rail_drop_index(chips, 3, Scale::ONE, Some(0), mid(1) + 1.0),
            1
        );
        assert_eq!(
            rail_drop_index(chips, 3, Scale::ONE, Some(0), mid(2) + 1.0),
            2
        );

        // Dragging the last chip up: its own slot is skipped, so only the chips above
        // it count and it can reach the top.
        assert_eq!(
            rail_drop_index(chips, 3, Scale::ONE, Some(2), mid(2) + 1.0),
            2
        );
        assert_eq!(rail_drop_index(chips, 3, Scale::ONE, Some(2), mid(0)), 0);
    }

    #[test]
    fn rail_drop_index_never_leaves_the_rail() {
        // A pointer dragged off the top or the bottom of the window still names a slot
        // the rail has, so a chip cannot be dropped out of it.
        let chips = rail(700);
        assert_eq!(rail_drop_index(chips, 3, Scale::ONE, Some(1), -500.0), 0);
        assert_eq!(rail_drop_index(chips, 3, Scale::ONE, Some(1), 5000.0), 2);
    }

    /// Three vaults and the fonts to draw their chips, for the rail-drag tests.
    fn rail_test_vaults() -> Vec<Vault> {
        crate::config::vaults(
            ["/home/user/notes", "/home/user/work", "/home/user/journal"]
                .iter()
                .map(std::path::PathBuf::from),
        )
    }

    #[test]
    fn a_dragged_chip_leaves_a_gap_and_is_drawn_over_the_rest() {
        // The chip in flight is not in the stack (its slot is the gap the drop will
        // fill) and is drawn last, so it passes over its neighbors.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        let vaults = rail_test_vaults();
        let chips = rail(700);
        let resting: Vec<Rect> = rail_buttons(chips, 3, Scale::ONE).map(|(_, r)| r).collect();

        let mut list = Vec::new();
        push_rail(
            &mut list,
            &ctx,
            chips,
            chips,
            &RailView {
                vaults: &vaults,
                active: Some(0),
                hovered: None,
                drag: Some(RailDragView { vault: 0, top: 400 }),
            },
        );
        // The chip bodies drawn, in the order they were pushed.
        let bodies: Vec<Rect> = list
            .iter()
            .filter_map(|cmd| match cmd {
                DrawCmd::RoundRect { rect, radius, .. }
                    if *radius == Scale::ONE.px(RAIL_RADIUS) && rect.w == resting[0].w =>
                {
                    Some(*rect)
                }
                _ => None,
            })
            .collect();
        // Two chips at rest (1 and 2), then the dragged one last, at the pointer.
        assert_eq!(bodies.len(), 3);
        assert_eq!(bodies[0].y, resting[1].y);
        assert_eq!(bodies[1].y, resting[2].y);
        assert_eq!(bodies[2].y, 400, "the chip in flight follows the pointer");
        // Nothing was drawn in the dragged chip's own slot: that gap is where it lands.
        assert!(!bodies[..2].iter().any(|r| r.y == resting[0].y));
    }

    #[test]
    fn a_dragged_chip_stops_at_the_ends_of_the_rail() {
        // Dragged past the rail's ends, the chip stops there rather than following the
        // pointer out of the column.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        let vaults = rail_test_vaults();
        let chips = rail(700);
        for (top, expect_within) in [(-9999, true), (9999, true)] {
            let mut list = Vec::new();
            push_rail(
                &mut list,
                &ctx,
                chips,
                chips,
                &RailView {
                    vaults: &vaults,
                    active: None,
                    hovered: None,
                    drag: Some(RailDragView { vault: 0, top }),
                },
            );
            let drawn = list
                .iter()
                .find_map(|cmd| match cmd {
                    DrawCmd::RoundRect { rect, .. } => Some(*rect),
                    _ => None,
                })
                .expect("the chip is drawn");
            assert_eq!(
                drawn.y >= chips.y && drawn.y + drawn.h <= chips.y + chips.h,
                expect_within,
                "a chip dragged to {top} stays in the rail"
            );
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
        let tabs = views(&["intro", "todo"], &[true, false], &[]);
        let strips = [
            Some(TabStrip {
                rect: strip_rect,
                tabs: &tabs,
                active: 1,
                hovered: None,
                close_hover: None,
                new_hover: false,
                scroll_x: 0,
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
        // The active tab (index 1) melts into the pane background over its whole
        // square box and shows a close glyph; the resting unsaved tab (index 0) shows
        // a dirty dot instead.
        let hits = tab_rects(strip_rect, &tabs, Scale::ONE, 0);
        let active = hits[1].rect;
        assert!(
            list.iter().any(
                |c| matches!(c, DrawCmd::Fill { rect, color } if *rect == active && *color == TAB_ACTIVE_BG)
            ),
            "active tab highlight"
        );
        assert!(has_text("×"), "close glyph on the active tab");
        assert!(
            has_round(TAB_DIRTY_COLOR),
            "dirty dot on the resting unsaved tab"
        );
        // The new-tab button sits at the end of the strip.
        assert!(has_text("+"), "new-tab button");
    }

    #[test]
    fn a_pinned_tab_wears_a_pin_badge_in_its_close_slot() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let top = top_bar_height(fonts.metrics(FONT_SIZE), Scale::ONE);
        let strip_rect = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: top,
        };
        // Two tabs: the first pinned and unsaved, the second active. The pin owns the
        // slot, so the pinned tab shows no unsaved dot and no close cross even though
        // it is dirty, and the badge sits in its close box.
        let tabs = views(&["ref", "todo"], &[true, false], &[true, false]);
        let strips = [
            Some(TabStrip {
                rect: strip_rect,
                tabs: &tabs,
                active: 1,
                hovered: None,
                close_hover: None,
                new_hover: false,
                scroll_x: 0,
                drag: None,
            }),
            None,
        ];
        let mut list: DisplayList = Vec::new();
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        push_top_bar(&mut list, &ctx, strip_rect, None, strips, &[]);

        let hits = tab_rects(strip_rect, &tabs, Scale::ONE, 0);
        let inside = |rect: &Rect, host: Rect| {
            rect.x >= host.x
                && rect.y >= host.y
                && rect.x + rect.w <= host.x + host.w
                && rect.y + rect.h <= host.y + host.h
        };
        // The pin badge lands inside the pinned tab's close box.
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Pin { rect, .. } if inside(rect, hits[0].close))),
            "pin badge on the pinned tab"
        );
        // The pinned tab is dirty, but the pin owns the slot, so no dirty dot is drawn
        // in its close box.
        assert!(
            !list.iter().any(|c| matches!(
                c,
                DrawCmd::RoundRect { rect, color, .. }
                    if *color == TAB_DIRTY_COLOR && inside(rect, hits[0].close)
            )),
            "no dirty dot behind the pin badge"
        );
        // The active tab (index 1) is not pinned, so it still shows its close cross.
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Text { text, .. } if text == "×")),
            "close cross on the unpinned active tab"
        );
    }

    #[test]
    fn tabs_share_one_width_that_shrinks_with_the_count() {
        let strip = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 30,
        };
        let avail = 800 - (TAB_CLOSE_GAP + TAB_NEW_W);
        // Few tabs: capped at the max width, leaving the rest of the lane empty.
        let two = tab_layout(strip, &plain_views(2), Scale::ONE);
        assert_eq!(two.unpinned_w, TAB_MAX_W);
        assert_eq!(two.avail, avail);
        assert!(
            two.content < two.avail,
            "two wide tabs do not fill the strip"
        );
        // Many tabs: an equal share strictly between the floor and the ceiling.
        let six = tab_layout(strip, &plain_views(6), Scale::ONE);
        assert_eq!(six.unpinned_w, avail / 6);
        assert!(six.unpinned_w > TAB_MIN_W && six.unpinned_w < TAB_MAX_W);
        // Too many: pinned at the floor and overflowing, so the strip must scroll.
        let many = tab_layout(strip, &plain_views(20), Scale::ONE);
        assert_eq!(many.unpinned_w, TAB_MIN_W);
        assert!(
            many.content > many.avail,
            "the run overflows the visible lane"
        );
    }

    #[test]
    fn a_pinned_tab_takes_the_compact_width_and_widens_the_rest() {
        // A strip whose two tabs, unpinned, would each land mid-range (not capped at
        // the ceiling), so pinning one visibly frees width to the other.
        let strip = Rect {
            x: 10,
            y: 0,
            w: 330, // avail = 330 - (TAB_CLOSE_GAP + TAB_NEW_W) = 300
            h: 30,
        };
        let both_free = tab_layout(strip, &plain_views(2), Scale::ONE);
        assert_eq!(
            both_free.unpinned_w, 150,
            "two unpinned split the lane evenly"
        );
        // Pin the first: it takes the fixed compact width, and the one unpinned tab
        // takes all the lane the pin left.
        let one_pinned = tab_layout(strip, &views(&["", ""], &[], &[true, false]), Scale::ONE);
        assert_eq!(one_pinned.pinned_w, TAB_PINNED_W);
        assert_eq!(one_pinned.unpinned_w, 300 - TAB_PINNED_W);
        assert!(
            one_pinned.unpinned_w > both_free.unpinned_w,
            "the freed space widens the remaining tab"
        );
        // The rects follow the widths: the compact pin sits first, the wide tab
        // begins where it ends.
        let tabs = tab_rects(
            strip,
            &views(&["ref", "work"], &[], &[true, false]),
            Scale::ONE,
            0,
        );
        assert_eq!(tabs[0].rect.w, TAB_PINNED_W);
        assert_eq!(tabs[1].rect.w, 300 - TAB_PINNED_W);
        assert_eq!(tabs[0].rect.x, strip.x);
        assert_eq!(tabs[1].rect.x, strip.x + TAB_PINNED_W);
    }

    #[test]
    fn tab_rects_are_uniform_index_aligned_and_shift_with_the_scroll() {
        let strip = Rect {
            x: 100,
            y: 0,
            w: 800,
            h: 30,
        };
        let tabs = plain_views(20);
        let w = tab_layout(strip, &tabs, Scale::ONE).unpinned_w;
        let unscrolled = tab_rects(strip, &tabs, Scale::ONE, 0);
        // One rect per label (a slot is kept even for a tab scrolled out of view),
        // all the same width, laid end to end from the strip's left edge.
        assert_eq!(unscrolled.len(), tabs.len());
        assert!(unscrolled.iter().all(|t| t.rect.w == w));
        assert_eq!(unscrolled[0].rect.x, strip.x);
        assert_eq!(unscrolled[3].rect.x, strip.x + 3 * w);
        // Scrolling shifts every tab left by the offset, keeping the slots aligned.
        let scrolled = tab_rects(strip, &tabs, Scale::ONE, 50);
        assert!(unscrolled
            .iter()
            .zip(&scrolled)
            .all(|(a, b)| b.rect.x == a.rect.x - 50));
    }

    #[test]
    fn the_new_tab_button_follows_the_tabs_then_pins_right_on_overflow() {
        let strip = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 30,
        };
        // Fitting: the + sits just past the last tab.
        let few = tab_layout(strip, &plain_views(2), Scale::ONE);
        let plus = new_tab_button_rect(strip, few, Scale::ONE);
        assert_eq!(plus.x, strip.x + few.content + TAB_CLOSE_GAP);
        // Overflowing: the + pins to the right of the visible lane and stays inside
        // the strip, so it is always reachable.
        let many = tab_layout(strip, &plain_views(20), Scale::ONE);
        let plus = new_tab_button_rect(strip, many, Scale::ONE);
        assert_eq!(plus.x, strip.x + many.avail + TAB_CLOSE_GAP);
        assert!(plus.x + plus.w <= strip.x + strip.w);
    }

    #[test]
    fn a_scrolled_strip_scissors_its_tabs_to_the_visible_lane() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let top = top_bar_height(fonts.metrics(FONT_SIZE), Scale::ONE);
        let strip_rect = Rect {
            x: 0,
            y: 0,
            w: 700,
            h: top,
        };
        let names: Vec<String> = (0..20).map(|i| format!("note-{i}")).collect();
        let labels: Vec<&str> = names.iter().map(String::as_str).collect();
        let tabs = views(&labels, &[], &[]);
        let layout = tab_layout(strip_rect, &tabs, Scale::ONE);
        let strips = [
            Some(TabStrip {
                rect: strip_rect,
                tabs: &tabs,
                active: 0,
                hovered: None,
                close_hover: None,
                new_hover: false,
                scroll_x: 40,
                drag: None,
            }),
            None,
        ];
        let mut list: DisplayList = Vec::new();
        let ctx = BarCtx::new(&fonts, Scale::ONE, FONT_SIZE);
        push_top_bar(
            &mut list,
            &ctx,
            Rect {
                x: 0,
                y: 0,
                w: 700,
                h: top,
            },
            None,
            strips,
            &[],
        );
        let vr = layout.viewport_right(strip_rect);
        // Every tab label is scissored between the strip's left edge and, at most,
        // the visible lane's right; its damage bounds are tightened to that range, so
        // a scrolled label never spills left of the strip nor past the + button.
        let mut clipped_labels = 0;
        for c in &list {
            if let DrawCmd::Text {
                bounds,
                clip_x: Some(cx),
                ..
            } = c
            {
                clipped_labels += 1;
                assert_eq!(cx[0], strip_rect.x, "scissored at the strip's left edge");
                assert!(cx[1] <= vr, "scissor stops at or before the + lane");
                assert!(
                    bounds.x >= strip_rect.x,
                    "bounds do not reach left of the strip"
                );
                assert!(
                    bounds.x + bounds.w <= vr,
                    "bounds do not reach into the + lane"
                );
            }
        }
        assert!(clipped_labels > 0, "tab labels are scissored");
        // The new-tab button is still drawn (pinned), unclipped.
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Text { text, clip_x: None, .. } if text == "+")),
            "pinned new-tab button"
        );
    }

    #[test]
    fn the_diff_pass_washes_insertions_green_and_deletions_red() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "teh the cat";
        let rope = Rope::from_str(src);
        let (w, h) = (600i32, 200i32);
        let wrap = w - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        // "teh" is the deletion, the following "the" the insertion.
        let deletes = [(0usize, 3usize)];
        let inserts = [(4usize, 7usize)];
        let mut params = paint_params(w, h);
        params.diff = Some(DiffTint {
            inserts: &inserts,
            deletes: &deletes,
        });
        let list = build_display_list(&fonts, &layout, &rope, &params);
        let fills: Vec<u32> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Fill { color, .. } => Some(*color),
                _ => None,
            })
            .collect();
        assert!(
            fills.contains(&AI_INSERT_BG),
            "an insertion gets a green wash"
        );
        assert!(fills.contains(&AI_DELETE_BG), "a deletion gets a red wash");
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

    #[test]
    fn conflict_banner_buttons_are_ordered_and_hit_where_drawn() {
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let view = Rect {
            x: 40,
            y: 12,
            w: 600,
            h: 400,
        };
        let banner = conflict_banner_layout(view, DiskConflict::Changed, FONT_SIZE, &fonts);

        // The strip spans the pane width along its top edge.
        assert_eq!(
            (banner.bar.x, banner.bar.y, banner.bar.w),
            (view.x, view.y, view.w)
        );

        // Primary sits left of secondary; both lie inside the strip and do not overlap.
        assert!(banner.primary.x + banner.primary.w <= banner.secondary.x);
        for b in [banner.primary, banner.secondary] {
            assert!(b.x >= banner.bar.x && b.x + b.w <= banner.bar.x + banner.bar.w);
            assert!(b.y >= banner.bar.y && b.y + b.h <= banner.bar.y + banner.bar.h);
        }

        // A click on each button resolves to it; the message area and points off the
        // strip resolve to nothing.
        let mid = |r: Rect| ((r.x + r.w / 2) as f32, (r.y + r.h / 2) as f32);
        let (px, py) = mid(banner.primary);
        assert_eq!(banner.button_at(px, py), Some(BannerButton::Primary));
        let (sx, sy) = mid(banner.secondary);
        assert_eq!(banner.button_at(sx, sy), Some(BannerButton::Secondary));
        assert_eq!(
            banner.button_at((view.x + 4) as f32, (view.y + 2) as f32),
            None,
            "the message side is not a button"
        );
        assert!(banner.contains(px, py));
        assert!(
            !banner.contains(px, (banner.bar.y + banner.bar.h + 5) as f32),
            "a point below the strip is not on the banner"
        );
    }

    #[test]
    fn conflict_banner_paints_a_strip_with_a_lit_button() {
        let fonts = Fonts::new(&font_sizes(FONT_SIZE)).expect("a default font");
        let view = Rect {
            x: 0,
            y: 0,
            w: 500,
            h: 300,
        };
        let banner = conflict_banner_layout(view, DiskConflict::Deleted, FONT_SIZE, &fonts);
        let mut list = DisplayList::new();
        push_conflict_banner(
            &mut list,
            &fonts,
            &banner,
            DiskConflict::Deleted,
            Some(BannerButton::Primary),
            FONT_SIZE,
        );

        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { rect, color }
                if *rect == banner.bar && *color == CONFLICT_BANNER_BG)),
            "the strip background is drawn"
        );
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { rect, color }
                if *rect == banner.primary && *color == CONFLICT_BUTTON_HOVER_BG)),
            "the hovered primary button lights up"
        );
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { rect, color }
                if *rect == banner.secondary && *color == CONFLICT_BUTTON_BG)),
            "the un-hovered secondary button uses the idle color"
        );
        assert!(
            list.iter().any(|c| matches!(c, DrawCmd::Text { .. })),
            "the message and labels are drawn"
        );
    }
}
