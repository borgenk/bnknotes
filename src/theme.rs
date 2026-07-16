/// Background, text, and selection colors (0x00RRGGBB), and the pixel size: a
/// dark palette on base hue ~212.
pub(crate) const BACKGROUND: u32 = 0x001c_2127;
pub(crate) const FOREGROUND: u32 = 0x00da_dada;
/// The accent blue (`hsl(215,75%,70%)`) pre-blended for selection: the fill is
/// opaque, so this is the accent at a set opacity, high enough to stay legible
/// over the darker code panel.
pub(crate) const SELECTION_COLOR: u32 = 0x0041_5776;
/// Fill behind find matches other than the current one; the current match is
/// the live selection, so a muted gold keeps the two apart at a glance.
pub(crate) const FIND_MATCH_COLOR: u32 = 0x005e_5128;
/// The find bar's floating panel.
pub(crate) const FIND_BG: u32 = 0x0031_3244;
/// An invalid regex query and its counter, drawn in the alert color.
pub(crate) const FIND_ERROR_COLOR: u32 = 0x00f3_8ba8;
/// The right-edge match ticks (brighter than the in-text match fill, since a
/// tick is a few pixels).
pub(crate) const FIND_TICK_COLOR: u32 = 0x00a8_9040;
/// Geometry of one right-edge match tick.
pub(crate) const TICK_WIDTH: i32 = 4;
pub(crate) const TICK_HEIGHT: i32 = 3;
/// The banner shown when a file changed outside the editor: an amber strip warm
/// enough to draw the eye without the alarm of a red error. Its buttons sit a
/// shade lighter and brighten under the pointer, the way a tab's controls do.
pub(crate) const CONFLICT_BANNER_BG: u32 = 0x0053_411a;
pub(crate) const CONFLICT_BANNER_TEXT: u32 = 0x00f2_d9a3;
pub(crate) const CONFLICT_BUTTON_BG: u32 = 0x0074_5c23;
pub(crate) const CONFLICT_BUTTON_HOVER_BG: u32 = 0x0091_7530;
/// Dimmed color for markdown syntax markers (a heading's `#`s, a code span's
/// backticks), so they read as syntax rather than content.
pub(crate) const MARKER_COLOR: u32 = 0x005d_6c7e;
/// Inline code: a tinted, padded panel with the body foreground drawn on top. A
/// span is set apart by its monospace face and the panel rather than by hue.
pub(crate) const CODE_COLOR: u32 = FOREGROUND;
pub(crate) const CODE_BG: u32 = 0x0028_2c34;
/// A code delimiter's backticks (an inline span's `` ` `` and a fence's ```), drawn
/// brighter than the muted [`MARKER_COLOR`] the other syntax markers use so the
/// delimiters stay legible against the code panel's darker tint. A touch below the
/// body foreground so they still read as syntax rather than content.
pub(crate) const CODE_MARKER_COLOR: u32 = 0x00c8_ccd4;
/// Syntax-highlighting colors for code-block tokens, a One Dark palette that pairs
/// with the code panel's [`CODE_BG`]. Plain code stays [`CODE_COLOR`].
pub(crate) const CODE_KEYWORD_COLOR: u32 = 0x00c6_78dd;
pub(crate) const CODE_STRING_COLOR: u32 = 0x0098_c379;
pub(crate) const CODE_COMMENT_COLOR: u32 = 0x005c_6370;
pub(crate) const CODE_NUMBER_COLOR: u32 = 0x00d1_9a66;
/// Corner radius of a code panel, fenced block or inline span: a subtle rounding.
pub(crate) const CODE_RADIUS: i32 = 4;
/// Padding baked into an inline code panel so the tint reads as a rounded pill
/// with breathing room around the glyphs rather than a box hugging them.
pub(crate) const CODE_PAD_X: i32 = 4;
pub(crate) const CODE_PAD_Y: i32 = 1;
/// An inline link's text color (the accent blue, `hsl(215,75%,70%)`); the
/// text is also underlined so a link reads as a link without its `[..](..)` markup.
pub(crate) const LINK_COLOR: u32 = 0x005c_9cf6;
/// A selected image's border (a bright accent so it reads as "selected") and its
/// thickness; the corner button's panel and corner radius. The button glyph reuses
/// [`FOREGROUND`].
pub(crate) const IMAGE_BORDER_COLOR: u32 = 0x005a_9ce0;
pub(crate) const IMAGE_BORDER_THICKNESS: i32 = 2;
pub(crate) const IMAGE_BUTTON_BG: u32 = 0x0020_242b;
pub(crate) const IMAGE_BUTTON_RADIUS: i32 = 4;
/// Per-level heading text colors (H1..H6): H1/H2 read as normal text, then blue,
/// yellow, red, and a muted gray for H6 (which is also smaller than body, so color
/// is its main cue). Indexed by level minus one.
pub(crate) const HEADING_COLORS: [u32; 6] = [
    FOREGROUND,
    FOREGROUND,
    0x002e_80f2,
    0x00e5_b567,
    0x00e8_3e3e,
    0x00bf_c6cf,
];
/// A blockquote's gutter bar, and the bullet a list item renders once the cursor
/// leaves its line. The bar is drawn at the text margin, in the empty gutter to
/// the left of the (indented) quote text.
pub(crate) const QUOTE_BAR_COLOR: u32 = 0x003f_444b;
pub(crate) const QUOTE_BAR_WIDTH: i32 = 3;
/// A gutter decoration (a list disc, a task checkbox) sits this fraction of the
/// text ascent short of the first character, so it reads as inset from the words
/// rather than crowding them. Shared by the bullet and the checkbox.
pub(crate) const DECOR_TEXT_GAP: f32 = 0.4;
/// A thematic break (`---`) renders as a thin horizontal line across the content
/// width once the cursor leaves it, in a muted line tone.
pub(crate) const RULE_COLOR: u32 = 0x003f_444b;
pub(crate) const RULE_THICKNESS: i32 = 2;
/// A markdown table's grid: the border/separator lines and a subtle tint behind
/// the header row so it reads as a header.
pub(crate) const TABLE_BORDER_COLOR: u32 = 0x003f_444b;
pub(crate) const TABLE_HEADER_BG: u32 = 0x0028_2c34;
/// Body text pixel size; headings scale up from it (see [`crate::markdown::font_size`]).
pub(crate) const FONT_SIZE: u32 = 14;
/// Inset of the text from the top-left corner (both axes).
pub(crate) const MARGIN: i32 = 40;
/// The caret: a thin vertical bar at the cursor position.
pub(crate) const CARET_COLOR: u32 = 0x00ff_ffff;
pub(crate) const CARET_WIDTH: i32 = 2;

/// The overlay scrollbar down the right edge of a content pane and of the sidebar.
/// It floats over the content (in the text margin, so it never covers a glyph) and
/// is drawn only while it is lit, so the layout never reflows around it.
///
/// The bar rests as a thin indicator and grows to [`SCROLL_WIDE`] when the pointer
/// enters its zone: [`SCROLL_NEAR`] of slack to the left of the lane, so it meets the
/// pointer before the pointer has to aim. The thumb is right-aligned in the lane and
/// grows leftward, staying anchored to the window edge (which is where a maximized
/// window's pointer lands).
pub(crate) const SCROLL_THIN: i32 = 4;
pub(crate) const SCROLL_WIDE: i32 = 10;
pub(crate) const SCROLL_NEAR: i32 = 24;
/// Gap between the lane and the column's right edge.
pub(crate) const SCROLL_INSET: i32 = 3;
/// Shortest the thumb gets, so even a huge document leaves something to grab.
pub(crate) const SCROLL_MIN_THUMB: i32 = 30;
/// The thumb and its trough are drawn as coverage of this tone over whichever
/// surface they sit on (the editor background, the sidebar's darker one), so both
/// columns get a bar with the same weight, and the fade is coverage falling to zero.
/// The thumb lifts under the pointer, the way a tab's buttons do.
pub(crate) const SCROLL_INK: u32 = 0x00cf_d8e3;
pub(crate) const SCROLL_THUMB_COVER: f32 = 0.20;
pub(crate) const SCROLL_THUMB_HOVER_COVER: f32 = 0.36;
pub(crate) const SCROLL_TROUGH_COVER: f32 = 0.05;

/// Top bar (the vault label plus each pane's tab strip) colors and metrics. The
/// bar shares the sidebar's darker tone; the active tab is filled with the editor
/// [`BACKGROUND`] so it reads as connected to the pane below it, while inactive
/// tabs sit muted on the bar. Its height derives from the body line height plus
/// [`TAB_VPAD`] above and below (see [`crate::paint::top_bar_height`]).
pub(crate) const TOP_BAR_BG: u32 = 0x0016_1a1f;
pub(crate) const TAB_ACTIVE_BG: u32 = BACKGROUND;
pub(crate) const TAB_ACTIVE_TEXT: u32 = FOREGROUND;
/// Inactive tab label and the close glyph: the muted sidebar-marker tone.
pub(crate) const TAB_INACTIVE_TEXT: u32 = 0x008c_a0b3;
/// The soft highlight behind a hovered inactive tab (the tree's hover tone).
pub(crate) const TAB_HOVER_BG: u32 = 0x0028_2c34;
/// The unsaved-changes dot shown on a tab that is neither active nor hovered.
pub(crate) const TAB_DIRTY_COLOR: u32 = 0x008c_a0b3;
/// The rounded button that lights up behind a tab's close glyph or the new-tab
/// (+) button under the pointer, and its corner radius.
pub(crate) const TAB_BUTTON_HOVER_BG: u32 = 0x0039_3f47;
pub(crate) const TAB_BUTTON_RADIUS: i32 = 4;
/// Vertical padding above and below a tab label; with the body line height it
/// sets the bar height. Horizontal padding inside a tab, and the gap between the
/// label and its close glyph.
pub(crate) const TAB_VPAD: i32 = 7;
pub(crate) const TAB_PAD_X: i32 = 12;
pub(crate) const TAB_CLOSE_GAP: i32 = 6;
/// The square side of a tab's close-glyph button and of the new-tab (+) button.
pub(crate) const TAB_CLOSE_W: i32 = 18;
pub(crate) const TAB_NEW_W: i32 = 24;
/// Every tab in a strip shares one width, shrinking from `TAB_MAX_W` toward
/// `TAB_MIN_W` as tabs pile up; past the floor the strip scrolls instead of
/// squeezing further. The floor is the tightest a tab reads at: its first letter
/// plus the close box (`TAB_PAD_X` + a glyph + `TAB_CLOSE_GAP` + `TAB_CLOSE_W` +
/// `TAB_PAD_X`, ~48 of chrome around one 14px letter). A longer label hard-cuts at
/// the close box, scissored by the painter, with no ellipsis.
pub(crate) const TAB_MIN_W: i32 = 58;
pub(crate) const TAB_MAX_W: i32 = 220;
/// How far the pointer must travel from a tab press before it counts as a drag to
/// reorder rather than a plain click that selects the tab.
pub(crate) const TAB_DRAG_THRESHOLD: i32 = 5;
/// Left padding of the vault label inside its region.
pub(crate) const VAULT_PAD_X: i32 = 14;

/// File-tree sidebar and splitter colors and metrics.
pub(crate) const TREE_BG: u32 = 0x0016_1a1f;
/// Folder and file names. The sidebar is chrome, not content, so it reads a step
/// below the body text rather than in [`FOREGROUND`].
pub(crate) const TREE_TEXT_COLOR: u32 = 0x008c_a0b3;
/// The disclosure chevron beside a folder, a step below the names again (the
/// same tone as [`MARKER_COLOR`]), so the icon column recedes behind what it
/// points at.
pub(crate) const TREE_MARKER_COLOR: u32 = MARKER_COLOR;
pub(crate) const TREE_HOVER_BG: u32 = 0x0028_2c34;
/// Every divider the chrome draws: the splitters between columns, the line under
/// the top bar, the hairlines between tabs, and the sidebar footer's rule. A
/// separator only has to be found when looked for, so it sits just above the
/// surfaces it parts.
pub(crate) const SPLITTER_COLOR: u32 = 0x002a_2f36;
/// Padding inside the sidebar column: the gap above the first row and below the
/// last, and the margin the row highlight leaves at the left and right edges.
pub(crate) const TREE_PAD: i32 = 8;
/// Horizontal padding inside a row highlight, from its edge to the chevron.
pub(crate) const TREE_ROW_PAD: i32 = 8;
/// Vertical padding added above and below each row's text, so items get more
/// breathing room and a taller hit area than the bare line height.
pub(crate) const TREE_ROW_VPAD: i32 = 3;
/// Corner radius of a row's selection/hover highlight.
pub(crate) const TREE_ROW_RADIUS: i32 = 5;
/// Vertical inset of the highlight within its row, so background shows above and
/// below it and the highlight reads as a pill.
pub(crate) const TREE_ROW_INSET_Y: i32 = 2;
/// Width reserved for a folder's chevron plus the gap to the name, so file and
/// folder names align in one column whether or not a chevron is drawn. Wider
/// than the chevron itself, so the label sits a little clear of the icon.
pub(crate) const TREE_MARKER_W: i32 = 18;
/// Largest a chevron's box grows to; capped below the reserved column so a gap to
/// the label always remains, even at large font sizes.
pub(crate) const TREE_CHEVRON_MAX: i32 = 15;
/// Per-depth indent step for nested rows.
pub(crate) const TREE_INDENT: i32 = 14;

/// The sidebar footer: the strip pinned below the file tree, holding the toggle for
/// whether folders with no notes in them are listed. Its button is a tree-sized row
/// with this padding above and below, and the gap sits between its checkbox and its
/// label. The label reads in the muted chrome tone ([`TREE_MARKER_COLOR`]) and
/// brightens to [`FOREGROUND`] under the pointer, the way a tab's buttons do.
pub(crate) const TREE_FOOTER_VPAD: i32 = 6;
pub(crate) const TREE_FOOTER_GAP: i32 = 8;
