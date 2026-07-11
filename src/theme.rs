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
/// Corner radius of a tab's fill (rounded at the top, square where it meets the
/// pane).
pub(crate) const TAB_RADIUS: i32 = 6;
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
/// A tab's minimum and maximum width; a longer label truncates with an ellipsis.
pub(crate) const TAB_MIN_W: i32 = 90;
pub(crate) const TAB_MAX_W: i32 = 220;
/// How far the pointer must travel from a tab press before it counts as a drag to
/// reorder rather than a plain click that selects the tab.
pub(crate) const TAB_DRAG_THRESHOLD: i32 = 5;
/// Left padding of the vault label inside its region.
pub(crate) const VAULT_PAD_X: i32 = 14;

/// File-tree sidebar and splitter colors and metrics.
pub(crate) const TREE_BG: u32 = 0x0016_1a1f;
/// The disclosure chevron beside a folder. Folder and file names share
/// [`FOREGROUND`]; only this marker is tinted, so it reads as chrome.
pub(crate) const TREE_MARKER_COLOR: u32 = 0x008c_a0b3;
pub(crate) const TREE_HOVER_BG: u32 = 0x0028_2c34;
pub(crate) const SPLITTER_COLOR: u32 = 0x003f_444b;
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
