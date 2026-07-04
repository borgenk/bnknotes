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
/// Inline code: a tinted background panel and a distinct foreground, so a code
/// span stands out from prose the way it does in a rendered document.
pub(crate) const CODE_COLOR: u32 = 0x00f3_8ba8;
pub(crate) const CODE_BG: u32 = 0x0028_2c34;
/// Corner radius of a fenced code block's panel: a very subtle rounding, no border.
pub(crate) const CODE_RADIUS: i32 = 4;
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
pub(crate) const BULLET: &str = "•";
/// A thematic break (`---`) renders as a thin horizontal line across the content
/// width once the cursor leaves it, in a muted line tone.
pub(crate) const RULE_COLOR: u32 = 0x003f_444b;
pub(crate) const RULE_THICKNESS: i32 = 2;
/// Body text pixel size; headings scale up from it (see [`crate::markdown::font_size`]).
pub(crate) const FONT_SIZE: u32 = 14;
/// Inset of the text from the top-left corner (both axes).
pub(crate) const MARGIN: i32 = 40;
/// The caret: a thin vertical bar at the cursor position.
pub(crate) const CARET_COLOR: u32 = 0x00ff_ffff;
pub(crate) const CARET_WIDTH: i32 = 2;

/// File-tree sidebar and splitter colors and metrics.
pub(crate) const TREE_BG: u32 = 0x0016_1a1f;
pub(crate) const TREE_DIR_COLOR: u32 = 0x008c_a0b3;
pub(crate) const TREE_HOVER_BG: u32 = 0x0028_2c34;
pub(crate) const SPLITTER_COLOR: u32 = 0x003f_444b;
/// Inset for tree rows from the column edges, and the per-depth indent step.
pub(crate) const TREE_PAD: i32 = 6;
pub(crate) const TREE_INDENT: i32 = 14;
