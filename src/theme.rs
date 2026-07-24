use crate::scale::Scale;

/// The side of a disclosure chevron on a row `row_h` pixels tall: a little over
/// half the row, floored so it stays legible and capped so it stays a marker.
/// Shared by the file tree's folder rows and a heading's fold chevron, so the two
/// read as one control wherever they appear.
pub(crate) fn chevron_side(row_h: i32, scale: Scale) -> i32 {
    (row_h * 9 / 16).clamp(scale.px(CHEVRON_MIN), scale.px(CHEVRON_MAX))
}

/// Vertical padding above and below a row's text, so it gets more breathing room
/// and a taller hit area than the bare line height.
pub(crate) const ROW_VPAD: i32 = 3;

/// The height of a row holding one line of `line_height` text, with breathing
/// room above and below. The file tree's rows are built from this, and a heading's
/// fold chevron is sized against the row a line of body text would make, so the
/// tree's marker and the editor's come out the same size.
pub(crate) fn row_height(line_height: i32, scale: Scale) -> i32 {
    (line_height + 2 * scale.px(ROW_VPAD)).max(1)
}

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
/// Space opened above a heading, as a fraction of that heading's own line height,
/// so a section reads as starting rather than as more of what came before. Scaling
/// it off the heading rather than the body gives an H1 more air than an H6, the
/// way a document's own hierarchy does. This is the knob to turn if headings feel
/// crowded or too loose.
pub(crate) const HEADING_SPACE_ABOVE: f32 = 0.4;
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
/// Every unpinned tab in a strip shares one width, shrinking from `TAB_MAX_W`
/// toward `TAB_MIN_W` as tabs pile up; past the floor the strip scrolls instead of
/// squeezing further. The floor is the tightest a tab reads at: its first letter
/// plus the close box (`TAB_PAD_X` + a glyph + `TAB_CLOSE_GAP` + `TAB_CLOSE_W` +
/// `TAB_PAD_X`, ~48 of chrome around one 14px letter). A longer label hard-cuts at
/// the close box, scissored by the painter, with no ellipsis.
pub(crate) const TAB_MIN_W: i32 = 58;
pub(crate) const TAB_MAX_W: i32 = 220;
/// A pinned tab takes this fixed compact width instead of sharing the unpinned
/// tabs' width, so it reads as a small anchor and hands its slack to the tabs being
/// switched between. It is the ~48 of chrome plus room for a few letters of the
/// name (hard-cut at the pin badge, no ellipsis), enough to tell one pin from
/// another where an icon-only tab could not.
pub(crate) const TAB_PINNED_W: i32 = 96;
/// How far the pointer must travel from a press before it counts as a drag to
/// reorder rather than a plain click. A hand is never quite still, so a click must
/// survive a few pixels of it. Shared by the tab strip and the vault rail, each
/// measuring along its own axis.
pub(crate) const DRAG_THRESHOLD: i32 = 5;
/// Left padding of the vault label inside its region.
pub(crate) const VAULT_PAD_X: i32 = 14;

/// The vault rail: the narrow fixed column outside the sidebar, holding one chip
/// per configured vault. It sits a shade below the sidebar's tone, so the chrome
/// reads as deepening toward the window edge, and runs the window's full height
/// (the top bar starts at its right edge rather than crossing it).
pub(crate) const RAIL_BG: u32 = 0x0010_1317;
/// The rail's width, and the padding from its edges to a chip. The chip is the
/// square that is left between them, so these two set its size.
pub(crate) const RAIL_W: i32 = 50;
pub(crate) const RAIL_PAD: i32 = 8;
/// Vertical gap between two chips.
pub(crate) const RAIL_GAP: i32 = 8;
/// Corner radius of a chip: rounded enough to read as a badge rather than a tile.
pub(crate) const RAIL_RADIUS: i32 = 9;
/// A hovered chip lifts toward this ink by this much. The lift is small on purpose:
/// a chip's color is the vault's identity, so the pointer brightens it rather than
/// recoloring it.
pub(crate) const RAIL_HOVER_INK: u32 = 0x00ff_ffff;
pub(crate) const RAIL_HOVER_LIFT: f32 = 0.12;
/// How much closer in a chip's picture is taken under the pointer: the thumbnail's
/// answer to a hover, where the picture pushes forward inside a frame that does not
/// move. A chip that grew instead would jostle the rail, and a picture cannot lift
/// toward white the way a flat fill does, so this is what a chip wearing one says
/// back. Small: enough to notice, not enough to lose what the picture was of.
pub(crate) const ICON_HOVER_ZOOM: f32 = 1.09;
/// The active vault's marker: a short bar pinned to the rail's left edge, centered
/// on its chip. Color and identity stay with the chip; this only says "you are
/// here", the way a tab's fill does for its pane.
pub(crate) const RAIL_ACTIVE_COLOR: u32 = FOREGROUND;
pub(crate) const RAIL_ACTIVE_W: i32 = 3;
/// The active marker's height as a fraction of the chip's, so it reads as an edge
/// tick against the chip rather than a second block beside it.
pub(crate) const RAIL_ACTIVE_SPAN: f32 = 0.55;
/// A chip's initials, inked with whichever of the two contrasts better against its
/// fill (see [`crate::paint::chip_ink`]), so a vault color of any tone stays
/// legible. Every color in [`VAULT_COLORS`] takes the dark one.
pub(crate) const RAIL_INK_LIGHT: u32 = 0x00f5_f7fa;
pub(crate) const RAIL_INK_DARK: u32 = 0x0016_1a1f;
/// A temporary vault's chip (a directory opened from the command line that the
/// config does not name) is drawn as a ring of its color this thick, with the
/// initials in that color rather than on a solid badge. An outline is the shape of
/// the thing: the vault is here for the session, not written down, and it reads as
/// provisional beside the saved chips without spending a second color to say so.
pub(crate) const RAIL_TEMP_RING: i32 = 2;

/// The chip colors, dealt to the vaults by their position in the rail (see
/// [`crate::config::Vault::new`]). One Dark accents, which sit against [`RAIL_BG`]
/// the way the code panel's do against the editor.
pub(crate) const VAULT_COLORS: [u32; 8] = [
    0x005c_9cf6, // blue
    0x0098_c379, // green
    0x00c6_78dd, // purple
    0x00d1_9a66, // orange
    0x00e0_6c75, // red
    0x0056_b6c2, // teal
    0x00e5_b567, // yellow
    0x00f3_8ba8, // pink
];

/// The context menu: the panel a right-click opens (today over a temporary vault's
/// chip). It floats over the chrome in the find bar's panel tone, the editor's one
/// other floating surface, lifted off what it covers by a hairline border a step
/// lighter than the panel. A hovered item takes the accent the editor selects text
/// with, so "this is what the click will act on" reads the same everywhere.
pub(crate) const MENU_BG: u32 = FIND_BG;
pub(crate) const MENU_BORDER: u32 = 0x0045_4a5e;
pub(crate) const MENU_TEXT: u32 = FOREGROUND;
pub(crate) const MENU_HOVER_BG: u32 = SELECTION_COLOR;
/// The panel's corner radius, its border thickness, and the padding between its
/// edge and the items inside it.
pub(crate) const MENU_RADIUS: i32 = 6;
pub(crate) const MENU_BORDER_W: i32 = 1;
pub(crate) const MENU_PAD: i32 = 4;
/// Padding inside one item: above and below its label (with the line height, this
/// sets the row height and so the hit area), and from the row's edge to its text.
pub(crate) const MENU_ITEM_VPAD: i32 = 6;
pub(crate) const MENU_ITEM_PAD_X: i32 = 12;
/// Corner radius of a hovered item's fill, the tab buttons' rounding.
pub(crate) const MENU_ITEM_RADIUS: i32 = TAB_BUTTON_RADIUS;

/// The palette: the searchable panel Ctrl+K opens to jump between vaults. It is the
/// context menu's surface at a larger size, so a panel floating over the window reads
/// the same wherever it came from, and the selected row takes the accent the editor
/// selects text with. A hairline divides the query from the matches, so the field
/// reads as a field rather than as the first row.
pub(crate) const PALETTE_BG: u32 = MENU_BG;
pub(crate) const PALETTE_BORDER: u32 = MENU_BORDER;
pub(crate) const PALETTE_TEXT: u32 = MENU_TEXT;
pub(crate) const PALETTE_SELECTED_BG: u32 = SELECTION_COLOR;
/// The prompt shown in place of an empty query, in the muted chrome tone an inactive
/// tab's label takes: it is there to say what to type, not to be read.
pub(crate) const PALETTE_HINT: u32 = TAB_INACTIVE_TEXT;
/// The fill behind the query where it occurs in a row, so a search row shows what it
/// matched. The gold the find bar marks a match with, so a match reads the same in the
/// palette as in the document.
pub(crate) const PALETTE_MATCH: u32 = FIND_MATCH_COLOR;
/// The panel's width and the gap above it, and its corner radius, border thickness,
/// and the padding between its edge and what is inside it.
pub(crate) const PALETTE_WIDTH: i32 = 520;
pub(crate) const PALETTE_TOP: i32 = 96;
pub(crate) const PALETTE_RADIUS: i32 = 8;
pub(crate) const PALETTE_BORDER_W: i32 = 1;
pub(crate) const PALETTE_PAD: i32 = 6;
/// Padding above and below the query and a row's label. With the line height these
/// set the field's height and a row's (and so its hit area); the query sits in the
/// taller of the two, which is what makes it look like the field it is.
pub(crate) const PALETTE_FIELD_VPAD: i32 = 9;
pub(crate) const PALETTE_ROW_VPAD: i32 = 6;
/// From a row's edge to its label, and the rounding of the selected row's fill.
pub(crate) const PALETTE_ROW_PAD_X: i32 = 10;
pub(crate) const PALETTE_ROW_RADIUS: i32 = MENU_ITEM_RADIUS;

/// The AI text-action preview: a floating panel, like the palette, holding the diff
/// between a selection and the tool's revision. It borrows the palette's panel look
/// and adds the diff's colors: an insertion in green, a deletion struck through in
/// red, over tints that separate the changed words from the unchanged text.
pub(crate) const AI_PANEL_BG: u32 = MENU_BG;
pub(crate) const AI_PANEL_BORDER: u32 = MENU_BORDER;
pub(crate) const AI_TEXT: u32 = FOREGROUND;
/// The subdued tone for the panel's title and the accept/reject hint.
pub(crate) const AI_HINT: u32 = TAB_INACTIVE_TEXT;
/// A failure message, in the same red the find bar shows a bad pattern in.
pub(crate) const AI_ERROR_TEXT: u32 = FIND_ERROR_COLOR;
/// The stronger washes behind the changed words in the diff view: a saturated green
/// behind an added word, a saturated red behind a removed one, drawn over the faint
/// full-line wash so the edited words stand out clearly and the two colors read apart
/// at a glance rather than blending. The text keeps its normal color; the wash carries
/// the meaning, the way a selection highlight does.
pub(crate) const AI_INSERT_BG: u32 = 0x002f_7a4a;
pub(crate) const AI_DELETE_BG: u32 = 0x008c_3a45;
/// The panel's width and the gap above it; it is wider than the palette because it
/// holds wrapped prose, not one-line rows.
pub(crate) const AI_PANEL_WIDTH: i32 = 640;
pub(crate) const AI_PANEL_TOP: i32 = 84;
pub(crate) const AI_PANEL_RADIUS: i32 = PALETTE_RADIUS;
pub(crate) const AI_PANEL_BORDER_W: i32 = 1;
pub(crate) const AI_PANEL_PAD: i32 = 14;
/// Padding above and below one wrapped diff line, and the most lines the panel shows
/// before it notes the rest are hidden.
pub(crate) const AI_LINE_VPAD: i32 = 3;
pub(crate) const AI_MAX_LINES: usize = 18;

/// The side-by-side diff view (the tab a Ready AI action opens into): two columns, the
/// old text on the left and the new on the right, aligned line for line so a change
/// reads across the divider, sharing one scroll.
///
/// A changed line carries two tints: a faint wash over the line (DIFF_LINE_*) inset a
/// little from the column edges, and, over it, the stronger word wash (AI_INSERT_BG /
/// AI_DELETE_BG) on just the words that changed, the way GitHub layers a line and its
/// intraline diff. A thin accent bar (DIFF_*_ACCENT) runs down the band's leading edge
/// and the line number is drawn in that accent. Where a change is one-sided (a line
/// added or removed, with no counterpart), the empty half is filled with diagonal
/// hatching (DIFF_HATCH) so it reads as absent rather than blank.
///
/// The text inset inside each column, narrower than the document's [`MARGIN`] because a
/// column is half the width. The gutter's padding each side of the number, the inset of
/// a changed line's band from the text edges (kept below [`DIFF_MARGIN`] so the word
/// washes stay inside it), and the divider between the two columns.
pub(crate) const DIFF_MARGIN: i32 = 16;
pub(crate) const DIFF_GUTTER_PAD: i32 = 5;
pub(crate) const DIFF_LINE_PAD: i32 = 6;
pub(crate) const DIFF_DIVIDER_W: i32 = 1;
/// The divider between the columns (the same tone every chrome divider uses), the dim
/// gutter text, and the subtle hairline parting a gutter from its text.
pub(crate) const DIFF_DIVIDER: u32 = SPLITTER_COLOR;
pub(crate) const DIFF_GUTTER_TEXT: u32 = TAB_INACTIVE_TEXT;
pub(crate) const DIFF_GUTTER_BORDER: u32 = 0x0024_2a31;
/// The faint full-line washes behind a changed line: a step up from the background
/// toward the stronger word washes above them, so the whole changed line reads as
/// changed while its edited words stand out within it.
pub(crate) const DIFF_LINE_INSERT_BG: u32 = 0x0022_3327;
pub(crate) const DIFF_LINE_DELETE_BG: u32 = 0x0033_2429;
/// The thin accent bar down a changed line's band edge, also the tone its line number is
/// drawn in: a saturated green for an added line, red for a removed one, so the change is
/// legible at the margin.
pub(crate) const DIFF_INSERT_ACCENT: u32 = 0x0098_c379;
pub(crate) const DIFF_DELETE_ACCENT: u32 = 0x00e0_6c75;
/// The empty half of a one-sided change (a line present on one side, absent on the
/// other): its text area is diagonal hatching, thin 45-degree strokes `PERIOD` device
/// pixels apart and `STROKE` wide, over the background; its gutter is a flat solid of the
/// same tone (see `push_diff_bands`), so the number column reads as a clean edge and only
/// the text area carries the "nothing here" texture.
pub(crate) const DIFF_HATCH: u32 = 0x0032_3d48;
pub(crate) const DIFF_HATCH_PERIOD: i32 = 6;
pub(crate) const DIFF_HATCH_STROKE: i32 = 1;

/// The full-window picture view: what a double-clicked image opens into. The
/// backdrop covers the whole window rather than dimming it, since the point is the
/// picture and everything else is in the way of it; near-black, so a picture's own
/// edges are where it ends. The padding is the clear space left around it.
pub(crate) const IMAGE_VIEW_BG: u32 = 0x0007_090b;
pub(crate) const IMAGE_VIEW_PAD: i32 = 28;

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
/// Corner radius of a row's selection/hover highlight.
pub(crate) const TREE_ROW_RADIUS: i32 = 5;
/// Vertical inset of the highlight within its row, so background shows above and
/// below it and the highlight reads as a pill.
pub(crate) const TREE_ROW_INSET_Y: i32 = 2;
/// Width reserved for a folder's chevron plus the gap to the name, so file and
/// folder names align in one column whether or not a chevron is drawn. Wider
/// than the chevron itself, so the label sits a little clear of the icon.
pub(crate) const TREE_MARKER_W: i32 = 18;
/// Smallest and largest a disclosure chevron's box grows to. The cap sits below
/// the tree's reserved column so a gap to the label always remains, even at large
/// font sizes.
pub(crate) const CHEVRON_MIN: i32 = 8;
pub(crate) const CHEVRON_MAX: i32 = 15;
/// Per-depth indent step for nested rows.
pub(crate) const TREE_INDENT: i32 = 14;

/// The sidebar footer: the strip pinned below the file tree, holding the toggle for
/// whether folders with no notes in them are listed. Its button is a tree-sized row
/// with this padding above and below, and the gap sits between its checkbox and its
/// label. The label reads in the muted chrome tone ([`TREE_MARKER_COLOR`]) and
/// brightens to [`FOREGROUND`] under the pointer, the way a tab's buttons do.
pub(crate) const TREE_FOOTER_VPAD: i32 = 6;
pub(crate) const TREE_FOOTER_GAP: i32 = 8;
