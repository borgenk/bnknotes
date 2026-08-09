//! Turning the flat text buffer into positioned visual lines, and mapping
//! between byte offsets and pixels.
//!
//! The buffer is one string with `'\n'` separators. This pass splits it on those
//! newlines into logical lines, then soft-wraps each to the available width with
//! a greedy word wrap (break after the last space that fits; an over-long word
//! breaks after a link's last URL separator, else mid-word at a grapheme
//! boundary). The result is a flat list of
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
//! heading's `#`s, a list/quote marker, a code span's backticks, emphasis
//! `*`/`_`). Markers live-preview: on a line the cursor has left they collapse
//! (`Run::visible` is false, zero width), drawing only the styled text; they
//! reappear when the cursor returns. The exception is a list or quote block
//! marker, which persists (it is the only sign the line is a list or quote).
//!
//! Wrapping measures what a row draws, so a collapsed marker takes no width and a
//! link's hidden destination does not blow a hole in the paragraph. Revealing a
//! line can therefore re-wrap it: the [`LineIndex`] measures every line collapsed
//! and records what its reveal would cost (`wrap_extra`), which [`Adjust`] adds
//! back for the lines the cursor or selection touches. A table cell is the
//! exception and wraps to its source, since its column width was fitted from the
//! source too.
//!
//! A list or quote item also hangs its wrapped continuations under the text: each
//! visual line past the first is inset by the marker's width (`VisualLine::indent`)
//! so the column aligns. The raw marker gives way to a [`Gutter`] decoration,
//! always the same width as the source marker so the text never shifts: a bullet
//! point `•` for a list and a drawn checkbox for a task item stand in while you
//! edit the label, yielding to the raw marker only where the caret or selection
//! reaches it ([`active_reveals_marker`]), while a quote's bar yields to its
//! editable `>` on the cursor's own line.
//!
//! Fenced code blocks are the one construct whose classification crosses line
//! boundaries: a ``` line opens a block and a later ``` closes it, and every line
//! between is code, not markdown. So `build` carries a fence flag down the loop,
//! and a line inside a fence (`VisualLine::code_block`) skips the inline pass
//! entirely, rendering in the distinct code face on a full-width padded panel.
//! The ``` fence lines hide their backticks off the cursor's line, like a heading
//! hides its `#`, so an unfocused block reads without visible fences; the panel
//! still draws those rows, leaving blank top and bottom padding.

use std::ops::Range;

use crate::fold::FoldMap;
use crate::markdown;
use crate::platform::freetype::{CodeInk, Face, FaceKey, FontStyle, Fonts, Metrics};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::platform::shape;
use crate::render::image::ImageStore;
use crate::rope::Rope;
use crate::scale::Scale;

/// Extra highlight width drawn past a line's text when the selection includes
/// that line's trailing newline, hinting the break itself is selected.
const NEWLINE_PAD: i32 = 8;

/// Horizontal padding inside a fenced code block's panel: the code text (and the
/// ``` fences) is inset this many pixels from the panel's left edge, and wrapping
/// keeps the same gap on the right, so the tinted background frames the code
/// rather than butting against it.
const CODE_PAD_X: i32 = 10;

/// A small left indent for list items (bullet and ordered), so a list sits inset
/// from the body margin. It applies to every visual line of the item; wrapped
/// continuations add the marker's width on top for the hanging indent.
const LIST_INDENT: i32 = 14;

/// Extra left indent per nesting level, on top of [`LIST_INDENT`]. The source's two
/// leading spaces per level advance only a few pixels, too little to read as nesting,
/// so each depth steps in by this fixed amount as well and the levels separate
/// clearly. Depth 0 (a top-level item) adds nothing.
const LIST_NEST_INDENT: i32 = 18;

/// Vertical breathing room above and below a rendered block image, so it does not
/// butt against the lines around it. Part of the image line's box height.
const IMAGE_PAD_Y: i32 = 8;

/// Extra pixels around a drawn task checkbox that still count as clicking it,
/// so the small square is comfortable to hit.
const CHECKBOX_HIT_PAD: i32 = 3;

/// The same slack around a heading's fold chevron. It is more generous: the
/// chevron is smaller, and it sits alone in the margin where a stray hit costs
/// nothing.
const FOLD_HIT_PAD: i32 = 6;

/// The gap the fold chevron keeps between itself and the text column. Small
/// enough that it plus the chevron stay well inside [`crate::theme::MARGIN`], so
/// the chevron always has margin to hang in.
const FOLD_CHEVRON_GAP: i32 = 8;

/// Horizontal padding between a table cell's border and its text, on each side of
/// the content; vertical padding above and below a row's text; and the width of a
/// grid border line. The cell content width plus twice the x padding plus the
/// borders is the column's drawn width.
const TABLE_CELL_PAD_X: i32 = 8;
pub(crate) const TABLE_CELL_PAD_Y: i32 = 4;
pub(crate) const TABLE_BORDER: i32 = 1;

/// The narrowest a table column's content is squeezed to when the natural widths
/// overflow the available width, so a column never collapses to nothing.
const TABLE_MIN_COL: i32 = 24;

/// Thickness of the add-row strip below a table and the add-column strip to its
/// right: the hover targets that grow the table by a row or a column.
const TABLE_ADD_THICKNESS: i32 = 16;

/// The selected-image toggle button: a rounded panel this wide and tall (wide
/// enough for the `</>` glyph), inset this far from the picture's top-right corner.
/// Shared by the painter (to draw it) and hit-testing (to catch clicks), so the
/// two always agree.
pub(crate) const IMAGE_BUTTON_W: i32 = 34;
pub(crate) const IMAGE_BUTTON_H: i32 = 22;
pub(crate) const IMAGE_BUTTON_MARGIN: i32 = 6;

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
    /// The line's inline runs, as a half-open range into the [`Layout`]'s shared
    /// `runs` pool (resolve with [`Layout::run_slice`]). In display order, tiling
    /// `[start, end)`; an empty line has an empty range. Each run is a styled
    /// stretch of text or a delimiter marker; a hidden marker (`!visible`) is
    /// collapsed to zero width. Flattening every line's runs into one pool per
    /// build means laying out a line appends to it rather than allocating a `Vec`
    /// per visual row.
    pub runs: Range<u32>,
    /// Pixels the line's content (its runs, caret, and selection) is inset past
    /// `left`, for the hanging indent of a wrapped list or quote item: `0` on the
    /// marker-bearing first line (whose marker run fills the gutter inline) and on
    /// non-list lines; the marker's width on continuation lines, so wrapped text
    /// aligns under the first line's text rather than under the marker.
    pub indent: i32,
    /// How this line's left gutter is decorated beyond its runs once the cursor
    /// leaves it: a bullet glyph for a list, a bar for a quote. [`Gutter::None`]
    /// on ordinary lines and on the cursor's own line (which shows its raw marker).
    pub gutter: Gutter,
    /// True for a line inside a fenced code block (the ``` fences and the code
    /// between them). It is drawn on a full-width tinted panel and its text is
    /// not parsed as inline markdown.
    pub code_block: bool,
    /// The ATX heading level (1..=6) if this line is a heading, else `None`. The
    /// app colors heading text by level, and the pixel `size` already carries the
    /// heading's scale.
    pub heading: Option<u8>,
    /// Whether this heading's section is folded, on the first visual row of a
    /// heading that owns one. `None` on every other row, and on a heading with an
    /// empty section: the margin draws a chevron exactly where this is `Some`, so a
    /// heading with nothing under it offers no fold to click.
    pub fold: Option<bool>,
    /// `Some(checked)` when this line belongs to a task-list item, on every
    /// visual line the item wraps into. A checked item's text renders dim; the
    /// checkbox itself is the first line's [`Gutter::Checkbox`]. `None` off task
    /// items (and on an over-nested one, which renders as plain body text).
    pub task: Option<bool>,
    /// True to draw a horizontal rule for this line: a thematic break (`---`)
    /// whose cursor has left, so its raw markers are collapsed. While the cursor is
    /// on the line this is false and the editable `---` shows instead.
    pub rule: bool,
    /// Set when this line renders a `![alt](path)` image as a picture box. The
    /// picture always shows; its editable source, when revealed, is a separate
    /// text line stacked *above* this one (see [`Layout::build_with_images`]). The
    /// box is taller than a text line, so the few places that measure a line's
    /// vertical extent consult this rather than the font metrics. A widget line is
    /// never a keyboard caret target (the source line above it is).
    pub image: Option<ImageBox>,
    /// Set when this line renders one grid row of a markdown table (a header or a
    /// body row). Like [`Self::image`], it is a widget line: taller than text, with
    /// its own internal cell layout. The caret sits in its cells (via the grid model
    /// in [`crate::table`]), and on the cursor's row the cells reveal their raw
    /// markers. A table's delimiter row emits no visual line at all. `None` on every
    /// other line.
    pub table: Option<TableRow>,
}

impl VisualLine {
    /// A picture-box widget line for a standalone `![alt](path)`: no runs, gutter,
    /// or markup, just the image at the given `baseline`. Used both for the
    /// picture-only case and the picture below a revealed source line, which
    /// differ only in `baseline` and whether they own a hard break.
    fn image_widget(
        start: usize,
        end: usize,
        baseline: i32,
        hard: bool,
        size: u32,
        image: ImageBox,
    ) -> Self {
        Self {
            start,
            end,
            baseline,
            hard,
            size,
            runs: 0..0,
            indent: 0,
            gutter: Gutter::None,
            code_block: false,
            heading: None,
            fold: None,
            task: None,
            rule: false,
            image: Some(image),
            table: None,
        }
    }
}

/// A block image laid out as a picture box for a logical line's `![alt](path)`.
/// The path is kept as a byte range into the buffer (sliced at paint time to look
/// the decoded image up in the [`ImageStore`]) rather than an owned string,
/// matching how [`Run`] carries text. The size is already scaled to fit the
/// content column, aspect preserved.
#[derive(Clone, Copy)]
pub struct ImageBox {
    /// Byte range of the destination path within the buffer.
    pub path_start: usize,
    pub path_end: usize,
    /// Display size in pixels (the painted rectangle, padding excluded).
    pub width: i32,
    pub height: i32,
    /// The vertical breathing room above and below the picture, in device pixels
    /// ([`IMAGE_PAD_Y`] scaled by the display). Stored so [`Self::box_height`] needs
    /// no scale at paint time.
    pad_y: i32,
    /// True while the user has this image selected: the painter draws a border
    /// and the corner toggle button, and `Ctrl+C` copies the image.
    pub selected: bool,
}

impl ImageBox {
    /// The full height of the image line's box: the image plus the vertical
    /// padding above and below it. This is the line's baseline-to-next-box advance
    /// and its clickable/cull height, standing in for a text line's line-height.
    /// The painter centers the image in this box and culls on it.
    pub(crate) fn box_height(&self) -> i32 {
        self.height + 2 * self.pad_y
    }
}

/// One grid row of a markdown table, laid out as a widget line. It owns its cells
/// and the table's shared column geometry, so the painter reads everything it
/// needs off the row without re-measuring. Every row of one table carries the same
/// `columns`, so a row can draw its own separators and the block's outer border.
#[derive(Clone)]
pub struct TableRow {
    /// True for the header row (the line above the delimiter), which draws bold and
    /// sits above the header rule. False for a body row.
    pub header: bool,
    /// True on the table's first drawn row (the header), so the painter draws the
    /// table's top edge; every row draws its own bottom edge. A table split across a
    /// window edge may have `first` unset on the rows in view.
    pub first: bool,
    /// Left edge and content width of each column, relative to the row's left
    /// (`Layout::left` plus the row's `indent`). Length matches `cells`.
    pub columns: Vec<Column>,
    /// One entry per column, in column order.
    pub cells: Vec<TableCell>,
    /// The x of every vertical grid line, relative to the row's left: the left
    /// outer edge, each column separator, and the right outer edge. Precomputed so
    /// the painter draws the borders without the layout's padding constants.
    pub borders: Vec<i32>,
    /// The table's total width in pixels (its right outer edge).
    pub width: i32,
    /// Full height of the row's box in pixels (the tallest cell plus vertical cell
    /// padding), standing in for a text line's line-height.
    pub height: i32,
}

/// A hover affordance that grows a table: an add-row strip just below its last
/// row, or an add-column strip just off its right edge. Carried from a hover/click
/// hit-test to the painter (which draws it) and the editor (which applies it).
#[derive(Clone, Copy)]
pub struct TableButton {
    pub kind: TableButtonKind,
    /// The strip's rectangle in content space (also the click target).
    pub rect: Rect,
    /// The table to grow, as its logical-line extent.
    pub block: crate::table::Block,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TableButtonKind {
    AddRow,
    AddColumn,
}

/// A table column's horizontal geometry, shared by every row of the table.
#[derive(Clone, Copy)]
pub struct Column {
    /// Left x of the cell's content, relative to the row's left.
    pub x: i32,
    /// Content width the cell text wraps within.
    pub width: i32,
    pub align: markdown::Align,
}

/// One cell of a table row: its trimmed content byte range (absolute) and its
/// wrapped text lines, each a run of styled spans. Runs carry absolute byte ranges
/// into the buffer, so a cell maps back to its source; the range lets the caret and
/// click hit-testing land in a cell even when it is empty (no runs).
#[derive(Clone)]
pub struct TableCell {
    pub start: usize,
    pub end: usize,
    pub lines: Vec<CellLine>,
}

/// One wrapped line within a table cell: the runs to draw and their pixel width
/// (so the painter can right-align or center without re-measuring).
#[derive(Clone)]
pub struct CellLine {
    pub runs: Vec<Run>,
    pub width: i32,
}

impl TableRow {
    /// The full box height of the row, standing in for a text line's line-height in
    /// the few places that measure a line's vertical extent.
    pub(crate) fn box_height(&self) -> i32 {
        self.height
    }
}

/// The per-image UI state the layout needs: which image (by its logical line's
/// start offset) is selected, and which has its raw source pinned open by the
/// corner button. Both are `None` by default (no image being interacted with).
/// Offsets are valid for the document version being laid out; the app clears them
/// on any edit, so they never go stale.
#[derive(Clone, Copy, Default)]
pub struct ImageUi {
    pub selected: Option<usize>,
    pub raw: Option<usize>,
    /// A selection drag is in progress. Image sources then hold their collapsed
    /// picture-only height even where the growing selection would reveal one, so
    /// the document geometry (and the pointer-to-offset mapping) stays put while
    /// dragging. Text markers still reveal, since revealing them never reflows.
    pub dragging: bool,
}

/// The live-preview decoration of a visual line's left gutter, chosen once the
/// cursor leaves the line. On the cursor's own line the raw marker shows so it
/// stays editable, so this is [`Gutter::None`] there. A decoration always occupies
/// the same width as the source marker, so toggling it never shifts the text (the
/// no-reflow invariant the whole module preserves).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gutter {
    /// No decoration: body and heading lines, ordered items (their number stays
    /// put), a quote line the cursor is on, and a bullet or task line whose caret
    /// or selection has reached its marker (so the raw marker shows for editing).
    None,
    /// Paint a filled disc in place of a `-`/`+`/`*` marker's glyph. Gives way to
    /// the raw marker (`Gutter::None`) once the caret or selection reaches it.
    Bullet,
    /// Draw a checkbox (built from fills, no glyph dependency) in place of a
    /// task item's `- [ ]`/`- [x]` marker. Gives way to the raw marker for editing
    /// once the caret or selection reaches it (see [`active_reveals_marker`]). The
    /// box geometry is [`Layout::checkbox_rect`], shared with click hit-testing.
    Checkbox { checked: bool },
    /// Draw a blockquote bar down the gutter. `marker` is true on the first line,
    /// where the bar stands in for the `>` marker (whose glyphs are not drawn),
    /// and false on wrapped continuations, which have no marker run of their own.
    QuoteBar { marker: bool },
}

/// One inline run of a visual line: an absolute byte range, what it is, whether it
/// is drawn, and which sides of it carry an inline code span's pill pad. A marker
/// run is hidden (`visible == false`) on a line the cursor has left, the
/// live-preview collapse; text runs are always visible.
#[derive(Clone, Copy)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    pub kind: markdown::SpanKind,
    pub visible: bool,
    pub pads: PillPads,
}

/// Which sides of a run reserve an inline code span's pill padding. The pad is
/// horizontal space outside the span's delimiters: with the backticks collapsed the
/// code run carries both sides, and on the cursor's line each revealed backtick takes
/// the outer one and the code run drops that side. The span reserves the same total
/// width either way, so a reveal moves the pad without reflowing the line.
///
/// Every walk (advance, hit-test, pill, draw pen) reads these flags, which is what
/// keeps the caret and a click on the same pixel as the glyph the painter put there.
///
/// ```text
///   collapsed:  [pad] c o d e [pad]
///   revealed:   [pad] ` c o d e ` [pad]
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PillPads {
    pub lead: bool,
    pub trail: bool,
}

impl PillPads {
    /// The leading and trailing pad in pixels at `scale`.
    pub(crate) fn px(self, scale: Scale) -> (i32, i32) {
        let pad = scale.px(crate::theme::CODE_PAD_X);
        (
            if self.lead { pad } else { 0 },
            if self.trail { pad } else { 0 },
        )
    }
}

/// What a walk over a line's runs needs to lay them the way the painter did: the
/// size and heading flag that pick each run's face.
#[derive(Clone, Copy)]
struct RunStyle {
    size: u32,
    heading: bool,
}

impl RunStyle {
    fn of(line: &VisualLine) -> Self {
        Self {
            size: line.size,
            heading: line.heading.is_some(),
        }
    }

    /// A table cell draws prose at the row's size, bold in a header row.
    fn of_cell(line: &VisualLine, t: &TableRow) -> Self {
        Self {
            size: line.size,
            heading: t.header,
        }
    }
}

/// Map a span/run kind to the face style it renders in within the prose family.
/// Markers use the regular face; only bold/italic emphasis selects a variant.
/// Code is not handled here (it draws in the separate code family, see
/// [`run_face`]); a stray code kind falls back to regular.
fn font_style(kind: markdown::SpanKind) -> FontStyle {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Bold) => FontStyle::Bold,
        markdown::SpanKind::Text(markdown::Emphasis::Italic) => FontStyle::Italic,
        markdown::SpanKind::Text(markdown::Emphasis::BoldItalic) => FontStyle::BoldItalic,
        _ => FontStyle::Regular,
    }
}

/// The face a run or span of `kind` is measured and drawn in: the distinct code
/// family for an inline-code or fenced-code run, otherwise the prose family in
/// the kind's emphasis style. On a heading line the prose text is emboldened (a
/// heading renders bold), composing with any inline emphasis; the `#` marker and
/// inline code keep their own face. The single source of truth so layout's
/// measuring and the app's drawing pick the same face and never disagree on widths.
pub(crate) fn run_face(fonts: &Fonts, size: u32, kind: markdown::SpanKind, heading: bool) -> &Face {
    fonts.face_for(face_key(size, kind, heading))
}

/// The pill padding an inline code span reserves on each side of itself: the tinted
/// panel extends [`crate::theme::CODE_PAD_X`] beyond the outermost glyphs it frames,
/// and that is real horizontal space, so the text around the span clears the pill
/// instead of sitting under it. Which run carries each side is [`PillPads`]; a
/// measure that only needs a span's total counts this twice. Zero for a fenced line
/// (its own full-width panel, drawn by the app) and for anything but inline code. The
/// local [`CODE_PAD_X`] here is the unrelated fenced-block inset.
pub(crate) fn inline_code_pad(kind: markdown::SpanKind, code_block: bool, scale: Scale) -> i32 {
    if !code_block && matches!(kind, markdown::SpanKind::Text(markdown::Emphasis::Code)) {
        scale.px(crate::theme::CODE_PAD_X)
    } else {
        0
    }
}

/// The drawn width of a code span's flanking backtick delimiter, for growing the
/// pill in [`Layout::code_rects`] to enclose the revealed delimiters. `after`
/// picks the run just after the code run at `code` rather than the one before.
/// Zero unless that neighbor is the span's revealed delimiter, which is exactly
/// the marker holding the pill's outer pad on that side: off the cursor's line the
/// delimiters are collapsed and the pill hugs the content.
fn backtick_marker_width(
    runs: &[Run],
    code: usize,
    after: bool,
    fonts: &Fonts,
    size: u32,
    heading: bool,
    rope: &Rope,
) -> f32 {
    let idx = if after {
        code + 1
    } else {
        code.wrapping_sub(1)
    };
    let Some(run) = runs.get(idx) else {
        return 0.0;
    };
    let delimiter = matches!(run.kind, markdown::SpanKind::Marker)
        && if after { run.pads.trail } else { run.pads.lead };
    if !delimiter {
        return 0.0;
    }
    let text = rope.slice(run.start..run.end).as_cow();
    shape::text_advance(run_face(fonts, size, run.kind, heading), &text)
}

/// True when a marker run begins with a backtick: an inline code delimiter or a
/// fence line (```rust and the like). Such markers recolor to
/// [`crate::theme::CODE_MARKER_COLOR`] so the delimiters stay legible against the
/// code panel's darker tint, while the other muted markers keep [`MARKER_COLOR`].
pub(crate) fn is_code_marker(rope: &Rope, run: &Run) -> bool {
    matches!(run.kind, markdown::SpanKind::Marker) && rope.byte(run.start) == Some(b'`')
}

/// The [`FaceKey`] a run of `kind` at `size` draws in, the borrow-free twin of
/// [`run_face`]: the display list records this so a painted run can resolve its
/// face later without holding a borrow of [`Fonts`]. Same decision as `run_face`,
/// so the two never disagree on which face (hence which width) a run uses. The two
/// code kinds pick the same face and part on ink alone: a fenced block's token is
/// [`CodeInk::Block`], an inline span is [`CodeInk::Inline`].
pub(crate) fn face_key(size: u32, kind: markdown::SpanKind, heading: bool) -> FaceKey {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Code) => FaceKey::Code {
            size,
            ink: CodeInk::Inline,
        },
        markdown::SpanKind::Code(_) => FaceKey::Code {
            size,
            ink: CodeInk::Block,
        },
        markdown::SpanKind::Text(_) if heading => FaceKey::Prose {
            size,
            style: embolden(font_style(kind)),
        },
        other => FaceKey::Prose {
            size,
            style: font_style(other),
        },
    }
}

/// Add weight to a prose style, so heading text renders bold and bold-italic
/// heading text keeps its slant. Already-bold styles are unchanged.
fn embolden(style: FontStyle) -> FontStyle {
    match style {
        FontStyle::Regular | FontStyle::Bold => FontStyle::Bold,
        FontStyle::Italic | FontStyle::BoldItalic => FontStyle::BoldItalic,
    }
}

/// The span kind covering byte `off`. `spans` tile `[0, len)` in order, so the
/// first span ending past `off` is the one holding it; a byte past the end
/// (never asked for in practice) falls back to regular text.
fn kind_at(spans: &[markdown::Span], off: usize) -> markdown::SpanKind {
    spans
        .iter()
        .find(|s| off < s.end)
        .map_or(markdown::SpanKind::Text(markdown::Emphasis::Regular), |s| {
            s.kind
        })
}

#[derive(Clone)]
pub struct Layout {
    lines: Vec<VisualLine>,
    /// Every visual line's runs, flattened into one pool; each [`VisualLine`]'s
    /// `runs` range indexes into this. One allocation per build instead of one
    /// per visual row.
    runs: Vec<Run>,
    left: i32,
    /// Total height of the text block in pixels (the sum of every visual line's
    /// line height), for the scroll extent.
    text_height: i32,
    /// The display scale this layout was built at, so its geometry methods
    /// (`checkbox_rect`, `code_rects`, table hit-testing) scale their pixel
    /// constants to match. Identity at [`Scale::ONE`].
    scale: Scale,
}

/// Shared layout/index inputs for one width and reveal state. Built once by a
/// caller (a pane, the perf harness) and threaded through the public entry points
/// so they take one context instead of re-exploding the same argument bundle.
#[derive(Clone, Copy)]
pub struct LayoutCtx<'a> {
    fonts: &'a Fonts,
    rope: &'a Rope,
    wrap_width: i32,
    body_size: u32,
    active: (usize, usize),
    images: &'a ImageStore,
    image_ui: ImageUi,
    /// The sections folded away right now. A build skips their lines outright; the
    /// index paths hold this empty so base geometry stays fold-free.
    folds: &'a FoldMap,
    /// The display scale, so the layout's pixel constants (indents, paddings,
    /// table borders) land at the right device size. Identity at [`Scale::ONE`].
    scale: Scale,
}

impl<'a> LayoutCtx<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fonts: &'a Fonts,
        rope: &'a Rope,
        wrap_width: i32,
        body_size: u32,
        active: (usize, usize),
        images: &'a ImageStore,
        image_ui: ImageUi,
        folds: &'a FoldMap,
        scale: Scale,
    ) -> Self {
        Self {
            fonts,
            rope,
            wrap_width,
            body_size,
            active,
            images,
            image_ui,
            folds,
            scale,
        }
    }

    fn with_active(self, active: (usize, usize)) -> Self {
        Self { active, ..self }
    }

    /// A ctx for base-geometry measurement (the [`LineIndex`] paths): reveal forced
    /// off ([`NO_REVEAL`], no image UI) and folds forced empty, so the geometry it
    /// feeds stays a pure function of `(version, width)` by construction rather than
    /// by each caller remembering to pass the neutral inputs.
    pub fn for_index(
        fonts: &'a Fonts,
        rope: &'a Rope,
        wrap_width: i32,
        body_size: u32,
        images: &'a ImageStore,
        scale: Scale,
    ) -> Self {
        Self::new(
            fonts,
            rope,
            wrap_width,
            body_size,
            NO_REVEAL,
            images,
            ImageUi::default(),
            FoldMap::NONE,
            scale,
        )
    }
}

#[derive(Clone, Copy)]
struct LogicalLine {
    start: usize,
    end: usize,
    has_newline: bool,
}

impl LogicalLine {
    fn at(rope: &Rope, line: usize, line_count: usize) -> Self {
        let start = rope.line_to_byte(line);
        let has_newline = line + 1 < line_count;
        let end = if has_newline {
            rope.line_to_byte(line + 1) - 1
        } else {
            rope.line_to_byte(line + 1)
        };
        Self {
            start,
            end,
            has_newline,
        }
    }
}

/// An open code fence carried across the lines of its block: how many backticks
/// opened it (a close needs at least as many) and the language its info string
/// selected, so every line inside is highlighted the same way.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Fence {
    ticks: usize,
    lang: crate::syntax::Lang,
}

impl Fence {
    /// Pack the fence into the line index's `u16` carry: ticks in the low byte,
    /// language in the high byte. Ticks are never zero inside a fence, so a zero
    /// packed value cleanly means "no open fence".
    fn packed(self) -> u16 {
        (self.ticks as u16 & 0xff) | ((self.lang as u16) << 8)
    }

    /// The fence a non-zero packed carry value describes.
    fn unpack(packed: u16) -> Option<Self> {
        (packed != 0).then(|| Fence {
            ticks: (packed & 0xff) as usize,
            lang: crate::syntax::Lang::from_u8((packed >> 8) as u8),
        })
    }
}

/// Cross-line parser state for constructs that carry over logical lines.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct LineCarry {
    fence: Option<Fence>,
    list_depth: Option<usize>,
}

impl LineCarry {
    fn new(fence: Option<Fence>, list_depth: Option<usize>) -> Self {
        Self { fence, list_depth }
    }

    /// The carry in the form [`LineIndex::entry_packed`] hands back, so a build
    /// that jumps a folded span resumes in the state its lines would have left.
    fn from_packed((fence, depth): (u16, i16)) -> Self {
        Self {
            fence: Fence::unpack(fence),
            list_depth: (depth >= 0).then_some(depth as usize),
        }
    }

    fn pair(self) -> (Option<Fence>, Option<usize>) {
        (self.fence, self.list_depth)
    }
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
        // No image store: any `![](...)` line lays out as its plain source text.
        // The bare build helper is the tests' entry point, so it fixes the display
        // scale at unity; the scale-aware paths go through a ctx the caller builds.
        let images = ImageStore::new(None);
        let ctx = LayoutCtx::new(
            fonts,
            rope,
            wrap_width,
            body_size,
            active,
            &images,
            ImageUi::default(),
            FoldMap::NONE,
            Scale::ONE,
        );
        Self::build_with_images(&ctx, left, top)
    }

    /// Like [`Self::build`], but resolves a standalone `![alt](path)` line to a
    /// rendered picture via `images`, scaled to fit the content column (aspect
    /// preserved, never upscaled). The picture always shows once decoded; its
    /// editable `![alt](path)` source, when revealed, is laid out as a text line
    /// *above* the picture, so editing happens with a live preview below. The
    /// source is revealed when the cursor is on the line or `image_ui.raw` pins it
    /// open (the corner button). A selected image (`image_ui.selected`) carries a
    /// border and button. A missing or undecodable image falls back to plain
    /// source, so a broken link stays visible and editable.
    pub fn build_with_images(ctx: &LayoutCtx, left: i32, top: i32) -> Self {
        let n = ctx.rope.len_lines();
        Self::build_range(ctx, left, top, 0, n, LineCarry::default())
    }

    /// Lay out only the logical lines `[band_start, band_end)`, seeding the running
    /// box top from `box_top_seed` (the caller reads it from the [`LineIndex`], which
    /// knows where the band sits without laying out the lines above it) and the
    /// cross-line carry state from `index.entry_state(band_start)`. The resulting
    /// `lines` hold absolute document baselines, so the existing consumers
    /// (`caret`, `offset_at`, `selection_rects`, …) work unchanged on the window;
    /// callers guarantee the offsets they ask about fall inside the band. This is
    /// the windowed counterpart of [`Self::build_with_images`]: same per-line
    /// [`lay_line`], a bounded slice of the loop.
    pub fn build_window(
        ctx: &LayoutCtx,
        left: i32,
        box_top_seed: i32,
        index: &LineIndex,
        band_start: usize,
        band_end: usize,
    ) -> Self {
        let n = ctx.rope.len_lines();
        let start = band_start.min(n);
        let end = band_end.min(n).max(start);
        let (fence, list_depth) = index.entry_state(start);
        Self::build_range(
            ctx,
            left,
            box_top_seed,
            start,
            end,
            LineCarry::new(fence, list_depth),
        )
    }

    /// Lay out logical lines `[start, end)` starting at content-space top
    /// `box_top_seed`, seeded with the cross-line carry (`fence`, `list_depth`)
    /// entering `start`. The one loop behind both the full build
    /// ([`Self::build_with_images`], `0..n` from ground state) and the windowed
    /// build ([`Self::build_window`], a band seeded from the index).
    fn build_range(
        ctx: &LayoutCtx,
        left: i32,
        box_top_seed: i32,
        start: usize,
        end: usize,
        mut carry: LineCarry,
    ) -> Self {
        let n = ctx.rope.len_lines();
        let mut box_top = box_top_seed;
        let mut pool = LayoutScratch::default();
        let mut tables = TableMemo::default();
        let mut lines = Vec::new();
        let mut i = start;
        while i < end {
            // A folded section is jumped whole: its lines emit nothing and take no
            // height. The carry comes from the index's record for the line after the
            // span, so a code fence or list that opened inside the fold still governs
            // the visible lines below it.
            if let Some((skip_to, exit)) = ctx.folds.skip_from(i) {
                carry = LineCarry::from_packed(exit);
                i = skip_to.max(i + 1);
                continue;
            }
            let line = LogicalLine::at(ctx.rope, i, n);
            box_top += lay_line(
                ctx,
                line,
                i,
                box_top,
                &mut carry,
                &mut pool,
                &mut tables,
                false,
                &mut lines,
            )
            .height;
            i += 1;
        }
        Self {
            lines,
            runs: pool.runs,
            left,
            text_height: box_top - box_top_seed,
            scale: ctx.scale,
        }
    }

    pub fn lines(&self) -> &[VisualLine] {
        &self.lines
    }

    /// Nudge each visual line's baseline by `shifts[i]` and set the total content
    /// height, re-placing the laid-out lines at externally computed positions. The diff
    /// view uses it to align its two columns line for line: it lays each side out
    /// normally, then shifts every line onto the shared row band both columns occupy.
    /// `shifts` has one entry per visual line, in order.
    pub fn shift_baselines(&mut self, shifts: &[i32], content_height: i32) {
        for (line, &dy) in self.lines.iter_mut().zip(shifts) {
            line.baseline += dy;
        }
        self.text_height = content_height;
    }

    /// The inline runs of `line`, resolved from the shared `runs` pool. `line`
    /// must belong to this layout (its `runs` range indexes this pool).
    pub fn run_slice(&self, line: &VisualLine) -> &[Run] {
        &self.runs[line.runs.start as usize..line.runs.end as usize]
    }

    /// Total height of the laid-out text block in pixels, for sizing the
    /// scrollable content.
    pub fn text_height(&self) -> i32 {
        self.text_height
    }

    /// The x of the content's left edge (the column's left margin), which the
    /// painter uses as the origin for text, panels, and images so a pane's
    /// content sits in its own column rather than at the window's left margin.
    pub fn left(&self) -> i32 {
        self.left
    }

    /// The byte range `(start, end)` of the visual line containing `offset`, for
    /// Home/End motion. In a table row this is the range of the cell holding the
    /// offset, so Home/End land at the cell's content edges rather than on the
    /// structural pipes (where a type would corrupt the row).
    pub fn line_bounds(&self, offset: usize) -> (usize, usize) {
        let line = self.line_for(offset);
        if let Some(t) = &line.table {
            if let Some(cell) = t
                .cells
                .iter()
                .find(|c| offset <= c.end)
                .or_else(|| t.cells.last())
            {
                return (cell.start, cell.end);
            }
        }
        (line.start, line.end)
    }

    /// The caret rectangle (its x, top y, and height) for the cursor at `offset`.
    /// The height spans the glyph box (ascent to descent) of the line it lands on,
    /// so the bar matches the text height rather than the taller padded line box.
    /// The caret only ever lands on a revealed line (the cursor's own line shows
    /// its markers), so its `offset` never falls inside a hidden run.
    pub fn caret(&self, fonts: &Fonts, rope: &Rope, offset: usize) -> (i32, i32, i32) {
        let line = self.line_for(offset);
        if let Some(t) = &line.table {
            return self.table_caret(fonts, rope, line, t, offset);
        }
        let m = fonts.metrics(line.size);
        let x = self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, offset);
        (
            x.round() as i32,
            line.baseline - m.ascent,
            m.ascent + m.descent,
        )
    }

    /// The byte offset nearest a click at content-space `(x, y)`.
    ///
    /// A click outside the text's vertical extent resolves in reading order rather
    /// than to the column the pointer's x happens to sit over on the edge line it
    /// would otherwise clamp to: below the last line lands at the document end,
    /// above the first line at its start. This is what lets a drag begun in the
    /// empty space below a note anchor at the end of the text, not mid-line. Within
    /// the text the band picks the line and x picks the column.
    pub fn offset_at(&self, fonts: &Fonts, rope: &Rope, x: f32, y: f32) -> usize {
        let last = self.last_line();
        if y >= self.line_bottom(fonts, last) as f32 {
            return last.end;
        }
        if let Some(first) = self.lines.first() {
            let top = first.baseline - fonts.metrics(first.size).ascent;
            if y < top as f32 {
                return first.start;
            }
        }
        let line = self.line_for_y(fonts, y);
        self.hit(fonts, rope, line, x, y)
    }

    /// Hit-test a click at content-space `(x, y)` against `line`, dispatching a
    /// table row to its cell-aware hit-test and any other line to the run-walk. A
    /// non-table line ignores `y` (its band already picked the line).
    fn hit(&self, fonts: &Fonts, rope: &Rope, line: &VisualLine, x: f32, y: f32) -> usize {
        match &line.table {
            Some(t) => self.table_hit(fonts, rope, line, t, x, y),
            None => self.hit_line(fonts, rope, line, x),
        }
    }

    /// If the visual line at content-space `y` is a rendered horizontal rule (a
    /// thematic break whose cursor has left), its content byte range
    /// `(start, end)`. A click there selects the whole rule rather than dropping a
    /// caret between its `---` markers; `None` for any other line.
    pub fn rule_at(&self, fonts: &Fonts, y: f32) -> Option<(usize, usize)> {
        let line = self.line_for_y(fonts, y);
        line.rule.then_some((line.start, line.end))
    }

    /// Whether content-space `(x, y)` lands on the glyphs the painter drew for the
    /// byte range `span` on `line`. The caller has the row from [`Self::row_at`] and
    /// the span from the markdown, so a link hit-tests against its own drawn box the
    /// way a checkbox does: the dead space a click otherwise clamps into (the left
    /// margin, the ragged run-out past the last glyph of a wrapped line, a widget
    /// row's padding) belongs to no span and lands the caret instead.
    pub(crate) fn span_hit(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
        span: (usize, usize),
        x: f32,
        y: f32,
    ) -> bool {
        let drawn = match &line.table {
            Some(t) => self.table_span_x(fonts, rope, (line, t), span, x, y),
            None => self.runs_span_x(
                fonts,
                rope,
                RunStyle::of(line),
                self.run_slice(line),
                (self.left + line.indent) as f32,
                span,
            ),
        };
        drawn.is_some_and(|(x0, x1)| x >= x0 && x < x1)
    }

    /// The x-range of `span` within the cell of table row `line` the pointer is
    /// over. A span in another cell, or on another of that cell's wrapped lines,
    /// draws none of its glyphs here and yields `None`.
    fn table_span_x(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        (line, t): (&VisualLine, &TableRow),
        span: (usize, usize),
        x: f32,
        y: f32,
    ) -> Option<(f32, f32)> {
        let col = self.table_col_at(line, t, x)?;
        let cell = t.cells.get(col)?;
        let k = self.cell_line_at(fonts, line, t, cell, y)?;
        let cl = cell.lines.get(k)?;
        let content_x = t.columns.get(col).map_or(0, |c| aligned_x(*c, cl.width));
        let left = (self.left + line.indent + content_x) as f32;
        self.runs_span_x(
            fonts,
            rope,
            RunStyle::of_cell(line, t),
            &cl.runs,
            left,
            span,
        )
    }

    /// The x-range the glyphs of `span` occupy within `runs`, laid from pen `left`,
    /// or `None` when the runs draw none of the span. A collapsed marker is
    /// zero-width, so a span reaches only as far as the text actually shown; a code
    /// run's pill pad frames its glyphs the way the painter lays it.
    fn runs_span_x(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        style: RunStyle,
        runs: &[Run],
        left: f32,
        span: (usize, usize),
    ) -> Option<(f32, f32)> {
        let mut pen = left;
        let mut drawn: Option<(f32, f32)> = None;
        for run in runs {
            if !run.visible {
                continue;
            }
            let face = run_face(fonts, style.size, run.kind, style.heading);
            let (lead, trail) = run.pads.px(self.scale);
            pen += lead as f32;
            let (s, e) = (span.0.max(run.start), span.1.min(run.end));
            if s < e {
                let x0 = pen + shape::text_advance(face, &rope.slice(run.start..s).as_cow());
                let x1 = x0 + shape::text_advance(face, &rope.slice(s..e).as_cow());
                drawn = Some((drawn.map_or(x0, |(a, _)| a), x1));
            }
            pen +=
                shape::text_advance(face, &rope.slice(run.start..run.end).as_cow()) + trail as f32;
        }
        drawn
    }

    /// The drawn checkbox's rectangle in content space for a task line whose
    /// gutter is [`Gutter::Checkbox`]: a square sized to the line's metrics, inset
    /// within the marker region a gap short of the text so it crowds neither the
    /// list indent nor the words. Its size and offset come from the canonical
    /// `[ ] ` width, not the live `[x]`/`[ ]`, so a checked and an unchecked box in
    /// a list line up in a column (the source `x` renders wider than a space in the
    /// proportional face). The painter draws the box here and [`Self::checkbox_at`]
    /// hit-tests clicks against it, so the two always agree. `None` for any other
    /// line (a task item's box shows on every line it occupies, the cursor's own
    /// line included).
    pub(crate) fn checkbox_rect(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
    ) -> Option<Rect> {
        if !matches!(line.gutter, Gutter::Checkbox { .. }) {
            return None;
        }
        // The block marker is the line's first run: `<indent>- [c] `. A wrap so
        // narrow it splits the marker mid-way (degenerate) just draws no box.
        let run = self.run_slice(line).first()?;
        let marker = rope.slice(run.start..run.end).as_cow();
        let indent = markdown::leading_indent(&marker);
        // Guard: a well-formed task marker has `[c]` here; a marker split mid-way by a
        // very narrow wrap draws no box.
        marker.get(indent + 2..indent + 5)?;
        let face = fonts.face(line.size, FontStyle::Regular);
        // `pre` is the marker up to the `[`; `tail_w` is the canonical `[ ] `
        // (brackets plus the trailing space before the text). Measuring the
        // canonical form, not the live `[x]`/`[ ]`, keeps checked and unchecked
        // boxes aligned in a column (source `x` renders wider than a space).
        let pre = shape::text_advance(face, marker.get(..indent + 2)?);
        let tail_w = shape::text_advance(face, "[ ] ");
        let m = fonts.metrics(line.size);
        // A touch taller than the ascent, so the box reads at a comfortable
        // weight beside the text rather than shrinking into it.
        let side = (m.ascent * 13 / 12).max(11);
        // The text begins at `pre + tail_w`. Right-anchor the box a gap short of it,
        // so it sits inset from both the list indent and the words rather than
        // crowding the text; clamp so it never runs off the marker's left.
        let gap = (m.ascent as f32 * crate::theme::DECOR_TEXT_GAP).max(4.0);
        let box_left = (pre + tail_w - gap - side as f32).max(0.0);
        let x = self.left + line.indent + box_left.round() as i32;
        // Centered on the middle of the ascent band, where the glyphs' bulk sits.
        let y = line.baseline - (m.ascent + side) / 2;
        Some(Rect {
            x,
            y,
            w: side,
            h: side,
        })
    }

    /// The logical-line start of the task item whose drawn checkbox contains
    /// content-space `(x, y)`, or `None`. The box is drawn on every task line,
    /// the cursor's own included, so a hit inside it always means "toggle" while
    /// a click elsewhere on the line still places the caret. The target is padded
    /// a little past the drawn box for clickability.
    pub fn checkbox_at(&self, fonts: &Fonts, rope: &Rope, x: f32, y: f32) -> Option<usize> {
        let line = self.line_for_y(fonts, y);
        let rect = self.checkbox_rect(fonts, rope, line)?;
        let pad = self.scale.px(CHECKBOX_HIT_PAD);
        let target = Rect {
            x: rect.x - pad,
            y: rect.y - pad,
            w: rect.w + 2 * pad,
            h: rect.h + 2 * pad,
        };
        rect_contains(&target, x, y).then_some(line.start)
    }

    /// The fold chevron's rectangle in content space for a heading that owns a
    /// section: a square hanging in the left margin, right-anchored a gap short of
    /// the text column and centered on the heading's ascent band. It sits entirely
    /// left of `left`, so showing or hiding it never moves a glyph. `None` on every
    /// line that offers no fold. The painter draws here and [`Self::fold_at`]
    /// hit-tests against it, so the two always agree.
    pub(crate) fn fold_rect(
        &self,
        fonts: &Fonts,
        line: &VisualLine,
        body_size: u32,
    ) -> Option<Rect> {
        line.fold?;
        // The size the file tree marks a folder with: the shared sizing rule over
        // the row a line of body text would make, so the two controls come out
        // alike. Sized off the body rather than the heading, so every level's
        // chevron matches and they line up in the margin.
        let row = crate::theme::row_height(fonts.metrics(body_size).line_height, self.scale);
        let side = crate::theme::chevron_side(row, self.scale);
        let gap = self.scale.px(FOLD_CHEVRON_GAP);
        // Centered on the heading's glyphs, the band from its ascent to its descent,
        // which is what the eye reads as the middle of the line. The full line box
        // is not that band: its surplus is leading, and leading all falls below the
        // text, so centering on the box would sit the chevron low.
        let m = fonts.metrics(line.size);
        let glyph_top = line.baseline - m.ascent;
        Some(Rect {
            x: self.left - gap - side,
            y: glyph_top + (m.ascent + m.descent - side) / 2,
            w: side,
            h: side,
        })
    }

    /// The buffer offset of the heading whose fold chevron contains content-space
    /// `(x, y)`, or `None`. The target is padded past the drawn shape, which is
    /// small and sits in the margin where nothing competes for the click.
    pub fn fold_at(&self, fonts: &Fonts, x: f32, y: f32, body_size: u32) -> Option<usize> {
        let line = self.row_at(fonts, y)?;
        let rect = self.fold_rect(fonts, line, body_size)?;
        let pad = self.scale.px(FOLD_HIT_PAD);
        let target = Rect {
            x: rect.x - pad,
            y: rect.y - pad,
            w: rect.w + 2 * pad,
            h: rect.h + 2 * pad,
        };
        rect_contains(&target, x, y).then_some(line.start)
    }

    /// The picture's rectangle in content space for an image `line` (the painter
    /// blits here and draws the border/button around it). `None` for a non-image
    /// line. The image is centered vertically in the line's box by [`IMAGE_PAD_Y`].
    pub(crate) fn image_rect(&self, fonts: &Fonts, line: &VisualLine) -> Option<Rect> {
        let img = line.image?;
        let m = fonts.metrics(line.size);
        Some(Rect {
            x: self.left + line.indent,
            y: line.baseline - m.ascent + self.scale.px(IMAGE_PAD_Y),
            w: img.width,
            h: img.height,
        })
    }

    /// The logical-line start of the image whose picture contains content-space
    /// `(x, y)`, for selecting it on click. Callers should test [`Self::image_button_at`]
    /// first, since the button overlays the picture's corner.
    pub fn image_at(&self, fonts: &Fonts, x: f32, y: f32) -> Option<usize> {
        self.lines.iter().find_map(|line| {
            let rect = self.image_rect(fonts, line)?;
            rect_contains(&rect, x, y).then_some(line.start)
        })
    }

    /// The logical-line start of the selected image whose corner toggle button
    /// contains content-space `(x, y)`. Only a selected image draws a button.
    pub fn image_button_at(&self, fonts: &Fonts, x: f32, y: f32) -> Option<usize> {
        self.lines.iter().find_map(|line| {
            let img = line.image?;
            let rect = self.image_rect(fonts, line)?;
            (img.selected && rect_contains(&image_button_rect(&rect, self.scale), x, y))
                .then_some(line.start)
        })
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
        let mut target = idx + dir;
        // Step over the picture half of a revealed image (its source line is the
        // caret target). A picture-only line is not skipped, so the caret can land
        // on it (at its start), revealing the source on the next build. Table rows
        // are caret targets and are not skipped: motion lands in a cell at goal_x.
        while target >= 0
            && (target as usize) < self.lines.len()
            && self.is_stacked_widget(target as usize)
        {
            target += dir;
        }
        if target < 0 || target as usize >= self.lines.len() {
            return offset;
        }
        // A table row needs a y to pick the wrapped line within a cell; entering it
        // from above/below lands on its first row of text (the box top).
        let line = &self.lines[target as usize];
        let y = (line.baseline - fonts.metrics(line.size).ascent) as f32;
        self.hit(fonts, rope, line, goal_x, y)
    }

    /// The highlight rectangles covering the byte range `[start, end)`, one per
    /// visual line it touches. Each fills the full line box but centered on the
    /// glyph box, splitting the leading above and below the text, so stacked lines
    /// meet with no gap while a single line's highlight stays balanced around it.
    pub fn selection_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        start: usize,
        end: usize,
    ) -> Vec<Rect> {
        self.rects(fonts, rope, start, end, false)
    }

    /// The highlight rectangles for a diff word wash, like [`Self::selection_rects`] but
    /// reaching the content-left edge on a line the range starts at or before. A change
    /// that spans a whole line (a wholly added or removed line) then washes its list
    /// bullet or quote marker too, rather than starting at the text past the collapsed
    /// marker and leaving the decoration bare. A mid-line change (a single reworded word)
    /// starts past the marker as before, so only a line-leading change reaches it.
    pub fn diff_wash_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        start: usize,
        end: usize,
    ) -> Vec<Rect> {
        self.rects(fonts, rope, start, end, true)
    }

    /// The highlight rectangles covering `[start, end)`. `cover_leading` reaches the
    /// content-left edge on a line the range begins at or before (the diff wash uses it
    /// to cover a line's marker); otherwise the left edge follows the text.
    fn rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        start: usize,
        end: usize,
        cover_leading: bool,
    ) -> Vec<Rect> {
        let mut rects = Vec::new();
        for (i, line) in self.lines.iter().enumerate() {
            let m = fonts.metrics(line.size);
            let sel_start = start.max(line.start);
            let sel_end = end.min(line.end);
            // A hard line's newline is selected when the range reaches past the
            // line's text into the break. Soft wraps have no newline to select.
            let newline_selected = line.hard && start <= line.end && end > line.end;
            if sel_start >= sel_end && !newline_selected {
                continue;
            }
            // A selection confined to one cell highlights just that text, like any
            // prose. One that leaves the cell (into another column or row) selects
            // row-granular: the whole box, so marking across cells reads as marking
            // rows, matching how a delete removes whole rows.
            if let Some(t) = &line.table {
                if start <= line.end
                    && end > line.start
                    && !self.table_selection_rects(fonts, rope, line, start, end, &mut rects)
                {
                    rects.push(Rect {
                        x: self.left + line.indent,
                        y: line.baseline - m.ascent,
                        w: t.width,
                        h: t.height,
                    });
                }
                continue;
            }
            // An image line has no text to run-walk. The picture is highlighted
            // only when it is the sole representation of its logical line (a text
            // selection spanning a picture-only image). When its editable source is
            // stacked above it (the reveal), that source line already carries the
            // highlight, so the picture is not boxed too: selecting the raw text
            // must not read as selecting the image.
            if let Some(img) = line.image {
                if !self.is_stacked_widget(i) {
                    rects.push(Rect {
                        x: self.left + line.indent,
                        y: line.baseline - m.ascent,
                        w: img.width.max(1),
                        h: img.box_height(),
                    });
                }
                continue;
            }
            // Measure the run-walked advance to each end of the selection; bytes
            // in a hidden marker collapse to the marker's edge, so a selection
            // touching one (in practice a revealed line) measures sensibly. When
            // covering the leading edge (a diff wash over a whole line), the left edge
            // is the content-left rather than the text past the marker, so the line's
            // bullet or quote decoration is washed too.
            let x0 = if cover_leading && start <= line.start {
                // The content-left edge, before the item's hanging indent, so the wash
                // reaches the bullet or quote marker sitting in that gutter.
                self.left as f32
            } else {
                self.left as f32
                    + line.indent as f32
                    + self.advance_to(fonts, rope, line, sel_start)
            };
            let x1 =
                self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, sel_end);
            let mut w = (x1 - x0).round() as i32;
            if newline_selected {
                w += self.scale.px(NEWLINE_PAD);
            }
            // The fill is the full line box centered on the glyph box: start a
            // leading pad above the text so the surplus splits above and below,
            // and stacked lines meet with no gap.
            rects.push(Rect {
                x: x0.round() as i32,
                y: line.baseline - m.ascent - leading_pad(m),
                w: w.max(1),
                h: m.line_height,
            });
        }
        rects
    }

    /// The tinted-panel rectangles behind every visible inline code span, one per
    /// run, each sized to its line's metrics. Fenced code blocks are excluded:
    /// they have their own full-width panel. Like [`Self::selection_rects`], these
    /// are background decorations the app paints beneath the selection and text, so
    /// a selection over inline code still shows through. Positions are run-walked
    /// the same way as the selection, so the two agree on a shared line.
    pub fn code_rects(&self, fonts: &Fonts, rope: &Rope) -> Vec<Rect> {
        let mut rects = Vec::new();
        for line in &self.lines {
            if line.code_block {
                continue;
            }
            let m = fonts.metrics(line.size);
            // Inline code inside a table lives in the cell runs, not the line's
            // own runs, so its pills come from the cell walk instead.
            if let Some(t) = &line.table {
                self.table_code_rects(fonts, rope, line, t, m, &mut rects);
                continue;
            }
            let heading = line.heading.is_some();
            let runs = self.run_slice(line);
            for (i, run) in runs.iter().enumerate() {
                if !run.visible
                    || !matches!(run.kind, markdown::SpanKind::Text(markdown::Emphasis::Code))
                {
                    continue;
                }
                // `advance_to(run.start)` lands at the run's flow start; the glyphs
                // (and so the panel the app expands from them) sit its leading pad in
                // from there, leaving the space before the span uncovered.
                let x0 = self.left as f32
                    + line.indent as f32
                    + self.advance_to(fonts, rope, line, run.start)
                    + run.pads.px(self.scale).0 as f32;
                let w = shape::text_advance(
                    run_face(fonts, line.size, run.kind, heading),
                    &rope.slice(run.start..run.end).as_cow(),
                );
                // On the cursor's line the span's backtick markers are revealed;
                // grow the pill out to enclose them so the delimiters sit inside
                // the tint. Off the line they are hidden (zero width), leaving the
                // pill hugging the content as before.
                let open = backtick_marker_width(runs, i, false, fonts, line.size, heading, rope);
                let close = backtick_marker_width(runs, i, true, fonts, line.size, heading, rope);
                rects.push(Rect {
                    x: (x0 - open).round() as i32,
                    y: line.baseline - m.ascent,
                    w: ((w + open + close).round() as i32).max(1),
                    h: m.ascent + m.descent,
                });
            }
        }
        rects
    }

    /// Inline-code pills for a table row: the same cell run-walk the painter does
    /// in [`crate::paint`], reproduced so each tint panel lands under the glyphs it
    /// backs. On the cursor's row the backtick markers are revealed, so each pill
    /// grows to enclose them the way [`Self::code_rects`] does for an ordinary line;
    /// off the row they are hidden and the pill hugs the content. Appends to `rects`.
    fn table_code_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
        t: &TableRow,
        m: Metrics,
        rects: &mut Vec<Rect>,
    ) {
        let row_x = self.left + line.indent;
        let box_top = line.baseline - m.ascent;
        let max_lines = row_max_lines(t.height, m.line_height, self.scale);
        for (col, cell) in t.columns.iter().zip(&t.cells) {
            let n_lines = cell.lines.len().max(1);
            for (k, cl) in cell.lines.iter().enumerate() {
                // The pill's glyph-box top matches the centered text baseline, so the
                // tint sits over the code instead of above it.
                let top = box_top
                    + self.scale.px(TABLE_CELL_PAD_Y)
                    + cell_line_top(k, n_lines, max_lines, m);
                let mut pen = row_x + aligned_x(*col, cl.width);
                for (i, run) in cl.runs.iter().enumerate() {
                    if !run.visible {
                        continue;
                    }
                    // A run's glyphs sit its leading pad in from the pen, matching the
                    // draw walk; the pill (which paint then expands by the pad) starts
                    // at the glyphs and grows over any revealed backticks.
                    let (lead, trail) = run.pads.px(self.scale);
                    let glyph_x = pen + lead;
                    let face = run_face(fonts, line.size, run.kind, t.header);
                    let w = shape::text_advance(face, &rope.slice(run.start..run.end).as_cow());
                    if matches!(run.kind, markdown::SpanKind::Text(markdown::Emphasis::Code)) {
                        let open = backtick_marker_width(
                            &cl.runs, i, false, fonts, line.size, t.header, rope,
                        );
                        let close = backtick_marker_width(
                            &cl.runs, i, true, fonts, line.size, t.header, rope,
                        );
                        rects.push(Rect {
                            x: (glyph_x as f32 - open).round() as i32,
                            y: top,
                            w: ((w + open + close).round() as i32).max(1),
                            h: m.ascent + m.descent,
                        });
                    }
                    pen = glyph_x + w.round() as i32 + trail;
                }
            }
        }
    }

    /// The visual line containing `offset`. A cursor at a soft-wrap boundary is
    /// given to the later line; one at a hard newline stays on its line. Image
    /// picture lines are skipped: they are never caret targets, so the caret lands
    /// on the source text line (which a revealed image always has above its
    /// picture). A table row is a caret target (its cells hold the caret), so it is
    /// not skipped. Falls back to the last text line for an offset at the document
    /// end.
    fn line_for(&self, offset: usize) -> &VisualLine {
        &self.lines[self.line_index_for(offset)]
    }

    fn line_index_for(&self, offset: usize) -> usize {
        self.lines
            .iter()
            .position(|l| l.image.is_none() && (offset < l.end || (offset == l.end && l.hard)))
            .or_else(|| self.lines.iter().rposition(|l| l.image.is_none()))
            .unwrap_or(self.lines.len().saturating_sub(1))
    }

    /// The visual line whose vertical band contains `y`, clamped to the first or
    /// last line for clicks above or below the text. Each line's band reaches to
    /// its own descent below the baseline.
    fn line_for_y(&self, fonts: &Fonts, y: f32) -> &VisualLine {
        self.lines
            .iter()
            .find(|l| y < self.line_bottom(fonts, l) as f32)
            .unwrap_or_else(|| self.last_line())
    }

    /// The bottom of `line`'s box in content space. A widget line's box is its full
    /// (taller) picture or table row, so a click anywhere on it lands on that line.
    fn line_bottom(&self, fonts: &Fonts, line: &VisualLine) -> i32 {
        let m = fonts.metrics(line.size);
        let height = match (line.image, &line.table) {
            (Some(img), _) => img.box_height(),
            (None, Some(table)) => table.box_height(),
            (None, None) => m.line_height,
        };
        line.baseline - m.ascent + height
    }

    /// The visual line whose box contains `y`, or `None` when `y` lands in no
    /// line's box at all: the margin above the first, the gap below the last, or
    /// the space between two blocks. [`Self::line_for_y`] clamps such a point onto
    /// the nearest line so a caret always has somewhere to go; this reports only
    /// what the painter actually drew there, which is what a click on a rendered
    /// widget (a link, say) has to hit.
    pub(crate) fn row_at(&self, fonts: &Fonts, y: f32) -> Option<&VisualLine> {
        let line = self.line_for_y(fonts, y);
        let top = (line.baseline - fonts.metrics(line.size).ascent) as f32;
        (y >= top && y < self.line_bottom(fonts, line) as f32).then_some(line)
    }

    /// The byte offset within `line` nearest window x `x`, snapped to a grapheme
    /// cluster boundary (a click in a cluster's left half lands before it). Hit
    /// testing walks only the visible runs, so a click on a line with hidden
    /// markers lands in the visible text, never inside a collapsed marker.
    fn hit_line(&self, fonts: &Fonts, rope: &Rope, line: &VisualLine, x: f32) -> usize {
        let mut acc = self.left as f32 + line.indent as f32;
        let mut drew_text = false;
        for run in self.run_slice(line) {
            if !run.visible {
                continue;
            }
            drew_text = true;
            let face = run_face(fonts, line.size, run.kind, line.heading.is_some());
            let text = rope.slice(run.start..run.end).as_cow();
            // A pill pad sits before the run's glyphs, so a click in that gap lands at
            // the run start; the trailing pad pushes the pen past the glyphs so a click
            // there lands after the span.
            let (lead, trail) = run.pads.px(self.scale);
            acc += lead as f32;
            for (off, cluster) in grapheme::graphemes(&text) {
                let w = shape::text_advance(face, cluster);
                if x < acc + w / 2.0 {
                    return run.start + off;
                }
                acc += w;
            }
            acc += trail as f32;
        }
        // Past every glyph: a line that drew text ends at its logical end, which
        // sits *after* any trailing collapsed marker (the closing `*` of `*word*`);
        // the last visible run's end would land the caret before that hidden marker.
        // A line with no visible glyphs (an empty line, or a picture row whose
        // source is collapsed) instead lands at its start, so arrowing onto a
        // picture reveals its source on the next build.
        if drew_text {
            line.end
        } else {
            line.start
        }
    }

    /// Pen advance in pixels from the left margin to absolute byte `offset`,
    /// summing the line's visible runs; a hidden (collapsed) marker contributes
    /// nothing. An `offset` at or past the line's end measures the whole line.
    fn advance_to(&self, fonts: &Fonts, rope: &Rope, line: &VisualLine, offset: usize) -> f32 {
        let mut x = 0.0;
        for run in self.run_slice(line) {
            // Runs are ordered, so once a run starts at or past `offset` the rest
            // do too; hidden runs add nothing.
            if !run.visible || offset <= run.start {
                continue;
            }
            let face = run_face(fonts, line.size, run.kind, line.heading.is_some());
            let (lead, trail) = run.pads.px(self.scale);
            // A run fully before the offset contributes its glyphs plus its pill pads;
            // one containing the offset contributes its leading pad plus the glyphs up
            // to the offset.
            if offset >= run.end {
                x += (lead + trail) as f32
                    + shape::text_advance(face, &rope.slice(run.start..run.end).as_cow());
            } else {
                x += lead as f32
                    + shape::text_advance(face, &rope.slice(run.start..offset).as_cow());
            }
        }
        x
    }

    fn last_line(&self) -> &VisualLine {
        // build always pushes at least one line, so this never panics.
        debug_assert!(!self.lines.is_empty());
        &self.lines[self.lines.len() - 1]
    }

    /// The caret rectangle for `offset` inside table row `line`: the cell holding
    /// the offset, the wrapped line within it, and the pen advance to the offset,
    /// resolved against the shared column geometry and each column's alignment. An
    /// empty cell places the caret at its content left.
    fn table_caret(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
        t: &TableRow,
        offset: usize,
    ) -> (i32, i32, i32) {
        let m = fonts.metrics(line.size);
        let box_top = line.baseline - m.ascent;
        let row_left = self.left + line.indent;
        let idx = t
            .cells
            .iter()
            .position(|c| offset <= c.end)
            .unwrap_or(t.cells.len().saturating_sub(1));
        let height = m.ascent + m.descent;
        let cell_pad_y = self.scale.px(TABLE_CELL_PAD_Y);
        let Some(cell) = t.cells.get(idx) else {
            return (row_left, box_top + cell_pad_y, height);
        };
        let (k, adv, line_w) = self.cell_offset_x(fonts, rope, line.size, t.header, cell, offset);
        let content_x = t.columns.get(idx).map_or(0, |c| aligned_x(*c, line_w));
        let x = row_left + content_x + adv.round() as i32;
        let n_lines = cell.lines.len().max(1);
        let max_lines = row_max_lines(t.height, m.line_height, self.scale);
        let y = box_top + cell_pad_y + cell_line_top(k, n_lines, max_lines, m);
        (x, y, height)
    }

    /// The wrapped line index within `cell` holding `offset`, the pen advance to
    /// the offset in that line's visible runs, and that line's drawn width (for
    /// alignment). An empty cell yields the first line, zero advance, zero width.
    fn cell_offset_x(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        size: u32,
        header: bool,
        cell: &TableCell,
        offset: usize,
    ) -> (usize, f32, i32) {
        if cell.lines.is_empty() {
            return (0, 0.0, 0);
        }
        // The last wrapped line starting at or before the offset.
        let mut k = 0;
        for (i, cl) in cell.lines.iter().enumerate() {
            let start = cl.runs.first().map_or(cell.start, |r| r.start);
            if offset >= start {
                k = i;
            }
        }
        let cl = &cell.lines[k];
        let mut adv = 0.0;
        for run in &cl.runs {
            if !run.visible || offset <= run.start {
                continue;
            }
            // A run's glyphs sit its leading pill pad in from the pen; add that on
            // entry, and the trailing pad once the offset is past the run.
            let (lead, trail) = run.pads.px(self.scale);
            let end = offset.min(run.end);
            let face = run_face(fonts, size, run.kind, header);
            adv += lead as f32 + shape::text_advance(face, &rope.slice(run.start..end).as_cow());
            if offset >= run.end {
                adv += trail as f32;
            }
        }
        (k, adv, cl.width)
    }

    /// Highlight rects for a selection `[start, end)` that lies wholly inside one
    /// cell of table row `line`: a precise per-wrapped-line rect, the way ordinary
    /// text selects, rather than the whole row. Returns `false` when the range is
    /// not confined to a single cell (it spans columns or rows), so the caller falls
    /// back to a row-granular highlight. Appends to `out`.
    fn table_selection_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
        start: usize,
        end: usize,
        out: &mut Vec<Rect>,
    ) -> bool {
        let Some(t) = &line.table else {
            return false;
        };
        let Some(idx) = t
            .cells
            .iter()
            .position(|c| c.start <= start && end <= c.end)
        else {
            return false;
        };
        let cell = &t.cells[idx];
        let m = fonts.metrics(line.size);
        let row_left = self.left + line.indent;
        let box_top = line.baseline - m.ascent;
        let n_lines = cell.lines.len().max(1);
        let max_lines = row_max_lines(t.height, m.line_height, self.scale);
        let cell_pad_y = self.scale.px(TABLE_CELL_PAD_Y);
        let (k0, adv0, _) = self.cell_offset_x(fonts, rope, line.size, t.header, cell, start);
        let (k1, adv1, _) = self.cell_offset_x(fonts, rope, line.size, t.header, cell, end);
        let column = t.columns.get(idx).copied();
        for k in k0..=k1 {
            let Some(cl) = cell.lines.get(k) else { break };
            let content_x = column.map_or(0, |c| aligned_x(c, cl.width));
            let left = row_left + content_x + if k == k0 { adv0.round() as i32 } else { 0 };
            let right = row_left
                + content_x
                + if k == k1 {
                    adv1.round() as i32
                } else {
                    cl.width
                };
            // Fill the line box centered on the glyph box, matching prose: the
            // cell's glyph box is already centered in its slot, so this fills the
            // whole slot and stays balanced around the text.
            let y = box_top + cell_pad_y + cell_line_top(k, n_lines, max_lines, m) - leading_pad(m);
            out.push(Rect {
                x: left,
                y,
                w: (right - left).max(1),
                h: m.line_height,
            });
        }
        true
    }

    /// The column of table row `line` whose grid slot holds content-space `x`; a
    /// point past the last column falls to it. `None` for a row with no cells.
    fn table_col_at(&self, line: &VisualLine, t: &TableRow, x: f32) -> Option<usize> {
        let last = t.cells.len().checked_sub(1)?;
        let rel_x = (x - (self.left + line.indent) as f32) as i32;
        Some(
            t.borders
                .windows(2)
                .position(|w| rel_x < w[1])
                .unwrap_or(last)
                .min(last),
        )
    }

    /// The wrapped line of `cell` that content-space `y` falls on, clamped to its
    /// last; `None` for an empty cell. The cell's text starts below the row's cell
    /// padding, plus the block pad that centers a short cell in a taller row.
    fn cell_line_at(
        &self,
        fonts: &Fonts,
        line: &VisualLine,
        t: &TableRow,
        cell: &TableCell,
        y: f32,
    ) -> Option<usize> {
        let last = cell.lines.len().checked_sub(1)?;
        let m = fonts.metrics(line.size);
        let box_top = line.baseline - m.ascent;
        let max_lines = row_max_lines(t.height, m.line_height, self.scale);
        let block_pad = cell_block_pad(cell.lines.len(), max_lines, m.line_height);
        let rel_y = (y - (box_top + self.scale.px(TABLE_CELL_PAD_Y) + block_pad) as f32).max(0.0);
        Some(((rel_y / m.line_height as f32) as usize).min(last))
    }

    /// The byte offset nearest content-space `(x, y)` inside table row `line`: the
    /// column under `x`, the wrapped line under `y`, then the nearest glyph in that
    /// cell line. An empty cell (or a click before its text) lands at the cell start.
    fn table_hit(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        line: &VisualLine,
        t: &TableRow,
        x: f32,
        y: f32,
    ) -> usize {
        let Some(col) = self.table_col_at(line, t, x) else {
            return line.start;
        };
        let cell = &t.cells[col];
        let Some(cl) = self
            .cell_line_at(fonts, line, t, cell, y)
            .and_then(|k| cell.lines.get(k))
        else {
            return cell.start;
        };
        let content_x = t.columns.get(col).map_or(0, |c| aligned_x(*c, cl.width));
        let mut acc = (self.left + line.indent + content_x) as f32;
        for run in &cl.runs {
            if !run.visible {
                continue;
            }
            // The pill pad frames a code span's glyphs; a click in the leading pad
            // falls to the run's first cluster, one in the trailing pad past it.
            let (lead, trail) = run.pads.px(self.scale);
            acc += lead as f32;
            let face = run_face(fonts, line.size, run.kind, t.header);
            let text = rope.slice(run.start..run.end).as_cow();
            for (off, cluster) in grapheme::graphemes(&text) {
                let w = shape::text_advance(face, cluster);
                if x < acc + w / 2.0 {
                    return run.start + off;
                }
                acc += w;
            }
            acc += trail as f32;
        }
        // Past every glyph: the end of the last visible run, else the cell start.
        cl.runs
            .iter()
            .rev()
            .find(|r| r.visible)
            .map_or(cell.start, |r| r.end)
    }

    /// The table-grow affordance under content-space `(x, y)`, if any: the add-row
    /// strip just below a table's last visible row, or the add-column strip just
    /// off its right edge. Each maximal run of table-row lines is one table; the
    /// strips sit outside its box so they never overlap a cell. `rope` resolves the
    /// run back to its logical-line block for the edit.
    pub fn table_button_at(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        x: f32,
        y: f32,
    ) -> Option<TableButton> {
        let mut i = 0;
        while i < self.lines.len() {
            if self.lines[i].table.is_none() {
                i += 1;
                continue;
            }
            let start = i;
            let mut end = i;
            while end + 1 < self.lines.len() && self.lines[end + 1].table.is_some() {
                end += 1;
            }
            i = end + 1;
            let (top, bot) = (&self.lines[start], &self.lines[end]);
            let width = top.table.as_ref().map_or(0, |t| t.width);
            let left = self.left + top.indent;
            let box_top = top.baseline - fonts.metrics(top.size).ascent;
            let box_bottom = bot.baseline - fonts.metrics(bot.size).ascent
                + bot.table.as_ref().map_or(0, |t| t.box_height());
            let add_thickness = self.scale.px(TABLE_ADD_THICKNESS);
            let add_row = Rect {
                x: left,
                y: box_bottom,
                w: width,
                h: add_thickness,
            };
            if rect_contains(&add_row, x, y) {
                let block = crate::table::block_at(rope, rope.byte_to_line(bot.start))?;
                return Some(TableButton {
                    kind: TableButtonKind::AddRow,
                    rect: add_row,
                    block,
                });
            }
            let add_col = Rect {
                x: left + width,
                y: box_top,
                w: add_thickness,
                h: box_bottom - box_top,
            };
            if rect_contains(&add_col, x, y) {
                let block = crate::table::block_at(rope, rope.byte_to_line(top.start))?;
                return Some(TableButton {
                    kind: TableButtonKind::AddColumn,
                    rect: add_col,
                    block,
                });
            }
        }
        None
    }

    /// Whether the visual line at `idx` is the picture half of a revealed image,
    /// i.e. its editable source line sits immediately above it (same logical line,
    /// so same `end`). Such a widget is stepped over by vertical caret motion; a
    /// picture-only line (no source above) is not, so the caret can land on it and
    /// reveal the source.
    fn is_stacked_widget(&self, idx: usize) -> bool {
        idx > 0
            && self.lines[idx].image.is_some()
            && self.lines[idx - 1].image.is_none()
            && self.lines[idx - 1].end == self.lines[idx].end
    }
}

/// Per-line scratch buffers, pre-allocated once per build and reused across every
/// line so laying out a line clears and refills them rather than allocating fresh
/// Vecs. The buffers grow to the largest line the build sees and then stay put, so
/// after the first few lines the per-line layout does no intermediate allocation
/// for wrapping. Held by the build loop (or, in the parallel build, per worker).
#[derive(Default)]
struct LayoutScratch {
    /// The line's inline spans (block marker plus code/emphasis/link runs),
    /// produced by [`line_spans_into`]/[`code_block_spans_into`].
    spans: Vec<markdown::Span>,
    /// One entry per grapheme cluster of the line being wrapped. Sized to the widest
    /// line seen.
    clusters: Vec<Cluster>,
    /// The wrapped visual segments' byte ranges, produced by [`wrap_segments`].
    segments: Vec<(usize, usize)>,
    /// The inline parser's own reusable buffers, so classifying a line's markup
    /// allocates nothing after warmup (see [`markdown::InlineScratch`]).
    inline: markdown::InlineScratch,
    /// The flattened run pool for a full/window build: each line appends its runs
    /// here and records the range. The finished build moves this into the
    /// [`Layout`]. Stays empty on the measure-only path (the index build).
    runs: Vec<Run>,
}

/// The per-line classification `lay_line` makes once, then threads through
/// wrapping and run-clipping: the resolved block `style` and its font `size`, the
/// leading block marker's byte length (`block_marker`) and whether it persists as
/// a hanging indent (`block_persists`), whether the line's markers are `revealed`
/// (shown verbatim rather than collapsed), and whether it is fenced `code_block`.
/// The named accessors derive everything else (`heading`, `task`, `rule`, the list
/// insets) so those rules live in one place instead of inline at each use.
#[derive(Clone, Copy)]
struct LineClass {
    style: markdown::LineStyle,
    size: u32,
    block_marker: usize,
    block_persists: bool,
    revealed: bool,
    code_block: bool,
    /// The list nesting depth (0 for a top-level item or a non-list line), which
    /// scales the list's left indent so nested items step in clearly.
    list_depth: usize,
    /// True for an indented plain line continuing a list item (a Shift+Enter
    /// continuation): it takes the list's left indent but draws no bullet.
    list_continuation: bool,
}

impl LineClass {
    /// The ATX heading level (1..=6) if this is a heading line, else `None`.
    fn heading(&self) -> Option<u8> {
        match self.style {
            markdown::LineStyle::Heading(level) => Some(level),
            _ => None,
        }
    }

    /// `Some(checked)` when this line belongs to a task item, else `None`.
    fn task(&self) -> Option<bool> {
        match self.style {
            markdown::LineStyle::Task { checked } => Some(checked),
            _ => None,
        }
    }

    /// Whether markup starting at line-relative `off` draws under this line's
    /// reveal state. Text always draws; a marker draws only on a revealed line,
    /// except the block marker (the leading `[0, block_marker)` bytes) when it
    /// persists as a list bullet or quote bar. Both the wrap (which measures what
    /// draws) and [`clip_runs_into`] (which decides what to paint) ask this, so a
    /// row can never be wrapped against markup it does not show.
    fn draws(&self, kind: markdown::SpanKind, off: usize) -> bool {
        if !matches!(kind, markdown::SpanKind::Marker) {
            return true;
        }
        if off < self.block_marker {
            self.revealed || self.block_persists
        } else {
            self.revealed
        }
    }

    /// True to draw a horizontal rule: a thematic break whose cursor has left, so
    /// its raw `---` is collapsed (on the line itself the editable source shows).
    fn rule(&self) -> bool {
        matches!(self.style, markdown::LineStyle::Rule) && !self.revealed
    }

    /// A list or task item hangs its wrapped continuations under the text.
    fn is_list(&self) -> bool {
        matches!(
            self.style,
            markdown::LineStyle::Bullet
                | markdown::LineStyle::Ordered
                | markdown::LineStyle::Task { .. }
        )
    }

    /// The left inset shared by every segment, and how much it narrows the wrap
    /// width: a code block pads both sides inside its panel; a list item shifts
    /// right by a small indent (one side only).
    fn seg_pad_and_wrap_reduce(&self, scale: Scale) -> (i32, i32) {
        if self.code_block {
            let pad = scale.px(CODE_PAD_X);
            (pad, 2 * pad)
        } else if self.is_list() || self.list_continuation {
            // A continuation takes the same left indent as its item, so its own
            // leading spaces then carry the text under the item's content.
            let pad = scale.px(LIST_INDENT) + self.list_depth as i32 * scale.px(LIST_NEST_INDENT);
            (pad, pad)
        } else {
            (0, 0)
        }
    }

    /// The hanging indent for wrapped continuations: the block marker's rendered
    /// width, measured on the source marker in the regular face at this line's
    /// size, so it matches the width the marker run itself advances.
    fn block_indent(&self, fonts: &Fonts, line: &str) -> i32 {
        if self.block_persists && self.block_marker > 0 {
            shape::text_advance(
                fonts.face(self.size, FontStyle::Regular),
                &line[..self.block_marker],
            )
            .round() as i32
        } else {
            0
        }
    }
}

/// Classify one logical `line` for [`lay_line`]: resolve its block style (fenced
/// code continuation, a fence opener, or ordinary markdown with list nesting),
/// fill `pool.spans` with its inline markup, and thread the cross-line carry
/// (`fence`, `list_depth`). `on_line` is whether the cursor/selection touches the
/// line, which controls whether a delimiter's markers are revealed.
fn classify_line(
    ctx: &LayoutCtx,
    line: &str,
    on_line: bool,
    carry: &mut LineCarry,
    pool: &mut LayoutScratch,
) -> LineClass {
    if let Some(fence) = carry.fence {
        // Inside a fence: literal code, not markdown. Only a delimiter line has
        // backticks to hide, and it reveals them on the cursor's own line.
        carry.list_depth = None;
        let delimiter = markdown::closes_code_fence(line, fence.ticks);
        if delimiter {
            carry.fence = None;
        }
        code_block_spans_into(line, delimiter, fence.lang, &mut pool.spans);
        LineClass {
            style: markdown::LineStyle::Body,
            size: markdown::code_size(ctx.body_size),
            block_marker: 0,
            block_persists: false,
            revealed: !delimiter || on_line,
            code_block: true,
            list_depth: 0,
            list_continuation: false,
        }
    } else if let Some(ticks) = markdown::opens_code_fence(line) {
        carry.list_depth = None;
        let lang = crate::syntax::Lang::from_info(&line[ticks..]);
        carry.fence = Some(Fence { ticks, lang });
        code_block_spans_into(line, true, lang, &mut pool.spans);
        LineClass {
            style: markdown::LineStyle::Body,
            size: markdown::code_size(ctx.body_size),
            block_marker: 0,
            block_persists: false,
            revealed: on_line,
            code_block: true,
            list_depth: 0,
            list_continuation: false,
        }
    } else {
        // Ordinary markdown. Resolve list nesting across lines: an item deeper than
        // one past the previous item has no parent and renders as body.
        let (style, marker, continuation) = resolve_list(line, &mut carry.list_depth);
        line_spans_into(line, marker, &mut pool.inline, &mut pool.spans);
        LineClass {
            style,
            size: markdown::font_size(style, ctx.body_size),
            block_marker: marker,
            block_persists: markdown::block_marker_persists(style),
            revealed: on_line,
            code_block: false,
            list_depth: carry.list_depth.unwrap_or(0),
            list_continuation: continuation,
        }
    }
}

/// Whether `active` reaches a bullet or task line's block marker, the finer
/// live-preview test those lists use for revealing their raw `- `/`- [ ] `. A
/// heading reveals its `#` across the whole line, but a list keeps its disc or
/// checkbox while you edit the label and gives way to the raw marker only where
/// the caret or selection actually touches it: a bare caret within the marker's
/// span (its first positions), or a selection overlapping its bytes. `marker_len`
/// is the marker's byte length ([`LineClass::block_marker`]), `ls` the line start.
/// A `dragging` selection holds every marker collapsed so the decorations stay put
/// under the pointer, giving way once the drag settles.
pub(crate) fn active_reveals_marker(
    active: (usize, usize),
    ls: usize,
    marker_len: usize,
    dragging: bool,
) -> bool {
    if marker_len == 0 || dragging {
        return false;
    }
    let (a, b) = active;
    let marker_end = ls + marker_len;
    if a == b {
        // A bare caret reveals the marker while it sits within the marker's span,
        // before the words, so the marker itself is editable.
        (ls..marker_end).contains(&a)
    } else {
        // A selection reveals the marker while it overlaps the marker's bytes.
        a < marker_end && b > ls
    }
}

/// What laying out one logical line produced.
struct Laid {
    /// The height it added: the amount to advance `box_top` before the next line.
    height: i32,
    /// The extra height the same line takes with its markup revealed, since a row
    /// is wrapped to what it draws. Zero on a line that is already revealed and on
    /// one whose markup fits the rows it already has. A revealed image's source
    /// rows are not counted here; [`LineIndex::measure_line`] prices those.
    wrap_extra: i32,
}

/// Lay out one logical line `[ls, le)` starting at content-space top `box_top`,
/// appending its visual line(s) to `out`. `fence` and `list_depth` are
/// the only cross-line carry, threaded by `&mut`; everything else is a pure
/// function of this line's text plus `active`/`image_ui`, which decide which markup
/// the line draws (and so where it wraps) and whether an image shows its source.
/// This is the single per-line unit shared by the full
/// [`Layout::build_with_images`] loop and the windowed build, so the two can never
/// disagree on a line's shape.
///
/// `pool` supplies the per-line scratch buffers (`clusters`, `segments`), owned by
/// the caller's build loop and reused across every line so laying out a line
/// clears and refills them instead of allocating fresh Vecs. `tables` is the same
/// idea for the one table being walked: per-table work reached from this per-line
/// path (block detection, column fitting) is paid by a table's first row and read
/// by the rest.
#[allow(clippy::too_many_arguments)]
fn lay_line(
    ctx: &LayoutCtx,
    logical: LogicalLine,
    line_no: usize,
    box_top: i32,
    carry: &mut LineCarry,
    pool: &mut LayoutScratch,
    tables: &mut TableMemo,
    // When true the caller only wants each line's height and image flag (the
    // `LineIndex` build), so the styled runs are skipped: `clip_runs` allocates a
    // `Vec` per visual row that measurement would immediately discard.
    measure_only: bool,
    out: &mut Vec<VisualLine>,
) -> Laid {
    let (ls, le, has_newline) = (logical.start, logical.end, logical.has_newline);
    // Running box top within this logical line; the return value is how far it
    // advanced past the incoming `box_top`.
    let mut y = box_top;
    let line = ctx.rope.slice(ls..le).as_cow();
    // Whether the cursor/selection touches this logical line: the same reveal test
    // headings use, applied to fence lines too.
    let on_line = ctx.active.0 <= le && ctx.active.1 >= ls;
    // A standalone `![alt](path)` line whose image has decoded renders as a picture
    // box. The picture always shows; when its source is revealed (cursor on the
    // line, or the corner button pinned it open) the editable `![alt](path)` is laid
    // out as a text line *above* the picture. Inside a code fence the syntax is
    // literal, not an image; a missing or undecodable image falls through to plain
    // source so a broken link stays visible and editable.
    //
    // `pending_image`, when set, is the widget to append after this line's source
    // text (the reveal case). The no-reveal case pushes its widget and returns,
    // never reaching the text path.
    let mut pending_image: Option<ImageBox> = None;
    if carry.fence.is_none() {
        // A table renders as a grid of widget rows: its delimiter row draws
        // nothing, and every other row is one table widget line. A table ends any
        // open list. Detection is by rope scan (`TableMemo::locate`), so no table
        // state threads the carry.
        if let Some((span, role)) = tables.locate(ctx.rope, line_no) {
            carry.list_depth = None;
            // A table cell wraps against its source, so a row's height is the same
            // revealed or not and there is nothing for a reveal to reflow.
            let height = match role {
                TableRole::Delimiter => 0,
                TableRole::Header | TableRole::Body => {
                    lay_table_row(ctx, span, line_no, box_top, measure_only, pool, tables, out)
                }
            };
            return Laid {
                height,
                wrap_extra: 0,
            };
        }
        if let Some(link) = markdown::standalone_image(&line) {
            if let Some((iw, ih)) = ctx.images.dimensions(&line[link.path.0..link.path.1]) {
                carry.list_depth = None;
                let column = if ctx.wrap_width > 0 {
                    ctx.wrap_width
                } else {
                    iw as i32
                };
                let (width, height) = fit_image(iw, ih, column);
                let image = ImageBox {
                    path_start: ls + link.path.0,
                    path_end: ls + link.path.1,
                    width,
                    height,
                    pad_y: ctx.scale.px(IMAGE_PAD_Y),
                    selected: ctx.image_ui.selected == Some(ls),
                };
                if (on_line && !ctx.image_ui.dragging) || ctx.image_ui.raw == Some(ls) {
                    // Source revealed: emit it (fall through), then the widget. A live
                    // drag holds it collapsed so the geometry under the pointer is
                    // stable, matching the index's `reveal_adjust`.
                    pending_image = Some(image);
                } else {
                    // Picture only: one widget line, no source text.
                    out.push(VisualLine::image_widget(
                        ls,
                        le,
                        y + ctx.fonts.metrics(ctx.body_size).ascent,
                        has_newline,
                        ctx.body_size,
                        image,
                    ));
                    y += image.box_height();
                    return Laid {
                        height: y - box_top,
                        wrap_extra: 0,
                    };
                }
            }
        }
    }
    // A line inside a fenced code block is code, not markdown: it skips the inline
    // pass and renders as one code run. The ``` fence lines hide their backticks off
    // the cursor's line, like a heading's `#`. A code line carries `Body` as an
    // inert style (its gutter is unused).
    let mut class = classify_line(ctx, &line, on_line, carry, pool);
    // A revealed image's source always shows verbatim (its markers, if any, are not
    // hidden), so the text above the picture reads as written.
    class.revealed |= pending_image.is_some();
    // The plain fields the wrap loop and visual lines read by name; `class` keeps
    // the classification for the derived accessors and `clip_runs_into`.
    let LineClass {
        style,
        size,
        revealed,
        code_block,
        ..
    } = class;
    let metrics = ctx.fonts.metrics(size);
    let heading = class.heading();
    // Space above a heading, so a section reads as starting rather than as more of
    // what came before. It shows most where nothing else separates two headings:
    // collapse a run of sections and their headings would otherwise stack flush.
    //
    // The space belongs to the heading's own logical line, measured down from its
    // top, so it is part of the height `lay_line` returns and the windowed build
    // reproduces it from the index like any other. It depends only on the line's
    // classification, never on reveal or on what is folded, which is what keeps the
    // index's base geometry a pure function of content and width. The document's
    // first line is exempt: there is nothing above it to separate from.
    let pad_above = match heading {
        Some(_) if line_no > 0 => {
            (metrics.line_height as f32 * crate::theme::HEADING_SPACE_ABOVE).round() as i32
        }
        _ => 0,
    };
    y += pad_above;
    // A heading offers a fold only where it owns a section; the marker rides its
    // first visual row, so a wrapped heading shows one chevron rather than one per
    // row.
    // The line below starts just past this one's newline, which the caller already
    // resolved, so asking whether the heading owns a section costs no tree descent.
    let fold = heading
        .filter(|&level| has_newline && crate::fold::owns_section(ctx.rope, le + 1, level))
        .map(|_| ctx.folds.is_folded(line_no));
    let task = class.task();
    let rule = class.rule();
    let block_indent = class.block_indent(ctx.fonts, &line);
    let (seg_pad, wrap_reduce) = class.seg_pad_and_wrap_reduce(ctx.scale);
    let line_wrap = if ctx.wrap_width > 0 {
        (ctx.wrap_width - wrap_reduce).max(1)
    } else {
        ctx.wrap_width
    };
    // `wrap_segments` returns ranges relative to the line; offset them back to
    // absolute buffer positions. It measures what the line draws, so a collapsed
    // marker's width is not reserved and the rows hold the text the reader sees;
    // revealing the line reflows it, by `reveal_rows` extra rows.
    let reveal_rows = wrap_segments(
        ctx.fonts,
        size,
        &line,
        &pool.spans,
        &class,
        WrapBasis::Drawn,
        line_wrap,
        block_indent,
        heading.is_some(),
        ctx.scale,
        &mut pool.clusters,
        &mut pool.segments,
    )
    .saturating_sub(pool.segments.len());
    let last = pool.segments.len() - 1;
    // A bullet keeps its dot and a task its checkbox while you edit the label, so a
    // list stays a list. The decoration gives way to the raw marker only where the
    // caret or selection reaches the marker (its first positions), so the `- `/`- [ ]`
    // is itself editable and a multi-item selection can take the markers with it.
    let marker_revealed =
        matches!(
            style,
            markdown::LineStyle::Bullet | markdown::LineStyle::Task { .. }
        ) && active_reveals_marker(ctx.active, ls, class.block_marker, ctx.image_ui.dragging);
    for k in 0..pool.segments.len() {
        let (start, end) = pool.segments[k];
        // Every segment carries the shared left inset (`seg_pad`); wrapped
        // continuations add the marker's width on top so their text hangs under the
        // first line's text.
        let indent = seg_pad + if k == 0 { 0 } else { block_indent };
        // A quote reveals its raw `> ` for editing, drawing the bar only once the
        // cursor leaves; a bullet or task gives way to the raw marker only when the
        // caret reaches it (`marker_revealed`), otherwise it keeps its decoration.
        let gutter = match style {
            markdown::LineStyle::Bullet if k == 0 && !marker_revealed => Gutter::Bullet,
            // A click on the box toggles it; a click elsewhere on the line places
            // the caret. The box gives way to the raw `- [ ]` on the same reveal a
            // bullet's dot does, so the whole marker is editable.
            markdown::LineStyle::Task { checked } if k == 0 && !marker_revealed => {
                Gutter::Checkbox { checked }
            }
            markdown::LineStyle::Quote if !revealed => Gutter::QuoteBar { marker: k == 0 },
            _ => Gutter::None,
        };
        // Append this segment's runs to the shared pool and record their range; the
        // measure-only path (the index build) wants no runs, leaving an empty range.
        let runs_start = pool.runs.len() as u32;
        if !measure_only {
            clip_runs_into(&pool.spans, &line, start, end, ls, &class, &mut pool.runs);
        }
        out.push(VisualLine {
            start: ls + start,
            end: ls + end,
            baseline: y + metrics.ascent,
            // Only the final segment of a logical line that has a trailing '\n' is a
            // hard break.
            hard: k == last && has_newline,
            size,
            runs: runs_start..pool.runs.len() as u32,
            indent,
            gutter,
            code_block,
            heading,
            fold: if k == 0 { fold } else { None },
            task,
            rule,
            image: None,
            table: None,
        });
        y += metrics.line_height;
    }
    // The picture sits below its just-emitted source line(s) (the reveal case). It is
    // a widget line: never a caret target, taller than text.
    if let Some(image) = pending_image {
        // The source line above owns the caret and the logical line's newline; this
        // widget carries neither.
        out.push(VisualLine::image_widget(
            ls,
            le,
            y + metrics.ascent,
            false,
            ctx.body_size,
            image,
        ));
        y += image.box_height();
    }
    Laid {
        height: y - box_top,
        wrap_extra: reveal_rows as i32 * metrics.line_height,
    }
}

/// An `active` range that reveals nothing: `MAX <= le` is always false, so every
/// line lays out with its markers collapsed and no image source shown. This is the
/// base geometry the [`LineIndex`] measures, reveal-independent by construction.
pub(crate) const NO_REVEAL: (usize, usize) = (usize::MAX, usize::MAX);

/// Below this line count, [`LineIndex::build_parallel`] just builds sequentially:
/// the per-worker font open and thread hand-off cost more than the work saved.
const PARALLEL_MIN_LINES: usize = 20_000;

/// One worker's output in [`LineIndex::build_parallel`]: the `LineRec`s it
/// measured for its chunk (speculatively, from ground entry state) and the entry
/// state it ended in, which the stitch pass checks against the true carry.
struct ChunkOut {
    recs: Vec<LineRec>,
    end: LineCarry,
}

/// Measure logical lines `[lo, hi)` of `rope` into `LineRec`s, starting from the
/// given entry state (`None`/`None` is ground, no fence or list), and report the
/// entry state left at the end. The single measure loop behind the sequential
/// [`LineIndex::build`], the parallel workers, and the stitch re-measure; a worker
/// runs it on its own `fonts`.
fn measure_chunk(
    ctx: &LayoutCtx,
    lo: usize,
    hi: usize,
    n: usize,
    mut carry: LineCarry,
) -> ChunkOut {
    let mut scratch: Vec<VisualLine> = Vec::new();
    let mut pool = LayoutScratch::default();
    let mut tables = TableMemo::default();
    let mut recs = Vec::with_capacity(hi - lo);
    for i in lo..hi {
        recs.push(LineIndex::measure_line(
            ctx,
            i,
            n,
            &mut carry,
            &mut pool,
            &mut tables,
            &mut scratch,
        ));
    }
    ChunkOut { recs, end: carry }
}

/// One logical line's contribution to the document's vertical geometry.
#[derive(Clone, Copy, PartialEq, Debug)]
struct LineRec {
    /// Base (unrevealed) pixel height: the sum of the line's visual rows.
    height: i32,
    /// Extra height the source text adds above a standalone picture when its line
    /// is revealed; 0 for every non-image line. Applied as a per-query overlay so
    /// the base height stays a pure function of `(version, width)`.
    image_extra: i32,
    /// Extra height the line's own text gains when its markup is revealed and the
    /// rows re-wrap against the wider source; 0 where the markup fits the rows it
    /// already has. The same overlay applies it, over the whole active range rather
    /// than its ends, since every line the range touches reveals.
    wrap_extra: i32,
    /// Open code-fence backtick count entering this line (0 = not in a fence).
    fence_in: u16,
    /// List nesting depth entering this line (-1 = not in a list).
    depth_in: i16,
    /// ATX heading level 1..=6, or 0 on a line that is not a heading. Recorded
    /// with the carry state in hand, so a `#` inside a code fence reads as the
    /// code it is. [`crate::fold`] resolves heading sections from this.
    heading: u8,
}

/// What is revealed right now, on top of the base [`LineIndex`] geometry: the
/// image sources shown (the cursor's image line and a raw-pinned one, never more
/// than two), and the logical lines whose markup shows raw. Empty is the base
/// geometry.
#[derive(Clone, Copy, Default)]
pub struct RevealAdjust {
    entries: [(usize, i32); 2],
    len: usize,
    /// The logical lines the cursor or selection touches, inclusive. Their markup
    /// draws raw, so each is as tall as the index recorded it would re-wrap to.
    active: Option<(usize, usize)>,
}

impl RevealAdjust {
    /// Record that logical line `line` reveals `extra` pixels of image source.
    /// Ignores zero deltas and duplicates, and never grows past two entries.
    pub fn push(&mut self, line: usize, extra: i32) {
        if extra == 0 || self.entries[..self.len].iter().any(|&(l, _)| l == line) {
            return;
        }
        if self.len < self.entries.len() {
            self.entries[self.len] = (line, extra);
            self.len += 1;
        }
    }

    /// Record that logical lines `lo..=hi` show their markup raw, so their rows
    /// wrap against the source rather than what they draw collapsed.
    pub fn reveal_lines(&mut self, lo: usize, hi: usize) {
        self.active = Some((lo.min(hi), hi.max(lo)));
    }

    /// The image height revealed strictly above `line`.
    fn images_before(&self, line: usize) -> i32 {
        self.entries[..self.len]
            .iter()
            .filter(|&&(l, _)| l < line)
            .map(|&(_, e)| e)
            .sum()
    }
}

/// The query-time overlay on a [`LineIndex`]'s base geometry: the image sources
/// revealed right now, and the sections folded away. The index measures a document
/// as if nothing were revealed and nothing folded, which is what keeps it a pure
/// function of content and width; both deltas are applied here instead, so a
/// reveal or a fold toggle costs a query rather than a re-measure.
#[derive(Clone, Copy)]
pub struct Adjust<'a> {
    pub reveal: RevealAdjust,
    pub folds: &'a FoldMap,
}

impl<'a> Adjust<'a> {
    pub fn new(reveal: RevealAdjust, folds: &'a FoldMap) -> Self {
        Self { reveal, folds }
    }
}

/// The document's vertical geometry as a per-logical-line index: each line's base
/// pixel height and the cross-line carry state entering it. Built by measuring
/// every line once via [`lay_line`], it computes total content height, maps a
/// logical line to a content-space position, and seeds a windowed layout that
/// starts mid-document, all without keeping the whole document's runs. A cursor
/// move or scroll reuses the index unchanged (the base heights are collapsed ones,
/// and what a reveal adds is priced per line and applied by [`Adjust`]); only an
/// edit or resize rebuilds it.
#[derive(Clone)]
pub struct LineIndex {
    recs: Vec<LineRec>,
    /// Cumulative base height: `tops[i]` is the content-space top of logical line
    /// `i` (0-relative); `tops[len]` is the total base height.
    tops: Vec<i32>,
    /// Cumulative [`LineRec::wrap_extra`], indexed the same way, so the re-wrap a
    /// reveal costs over any run of lines is one subtraction rather than a walk. A
    /// selection may reveal the whole document, and a scroll queries the geometry
    /// per frame, so this stays O(1).
    wrap_tops: Vec<i32>,
}

impl LineIndex {
    /// Measure one logical line `i`'s record, advancing the carry state
    /// (`fence`/`list_depth`) through it. Shared by [`Self::build`] and
    /// [`Self::update`] so the incremental path can never disagree with a rebuild.
    #[allow(clippy::too_many_arguments)]
    fn measure_line(
        ctx: &LayoutCtx,
        i: usize,
        n: usize,
        carry: &mut LineCarry,
        pool: &mut LayoutScratch,
        tables: &mut TableMemo,
        scratch: &mut Vec<VisualLine>,
    ) -> LineRec {
        let logical = LogicalLine::at(ctx.rope, i, n);
        let carry_in = *carry;
        scratch.clear();
        let laid = lay_line(ctx, logical, i, 0, carry, pool, tables, true, scratch);
        // Read the heading level off the measured rows before the reveal pass
        // below reuses `scratch`. A wrapped heading repeats its level on every row.
        let heading_level = scratch.iter().find_map(|v| v.heading).unwrap_or(0);
        // A decoded standalone image is the one line whose reveal adds whole rows of
        // its own (its editable source), so it is measured revealed here. Re-running
        // from the pre-line carry reproduces the same state mutation (an image forces
        // list_depth=None, fence must be None), so the throwaway `revealed_carry` is
        // discarded. Every other line's reveal only re-wraps, which `lay_line`
        // already priced.
        let image_extra = if scratch.iter().any(|v| v.image.is_some()) {
            let mut revealed_carry = carry_in;
            let revealed_ctx = ctx.with_active((logical.start, logical.end));
            scratch.clear();
            let revealed = lay_line(
                &revealed_ctx,
                logical,
                i,
                0,
                &mut revealed_carry,
                pool,
                tables,
                true,
                scratch,
            );
            revealed.height - laid.height
        } else {
            0
        };
        LineRec {
            height: laid.height,
            image_extra,
            wrap_extra: laid.wrap_extra,
            fence_in: carry_in.fence.map_or(0, |f| f.packed()),
            depth_in: carry_in.list_depth.map_or(-1, |d| d as i16),
            heading: heading_level,
        }
    }

    /// Build the cumulative prefixes (`tops`, `wrap_tops`) from a record list.
    fn from_recs(recs: Vec<LineRec>) -> Self {
        let mut tops = Vec::with_capacity(recs.len() + 1);
        let mut wrap_tops = Vec::with_capacity(recs.len() + 1);
        let (mut acc, mut wrap_acc) = (0, 0);
        for rec in &recs {
            tops.push(acc);
            wrap_tops.push(wrap_acc);
            acc += rec.height;
            wrap_acc += rec.wrap_extra;
        }
        tops.push(acc);
        wrap_tops.push(wrap_acc);
        LineIndex {
            recs,
            tops,
            wrap_tops,
        }
    }

    /// Measure every logical line's base geometry and carry state. `images` is the
    /// pane's decoded-image store, so a standalone `![](...)` line is recognized as
    /// a picture (and its `image_extra` recorded) exactly as the full build sees it.
    pub fn build(
        fonts: &Fonts,
        rope: &Rope,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
        scale: Scale,
    ) -> Self {
        let n = rope.len_lines();
        let ctx = LayoutCtx::for_index(fonts, rope, wrap_width, body_size, images, scale);
        let out = measure_chunk(&ctx, 0, n, n, LineCarry::default());
        Self::from_recs(out.recs)
    }

    /// Build the index across up to `threads` worker threads, for a large
    /// document where the sequential [`build`](Self::build) would stall (a huge
    /// file open, a resize on a long file). Each worker lays out a contiguous
    /// chunk of lines with its own [`Fonts`] (FreeType faces are not shareable),
    /// speculating that its chunk begins in ground state (no open code fence, no
    /// list). A short sequential pass then stitches the chunks in order, accepting
    /// a chunk whose ground-state guess held and re-measuring only the rare chunk
    /// a fence or list carried across its boundary. Plain prose starts every chunk
    /// in ground state, so nothing is re-measured and the build scales with the
    /// core count.
    ///
    /// The result is field-for-field identical to [`build`](Self::build) (a golden
    /// test pins this); only the scheduling differs. Falls back to a sequential
    /// build for a small document, a single thread, or if a worker cannot open its
    /// own fonts.
    pub fn build_parallel(
        fonts: &Fonts,
        rope: &Rope,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
        scale: Scale,
        threads: usize,
    ) -> Self {
        let n = rope.len_lines();
        let threads = threads.max(1);
        if threads == 1 || n < PARALLEL_MIN_LINES {
            return Self::build(fonts, rope, wrap_width, body_size, images, scale);
        }
        let ctx = LayoutCtx::for_index(fonts, rope, wrap_width, body_size, images, scale);
        Self::build_chunked(&ctx, threads)
    }

    /// The chunked build behind [`build_parallel`](Self::build_parallel), without
    /// the small-document threshold, so a test can force many small chunks to
    /// exercise the stitch and fence/list fix-up paths.
    fn build_chunked(ctx: &LayoutCtx, threads: usize) -> Self {
        debug_assert!(
            ctx.active == NO_REVEAL,
            "build_chunked measures base index geometry; its ctx must be reveal-free"
        );
        let n = ctx.rope.len_lines();
        let chunk = n.div_ceil(threads);
        let ranges: Vec<(usize, usize)> = (0..threads)
            .map(|t| (t * chunk, ((t + 1) * chunk).min(n)))
            .filter(|&(lo, hi)| lo < hi)
            .collect();
        // Copy out the ctx's `Send` fields so each worker rebuilds its ctx from
        // these rather than capturing `ctx` itself: `Fonts` is `!Sync` (FreeType
        // faces are not shareable), so `&LayoutCtx` is not `Send` and cannot cross
        // the thread boundary. Each worker opens its own `fonts` and rebuilds a
        // reveal-free ctx via `for_index`, so the base-geometry precondition holds
        // by construction rather than by copying `active`/`image_ui` across.
        let rope = ctx.rope;
        let wrap_width = ctx.wrap_width;
        let body_size = ctx.body_size;
        let images = ctx.images;
        let scale = ctx.scale;

        // Measure every chunk from ground state. The main thread takes the first
        // chunk with the caller's fonts; each other chunk gets a worker that opens
        // its own fonts (`None` if it cannot, which forces the fallback below).
        let outs: Vec<Option<ChunkOut>> = std::thread::scope(|s| {
            let handles: Vec<_> = ranges
                .iter()
                .skip(1)
                .map(|&(lo, hi)| {
                    s.spawn(move || {
                        let fonts = Fonts::new(&markdown::all_sizes(body_size)).ok()?;
                        let worker_ctx = LayoutCtx::for_index(
                            &fonts, rope, wrap_width, body_size, images, scale,
                        );
                        Some(measure_chunk(&worker_ctx, lo, hi, n, LineCarry::default()))
                    })
                })
                .collect();
            let (lo0, hi0) = ranges[0];
            let mut outs = vec![Some(measure_chunk(ctx, lo0, hi0, n, LineCarry::default()))];
            for h in handles {
                outs.push(h.join().ok().flatten());
            }
            outs
        });

        // A worker that could not open fonts leaves a hole; rather than a partial
        // index, fall back to a full sequential build.
        if outs.iter().any(Option::is_none) {
            return Self::build(
                ctx.fonts,
                ctx.rope,
                ctx.wrap_width,
                ctx.body_size,
                ctx.images,
                ctx.scale,
            );
        }
        let outs: Vec<ChunkOut> = outs.into_iter().flatten().collect();

        // Stitch in order: accept a chunk whose ground-state guess matched its true
        // entry state, and re-measure (sequentially, with the caller's fonts) any
        // chunk a fence or list carried into.
        let mut recs: Vec<LineRec> = Vec::with_capacity(n);
        let mut carry = LineCarry::default();
        for (out, &(lo, hi)) in outs.iter().zip(&ranges) {
            if carry == LineCarry::default() {
                recs.extend_from_slice(&out.recs);
                carry = out.end;
            } else {
                let re = measure_chunk(ctx, lo, hi, n, carry);
                recs.extend_from_slice(&re.recs);
                carry = re.end;
            }
        }
        Self::from_recs(recs)
    }

    /// Incrementally bring the index up to date after an edit turned `prev` into
    /// `new`, re-measuring only the changed region instead of the whole document.
    ///
    /// The changed byte span is found by a byte-cheap line diff (matching prefix
    /// from the front, matching suffix from the back). Re-measurement runs forward
    /// from the first changed line until the carry state entering a line re-syncs
    /// with the old index past the changed text: from there on, identical text fed
    /// identical state yields identical records, so the untouched tail is reused
    /// (only its position shifts, recomputed by `from_recs`). A change that never
    /// re-syncs (an unterminated fence flipping every following line) re-measures to
    /// EOF, exactly as a full rebuild would. If `prev` does not match this index,
    /// it falls back to a full rebuild.
    pub fn update(&mut self, ctx: &LayoutCtx, prev: &Rope) {
        debug_assert!(
            ctx.active == NO_REVEAL,
            "LineIndex::update measures base geometry; its ctx must be reveal-free"
        );
        let new = ctx.rope;
        let new_n = new.len_lines();
        let old_n = self.recs.len();
        if old_n != prev.len_lines() {
            *self = Self::build(
                ctx.fonts,
                new,
                ctx.wrap_width,
                ctx.body_size,
                ctx.images,
                ctx.scale,
            );
            return;
        }
        // Find the changed byte span cheaply (shared leaves compared as raw bytes,
        // never materialized), then map it to a conservative changed line range.
        let prefix = new.common_prefix_len(prev);
        let suffix = new.common_suffix_len(prev, new.len().min(prev.len()) - prefix);
        let front = new.byte_to_line(prefix);
        // Trailing lines strictly after the one holding the suffix boundary are
        // unchanged; the boundary line itself is re-measured (conservative). Clamp
        // so both the new and old resync indices stay valid.
        let suffix_start = new.len() - suffix;
        let back = (new_n - 1)
            .saturating_sub(new.byte_to_line(suffix_start))
            .min(new_n.saturating_sub(front))
            .min(old_n.saturating_sub(front));
        let delta = new_n as isize - old_n as isize;
        // First line of the unchanged suffix; re-measurement may converge here.
        let resync = new_n - back;

        let (fence, list_depth) = self.entry_state(front);
        let mut carry = LineCarry::new(fence, list_depth);
        let mut mid: Vec<LineRec> = Vec::new();
        let mut scratch: Vec<VisualLine> = Vec::new();
        let mut pool = LayoutScratch::default();
        let mut tables = TableMemo::default();
        let mut i = front;
        let converge_old = loop {
            let old_i = i as isize - delta;
            // Converge only once past the changed text (i in the suffix) and the
            // incoming carry state matches the old index there; then the rest is
            // provably identical and reused.
            if i >= resync && old_i >= front as isize && (old_i as usize) < old_n {
                let old_state = self.entry_state(old_i as usize);
                if carry.pair() == old_state {
                    break old_i as usize;
                }
            }
            if i >= new_n {
                break old_n;
            }
            mid.push(Self::measure_line(
                ctx,
                i,
                new_n,
                &mut carry,
                &mut pool,
                &mut tables,
                &mut scratch,
            ));
            i += 1;
        };

        let mut recs = Vec::with_capacity(front + mid.len() + old_n.saturating_sub(converge_old));
        recs.extend_from_slice(&self.recs[..front]);
        recs.append(&mut mid);
        recs.extend_from_slice(&self.recs[converge_old..]);
        *self = Self::from_recs(recs);
    }

    // A document always has at least one logical line, so there is no meaningful
    // empty index; `len` stands alone.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.recs.len()
    }

    /// The heap bytes the index holds: the per-line records and their prefix sums.
    /// For the memory report.
    pub fn heap_bytes(&self) -> usize {
        self.recs.capacity() * std::mem::size_of::<LineRec>()
            + (self.tops.capacity() + self.wrap_tops.capacity()) * std::mem::size_of::<i32>()
    }

    /// Total content height, less what the overlay's folds hide and plus everything
    /// it reveals.
    pub fn content_height(&self, adjust: Adjust) -> i32 {
        self.tops[self.recs.len()] - adjust.folds.hidden_total()
            + self.revealed_before(self.recs.len(), adjust)
    }

    /// Content-space top of logical line `line` (0-relative), pulled up by every
    /// fold above it and pushed down by everything revealed above it. A line inside
    /// a fold resolves to the position the fold itself occupies, so all of a hidden
    /// section shares one top and the result stays monotonic in `line` (which is
    /// what lets [`Self::line_at_y`] binary search it).
    pub fn logical_top(&self, line: usize, adjust: Adjust) -> i32 {
        let line = line.min(self.recs.len());
        let base = match adjust.folds.collapsed_top(line) {
            Some((start, hidden_above)) => self.tops[start] - hidden_above,
            None => self.tops[line] - adjust.folds.hidden_before(line),
        };
        base + self.revealed_before(line, adjust)
    }

    /// The height the current reveal adds strictly above `line`: the image sources
    /// shown there, plus the re-wrap of every active line above it that a fold does
    /// not hide (a hidden line draws nothing, so its markup reflows nothing).
    fn revealed_before(&self, line: usize, adjust: Adjust) -> i32 {
        let images = adjust.reveal.images_before(line);
        let Some((lo, hi)) = adjust.reveal.active else {
            return images;
        };
        // `hi` is inclusive; the prefix sums want a half-open range, clamped to the
        // lines that lie above `line` and inside the document.
        let end = (hi + 1).min(line).min(self.recs.len());
        if end <= lo {
            return images;
        }
        let hidden: i32 = adjust
            .folds
            .hidden_spans_in(lo, end)
            .map(|(s, e)| self.wrap_tops[e] - self.wrap_tops[s])
            .sum();
        images + (self.wrap_tops[end] - self.wrap_tops[lo]) - hidden
    }

    /// The logical line whose row band contains content-space `y`: the last line
    /// whose top is `<= y`, clamped into range. Turns a scroll offset into a window
    /// band edge. `logical_top` is monotonic in `line`, so a binary search finds it.
    pub fn line_at_y(&self, y: i32, adjust: Adjust) -> usize {
        let n = self.recs.len();
        if n == 0 {
            return 0;
        }
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.logical_top(mid, adjust) <= y {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo.saturating_sub(1)
    }

    /// The extra height logical line `line` gains when its image source is revealed
    /// (0 for a non-image line). The pane uses this to build a [`RevealAdjust`].
    pub fn image_extra(&self, line: usize) -> i32 {
        self.recs.get(line).map_or(0, |r| r.image_extra)
    }

    /// The cross-line carry state entering logical line `line`: the open fence (its
    /// backtick count and language) and the list nesting depth. Used to seed a
    /// mid-document window and to detect re-sync during an incremental update.
    fn entry_state(&self, line: usize) -> (Option<Fence>, Option<usize>) {
        match self.recs.get(line) {
            Some(rec) => (
                Fence::unpack(rec.fence_in),
                (rec.depth_in >= 0).then_some(rec.depth_in as usize),
            ),
            None => (None, None),
        }
    }

    /// [`Self::entry_state`] in its stored form, for a [`crate::fold::FoldMap`] to
    /// carry across a span it hides and hand back to the layout walk.
    pub fn entry_packed(&self, line: usize) -> (u16, i16) {
        self.recs
            .get(line)
            .map_or((0, -1), |rec| (rec.fence_in, rec.depth_in))
    }

    /// The ATX heading level of logical line `line`, or `None` where it is not a
    /// heading. Measured with the fence carry in hand, so a `#` inside a code block
    /// is not one.
    pub fn heading_at(&self, line: usize) -> Option<u8> {
        self.recs
            .get(line)
            .map(|rec| rec.heading)
            .filter(|&l| l > 0)
    }

    /// Base height of logical lines `[start, end)`, the pixels a fold over that
    /// range removes from the document.
    pub fn span_height(&self, start: usize, end: usize) -> i32 {
        let n = self.recs.len();
        let (start, end) = (start.min(n), end.min(n));
        self.tops[end.max(start)] - self.tops[start]
    }
}

/// Display size for an image of intrinsic size `iw` x `ih` shown in a `column`
/// of pixels: the intrinsic size when it already fits, otherwise scaled down to
/// the column width with its aspect ratio preserved. Never upscaled (a small
/// image keeps its size rather than blurring to fill the column). Integer math,
/// in `i64` so the intermediate product cannot overflow.
fn fit_image(iw: u32, ih: u32, column: i32) -> (i32, i32) {
    let (iw, ih) = (iw.max(1) as i64, ih.max(1) as i64);
    let column = column.max(1) as i64;
    let (w, h) = if iw <= column {
        (iw, ih)
    } else {
        (column, (ih * column / iw).max(1))
    };
    (w as i32, h as i32)
}

/// The toggle button's rectangle, inset into the top-right corner of a picture
/// `image` rect. Shared by the painter and hit-testing so a click lands where the
/// button is drawn.
pub(crate) fn image_button_rect(image: &Rect, scale: Scale) -> Rect {
    let (w, h, margin) = (
        scale.px(IMAGE_BUTTON_W),
        scale.px(IMAGE_BUTTON_H),
        scale.px(IMAGE_BUTTON_MARGIN),
    );
    Rect {
        x: image.x + image.w - w - margin,
        y: image.y + margin,
        w,
        h,
    }
}

/// Whether content-space point `(x, y)` lies within `rect`.
fn rect_contains(rect: &Rect, x: f32, y: f32) -> bool {
    x >= rect.x as f32
        && x < (rect.x + rect.w) as f32
        && y >= rect.y as f32
        && y < (rect.y + rect.h) as f32
}

/// Resolve a line's block style against the running list nesting `list_depth`
/// (the depth of the most recent valid list item, or `None` outside a list),
/// returning the effective style, its block-marker length, and whether the line
/// is a list continuation. A list item nests by leading indentation
/// ([`markdown::LIST_INDENT_SPACES`] per level); it only renders as a list item
/// when its parent, one level shallower, is present above it, otherwise it falls
/// back to plain body text (marker length `0`). An indented plain line under an
/// open list continues the current item (a Shift+Enter continuation): the nesting
/// is kept and the line is flagged so the layout insets it under the item text
/// with no bullet. Blank lines leave the nesting untouched; any other non-list
/// line ends it.
fn resolve_list(line: &str, list_depth: &mut Option<usize>) -> (markdown::LineStyle, usize, bool) {
    let (style, marker) = markdown::classify(line);
    match style {
        markdown::LineStyle::Bullet
        | markdown::LineStyle::Ordered
        | markdown::LineStyle::Task { .. } => {
            let depth = markdown::leading_indent(line) / markdown::LIST_INDENT_SPACES;
            let allowed = list_depth.map_or(0, |d| d + 1);
            if depth <= allowed {
                *list_depth = Some(depth);
                (style, marker, false)
            } else {
                // Over-nested: no parent one level up, so render as plain text.
                *list_depth = None;
                (markdown::LineStyle::Body, 0, false)
            }
        }
        markdown::LineStyle::Body
            if !line.trim().is_empty()
                && list_depth.is_some()
                && markdown::leading_indent(line) >= markdown::LIST_INDENT_SPACES =>
        {
            // Indented plain text under an open list: a continuation of the current
            // item. Keep the nesting so a following sibling still nests correctly.
            (markdown::LineStyle::Body, 0, true)
        }
        other => {
            if !line.trim().is_empty() {
                *list_depth = None;
            }
            (other, marker, false)
        }
    }
}

/// Fill `out` with the full inline span list of a logical `line`, line-relative:
/// a leading block marker of `marker` bytes (a heading's `#`s, a list `- `, etc.,
/// if any) followed by the inline spans of the content past it, offset back over
/// the marker. Tiles `[0, line.len())`. `marker` is passed in (not recomputed) so
/// a caller can suppress it, e.g. an over-nested list item that renders as plain
/// body text gets `marker == 0`. Reuses the pooled `out` buffer rather than
/// allocating a fresh Vec per line.
fn line_spans_into(
    line: &str,
    marker: usize,
    inline: &mut markdown::InlineScratch,
    out: &mut Vec<markdown::Span>,
) {
    out.clear();
    if marker > 0 {
        out.push(markdown::Span {
            start: 0,
            end: marker,
            kind: markdown::SpanKind::Marker,
        });
    }
    // Append the content's spans (content-relative), then shift them past the
    // marker in place, so no intermediate Vec is allocated for them.
    let base = out.len();
    markdown::inline_spans_append(&line[marker..], inline, out);
    for span in &mut out[base..] {
        span.start += marker;
        span.end += marker;
    }
}

/// Fill `out` with the spans of a line inside a fenced code block: a whole-line
/// code run, or, on a fence delimiter line (the opening or closing ```), a
/// whole-line marker run so the fence and any info string read as dimmed syntax.
/// No inline parsing happens inside a code block. An empty line has no runs (the
/// panel still draws). Reuses the pooled `out` buffer.
fn code_block_spans_into(
    line: &str,
    delimiter: bool,
    lang: crate::syntax::Lang,
    out: &mut Vec<markdown::Span>,
) {
    out.clear();
    if line.is_empty() {
        return;
    }
    if delimiter {
        // The ``` fence line (with its info string) is one whole-line marker, drawn
        // dimmed as syntax and hidden off the cursor's own line.
        out.push(markdown::Span {
            start: 0,
            end: line.len(),
            kind: markdown::SpanKind::Marker,
        });
    } else {
        // A code line is tokenized into colored spans; an unknown language yields
        // a single plain span, the same flat look as before highlighting.
        crate::syntax::lex_into(lang, line, out);
    }
}

/// Clip line-relative `spans` to the visual segment `[seg_start, seg_end)`,
/// offsetting to absolute by `ls`, marking each run visible per [`LineClass::draws`]
/// and giving it its pill pads. Empty overlaps are dropped, so the runs tile the
/// segment. `line` is the logical line the spans index into.
#[allow(clippy::too_many_arguments)]
fn clip_runs_into(
    spans: &[markdown::Span],
    line: &str,
    seg_start: usize,
    seg_end: usize,
    ls: usize,
    class: &LineClass,
    out: &mut Vec<Run>,
) {
    let base = out.len();
    for span in spans {
        let s = span.start.max(seg_start);
        let e = span.end.min(seg_end);
        if s >= e {
            continue;
        }
        out.push(Run {
            start: ls + s,
            end: ls + e,
            kind: span.kind,
            visible: class.draws(span.kind, span.start),
            pads: PillPads::default(),
        });
    }
    assign_pill_pads(&mut out[base..], line, ls, class.code_block);
}

/// Give each run the sides on which it carries an inline code span's pill pad. A
/// code run takes the pad on every side its delimiter does not cover, and a revealed
/// backtick takes it on its outer side, so the pad always falls outside the
/// delimiters and a span reserves the same total width either way. `runs` is one
/// visual row's slice, `line` the logical line at buffer offset `ls`. A fenced line
/// draws no pill and so reserves nothing.
fn assign_pill_pads(runs: &mut [Run], line: &str, ls: usize, code_block: bool) {
    if code_block {
        return;
    }
    let text_of = |r: &Run| -> &str {
        r.start
            .checked_sub(ls)
            .zip(r.end.checked_sub(ls))
            .and_then(|(s, e)| line.get(s..e))
            .unwrap_or("")
    };
    let is_code =
        |r: &Run| r.visible && matches!(r.kind, markdown::SpanKind::Text(markdown::Emphasis::Code));
    // A delimiter is a shown marker of nothing but backticks, the run the inline
    // parser emits on each side of a code span.
    let is_tick = |r: &Run| {
        r.visible
            && matches!(r.kind, markdown::SpanKind::Marker)
            && !text_of(r).is_empty()
            && text_of(r).bytes().all(|b| b == b'`')
    };
    for i in 0..runs.len() {
        let before = i.checked_sub(1).map(|j| runs[j]);
        let after = runs.get(i + 1).copied();
        let run = runs[i];
        runs[i].pads = if is_code(&run) {
            PillPads {
                lead: !before.is_some_and(|r| is_tick(&r)),
                trail: !after.is_some_and(|r| is_tick(&r)),
            }
        } else if is_tick(&run) {
            PillPads {
                lead: after.is_some_and(|r| is_code(&r)),
                trail: before.is_some_and(|r| is_code(&r)),
            }
        } else {
            PillPads::default()
        };
    }
}

/// One measured grapheme cluster of the line being wrapped.
#[derive(Clone, Copy)]
struct Cluster {
    /// Byte offset within the line.
    off: usize,
    /// Advance in the cluster's own face, with an inline code span's pill pads
    /// folded onto the span's first cluster. What the cluster costs a row that shows
    /// it; a row that collapses it (see `drawn`) is charged nothing.
    w: f32,
    /// Whether the row draws this cluster: false for markup the line collapses,
    /// which takes no width until the line is revealed.
    drawn: bool,
    /// A space, so a wrap may break after it.
    space: bool,
    /// Markup rather than text: a block marker (`> `, `# `, `- `) or an inline
    /// delimiter. It collapses off the cursor's line, so a segment holding nothing
    /// but markers draws an empty row.
    marker: bool,
    /// A separator inside a link, so a URL with no space in it may break after this
    /// cluster instead of mid-token.
    url_sep: bool,
}

/// The punctuation a URL may break after: every ASCII separator RFC 3986 builds a
/// URL from. What lies between them is a name, which breaks no better than any
/// other word.
const URL_SEPARATORS: &[u8] = b"/-_.~:?#[]@!$&'()*+,;=%";

/// Which widths a wrap measures.
#[derive(Clone, Copy, PartialEq)]
enum WrapBasis {
    /// What the row draws: markup the line collapses measures zero, so a row is
    /// wrapped to the text the reader sees.
    Drawn,
    /// The source in full, markup included. A table cell wraps this way, so the
    /// column width fitted from its natural (source) width and the wrap inside it
    /// agree, and a table's geometry does not shift as the caret enters a row. It is
    /// also what a revealed line costs, which is how the reveal's reflow is priced.
    Source,
}

/// Greedily word-wrap the logical `line` to `wrap_width` pixels, filling `segments`
/// with the byte ranges of its visual rows and returning how many rows the same line
/// takes once its markup is revealed (equal to `segments.len()` on an already
/// revealed line, and under [`WrapBasis::Source`]).
///
/// Breaks after the last space that fits and has text before it; a word wider than
/// `wrap_width` breaks after the last URL separator inside a link, else at a
/// grapheme boundary, so every row holds at least one cluster and no row is markup
/// alone. `wrap_width <= 0` disables wrapping. Each cluster is measured in its run's
/// face (per `spans` and `size`), so bold/italic widths wrap correctly.
///
/// `block_indent` is the hanging indent of a list or quote item: the first
/// segment uses the full `wrap_width` (its marker sits in that width), and every
/// continuation is wrapped `block_indent` pixels narrower, since it renders inset
/// by that much. So the text column is the same width on every line of the item.
#[allow(clippy::too_many_arguments)]
fn wrap_segments(
    fonts: &Fonts,
    size: u32,
    line: &str,
    spans: &[markdown::Span],
    class: &LineClass,
    basis: WrapBasis,
    wrap_width: i32,
    block_indent: i32,
    heading: bool,
    scale: Scale,
    clusters: &mut Vec<Cluster>,
    segments: &mut Vec<(usize, usize)>,
) -> usize {
    segments.clear();
    if wrap_width <= 0 || line.is_empty() {
        segments.push((0, line.len()));
        return 1;
    }
    // Measure each cluster once. The pooled buffer is cleared and refilled, so it
    // reuses its capacity across lines instead of allocating one sized to the line
    // on every call.
    clusters.clear();
    let mut prev_code = false;
    // Whether anything on the line is collapsed, so revealing it could reflow. A
    // plain line (most of a document) has nothing to reveal and skips the second
    // pass entirely.
    let mut collapses = false;
    clusters.extend(grapheme::graphemes(line).map(|(off, cluster)| {
        let kind = kind_at(spans, off);
        let face = run_face(fonts, size, kind, heading);
        let mut w = shape::text_advance(face, cluster);
        // Fold both of an inline code span's pill pads onto its first cluster, so a
        // wrap keeps the pill's full width on one line. The exact placement within
        // the run does not change the total width the line has to fit. A fenced
        // block draws no pill, so it reserves no pad.
        let is_code = matches!(kind, markdown::SpanKind::Text(markdown::Emphasis::Code));
        if !class.code_block && is_code && !prev_code {
            w += (2 * scale.px(crate::theme::CODE_PAD_X)) as f32;
        }
        prev_code = is_code;
        let drawn = basis == WrapBasis::Source || class.draws(kind, off);
        collapses |= !drawn;
        Cluster {
            off,
            w,
            drawn,
            space: cluster == " ",
            marker: matches!(kind, markdown::SpanKind::Marker),
            url_sep: matches!(kind, markdown::SpanKind::Text(markdown::Emphasis::Link))
                && matches!(cluster.as_bytes(), [b] if URL_SEPARATORS.contains(b)),
        }
    }));
    if clusters.is_empty() {
        segments.push((0, line.len()));
        return 1;
    }
    let rows = wrap_rows(
        clusters,
        basis,
        line.len(),
        wrap_width,
        block_indent,
        Some(segments),
    );
    // What the line would cost with its markup shown: a second walk over clusters
    // already measured, so the reflow a reveal causes is priced without laying the
    // line out twice. Nothing collapsed means nothing to reflow.
    if collapses {
        wrap_rows(
            clusters,
            WrapBasis::Source,
            line.len(),
            wrap_width,
            block_indent,
            None,
        )
    } else {
        rows
    }
}

/// One greedy pass over the measured `clusters` on `basis`, filling rows to
/// `wrap_width` and returning the row count. `out`, when given, collects each row's
/// byte range; without it the pass only counts, which is how the reveal's reflow is
/// priced.
fn wrap_rows(
    clusters: &[Cluster],
    basis: WrapBasis,
    line_len: usize,
    wrap_width: i32,
    block_indent: i32,
    mut out: Option<&mut Vec<(usize, usize)>>,
) -> usize {
    let mut rows = 0;
    let mut i = 0;
    while i < clusters.len() {
        // The first segment gets the full width; continuations are narrower by
        // the hanging indent, since they render inset by it.
        let limit = (wrap_width - if rows == 0 { 0 } else { block_indent }).max(1) as f32;
        let seg_start = clusters[i].off;
        let mut width = 0.0;
        // Cluster index that would start the next segment if we break at the
        // last fitting space; `None` until a space is seen in this segment.
        let mut last_break: Option<usize> = None;
        // The same, for the last URL separator that fits. A long link holds no
        // space to break at, so it breaks after its last `/` or `-` instead of
        // mid-token. Only reached when there is no space break, so a link that
        // would fit whole on the next row still moves there in one piece.
        let mut last_url_break: Option<usize> = None;
        // Whether the segment has any text to keep yet. A space before there is any
        // is no break opportunity: breaking there would leave a segment of markup
        // that draws nothing, so `> <long url>` would hand the quote's first row to
        // the collapsed `> ` and drop the URL, blank row and all, onto the next.
        let mut text = false;
        let mut j = i;
        while j < clusters.len() {
            let c = clusters[j];
            let w = if c.drawn || basis == WrapBasis::Source {
                c.w
            } else {
                0.0
            };
            // Always keep at least one cluster (j > i) before allowing a break.
            if width + w > limit && j > i {
                break;
            }
            width += w;
            j += 1;
            match (c.space, c.marker) {
                (true, _) if text => last_break = Some(j),
                (false, false) => {
                    text = true;
                    if c.url_sep {
                        last_url_break = Some(j);
                    }
                }
                _ => {}
            }
        }
        rows += 1;
        if j >= clusters.len() {
            if let Some(out) = out.as_mut() {
                out.push((seg_start, line_len));
            }
            break;
        }
        // Break after the last space if there was one, else after the last URL
        // separator, else mid-word at j.
        let next = match (last_break, last_url_break) {
            (Some(b), _) if b > i => b,
            (_, Some(b)) if b > i => b,
            _ => j,
        };
        if let Some(out) = out.as_mut() {
            out.push((seg_start, clusters[next].off));
        }
        i = next;
    }
    rows
}

// --- Table layout -----------------------------------------------------------
//
// A markdown table lays out as a run of widget lines, one per source row (the
// header and each body row; the delimiter row is structural and draws nothing).
// Column widths are shared across the whole table, so the block is measured once to
// fit them, then each row wraps its own cells to those widths and takes the tallest
// cell as its height. The block is recovered from any of its lines by a bounded rope
// scan, so a row's geometry is a pure function of the table text and the wrap width,
// and the windowed index needs no cross-line table state.
//
// Both of those are per-table work reached from a per-line path, so a build carries
// a `TableMemo` beside its scratch: it holds the one table being walked, and every
// row of that table reads the detection and the fitted widths the first row paid for.

/// Where a queried line sits within a table block.
enum TableRole {
    Header,
    Delimiter,
    Body,
}

/// A table block's logical-line extent: the header row, the delimiter row below it,
/// and one past the last body row.
#[derive(Clone, Copy)]
struct TableSpan {
    header: usize,
    delim: usize,
    end: usize,
}

impl TableSpan {
    /// The drawn rows in order: the header, then every body row. The delimiter is
    /// skipped, since it renders nothing.
    fn drawn_rows(&self) -> impl Iterator<Item = usize> {
        std::iter::once(self.header).chain(self.delim + 1..self.end)
    }

    /// Where logical line `i` sits in this block.
    fn role(&self, i: usize) -> TableRole {
        if i == self.header {
            TableRole::Header
        } else if i == self.delim {
            TableRole::Delimiter
        } else {
            TableRole::Body
        }
    }
}

/// The one table a build is walking: the block it last located, that block's column
/// alignments, and the fitted column geometry once a row has asked for it. A build
/// visits a table's rows in order, so holding a single table collapses what was a
/// block re-detection and a re-measure of every cell in the table, per row.
///
/// Created and dropped with the [`LayoutScratch`] beside it, so it lives for exactly
/// one build. The rope cannot change inside a build; the geometry can (the diff view
/// lays the same text out at two widths through one build), so the stored widths
/// carry the geometry they were fitted under and a change re-measures.
#[derive(Default)]
struct TableMemo {
    block: Option<MemoBlock>,
}

/// The located block: its extent, its per-column alignments, and its fitted columns
/// keyed by the geometry they were measured under.
struct MemoBlock {
    span: TableSpan,
    aligns: Vec<markdown::Align>,
    columns: Option<(ColumnKey, Vec<Column>)>,
}

/// The geometry [`table_columns`] reads besides the table's own text: the wrap width,
/// the body font size, and the display scale. Everything else it touches (the fonts,
/// the rope) is fixed for a build.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ColumnKey {
    wrap_width: i32,
    body_size: u32,
    scale_120: u32,
}

impl TableMemo {
    /// Point the memo at the table containing logical line `i` and report the line's
    /// role in it, or `None` when the line is not in a table. A line inside the block
    /// already held costs nothing; anything else re-detects and drops the old block's
    /// measurements. The extent comes from [`crate::table::block_at`] (shared with
    /// the editing model, so caret navigation and rendering agree on a table's
    /// bounds); the alignments are resolved here for layout.
    fn locate(&mut self, rope: &Rope, i: usize) -> Option<(TableSpan, TableRole)> {
        if let Some(held) = &self.block {
            if i >= held.span.header && i < held.span.end {
                return Some((held.span, held.span.role(i)));
            }
        }
        let block = crate::table::block_at(rope, i)?;
        let text = |k: usize| {
            let l = LogicalLine::at(rope, k, rope.len_lines());
            rope.slice(l.start..l.end).as_cow()
        };
        let aligns = markdown::table_head(&text(block.header), &text(block.delim))?;
        let span = TableSpan {
            header: block.header,
            delim: block.delim,
            end: block.end,
        };
        self.block = Some(MemoBlock {
            span,
            aligns,
            columns: None,
        });
        Some((span, span.role(i)))
    }

    /// The shared column geometry of the block the memo holds, fitted to `ctx`'s wrap
    /// width, measured on the first row that asks and reused by the rest. Empty when
    /// no block is held (`locate` returned `None`), which no caller reaches.
    fn columns(&mut self, ctx: &LayoutCtx, pool: &mut LayoutScratch) -> &[Column] {
        let Some(block) = &mut self.block else {
            return &[];
        };
        let key = ColumnKey {
            wrap_width: ctx.wrap_width,
            body_size: ctx.body_size,
            scale_120: ctx.scale.factor_120(),
        };
        if block.columns.as_ref().is_none_or(|(k, _)| *k != key) {
            let columns = table_columns(ctx, block.span, &block.aligns, pool);
            block.columns = Some((key, columns));
        }
        block.columns.as_ref().map_or(&[], |(_, c)| c.as_slice())
    }
}

/// The shared, positioned column geometry of the block at `span`, fitted to the wrap
/// width. Each column's natural width is its widest unwrapped cell; if the columns
/// overflow the available width they shrink proportionally, never past
/// [`TABLE_MIN_COL`], so wide cells wrap. Measures every cell of the table, so it is
/// reached through [`TableMemo::columns`] rather than called per row.
fn table_columns(
    ctx: &LayoutCtx,
    span: TableSpan,
    aligns: &[markdown::Align],
    pool: &mut LayoutScratch,
) -> Vec<Column> {
    let n = ctx.rope.len_lines();
    let cols = aligns.len();
    let mut natural = vec![0i32; cols];
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for row in span.drawn_rows() {
        let l = LogicalLine::at(ctx.rope, row, n);
        let line = ctx.rope.slice(l.start..l.end).as_cow();
        markdown::table_row_cells(&line, &mut ranges);
        let header = row == span.header;
        for (c, &(s, e)) in ranges.iter().take(cols).enumerate() {
            let w = cell_measure_width(ctx, &line[s..e], header, pool);
            natural[c] = natural[c].max(w);
        }
    }
    let border = ctx.scale.px(TABLE_BORDER);
    let cell_pad_x = ctx.scale.px(TABLE_CELL_PAD_X);
    let min_col = ctx.scale.px(TABLE_MIN_COL);
    let border_pad = (cols as i32 + 1) * border + cols as i32 * 2 * cell_pad_x;
    let avail = if ctx.wrap_width > 0 {
        (ctx.wrap_width - border_pad).max(cols as i32 * min_col)
    } else {
        natural.iter().sum::<i32>().max(1)
    };
    let widths = fit_columns(&natural, avail, min_col);
    let mut out = Vec::with_capacity(cols);
    let mut x = border;
    for (&w, &align) in widths.iter().zip(aligns) {
        let cx = x + cell_pad_x;
        let w = w.max(1);
        out.push(Column {
            x: cx,
            width: w,
            align,
        });
        x = cx + w + cell_pad_x + border;
    }
    out
}

/// The left x of a cell line within its column, honoring the column's alignment:
/// left sits at the content edge, right and center shift by the free space. Shared
/// by the painter (drawing) and the caret/hit-test (positioning) so they agree.
pub(crate) fn aligned_x(col: Column, line_w: i32) -> i32 {
    match col.align {
        markdown::Align::Left => col.x,
        markdown::Align::Right => col.x + col.width - line_w,
        markdown::Align::Center => col.x + (col.width - line_w) / 2,
    }
}

/// Vertical offset that centers a cell's `n_lines`-line block within a row whose
/// tallest cell has `max_lines` lines. Zero when the cell already fills the row.
fn cell_block_pad(n_lines: usize, max_lines: usize, line_height: i32) -> i32 {
    max_lines.saturating_sub(n_lines) as i32 * line_height / 2
}

/// The vertical offset from a table row's content top (`box_top + TABLE_CELL_PAD_Y`)
/// to the top of cell line `k`'s glyph box. It centers the cell's block within the
/// row (a short cell sits in the middle of a taller row), then centers each line's
/// glyphs within their line-height slot, so the leading slack splits above and below
/// instead of all falling below and leaving the text riding high. Shared by the
/// painter, the caret, and the click hit-test so the three stay in agreement.
pub(crate) fn cell_line_top(k: usize, n_lines: usize, max_lines: usize, m: Metrics) -> i32 {
    cell_block_pad(n_lines, max_lines, m.line_height) + k as i32 * m.line_height + leading_pad(m)
}

/// Half the leading a taller line box adds around the glyph box: the slack above
/// (and below) ascent + descent once the surplus is split evenly. Centering a
/// full-line-box fill on the text means starting it this far above the glyph box.
pub(crate) fn leading_pad(m: Metrics) -> i32 {
    (m.line_height - (m.ascent + m.descent)) / 2
}

/// The count of the tallest cell's wrapped lines in a laid-out row, recovered from
/// its box height (`max_lines * line_height + 2 * pad`).
pub(crate) fn row_max_lines(height: i32, line_height: i32, scale: Scale) -> usize {
    (((height - 2 * scale.px(TABLE_CELL_PAD_Y)) / line_height).max(1)) as usize
}

/// Fit natural column widths to `avail` pixels: keep them if they fit, else shrink
/// each in proportion to its share, clamped to a floor so none vanishes.
fn fit_columns(natural: &[i32], avail: i32, min_col: i32) -> Vec<i32> {
    let total: i32 = natural.iter().sum();
    if total <= avail || total == 0 {
        return natural.to_vec();
    }
    natural
        .iter()
        .map(|&w| ((w as i64 * avail as i64 / total as i64) as i32).max(min_col))
        .collect()
}

/// Lay out one table row (`row_line`, a header or body row of the block `tables`
/// holds) at content-space top `box_top`, appending its widget line to `out` (unless
/// `measure_only`) and returning the row's box height. Cells wrap to the shared
/// column widths; the row is as tall as its tallest cell.
#[allow(clippy::too_many_arguments)]
fn lay_table_row(
    ctx: &LayoutCtx,
    span: TableSpan,
    row_line: usize,
    box_top: i32,
    measure_only: bool,
    pool: &mut LayoutScratch,
    tables: &mut TableMemo,
    out: &mut Vec<VisualLine>,
) -> i32 {
    let n = ctx.rope.len_lines();
    let size = ctx.body_size;
    let metrics = ctx.fonts.metrics(size);
    // The whole table's fitted widths, measured by whichever of its rows got here
    // first. `tables` stays borrowed for the cell loop below; `pool` is a separate
    // value, so the cells still take it mutably.
    let columns = tables.columns(ctx, pool);
    let cols = columns.len();
    let l = LogicalLine::at(ctx.rope, row_line, n);
    let line = ctx.rope.slice(l.start..l.end).as_cow();
    let header = row_line == span.header;
    // The cursor is on this row: reveal its cells' raw markers, the same live-preview
    // test an ordinary line uses, so the grid stays but the backticks become visible.
    let revealed = ctx.active.0 <= l.end && ctx.active.1 >= l.start;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    markdown::table_row_cells(&line, &mut ranges);
    let mut cells = Vec::with_capacity(cols);
    let mut max_lines = 1usize;
    for (c, col) in columns.iter().enumerate() {
        // A row shorter than the header pads with empty cells; a longer one is
        // truncated to the column count.
        let (s, e) = ranges.get(c).copied().unwrap_or((line.len(), line.len()));
        let cell = build_cell(
            ctx,
            &line[s..e],
            l.start + s,
            l.start + e,
            col.width,
            header,
            revealed,
            measure_only,
            pool,
        );
        max_lines = max_lines.max(cell.lines.len().max(1));
        cells.push(cell);
    }
    let height = max_lines as i32 * metrics.line_height + 2 * ctx.scale.px(TABLE_CELL_PAD_Y);
    if !measure_only {
        // Vertical grid lines: the left outer edge, each column separator (just past
        // a column's content and right padding), and the right outer edge.
        let mut borders = Vec::with_capacity(columns.len() + 1);
        borders.push(0);
        for col in columns {
            borders.push(col.x + col.width + ctx.scale.px(TABLE_CELL_PAD_X));
        }
        let width = borders.last().copied().unwrap_or(0) + ctx.scale.px(TABLE_BORDER);
        out.push(VisualLine {
            start: l.start,
            end: l.end,
            // Baseline sits at the row's top plus the ascent, so `baseline - ascent`
            // is the box top (the widget convention images use); the painter applies
            // the vertical cell padding when placing text.
            baseline: box_top + metrics.ascent,
            hard: l.has_newline,
            size,
            runs: 0..0,
            indent: 0,
            gutter: Gutter::None,
            code_block: false,
            heading: None,
            fold: None,
            task: None,
            rule: false,
            image: None,
            table: Some(TableRow {
                header,
                first: row_line == span.header,
                // The drawn row keeps its own copy of the shared geometry, so the
                // painter and the hit-test read a line's columns off the line.
                columns: columns.to_vec(),
                cells,
                borders,
                width,
                height,
            }),
        });
    }
    height
}

/// The pieces of a cell's text between `<br>` hard breaks, as byte ranges into it.
/// A cell with no `<br>` yields one piece spanning the whole text, so the common
/// case is unchanged; each `<br>` (the in-cell line break Shift+Enter inserts)
/// forces the next piece onto its own line.
fn br_pieces(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    const BR: &str = "<br>";
    let mut start = 0;
    let mut done = false;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        match text[start..].find(BR) {
            Some(rel) => {
                let at = start + rel;
                let piece = (start, at);
                start = at + BR.len();
                Some(piece)
            }
            None => {
                done = true;
                Some((start, text.len()))
            }
        }
    })
}

/// The unwrapped width of a cell's source text, measured the way [`build_cell`]
/// lays it out: each span in its own face, so a code span counts in the wider
/// monospace face it draws in, not the prose face, plus its pill pad. The markers a
/// revealed row shows (backticks, `*`) are counted too, so the column reserves their
/// room and revealing a row never reflows it. A `<br>` splits the cell; the widest
/// piece wins. Mirrors [`wrap_segments`]' own measurement (`pad_code` true), so a
/// column sized to it never forces a cell to wrap against its natural width.
fn cell_measure_width(
    ctx: &LayoutCtx,
    cell_text: &str,
    header: bool,
    pool: &mut LayoutScratch,
) -> i32 {
    let size = ctx.body_size;
    let mut max_w = 0.0f32;
    for (ps, pe) in br_pieces(cell_text) {
        let piece = &cell_text[ps..pe];
        pool.spans.clear();
        markdown::inline_spans_append(piece, &mut pool.inline, &mut pool.spans);
        // Sum each span in its own face. text_advance sums per-cluster advances with
        // no cross-cluster kerning, so a whole span measures the same as wrap_segments'
        // per-cluster walk, and a plain cell costs the one measure it always did. A
        // code span reserves its pill pad on each side, matching the run-walks.
        let mut w = 0.0;
        for span in &pool.spans {
            let face = run_face(ctx.fonts, size, span.kind, header);
            w += shape::text_advance(face, &piece[span.start..span.end]);
            w += 2.0 * inline_code_pad(span.kind, false, ctx.scale) as f32;
        }
        max_w = max_w.max(w);
    }
    max_w.ceil() as i32
}

/// Lay out one table cell: its inline markup wrapped to the column `width`, as a
/// list of drawn lines. A `<br>` splits the cell onto a fresh line; header cells
/// draw in the bold face. When `revealed` (the cursor is on this row) the cell's
/// markers show raw so the backticks and emphasis are editable, matching an ordinary
/// active line; off the row they collapse. On `measure_only` the runs and widths are
/// skipped (the caller wants only the wrapped line count for the row height),
/// matching the index's measure pass.
#[allow(clippy::too_many_arguments)]
fn build_cell(
    ctx: &LayoutCtx,
    cell_text: &str,
    start: usize,
    end: usize,
    width: i32,
    header: bool,
    revealed: bool,
    measure_only: bool,
    pool: &mut LayoutScratch,
) -> TableCell {
    let size = ctx.body_size;
    // Off the cursor's row a cell collapses its markers like any off-cursor line;
    // on the row (`revealed`) they show raw so the backticks and emphasis are
    // editable. The cell carries no block marker of its own.
    let class = LineClass {
        style: markdown::LineStyle::Body,
        size,
        block_marker: 0,
        block_persists: false,
        revealed,
        code_block: false,
        list_depth: 0,
        list_continuation: false,
    };
    let mut lines = Vec::new();
    for (ps, pe) in br_pieces(cell_text) {
        let piece = &cell_text[ps..pe];
        let piece_abs = start + ps;
        pool.spans.clear();
        markdown::inline_spans_append(piece, &mut pool.inline, &mut pool.spans);
        wrap_segments(
            ctx.fonts,
            size,
            piece,
            &pool.spans,
            &class,
            // A column is fitted to its cells' natural (source) widths, so the wrap
            // inside one measures the source too. Otherwise a row's cells would
            // re-wrap as the caret entered it, under a column width fitted for text
            // that is no longer what the cell draws.
            WrapBasis::Source,
            width,
            0,
            header,
            ctx.scale,
            &mut pool.clusters,
            &mut pool.segments,
        );
        for k in 0..pool.segments.len() {
            let (ss, se) = pool.segments[k];
            let mut runs = Vec::new();
            if !measure_only {
                clip_runs_into(&pool.spans, piece, ss, se, piece_abs, &class, &mut runs);
            }
            let width = if measure_only {
                0
            } else {
                cell_line_width(ctx, size, &runs, header)
            };
            lines.push(CellLine { runs, width });
        }
    }
    TableCell { start, end, lines }
}

/// The drawn width of a cell line: the sum of its visible runs' advances, in the
/// faces they draw in, each widened by the pill pads it carries. Feeds the painter's
/// right/center alignment.
fn cell_line_width(ctx: &LayoutCtx, size: u32, runs: &[Run], header: bool) -> i32 {
    let mut w = 0.0;
    for run in runs {
        if !run.visible {
            continue;
        }
        let (lead, trail) = run.pads.px(ctx.scale);
        let face = run_face(ctx.fonts, size, run.kind, header);
        w += (lead + trail) as f32
            + shape::text_advance(face, &ctx.rope.slice(run.start..run.end).as_cow());
    }
    w.round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::Folds;
    use crate::markdown::LineStyle;
    use crate::theme::MARGIN;

    const BODY: u32 = 32;
    const LEFT: i32 = 40;
    const TOP: i32 = 10;

    /// The geometry overlay with nothing revealed, so a test reads base heights
    /// under `folds` alone.
    fn base_adjust(folds: &FoldMap) -> Adjust<'_> {
        Adjust::new(RevealAdjust::default(), folds)
    }

    fn fonts() -> Fonts {
        Fonts::new(&markdown::all_sizes(BODY)).expect("a default font")
    }

    /// Fonts at a chosen line-height scale, so a test can exercise the leading the
    /// editor's 1.5 spacing adds below the glyph box. Plain `fonts` runs at scale
    /// 1.0, where line_height equals ascent + descent and the surplus is invisible.
    fn fonts_scaled(scale: f32) -> Fonts {
        Fonts::with_config(
            &crate::platform::freetype::FontConfig::default(),
            &markdown::all_sizes(BODY),
            scale,
        )
        .expect("a default font")
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

    /// Lay out the whole document the way [`Layout::build_range`] does, but with a
    /// fresh [`TableMemo`] per line, so a table's rows share no measurement: the
    /// shape the build had before the memo existed.
    fn build_unmemoized(ctx: &LayoutCtx, left: i32, top: i32) -> Layout {
        let n = ctx.rope.len_lines();
        let mut pool = LayoutScratch::default();
        let mut carry = LineCarry::default();
        let mut lines = Vec::new();
        let mut box_top = top;
        for i in 0..n {
            let logical = LogicalLine::at(ctx.rope, i, n);
            let mut tables = TableMemo::default();
            box_top += lay_line(
                ctx,
                logical,
                i,
                box_top,
                &mut carry,
                &mut pool,
                &mut tables,
                false,
                &mut lines,
            )
            .height;
        }
        Layout {
            lines,
            runs: pool.runs,
            left,
            text_height: box_top - top,
            scale: ctx.scale,
        }
    }

    /// A ctx over `r` at `wrap`, reveal-free, with no images and no folds.
    fn table_ctx<'a>(
        f: &'a Fonts,
        r: &'a Rope,
        images: &'a ImageStore,
        wrap: i32,
    ) -> LayoutCtx<'a> {
        LayoutCtx::new(
            f,
            r,
            wrap,
            BODY,
            NO_REVEAL,
            images,
            ImageUi::default(),
            FoldMap::NONE,
            Scale::ONE,
        )
    }

    /// The total x extent of a table row's columns, which is what the fitted widths
    /// move when the geometry changes.
    fn columns_span(v: &VisualLine) -> i32 {
        let t = v.table.as_ref().expect("a table row");
        t.columns.last().map_or(0, |c| c.x + c.width)
    }

    #[test]
    fn a_table_lays_out_the_same_with_and_without_the_memo() {
        // The memo may only save work, never change output: every row of the memoized
        // build must be pixel-identical to one measured on its own. Two tables of
        // different natural widths, so a leaked measurement would show.
        let f = fonts();
        let images = ImageStore::new(None);
        let r = Rope::from_str(&format!(
            "intro\n\n{}\nbetween\n\n| x | y |\n| - | - |\n| 1 | 2 |\n\nafter\n",
            table_doc(6)
        ));
        for &w in &[240, 500, 1_000_000] {
            let ctx = table_ctx(&f, &r, &images, w);
            assert_eq!(
                fingerprint(&Layout::build_with_images(&ctx, LEFT, TOP)),
                fingerprint(&build_unmemoized(&ctx, LEFT, TOP)),
                "memoized and per-row builds must agree (width {w})"
            );
        }
    }

    #[test]
    fn a_table_memo_survives_a_geometry_change() {
        // The stored widths carry the geometry they were fitted under, so the same
        // table laid out at a second wrap width through the *same* memo re-measures
        // instead of handing back the first width's columns.
        let f = fonts();
        let images = ImageStore::new(None);
        let r = Rope::from_str(&table_doc(3));
        let mut pool = LayoutScratch::default();
        let mut tables = TableMemo::default();
        let n = r.len_lines();
        let header = LogicalLine::at(&r, 0, n);
        let lay = |wrap: i32, pool: &mut LayoutScratch, tables: &mut TableMemo| {
            let ctx = table_ctx(&f, &r, &images, wrap);
            let mut out = Vec::new();
            let mut carry = LineCarry::default();
            lay_line(
                &ctx, header, 0, 0, &mut carry, pool, tables, false, &mut out,
            );
            out
        };
        let wide = lay(1_000_000, &mut pool, &mut tables);
        let narrow = lay(300, &mut pool, &mut tables);
        assert!(
            columns_span(&narrow[0]) < columns_span(&wide[0]),
            "the narrow build must refit: {} vs {}",
            columns_span(&narrow[0]),
            columns_span(&wide[0])
        );
        // And the refit is the one a fresh memo would produce.
        let fresh = lay(300, &mut pool, &mut TableMemo::default());
        assert_eq!(columns_span(&narrow[0]), columns_span(&fresh[0]));
    }

    #[test]
    fn two_tables_in_one_document_each_get_their_own_columns() {
        // Walking off one table and onto another must drop the first table's widths:
        // each row lands under its own table's geometry. A wide wrap keeps both at
        // their natural widths, so the two are far apart.
        let f = fonts();
        let images = ImageStore::new(None);
        let r = Rope::from_str(
            "| short | s |\n| - | - |\n| a | b |\n\n\
             | a much wider header cell | and another wide one |\n| - | - |\n\
             | with equally wide body text in it | and more of the same here |\n",
        );
        let ctx = table_ctx(&f, &r, &images, 1_000_000);
        let l = Layout::build_with_images(&ctx, LEFT, TOP);
        let rows: Vec<&VisualLine> = l.lines().iter().filter(|v| v.table.is_some()).collect();
        assert_eq!(rows.len(), 4, "two header rows and two body rows");
        let (narrow, wide) = (columns_span(rows[0]), columns_span(rows[2]));
        assert_eq!(columns_span(rows[1]), narrow, "the first table's body row");
        assert_eq!(columns_span(rows[3]), wide, "the second table's body row");
        assert!(
            narrow < wide,
            "each table keeps its own width: {narrow} vs {wide}"
        );
    }

    /// A generated table document of `rows` body rows and three columns.
    fn table_doc(rows: usize) -> String {
        let mut s = String::from("| head one | head two | head three |\n| --- | --- | --- |\n");
        for i in 0..rows {
            s.push_str(&format!("| cell {i} a | cell {i} b | cell {i} c |\n"));
        }
        s
    }

    #[test]
    fn image_button_rect_scales_exactly_with_the_display() {
        let pic = Rect {
            x: 0,
            y: 0,
            w: 400,
            h: 300,
        };
        let one = image_button_rect(&pic, Scale::ONE);
        let two = image_button_rect(&pic, Scale::from_120(240));
        // The button doubles in size at 2x, and its inset from the top-right corner
        // doubles with it, so it stays pinned to the picture's corner.
        assert_eq!((two.w, two.h), (2 * one.w, 2 * one.h));
        assert_eq!(pic.x + pic.w - two.x, 2 * (pic.x + pic.w - one.x));
        assert_eq!(two.y, 2 * one.y);
    }

    #[test]
    fn table_geometry_scales_with_the_display() {
        let f = fonts();
        let r = Rope::from_str("| a | b |\n| - | - |\n| c | d |\n");
        let images = ImageStore::new(None);
        // Same font size both times, so only the scaled cell padding, borders, and
        // column geometry move the grid width; the display scale flows in through the
        // ctx. A wide wrap so the columns keep their natural (unshrunk) widths.
        let build_at = |scale: Scale| {
            let ctx = LayoutCtx::new(
                &f,
                &r,
                1_000_000,
                BODY,
                (0, 0),
                &images,
                ImageUi::default(),
                FoldMap::NONE,
                scale,
            );
            Layout::build_with_images(&ctx, LEFT, TOP)
        };
        let width_of = |l: &Layout| {
            l.lines()
                .iter()
                .find_map(|v| v.table.as_ref().map(|t| t.width))
                .expect("a table row")
        };
        let one = width_of(&build_at(Scale::ONE));
        let two = width_of(&build_at(Scale::from_120(240)));
        assert!(
            two > one,
            "table width scales with the display: {two} vs {one}"
        );
    }

    #[test]
    fn inline_code_pad_scales_with_the_display() {
        let code = markdown::SpanKind::Text(markdown::Emphasis::Code);
        let one = inline_code_pad(code, false, Scale::ONE);
        assert!(one > 0);
        assert_eq!(inline_code_pad(code, false, Scale::from_120(240)), 2 * one);
        // A fenced-block code run draws no pill, so no pad at any scale.
        assert_eq!(inline_code_pad(code, true, Scale::from_120(240)), 0);
    }

    /// A rope from a string literal, for the geometry tests.
    fn rope(s: &str) -> Rope {
        Rope::from_str(s)
    }

    /// A wrap width that fits "aaa " but not a following cluster, so "aaa bbb"
    /// splits deterministically into "aaa " and "bbb".
    fn tight_wrap(f: &Fonts) -> i32 {
        let face = f.face(BODY, FontStyle::Regular);
        (shape::text_advance(face, "aaa ") + shape::text_advance(face, "b") / 2.0) as i32
    }

    /// A pixel-complete fingerprint of one visual line: every field that reaches
    /// the screen. Two layouts with equal fingerprints render identically, so this
    /// is the oracle every later stage (the windowed build, the incremental index)
    /// and any future allocation refactor is diffed against.
    #[derive(Clone, PartialEq, Debug)]
    struct LineFp {
        start: usize,
        end: usize,
        baseline: i32,
        hard: bool,
        size: u32,
        indent: i32,
        gutter: Gutter,
        code_block: bool,
        heading: Option<u8>,
        task: Option<bool>,
        rule: bool,
        image: Option<(usize, usize, i32, i32, bool)>,
        runs: Vec<(usize, usize, markdown::SpanKind, bool)>,
        /// A table row's pixel-complete shape: header/first flags, width, height,
        /// each column's (x, width, align), and every cell's wrapped lines as their
        /// runs. `None` off table rows.
        table: Option<TableFp>,
    }

    /// One run in a fingerprinted cell line: byte range, kind, visibility, and its
    /// line's drawn width.
    type CellRunFp = (usize, usize, markdown::SpanKind, bool, i32);

    #[derive(Clone, PartialEq, Debug)]
    struct TableFp {
        header: bool,
        first: bool,
        width: i32,
        height: i32,
        borders: Vec<i32>,
        columns: Vec<(i32, i32, markdown::Align)>,
        /// Per cell, per wrapped line, the line's runs.
        cells: Vec<Vec<Vec<CellRunFp>>>,
    }

    fn fingerprint(l: &Layout) -> Vec<LineFp> {
        l.lines()
            .iter()
            .map(|v| LineFp {
                start: v.start,
                end: v.end,
                baseline: v.baseline,
                hard: v.hard,
                size: v.size,
                indent: v.indent,
                gutter: v.gutter,
                code_block: v.code_block,
                heading: v.heading,
                task: v.task,
                rule: v.rule,
                image: v
                    .image
                    .map(|im| (im.path_start, im.path_end, im.width, im.height, im.selected)),
                runs: l
                    .run_slice(v)
                    .iter()
                    .map(|r| (r.start, r.end, r.kind, r.visible))
                    .collect(),
                table: v.table.as_ref().map(|t| TableFp {
                    header: t.header,
                    first: t.first,
                    width: t.width,
                    height: t.height,
                    borders: t.borders.clone(),
                    columns: t.columns.iter().map(|c| (c.x, c.width, c.align)).collect(),
                    cells: t
                        .cells
                        .iter()
                        .map(|cell| {
                            cell.lines
                                .iter()
                                .map(|cl| {
                                    cl.runs
                                        .iter()
                                        .map(|r| (r.start, r.end, r.kind, r.visible, cl.width))
                                        .collect()
                                })
                                .collect()
                        })
                        .collect(),
                }),
            })
            .collect()
    }

    /// Curated fixtures that exercise the constructs the index and windowing must
    /// get right (nested lists, fences closed and unterminated, quotes, tasks,
    /// rules, CRLF, trailing-newline, inline spans), paired with the seeded corpus
    /// generators for scale.
    const FIXTURES: &[&str] = &[
        "",
        "plain paragraph with no markup at all here",
        "# Heading one\n\nbody text\n\n## Two\n\nmore\n",
        "- a\n- b\n  - nested\n  - deeper still and long enough to wrap somewhere\n- c\n",
        "1. one\n2. two\n3. three\n",
        "- [ ] todo item\n- [x] done item\n",
        "> a quote long enough to wrap across a narrow column of pixels here now\n",
        "```\ncode line\nmore code\n```\nafter the fence\n",
        "```rust\nunterminated fence runs to EOF\nstill code\nand more\n",
        "---\n***\n___\n",
        "a\nb\nc\nd\ne\nf\ng\nh\n",
        "trailing line with no newline",
        "windows\r\nline\r\nendings\r\n",
        "mix **bold** and *italic* and `code` and [link](x) inline here\n",
        "| Name | Role | Score |\n| --- | :---: | ---: |\n| Ada | engineer who writes prose long enough to wrap a cell | 92 |\n| Linus | kernel | 100 |\n",
        "before the table\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\nafter the table\n",
    ];

    /// Every fixture plus each seeded corpus (kept small for fast tests) as a doc
    /// to diff layout variants against.
    fn oracle_docs() -> Vec<Rope> {
        let mut docs: Vec<Rope> = FIXTURES.iter().map(|s| Rope::from_str(s)).collect();
        for name in crate::dev::corpus::NAMES {
            if let Some(text) = crate::dev::corpus::generate(name, 12 * 1024) {
                docs.push(Rope::from_str(&text));
            }
        }
        docs
    }

    /// Representative wrap widths: 0 disables wrapping, the rest span tight to wide.
    const ORACLE_WIDTHS: &[i32] = &[0, 140, 300, 560, 1_000_000];

    #[test]
    fn fingerprint_is_deterministic_across_builds() {
        // Idempotence: the same inputs must fingerprint identically, so a later
        // stage's "unchanged output" assertion is meaningful and nondeterminism
        // (e.g. a hash-order leak) is caught here first.
        let f = fonts();
        for r in oracle_docs() {
            for &w in ORACLE_WIDTHS {
                let a = fingerprint(&Layout::build(&f, &r, LEFT, TOP, w, BODY, (0, 0)));
                let b = fingerprint(&Layout::build(&f, &r, LEFT, TOP, w, BODY, (0, 0)));
                assert_eq!(a, b, "layout build must be deterministic (width {w})");
            }
        }
    }

    #[test]
    fn the_reveal_overlay_matches_a_revealed_build() {
        // The index measures collapsed lines, so revealing one re-wraps it. What the
        // overlay adds for that must be exactly what a revealed build lays out, or a
        // scroll position drifts the moment the caret lands on a link. Checked with
        // the caret on a spread of lines, and with the whole document selected.
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            let n = r.len_lines();
            let spots: Vec<(usize, usize)> = (0..n)
                .step_by((n / 6).max(1))
                .map(|l| (l, l))
                .chain(std::iter::once((0, n - 1)))
                .collect();
            for &w in ORACLE_WIDTHS {
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                for &(lo, hi) in &spots {
                    let active = (r.line_content_range(lo).0, r.line_content_range(hi).1);
                    let full = Layout::build(&f, &r, LEFT, TOP, w, BODY, active);
                    let mut reveal = RevealAdjust::default();
                    reveal.reveal_lines(lo, hi);
                    let adjust = Adjust::new(reveal, FoldMap::NONE);
                    assert_eq!(
                        idx.content_height(adjust),
                        full.text_height(),
                        "content height revealing lines {lo}..={hi} (width {w})"
                    );
                    let mut last = None;
                    for v in full.lines() {
                        let li = r.byte_to_line(v.start);
                        if Some(li) != last {
                            let top = v.baseline
                                - f.metrics(v.size).ascent
                                - heading_pad(&f, v, li)
                                - TOP;
                            assert_eq!(
                                idx.logical_top(li, adjust),
                                top,
                                "top of line {li} revealing {lo}..={hi} (width {w})"
                            );
                            last = Some(li);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_revealed_link_line_costs_the_rows_its_source_adds() {
        // The concrete case the overlay exists for: collapsed, the line is one row
        // of drawn text; revealed, the destination shows and pushes it to more.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let src = "see [docs](https://example.com/a/very/long/path/goes/on/forever) ok\n";
        let r = rope(src);
        let w = wrap_after(&f, "see docs ok and some room to spare");
        let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
        let collapsed = Layout::build(&f, &r, LEFT, TOP, w, BODY, NO_REVEAL);
        assert_eq!(
            collapsed.lines().len(),
            2,
            "collapsed, the drawn text fits one row (plus the empty last line)"
        );
        let revealed = Layout::build(&f, &r, LEFT, TOP, w, BODY, (0, 0));
        assert!(
            revealed.lines().len() > collapsed.lines().len(),
            "the raw destination pushes the line onto more rows"
        );
        let mut reveal = RevealAdjust::default();
        reveal.reveal_lines(0, 0);
        assert_eq!(
            idx.content_height(Adjust::new(reveal, FoldMap::NONE)),
            revealed.text_height(),
            "the overlay prices exactly the rows the reveal added"
        );
        assert_eq!(
            idx.content_height(base_adjust(FoldMap::NONE)),
            collapsed.text_height(),
            "and the base geometry stays the collapsed one"
        );
    }

    #[test]
    fn a_folded_lines_reveal_costs_nothing() {
        // A selection can run across a folded section. Those lines draw nothing, so
        // their markup reflows nothing and the overlay must not charge for it.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let src = "# H\nsee [docs](https://example.com/a/very/long/path/goes/on) ok\n# H2\n";
        let r = rope(src);
        let w = wrap_after(&f, "see docs ok and some room to spare");
        let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
        let map = FoldMap::build(&crate::fold::Folds::from_lines([0]), &idx);
        assert!(map.hides(1), "the link line is inside the folded section");
        let mut reveal = RevealAdjust::default();
        reveal.reveal_lines(0, r.len_lines() - 1);
        assert!(
            idx.content_height(Adjust::new(reveal, FoldMap::NONE))
                > idx.content_height(base_adjust(FoldMap::NONE)),
            "unfolded, the same reveal does cost rows"
        );
        assert_eq!(
            idx.content_height(Adjust::new(reveal, &map)),
            idx.content_height(base_adjust(&map)),
            "a hidden line's re-wrap is not part of the document's height"
        );
    }

    #[test]
    fn line_index_matches_the_full_build() {
        // Index oracle: the index's base heights reproduce the full build's
        // total height and every logical line's top, across corpora and widths.
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            for &w in ORACLE_WIDTHS {
                let full = Layout::build(&f, &r, LEFT, TOP, w, BODY, NO_REVEAL);
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                assert_eq!(idx.len(), r.len_lines(), "one record per logical line");
                assert_eq!(
                    idx.content_height(base_adjust(FoldMap::NONE)),
                    full.text_height(),
                    "content height (width {w})"
                );
                // The first visual row of each logical line sits at the index's top,
                // less any space the line opens above itself (a heading's), which is
                // part of the logical line's extent but sits above its text box.
                let mut last = None;
                for v in full.lines() {
                    let li = r.byte_to_line(v.start);
                    if Some(li) != last {
                        let top =
                            v.baseline - f.metrics(v.size).ascent - heading_pad(&f, v, li) - TOP;
                        assert_eq!(
                            idx.logical_top(li, base_adjust(FoldMap::NONE)),
                            top,
                            "top of logical line {li} (width {w})"
                        );
                        last = Some(li);
                    }
                }
            }
        }
    }

    /// Build the whole document with `map`'s sections folded away.
    fn folded_layout(f: &Fonts, r: &Rope, w: i32, map: &FoldMap) -> Layout {
        let images = ImageStore::new(None);
        let ctx = LayoutCtx::new(
            f,
            r,
            w,
            BODY,
            NO_REVEAL,
            &images,
            ImageUi::default(),
            map,
            Scale::ONE,
        );
        Layout::build_with_images(&ctx, LEFT, TOP)
    }

    /// The space a heading on logical line `li` should open above itself, recomputed
    /// from the theme rule so a test checks the value rather than trusting the code.
    fn heading_pad(f: &Fonts, v: &VisualLine, li: usize) -> i32 {
        match v.heading {
            Some(_) if li > 0 => (f.metrics(v.size).line_height as f32
                * crate::theme::HEADING_SPACE_ABOVE)
                .round() as i32,
            _ => 0,
        }
    }

    /// The blank pixels between visual row `i`'s box and the one above it: what the
    /// reader actually sees as a gap. `0` for the first row.
    fn gap_above(f: &Fonts, l: &Layout, i: usize) -> i32 {
        if i == 0 {
            return 0;
        }
        let (prev, cur) = (&l.lines()[i - 1], &l.lines()[i]);
        let pm = f.metrics(prev.size);
        let prev_bottom = prev.baseline - pm.ascent + pm.line_height;
        (cur.baseline - f.metrics(cur.size).ascent) - prev_bottom
    }

    /// Every heading in a document, for a test that wants to fold them all (maximal
    /// nesting, so spans merge) or a thinned-out subset.
    fn heading_lines(idx: &LineIndex, every: usize) -> Vec<usize> {
        (0..idx.len())
            .filter(|&l| idx.heading_at(l).is_some())
            .step_by(every)
            .collect()
    }

    #[test]
    fn a_folded_build_is_the_unfolded_build_less_its_hidden_lines() {
        // The fold oracle. Folding must not re-flow anything: every row the folded
        // build emits is byte-for-byte the row the unfolded build emitted for that
        // logical line, only lifted by the height hidden above it. And no row for a
        // hidden line survives. Checked across the corpora, both wrap widths, and
        // two anchor sets (every heading, so spans nest and merge; every third, so
        // folded and open sections interleave).
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            for &w in ORACLE_WIDTHS {
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                let unfolded = folded_layout(&f, &r, w, FoldMap::NONE);
                for every in [1, 3] {
                    let anchors = heading_lines(&idx, every);
                    if anchors.is_empty() {
                        continue;
                    }
                    let map = FoldMap::build(&Folds::from_lines(anchors), &idx);
                    let folded = folded_layout(&f, &r, w, &map);

                    // The document loses exactly the height of what it hides.
                    assert_eq!(
                        folded.text_height(),
                        unfolded.text_height() - map.hidden_total(),
                        "folded height (width {w}, every {every})"
                    );

                    // The surviving rows, with each baseline lifted by the height
                    // hidden above its logical line.
                    let mut want = Vec::new();
                    for fp in fingerprint(&unfolded) {
                        let li = r.byte_to_line(fp.start);
                        if map.hides(li) {
                            continue;
                        }
                        want.push(LineFp {
                            baseline: fp.baseline - map.hidden_before(li),
                            ..fp
                        });
                    }
                    assert_eq!(
                        fingerprint(&folded),
                        want,
                        "folded rows (width {w}, every {every})"
                    );
                }
            }
        }
    }

    #[test]
    fn a_folded_window_matches_the_folded_full_build() {
        // The windowed path jumps a hidden span whole and resumes its cross-line
        // carry from the index, so a code fence or list that opened inside a fold
        // still governs the visible lines below it. Every band of a folded document
        // must therefore reproduce the folded full build's rows over that band.
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            for w in [300, 1_000_000] {
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                let anchors = heading_lines(&idx, 2);
                if anchors.is_empty() {
                    continue;
                }
                let map = FoldMap::build(&Folds::from_lines(anchors), &idx);
                let full = folded_layout(&f, &r, w, &map);
                let full_fp = fingerprint(&full);
                let adjust = base_adjust(&map);
                let n = idx.len();
                let images = ImageStore::new(None);
                let ctx = LayoutCtx::new(
                    &f,
                    &r,
                    w,
                    BODY,
                    NO_REVEAL,
                    &images,
                    ImageUi::default(),
                    &map,
                    Scale::ONE,
                );
                for (start, end) in [(0, n / 3), (n / 3, 2 * n / 3), (2 * n / 3, n)] {
                    if start >= end {
                        continue;
                    }
                    let seed = TOP + idx.logical_top(start, adjust);
                    let win = Layout::build_window(&ctx, LEFT, seed, &idx, start, end);
                    // The full build's rows over the same logical band.
                    let want: Vec<LineFp> = full_fp
                        .iter()
                        .filter(|fp| (start..end).contains(&r.byte_to_line(fp.start)))
                        .cloned()
                        .collect();
                    assert_eq!(
                        fingerprint(&win),
                        want,
                        "folded window {start}..{end} (width {w})"
                    );
                }
            }
        }
    }

    #[test]
    fn a_fence_opened_inside_a_fold_still_closes_the_lines_below_it() {
        // The carry hazard, pinned on its own so a failure names it. The fence opens
        // inside `## Hidden`'s section and never closes, so every line after it is
        // code. Folding that section skips those lines entirely; without the span's
        // recorded exit carry the build would resume thinking no fence was open and
        // render the tail as prose.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let text = "# Doc\n## Hidden\n```rust\nlet x = 1;\n## still code\ntail code\n";
        let r = rope(text);
        let w = 1_000_000;
        let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
        // The `##` on line 4 is inside the fence, so it is code, not a heading, and
        // does not end the section: `## Hidden` runs to the end of the document.
        assert_eq!(idx.heading_at(4), None);
        let map = FoldMap::build(&Folds::from_lines([1]), &idx);
        assert!(map.hides(2) && map.hides(5));

        // Nothing is left to check below the fold in this document, so verify the
        // carry directly: a window seeded past the span sees the open fence.
        let unfolded = folded_layout(&f, &r, w, FoldMap::NONE);
        assert!(
            unfolded
                .lines()
                .iter()
                .filter(|v| r.byte_to_line(v.start) >= 3)
                .all(|v| v.code_block),
            "every line after the fence is code"
        );
        let folded = folded_layout(&f, &r, w, &map);
        assert_eq!(
            folded.lines().len(),
            2,
            "only `# Doc` and the folded `## Hidden` survive"
        );
        assert_eq!(
            folded.lines()[1].fold,
            Some(true),
            "the heading reads folded"
        );
    }

    #[test]
    fn a_heading_carries_a_fold_marker_only_where_it_owns_a_section() {
        let f = fonts();
        // `## Empty` is followed by a sibling, so it owns nothing and offers no
        // fold; `## Full` owns the body line under it.
        let r = rope("# Top\n## Empty\n## Full\nbody\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        let marks: Vec<Option<bool>> = l.lines().iter().map(|v| v.fold).collect();
        assert_eq!(
            marks,
            // The trailing newline leaves a final empty line, which is what `## Full`
            // owns and why it offers a fold at all.
            vec![Some(false), None, Some(false), None, None],
            "a chevron only where there is a section to collapse"
        );
    }

    #[test]
    fn a_fold_chevron_sits_left_of_the_text_column() {
        // The no-reflow invariant for the margin affordance: the chevron hangs
        // entirely left of `left`, so showing or hiding it never moves a glyph.
        let f = fonts();
        let r = rope("# Heading\nbody\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        let heading = &l.lines()[0];
        let rect = l.fold_rect(&f, heading, BODY).expect("a foldable heading");
        assert!(
            rect.x + rect.w <= l.left(),
            "chevron right edge {} must not reach the text column at {}",
            rect.x + rect.w,
            l.left()
        );
        assert!(
            rect.x >= l.left() - MARGIN,
            "chevron stays inside the margin"
        );
        // A body line offers none.
        assert!(l.fold_rect(&f, &l.lines()[1], BODY).is_none());
    }

    #[test]
    fn a_fold_chevron_is_centered_in_the_heading_line() {
        // It must sit on the line's middle, not the middle of its ascent, which
        // rides high because the line box also carries the descent and the leading.
        // Checked at a line-height scale where the two visibly differ.
        let f = fonts_scaled(1.5);
        let r = rope("# Heading\nbody\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        let heading = &l.lines()[0];
        let rect = l.fold_rect(&f, heading, BODY).expect("a foldable heading");
        let m = f.metrics(heading.size);
        let glyph_top = heading.baseline - m.ascent;
        // Equal slack above and below the glyphs, to the pixel integer division
        // allows. The glyph band, not the line box: the box's surplus is leading,
        // which all falls below the text, so centering on it would sit low.
        let above = rect.y - glyph_top;
        let below = (heading.baseline + m.descent) - (rect.y + rect.h);
        assert!(
            (above - below).abs() <= 1,
            "chevron off center: {above}px above the glyphs, {below}px below"
        );
        // And that is genuinely above the box's own middle, so the two rules cannot
        // be confused for one another at this line height.
        assert!(
            rect.y + rect.h / 2 < glyph_top + m.line_height / 2,
            "leading falls below the text, so the glyph center sits above the box center"
        );
    }

    #[test]
    fn a_heading_opens_space_above_itself() {
        // Every heading but the document's first separates itself from what came
        // before, and the space scales with the heading, so an H1 gets more air
        // than an H4.
        let f = fonts();
        let r = rope("# First\nbody\n## Second\n#### Fourth\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        let gaps: Vec<i32> = (0..l.lines().len()).map(|i| gap_above(&f, &l, i)).collect();
        assert_eq!(gaps[0], 0, "the first line has nothing above it to clear");
        assert_eq!(gaps[1], 0, "body text opens no space");
        assert!(gaps[2] > 0 && gaps[3] > 0, "later headings do");
        assert!(
            gaps[2] > gaps[3],
            "an H2 takes more space than an H4: {} vs {}",
            gaps[2],
            gaps[3]
        );
    }

    #[test]
    fn collapsed_headings_do_not_stack_flush() {
        // The case the spacing exists for. Fold two sections and only their headings
        // remain, with every blank line that used to separate them hidden inside the
        // folds. Without the space above a heading they would butt together.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let w = 1_000_000;
        let r = rope("# Guide\n\n## Setup\ninstall\n\n## Usage\nrun\n");
        let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
        // `## Setup` is line 2 and `## Usage` line 5.
        let map = FoldMap::build(&Folds::from_lines([2, 5]), &idx);
        let l = folded_layout(&f, &r, w, &map);

        let headings: Vec<&VisualLine> =
            l.lines().iter().filter(|v| v.heading == Some(2)).collect();
        assert_eq!(headings.len(), 2, "both folded headings still draw");
        // The two are adjacent rows now, so the only thing between them is the
        // second's own space above.
        let first = headings[0];
        let second = headings[1];
        let first_bottom =
            first.baseline - f.metrics(first.size).ascent + f.metrics(first.size).line_height;
        let second_top = second.baseline - f.metrics(second.size).ascent;
        let gap = second_top - first_bottom;
        assert!(gap > 0, "collapsed headings must not butt together");
        assert_eq!(
            gap,
            heading_pad(&f, second, 5),
            "the gap is the space the second heading opens above itself"
        );
    }

    #[test]
    fn a_hash_inside_a_code_fence_opens_no_space() {
        // The space keys off the line's classification, so a `#` that is code gets
        // none and a fenced block keeps its even line rhythm.
        let f = fonts();
        let r = rope("text\n```rust\n# a comment\n# another\n```\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        assert!(
            (1..l.lines().len()).all(|i| gap_above(&f, &l, i) == 0),
            "no line in a fenced block opens space"
        );
    }

    #[test]
    fn every_heading_level_gets_the_same_chevron() {
        // Sized off the body, so the chevrons line up in a column down the margin
        // rather than growing with the heading they mark, and each is the size the
        // file tree marks a folder with.
        let f = fonts();
        let r = rope("# One\ntext\n## Two\ntext\n###### Six\ntext\n");
        let l = folded_layout(&f, &r, 1_000_000, FoldMap::NONE);
        let boxes: Vec<Rect> = l
            .lines()
            .iter()
            .filter_map(|v| l.fold_rect(&f, v, BODY))
            .collect();
        assert_eq!(boxes.len(), 3, "one chevron per heading");
        // The size the file tree draws for a row in this font, derived through the
        // tree's own helper so the claim is checked against the tree, not restated.
        let want = crate::theme::chevron_side(
            crate::paint::tree_row_height(f.metrics(BODY), Scale::ONE),
            Scale::ONE,
        );
        for b in &boxes {
            assert_eq!((b.w, b.h), (want, want), "the tree's chevron size");
            assert_eq!(
                b.x + b.w,
                boxes[0].x + boxes[0].w,
                "right-aligned against the text column, so they form a column"
            );
        }
    }

    #[test]
    fn line_index_records_image_reveal_extra() {
        // A decoded standalone image is the one line whose height depends on reveal;
        // the index carries that delta as a per-line overlay.
        let f = fonts();
        let s = ImageStore::from_decoded(vec![(
            "cat.png".to_string(),
            crate::render::image::Image {
                pixels: vec![0u32; 80 * 40],
                width: 80,
                height: 40,
            },
        )]);
        let r = rope("intro line\n![cat](cat.png)\ntrailing line\n");
        let w = 560;
        let img_line = 1;
        let idx = LineIndex::build(&f, &r, w, BODY, &s, Scale::ONE);
        assert!(
            idx.image_extra(img_line) > 0,
            "revealing the source adds height"
        );
        assert_eq!(idx.image_extra(0), 0, "a text line has no reveal extra");
        assert_eq!(idx.image_extra(2), 0);

        // Base height (nothing revealed) matches a full no-reveal build.
        let base_ctx = LayoutCtx::for_index(&f, &r, w, BODY, &s, Scale::ONE);
        let base = Layout::build_with_images(&base_ctx, LEFT, TOP);
        assert_eq!(
            idx.content_height(base_adjust(FoldMap::NONE)),
            base.text_height()
        );

        // With the image's source revealed, the overlay height matches a full build
        // that reveals the same line.
        let ls = r.line_to_byte(img_line);
        let le = r.line_to_byte(img_line + 1) - 1;
        let revealed_ctx = LayoutCtx::new(
            &f,
            &r,
            w,
            BODY,
            (ls, le),
            &s,
            ImageUi::default(),
            FoldMap::NONE,
            Scale::ONE,
        );
        let revealed = Layout::build_with_images(&revealed_ctx, LEFT, TOP);
        let mut adj = RevealAdjust::default();
        adj.push(img_line, idx.image_extra(img_line));
        assert_eq!(
            idx.content_height(Adjust::new(adj, FoldMap::NONE)),
            revealed.text_height()
        );
    }

    #[test]
    fn window_matches_the_full_build_slice() {
        // Window oracle: a windowed build is byte-identical to the slice of the
        // full build covering the same logical lines, at absolute document
        // positions, across corpora, widths, and several bands.
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            let n = r.len_lines();
            for &w in ORACLE_WIDTHS {
                let full = Layout::build(&f, &r, LEFT, TOP, w, BODY, NO_REVEAL);
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                let full_fp = fingerprint(&full);
                let bands: [(usize, usize); 5] = [
                    (0, n),
                    (0, (n / 2).max(1).min(n)),
                    (n / 3, (2 * n / 3).max(n / 3)),
                    (n.saturating_sub(4), n),
                    (n / 2, n / 2), // empty band
                ];
                for (bs, be) in bands {
                    let seed = TOP + idx.logical_top(bs, base_adjust(FoldMap::NONE));
                    let win_ctx = LayoutCtx::for_index(&f, &r, w, BODY, &no_images, Scale::ONE);
                    let win = Layout::build_window(&win_ctx, LEFT, seed, &idx, bs, be);
                    let expected: Vec<_> = full_fp
                        .iter()
                        .filter(|fp| {
                            let li = r.byte_to_line(fp.start);
                            li >= bs && li < be
                        })
                        .cloned()
                        .collect();
                    assert_eq!(fingerprint(&win), expected, "window [{bs},{be}) width {w}");
                }
            }
        }
    }

    #[test]
    fn a_revealed_window_matches_the_revealed_full_build() {
        // What the pane actually runs: a window laid out under the live reveal,
        // seeded from the index's adjusted top. The caret sits on a link line, so
        // that line re-wraps and every band below it must still land where the full
        // build puts it.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let src = "intro line here\n\
                   see [docs](https://example.com/a/very/long/path/goes/on) ok\n\
                   another [ref](https://example.com/some/other/long/path/here) line\n\
                   tail\n";
        let r = rope(src);
        let n = r.len_lines();
        let w = wrap_after(&f, "see docs ok and some room to spare");
        let idx = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
        let caret = r.line_content_range(1).0;
        let active = (caret, caret);
        let full = Layout::build(&f, &r, LEFT, TOP, w, BODY, active);
        assert!(
            full.lines().len() > n,
            "the revealed line must actually re-wrap, or the bands below it never move"
        );
        let full_fp = fingerprint(&full);
        let mut reveal = RevealAdjust::default();
        reveal.reveal_lines(1, 1);
        let adjust = Adjust::new(reveal, FoldMap::NONE);
        for (bs, be) in [(0, n), (1, n), (2, n), (n - 1, n)] {
            let seed = TOP + idx.logical_top(bs, adjust);
            let win_ctx =
                LayoutCtx::for_index(&f, &r, w, BODY, &no_images, Scale::ONE).with_active(active);
            let win = Layout::build_window(&win_ctx, LEFT, seed, &idx, bs, be);
            let expected: Vec<_> = full_fp
                .iter()
                .filter(|fp| {
                    let li = r.byte_to_line(fp.start);
                    li >= bs && li < be
                })
                .cloned()
                .collect();
            assert_eq!(fingerprint(&win), expected, "revealed window [{bs},{be})");
        }
    }

    #[test]
    fn incremental_index_matches_a_full_rebuild() {
        // Incremental oracle: after each edit, the incrementally updated index must be
        // byte-identical to a fresh full build. The edit stream covers in-line
        // edits, a newline split, a whole-line delete, a fence change, a line join,
        // and an unterminated fence that flips state to EOF (the non-convergence
        // path). A convergence bug (stale heights/state below the edit) shows here.
        let f = fonts();
        let no_images = ImageStore::new(None);
        let w = 300;
        let mut text = String::from(
            "# Title\n\nintro paragraph here\n\n- alpha\n- beta\n  - nested\n\n```\nlet x = 1;\n```\n\ntail line\nlast\n",
        );
        let mut rope = Rope::from_str(&text);
        let mut index = LineIndex::build(&f, &rope, w, BODY, &no_images, Scale::ONE);

        let edits: &[(&str, &str)] = &[
            ("intro", "INTRO"),                                  // in-line replace
            ("INTRO paragraph", "INTRO paragraph\nsecond line"), // insert a newline
            ("- beta\n", ""),                                    // delete a whole line
            ("```\nlet", "````\nlet"),                           // change the fence marker
            ("tail line\nlast", "taillast"),                     // join two lines
            ("# Title", "```\n# Title"),                         // open a fence: state to EOF
        ];
        for (needle, repl) in edits {
            let pos = text
                .find(needle)
                .unwrap_or_else(|| panic!("needle {needle:?} not found"));
            let prev = rope.clone();
            text.replace_range(pos..pos + needle.len(), repl);
            rope = Rope::from_str(&text);
            let update_ctx = LayoutCtx::for_index(&f, &rope, w, BODY, &no_images, Scale::ONE);
            index.update(&update_ctx, &prev);
            let rebuilt = LineIndex::build(&f, &rope, w, BODY, &no_images, Scale::ONE);
            assert_eq!(
                index.recs, rebuilt.recs,
                "records after {needle:?} -> {repl:?}"
            );
            assert_eq!(index.tops, rebuilt.tops, "prefix sums after {needle:?}");
        }
    }

    #[test]
    fn parallel_index_matches_the_sequential_build() {
        // The parallel build must be field-for-field identical to the sequential
        // one. Forcing a high chunk count (`build_chunked` skips the size
        // threshold) puts several chunks inside each fixture, so the fence/list
        // fix-up path runs on the fenced and nested-list fixtures where a code
        // fence or list carries across a chunk boundary (a stitch bug that reused
        // stale ground-state chunks would surface as mismatched heights).
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            for &w in ORACLE_WIDTHS {
                let seq = LineIndex::build(&f, &r, w, BODY, &no_images, Scale::ONE);
                let ctx = LayoutCtx::for_index(&f, &r, w, BODY, &no_images, Scale::ONE);
                for threads in [2, 3, 7, 32] {
                    let par = LineIndex::build_chunked(&ctx, threads);
                    assert_eq!(
                        par.recs,
                        seq.recs,
                        "records (width {w}, {threads} chunks, {} lines)",
                        r.len_lines()
                    );
                    assert_eq!(
                        par.tops, seq.tops,
                        "prefix sums (width {w}, {threads} chunks)"
                    );
                }
            }
        }
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
        assert_eq!(h, m.line_height, "caret spans the line box");
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
    }

    #[test]
    fn a_click_past_the_text_vertically_lands_at_the_document_edge() {
        // A drag started in the empty space below a note must anchor at the end of
        // the text, not at the column the pointer's x sits over on the last line;
        // above the first line is the mirror, the document start. The x is placed
        // squarely mid-line so a column-based hit would land there instead.
        let f = fonts();
        let text = "one two\nthree four";
        let r = rope(text);
        let l = layout(&f, &r);
        let mid_x = l.caret(&f, &r, text.len() - 4).0 as f32;
        assert_eq!(
            l.offset_at(&f, &r, mid_x, 9_000.0),
            text.len(),
            "below the text -> document end, whatever the x"
        );
        assert_eq!(
            l.offset_at(&f, &r, mid_x, -100.0),
            0,
            "above the text -> document start, whatever the x"
        );
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
    fn selection_fills_the_line_box_centered_on_the_glyph_box() {
        // At the editor's 1.5 spacing the line box is taller than the glyph box.
        // The highlight fills the full line box, so stacked lines meet with no gap,
        // but centers it on the text (leading split above and below) so a single
        // line's highlight is not bottom-heavy.
        let f = fonts_scaled(1.5);
        let m = f.metrics(BODY);
        let pad = leading_pad(m);
        assert!(pad > 0, "1.5 spacing leaves leading around the glyph box");
        let r = rope("ab\ncd");
        let l = layout(&f, &r);
        let rects = l.selection_rects(&f, &r, 0, 5);
        assert_eq!(rects.len(), 2, "one rect per line");
        // The glyph-box top (the caret top) sits one leading pad below the fill top.
        let (_, caret_top, _) = l.caret(&f, &r, 0);
        assert_eq!(rects[0].h, m.line_height, "fills the full line box");
        assert_eq!(rects[0].y, caret_top - pad, "centered on the glyph box");
        // Stacked lines meet exactly: no gap, no overlap.
        assert_eq!(
            rects[0].y + rects[0].h,
            rects[1].y,
            "consecutive line highlights are contiguous"
        );
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
        let wrap = (shape::text_advance(f.face(BODY, FontStyle::Regular), "m") * 3.0) as i32;
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
    fn a_long_url_breaks_after_its_last_separator() {
        let f = fonts();
        // The row fits two more clusters than the last `/`, so a mid-word break
        // would cut "bbbb" in half.
        let src = "https://example.com/aaaa/bbbb";
        let r = rope(src);
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "https://example.com/aaaa/bb"),
            BODY,
            (usize::MAX, usize::MAX),
        );
        let lines = l.lines();
        assert_eq!(
            &src[..lines[0].end],
            "https://example.com/aaaa/",
            "the row ends after the last separator that fits"
        );
        assert_eq!(&src[lines[1].start..], "bbbb", "the name stays whole");
    }

    #[test]
    fn a_url_that_fits_the_next_row_moves_there_whole() {
        let f = fonts();
        // A separator break is the fallback, not a preference: the space still
        // wins, so a link only splits when it cannot fit a row of its own.
        let src = "aaa https://example.com/b";
        let r = rope(src);
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "https://example.com/b"),
            BODY,
            (usize::MAX, usize::MAX),
        );
        let lines = l.lines();
        assert_eq!(lines.len(), 2, "one break, at the space");
        assert_eq!(
            &src[..lines[0].end],
            "aaa ",
            "the space keeps the first row"
        );
        assert_eq!(
            &src[lines[1].start..],
            "https://example.com/b",
            "the whole URL on the second"
        );
    }

    #[test]
    fn a_long_word_that_is_not_a_link_still_breaks_mid_word() {
        let f = fonts();
        // The same punctuation outside a link is ordinary prose, so it is no break
        // opportunity and the word breaks where it stops fitting.
        let src = "path/to/aaaa/bbbb";
        let r = rope(src);
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "path/to/aaaa/bb"),
            BODY,
            (usize::MAX, usize::MAX),
        );
        let lines = l.lines();
        assert_eq!(&src[..lines[0].end], "path/to/aaaa/bb", "a mid-word break");
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
            shape::text_advance(f.face(BODY, FontStyle::Regular), "aaa ").round() as i32;
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
        let body_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "Title");
        let h1_w = shape::text_advance(f.face(h1_size(), FontStyle::Regular), "Title");
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
    fn marker_hidden(l: &Layout, line: &VisualLine) -> bool {
        let run = l.run_slice(line).first().expect("a heading line has runs");
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
        assert!(
            marker_hidden(&l, &l.lines()[0]),
            "the '# ' marker collapses"
        );
        assert!(
            l.run_slice(&l.lines()[1])
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
        assert!(
            !marker_hidden(&l, &l.lines()[0]),
            "markers stay while editing"
        );
    }

    #[test]
    fn markers_show_at_the_very_end_of_the_heading_line() {
        let f = fonts();
        // Cursor at offset 7, right before the '\n': still on the heading line.
        let l = heading_and_body(&f, (7, 7));
        assert!(!marker_hidden(&l, &l.lines()[0]));
    }

    #[test]
    fn a_selection_touching_a_heading_reveals_it() {
        let f = fonts();
        // Select the whole buffer: the heading is within it, so it reveals.
        let l = heading_and_body(&f, (0, "# Title\nbody".len()));
        assert!(
            !marker_hidden(&l, &l.lines()[0]),
            "a selected heading shows markers"
        );
    }

    #[test]
    fn hidden_markers_shift_clicks_into_the_visible_text() {
        let f = fonts();
        let r = rope("# Title");
        // No cursor on the line, so the markers hide.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        assert!(marker_hidden(&l, &l.lines()[0]));
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
            shape::text_advance(f.face(h1_size(), FontStyle::Regular), "# ").round() as i32;
        assert_eq!(x_shown - x_hidden, marker_w, "expanded by the marker width");
    }

    #[test]
    fn a_reveal_reflows_its_own_line_and_pushes_what_follows() {
        // Wrapping measures what a line draws, so revealing its markup can cost rows.
        // The reflow is local: nothing above the revealed line moves, and everything
        // below it shifts down by exactly the rows it gained.
        let f = fonts();
        let src = "intro\nsee [docs](https://example.com/a/very/long/path/goes/on) ok\ntail";
        let r = rope(src);
        let w = wrap_after(&f, "see docs ok and some room to spare");
        let hidden = Layout::build(&f, &r, LEFT, TOP, w, BODY, NO_REVEAL);
        let caret = r.line_content_range(1).0;
        let shown = Layout::build(&f, &r, LEFT, TOP, w, BODY, (caret, caret));
        let gained = shown.lines().len() - hidden.lines().len();
        assert!(gained > 0, "the raw destination costs the line a row");
        assert_eq!(
            hidden.lines()[0].baseline,
            shown.lines()[0].baseline,
            "the line above the reveal does not move"
        );
        let line_height = f.metrics(BODY).line_height;
        assert_eq!(
            shown.lines().last().map(|l| l.baseline),
            hidden
                .lines()
                .last()
                .map(|l| l.baseline + gained as i32 * line_height),
            "the tail moves down by the rows the reveal added"
        );
        assert_eq!(
            shown.text_height(),
            hidden.text_height() + gained as i32 * line_height
        );
    }

    // --- List marker reveal -----------------------------------------------

    fn gutter_of(l: &Layout, line: usize) -> Gutter {
        l.lines()[line].gutter
    }

    /// Build at unity scale with a selection drag in flight, so list markers stay
    /// collapsed until the drag settles (the `image_ui.dragging` gate).
    fn build_dragging(f: &Fonts, r: &Rope, active: (usize, usize)) -> Layout {
        let images = ImageStore::new(None);
        let ctx = LayoutCtx::new(
            f,
            r,
            1_000_000,
            BODY,
            active,
            &images,
            ImageUi {
                dragging: true,
                ..ImageUi::default()
            },
            FoldMap::NONE,
            Scale::ONE,
        );
        Layout::build_with_images(&ctx, LEFT, TOP)
    }

    #[test]
    fn a_bullet_keeps_its_disc_while_the_caret_edits_the_label() {
        let f = fonts();
        let r = rope("- item");
        // Caret at byte 3, inside "item" past the "- " marker.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (3, 3));
        assert_eq!(
            gutter_of(&l, 0),
            Gutter::Bullet,
            "editing the label keeps the disc"
        );
    }

    #[test]
    fn a_bullet_reveals_its_marker_at_the_first_two_positions() {
        let f = fonts();
        // "- item": "-" is byte 0, the space byte 1, "item" from byte 2.
        let r = rope("- item");
        for caret in [0, 1] {
            let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (caret, caret));
            assert_eq!(
                gutter_of(&l, 0),
                Gutter::None,
                "caret at position {} reveals the raw marker",
                caret + 1
            );
        }
        // At byte 2 (on "item") the disc is back: the caret has left the marker.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (2, 2));
        assert_eq!(gutter_of(&l, 0), Gutter::Bullet);
    }

    #[test]
    fn a_task_item_reveals_its_whole_marker_around_the_bracket() {
        let f = fonts();
        // Marker "- [ ] " is bytes 0..6; a caret anywhere within it reveals it, so
        // the checkbox brackets are editable, not just the leading dash.
        let r = rope("- [ ] do");
        for caret in [0, 1, 3, 5] {
            let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (caret, caret));
            assert_eq!(
                gutter_of(&l, 0),
                Gutter::None,
                "caret {caret} reveals the marker"
            );
        }
        // At byte 6 (on "do") the checkbox is back.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (6, 6));
        assert!(matches!(gutter_of(&l, 0), Gutter::Checkbox { .. }));
    }

    #[test]
    fn a_selection_reveals_a_bullet_marker_only_when_it_overlaps_it() {
        let f = fonts();
        let r = rope("- item");
        // A selection from the start over the marker reveals it.
        let over = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 4));
        assert_eq!(gutter_of(&over, 0), Gutter::None);
        // One confined to the label (bytes 2..6) leaves the disc.
        let label = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (2, 6));
        assert_eq!(gutter_of(&label, 0), Gutter::Bullet);
    }

    #[test]
    fn a_multi_item_selection_reveals_every_covered_marker() {
        let f = fonts();
        let r = rope("- one\n- two\n- three");
        // Select the whole buffer: every bullet's marker is within it, so each
        // reveals.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, r.len()));
        for line in 0..3 {
            assert_eq!(
                gutter_of(&l, line),
                Gutter::None,
                "list line {line} reveals"
            );
        }
    }

    #[test]
    fn a_drag_holds_list_markers_collapsed_until_release() {
        let f = fonts();
        let r = rope("- one\n- two\n- three");
        // The whole-buffer selection mid-drag: every marker stays a disc.
        let dragging = build_dragging(&f, &r, (0, r.len()));
        for line in 0..3 {
            assert_eq!(
                gutter_of(&dragging, line),
                Gutter::Bullet,
                "line {line} holds its disc during the drag"
            );
        }
        // Released (no drag flag), the same selection reveals them.
        let released = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, r.len()));
        for line in 0..3 {
            assert_eq!(gutter_of(&released, line), Gutter::None);
        }
    }

    #[test]
    fn revealing_a_bullet_marker_does_not_reflow() {
        // Like a heading's, a list marker's reveal is a gutter swap at the same
        // width, so the label's caret column and the height stay put.
        let f = fonts();
        let r = rope("- item");
        let disc = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (3, 3));
        let raw = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let (x_disc, _, _) = disc.caret(&f, &r, 2);
        let (x_raw, _, _) = raw.caret(&f, &r, 2);
        assert_eq!(x_disc, x_raw, "the label column never shifts");
        assert_eq!(disc.text_height(), raw.text_height());
    }

    // --- inline code runs -------------------------------------------------

    #[test]
    fn a_code_span_becomes_marker_code_marker_runs() {
        let f = fonts();
        let r = rope("a `b` c");
        // Cursor on the line: every run is shown.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let kinds: Vec<_> = l
            .run_slice(&l.lines()[0])
            .iter()
            .map(|run| run.kind)
            .collect();
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
            l.run_slice(&l.lines()[0]).iter().all(|run| run.visible),
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
            l.run_slice(line)
                .iter()
                .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
                .all(|run| !run.visible),
            "the backticks collapse off the line"
        );
        assert!(
            l.run_slice(line)
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
    fn an_inline_code_pill_frames_only_the_code_leaving_spaces_uncovered() {
        let f = fonts();
        // Cursor on line two, so line one's backticks collapse and the pill (not the
        // raw source) is what lays out.
        let r = rope("a `b` c\nx");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (8, 8));
        let pad = crate::theme::CODE_PAD_X;
        let pill = l.code_rects(&f, &r);
        assert_eq!(pill.len(), 1, "one inline code pill");
        // The drawn panel is the glyph rect expanded by the pad on each side. Its
        // left edge sits exactly where the space before the span ends (byte 3, the
        // "b" run's flow start), so that space stays visible rather than tinted.
        let space_right = l.caret(&f, &r, 3).0;
        assert_eq!(
            pill[0].x - pad,
            space_right,
            "the panel starts at the preceding space's right edge"
        );
        // And its right edge sits where the trailing space begins (byte 5, the " c"
        // run's flow start), within the 1 px slack from rounding the glyph width and
        // the integer pad separately.
        let next_space_left = l.caret(&f, &r, 5).0;
        assert!(
            ((pill[0].x + pill[0].w + pad) - next_space_left).abs() <= 1,
            "the panel ends at the trailing space's left edge"
        );
    }

    #[test]
    fn clicks_round_trip_across_an_inline_code_pill() {
        let f = fonts();
        let r = rope("a `b` c\nx");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (8, 8));
        let y = l.lines()[0].baseline as f32;
        // Every visible offset around the span round-trips through the caret and back,
        // so the pad folded into the advance and into the hit-test agree.
        for off in [0usize, 3, 6, 7] {
            let x = l.caret(&f, &r, off).0 as f32;
            assert_eq!(
                l.offset_at(&f, &r, x, y),
                off,
                "caret x at {off} maps back to {off}"
            );
        }
        // A click in the pad before the glyphs lands at the code start, not short.
        let flow_start = l.caret(&f, &r, 3).0 as f32;
        assert_eq!(
            l.offset_at(
                &f,
                &r,
                flow_start + crate::theme::CODE_PAD_X as f32 / 2.0,
                y
            ),
            3,
            "the leading pad belongs to the code start"
        );
    }

    #[test]
    fn a_fenced_code_line_gets_no_inline_pad() {
        let f = fonts();
        // Cursor on the trailing line, off the fence.
        let r = rope("```\nmn\n```\nx");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (11, 11));
        // A fence draws one full-width panel, not per-span pills, so it lists none.
        assert!(
            l.code_rects(&f, &r).is_empty(),
            "fenced code is not an inline pill"
        );
        // Its glyphs advance by pure width with no pad folded in: the caret at the end
        // of "mn" sits exactly its two-glyph advance past the fenced inset.
        let content = &l.lines()[1];
        assert!(content.code_block, "the 'mn' line is fenced");
        let face = run_face(
            &f,
            content.size,
            markdown::SpanKind::Text(markdown::Emphasis::Code),
            false,
        );
        let mn_w = shape::text_advance(face, "mn").round() as i32;
        assert_eq!(
            l.caret(&f, &r, 6).0,
            LEFT + content.indent + mn_w,
            "no pad on a fenced line"
        );
    }

    #[test]
    fn clicking_past_a_trailing_collapsed_marker_lands_at_the_line_end() {
        let f = fonts();
        // With the cursor away, "`b`" renders as just "b"; its closing backtick
        // is a hidden marker at the line's end.
        let r = rope("`b`");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let y = l.lines()[0].baseline as f32;
        // A click far to the right resolves to the line's logical end (3), after
        // the collapsed backtick, not the end of the visible "b" (2).
        assert_eq!(l.offset_at(&f, &r, 10_000.0, y), 3);
    }

    #[test]
    fn hiding_a_code_marker_shifts_the_following_text_left() {
        let f = fonts();
        let r = rope("`b`");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        // Offset 1 is "b", just past the opening backtick. Collapsed it sits at the
        // margin (the pill's leading pad is drawn space the caret sits before); shown
        // it sits past that pad and the backtick, which the pad now falls outside of.
        let (x_hidden, _, _) = hidden.caret(&f, &r, 1);
        let (x_shown, _, _) = shown.caret(&f, &r, 1);
        assert_eq!(x_hidden, LEFT, "collapsed: code text at the margin");
        let tick_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        assert_eq!(
            x_shown - x_hidden,
            tick_w + crate::theme::CODE_PAD_X,
            "shown: shifted by the backtick and the pad that moved outside it"
        );
    }

    /// The four caret stops around a revealed `` `cd` ``, read off the pill the
    /// painter draws: its edges are the delimiters' outer edges, so an offset that
    /// agrees with them agrees with the glyphs. Byte layout is `ab `cd``, so 3 is the
    /// opening backtick, 4 the code start, 6 the code end, 7 past the closing one.
    fn revealed_span(l: &Layout, f: &Fonts, r: &Rope) -> ([i32; 4], Rect) {
        let pill = l.code_rects(f, r);
        let carets = [
            l.caret(f, r, 3).0,
            l.caret(f, r, 4).0,
            l.caret(f, r, 6).0,
            l.caret(f, r, 7).0,
        ];
        (carets, pill[0])
    }

    #[test]
    fn a_caret_inside_a_revealed_code_span_lands_on_the_drawn_delimiters() {
        let f = fonts();
        let src = "ab `cd`";
        let r = rope(src);
        // Cursor inside the span, so both backticks are revealed and drawn.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (5, 5));
        let ([before, code_start, code_end, after], pill) = revealed_span(&l, &f, &r);
        let tick = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        let pad = crate::theme::CODE_PAD_X;
        // The pill runs from the opening backtick's left edge to the closing one's
        // right edge, so each caret stop is a fixed step in from an edge.
        assert_eq!(before, pill.x - pad, "before the opening backtick");
        assert_eq!(
            code_start,
            pill.x + tick,
            "between the backtick and the code"
        );
        assert_eq!(
            code_end,
            pill.x + pill.w - tick,
            "between the code and the closing backtick"
        );
        assert_eq!(after, pill.x + pill.w + pad, "past the closing backtick");
    }

    #[test]
    fn a_click_on_a_revealed_backtick_can_land_on_either_side_of_it() {
        let f = fonts();
        let src = "ab `cd`";
        let r = rope(src);
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (5, 5));
        let y = l.lines()[0].baseline as f32;
        let (_, pill) = revealed_span(&l, &f, &r);
        let tick = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        // The opening backtick's own glyph box: its left half places the caret before
        // it and its right half after it, so a drag over the delimiter selects it.
        let open = pill.x as f32;
        assert_eq!(
            l.offset_at(&f, &r, open + 1.0, y),
            3,
            "left of the opening `"
        );
        assert_eq!(
            l.offset_at(&f, &r, open + tick as f32 - 1.0, y),
            4,
            "right of the opening `"
        );
        // And the closing one, whose right edge is the pill's.
        let close = (pill.x + pill.w - tick) as f32;
        assert_eq!(
            l.offset_at(&f, &r, close + 1.0, y),
            6,
            "left of the closing `"
        );
        assert_eq!(
            l.offset_at(&f, &r, close + tick as f32 - 1.0, y),
            7,
            "right of the closing `"
        );
    }

    #[test]
    fn clicks_round_trip_across_a_revealed_code_span() {
        let f = fonts();
        let r = rope("ab `cd`");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (5, 5));
        let y = l.lines()[0].baseline as f32;
        // Every offset in and around the revealed span, the delimiters' own included,
        // is a caret stop a click can reach again.
        for off in 0..=7 {
            let x = l.caret(&f, &r, off).0 as f32;
            assert_eq!(
                l.offset_at(&f, &r, x, y),
                off,
                "caret x at {off} maps back to {off}"
            );
        }
    }

    #[test]
    fn revealing_a_code_span_leaves_the_text_after_it_put() {
        let f = fonts();
        let src = "ab `cd` ef";
        let r = rope(src);
        let hidden = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (5, 5));
        let tick = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        // The span reserves the same pad either way, so revealing the delimiters
        // pushes what follows right by exactly the two backticks, no more.
        let ef = src.find("ef").expect("the trailing word");
        assert_eq!(
            shown.caret(&f, &r, ef).0 - hidden.caret(&f, &r, ef).0,
            2 * tick,
            "the following text shifts by the delimiters alone"
        );
    }

    #[test]
    fn a_revealed_cell_code_span_agrees_with_its_pill() {
        let f = fonts();
        // Cursor inside the cell's code span, so the row reveals its backticks.
        let src = "| h |\n| - |\n| `cd` |\n";
        let r = rope(src);
        let tick_open = src.rfind("`cd`").expect("the cell's code span");
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            1_000_000,
            BODY,
            (tick_open + 2, tick_open + 2),
        );
        let pill = l.code_rects(&f, &r);
        assert_eq!(pill.len(), 1, "one pill behind the cell's inline code");
        let tick = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        // The cell walk lays runs the same way the line walk does, so the caret stops
        // sit on the drawn delimiters here too.
        assert_eq!(
            l.caret(&f, &r, tick_open + 1).0,
            pill[0].x + tick,
            "between the backtick and the code"
        );
        assert_eq!(
            l.caret(&f, &r, tick_open + 3).0,
            pill[0].x + pill[0].w - tick,
            "between the code and the closing backtick"
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
        let runs: Vec<_> = l
            .run_slice(line)
            .iter()
            .map(|run| (run.start, run.kind))
            .collect();
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
        assert!(
            l.run_slice(line).iter().all(|run| run.visible),
            "active: all shown"
        );
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
            l.run_slice(line)
                .iter()
                .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
                .all(|run| !run.visible),
            "every marker (heading and emphasis) collapses off the line"
        );
        assert!(
            l.run_slice(line)
                .iter()
                .filter(|run| matches!(
                    run.kind,
                    markdown::SpanKind::Text(markdown::Emphasis::Bold)
                ))
                .all(|run| run.visible),
            "the bold text stays shown"
        );
    }

    // --- lists & blockquotes ----------------------------------------------

    #[test]
    fn a_list_bullet_persists_when_the_cursor_leaves() {
        let f = fonts();
        let src = "- item\nx";
        let r = rope(src);
        // Cursor on the second line, so a heading would hide its marker.
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let bullet = l
            .run_slice(&l.lines()[0])
            .first()
            .expect("a bullet line has runs");
        assert!(
            matches!(bullet.kind, markdown::SpanKind::Marker),
            "the line starts with the bullet marker"
        );
        assert!(
            bullet.visible,
            "the bullet stays on screen off the line (unlike a heading)"
        );
    }

    #[test]
    fn a_quote_marker_persists_while_its_inline_markers_still_hide() {
        let f = fonts();
        let src = "> a `b` c\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let runs = l.run_slice(&l.lines()[0]);
        let quote = runs.first().expect("a quote line has runs");
        assert!(
            matches!(quote.kind, markdown::SpanKind::Marker) && quote.visible,
            "the quote '> ' persists"
        );
        // The code span's backticks (inline markers past the block marker) still
        // collapse off the line.
        let inline_markers: Vec<_> = runs
            .iter()
            .skip(1)
            .filter(|run| matches!(run.kind, markdown::SpanKind::Marker))
            .collect();
        assert!(
            !inline_markers.is_empty() && inline_markers.iter().all(|run| !run.visible),
            "inline markers hide even though the block marker persists"
        );
    }

    // --- hanging indent & gutter decoration -------------------------------

    #[test]
    fn body_lines_have_no_indent_or_gutter() {
        let f = fonts();
        let l = layout(&f, &rope("plain text\nmore"));
        for line in l.lines() {
            assert_eq!(line.indent, 0, "body text is flush to the margin");
            assert_eq!(line.gutter, Gutter::None, "body text has no gutter");
        }
    }

    /// A wrap width that fits `prefix` but not one more cluster, so a line whose
    /// text begins with `prefix` breaks right after it.
    fn wrap_after(f: &Fonts, prefix: &str) -> i32 {
        let face = f.face(BODY, FontStyle::Regular);
        (shape::text_advance(face, prefix) + shape::text_advance(face, "b") / 2.0) as i32
    }

    #[test]
    fn a_wrapped_list_item_hangs_its_continuation_under_the_text() {
        let f = fonts();
        let marker_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "- ").round() as i32;
        let src = "- aaaaa bbbbb";
        let r = rope(src);
        // The list indent narrows the wrap, so widen by it to keep the split.
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "- aaaaa ") + LIST_INDENT,
            BODY,
            (0, r.len()),
        );
        let lines = l.lines();
        assert_eq!(
            lines.len(),
            2,
            "the bullet item wraps into two visual lines"
        );
        assert_eq!(
            lines[0].indent, LIST_INDENT,
            "the marker line sits at the small list indent"
        );
        assert_eq!(
            lines[1].indent,
            LIST_INDENT + marker_w,
            "the continuation hangs under the text, past indent and marker"
        );
        // The caret at the start of the continuation sits at the indented column,
        // not back at the margin.
        let cont = src.find("bbbbb").unwrap();
        let (x, _, _) = l.caret(&f, &r, cont);
        assert_eq!(
            x,
            LEFT + LIST_INDENT + marker_w,
            "continuation text starts past the indent and marker"
        );
    }

    #[test]
    fn a_nested_item_with_a_parent_is_a_list_item() {
        let f = fonts();
        // Child at depth 1 (two spaces) under a depth-0 parent.
        let r = rope("- a\n  - b");
        // Caret on the first line, so the child's marker is untouched and decorated.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let child = &l.lines()[1];
        assert_eq!(
            child.gutter,
            Gutter::Bullet,
            "a nested item with a parent renders as a bullet"
        );
        assert_eq!(
            child.indent,
            LIST_INDENT + LIST_NEST_INDENT,
            "and carries the list indent stepped in by one nesting level"
        );
        let marker = l.run_slice(child).first().expect("a marker run");
        assert!(matches!(marker.kind, markdown::SpanKind::Marker));
        assert_eq!(
            marker.end - marker.start,
            4,
            "the marker spans the indentation and the dash (`  - `)"
        );
    }

    #[test]
    fn a_continuation_line_hangs_under_its_list_item_with_no_bullet() {
        let f = fonts();
        // An indented plain line under a bullet is a Shift+Enter continuation: it
        // takes the item's list indent but draws no bullet of its own.
        let r = rope("- item\n  cont");
        let l = layout(&f, &r);
        let cont = &l.lines()[1];
        assert_eq!(cont.gutter, Gutter::None, "a continuation draws no bullet");
        assert_eq!(
            cont.indent, LIST_INDENT,
            "but carries the item's list indent"
        );
        // A following sibling still nests, so the continuation kept the list open.
        let r2 = rope("- a\n  cont\n- b");
        // Caret on the first line, so the sibling's marker is untouched and decorated.
        let l2 = Layout::build(&f, &r2, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            l2.lines()[2].gutter,
            Gutter::Bullet,
            "the item after a continuation still renders as a list item"
        );
    }

    #[test]
    fn an_over_nested_item_without_a_parent_renders_as_body() {
        let f = fonts();
        // Depth 2 (four spaces) right after a depth-0 item: no depth-1 parent.
        let r = rope("- a\n    - b");
        let l = layout(&f, &r);
        let over = &l.lines()[1];
        assert_eq!(over.gutter, Gutter::None, "no bullet without a parent");
        assert_eq!(over.indent, 0, "and not given the list indent");
        assert!(
            l.run_slice(over)
                .iter()
                .all(|run| matches!(run.kind, markdown::SpanKind::Text(_))),
            "the whole line is plain body text, no marker"
        );
    }

    #[test]
    fn outdenting_a_nested_item_reparents_its_child() {
        // Three items nested one inside the next: a (depth 0) > b (1) > c (2).
        // Reverse-indent the middle item b. Its child c should come along, so c
        // renders as a list item at one indent (under b), not orphaned into plain
        // text by losing its depth-1 parent.
        let f = fonts();
        let mut buf = crate::text::TextBuffer::new();
        buf.insert_str("- a\n  - b\n    - c");
        buf.set_cursor(8, false); // on the middle item "  - b"
        buf.nest_list_item(true); // Shift+Tab: outdent b
        let l = Layout::build(&f, buf.rope(), LEFT, TOP, 1_000_000, BODY, (0, 0));
        let third = &l.lines()[2];
        assert_eq!(
            third.gutter,
            Gutter::Bullet,
            "the third item still renders as a list item"
        );
        assert_eq!(
            third.indent,
            LIST_INDENT + LIST_NEST_INDENT,
            "and carries the list indent stepped in by one nesting level"
        );
        let marker = l.run_slice(third).first().expect("a marker run");
        assert_eq!(
            marker.end - marker.start,
            4,
            "rendered at one indent (`  - `), reparented under b"
        );
    }

    #[test]
    fn a_bullet_shows_its_dot_on_and_off_its_line() {
        let f = fonts();
        let src = "- item\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::Bullet,
            "a bullet point stands in once the cursor leaves"
        );
        // Editing the label (caret at byte 3, on "item") keeps the dot; only the
        // caret reaching the marker reveals the raw "- " (the marker-reveal tests).
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (3, 3));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::Bullet,
            "and the dot stays while the cursor edits the label"
        );
    }

    #[test]
    fn a_quote_shows_a_bar_off_its_line_and_a_raw_marker_on_it() {
        let f = fonts();
        let src = "> quoted\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::QuoteBar { marker: true },
            "the bar stands in for the '>' off-line"
        );
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::None,
            "raw '> ' shows while editing"
        );
    }

    #[test]
    fn a_long_word_after_a_marker_starts_on_the_markers_own_row() {
        let f = fonts();
        // The URL is one unbreakable word, wider than the column. The space in `> `
        // is no place to break: the first row would hold nothing but a marker that
        // collapses to a quote bar, so the quote would open with a blank line and
        // the URL would break mid-word on the next row anyway. It starts on the
        // marker's row instead.
        let src = "> https://example.com/a-very-long-path-that-cannot-fit\nz";
        let off = src.find('z').expect("the plain line");
        let r = rope(src);
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "> https://exam"),
            BODY,
            (off, off),
        );
        let lines = l.lines();
        let marker = "> ".len();
        assert!(
            lines[0].end > marker,
            "the URL starts on the quote's first row, not below it"
        );
        assert_eq!(lines[0].gutter, Gutter::QuoteBar { marker: true });
        // The word breaks mid-word from there, and the whole URL is still laid out.
        assert!(lines.len() > 2, "the URL wraps on past the first row");
        assert_eq!(
            lines[..lines.len() - 1].last().map(|l| l.end),
            Some(src.find('\n').expect("the newline")),
            "the last quote row ends the URL"
        );
    }

    #[test]
    fn a_long_word_still_moves_below_the_text_it_follows() {
        let f = fonts();
        // With text on the row already, the space is a real break: "aaa" keeps the
        // row and the long word starts on the next one (and breaks up from there).
        let src = "aaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let r = rope(src);
        let l = Layout::build(&f, &r, LEFT, TOP, wrap_after(&f, "aaa "), BODY, (0, 0));
        let lines = l.lines();
        assert_eq!(
            (lines[0].start, lines[0].end),
            (0, 4),
            "'aaa ' keeps the first row"
        );
        assert!(lines.len() > 2, "the long word breaks across the rest");
    }

    #[test]
    fn a_wrapped_quote_keeps_the_bar_without_a_second_marker() {
        let f = fonts();
        let marker_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "> ").round() as i32;
        let src = "> aaaaa bbbbb\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "> aaaaa "),
            BODY,
            (off, off),
        );
        let lines = l.lines();
        assert!(
            lines.len() >= 3,
            "the quote wraps in two, then a plain line"
        );
        assert_eq!(lines[0].gutter, Gutter::QuoteBar { marker: true });
        assert_eq!(
            lines[1].gutter,
            Gutter::QuoteBar { marker: false },
            "the continuation keeps the bar but has no marker run of its own"
        );
        assert_eq!(lines[1].indent, marker_w, "and hangs under the text");
    }

    #[test]
    fn an_ordered_item_keeps_a_plain_gutter_but_still_indents() {
        let f = fonts();
        let marker_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "1. ").round() as i32;
        let r = rope("1. aaaaa bbbbb");
        // Cursor off the line: the number must still show (never substituted).
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "1. aaaaa ") + LIST_INDENT,
            BODY,
            (99, 99),
        );
        let lines = l.lines();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0].gutter,
            Gutter::None,
            "an ordered number is never replaced"
        );
        assert_eq!(lines[1].gutter, Gutter::None);
        assert_eq!(
            lines[0].indent, LIST_INDENT,
            "the number sits at the small list indent"
        );
        assert_eq!(
            lines[1].indent,
            LIST_INDENT + marker_w,
            "and wrapped lines still hang-indent past it"
        );
    }

    // --- Task lists ---------------------------------------------------------

    #[test]
    fn a_task_line_draws_its_checkbox_while_editing_the_label() {
        let f = fonts();
        let src = "- [ ] todo\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::Checkbox { checked: false },
            "the checkbox stands in with the cursor away"
        );
        assert!(
            away.run_slice(&away.lines()[0])[0].visible,
            "the marker run persists (width-matched under the drawn box)"
        );
        // The box stays put while the cursor edits the label (byte 8, on "todo"):
        // it flips back to the raw `- [ ]` only when the caret reaches the marker,
        // mirroring how a bullet keeps its dot while editing the label.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (8, 8));
        assert_eq!(on.lines()[0].gutter, Gutter::Checkbox { checked: false });
        assert!(on.run_slice(&on.lines()[0])[0].visible);
    }

    #[test]
    fn a_checked_task_carries_its_state_into_gutter_and_line() {
        let f = fonts();
        let src = "- [x] done\nx";
        let r = rope(src);
        // Cursor on the trailing body line, so the task line renders.
        let off = src.len() - 1;
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(l.lines()[0].gutter, Gutter::Checkbox { checked: true });
        assert_eq!(l.lines()[0].task, Some(true), "the line knows it is done");
        assert_eq!(l.lines()[1].task, None, "body lines are not tasks");
    }

    #[test]
    fn a_wrapped_task_item_stamps_task_on_continuations_without_a_second_box() {
        let f = fonts();
        let src = "- [x] aaaaa bbbbb\nx";
        let r = rope(src);
        // Cursor on the trailing body line, so the task item renders.
        let off = src.len() - 1;
        let l = Layout::build(
            &f,
            &r,
            LEFT,
            TOP,
            wrap_after(&f, "- [x] aaaaa ") + LIST_INDENT,
            BODY,
            (off, off),
        );
        let lines = l.lines();
        assert!(lines.len() >= 3, "the item wraps, then the plain line");
        assert_eq!(lines[0].gutter, Gutter::Checkbox { checked: true });
        assert_eq!(
            lines[1].gutter,
            Gutter::None,
            "no second checkbox on the continuation"
        );
        assert_eq!(
            lines[1].task,
            Some(true),
            "the continuation still dims with the item"
        );
        let marker_w =
            shape::text_advance(f.face(BODY, FontStyle::Regular), "- [x] ").round() as i32;
        assert_eq!(
            lines[1].indent,
            LIST_INDENT + marker_w,
            "continuations hang under the text, past the whole task marker"
        );
    }

    #[test]
    fn nested_tasks_follow_list_nesting() {
        let f = fonts();
        // A task under a bullet parent renders; an over-nested one is body.
        let r = rope("- a\n  - [ ] sub\n    - [ ] fine");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (99, 99));
        assert_eq!(
            l.lines()[1].gutter,
            Gutter::Checkbox { checked: false },
            "a nested task with a parent draws its box"
        );
        assert_eq!(
            l.lines()[2].gutter,
            Gutter::Checkbox { checked: false },
            "each level deeper by one keeps nesting"
        );
        let over = rope("- a\n    - [ ] orphan");
        let l = Layout::build(&f, &over, LEFT, TOP, 1_000_000, BODY, (99, 99));
        assert_eq!(l.lines()[1].gutter, Gutter::None, "no parent, no box");
        assert_eq!(l.lines()[1].task, None, "an orphan renders as plain body");
    }

    #[test]
    fn checkbox_rect_sits_inset_a_gap_short_of_the_text() {
        let f = fonts();
        // A nested task under a parent, so its indentation offsets the box.
        let r = rope("- a\n  - [ ] sub\nx");
        let off = r.len() - 1;
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let line = &l.lines()[1];
        let rect = l
            .checkbox_rect(&f, &r, line)
            .expect("a non-revealed task line has a box");
        let face = f.face(BODY, FontStyle::Regular);
        // The text begins after the whole `  - [ ] ` marker; the box sits a gap
        // short of it and stays within the item's content.
        let text_x = LEFT as f32 + line.indent as f32 + shape::text_advance(face, "  - [ ] ");
        assert!(
            rect.x as f32 >= LEFT as f32 + line.indent as f32,
            "the box stays within the item's content, not off the left"
        );
        assert!(
            (rect.x + rect.w) as f32 <= text_x - 2.0,
            "the box ends a gap short of the text"
        );
        assert!(rect.w >= 8 && rect.h == rect.w, "a sensible square");
        let m = f.metrics(BODY);
        // Centered on the ascent band (being a touch taller than the ascent, it
        // may sit a hair above the cap line) and never dipping past the descender.
        let mid = line.baseline - m.ascent / 2;
        assert!(
            (rect.y + rect.h / 2 - mid).abs() <= 1,
            "vertically centered on the ascent band"
        );
        assert!(
            rect.y + rect.h <= line.baseline + m.descent,
            "stays above the descender"
        );
        // The box shows while editing the label (byte 13, on "sub"), at the same
        // geometry (reveal never moves it).
        let label = 13;
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (label, label));
        assert_eq!(on.checkbox_rect(&f, &r, &on.lines()[1]), Some(rect));
    }

    #[test]
    fn checkbox_at_hits_the_box_and_only_the_box() {
        let f = fonts();
        let src = "- [ ] todo\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let rect = l
            .checkbox_rect(&f, &r, &l.lines()[0])
            .expect("the box exists");
        let (cx, cy) = ((rect.x + rect.w / 2) as f32, (rect.y + rect.h / 2) as f32);
        assert_eq!(l.checkbox_at(&f, &r, cx, cy), Some(0), "center hit");
        assert_eq!(
            l.checkbox_at(&f, &r, cx + 200.0, cy),
            None,
            "the item's text is not the checkbox"
        );
        assert_eq!(
            l.checkbox_at(&f, &r, cx, cy + 200.0),
            None,
            "another line is not the checkbox"
        );
        // While editing the label (byte 8, on "todo") the box still hit-tests, so a
        // click there toggles it rather than only placing the caret.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (8, 8));
        assert_eq!(on.checkbox_at(&f, &r, cx, cy), Some(0));
    }

    #[test]
    fn task_reveal_state_does_not_change_vertical_layout() {
        let f = fonts();
        let r = rope("- [x] a task long enough to wrap somewhere\nbody");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (0, 0));
        assert_eq!(hidden.lines().len(), shown.lines().len());
        for (h, s) in hidden.lines().iter().zip(shown.lines()) {
            assert_eq!(h.baseline, s.baseline, "baselines are reveal-independent");
            assert_eq!((h.start, h.end), (s.start, s.end), "and so are the breaks");
        }
        assert_eq!(hidden.text_height(), shown.text_height());
    }

    // --- Fenced code blocks -----------------------------------------------

    /// The first run kind of each visual line, for terse fence assertions.
    fn first_kinds(l: &Layout) -> Vec<Option<markdown::SpanKind>> {
        l.lines()
            .iter()
            .map(|line| l.run_slice(line).first().map(|r| r.kind))
            .collect()
    }

    #[test]
    fn a_fenced_block_marks_its_lines_as_code() {
        let f = fonts();
        let r = rope("```rust\nfn x() {}\n```\nbody");
        let l = layout(&f, &r);
        let code: Vec<bool> = l.lines().iter().map(|line| line.code_block).collect();
        assert_eq!(
            code,
            vec![true, true, true, false],
            "the fence lines and the code between are a block; the body is not"
        );
        // The ``` fences read as markers; the code line tokenizes (its first token
        // is the `fn` keyword); the body is prose.
        let kinds = first_kinds(&l);
        assert!(
            matches!(kinds[0], Some(markdown::SpanKind::Marker)),
            "```rust"
        );
        assert!(
            matches!(kinds[1], Some(markdown::SpanKind::Code(_))),
            "fn x() {{}} is a code token"
        );
        assert!(matches!(kinds[2], Some(markdown::SpanKind::Marker)), "```");
        assert!(
            matches!(
                kinds[3],
                Some(markdown::SpanKind::Text(markdown::Emphasis::Regular))
            ),
            "body is prose"
        );
    }

    #[test]
    fn markdown_inside_a_fence_is_literal() {
        // A '#' line that would be an H1 is plain code inside a fence: code size,
        // a single code run, no heading marker.
        let f = fonts();
        let r = rope("```\n# not a heading\n```");
        let l = layout(&f, &r);
        let inner = &l.lines()[1];
        assert!(inner.code_block);
        assert_eq!(
            inner.size,
            markdown::code_size(BODY),
            "no heading scaling inside a code block"
        );
        assert_eq!(l.run_slice(inner).len(), 1);
        assert_eq!(
            l.run_slice(inner)[0].kind,
            markdown::SpanKind::Code(markdown::CodeToken::Plain),
            "the '#' is plain code (no language), not a heading marker"
        );
    }

    #[test]
    fn a_rust_code_line_tokenizes_into_colored_runs() {
        let f = fonts();
        let r = rope("```rust\nlet x = 3;\n```");
        let l = layout(&f, &r);
        let code = &l.lines()[1]; // "let x = 3;"
        let runs = l.run_slice(code);
        assert!(runs.len() > 1, "the code line splits into token runs");
        assert!(
            runs.iter().any(|run| matches!(
                run.kind,
                markdown::SpanKind::Code(markdown::CodeToken::Keyword)
            )),
            "the `let` keyword is highlighted"
        );
        assert!(
            runs.iter().any(|run| matches!(
                run.kind,
                markdown::SpanKind::Code(markdown::CodeToken::Number)
            )),
            "the `3` number is highlighted"
        );
    }

    #[test]
    fn an_unclosed_fence_runs_to_the_end() {
        let f = fonts();
        let r = rope("```\n# H");
        let l = layout(&f, &r);
        assert!(
            l.lines().iter().all(|line| line.code_block),
            "with no closing fence every following line stays code"
        );
    }

    #[test]
    fn text_resumes_after_a_closed_fence() {
        let f = fonts();
        let r = rope("```\nx\n```\n# H");
        let l = layout(&f, &r);
        let last = l.lines().last().expect("a last line");
        assert!(
            !last.code_block,
            "the heading after the fence is normal again"
        );
        assert_eq!(last.size, h1_size(), "and is scaled as an H1");
    }

    /// The first run of a line (its fence marker, for fence lines).
    fn first_run<'a>(l: &'a Layout, line: &VisualLine) -> &'a Run {
        l.run_slice(line).first().expect("a fence line has a run")
    }

    #[test]
    fn fence_backticks_hide_off_their_line_and_show_on_it() {
        let f = fonts();
        let src = "```rust\nfn x() {}\n```\nbody";
        let r = rope(src);
        // Cursor on the trailing body line: both fence lines are off-cursor.
        let off = src.find("body").unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        let (open, close) = (
            first_run(&away, &away.lines()[0]),
            first_run(&away, &away.lines()[2]),
        );
        assert!(matches!(open.kind, markdown::SpanKind::Marker) && !open.visible);
        assert!(
            matches!(close.kind, markdown::SpanKind::Marker) && !close.visible,
            "both ``` fences collapse when the cursor is outside the block"
        );
        // The code line between always shows, fence or no fence.
        assert!(away
            .run_slice(&away.lines()[1])
            .iter()
            .all(|run| run.visible));
        // Cursor on the opening fence line reveals only that fence (per-line, like
        // a heading): the closing fence on another line stays hidden.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert!(
            first_run(&on, &on.lines()[0]).visible,
            "the ``` shows on its line"
        );
        assert!(
            !first_run(&on, &on.lines()[2]).visible,
            "the closing fence is on another line, still hidden"
        );
    }

    #[test]
    fn code_lines_inset_by_the_panel_padding() {
        let f = fonts();
        let r = rope("```\ncode here\n```");
        let l = layout(&f, &r);
        assert!(
            l.lines().iter().all(|line| line.indent == CODE_PAD_X),
            "every code-block line (fences and code) insets by the panel padding"
        );
        // A plain line carries no such inset.
        assert_eq!(layout(&f, &rope("plain")).lines()[0].indent, 0);
    }

    #[test]
    fn caret_on_a_code_line_starts_past_the_padding() {
        let f = fonts();
        let r = rope("```\nx\n```");
        let code_start = "```\n".len();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (code_start, code_start));
        let (x, _, _) = l.caret(&f, &r, code_start);
        assert_eq!(
            x,
            LEFT + CODE_PAD_X,
            "code text begins past the left panel padding"
        );
    }

    #[test]
    fn a_selection_over_a_code_line_yields_a_padded_rect() {
        // The selection rect for a code line sits at the padded column and has
        // width, so the highlight is visible inside the panel (the app paints the
        // panel beneath it). Regression guard for the selection-in-code-block fix.
        let f = fonts();
        let src = "```\nxyz\n```";
        let r = rope(src);
        let start = src.find("xyz").unwrap();
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (start, start + 3));
        let rects = l.selection_rects(&f, &r, start, start + 3);
        assert_eq!(rects.len(), 1, "one rect for the single code line");
        assert_eq!(
            rects[0].x,
            LEFT + CODE_PAD_X,
            "rect starts past the padding"
        );
        assert!(rects[0].w > 0, "and has positive width");
    }

    #[test]
    fn inline_code_has_a_background_rect_but_a_fenced_block_does_not() {
        let f = fonts();
        // An inline code span yields one background panel, sitting past the line
        // start (after "a " and the opening backtick) with real size.
        let r = rope("a `b` c");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let rects = l.code_rects(&f, &r);
        assert_eq!(rects.len(), 1, "one panel behind the inline code span");
        assert!(rects[0].x > LEFT && rects[0].w > 0 && rects[0].h > 0);
        // A fenced block has no inline rects: its full-width panel covers it.
        let r2 = rope("```\ncode\n```");
        let l2 = layout(&f, &r2);
        assert!(
            l2.code_rects(&f, &r2).is_empty(),
            "block code uses the full-width panel, not per-run inline rects"
        );
    }

    #[test]
    fn revealing_the_backticks_grows_the_code_pill_to_enclose_them() {
        let f = fonts();
        // The code span is on line 2 after "ab ". With the cursor on line 1 its
        // backticks are hidden; with the cursor inside the span they reveal.
        let src = "top\nab `cd`";
        let r = rope(src);
        let span = src.find("`cd`").unwrap();
        let hidden = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0)).code_rects(&f, &r);
        let shown = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (span + 1, span + 1))
            .code_rects(&f, &r);
        assert_eq!(hidden.len(), 1);
        assert_eq!(shown.len(), 1);
        // The left edge stays anchored (the hidden opening backtick collapsed to
        // exactly where the revealed one now begins), and the pill widens to hold
        // both delimiters.
        assert_eq!(shown[0].x, hidden[0].x, "the pill's left edge stays put");
        assert!(
            shown[0].w > hidden[0].w,
            "revealed, the pill widens to enclose both backticks"
        );
    }

    #[test]
    fn inline_code_in_a_table_cell_gets_a_pill() {
        let f = fonts();
        // Inline code in a cell lives in the cell runs, not the line's own runs,
        // so it must still pick up its tint panel through the table walk. Cursor is
        // outside the table, so the cell is cooked (its backticks hidden).
        let r = rope("| h | j |\n| - | - |\n| `x` | b |\n");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let rects = l.code_rects(&f, &r);
        assert_eq!(rects.len(), 1, "one pill behind the cell's inline code");
        assert!(rects[0].w > 0 && rects[0].h > 0);
        assert!(
            rects[0].x > LEFT,
            "positioned inside the cell, past the margin"
        );
    }

    #[test]
    fn a_table_cell_code_pill_lines_up_with_its_centered_text() {
        let f = fonts();
        // The second cell has two lines (a <br> b), so the first (code) cell is
        // block-centered in a taller row. The inline-code pill must drop with the
        // centered text, not stay pinned to the row's top.
        let src = "| `x` | a<br>b |\n| - | - |\n";
        let r = rope(src);
        let end = src.len();
        // Cursor outside the table, so the cell is cooked and the pill is drawn.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (end, end));
        let pills = l.code_rects(&f, &r);
        assert_eq!(pills.len(), 1, "one pill for the `x` code span");
        let x_off = src.find('x').unwrap();
        let (_, caret_y, _) = l.caret(&f, &r, x_off);
        assert_eq!(
            pills[0].y, caret_y,
            "the code pill sits over the code text, not above it"
        );
    }

    #[test]
    fn adjacent_code_spans_in_a_cell_stay_on_one_line() {
        let f = fonts();
        // The natural column width once measured the raw cell in the prose face, but
        // the code spans draw in the wider monospace face, so this cell used to wrap
        // apart at every width. It must now sit on one line with two distinct pills.
        let r = rope("| File | n |\n| - | - |\n| `main.rs`, `error.rs` | x |\n");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let (_, t) = table_rows(&l)[1];
        assert_eq!(t.cells[0].lines.len(), 1, "the cell stays on one line");
        let rects = l.code_rects(&f, &r);
        assert_eq!(rects.len(), 2, "two separate pills, one per code span");
        assert_eq!(rects[0].y, rects[1].y, "both on the same cell line");
        assert!(
            rects[1].x >= rects[0].x + rects[0].w,
            "the pills do not overlap"
        );
    }

    #[test]
    fn a_cell_pill_does_not_cover_the_text_between_two_code_spans() {
        let f = fonts();
        // The ", " between the spans is not code; the painter expands each pill by the
        // pill pad, so that pad must be reserved space, not paint over the comma.
        let r = rope("| File | n |\n| - | - |\n| `a`, `b` | x |\n");
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let rects = l.code_rects(&f, &r);
        assert_eq!(rects.len(), 2);
        let pad = crate::theme::CODE_PAD_X;
        // The pills as the painter draws them, grown by the pad on each side.
        let gap = (rects[1].x - pad) - (rects[0].x + rects[0].w + pad);
        let comma = shape::text_advance(f.face(BODY, FontStyle::Regular), ", ").round() as i32;
        assert!(
            gap > 0,
            "the drawn pills leave a gap for the comma, got {gap}"
        );
        assert!(
            (gap - comma).abs() <= 2,
            "the gap holds exactly \", \" ({comma}px), got {gap}"
        );
    }

    #[test]
    fn a_table_row_reveals_its_markers_when_the_cursor_is_on_it() {
        let f = fonts();
        let r = rope("| h |\n| - |\n| `x` |\n");
        let markers = |l: &Layout| -> (usize, usize) {
            let (_, t) = table_rows(l)[1];
            let runs = &t.cells[0].lines[0].runs;
            let shown = runs
                .iter()
                .filter(|run| matches!(run.kind, markdown::SpanKind::Marker) && run.visible)
                .count();
            (shown, t.cells[0].lines.len())
        };
        let body = r.line_to_byte(2);
        // Cursor in the body cell: its two backtick markers reveal.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (body, body));
        let (shown_on, lines_on) = markers(&on);
        assert_eq!(shown_on, 2, "the active row shows both backticks");
        // Cursor outside the table: the same markers collapse.
        let off = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let (shown_off, lines_off) = markers(&off);
        assert_eq!(shown_off, 0, "an off-cursor row hides its backticks");
        assert_eq!(lines_on, lines_off, "revealing the markers does not reflow");
    }

    #[test]
    fn fenced_code_block_content_draws_in_the_code_face() {
        let f = fonts();
        let r = rope("```\ncode\n```");
        let l = layout(&f, &r);
        // The middle line is the code content; its run resolves to the code face,
        // the same one inline code uses, not the prose face. It inks as a block, so
        // the renderer thins it less than the prose (and than an inline span).
        let content = &l.lines()[1];
        let run = l.run_slice(content).first().expect("a content run");
        assert_eq!(
            face_key(content.size, run.kind, content.heading.is_some()),
            FaceKey::Code {
                size: content.size,
                ink: CodeInk::Block
            },
            "code block text draws in the code face, matching inline code"
        );
    }

    #[test]
    fn an_inline_code_span_inks_like_the_prose_it_sits_in() {
        let f = fonts();
        let r = rope("a `b` c");
        let l = layout(&f, &r);
        let line = &l.lines()[0];
        let code = l
            .run_slice(line)
            .iter()
            .find(|run| matches!(run.kind, markdown::SpanKind::Text(markdown::Emphasis::Code)))
            .expect("the code run");
        // Same face and size as a fenced block's tokens, but inline ink: it sits at
        // the prose size in a line of prose, so it takes the prose thinning rather
        // than the gentler one a block's smaller stems need.
        assert_eq!(
            face_key(line.size, code.kind, line.heading.is_some()),
            FaceKey::Code {
                size: line.size,
                ink: CodeInk::Inline
            }
        );
    }

    #[test]
    fn heading_lines_carry_their_level_others_do_not() {
        let f = fonts();
        let r = rope("### h\nbody\n```\ncode\n```");
        let l = layout(&f, &r);
        let lines = l.lines();
        assert_eq!(lines[0].heading, Some(3), "the ### line is an H3");
        assert_eq!(lines[1].heading, None, "body has no heading level");
        assert!(
            lines[2..].iter().all(|ln| ln.heading.is_none()),
            "fence and code lines are never headings"
        );
    }

    #[test]
    fn heading_text_routes_to_the_bold_face() {
        // Monospace bold and regular share advances, so the bolding can't be seen
        // in widths; assert the face routing directly. (Robust to a family that
        // lacks a bold variant: face() then falls back, and both sides agree.)
        let f = fonts();
        let size = h1_size();
        let text = |em| markdown::SpanKind::Text(em);
        // On a heading: prose is bold, italic prose is bold-italic, already-bold
        // stays bold, but the `#` marker is never emboldened.
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Regular), true),
            f.face(size, FontStyle::Bold),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Italic), true),
            f.face(size, FontStyle::BoldItalic),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Bold), true),
            f.face(size, FontStyle::Bold),
        ));
        assert!(std::ptr::eq(
            run_face(&f, size, markdown::SpanKind::Marker, true),
            f.face(size, FontStyle::Regular),
        ));
        // Off a heading, prose stays regular.
        assert!(std::ptr::eq(
            run_face(&f, size, text(markdown::Emphasis::Regular), false),
            f.face(size, FontStyle::Regular),
        ));
    }

    #[test]
    fn a_thematic_break_draws_a_rule_off_its_line_and_dashes_on_it() {
        let f = fonts();
        let src = "a\n---\nb";
        let r = rope(src);
        // Cursor on "a": the rule line (line 1) is off-cursor, so it draws a rule
        // and its raw `---` collapses.
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        let line = &away.lines()[1];
        assert!(line.rule, "the rule draws when the cursor is elsewhere");
        assert!(
            away.run_slice(line).iter().all(|run| !run.visible),
            "the raw --- is collapsed off the line"
        );
        // Cursor on the --- line: it shows the editable dashes, no rule.
        let on_off = src.find("---").unwrap();
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (on_off, on_off));
        let line = &on.lines()[1];
        assert!(!line.rule, "no rule while the cursor edits the line");
        assert!(
            on.run_slice(line)
                .iter()
                .any(|run| run.visible && matches!(run.kind, markdown::SpanKind::Marker)),
            "the raw --- shows for editing"
        );
    }

    #[test]
    fn rule_at_reports_a_rendered_rule_for_a_click() {
        let f = fonts();
        let src = "a\n---\nb";
        let r = rope(src);
        // Cursor on "a": the rule line (line 1, bytes 2..5) is rendered.
        let l = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            l.rule_at(&f, l.lines()[1].baseline as f32),
            Some((2, 5)),
            "a click on the rendered rule yields its content range"
        );
        // A body line is never a click-selectable rule.
        assert_eq!(l.rule_at(&f, l.lines()[0].baseline as f32), None);
        // Once the cursor edits the rule it shows the raw ---, so a click there
        // edits rather than reselecting.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (2, 2));
        assert_eq!(on.rule_at(&f, on.lines()[1].baseline as f32), None);
    }

    /// A decoded image of the given size, pixels zeroed (layout only reads dims).
    fn decoded(w: u32, h: u32) -> crate::render::image::Image {
        crate::render::image::Image {
            pixels: vec![0; (w * h) as usize],
            width: w,
            height: h,
        }
    }

    fn store(path: &str, w: u32, h: u32) -> ImageStore {
        ImageStore::from_decoded(vec![(path.to_string(), decoded(w, h))])
    }

    /// Build a layout over `r` with the image store `s` and UI state `ui`, at a
    /// wide wrap so images show at intrinsic size.
    fn layout_img(
        f: &Fonts,
        r: &Rope,
        active: (usize, usize),
        s: &ImageStore,
        ui: ImageUi,
    ) -> Layout {
        let ctx = LayoutCtx::new(
            f,
            r,
            1_000_000,
            BODY,
            active,
            s,
            ui,
            FoldMap::NONE,
            Scale::ONE,
        );
        Layout::build_with_images(&ctx, LEFT, TOP)
    }

    #[test]
    fn fit_image_caps_width_and_preserves_aspect() {
        assert_eq!(
            fit_image(100, 50, 1000),
            (100, 50),
            "fits the column: intrinsic"
        );
        assert_eq!(
            fit_image(200, 100, 100),
            (100, 50),
            "too wide: scaled, aspect kept"
        );
        assert_eq!(fit_image(0, 0, 100), (1, 1), "degenerate dims clamp to 1");
    }

    #[test]
    fn an_idle_image_is_a_picture_only() {
        let f = fonts();
        let r = rope("![cat](cat.png)\nafter");
        let s = store("cat.png", 80, 40);
        // Nothing selected, cursor on "after": just the picture, no source line.
        let l = layout_img(&f, &r, (20, 20), &s, ImageUi::default());
        let img = l.lines()[0].image.expect("first line renders as a picture");
        assert_eq!((img.width, img.height), (80, 40));
        assert!(!img.selected, "not selected by default");
        assert_eq!(r.byte_slice(img.path_start..img.path_end), "cat.png");
        assert!(
            l.run_slice(&l.lines()[0]).is_empty(),
            "no source text for an idle image"
        );
    }

    #[test]
    fn the_cursor_stacks_the_source_above_the_still_shown_picture() {
        let f = fonts();
        let r = rope("![cat](cat.png)\nafter");
        let s = store("cat.png", 80, 40);
        // Cursor on the image line: source text on top, picture still below it.
        let l = layout_img(&f, &r, (0, 0), &s, ImageUi::default());
        assert!(l.lines()[0].image.is_none(), "row 0 is the editable source");
        assert!(
            !l.run_slice(&l.lines()[0]).is_empty(),
            "the raw markdown is laid out"
        );
        assert!(
            l.lines()[1].image.is_some(),
            "the picture still shows below it"
        );
    }

    #[test]
    fn a_drag_holds_an_image_source_collapsed() {
        let f = fonts();
        let r = rope("![cat](cat.png)\nafter");
        let s = store("cat.png", 80, 40);
        // The selection covers the image line, but a drag is in progress: the source
        // stays collapsed to the picture so the geometry under the pointer holds
        // still. Text markers still reveal (a separate `active`-driven decision).
        let ui = ImageUi {
            dragging: true,
            ..ImageUi::default()
        };
        let l = layout_img(&f, &r, (0, 15), &s, ui);
        assert!(
            l.lines()[0].image.is_some(),
            "the image stays a picture-only row while dragging"
        );
        assert!(
            l.run_slice(&l.lines()[0]).is_empty(),
            "no source text expands under the drag"
        );
    }

    #[test]
    fn the_button_pin_reveals_source_without_the_cursor() {
        let f = fonts();
        let r = rope("![cat](cat.png)\nafter");
        let s = store("cat.png", 80, 40);
        // Cursor on "after" (off the image), but the corner button pinned raw open.
        let ui = ImageUi {
            selected: Some(0),
            raw: Some(0),
            dragging: false,
        };
        let l = layout_img(&f, &r, (20, 20), &s, ui);
        assert!(
            l.lines()[0].image.is_none(),
            "source shows though the cursor is away"
        );
        assert!(
            l.lines()[1].image.is_some(),
            "picture below the pinned source"
        );
    }

    #[test]
    fn a_selected_image_carries_the_flag() {
        let f = fonts();
        let r = rope("![x](cat.png)");
        let s = store("cat.png", 80, 80);
        let idle = layout_img(&f, &r, (99, 99), &s, ImageUi::default());
        assert!(!idle.lines()[0].image.unwrap().selected);
        let sel = layout_img(
            &f,
            &r,
            (99, 99),
            &s,
            ImageUi {
                selected: Some(0),
                raw: None,
                dragging: false,
            },
        );
        assert!(
            sel.lines()[0].image.unwrap().selected,
            "the flag drives the border/button"
        );
    }

    #[test]
    fn a_missing_image_stays_plain_source() {
        let f = fonts();
        let r = rope("![x](gone.png)\nafter");
        let s = ImageStore::from_decoded(vec![]); // nothing decoded
        let l = layout_img(&f, &r, (20, 20), &s, ImageUi::default());
        assert!(
            l.lines().iter().all(|line| line.image.is_none()),
            "an undecodable image is never a picture box, just editable source"
        );
        assert!(
            !l.run_slice(&l.lines()[0]).is_empty(),
            "the source is laid out as text"
        );
    }

    #[test]
    fn an_image_box_pushes_the_following_line_down() {
        let f = fonts();
        let r = rope("![x](cat.png)\nafter");
        let s = store("cat.png", 40, 200); // a tall image
        let l = layout_img(&f, &r, (20, 20), &s, ImageUi::default());
        let img = l.lines()[0].image.unwrap();
        let after = &l.lines()[1];
        let m = f.metrics(BODY);
        // "after" sits a full image-box height below the picture's top at TOP.
        assert_eq!(after.baseline - m.ascent, TOP + img.box_height());
    }

    #[test]
    fn image_at_hits_the_picture_and_button_only_when_selected() {
        let f = fonts();
        let r = rope("![x](cat.png)\nafter");
        let s = store("cat.png", 80, 80);
        let sel = layout_img(
            &f,
            &r,
            (20, 20),
            &s,
            ImageUi {
                selected: Some(0),
                raw: None,
                dragging: false,
            },
        );
        let rect = sel
            .image_rect(&f, &sel.lines()[0])
            .expect("the picture rect");
        // A point inside the picture selects that image.
        let inside = ((rect.x + 5) as f32, (rect.y + 5) as f32);
        assert_eq!(sel.image_at(&f, inside.0, inside.1), Some(0));
        // The corner button is hit only when the image is selected.
        let btn = image_button_rect(&rect, Scale::ONE);
        let bpt = ((btn.x + 1) as f32, (btn.y + 1) as f32);
        assert_eq!(sel.image_button_at(&f, bpt.0, bpt.1), Some(0));
        let idle = layout_img(&f, &r, (20, 20), &s, ImageUi::default());
        assert_eq!(
            idle.image_button_at(&f, bpt.0, bpt.1),
            None,
            "no button when idle"
        );
        // A point well below the picture hits nothing.
        assert_eq!(
            sel.image_at(&f, inside.0, (rect.y + rect.h + 500) as f32),
            None
        );
    }

    #[test]
    fn vertical_motion_steps_over_a_revealed_picture() {
        let f = fonts();
        let r = rope("![x](cat.png)\nafter");
        let s = store("cat.png", 80, 80);
        // Source revealed (cursor on the image line): rows are [source, picture,
        // after]. "![x](cat.png)" is 13 bytes, so "after" starts at byte 14.
        let l = layout_img(&f, &r, (0, 0), &s, ImageUi::default());
        // Down from the source skips the picture row and lands in "after" (>= 14).
        assert!(
            l.offset_vertical(&f, &r, 0, 1, 0.0) >= 14,
            "down skips the picture"
        );
        // Up from "after" skips the picture and lands back in the source (<= 13).
        assert!(
            l.offset_vertical(&f, &r, 16, -1, 0.0) <= 13,
            "up skips the picture"
        );
    }

    #[test]
    fn arrowing_onto_a_picture_only_image_lands_at_its_start() {
        let f = fonts();
        let r = rope("top\n![x](cat.png)");
        let s = store("cat.png", 80, 80);
        // Cursor on "top"; the image below is a picture-only row (not revealed).
        let l = layout_img(&f, &r, (0, 0), &s, ImageUi::default());
        // Down lands on the image's logical start (4), so the next build reveals it.
        assert_eq!(l.offset_vertical(&f, &r, 0, 1, 0.0), 4);
    }

    #[test]
    fn a_selection_over_an_image_covers_its_whole_box() {
        let f = fonts();
        let r = rope("![x](cat.png)\nafter");
        let s = store("cat.png", 80, 60);
        let l = layout_img(&f, &r, (20, 20), &s, ImageUi::default());
        // Select from the document start into "after": the image row's rect spans
        // the whole picture box, not a thin text-height strip.
        let rects = l.selection_rects(&f, &r, 0, 18);
        let img = l.lines()[0].image.unwrap();
        assert!(
            rects
                .iter()
                .any(|rc| rc.h == img.box_height() && rc.w == img.width),
            "the image's selection rect matches its box"
        );
    }

    #[test]
    fn selecting_a_revealed_source_does_not_box_the_picture() {
        let f = fonts();
        let r = rope("![x](cat.png)\nafter");
        let s = store("cat.png", 80, 60);
        // Raw revealed (cursor on the image line): rows are [source, picture, after].
        let l = layout_img(&f, &r, (0, 0), &s, ImageUi::default());
        let picture = l.lines()[1].image.expect("row 1 is the picture");
        // Select the source line. The picture row must not add its box: that would
        // make highlighting the raw text look like selecting the image.
        let rects = l.selection_rects(&f, &r, 0, 13);
        assert!(
            !rects.iter().any(|rc| rc.h == picture.box_height()),
            "the stacked picture is not boxed by a text selection of its source"
        );
        assert!(
            !rects.is_empty(),
            "the source text itself is still highlighted"
        );
    }

    #[test]
    fn hit_testing_an_emoji_line_snaps_to_cluster_boundaries() {
        let f = fonts();
        let r = rope("a😀b");
        let l = layout(&f, &r);
        let face = f.face(BODY, FontStyle::Regular);
        let m = f.metrics(BODY);
        let y = TOP as f32 + m.ascent as f32 / 2.0;
        // Sweep x across the whole line: every hit must land on a cluster
        // boundary (the emoji is bytes 1..5), never inside the cluster, whether
        // the emoji renders in color or as tofu.
        let end = LEFT as f32 + shape::text_advance(face, "a😀b") + 20.0;
        let mut x = 0.0f32;
        while x < end {
            let off = l.offset_at(&f, &r, x, y);
            assert!(
                [0usize, 1, 5, 6].contains(&off),
                "hit at x={x} returned mid-cluster offset {off}"
            );
            x += 3.0;
        }
    }

    // --- Tables -------------------------------------------------------------

    /// The table rows of a layout, in order, paired with the visual line's start.
    fn table_rows(l: &Layout) -> Vec<(usize, &TableRow)> {
        l.lines()
            .iter()
            .filter_map(|v| v.table.as_ref().map(|t| (v.start, t)))
            .collect()
    }

    #[test]
    fn a_table_lays_out_as_grid_rows_with_no_delimiter_line() {
        let f = fonts();
        let r = rope("| A | B |\n| --- | ---: |\n| 1 | 2 |\n");
        let l = layout(&f, &r);
        let rows = table_rows(&l);
        // Header and one body row become grid rows; the delimiter row draws nothing.
        assert_eq!(rows.len(), 2, "header + body, no delimiter row");
        assert_eq!(
            rows.iter().map(|&(s, _)| s).collect::<Vec<_>>(),
            vec![r.line_to_byte(0), r.line_to_byte(2)],
            "grid rows are the header and body lines"
        );
        let header = rows[0].1;
        assert!(header.header && header.first);
        assert_eq!(header.columns.len(), 2);
        assert_eq!(header.columns[0].align, markdown::Align::Left);
        assert_eq!(header.columns[1].align, markdown::Align::Right);
        assert!(!rows[1].1.header && !rows[1].1.first, "body row");
        // Every row of a table shares the same column geometry and width.
        assert_eq!(header.width, rows[1].1.width);
        assert_eq!(
            header.columns.iter().map(|c| c.x).collect::<Vec<_>>(),
            rows[1].1.columns.iter().map(|c| c.x).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_wide_cell_wraps_and_grows_the_row() {
        let f = fonts();
        let r = rope("| h |\n| --- |\n| several words that must wrap in a narrow column |\n");
        let l = Layout::build(&f, &r, LEFT, TOP, 160, BODY, NO_REVEAL);
        let body = table_rows(&l)
            .into_iter()
            .find(|(_, t)| !t.header)
            .expect("a body row")
            .1;
        assert!(
            body.cells[0].lines.len() > 1,
            "the long cell wrapped to several lines"
        );
        let lh = f.metrics(BODY).line_height;
        assert!(
            body.height >= 2 * lh,
            "the row grew to fit the wrapped cell"
        );
    }

    #[test]
    fn prose_around_a_table_is_not_swallowed() {
        let f = fonts();
        let r = rope("intro\n\n| a | b |\n| - | - |\n| 1 | 2 |\n\noutro\n");
        let l = layout(&f, &r);
        // The paragraphs above and below still render as ordinary text lines.
        let has = |s: &str| {
            l.lines()
                .iter()
                .any(|v| v.table.is_none() && r.byte_slice(v.start..v.end) == s)
        };
        assert!(has("intro") && has("outro"), "prose kept as text");
        assert_eq!(table_rows(&l).len(), 2, "only the header and body row");
    }

    #[test]
    fn pipe_lines_without_a_delimiter_stay_source() {
        let f = fonts();
        // Two piped lines but no delimiter row: ordinary text, never a table.
        let r = rope("| a | b |\n| c | d |\n");
        let l = layout(&f, &r);
        assert!(l.lines().iter().all(|v| v.table.is_none()));
    }

    #[test]
    fn a_body_row_short_or_long_pads_or_truncates_to_the_columns() {
        let f = fonts();
        let r = rope("| a | b | c |\n| - | - | - |\n| x |\n| p | q | r | s |\n");
        let l = layout(&f, &r);
        for (_, t) in table_rows(&l) {
            assert_eq!(
                t.cells.len(),
                3,
                "every row has exactly the header's columns"
            );
        }
    }

    #[test]
    fn caret_and_click_map_into_the_right_cell() {
        let f = fonts();
        let r = rope("| A | B |\n| --- | --- |\n| ab | cd |\n");
        let l = layout(&f, &r);
        let body = l
            .lines()
            .iter()
            .find(|v| v.table.as_ref().is_some_and(|t| !t.header))
            .expect("a body row");
        let t = body.table.as_ref().unwrap();
        let cells = crate::table::row_cells(&r, r.byte_to_line(body.start));
        let col1 = t.columns[1];
        let row_left = l.left() + body.indent;
        // The caret at the start of the "cd" cell lands at column 1's left edge
        // (both columns left-aligned) and within the row's vertical box.
        let (cx, cy, ch) = l.caret(&f, &r, cells[1].0);
        assert_eq!(cx, row_left + col1.x, "caret at column 1's content left");
        let box_top = body.baseline - f.metrics(body.size).ascent;
        assert!(
            cy >= box_top && cy + ch <= box_top + t.height,
            "caret within the row box"
        );
        // A click just inside that cell returns an offset within the "cd" cell.
        let hit = l.offset_at(&f, &r, cx as f32 + 2.0, cy as f32 + 2.0);
        assert!(
            hit >= cells[1].0 && hit <= cells[1].1,
            "click at {hit} lands inside the cd cell {:?}",
            cells[1]
        );
    }

    #[test]
    fn vertical_motion_enters_table_rows_skips_the_delimiter_and_exits() {
        let f = fonts();
        let r = rope("top\n| aa | bb |\n| - | - |\n| cc | dd |\nbot\n");
        let l = layout(&f, &r);
        // Aim for the caret's own column from inside "top".
        let start = 1;
        let goal = l.caret(&f, &r, start).0 as f32;
        // Down enters the header row's first cell.
        let header = l.offset_vertical(&f, &r, start, 1, goal);
        assert!(
            crate::table::cell_at(&r, header).is_some(),
            "Down lands in the header row"
        );
        // Down again skips the (invisible) delimiter row and lands in the body.
        let body = l.offset_vertical(&f, &r, header, 1, goal);
        assert_eq!(
            r.byte_to_line(body),
            3,
            "in the body row, past the delimiter"
        );
        // Down once more leaves the table for the prose below.
        let out = l.offset_vertical(&f, &r, body, 1, goal);
        assert!(
            crate::table::cell_at(&r, out).is_none(),
            "Down leaves the table"
        );
        assert_eq!(
            r.byte_slice(l.line_bounds(out).0..l.line_bounds(out).1),
            "bot"
        );
    }

    #[test]
    fn a_br_breaks_a_cell_onto_another_line() {
        let f = fonts();
        let r = rope("| a<br>b | c |\n| - | - |\n");
        let l = layout(&f, &r);
        let (_, t) = table_rows(&l)[0];
        assert_eq!(t.cells[0].lines.len(), 2, "the <br> split the cell");
        assert_eq!(t.cells[1].lines.len(), 1, "the other cell is one line");
        let lh = f.metrics(BODY).line_height;
        assert!(t.height >= 2 * lh, "the row is two text lines tall");
    }

    #[test]
    fn hover_finds_the_add_row_and_add_column_zones() {
        let f = fonts();
        let r = rope("| a | b |\n| - | - |\n| c | d |\n");
        let l = layout(&f, &r);
        let rows: Vec<&VisualLine> = l.lines().iter().filter(|v| v.table.is_some()).collect();
        let (top, bot) = (rows[0], *rows.last().unwrap());
        let width = top.table.as_ref().unwrap().width;
        let left = l.left() + top.indent;
        let box_top = top.baseline - f.metrics(top.size).ascent;
        let box_bottom =
            bot.baseline - f.metrics(bot.size).ascent + bot.table.as_ref().unwrap().box_height();
        // Just below the last row: the add-row strip, tied to this table.
        let add_row = l
            .table_button_at(&f, &r, (left + 5) as f32, (box_bottom + 4) as f32)
            .expect("add-row zone");
        assert_eq!(add_row.kind, TableButtonKind::AddRow);
        assert_eq!(add_row.block, crate::table::block_at(&r, 0).unwrap());
        // Just right of the box: the add-column strip.
        let add_col = l
            .table_button_at(&f, &r, (left + width + 4) as f32, (box_top + 4) as f32)
            .expect("add-column zone");
        assert_eq!(add_col.kind, TableButtonKind::AddColumn);
        // Inside a cell: no button.
        assert!(l
            .table_button_at(&f, &r, (left + 5) as f32, (box_top + 4) as f32)
            .is_none());
    }

    #[test]
    fn a_table_selection_highlights_whole_rows() {
        let f = fonts();
        let r = rope("| a | b |\n| - | - |\n| c | d |\n");
        let l = layout(&f, &r);
        let cells = crate::table::row_cells(&r, 2);
        // A selection within the body row's cells lights up the whole row box.
        let rects = l.selection_rects(&f, &r, cells[0].0, cells[1].0);
        assert_eq!(rects.len(), 1, "one full-row highlight, not per-cell");
        let (_, t) = table_rows(&l)[1]; // the body row
        assert_eq!(rects[0].w, t.width, "spans the table width");
        assert_eq!(rects[0].h, t.height, "spans the row height");
    }

    #[test]
    fn a_selection_inside_one_table_cell_is_precise_not_the_whole_row() {
        // Text selected inside a single cell highlights just that text, like prose;
        // the row-granular box is reserved for a selection that leaves the cell.
        let f = fonts();
        let r = rope("| hello | b |\n| - | - |\n| c | d |\n");
        let l = layout(&f, &r);
        let (_, t) = table_rows(&l)[0]; // the header row
        let cell0 = &t.cells[0];
        let start = cell0.start;
        let end = (cell0.start + 3).min(cell0.end); // a few chars within "hello"
        assert!(end > start, "the cell has content to select");
        let rects = l.selection_rects(&f, &r, start, end);
        assert_eq!(rects.len(), 1, "one precise rect inside the cell");
        assert!(rects[0].w > 0, "with real width");
        assert!(rects[0].w < t.width, "narrower than the whole row");
        assert!(
            rects[0].h < t.height,
            "a text-line height, not the whole box"
        );
    }

    #[test]
    fn a_table_cell_selection_fills_the_slot_centered_on_the_text() {
        // In a cell the glyph box is already centered in its line slot, so the
        // centered fill covers the whole slot (line-box height) and stays balanced
        // around the text, the same fill prose gets.
        let f = fonts_scaled(1.5);
        let m = f.metrics(BODY);
        let pad = leading_pad(m);
        let r = rope("| hello | b |\n| - | - |\n| c | d |\n");
        let l = layout(&f, &r);
        let (_, t) = table_rows(&l)[0]; // the header row
        let cell0 = &t.cells[0];
        let start = cell0.start;
        let end = (cell0.start + 3).min(cell0.end);
        let rects = l.selection_rects(&f, &r, start, end);
        assert_eq!(rects.len(), 1, "one precise rect inside the cell");
        let (_, caret_top, _) = l.caret(&f, &r, start);
        assert_eq!(rects[0].h, m.line_height, "fills the cell line slot");
        assert_eq!(rects[0].y, caret_top - pad, "centered on the glyph box");
        assert!(rects[0].w < t.width, "still narrower than the whole row");
    }
}
