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
//! marker, which persists (it is the only sign the line is a list or quote). Wrapping
//! always measures the source text, so this collapse only shifts runs
//! horizontally within a line, never reflowing the document.
//!
//! A list or quote item also hangs its wrapped continuations under the text: each
//! visual line past the first is inset by the marker's width (`VisualLine::indent`)
//! so the column aligns. Once the cursor leaves the line the raw marker gives way
//! to a [`Gutter`] decoration, a bullet point `•` for a list, a drawn checkbox
//! for a task item, or a bar for a quote, always the same width as the source
//! marker so the text never shifts; on the cursor's own line the raw, editable
//! marker shows instead.
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

use crate::freetype::{Face, FaceKey, FontStyle, Fonts};
use crate::grapheme;
use crate::image::ImageStore;
use crate::markdown;
use crate::rope::Rope;
use crate::shape;

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

/// Vertical breathing room above and below a rendered block image, so it does not
/// butt against the lines around it. Part of the image line's box height.
const IMAGE_PAD_Y: i32 = 8;

/// Extra pixels around a drawn task checkbox that still count as clicking it,
/// so the small square is comfortable to hit.
const CHECKBOX_HIT_PAD: i32 = 3;

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
    /// `Some(checked)` when this line belongs to a task-list item, on every
    /// visual line the item wraps into. A checked item's text renders dim and
    /// struck through; the checkbox itself is the first line's
    /// [`Gutter::Checkbox`]. `None` off task items (and on an over-nested one,
    /// which renders as plain body text).
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
        self.height + 2 * IMAGE_PAD_Y
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
}

/// The live-preview decoration of a visual line's left gutter, chosen once the
/// cursor leaves the line. On the cursor's own line the raw marker shows so it
/// stays editable, so this is [`Gutter::None`] there. A decoration always occupies
/// the same width as the source marker, so toggling it never shifts the text (the
/// no-reflow invariant the whole module preserves).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gutter {
    /// No decoration: body and heading lines, ordered items (their number stays
    /// put), and any list or quote line the cursor is currently on.
    None,
    /// Paint a bullet point `•` in place of a `-`/`+`/`*` marker's glyph.
    Bullet,
    /// Draw a checkbox (built from fills, no glyph dependency) in place of a
    /// task item's `- [ ]`/`- [x]` marker, once the cursor leaves the line; on
    /// the cursor's own line the raw marker shows for editing instead. The box
    /// geometry is [`Layout::checkbox_rect`], shared with click hit-testing.
    Checkbox { checked: bool },
    /// Draw a blockquote bar down the gutter. `marker` is true on the first line,
    /// where the bar stands in for the `>` marker (whose glyphs are not drawn),
    /// and false on wrapped continuations, which have no marker run of their own.
    QuoteBar { marker: bool },
}

/// One inline run of a visual line: an absolute byte range, what it is, and
/// whether it is drawn. A marker run is hidden (`visible == false`) on a line
/// the cursor has left, the live-preview collapse; text runs are always visible.
#[derive(Clone, Copy)]
pub struct Run {
    pub start: usize,
    pub end: usize,
    pub kind: markdown::SpanKind,
    pub visible: bool,
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

/// The [`FaceKey`] a run of `kind` at `size` draws in, the borrow-free twin of
/// [`run_face`]: the display list records this so a painted run can resolve its
/// face later without holding a borrow of [`Fonts`]. Same decision as `run_face`,
/// so the two never disagree on which face (hence which width) a run uses.
pub(crate) fn face_key(size: u32, kind: markdown::SpanKind, heading: bool) -> FaceKey {
    match kind {
        markdown::SpanKind::Text(markdown::Emphasis::Code) => FaceKey::Code { size },
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

/// An axis-aligned rectangle in window pixels.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
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
        Self::build_with_images(
            fonts,
            rope,
            left,
            top,
            wrap_width,
            body_size,
            active,
            &ImageStore::new(None),
            ImageUi::default(),
        )
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
    #[allow(clippy::too_many_arguments)]
    pub fn build_with_images(
        fonts: &Fonts,
        rope: &Rope,
        left: i32,
        top: i32,
        wrap_width: i32,
        body_size: u32,
        active: (usize, usize),
        images: &ImageStore,
        image_ui: ImageUi,
    ) -> Self {
        let n = rope.len_lines();
        Self::build_range(
            fonts, rope, left, top, wrap_width, body_size, active, images, image_ui, 0, n, None,
            None,
        )
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
    #[allow(clippy::too_many_arguments)]
    pub fn build_window(
        fonts: &Fonts,
        rope: &Rope,
        left: i32,
        box_top_seed: i32,
        wrap_width: i32,
        body_size: u32,
        active: (usize, usize),
        images: &ImageStore,
        image_ui: ImageUi,
        index: &LineIndex,
        band_start: usize,
        band_end: usize,
    ) -> Self {
        let n = rope.len_lines();
        let start = band_start.min(n);
        let end = band_end.min(n).max(start);
        let (fence, list_depth) = index.entry_state(start);
        Self::build_range(
            fonts,
            rope,
            left,
            box_top_seed,
            wrap_width,
            body_size,
            active,
            images,
            image_ui,
            start,
            end,
            fence,
            list_depth,
        )
    }

    /// Lay out logical lines `[start, end)` starting at content-space top
    /// `box_top_seed`, seeded with the cross-line carry (`fence`, `list_depth`)
    /// entering `start`. The one loop behind both the full build
    /// ([`Self::build_with_images`], `0..n` from ground state) and the windowed
    /// build ([`Self::build_window`], a band seeded from the index).
    #[allow(clippy::too_many_arguments)]
    fn build_range(
        fonts: &Fonts,
        rope: &Rope,
        left: i32,
        box_top_seed: i32,
        wrap_width: i32,
        body_size: u32,
        active: (usize, usize),
        images: &ImageStore,
        image_ui: ImageUi,
        start: usize,
        end: usize,
        mut fence: Option<usize>,
        mut list_depth: Option<usize>,
    ) -> Self {
        let n = rope.len_lines();
        let mut box_top = box_top_seed;
        let mut pool = LayoutScratch::default();
        let mut lines = Vec::new();
        for i in start..end {
            let ls = rope.line_to_byte(i);
            // Every logical line but the last ends in a '\n'; its content excludes
            // that byte. The rope materializes just this line inside `lay_line`.
            let has_newline = i + 1 < n;
            let le = if has_newline {
                rope.line_to_byte(i + 1) - 1
            } else {
                rope.line_to_byte(i + 1)
            };
            box_top += lay_line(
                fonts,
                rope,
                ls,
                le,
                has_newline,
                wrap_width,
                body_size,
                active,
                images,
                image_ui,
                box_top,
                &mut fence,
                &mut list_depth,
                &mut pool,
                false,
                &mut lines,
            );
        }
        Self {
            lines,
            runs: pool.runs,
            left,
            text_height: box_top - box_top_seed,
        }
    }

    pub fn lines(&self) -> &[VisualLine] {
        &self.lines
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

    /// The byte range `(start, end)` of the visual line containing `offset`,
    /// for Home/End motion.
    pub fn line_bounds(&self, offset: usize) -> (usize, usize) {
        let line = self.line_for(offset);
        (line.start, line.end)
    }

    /// The caret rectangle (its x, top y, and height) for the cursor at `offset`,
    /// sized to the metrics of the line it lands on. The caret only ever lands on
    /// a revealed line (the cursor's own line shows its markers), so its `offset`
    /// never falls inside a hidden run.
    pub fn caret(&self, fonts: &Fonts, rope: &Rope, offset: usize) -> (i32, i32, i32) {
        let line = self.line_for(offset);
        let m = fonts.metrics(line.size);
        let x = self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, offset);
        (
            x.round() as i32,
            line.baseline - m.ascent,
            m.ascent + m.descent,
        )
    }

    /// The byte offset nearest a click at window `(x, y)`.
    pub fn offset_at(&self, fonts: &Fonts, rope: &Rope, x: f32, y: f32) -> usize {
        let line = self.line_for_y(fonts, y);
        self.hit_line(fonts, rope, line, x)
    }

    /// If the visual line at content-space `y` is a rendered horizontal rule (a
    /// thematic break whose cursor has left), its content byte range
    /// `(start, end)`. A click there selects the whole rule rather than dropping a
    /// caret between its `---` markers; `None` for any other line.
    pub fn rule_at(&self, fonts: &Fonts, y: f32) -> Option<(usize, usize)> {
        let line = self.line_for_y(fonts, y);
        line.rule.then_some((line.start, line.end))
    }

    /// The drawn checkbox's rectangle in content space for a task line whose
    /// gutter is [`Gutter::Checkbox`]: a square centered over the marker's
    /// `[ ]`/`[x]` brackets (so toggling the reveal barely shifts it), sized to
    /// the line's metrics and never wider than the marker it replaces (the
    /// width-matched substitution the no-reflow invariant needs). The painter
    /// draws the box here and [`Self::checkbox_at`] hit-tests clicks against it,
    /// so the two always agree. `None` for any other line, including a revealed
    /// task line, which shows its raw editable marker instead.
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
        let brackets = marker.get(indent + 2..indent + 5)?;
        let face = fonts.face(line.size, FontStyle::Regular);
        let pre = shape::text_advance(face, marker.get(..indent + 2)?);
        let brackets_w = shape::text_advance(face, brackets);
        let m = fonts.metrics(line.size);
        let side = (m.ascent * 4 / 5).max(8);
        let x = self.left + line.indent + (pre + (brackets_w - side as f32) / 2.0).round() as i32;
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
    /// content-space `(x, y)`, or `None`. Only a non-revealed task line draws a
    /// checkbox (the cursor's own line shows its raw marker, where clicks
    /// edit), so a hit here always means "toggle", never "place the caret".
    /// The target is padded a little past the drawn box for clickability.
    pub fn checkbox_at(&self, fonts: &Fonts, rope: &Rope, x: f32, y: f32) -> Option<usize> {
        let line = self.line_for_y(fonts, y);
        let rect = self.checkbox_rect(fonts, rope, line)?;
        let target = Rect {
            x: rect.x - CHECKBOX_HIT_PAD,
            y: rect.y - CHECKBOX_HIT_PAD,
            w: rect.w + 2 * CHECKBOX_HIT_PAD,
            h: rect.h + 2 * CHECKBOX_HIT_PAD,
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
            y: line.baseline - m.ascent + IMAGE_PAD_Y,
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
            (img.selected && rect_contains(&image_button_rect(&rect), x, y)).then_some(line.start)
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
        // on it (at its start), revealing the source on the next build.
        while target >= 0
            && (target as usize) < self.lines.len()
            && self.is_stacked_widget(target as usize)
        {
            target += dir;
        }
        if target < 0 || target as usize >= self.lines.len() {
            return offset;
        }
        self.hit_line(fonts, rope, &self.lines[target as usize], goal_x)
    }

    /// The highlight rectangles covering the byte range `[start, end)`, one per
    /// visual line it touches, each sized to that line's metrics.
    pub fn selection_rects(
        &self,
        fonts: &Fonts,
        rope: &Rope,
        start: usize,
        end: usize,
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
            // touching one (in practice a revealed line) measures sensibly.
            let x0 = self.left as f32
                + line.indent as f32
                + self.advance_to(fonts, rope, line, sel_start);
            let x1 =
                self.left as f32 + line.indent as f32 + self.advance_to(fonts, rope, line, sel_end);
            let mut w = (x1 - x0).round() as i32;
            if newline_selected {
                w += NEWLINE_PAD;
            }
            rects.push(Rect {
                x: x0.round() as i32,
                y: line.baseline - m.ascent,
                w: w.max(1),
                h: m.ascent + m.descent,
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
            for run in self.run_slice(line) {
                if !run.visible
                    || !matches!(run.kind, markdown::SpanKind::Text(markdown::Emphasis::Code))
                {
                    continue;
                }
                let x0 = self.left as f32
                    + line.indent as f32
                    + self.advance_to(fonts, rope, line, run.start);
                let w = shape::text_advance(
                    run_face(fonts, line.size, run.kind, line.heading.is_some()),
                    &rope.slice(run.start..run.end).as_cow(),
                );
                rects.push(Rect {
                    x: x0.round() as i32,
                    y: line.baseline - m.ascent,
                    w: (w.round() as i32).max(1),
                    h: m.ascent + m.descent,
                });
            }
        }
        rects
    }

    /// The visual line containing `offset`. A cursor at a soft-wrap boundary is
    /// given to the later line; one at a hard newline stays on its line. Image
    /// picture lines are skipped: they are never caret targets, so the caret lands
    /// on the source text line (which a revealed image always has above its
    /// picture). Falls back to the last text line for an offset at the document end.
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
            .find(|l| {
                let m = fonts.metrics(l.size);
                let bottom = match l.image {
                    // An image line's band spans its full (taller) box, so a click
                    // anywhere on the picture lands on it and reveals its source.
                    Some(img) => l.baseline - m.ascent + img.box_height(),
                    None => l.baseline + m.descent,
                };
                y < bottom as f32
            })
            .unwrap_or_else(|| self.last_line())
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
            for (off, cluster) in grapheme::graphemes(&text) {
                let w = shape::text_advance(face, cluster);
                if x < acc + w / 2.0 {
                    return run.start + off;
                }
                acc += w;
            }
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
            let end = offset.min(run.end);
            let face = run_face(fonts, line.size, run.kind, line.heading.is_some());
            x += shape::text_advance(face, &rope.slice(run.start..end).as_cow());
        }
        x
    }

    fn last_line(&self) -> &VisualLine {
        // build always pushes at least one line, so this never panics.
        debug_assert!(!self.lines.is_empty());
        &self.lines[self.lines.len() - 1]
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
    /// One entry per grapheme cluster: its byte offset, pixel width, and whether it
    /// is a space (a wrap opportunity). Sized to the widest line seen.
    clusters: Vec<(usize, f32, bool)>,
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

/// Lay out one logical line `[ls, le)` starting at content-space top `box_top`,
/// appending its visual line(s) to `out` and returning the height it added (the
/// amount to advance `box_top` before the next line). `fence` and `list_depth` are
/// the only cross-line carry, threaded by `&mut`; everything else is a pure
/// function of this line's text plus `active`/`image_ui`, which affect reveal and
/// image widgets only, never wrap points or baselines. This is the single per-line
/// unit shared by the full [`Layout::build_with_images`] loop and (later) the
/// windowed build, so the two can never disagree on a line's shape.
///
/// `pool` supplies the per-line scratch buffers (`clusters`, `segments`), owned by
/// the caller's build loop and reused across every line so laying out a line
/// clears and refills them instead of allocating fresh Vecs.
#[allow(clippy::too_many_arguments)]
fn lay_line(
    fonts: &Fonts,
    rope: &Rope,
    ls: usize,
    le: usize,
    has_newline: bool,
    wrap_width: i32,
    body_size: u32,
    active: (usize, usize),
    images: &ImageStore,
    image_ui: ImageUi,
    box_top: i32,
    fence: &mut Option<usize>,
    list_depth: &mut Option<usize>,
    pool: &mut LayoutScratch,
    // When true the caller only wants each line's height and image flag (the
    // `LineIndex` build), so the styled runs are skipped: `clip_runs` allocates a
    // `Vec` per visual row that measurement would immediately discard.
    measure_only: bool,
    out: &mut Vec<VisualLine>,
) -> i32 {
    // Running box top within this logical line; the return value is how far it
    // advanced past the incoming `box_top`.
    let mut y = box_top;
    let line = rope.slice(ls..le).as_cow();
    // Whether the cursor/selection touches this logical line: the same reveal test
    // headings use, applied to fence lines too.
    let on_line = active.0 <= le && active.1 >= ls;
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
    if fence.is_none() {
        if let Some(link) = markdown::standalone_image(&line) {
            if let Some((iw, ih)) = images.dimensions(&line[link.path.0..link.path.1]) {
                *list_depth = None;
                let column = if wrap_width > 0 {
                    wrap_width
                } else {
                    iw as i32
                };
                let (width, height) = fit_image(iw, ih, column);
                let image = ImageBox {
                    path_start: ls + link.path.0,
                    path_end: ls + link.path.1,
                    width,
                    height,
                    selected: image_ui.selected == Some(ls),
                };
                if on_line || image_ui.raw == Some(ls) {
                    // Source revealed: emit it (fall through), then the widget.
                    pending_image = Some(image);
                } else {
                    // Picture only: one widget line, no source text.
                    out.push(VisualLine {
                        start: ls,
                        end: le,
                        baseline: y + fonts.metrics(body_size).ascent,
                        hard: has_newline,
                        size: body_size,
                        runs: 0..0,
                        indent: 0,
                        gutter: Gutter::None,
                        code_block: false,
                        heading: None,
                        task: None,
                        rule: false,
                        image: Some(image),
                    });
                    y += image.box_height();
                    return y - box_top;
                }
            }
        }
    }
    // A line inside a fenced code block is code, not markdown: it skips the inline
    // pass and renders as one code run. The ``` fence lines hide their backticks off
    // the cursor's line, like a heading's `#`. A code line carries `Body` as an
    // inert style (its gutter is unused).
    let (style, size, block_marker, block_persists, revealed, code_block) =
        if let Some(open) = *fence {
            *list_depth = None;
            let delimiter = markdown::closes_code_fence(&line, open);
            if delimiter {
                *fence = None;
            }
            code_block_spans_into(&line, delimiter, &mut pool.spans);
            // Only a fence delimiter line has backticks to hide.
            let revealed = !delimiter || on_line;
            (
                markdown::LineStyle::Body,
                body_size,
                0,
                false,
                revealed,
                true,
            )
        } else if let Some(open) = markdown::opens_code_fence(&line) {
            *list_depth = None;
            *fence = Some(open);
            code_block_spans_into(&line, true, &mut pool.spans);
            (
                markdown::LineStyle::Body,
                body_size,
                0,
                false,
                on_line,
                true,
            )
        } else {
            // Resolve list nesting across lines: an item deeper than one past the
            // previous item has no parent and renders as body.
            let (style, marker) = resolve_list(&line, list_depth);
            line_spans_into(&line, marker, &mut pool.inline, &mut pool.spans);
            (
                style,
                markdown::font_size(style, body_size),
                marker,
                markdown::block_marker_persists(style),
                on_line,
                false,
            )
        };
    // A revealed image's source always shows verbatim (its markers, if any, are not
    // hidden), so the text above the picture reads as written.
    let revealed = revealed || pending_image.is_some();
    let metrics = fonts.metrics(size);
    // Heading lines color their text by level (fence/code lines carry the inert
    // `Body` style, so they are never headings here).
    let heading = match style {
        markdown::LineStyle::Heading(level) => Some(level),
        _ => None,
    };
    // Stamped on every segment of a task item, so a checked item's text dims and
    // strikes through wrapped continuations too.
    let task = match style {
        markdown::LineStyle::Task { checked } => Some(checked),
        _ => None,
    };
    // A thematic break draws as a horizontal rule once the cursor leaves it; on the
    // line itself the raw `---` shows (and `revealed` is true).
    let rule = matches!(style, markdown::LineStyle::Rule) && !revealed;
    // A list or quote item hangs its wrapped continuations under the text: the inset
    // is the marker's width, measured on the source marker in the regular face at
    // this line's size, so it matches the width the marker run itself advances.
    let block_indent = if block_persists && block_marker > 0 {
        shape::text_advance(fonts.face(size, FontStyle::Regular), &line[..block_marker]).round()
            as i32
    } else {
        0
    };
    // A left inset shared by every segment of the line, and how much it narrows the
    // wrap width: a code block pads both sides inside its panel; a list item shifts
    // right by a small indent (one side only).
    let is_list = matches!(
        style,
        markdown::LineStyle::Bullet
            | markdown::LineStyle::Ordered
            | markdown::LineStyle::Task { .. }
    );
    let (seg_pad, wrap_reduce) = if code_block {
        (CODE_PAD_X, 2 * CODE_PAD_X)
    } else if is_list {
        (LIST_INDENT, LIST_INDENT)
    } else {
        (0, 0)
    };
    let line_wrap = if wrap_width > 0 {
        (wrap_width - wrap_reduce).max(1)
    } else {
        wrap_width
    };
    // `wrap_segments` returns ranges relative to the line; offset them back to
    // absolute buffer positions. It measures the source text, so clipping spans to
    // each segment never disagrees with the breaks, and reveal state (which only
    // hides markers) never reflows the document.
    wrap_segments(
        fonts,
        size,
        &line,
        &pool.spans,
        line_wrap,
        block_indent,
        heading.is_some(),
        &mut pool.clusters,
        &mut pool.segments,
    );
    let last = pool.segments.len() - 1;
    for k in 0..pool.segments.len() {
        let (start, end) = pool.segments[k];
        // Every segment carries the shared left inset (`seg_pad`); wrapped
        // continuations add the marker's width on top so their text hangs under the
        // first line's text.
        let indent = seg_pad + if k == 0 { 0 } else { block_indent };
        // A bullet shows its dot even on the cursor's own line, so a list stays a
        // list while you edit it. A quote reveals its raw `> ` for editing, drawing
        // the bar only once the cursor leaves.
        let gutter = match style {
            markdown::LineStyle::Bullet if k == 0 => Gutter::Bullet,
            // A task item draws its checkbox only once the cursor leaves: on the line
            // itself the raw `- [ ]` is the editable source, and a click there edits.
            markdown::LineStyle::Task { checked } if k == 0 && !revealed => {
                Gutter::Checkbox { checked }
            }
            markdown::LineStyle::Quote if !revealed => Gutter::QuoteBar { marker: k == 0 },
            _ => Gutter::None,
        };
        // Append this segment's runs to the shared pool and record their range; the
        // measure-only path (the index build) wants no runs, leaving an empty range.
        let runs_start = pool.runs.len() as u32;
        if !measure_only {
            clip_runs_into(
                &pool.spans,
                start,
                end,
                ls,
                revealed,
                block_marker,
                block_persists,
                &mut pool.runs,
            );
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
            task,
            rule,
            image: None,
        });
        y += metrics.line_height;
    }
    // The picture sits below its just-emitted source line(s) (the reveal case). It is
    // a widget line: never a caret target, taller than text.
    if let Some(image) = pending_image {
        out.push(VisualLine {
            start: ls,
            end: le,
            // The source line above owns the caret and the logical line's newline;
            // this widget carries neither.
            baseline: y + metrics.ascent,
            hard: false,
            size: body_size,
            runs: 0..0,
            indent: 0,
            gutter: Gutter::None,
            code_block: false,
            heading: None,
            task: None,
            rule: false,
            image: Some(image),
        });
        y += image.box_height();
    }
    y - box_top
}

/// An `active` range that reveals nothing: `MAX <= le` is always false, so every
/// line lays out with its markers collapsed and no image source shown. This is the
/// base geometry the [`LineIndex`] measures, reveal-independent by construction.
const NO_REVEAL: (usize, usize) = (usize::MAX, usize::MAX);

/// Below this line count, [`LineIndex::build_parallel`] just builds sequentially:
/// the per-worker font open and thread hand-off cost more than the work saved.
const PARALLEL_MIN_LINES: usize = 20_000;

/// One worker's output in [`LineIndex::build_parallel`]: the `LineRec`s it
/// measured for its chunk (speculatively, from ground entry state) and the entry
/// state it ended in, which the stitch pass checks against the true carry.
struct ChunkOut {
    recs: Vec<LineRec>,
    end_fence: Option<usize>,
    end_depth: Option<usize>,
}

/// Measure logical lines `[lo, hi)` of `rope` into `LineRec`s, starting from the
/// given entry state (`None`/`None` is ground, no fence or list), and report the
/// entry state left at the end. The single measure loop behind the sequential
/// [`LineIndex::build`], the parallel workers, and the stitch re-measure; a worker
/// runs it on its own `fonts`.
#[allow(clippy::too_many_arguments)]
fn measure_chunk(
    fonts: &Fonts,
    rope: &Rope,
    lo: usize,
    hi: usize,
    n: usize,
    wrap_width: i32,
    body_size: u32,
    images: &ImageStore,
    mut fence: Option<usize>,
    mut list_depth: Option<usize>,
) -> ChunkOut {
    let mut scratch: Vec<VisualLine> = Vec::new();
    let mut pool = LayoutScratch::default();
    let mut recs = Vec::with_capacity(hi - lo);
    for i in lo..hi {
        recs.push(LineIndex::measure_line(
            fonts,
            rope,
            i,
            n,
            wrap_width,
            body_size,
            images,
            &mut fence,
            &mut list_depth,
            &mut pool,
            &mut scratch,
        ));
    }
    ChunkOut {
        recs,
        end_fence: fence,
        end_depth: list_depth,
    }
}

/// One logical line's contribution to the document's vertical geometry.
#[derive(Clone, Copy, PartialEq, Debug)]
struct LineRec {
    /// Base (unrevealed) pixel height: the sum of the line's visual rows.
    height: i32,
    /// Extra height the source text adds above a standalone picture when its line
    /// is revealed; 0 for every non-image line. Applied as a per-query overlay so
    /// the base height stays a pure function of `(version, width)`.
    reveal_extra: i32,
    /// Open code-fence backtick count entering this line (0 = not in a fence).
    fence_in: u16,
    /// List nesting depth entering this line (-1 = not in a list).
    depth_in: i16,
}

/// A ≤2-line overlay of the extra height revealed image sources add on top of the
/// base [`LineIndex`] geometry (the cursor's image line, and a raw-pinned one).
/// Empty is the base geometry.
#[derive(Clone, Copy, Default)]
pub struct RevealAdjust {
    entries: [(usize, i32); 2],
    len: usize,
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

    fn total(&self) -> i32 {
        self.entries[..self.len].iter().map(|&(_, e)| e).sum()
    }

    fn before(&self, line: usize) -> i32 {
        self.entries[..self.len]
            .iter()
            .filter(|&&(l, _)| l < line)
            .map(|&(_, e)| e)
            .sum()
    }
}

/// The document's vertical geometry as a per-logical-line index: each line's base
/// pixel height and the cross-line carry state entering it. Built by measuring
/// every line once via [`lay_line`], it computes total content height, maps a
/// logical line to a content-space position, and seeds a windowed layout that
/// starts mid-document, all without keeping the whole document's runs. A cursor
/// move or scroll reuses the index unchanged (heights do not depend on reveal);
/// only an edit or resize rebuilds it.
#[derive(Clone)]
pub struct LineIndex {
    recs: Vec<LineRec>,
    /// Cumulative base height: `tops[i]` is the content-space top of logical line
    /// `i` (0-relative); `tops[len]` is the total base height.
    tops: Vec<i32>,
}

impl LineIndex {
    /// Measure one logical line `i`'s record, advancing the carry state
    /// (`fence`/`list_depth`) through it. Shared by [`Self::build`] and
    /// [`Self::update`] so the incremental path can never disagree with a rebuild.
    #[allow(clippy::too_many_arguments)]
    fn measure_line(
        fonts: &Fonts,
        rope: &Rope,
        i: usize,
        n: usize,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
        fence: &mut Option<usize>,
        list_depth: &mut Option<usize>,
        pool: &mut LayoutScratch,
        scratch: &mut Vec<VisualLine>,
    ) -> LineRec {
        let ls = rope.line_to_byte(i);
        let has_newline = i + 1 < n;
        let le = if has_newline {
            rope.line_to_byte(i + 1) - 1
        } else {
            rope.line_to_byte(i + 1)
        };
        let fence_in = *fence;
        let depth_in = *list_depth;
        scratch.clear();
        let height = lay_line(
            fonts,
            rope,
            ls,
            le,
            has_newline,
            wrap_width,
            body_size,
            NO_REVEAL,
            images,
            ImageUi::default(),
            0,
            fence,
            list_depth,
            pool,
            true,
            scratch,
        );
        // Only a decoded standalone image changes height when revealed; measure
        // that once here. Re-running from the pre-line carry reproduces the same
        // state mutation (an image forces list_depth=None, fence must be None), so
        // the throwaway `f2`/`d2` are discarded.
        let reveal_extra = if scratch.iter().any(|v| v.image.is_some()) {
            let mut f2 = fence_in;
            let mut d2 = depth_in;
            scratch.clear();
            let revealed = lay_line(
                fonts,
                rope,
                ls,
                le,
                has_newline,
                wrap_width,
                body_size,
                (ls, le),
                images,
                ImageUi::default(),
                0,
                &mut f2,
                &mut d2,
                pool,
                true,
                scratch,
            );
            revealed - height
        } else {
            0
        };
        LineRec {
            height,
            reveal_extra,
            fence_in: fence_in.map_or(0, |c| c as u16),
            depth_in: depth_in.map_or(-1, |d| d as i16),
        }
    }

    /// Build the cumulative-height prefix (`tops`) from a record list.
    fn from_recs(recs: Vec<LineRec>) -> Self {
        let mut tops = Vec::with_capacity(recs.len() + 1);
        let mut acc = 0;
        for rec in &recs {
            tops.push(acc);
            acc += rec.height;
        }
        tops.push(acc);
        LineIndex { recs, tops }
    }

    /// Measure every logical line's base geometry and carry state. `images` is the
    /// pane's decoded-image store, so a standalone `![](...)` line is recognized as
    /// a picture (and its `reveal_extra` recorded) exactly as the full build sees it.
    pub fn build(
        fonts: &Fonts,
        rope: &Rope,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
    ) -> Self {
        let n = rope.len_lines();
        let out = measure_chunk(
            fonts, rope, 0, n, n, wrap_width, body_size, images, None, None,
        );
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
        threads: usize,
    ) -> Self {
        let n = rope.len_lines();
        let threads = threads.max(1);
        if threads == 1 || n < PARALLEL_MIN_LINES {
            return Self::build(fonts, rope, wrap_width, body_size, images);
        }
        Self::build_chunked(fonts, rope, wrap_width, body_size, images, threads)
    }

    /// The chunked build behind [`build_parallel`](Self::build_parallel), without
    /// the small-document threshold, so a test can force many small chunks to
    /// exercise the stitch and fence/list fix-up paths.
    fn build_chunked(
        fonts: &Fonts,
        rope: &Rope,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
        threads: usize,
    ) -> Self {
        let n = rope.len_lines();
        let chunk = n.div_ceil(threads);
        let ranges: Vec<(usize, usize)> = (0..threads)
            .map(|t| (t * chunk, ((t + 1) * chunk).min(n)))
            .filter(|&(lo, hi)| lo < hi)
            .collect();

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
                        Some(measure_chunk(
                            &fonts, rope, lo, hi, n, wrap_width, body_size, images, None, None,
                        ))
                    })
                })
                .collect();
            let (lo0, hi0) = ranges[0];
            let mut outs = vec![Some(measure_chunk(
                fonts, rope, lo0, hi0, n, wrap_width, body_size, images, None, None,
            ))];
            for h in handles {
                outs.push(h.join().ok().flatten());
            }
            outs
        });

        // A worker that could not open fonts leaves a hole; rather than a partial
        // index, fall back to a full sequential build.
        if outs.iter().any(Option::is_none) {
            return Self::build(fonts, rope, wrap_width, body_size, images);
        }
        let outs: Vec<ChunkOut> = outs.into_iter().flatten().collect();

        // Stitch in order: accept a chunk whose ground-state guess matched its true
        // entry state, and re-measure (sequentially, with the caller's fonts) any
        // chunk a fence or list carried into.
        let mut recs: Vec<LineRec> = Vec::with_capacity(n);
        let mut fence: Option<usize> = None;
        let mut list_depth: Option<usize> = None;
        for (out, &(lo, hi)) in outs.iter().zip(&ranges) {
            if fence.is_none() && list_depth.is_none() {
                recs.extend_from_slice(&out.recs);
                fence = out.end_fence;
                list_depth = out.end_depth;
            } else {
                let re = measure_chunk(
                    fonts, rope, lo, hi, n, wrap_width, body_size, images, fence, list_depth,
                );
                recs.extend_from_slice(&re.recs);
                fence = re.end_fence;
                list_depth = re.end_depth;
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
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        fonts: &Fonts,
        new: &Rope,
        prev: &Rope,
        wrap_width: i32,
        body_size: u32,
        images: &ImageStore,
    ) {
        let new_n = new.len_lines();
        let old_n = self.recs.len();
        if old_n != prev.len_lines() {
            *self = Self::build(fonts, new, wrap_width, body_size, images);
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

        let (mut fence, mut list_depth) = self.entry_state(front);
        let mut mid: Vec<LineRec> = Vec::new();
        let mut scratch: Vec<VisualLine> = Vec::new();
        let mut pool = LayoutScratch::default();
        let mut i = front;
        let converge_old = loop {
            let old_i = i as isize - delta;
            // Converge only once past the changed text (i in the suffix) and the
            // incoming carry state matches the old index there; then the rest is
            // provably identical and reused.
            if i >= resync && old_i >= front as isize && (old_i as usize) < old_n {
                let old_state = self.entry_state(old_i as usize);
                if (fence, list_depth) == old_state {
                    break old_i as usize;
                }
            }
            if i >= new_n {
                break old_n;
            }
            mid.push(Self::measure_line(
                fonts,
                new,
                i,
                new_n,
                wrap_width,
                body_size,
                images,
                &mut fence,
                &mut list_depth,
                &mut pool,
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

    /// The heap bytes the index holds: the per-line records and their prefix-sum
    /// tops. For the memory report.
    pub fn heap_bytes(&self) -> usize {
        self.recs.capacity() * std::mem::size_of::<LineRec>()
            + self.tops.capacity() * std::mem::size_of::<i32>()
    }

    /// Total content height, including any image sources the overlay reveals.
    pub fn content_height(&self, adjust: &RevealAdjust) -> i32 {
        self.tops[self.recs.len()] + adjust.total()
    }

    /// Content-space top of logical line `line` (0-relative), pushed down by any
    /// earlier revealed image.
    pub fn logical_top(&self, line: usize, adjust: &RevealAdjust) -> i32 {
        let line = line.min(self.recs.len());
        self.tops[line] + adjust.before(line)
    }

    /// The logical line whose row band contains content-space `y`: the last line
    /// whose top is `<= y`, clamped into range. Turns a scroll offset into a window
    /// band edge. `logical_top` is monotonic in `line`, so a binary search finds it.
    pub fn line_at_y(&self, y: i32, adjust: &RevealAdjust) -> usize {
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
    pub fn reveal_extra(&self, line: usize) -> i32 {
        self.recs.get(line).map_or(0, |r| r.reveal_extra)
    }

    /// The cross-line carry state entering logical line `line`: the open fence's
    /// backtick count and the list nesting depth. Used to seed a mid-document
    /// window and to detect re-sync during an incremental update.
    pub fn entry_state(&self, line: usize) -> (Option<usize>, Option<usize>) {
        match self.recs.get(line) {
            Some(rec) => (
                (rec.fence_in != 0).then_some(rec.fence_in as usize),
                (rec.depth_in >= 0).then_some(rec.depth_in as usize),
            ),
            None => (None, None),
        }
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
pub(crate) fn image_button_rect(image: &Rect) -> Rect {
    Rect {
        x: image.x + image.w - IMAGE_BUTTON_W - IMAGE_BUTTON_MARGIN,
        y: image.y + IMAGE_BUTTON_MARGIN,
        w: IMAGE_BUTTON_W,
        h: IMAGE_BUTTON_H,
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
/// returning the effective style and its block-marker length. A list item nests
/// by leading indentation ([`markdown::LIST_INDENT_SPACES`] per level); it only
/// renders as a list item when its parent, one level shallower, is present above
/// it, otherwise it falls back to plain body text (marker length `0`). Blank
/// lines leave the nesting untouched; any other non-list line ends it.
fn resolve_list(line: &str, list_depth: &mut Option<usize>) -> (markdown::LineStyle, usize) {
    let (style, marker) = markdown::classify(line);
    match style {
        markdown::LineStyle::Bullet
        | markdown::LineStyle::Ordered
        | markdown::LineStyle::Task { .. } => {
            let depth = markdown::leading_indent(line) / markdown::LIST_INDENT_SPACES;
            let allowed = list_depth.map_or(0, |d| d + 1);
            if depth <= allowed {
                *list_depth = Some(depth);
                (style, marker)
            } else {
                // Over-nested: no parent one level up, so render as plain text.
                *list_depth = None;
                (markdown::LineStyle::Body, 0)
            }
        }
        other => {
            if !line.trim().is_empty() {
                *list_depth = None;
            }
            (other, marker)
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
fn code_block_spans_into(line: &str, delimiter: bool, out: &mut Vec<markdown::Span>) {
    out.clear();
    if line.is_empty() {
        return;
    }
    let kind = if delimiter {
        markdown::SpanKind::Marker
    } else {
        markdown::SpanKind::Text(markdown::Emphasis::Code)
    };
    out.push(markdown::Span {
        start: 0,
        end: line.len(),
        kind,
    });
}

/// Clip line-relative `spans` to the visual segment `[seg_start, seg_end)`,
/// offsetting to absolute by `ls`. Text runs always show. A marker shows only on
/// a revealed line, except the block marker (the leading `[0, block_marker)`
/// bytes) when `block_persists`: a list bullet or quote `>` stays on screen even
/// off the active line. Empty overlaps are dropped, so the runs tile the segment.
#[allow(clippy::too_many_arguments)]
fn clip_runs_into(
    spans: &[markdown::Span],
    seg_start: usize,
    seg_end: usize,
    ls: usize,
    revealed: bool,
    block_marker: usize,
    block_persists: bool,
    out: &mut Vec<Run>,
) {
    for span in spans {
        let s = span.start.max(seg_start);
        let e = span.end.min(seg_end);
        if s >= e {
            continue;
        }
        let is_marker = matches!(span.kind, markdown::SpanKind::Marker);
        // The block marker occupies the line's leading `[0, block_marker)` bytes;
        // inline markers (code, emphasis) start past it.
        let is_block_marker = is_marker && span.start < block_marker;
        let visible = if is_block_marker {
            revealed || block_persists
        } else {
            !is_marker || revealed
        };
        out.push(Run {
            start: ls + s,
            end: ls + e,
            kind: span.kind,
            visible,
        });
    }
}

/// Greedily word-wrap the logical `line` to `wrap_width` pixels, returning the
/// byte ranges of its visual segments. Breaks after the last space that fits; a
/// word wider than `wrap_width` breaks at a grapheme boundary so every segment
/// holds at least one cluster. `wrap_width <= 0` disables wrapping. Each cluster
/// is measured in its run's face (per `spans` and `size`), so bold/italic widths
/// wrap correctly; the source text is measured in full (markers included), so
/// hiding markers later never reflows the line.
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
    wrap_width: i32,
    block_indent: i32,
    heading: bool,
    clusters: &mut Vec<(usize, f32, bool)>,
    segments: &mut Vec<(usize, usize)>,
) {
    segments.clear();
    if wrap_width <= 0 || line.is_empty() {
        segments.push((0, line.len()));
        return;
    }
    // Measure each cluster once: its offset within the line, width in its run's
    // face, and whether it is a space (a wrap opportunity). The pooled buffer is
    // cleared and refilled, so it reuses its capacity across lines instead of
    // allocating one sized to the line on every call.
    clusters.clear();
    clusters.extend(grapheme::graphemes(line).map(|(off, cluster)| {
        let face = run_face(fonts, size, kind_at(spans, off), heading);
        (off, shape::text_advance(face, cluster), cluster == " ")
    }));
    if clusters.is_empty() {
        segments.push((0, line.len()));
        return;
    }

    let mut i = 0;
    while i < clusters.len() {
        // The first segment gets the full width; continuations are narrower by
        // the hanging indent, since they render inset by it.
        let limit = (wrap_width - if segments.is_empty() { 0 } else { block_indent }).max(1) as f32;
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
            segments.push((seg_start, line.len()));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown::LineStyle;

    const BODY: u32 = 32;
    const LEFT: i32 = 40;
    const TOP: i32 = 10;

    fn fonts() -> Fonts {
        Fonts::new(&markdown::all_sizes(BODY)).expect("a default font")
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
    ];

    /// Every fixture plus each seeded corpus (kept small for fast tests) as a doc
    /// to diff layout variants against.
    fn oracle_docs() -> Vec<Rope> {
        let mut docs: Vec<Rope> = FIXTURES.iter().map(|s| Rope::from_str(s)).collect();
        for name in crate::corpus::NAMES {
            if let Some(text) = crate::corpus::generate(name, 12 * 1024) {
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
    fn line_index_matches_the_full_build() {
        // Index oracle: the index's base heights reproduce the full build's
        // total height and every logical line's top, across corpora and widths.
        let f = fonts();
        let no_images = ImageStore::new(None);
        for r in oracle_docs() {
            for &w in ORACLE_WIDTHS {
                let full = Layout::build(&f, &r, LEFT, TOP, w, BODY, NO_REVEAL);
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images);
                assert_eq!(idx.len(), r.len_lines(), "one record per logical line");
                assert_eq!(
                    idx.content_height(&RevealAdjust::default()),
                    full.text_height(),
                    "content height (width {w})"
                );
                // The first visual row of each logical line sits at the index's top.
                let mut last = None;
                for v in full.lines() {
                    let li = r.byte_to_line(v.start);
                    if Some(li) != last {
                        let top = v.baseline - f.metrics(v.size).ascent - TOP;
                        assert_eq!(
                            idx.logical_top(li, &RevealAdjust::default()),
                            top,
                            "top of logical line {li} (width {w})"
                        );
                        last = Some(li);
                    }
                }
            }
        }
    }

    #[test]
    fn line_index_records_image_reveal_extra() {
        // A decoded standalone image is the one line whose height depends on reveal;
        // the index carries that delta as a per-line overlay.
        let f = fonts();
        let s = ImageStore::from_decoded(vec![(
            "cat.png".to_string(),
            crate::image::Image {
                pixels: vec![0u32; 80 * 40],
                width: 80,
                height: 40,
            },
        )]);
        let r = rope("intro line\n![cat](cat.png)\ntrailing line\n");
        let w = 560;
        let img_line = 1;
        let idx = LineIndex::build(&f, &r, w, BODY, &s);
        assert!(
            idx.reveal_extra(img_line) > 0,
            "revealing the source adds height"
        );
        assert_eq!(idx.reveal_extra(0), 0, "a text line has no reveal extra");
        assert_eq!(idx.reveal_extra(2), 0);

        // Base height (nothing revealed) matches a full no-reveal build.
        let base = Layout::build_with_images(
            &f,
            &r,
            LEFT,
            TOP,
            w,
            BODY,
            NO_REVEAL,
            &s,
            ImageUi::default(),
        );
        assert_eq!(
            idx.content_height(&RevealAdjust::default()),
            base.text_height()
        );

        // With the image's source revealed, the overlay height matches a full build
        // that reveals the same line.
        let ls = r.line_to_byte(img_line);
        let le = r.line_to_byte(img_line + 1) - 1;
        let revealed =
            Layout::build_with_images(&f, &r, LEFT, TOP, w, BODY, (ls, le), &s, ImageUi::default());
        let mut adj = RevealAdjust::default();
        adj.push(img_line, idx.reveal_extra(img_line));
        assert_eq!(idx.content_height(&adj), revealed.text_height());
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
                let idx = LineIndex::build(&f, &r, w, BODY, &no_images);
                let full_fp = fingerprint(&full);
                let bands: [(usize, usize); 5] = [
                    (0, n),
                    (0, (n / 2).max(1).min(n)),
                    (n / 3, (2 * n / 3).max(n / 3)),
                    (n.saturating_sub(4), n),
                    (n / 2, n / 2), // empty band
                ];
                for (bs, be) in bands {
                    let seed = TOP + idx.logical_top(bs, &RevealAdjust::default());
                    let win = Layout::build_window(
                        &f,
                        &r,
                        LEFT,
                        seed,
                        w,
                        BODY,
                        NO_REVEAL,
                        &no_images,
                        ImageUi::default(),
                        &idx,
                        bs,
                        be,
                    );
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
        let mut index = LineIndex::build(&f, &rope, w, BODY, &no_images);

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
            index.update(&f, &rope, &prev, w, BODY, &no_images);
            let rebuilt = LineIndex::build(&f, &rope, w, BODY, &no_images);
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
                let seq = LineIndex::build(&f, &r, w, BODY, &no_images);
                for threads in [2, 3, 7, 32] {
                    let par = LineIndex::build_chunked(&f, &r, w, BODY, &no_images, threads);
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
        assert_eq!(h, m.ascent + m.descent, "caret spans ascent + descent");
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
        let off = l.offset_at(&f, &r, -100.0, 9_000.0);
        assert_eq!(off, 4, "start of the last line");
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
    fn reveal_state_does_not_change_vertical_layout() {
        // Wrapping uses the source text, so collapsing markers never reflows the
        // document: line count, baselines, and height are reveal-independent.
        let f = fonts();
        let r = rope("# A longer heading that might wrap\nbody text below the heading");
        let hidden = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (99, 99));
        let shown = Layout::build(&f, &r, LEFT, TOP, 200, BODY, (0, 0));
        assert_eq!(hidden.lines().len(), shown.lines().len());
        for (h, s) in hidden.lines().iter().zip(shown.lines()) {
            assert_eq!(h.baseline, s.baseline, "baselines are reveal-independent");
        }
        assert_eq!(hidden.text_height(), shown.text_height());
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
        // Offset 1 is "b", just past the opening backtick. Collapsed it sits at
        // the margin; shown it sits one backtick-width to the right.
        let (x_hidden, _, _) = hidden.caret(&f, &r, 1);
        let (x_shown, _, _) = shown.caret(&f, &r, 1);
        assert_eq!(x_hidden, LEFT, "collapsed: code text at the margin");
        let tick_w = shape::text_advance(f.face(BODY, FontStyle::Regular), "`").round() as i32;
        assert_eq!(
            x_shown - x_hidden,
            tick_w,
            "shown: shifted by the backtick width"
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
        let l = layout(&f, &r);
        let child = &l.lines()[1];
        assert_eq!(
            child.gutter,
            Gutter::Bullet,
            "a nested item with a parent renders as a bullet"
        );
        assert_eq!(child.indent, LIST_INDENT, "and carries the list indent");
        let marker = l.run_slice(child).first().expect("a marker run");
        assert!(matches!(marker.kind, markdown::SpanKind::Marker));
        assert_eq!(
            marker.end - marker.start,
            4,
            "the marker spans the indentation and the dash (`  - `)"
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
        assert_eq!(third.indent, LIST_INDENT, "and carries the list indent");
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
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::Bullet,
            "and the dot stays while the cursor edits the line"
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
    fn a_task_line_draws_a_checkbox_off_line_and_raw_marker_on_it() {
        let f = fonts();
        let src = "- [ ] todo\nx";
        let r = rope(src);
        let off = src.find('x').unwrap();
        let away = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (off, off));
        assert_eq!(
            away.lines()[0].gutter,
            Gutter::Checkbox { checked: false },
            "the checkbox stands in once the cursor leaves"
        );
        assert!(
            away.run_slice(&away.lines()[0])[0].visible,
            "the marker run persists (width-matched under the drawn box)"
        );
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(
            on.lines()[0].gutter,
            Gutter::None,
            "the raw `- [ ]` shows while editing"
        );
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
            "the continuation still dims and strikes"
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
    fn checkbox_rect_sits_over_the_brackets_within_the_marker() {
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
        let pre = shape::text_advance(face, "  - ");
        let marker = shape::text_advance(face, "  - [ ] ");
        assert!(
            rect.x as f32 >= LEFT as f32 + line.indent as f32 + pre - 1.0,
            "the box starts at or after the brackets (past the indent and dash)"
        );
        assert!(
            (rect.x + rect.w) as f32 <= LEFT as f32 + line.indent as f32 + marker + 1.0,
            "and ends within the marker's advance (width-matched substitution)"
        );
        assert!(rect.w >= 8 && rect.h == rect.w, "a sensible square");
        let m = f.metrics(BODY);
        assert!(
            rect.y >= line.baseline - m.ascent && rect.y + rect.h <= line.baseline + m.descent,
            "vertically inside the line's box"
        );
        // With the cursor on the task line the raw marker shows: no drawn box.
        let task_start = 4;
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (task_start, task_start));
        assert!(on.checkbox_rect(&f, &r, &on.lines()[1]).is_none());
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
        // On the revealed line the raw marker shows and clicks edit, not toggle.
        let on = Layout::build(&f, &r, LEFT, TOP, 1_000_000, BODY, (0, 0));
        assert_eq!(on.checkbox_at(&f, &r, cx, cy), None);
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
        // The ``` fences read as markers; the code line is a code run.
        assert_eq!(
            first_kinds(&l),
            vec![
                Some(markdown::SpanKind::Marker),                         // ```rust
                Some(markdown::SpanKind::Text(markdown::Emphasis::Code)), // fn x() {}
                Some(markdown::SpanKind::Marker),                         // ```
                Some(markdown::SpanKind::Text(markdown::Emphasis::Regular)), // body
            ]
        );
    }

    #[test]
    fn markdown_inside_a_fence_is_literal() {
        // A '#' line that would be an H1 is plain code inside a fence: body size,
        // a single code run, no heading marker.
        let f = fonts();
        let r = rope("```\n# not a heading\n```");
        let l = layout(&f, &r);
        let inner = &l.lines()[1];
        assert!(inner.code_block);
        assert_eq!(inner.size, BODY, "no heading scaling inside a code block");
        assert_eq!(l.run_slice(inner).len(), 1);
        assert_eq!(
            l.run_slice(inner)[0].kind,
            markdown::SpanKind::Text(markdown::Emphasis::Code),
            "the '#' is code, not a heading marker"
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
    fn decoded(w: u32, h: u32) -> crate::image::Image {
        crate::image::Image {
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
        Layout::build_with_images(f, r, LEFT, TOP, 1_000_000, BODY, active, s, ui)
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
    fn the_button_pin_reveals_source_without_the_cursor() {
        let f = fonts();
        let r = rope("![cat](cat.png)\nafter");
        let s = store("cat.png", 80, 40);
        // Cursor on "after" (off the image), but the corner button pinned raw open.
        let ui = ImageUi {
            selected: Some(0),
            raw: Some(0),
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
            },
        );
        let rect = sel
            .image_rect(&f, &sel.lines()[0])
            .expect("the picture rect");
        // A point inside the picture selects that image.
        let inside = ((rect.x + 5) as f32, (rect.y + 5) as f32);
        assert_eq!(sel.image_at(&f, inside.0, inside.1), Some(0));
        // The corner button is hit only when the image is selected.
        let btn = image_button_rect(&rect);
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
}
