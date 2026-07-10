use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::file::{self, FileFormat};
use crate::find::FindState;
use crate::layout::{ImageUi, Layout, LayoutCtx, LineIndex, RevealAdjust};
use crate::markdown;
use crate::platform::error::{Error, Result};
use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::platform::protocol::wp_cursor_shape_device_v1;
use crate::platform::scroll;
use crate::render::image::ImageStore;
use crate::rope::Rope;
use crate::scale::Scale;
use crate::store;
use crate::text::TextBuffer;
use crate::theme::{FONT_SIZE, MARGIN};
use crate::tree;

/// Largest file the editor will open, as a guard against mapping a huge file into
/// memory on startup. Generous for a text editor; a bigger file is a clean error,
/// not an out-of-memory crash.
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// While drag-selecting, a pointer this close to the top or bottom edge (in
/// pixels) auto-scrolls the view so the selection can extend past what is shown.
const AUTOSCROLL_MARGIN: i32 = 24;
/// Pixels scrolled per auto-scroll tick while the pointer is held in the edge zone.
const AUTOSCROLL_STEP: i32 = 24;
/// How often an auto-scroll tick fires while the pointer is held in the edge zone.
const AUTOSCROLL_INTERVAL: Duration = Duration::from_millis(16);

/// Two clicks within this window, and within [`MULTI_CLICK_RADIUS`] pixels,
/// count as a double-click (then triple, ...).
const MULTI_CLICK_INTERVAL: Duration = Duration::from_millis(400);
/// How far apart successive clicks may land yet still chain into a multi-click.
const MULTI_CLICK_RADIUS: f32 = 5.0;

/// Multi-click chaining. Successive clicks landing close in time (within
/// [`MULTI_CLICK_INTERVAL`]) and space (within [`MULTI_CLICK_RADIUS`]) escalate a
/// single click to a double, then a triple, and so on; a click outside either
/// window starts a fresh chain at one.
pub(crate) struct ClickChain {
    /// When the last click landed, or `None` before the first click.
    last: Option<Instant>,
    /// Where the last click landed, for the radius test.
    last_pos: (f32, f32),
    /// The current run length: 1 = single, 2 = double, 3 = triple, ...
    count: u32,
}

impl ClickChain {
    fn new() -> Self {
        Self {
            last: None,
            last_pos: (0.0, 0.0),
            count: 0,
        }
    }

    /// Record a click at window position `(px, py)` occurring at `now`. Returns
    /// the resulting click count (1 = single, 2 = double, ...) and whether it
    /// chained onto the previous click.
    pub(crate) fn register(&mut self, px: f32, py: f32, now: Instant) -> (u32, bool) {
        let (lx, ly) = self.last_pos;
        let near = (px - lx).abs() <= MULTI_CLICK_RADIUS && (py - ly).abs() <= MULTI_CLICK_RADIUS;
        let chained = near
            && self
                .last
                .is_some_and(|t| now.saturating_duration_since(t) <= MULTI_CLICK_INTERVAL);
        self.count = if chained { self.count + 1 } else { 1 };
        self.last = Some(now);
        self.last_pos = (px, py);
        (self.count, chained)
    }
}

/// Read `path` into buffer text and its on-disk [`FileFormat`]. A path that does
/// not exist yet is not an error: it opens an empty buffer that a later save
/// creates, so `bnknotes new-note.md` works.
fn load(path: &Path) -> Result<(String, FileFormat)> {
    // Reject an oversized file up front (by its stat size) rather than reading it
    // all into memory. A missing file is not an error here, so a stat failure
    // falls through to the read, which maps NotFound to an empty buffer.
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > MAX_FILE_BYTES {
            return Err(Error::msg(format!(
                "file is {} bytes; refusing to open files larger than {} MiB",
                meta.len(),
                MAX_FILE_BYTES / (1024 * 1024)
            )));
        }
    }
    match std::fs::read(path) {
        Ok(bytes) => file::decode(&bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok((String::new(), FileFormat::default()))
        }
        Err(e) => Err(e.into()),
    }
}

/// The inputs the laid-out geometry depends on, used to memoize it: content
/// version, surface width, body font size, the display scale (as its 120ths
/// factor, so a DPI change reflows even at an unchanged column width and body
/// size), the active range's logical line span (the marker reveal depends only on
/// which lines the range touches, not the exact byte offsets), and the selected /
/// raw-pinned image (by line-start offset). A scroll, idle redraw, or cursor move
/// within one line leaves all of these unchanged and reuses the cached layout.
type LayoutKey = (
    u64,
    i32,
    i32,
    u32,
    u32,
    usize,
    usize,
    Option<usize>,
    Option<usize>,
);

/// The inputs the line-metrics index depends on: content version, column x and
/// width, body font size, and the display scale (its 120ths factor). Reveal and
/// scroll leave these unchanged, so a cursor move or a scroll reuses the index and
/// only the visible window rebuilds.
type IndexKey = (u64, i32, i32, u32, u32);

/// Cap on worker threads for a parallel index build. Bounds the transient memory
/// of the per-worker font sets (each worker opens its own FreeType faces).
const INDEX_MAX_THREADS: usize = 16;
/// Lines of work to justify one extra worker's font open. Below one worker's
/// worth the build stays sequential; a huge document scales up to the cap.
const LINES_PER_WORKER: usize = 50_000;

/// Which region holds keyboard focus. Editing keys go to the focused editor;
/// the file tree takes navigation keys when it is focused.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focus {
    Tree,
    /// Content pane 0 (left/A) or 1 (right/B).
    Editor(usize),
}

/// Width of the grab bar between two columns: a few pixels of hit zone, with a
/// thin divider line drawn down its middle.
pub(crate) const SPLITTER: i32 = 6;

/// The adjustable column layout. The sidebar has a logical pixel width; the two
/// content panes split the remaining width by a ratio. Splitters drag these
/// values. `tree_width` is logical (the persisted, scale-free unit); [`Self::columns`]
/// scales it to device pixels.
pub(crate) struct PaneLayout {
    pub(crate) tree_width: i32,
    /// Pane A's share of the content area (right of the tree), clamped 0.1..0.9.
    pub(crate) split_ratio: f32,
    pub(crate) tree_visible: bool,
    pub(crate) second_open: bool,
}

/// The pixel rectangles the columns and their splitter bars occupy, derived
/// from the window size and the [`PaneLayout`]. A hidden column is `None`.
pub(crate) struct Columns {
    pub(crate) tree: Option<Rect>,
    /// Editor A is always present; editor B only when `second_open`.
    pub(crate) editors: [Option<Rect>; 2],
    /// Grab bars: `[0]` between tree and A, `[1]` between A and B.
    pub(crate) splitters: [Option<Rect>; 2],
}

impl PaneLayout {
    /// Lay the columns out across the device `width`×`height` at display `scale`.
    /// The logical sidebar width and splitter thickness are scaled to device pixels
    /// here, so the persisted layout stays scale-free. Editor A always takes what is
    /// left after an optional sidebar and an optional second pane; each column is
    /// full height, so content stays in its column by construction (no clip).
    pub(crate) fn columns(&self, width: i32, height: i32, scale: Scale) -> Columns {
        let splitter = scale.px(SPLITTER);
        let tree_width = scale.px(self.tree_width);
        let mut x = 0;
        let mut tree = None;
        let mut splitters = [None, None];
        if self.tree_visible {
            tree = Some(Rect {
                x,
                y: 0,
                w: tree_width,
                h: height,
            });
            x += tree_width;
            splitters[0] = Some(Rect {
                x,
                y: 0,
                w: splitter,
                h: height,
            });
            x += splitter;
        }
        let mut editors = [None, None];
        if self.second_open {
            let content = (width - x - splitter).max(0);
            let a_w = ((content as f32 * self.split_ratio) as i32).clamp(0, content);
            editors[0] = Some(Rect {
                x,
                y: 0,
                w: a_w,
                h: height,
            });
            x += a_w;
            splitters[1] = Some(Rect {
                x,
                y: 0,
                w: splitter,
                h: height,
            });
            x += splitter;
            editors[1] = Some(Rect {
                x,
                y: 0,
                w: (width - x).max(0),
                h: height,
            });
        } else {
            editors[0] = Some(Rect {
                x,
                y: 0,
                w: (width - x).max(0),
                h: height,
            });
        }
        Columns {
            tree,
            editors,
            splitters,
        }
    }
}

/// What a window position lands on, for routing pointer events to the right
/// region. Splitters are checked first (they sit in the gaps between columns).
pub(crate) enum Hit {
    Splitter(usize),
    Tree(Rect),
    Editor(usize, Rect),
    None,
}

/// One content pane: an independent editor over its own buffer, carrying its own
/// file, inline images, memoized layout, scroll, find bar, and the drag/selection
/// transients a click-drag needs. Two of these live on [`State`].
pub(crate) struct EditorPane {
    pub(crate) text: TextBuffer,
    pub(crate) path: Option<PathBuf>,
    pub(crate) format: FileFormat,
    pub(crate) images: ImageStore,
    pub(crate) selected_image: Option<usize>,
    pub(crate) raw_image: Option<usize>,
    pub(crate) find: FindState,
    pub(crate) scroll_y: i32,
    pub(crate) content_height: i32,
    pub(crate) line_index: Option<LineIndex>,
    pub(crate) index_key: Option<IndexKey>,
    /// The rope the current index was built from, so an edit can be re-measured
    /// incrementally by diffing against it. An `Rc` clone, so keeping it is free.
    pub(crate) prev_rope: Option<Rope>,
    pub(crate) layout_cache: Option<Layout>,
    pub(crate) layout_key: Option<LayoutKey>,
    /// The logical-line range `[start, end)` the cached window covers, so a caller
    /// can tell whether an offset (the cursor, a click) falls inside it.
    pub(crate) layout_band: (usize, usize),
    pub(crate) goal_x: Option<f32>,
    pub(crate) selecting: bool,
    pub(crate) frozen_reveal: Option<(usize, usize)>,
    pub(crate) clicks: ClickChain,
    /// Drag auto-scroll step: pixels per tick (negative up, positive down), 0 when
    /// disarmed. Armed while a selection drag holds the pointer in a column's edge
    /// zone; see [`Self::update_autoscroll`] and [`Self::autoscroll_due`].
    pub(crate) autoscroll: i32,
    /// When the next auto-scroll tick is due; `None` while disarmed.
    pub(crate) autoscroll_at: Option<Instant>,
    /// Pooled inline-markup scratch for [`Self::link_under_pointer`], so hit-testing
    /// a hovered line's links reuses its buffers instead of reallocating per event.
    link_scratch: markdown::InlineScratch,
    /// The display scale and the device-pixel body font size this pane lays out at,
    /// set by [`State`] and refreshed on a DPI change. At [`Scale::ONE`] the body is
    /// [`FONT_SIZE`] and every geometry constant is its authored value.
    scale: Scale,
    body_size: u32,
}

impl EditorPane {
    /// An empty, file-less pane (blank buffer, no images).
    pub(crate) fn empty() -> Self {
        Self {
            text: TextBuffer::new(),
            path: None,
            format: FileFormat::default(),
            images: ImageStore::new(None),
            selected_image: None,
            raw_image: None,
            find: FindState::new(),
            scroll_y: 0,
            content_height: 0,
            line_index: None,
            index_key: None,
            prev_rope: None,
            layout_cache: None,
            layout_key: None,
            layout_band: (0, 0),
            goal_x: None,
            selecting: false,
            frozen_reveal: None,
            clicks: ClickChain::new(),
            autoscroll: 0,
            autoscroll_at: None,
            link_scratch: markdown::InlineScratch::default(),
            scale: Scale::ONE,
            body_size: FONT_SIZE,
        }
    }

    /// Adopt a new display scale and device-pixel body size (a DPI change), so the
    /// next layout re-measures at the new size. The memoization keys carry both, so
    /// simply setting them invalidates the cached index and window.
    pub(crate) fn set_scale(&mut self, scale: Scale, body_size: u32) {
        self.scale = scale;
        self.body_size = body_size;
    }

    /// Load `path` into a fresh pane, resolving inline images against its
    /// directory. An unreadable file is an error the caller surfaces.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let (contents, format) = load(path)?;
        let image_dir = path.parent().map(Path::to_path_buf);
        Ok(Self {
            text: TextBuffer::from_text(contents),
            path: Some(path.to_path_buf()),
            format,
            images: ImageStore::new(image_dir),
            ..Self::empty()
        })
    }

    /// The active range for live-preview marker reveal: the live selection (or
    /// the bare cursor), pinned to `frozen_reveal` while a drag extends a
    /// selection so the markdown under the pointer does not flip rendered/raw.
    pub(crate) fn reveal_range(&self) -> (usize, usize) {
        let cursor = self.text.cursor();
        match (self.selecting, self.frozen_reveal) {
            (true, Some(range)) => range,
            _ => self.text.selection().unwrap_or((cursor, cursor)),
        }
    }

    /// The per-image UI state (selected border, pinned raw source) for the layout.
    pub(crate) fn image_ui(&self) -> ImageUi {
        ImageUi {
            selected: self.selected_image,
            raw: self.raw_image,
        }
    }

    /// Decode every standalone `![](path)` line so [`Self::ensure_layout`] can
    /// size the image lines. A path already attempted is a cache hit.
    pub(crate) fn scan_images(&mut self) {
        let rope = self.text.rope();
        for (ls, le, _) in rope.lines() {
            let line = rope.slice(ls..le).as_cow();
            if let Some(link) = markdown::standalone_image(&line) {
                let raw = line[link.path.0..link.path.1].to_string();
                self.images.ensure(&raw);
            }
        }
    }

    /// The ≤2-line overlay of image sources revealed right now (the reveal range's
    /// ends and the raw-pinned line), so the index reports heights and positions
    /// under the current reveal. Empty (and free) for the common no-image case.
    fn reveal_adjust(&self) -> RevealAdjust {
        let mut adjust = RevealAdjust::default();
        let Some(index) = self.line_index.as_ref() else {
            return adjust;
        };
        let rope = self.text.rope();
        let (lo, hi) = self.reveal_range();
        let l0 = rope.byte_to_line(lo);
        adjust.push(l0, index.reveal_extra(l0));
        let l1 = rope.byte_to_line(hi);
        if l1 != l0 {
            adjust.push(l1, index.reveal_extra(l1));
        }
        if let Some(raw) = self.raw_image {
            let lr = rope.byte_to_line(raw);
            adjust.push(lr, index.reveal_extra(lr));
        }
        adjust
    }

    /// Rebuild the line-metrics index when the content, column, or font changes
    /// (not on a scroll or cursor move), then refresh `content_height` from it plus
    /// any revealed image sources. The index is where the O(document) measuring
    /// lives now, paid once per edit or resize rather than per keystroke.
    pub(crate) fn ensure_index(&mut self, fonts: &Fonts, rect: Rect) {
        let margin = self.scale.px(MARGIN);
        let key = (
            self.text.version(),
            rect.x,
            rect.w,
            self.body_size,
            self.scale.factor_120(),
        );
        if self.index_key != Some(key) {
            let wrap_width = rect.w - 2 * margin;
            self.scan_images();
            // When only the content changed (same column, font, and scale), re-measure
            // just the edited region against the previous rope; otherwise (first build,
            // or a resize) rebuild in full.
            let geometry_same = matches!(
                self.index_key,
                Some((_, x, w, fsize, f120))
                    if x == rect.x && w == rect.w && fsize == self.body_size
                        && f120 == self.scale.factor_120()
            );
            let incremental =
                geometry_same && self.line_index.is_some() && self.prev_rope.is_some();
            if incremental {
                let mut index = self.line_index.take().expect("checked");
                let prev = self.prev_rope.take().expect("checked");
                let ctx = LayoutCtx::for_index(
                    fonts,
                    self.text.rope(),
                    wrap_width,
                    self.body_size,
                    &self.images,
                    self.scale,
                );
                index.update(&ctx, &prev);
                self.line_index = Some(index);
            } else {
                // A first build or a resize: no prior index to re-measure against,
                // so build in full, in parallel for a large document (a huge file
                // open or a resize on a long one). Worker count scales with the
                // work and caps out, so a modest document does not pay to open a
                // dozen font sets.
                let cores = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1);
                let threads = (self.text.rope().len_lines() / LINES_PER_WORKER)
                    .clamp(1, INDEX_MAX_THREADS.min(cores));
                self.line_index = Some(LineIndex::build_parallel(
                    fonts,
                    self.text.rope(),
                    wrap_width,
                    self.body_size,
                    &self.images,
                    self.scale,
                    threads,
                ));
            }
            self.prev_rope = Some(self.text.rope().clone());
            self.index_key = Some(key);
        }
        let adjust = self.reveal_adjust();
        if let Some(index) = self.line_index.as_ref() {
            self.content_height = index.content_height(&adjust) + 2 * margin;
        }
    }

    /// The shared layout context for a windowed build: this pane's rope and image
    /// store at `wrap_width`, with the reveal `active` range and current image UI.
    /// Both windowed builders go through this so their inputs cannot drift apart.
    fn layout_ctx<'a>(
        &'a self,
        fonts: &'a Fonts,
        active: (usize, usize),
        wrap_width: i32,
    ) -> LayoutCtx<'a> {
        LayoutCtx::new(
            fonts,
            self.text.rope(),
            wrap_width,
            self.body_size,
            active,
            &self.images,
            self.image_ui(),
            self.scale,
        )
    }

    /// Rebuild the windowed layout when the viewport leaves the built band or the
    /// reveal changes. The window covers the viewport plus a page of overscan each
    /// side (so all incremental motion and the caret sit inside it) with absolute
    /// positions seeded from the index, so lines above the window are never laid
    /// out. A scroll or cursor move that stays within the band reuses the cache.
    pub(crate) fn ensure_window(&mut self, fonts: &Fonts, rect: Rect) {
        let active = self.reveal_range();
        let active_lines = {
            let rope = self.text.rope();
            (rope.byte_to_line(active.0), rope.byte_to_line(active.1))
        };
        let key = (
            self.text.version(),
            rect.x,
            rect.w,
            self.body_size,
            self.scale.factor_120(),
            active_lines.0,
            active_lines.1,
            self.selected_image,
            self.raw_image,
        );
        let adjust = self.reveal_adjust();
        let (vis_top, vis_bot, over, n) = {
            let Some(index) = self.line_index.as_ref() else {
                return;
            };
            (
                index.line_at_y(self.scroll_y, &adjust),
                index.line_at_y(self.scroll_y + rect.h.max(0), &adjust),
                self.page_lines(fonts, rect),
                index.len(),
            )
        };
        let covered = self.layout_key == Some(key)
            && self.layout_cache.is_some()
            && self.layout_band.0 <= vis_top
            && self.layout_band.1 > vis_bot;
        if covered {
            return;
        }
        let start = vis_top.saturating_sub(over);
        let end = (vis_bot + over + 1).min(n).max(start);
        let margin = self.scale.px(MARGIN);
        let wrap_width = rect.w - 2 * margin;
        let layout = {
            let ctx = self.layout_ctx(fonts, active, wrap_width);
            let index = self.line_index.as_ref().expect("index built above");
            let seed = rect.y + margin + index.logical_top(start, &adjust);
            Layout::build_window(&ctx, rect.x + margin, seed, index, start, end)
        };
        self.layout_cache = Some(layout);
        self.layout_band = (start, end);
        self.layout_key = Some(key);
    }

    /// Ensure both the index and the visible window are current for `rect`. Callers
    /// that paint or hit-test the viewport use this; cursor-only operations use
    /// [`Self::ensure_index`] with [`Self::cursor_window`] instead.
    pub(crate) fn ensure_layout(&mut self, fonts: &Fonts, rect: Rect) {
        self.ensure_index(fonts, rect);
        self.ensure_window(fonts, rect);
    }

    /// Lay out a window centered on the cursor (a page each side plus slack for
    /// page-motion targets landing at the edges), with absolute document positions
    /// from the index. Cursor operations run the ordinary [`Layout`] methods on
    /// this, so their logic is shared with the full build; it is built on demand and
    /// is O(viewport), never O(document). Requires the index (call `ensure_index`).
    fn cursor_window(&self, fonts: &Fonts, rect: Rect) -> Option<Layout> {
        let index = self.line_index.as_ref()?;
        let adjust = self.reveal_adjust();
        let active = self.reveal_range();
        let cursor_line = self.text.rope().byte_to_line(self.text.cursor());
        let span = self.page_lines(fonts, rect) + 4;
        let n = index.len();
        let start = cursor_line.saturating_sub(span);
        let end = (cursor_line + span + 1).min(n).max(start);
        let margin = self.scale.px(MARGIN);
        let seed = rect.y + margin + index.logical_top(start, &adjust);
        let wrap_width = rect.w - 2 * margin;
        let ctx = self.layout_ctx(fonts, active, wrap_width);
        Some(Layout::build_window(
            &ctx,
            rect.x + margin,
            seed,
            index,
            start,
            end,
        ))
    }

    /// The cursor byte offset paired with a window laid out around it, after
    /// refreshing the index. `None` only when the index is unavailable. The shared
    /// preamble of the keyboard cursor-motions.
    fn motion_window(&mut self, fonts: &Fonts, rect: Rect) -> Option<(Layout, usize)> {
        self.ensure_index(fonts, rect);
        let cursor = self.text.cursor();
        self.cursor_window(fonts, rect).map(|win| (win, cursor))
    }

    /// Scroll by `delta` px within the column `rect`, clamped to the content.
    /// Returns whether the offset changed (so the caller can mark a repaint).
    pub(crate) fn scroll_by(&mut self, fonts: &Fonts, rect: Rect, delta: i32) -> bool {
        // Only the content height (from the index) is needed to clamp; the visible
        // window rebuilds on the next frame at the new offset.
        self.ensure_index(fonts, rect);
        let sy = scroll::clamp(self.scroll_y + delta, self.content_height, rect.h);
        let moved = sy != self.scroll_y;
        self.scroll_y = sy;
        moved
    }

    /// Scroll the minimum to keep the caret visible after a cursor move; returns
    /// whether the offset changed. The caret's band comes from a window built
    /// around the cursor, so this works even when the cursor sits far outside the
    /// currently painted window.
    pub(crate) fn scroll_caret_into_view(&mut self, fonts: &Fonts, rect: Rect) -> bool {
        self.ensure_index(fonts, rect);
        let cursor = self.text.cursor();
        let Some(win) = self.cursor_window(fonts, rect) else {
            return false;
        };
        let (_, caret_top, caret_h) = win.caret(fonts, self.text.rope(), cursor);
        let sy = scroll::reveal(
            self.scroll_y,
            caret_top,
            caret_h,
            rect.h,
            self.content_height,
            self.scale.px(MARGIN),
        );
        let moved = sy != self.scroll_y;
        self.scroll_y = sy;
        moved
    }

    /// Move the cursor `dir` lines (-1 up, +1 down), keeping a sticky goal x.
    pub(crate) fn move_vertical(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        dir: i32,
        extend: bool,
    ) -> bool {
        let Some((win, cursor)) = self.motion_window(fonts, rect) else {
            return false;
        };
        let (caret_x, _, _) = win.caret(fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        let target = win.offset_vertical(fonts, self.text.rope(), cursor, dir, goal);
        self.goal_x = Some(goal);
        self.text.set_cursor_changed(target, extend)
    }

    /// Move the cursor to the start (`to_end = false`) or end of its visual line.
    pub(crate) fn move_line_edge(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        to_end: bool,
        extend: bool,
    ) -> bool {
        let Some((win, cursor)) = self.motion_window(fonts, rect) else {
            return false;
        };
        let (start, end) = win.line_bounds(cursor);
        let target = if to_end {
            end
        } else {
            // Smart Home. Visual bounds, so it works on wrapped rows too.
            let line = self.text.rope().slice(start..end).as_cow();
            smart_home_target(&line, start, end, cursor)
        };
        self.text.set_cursor_changed(target, extend)
    }

    /// Move the cursor one column-height up/down, keeping the sticky goal column.
    pub(crate) fn move_page(&mut self, fonts: &Fonts, rect: Rect, dir: i32, extend: bool) -> bool {
        let page_lines = self.page_lines(fonts, rect);
        let Some((win, cursor)) = self.motion_window(fonts, rect) else {
            return false;
        };
        let (caret_x, _, _) = win.caret(fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        // Step a line at a time so each hop honors the goal column and the
        // varying per-line heights; stop early at the first or last line. The
        // cursor window spans a page plus slack, so a full page of steps stays
        // inside it.
        let mut target = cursor;
        for _ in 0..page_lines {
            let next = win.offset_vertical(fonts, self.text.rope(), target, dir, goal);
            if next == target {
                break;
            }
            target = next;
        }
        self.goal_x = Some(goal);
        self.text.set_cursor_changed(target, extend)
    }

    /// How many visual lines make up a Page Up/Down jump in this column: its
    /// height in body line-heights, less one row of overlap, at least one.
    pub(crate) fn page_lines(&self, fonts: &Fonts, rect: Rect) -> usize {
        let line_height = fonts.metrics(self.body_size).line_height.max(1);
        (rect.h / line_height - 1).max(1) as usize
    }

    /// Content-space top for each current find match, from the index, so the tick
    /// strip places a match outside the painted window at its true position. Empty
    /// when find is closed or the index is not built. Call after `ensure_index`.
    pub(crate) fn find_tick_tops(&self) -> Vec<i32> {
        let Some(index) = self.line_index.as_ref() else {
            return Vec::new();
        };
        if !self.find.is_open() {
            return Vec::new();
        }
        let rope = self.text.rope();
        let adjust = self.reveal_adjust();
        let margin = self.scale.px(MARGIN);
        self.find
            .matches()
            .iter()
            .map(|&(start, _)| margin + index.logical_top(rope.byte_to_line(start), &adjust))
            .collect()
    }

    /// Whether the cursor's logical line falls inside the painted window, so a caret
    /// drawn from that window sits at the true position rather than a clamped
    /// fallback. Off-window (scrolled far away) means the caret is simply not drawn.
    pub(crate) fn cursor_in_window(&self) -> bool {
        let line = self.text.rope().byte_to_line(self.text.cursor());
        self.layout_band.0 <= line && line < self.layout_band.1
    }

    /// Content-space y for a window y inside this pane's column (the column top
    /// and this pane's scroll offset removed).
    pub(crate) fn content_y(&self, rect: Rect, py: f32) -> f32 {
        py - rect.y as f32 + self.scroll_y as f32
    }

    /// Ensure the windowed layout covers `rect`, then return it (with the buffer
    /// rope, since the hit-tests need both) and the content-space y for the window
    /// position `py`. `None` when no layout could be built. The shared preamble of
    /// the pointer-query methods below; it hands back the rope so a caller need
    /// not re-borrow `self` (which the returned layout reference locks) to reach
    /// it.
    fn layout_at(&mut self, fonts: &Fonts, rect: Rect, py: f32) -> Option<(&Layout, &Rope, f32)> {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        let layout = self.layout_cache.as_ref()?;
        Some((layout, self.text.rope(), y))
    }

    /// The byte offset under a window position within this pane's column.
    pub(crate) fn pointer_offset(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> usize {
        match self.layout_at(fonts, rect, py) {
            Some((layout, rope, y)) => layout.offset_at(fonts, rope, px, y),
            None => 0,
        }
    }

    /// Place the cursor (or, with `extend`, the selection's free end) at a window
    /// position within this pane's column. Returns whether it moved.
    pub(crate) fn place_cursor_at(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
        extend: bool,
    ) -> bool {
        let Some((layout, rope, y)) = self.layout_at(fonts, rect, py) else {
            return false;
        };
        let offset = layout.offset_at(fonts, rope, px, y);
        self.text.set_cursor_changed(offset, extend)
    }

    /// The destination of the openable link under a window position, if its line
    /// is rendered (markup collapsed) and the scheme is one the browser opens.
    pub(crate) fn link_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<String> {
        let off = self.pointer_offset(fonts, rect, px, py);
        let (a0, a1) = self.reveal_range();
        let rope = self.text.rope();
        let line = rope.byte_to_line(off);
        let (ls, le) = rope.line_content_range(line);
        if a0 <= le && a1 >= ls {
            return None;
        }
        let content = rope.slice(ls..le).as_cow();
        let rel = off - ls;
        let link = markdown::links(&content, &mut self.link_scratch)
            .into_iter()
            .find(|l| l.text.0 <= rel && rel <= l.text.1)?;
        let dest = content[link.dest.0..link.dest.1].to_string();
        crate::platform::browser::can_open(&dest).then_some(dest)
    }

    /// The logical-line start of the rendered task checkbox under a window
    /// position, if any.
    pub(crate) fn checkbox_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<usize> {
        let (layout, rope, y) = self.layout_at(fonts, rect, py)?;
        layout.checkbox_at(fonts, rope, px, y)
    }

    /// Handle a press on an image button (toggles its raw source) or picture
    /// (selects it). Returns whether it was consumed.
    pub(crate) fn image_press(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> bool {
        let Some((layout, _, y)) = self.layout_at(fonts, rect, py) else {
            return false;
        };
        let button = layout.image_button_at(fonts, px, y);
        let image = layout.image_at(fonts, px, y);
        if let Some(ls) = button {
            self.raw_image = if self.raw_image == Some(ls) {
                None
            } else {
                Some(ls)
            };
            return true;
        }
        if let Some(ls) = image {
            self.selected_image = Some(ls);
            self.raw_image = None;
            let cursor = self.text.cursor();
            self.text.set_cursor(cursor, false);
            self.selecting = false;
            self.frozen_reveal = None;
            return true;
        }
        false
    }

    /// Handle a press on a table's add-row / add-column hover strip: grow the table
    /// by a row or a column. Returns whether it was consumed.
    pub(crate) fn press_table_button(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> bool {
        let button = {
            let Some((layout, rope, y)) = self.layout_at(fonts, rect, py) else {
                return false;
            };
            layout.table_button_at(fonts, rope, px, y)
        };
        let Some(button) = button else {
            return false;
        };
        match button.kind {
            crate::layout::TableButtonKind::AddRow => self.text.add_table_row(button.block),
            crate::layout::TableButtonKind::AddColumn => self.text.add_table_column(button.block),
        }
    }

    /// The pointer shape for a window position within this pane: a hand over a
    /// link/checkbox/image-button or a table add-row/add-column strip, the arrow
    /// over a picture, the I-beam elsewhere.
    pub(crate) fn cursor_shape_at(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> u32 {
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_POINTER, SHAPE_TEXT};
        let over_link = self.link_under_pointer(fonts, rect, px, py).is_some();
        let over_checkbox = self.checkbox_under_pointer(fonts, rect, px, py).is_some();
        if over_link || over_checkbox {
            return SHAPE_POINTER;
        }
        match self.layout_at(fonts, rect, py) {
            Some((layout, rope, y)) if layout.table_button_at(fonts, rope, px, y).is_some() => {
                SHAPE_POINTER
            }
            Some((layout, _, y)) if layout.image_button_at(fonts, px, y).is_some() => SHAPE_POINTER,
            Some((layout, _, y)) if layout.image_at(fonts, px, y).is_some() => SHAPE_DEFAULT,
            _ => SHAPE_TEXT,
        }
    }

    /// The byte range of the rendered horizontal rule under a window position, if
    /// its line is a collapsed thematic break, so a click can select the whole
    /// rule line without the caller reaching into the layout cache.
    pub(crate) fn rule_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        py: f32,
    ) -> Option<(usize, usize)> {
        let (layout, _, y) = self.layout_at(fonts, rect, py)?;
        layout.rule_at(fonts, y)
    }

    /// Arm or disarm drag auto-scroll from the pointer's distance to this
    /// column's top/bottom edge. The first arm seeds the tick deadline at now, so
    /// the first tick fires on the next [`Self::autoscroll_due`] check.
    pub(crate) fn update_autoscroll(&mut self, rect: Rect, py: f32) {
        let y = py as i32 - rect.y;
        let step = if y < AUTOSCROLL_MARGIN {
            -AUTOSCROLL_STEP
        } else if y > rect.h - AUTOSCROLL_MARGIN {
            AUTOSCROLL_STEP
        } else {
            0
        };
        if step == 0 {
            self.disarm_autoscroll();
        } else {
            self.autoscroll = step;
            self.autoscroll_at.get_or_insert_with(Instant::now);
        }
    }

    /// Stop drag auto-scroll: no step, no pending tick.
    pub(crate) fn disarm_autoscroll(&mut self) {
        self.autoscroll = 0;
        self.autoscroll_at = None;
    }

    /// The scroll step if an auto-scroll tick is due at `now`, advancing the
    /// deadline by one [`AUTOSCROLL_INTERVAL`]; `None` while disarmed or before
    /// the next tick is due.
    pub(crate) fn autoscroll_due(&mut self, now: Instant) -> Option<i32> {
        if self.autoscroll == 0 {
            return None;
        }
        let at = self.autoscroll_at?;
        if at > now {
            return None;
        }
        self.autoscroll_at = Some(now + AUTOSCROLL_INTERVAL);
        Some(self.autoscroll)
    }
}

/// The file-tree sidebar: the directory model plus its own scroll and the row
/// the pointer last hovered (for highlight). Rows come from [`tree::Tree`].
pub(crate) struct TreePane {
    pub(crate) tree: tree::Tree,
    pub(crate) scroll_y: i32,
    pub(crate) content_height: i32,
    /// The row the pointer last hovered (mouse highlight).
    pub(crate) hovered: Option<usize>,
    /// The keyboard-selected row, moved by Up/Down and acted on by Enter while
    /// the tree holds focus.
    pub(crate) selected: usize,
}

/// Build an editor pane from a saved path, restoring its caret and scroll.
/// Best-effort: an unreadable or missing file yields an empty pane (logged, not
/// fatal) so a moved-away note never stops the editor opening. `set_cursor`
/// clamps the offset into the (possibly changed) file.
pub(crate) fn restore_editor(path: Option<&Path>, saved: &store::PaneSession) -> EditorPane {
    let mut pane = match path {
        Some(p) => EditorPane::open(p).unwrap_or_else(|e| {
            eprintln!("bnknotes: could not reopen {}: {e}", p.display());
            EditorPane::empty()
        }),
        None => EditorPane::empty(),
    };
    pane.text.set_cursor(saved.cursor as usize, false);
    pane.scroll_y = saved.scroll_y.max(0);
    pane
}

/// The Smart Home target within a visual line spanning bytes `[start, end)`
/// whose text is `line`: the first non-whitespace character, or the line start
/// when the cursor is already there (a second press) or the line has no leading
/// content to skip. Pure so the toggle is unit-tested without a layout.
fn smart_home_target(line: &str, start: usize, end: usize, cursor: usize) -> usize {
    let ws = line
        .bytes()
        .take_while(|&b| b == b' ' || b == b'\t')
        .count();
    let first_non_ws = start + ws;
    if cursor != first_non_ws && first_non_ws < end {
        first_non_ws
    } else {
        start
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_lay_out_tree_and_two_editors_with_splitters() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        let cols = panes.columns(1000, 700, Scale::ONE);
        let tree = cols.tree.expect("tree shown");
        assert_eq!((tree.x, tree.w, tree.h), (0, 200, 700));
        assert_eq!(cols.splitters[0].expect("tree splitter").x, 200);
        let a = cols.editors[0].expect("editor A");
        let b = cols.editors[1].expect("editor B");
        assert_eq!(a.x, 200 + SPLITTER);
        assert_eq!(cols.splitters[1].expect("content splitter").x, a.x + a.w);
        // The content area splits evenly, and every column is full height.
        assert!((a.w - b.w).abs() <= 1);
        assert_eq!(a.h, 700);
        assert_eq!(a.x + a.w + SPLITTER + b.w, 1000);
    }

    #[test]
    fn columns_hide_tree_and_second_pane_when_off() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: false,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, Scale::ONE);
        assert!(cols.tree.is_none());
        assert!(cols.splitters[0].is_none() && cols.splitters[1].is_none());
        let a = cols.editors[0].expect("editor A");
        assert_eq!((a.x, a.w), (0, 1000));
        assert!(cols.editors[1].is_none());
    }

    #[test]
    fn smart_home_toggles_between_first_glyph_and_line_start() {
        // "    foo" occupying bytes [10, 17); its indent ends at byte 14.
        let (line, start, end) = ("    foo", 10, 17);
        // From the line end, jump to the first non-whitespace character.
        assert_eq!(smart_home_target(line, start, end, end), 14);
        // Already there (a second press): fall back to column 0.
        assert_eq!(smart_home_target(line, start, end, 14), start);
        // No indent: the first press already lands on the start.
        assert_eq!(smart_home_target("foo", 0, 3, 3), 0);
        // Blank and all-whitespace lines: only the start to go to.
        assert_eq!(smart_home_target("", 5, 5, 5), 5);
        assert_eq!(smart_home_target("   ", 0, 3, 3), 0);
    }

    #[test]
    fn clicks_chain_within_the_time_and_radius_window() {
        let mut chain = ClickChain::new();
        let t0 = Instant::now();
        // The first click is a bare single that chains onto nothing.
        assert_eq!(chain.register(10.0, 10.0, t0), (1, false));
        // A second click just inside the interval and radius chains to a double.
        let near = t0 + MULTI_CLICK_INTERVAL - Duration::from_millis(1);
        assert_eq!(chain.register(12.0, 12.0, near), (2, true));
        // A third, still in-window, escalates to a triple.
        assert_eq!(
            chain.register(12.0, 12.0, near + Duration::from_millis(1)),
            (3, true)
        );
    }

    #[test]
    fn clicks_reset_outside_the_window() {
        let mut chain = ClickChain::new();
        let t0 = Instant::now();
        assert_eq!(chain.register(10.0, 10.0, t0), (1, false));
        // Too slow: past the interval starts a fresh single.
        let slow = t0 + MULTI_CLICK_INTERVAL + Duration::from_millis(1);
        assert_eq!(chain.register(10.0, 10.0, slow), (1, false));
        // In time but too far: past the radius also starts a fresh single.
        let far = MULTI_CLICK_RADIUS + 1.0;
        assert_eq!(
            chain.register(10.0 + far, 10.0, slow + Duration::from_millis(1)),
            (1, false)
        );
    }

    #[test]
    fn autoscroll_arms_in_the_edge_zone_and_disarms_outside() {
        let mut pane = EditorPane::empty();
        let rect = Rect {
            x: 0,
            y: 0,
            w: 200,
            h: 400,
        };
        // Near the top edge: scroll up.
        pane.update_autoscroll(rect, (AUTOSCROLL_MARGIN - 1) as f32);
        assert_eq!(pane.autoscroll, -AUTOSCROLL_STEP);
        assert!(pane.autoscroll_at.is_some());
        // The interior: disarmed.
        pane.update_autoscroll(rect, (rect.h / 2) as f32);
        assert_eq!(pane.autoscroll, 0);
        assert!(pane.autoscroll_at.is_none());
        // Near the bottom edge: scroll down.
        pane.update_autoscroll(rect, (rect.h - AUTOSCROLL_MARGIN + 1) as f32);
        assert_eq!(pane.autoscroll, AUTOSCROLL_STEP);
    }

    #[test]
    fn autoscroll_fires_once_per_interval_and_re_arms() {
        let mut pane = EditorPane::empty();
        // Disarmed: never due.
        assert_eq!(pane.autoscroll_due(Instant::now()), None);
        // Arm with a known step and a deadline one interval out.
        pane.autoscroll = AUTOSCROLL_STEP;
        let now = Instant::now();
        pane.autoscroll_at = Some(now + AUTOSCROLL_INTERVAL);
        // Before the deadline: not yet due.
        assert_eq!(pane.autoscroll_due(now), None);
        // At the deadline: due, and the deadline advances by one interval.
        let due_at = now + AUTOSCROLL_INTERVAL;
        assert_eq!(pane.autoscroll_due(due_at), Some(AUTOSCROLL_STEP));
        assert_eq!(pane.autoscroll_at, Some(due_at + AUTOSCROLL_INTERVAL));
        // Disarming stops it.
        pane.disarm_autoscroll();
        assert_eq!(pane.autoscroll, 0);
        assert_eq!(pane.autoscroll_due(due_at + AUTOSCROLL_INTERVAL), None);
    }
}
