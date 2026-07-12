use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::config::CaretConfig;
use crate::file::{self, FileFormat};
use crate::find::FindState;
use crate::layout::{ImageUi, Layout, LayoutCtx, LineIndex, RevealAdjust, NO_REVEAL};
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

/// The decay step treats a stall longer than this as one bounded update, so a long
/// idle never lets the caret leap most of the way in a single frame. The
/// exponential step never overshoots regardless.
const CARET_DT_CAP: Duration = Duration::from_millis(50);
/// While the caret eases, the loop retries at least this often so a frame whose
/// integer rectangle did not change (and so presented nothing and armed no
/// compositor callback) still gets a follow-up tick.
const CARET_RETRY: Duration = Duration::from_millis(16);
/// Within half a device pixel of the target the caret snaps and settles, cutting
/// the asymptotic tail so the loop can go idle.
const CARET_SETTLE_PX: f32 = 0.5;

/// Content-space geometry the caret target was computed under: column left and
/// width, body font size, and the display scale (its 120ths factor). A change means
/// the document reflowed (resize, DPI change, column move, show/hide a pane), so the
/// caret snaps to its new target rather than smearing across the jump.
type CaretGeomKey = (i32, i32, u32, u32);

/// Euclidean distance between two content-space points.
fn caret_distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

/// The smooth-caret animation for one document: the rendered caret position that
/// eases toward its live target, plus the bookkeeping the app loop reads to keep
/// feeding it frames until it settles. Positions are content space (pre-scroll); the
/// painter shifts by scroll at draw. Inert unless [`CaretConfig::animate`] is on, in
/// which case every update snaps.
///
/// ```text
///   settled ──new target──▶ in flight ──ease each frame──▶ within 0.5px ──▶ settled
///     ▲                        │                                              │
///     └───────── snap (disabled / reflow / requested / jump > cap) ◀──────────┘
/// ```
pub(crate) struct CaretAnim {
    /// The rendered caret's content-space top-left, or `None` until the first update
    /// seeds it from the live target.
    pos: Option<(f32, f32)>,
    /// The geometry the last target was laid out under, to detect a reflow.
    key: Option<CaretGeomKey>,
    /// When the last decay step ran, or `None` while settled, so a move after an idle
    /// stretch starts its clock fresh instead of consuming one whole capped step.
    last_tick: Option<Instant>,
    /// Whether the caret is still easing toward its target. Drives frame requests.
    in_flight: bool,
    /// Force the next update to snap (a focus, tab, or file-open transition).
    snap_next: bool,
}

impl CaretAnim {
    fn new() -> Self {
        Self {
            pos: None,
            key: None,
            last_tick: None,
            in_flight: false,
            snap_next: false,
        }
    }

    /// Advance the rendered caret one frame toward `target` (content-space top-left,
    /// caret `height`) laid out under `key`, at `now`. Snaps (no smear) when
    /// animation is off, on the first update, after a requested snap, when the
    /// geometry reflowed, or when the jump exceeds the configured cap. Otherwise
    /// eases with frame-rate-independent exponential decay and settles within half a
    /// pixel.
    pub(crate) fn update(
        &mut self,
        target: (f32, f32),
        height: f32,
        key: CaretGeomKey,
        config: &CaretConfig,
        now: Instant,
    ) {
        let reflowed = self.key != Some(key);
        self.key = Some(key);
        let snap = !config.animate
            || self.snap_next
            || self.pos.is_none()
            || reflowed
            || self
                .pos
                .is_some_and(|pos| caret_distance(pos, target) > config.max_jump_lines * height);
        if snap {
            self.pos = Some(target);
            self.in_flight = false;
            self.last_tick = None;
            self.snap_next = false;
            return;
        }
        let pos = self.pos.expect("pos is Some past the snap guard");
        if pos == target {
            self.in_flight = false;
            self.last_tick = None;
            return;
        }
        if !self.in_flight {
            // A settled caret just got a new target: keep the old rendered position,
            // start the clock, and draw the full smear this frame. The first decay
            // step lands next frame, so a move after a long idle never consumes a
            // whole capped step at once.
            self.in_flight = true;
            self.last_tick = Some(now);
            return;
        }
        let dt = self
            .last_tick
            .map_or(CARET_DT_CAP, |t| now.saturating_duration_since(t))
            .min(CARET_DT_CAP);
        self.last_tick = Some(now);
        let factor = 1.0 - (-config.rate * dt.as_secs_f32()).exp();
        let next = (
            pos.0 + (target.0 - pos.0) * factor,
            pos.1 + (target.1 - pos.1) * factor,
        );
        if caret_distance(next, target) < CARET_SETTLE_PX {
            self.pos = Some(target);
            self.in_flight = false;
            self.last_tick = None;
        } else {
            self.pos = Some(next);
        }
    }

    /// Mark the next update to snap to its live target, and clear flight and timing
    /// now so a hidden, off-window, or unfocused caret cannot keep scheduling
    /// retries. Cheaper than computing layout geometry at the transition call site.
    pub(crate) fn request_snap(&mut self) {
        self.snap_next = true;
        self.in_flight = false;
        self.last_tick = None;
    }

    /// Whether the caret is still easing toward its target.
    pub(crate) fn in_flight(&self) -> bool {
        self.in_flight
    }

    /// When the loop should next tick this caret while it eases, so a quantized frame
    /// that presented nothing still gets a follow-up. `None` once settled.
    pub(crate) fn retry_at(&self) -> Option<Instant> {
        self.in_flight
            .then_some(self.last_tick)
            .flatten()
            .map(|t| t + CARET_RETRY)
    }

    /// The content-space rectangle to draw this frame: the outward-rounded bounding
    /// box spanning the rendered position and `target`, each a caret bar of width
    /// `caret_width` and the given `height`. Flooring the left/top and ceiling the
    /// right/bottom never drops the target pixel and yields exactly the caret bar at
    /// rest. Pre-scroll; the painter shifts it by scroll at draw.
    pub(crate) fn smear_rect(&self, target: (f32, f32), height: f32, caret_width: i32) -> Rect {
        let (px, py) = self.pos.unwrap_or(target);
        let (tx, ty) = target;
        let cw = caret_width as f32;
        let left = px.min(tx).floor() as i32;
        let top = py.min(ty).floor() as i32;
        let right = (px.max(tx) + cw).ceil() as i32;
        let bottom = (py.max(ty) + height).ceil() as i32;
        Rect {
            x: left,
            y: top,
            w: (right - left).max(0),
            h: (bottom - top).max(0),
        }
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
/// from the window size and the [`PaneLayout`]. A hidden column is `None`. The
/// columns start below the top bar; the bar's own regions (`vault`, `tab_strips`)
/// span the same x as the columns beneath them, so a pane's tabs sit above it.
pub(crate) struct Columns {
    pub(crate) tree: Option<Rect>,
    /// Editor A is always present; editor B only when `second_open`.
    pub(crate) editors: [Option<Rect>; 2],
    /// Grab bars: `[0]` between tree and A, `[1]` between A and B.
    pub(crate) splitters: [Option<Rect>; 2],
    /// Each pane's tab-strip region of the top bar, spanning that editor column.
    pub(crate) tab_strips: [Option<Rect>; 2],
}

impl PaneLayout {
    /// Lay the columns out across the device `width`×`height` at display `scale`,
    /// below a top bar of height `top`. The logical sidebar width and splitter
    /// thickness are scaled to device pixels here, so the persisted layout stays
    /// scale-free. Editor A always takes what is left after an optional sidebar and
    /// an optional second pane; each column is full height below the bar, so content
    /// stays in its column by construction (no clip). The bar's vault and tab-strip
    /// regions mirror the column x-spans, so a pane's tabs sit directly above it.
    pub(crate) fn columns(&self, width: i32, height: i32, top: i32, scale: Scale) -> Columns {
        let splitter = scale.px(SPLITTER);
        let tree_width = scale.px(self.tree_width);
        let body = (height - top).max(0);
        let mut x = 0;
        let mut tree = None;
        let mut splitters = [None, None];
        if self.tree_visible {
            tree = Some(Rect {
                x,
                y: top,
                w: tree_width,
                h: body,
            });
            x += tree_width;
            splitters[0] = Some(Rect {
                x,
                y: top,
                w: splitter,
                h: body,
            });
            x += splitter;
        }
        let mut editors = [None, None];
        if self.second_open {
            let content = (width - x - splitter).max(0);
            let a_w = ((content as f32 * self.split_ratio) as i32).clamp(0, content);
            editors[0] = Some(Rect {
                x,
                y: top,
                w: a_w,
                h: body,
            });
            x += a_w;
            splitters[1] = Some(Rect {
                x,
                y: top,
                w: splitter,
                h: body,
            });
            x += splitter;
            editors[1] = Some(Rect {
                x,
                y: top,
                w: (width - x).max(0),
                h: body,
            });
        } else {
            editors[0] = Some(Rect {
                x,
                y: top,
                w: (width - x).max(0),
                h: body,
            });
        }
        // Each pane's tab strip takes its column's x-span at the top strip's height.
        let strip = |col: Option<Rect>| {
            col.map(|c| Rect {
                x: c.x,
                y: 0,
                w: c.w,
                h: top,
            })
        };
        let tab_strips = [strip(editors[0]), strip(editors[1])];
        Columns {
            tree,
            editors,
            splitters,
            tab_strips,
        }
    }
}

/// What a window position lands on, for routing pointer events to the right
/// region. The top bar is checked first (it sits above every column), then
/// splitters (they sit in the gaps between columns).
pub(crate) enum Hit {
    Splitter(usize),
    Tree(Rect),
    Editor(usize, Rect),
    /// A tab in pane `group`'s strip: switch to it.
    Tab(usize, usize),
    /// The close box of a tab in pane `group`'s strip: close it.
    TabClose(usize, usize),
    /// The new-tab (+) button at the end of pane `group`'s strip: open a blank tab.
    NewTab(usize),
    /// Empty area of pane `group`'s tab strip; consumes the click by focusing the
    /// pane without acting on a tab.
    TabStrip(usize),
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
    pub(crate) clicks: ClickChain,
    /// Drag auto-scroll step: pixels per tick (negative up, positive down), 0 when
    /// disarmed. Armed while a selection drag holds the pointer in a column's edge
    /// zone; see [`Self::update_autoscroll`] and [`Self::autoscroll_due`].
    pub(crate) autoscroll: i32,
    /// When the next auto-scroll tick is due; `None` while disarmed.
    pub(crate) autoscroll_at: Option<Instant>,
    /// The smooth-caret animation for this document: the rendered caret easing
    /// toward the cursor's live position. Inert unless the config enables it.
    caret_anim: CaretAnim,
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
            clicks: ClickChain::new(),
            autoscroll: 0,
            autoscroll_at: None,
            caret_anim: CaretAnim::new(),
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

    /// The tab's label: the file's name without its .md suffix (matching the
    /// tree), or "untitled" for a file-less scratch buffer.
    fn tab_name(&self) -> String {
        self.path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|s| s.to_str())
            .map(str::to_string)
            .unwrap_or_else(|| "untitled".to_string())
    }

    /// Whether this is a fresh, untouched scratch tab (no file, empty buffer), so
    /// opening a note into a pane showing one replaces it rather than stacking a
    /// blank tab beside it.
    fn is_blank(&self) -> bool {
        self.path.is_none() && self.text.rope().len() == 0
    }

    /// The active range for live-preview marker reveal: the live selection, or the
    /// bare cursor when there is none. A drag in progress reveals nothing
    /// ([`NO_REVEAL`]): the whole document stays rendered while selecting, and the
    /// markers under the selection reveal only once the button is released.
    pub(crate) fn reveal_range(&self) -> (usize, usize) {
        if self.selecting {
            return NO_REVEAL;
        }
        let cursor = self.text.cursor();
        self.text.selection().unwrap_or((cursor, cursor))
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
        // A drag suppresses selection-driven reveal (NO_REVEAL touches no line), so
        // no image source expands while selecting; a corner-pinned source below
        // still does.
        if (lo, hi) != NO_REVEAL {
            let l0 = rope.byte_to_line(lo);
            adjust.push(l0, index.reveal_extra(l0));
            let l1 = rope.byte_to_line(hi);
            if l1 != l0 {
                adjust.push(l1, index.reveal_extra(l1));
            }
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
        // Reveal depends only on which lines the active range touches, so the key
        // is coarsened to lines. NO_REVEAL is kept verbatim: mapping it through
        // byte_to_line would clamp to the last line and could collide with a real
        // cursor there, so a drag starting on that line would not un-reveal it.
        let active_lines = if active == NO_REVEAL {
            NO_REVEAL
        } else {
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
        // The caret comes from a window built at absolute positions (the column top
        // baked in), but scroll::reveal works in content space against the pane's
        // scroll and content height, so drop the column top first.
        let (_, caret_top, caret_h) = win.caret(fonts, self.text.rope(), cursor);
        let sy = scroll::reveal(
            self.scroll_y,
            caret_top - rect.y,
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

    /// This document's caret animation, for the app loop to read its flight state
    /// and retry deadline.
    pub(crate) fn caret_anim(&self) -> &CaretAnim {
        &self.caret_anim
    }

    /// This document's caret animation, for the frame builder to tick it.
    pub(crate) fn caret_anim_mut(&mut self) -> &mut CaretAnim {
        &mut self.caret_anim
    }

    /// Snap the caret to its live target on the next frame, so a focus, tab, or
    /// file-open change never resumes a stale mid-flight smear.
    pub(crate) fn request_caret_snap(&mut self) {
        self.caret_anim.request_snap();
    }

    /// The layout-space y for a window y inside this pane's column. The windowed
    /// layout is built at absolute positions (the column top and margin baked into
    /// its baselines by the seed), with the pane's scroll applied only at paint, so
    /// a hit-test recovers the layout frame by adding the scroll back. This mirrors
    /// the x side, where the raw window x is passed straight through (the column
    /// left is baked into the layout too); only y is scrolled.
    pub(crate) fn content_y(&self, py: f32) -> f32 {
        py + self.scroll_y as f32
    }

    /// Ensure the windowed layout covers `rect`, then return it (with the buffer
    /// rope, since the hit-tests need both) and the content-space y for the window
    /// position `py`. `None` when no layout could be built. The shared preamble of
    /// the pointer-query methods below; it hands back the rope so a caller need
    /// not re-borrow `self` (which the returned layout reference locks) to reach
    /// it.
    fn layout_at(&mut self, fonts: &Fonts, rect: Rect, py: f32) -> Option<(&Layout, &Rope, f32)> {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(py);
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

/// One content pane as a stack of open documents with one active. A pane may hold
/// no documents at all (nothing open), in which case it shows an empty column and
/// a bare tab strip with just the new-tab button. The group forwards to its active
/// document (via [`Deref`]), so a pane's active tab is simply "the pane" to every
/// caller that edits, scrolls, or hit-tests it; only tab management (open, close,
/// switch) and the DPI fan-out are group-level. So the deref stays total even with
/// no tabs, an empty pane forwards to a scratch `placeholder` document; it is never
/// drawn (an empty pane paints no content), so editing it is a harmless no-op until
/// the first keystroke materializes a real tab.
pub(crate) struct PaneGroup {
    docs: Vec<EditorPane>,
    active: usize,
    /// The document an empty pane's deref targets, so the `pane.field` sites stay
    /// panic-free with no tabs open. Never painted.
    placeholder: EditorPane,
    /// The display scale and body size the group lays out at, so a tab opened
    /// after a DPI change is stamped with the current scale rather than the unity
    /// default a fresh [`EditorPane`] carries.
    scale: Scale,
    body_size: u32,
}

impl PaneGroup {
    /// A pane with nothing open (no tabs).
    pub(crate) fn empty() -> Self {
        Self {
            docs: Vec::new(),
            active: 0,
            placeholder: EditorPane::empty(),
            scale: Scale::ONE,
            body_size: FONT_SIZE,
        }
    }

    /// The active tab index (0 and meaningless when the pane has no tabs).
    pub(crate) fn active_index(&self) -> usize {
        self.active
    }

    /// Whether the pane has any open tab. An empty pane paints no content and no
    /// caret, and its title falls back to the app name.
    pub(crate) fn has_tabs(&self) -> bool {
        !self.docs.is_empty()
    }

    /// The active document, or the placeholder when the pane is empty. Most callers
    /// reach it implicitly through [`Deref`]; this explicit borrow is for the few
    /// sites that mutate one field while reading another, which a deref cannot split
    /// into disjoint borrows.
    pub(crate) fn active_doc_mut(&mut self) -> &mut EditorPane {
        self.docs
            .get_mut(self.active)
            .unwrap_or(&mut self.placeholder)
    }

    /// Each tab's label, in order, for the tab strip.
    pub(crate) fn tab_names(&self) -> Vec<String> {
        self.docs.iter().map(EditorPane::tab_name).collect()
    }

    /// Whether each tab has unsaved edits, in order, for the tab strip's dot.
    pub(crate) fn tab_dirty(&self) -> Vec<bool> {
        self.docs.iter().map(|d| d.text.is_modified()).collect()
    }

    /// Switch the active tab. Out-of-range indices are ignored.
    pub(crate) fn activate(&mut self, i: usize) {
        if i < self.docs.len() {
            self.active = i;
            self.active_doc_mut().request_caret_snap();
        }
    }

    /// Open a fresh blank tab and focus it, stamped with the group's scale. Backs
    /// the new-tab button and the first keystroke into an empty pane.
    pub(crate) fn new_tab(&mut self, scale: Scale, body_size: u32) {
        let mut doc = EditorPane::empty();
        doc.set_scale(scale, body_size);
        self.docs.push(doc);
        self.active = self.docs.len() - 1;
    }

    /// Open `path`: focus an existing tab holding it, replace a blank scratch tab
    /// in place, or stack a new tab and focus it. The new document adopts the
    /// group's current scale so it renders crisp immediately at any DPI.
    pub(crate) fn open_or_focus(&mut self, path: &Path) -> Result<()> {
        if let Some(i) = self
            .docs
            .iter()
            .position(|d| d.path.as_deref() == Some(path))
        {
            self.active = i;
            self.active_doc_mut().request_caret_snap();
            return Ok(());
        }
        let mut doc = EditorPane::open(path)?;
        doc.set_scale(self.scale, self.body_size);
        // Replace a lone blank scratch tab (a just-opened new tab); otherwise stack.
        if self.docs.get(self.active).is_some_and(EditorPane::is_blank) {
            self.docs[self.active] = doc;
        } else {
            self.docs.push(doc);
            self.active = self.docs.len() - 1;
        }
        Ok(())
    }

    /// Close tab `i`. Closing the last tab empties the pane (no tabs). The active
    /// tab stays put where it can; closing it falls to the neighbor that slides
    /// into its place.
    pub(crate) fn close(&mut self, i: usize) {
        self.detach(i);
    }

    /// Remove tab `i` and hand back its document, so it can be re-homed in another
    /// pane (a tab dragged across the split). The active tab shifts the same way
    /// [`Self::close`] moves it. An out-of-range index yields `None`.
    pub(crate) fn detach(&mut self, i: usize) -> Option<EditorPane> {
        if i >= self.docs.len() {
            return None;
        }
        let doc = self.docs.remove(i);
        if i < self.active || self.active >= self.docs.len() {
            self.active = self.active.saturating_sub(1);
        }
        // The tab that slid into focus is a different document; snap its caret so it
        // does not resume a smear from wherever it last left off.
        self.active_doc_mut().request_caret_snap();
        Some(doc)
    }

    /// Insert `doc` as a tab at index `i` (clamped to the end), adopt it under the
    /// group's scale, make it active, and return the index it landed at. Receives a
    /// tab dragged in from the other pane.
    pub(crate) fn insert_tab(&mut self, i: usize, mut doc: EditorPane) -> usize {
        doc.set_scale(self.scale, self.body_size);
        let i = i.min(self.docs.len());
        self.docs.insert(i, doc);
        self.active = i;
        i
    }

    /// Move the tab at `from` to index `to`, sliding the tabs in between over to
    /// fill the gap. The same document stays active across the move, whichever slot
    /// it lands in. Out-of-range or no-op moves are ignored.
    pub(crate) fn move_tab(&mut self, from: usize, to: usize) {
        if from >= self.docs.len() || to >= self.docs.len() || from == to {
            return;
        }
        let doc = self.docs.remove(from);
        self.docs.insert(to, doc);
        self.active = if self.active == from {
            to
        } else {
            let mut a = self.active;
            if a > from {
                a -= 1;
            }
            if a >= to {
                a += 1;
            }
            a
        };
    }

    /// Adopt a new display scale and body size (a DPI change), fanning it out to
    /// every tab (and the placeholder) so an inactive tab re-measures correctly
    /// when next shown.
    pub(crate) fn set_scale(&mut self, scale: Scale, body_size: u32) {
        self.scale = scale;
        self.body_size = body_size;
        for doc in &mut self.docs {
            doc.set_scale(scale, body_size);
        }
        self.placeholder.set_scale(scale, body_size);
    }

    /// The persisted form of this pane: each tab's file, caret, and scroll, plus
    /// the active index.
    pub(crate) fn to_session(&self) -> store::PaneGroupSession {
        store::PaneGroupSession {
            tabs: self
                .docs
                .iter()
                .map(|d| store::PaneSession {
                    path: d.path.clone(),
                    cursor: d.text.cursor() as u64,
                    scroll_y: d.scroll_y,
                })
                .collect(),
            active: self.active as u32,
        }
    }
}

impl Deref for PaneGroup {
    type Target = EditorPane;

    fn deref(&self) -> &EditorPane {
        self.docs.get(self.active).unwrap_or(&self.placeholder)
    }
}

impl DerefMut for PaneGroup {
    fn deref_mut(&mut self) -> &mut EditorPane {
        self.docs
            .get_mut(self.active)
            .unwrap_or(&mut self.placeholder)
    }
}

/// The file-tree sidebar: the directory model plus its own scroll and the row
/// the pointer last hovered (for highlight). Rows come from [`tree::Tree`].
pub(crate) struct TreePane {
    pub(crate) tree: tree::Tree,
    pub(crate) scroll_y: i32,
    pub(crate) content_height: i32,
    /// The row the pointer last hovered (mouse highlight), or `None` when the
    /// pointer is off the tree.
    pub(crate) hovered: Option<usize>,
    /// The active row: the last one clicked or opened, and the target of Up/Down
    /// and Enter. `None` until the user picks one, so nothing is marked at launch.
    pub(crate) selected: Option<usize>,
}

/// Build a pane from its saved tabs, restoring each file's caret and scroll and
/// the active tab. An empty tab list (a never-opened pane) yields an empty pane
/// with no tabs. The active index is clamped into the restored tabs.
pub(crate) fn restore_group(tabs: &[store::PaneSession], active: usize) -> PaneGroup {
    if tabs.is_empty() {
        return PaneGroup::empty();
    }
    let docs: Vec<EditorPane> = tabs.iter().map(restore_doc).collect();
    let active = active.min(docs.len() - 1);
    PaneGroup {
        docs,
        active,
        placeholder: EditorPane::empty(),
        scale: Scale::ONE,
        body_size: FONT_SIZE,
    }
}

/// Build one document from a saved tab, restoring its caret and scroll.
/// Best-effort: an unreadable or missing file yields an empty document (logged,
/// not fatal) so a moved-away note never stops the editor opening. `set_cursor`
/// clamps the offset into the (possibly changed) file.
fn restore_doc(saved: &store::PaneSession) -> EditorPane {
    let mut doc = match saved.path.as_deref() {
        Some(p) => EditorPane::open(p).unwrap_or_else(|e| {
            eprintln!("bnknotes: could not reopen {}: {e}", p.display());
            EditorPane::empty()
        }),
        None => EditorPane::empty(),
    };
    doc.text.set_cursor(saved.cursor as usize, false);
    doc.scroll_y = saved.scroll_y.max(0);
    doc
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
    fn a_drag_suppresses_reveal_until_release() {
        let mut ed = EditorPane::empty();
        ed.text.insert_str("# Title\nbody");
        // A settled selection over the heading is its own reveal range, so the
        // heading shows its markers.
        ed.text.set_cursor(0, false);
        ed.text.set_cursor(7, true);
        assert_eq!(ed.reveal_range(), (0, 7));
        // While the drag is live, nothing reveals: the whole document stays
        // rendered, headings included.
        ed.selecting = true;
        assert_eq!(ed.reveal_range(), NO_REVEAL);
        // Releasing lets the reveal catch up to the settled selection.
        ed.selecting = false;
        assert_eq!(ed.reveal_range(), (0, 7));
    }

    #[test]
    fn reveal_follows_the_bare_cursor_when_not_selecting() {
        let mut ed = EditorPane::empty();
        ed.text.insert_str("# Title");
        ed.text.set_cursor(3, false);
        assert_eq!(ed.reveal_range(), (3, 3));
    }

    /// Whether the heading logical line starting at `heading_start` has its `#`
    /// marker collapsed in the pane's freshly built layout.
    fn heading_marker_hidden(pane: &EditorPane, heading_start: usize) -> bool {
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        let line = layout
            .lines()
            .iter()
            .find(|l| l.start == heading_start)
            .expect("the heading's visual line");
        let run = layout.run_slice(line).first().expect("a marker run");
        assert!(
            matches!(run.kind, markdown::SpanKind::Marker),
            "the first run is the heading marker"
        );
        !run.visible
    }

    #[test]
    fn a_drag_beginning_on_the_last_line_still_un_reveals_it() {
        // A heading on the final line: with the cursor there it reveals its `#`.
        // Starting a drag from it must collapse the marker again, though NO_REVEAL
        // and a last-line cursor both fold to the last line index, so the layout
        // cache key must keep them apart.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let text = "body line\n# Heading";
        let heading_start = text.find("# Heading").expect("heading present");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 600,
            h: 400,
        };
        let mut pane = EditorPane::empty();
        pane.text = TextBuffer::from_text(text.to_string());
        pane.set_scale(Scale::ONE, FONT_SIZE);

        pane.text.set_cursor(heading_start + 2, false);
        pane.ensure_layout(&fonts, rect);
        assert!(
            !heading_marker_hidden(&pane, heading_start),
            "cursor on the heading reveals its '#'"
        );

        // A drag now begins on that same line; the reveal must switch off.
        pane.selecting = true;
        pane.ensure_layout(&fonts, rect);
        assert!(
            heading_marker_hidden(&pane, heading_start),
            "a live drag collapses the heading marker again"
        );
    }

    /// A group of `n` tabs, each stamped with a distinct `scroll_y` so a test can
    /// track where a given document lands after a reorder.
    fn marked_group(n: i32) -> PaneGroup {
        let mut g = PaneGroup::empty();
        for i in 0..n {
            g.new_tab(Scale::ONE, FONT_SIZE);
            g.active_doc_mut().scroll_y = i;
        }
        g
    }

    fn order(g: &PaneGroup) -> Vec<i32> {
        g.docs.iter().map(|d| d.scroll_y).collect()
    }

    #[test]
    fn move_tab_reorders_and_carries_the_active_document() {
        // Documents 0..4; make document 1 (the second) active, then drag the first
        // tab to the third slot. The middle tabs slide left, and the active follows
        // its document to the slot it now occupies.
        let mut g = marked_group(4);
        g.activate(1);
        g.move_tab(0, 2);
        assert_eq!(order(&g), vec![1, 2, 0, 3]);
        assert_eq!(g.active_index(), 0);
    }

    #[test]
    fn move_tab_follows_the_dragged_tab_when_it_is_active() {
        // Dragging the active tab itself: it stays active in its new slot.
        let mut g = marked_group(4);
        g.activate(0);
        g.move_tab(0, 3);
        assert_eq!(order(&g), vec![1, 2, 3, 0]);
        assert_eq!(g.active_index(), 3);
    }

    #[test]
    fn detach_and_insert_carry_a_tab_across_panes() {
        // Pane A holds tabs 0,1,2 with the middle one active; pane B holds two tabs
        // marked apart (10, 11) with the second active.
        let mut a = marked_group(3);
        a.activate(1);
        let mut b = marked_group(2);
        b.docs[0].scroll_y = 10;
        b.docs[1].scroll_y = 11;

        // Drag A's active tab out: it leaves A, whose active falls to the neighbor
        // that slid into its place.
        let doc = a.detach(1).expect("a tab to detach");
        assert_eq!(doc.scroll_y, 1);
        assert_eq!(order(&a), vec![0, 2]);
        assert_eq!(a.active_index(), 1);

        // Drop it at the front of B: B now owns it and makes it active.
        let landed = b.insert_tab(0, doc);
        assert_eq!(landed, 0);
        assert_eq!(order(&b), vec![1, 10, 11]);
        assert_eq!(b.active_index(), 0);

        // An out-of-range detach is a no-op that hands back nothing.
        assert!(a.detach(9).is_none());
        assert_eq!(order(&a), vec![0, 2]);
    }

    #[test]
    fn insert_tab_clamps_past_the_end() {
        let mut b = marked_group(2);
        let doc = EditorPane::empty();
        let landed = b.insert_tab(99, doc);
        assert_eq!(landed, 2);
        assert_eq!(b.active_index(), 2);
    }

    #[test]
    fn move_tab_ignores_out_of_range_and_no_op_moves() {
        let mut g = marked_group(3);
        g.activate(2);
        g.move_tab(1, 1); // no-op
        g.move_tab(0, 9); // out of range
        g.move_tab(9, 0); // out of range
        assert_eq!(order(&g), vec![0, 1, 2]);
        assert_eq!(g.active_index(), 2);
    }

    #[test]
    fn columns_lay_out_tree_and_two_editors_with_splitters() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        let cols = panes.columns(1000, 700, 0, Scale::ONE);
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
    fn columns_reserve_the_top_bar_and_align_strips() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE);
        // Every column starts below the bar and is that much shorter.
        let tree = cols.tree.expect("tree shown");
        assert_eq!((tree.y, tree.h), (30, 670));
        let a = cols.editors[0].expect("editor A");
        let b = cols.editors[1].expect("editor B");
        assert_eq!((a.y, a.h), (30, 670));
        // Each pane's tab strip sits above its column at the bar height.
        let strip_a = cols.tab_strips[0].expect("strip A");
        let strip_b = cols.tab_strips[1].expect("strip B");
        assert_eq!(
            (strip_a.x, strip_a.y, strip_a.w, strip_a.h),
            (a.x, 0, a.w, 30)
        );
        assert_eq!((strip_b.x, strip_b.w, strip_b.h), (b.x, b.w, 30));
    }

    #[test]
    fn hidden_sidebar_fills_the_strip_full_width() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: false,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE);
        let strip_a = cols.tab_strips[0].expect("strip A");
        assert_eq!((strip_a.x, strip_a.w), (0, 1000));
        assert!(cols.tab_strips[1].is_none());
    }

    #[test]
    fn open_replaces_a_blank_new_tab_then_stacks_and_dedups() {
        let dir = std::env::temp_dir().join(format!("bnknotes_group_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let a = dir.join("a.md");
        let b = dir.join("b.md");
        std::fs::write(&a, b"alpha").expect("write a");
        std::fs::write(&b, b"beta").expect("write b");

        let mut group = PaneGroup::empty();
        // An empty pane has no tabs; opening a file gives it its first tab.
        assert!(!group.has_tabs());
        group.open_or_focus(&a).expect("open a");
        assert_eq!(group.tab_names(), vec!["a".to_string()]);
        assert_eq!(group.active_index(), 0);
        // A second, different file stacks a new tab and focuses it.
        group.open_or_focus(&b).expect("open b");
        assert_eq!(group.tab_names(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(group.active_index(), 1);
        // Reopening the first file focuses its existing tab, no duplicate.
        group.open_or_focus(&a).expect("reopen a");
        assert_eq!(group.tab_names().len(), 2);
        assert_eq!(group.active_index(), 0);
        // A fresh blank tab (the new-tab button) is replaced in place by an open.
        group.new_tab(Scale::ONE, FONT_SIZE);
        assert_eq!(group.tab_names(), vec!["a", "b", "untitled"]);
        let c = dir.join("c.md");
        std::fs::write(&c, b"gamma").expect("write c");
        group.open_or_focus(&c).expect("open c");
        assert_eq!(group.tab_names(), vec!["a", "b", "c"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn closing_the_last_tab_empties_the_pane() {
        let mut group = PaneGroup::empty();
        group.new_tab(Scale::ONE, FONT_SIZE);
        assert!(group.has_tabs());
        group.close(0);
        assert!(!group.has_tabs());
        assert!(group.tab_names().is_empty());
    }

    #[test]
    fn closing_a_tab_before_the_active_one_shifts_the_active_index() {
        let dir = std::env::temp_dir().join(format!("bnknotes_close_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let paths: Vec<PathBuf> = ["a", "b", "c"]
            .iter()
            .map(|n| {
                let p = dir.join(format!("{n}.md"));
                std::fs::write(&p, n.as_bytes()).expect("write");
                p
            })
            .collect();
        let mut group = PaneGroup::empty();
        for p in &paths {
            group.open_or_focus(p).expect("open");
        }
        // Active is the third tab (index 2); closing the first slides it to 1.
        assert_eq!(group.active_index(), 2);
        group.close(0);
        assert_eq!(group.tab_names(), vec!["b".to_string(), "c".to_string()]);
        assert_eq!(group.active_index(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_scale_reaches_every_tab_and_new_tabs() {
        let dir = std::env::temp_dir().join(format!("bnknotes_scale_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let a = dir.join("a.md");
        std::fs::write(&a, b"alpha").expect("write");
        let mut group = PaneGroup::empty();
        let scale = Scale::from_120(180);
        group.set_scale(scale, 21);
        // Every existing tab adopts the new body size, and so does one opened after.
        assert_eq!(group.body_size, 21);
        group.open_or_focus(&a).expect("open");
        assert_eq!(group.docs[group.active].body_size, 21);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn columns_hide_tree_and_second_pane_when_off() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: false,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, 0, Scale::ONE);
        assert!(cols.tree.is_none());
        assert!(cols.splitters[0].is_none() && cols.splitters[1].is_none());
        let a = cols.editors[0].expect("editor A");
        assert_eq!((a.x, a.w), (0, 1000));
        assert!(cols.editors[1].is_none());
    }

    #[test]
    fn hit_test_is_invariant_to_the_column_top_offset() {
        // A click over a given content position must land on the same byte offset
        // whether or not the column starts below a top bar. The layout bakes the
        // column top into its baselines, so the hit-test must not double-count it.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let text = "a normal paragraph line of prose\n# A tall heading here\nfinal line after it\n";
        let pane_at = |top: i32| {
            let mut pane = EditorPane::empty();
            pane.text = TextBuffer::from_text(text.to_string());
            pane.set_scale(Scale::ONE, FONT_SIZE);
            (
                pane,
                Rect {
                    x: 0,
                    y: top,
                    w: 600,
                    h: 400,
                },
            )
        };
        let (mut flush, r_flush) = pane_at(0);
        let (mut barred, r_barred) = pane_at(40);
        for dy in [4.0, 24.0, 48.0, 80.0, 120.0] {
            let a = flush.pointer_offset(&fonts, r_flush, 120.0, dy);
            // The barred column and the pointer both shift down by the bar height.
            let b = barred.pointer_offset(&fonts, r_barred, 120.0, dy + 40.0);
            assert_eq!(a, b, "hit at content y {dy} must ignore the column top");
        }
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

    // --- Caret animation --------------------------------------------------

    /// A geometry key standing in for one column, body size, and scale; the tests
    /// only care that it stays constant (a change would force a snap).
    const CARET_KEY: CaretGeomKey = (0, 600, FONT_SIZE, 120);
    /// The caret height the tests animate with.
    const CARET_H: f32 = 20.0;

    /// An animating config with the given jump cap (in caret heights). A large cap
    /// takes the distance guard out of the way for the easing tests.
    fn anim_cfg(max_jump_lines: f32) -> CaretConfig {
        CaretConfig {
            animate: true,
            rate: 40.0,
            max_jump_lines,
        }
    }

    #[test]
    fn caret_initializes_at_target() {
        let mut a = CaretAnim::new();
        a.update(
            (100.0, 50.0),
            CARET_H,
            CARET_KEY,
            &anim_cfg(1000.0),
            Instant::now(),
        );
        // First update seeds the rendered position at the live target: settled, no
        // smear from the default origin.
        assert_eq!(a.pos, Some((100.0, 50.0)));
        assert!(!a.in_flight());
        let r = a.smear_rect((100.0, 50.0), CARET_H, 2);
        assert_eq!(r.w, 2);
        assert_eq!(r.h, 20);
    }

    #[test]
    fn caret_converges_to_target() {
        let cfg = anim_cfg(1000.0);
        let mut a = CaretAnim::new();
        let mut t = Instant::now();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t);
        a.update((200.0, 0.0), CARET_H, CARET_KEY, &cfg, t); // flight starts, no step
        for _ in 0..200 {
            t += Duration::from_millis(16);
            a.update((200.0, 0.0), CARET_H, CARET_KEY, &cfg, t);
            if !a.in_flight() {
                break;
            }
        }
        assert_eq!(a.pos, Some((200.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_easing_is_frame_rate_independent() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        // One 32ms step.
        let mut one = CaretAnim::new();
        one.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        one.update((300.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        one.update(
            (300.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(32),
        );
        // Two 16ms steps.
        let mut two = CaretAnim::new();
        two.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        two.update((300.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        two.update(
            (300.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(16),
        );
        two.update(
            (300.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(32),
        );
        // Exponential decay composes over dt, so the same elapsed time lands in the
        // same place however it is subdivided.
        assert!((one.pos.unwrap().0 - two.pos.unwrap().0).abs() < 0.5);
    }

    #[test]
    fn caret_settles_within_half_a_pixel() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        // Sit just under a pixel from the target while in flight; the next step
        // crosses the half-pixel threshold and snaps exactly, with no asymptotic
        // tail.
        a.pos = Some((99.7, 0.0));
        a.in_flight = true;
        a.last_tick = Some(t0);
        a.update(
            (100.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(16),
        );
        assert_eq!(a.pos, Some((100.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_requested_snap_lands_at_target() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        a.update((50.0, 0.0), CARET_H, CARET_KEY, &cfg, t0); // in flight
        a.request_snap();
        a.update((50.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        assert_eq!(a.pos, Some((50.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_jump_beyond_cap_snaps() {
        let cfg = anim_cfg(6.0); // cap = 6 * 20px = 120px
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        a.update((500.0, 0.0), CARET_H, CARET_KEY, &cfg, t0); // 500px > cap
        assert_eq!(a.pos, Some((500.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_dt_is_capped_and_never_overshoots() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        // A capped (50ms) step.
        let mut capped = CaretAnim::new();
        capped.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        capped.update((400.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        capped.update(
            (400.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(50),
        );
        // A huge elapsed duration clamps to the same cap.
        let mut huge = CaretAnim::new();
        huge.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        huge.update((400.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        huge.update(
            (400.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_secs(10),
        );
        assert!((capped.pos.unwrap().0 - huge.pos.unwrap().0).abs() < 1e-3);
        // The exponential step approaches but never passes the target.
        assert!(huge.pos.unwrap().0 < 400.0);
    }

    #[test]
    fn caret_smear_spans_pos_to_target_and_is_the_bar_at_rest() {
        let cfg = anim_cfg(1000.0);
        let mut a = CaretAnim::new();
        a.update((100.0, 40.0), CARET_H, CARET_KEY, &cfg, Instant::now());
        // At rest the box is exactly the caret bar.
        let rest = a.smear_rect((100.0, 40.0), CARET_H, 2);
        assert_eq!(
            rest,
            Rect {
                x: 100,
                y: 40,
                w: 2,
                h: 20
            }
        );
        // A rightward in-flight state spans pos.x to target.x + CARET_WIDTH.
        a.pos = Some((100.0, 40.0));
        let smear = a.smear_rect((150.0, 40.0), CARET_H, 2);
        assert_eq!(smear.x, 100);
        assert_eq!(smear.x + smear.w, 152);
        assert_eq!(smear.y, 40);
        assert_eq!(smear.h, 20);
    }

    #[test]
    fn caret_disabled_is_the_plain_bar() {
        let cfg = CaretConfig {
            animate: false,
            rate: 40.0,
            max_jump_lines: 6.0,
        };
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((100.0, 40.0), CARET_H, CARET_KEY, &cfg, t0);
        assert!(!a.in_flight());
        // With animation off the paint rect is the same Fill bar as the pre-animation
        // path (the caller draws radius zero because it never flies).
        assert_eq!(
            a.smear_rect((100.0, 40.0), CARET_H, 2),
            Rect {
                x: 100,
                y: 40,
                w: 2,
                h: 20
            }
        );
        // A move snaps immediately; it never eases.
        a.update(
            (300.0, 40.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t0 + Duration::from_millis(16),
        );
        assert_eq!(a.pos, Some((300.0, 40.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_exposes_a_retry_only_while_in_flight() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        assert!(a.retry_at().is_none()); // settled
        a.update((80.0, 0.0), CARET_H, CARET_KEY, &cfg, t0); // in flight
        assert!(a.retry_at().is_some_and(|at| at > t0));
    }

    #[test]
    fn caret_move_after_idle_starts_the_clock_fresh() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        // A move long after settling: the first frame keeps the old position and just
        // starts the clock, so no animation time is consumed by the idle stretch.
        let t1 = t0 + Duration::from_secs(100);
        a.update((60.0, 0.0), CARET_H, CARET_KEY, &cfg, t1);
        assert_eq!(a.pos, Some((0.0, 0.0)));
        assert!(a.in_flight());
        assert_eq!(a.last_tick, Some(t1));
        // The next tick uses the real 16ms elapsed, not the 100s idle, so it eases
        // instead of leaping most of the way.
        a.update(
            (60.0, 0.0),
            CARET_H,
            CARET_KEY,
            &cfg,
            t1 + Duration::from_millis(16),
        );
        let moved = a.pos.unwrap().0;
        assert!(moved > 0.0 && moved < 40.0);
    }

    #[test]
    fn caret_request_snap_clears_flight_immediately() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update((0.0, 0.0), CARET_H, CARET_KEY, &cfg, t0);
        a.update((80.0, 0.0), CARET_H, CARET_KEY, &cfg, t0); // in flight
        assert!(a.in_flight());
        // A hidden or off-window caret requests a snap; flight and timing clear at
        // once so it cannot keep scheduling retries.
        a.request_snap();
        assert!(!a.in_flight());
        assert!(a.retry_at().is_none());
    }
}
