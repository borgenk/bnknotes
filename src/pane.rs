use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::config::CaretConfig;
use crate::file::{self, FileFormat};
use crate::find::FindState;
use crate::fold::{self, FoldMap, Folds};
use crate::frontmatter;
use crate::layout::{
    Adjust, ImageUi, Layout, LayoutCtx, LineIndex, NO_REVEAL, RevealAdjust, VisualLine,
};
use crate::markdown;
use crate::platform::error::{Error, Result};
use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::platform::protocol::wp_cursor_shape_device_v1;
use crate::platform::scroll::{self, Scrollbar};
use crate::render::image::ImageStore;
use crate::rope::Rope;
use crate::scale::Scale;
use crate::store;
use crate::text::{DragGrain, TextBuffer};
use crate::theme::{DIFF_MARGIN, FONT_SIZE, MARGIN, RAIL_W};
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
/// The first easing step after a settle uses this nominal frame time rather than the
/// real elapsed gap, so a move after a long idle grows its smear from the caret
/// instead of leaping most of the way in one capped step. Steps in flight use the
/// real elapsed time.
const CARET_FIRST_STEP: Duration = Duration::from_millis(16);
/// The vertical axis eases faster than the horizontal by this factor. An up/down line
/// change covers about three times the pixel distance of a left/right character step,
/// so at a shared rate it would drag a tall smear and read as sluggish. Boosting the
/// vertical rate drops the caret into the new line quickly while it still slides
/// horizontally.
const CARET_VERTICAL_BOOST: f32 = 1.6;

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

/// Linear interpolation from `a` to `b` by `f` in 0..=1.
fn lerp(a: f32, b: f32, f: f32) -> f32 {
    a + (b - a) * f
}

/// The leading- and trailing-edge fractions to ease one axis this step. The edge
/// leading the direction of travel moves by `lead`; the trailing edge lags at
/// `trail`, and that gap is the stretch. The two axes carry their own factors so the
/// vertical axis can ease faster than the horizontal.
#[derive(Clone, Copy)]
struct AxisEase {
    lead: f32,
    trail: f32,
}

/// The rendered caret as its four content-space edges. At rest right - left is the
/// caret width and bottom - top the line height; in flight the edge leading the
/// direction of travel pulls ahead of the trailing one, so the box stretches the way
/// the caret is going and collapses back to a bar as it arrives.
#[derive(Clone, Copy, PartialEq, Debug)]
struct CaretBox {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl CaretBox {
    /// The caret bar at content-space `(x, y)` with the given device width and line
    /// height. This is the caret at rest and the target every step eases toward.
    fn bar(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        }
    }

    /// Ease each edge one step toward `target`, the horizontal edges by `x` and the
    /// vertical edges by `y`. On each axis the edge leading the direction of travel
    /// moves at `lead` and the trailing edge lags at `trail`: moving right, the right
    /// edge races ahead while the left edge trails, so the box elongates rightward,
    /// then the left edge catches up.
    fn ease_toward(self, target: Self, x: AxisEase, y: AxisEase) -> Self {
        let (fl, fr) = if target.left >= self.left {
            (x.trail, x.lead)
        } else {
            (x.lead, x.trail)
        };
        let (ft, fb) = if target.top >= self.top {
            (y.trail, y.lead)
        } else {
            (y.lead, y.trail)
        };
        Self {
            left: lerp(self.left, target.left, fl),
            top: lerp(self.top, target.top, ft),
            right: lerp(self.right, target.right, fr),
            bottom: lerp(self.bottom, target.bottom, fb),
        }
    }

    /// The largest distance any edge sits from `target`, for the settle test.
    fn max_edge_gap(self, target: Self) -> f32 {
        (self.left - target.left)
            .abs()
            .max((self.top - target.top).abs())
            .max((self.right - target.right).abs())
            .max((self.bottom - target.bottom).abs())
    }

    /// The integer rectangle to draw: floor the left/top and ceil the right/bottom, so
    /// the outward rounding never drops the target pixel and yields exactly the caret
    /// bar at rest. Pre-scroll; the painter shifts it by scroll at draw.
    fn to_rect(self) -> Rect {
        let left = self.left.floor() as i32;
        let top = self.top.floor() as i32;
        let right = self.right.ceil() as i32;
        let bottom = self.bottom.ceil() as i32;
        Rect {
            x: left,
            y: top,
            w: (right - left).max(0),
            h: (bottom - top).max(0),
        }
    }
}

/// What the caret builder observes this frame, for the animation to classify the
/// move. The smear plays for any keyboard-driven move (typing, Enter, navigation);
/// a pointer-driven move, a live-preview reveal or reflow (the target shifts while
/// the logical offset stays put), a scroll, and an over-cap jump all snap.
pub(crate) struct CaretFrame {
    /// The target caret bar's content-space top-left.
    pub(crate) target: (f32, f32),
    /// The target line height and device caret width, sizing the target box.
    pub(crate) height: f32,
    pub(crate) width: i32,
    /// The logical cursor offset. A changed offset is a real caret move (a keystroke,
    /// Enter, or a navigation); an unchanged offset with a moved target is a reflow or
    /// scroll, so the caret snaps rather than smears.
    pub(crate) cursor: usize,
    /// Whether the last input that moved the caret was a pointer event (click, drag),
    /// which snaps.
    pub(crate) from_pointer: bool,
    /// The layout geometry key; a change is a reflow (resize, DPI, column move).
    pub(crate) key: CaretGeomKey,
}

/// How long the caret stays shown, then hidden, while idle. Activity resets it to
/// shown, so it never blinks out mid-keystroke.
const CARET_BLINK: Duration = Duration::from_millis(500);
/// How many toggles a blink run lasts before the caret rests shown, about ten
/// seconds at the cadence above. Resting takes an untouched focused window to no
/// timers at all, so it stops waking the loop, and a solid caret still marks where
/// typing goes.
const CARET_BLINK_TOGGLES: u32 = 20;

/// The window's blink clock: whether the caret shows this frame, when it next
/// toggles, and how many toggles the run has left. A run is spent by the toggles
/// alone, so an untouched window ends up solid and timerless; any caret activity
/// ([`Self::bump`]) starts a fresh one.
///
/// ```text
///   bump ──▶ blinking ──20 toggles──▶ rested (shown, no deadline)
///     ▲                                   │
///     └────────── caret activity ◀────────┘
/// ```
pub(crate) struct CaretBlink {
    /// Whether the caret is drawn this frame.
    visible: bool,
    /// When the next toggle is due, or `None` while rested or hidden.
    at: Option<Instant>,
    /// Toggles left in this run; zero means rested, and nothing is armed.
    left: u32,
}

impl CaretBlink {
    pub(crate) fn new() -> Self {
        Self {
            visible: true,
            at: None,
            left: CARET_BLINK_TOGGLES,
        }
    }

    /// Whether the caret is drawn this frame.
    pub(crate) fn visible(&self) -> bool {
        self.visible
    }

    /// When the loop must come back to toggle, or `None` when nothing is armed (the
    /// run is spent, or there is no caret to blink).
    pub(crate) fn wake_at(&self) -> Option<Instant> {
        self.at
    }

    /// Activity moved the caret: show it and start a fresh run, so it is visible at
    /// once rather than caught mid-blink.
    pub(crate) fn bump(&mut self, now: Instant) {
        self.visible = true;
        self.at = Some(now + CARET_BLINK);
        self.left = CARET_BLINK_TOGGLES;
    }

    /// The window lost keyboard focus: no caret, no timer.
    pub(crate) fn hide(&mut self) {
        self.visible = false;
        self.at = None;
        self.left = 0;
    }

    /// Nothing draws the caret right now (focus is on the tree or a diff tab, or the
    /// cursor scrolled out of the painted band): rest it shown and armed for nothing,
    /// so no frame is rebuilt for a caret that is not there and it comes back solid.
    pub(crate) fn rest(&mut self) {
        self.visible = true;
        self.at = None;
        self.left = 0;
    }

    /// Advance the clock, returning whether the caret's visibility changed (the only
    /// reason a blink repaints). `in_flight` is the smooth caret still easing, which
    /// holds the caret shown and re-runs the cycle, so it never blinks out mid-motion
    /// and gets a full run once it settles.
    pub(crate) fn tick(&mut self, now: Instant, in_flight: bool) -> bool {
        if in_flight {
            let changed = !self.visible;
            self.bump(now);
            return changed;
        }
        if self.left == 0 {
            return false;
        }
        match self.at {
            // A run with no deadline yet: the first tick only arms it.
            None => {
                self.at = Some(now + CARET_BLINK);
                false
            }
            Some(at) if at <= now => {
                self.left -= 1;
                let was = self.visible;
                if self.left == 0 {
                    // The run is spent: rest shown, so the caret settles solid rather
                    // than on whichever half the last toggle landed.
                    self.visible = true;
                    self.at = None;
                } else {
                    self.visible = !self.visible;
                    self.at = Some(now + CARET_BLINK);
                }
                self.visible != was
            }
            Some(_) => false,
        }
    }
}

/// The smooth-caret animation for one document: the rendered caret box whose edges
/// ease toward its live target, plus the bookkeeping the app loop reads to keep
/// feeding it frames until it settles. The edge leading the direction of travel
/// outruns the trailing edge, so the box stretches the way the caret is going and
/// collapses back to a bar as it arrives. Boxes are content space (pre-scroll); the
/// painter shifts by scroll at draw. The smear plays for keyboard-driven moves (and
/// only when [`CaretConfig::animate`] is on); pointer moves and reflows snap.
///
/// ```text
///   settled ──keyboard move──▶ in flight ──ease each frame──▶ within 0.5px ──▶ settled
///     ▲                           │                                             │
///     └──── snap (off / pointer / reflow / scroll / requested / jump>cap) ◀──────┘
/// ```
pub(crate) struct CaretAnim {
    /// The rendered caret box, or `None` until the first update seeds it from the live
    /// target.
    pos: Option<CaretBox>,
    /// The geometry the last target was laid out under, to detect a reflow.
    key: Option<CaretGeomKey>,
    /// The logical cursor offset the last update saw, to tell a real caret move (offset
    /// moved) from a reflow or scroll (offset unchanged).
    last_cursor: Option<usize>,
    /// When the last decay step ran, or `None` while settled.
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
            last_cursor: None,
            last_tick: None,
            in_flight: false,
            snap_next: false,
        }
    }

    /// Advance the rendered caret one frame from the observed `frame`, at `now`. Any
    /// keyboard-driven move (the logical offset changed and it was not pointer-driven)
    /// eases each edge with frame-rate-independent exponential decay (the leading edge
    /// at `rate`, the trailing edge at `rate * spring`, so the box stretches along its
    /// path) and settles within half a pixel. Everything else snaps: animation off, a
    /// requested snap, the first frame, a geometry reflow, a pointer move, a reveal or
    /// scroll that shifts the target without moving the offset, or a jump beyond the
    /// configured cap.
    pub(crate) fn update(&mut self, frame: CaretFrame, config: &CaretConfig, now: Instant) {
        let CaretFrame {
            target,
            height,
            width,
            cursor,
            from_pointer,
            key,
        } = frame;
        let target_box = CaretBox::bar(target.0, target.1, width as f32, height);
        let reflowed = self.key != Some(key);
        let moved = self.last_cursor.is_some_and(|c| c != cursor);
        self.key = Some(key);
        self.last_cursor = Some(cursor);

        // Hard snap: no smear ever. Animation off, a requested snap (focus/tab/open),
        // or a geometry reflow (resize, DPI, column move).
        if !config.animate || self.snap_next || reflowed {
            self.snap_to(target_box);
            return;
        }
        // The first frame has no rendered box to ease from, so it seeds at the target.
        let Some(cur) = self.pos else {
            self.snap_to(target_box);
            return;
        };

        if self.in_flight {
            // Mid-smear. A pointer move cuts it short and snaps to where it pointed; a
            // fresh keyboard move just re-aims the destination; otherwise keep easing
            // toward the target (which a reveal reflow may have nudged).
            if moved && from_pointer {
                self.snap_to(target_box);
            } else {
                self.ease(cur, target_box, config, now);
            }
            return;
        }

        // Settled. A keyboard-driven move within the jump cap starts a smear; a
        // pointer move, a reveal or scroll (offset unchanged), and an over-cap jump
        // snap.
        let keyboard_move = moved && !from_pointer;
        let within_cap =
            caret_distance((cur.left, cur.top), target) <= config.max_jump_lines * height;
        if keyboard_move && within_cap {
            self.ease(cur, target_box, config, now);
        } else {
            self.snap_to(target_box);
        }
    }

    /// Jump to `target` and settle: no smear, flight and timing cleared.
    fn snap_to(&mut self, target: CaretBox) {
        self.pos = Some(target);
        self.in_flight = false;
        self.last_tick = None;
        self.snap_next = false;
    }

    /// Ease one step from `cur` toward `target`. The first step after a settle uses a
    /// nominal frame time; steps already in flight use the real elapsed gap (capped).
    /// This starts the smear from the caret even after a long idle, rather than
    /// spending a full capped step. Settles exactly within half a pixel.
    fn ease(&mut self, cur: CaretBox, target: CaretBox, config: &CaretConfig, now: Instant) {
        if cur == target {
            self.snap_to(target);
            return;
        }
        let dt = if self.in_flight {
            self.last_tick
                .map_or(CARET_DT_CAP, |t| now.saturating_duration_since(t))
                .min(CARET_DT_CAP)
        } else {
            CARET_FIRST_STEP
        };
        self.in_flight = true;
        self.last_tick = Some(now);
        let secs = dt.as_secs_f32();
        // The leading edge decays at `rate`, the trailing edge at `rate * spring` (the
        // stretch). The vertical axis runs faster so a line change drops in quickly.
        let axis = |rate: f32| AxisEase {
            lead: 1.0 - (-rate * secs).exp(),
            trail: 1.0 - (-rate * config.spring * secs).exp(),
        };
        let next = cur.ease_toward(
            target,
            axis(config.rate),
            axis(config.rate * CARET_VERTICAL_BOOST),
        );
        if next.max_edge_gap(target) < CARET_SETTLE_PX {
            self.snap_to(target);
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

    /// The content-space rectangle to draw this frame: the current caret box, outward-
    /// rounded to integer pixels. At rest it is exactly the caret bar; in flight it is
    /// the stretched smear. Pre-scroll; the painter shifts it by scroll at draw.
    pub(crate) fn draw_rect(&self) -> Rect {
        self.pos.map_or(
            Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
            CaretBox::to_rect,
        )
    }
}

/// Read `path` into buffer text and its on-disk [`FileFormat`]. A path that does
/// not exist yet is not an error: it opens an empty buffer that a later save
/// creates, so `bnknotes new-note.md` works.
fn load(path: &Path) -> Result<(String, FileFormat)> {
    // Reject an oversized file up front (by its stat size) rather than reading it
    // all into memory. A missing file is not an error here, so a stat failure
    // falls through to the read, which maps NotFound to an empty buffer.
    if let Ok(meta) = std::fs::metadata(path)
        && meta.len() > MAX_FILE_BYTES
    {
        return Err(Error::msg(format!(
            "file is {} bytes; refusing to open files larger than {} MiB",
            meta.len(),
            MAX_FILE_BYTES / (1024 * 1024)
        )));
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
/// size), the active range's logical line span (a heading or fence reveal depends
/// only on which lines the range touches, not the exact byte offsets), whether
/// each active end reaches a bullet/task marker (the one reveal that turns on the
/// column, not just the line, so it rides the key on its own), the selected /
/// raw-pinned image (by line-start offset), whether a drag is in progress (which
/// holds image sources and list markers collapsed), and the fold revision (folding
/// removes lines from the build). A scroll, idle redraw, or cursor move within one
/// line that does not cross a marker boundary reuses the cached layout.
type LayoutKey = (
    u64,
    i32,
    i32,
    u32,
    u32,
    usize,
    usize,
    ((bool, bool), (bool, bool)),
    Option<usize>,
    Option<usize>,
    bool,
    u64,
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Focus {
    Tree,
    /// Content pane 0 (left/A) or 1 (right/B).
    Editor(usize),
}

/// Width of the grab bar between two columns: a few pixels of hit zone, with a
/// thin divider line drawn down its middle.
pub(crate) const SPLITTER: i32 = 6;

/// The visible divider inside a splitter's grab bar: one device pixel down its
/// middle, spanning the bar's height. The rest of the bar is hit zone.
pub(crate) fn divider_line(splitter: Rect, scale: Scale) -> Rect {
    let w = scale.px(1).max(1);
    Rect {
        x: splitter.x + (splitter.w - w) / 2,
        w,
        ..splitter
    }
}

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
    /// The vault rail, the outermost column and the only full-height one: the top
    /// bar runs beside it, not over it, so it is never `None` and never covered.
    pub(crate) rail: Rect,
    /// The rail's chip region: the rail below the bar's height, so the chips line up
    /// with the tree rows and the pane text rather than with the tabs.
    pub(crate) rail_chips: Rect,
    /// The top bar's strip, from the rail's right edge to the window's. Empty
    /// (`h == 0`) when no bar is reserved.
    pub(crate) bar: Rect,
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
    ///
    /// The vault rail is the exception to the bar: it is fixed width, always shown,
    /// and full height, so the bar starts at its right edge and everything else
    /// follows from there. Its chips still keep the bar's line: they sit in
    /// `rail_chips`, the rail below `top`, so they start where the columns do.
    ///
    /// ```text
    ///   0    rail                                        width
    ///   ├─────┼─────────────┬───┬───────────┬───┬──────────┤
    ///   │     │    vault    │   │  strip A  │   │ strip B  │  ← bar (h = top)
    ///   │rail ├─────────────┤   ├───────────┤   ├──────────┤
    ///   │chips│    tree     │ s │ editor A  │ s │ editor B │
    ///   └─────┴─────────────┴───┴───────────┴───┴──────────┘
    /// ```
    /// When `solo` is `Some(i)`, editor `i` takes over the whole content area: the
    /// sidebar and the other pane are hidden and no splitters are drawn, so a diff
    /// review gets the full width to read across. Everything else (the rail, that pane's
    /// tab strip) stays. `None` is the ordinary layout below.
    pub(crate) fn columns(
        &self,
        width: i32,
        height: i32,
        top: i32,
        scale: Scale,
        solo: Option<usize>,
    ) -> Columns {
        let splitter = scale.px(SPLITTER);
        let tree_width = scale.px(self.tree_width);
        let body = (height - top).max(0);
        // Never wider than the window, so a window squeezed narrower than the rail
        // leaves the columns beside it empty rather than negative.
        let rail = Rect {
            x: 0,
            y: 0,
            w: scale.px(RAIL_W).min(width.max(0)),
            h: height,
        };
        let rail_chips = Rect {
            y: top,
            h: body,
            ..rail
        };
        let bar = Rect {
            x: rail.w,
            y: 0,
            w: (width - rail.w).max(0),
            h: top,
        };
        if let Some(i) = solo.filter(|&i| i < 2) {
            // A soloed pane takes the whole content area beside the sidebar: the other
            // pane and the pane splitter fall away, but the file tree stays (it is kept
            // visible during a diff review), so the pane spans from the tree's right edge
            // to the window's.
            let mut x = rail.w;
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
            let editor = Rect {
                x,
                y: top,
                w: (width - x).max(0),
                h: body,
            };
            // The tab strip reaches left across the tree splitter's grab margin so its
            // first tab sits flush against the divider, matching the ordinary layout.
            let strip_x = splitters[0].map_or(x, |s| {
                let line = divider_line(s, scale);
                line.x + line.w
            });
            let strip = Rect {
                x: strip_x,
                y: 0,
                w: (width - strip_x).max(0),
                h: top,
            };
            let mut editors = [None, None];
            let mut tab_strips = [None, None];
            editors[i] = Some(editor);
            tab_strips[i] = Some(strip);
            return Columns {
                rail,
                rail_chips,
                bar,
                tree,
                editors,
                splitters,
                tab_strips,
            };
        }
        let mut x = rail.w;
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
        // Each pane's tab strip takes its column's x-span at the top strip's height,
        // reaching left across the splitter's grab margin so the first tab sits flush
        // against the divider line that continues up through the bar.
        let strip = |col: Option<Rect>, left: Option<Rect>| {
            col.map(|c| {
                let x = left.map_or(c.x, |s| {
                    let line = divider_line(s, scale);
                    line.x + line.w
                });
                Rect {
                    x,
                    y: 0,
                    w: c.w + (c.x - x),
                    h: top,
                }
            })
        };
        let tab_strips = [
            strip(editors[0], splitters[0]),
            strip(editors[1], splitters[1]),
        ];
        Columns {
            rail,
            rail_chips,
            bar,
            tree,
            editors,
            splitters,
            tab_strips,
        }
    }
}

/// What a window position lands on, for routing pointer events to the right
/// region. The vault rail is checked first (it is full height, so it wins beside
/// the bar as well as beside the columns), then the top bar (it sits above every
/// remaining column), then splitters (they sit in the gaps between columns).
pub(crate) enum Hit {
    /// A row of the open palette, by its position in the palette's matches: go there.
    PaletteRow(usize),
    /// The open palette's panel, off its rows: the press belongs to the field, so it
    /// consumes the click and leaves the palette up.
    Palette,
    /// Anywhere outside the open palette: the press puts it away and goes no further,
    /// the way a press off the context menu dismisses that.
    PaletteOutside,
    /// An item of the open context menu: run it.
    MenuItem(usize),
    /// Anywhere else while a context menu is open: the press dismisses the menu and
    /// goes no further, the way a pop-up's grab swallows the click that closes it.
    Menu,
    /// Anywhere while a picture is open over the whole window: the press puts it away
    /// and goes no further.
    ImageView,
    /// The rail's chip for vault `i`: switch the window to it.
    Vault(usize),
    /// The rest of the vault rail; consumes the click, so a miss between chips does
    /// not fall through to the column beside it.
    Rail,
    Splitter(usize),
    /// A row of the sidebar's file tree, carrying the rows' viewport (the column
    /// above the footer), which is what a row index and the scroll clamp measure
    /// against.
    Tree(Rect),
    /// The sidebar footer's folder-filter toggle: list, or hide, the folders with no
    /// notes in them.
    TreeFilter,
    /// The rest of the sidebar footer; consumes the click so it cannot fall through
    /// to a tree row scrolled beneath the strip.
    TreeFooter,
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

/// A live pointer drag-selection: what the press took hold of, and how a sweep
/// from there grows the selection.
#[derive(Clone, Copy)]
struct DragSelect {
    /// The byte range the press took as a unit: the caret for a plain click, the
    /// word or line for a double/triple-click, the whole line for a press on a
    /// rendered horizontal rule or picture. A sweep grows out of this range and
    /// never cuts into it, so a jitter during the click cannot break the press's
    /// own selection apart.
    anchor: (usize, usize),
    /// The unit the sweep extends by, from the press that started it.
    grain: DragGrain,
    /// The caret's place on the picture the press selected, if it landed on one
    /// (its line's start or end). The picture stays selected while the pointer
    /// stays within the anchor and gives way to a text selection once the sweep
    /// leaves the image's line.
    image: Option<usize>,
}

/// The side of a selected picture the caret holds, which decides where typing
/// goes: Above is the picture line's start, Below its end.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PictureSide {
    Above,
    Below,
}

/// The four delete keys: Backspace and Delete, by grapheme or by word.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DeleteKey {
    Back,
    Forward,
    WordBack,
    WordForward,
}

impl DragSelect {
    /// A drag over text: no picture is selected under it.
    fn text(anchor: (usize, usize), grain: DragGrain) -> Self {
        Self {
            anchor,
            grain,
            image: None,
        }
    }

    /// A drag off the picture on the logical line start..end, with the caret at
    /// image on it.
    fn image(image: usize, (start, end): (usize, usize)) -> Self {
        Self {
            anchor: (start, end),
            grain: DragGrain::Char,
            image: Some(image),
        }
    }

    /// Whether `offset` is still within what the press took hold of.
    fn holds(&self, offset: usize) -> bool {
        (self.anchor.0..=self.anchor.1).contains(&offset)
    }
}

/// One content pane: an independent editor over its own buffer, carrying its own
/// file, inline images, memoized layout, scroll, find bar, and the drag/selection
/// transients a click-drag needs. Two of these live on the app's State.
/// How a document's file on disk has diverged from the buffer, once the editor
/// notices a change made outside it that it cannot silently absorb. `None` while
/// the buffer and the file agree (or the buffer was cleanly reloaded).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum DiskConflict {
    /// The file changed on disk while the buffer held unsaved edits. Reloading
    /// would lose those edits and saving would clobber the other writer, so the
    /// choice is the user's.
    Changed,
    /// The file the buffer tracks was deleted or moved away on disk.
    Deleted,
}

/// What a press landed on in a picture's neighborhood, for a caller that has to
/// treat the picture and its raw-source button differently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ImagePress {
    /// Neither: the press is the text's to deal with. In the space around a
    /// picture that puts the caret on the picture, selecting it.
    Missed,
    /// The corner button that pins the picture's raw source open. Toggled.
    Button,
    /// The picture itself. Selected, with a drag anchored on its line.
    Picture,
}

/// What a call to [`EditorPane::save`] came to. A save is asked for by a keystroke,
/// by the idle autosave, and by everything that lets a document go (a tab closing,
/// a vault switching, the window shutting), so most of these are the ordinary
/// answers to "save this" rather than failures: only the caller knows which are
/// worth a word to the user.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Saved {
    /// The buffer is on disk now.
    Written,
    /// Nothing to write: the buffer has not changed since it was last saved.
    Unchanged,
    /// The buffer has no file to write to (a blank tab), so it stays put.
    NoPath,
    /// The file changed on disk since the buffer last agreed with it, so it was
    /// left alone and a conflict was raised for the user.
    Conflict,
}

/// A process-unique handle to one open document, stamped by its [`PaneGroup`] when the
/// tab is created and kept as the tab moves, reorders, or crosses the split. A view
/// that is not itself the document (a diff tab) names its source this way, so an
/// accept finds the right buffer wherever its tab now sits, or learns it was closed.
/// The two panes mint from distinct residues (see [`PaneGroup::mint_id`]) so their
/// ids never collide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DocId(u64);

impl DocId {
    /// A document not yet owned by a pane: a freshly built [`EditorPane`] before it
    /// is stamped, and the never-drawn placeholder. A diff view never targets one, so
    /// it reads as a safe non-match.
    const UNSTAMPED: DocId = DocId(u64::MAX);

    /// A fixed id for a test that builds a view in isolation, without a pane to mint one.
    #[cfg(test)]
    pub(crate) fn from_raw(n: u64) -> DocId {
        DocId(n)
    }
}

pub(crate) struct EditorPane {
    /// This document's stable identity, stamped by the [`PaneGroup`] that opens it so
    /// a diff view can name the buffer it applies back to. [`DocId::UNSTAMPED`] until a
    /// pane takes ownership.
    pub(crate) id: DocId,
    pub(crate) text: TextBuffer,
    pub(crate) path: Option<PathBuf>,
    pub(crate) format: FileFormat,
    /// The file's on-disk identity the last time the buffer and the file agreed
    /// (at open, after a save, or after a reload), against which an external
    /// change is measured. `None` for a buffer whose file does not exist yet.
    pub(crate) disk: Option<file::DiskSnapshot>,
    /// An unresolved external change awaiting the user's decision; drawn as a
    /// banner over the pane. `None` when the buffer and the file are in sync.
    pub(crate) conflict: Option<DiskConflict>,
    pub(crate) images: ImageStore,
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
    /// The headings this document has folded, persisted with the tab.
    pub(crate) folds: Folds,
    /// What those folds hide, resolved against the current index. Rebuilt by
    /// [`Self::ensure_index`] whenever `fold_map_key` goes stale.
    fold_map: FoldMap,
    /// The content version and fold revision `fold_map` was resolved at.
    fold_map_key: Option<(u64, u64)>,
    pub(crate) goal_x: Option<f32>,
    /// The live pointer drag-selection, or `None` when no drag is in flight.
    drag: Option<DragSelect>,
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
    /// set by the app's State and refreshed on a DPI change. At [`Scale::ONE`] the body is
    /// [`FONT_SIZE`] and every geometry constant is its authored value.
    scale: Scale,
    body_size: u32,
    /// The buffer version the autosave last looked at, for [`Self::poll_edited`].
    autosave_seen: u64,
}

impl EditorPane {
    /// An empty, file-less pane (blank buffer, no images).
    pub(crate) fn empty() -> Self {
        Self {
            id: DocId::UNSTAMPED,
            text: TextBuffer::new(),
            path: None,
            format: FileFormat::default(),
            disk: None,
            conflict: None,
            images: ImageStore::new(None),
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
            folds: Folds::default(),
            fold_map: FoldMap::default(),
            fold_map_key: None,
            goal_x: None,
            drag: None,
            clicks: ClickChain::new(),
            autoscroll: 0,
            autoscroll_at: None,
            caret_anim: CaretAnim::new(),
            link_scratch: markdown::InlineScratch::default(),
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            autosave_seen: 0,
        }
    }

    /// Adopt a new display scale and device-pixel body size (a DPI change), so the
    /// next layout re-measures at the new size. The memoization keys carry both, so
    /// simply setting them invalidates the cached index and window.
    pub(crate) fn set_scale(&mut self, scale: Scale, body_size: u32) {
        self.scale = scale;
        self.body_size = body_size;
    }

    /// Point this pane's image store at the vault root, so a wikilink embed
    /// `![[name]]` resolves its picture against the vault and its attachment folders
    /// as well as the note's own directory.
    pub(crate) fn set_vault_root(&mut self, root: Option<&Path>) {
        self.images.set_vault(root.map(Path::to_path_buf));
    }

    /// Load `path` into a fresh pane, resolving inline images against its
    /// directory. An unreadable file is an error the caller surfaces.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let (contents, format) = load(path)?;
        let image_dir = path.parent().map(Path::to_path_buf);
        let mut doc = Self {
            text: TextBuffer::from_text(contents),
            path: Some(path.to_path_buf()),
            format,
            disk: file::DiskSnapshot::of(path),
            images: ImageStore::new(image_dir),
            ..Self::empty()
        };
        doc.park_past_frontmatter();
        Ok(doc)
    }

    /// Move a caret sitting in the frontmatter block down to the note's first
    /// content line, so opening a note shows the note. The block hides while the
    /// caret is outside it, and a caret at the document start (a fresh open, or a
    /// session that saved one there) would be inside and hold it open.
    ///
    /// Every caret inside the block parks, not just one at the start: a saved caret
    /// is not a deliberate gesture the way a fold is, and a note that always reopens
    /// on its content beats one that sometimes reopens on its metadata.
    ///
    /// A document that is nothing but frontmatter has no content line to park on, so
    /// the caret stays inside and the block shows: there is nothing else to see.
    fn park_past_frontmatter(&mut self) {
        let rope = self.text.rope();
        let Some(end) = frontmatter::extent(rope) else {
            return;
        };
        if rope.byte_to_line(self.text.cursor()) >= end {
            return;
        }
        let target = rope.line_to_byte(end);
        self.text.set_cursor(target, false);
    }

    /// Whether the buffer changed since this was last asked, latching the version it
    /// reports on. The autosave's idle window re-arms on this rather than on each
    /// edit's own call site: it watches the buffer itself, so a typed character, a
    /// paste, a replace-all, and a table edit all count without every one of them
    /// having to remember to say so.
    pub(crate) fn poll_edited(&mut self) -> bool {
        let version = self.text.version();
        let edited = version != self.autosave_seen;
        self.autosave_seen = version;
        edited
    }

    /// Write the buffer to its backing file, atomically and in the file's original
    /// on-disk format, and report what that came to. Saving is the document's own
    /// business (the text, the path, the format, and the disk snapshot are all
    /// here), so a caller only has to decide which documents to ask and what to say
    /// about the answer.
    ///
    /// A file that changed on disk since this buffer last agreed with it is not
    /// overwritten: another writer's version is sitting there, so this raises a
    /// conflict for the user instead. A missing file is safe to recreate, and an
    /// unchanged snapshot is our own last save.
    pub(crate) fn save(&mut self) -> Result<Saved> {
        let Some(path) = self.path.clone() else {
            return Ok(Saved::NoPath);
        };
        if !self.text.is_modified() {
            return Ok(Saved::Unchanged);
        }
        let current = file::DiskSnapshot::of(&path);
        if current.is_some() && current != self.disk {
            self.conflict = Some(DiskConflict::Changed);
            return Ok(Saved::Conflict);
        }
        let bytes = file::encode(&self.text.text(), &self.format);
        file::save_atomic(&path, &bytes)?;
        self.text.mark_saved();
        // Record the just-written file as the new synced state, so its own change
        // events (and any conflict) clear rather than bounce back as external.
        self.sync_disk();
        Ok(Saved::Written)
    }

    /// Record the file's current on-disk identity as the last-synced state and
    /// clear any conflict, so a later external change is measured against this
    /// point. Called after a save writes the buffer out, and when the user keeps
    /// their version in a conflict (accepting that the next save overwrites).
    pub(crate) fn sync_disk(&mut self) {
        self.disk = self.path.as_deref().and_then(file::DiskSnapshot::of);
        self.conflict = None;
    }

    /// Replace the buffer with the file's current on-disk contents, re-syncing the
    /// change-tracking snapshot and clearing any conflict. Used when the file
    /// changed outside the editor and the buffer's edits are being discarded (or
    /// there were none). A file-less buffer has nothing to reload.
    pub(crate) fn reload_from_disk(&mut self) -> Result<()> {
        let Some(path) = self.path.clone() else {
            return Ok(());
        };
        let (contents, format) = load(&path)?;
        self.format = format;
        // A fresh version invalidates the layout and index caches (their keys
        // carry it), so the next frame re-measures against the new contents.
        self.text.reload(contents);
        self.disk = file::DiskSnapshot::of(&path);
        self.conflict = None;
        self.scan_images();
        self.request_caret_snap();
        Ok(())
    }

    /// Compare the file on disk against the last-synced snapshot and react to any
    /// change made outside the editor: reload silently when the buffer is clean,
    /// raise a conflict for the user when it holds unsaved edits, or flag a delete
    /// when the file is gone. Returns whether the document changed, so the caller
    /// repaints. A file-less buffer, an in-sync file, or one already in conflict is
    /// left untouched.
    pub(crate) fn recheck_disk(&mut self) -> bool {
        let Some(path) = self.path.clone() else {
            return false;
        };
        // A conflict is already parked for the user; don't re-raise it each event.
        if self.conflict.is_some() {
            return false;
        }
        let current = file::DiskSnapshot::of(&path);
        if current == self.disk {
            return false;
        }
        if current.is_none() {
            self.conflict = Some(DiskConflict::Deleted);
            return true;
        }
        if self.text.is_modified() {
            self.conflict = Some(DiskConflict::Changed);
        } else if self.reload_from_disk().is_err() {
            // The file changed but a re-read failed (a transient rename in
            // flight, a permission flip): surface it rather than sit out of sync.
            self.conflict = Some(DiskConflict::Changed);
        }
        true
    }

    /// The tab's label: the file's name without its .md suffix (matching the
    /// tree), or "untitled" for a file-less scratch buffer.
    fn tab_name(&self) -> &str {
        self.path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|s| s.to_str())
            .unwrap_or("untitled")
    }

    /// Whether this is a fresh, untouched scratch tab (no file, empty buffer), so
    /// opening a note into a pane showing one replaces it rather than stacking a
    /// blank tab beside it.
    fn is_blank(&self) -> bool {
        self.path.is_none() && self.text.rope().len() == 0
    }

    /// The active range for live-preview marker reveal: the live selection, or the
    /// bare cursor when there is none. Markers reveal even mid-drag, so a selection
    /// shows the raw source it grabs (the ``` of a fenced block, a heading's `#`)
    /// as it sweeps over it, rather than dragging across blank collapsed glyphs.
    /// Revealing a text marker never reflows the line, so this is safe under the
    /// pointer; image sources, which would reflow, are held collapsed instead (see
    /// [`Self::image_ui`] and [`Self::reveal_adjust`]).
    pub(crate) fn reveal_range(&self) -> (usize, usize) {
        let cursor = self.text.cursor();
        self.text.selection().unwrap_or((cursor, cursor))
    }

    /// Whether each end of the active range reveals a bullet or task line's raw
    /// marker: the finer live-preview test lists use, where a heading reveals its
    /// `#` across the whole line but a list gives way to its `- `/`- [ ] ` only
    /// where the range reaches the marker. Folded into the layout cache key, since
    /// moving the caret across a list's marker boundary flips the reveal without
    /// changing which lines the range touches, which the coarse line span misses.
    /// Held false throughout a drag, when every marker stays collapsed until
    /// release (matching the layout's `dragging` gate).
    fn marker_reveal(&self, active: (usize, usize)) -> (bool, bool) {
        if self.dragging() {
            return (false, false);
        }
        let rope = self.text.rope();
        let reaches = |off: usize| {
            let (ls, le) = rope.line_content_range(rope.byte_to_line(off));
            let (style, marker_len) = markdown::classify(&rope.slice(ls..le).as_cow());
            matches!(
                style,
                markdown::LineStyle::Bullet | markdown::LineStyle::Task { .. }
            ) && crate::layout::active_reveals_marker(active, ls, marker_len, false)
        };
        (reaches(active.0), reaches(active.1))
    }

    /// Whether each end of the active range sits inside the source of a picture,
    /// which reveals that source; the edges of a picture's line are where a caret
    /// on the picture sits, and reveal nothing. Folded into the layout cache key, since
    /// stepping between a picture's edge and its inside flips the reveal without
    /// changing which lines the range touches. Reads the index for which lines are
    /// pictures, so a caret leaving a text line's start rebuilds nothing.
    fn image_reveal(&self, active: (usize, usize)) -> (bool, bool) {
        let Some(index) = self.line_index.as_ref() else {
            return (false, false);
        };
        let rope = self.text.rope();
        let inside = |off: usize| {
            let line = rope.byte_to_line(off);
            let (ls, le) = rope.line_content_range(line);
            index.image_extra(line) > 0
                && crate::layout::active_reveals_image_source((off, off), ls, le)
        };
        (inside(active.0), inside(active.1))
    }

    /// The per-image UI state (selected border, pinned raw source, drag-freeze) for
    /// the layout.
    pub(crate) fn image_ui(&self) -> ImageUi {
        ImageUi {
            selected: self.selected_picture(),
            raw: self.raw_image,
            dragging: self.dragging(),
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

    /// The overlay the index's geometry queries run under: the image sources
    /// revealed right now, over the sections currently folded away. Both are cheap
    /// and empty in the common case.
    fn adjust(&self) -> Adjust<'_> {
        Adjust::new(self.reveal_adjust(), &self.fold_map)
    }

    /// What the index's geometry queries must add for the current reveal: the lines
    /// the cursor or selection touches (whose markup draws raw, so their rows wrap
    /// wider), and the image sources shown (one an end of the range sits inside, and
    /// the pinned one). A source inside a folded section is skipped: it is not on
    /// screen to reveal.
    fn reveal_adjust(&self) -> RevealAdjust {
        let mut adjust = RevealAdjust::default();
        let Some(index) = self.line_index.as_ref() else {
            return adjust;
        };
        let rope = self.text.rope();
        let (lo, hi) = self.reveal_range();
        // Every line the range touches shows its markup, drag or not, so every one
        // of them re-wraps.
        adjust.reveal_lines(rope.byte_to_line(lo), rope.byte_to_line(hi));
        // A drag holds every image source collapsed so the geometry under the
        // pointer stays put (matching the window build's image_ui.dragging); only a
        // corner-pinned source still expands. Off a drag an end of the range inside
        // an image source expands it, as the window build's reveal test does.
        let mut expand = |line: usize| {
            if !self.fold_map.hides(line) {
                adjust.push(line, index.image_extra(line));
            }
        };
        if !self.dragging() {
            let (lo_inside, hi_inside) = self.image_reveal((lo, hi));
            if lo_inside {
                expand(rope.byte_to_line(lo));
            }
            if hi_inside {
                expand(rope.byte_to_line(hi));
            }
        }
        if let Some(raw) = self.raw_image {
            expand(rope.byte_to_line(raw));
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
            // Shift the fold anchors by whatever edit happened since the last index,
            // so a fold rides its heading as text above it grows or shrinks. This
            // runs before either rebuild path below: a resize batched with an edit
            // takes the full rebuild yet still moves headings, so remapping only on
            // the incremental path would drop those folds.
            if let Some(prev) = self.prev_rope.as_ref() {
                self.folds.remap(prev, self.text.rope());
            }
            // When only the content changed (same column, font, and scale), re-measure
            // just the edited region against the previous rope; otherwise (first build,
            // or a resize) rebuild in full.
            let geometry_same = matches!(
                self.index_key,
                Some((_, x, w, fsize, f120))
                    if x == rect.x && w == rect.w && fsize == self.body_size
                        && f120 == self.scale.factor_120()
            );
            let incremental = geometry_same
                .then(|| self.line_index.take().zip(self.prev_rope.take()))
                .flatten();
            if let Some((mut index, prev)) = incremental {
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
        self.ensure_fold_map();
        let adjust = self.adjust();
        if let Some(index) = self.line_index.as_ref() {
            self.content_height = index.content_height(adjust) + 2 * margin;
        }
    }

    /// Re-resolve what the document hides when the content, the fold set, or the
    /// caret has moved under them, dropping any anchor whose heading is gone. Cheap
    /// to check and, with nothing hidden, cheap to do.
    ///
    /// The frontmatter block's state is settled here first, because it is the caret
    /// that decides it: the block shows while the caret is inside it and hides again
    /// when it leaves. Writing it into [`Folds`] moves the revision, so the fold map
    /// and the windowed layout both rebuild on the same key they already watch.
    fn ensure_fold_map(&mut self) {
        let Some(index) = self.line_index.as_ref() else {
            return;
        };
        let cursor_line = self.text.rope().byte_to_line(self.text.cursor());
        let in_frontmatter = index.frontmatter().is_some_and(|end| cursor_line < end);
        self.folds.set_frontmatter_open(in_frontmatter);
        let key = (self.text.version(), self.folds.revision());
        if self.fold_map_key == Some(key) {
            return;
        }
        self.folds.retain_headings(index);
        self.fold_map = FoldMap::build(&self.folds, index);
        // `retain_headings` can bump the revision, so key off where it landed.
        self.fold_map_key = Some((self.text.version(), self.folds.revision()));
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
            &self.fold_map,
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
        // A heading or fence reveal depends only on which lines the active range
        // touches, so the key coarsens those to lines. A bullet or task reveal is
        // finer (it turns on where the range reaches the marker, not just the line),
        // so its two-ended state rides the key on its own, and so does a picture's,
        // which turns on whether an end sits inside its source or at its edges. The
        // live drag rides the key because it holds image sources and list markers
        // collapsed, so releasing a drag must rebuild to let those reveal.
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
            (self.marker_reveal(active), self.image_reveal(active)),
            self.selected_picture(),
            self.raw_image,
            self.dragging(),
            self.folds.revision(),
        );
        let adjust = self.adjust();
        let (vis_top, vis_bot, over, n) = {
            let Some(index) = self.line_index.as_ref() else {
                return;
            };
            (
                index.line_at_y(self.scroll_y, adjust),
                index.line_at_y(self.scroll_y + rect.h.max(0), adjust),
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
            let Some(index) = self.line_index.as_ref() else {
                return;
            };
            let seed = rect.y + margin + index.logical_top(start, adjust);
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
    ///
    /// `None` when the index is unavailable, and when a fold hides the cursor's
    /// line: it has no caret box to answer with, and a section deeper than the
    /// window would leave nothing laid out to ask. Callers that move or reveal the
    /// caret restore it to a visible line first (see
    /// [`Self::open_folds_at_cursor`]), so this is the floor under them, not a path
    /// the editor works from.
    fn cursor_window(&self, fonts: &Fonts, rect: Rect) -> Option<Layout> {
        let index = self.line_index.as_ref()?;
        let adjust = self.adjust();
        let active = self.reveal_range();
        let cursor_line = self.text.rope().byte_to_line(self.text.cursor());
        if self.fold_map.hides(cursor_line) {
            return None;
        }
        let span = self.page_lines(fonts, rect) + 4;
        let n = index.len();
        let start = cursor_line.saturating_sub(span);
        let end = (cursor_line + span + 1).min(n).max(start);
        let margin = self.scale.px(MARGIN);
        let seed = rect.y + margin + index.logical_top(start, adjust);
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

    /// Fold or unfold the section the cursor sits in: the heading on the cursor's
    /// own line, else the nearest heading above it. Returns whether anything
    /// changed. A heading with no lines under it has no section, so nothing folds.
    pub(crate) fn toggle_fold_at_cursor(&mut self, fonts: &Fonts, rect: Rect) -> bool {
        self.ensure_index(fonts, rect);
        let Some(index) = self.line_index.as_ref() else {
            return false;
        };
        let line = self.text.rope().byte_to_line(self.text.cursor());
        let Some(heading) = fold::enclosing_heading(index, line) else {
            return false;
        };
        self.set_fold(heading)
    }

    /// Fold or unfold the heading whose logical line starts at buffer offset `ls`,
    /// the chevron's click target. Returns whether anything changed.
    pub(crate) fn toggle_fold_at(&mut self, fonts: &Fonts, rect: Rect, ls: usize) -> bool {
        self.ensure_index(fonts, rect);
        self.set_fold(self.text.rope().byte_to_line(ls))
    }

    /// Toggle the fold on heading `line` and settle the caret afterwards. The map is
    /// rebuilt immediately so the caret check sees the fold that just closed.
    fn set_fold(&mut self, line: usize) -> bool {
        if !self.folds.toggle(line) {
            return false;
        }
        self.fold_map_key = None;
        self.ensure_fold_map();
        self.lift_cursor_out_of_folds();
        true
    }

    /// Unfold every section in this document. Returns whether anything was folded.
    pub(crate) fn unfold_all(&mut self) -> bool {
        if !self.folds.clear() {
            return false;
        }
        self.fold_map_key = None;
        self.ensure_fold_map();
        true
    }

    /// Move the cursor onto the folded heading above it when the section it was in
    /// has just closed, so the caret never parks on text nobody can see. Reports
    /// whether the cursor moved.
    fn lift_cursor_out_of_folds(&mut self) -> bool {
        let rope = self.text.rope();
        let line = rope.byte_to_line(self.text.cursor());
        if !self.fold_map.hides(line) {
            return false;
        }
        let heading = self.fold_map.first_visible_at_or_before(line);
        let target = rope.line_content_range(heading).1;
        self.text.set_cursor_changed(target, false)
    }

    /// Open every fold hiding the cursor, so an edit that landed the caret inside a
    /// collapsed section puts what it wrote on screen. Pressing Enter on a folded
    /// heading is the case that matters: the line it opens is the section's own
    /// first line, so there is nowhere to put it that the fold does not hide.
    ///
    /// Each pass drops the outermost heading still hiding the cursor and re-resolves,
    /// which in turn opens a nested fold underneath it; the chain is at most one fold
    /// per heading level, and stops as soon as the cursor is visible, so a nested
    /// section the edit did not reach stays folded.
    pub(crate) fn open_folds_at_cursor(&mut self, fonts: &Fonts, rect: Rect) {
        // Only sections reach here. A frontmatter block hiding the caret cannot
        // happen: resolving what the document hides shows the block wherever the
        // caret is inside it, which is settled before this asks.
        if self.folds.is_empty() {
            return;
        }
        // The map has to answer for the text as the edit left it, anchors and all.
        self.ensure_index(fonts, rect);
        loop {
            let line = self.text.rope().byte_to_line(self.text.cursor());
            if !self.fold_map.hides(line) {
                return;
            }
            let heading = self.fold_map.first_visible_at_or_before(line);
            if !self.folds.open(heading) {
                return;
            }
            self.fold_map_key = None;
            self.ensure_fold_map();
        }
    }

    /// Step the cursor clear of a folded section it just walked into, continuing the
    /// direction it was traveling: forward lands on the first line past the fold,
    /// backward on the end of the heading that hides it. Arrowing over a folded
    /// heading therefore steps across the whole section rather than through its
    /// hidden text. Reports whether the cursor moved.
    ///
    /// A hidden frontmatter block is not stepped over: it is hidden only until the
    /// caret arrives, so arrowing back into it from the first line of the note is
    /// how you open it.
    pub(crate) fn step_over_folds(&mut self, forward: bool, extend: bool) -> bool {
        if self.fold_map.is_empty() {
            return false;
        }
        let rope = self.text.rope();
        let line = rope.byte_to_line(self.text.cursor());
        if !self.fold_map.steps_over(line) {
            return false;
        }
        let target = if forward {
            let out = self.fold_map.first_visible_at_or_after(line);
            if out >= rope.len_lines() {
                rope.len()
            } else {
                rope.line_to_byte(out)
            }
        } else {
            let heading = self.fold_map.first_visible_at_or_before(line);
            rope.line_content_range(heading).1
        };
        self.text.set_cursor_changed(target, extend)
    }

    /// The logical-line start of the heading whose fold chevron is under a window
    /// position, if any.
    pub(crate) fn fold_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<usize> {
        let body_size = self.body_size;
        let (layout, _, y) = self.layout_at(fonts, rect, py)?;
        layout.fold_at(fonts, px, y, body_size)
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
        let adjust = self.adjust();
        let margin = self.scale.px(MARGIN);
        self.find
            .matches()
            .iter()
            .map(|&(start, _)| margin + index.logical_top(rope.byte_to_line(start), adjust))
            .collect()
    }

    /// Whether the cursor's logical line falls inside the painted window and is on
    /// screen at all, so a caret drawn from that window sits at the true position
    /// rather than a clamped fallback. Off-window (scrolled far away) or behind a
    /// fold means the caret is simply not drawn.
    pub(crate) fn cursor_in_window(&self) -> bool {
        let line = self.text.rope().byte_to_line(self.text.cursor());
        self.layout_band.0 <= line && line < self.layout_band.1 && !self.fold_map.hides(line)
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
    /// position within this pane's column, anchoring a drag-selection at the
    /// selection's fixed end so holding the button and moving extends from there.
    /// Returns whether the cursor moved.
    pub(crate) fn press_at(
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
        let moved = self.text.set_cursor_changed(offset, extend);
        let anchor = self.text.selection_anchor();
        self.drag = Some(DragSelect::text((anchor, anchor), DragGrain::Char));
        moved
    }

    /// A double-click press: select the word under the pointer and drag on by whole
    /// words, so a sweep keeps snapping to word bounds. Around a collapsed picture
    /// it selects the picture as a single click does, since a word of its hidden
    /// source would reveal it.
    pub(crate) fn press_word(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) {
        let offset = self.pointer_offset(fonts, rect, px, py);
        self.ensure_index(fonts, rect);
        let word = match self.collapsed_picture(self.text.rope().byte_to_line(offset)) {
            Some(_) => {
                self.text.set_cursor(offset, false);
                (offset, offset)
            }
            None => self.text.select_word_at(offset),
        };
        self.drag = Some(DragSelect::text(word, DragGrain::Word));
    }

    /// A triple-click press: select the line under the pointer and drag on by whole
    /// lines.
    pub(crate) fn press_line(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) {
        let offset = self.pointer_offset(fonts, rect, px, py);
        let line = self.text.select_line_at(offset);
        self.drag = Some(DragSelect::text(line, DragGrain::Line));
    }

    /// A press on a rendered horizontal rule: select the whole rule line (its
    /// `---` is collapsed, so there is no caret position between the markers to
    /// land on) and anchor the drag on it, so a sweep off the rule grows the
    /// selection out of it. Returns whether the press was the rule's.
    pub(crate) fn press_rule(&mut self, fonts: &Fonts, rect: Rect, py: f32) -> bool {
        let rule = {
            let Some((layout, _, y)) = self.layout_at(fonts, rect, py) else {
                return false;
            };
            layout.rule_at(fonts, y)
        };
        let Some((start, end)) = rule else {
            return false;
        };
        self.text.select_range(start, end);
        self.drag = Some(DragSelect::text((start, end), DragGrain::Char));
        true
    }

    /// Extend the live drag-selection to a window position within this pane's
    /// column. Returns whether anything changed; `false` with no drag in flight.
    pub(crate) fn drag_to(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> bool {
        let Some(drag) = self.drag else {
            return false;
        };
        let offset = {
            let Some((layout, rope, y)) = self.layout_at(fonts, rect, py) else {
                return false;
            };
            layout.offset_at(fonts, rope, px, y)
        };
        // A picture the press selected stays selected while the pointer stays on
        // its line: a drag has to leave the line before the text selection takes
        // over, so a jitter during the click does not turn the selected picture into
        // a selected line of collapsed source.
        if let Some(at) = drag.image
            && drag.holds(offset)
        {
            return self.text.set_cursor_changed(at, false);
        }
        self.text.drag_to(drag.anchor, offset, drag.grain)
    }

    /// Whether a pointer drag-selection is in flight in this pane.
    pub(crate) fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Let go of the drag-selection: the button came up, or another gesture (a
    /// followed link) took the press.
    pub(crate) fn end_drag(&mut self) {
        self.drag = None;
    }

    /// Let go of a drag-selection because the button came up, reporting whether one
    /// was in flight.
    ///
    /// A sweep by character that starts in a block marker takes the whole marker
    /// with it (see TextBuffer::include_block_marker). A double or triple click has
    /// asked for a word or a line, so it keeps the marker out.
    pub(crate) fn release_drag(&mut self) -> bool {
        let Some(drag) = self.drag.take() else {
            return false;
        };
        if drag.grain == DragGrain::Char {
            self.text.include_block_marker();
        }
        true
    }

    /// The destination of the openable link under a window position, if its line
    /// is rendered (markup collapsed) and the scheme is one the browser opens. An
    /// inline `[text](dest)` and a bare URL both resolve here, since a bare URL is
    /// just the link whose text is its own destination.
    ///
    /// The pointer has to sit on the link's own glyphs. The empty space around them
    /// (the margins, the padding above the first line, the run-out past the end of a
    /// wrapped URL) is space the painter drew no link on, so a click there drops the
    /// caret rather than opening a browser: a note that is mostly link stays a note
    /// you can edit.
    pub(crate) fn link_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<String> {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(py);
        let (a0, a1) = self.reveal_range();
        let layout = self.layout_cache.as_ref()?;
        let row = layout.row_at(fonts, y)?;
        // A fenced code block is not parsed as inline markdown, so nothing in it is a
        // link however much it looks like one: a `curl https://…` renders as code and
        // must stay code when clicked.
        if row.code_block {
            return None;
        }
        let rope = self.text.rope();
        let (ls, le) = rope.line_content_range(rope.byte_to_line(row.start));
        // The revealed line shows its raw source, which the caret is editing.
        if a0 <= le && a1 >= ls {
            return None;
        }
        let content = rope.slice(ls..le).as_cow();
        let link = markdown::links(&content, &mut self.link_scratch)
            .into_iter()
            .find(|l| layout.span_hit(fonts, rope, row, (ls + l.text.0, ls + l.text.1), px, y))?;
        let dest = content[link.dest.0..link.dest.1].to_string();
        let text = &content[link.text.0..link.text.1];
        (crate::platform::browser::can_open(&dest)
            && crate::platform::browser::text_agrees(text, &dest))
        .then_some(dest)
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

    /// The logical-line start of the picture drawn under a window position, if
    /// any. Its corner button counts as the picture, since it sits on top of it.
    pub(crate) fn picture_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<usize> {
        let (layout, _, y) = self.layout_at(fonts, rect, py)?;
        layout.image_at(fonts, px, y)
    }

    /// Handle a press on a picture's corner button or on the picture, and report
    /// which it was. The button toggles the raw source open, and closing it from
    /// inside puts the caret back on the picture. The picture is selected: the caret
    /// goes onto it, on its side above from the upper half and below from the lower,
    /// which also takes the caret out of any source it was editing; and a drag is
    /// anchored on its line so a sweep off the picture selects it along with the
    /// text it reaches. A caller counting clicks needs to know which it was, since
    /// only a second click on the picture opens it.
    pub(crate) fn image_press(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> ImagePress {
        let (button, picture, on) = {
            let Some((layout, rope, y)) = self.layout_at(fonts, rect, py) else {
                return ImagePress::Missed;
            };
            (
                layout.image_button_at(fonts, px, y),
                layout.image_at(fonts, px, y),
                layout.offset_at(fonts, rope, px, y),
            )
        };
        if let Some(ls) = button {
            self.toggle_raw_pin(ls);
            return ImagePress::Button;
        }
        let Some(ls) = picture else {
            return ImagePress::Missed;
        };
        let rope = self.text.rope();
        let line = rope.line_content_range(rope.byte_to_line(ls));
        self.raw_image = None;
        self.text.set_cursor(on, false);
        self.drag = Some(DragSelect::image(on, line));
        ImagePress::Picture
    }

    /// Open or close the source of the picture on the line starting at ls. Closing
    /// it with the caret in it puts the caret back on the picture, keeping the side
    /// above when the caret was at the source's start, so the picture shows
    /// selected again rather than the source staying open around the caret.
    fn toggle_raw_pin(&mut self, ls: usize) {
        if self.raw_image != Some(ls) {
            self.raw_image = Some(ls);
            return;
        }
        self.raw_image = None;
        let rope = self.text.rope();
        let cursor = self.text.cursor();
        let (start, end) = rope.line_content_range(rope.byte_to_line(ls));
        if (start..=end).contains(&cursor) {
            let on = if cursor == start { start } else { end };
            self.text.set_cursor(on, false);
        }
    }

    /// The logical-line start of the picture the caret is on, which selects it: a
    /// bare caret at the start or end of a collapsed picture's line. The caret there
    /// is not drawn; the picture shows a border and its corner button instead. Reads
    /// the index, which the caller has brought current.
    pub(crate) fn selected_picture(&self) -> Option<usize> {
        if self.text.selection().is_some() {
            return None;
        }
        let cursor = self.text.cursor();
        let (ls, le) = self.collapsed_picture(self.text.rope().byte_to_line(cursor))?;
        (cursor == ls || cursor == le).then_some(ls)
    }

    /// The content range ls..le of a logical line when it draws as a picture
    /// with its source collapsed, so the caret's only places on it are the line's
    /// start and end, the picture selected at either. Reads the index, which the
    /// caller has brought current.
    fn collapsed_picture(&self, line: usize) -> Option<(usize, usize)> {
        let index = self.line_index.as_ref()?;
        if index.image_extra(line) == 0 {
            return None;
        }
        let (ls, le) = self.text.rope().line_content_range(line);
        (self.raw_image != Some(ls)).then_some((ls, le))
    }

    /// The side the caret holds on the selected picture, if one is selected.
    fn picture_side(&mut self, fonts: &Fonts, rect: Rect) -> Option<PictureSide> {
        self.ensure_index(fonts, rect);
        let ls = self.selected_picture()?;
        Some(if self.text.cursor() == ls {
            PictureSide::Above
        } else {
            PictureSide::Below
        })
    }

    /// Insert typed or pasted text at the caret. On a selected picture the text
    /// goes on a line of its own, above or below the picture by the side the caret
    /// holds, so the picture's line stays a picture.
    pub(crate) fn insert_text(&mut self, fonts: &Fonts, rect: Rect, s: &str) {
        match self.picture_side(fonts, rect) {
            Some(side) => self.text.insert_line(s, side == PictureSide::Above),
            None => self.text.insert_str(s),
        }
    }

    /// Insert pasted text at the caret, placed as typed text is on a selected
    /// picture. On a quote line the paste stays in the quote.
    pub(crate) fn paste_text(&mut self, fonts: &Fonts, rect: Rect, s: &str) {
        match self.picture_side(fonts, rect) {
            Some(side) => self.text.insert_line(s, side == PictureSide::Above),
            None => self.text.paste_str(s),
        }
    }

    /// The Enter key. On a selected picture holding its side above, it opens an
    /// empty line above the picture and moves onto it, ready to type there.
    pub(crate) fn enter(&mut self, fonts: &Fonts, rect: Rect) -> bool {
        if self.picture_side(fonts, rect) == Some(PictureSide::Above) {
            self.text.insert_line("", true);
            return true;
        }
        self.text.enter()
    }

    /// One of the delete keys. On a selected picture any of them removes the
    /// picture with its line. From a neighboring line, a delete that would reach a
    /// collapsed picture's line stops at the line break, and with nothing left to
    /// delete the caret goes onto the picture, selecting it; a second press then
    /// removes it. A partial delete would break the picture unseen and a join would
    /// turn it back into markdown. An empty line between still goes, since removing
    /// it leaves the picture's line as it was. Otherwise the plain delete runs.
    pub(crate) fn delete(&mut self, fonts: &Fonts, rect: Rect, key: DeleteKey) -> bool {
        let forward = matches!(key, DeleteKey::Forward | DeleteKey::WordForward);
        if self.picture_side(fonts, rect).is_some() {
            return self.delete_selected_picture(forward);
        }
        if let Some(changed) = self.delete_short_of_picture(fonts, rect, key) {
            return changed;
        }
        match key {
            DeleteKey::Back => self.text.backspace(),
            DeleteKey::Forward => self.text.delete_forward(),
            DeleteKey::WordBack => self.text.delete_word_back(),
            DeleteKey::WordForward => self.text.delete_word_forward(),
        }
    }

    /// Remove the selected picture and its line as one undo step. Backspace leaves
    /// the caret at the end of the line above, Delete at the start of the line
    /// below, each falling back to the other side at the document's edge.
    fn delete_selected_picture(&mut self, forward: bool) -> bool {
        let Some(ls) = self.selected_picture() else {
            return false;
        };
        let rope = self.text.rope();
        let le = rope.line_content_range(rope.byte_to_line(ls)).1;
        // The line goes with its own line break, or the one before it when it is
        // the last line.
        let (lo, hi) = if le < rope.len() {
            (ls, le + 1)
        } else {
            (ls.saturating_sub(1), le)
        };
        self.text.delete_range(lo, hi);
        let caret = if forward { lo } else { ls.saturating_sub(1) };
        self.text.set_cursor(caret, false);
        true
    }

    /// A delete from a bare caret on another line whose reach crosses into a
    /// collapsed picture's line: cut short at that line's break, or the caret onto
    /// the picture when nothing is left. None when the delete reaches no picture,
    /// or only removes the empty line the caret is on.
    fn delete_short_of_picture(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        key: DeleteKey,
    ) -> Option<bool> {
        if self.text.selection().is_some() {
            return None;
        }
        self.ensure_index(fonts, rect);
        let forward = matches!(key, DeleteKey::Forward | DeleteKey::WordForward);
        let word = matches!(key, DeleteKey::WordBack | DeleteKey::WordForward);
        let cursor = self.text.cursor();
        let reach = self.text.delete_target(forward, word);
        let rope = self.text.rope();
        let line = rope.byte_to_line(cursor);
        let (cs, ce) = rope.line_content_range(line);
        // The nearest collapsed picture between the caret's line and the reach.
        let picture = if forward {
            (line + 1..=rope.byte_to_line(reach)).find_map(|l| self.collapsed_picture(l))
        } else {
            (rope.byte_to_line(reach)..line)
                .rev()
                .find_map(|l| self.collapsed_picture(l))
        }?;
        if !word && cs == ce {
            return None;
        }
        let (ls, le) = picture;
        let (lo, hi) = if forward {
            (cursor, reach.min(ls - 1))
        } else {
            (reach.max(le + 1), cursor)
        };
        if lo >= hi {
            return Some(
                self.text
                    .set_cursor_changed(if forward { ls } else { le }, false),
            );
        }
        Some(self.text.delete_range(lo, hi))
    }

    /// A Left or Right step, by grapheme or by word, that treats a collapsed
    /// picture as one stop. A step onto the picture, or into its hidden source,
    /// selects it on the side it came from: Right arrives on its side below and
    /// Left on its side above, which is where typing then goes. The next step
    /// leaves it. Returns whether the cursor or selection changed.
    pub(crate) fn move_horizontal(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        forward: bool,
        word: bool,
        extend: bool,
    ) -> bool {
        self.ensure_index(fonts, rect);
        let from = self.text.cursor();
        let rope = self.text.rope();
        if let Some((ls, le)) = self
            .collapsed_picture(rope.byte_to_line(from))
            .filter(|&(ls, le)| from == ls || from == le)
        {
            let past = if forward {
                (le < rope.len()).then_some(le + 1)
            } else {
                ls.checked_sub(1)
            };
            return self.text.set_cursor_changed(past.unwrap_or(from), extend);
        }
        let moved = match (forward, word) {
            (false, false) => self.text.move_left(extend),
            (true, false) => self.text.move_right(extend),
            (false, true) => self.text.move_word_left(extend),
            (true, true) => self.text.move_word_right(extend),
        };
        moved | self.step_over_folds(forward, extend) | self.land_on_picture(forward, extend)
    }

    /// Settle a caret that came to rest on a collapsed picture's line onto the
    /// picture: on its side below when moving forward, above when moving back.
    /// Reads the index, which the caller has brought current. Returns whether the
    /// cursor or selection changed.
    pub(crate) fn land_on_picture(&mut self, forward: bool, extend: bool) -> bool {
        let cursor = self.text.cursor();
        let Some((ls, le)) = self.collapsed_picture(self.text.rope().byte_to_line(cursor)) else {
            return false;
        };
        self.text
            .set_cursor_changed(if forward { le } else { ls }, extend)
    }

    /// Close the source the corner button pinned open unless the caret is on its
    /// line: the pin holds while that source is being edited and closes once the
    /// caret leaves it. Returns whether it closed.
    pub(crate) fn settle_raw_pin(&mut self) -> bool {
        let Some(raw) = self.raw_image else {
            return false;
        };
        let rope = self.text.rope();
        if rope.line_to_byte(rope.byte_to_line(self.text.cursor())) == raw {
            return false;
        }
        self.raw_image = None;
        true
    }

    /// Whether a window position lands on the line whose source is pinned open, so
    /// a click there edits that source and keeps it open.
    pub(crate) fn pointer_on_pinned_source(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> bool {
        let Some(raw) = self.raw_image else {
            return false;
        };
        let offset = self.pointer_offset(fonts, rect, px, py);
        let rope = self.text.rope();
        rope.line_to_byte(rope.byte_to_line(offset)) == raw
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
    /// link, checkbox, image button, or table add-row/add-column strip, the arrow
    /// over a picture, the I-beam elsewhere.
    pub(crate) fn cursor_shape_at(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> u32 {
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_POINTER, SHAPE_TEXT};
        let over_link = self.link_under_pointer(fonts, rect, px, py).is_some();
        let over_checkbox = self.checkbox_under_pointer(fonts, rect, px, py).is_some();
        // A fold chevron is a control, not text, so the I-beam gives way over it the
        // way it does over a link or a checkbox.
        let over_fold = self.fold_under_pointer(fonts, rect, px, py).is_some();
        if over_link || over_checkbox || over_fold {
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

/// The two columns' aligned layout is memoized on the geometry it was built for: the
/// old and new text-area widths (a resize or a split-ratio drag changes them), the body
/// size, and the display scale. A scroll leaves all of these unchanged and reuses the
/// layout; only the visible window shifts.
type DiffLayoutKey = (i32, i32, u32, u32);

/// The two text rectangles a [`DiffView`] lays its columns into: the old (left) and the
/// new (right). Each is the text area only, its line-number gutter already carved off.
/// The renderer computes these; the view lays out to them so the tinted words land where
/// they are drawn.
#[derive(Clone, Copy)]
pub(crate) struct DiffColumns {
    pub(crate) old_text: Rect,
    pub(crate) new_text: Rect,
}

/// One side of one aligned row: whether this side has a line here, whether that line is
/// a change (tinted and accented) rather than carried through, its content height within
/// the band, its 1-based line number, and the first visual row's baseline (for the
/// gutter number). An absent cell (`!present`) is the empty half of a one-sided change,
/// which the renderer hatches.
#[derive(Clone, Copy)]
pub(crate) struct DiffCell {
    pub(crate) present: bool,
    pub(crate) changed: bool,
    pub(crate) height: i32,
    pub(crate) number: usize,
    pub(crate) baseline: i32,
}

impl DiffCell {
    const ABSENT: DiffCell = DiffCell {
        present: false,
        changed: false,
        height: 0,
        number: 0,
        baseline: 0,
    };
}

/// One aligned row of the diff: the content-space band both columns share, and each
/// side's cell within it. The old and new lines of a row sit at the same `y` so a change
/// reads straight across the divider; the taller side sets the band height and the
/// shorter side leaves the surplus blank (a changed pair) or hatched (a one-sided
/// change).
#[derive(Clone, Copy)]
pub(crate) struct DiffBand {
    pub(crate) y: i32,
    pub(crate) height: i32,
    pub(crate) old: DiffCell,
    pub(crate) new: DiffCell,
}

/// A side-by-side diff shown in its own tab: the old text on the left, the new on the
/// right, aligned line for line so a change reads across the divider. It is a pure
/// function of the two texts (the line and word diffs decide the tint; nothing massages
/// the tool's answer), and it remembers only what an accept needs: the source document,
/// the span it replaces, the version that span was captured at, and the revised text to
/// write. Accept applies that edit to the source wherever its tab now sits; reject just
/// closes this one.
pub(crate) struct DiffView {
    /// The document this diff applies back to, named by id so it is found after a tab
    /// move or reorder, or known to be gone.
    source: DocId,
    /// The byte range in the source the revised text replaces.
    span: (usize, usize),
    /// The source's content version when the action ran. A later version means the note
    /// changed underneath, so the span is stale and accept is refused.
    version: u64,
    /// What accept writes over `span`: the tool's answer verbatim.
    revised: String,
    /// The action's name, shown as the tab's title.
    label: &'static str,
    /// The two sides as ropes: `old` is the original fragment, `new` the revision.
    old: Rope,
    new: Rope,
    /// The removed word ranges in `old` and the added word ranges in `new`, tinted red
    /// on the left and green on the right. A one-sided line contributes its whole range
    /// (it is wholly added or removed); a changed pair contributes only its edited words.
    deletes: Vec<(usize, usize)>,
    inserts: Vec<(usize, usize)>,
    /// The line alignment: one row per aligned old/new line pair.
    rows: Vec<crate::ai::DiffRow>,
    /// The shared vertical scroll, applied to both sides so they move together.
    pub(crate) scroll_y: i32,
    /// The scrollable extent (aligned content plus top and bottom margin), for the
    /// shared scroll clamp and the column scrollbar. Refreshed with the layout.
    content_height: i32,
    /// The two columns' laid-out text, each line nudged to the row band both columns
    /// share. `None` until first built; rebuilt on a resize, reused on a scroll.
    old_layout: Option<Layout>,
    new_layout: Option<Layout>,
    /// The aligned row bands, in reading order, alongside the layouts they were placed
    /// under, so the renderer draws the tints, hatching, and gutter numbers.
    bands: Vec<DiffBand>,
    layout_key: Option<DiffLayoutKey>,
    /// An empty image store: a prose fragment carries no pictures to decode, so an
    /// image line (rare in a selection) lays out as its raw source text.
    images: ImageStore,
}

impl DiffView {
    /// Build a diff of `old` into `revised`, targeting document `source` at `span`
    /// (captured at content `version`). The line and word diffs run once here; the
    /// layout builds lazily on the first frame that knows the column geometry.
    pub(crate) fn new(
        source: DocId,
        span: (usize, usize),
        version: u64,
        old: String,
        revised: String,
        label: &'static str,
    ) -> Self {
        let rows = crate::ai::line_diff(&old, &revised);
        let (deletes, inserts) = diff_word_ranges(&old, &revised, &rows);
        Self {
            source,
            span,
            version,
            old: Rope::from_str(&old),
            new: Rope::from_str(&revised),
            revised,
            label,
            deletes,
            inserts,
            rows,
            scroll_y: 0,
            content_height: 0,
            old_layout: None,
            new_layout: None,
            bands: Vec::new(),
            layout_key: None,
            images: ImageStore::new(None),
        }
    }

    /// The document this diff applies to, the range it replaces, the version that range
    /// was captured at, and the text to write. The app resolves the source by id, checks
    /// the version still matches, and splices `revised` over `span` as one undo step.
    pub(crate) fn apply(&self) -> (DocId, (usize, usize), u64, &str) {
        (self.source, self.span, self.version, &self.revised)
    }

    /// Build or refresh the aligned layout for the geometry in `cols`, and refresh the
    /// content height and row bands. Each side is laid out on its own, then every line is
    /// nudged so the two columns share one row band per aligned line pair. Keyed on the
    /// geometry, so a scroll reuses the cached layout and a resize rebuilds it. Called
    /// each frame before the view is drawn.
    pub(crate) fn ensure_layouts(
        &mut self,
        fonts: &Fonts,
        cols: &DiffColumns,
        scale: Scale,
        body_size: u32,
    ) {
        let key = (
            cols.old_text.w,
            cols.new_text.w,
            body_size,
            scale.factor_120(),
        );
        if self.layout_key == Some(key) {
            return;
        }
        let margin = scale.px(DIFF_MARGIN);
        let build = |rope: &Rope, rect: Rect| {
            let ctx = LayoutCtx::new(
                fonts,
                rope,
                (rect.w - 2 * margin).max(0),
                body_size,
                NO_REVEAL,
                &self.images,
                ImageUi::default(),
                FoldMap::NONE,
                scale,
            );
            Layout::build_with_images(&ctx, rect.x + margin, rect.y + margin)
        };
        let mut old_layout = build(&self.old, cols.old_text);
        let mut new_layout = build(&self.new, cols.new_text);

        // Group each side into its logical lines, then walk the aligned rows: each row's
        // band is as tall as its taller side, and every line is shifted so its box top
        // lands at the band both columns share.
        let old_groups = logical_groups(old_layout.lines(), fonts);
        let new_groups = logical_groups(new_layout.lines(), fonts);
        let mut old_shift = vec![0i32; old_layout.lines().len()];
        let mut new_shift = vec![0i32; new_layout.lines().len()];
        let mut bands = Vec::with_capacity(self.rows.len());
        // The bands are placed in the same window space the raw layouts were built in
        // (both columns start at the pane's top plus a margin), so a scroll of 0 shows
        // the first line a margin below the pane top, as the document view does.
        let top = cols.old_text.y + margin;
        let mut y = top;
        for row in &self.rows {
            let old_g = row.old.and_then(|i| old_groups.get(i)).copied();
            let new_g = row.new.and_then(|j| new_groups.get(j)).copied();
            let band_h = old_g
                .map_or(0, |g| g.height)
                .max(new_g.map_or(0, |g| g.height));
            let cell =
                |group: Option<LineGroup>, number: Option<usize>, shift: &mut [i32]| -> DiffCell {
                    match (group, number) {
                        (Some(g), Some(idx)) => {
                            let dy = y - g.top;
                            for s in shift.iter_mut().take(g.last + 1).skip(g.first) {
                                *s = dy;
                            }
                            DiffCell {
                                present: true,
                                changed: row.changed,
                                height: g.height,
                                number: idx + 1,
                                baseline: g.baseline + dy,
                            }
                        }
                        _ => DiffCell::ABSENT,
                    }
                };
            let old = cell(old_g, row.old, &mut old_shift);
            let new = cell(new_g, row.new, &mut new_shift);
            bands.push(DiffBand {
                y,
                height: band_h,
                old,
                new,
            });
            y += band_h;
        }
        // The scrollable extent runs from the pane top to the last band's bottom, plus a
        // margin each end (the top margin is the gap from the pane top to the first band).
        let content_height = (y - cols.old_text.y) + margin;
        old_layout.shift_baselines(&old_shift, content_height);
        new_layout.shift_baselines(&new_shift, content_height);

        self.content_height = content_height;
        self.bands = bands;
        self.old_layout = Some(old_layout);
        self.new_layout = Some(new_layout);
        self.layout_key = Some(key);
    }

    /// The scrollable extent, for the scroll clamp and the scrollbar.
    pub(crate) fn content_height(&self) -> i32 {
        self.content_height
    }

    /// The aligned row bands, for the tints, hatching, and gutter numbers.
    pub(crate) fn bands(&self) -> &[DiffBand] {
        &self.bands
    }

    /// The old side (removed words tinted), or `None` before the first
    /// [`Self::ensure_layouts`].
    pub(crate) fn old_side(&self) -> Option<DiffSide<'_>> {
        self.old_layout.as_ref().map(|layout| DiffSide {
            layout,
            rope: &self.old,
            ranges: &self.deletes,
        })
    }

    /// The new side (added words tinted).
    pub(crate) fn new_side(&self) -> Option<DiffSide<'_>> {
        self.new_layout.as_ref().map(|layout| DiffSide {
            layout,
            rope: &self.new,
            ranges: &self.inserts,
        })
    }
}

/// One logical line's placement within a laid-out side: the visual-row range it spans
/// (`[first, last]`, inclusive), its box top, its first row's baseline, and its total
/// height. The diff alignment shifts each group as a unit onto its row band.
#[derive(Clone, Copy)]
struct LineGroup {
    first: usize,
    last: usize,
    top: i32,
    baseline: i32,
    height: i32,
}

/// Group a side's visual lines into logical lines, one [`LineGroup`] per `'\n'`-
/// delimited line (matching [`crate::ai::line_diff`]'s split). A logical line ends at
/// the first row whose hard break follows it, or at the last row of the layout.
fn logical_groups(lines: &[VisualLine], fonts: &Fonts) -> Vec<LineGroup> {
    let mut groups = Vec::new();
    let mut first = 0;
    for (i, line) in lines.iter().enumerate() {
        if !(line.hard || i + 1 == lines.len()) {
            continue;
        }
        let seed = &lines[first];
        let top = seed.baseline - fonts.metrics(seed.size).ascent;
        let height = lines[first..=i]
            .iter()
            .map(|l| row_box_height(l, fonts))
            .sum();
        groups.push(LineGroup {
            first,
            last: i,
            top,
            baseline: seed.baseline,
            height,
        });
        first = i + 1;
    }
    groups
}

/// A visual line's box height, the baseline-to-next advance the layout stacked it by: an
/// image or table row's taller box, else the line-height of its font size.
fn row_box_height(line: &VisualLine, fonts: &Fonts) -> i32 {
    if let Some(img) = &line.image {
        img.box_height()
    } else if let Some(table) = &line.table {
        table.box_height()
    } else {
        fonts.metrics(line.size).line_height
    }
}

/// The word ranges to tint over each side, from the aligned rows: a changed pair (both
/// lines present) contributes only the words that differ, while a one-sided line (wholly
/// added or removed) contributes its whole extent. Ranges are byte offsets into `old`
/// and `new`, the space the layout and its tinting read.
fn diff_word_ranges(
    old: &str,
    new: &str,
    rows: &[crate::ai::DiffRow],
) -> (crate::ai::WordRanges, crate::ai::WordRanges) {
    let old_spans = line_spans(old);
    let new_spans = line_spans(new);
    let (mut deletes, mut inserts) = (Vec::new(), Vec::new());
    for row in rows {
        if !row.changed {
            continue;
        }
        let old_line = row.old.map(|i| old_spans[i]);
        let new_line = row.new.map(|j| new_spans[j]);
        match (old_line, new_line) {
            (Some((os, oe)), Some((ns, ne))) => {
                // A changed pair: tint only the words that differ, shifted into each
                // side's byte space.
                let (dels, inss) = crate::ai::side_ranges(&old[os..oe], &new[ns..ne]);
                deletes.extend(dels.into_iter().map(|(a, b)| (os + a, os + b)));
                inserts.extend(inss.into_iter().map(|(a, b)| (ns + a, ns + b)));
            }
            (Some(span), None) => deletes.push(span),
            (None, Some(span)) => inserts.push(span),
            (None, None) => {}
        }
    }
    (deletes, inserts)
}

/// The byte range of each `'\n'`-delimited line in `s` (the newline excluded), matching
/// the logical lines [`crate::ai::line_diff`] indexes.
fn line_spans(s: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = 0;
    for (i, b) in s.bytes().enumerate() {
        if b == b'\n' {
            spans.push((start, i));
            start = i + 1;
        }
    }
    spans.push((start, s.len()));
    spans
}

/// One side of a diff for the renderer: its laid-out text, the rope it was built from,
/// and the changed word ranges to tint (removed on the old side, added on the new).
pub(crate) struct DiffSide<'a> {
    pub(crate) layout: &'a Layout,
    pub(crate) rope: &'a Rope,
    pub(crate) ranges: &'a [(usize, usize)],
}

/// What a tab shows. Tabs are managed uniformly by [`PaneGroup`] (order, active tab,
/// open/close/move); only the content varies: a document, or a side-by-side diff
/// awaiting accept or reject. Both are heavy, so each is boxed to keep a `Tab` (and the
/// `Vec<Tab>` a pane holds) a small tagged pointer.
pub(crate) enum TabContent {
    Doc(Box<EditorPane>),
    Diff(Box<DiffView>),
}

/// One tab as the strip reads it: what to draw, without the content behind it. The
/// label borrows from the tab (a document's path, a review's own label), so a frame
/// composes the strip without building a string per tab.
#[derive(Clone, Copy)]
pub(crate) struct TabView<'a> {
    pub label: &'a str,
    /// Whether the tab has unsaved edits, for the strip's dot.
    pub dirty: bool,
    /// Whether the tab is pinned, so it takes the compact width and wears a pin
    /// badge in its close slot.
    pub pinned: bool,
}

/// One tab: its content, kept apart from the strip identity (the title and unsaved
/// dot the strip draws) so a pane can hold a view that is not a document. The content
/// answers what the strip needs through [`Self::title`] and [`Self::is_dirty`].
pub(crate) struct Tab {
    content: TabContent,
    /// Whether the tab is pinned: kept where it is, marked with a pin badge instead
    /// of a close cross, and guarded against the click and the Ctrl+W that would
    /// otherwise close it. Only a document tab is ever pinned; a diff review is
    /// transient. Persisted with the tab.
    pinned: bool,
}

impl Tab {
    /// A document tab.
    fn doc(doc: EditorPane) -> Self {
        Self {
            content: TabContent::Doc(Box::new(doc)),
            pinned: false,
        }
    }

    /// A diff tab: a side-by-side review awaiting accept or reject.
    fn diff(view: DiffView) -> Self {
        Self {
            content: TabContent::Diff(Box::new(view)),
            pinned: false,
        }
    }

    /// The document this tab holds, when it holds one (not a diff).
    fn as_doc(&self) -> Option<&EditorPane> {
        match &self.content {
            TabContent::Doc(doc) => Some(doc),
            TabContent::Diff(_) => None,
        }
    }

    fn as_doc_mut(&mut self) -> Option<&mut EditorPane> {
        match &mut self.content {
            TabContent::Doc(doc) => Some(doc),
            TabContent::Diff(_) => None,
        }
    }

    /// The diff this tab shows, when it is a diff tab.
    fn as_diff(&self) -> Option<&DiffView> {
        match &self.content {
            TabContent::Diff(view) => Some(view),
            TabContent::Doc(_) => None,
        }
    }

    fn as_diff_mut(&mut self) -> Option<&mut DiffView> {
        match &mut self.content {
            TabContent::Diff(view) => Some(view),
            TabContent::Doc(_) => None,
        }
    }

    /// The document, consuming the tab, for a tab dragged out of the pane. A diff tab
    /// yields nothing: it is transient and never re-homed.
    fn into_doc(self) -> Option<EditorPane> {
        match self.content {
            TabContent::Doc(doc) => Some(*doc),
            TabContent::Diff(_) => None,
        }
    }

    /// The label the tab strip draws, borrowed from the document's path (or the
    /// review's own label), so a frame does not build a string per tab.
    fn title(&self) -> &str {
        match &self.content {
            TabContent::Doc(doc) => doc.tab_name(),
            TabContent::Diff(view) => view.label,
        }
    }

    /// Whether the strip marks the tab with an unsaved-changes dot. A diff carries no
    /// file, so it is never dirty.
    fn is_dirty(&self) -> bool {
        match &self.content {
            TabContent::Doc(doc) => doc.text.is_modified(),
            TabContent::Diff(_) => false,
        }
    }
}

/// One content pane as a stack of open tabs with one active. A pane may hold no tabs
/// at all (nothing open), in which case it shows an empty column and a bare tab strip
/// with just the new-tab button. The group forwards to its active tab's *document*
/// (via [`Deref`]), so a document tab is simply "the pane" to every caller that edits,
/// scrolls, or hit-tests it; tab management and the DPI fan-out are group-level. The
/// deref stays total, an empty pane, or a non-document active tab, forwards to a
/// scratch `placeholder` document that is never drawn, so editing it is a harmless
/// no-op.
pub(crate) struct PaneGroup {
    tabs: Vec<Tab>,
    active: usize,
    /// The document an empty pane's deref targets, so the `pane.field` sites stay
    /// panic-free with no tabs open. Never painted.
    placeholder: EditorPane,
    /// The display scale and body size the group lays out at, so a tab opened
    /// after a DPI change is stamped with the current scale rather than the unity
    /// default a fresh [`EditorPane`] carries.
    scale: Scale,
    body_size: u32,
    /// The vault root every tab resolves wikilink embeds against, stamped onto each
    /// document as it opens so a `![[name]]` finds its picture. `None` outside a
    /// vault (a bare file argument with no session root).
    vault: Option<PathBuf>,
    /// The column's overlay scrollbar. It belongs to the column rather than to a
    /// document, so switching tabs slides the same bar onto the new document's scroll
    /// instead of resurrecting the one the old tab left half-faded.
    pub(crate) scrollbar: Scrollbar,
    /// The next document id this pane mints, stepped by two from a per-pane residue so
    /// the two panes' ids never collide and a tab dragged across the split keeps its
    /// own. See [`Self::mint_id`].
    next_id: u64,
}

impl PaneGroup {
    /// A pane with nothing open (no tabs), minting ids from `seed`. The two panes seed
    /// from distinct residues (0 and 1) and every id steps by two, so their documents
    /// are globally distinguishable.
    pub(crate) fn with_seed(seed: u64) -> Self {
        Self {
            tabs: Vec::new(),
            active: 0,
            placeholder: EditorPane::empty(),
            scale: Scale::ONE,
            body_size: FONT_SIZE,
            vault: None,
            scrollbar: Scrollbar::default(),
            next_id: seed,
        }
    }

    /// A pane with nothing open, seeded to mint even ids. A test convenience for a
    /// single-pane context; the app builds both panes through [`Self::with_seed`] so
    /// the second seeds odd.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self::with_seed(0)
    }

    /// Mint the next id for a document this pane opens, stepping the counter by two so
    /// the sibling pane (seeded to the other residue) can never mint the same value.
    fn mint_id(&mut self) -> DocId {
        let id = DocId(self.next_id);
        self.next_id += 2;
        id
    }

    /// Stamp `doc` with a fresh id and append it as a tab, returning its index. The one
    /// place a brand-new document joins the pane, so id assignment lives here rather
    /// than at each call site. A tab arriving from the sibling pane (see
    /// [`Self::insert_tab`]) keeps the id it was already stamped with.
    fn push_doc(&mut self, mut doc: EditorPane) -> usize {
        doc.id = self.mint_id();
        self.tabs.push(Tab::doc(doc));
        self.tabs.len() - 1
    }

    /// The active tab index (0 and meaningless when the pane has no tabs).
    pub(crate) fn active_index(&self) -> usize {
        self.active
    }

    /// The number of open tabs, for laying out and scrolling the tab strip.
    pub(crate) fn tab_count(&self) -> usize {
        self.tabs.len()
    }

    /// Whether the pane has any open tab. An empty pane paints no content and no
    /// caret, and its title falls back to the app name.
    pub(crate) fn has_tabs(&self) -> bool {
        !self.tabs.is_empty()
    }

    /// The active document, or the placeholder when the pane is empty or its active
    /// tab is not a document. Most callers reach it implicitly through [`Deref`]; this
    /// explicit borrow is for the few sites that mutate one field while reading
    /// another, which a deref cannot split into disjoint borrows.
    pub(crate) fn active_doc_mut(&mut self) -> &mut EditorPane {
        self.tabs
            .get_mut(self.active)
            .and_then(Tab::as_doc_mut)
            .unwrap_or(&mut self.placeholder)
    }

    /// Open `view` as a diff tab and focus it. It lands directly after the document it
    /// reviews when that document is in this pane, else at the end, so the review sits
    /// beside its source.
    pub(crate) fn open_diff(&mut self, view: DiffView) {
        let source = view.apply().0;
        let at = self
            .tabs
            .iter()
            .position(|t| t.as_doc().is_some_and(|d| d.id == source))
            .map_or(self.tabs.len(), |i| i + 1);
        self.tabs.insert(at, Tab::diff(view));
        self.active = at;
    }

    /// Whether the active tab is a diff review rather than a document, so the frame
    /// draws the two columns and the keyboard resolves accept or reject.
    pub(crate) fn active_is_diff(&self) -> bool {
        self.tabs
            .get(self.active)
            .is_some_and(|t| t.as_diff().is_some())
    }

    /// Whether tab `i` is a diff review. A diff tab is transient and re-homes to no
    /// document, so a cross-split drag leaves it where it is (see the tab-drag path).
    pub(crate) fn tab_is_diff(&self, i: usize) -> bool {
        self.tabs.get(i).is_some_and(|t| t.as_diff().is_some())
    }

    /// The active diff review, when the active tab is one.
    pub(crate) fn active_diff(&self) -> Option<&DiffView> {
        self.tabs.get(self.active).and_then(Tab::as_diff)
    }

    /// The active diff review mutably, for refreshing its layouts and scroll.
    pub(crate) fn active_diff_mut(&mut self) -> Option<&mut DiffView> {
        self.tabs.get_mut(self.active).and_then(Tab::as_diff_mut)
    }

    /// The document with id `id`, if a tab in this pane holds it, for a diff applying
    /// its edit back to its source.
    pub(crate) fn doc_by_id_mut(&mut self, id: DocId) -> Option<&mut EditorPane> {
        self.docs_mut().find(|d| d.id == id)
    }

    /// Focus the tab holding document `id`, snapping its caret, and report whether it
    /// was found here.
    pub(crate) fn focus_doc(&mut self, id: DocId) -> bool {
        let Some(i) = self
            .tabs
            .iter()
            .position(|t| t.as_doc().is_some_and(|d| d.id == id))
        else {
            return false;
        };
        self.active = i;
        self.active_doc_mut().request_caret_snap();
        true
    }

    /// Close the active tab, if there is one. Used to dismiss a resolved diff review.
    pub(crate) fn close_active(&mut self) {
        if self.active < self.tabs.len() {
            self.close(self.active);
        }
    }

    /// Each tab as the strip sees it, in order: its label, whether it has unsaved
    /// edits (the dot), and whether it is pinned (the badge, and the compact width).
    /// Borrowed, so composing a frame costs no string per tab.
    pub(crate) fn tab_views(&self) -> impl Iterator<Item = TabView<'_>> {
        self.tabs.iter().map(|t| TabView {
            label: t.title(),
            dirty: t.is_dirty(),
            pinned: t.pinned,
        })
    }

    /// Whether tab `i` is pinned. An out-of-range index reads as unpinned, so a
    /// close or a Ctrl+W on a pane with no tabs falls through to its normal no-op.
    pub(crate) fn is_pinned(&self, i: usize) -> bool {
        self.tabs.get(i).is_some_and(|t| t.pinned)
    }

    /// Flip tab `i`'s pinned state. A diff review is transient and never re-homed,
    /// so it is never pinned; the toggle ignores one (and any out-of-range index).
    pub(crate) fn toggle_pin(&mut self, i: usize) {
        if let Some(tab) = self.tabs.get_mut(i)
            && tab.as_doc().is_some()
        {
            tab.pinned = !tab.pinned;
        }
    }

    /// Set tab `i`'s pinned state, for carrying a pin across the split when a tab is
    /// dragged into this pane (where it arrives as a fresh, unpinned tab).
    pub(crate) fn set_pinned(&mut self, i: usize, pinned: bool) {
        if let Some(tab) = self.tabs.get_mut(i) {
            tab.pinned = pinned;
        }
    }

    /// Every open document tab's file path, for the file watcher's set of directories
    /// to watch. Scratch tabs (no path) and non-document tabs contribute nothing.
    pub(crate) fn paths(&self) -> impl Iterator<Item = &Path> {
        self.docs().filter_map(|d| d.path.as_deref())
    }

    /// Every open document tab, for what the whole group has to answer for rather than
    /// just the tab on top: saving each with unsaved edits, and watching each for one.
    pub(crate) fn docs_mut(&mut self) -> impl Iterator<Item = &mut EditorPane> {
        self.tabs.iter_mut().filter_map(Tab::as_doc_mut)
    }

    /// The group's open documents, for a reader that needs more of one than its path
    /// (a search wanting the buffer's text, which is what is on screen).
    pub(crate) fn docs(&self) -> impl Iterator<Item = &EditorPane> {
        self.tabs.iter().filter_map(Tab::as_doc)
    }

    /// Re-check every open document tab against its file on disk, reloading clean
    /// buffers and flagging conflicts on dirty ones. Returns whether any tab changed.
    pub(crate) fn recheck_disk(&mut self) -> bool {
        let mut changed = false;
        for doc in self.docs_mut() {
            changed |= doc.recheck_disk();
        }
        changed
    }

    /// Switch the active tab. Out-of-range indices are ignored.
    pub(crate) fn activate(&mut self, i: usize) {
        if i < self.tabs.len() {
            self.active = i;
            self.active_doc_mut().request_caret_snap();
        }
    }

    /// Open a fresh blank tab and focus it, stamped with the group's scale. Backs
    /// the new-tab button and the first keystroke into an empty pane.
    pub(crate) fn new_tab(&mut self, scale: Scale, body_size: u32) {
        let mut doc = EditorPane::empty();
        doc.set_scale(scale, body_size);
        doc.set_vault_root(self.vault.as_deref());
        self.active = self.push_doc(doc);
    }

    /// Open `path`: focus an existing tab holding it, replace a blank scratch tab
    /// in place, or stack a new tab and focus it. The new document adopts the
    /// group's current scale so it renders crisp immediately at any DPI.
    pub(crate) fn open_or_focus(&mut self, path: &Path) -> Result<()> {
        if let Some(i) = self
            .tabs
            .iter()
            .position(|t| t.as_doc().is_some_and(|d| d.path.as_deref() == Some(path)))
        {
            self.active = i;
            self.active_doc_mut().request_caret_snap();
            return Ok(());
        }
        let mut doc = EditorPane::open(path)?;
        doc.set_scale(self.scale, self.body_size);
        doc.set_vault_root(self.vault.as_deref());
        // Replace a lone blank scratch tab (a just-opened new tab); otherwise stack.
        // A pinned blank tab is left alone, since pinning it asks to keep it.
        if self
            .tabs
            .get(self.active)
            .is_some_and(|t| !t.pinned && t.as_doc().is_some_and(EditorPane::is_blank))
        {
            doc.id = self.mint_id();
            self.tabs[self.active] = Tab::doc(doc);
        } else {
            self.active = self.push_doc(doc);
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
    /// pane (a tab dragged across the split). A non-document tab yields `None` but is
    /// still removed. The active tab shifts the same way [`Self::close`] moves it. An
    /// out-of-range index yields `None` and removes nothing.
    pub(crate) fn detach(&mut self, i: usize) -> Option<EditorPane> {
        if i >= self.tabs.len() {
            return None;
        }
        let tab = self.tabs.remove(i);
        if i < self.active || self.active >= self.tabs.len() {
            self.active = self.active.saturating_sub(1);
        }
        // The tab that slid into focus is a different document; snap its caret so it
        // does not resume a smear from wherever it last left off.
        self.active_doc_mut().request_caret_snap();
        tab.into_doc()
    }

    /// Insert `doc` as a tab at index `i` (clamped to the end), adopt it under the
    /// group's scale, make it active, and return the index it landed at. Receives a
    /// tab dragged in from the other pane.
    pub(crate) fn insert_tab(&mut self, i: usize, mut doc: EditorPane) -> usize {
        doc.set_scale(self.scale, self.body_size);
        let i = i.min(self.tabs.len());
        self.tabs.insert(i, Tab::doc(doc));
        self.active = i;
        i
    }

    /// Move the tab at `from` to index `to`, sliding the tabs in between over to
    /// fill the gap. The same tab stays active across the move, whichever slot it
    /// lands in. Out-of-range or no-op moves are ignored.
    pub(crate) fn move_tab(&mut self, from: usize, to: usize) {
        if from >= self.tabs.len() || to >= self.tabs.len() || from == to {
            return;
        }
        let tab = self.tabs.remove(from);
        self.tabs.insert(to, tab);
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
    /// every document tab (and the placeholder) so an inactive tab re-measures
    /// correctly when next shown.
    pub(crate) fn set_scale(&mut self, scale: Scale, body_size: u32) {
        self.scale = scale;
        self.body_size = body_size;
        for doc in self.docs_mut() {
            doc.set_scale(scale, body_size);
        }
        self.placeholder.set_scale(scale, body_size);
    }

    /// The persisted form of this pane: each document tab's file, caret, scroll, and
    /// folded headings, plus the active index. Non-document tabs (a diff view) are
    /// transient and not persisted.
    pub(crate) fn to_session(&self) -> store::PaneGroupSession {
        // Only documents persist, so the active index is measured in the document-only
        // list: the count of documents ahead of the active tab, clamped into that list.
        // When the active tab is itself a diff (transient), this lands on the nearest
        // saved document rather than off the end.
        let doc_count = self.docs().count();
        let active = self.tabs[..self.active.min(self.tabs.len())]
            .iter()
            .filter(|t| t.as_doc().is_some())
            .count()
            .min(doc_count.saturating_sub(1));
        store::PaneGroupSession {
            tabs: self
                .tabs
                .iter()
                .filter_map(|t| {
                    t.as_doc().map(|d| store::PaneSession {
                        path: d.path.clone(),
                        cursor: d.text.cursor() as u64,
                        scroll_y: d.scroll_y,
                        folds: d.folds.lines().map(|l| l as u32).collect(),
                        pinned: t.pinned,
                    })
                })
                .collect(),
            active: active as u32,
        }
    }
}

impl Deref for PaneGroup {
    type Target = EditorPane;

    fn deref(&self) -> &EditorPane {
        self.tabs
            .get(self.active)
            .and_then(Tab::as_doc)
            .unwrap_or(&self.placeholder)
    }
}

impl DerefMut for PaneGroup {
    fn deref_mut(&mut self) -> &mut EditorPane {
        self.tabs
            .get_mut(self.active)
            .and_then(Tab::as_doc_mut)
            .unwrap_or(&mut self.placeholder)
    }
}

/// The file-tree sidebar: the directory model plus its own scroll and the row
/// the pointer last hovered (for highlight). Rows come from [`tree::Tree`].
pub(crate) struct TreePane {
    pub(crate) tree: tree::Tree,
    pub(crate) scroll_y: i32,
    pub(crate) content_height: i32,
    /// The sidebar's overlay scrollbar, lit while the rows scroll and grabbable when
    /// the pointer comes near it.
    pub(crate) scrollbar: Scrollbar,
    /// The row the pointer last hovered (mouse highlight), or `None` when the
    /// pointer is off the tree.
    pub(crate) hovered: Option<usize>,
    /// The active row: the last one clicked or opened, and the target of Up/Down
    /// and Enter. `None` until the user picks one, so nothing is marked at launch.
    pub(crate) selected: Option<usize>,
    /// Whether the pointer is over the footer's folder-filter toggle, which lights
    /// up under it.
    pub(crate) filter_hovered: bool,
}

/// Build a pane from its saved tabs, restoring each file's caret and scroll and
/// the active tab. An empty tab list (a never-opened pane) yields an empty pane
/// with no tabs. The active index is clamped into the restored tabs.
pub(crate) fn restore_group(
    tabs: &[store::PaneSession],
    active: usize,
    vault: Option<&Path>,
    seed: u64,
) -> PaneGroup {
    let mut group = PaneGroup::with_seed(seed);
    group.vault = vault.map(Path::to_path_buf);
    if tabs.is_empty() {
        return group;
    }
    for saved in tabs {
        let mut doc = restore_doc(saved);
        doc.set_vault_root(vault);
        let idx = group.push_doc(doc);
        group.tabs[idx].pinned = saved.pinned;
    }
    group.active = active.min(group.tabs.len() - 1);
    group
}

/// Build one document from a saved tab, restoring its caret, scroll, and folds.
/// Best-effort: an unreadable or missing file yields an empty document (logged,
/// not fatal) so a moved-away note never stops the editor opening. `set_cursor`
/// clamps the offset into the (possibly changed) file, and a fold anchor that no
/// longer lands on a heading is dropped once the document is first indexed.
fn restore_doc(saved: &store::PaneSession) -> EditorPane {
    let mut doc = match saved.path.as_deref() {
        Some(p) => EditorPane::open(p).unwrap_or_else(|e| {
            eprintln!("bnknotes: could not reopen {}: {e}", p.display());
            EditorPane::empty()
        }),
        None => EditorPane::empty(),
    };
    doc.text.set_cursor(saved.cursor as usize, false);
    doc.park_past_frontmatter();
    doc.scroll_y = saved.scroll_y.max(0);
    doc.folds = Folds::from_lines(saved.folds.iter().map(|&l| l as usize));
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
    use crate::pane::*;

    #[test]
    fn reveal_follows_the_selection_through_a_drag() {
        let mut ed = EditorPane::empty();
        ed.text.insert_str("# Title\nbody");
        // A settled selection over the heading is its own reveal range, so the
        // heading shows its markers.
        ed.text.set_cursor(0, false);
        ed.text.set_cursor(7, true);
        assert_eq!(ed.reveal_range(), (0, 7));
        // A live drag still reveals the selection, so the source you are grabbing
        // stays visible under the pointer rather than blanking out.
        ed.drag = Some(DragSelect::text((0, 0), DragGrain::Char));
        assert_eq!(ed.reveal_range(), (0, 7));
        ed.end_drag();
        assert_eq!(ed.reveal_range(), (0, 7));
    }

    #[test]
    fn reveal_follows_the_bare_cursor_when_not_selecting() {
        let mut ed = EditorPane::empty();
        ed.text.insert_str("# Title");
        ed.text.set_cursor(3, false);
        assert_eq!(ed.reveal_range(), (3, 3));
    }

    /// A scratch note path under the system temp dir, seeded with `contents`.
    fn tmp_note(name: &str, contents: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-pane-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("note.md");
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn recheck_reloads_a_clean_buffer_changed_on_disk() {
        let path = tmp_note("clean-reload", "original\n");
        let mut ed = EditorPane::open(&path).unwrap();
        assert_eq!(ed.text.text(), "original\n");

        // Another writer changes the file; the buffer has no edits.
        std::fs::write(&path, "rewritten from elsewhere\n").unwrap();
        assert!(ed.recheck_disk(), "an external change is detected");
        assert_eq!(
            ed.text.text(),
            "rewritten from elsewhere\n",
            "a clean buffer reloads"
        );
        assert!(ed.conflict.is_none(), "a silent reload raises no conflict");

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn recheck_flags_a_conflict_on_a_dirty_buffer() {
        let path = tmp_note("dirty-conflict", "original\n");
        let mut ed = EditorPane::open(&path).unwrap();
        // The buffer has unsaved edits.
        ed.text.insert_str("my local edit ");
        assert!(ed.text.is_modified());

        std::fs::write(&path, "someone else's version\n").unwrap();
        assert!(ed.recheck_disk());
        assert_eq!(
            ed.conflict,
            Some(DiskConflict::Changed),
            "a dirty buffer conflicts"
        );
        assert!(
            ed.text.text().starts_with("my local edit "),
            "the edits are kept, not clobbered by a reload"
        );

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn recheck_flags_a_delete() {
        let path = tmp_note("deleted", "here for now\n");
        let mut ed = EditorPane::open(&path).unwrap();

        std::fs::remove_file(&path).unwrap();
        assert!(ed.recheck_disk());
        assert_eq!(ed.conflict, Some(DiskConflict::Deleted));

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn recheck_is_a_no_op_when_the_file_is_unchanged() {
        let path = tmp_note("unchanged", "steady\n");
        let mut ed = EditorPane::open(&path).unwrap();
        assert!(!ed.recheck_disk(), "no external change means no work");
        assert!(ed.conflict.is_none());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn recheck_holds_a_raised_conflict_until_resolved() {
        // Once a conflict is parked for the user, further events must not thrash it
        // (or silently reload over the pending decision).
        let path = tmp_note("held", "v1\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("edit ");
        std::fs::write(&path, "v2\n").unwrap();
        assert!(ed.recheck_disk());
        assert_eq!(ed.conflict, Some(DiskConflict::Changed));

        // A second external change while the conflict stands changes nothing.
        std::fs::write(&path, "v3\n").unwrap();
        assert!(!ed.recheck_disk(), "a standing conflict is not re-raised");
        assert_eq!(ed.conflict, Some(DiskConflict::Changed));

        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_writes_the_buffer_and_marks_it_clean() {
        let path = tmp_note("save-writes", "before\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("after ");
        assert!(ed.text.is_modified());

        assert_eq!(ed.save().unwrap(), Saved::Written);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after before\n");
        assert!(!ed.text.is_modified(), "a written buffer is clean");
        // The file it just wrote is the synced state, so its own change does not
        // come back as an external one.
        assert!(!ed.recheck_disk());
        assert!(ed.conflict.is_none());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_is_nothing_to_do_for_an_unchanged_buffer() {
        // What makes an autosave on a timer affordable: an idle document costs a
        // version compare, not a write.
        let path = tmp_note("save-clean", "steady\n");
        let mut ed = EditorPane::open(&path).unwrap();
        assert_eq!(ed.save().unwrap(), Saved::Unchanged);

        // Nothing was rewritten: the file's identity is untouched.
        let before = file::DiskSnapshot::of(&path);
        assert_eq!(ed.save().unwrap(), Saved::Unchanged);
        assert_eq!(file::DiskSnapshot::of(&path), before);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_refuses_a_file_that_changed_underneath_it() {
        // The guard the autosave leans on hardest: firing on a timer must never be
        // how another writer's edit gets clobbered.
        let path = tmp_note("save-conflict", "v1\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("mine ");
        std::fs::write(&path, "theirs\n").unwrap();

        assert_eq!(ed.save().unwrap(), Saved::Conflict);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "theirs\n",
            "the other writer's file stands"
        );
        assert_eq!(ed.conflict, Some(DiskConflict::Changed));
        assert!(ed.text.is_modified(), "the buffer keeps its edits");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_recreates_a_file_deleted_underneath_it() {
        // A missing file is not another writer's version, so saving puts it back.
        let path = tmp_note("save-deleted", "v1\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("kept ");
        std::fs::remove_file(&path).unwrap();

        assert_eq!(ed.save().unwrap(), Saved::Written);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "kept v1\n");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_has_nowhere_to_put_a_blank_tab() {
        let mut ed = EditorPane::empty();
        ed.text.insert_str("scratch");
        assert_eq!(ed.save().unwrap(), Saved::NoPath);
        assert!(ed.text.is_modified(), "it stays put, and stays dirty");
    }

    #[test]
    fn save_keeps_the_files_own_line_endings() {
        // The autosave writes as often as the typing pauses, so it must not be what
        // silently rewrites a CRLF file to LF.
        let path = tmp_note("save-crlf", "one\r\ntwo\r\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("zero\n");
        assert_eq!(ed.save().unwrap(), Saved::Written);
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "zero\r\none\r\ntwo\r\n"
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn poll_edited_reports_an_edit_once() {
        // The autosave's idle window re-arms on this: an edit pushes it back, and a
        // pass with no edit lets it run out.
        let mut ed = EditorPane::empty();
        assert!(!ed.poll_edited(), "a fresh buffer has not been edited");
        ed.text.insert_str("hello");
        assert!(ed.poll_edited(), "the edit is reported");
        assert!(!ed.poll_edited(), "and only once");
    }

    #[test]
    fn poll_edited_sees_every_kind_of_change() {
        // Watching the buffer rather than the edit call sites is the point: an undo
        // and a reload move the text too, and each re-arms the window.
        let mut ed = EditorPane::empty();
        ed.text.insert_str("hello");
        assert!(ed.poll_edited());
        ed.text.undo();
        assert!(ed.poll_edited(), "an undo is a change like any other");
        ed.text.reload("from disk".to_string());
        assert!(ed.poll_edited(), "so is a reload");
    }

    #[test]
    fn poll_edited_ignores_a_save() {
        // A save does not move the text, so it must not re-arm the window it just
        // satisfied (which would autosave on a loop forever).
        let path = tmp_note("poll-save", "note\n");
        let mut ed = EditorPane::open(&path).unwrap();
        ed.text.insert_str("edit ");
        assert!(ed.poll_edited());
        assert_eq!(ed.save().unwrap(), Saved::Written);
        assert!(!ed.poll_edited(), "the save is not itself an edit");
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
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
    fn a_drag_over_a_marker_keeps_it_revealed() {
        // Dragging a selection across a heading keeps its `#` shown, so the source
        // being grabbed stays visible under the pointer rather than blanking out.
        // The heading is on the final line, where a last-line cursor and its
        // selection both fold to the same line index, so the rebuild still lands.
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

        // A drag now extends a selection across that line; the marker stays shown.
        let press = heading_start + 2;
        pane.drag = Some(DragSelect::text((press, press), DragGrain::Char));
        pane.text.set_cursor(text.len(), true);
        pane.ensure_layout(&fonts, rect);
        assert!(
            !heading_marker_hidden(&pane, heading_start),
            "a live drag keeps the heading marker revealed"
        );
    }

    #[test]
    fn double_clicking_a_headings_first_word_leaves_the_marker_out() {
        // The reported bug: a double-click selected the word, then on release the
        // selection jumped to swallow the `##` too. The marker rule exists for a
        // character sweep, which has no other way to reach past a collapsed marker;
        // a double-click asked for a word and gets one.
        let mut pane = pane_of("## Some heading\nbody\n");
        let word = pane.text.select_word_at(3);
        pane.drag = Some(DragSelect::text(word, DragGrain::Word));
        assert!(pane.release_drag());
        assert_eq!(
            pane.text.selection(),
            Some((3, 7)),
            "the word alone, `Some`, with the `## ` left out"
        );
    }

    #[test]
    fn sweeping_from_a_headings_text_start_still_takes_the_marker() {
        // The behavior the rule is for, unchanged: a press at the line's left edge
        // lands past the zero-width collapsed marker, so pulling the selection back
        // over it on release is the only way to select a heading with its level.
        let mut pane = pane_of("## Some heading\nbody\n");
        pane.text.set_cursor(3, false);
        pane.drag = Some(DragSelect::text((3, 3), DragGrain::Char));
        pane.text.set_cursor(7, true);
        assert!(pane.release_drag());
        assert_eq!(
            pane.text.selection(),
            Some((0, 7)),
            "the sweep reaches back over `## `"
        );
    }

    #[test]
    fn sweeping_a_list_items_label_selects_only_the_label() {
        // A link at the start of a bullet, swept from its first letter to past the
        // line's end, copies the link and not the "- " the disc stands in for.
        let (mut pane, fonts, rect) = laid_out("- example.com/a\nafter\n", 21);
        let (px, py) = over(&pane, &fonts, 2);
        pane.press_at(&fonts, rect, px, py, false);
        assert!(pane.drag_to(&fonts, rect, 700.0, py));
        assert!(pane.release_drag());
        assert_eq!(pane.text.selected_text().as_deref(), Some("example.com/a"));
    }

    #[test]
    fn sweeping_from_before_a_list_items_label_takes_whole_items() {
        // A press in the marker, short of the label, pulls the selection back over
        // the first "- " on release, so a multi-item sweep copies whole items.
        let (mut pane, fonts, rect) = laid_out("- one\n- two\n- three\nafter\n", 24);
        let (px, py) = over(&pane, &fonts, 1);
        pane.press_at(&fonts, rect, px, py, false);
        let (_, ty) = over(&pane, &fonts, 15);
        assert!(pane.drag_to(&fonts, rect, 700.0, ty));
        assert!(pane.release_drag());
        assert_eq!(
            pane.text.selected_text().as_deref(),
            Some("- one\n- two\n- three")
        );
    }

    #[test]
    fn marker_reveal_flips_across_a_list_lines_marker_boundary() {
        // Moving the caret between the marker and the label flips the reveal without
        // changing which line is touched, so the layout cache key carries this pair
        // to force a rebuild the coarse line span would miss.
        let mut pane = pane_of("- item\n");
        // Caret within the marker ("- " is bytes 0..2): revealed.
        pane.text.set_cursor(1, false);
        assert_eq!(pane.marker_reveal(pane.reveal_range()), (true, true));
        // Caret in the label: collapsed, though the touched line is unchanged.
        pane.text.set_cursor(4, false);
        assert_eq!(pane.marker_reveal(pane.reveal_range()), (false, false));
        // A drag holds it collapsed wherever the ends sit.
        pane.text.set_cursor(1, false);
        pane.drag = Some(DragSelect::text((1, 1), DragGrain::Char));
        assert_eq!(pane.marker_reveal(pane.reveal_range()), (false, false));
    }

    #[test]
    fn triple_clicking_a_heading_takes_the_whole_line_either_way() {
        // A line-grain press already starts at the line's first byte, so the marker
        // rule has nothing to add and the grain check changes nothing here.
        let mut pane = pane_of("## Some heading\nbody\n");
        let line = pane.text.select_line_at(3);
        pane.drag = Some(DragSelect::text(line, DragGrain::Line));
        assert!(pane.release_drag());
        assert_eq!(pane.text.selection(), Some((0, 15)));
    }

    #[test]
    fn releasing_without_a_drag_reports_nothing_happened() {
        let mut pane = pane_of("## Some heading\nbody\n");
        assert!(!pane.release_drag(), "no drag was in flight");
    }

    /// The document the fold tests drive, with a section deep enough to nest.
    const FOLDABLE: &str = "\
# Guide
intro
## Setup
install
### Notes
detail
## Usage
run
";

    const WITH_FRONT: &str = "\
---
title: Work log
tags: [Notebooks/Work]
---

# Monday
ran the numbers
";

    fn fold_rect() -> Rect {
        Rect {
            x: 0,
            y: 0,
            w: 600,
            h: 400,
        }
    }

    /// The logical lines the pane's laid-out window actually drew.
    fn drawn_lines(pane: &EditorPane) -> Vec<usize> {
        let rope = pane.text.rope();
        let mut seen: Vec<usize> = pane
            .layout_cache
            .as_ref()
            .expect("a built window")
            .lines()
            .iter()
            .map(|v| rope.byte_to_line(v.start))
            .collect();
        seen.dedup();
        seen
    }

    #[test]
    fn folding_a_section_removes_its_lines_and_its_height() {
        // The whole feature, driven through the pane the app drives: fold, and the
        // section's lines stop being laid out while the document gets shorter by
        // exactly their height.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        pane.ensure_layout(&fonts, rect);
        let open_height = pane.content_height;
        assert_eq!(drawn_lines(&pane), (0..9).collect::<Vec<_>>());

        // Fold `## Setup` (line 2) via the chevron's entry point.
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            drawn_lines(&pane),
            vec![0, 1, 2, 6, 7, 8],
            "the section's body and its nested heading are gone"
        );
        assert!(pane.content_height < open_height);

        // Unfolding restores the document exactly.
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), (0..9).collect::<Vec<_>>());
        assert_eq!(pane.content_height, open_height);
    }

    #[test]
    fn folding_lifts_a_caret_that_was_inside_the_section() {
        // The caret must never park on text nobody can see, so closing the section
        // under it moves it up to the heading that now hides it.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        // Park the caret on `detail`, inside `## Setup`'s section.
        let detail = pane.text.rope().line_to_byte(5);
        pane.text.set_cursor(detail + 3, false);
        pane.ensure_layout(&fonts, rect);

        // Ctrl+period folds the section the caret sits in, which is `### Notes`.
        assert!(pane.toggle_fold_at_cursor(&fonts, rect));
        let line = pane.text.rope().byte_to_line(pane.text.cursor());
        assert_eq!(line, 4, "the caret lands on the heading that hid it");

        // Folding again from there closes `### Notes`' parent, and the caret walks
        // up again rather than being left inside.
        assert!(pane.toggle_fold_at(&fonts, rect, pane.text.rope().line_to_byte(2)));
        let line = pane.text.rope().byte_to_line(pane.text.cursor());
        assert_eq!(line, 2);
    }

    #[test]
    fn a_note_opens_on_its_content_with_its_metadata_hidden() {
        // Opening a note shows the note: the caret is parked past the block, which
        // is what keeps the block shut, and no line of it is drawn.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(WITH_FRONT);
        pane.park_past_frontmatter();
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            pane.text.rope().byte_to_line(pane.text.cursor()),
            4,
            "the caret sits on the first line below the block"
        );
        assert_eq!(
            drawn_lines(&pane).first().copied(),
            Some(4),
            "the document starts at its content"
        );
    }

    #[test]
    fn the_caret_entering_the_block_shows_it_and_leaving_shuts_it_again() {
        // The live-preview rule at block scale. Arrowing back off the note's first
        // line is the way in: the block is not stepped over the way a folded section
        // is, so the caret lands on its closing delimiter and the metadata appears.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(WITH_FRONT);
        pane.park_past_frontmatter();
        pane.ensure_layout(&fonts, rect);

        pane.text.move_left(false);
        assert!(
            !pane.step_over_folds(false, false),
            "the block opens rather than being stepped over"
        );
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            pane.text.rope().byte_to_line(pane.text.cursor()),
            3,
            "the caret is on the closing delimiter"
        );
        assert_eq!(
            drawn_lines(&pane).first().copied(),
            Some(0),
            "the whole block is on screen to edit"
        );
        assert!(
            pane.cursor_window(&fonts, rect).is_some(),
            "and the caret has a line to measure against"
        );

        // Back onto the note and the block shuts behind it.
        let content = pane.text.rope().line_to_byte(5);
        pane.text.set_cursor(content, false);
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane).first().copied(), Some(4));
    }

    #[test]
    fn an_arrival_inside_the_block_finds_a_caret_to_scroll_to() {
        // The twin of the collapsed-section search bug: a caret that arrives at an
        // offset the view did not pick (a find match, a search hit, an applied
        // revision) must not land on a line nothing draws. Settling the block is part
        // of resolving what the document hides, so it happens before anything asks
        // where the caret is.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(WITH_FRONT);
        pane.park_past_frontmatter();
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            drawn_lines(&pane).first().copied(),
            Some(4),
            "shut to begin"
        );

        let hit = pane.text.rope().line_to_byte(2) + 3;
        pane.text.set_cursor(hit, false);
        pane.scroll_caret_into_view(&fonts, rect);
        assert!(
            pane.cursor_window(&fonts, rect).is_some(),
            "the hit's line is laid out"
        );
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane).first().copied(), Some(0));
    }

    #[test]
    fn any_restored_caret_inside_the_block_is_parked_on_the_content() {
        // A session's caret is not a gesture the way a fold is: the common saved
        // value is the document start, from a note opened and never moved in, and
        // restoring that with the metadata showing is worse than losing the rarer
        // caret someone really left in a tag. Both park, so a note always reopens
        // the same way.
        let mut pane = pane_of(WITH_FRONT);
        for saved in [0, pane.text.rope().line_to_byte(1)] {
            pane.text.set_cursor(saved, false);
            pane.park_past_frontmatter();
            assert_eq!(
                pane.text.rope().byte_to_line(pane.text.cursor()),
                4,
                "a caret saved at {saved}"
            );
        }
    }

    #[test]
    fn a_document_that_is_only_metadata_still_shows_it() {
        // There is no content line to park on, so the caret stays inside and the
        // block draws: hiding it would leave an empty pane with nowhere to type.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of("---\ntitle: Work log\n---");
        pane.park_past_frontmatter();
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2]);
    }

    #[test]
    fn pressing_enter_on_a_folded_heading_opens_the_section() {
        // The reported bug: Enter at the end of a collapsed heading put the new line
        // inside the section, where the fold hid it, so the caret and everything
        // typed next vanished behind the heading. The section it belongs to opens.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 6, 7, 8]);

        pane.text
            .set_cursor(pane.text.rope().line_content_range(2).1, false);
        pane.text.enter();
        pane.open_folds_at_cursor(&fonts, rect);
        pane.ensure_layout(&fonts, rect);
        assert!(
            pane.folds.is_empty(),
            "`## Setup` opened around the new line"
        );
        assert_eq!(
            pane.text.rope().byte_to_line(pane.text.cursor()),
            3,
            "the caret is on the line Enter made"
        );
        assert_eq!(
            drawn_lines(&pane),
            (0..10).collect::<Vec<_>>(),
            "and that line is on screen"
        );
    }

    #[test]
    fn opening_a_section_around_an_edit_leaves_the_folds_it_did_not_reach() {
        // Only what hides the caret opens: the new line sits in `# Guide`'s section,
        // so `# Guide` opens, while `## Setup` nested below it stays collapsed.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        for line in [0, 2] {
            let ls = pane.text.rope().line_to_byte(line);
            assert!(pane.toggle_fold_at(&fonts, rect, ls));
        }
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0]);

        pane.text
            .set_cursor(pane.text.rope().line_content_range(0).1, false);
        pane.text.enter();
        pane.open_folds_at_cursor(&fonts, rect);
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            pane.folds.lines().collect::<Vec<_>>(),
            vec![3],
            "`## Setup`, ridden down a line by the edit, is still folded"
        );
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 3, 7, 8, 9]);
    }

    #[test]
    fn backspacing_into_a_folded_section_opens_every_fold_over_it() {
        // Joining the first visible line back into the section above drags the caret
        // into text two folds deep, so both open: one pass per level, until the
        // caret is somewhere the user can see it.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        for line in [2, 4] {
            let ls = pane.text.rope().line_to_byte(line);
            assert!(pane.toggle_fold_at(&fonts, rect, ls));
        }
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 6, 7, 8]);

        // At the start of `## Usage`, the first line past both folds.
        pane.text
            .set_cursor(pane.text.rope().line_to_byte(6), false);
        pane.text.backspace();
        pane.open_folds_at_cursor(&fonts, rect);
        pane.ensure_layout(&fonts, rect);
        assert!(
            pane.folds.is_empty(),
            "both `## Setup` and `### Notes` open"
        );
        assert_eq!(drawn_lines(&pane), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn arrowing_over_a_folded_heading_steps_across_the_whole_section() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));

        // At the end of the folded `## Setup` line, one step right lands on the
        // first visible line past the section, not on its hidden first line.
        let end = pane.text.rope().line_content_range(2).1;
        pane.text.set_cursor(end, false);
        pane.text.move_right(false);
        pane.step_over_folds(true, false);
        assert_eq!(pane.text.rope().byte_to_line(pane.text.cursor()), 6);

        // And stepping back left returns to the folded heading's end, not into it.
        pane.text.move_left(false);
        pane.step_over_folds(false, false);
        assert_eq!(pane.text.cursor(), end);
    }

    /// A note whose one section is longer than any cursor window, so a caret parked
    /// in the middle of it has no laid-out line anywhere near it while it is folded.
    fn long_section() -> String {
        let mut text = String::from("# Guide\n");
        for i in 0..200 {
            text.push_str(&format!("body line {i}\n"));
        }
        text
    }

    /// What the app does to bring a caret on screen after landing it at an offset
    /// the view did not pick: `State::reveal_active_caret`.
    fn reveal_caret(pane: &mut EditorPane, fonts: &Fonts, rect: Rect) {
        pane.open_folds_at_cursor(fonts, rect);
        pane.scroll_caret_into_view(fonts, rect);
    }

    #[test]
    fn a_search_hit_inside_a_folded_section_opens_it() {
        // The reported crash: a Ctrl+Shift+F hit landed the caret deep inside a
        // collapsed section, where the window built around it laid out nothing (every
        // line in the band was hidden) and asking that empty window for the caret's
        // box took the process down. The section the hit is in opens, so the caret has
        // a line on screen to sit on.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(&long_section());
        assert!(pane.toggle_fold_at(&fonts, rect, 0));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0], "the whole body is folded away");

        // The palette's OpenMatch: the caret goes straight to the hit's offset.
        let hit = pane.text.rope().line_to_byte(150);
        pane.text.set_cursor(hit, false);
        reveal_caret(&mut pane, &fonts, rect);
        pane.ensure_layout(&fonts, rect);

        assert!(pane.folds.is_empty(), "the section holding the hit opened");
        assert!(
            drawn_lines(&pane).contains(&150),
            "the hit's line is laid out"
        );
        assert!(pane.cursor_in_window(), "and the caret is drawn on it");
    }

    #[test]
    fn a_find_match_inside_a_folded_section_opens_it() {
        // The same for Ctrl+F: the bar's match is an offset the view did not pick, so
        // a match inside a collapsed section has to open it rather than select text
        // nobody can see.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 6, 7, 8]);

        // Ctrl+F, then a query that only matches inside the folded section.
        let target = {
            let cursor = pane.text.cursor();
            pane.find.open_at(cursor, &pane.text);
            let mut target = None;
            for c in "detail".chars() {
                target = pane.find.push(c, &pane.text);
            }
            target.expect("the query matches inside the fold")
        };
        pane.text.select_range(target.0, target.1);
        reveal_caret(&mut pane, &fonts, rect);
        pane.ensure_layout(&fonts, rect);

        assert!(pane.folds.is_empty(), "the match's section opened");
        assert!(drawn_lines(&pane).contains(&5), "the match's line is drawn");
        assert!(pane.cursor_in_window());
    }

    #[test]
    fn a_caret_left_behind_a_fold_has_no_caret_box_rather_than_no_editor() {
        // The floor under the openers above, for any offset that reaches the caret
        // without passing them (a session restored against a note edited elsewhere).
        // A hidden line is not on screen, so it has no caret box and nothing scrolls;
        // what it must never do is ask a window that laid out nothing.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(&long_section());
        assert!(pane.toggle_fold_at(&fonts, rect, 0));
        pane.ensure_layout(&fonts, rect);

        pane.text
            .set_cursor(pane.text.rope().line_to_byte(150), false);
        assert!(!pane.scroll_caret_into_view(&fonts, rect));
        assert!(!pane.move_vertical(&fonts, rect, 1, false));
        assert!(!pane.move_line_edge(&fonts, rect, true, false));
        pane.ensure_layout(&fonts, rect);
        assert!(!pane.cursor_in_window(), "no caret is drawn for it");
    }

    #[test]
    fn a_fold_rides_its_heading_through_an_edit_above_it() {
        // The anchor is a line number, so text inserted above has to move it. This
        // drives the remap through `ensure_index`, where the app triggers it.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 6, 7, 8]);

        // Type two fresh lines at the very top.
        pane.text.set_cursor(0, false);
        pane.text.insert_str("added\nlines\n");
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            pane.folds.lines().collect::<Vec<_>>(),
            vec![4],
            "the anchor moved with its heading"
        );
        assert_eq!(
            drawn_lines(&pane),
            vec![0, 1, 2, 3, 4, 8, 9, 10],
            "the same section is still the hidden one"
        );
    }

    #[test]
    fn a_fold_survives_a_line_typed_directly_above_it() {
        // The reported bug, end to end: fold a section, then open a blank line right
        // above its heading. The heading moves down one line and the fold must ride
        // with it, not spring open.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0, 1, 2, 6, 7, 8]);

        // Insert a blank line immediately above `## Setup`, the adjacent case the
        // old boundary math misplaced.
        pane.text
            .set_cursor(pane.text.rope().line_to_byte(2), false);
        pane.text.insert_str("\n");
        pane.ensure_layout(&fonts, rect);
        assert_eq!(
            pane.folds.lines().collect::<Vec<_>>(),
            vec![3],
            "the fold rode its heading down instead of popping open"
        );
        assert_eq!(
            drawn_lines(&pane),
            vec![0, 1, 2, 3, 7, 8, 9],
            "the same section is still the hidden one"
        );
    }

    #[test]
    fn deleting_a_folded_heading_drops_its_fold() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        let ls = pane.text.rope().line_to_byte(2);
        assert!(pane.toggle_fold_at(&fonts, rect, ls));
        pane.ensure_layout(&fonts, rect);

        // Remove the `## ` so line 2 is ordinary prose; the section unfolds.
        pane.text.set_cursor(ls, false);
        for _ in 0..3 {
            pane.text.delete_forward();
        }
        pane.ensure_layout(&fonts, rect);
        assert!(pane.folds.is_empty(), "an anchor off a heading is dropped");
        assert_eq!(drawn_lines(&pane), (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn unfold_all_opens_every_section() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let mut pane = pane_of(FOLDABLE);
        for line in [0, 2, 4] {
            let ls = pane.text.rope().line_to_byte(line);
            pane.toggle_fold_at(&fonts, rect, ls);
        }
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), vec![0], "everything is under `# Guide`");
        assert!(pane.unfold_all());
        pane.ensure_layout(&fonts, rect);
        assert_eq!(drawn_lines(&pane), (0..9).collect::<Vec<_>>());
        assert!(!pane.unfold_all(), "nothing left to unfold");
    }

    #[test]
    fn folds_survive_a_session_round_trip() {
        // What the state file carries, driven through the same conversion the app
        // saves and restores with.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = fold_rect();
        let path = tmp_note("folds-session", FOLDABLE);
        let mut group = restore_group(
            &[store::PaneSession {
                path: Some(path.clone()),
                cursor: 0,
                scroll_y: 0,
                folds: Vec::new(),
                pinned: false,
            }],
            0,
            None,
            0,
        );
        let ls = group.active_doc_mut().text.rope().line_to_byte(2);
        assert!(group.active_doc_mut().toggle_fold_at(&fonts, rect, ls));

        let saved = group.to_session();
        assert_eq!(saved.tabs[0].folds, vec![2], "the fold is persisted");

        // Reopening the same file restores it, and it hides the same lines.
        let mut reopened = restore_group(&saved.tabs, saved.active as usize, None, 0);
        assert_eq!(
            reopened.active_doc_mut().folds.lines().collect::<Vec<_>>(),
            vec![2]
        );
        reopened.active_doc_mut().set_scale(Scale::ONE, FONT_SIZE);
        reopened.active_doc_mut().ensure_layout(&fonts, rect);
        assert_eq!(
            drawn_lines(reopened.active_doc_mut()),
            vec![0, 1, 2, 6, 7, 8]
        );
    }

    /// A pane holding `text`, with the cursor parked at the very start so no line
    /// under test is the revealed (active) one.
    fn pane_of(text: &str) -> EditorPane {
        let mut pane = EditorPane::empty();
        pane.text = TextBuffer::from_text(text.to_string());
        pane.set_scale(Scale::ONE, FONT_SIZE);
        pane.text.set_cursor(0, false);
        pane
    }

    /// A pane holding `text`, laid out in an 800x600 column with the cursor at
    /// `cursor`, ready for pointer probes against the real layout.
    fn laid_out(text: &str, cursor: usize) -> (EditorPane, Fonts, Rect) {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 600,
        };
        let mut pane = pane_of(text);
        pane.text.set_cursor(cursor, false);
        pane.ensure_layout(&fonts, rect);
        (pane, fonts, rect)
    }

    /// The pointer position over the glyph at byte `at`: a nudge right of its caret,
    /// so the hit lands on the glyph and not on the boundary with the one before, and
    /// halfway down the line.
    fn over(pane: &EditorPane, fonts: &Fonts, at: usize) -> (f32, f32) {
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        let (x, y, h) = layout.caret(fonts, pane.text.rope(), at);
        (x as f32 + 1.0, (y + h / 2) as f32)
    }

    /// The link the pointer finds over the first character of `needle` in `text`,
    /// driving the real layout so the hit-test, the reveal rule, and the code-block
    /// rule all run. `text` is laid under a spacer line that the cursor parks on, so
    /// no line under test is the revealed one (that rule has its own case below).
    fn link_over(text: &str, needle: &str) -> Option<String> {
        let doc = format!("spacer\n{text}");
        let (mut pane, fonts, rect) = laid_out(&doc, 0);
        let at = doc.find(needle).expect("needle not in text");
        let (px, py) = over(&pane, &fonts, at);
        pane.link_under_pointer(&fonts, rect, px, py)
    }

    #[test]
    fn a_bare_url_follows_on_a_click() {
        // Typed straight into the prose, with no markup at all.
        assert_eq!(
            link_over("see https://example.com/a now", "https"),
            Some("https://example.com/a".into())
        );
        // The whole URL is clickable, not just its first character.
        assert_eq!(
            link_over("see https://example.com/a now", "example"),
            Some("https://example.com/a".into())
        );
        // An inline link still resolves to its destination, not its text.
        assert_eq!(
            link_over("see [docs](https://example.com/d) now", "docs"),
            Some("https://example.com/d".into())
        );
        // Ordinary prose is not a link, so a click there just places the caret.
        assert_eq!(link_over("see https://example.com/a now", "now"), None);
    }

    #[test]
    fn a_link_whose_text_shows_another_host_does_not_follow() {
        // The reader sees paypal.com; the click would go to evil.example.
        let doc = "pay at [https://paypal.com/login](https://evil.example/) now";
        assert_eq!(link_over(doc, "paypal"), None);
        // Text that names the target's own host still follows, and so does a bare
        // URL whose path quotes another URL.
        let doc = "see [https://example.com/a](https://example.com/b) now";
        assert_eq!(
            link_over(doc, "example"),
            Some("https://example.com/b".into())
        );
        let url = "https://web.archive.org/web/2020/https://example.com/";
        let doc = format!("see {url} now");
        assert_eq!(link_over(&doc, "archive"), Some(url.into()));
    }

    #[test]
    fn only_a_links_own_glyphs_follow_it() {
        // A URL alone on a line leaves most of the line, and the margins around it,
        // empty. A click on that empty space is a click into the text, so it drops
        // the caret; only the drawn URL follows. Otherwise a note whose first line
        // is a link is nearly impossible to click into.
        let url = "https://example.com/a";
        let doc = format!("{url}\ntail");
        let (mut pane, fonts, rect) = laid_out(&doc, doc.len());
        let (px, py) = over(&pane, &fonts, 0);
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, px, py),
            Some(url.into()),
            "the URL's own glyphs follow"
        );
        let (end_x, _) = over(&pane, &fonts, url.len());
        for (name, x, y) in [
            ("past the last character", end_x + 8.0, py),
            ("the far right of the line", 780.0, py),
            ("the left margin", 1.0, py),
            ("the padding above the first line", px, 1.0),
        ] {
            assert_eq!(
                pane.link_under_pointer(&fonts, rect, x, y),
                None,
                "{name} is not the link"
            );
        }
    }

    #[test]
    fn the_run_out_past_a_wrapped_url_is_not_the_link() {
        // A URL long enough to wrap ends its last row early, leaving the rest of that
        // row blank. The blank part belongs to no glyph, so it is not the link.
        let url = "https://example.com/en/articles/9267400-move-your-personal-account-to-a-team-or-an-enterprise-organization-with-the-existing-notes";
        let doc = format!("spacer\n{url}");
        let (mut pane, fonts, rect) = laid_out(&doc, 0);
        let rows = pane
            .layout_cache
            .as_ref()
            .expect("a built layout")
            .lines()
            .len();
        // The spacer, then the URL over at least two rows.
        assert!(
            rows > 2,
            "the URL has to wrap for this case to mean anything"
        );
        // The tail row: its last glyph is the link, the blank rest of the row is not.
        let (tail_x, tail_y) = over(&pane, &fonts, doc.len() - 1);
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, tail_x, tail_y),
            Some(url.into()),
            "the wrapped tail still follows"
        );
        let (end_x, _) = over(&pane, &fonts, doc.len());
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, end_x + 8.0, tail_y),
            None,
            "the blank run-out after the tail is not the link"
        );
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, 700.0, tail_y),
            None,
            "nor is the far end of that row"
        );
    }

    #[test]
    fn a_link_in_a_table_cell_follows_from_its_own_cell() {
        // A cell's link hit-tests against the glyphs drawn in that cell, so the link
        // follows from its own text and from nowhere else in the row.
        let doc = "| a | b |\n| --- | --- |\n| [docs](https://example.com/t) | plain |\ntail";
        let (mut pane, fonts, rect) = laid_out(doc, doc.len());
        let (px, py) = over(&pane, &fonts, doc.find("docs").expect("link text"));
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, px, py),
            Some("https://example.com/t".into()),
            "the cell's link text follows"
        );
        let (qx, qy) = over(&pane, &fonts, doc.find("plain").expect("plain cell"));
        assert_eq!(
            pane.link_under_pointer(&fonts, rect, qx, qy),
            None,
            "the next cell over is not the link"
        );
    }

    #[test]
    fn a_url_in_a_fenced_code_block_is_code_not_a_link() {
        // The block is not parsed as inline markdown, so the URL renders as plain
        // code; following it would mean the pointer offering a link the painter never
        // drew. The same URL outside the fence is still a link.
        let text = "```sh\ncurl https://example.com/a\n```\nsee https://example.com/b\n";
        assert_eq!(link_over(text, "https://example.com/a"), None);
        assert_eq!(
            link_over(text, "https://example.com/b"),
            Some("https://example.com/b".into())
        );
    }

    #[test]
    fn a_url_on_the_cursors_own_line_stays_editable() {
        // The active line shows its raw source and belongs to the keyboard: clicking
        // a URL there places the caret rather than launching a browser, which is what
        // lets a URL be edited at all.
        let text = "see https://example.com/a now";
        let at = text.find("https").expect("url present");
        let (mut pane, fonts, rect) = laid_out(text, at);
        let (px, py) = over(&pane, &fonts, at);
        assert_eq!(pane.link_under_pointer(&fonts, rect, px, py), None);
    }

    /// Whether the fenced-block delimiter line starting at `fence_start` shows its
    /// ``` backticks (any visible marker run) in the pane's built layout.
    fn fence_backticks_shown(pane: &EditorPane, fence_start: usize) -> bool {
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        let line = layout
            .lines()
            .iter()
            .find(|l| l.start == fence_start)
            .expect("the fence's visual line");
        layout
            .run_slice(line)
            .iter()
            .any(|r| matches!(r.kind, markdown::SpanKind::Marker) && r.visible)
    }

    #[test]
    fn dragging_a_selection_over_a_fence_keeps_the_backticks_visible() {
        // The reported bug: selecting the ``` of a fenced block used to blank the
        // whole delimiter line for the length of the drag, so you dragged over
        // nothing and the backticks only reappeared on release.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let text = "```\ncode\n```\n";
        let rect = Rect {
            x: 0,
            y: 0,
            w: 600,
            h: 400,
        };
        let mut pane = EditorPane::empty();
        pane.text = TextBuffer::from_text(text.to_string());
        pane.set_scale(Scale::ONE, FONT_SIZE);

        // Anchor at the top and drag down across the opening fence.
        pane.text.set_cursor(0, false);
        pane.drag = Some(DragSelect::text((0, 0), DragGrain::Char));
        pane.text.set_cursor(text.len(), true);
        pane.ensure_layout(&fonts, rect);
        assert!(
            fence_backticks_shown(&pane, 0),
            "the ``` stays visible while the drag sweeps over it"
        );
    }

    #[test]
    fn a_drag_that_starts_on_a_rendered_rule_selects_out_of_it() {
        // The reported bug: pressing a rendered `---` selected the rule but armed no
        // drag, so sweeping off it selected nothing at all.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let text = "above\n---\nbelow\n";
        let rect = Rect {
            x: 0,
            y: 0,
            w: 600,
            h: 400,
        };
        let mut pane = EditorPane::empty();
        pane.text = TextBuffer::from_text(text.to_string());
        pane.set_scale(Scale::ONE, FONT_SIZE);
        pane.ensure_layout(&fonts, rect);
        let (above_y, rule_y, below_y) = {
            let layout = pane.layout_cache.as_ref().expect("a layout");
            let y = |i: usize| layout.lines()[i].baseline as f32;
            (y(0), y(1), y(2))
        };

        assert!(
            pane.press_rule(&fonts, rect, rule_y),
            "the press lands on the rendered rule"
        );
        assert_eq!(
            pane.text.selection(),
            Some((6, 9)),
            "the press selects the whole rule"
        );
        assert!(pane.dragging(), "and arms a drag anchored on it");

        // Sweeping down past the end of the line below takes the rule with it.
        assert!(pane.drag_to(&fonts, rect, 10_000.0, below_y));
        assert_eq!(pane.text.selection(), Some((6, 15)));
        // Sweeping the other way keeps the rule whole and grows upward instead.
        assert!(pane.drag_to(&fonts, rect, 0.0, above_y));
        assert_eq!(pane.text.selection(), Some((0, 9)));

        pane.end_drag();
        assert!(!pane.dragging());
        assert!(
            !pane.drag_to(&fonts, rect, 0.0, below_y),
            "a released drag no longer follows the pointer"
        );
    }

    #[test]
    fn double_clicking_a_rendered_rule_keeps_the_selection_on_the_rule() {
        // The reported bug: the first click selects the rule, and that selection
        // reveals its raw `---`, so the second click finds text where the rule was
        // and falls through to the word gesture. The dashes are separators like the
        // blank lines flanking them, so the selection ran out through both
        // paragraphs.
        let (mut pane, fonts, rect) = laid_out("above\n\n---\n\nbelow\n", 0);
        let rule_y = {
            let layout = pane.layout_cache.as_ref().expect("a layout");
            layout.lines()[2].baseline as f32
        };

        assert!(
            pane.press_rule(&fonts, rect, rule_y),
            "the first click lands on the rendered rule"
        );
        assert_eq!(pane.text.selection(), Some((7, 10)));

        // Same pointer, second click: the source now shows, so no rule is hit.
        pane.ensure_layout(&fonts, rect);
        assert!(
            !pane.press_rule(&fonts, rect, rule_y),
            "the revealed line is editable text, not a rule"
        );
        pane.press_word(&fonts, rect, 10_000.0, rule_y);
        assert_eq!(pane.text.selected_text().as_deref(), Some("---"));
    }

    #[test]
    fn a_drag_after_a_double_click_extends_by_whole_words() {
        let (mut pane, fonts, rect) = laid_out("alpha beta gamma delta\n", 0);
        // The double-click takes "beta" (bytes 6..10).
        let (px, py) = over(&pane, &fonts, 7);
        pane.press_word(&fonts, rect, px, py);
        assert_eq!(pane.text.selected_text().as_deref(), Some("beta"));
        assert!(pane.dragging(), "the double-click arms a word-grained drag");
        // Sweeping part-way into "gamma" takes the whole word, not the letters swept.
        let (gx, gy) = over(&pane, &fonts, 13);
        assert!(pane.drag_to(&fonts, rect, gx, gy));
        assert_eq!(pane.text.selected_text().as_deref(), Some("beta gamma"));
        // Sweeping back past the press pulls in "alpha" whole and keeps "beta".
        let (ax, ay) = over(&pane, &fonts, 2);
        assert!(pane.drag_to(&fonts, rect, ax, ay));
        assert_eq!(pane.text.selected_text().as_deref(), Some("alpha beta"));
    }

    #[test]
    fn a_drag_after_a_triple_click_extends_by_whole_lines() {
        let (mut pane, fonts, rect) = laid_out("first line\nsecond line\nthird line\n", 0);
        // The triple-click takes the second line (bytes 11..22).
        let (px, py) = over(&pane, &fonts, 13);
        pane.press_line(&fonts, rect, px, py);
        assert_eq!(pane.text.selected_text().as_deref(), Some("second line"));
        // Sweeping part-way into the third line takes all of it.
        let (tx, ty) = over(&pane, &fonts, 25);
        assert!(pane.drag_to(&fonts, rect, tx, ty));
        assert_eq!(
            pane.text.selected_text().as_deref(),
            Some("second line\nthird line")
        );
    }

    /// An image store holding one decoded 80x80 picture under cat.png.
    fn cat_store() -> ImageStore {
        ImageStore::from_decoded(vec![(
            "cat.png".to_string(),
            crate::render::image::Image {
                pixels: vec![0; 80 * 80 * 4],
                width: 80,
                height: 80,
            },
        )])
    }

    #[test]
    fn a_drag_off_a_pressed_picture_hands_the_selection_to_the_text() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 600,
        };
        let mut pane = pane_of("![x](cat.png)\nafter\n");
        pane.images = cat_store();
        pane.ensure_layout(&fonts, rect);
        // A point on the picture itself, found the way the pointer finds it.
        let (ix, iy) = {
            let layout = pane.layout_cache.as_ref().expect("a built layout");
            let y = (0..rect.h)
                .map(|y| y as f32)
                .find(|&y| layout.image_at(&fonts, 60.0, y).is_some())
                .expect("the picture's rows");
            (60.0, y + 4.0)
        };

        assert_eq!(
            pane.image_press(&fonts, rect, ix, iy),
            ImagePress::Picture,
            "the press lands on the picture"
        );
        assert_eq!(pane.selected_picture(), Some(0), "which the press selects");
        assert!(pane.text.selection().is_none(), "with no text selected yet");
        assert!(pane.dragging(), "and a drag anchored on the image's line");

        // A jitter that stays on the picture leaves it selected, rather than
        // text-selecting the collapsed source line under it.
        assert!(!pane.drag_to(&fonts, rect, ix + 1.0, iy + 1.0));
        assert_eq!(pane.selected_picture(), Some(0));
        assert!(pane.text.selection().is_none());

        // Sweeping into the text below hands over to a text selection, which carries
        // the image line with it.
        let (tx, ty) = over(&pane, &fonts, 19);
        assert!(pane.drag_to(&fonts, rect, tx, ty));
        assert_eq!(
            pane.selected_picture(),
            None,
            "the picture's border gives way"
        );
        assert_eq!(pane.text.selection(), Some((0, 19)));

        // Sweeping back onto the picture takes it as a picture again.
        assert!(pane.drag_to(&fonts, rect, ix, iy));
        assert_eq!(pane.selected_picture(), Some(0));
        assert!(pane.text.selection().is_none());
    }

    /// A note that opens with an 80px picture over a line of text, laid out in an
    /// 800x600 column with the caret parked in that text at byte 16, so the picture
    /// shows alone with its source collapsed. Also returns the picture's rectangle.
    fn picture_note() -> (EditorPane, Fonts, Rect, Rect) {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 600,
        };
        let mut pane = pane_of("![x](cat.png)\nafter\n");
        pane.images = cat_store();
        pane.text.set_cursor(16, false);
        pane.ensure_layout(&fonts, rect);
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        let picture = layout
            .image_rect(&fonts, &layout.lines()[0])
            .expect("the note opens with a picture alone");
        (pane, fonts, rect, picture)
    }

    #[test]
    fn only_a_point_on_the_picture_is_under_the_pointer() {
        let (mut pane, fonts, rect, pic) = picture_note();
        let mid_y = (pic.y + pic.h / 2) as f32;
        let mid = ((pic.x + pic.w / 2) as f32, mid_y);
        assert_eq!(
            pane.picture_under_pointer(&fonts, rect, mid.0, mid.1),
            Some(0)
        );
        // Beside the picture on its own line, and on the text below it, the right
        // click is on the note, not on the picture.
        let beside = ((pic.x + pic.w + 40) as f32, mid_y);
        assert_eq!(
            pane.picture_under_pointer(&fonts, rect, beside.0, beside.1),
            None
        );
        let (tx, ty) = over(&pane, &fonts, 16);
        assert_eq!(pane.picture_under_pointer(&fonts, rect, tx, ty), None);
    }

    #[test]
    fn a_press_around_a_picture_selects_it() {
        // Above the picture or level with its upper half selects it on its side
        // above (0); below it or level with its lower half on its side below (13).
        // Neither reveals the source.
        let (_, _, _, pic) = picture_note();
        let (upper, lower) = ((pic.y + 5) as f32, (pic.y + pic.h - 5) as f32);
        let probes = [
            ("the margin above the note", ((pic.x + 10) as f32, 1.0), 0),
            (
                "the padding above it",
                ((pic.x + pic.w - 5) as f32, (pic.y - 2) as f32),
                0,
            ),
            (
                "the margin left of its upper half",
                ((pic.x - 10) as f32, upper),
                0,
            ),
            (
                "the space right of its lower half",
                ((pic.x + pic.w + 40) as f32, lower),
                13,
            ),
            (
                "the padding below it",
                ((pic.x + 5) as f32, (pic.y + pic.h + 2) as f32),
                13,
            ),
        ];
        for (place, (px, py), on) in probes {
            let (mut pane, fonts, rect, _) = picture_note();
            assert_eq!(
                pane.image_press(&fonts, rect, px, py),
                ImagePress::Missed,
                "{place}: off the picture, so the press is the text's"
            );
            assert_eq!(
                pane.cursor_shape_at(&fonts, rect, px, py),
                wp_cursor_shape_device_v1::SHAPE_TEXT,
                "{place}: the I-beam, since a click there places the caret"
            );
            pane.press_at(&fonts, rect, px, py, false);
            pane.release_drag();
            assert_eq!(pane.text.cursor(), on, "{place}");
            assert_eq!(pane.selected_picture(), Some(0), "{place}: selected");
            pane.ensure_layout(&fonts, rect);
            let layout = pane.layout_cache.as_ref().expect("a built layout");
            assert!(
                layout.lines()[0].image.is_some_and(|img| img.selected),
                "{place}: the picture shows alone and selected, its source collapsed"
            );
        }
    }

    #[test]
    fn a_double_click_around_a_picture_selects_it_as_a_click_does() {
        // A word of the hidden source would reveal it.
        let (mut pane, fonts, rect, pic) = picture_note();
        let (px, py) = ((pic.x + pic.w + 40) as f32, (pic.y + pic.h - 5) as f32);
        pane.press_word(&fonts, rect, px, py);
        pane.release_drag();
        assert_eq!(pane.text.selection(), None);
        assert_eq!((pane.text.cursor(), pane.selected_picture()), (13, Some(0)));
    }

    #[test]
    fn clicking_a_picture_closes_the_source_the_caret_was_editing() {
        // Two pictures, a on 0..13 and b on 14..27. The caret inside b's source shows
        // it. A click on a moves the caret onto a, so b's source closes and stays
        // closed through the clicks that follow.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 600,
        };
        let mut pane = pane_of("![a](cat.png)\n![b](cat.png)\n");
        pane.images = cat_store();
        pane.text.set_cursor(18, false);
        let b_open = |pane: &mut EditorPane| {
            pane.ensure_layout(&fonts, rect);
            let layout = pane.layout_cache.as_ref().expect("a built layout");
            layout
                .lines()
                .iter()
                .any(|l| l.start == 14 && l.image.is_none())
        };
        assert!(
            b_open(&mut pane),
            "b's source shows while the caret is in it"
        );
        for (click, height, on) in [(0, 0.25, 0), (1, 0.25, 0), (2, 0.75, 13)] {
            let a = {
                let layout = pane.layout_cache.as_ref().expect("a built layout");
                layout
                    .image_rect(&fonts, &layout.lines()[0])
                    .expect("a is the first row")
            };
            // Clear of the corner button, which a selected picture draws.
            let (px, py) = ((a.x + 10) as f32, a.y as f32 + a.h as f32 * height);
            assert_eq!(pane.image_press(&fonts, rect, px, py), ImagePress::Picture);
            pane.release_drag();
            assert_eq!(pane.selected_picture(), Some(0), "click {click} selects a");
            assert_eq!(
                pane.text.cursor(),
                on,
                "click {click}: the side by the half"
            );
            assert!(!b_open(&mut pane), "click {click}: b's source stays closed");
        }
    }

    #[test]
    fn the_button_opens_a_selected_pictures_source_and_closes_it_again() {
        let (mut pane, fonts, rect, pic) = picture_note();
        let button = |pane: &mut EditorPane| {
            pane.ensure_layout(&fonts, rect);
            let layout = pane.layout_cache.as_ref().expect("a built layout");
            let row = layout
                .lines()
                .iter()
                .find(|l| l.image.is_some())
                .expect("the picture's row");
            let pic = layout.image_rect(&fonts, row).expect("the picture");
            let b = crate::layout::image_button_rect(&pic, Scale::ONE);
            ((b.x + 2) as f32, (b.y + 2) as f32)
        };
        // A click on the upper half selects the picture, which draws the button.
        let (px, py) = ((pic.x + 10) as f32, (pic.y + 10) as f32);
        assert_eq!(pane.image_press(&fonts, rect, px, py), ImagePress::Picture);
        pane.release_drag();
        assert_eq!(pane.selected_picture(), Some(0));

        // The button opens the source: the caret is now at its start, as text.
        let (bx, by) = button(&mut pane);
        assert_eq!(pane.image_press(&fonts, rect, bx, by), ImagePress::Button);
        assert_eq!(pane.raw_image, Some(0));
        assert_eq!(pane.selected_picture(), None, "the caret is in the source");

        // The button stays drawn while the source is open, and closes it from
        // inside, putting the caret back on the picture.
        pane.text.set_cursor(5, false);
        let (bx, by) = button(&mut pane);
        assert_eq!(pane.image_press(&fonts, rect, bx, by), ImagePress::Button);
        assert_eq!(pane.raw_image, None);
        assert_eq!((pane.text.cursor(), pane.selected_picture()), (13, Some(0)));
        pane.ensure_layout(&fonts, rect);
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        assert!(
            layout.lines()[0].image.is_some(),
            "the source is collapsed again"
        );
    }

    #[test]
    fn a_press_below_a_note_ending_on_a_picture_goes_to_the_end() {
        // The page under the note belongs to the text: a press there lands at the
        // document's end, after the picture, which reveals nothing.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 800,
            h: 600,
        };
        let mut pane = pane_of("top\n![x](cat.png)");
        pane.images = cat_store();
        pane.ensure_layout(&fonts, rect);
        let (px, py) = (60.0, 560.0);
        assert_eq!(pane.image_press(&fonts, rect, px, py), ImagePress::Missed);
        pane.press_at(&fonts, rect, px, py, false);
        pane.release_drag();
        assert_eq!(pane.text.cursor(), 17, "after the picture");
        pane.ensure_layout(&fonts, rect);
        let layout = pane.layout_cache.as_ref().expect("a built layout");
        assert!(layout.lines()[1].image.is_some() && layout.lines().len() == 2);
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
        g.docs().map(|d| d.scroll_y).collect()
    }

    /// The strip's labels, in order.
    fn tab_labels(g: &PaneGroup) -> Vec<&str> {
        g.tab_views().map(|t| t.label).collect()
    }

    /// Which tabs are pinned, in order.
    fn pinned_flags(g: &PaneGroup) -> Vec<bool> {
        g.tab_views().map(|t| t.pinned).collect()
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
        {
            let mut docs = b.docs_mut();
            docs.next().unwrap().scroll_y = 10;
            docs.next().unwrap().scroll_y = 11;
        }

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
    fn toggle_pin_flips_only_the_named_tab() {
        let mut g = marked_group(3);
        assert!(!g.is_pinned(1));
        g.toggle_pin(1);
        assert_eq!(pinned_flags(&g), vec![false, true, false]);
        g.toggle_pin(1);
        assert!(!g.is_pinned(1));
        // An out-of-range toggle changes nothing and reads as unpinned.
        g.toggle_pin(9);
        assert_eq!(pinned_flags(&g), vec![false, false, false]);
        assert!(!g.is_pinned(9));
    }

    #[test]
    fn a_pin_rides_its_tab_through_a_reorder() {
        // Pin the first tab, then drag it to the last slot: the pin follows the tab,
        // not the slot it left.
        let mut g = marked_group(4);
        g.toggle_pin(0);
        g.move_tab(0, 3);
        assert_eq!(order(&g), vec![1, 2, 3, 0]);
        assert_eq!(pinned_flags(&g), vec![false, false, false, true]);
    }

    #[test]
    fn a_pin_crosses_the_split_with_its_tab() {
        // A tab dragged to the other pane arrives as a fresh, unpinned tab; the drag
        // path re-applies the pin, so carrying its state is set_pinned's job.
        let mut a = marked_group(2);
        a.toggle_pin(0);
        let pinned = a.is_pinned(0);
        let doc = a.detach(0).expect("a tab to detach");
        let mut b = marked_group(1);
        let landed = b.insert_tab(0, doc);
        assert!(!b.is_pinned(landed), "arrives unpinned");
        b.set_pinned(landed, pinned);
        assert!(b.is_pinned(landed), "the pin is carried over");
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
        let cols = panes.columns(1000, 700, 0, Scale::ONE, None);
        // Every column is measured from the rail's right edge, not the window's.
        let tree = cols.tree.expect("tree shown");
        assert_eq!((tree.x, tree.w, tree.h), (RAIL_W, 200, 700));
        assert_eq!(cols.splitters[0].expect("tree splitter").x, RAIL_W + 200);
        let a = cols.editors[0].expect("editor A");
        let b = cols.editors[1].expect("editor B");
        assert_eq!(a.x, RAIL_W + 200 + SPLITTER);
        assert_eq!(cols.splitters[1].expect("content splitter").x, a.x + a.w);
        // The content area splits evenly, and every column is full height.
        assert!((a.w - b.w).abs() <= 1);
        assert_eq!(a.h, 700);
        assert_eq!(a.x + a.w + SPLITTER + b.w, 1000);
    }

    #[test]
    fn soloing_a_pane_keeps_the_tree_and_drops_the_other_pane() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        // A diff review solos pane 1: the tree stays (the file explorer is not hidden),
        // the other editor and the pane splitter fall away, and the diff pane spans from
        // the tree's right edge to the window.
        let cols = panes.columns(1000, 700, 30, Scale::ONE, Some(1));
        let tree = cols
            .tree
            .expect("the tree stays visible during a diff review");
        assert_eq!((tree.x, tree.w), (RAIL_W, 200));
        assert!(cols.editors[0].is_none(), "the other pane is hidden");
        assert!(cols.splitters[1].is_none(), "no pane splitter");
        let diff = cols.editors[1].expect("the soloed pane");
        assert_eq!(diff.x, RAIL_W + 200 + SPLITTER);
        assert_eq!(diff.x + diff.w, 1000);
        assert!(cols.tab_strips[1].is_some() && cols.tab_strips[0].is_none());
    }

    #[test]
    fn the_rail_is_the_outermost_column_and_the_only_full_height_one() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE, None);
        // The rail runs the whole window height, beside the bar rather than under
        // it, and the bar starts where the rail ends.
        assert_eq!(
            (cols.rail.x, cols.rail.y, cols.rail.w, cols.rail.h),
            (0, 0, RAIL_W, 700)
        );
        assert_eq!(
            (cols.bar.x, cols.bar.y, cols.bar.w, cols.bar.h),
            (RAIL_W, 0, 1000 - RAIL_W, 30)
        );
        // Nothing else reaches into it: the sidebar starts at its edge, every
        // column still ends at the window's, and the bar's strip stays clear of it.
        let tree = cols.tree.expect("tree shown");
        let a = cols.editors[0].expect("editor A");
        let strip = cols.tab_strips[0].expect("strip A");
        assert_eq!(tree.x, cols.rail.x + cols.rail.w);
        assert!(strip.x >= cols.rail.x + cols.rail.w);
        assert_eq!(strip.x + strip.w, a.x + a.w);
        assert_eq!(a.x + a.w, 1000);
    }

    #[test]
    fn the_rail_keeps_its_chips_on_the_columns_first_line() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE, None);
        // The rail is drawn full height, but its chips take the column beneath the
        // bar, so they start where the tree and the editors do.
        let chips = cols.rail_chips;
        assert_eq!((chips.x, chips.w), (cols.rail.x, cols.rail.w));
        assert_eq!((chips.y, chips.h), (30, 670));
        assert_eq!(chips.y, cols.tree.expect("tree shown").y);
        assert_eq!(chips.y, cols.editors[0].expect("editor A").y);
        assert_eq!(chips.y + chips.h, cols.rail.y + cols.rail.h);
    }

    #[test]
    fn the_rail_never_outgrows_a_narrow_window() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: false,
        };
        // A window squeezed narrower than the rail gives it what there is, and the
        // columns beside it collapse to nothing rather than to a negative width.
        let cols = panes.columns(20, 700, 30, Scale::ONE, None);
        assert_eq!(cols.rail.w, 20);
        assert_eq!((cols.bar.x, cols.bar.w), (20, 0));
        let a = cols.editors[0].expect("editor A");
        assert_eq!(a.w, 0);
    }

    #[test]
    fn the_rail_scales_with_the_display() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: false,
        };
        let scale = Scale::from_120(240);
        let cols = panes.columns(2000, 1400, 60, scale, None);
        assert_eq!(cols.rail.w, scale.px(RAIL_W));
        assert_eq!(cols.bar.x, scale.px(RAIL_W));
        assert_eq!(cols.tree.expect("tree shown").x, scale.px(RAIL_W));
    }

    #[test]
    fn columns_reserve_the_top_bar_and_align_strips() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE, None);
        // Every column starts below the bar and is that much shorter.
        let tree = cols.tree.expect("tree shown");
        assert_eq!((tree.y, tree.h), (30, 670));
        let a = cols.editors[0].expect("editor A");
        let b = cols.editors[1].expect("editor B");
        assert_eq!((a.y, a.h), (30, 670));
        // Each pane's tab strip sits above its column at the bar height, starting at
        // the divider line so the first tab abuts it with no gap.
        let strip_a = cols.tab_strips[0].expect("strip A");
        let strip_b = cols.tab_strips[1].expect("strip B");
        let line_a = divider_line(cols.splitters[0].expect("splitter 0"), Scale::ONE);
        let line_b = divider_line(cols.splitters[1].expect("splitter 1"), Scale::ONE);
        let (left_a, left_b) = (line_a.x + line_a.w, line_b.x + line_b.w);
        assert_eq!(
            (strip_a.x, strip_a.y, strip_a.w, strip_a.h),
            (left_a, 0, a.w + (a.x - left_a), 30)
        );
        assert_eq!(
            (strip_b.x, strip_b.w, strip_b.h),
            (left_b, b.w + (b.x - left_b), 30)
        );
        // Each strip still ends at its column's right edge, so they never overlap.
        assert_eq!(strip_a.x + strip_a.w, a.x + a.w);
        assert_eq!(strip_b.x + strip_b.w, b.x + b.w);
        assert!(strip_a.x + strip_a.w <= strip_b.x);
    }

    #[test]
    fn hidden_sidebar_fills_the_strip_beside_the_rail() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: false,
            second_open: false,
        };
        let cols = panes.columns(1000, 700, 30, Scale::ONE, None);
        let strip_a = cols.tab_strips[0].expect("strip A");
        assert_eq!((strip_a.x, strip_a.w), (RAIL_W, 1000 - RAIL_W));
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
        assert_eq!(tab_labels(&group), vec!["a"]);
        assert_eq!(group.active_index(), 0);
        // A second, different file stacks a new tab and focuses it.
        group.open_or_focus(&b).expect("open b");
        assert_eq!(tab_labels(&group), vec!["a", "b"]);
        assert_eq!(group.active_index(), 1);
        // Reopening the first file focuses its existing tab, no duplicate.
        group.open_or_focus(&a).expect("reopen a");
        assert_eq!(tab_labels(&group).len(), 2);
        assert_eq!(group.active_index(), 0);
        // A fresh blank tab (the new-tab button) is replaced in place by an open.
        group.new_tab(Scale::ONE, FONT_SIZE);
        assert_eq!(tab_labels(&group), vec!["a", "b", "untitled"]);
        let c = dir.join("c.md");
        std::fs::write(&c, b"gamma").expect("write c");
        group.open_or_focus(&c).expect("open c");
        assert_eq!(tab_labels(&group), vec!["a", "b", "c"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_pinned_blank_tab_is_not_replaced_by_an_open() {
        let dir = std::env::temp_dir().join(format!("bnknotes_pinblank_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let a = dir.join("a.md");
        std::fs::write(&a, b"alpha").expect("write a");

        let mut group = PaneGroup::empty();
        group.new_tab(Scale::ONE, FONT_SIZE);
        group.toggle_pin(0);
        // Pinning the blank tab asks to keep it: the open stacks a new tab rather
        // than filling the blank in place.
        group.open_or_focus(&a).expect("open a");
        assert_eq!(tab_labels(&group), vec!["untitled", "a"]);
        assert_eq!(pinned_flags(&group), vec![true, false]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn to_session_carries_each_tabs_pin() {
        let mut group = PaneGroup::empty();
        group.new_tab(Scale::ONE, FONT_SIZE);
        group.new_tab(Scale::ONE, FONT_SIZE);
        group.toggle_pin(0);
        let session = group.to_session();
        assert_eq!(
            session.tabs.iter().map(|t| t.pinned).collect::<Vec<_>>(),
            vec![true, false]
        );
    }

    #[test]
    fn closing_the_last_tab_empties_the_pane() {
        let mut group = PaneGroup::empty();
        group.new_tab(Scale::ONE, FONT_SIZE);
        assert!(group.has_tabs());
        group.close(0);
        assert!(!group.has_tabs());
        assert!(tab_labels(&group).is_empty());
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
        assert_eq!(tab_labels(&group), vec!["b", "c"]);
        assert_eq!(group.active_index(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A diff review over the given group's active document.
    fn open_review(group: &mut PaneGroup) {
        let source = group.id;
        let version = group.text.version();
        let view = DiffView::new(
            source,
            (0, 7),
            version,
            "teh cat".into(),
            "the cat".into(),
            "Fix spelling & grammar",
        );
        group.open_diff(view);
    }

    #[test]
    fn a_diff_review_sits_by_its_source_and_never_persists() {
        let a = tmp_note("diff-src", "teh cat\n");
        let mut group = PaneGroup::empty();
        group.open_or_focus(&a).expect("open the source");
        open_review(&mut group);
        // It is active, is a diff, sits right after its source, and shows the label.
        assert!(group.active_is_diff());
        assert_eq!(group.active_index(), 1);
        assert_eq!(tab_labels(&group), vec!["note", "Fix spelling & grammar"]);
        // The document list skips it, so a search or the watcher never sees it.
        assert_eq!(group.docs().count(), 1);
        // Only the document persists; the active index maps into the doc-only list, so
        // an active diff points at its source rather than off the end.
        let session = group.to_session();
        assert_eq!(session.tabs.len(), 1);
        assert_eq!(session.active, 0);
    }

    #[test]
    fn detaching_a_diff_review_removes_it_and_yields_no_document() {
        let a = tmp_note("diff-detach", "teh cat\n");
        let mut group = PaneGroup::empty();
        group.open_or_focus(&a).expect("open the source");
        open_review(&mut group);
        assert!(group.tab_is_diff(1));
        // A diff re-homes to no document: detach removes it but hands back nothing.
        assert!(group.detach(1).is_none());
        assert_eq!(group.tab_count(), 1);
        assert!(!group.active_is_diff());
    }

    #[test]
    fn the_two_panes_mint_distinct_ids_that_survive_a_move() {
        let mut a = PaneGroup::with_seed(0);
        let mut b = PaneGroup::with_seed(1);
        a.new_tab(Scale::ONE, FONT_SIZE);
        b.new_tab(Scale::ONE, FONT_SIZE);
        // Even in A, odd in B: the two panes can never mint the same id.
        assert_ne!(a.id, b.id);
        // A document keeps its id when it is dragged across the split.
        let before = a.id;
        let doc = a.detach(0).expect("a document tab");
        assert_eq!(doc.id, before);
        b.insert_tab(0, doc);
        assert_eq!(b.id, before, "the id survives the move");
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
        assert_eq!(group.active_doc_mut().body_size, 21);
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
        let cols = panes.columns(1000, 700, 0, Scale::ONE, None);
        assert!(cols.tree.is_none());
        assert!(cols.splitters[0].is_none() && cols.splitters[1].is_none());
        let a = cols.editors[0].expect("editor A");
        // Editor A takes everything beside the rail, which is never hidden.
        assert_eq!((a.x, a.w), (RAIL_W, 1000 - RAIL_W));
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
    /// The caret width the tests animate with.
    const CARET_W: i32 = 2;

    /// The target bar the tests ease toward, at the given content-space top-left.
    fn bar(x: f32, y: f32) -> CaretBox {
        CaretBox::bar(x, y, CARET_W as f32, CARET_H)
    }

    /// A frame under the constant geometry key: a caret at `target` with logical
    /// offset `cursor`, and whether a pointer drove the move.
    fn frame(target: (f32, f32), cursor: usize, from_pointer: bool) -> CaretFrame {
        CaretFrame {
            target,
            height: CARET_H,
            width: CARET_W,
            cursor,
            from_pointer,
            key: CARET_KEY,
        }
    }

    /// A keyboard move frame: the offset moved and no pointer drove it.
    fn nav(target: (f32, f32), cursor: usize) -> CaretFrame {
        frame(target, cursor, false)
    }

    /// An animating config with the given jump cap (in caret heights). A large cap
    /// takes the distance guard out of the way for the easing tests.
    fn anim_cfg(max_jump_lines: f32) -> CaretConfig {
        CaretConfig {
            animate: true,
            rate: 40.0,
            spring: 0.65,
            max_jump_lines,
        }
    }

    /// Drive the blink past one interval, the way the loop's timer wake does.
    fn blink_step(b: &mut CaretBlink, t: &mut Instant) -> bool {
        *t += CARET_BLINK;
        b.tick(*t, false)
    }

    #[test]
    fn blink_toggles_on_the_interval() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        assert!(b.visible(), "activity shows the caret");
        // Short of the deadline nothing moves; past it the caret toggles and asks
        // for the repaint.
        t += CARET_BLINK / 2;
        assert!(!b.tick(t, false));
        assert!(b.visible());
        assert!(blink_step(&mut b, &mut t), "a toggle repaints");
        assert!(!b.visible(), "the caret blinks off");
        assert!(blink_step(&mut b, &mut t));
        assert!(b.visible(), "and back on");
    }

    #[test]
    fn blink_holds_the_caret_shown_while_it_eases() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        // Long enough to be several toggles overdue: an easing caret must not blink
        // out mid-motion, and it comes out of the ease with a full run.
        t += CARET_BLINK * 3;
        assert!(!b.tick(t, true), "already shown, so nothing repaints");
        assert!(b.visible());
        assert_eq!(b.wake_at(), Some(t + CARET_BLINK));
        for _ in 0..(CARET_BLINK_TOGGLES - 1) {
            blink_step(&mut b, &mut t);
        }
        assert!(b.wake_at().is_some(), "the ease re-ran the whole run");
    }

    #[test]
    fn blink_rests_shown_and_timerless_after_its_run() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        for _ in 0..CARET_BLINK_TOGGLES {
            blink_step(&mut b, &mut t);
        }
        assert!(
            b.visible(),
            "an untouched caret rests shown, not blinked off"
        );
        assert_eq!(b.wake_at(), None, "a rested caret arms no timer");
        // Later ticks stay quiet: this is the state that takes an idle focused
        // window to zero wake-ups.
        assert!(!blink_step(&mut b, &mut t));
        assert!(!blink_step(&mut b, &mut t));
        assert!(b.visible());
        assert_eq!(b.wake_at(), None);
    }

    #[test]
    fn blink_resumes_on_activity_after_resting() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        for _ in 0..CARET_BLINK_TOGGLES {
            blink_step(&mut b, &mut t);
        }
        assert_eq!(b.wake_at(), None);
        b.bump(t);
        assert_eq!(b.wake_at(), Some(t + CARET_BLINK));
        assert!(blink_step(&mut b, &mut t), "typing starts a fresh run");
        assert!(!b.visible());
    }

    #[test]
    fn blink_hidden_by_a_lost_focus_shows_nothing_and_arms_nothing() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        b.hide();
        assert!(!b.visible());
        assert_eq!(b.wake_at(), None);
        assert!(
            !blink_step(&mut b, &mut t),
            "an unfocused window never toggles"
        );
        assert!(!b.visible());
    }

    #[test]
    fn blink_rest_brings_the_caret_back_shown() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        blink_step(&mut b, &mut t);
        assert!(!b.visible(), "caught on the blinked-off half");
        // Focus moves to the tree: the caret rests shown, so it is solid when the
        // editor takes focus back rather than invisible for half a cycle.
        b.rest();
        assert!(b.visible());
        assert_eq!(b.wake_at(), None);
    }

    #[test]
    fn blink_reports_a_repaint_only_when_the_caret_changed() {
        let mut t = Instant::now();
        let mut b = CaretBlink::new();
        b.bump(t);
        // The tick that only arms, and every tick short of the deadline, changes
        // nothing: a blink must not rebuild a frame that comes out identical.
        assert!(!b.tick(t, false));
        t += CARET_BLINK / 4;
        assert!(!b.tick(t, false));
        assert!(blink_step(&mut b, &mut t));
        // The last toggle of a run lands on shown, which is a change only when the
        // caret was off.
        let mut spent = CaretBlink::new();
        let mut u = Instant::now();
        spent.bump(u);
        let changes = (0..CARET_BLINK_TOGGLES)
            .filter(|_| blink_step(&mut spent, &mut u))
            .count();
        assert_eq!(
            changes as u32, CARET_BLINK_TOGGLES,
            "every toggle of a run flips the caret"
        );
    }

    #[test]
    fn caret_initializes_at_target() {
        let mut a = CaretAnim::new();
        a.update(nav((100.0, 50.0), 0), &anim_cfg(1000.0), Instant::now());
        // First update seeds the rendered box at the live target: settled, no smear
        // from the default origin.
        assert_eq!(a.pos, Some(bar(100.0, 50.0)));
        assert!(!a.in_flight());
        let r = a.draw_rect();
        assert_eq!(r.w, 2);
        assert_eq!(r.h, 20);
    }

    #[test]
    fn caret_converges_to_target() {
        let cfg = anim_cfg(1000.0);
        let mut a = CaretAnim::new();
        let mut t = Instant::now();
        a.update(nav((0.0, 0.0), 0), &cfg, t);
        a.update(nav((200.0, 0.0), 1), &cfg, t); // keyboard nav: flight begins
        for _ in 0..200 {
            t += Duration::from_millis(16);
            a.update(nav((200.0, 0.0), 1), &cfg, t); // same offset: keep easing
            if !a.in_flight() {
                break;
            }
        }
        assert_eq!(a.pos, Some(bar(200.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_easing_is_frame_rate_independent() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        // One 32ms step (after the shared nominal first step).
        let mut one = CaretAnim::new();
        one.update(nav((0.0, 0.0), 0), &cfg, t0);
        one.update(nav((300.0, 0.0), 1), &cfg, t0);
        one.update(nav((300.0, 0.0), 1), &cfg, t0 + Duration::from_millis(32));
        // Two 16ms steps.
        let mut two = CaretAnim::new();
        two.update(nav((0.0, 0.0), 0), &cfg, t0);
        two.update(nav((300.0, 0.0), 1), &cfg, t0);
        two.update(nav((300.0, 0.0), 1), &cfg, t0 + Duration::from_millis(16));
        two.update(nav((300.0, 0.0), 1), &cfg, t0 + Duration::from_millis(32));
        // Exponential decay composes over dt, so from the shared first step the same
        // elapsed time lands in the same place however it is subdivided.
        assert!((one.pos.unwrap().left - two.pos.unwrap().left).abs() < 0.5);
    }

    #[test]
    fn caret_settles_within_half_a_pixel() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // Sit just under a pixel from the target while easing (offset unchanged from
        // the move, so the in-flight step continues); the next step crosses the
        // half-pixel threshold and snaps exactly, with no asymptotic tail.
        a.pos = Some(bar(99.7, 0.0));
        a.in_flight = true;
        a.last_tick = Some(t0);
        a.last_cursor = Some(1);
        a.update(nav((100.0, 0.0), 1), &cfg, t0 + Duration::from_millis(16));
        assert_eq!(a.pos, Some(bar(100.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_requested_snap_lands_at_target() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        a.update(nav((50.0, 0.0), 1), &cfg, t0); // in flight
        a.request_snap();
        a.update(nav((50.0, 0.0), 1), &cfg, t0);
        assert_eq!(a.pos, Some(bar(50.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_jump_beyond_cap_snaps() {
        let cfg = anim_cfg(6.0); // cap = 6 * 20px = 120px
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        a.update(nav((500.0, 0.0), 1), &cfg, t0); // keyboard nav 500px > cap
        assert_eq!(a.pos, Some(bar(500.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_dt_is_capped_and_never_overshoots() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        // A capped (50ms) step, after the nominal first step.
        let mut capped = CaretAnim::new();
        capped.update(nav((0.0, 0.0), 0), &cfg, t0);
        capped.update(nav((400.0, 0.0), 1), &cfg, t0);
        capped.update(nav((400.0, 0.0), 1), &cfg, t0 + Duration::from_millis(50));
        // A huge elapsed duration clamps to the same cap.
        let mut huge = CaretAnim::new();
        huge.update(nav((0.0, 0.0), 0), &cfg, t0);
        huge.update(nav((400.0, 0.0), 1), &cfg, t0);
        huge.update(nav((400.0, 0.0), 1), &cfg, t0 + Duration::from_secs(10));
        assert!((capped.pos.unwrap().left - huge.pos.unwrap().left).abs() < 1e-3);
        // The exponential step approaches but never passes the target.
        assert!(huge.pos.unwrap().left < 400.0);
    }

    #[test]
    fn caret_draw_rect_is_the_bar_at_rest_and_a_smear_in_flight() {
        let cfg = anim_cfg(1000.0);
        let mut a = CaretAnim::new();
        a.update(nav((100.0, 40.0), 0), &cfg, Instant::now());
        // At rest the box is exactly the caret bar.
        assert_eq!(
            a.draw_rect(),
            Rect {
                x: 100,
                y: 40,
                w: 2,
                h: 20
            }
        );
        // A rightward in-flight box: the leading right edge has run ahead of the
        // trailing left edge, so the drawn rect spans from left to right.
        a.pos = Some(CaretBox {
            left: 100.0,
            top: 40.0,
            right: 148.0,
            bottom: 60.0,
        });
        a.in_flight = true;
        let smear = a.draw_rect();
        assert_eq!(smear.x, 100);
        assert_eq!(smear.x + smear.w, 148);
        assert_eq!(smear.y, 40);
        assert_eq!(smear.h, 20);
    }

    #[test]
    fn caret_smear_leads_in_the_travel_direction() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // Moving right, the first in-flight step stretches: both edges leave the
        // origin, but the leading right edge outruns the trailing left edge, so the
        // box is wider than the caret bar and starts near the origin, not the target.
        a.update(nav((200.0, 0.0), 1), &cfg, t0);
        let b = a.pos.unwrap();
        assert!(b.left > 0.0, "trailing edge left the origin");
        assert!(b.left < 200.0, "trailing edge has not reached the target");
        assert!(
            b.right - CARET_W as f32 > b.left,
            "leading edge outran the trailing edge (a stretch)"
        );
        assert!(
            b.right < 200.0 + CARET_W as f32,
            "leading edge still short of target"
        );
    }

    #[test]
    fn caret_drops_vertically_faster_than_it_slides() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        // The same distance, once horizontally and once vertically.
        let mut h = CaretAnim::new();
        h.update(nav((0.0, 0.0), 0), &cfg, t0);
        h.update(nav((100.0, 0.0), 1), &cfg, t0);
        let mut v = CaretAnim::new();
        v.update(nav((0.0, 0.0), 0), &cfg, t0);
        v.update(nav((0.0, 100.0), 1), &cfg, t0);
        // The vertical axis eases faster, so after the same first step the vertical
        // caret's trailing edge has covered more of the distance than the horizontal
        // one's, keeping a line change from dragging a tall smear.
        assert!(
            v.pos.unwrap().top > h.pos.unwrap().left,
            "vertical drops faster than horizontal slides"
        );
    }

    #[test]
    fn caret_typing_smears() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // Typing advances the offset with no pointer, so it is a keyboard move and
        // eases like navigation: the caret flows to the insertion point.
        a.update(nav((10.0, 0.0), 1), &cfg, t0 + Duration::from_millis(16));
        assert!(a.in_flight());
        assert!(a.pos.unwrap().left < 10.0, "still easing toward the caret");
    }

    #[test]
    fn caret_mouse_click_snaps() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // A click moves the offset, but the pointer flag makes it snap: the smear is a
        // keyboard affordance, not a mouse one.
        a.update(
            frame((300.0, 0.0), 9, true),
            &cfg,
            t0 + Duration::from_millis(16),
        );
        assert_eq!(a.pos, Some(bar(300.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_reveal_reflow_snaps() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((100.0, 0.0), 3), &cfg, t0);
        // The active line reveals its raw markers: the caret's screen x shifts right
        // but its logical offset is unchanged. That is a reflow, not a navigation, so
        // the caret snaps to the new x with no smear. (A pure scroll is the same case
        // with an unchanged content-space target, and is likewise a no-op.)
        a.update(nav((140.0, 0.0), 3), &cfg, t0 + Duration::from_millis(16));
        assert_eq!(a.pos, Some(bar(140.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_geometry_reflow_snaps() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // A resize/DPI/column move changes the geometry key: the whole document
        // reflows, so even a keyboard offset change snaps rather than smearing across
        // the jump.
        let reflowed = CaretFrame {
            key: (0, 700, FONT_SIZE, 120),
            ..nav((200.0, 0.0), 1)
        };
        a.update(reflowed, &cfg, t0 + Duration::from_millis(16));
        assert_eq!(a.pos, Some(bar(200.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_pointer_interrupts_a_smear() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        a.update(nav((200.0, 0.0), 1), &cfg, t0); // smear in flight
        assert!(a.in_flight());
        // A click mid-smear cuts it short and snaps to where it pointed.
        a.update(
            frame((30.0, 0.0), 2, true),
            &cfg,
            t0 + Duration::from_millis(16),
        );
        assert_eq!(a.pos, Some(bar(30.0, 0.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_keyboard_move_midflight_reaims() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        a.update(nav((200.0, 0.0), 1), &cfg, t0); // smear in flight
        assert!(a.in_flight());
        // Typing (or another arrow) mid-smear re-aims the destination and keeps
        // easing: consecutive keystrokes flow into one continuous smear.
        a.update(nav((30.0, 0.0), 2), &cfg, t0 + Duration::from_millis(16));
        assert!(a.in_flight());
    }

    #[test]
    fn caret_disabled_is_the_plain_bar() {
        let cfg = CaretConfig {
            animate: false,
            rate: 40.0,
            spring: 0.65,
            max_jump_lines: 6.0,
        };
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((100.0, 40.0), 0), &cfg, t0);
        assert!(!a.in_flight());
        // With animation off the paint rect is the same Fill bar as the pre-animation
        // path (the caller draws radius zero because it never flies).
        assert_eq!(
            a.draw_rect(),
            Rect {
                x: 100,
                y: 40,
                w: 2,
                h: 20
            }
        );
        // A keyboard move snaps immediately; it never eases.
        a.update(nav((300.0, 40.0), 1), &cfg, t0 + Duration::from_millis(16));
        assert_eq!(a.pos, Some(bar(300.0, 40.0)));
        assert!(!a.in_flight());
    }

    #[test]
    fn caret_exposes_a_retry_only_while_in_flight() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        assert!(a.retry_at().is_none()); // settled
        a.update(nav((80.0, 0.0), 1), &cfg, t0); // in flight
        assert!(a.retry_at().is_some_and(|at| at > t0));
    }

    #[test]
    fn caret_move_after_idle_does_not_spend_the_idle_gap() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        // A move long after settling: the first step uses the nominal frame time, not
        // the 100s idle, so it eases a little rather than leaping most of the way.
        let t1 = t0 + Duration::from_secs(100);
        a.update(nav((60.0, 0.0), 1), &cfg, t1);
        assert!(a.in_flight());
        assert_eq!(a.last_tick, Some(t1));
        let moved = a.pos.unwrap().left;
        assert!(moved > 0.0 && moved < 40.0);
        // The next tick continues easing with the real 16ms elapsed.
        a.update(nav((60.0, 0.0), 1), &cfg, t1 + Duration::from_millis(16));
        assert!(a.pos.unwrap().left > moved);
    }

    #[test]
    fn caret_request_snap_clears_flight_immediately() {
        let cfg = anim_cfg(1000.0);
        let t0 = Instant::now();
        let mut a = CaretAnim::new();
        a.update(nav((0.0, 0.0), 0), &cfg, t0);
        a.update(nav((80.0, 0.0), 1), &cfg, t0); // in flight
        assert!(a.in_flight());
        // A hidden or off-window caret requests a snap; flight and timing clear at
        // once so it cannot keep scheduling retries.
        a.request_snap();
        assert!(!a.in_flight());
        assert!(a.retry_at().is_none());
    }
}
