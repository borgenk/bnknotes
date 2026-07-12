//! The editor's orchestration: it binds the compositor globals, builds the
//! xdg-shell surface stack, and runs the event loop.
//!
//! On top of the earlier single-line field, this drives a small but real
//! multi-line editor. A keyboard (via libxkbcommon) inserts text, newlines, and
//! soft-tab indentation, and moves the cursor by grapheme cluster; Up/Down/
//! Home/End and Page Up/Down navigate the visual lines a [`Layout`] pass
//! computes, while Ctrl jumps by word (arrows) or to the document ends
//! (Home/End) and deletes whole words (Backspace/Delete); a held Shift turns
//! motion into a selection, which a pointer can also drag out (double- and
//! triple-click select a word or line, Shift+click extends, and a drag past an
//! edge auto-scrolls), with an I-beam shown over the text; the selection
//! highlights behind the text and Backspace or typing replaces it. Ctrl shortcuts
//! select-all,
//! copy, cut, paste (over wl_data_device), and undo/redo; a held key repeats on a
//! timer. Content taller than the window scrolls (mouse wheel, or automatically
//! to keep the caret in view), and the window honors compositor resizes by
//! reallocating its GPU buffers. Rendering double-buffers so the compositor never
//! reads a frame we are still painting.

mod clipboard;
mod input;
mod present;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use self::clipboard::{is_text_mime, ClipboardState, CLIPBOARD_MIMES};
use self::input::KeyRepeat;
use self::present::GpuPresentation;
use crate::config::{CaretConfig, Config};
use crate::file;
use crate::markdown;
use crate::paint::{
    build_display_list, build_find_paint, push_splitter, push_top_bar, push_tree, top_bar_height,
    BarCtx, CaretPaint, PaintParams, TabStrip,
};
use crate::pane::{
    restore_group, CaretAnim, CaretFrame, Columns, Focus, PaneGroup, PaneLayout, TreePane,
};
use crate::platform::conn::{Connection, Fill};
use crate::platform::error::{Error, Result};
use crate::platform::freetype::{FontConfig, Fonts};
use crate::platform::geom::Rect;
use crate::platform::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device_manager, wl_data_offer, wl_display,
    wl_output, wl_registry, wl_seat, wl_surface, wp_fractional_scale_manager_v1,
    wp_fractional_scale_v1, wp_viewporter, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::platform::scroll;
use crate::platform::wire::{Arg, Message, Reader};
use crate::platform::xkb::Xkb;
use crate::render::display::DisplayList;
use crate::scale::Scale;
use crate::store;
use crate::theme::{CARET_WIDTH, FONT_SIZE};
use crate::tree;

/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display (8K is 7680 wide), it just keeps a
/// bogus or hostile size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

/// A compositor scale of 1.0, in the fractional-scale protocol's 120ths unit. The
/// scale is unknown until the compositor reports it, so the window opens at unity.
const SCALE_120_UNITY: u32 = 120;

/// The sane range a resolved device-pixel body font size is clamped to, guarding a
/// hostile scale or a bogus `BNKNOTES_FONT_SIZE` from opening an unusable face.
const FONT_SIZE_RANGE: std::ops::RangeInclusive<u32> = 6..=72;

/// The device-pixel body font size at compositor scale `scale_120` (120ths). The
/// logical body [`FONT_SIZE`] is scaled by the display, so text keeps its physical
/// size as DPI changes. `BNKNOTES_FONT_SIZE` pins an explicit pixel height and opts
/// out of scaling (the escape hatch).
fn config_font_px(scale_120: u32) -> u32 {
    if let Some(px) = std::env::var("BNKNOTES_FONT_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&px| px > 0)
    {
        return px.clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end());
    }
    Scale::from_120(scale_120)
        .font_px(FONT_SIZE)
        .clamp(*FONT_SIZE_RANGE.start(), *FONT_SIZE_RANGE.end())
}

/// Compositor scale tracking and the objects that deliver it.
///
/// Two mechanisms, in priority order. The fractional path is primary: a
/// `wp_fractional_scale_v1` reports `preferred_scale` in 120ths and a `wp_viewport`
/// maps the device-pixel buffer down to the logical window size, so a 1.25 or 1.5
/// display renders pixel-exact. The integer path is the fallback for compositors
/// with no fractional-scale global: each `wl_output`'s `scale` event, the surface's
/// scale being the max over the outputs it spans (`wl_surface.enter`/`leave`), and
/// `wl_surface.set_buffer_scale`. `viewport != 0` means the fractional path is live.
struct Scaling {
    /// Effective scale as 120ths (120 = 1.0); drives the font pixels and buffer size.
    factor_120: u32,
    /// Logical window size (surface-local px) from the last xdg configure. The
    /// device buffer is this scaled up; the viewport (or buffer scale) maps back.
    logical: (u32, u32),
    /// Whether the surface's geometry + viewport/buffer-scale state has been resent
    /// since `factor_120` or `logical` last changed. Cleared on change, set on sync.
    synced: bool,
    fractional_manager: Option<u32>,
    viewporter: Option<u32>,
    /// Per-surface objects the managers mint (0 when unavailable).
    fractional_scale: u32,
    viewport: u32,
    /// Integer fallback: each bound `wl_output`'s scale, and the ids the surface
    /// currently spans. Unused once the fractional path is live.
    outputs: Vec<(u32, i32)>,
    entered: Vec<u32>,
}

impl Scaling {
    fn new(logical: (u32, u32)) -> Self {
        Scaling {
            factor_120: SCALE_120_UNITY,
            logical,
            synced: false,
            fractional_manager: None,
            viewporter: None,
            fractional_scale: 0,
            viewport: 0,
            outputs: Vec::new(),
            entered: Vec::new(),
        }
    }

    /// Whether the fractional path is live (a viewport was created for the surface).
    fn is_fractional(&self) -> bool {
        self.viewport != 0
    }

    /// The integer scale the surface spans: the max scale over the outputs it is
    /// on, at least 1. Used only on the fallback path.
    fn integer_scale(&self) -> i32 {
        self.entered
            .iter()
            .filter_map(|id| self.outputs.iter().find(|(oid, _)| oid == id))
            .map(|(_, scale)| *scale)
            .max()
            .unwrap_or(1)
            .max(1)
    }
}

pub fn run() -> Result<()> {
    // An optional path to open; everything after it is ignored for now.
    let path = std::env::args_os().nth(1).map(PathBuf::from);
    State::new(path)?.bring_up()
}

/// `--gpu-probe`: report what the compositor offers the dmabuf presentation
/// path (and, in later milestones, what the GPU offers), without opening a
/// window. This is the same negotiation the editor performs at startup, so a
/// successful probe means the GPU backend's protocol prerequisites hold.
pub fn gpu_probe() -> Result<()> {
    State::new(None)?.probe_dmabuf()
}

/// What the pointer is hovering in a pane's tab strip, so the painter can light
/// the right element: a tab body, that tab's close glyph, or the new-tab button.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TabHover {
    Tab {
        pane: usize,
        tab: usize,
        close: bool,
    },
    New {
        pane: usize,
    },
}

/// An in-progress tab drag-to-reorder. Records which pane and which tab (by its
/// current index, which the reorder keeps up to date), the pointer x at press and
/// the grab offset within the tab (so the floating tab stays under the cursor), and
/// whether the pointer has crossed the click threshold yet: a press that never
/// does is a plain tab click, not a drag.
#[derive(Clone, Copy)]
struct TabDrag {
    pane: usize,
    tab: usize,
    grab_x: f32,
    grab_dx: i32,
    moved: bool,
}

impl TabHover {
    /// The hovered tab index in pane `g` (its body highlight), if any.
    fn tab_in(self, g: usize) -> Option<usize> {
        match self {
            TabHover::Tab { pane, tab, .. } if pane == g => Some(tab),
            _ => None,
        }
    }

    /// The tab whose close glyph is hovered in pane `g`, if any.
    fn close_in(self, g: usize) -> Option<usize> {
        match self {
            TabHover::Tab {
                pane,
                tab,
                close: true,
            } if pane == g => Some(tab),
            _ => None,
        }
    }
}

struct State {
    conn: Connection,
    /// The font opened at every size the styles need (body and heading sizes),
    /// so each line can render at its own size.
    fonts: Fonts,
    /// Mask gamma for glyph coverage, read from the app config at startup and
    /// handed to the shader each frame (see [`crate::config::Config`]).
    glyph_gamma: f32,
    /// Caret motion settings, read from the app config at startup. When animation is
    /// off the caret snaps cell to cell; otherwise it eases toward its target.
    caret_config: CaretConfig,
    xkb: Xkb,
    /// The two content panes (editor A and B) and the file-tree sidebar, laid
    /// out as adjustable columns below the top bar. Each pane is a stack of open
    /// tabs with one active; `focus` selects which region takes keyboard input; a
    /// click moves focus (and, on a splitter, starts a drag).
    editors: [PaneGroup; 2],
    tree: TreePane,
    panes: PaneLayout,
    focus: Focus,
    /// The editor last given focus, so a file opened from the tree (which takes
    /// focus itself) lands in the content pane the user was last editing.
    last_editor: usize,
    /// What the pointer is over in a tab strip (a tab, its close glyph, or the
    /// new-tab button), for the hover highlights; `None` when off every strip.
    hovered_tab: Option<TabHover>,
    /// While a splitter is dragged, which one: 0 = tree|A edge, 1 = A|B edge.
    drag_splitter: Option<usize>,
    /// While a tab is dragged to reorder it: which pane and tab, and the grab state.
    drag_tab: Option<TabDrag>,
    /// The toplevel title last sent, so it is only re-set when it changes.
    title: String,

    /// Current surface size in device pixels (the buffer size). The logical window
    /// the compositor sees lives in `scaling.logical`; at scale 1.0 the two match.
    width: u32,
    height: u32,
    /// Compositor scale tracking and the objects that deliver it.
    scaling: Scaling,
    /// Device-pixel body font size, derived from the scale (or pinned by
    /// `BNKNOTES_FONT_SIZE`). The fonts are opened at this size and every heading
    /// scales off it; [`FONT_SIZE`] at scale 1.0.
    body_size: u32,
    /// The font paths and line-height scale, retained so a DPI change can reopen the
    /// font set at the new device body size without rebuilding the whole config.
    font_config: FontConfig,
    line_height_scale: f32,

    // Object ids. Globals are Option (discovered via the registry); ids we
    // create default to 0 (never valid) until assigned.
    registry: u32,
    compositor: Option<u32>,
    wm_base: Option<u32>,
    seat: Option<u32>,
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    keyboard: u32,
    pointer: u32,
    data_device_manager: Option<u32>,
    data_device: u32,
    /// GPU/dmabuf presentation resources: feedback, backend, swap buffers,
    /// explicit sync, and per-frame presentation counters.
    presentation: GpuPresentation,
    /// The cursor-shape manager (if the compositor offers it) and the shape
    /// device we create for our pointer. The shape varies by what is under the
    /// pointer: a hand over an image's button, the plain arrow over a picture, the
    /// I-beam over text.
    cursor_shape_manager: Option<u32>,
    cursor_shape_device: u32,
    /// The most recent pointer `enter` serial (cursor-shape requests need it) and
    /// the shape currently set, so motion only re-requests when the shape changes.
    pointer_serial: u32,
    cursor_shape: u32,

    // Pointer state: latest position in surface-local pixels. Which pane the
    // pointer falls in (and any in-progress drag) is resolved per-pane at use;
    // the per-editor drag/selection transients live on `EditorPane`.
    pointer_x: f32,
    pointer_y: f32,
    /// Caret blink: whether the caret is currently shown, and when it next
    /// toggles. Activity (`bump_caret`) resets it to shown. Applies to the
    /// focused editor's caret; other panes draw a dimmed, static caret.
    caret_visible: bool,
    caret_blink_at: Option<Instant>,
    /// Whether the last input that moved the caret came from the pointer. The smear
    /// is a keyboard-navigation affordance, so a pointer-driven move (click, drag)
    /// snaps instead of easing. Set false by any keyboard action and true when the
    /// pointer places the caret; only consulted when the caret's logical offset
    /// actually moved.
    caret_from_pointer: bool,
    /// Whether the surface holds the keyboard focus (`wl_keyboard.enter` to
    /// `leave`). The caret only blinks (and only wakes the loop on its timer)
    /// while focused; an unfocused window hides the caret and stays idle.
    focused: bool,

    /// Key auto-repeat timers (delay, interval, held key, next deadline).
    key_repeat: KeyRepeat,
    /// The latest input event serial, needed to set the clipboard selection.
    last_serial: u32,

    /// Clipboard: the source we own while holding the selection, the bytes and
    /// MIME types it serves, and the incoming/selection offers a paste reads.
    clipboard: ClipboardState,

    // Id allocation: a bump counter plus a free list fed by `delete_id`.
    next_id: u32,
    free_ids: Vec<u32>,

    // Loop state.
    pending_sync: Option<u32>,
    /// A compositor-requested size awaiting the next configure ack.
    pending_size: Option<(u32, u32)>,
    /// The serial of the latest xdg_surface.configure not yet acked. The ack is
    /// deferred to `render_frame` so it commits together with the buffer sized to
    /// that configure; acking on arrival lets the compositor reposition an
    /// anchored (left/top) resize edge a frame before the matching buffer lands,
    /// which reads as jitter. Only the latest serial needs acking.
    pending_configure: Option<u32>,
    configured: bool,
    closed: bool,
    /// The content changed and a frame should be drawn.
    dirty: bool,
    /// The in-flight frame callback's id, or 0 when none is pending. Set when a
    /// frame is committed, cleared when the compositor's `done` fires. While it is
    /// nonzero the loop holds off redrawing, so input bursts, key repeat, and
    /// caret blink coalesce into at most one frame per refresh (frame pacing).
    frame_callback: u32,
    /// When set (the `BNKNOTES_STATS` env var is present), each presented frame
    /// prints its render time and the glyph-cache hit rate to stderr. Plain
    /// instrumentation so an optimization can be measured rather than guessed at.
    stats: bool,
}

/// Resolve the file-tree root and any file to open, from the CLI argument and the
/// saved session root. Precedence: a directory argument roots there and opens no
/// file; a file argument roots at its parent (or `cwd` if it has none) and opens
/// the file; no argument uses the last session's root when it is still a
/// directory, else `cwd`. Pure precedence, split out of `State::new` so it is
/// testable without a live connection.
fn resolve_root(
    arg: Option<&Path>,
    session_root: Option<&Path>,
    cwd: &Path,
) -> (PathBuf, Option<PathBuf>) {
    match arg {
        Some(p) if p.is_dir() => (p.to_path_buf(), None),
        Some(p) => (
            p.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| cwd.to_path_buf()),
            Some(p.to_path_buf()),
        ),
        None => (
            session_root
                .filter(|r| r.is_dir())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| cwd.to_path_buf()),
            None,
        ),
    }
}

/// The region that holds focus at startup: a file argument focuses its editor
/// pane; otherwise the saved focus, guarded so it never points at a hidden region
/// (a saved tree or pane-B focus falls back to editor A when that region is not
/// shown). Pure precedence, split out for testing.
fn initial_focus(
    has_cli_file: bool,
    saved: store::Focus,
    tree_visible: bool,
    second_open: bool,
) -> Focus {
    if has_cli_file {
        return Focus::Editor(0);
    }
    match saved {
        store::Focus::Tree if tree_visible => Focus::Tree,
        store::Focus::PaneB if second_open => Focus::Editor(1),
        store::Focus::Tree | store::Focus::PaneA | store::Focus::PaneB => Focus::Editor(0),
    }
}

impl State {
    fn new(path: Option<PathBuf>) -> Result<Self> {
        let conn = Connection::connect()?;
        // Font selection lives in the app config (defaults for now; no on-disk
        // config yet). Open every size the styles use so headings render larger
        // than body text without re-opening mid-frame.
        let config = Config::default();
        // The window opens at scale 1.0 (the compositor reports the real scale
        // after the first configure), so the body size starts at the logical
        // FONT_SIZE unless BNKNOTES_FONT_SIZE pins it.
        let body_size = config_font_px(SCALE_120_UNITY);
        let fonts = Fonts::with_config(
            &config.fonts,
            &markdown::all_sizes(body_size),
            config.line_height_scale,
        )?;
        // The saved layout: window/column sizes, tree root and open folders, and
        // each pane's last file with its caret and scroll. Missing or corrupt
        // falls back to the first-run default without failing.
        let session = store::load();

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let (root, cli_file) = resolve_root(path.as_deref(), session.tree_root.as_deref(), &cwd);

        // Editor A: a file argument opens as a single tab; otherwise restore the
        // session's pane A tabs. Editor B always restores from the session (one
        // blank tab if it had none).
        let editor_a = match &cli_file {
            Some(p) => restore_group(
                &[store::PaneSession {
                    path: Some(p.to_path_buf()),
                    ..Default::default()
                }],
                0,
            ),
            None => restore_group(&session.panes[0].tabs, session.panes[0].active as usize),
        };
        let editor_b = restore_group(&session.panes[1].tabs, session.panes[1].active as usize);

        let expanded: BTreeSet<PathBuf> = session.expanded.iter().cloned().collect();
        let tree = TreePane {
            tree: tree::Tree::new(root, expanded),
            scroll_y: 0,
            content_height: 0,
            hovered: None,
            selected: None,
        };
        let panes = PaneLayout {
            tree_width: session.tree_width as i32,
            split_ratio: session.split_ratio,
            // Opening a file from the CLI still shows the sidebar, so its folder
            // (the file's parent) is there to browse, even if it was last hidden.
            tree_visible: session.tree_visible || cli_file.is_some(),
            second_open: session.second_open,
        };
        let focus = initial_focus(
            cli_file.is_some(),
            session.focus,
            panes.tree_visible,
            panes.second_open,
        );

        Ok(Self {
            conn,
            fonts,
            glyph_gamma: config.glyph_coverage_gamma,
            caret_config: config.caret,
            xkb: Xkb::new()?,
            editors: [editor_a, editor_b],
            tree,
            panes,
            focus,
            last_editor: match focus {
                Focus::Editor(i) => i,
                Focus::Tree => 0,
            },
            hovered_tab: None,
            drag_splitter: None,
            drag_tab: None,
            title: String::new(),
            width: session.window.0,
            height: session.window.1,
            scaling: Scaling::new(session.window),
            body_size,
            font_config: config.fonts,
            line_height_scale: config.line_height_scale,
            registry: 0,
            compositor: None,
            wm_base: None,
            seat: None,
            surface: 0,
            xdg_surface: 0,
            toplevel: 0,
            keyboard: 0,
            pointer: 0,
            data_device_manager: None,
            data_device: 0,
            presentation: GpuPresentation::new(),
            cursor_shape_manager: None,
            cursor_shape_device: 0,
            pointer_serial: 0,
            cursor_shape: 0,
            pointer_x: 0.0,
            pointer_y: 0.0,
            caret_visible: true,
            caret_blink_at: None,
            caret_from_pointer: false,
            focused: false,
            key_repeat: KeyRepeat::new(),
            last_serial: 0,
            clipboard: ClipboardState::new(),
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            pending_configure: None,
            configured: false,
            closed: false,
            dirty: true,
            frame_callback: 0,
            stats: std::env::var_os("BNKNOTES_STATS").is_some(),
        })
    }

    fn alloc_id(&mut self) -> u32 {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        })
    }

    /// The editor pane that currently takes keyboard input. Falls back to editor
    /// 0 when the tree holds focus, a harmless default since the edit paths that
    /// call this only run while an editor is focused.
    fn active_editor(&self) -> usize {
        match self.focus {
            Focus::Editor(i) => i,
            Focus::Tree => 0,
        }
    }

    /// The column rectangle for editor `i` at the current window size, or an
    /// empty rect when that pane is hidden (only editor 1 can be).
    fn editor_rect(&self, i: usize) -> Rect {
        self.columns().editors[i].unwrap_or(Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        })
    }

    /// The current column layout for this window size, below the top bar.
    fn columns(&self) -> Columns {
        self.panes.columns(
            self.width as i32,
            self.height as i32,
            self.top_bar_h(),
            self.scale(),
        )
    }

    /// The top bar's device-pixel height at the current body size and scale.
    fn top_bar_h(&self) -> i32 {
        top_bar_height(self.fonts.metrics(self.body_size), self.scale())
    }

    /// The current display scale, derived from the compositor's reported factor.
    fn scale(&self) -> Scale {
        Scale::from_120(self.scaling.factor_120)
    }

    /// A logical (surface-local) length in device pixels at the current scale.
    fn to_device(&self, logical: u32) -> u32 {
        self.scale().px(logical as i32).max(0) as u32
    }

    /// The logical window size scaled to the device buffer size, clamped so a bogus
    /// configure cannot blow up the buffer arithmetic.
    fn device_size(&self, logical_w: u32, logical_h: u32) -> (u32, u32) {
        (
            self.to_device(logical_w).clamp(1, MAX_DIMENSION),
            self.to_device(logical_h).clamp(1, MAX_DIMENSION),
        )
    }

    /// Adopt a new compositor scale (in 120ths): reopen the fonts at the size it
    /// calls for, re-derive the device buffer size from the logical window, and
    /// mark the surface state unsynced so the next frame resends it. A no-op if the
    /// scale is unchanged.
    fn set_scale_120(&mut self, factor_120: u32) {
        let factor_120 = factor_120.max(1);
        if factor_120 == self.scaling.factor_120 {
            return;
        }
        self.scaling.factor_120 = factor_120;
        self.scaling.synced = false; // geometry + viewport/buffer-scale must be resent
        self.apply_font_size(config_font_px(factor_120));
        let (lw, lh) = self.scaling.logical;
        let (dw, dh) = self.device_size(lw, lh);
        self.resize_to(dw, dh);
        // Force the next frame even if the device size happened to land the same, so
        // the resent scale state (and rescaled glyphs) reach the screen.
        self.dirty = true;
    }

    /// Reopen the fonts at device-pixel `size` (keeping the working fonts on a
    /// font-open failure, so a DPI change never blanks the window), then push the
    /// current scale and effective body size to both panes so their cached layouts
    /// invalidate and re-measure at the new size.
    fn apply_font_size(&mut self, size: u32) {
        if size != self.body_size {
            if let Ok(fonts) = Fonts::with_config(
                &self.font_config,
                &markdown::all_sizes(size),
                self.line_height_scale,
            ) {
                self.fonts = fonts;
                self.body_size = size;
            }
        }
        let (scale, body) = (self.scale(), self.body_size);
        for ed in &mut self.editors {
            ed.set_scale(scale, body);
        }
    }

    /// On the integer-scale fallback, recompute the surface scale as the max over
    /// the outputs it currently spans and adopt it. A no-op on the fractional path.
    fn refresh_integer_scale(&mut self) {
        if self.scaling.is_fractional() {
            return;
        }
        let factor_120 = self.scaling.integer_scale() as u32 * 120;
        self.set_scale_120(factor_120);
    }

    /// Ensure the focused editor's layout is current, then keep its caret on
    /// screen and repaint if that scrolled. The routed equivalent of the old
    /// `scroll_caret_into_view`, for the many call sites that acted on "the
    /// current editor".
    fn reveal_active_caret(&mut self) {
        let i = self.active_editor();
        let rect = self.editor_rect(i);
        if self.editors[i].scroll_caret_into_view(&self.fonts, rect) {
            self.dirty = true;
        }
    }

    fn bring_up(&mut self) -> Result<()> {
        // Discover and bind globals.
        self.registry = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::GET_REGISTRY,
            &[Arg::NewId(self.registry)],
        );
        self.roundtrip()?;

        let compositor = self
            .compositor
            .ok_or_else(|| Error::msg("compositor has no wl_compositor"))?;
        let wm_base = self
            .wm_base
            .ok_or_else(|| Error::msg("compositor has no xdg_wm_base"))?;
        self.seat
            .ok_or_else(|| Error::msg("compositor has no wl_seat"))?;

        // The GPU (Vulkan + dmabuf) is mandatory: negotiate over dmabuf feedback
        // and bring Vulkan up, or exit with a clear message. There is no software
        // fallback. The feedback (device, modifiers) must be committed before
        // init_gpu can allocate, so wait out the burst here first.
        self.request_dmabuf_feedback();
        self.roundtrip()?;
        self.init_gpu()
            .map_err(|e| Error::msg(format!("requires a Vulkan device; none available: {e}")))?;

        // Surface stack: wl_surface -> xdg_surface -> xdg_toplevel.
        self.surface = self.alloc_id();
        self.conn.request(
            compositor,
            wl_compositor::CREATE_SURFACE,
            &[Arg::NewId(self.surface)],
        );
        self.xdg_surface = self.alloc_id();
        self.conn.request(
            wm_base,
            xdg_wm_base::GET_XDG_SURFACE,
            &[Arg::NewId(self.xdg_surface), Arg::Object(self.surface)],
        );
        self.toplevel = self.alloc_id();
        self.conn.request(
            self.xdg_surface,
            xdg_surface::GET_TOPLEVEL,
            &[Arg::NewId(self.toplevel)],
        );
        // The app_id ties the window to bnknotes.desktop, so the compositor can
        // show the right icon and group windows in the taskbar.
        self.conn.request(
            self.toplevel,
            xdg_toplevel::SET_APP_ID,
            &[Arg::Str("bnknotes")],
        );
        self.refresh_title();

        // Fractional scaling: a per-surface fractional-scale object delivers the
        // preferred scale, and a viewport maps the device-pixel buffer onto the
        // logical window size. They come as a pair; without both, the integer
        // fallback (wl_output scale + set_buffer_scale) is used instead.
        if let (Some(fmgr), Some(vp)) = (self.scaling.fractional_manager, self.scaling.viewporter) {
            let fractional = self.alloc_id();
            self.conn.request(
                fmgr,
                wp_fractional_scale_manager_v1::GET_FRACTIONAL_SCALE,
                &[Arg::NewId(fractional), Arg::Object(self.surface)],
            );
            self.scaling.fractional_scale = fractional;
            let viewport = self.alloc_id();
            self.conn.request(
                vp,
                wp_viewporter::GET_VIEWPORT,
                &[Arg::NewId(viewport), Arg::Object(self.surface)],
            );
            self.scaling.viewport = viewport;
        }

        // The data device drives the clipboard; skip it if the compositor has no
        // wl_data_device_manager (editing still works, just no copy/paste).
        if let (Some(manager), Some(seat)) = (self.data_device_manager, self.seat) {
            let device = self.alloc_id();
            self.conn.request(
                manager,
                wl_data_device_manager::GET_DATA_DEVICE,
                &[Arg::NewId(device), Arg::Object(seat)],
            );
            self.data_device = device;
        }

        self.create_buffers()?;
        // With buffers and a surface in place, negotiate explicit sync if the
        // compositor offers it and the GPU path is live; otherwise the bridge
        // stays. Best-effort: any failure leaves the bridge running.
        self.init_explicit_sync();

        // Empty commit to get the initial configure; the first frame is drawn
        // by the loop once `configured` is set.
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.run_until(|s| s.configured)?;

        eprintln!(
            "bnknotes: type to edit; Enter, Tab, arrows/Home/End, PageUp/Down, \
             Ctrl+arrows by word, Ctrl+Home/End to ends, Shift selects; \
             click/drag, Shift+click extends, double/triple-click select word/line; \
             Ctrl+Backspace/Delete delete a word, Ctrl+Shift+K deletes a line; \
             Ctrl+A/C/X/V select-all/copy/cut/paste, \
             Ctrl+Z/Y undo/redo, Ctrl+S save; \
             Ctrl+B toggle tree, Ctrl+\\ toggle 2nd pane, Ctrl+0/1/2 focus tree/A/B, \
             drag splitters to resize; held keys repeat; close to exit"
        );
        // Name the GPU device, so "which GPU?" never needs a probe. The backend
        // is mandatory (bring_up exits above if it is missing), so it is present.
        if let Some(gpu) = &self.presentation.backend {
            eprintln!("bnknotes: renderer: gpu ({})", gpu.name());
        }
        self.run_until(|s| s.closed)?;
        // Remember the layout for next launch: column sizes, tree root and open
        // folders, and each pane's file with its caret and scroll.
        self.save_session();
        Ok(())
    }

    /// The current layout as a persistable session (see [`crate::store`]).
    fn current_session(&self) -> store::Session {
        let focus = match self.focus {
            Focus::Tree => store::Focus::Tree,
            Focus::Editor(1) => store::Focus::PaneB,
            Focus::Editor(_) => store::Focus::PaneA,
        };
        store::Session {
            // The persisted window is the logical size (scale-free); the device
            // buffer size lives in width/height.
            window: self.scaling.logical,
            tree_width: self.panes.tree_width.max(0) as u32,
            split_ratio: self.panes.split_ratio,
            tree_visible: self.panes.tree_visible,
            second_open: self.panes.second_open,
            focus,
            tree_root: Some(self.tree.tree.root().to_path_buf()),
            expanded: self.tree.tree.expanded_dirs(),
            panes: [self.editors[0].to_session(), self.editors[1].to_session()],
        }
    }

    /// Persist the layout, reporting (not failing) if the state file cannot be
    /// written; a lost layout must never take the editor down.
    fn save_session(&self) {
        if let Err(e) = store::save(&self.current_session()) {
            eprintln!("bnknotes: could not save layout: {e}");
        }
    }

    fn roundtrip(&mut self) -> Result<()> {
        let callback = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::SYNC,
            &[Arg::NewId(callback)],
        );
        self.pending_sync = Some(callback);
        self.run_until(|s| s.pending_sync.is_none())
    }

    /// Drain buffered events, service key repeats, draw a frame if one is due,
    /// then wait for more (waking at the next repeat deadline).
    fn run_until(&mut self, done: impl Fn(&State) -> bool) -> Result<()> {
        loop {
            while let Some(msg) = self.conn.next_message()? {
                self.handle(msg)?;
            }
            self.fire_repeats()?;
            self.fire_autoscroll();
            self.fire_caret_blink();
            self.fire_caret_anim();
            // Pace to the compositor's clock: only draw when no frame callback is
            // outstanding, so a burst of edits or held keys between refreshes
            // collapses into a single repaint instead of racing both buffers.
            if self.configured && self.dirty && self.frame_callback == 0 && self.render_frame()? {
                self.dirty = false;
            }
            if done(self) {
                return Ok(());
            }
            self.conn.flush()?;
            let timeout = self.next_wake();
            if let Fill::Bytes(0) = self.conn.fill(timeout)? {
                return Err(Error::msg("compositor closed the connection"));
            }
        }
    }

    /// Compose the whole window's display list: the file tree, then each visible
    /// editor pane built into its column, then the splitter bars. Refreshes each
    /// pane's layout and clamps its scroll first, so the list matches the current
    /// column sizes. Tree first so an editor's background covers any overhanging
    /// filename; splitters last so they cover the inter-column gaps.
    fn build_frame_list(&mut self) -> Rc<DisplayList> {
        let cols = self.columns();
        let now = Instant::now();
        let mut list: DisplayList = Vec::new();
        if let Some(rect) = cols.tree {
            let (scale, body_size) = (self.scale(), self.body_size);
            push_tree(
                &mut list,
                &self.fonts,
                &mut self.tree,
                rect,
                scale,
                body_size,
            );
        }
        for i in 0..self.editors.len() {
            let Some(rect) = cols.editors[i] else {
                continue;
            };
            // An empty pane (no tabs open) paints no content and no caret.
            if !self.editors[i].has_tabs() {
                continue;
            }
            self.editors[i].ensure_layout(&self.fonts, rect);
            let ch = self.editors[i].content_height;
            self.editors[i].scroll_y = scroll::clamp(self.editors[i].scroll_y, ch, rect.h);
            let focused = self.focus == Focus::Editor(i);
            // Tick this pane's caret animation and reduce it to a paint rectangle
            // before borrowing the pane immutably for the rest of the frame.
            let caret = self.build_caret(i, rect, focused, now);
            let ed = &self.editors[i];
            // Owned so it outlives the `FindPaint` that borrows it for this frame.
            let tick_tops = ed.find_tick_tops();
            // The table add-row/add-column strip under the pointer, when it hovers
            // this pane and no drag is in flight, recomputed each frame so it never
            // goes stale after a scroll or edit.
            let (px, py) = (self.pointer_x, self.pointer_y);
            let over = px >= rect.x as f32
                && px < (rect.x + rect.w) as f32
                && py >= rect.y as f32
                && py < (rect.y + rect.h) as f32;
            let table_hover = if over && !ed.selecting && self.drag_splitter.is_none() {
                ed.layout_cache.as_ref().and_then(|l| {
                    l.table_button_at(&self.fonts, ed.text.rope(), px, ed.content_y(py))
                })
            } else {
                None
            };
            let params = PaintParams {
                width: rect.w as usize,
                height: rect.h as usize,
                scale: self.scale(),
                body_size: self.body_size,
                scroll_y: ed.scroll_y,
                selection: ed.text.selection(),
                caret,
                find: build_find_paint(&ed.find, &tick_tops, ed.content_height),
                table_hover,
            };
            if let Some(layout) = ed.layout_cache.as_ref() {
                list.append(&mut build_display_list(
                    &self.fonts,
                    layout,
                    ed.text.rope(),
                    &params,
                ));
            }
        }
        let scale = self.scale();
        // The column splitters, kept (Copy) so the top bar can continue their
        // divider lines up through the reserved strip.
        let splitter_rects = cols.splitters;
        for s in cols.splitters.into_iter().flatten() {
            push_splitter(&mut list, s, scale);
        }
        // The top bar: a tab strip above each pane, segmented by the continued
        // column dividers. Drawn last; it occupies the reserved strip above every
        // column. The vault label is left off for now (a later step brings the
        // dropdown), so `None` is passed for it.
        let top = self.top_bar_h();
        if top > 0 {
            let bar = Rect {
                x: 0,
                y: 0,
                w: self.width as i32,
                h: top,
            };
            let labels = [self.editors[0].tab_names(), self.editors[1].tab_names()];
            let dirty = [self.editors[0].tab_dirty(), self.editors[1].tab_dirty()];
            let strip = |g: usize| {
                cols.tab_strips[g].map(|rect| TabStrip {
                    rect,
                    labels: &labels[g],
                    dirty: &dirty[g],
                    active: self.editors[g].active_index(),
                    hovered: self.hovered_tab.and_then(|h| h.tab_in(g)),
                    close_hover: self.hovered_tab.and_then(|h| h.close_in(g)),
                    new_hover: self.hovered_tab == Some(TabHover::New { pane: g }),
                    // A tab reorder in progress lifts its tab to follow the pointer,
                    // but only once the press has grown into an actual drag.
                    drag: self.drag_tab.filter(|d| d.pane == g && d.moved).map(|d| {
                        crate::paint::TabDragView {
                            tab: d.tab,
                            left: (self.pointer_x - d.grab_dx as f32).round() as i32,
                        }
                    }),
                })
            };
            let dividers: Vec<Rect> = splitter_rects.into_iter().flatten().collect();
            let ctx = BarCtx::new(&self.fonts, scale, self.body_size);
            push_top_bar(&mut list, &ctx, bar, None, [strip(0), strip(1)], &dividers);
        }
        Rc::new(list)
    }

    /// Tick pane `i`'s caret animation and reduce it to the rectangle to draw this
    /// frame, or `None` when no caret is drawn. The animation is ticked even when
    /// the caret is hidden (unfocused or blinked off) so it still settles; when the
    /// cursor has scrolled off the painted window it is snapped, so it never resumes
    /// a stale smear on the way back. The returned rectangle is content space; the
    /// painter shifts it by scroll.
    fn build_caret(
        &mut self,
        i: usize,
        rect: Rect,
        focused: bool,
        now: Instant,
    ) -> Option<CaretPaint> {
        let scale = self.scale();
        let body_size = self.body_size;
        let config = self.caret_config;
        let visible = self.caret_visible;
        let from_pointer = self.caret_from_pointer;
        let fonts = &self.fonts;
        let ed = &mut self.editors[i];
        if !ed.has_tabs() {
            return None;
        }
        // Off-window: the cursor scrolled outside the painted band. Snap so a later
        // scroll back does not smear from a stale position, and draw nothing.
        if !ed.cursor_in_window() {
            ed.request_caret_snap();
            return None;
        }
        let cursor = ed.text.cursor();
        let (cx, cy, ch) = ed
            .layout_cache
            .as_ref()?
            .caret(fonts, ed.text.rope(), cursor);
        let caret_width = scale.px(CARET_WIDTH);
        let frame = CaretFrame {
            target: (cx as f32, cy as f32),
            height: ch as f32,
            width: caret_width,
            cursor,
            from_pointer,
            key: (rect.x, rect.w, body_size, scale.factor_120()),
        };
        let anim = ed.caret_anim_mut();
        anim.update(frame, &config, now);
        // Hidden this frame (unfocused window or blinked-off half): the anim is still
        // ticked above so it settles, but nothing is drawn.
        if !(focused && visible) {
            return None;
        }
        let paint_rect = anim.draw_rect();
        // A moving caret is a rounded smear; a settled one (and the whole
        // animation-off path) is the plain bar, drawn as a sharp Fill.
        let radius = if anim.in_flight() {
            caret_width.min(paint_rect.w.min(paint_rect.h) / 2)
        } else {
            0
        };
        Some(CaretPaint {
            rect: paint_rect,
            radius,
        })
    }

    /// The focused editor's caret animation when it is drawable this frame: the
    /// window has keyboard focus, an editor pane with tabs holds focus, and the
    /// cursor sits in its painted window. This is the caret whose in-flight motion
    /// keeps the loop ticking; nothing else animates the caret.
    fn drawable_caret_anim(&self) -> Option<&CaretAnim> {
        if !self.focused {
            return None;
        }
        let Focus::Editor(i) = self.focus else {
            return None;
        };
        let ed = &self.editors[i];
        (ed.has_tabs() && ed.cursor_in_window()).then(|| ed.caret_anim())
    }

    /// When the loop should next tick the caret while it eases, or `None` when it is
    /// settled or not drawable.
    fn caret_retry_at(&self) -> Option<Instant> {
        self.drawable_caret_anim().and_then(CaretAnim::retry_at)
    }

    /// While the caret eases and no compositor frame is pending, a tick whose integer
    /// rectangle did not change presents nothing and arms no frame callback, so the
    /// loop would sleep mid-motion. When its short retry deadline passes, mark a
    /// repaint so the next iteration ticks it again, until the rectangle moves (back
    /// to callback pacing) or the caret settles.
    fn fire_caret_anim(&mut self) {
        if self.frame_callback != 0 {
            return;
        }
        if self.caret_retry_at().is_some_and(|at| at <= Instant::now()) {
            self.dirty = true;
        }
    }

    fn handle(&mut self, msg: Message) -> Result<()> {
        let mut r = Reader::new(&msg.body);

        if msg.object == protocol::WL_DISPLAY {
            match msg.opcode {
                wl_display::EV_ERROR => {
                    let object = r.u32()?;
                    let code = r.u32()?;
                    let message = r.string()?;
                    return Err(Error::msg(format!(
                        "wayland protocol error: object {object} code {code}: {message}"
                    )));
                }
                wl_display::EV_DELETE_ID => {
                    let id = r.u32()?;
                    // Only recycle ids from the client range. A server-range id
                    // (>= 0xff000000) must never be handed back out as a fresh
                    // object; the server allocates those itself.
                    if id < SERVER_ID_BASE {
                        self.free_ids.push(id);
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        if msg.object == self.registry && msg.opcode == wl_registry::EV_GLOBAL {
            return self.on_global(&mut r);
        }

        if Some(msg.object) == self.pending_sync && msg.opcode == wl_callback::EV_DONE {
            self.pending_sync = None;
            return Ok(());
        }

        // The frame callback fired: the compositor is ready for the next frame, so
        // lift the pacing gate. The callback is one-shot; `render_frame` arms a
        // fresh one each time it presents.
        if self.frame_callback != 0
            && msg.object == self.frame_callback
            && msg.opcode == wl_callback::EV_DONE
        {
            self.frame_callback = 0;
            // Keep the animation going: if the caret is still easing, request the
            // next frame now that the compositor is ready for it.
            if self.drawable_caret_anim().is_some_and(CaretAnim::in_flight) {
                self.dirty = true;
            }
            return Ok(());
        }

        if msg.object == self.xdg_surface && msg.opcode == xdg_surface::EV_CONFIGURE {
            // Defer the ack to the frame that commits the matching-size buffer
            // (see `pending_configure`); keep only the latest serial.
            self.pending_configure = Some(r.u32()?);
            if let Some((lw, lh)) = self.pending_size.take() {
                // The configure size is logical (surface-local); the buffer is device
                // pixels. Record the logical size (for the viewport destination) and
                // resize the buffer to the scaled device size.
                self.scaling.logical = (lw, lh);
                self.scaling.synced = false;
                let (dw, dh) = self.device_size(lw, lh);
                self.resize_to(dw, dh);
            }
            self.configured = true;
            self.dirty = true;
            return Ok(());
        }

        // wp_fractional_scale_v1.preferred_scale: the compositor's scale as 120ths.
        // This is the primary scale source; adopting it reopens the fonts and
        // resizes the buffer to exact device resolution.
        if self.scaling.fractional_scale != 0
            && msg.object == self.scaling.fractional_scale
            && msg.opcode == wp_fractional_scale_v1::EV_PREFERRED_SCALE
        {
            let scale_120 = r.u32()?;
            self.set_scale_120(scale_120);
            return Ok(());
        }

        // wl_surface.enter/leave and wl_output.scale drive the integer fallback
        // (used only when the fractional path is absent).
        if msg.object == self.surface
            && (msg.opcode == wl_surface::EV_ENTER || msg.opcode == wl_surface::EV_LEAVE)
        {
            let output = r.u32()?;
            if msg.opcode == wl_surface::EV_ENTER {
                if !self.scaling.entered.contains(&output) {
                    self.scaling.entered.push(output);
                }
            } else {
                self.scaling.entered.retain(|&o| o != output);
            }
            self.refresh_integer_scale();
            return Ok(());
        }
        if msg.opcode == wl_output::EV_SCALE
            && self.scaling.outputs.iter().any(|(id, _)| *id == msg.object)
        {
            let factor = r.u32()? as i32;
            for out in &mut self.scaling.outputs {
                if out.0 == msg.object {
                    out.1 = factor.max(1);
                }
            }
            self.refresh_integer_scale();
            return Ok(());
        }

        if msg.object == self.toplevel {
            match msg.opcode {
                xdg_toplevel::EV_CONFIGURE => {
                    let w = r.u32()?;
                    let h = r.u32()?;
                    let _states = r.array()?;
                    // A zero dimension means the compositor leaves it to us. Clamp
                    // to MAX_DIMENSION so the buffer geometry stays well inside the
                    // buffer-size arithmetic (see `create_buffers`).
                    if w > 0 && h > 0 {
                        self.pending_size = Some((w.min(MAX_DIMENSION), h.min(MAX_DIMENSION)));
                    }
                }
                xdg_toplevel::EV_CLOSE => self.closed = true,
                _ => {}
            }
            return Ok(());
        }

        if Some(msg.object) == self.wm_base && msg.opcode == xdg_wm_base::EV_PING {
            let serial = r.u32()?;
            self.conn
                .request(msg.object, xdg_wm_base::PONG, &[Arg::Uint(serial)]);
            return Ok(());
        }

        if Some(msg.object) == self.seat && msg.opcode == wl_seat::EV_CAPABILITIES {
            let caps = r.u32()?;
            if caps & wl_seat::CAP_KEYBOARD != 0 && self.keyboard == 0 {
                let keyboard = self.alloc_id();
                self.conn
                    .request(msg.object, wl_seat::GET_KEYBOARD, &[Arg::NewId(keyboard)]);
                self.keyboard = keyboard;
            }
            if caps & wl_seat::CAP_POINTER != 0 && self.pointer == 0 {
                let pointer = self.alloc_id();
                self.conn
                    .request(msg.object, wl_seat::GET_POINTER, &[Arg::NewId(pointer)]);
                self.pointer = pointer;
                self.init_cursor_shape();
            }
            return Ok(());
        }

        if msg.object == self.keyboard {
            return self.on_keyboard(msg.opcode, &mut r);
        }

        if msg.object == self.pointer {
            return self.on_pointer(msg.opcode, &mut r);
        }

        if self.data_device != 0 && msg.object == self.data_device {
            return self.on_data_device(msg.opcode, &mut r);
        }

        if self.clipboard.source != 0 && msg.object == self.clipboard.source {
            return self.on_data_source(msg.opcode, &mut r);
        }

        if msg.object == self.clipboard.incoming_offer && msg.opcode == wl_data_offer::EV_OFFER {
            let mime = r.string()?;
            if is_text_mime(mime)
                && (self.clipboard.incoming_text_mime.is_none() || mime == CLIPBOARD_MIMES[0])
            {
                self.clipboard.incoming_text_mime = Some(mime.to_string());
            }
            return Ok(());
        }

        if self.presentation.feedback_id != 0 && msg.object == self.presentation.feedback_id {
            return self.on_dmabuf_feedback(msg.opcode, &mut r);
        }

        if msg.opcode == wl_buffer::EV_RELEASE {
            if msg.object == self.presentation.buffers[0] {
                self.presentation.busy[0] = false;
            } else if msg.object == self.presentation.buffers[1] {
                self.presentation.busy[1] = false;
            }
        }
        Ok(())
    }

    fn on_global(&mut self, r: &mut Reader) -> Result<()> {
        let name = r.u32()?;
        let interface = r.string()?;
        let version = r.u32()?;
        match interface {
            protocol::IFACE_COMPOSITOR => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_COMPOSITOR);
                self.compositor = Some(id);
            }
            protocol::IFACE_WM_BASE => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_WM_BASE);
                self.wm_base = Some(id);
            }
            protocol::IFACE_SEAT => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_SEAT);
                self.seat = Some(id);
            }
            protocol::IFACE_DATA_DEVICE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_DATA_DEVICE_MANAGER,
                );
                self.data_device_manager = Some(id);
            }
            protocol::IFACE_CURSOR_SHAPE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_CURSOR_SHAPE_MANAGER,
                );
                self.cursor_shape_manager = Some(id);
            }
            // Only useful at v4+ (the feedback object); an older advertisement
            // is left unbound and the GPU path simply stays unavailable.
            protocol::IFACE_DMABUF if version >= protocol::VERSION_DMABUF => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DMABUF);
                self.presentation.dmabuf = Some(id);
                self.presentation.dmabuf_version = version.min(protocol::VERSION_DMABUF);
            }
            // Explicit sync: bound only to be wired up
            // once a GPU backend exists; absent it, the bridge path is used.
            protocol::IFACE_DRM_SYNCOBJ => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_DRM_SYNCOBJ);
                self.presentation.syncobj_manager = Some(id);
            }
            // Display scaling. The fractional-scale manager and viewporter are the
            // primary path; each bound wl_output feeds the integer fallback (its
            // scale defaults to 1 until a `scale` event refines it).
            protocol::IFACE_OUTPUT => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_OUTPUT);
                self.scaling.outputs.push((id, 1));
            }
            protocol::IFACE_FRACTIONAL_SCALE_MANAGER => {
                let id = self.bind_capped(
                    name,
                    interface,
                    version,
                    protocol::VERSION_FRACTIONAL_SCALE_MANAGER,
                );
                self.scaling.fractional_manager = Some(id);
            }
            protocol::IFACE_VIEWPORTER => {
                let id = self.bind_capped(name, interface, version, protocol::VERSION_VIEWPORTER);
                self.scaling.viewporter = Some(id);
            }
            _ => {}
        }
        Ok(())
    }

    /// Common bookkeeping after a buffer edit: request a redraw, keep the caret
    /// on screen, and refresh the title's modified indicator.
    fn after_edit(&mut self) {
        self.dirty = true;
        self.bump_caret();
        self.reveal_active_caret();
        self.refresh_title();
    }

    /// Write the buffer to its backing file, atomically and in the file's
    /// original on-disk format. A buffer with no file (started without a path)
    /// has nowhere to go yet, so it reports that and stays put.
    fn save(&mut self) -> Result<()> {
        let i = self.active_editor();
        let Some(path) = self.editors[i].path.clone() else {
            eprintln!("bnknotes: no file to save to; start bnknotes with a path");
            return Ok(());
        };
        let bytes = file::encode(&self.editors[i].text.text(), &self.editors[i].format);
        file::save_atomic(&path, &bytes)?;
        self.editors[i].text.mark_saved();
        self.refresh_title();
        eprintln!("bnknotes: saved {}", path.display());
        Ok(())
    }

    /// The toplevel title: the focused editor's file name (or just the app name
    /// for an unnamed buffer), with a leading dot when it has unsaved changes.
    fn window_title(&self) -> String {
        let pane = &self.editors[self.active_editor()];
        if !pane.has_tabs() {
            return "bnknotes".to_string();
        }
        let modified = if pane.text.is_modified() { "• " } else { "" };
        match pane.path.as_ref().and_then(|p| p.file_name()) {
            Some(name) => format!("{modified}{} - bnknotes", name.to_string_lossy()),
            None => format!("{modified}bnknotes"),
        }
    }

    /// Send the toplevel title only if it changed, so the modified indicator and
    /// filename stay current without re-setting an identical string each frame.
    /// The title tracks the active pane, its file, and its modified flag, so every
    /// focus change routes through `focus_pane` to call this; the `desired`
    /// compare then dedupes the protocol request on the unchanged common path.
    fn refresh_title(&mut self) {
        let desired = self.window_title();
        if desired != self.title {
            self.conn.request(
                self.toplevel,
                xdg_toplevel::SET_TITLE,
                &[Arg::Str(&desired)],
            );
            self.title = desired;
        }
    }

    /// How long to block for input: the soonest of the next key-repeat, the next
    /// auto-scroll tick, and the next caret blink, or `None` when none is due.
    fn next_wake(&self) -> Option<Duration> {
        let now = Instant::now();
        let until = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        // Auto-scroll only ever runs in the focused editor (the one being dragged).
        let ed = &self.editors[self.active_editor()];
        let autoscroll = (ed.autoscroll != 0 && ed.selecting)
            .then_some(ed.autoscroll_at)
            .flatten()
            .map(until);
        // Only wake for the caret while focused, matching `fire_caret_blink`: a
        // stray `bump_caret` off a pointer event on an unfocused window must not
        // leave a past deadline that spins the loop.
        let caret = self
            .focused
            .then_some(self.caret_blink_at)
            .flatten()
            .map(until);
        // A caret still easing needs a follow-up tick, but only when no compositor
        // frame is pending: while a callback is outstanding, the frame-done event
        // paces the animation, so adding a deadline here would just wake early.
        let caret_anim = (self.frame_callback == 0)
            .then(|| self.caret_retry_at())
            .flatten()
            .map(until);
        [self.repeat_timeout(), autoscroll, caret, caret_anim]
            .into_iter()
            .flatten()
            .min()
    }

    /// Bind the just-advertised global at the version we negotiate: the
    /// compositor's advertised `version` capped at `max`, the highest interface
    /// version we support. Allocates the object id, sends the registry bind, and
    /// returns the id for the caller to store. The shared body of every
    /// [`on_global`](Self::on_global) arm.
    fn bind_capped(&mut self, name: u32, interface: &str, version: u32, max: u32) -> u32 {
        let id = self.alloc_id();
        self.conn.request(
            self.registry,
            wl_registry::BIND,
            &[
                Arg::Uint(name),
                Arg::Bind {
                    interface,
                    version: version.min(max),
                    new_id: id,
                },
            ],
        );
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `Focus` is not `Debug`, so compare with `==` inside `assert!`.

    #[test]
    fn resolve_root_prefers_a_directory_argument() {
        let dir = std::env::temp_dir();
        let (root, file) = resolve_root(Some(&dir), None, Path::new("/cwd"));
        assert_eq!(root, dir);
        assert_eq!(file, None);
    }

    #[test]
    fn resolve_root_roots_a_file_argument_at_its_parent() {
        let dir = std::env::temp_dir();
        let arg = dir.join("note.md");
        let (root, opened) = resolve_root(Some(&arg), None, Path::new("/cwd"));
        assert_eq!(root, dir);
        assert_eq!(opened.as_deref(), Some(arg.as_path()));
    }

    #[test]
    fn resolve_root_uses_the_session_root_when_it_is_a_directory() {
        let dir = std::env::temp_dir();
        let (root, file) = resolve_root(None, Some(&dir), Path::new("/cwd"));
        assert_eq!(root, dir);
        assert_eq!(file, None);
    }

    #[test]
    fn resolve_root_ignores_a_nonexistent_session_root() {
        let bogus = Path::new("/does/not/exist/anywhere");
        let (root, file) = resolve_root(None, Some(bogus), Path::new("/cwd"));
        assert_eq!(root, Path::new("/cwd"));
        assert_eq!(file, None);
    }

    #[test]
    fn resolve_root_with_no_argument_or_session_uses_cwd() {
        let (root, file) = resolve_root(None, None, Path::new("/here"));
        assert_eq!(root, Path::new("/here"));
        assert_eq!(file, None);
    }

    #[test]
    fn initial_focus_a_file_argument_always_focuses_editor_a() {
        // A file argument overrides any saved focus, even tree or pane B.
        assert!(initial_focus(true, store::Focus::Tree, true, true) == Focus::Editor(0));
        assert!(initial_focus(true, store::Focus::PaneB, true, true) == Focus::Editor(0));
    }

    #[test]
    fn initial_focus_restores_the_saved_region_when_shown() {
        assert!(initial_focus(false, store::Focus::Tree, true, true) == Focus::Tree);
        assert!(initial_focus(false, store::Focus::PaneB, true, true) == Focus::Editor(1));
        assert!(initial_focus(false, store::Focus::PaneA, true, true) == Focus::Editor(0));
    }

    #[test]
    fn initial_focus_falls_back_to_editor_a_for_a_hidden_region() {
        // Saved tree focus but the tree is hidden -> editor A.
        assert!(initial_focus(false, store::Focus::Tree, false, true) == Focus::Editor(0));
        // Saved pane-B focus but the second pane is closed -> editor A.
        assert!(initial_focus(false, store::Focus::PaneB, true, false) == Focus::Editor(0));
    }
}
