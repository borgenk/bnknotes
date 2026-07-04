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

use self::clipboard::{is_text_mime, CLIPBOARD_MIMES};
use self::input::{DEFAULT_REPEAT_DELAY_MS, DEFAULT_REPEAT_RATE};
use self::present::ExplicitSync;
use crate::conn::{Connection, Fill};
use crate::display::DisplayList;
use crate::dmabuf;
use crate::error::{Error, Result};
use crate::file;
use crate::freetype::Fonts;
use crate::gpu;
use crate::layout::Rect;
use crate::markdown;
use crate::paint::{build_display_list, build_find_paint, push_splitter, push_tree, PaintParams};
use crate::pane::{restore_editor, Columns, EditorPane, Focus, PaneLayout, TreePane};
use crate::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device_manager, wl_data_offer, wl_display,
    wl_registry, wl_seat, wl_surface, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::scroll;
use crate::store;
use crate::theme::FONT_SIZE;
use crate::tree;
use crate::vulkan;
use crate::wire::{Arg, Message, Reader};
use crate::xkb::Xkb;

/// Initial window size; the compositor may resize us to whatever it likes.
const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;
/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display (8K is 7680 wide), it just keeps a
/// bogus or hostile size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

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

struct State {
    conn: Connection,
    /// The font opened at every size the styles need (body and heading sizes),
    /// so each line can render at its own size.
    fonts: Fonts,
    xkb: Xkb,
    /// The two content panes (editor A and B) and the file-tree sidebar, laid
    /// out as adjustable full-height columns. `focus` selects which region takes
    /// keyboard input; a click moves focus (and, on a splitter, starts a drag).
    editors: [EditorPane; 2],
    tree: TreePane,
    panes: PaneLayout,
    focus: Focus,
    /// The editor last given focus, so a file opened from the tree (which takes
    /// focus itself) lands in the content pane the user was last editing.
    last_editor: usize,
    /// While a splitter is dragged, which one: 0 = tree|A edge, 1 = A|B edge.
    drag_splitter: Option<usize>,
    /// The toplevel title last sent, so it is only re-set when it changes.
    title: String,
    /// The modified flag the current `title` reflects, so `refresh_title` can skip
    /// the format!+compare on the common case (a cursor move leaves it unchanged).
    /// `None` until the first title is sent.
    title_modified: Option<bool>,

    /// Current surface size in pixels.
    width: u32,
    height: u32,

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
    /// The dmabuf global (None when absent or below v4, which the GPU path
    /// requires), the version it was bound at, the feedback object created from
    /// it, and the accumulating feedback the GPU backend consumes. They are the
    /// negotiation half of the GPU path.
    dmabuf: Option<u32>,
    dmabuf_version: u32,
    dmabuf_feedback: u32,
    dmabuf_fb: dmabuf::FeedbackState,
    /// The Vulkan backend: the device plus one exportable image per swap slot.
    /// Always `Some` once `bring_up` succeeds (a missing device hard-exits at
    /// startup, so the editor never runs without a GPU); it stays an `Option`
    /// only because it is created mid-bring-up, after the dmabuf feedback the
    /// device selection needs has been committed.
    gpu: Option<vulkan::Gpu>,
    gpu_images: Vec<vulkan::GpuImage>,
    /// The compositor∩device modifier list images are allocated with,
    /// computed once at GPU init and reused across resizes.
    gpu_modifiers: Vec<u64>,
    /// The GPU path's persistent glyph/emoji atlases (CPU mirrors plus slot
    /// records); steady-state frames insert nothing and upload nothing.
    gpu_glyphs: gpu::GlyphCache,
    /// The explicit-sync manager global (`wp_linux_drm_syncobj_manager_v1`) when
    /// the compositor offers it; `None` means the GPU path uses the implicit-sync
    /// bridge.
    drm_syncobj_manager: Option<u32>,
    /// Active explicit-sync objects, negotiated once a GPU backend is up and the
    /// manager is present. `None` keeps the bridge.
    explicit_sync: Option<ExplicitSync>,
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
    /// Whether the surface holds the keyboard focus (`wl_keyboard.enter` to
    /// `leave`). The caret only blinks (and only wakes the loop on its timer)
    /// while focused; an unfocused window hides the caret and stays idle.
    focused: bool,

    // Key auto-repeat. The compositor sends one press; the client repeats the
    // held key on a timer until it is released.
    repeat_delay: Duration,
    /// `None` disables repeat (the compositor reported a zero rate).
    repeat_interval: Option<Duration>,
    repeat_key: Option<u32>,
    repeat_at: Option<Instant>,
    /// The latest input event serial, needed to set the clipboard selection.
    last_serial: u32,

    // Clipboard. When we own the selection, `clipboard_source` is our live
    // wl_data_source; `clipboard_data` is the bytes it serves and `clipboard_mimes`
    // the types it advertises (text for a text selection, `image/png`/`image/jpeg`
    // for a copied image). Incoming offers are tracked so a paste can request the
    // right MIME type.
    clipboard_source: u32,
    clipboard_data: Vec<u8>,
    clipboard_mimes: Vec<&'static str>,
    incoming_offer: u32,
    incoming_text_mime: Option<String>,
    selection_offer: u32,
    selection_text_mime: Option<String>,

    // Double-buffered GPU images: two exportable dmabuf-backed buffers,
    // ping-ponged so we only render into a buffer the compositor has released.
    buffers: [u32; 2],
    busy: [bool; 2],
    /// The `(width, height)` the current buffers were allocated at. A configure
    /// only records the new surface size; the buffers are reallocated lazily in
    /// `render_frame` when this no longer matches, so a burst of configures
    /// during an interactive resize costs one reallocation per painted frame
    /// rather than one per configure.
    buffer_size: (u32, u32),
    /// The display list last committed to the compositor, i.e. what is on screen.
    /// The new frame is diffed against this to decide whether to present at all
    /// and which regions to report as damaged.
    present_list: Rc<DisplayList>,

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
    /// Count of frames presented, for the stats line.
    frame_count: u64,
    /// When the previous frame was presented, for the inter-frame gap in stats.
    last_present: Option<Instant>,
    /// Explicit-sync accounting for the stats line: frames that published a real
    /// render-done fence as the acquire point, versus frames where the driver
    /// could not export one so the GPU path CPU-waited the frame to completion
    /// instead. A climbing `cpu_wait` count means explicit sync negotiated but is
    /// degrading to a synchronous wait (fence export is the usual suspect on some
    /// drivers); both stay at zero on the implicit-sync bridge.
    explicit_fence_frames: u64,
    explicit_cpu_wait_frames: u64,
}

impl State {
    fn new(path: Option<PathBuf>) -> Result<Self> {
        let conn = Connection::connect()?;
        // Open the font at every size the styles use, so headings can render
        // larger than body text without re-opening mid-frame.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
        // The saved layout: window/column sizes, tree root and open folders, and
        // each pane's last file with its caret and scroll. Missing or corrupt
        // falls back to the first-run default without failing.
        let session = store::load();

        // Resolve the tree root: a directory argument roots there, a file roots
        // at its parent, and no argument uses the last session's root or the
        // working directory. A file argument also opens into editor A.
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let (root, cli_file) = match &path {
            Some(p) if p.is_dir() => (p.clone(), None),
            Some(p) => (
                p.parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| cwd.clone()),
                Some(p.clone()),
            ),
            None => (
                session
                    .tree_root
                    .clone()
                    .filter(|r| r.is_dir())
                    .unwrap_or(cwd),
                None,
            ),
        };

        // Editor A: the file argument wins; otherwise restore the session's
        // pane A. Editor B always restores from the session (empty if none).
        let editor_a = match &cli_file {
            Some(p) => restore_editor(Some(p), &store::PaneSession::default()),
            None => restore_editor(session.panes[0].path.as_deref(), &session.panes[0]),
        };
        let editor_b = restore_editor(session.panes[1].path.as_deref(), &session.panes[1]);

        let expanded: BTreeSet<PathBuf> = session.expanded.iter().cloned().collect();
        let tree = TreePane {
            tree: tree::Tree::new(root, expanded),
            scroll_y: 0,
            content_height: 0,
            hovered: None,
            selected: 0,
        };
        let panes = PaneLayout {
            tree_width: session.tree_width as i32,
            split_ratio: session.split_ratio,
            tree_visible: session.tree_visible,
            second_open: session.second_open,
        };
        // A file argument focuses its pane; otherwise the saved focus, guarded so
        // it never points at a hidden region.
        let focus = if cli_file.is_some() {
            Focus::Editor(0)
        } else {
            match session.focus {
                store::Focus::Tree if panes.tree_visible => Focus::Tree,
                store::Focus::PaneB if panes.second_open => Focus::Editor(1),
                store::Focus::Tree | store::Focus::PaneA | store::Focus::PaneB => Focus::Editor(0),
            }
        };

        Ok(Self {
            conn,
            fonts,
            xkb: Xkb::new()?,
            editors: [editor_a, editor_b],
            tree,
            panes,
            focus,
            last_editor: match focus {
                Focus::Editor(i) => i,
                Focus::Tree => 0,
            },
            drag_splitter: None,
            title: String::new(),
            title_modified: None,
            width: INIT_WIDTH,
            height: INIT_HEIGHT,
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
            dmabuf: None,
            dmabuf_version: 0,
            dmabuf_feedback: 0,
            dmabuf_fb: dmabuf::FeedbackState::default(),
            gpu: None,
            gpu_images: Vec::new(),
            gpu_modifiers: Vec::new(),
            gpu_glyphs: gpu::GlyphCache::new(),
            drm_syncobj_manager: None,
            explicit_sync: None,
            cursor_shape_manager: None,
            cursor_shape_device: 0,
            pointer_serial: 0,
            cursor_shape: 0,
            pointer_x: 0.0,
            pointer_y: 0.0,
            caret_visible: true,
            caret_blink_at: None,
            focused: false,
            repeat_delay: Duration::from_millis(DEFAULT_REPEAT_DELAY_MS),
            repeat_interval: Some(Duration::from_secs(1) / DEFAULT_REPEAT_RATE),
            repeat_key: None,
            repeat_at: None,
            last_serial: 0,
            clipboard_source: 0,
            clipboard_data: Vec::new(),
            clipboard_mimes: Vec::new(),
            incoming_offer: 0,
            incoming_text_mime: None,
            selection_offer: 0,
            selection_text_mime: None,
            buffers: [0, 0],
            busy: [false, false],
            buffer_size: (0, 0),
            present_list: Rc::default(),
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
            frame_count: 0,
            last_present: None,
            explicit_fence_frames: 0,
            explicit_cpu_wait_frames: 0,
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

    /// The current column layout for this window size.
    fn columns(&self) -> Columns {
        self.panes.columns(self.width as i32, self.height as i32)
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
        if let Some(gpu) = &self.gpu {
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
        let pane = |i: usize| store::PaneSession {
            path: self.editors[i].path.clone(),
            cursor: self.editors[i].text.cursor() as u64,
            scroll_y: self.editors[i].scroll_y,
        };
        store::Session {
            window: (self.width, self.height),
            tree_width: self.panes.tree_width.max(0) as u32,
            split_ratio: self.panes.split_ratio,
            tree_visible: self.panes.tree_visible,
            second_open: self.panes.second_open,
            focus,
            tree_root: Some(self.tree.tree.root().to_path_buf()),
            expanded: self.tree.tree.expanded_dirs(),
            panes: [pane(0), pane(1)],
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
        let mut list: DisplayList = Vec::new();
        if let Some(rect) = cols.tree {
            let focused = self.focus == Focus::Tree;
            push_tree(&mut list, &self.fonts, &mut self.tree, rect, focused);
        }
        for i in 0..self.editors.len() {
            let Some(rect) = cols.editors[i] else {
                continue;
            };
            self.editors[i].ensure_layout(&self.fonts, rect);
            let ch = self.editors[i].content_height;
            self.editors[i].scroll_y = scroll::clamp(self.editors[i].scroll_y, ch, rect.h);
            let focused = self.focus == Focus::Editor(i);
            let params = PaintParams {
                width: rect.w as usize,
                height: rect.h as usize,
                scroll_y: self.editors[i].scroll_y,
                selection: self.editors[i].text.selection(),
                cursor: self.editors[i].text.cursor(),
                // Only the focused pane blinks a caret; others show none.
                caret_visible: focused && self.caret_visible,
                find: build_find_paint(&self.editors[i].find),
            };
            if let Some(layout) = self.editors[i].layout_cache.as_ref() {
                list.append(&mut build_display_list(
                    &self.fonts,
                    layout,
                    self.editors[i].text.rope(),
                    &params,
                ));
            }
        }
        for s in cols.splitters.into_iter().flatten() {
            push_splitter(&mut list, s);
        }
        Rc::new(list)
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
            return Ok(());
        }

        if msg.object == self.xdg_surface && msg.opcode == xdg_surface::EV_CONFIGURE {
            // Defer the ack to the frame that commits the matching-size buffer
            // (see `pending_configure`); keep only the latest serial.
            self.pending_configure = Some(r.u32()?);
            if let Some((w, h)) = self.pending_size.take() {
                self.resize_to(w, h);
            }
            self.configured = true;
            self.dirty = true;
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

        if self.clipboard_source != 0 && msg.object == self.clipboard_source {
            return self.on_data_source(msg.opcode, &mut r);
        }

        if msg.object == self.incoming_offer && msg.opcode == wl_data_offer::EV_OFFER {
            let mime = r.string()?;
            if is_text_mime(mime)
                && (self.incoming_text_mime.is_none() || mime == CLIPBOARD_MIMES[0])
            {
                self.incoming_text_mime = Some(mime.to_string());
            }
            return Ok(());
        }

        if self.dmabuf_feedback != 0 && msg.object == self.dmabuf_feedback {
            return self.on_dmabuf_feedback(msg.opcode, &mut r);
        }

        if msg.opcode == wl_buffer::EV_RELEASE {
            if msg.object == self.buffers[0] {
                self.busy[0] = false;
            } else if msg.object == self.buffers[1] {
                self.busy[1] = false;
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
                let id = self.alloc_id();
                self.bind(
                    name,
                    interface,
                    version.min(protocol::VERSION_COMPOSITOR),
                    id,
                );
                self.compositor = Some(id);
            }
            protocol::IFACE_WM_BASE => {
                let id = self.alloc_id();
                self.bind(name, interface, version.min(protocol::VERSION_WM_BASE), id);
                self.wm_base = Some(id);
            }
            protocol::IFACE_SEAT => {
                let id = self.alloc_id();
                self.bind(name, interface, version.min(protocol::VERSION_SEAT), id);
                self.seat = Some(id);
            }
            protocol::IFACE_DATA_DEVICE_MANAGER => {
                let id = self.alloc_id();
                self.bind(
                    name,
                    interface,
                    version.min(protocol::VERSION_DATA_DEVICE_MANAGER),
                    id,
                );
                self.data_device_manager = Some(id);
            }
            protocol::IFACE_CURSOR_SHAPE_MANAGER => {
                let id = self.alloc_id();
                self.bind(
                    name,
                    interface,
                    version.min(protocol::VERSION_CURSOR_SHAPE_MANAGER),
                    id,
                );
                self.cursor_shape_manager = Some(id);
            }
            // Only useful at v4+ (the feedback object); an older advertisement
            // is left unbound and the GPU path simply stays unavailable.
            protocol::IFACE_DMABUF if version >= protocol::VERSION_DMABUF => {
                let id = self.alloc_id();
                let bound = version.min(protocol::VERSION_DMABUF);
                self.bind(name, interface, bound, id);
                self.dmabuf = Some(id);
                self.dmabuf_version = bound;
            }
            // Explicit sync: bound only to be wired up
            // once a GPU backend exists; absent it, the bridge path is used.
            protocol::IFACE_DRM_SYNCOBJ => {
                let id = self.alloc_id();
                self.bind(
                    name,
                    interface,
                    version.min(protocol::VERSION_DRM_SYNCOBJ),
                    id,
                );
                self.drm_syncobj_manager = Some(id);
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
        let ed = &self.editors[self.active_editor()];
        let modified = if ed.text.is_modified() { "• " } else { "" };
        match ed.path.as_ref().and_then(|p| p.file_name()) {
            Some(name) => format!("{modified}{} - bnknotes", name.to_string_lossy()),
            None => format!("{modified}bnknotes"),
        }
    }

    /// Send the toplevel title only if it changed, so the modified indicator and
    /// filename stay current without re-setting an identical string each frame.
    fn refresh_title(&mut self) {
        // The title depends only on the modified flag and the (session-fixed)
        // filename, so a cursor move that left the modified state untouched cannot
        // change it: skip the format!+compare on that common path.
        let modified = self.editors[self.active_editor()].text.is_modified();
        if self.title_modified == Some(modified) {
            return;
        }
        let desired = self.window_title();
        if desired != self.title {
            self.conn.request(
                self.toplevel,
                xdg_toplevel::SET_TITLE,
                &[Arg::Str(&desired)],
            );
            self.title = desired;
        }
        self.title_modified = Some(modified);
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
        [self.repeat_timeout(), autoscroll, caret]
            .into_iter()
            .flatten()
            .min()
    }

    fn bind(&mut self, name: u32, interface: &str, version: u32, id: u32) {
        self.conn.request(
            self.registry,
            wl_registry::BIND,
            &[
                Arg::Uint(name),
                Arg::Bind {
                    interface,
                    version,
                    new_id: id,
                },
            ],
        );
    }
}
