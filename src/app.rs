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
//! reallocating its shm buffers. Rendering double-buffers so the compositor never
//! reads a frame we are still painting.

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::conn::{Connection, Fill};
use crate::display::{self, DisplayList, DrawCmd};
use crate::error::{Error, Result};
use crate::ffi;
use crate::file::{self, FileFormat};
use crate::freetype::{FaceKey, Fonts, Metrics};
use crate::image::ImageStore;
use crate::layout::{face_key, image_button_rect, Gutter, ImageUi, Layout, Rect};
use crate::markdown;
use crate::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device, wl_data_device_manager,
    wl_data_offer, wl_data_source, wl_display, wl_keyboard, wl_pointer, wl_registry, wl_seat,
    wl_shm, wl_shm_pool, wl_surface, wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
    xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::render::{self, Canvas};
use crate::scroll;
use crate::shm::Pool;
use crate::text::TextBuffer;
use crate::wire::{Arg, Message, Reader};
use crate::xkb::{KeyAction, Xkb};

/// Initial window size; the compositor may resize us to whatever it likes.
const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;
/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display (8K is 7680 wide), it just keeps a
/// bogus or hostile size from blowing up the shm-pool arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// Background, text, and selection colors (0x00RRGGBB), and the pixel size: a
/// dark palette on base hue ~212.
const BACKGROUND: u32 = 0x001c_2127;
const FOREGROUND: u32 = 0x00da_dada;
/// The accent blue (`hsl(215,75%,70%)`) pre-blended for selection: the fill is
/// opaque, so this is the accent at a set opacity, high enough to stay legible
/// over the darker code panel.
const SELECTION_COLOR: u32 = 0x0041_5776;
/// Dimmed color for markdown syntax markers (a heading's `#`s, a code span's
/// backticks), so they read as syntax rather than content.
const MARKER_COLOR: u32 = 0x005d_6c7e;
/// Inline code: a tinted background panel and a distinct foreground, so a code
/// span stands out from prose the way it does in a rendered document.
const CODE_COLOR: u32 = 0x00f3_8ba8;
const CODE_BG: u32 = 0x0028_2c34;
/// Corner radius of a fenced code block's panel: a very subtle rounding, no border.
const CODE_RADIUS: i32 = 4;
/// An inline link's text color (the accent blue, `hsl(215,75%,70%)`); the
/// text is also underlined so a link reads as a link without its `[..](..)` markup.
const LINK_COLOR: u32 = 0x005c_9cf6;
/// A selected image's border (a bright accent so it reads as "selected") and its
/// thickness; the corner button's panel and corner radius. The button glyph reuses
/// [`FOREGROUND`].
const IMAGE_BORDER_COLOR: u32 = 0x005a_9ce0;
const IMAGE_BORDER_THICKNESS: i32 = 2;
const IMAGE_BUTTON_BG: u32 = 0x0020_242b;
const IMAGE_BUTTON_RADIUS: i32 = 4;
/// Per-level heading text colors (H1..H6): H1/H2 read as normal text, then blue,
/// yellow, red, and a muted gray for H6 (which is also smaller than body, so color
/// is its main cue). Indexed by level minus one.
const HEADING_COLORS: [u32; 6] = [
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
const QUOTE_BAR_COLOR: u32 = 0x003f_444b;
const QUOTE_BAR_WIDTH: i32 = 3;
const BULLET: &str = "•";
/// A thematic break (`---`) renders as a thin horizontal line across the content
/// width once the cursor leaves it, in a muted line tone.
const RULE_COLOR: u32 = 0x003f_444b;
const RULE_THICKNESS: i32 = 2;
/// Body text pixel size; headings scale up from it (see [`markdown::font_size`]).
pub(crate) const FONT_SIZE: u32 = 14;
/// Inset of the text from the top-left corner (both axes).
pub(crate) const MARGIN: i32 = 40;
/// What the Tab key inserts: spaces (a "soft tab"), not a literal `'\t'`. This
/// keeps the buffer free of tab characters and spares the renderer any tab-stop
/// logic. Rendering literal tabs found in a *loaded* file is a separate concern.
const SOFT_TAB: &str = "    ";
/// The caret: a thin vertical bar at the cursor position.
const CARET_COLOR: u32 = 0x00ff_ffff;
const CARET_WIDTH: i32 = 2;
/// How long the caret stays shown, then hidden, while idle. Activity resets it
/// to shown, so it never blinks out mid-keystroke.
const CARET_BLINK: Duration = Duration::from_millis(500);
/// Pixels scrolled per unit of wl_pointer.axis value (wheel/touchpad). One wheel
/// detent is ~15 axis units, so this lands a notch at roughly three body lines.
const SCROLL_FACTOR: f32 = 4.0;

/// Two clicks within this window, and within [`MULTI_CLICK_RADIUS`] pixels,
/// count as a double-click (then triple, ...).
const MULTI_CLICK_INTERVAL: Duration = Duration::from_millis(400);
/// How far apart successive clicks may land yet still chain into a multi-click.
const MULTI_CLICK_RADIUS: f32 = 5.0;

/// While drag-selecting, a pointer this close to the top or bottom edge (in
/// pixels) auto-scrolls the view so the selection can extend past what is shown.
const AUTOSCROLL_MARGIN: i32 = 24;
/// Pixels scrolled per auto-scroll tick, and how often a tick fires while the
/// pointer is held in the edge zone.
const AUTOSCROLL_STEP: i32 = 24;
const AUTOSCROLL_INTERVAL: Duration = Duration::from_millis(16);

/// Fallback auto-repeat timing if the compositor never sends repeat_info.
const DEFAULT_REPEAT_DELAY_MS: u64 = 400;
const DEFAULT_REPEAT_RATE: u32 = 25;
/// Cap on repeats fired per wake-up, so a stalled frame cannot flood input.
const REPEAT_CATCHUP_CAP: u32 = 16;

/// MIME types we advertise when putting text on the clipboard, and accept when
/// pasting. The charset variant is listed first as the preferred form.
const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

/// Largest file the editor will open, as a guard against mapping a huge file into
/// memory on startup. Generous for a text editor; a bigger file is a clean error,
/// not an out-of-memory crash.
const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

pub fn run() -> Result<()> {
    // An optional path to open; everything after it is ignored for now.
    let path = std::env::args_os().nth(1).map(PathBuf::from);
    State::new(path)?.bring_up()
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
/// version, surface width, body font size, the active range's logical line span
/// (the marker reveal depends only on which lines the range touches, not the
/// exact byte offsets), and the selected / raw-pinned image (by line-start
/// offset). A scroll, idle redraw, or cursor move within one line leaves all of
/// these unchanged and reuses the cached layout.
type LayoutKey = (u64, u32, u32, usize, usize, Option<usize>, Option<usize>);

struct State {
    conn: Connection,
    /// The font opened at every size the styles need (body and heading sizes),
    /// so each line can render at its own size.
    fonts: Fonts,
    xkb: Xkb,
    text: TextBuffer,
    /// The file backing the buffer, if any, and the on-disk conventions to
    /// reproduce when saving it.
    path: Option<PathBuf>,
    format: FileFormat,
    /// Decoded inline images, keyed by their `![](path)` destination, with
    /// relative paths resolved against the document's directory. Refreshed on each
    /// layout rebuild and borrowed by the painter to blit the pictures.
    images: ImageStore,
    /// The image (by its logical line's start offset) the user has selected: it
    /// draws a border and corner button, and `Ctrl+C` copies it. `None` when no
    /// image is selected. Cleared by any keyboard action or a click off the image.
    selected_image: Option<usize>,
    /// The image (by line start) whose raw `![alt](path)` source is pinned open by
    /// its corner button, shown above the picture even with the cursor away. The
    /// cursor-on-line reveal is separate; this survives until cleared.
    raw_image: Option<usize>,
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
    shm: Option<u32>,
    wm_base: Option<u32>,
    seat: Option<u32>,
    surface: u32,
    xdg_surface: u32,
    toplevel: u32,
    keyboard: u32,
    pointer: u32,
    data_device_manager: Option<u32>,
    data_device: u32,
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

    // Pointer state: latest position (surface-local pixels) and whether a left
    // button drag is currently extending a selection.
    pointer_x: f32,
    pointer_y: f32,
    selecting: bool,
    /// While a drag is extending a selection, the active range used for marker
    /// reveal is frozen to this (the press point), so dragging across the document
    /// does not keep flipping headings/lists/code between rendered and raw markdown
    /// under the pointer. The live selection still highlights; only the reveal is
    /// pinned. Cleared on button release, when reveal catches up to the final
    /// selection. Pinning it also keeps the layout cache key constant for the whole
    /// drag, so no re-layout happens per motion event.
    frozen_reveal: Option<(usize, usize)>,
    /// Click chaining for double/triple-click: when the last left press fired,
    /// where, and how many have chained so far (1 = single, 2 = double, ...).
    last_click: Option<Instant>,
    last_click_pos: (f32, f32),
    click_count: u32,
    /// Auto-scroll while drag-selecting at an edge: pixels per tick (0 = off,
    /// sign = direction) and when the next tick is due.
    autoscroll: i32,
    autoscroll_at: Option<Instant>,
    /// Sticky x for vertical motion, so repeated Up/Down keep a column.
    goal_x: Option<f32>,
    /// How far the viewport is scrolled down the document, in pixels.
    scroll_y: i32,
    /// Caret blink: whether the caret is currently shown, and when it next
    /// toggles. Activity (`bump_caret`) resets it to shown.
    caret_visible: bool,
    caret_blink_at: Option<Instant>,
    /// Whether the surface holds the keyboard focus (`wl_keyboard.enter` to
    /// `leave`). The caret only blinks (and only wakes the loop on its timer)
    /// while focused; an unfocused window hides the caret and stays idle.
    focused: bool,

    /// Memoized layout, reused until its inputs change. The key is the content
    /// version, the surface width, and the active (cursor/selection) range, the
    /// only things the layout's geometry depends on; a scroll or idle redraw
    /// reuses it instead of re-wrapping and re-measuring the whole document.
    layout_cache: Option<Layout>,
    layout_key: Option<LayoutKey>,
    /// The laid-out document's pixel height plus top and bottom margins, refreshed
    /// whenever the layout is rebuilt. Cached so the scroll path can clamp without
    /// cloning the whole layout just to read its extent.
    content_height: i32,

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

    // Double-buffered shm: one pool holding two frames, ping-ponged so we only
    // paint a buffer the compositor has released.
    pool: Option<Pool>,
    pool_id: u32,
    buffers: [u32; 2],
    busy: [bool; 2],
    frame_len: usize,
    /// The display list each back buffer currently holds, so a repaint diffs the
    /// new frame against what is already in *that* buffer and clears only the
    /// changed regions. The two age independently (each is painted on alternate
    /// frames), which is why damage is tracked per buffer. Reset to empty whenever
    /// the buffers are recreated, forcing a full repaint of the fresh memory.
    buffer_lists: [DisplayList; 2],
    /// The display list last committed to the compositor, i.e. what is on screen.
    /// The new frame is diffed against this to decide whether to present at all
    /// and which regions to report as damaged.
    present_list: DisplayList,

    // Id allocation: a bump counter plus a free list fed by `delete_id`.
    next_id: u32,
    free_ids: Vec<u32>,

    // Loop state.
    pending_sync: Option<u32>,
    /// A compositor-requested size awaiting the next configure ack.
    pending_size: Option<(u32, u32)>,
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
}

impl State {
    fn new(path: Option<PathBuf>) -> Result<Self> {
        let conn = Connection::connect()?;
        // Open the font at every size the styles use, so headings can render
        // larger than body text without re-opening mid-frame.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
        // Load the file (if one was named) before showing the window, so a read
        // error surfaces as a clean startup failure instead of an empty editor.
        let (text, format) = match &path {
            Some(p) => {
                let (contents, format) = load(p)?;
                (TextBuffer::from_text(contents), format)
            }
            None => (TextBuffer::new(), FileFormat::default()),
        };
        // Inline images resolve relative paths against the document's directory.
        // An unsaved buffer has none, so only absolute paths resolve there.
        let image_dir = path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        Ok(Self {
            conn,
            fonts,
            xkb: Xkb::new()?,
            text,
            path,
            format,
            images: ImageStore::new(image_dir),
            selected_image: None,
            raw_image: None,
            title: String::new(),
            title_modified: None,
            width: INIT_WIDTH,
            height: INIT_HEIGHT,
            registry: 0,
            compositor: None,
            shm: None,
            wm_base: None,
            seat: None,
            surface: 0,
            xdg_surface: 0,
            toplevel: 0,
            keyboard: 0,
            pointer: 0,
            data_device_manager: None,
            data_device: 0,
            cursor_shape_manager: None,
            cursor_shape_device: 0,
            pointer_serial: 0,
            cursor_shape: 0,
            pointer_x: 0.0,
            pointer_y: 0.0,
            selecting: false,
            frozen_reveal: None,
            last_click: None,
            last_click_pos: (0.0, 0.0),
            click_count: 0,
            autoscroll: 0,
            autoscroll_at: None,
            goal_x: None,
            scroll_y: 0,
            caret_visible: true,
            caret_blink_at: None,
            focused: false,
            layout_cache: None,
            layout_key: None,
            content_height: 0,
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
            pool: None,
            pool_id: 0,
            buffers: [0, 0],
            busy: [false, false],
            frame_len: 0,
            buffer_lists: [Vec::new(), Vec::new()],
            present_list: Vec::new(),
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            configured: false,
            closed: false,
            dirty: true,
            frame_callback: 0,
            stats: std::env::var_os("BNKNOTES_STATS").is_some(),
            frame_count: 0,
            last_present: None,
        })
    }

    fn alloc_id(&mut self) -> u32 {
        self.free_ids.pop().unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        })
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
        self.shm
            .ok_or_else(|| Error::msg("compositor has no wl_shm"))?;
        self.seat
            .ok_or_else(|| Error::msg("compositor has no wl_seat"))?;

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

        // Empty commit to get the initial configure; the first frame is drawn
        // by the loop once `configured` is set.
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.run_until(|s| s.configured)?;

        eprintln!(
            "bnknotes: type to edit; Enter, Tab, arrows/Home/End, PageUp/Down, \
             Ctrl+arrows by word, Ctrl+Home/End to ends, Shift selects; \
             click/drag, Shift+click extends, double/triple-click select word/line; \
             Ctrl+Backspace/Delete delete a word; Ctrl+A/C/X/V select-all/copy/cut/paste, \
             Ctrl+Z/Y undo/redo, Ctrl+S save; held keys repeat; close to exit"
        );
        self.run_until(|s| s.closed)?;
        Ok(())
    }

    /// Allocate the shm pool and its two buffers at the current size.
    fn create_buffers(&mut self) -> Result<()> {
        let shm = self
            .shm
            .ok_or_else(|| Error::msg("creating buffers before wl_shm was bound"))?;
        // Compute the geometry in usize: a u32 `width * 4 * height` wraps near a
        // ~46k-px square, which would size the pool too small and let the
        // compositor read past the mapping. The configure path clamps dimensions
        // to MAX_DIMENSION, so the only remaining bound is the wire's i32 pool-size
        // argument, which a near-MAX_DIMENSION square could still exceed.
        let width = self.width as usize;
        let height = self.height as usize;
        let stride = width * 4;
        let frame_len = stride * height;
        let pool_size = frame_len * 2;
        if pool_size > i32::MAX as usize {
            return Err(Error::msg(
                "requested window size is too large for an shm pool",
            ));
        }
        let pool = Pool::new(pool_size)?;
        self.frame_len = frame_len;
        let pool_id = self.alloc_id();
        self.conn.request_with_fd(
            shm,
            wl_shm::CREATE_POOL,
            &[Arg::NewId(pool_id), Arg::Int(pool_size as i32)],
            pool.fd(),
        )?;
        for i in 0..2 {
            let buffer = self.alloc_id();
            self.conn.request(
                pool_id,
                wl_shm_pool::CREATE_BUFFER,
                &[
                    Arg::NewId(buffer),
                    Arg::Int((i * frame_len) as i32),
                    Arg::Int(self.width as i32),
                    Arg::Int(self.height as i32),
                    Arg::Int(stride as i32),
                    Arg::Uint(wl_shm::FORMAT_XRGB8888),
                ],
            );
            self.buffers[i] = buffer;
        }
        self.pool = Some(pool);
        self.pool_id = pool_id;
        self.busy = [false, false];
        // The new buffers hold undefined memory and the surface size may have
        // changed, so no stored frame describes them: clear the damage history to
        // force the next frame to repaint each buffer (and the screen) in full.
        self.buffer_lists = [Vec::new(), Vec::new()];
        self.present_list = Vec::new();
        Ok(())
    }

    /// Destroy the current buffers and pool, freeing the mapping. The server
    /// recycles the ids via `delete_id` into our free list.
    fn destroy_buffers(&mut self) {
        let buffers = self.buffers;
        for b in buffers {
            if b != 0 {
                self.conn.request(b, wl_buffer::DESTROY, &[]);
            }
        }
        if self.pool_id != 0 {
            self.conn.request(self.pool_id, wl_shm_pool::DESTROY, &[]);
        }
        self.pool = None; // Drop unmaps and closes the fd.
        self.pool_id = 0;
        self.buffers = [0, 0];
        self.busy = [false, false];
    }

    /// Resize to `(w, h)`: tear down the buffers and rebuild them at the new
    /// size. A no-op if the size is unchanged.
    fn apply_resize(&mut self, w: u32, h: u32) -> Result<()> {
        if (w, h) == (self.width, self.height) {
            return Ok(());
        }
        self.destroy_buffers();
        self.width = w;
        self.height = h;
        self.create_buffers()?;
        self.dirty = true;
        Ok(())
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

    /// Build the current frame, repaint only what changed in a free buffer, and
    /// present it. Returns false if both buffers are still held by the compositor,
    /// leaving `dirty` set so the next buffer release retries.
    ///
    /// Two diffs over the display list drive the damage tracking:
    ///
    /// - against `present_list` (what is on screen) to decide whether to present at
    ///   all and which regions to report as damaged to the compositor;
    /// - against this buffer's last frame to decide which regions to repaint into
    ///   it, since the two back buffers age independently.
    ///
    /// An idle caret blink thus repaints a caret-sized region (often nothing, when
    /// the chosen buffer already holds that state) instead of the whole surface.
    fn render_frame(&mut self) -> Result<bool> {
        let Some(idx) = (0..2).find(|&i| !self.busy[i]) else {
            return Ok(false);
        };
        // Time the whole frame (layout reuse/rebuild plus paint) when stats are on.
        let started = self.stats.then(Instant::now);
        // Build (or reuse) the layout up front, then build the frame *by reference*
        // off the cached layout, so a frame never clones the document's visual lines.
        self.ensure_layout();
        // Keep the scroll offset valid for the current content and window size.
        self.scroll_y = scroll::clamp(self.scroll_y, self.content_height, self.height as i32);
        let width = self.width as usize;
        let height = self.height as usize;
        let (sw, sh) = (self.width as i32, self.height as i32);
        let params = PaintParams {
            width,
            height,
            scroll_y: self.scroll_y,
            selection: self.text.selection(),
            cursor: self.text.cursor(),
            caret_visible: self.caret_visible,
        };

        let new_list = {
            let Some(layout) = self.layout_cache.as_ref() else {
                return Ok(false);
            };
            build_display_list(&self.fonts, layout, self.text.rope(), &params)
        };

        // Nothing changed since the on-screen frame: it already shows this content,
        // so there is nothing to present (the common idle case).
        let screen_dmg = display::damage(&self.present_list, &new_list, sw, sh);
        if screen_dmg.is_empty() {
            return Ok(true);
        }

        // Repaint only the regions of this buffer that differ from what it already
        // holds. The background fill at the head of the list clears each clip, then
        // the rest of the list repaints it, so a clipped region matches a full
        // repaint exactly.
        let buf_dmg = display::damage(&self.buffer_lists[idx], &new_list, sw, sh);
        if !buf_dmg.is_empty() {
            let frame_len = self.frame_len;
            let offset = idx * frame_len;
            let Some(pool) = self.pool.as_mut() else {
                return Ok(false);
            };
            let frame = &mut pool.data_mut()[offset..offset + frame_len];
            let mut canvas = Canvas::new(frame, width, height);
            for clip in &buf_dmg {
                display::execute(
                    &mut canvas,
                    &self.fonts,
                    &self.images,
                    &new_list,
                    Some(*clip),
                );
            }
        }
        self.buffer_lists[idx] = new_list.clone();
        self.present_list = new_list;

        let buffer = self.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Report just the regions that changed on screen, so the compositor
        // recomposites only those rather than the whole surface.
        for r in &screen_dmg {
            self.conn.request(
                self.surface,
                wl_surface::DAMAGE,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
        // Arm a frame callback before committing, so the compositor tells us when
        // the next frame is due. `run_until` won't draw again until its `done`
        // clears `frame_callback`, which is what paces redraws to the refresh.
        let callback = self.alloc_id();
        self.conn
            .request(self.surface, wl_surface::FRAME, &[Arg::NewId(callback)]);
        self.frame_callback = callback;
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.busy[idx] = true;

        if let Some(started) = started {
            let cache = self.fonts.cache_stats();
            let gap = self
                .last_present
                .map(|t| started.duration_since(t).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            eprintln!(
                "[stats] frame {} {:.2}ms paint, {:.1}ms since last | glyphs hit {} miss {} ({:.1}% of {})",
                self.frame_count,
                started.elapsed().as_secs_f64() * 1000.0,
                gap,
                cache.hits,
                cache.misses,
                cache.hit_rate() * 100.0,
                cache.lookups(),
            );
            self.last_present = Some(started);
        }
        self.frame_count += 1;
        Ok(true)
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
            let serial = r.u32()?;
            self.conn.request(
                self.xdg_surface,
                xdg_surface::ACK_CONFIGURE,
                &[Arg::Uint(serial)],
            );
            if let Some((w, h)) = self.pending_size.take() {
                self.apply_resize(w, h)?;
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
                    // shm-pool size arithmetic (see `create_buffers`).
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
            protocol::IFACE_SHM => {
                let id = self.alloc_id();
                self.bind(name, interface, version.min(protocol::VERSION_SHM), id);
                self.shm = Some(id);
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
            _ => {}
        }
        Ok(())
    }

    fn on_keyboard(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_keyboard::EV_KEYMAP => {
                let format = r.u32()?;
                let size = r.u32()? as usize;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("wl_keyboard.keymap arrived without its fd"))?;
                let ptr = ffi::mmap_ro(fd.as_raw_fd(), size)?;
                // SAFETY: ptr maps `size` readable bytes from the keymap fd; the
                // bytes are copied out before the mapping is released.
                let bytes = unsafe { core::slice::from_raw_parts(ptr, size) }.to_vec();
                // SAFETY: ptr/size came from mmap_ro and are not used again.
                unsafe { ffi::unmap(ptr, size) };
                self.xkb.load_keymap(&bytes, format)?;
            }
            wl_keyboard::EV_KEY => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let keycode = r.u32()?;
                let key_state = r.u32()?;
                if key_state == wl_keyboard::KEY_STATE_PRESSED {
                    self.apply_action(self.xkb.action(keycode))?;
                    self.start_repeat(keycode);
                } else if self.repeat_key == Some(keycode) {
                    self.stop_repeat();
                }
            }
            wl_keyboard::EV_MODIFIERS => {
                self.last_serial = r.u32()?;
                let depressed = r.u32()?;
                let latched = r.u32()?;
                let locked = r.u32()?;
                let group = r.u32()?;
                self.xkb.update_modifiers(depressed, latched, locked, group);
            }
            wl_keyboard::EV_REPEAT_INFO => {
                let rate = r.u32()?;
                let delay = r.u32()?;
                self.repeat_interval = if rate == 0 {
                    None
                } else {
                    Some(Duration::from_secs(1) / rate)
                };
                self.repeat_delay = Duration::from_millis(u64::from(delay));
            }
            wl_keyboard::EV_ENTER => {
                self.last_serial = r.u32()?;
                // Regained keyboard focus: show the caret and resume its blink.
                self.focused = true;
                self.bump_caret();
                self.dirty = true;
            }
            wl_keyboard::EV_LEAVE => {
                self.stop_repeat();
                // Lost keyboard focus: hide the caret, stop blinking, and let the
                // loop go idle (no 500ms blink wake-ups while unfocused).
                self.focused = false;
                self.caret_visible = false;
                self.caret_blink_at = None;
                self.dirty = true;
            }
            _ => {}
        }
        Ok(())
    }

    fn apply_action(&mut self, action: KeyAction) -> Result<()> {
        // Any keyboard action but Copy drops an image selection and its pinned raw
        // source (using the keyboard means the user has left the image widget). A
        // revealed source the caret sits in stays shown via the cursor-on-line
        // reveal. Copy must see the selection to choose image-vs-text, so it is
        // exempt. Repaint so the border and button disappear immediately.
        if !matches!(action, KeyAction::Copy)
            && (self.selected_image.is_some() || self.raw_image.is_some())
        {
            self.selected_image = None;
            self.raw_image = None;
            self.dirty = true;
        }
        let extend = self.xkb.shift_active();
        let changed = match action {
            KeyAction::Char(c) => {
                let mut encoded = [0u8; 4];
                self.text.insert_str(c.encode_utf8(&mut encoded));
                self.goal_x = None;
                true
            }
            KeyAction::Enter => {
                self.goal_x = None;
                self.text.enter()
            }
            // On a list item, Tab nests it one level deeper and Shift+Tab one
            // shallower. Anywhere else, Tab inserts a soft tab and Shift+Tab is a
            // no-op (there is nothing to dedent).
            KeyAction::Tab => {
                self.goal_x = None;
                if self.text.cursor_on_list_item() {
                    self.text.nest_list_item(extend)
                } else if extend {
                    false
                } else {
                    self.text.insert_str(SOFT_TAB);
                    true
                }
            }
            KeyAction::Backspace => {
                self.goal_x = None;
                self.text.backspace()
            }
            KeyAction::Delete => {
                self.goal_x = None;
                self.text.delete_forward()
            }
            KeyAction::DeleteWordBack => {
                self.goal_x = None;
                self.text.delete_word_back()
            }
            KeyAction::DeleteWordForward => {
                self.goal_x = None;
                self.text.delete_word_forward()
            }
            KeyAction::Left => {
                self.goal_x = None;
                self.text.move_left(extend)
            }
            KeyAction::Right => {
                self.goal_x = None;
                self.text.move_right(extend)
            }
            KeyAction::WordLeft => {
                self.goal_x = None;
                self.text.move_word_left(extend)
            }
            KeyAction::WordRight => {
                self.goal_x = None;
                self.text.move_word_right(extend)
            }
            KeyAction::Up => self.move_vertical(-1, extend),
            KeyAction::Down => self.move_vertical(1, extend),
            KeyAction::PageUp => self.move_page(-1, extend),
            KeyAction::PageDown => self.move_page(1, extend),
            KeyAction::Home => {
                self.goal_x = None;
                self.move_line_edge(false, extend)
            }
            KeyAction::End => {
                self.goal_x = None;
                self.move_line_edge(true, extend)
            }
            KeyAction::DocStart => {
                self.goal_x = None;
                self.text.move_doc_start(extend)
            }
            KeyAction::DocEnd => {
                self.goal_x = None;
                self.text.move_doc_end(extend)
            }
            KeyAction::SelectAll => {
                self.text.select_all();
                true
            }
            KeyAction::Undo => {
                self.goal_x = None;
                self.text.undo()
            }
            KeyAction::Redo => {
                self.goal_x = None;
                self.text.redo()
            }
            // The clipboard actions manage their own redraw and may do I/O over a
            // pipe to another app. As with Save, an I/O failure (e.g. a peer that
            // closes the pipe early, EPIPE) must not bring the editor down and lose
            // the user's unsaved work, so it is reported and swallowed rather than
            // propagated out of the loop. With an image selected, Copy puts the
            // picture on the clipboard instead of text.
            KeyAction::Copy => {
                let result = if self.selected_image.is_some() {
                    self.copy_selected_image()
                } else {
                    self.copy_selection()
                };
                if let Err(e) = result {
                    eprintln!("bnknotes: copy failed: {e}");
                }
                false
            }
            KeyAction::Cut => {
                if let Err(e) = self.cut_selection() {
                    eprintln!("bnknotes: cut failed: {e}");
                }
                false
            }
            KeyAction::Paste => {
                if let Err(e) = self.paste() {
                    eprintln!("bnknotes: paste failed: {e}");
                }
                false
            }
            // Saving touches the disk but not the buffer; a failure must not
            // bring the editor down and lose the user's unsaved work, so it is
            // reported and swallowed rather than propagated out of the loop.
            KeyAction::Save => {
                if let Err(e) = self.save() {
                    eprintln!("bnknotes: save failed: {e}");
                }
                false
            }
            KeyAction::None => false,
        };
        if changed {
            self.after_edit();
        }
        Ok(())
    }

    /// Common bookkeeping after a buffer edit: request a redraw, keep the caret
    /// on screen, and refresh the title's modified indicator.
    fn after_edit(&mut self) {
        self.dirty = true;
        self.bump_caret();
        self.scroll_caret_into_view();
        self.refresh_title();
    }

    /// Write the buffer to its backing file, atomically and in the file's
    /// original on-disk format. A buffer with no file (started without a path)
    /// has nowhere to go yet, so it reports that and stays put.
    fn save(&mut self) -> Result<()> {
        let Some(path) = self.path.clone() else {
            eprintln!("bnknotes: no file to save to; start bnknotes with a path");
            return Ok(());
        };
        let bytes = file::encode(&self.text.text(), &self.format);
        file::save_atomic(&path, &bytes)?;
        self.text.mark_saved();
        self.refresh_title();
        eprintln!("bnknotes: saved {}", path.display());
        Ok(())
    }

    /// The toplevel title: the file's name (or just the app name for an unnamed
    /// buffer), with a leading dot when there are unsaved changes.
    fn window_title(&self) -> String {
        let modified = if self.text.is_modified() { "• " } else { "" };
        match self.path.as_ref().and_then(|p| p.file_name()) {
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
        let modified = self.text.is_modified();
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

    /// Move the cursor `dir` lines (-1 up, +1 down), keeping a sticky goal x.
    fn move_vertical(&mut self, dir: i32, extend: bool) -> bool {
        self.ensure_layout();
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        // Read the cached layout by reference; `goal_x`/`text` are disjoint fields,
        // so they can be touched while the layout borrow is live.
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (caret_x, _, _) = layout.caret(&self.fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        let target = layout.offset_vertical(&self.fonts, self.text.rope(), cursor, dir, goal);
        self.goal_x = Some(goal);
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// Move the cursor to the start (`to_end = false`) or end of its line.
    fn move_line_edge(&mut self, to_end: bool, extend: bool) -> bool {
        self.ensure_layout();
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (start, end) = layout.line_bounds(cursor);
        let target = if to_end { end } else { start };
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// Move the cursor one viewport height up (`dir = -1`) or down (`dir = 1`),
    /// keeping the sticky goal column, like a run of Up/Down. The view follows
    /// via the caret-reveal scroll in `after_edit`. Returns whether it moved.
    fn move_page(&mut self, dir: i32, extend: bool) -> bool {
        self.ensure_layout();
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        let page_lines = self.page_lines();
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (caret_x, _, _) = layout.caret(&self.fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        // Step a line at a time so each hop honors the goal column and the
        // varying per-line heights; stop early at the first or last line.
        let mut target = cursor;
        for _ in 0..page_lines {
            let next = layout.offset_vertical(&self.fonts, self.text.rope(), target, dir, goal);
            if next == target {
                break;
            }
            target = next;
        }
        self.goal_x = Some(goal);
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// How many visual lines make up a Page Up/Down jump: the window height in
    /// body line-heights, less one row of overlap for continuity, at least one.
    fn page_lines(&self) -> usize {
        let line_height = self.fonts.metrics(FONT_SIZE).line_height.max(1);
        (self.height as i32 / line_height - 1).max(1) as usize
    }

    fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                let serial = r.u32()?;
                let _surface = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                self.pointer_serial = serial;
                // The compositor resets the cursor on enter, so force a re-set by
                // invalidating the remembered shape, then pick one for the position.
                self.cursor_shape = 0;
                self.update_cursor_shape();
            }
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                if self.selecting {
                    // Drag extends the selection from the press point, and arms
                    // auto-scroll when the pointer sits at a vertical edge.
                    self.place_cursor_at(self.pointer_x, self.pointer_y, true);
                    self.update_autoscroll();
                } else {
                    // Match the pointer shape to whatever is now under it.
                    self.update_cursor_shape();
                }
            }
            wl_pointer::EV_BUTTON => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let button = r.u32()?;
                let state = r.u32()?;
                if button == wl_pointer::BTN_LEFT {
                    if state == wl_pointer::BUTTON_STATE_PRESSED {
                        self.goal_x = None;
                        self.on_left_press();
                    } else {
                        self.selecting = false;
                        self.stop_autoscroll();
                        // Drag over: let the reveal catch up to the final selection.
                        if self.frozen_reveal.take().is_some() {
                            self.dirty = true;
                        }
                    }
                }
            }
            wl_pointer::EV_AXIS => {
                let _time = r.u32()?;
                let axis = r.u32()?;
                let value = r.fixed()?;
                if axis == wl_pointer::AXIS_VERTICAL_SCROLL {
                    self.scroll_by((value * SCROLL_FACTOR).round() as i32);
                }
            }
            _ => {} // leave, frame, axis_source, and friends
        }
        Ok(())
    }

    /// Place the cursor (or, with `extend`, the selection's free end) at a click
    /// or drag position in surface-local pixels. The window y is shifted into
    /// content space by the scroll offset before hit-testing.
    fn place_cursor_at(&mut self, x: f32, y: f32, extend: bool) {
        // Hit-test against the cached layout by reference; the hot drag path must
        // not clone or rebuild it. The reveal pin keeps the cache valid throughout.
        self.ensure_layout();
        let Some(layout) = self.layout_cache.as_ref() else {
            return;
        };
        let offset = layout.offset_at(&self.fonts, self.text.rope(), x, y + self.scroll_y as f32);
        let before = self.text.cursor();
        let before_sel = self.text.selection();
        self.text.set_cursor(offset, extend);
        self.bump_caret();
        if self.text.cursor() != before || self.text.selection() != before_sel {
            self.dirty = true;
        }
    }

    /// Create the cursor-shape device for our pointer, if the compositor offers
    /// the manager. Without it the pointer just keeps the compositor's default.
    fn init_cursor_shape(&mut self) {
        let (Some(manager), true) = (self.cursor_shape_manager, self.pointer != 0) else {
            return;
        };
        let device = self.alloc_id();
        self.conn.request(
            manager,
            wp_cursor_shape_manager_v1::GET_POINTER,
            &[Arg::NewId(device), Arg::Object(self.pointer)],
        );
        self.cursor_shape_device = device;
    }

    /// Ask the compositor to draw `shape` for the pointer, using the latest `enter`
    /// serial. A no-op when the compositor gave us no cursor-shape device.
    fn set_cursor_shape(&mut self, shape: u32) {
        if self.cursor_shape_device == 0 {
            return;
        }
        self.conn.request(
            self.cursor_shape_device,
            wp_cursor_shape_device_v1::SET_SHAPE,
            &[Arg::Uint(self.pointer_serial), Arg::Uint(shape)],
        );
    }

    /// Pick and apply the pointer shape for its current position: a hand over a
    /// selected image's button, the plain arrow over a picture, the I-beam over
    /// text. Only re-requests when the shape changes, so it is cheap to call on
    /// every motion event. Uses the cached layout (no clone).
    fn update_cursor_shape(&mut self) {
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_POINTER, SHAPE_TEXT};
        // A rendered link reads as clickable: the hand pointer, the same affordance
        // the image button uses. Checked first so it wins over the I-beam.
        let over_link = self.link_under_pointer().is_some();
        self.ensure_layout();
        let (x, y) = (self.pointer_x, self.pointer_y + self.scroll_y as f32);
        let shape = match self.layout_cache.as_ref() {
            _ if over_link => SHAPE_POINTER,
            Some(layout) if layout.image_button_at(&self.fonts, x, y).is_some() => SHAPE_POINTER,
            Some(layout) if layout.image_at(&self.fonts, x, y).is_some() => SHAPE_DEFAULT,
            _ => SHAPE_TEXT,
        };
        if shape != self.cursor_shape {
            self.cursor_shape = shape;
            self.set_cursor_shape(shape);
        }
    }

    /// Handle a left press that landed on an image's picture or its toggle button,
    /// returning whether it was consumed (so the normal text-click path is skipped).
    /// The button toggles the raw source; the picture selects the image (arming
    /// `Ctrl+C` to copy it) without moving the text caret, so the source does not
    /// auto-reveal.
    fn on_image_press(&mut self) -> bool {
        let (x, y) = (self.pointer_x, self.pointer_y + self.scroll_y as f32);
        self.ensure_layout();
        // Query both hits up front (each a copyable `Option<usize>`) so the layout
        // borrow ends before the mutations below, which touch other `State` fields.
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let button = layout.image_button_at(&self.fonts, x, y);
        let image = layout.image_at(&self.fonts, x, y);
        if let Some(ls) = button {
            self.raw_image = if self.raw_image == Some(ls) {
                None
            } else {
                Some(ls)
            };
            self.dirty = true;
            return true;
        }
        if let Some(ls) = image {
            self.selected_image = Some(ls);
            self.raw_image = None;
            // Collapse any text selection so a copy targets the image, and leave
            // the caret where it is (off the image) so its source stays hidden.
            let cursor = self.text.cursor();
            self.text.set_cursor(cursor, false);
            self.selecting = false;
            self.frozen_reveal = None;
            self.dirty = true;
            return true;
        }
        false
    }

    /// Handle a left-button press: chain it into the click count and act on it.
    /// A single click places the cursor (extending the selection if Shift is
    /// held) and starts a drag; a double-click selects the word under the
    /// pointer; a triple (or more) selects the whole logical line.
    fn on_left_press(&mut self) {
        // Image interactions take precedence over text: a click on a selected
        // image's button toggles its raw source, a click on a picture selects it.
        if self.on_image_press() {
            return;
        }
        // Any other click deselects the image and unpins its raw source; if the
        // click lands in a revealed source line, the cursor-on-line reveal keeps
        // that source shown, otherwise it hides (matching "click away to dismiss").
        self.selected_image = None;
        self.raw_image = None;

        let now = Instant::now();
        let (lx, ly) = self.last_click_pos;
        let near = (self.pointer_x - lx).abs() <= MULTI_CLICK_RADIUS
            && (self.pointer_y - ly).abs() <= MULTI_CLICK_RADIUS;
        let chained = near
            && self
                .last_click
                .is_some_and(|t| now.saturating_duration_since(t) <= MULTI_CLICK_INTERVAL);
        self.click_count = if chained { self.click_count + 1 } else { 1 };
        self.last_click = Some(now);
        self.last_click_pos = (self.pointer_x, self.pointer_y);

        // A plain click on a rendered horizontal rule selects the whole rule line
        // (cursor at its end) rather than dropping a caret between its `---`
        // markers. Shift-click falls through so it can still extend a selection
        // across one.
        if !self.xkb.shift_active() {
            let cy = self.pointer_y + self.scroll_y as f32;
            self.ensure_layout();
            let rule = self
                .layout_cache
                .as_ref()
                .and_then(|l| l.rule_at(&self.fonts, cy));
            if let Some((start, end)) = rule {
                self.text.select_range(start, end);
                self.selecting = false;
                self.dirty = true;
                self.bump_caret();
                return;
            }
        }

        // A plain click on a link while it is rendered (its `[..](..)` markup
        // collapsed because the cursor is on another line) follows the link instead
        // of dropping a caret; once the cursor is on the line the markup shows and
        // clicks edit normally. Gated to an un-chained, unmodified press so a
        // double-click cannot open it twice.
        if !chained && !self.xkb.shift_active() {
            if let Some(dest) = self.link_under_pointer() {
                // If the browser genuinely cannot launch, fall through so the click
                // still places a caret rather than doing nothing.
                if crate::browser::open_url(&dest).is_ok() {
                    self.selecting = false;
                    return;
                }
            }
        }

        match self.click_count {
            1 => {
                let extend = self.xkb.shift_active();
                self.place_cursor_at(self.pointer_x, self.pointer_y, extend);
                self.selecting = true;
                // Pin marker reveal to the press point for the whole drag.
                let c = self.text.cursor();
                self.frozen_reveal = Some(self.text.selection().unwrap_or((c, c)));
            }
            2 => {
                let offset = self.pointer_offset();
                self.text.select_word_at(offset);
                self.dirty = true;
            }
            _ => {
                let offset = self.pointer_offset();
                self.text.select_line_at(offset);
                self.dirty = true;
            }
        }
        self.bump_caret();
    }

    /// The destination of the openable link under the pointer, if any. A link
    /// qualifies only when its line is *rendered* (the reveal range does not touch
    /// it, so its `[..](..)` markup is collapsed), the pointer is over its text, and
    /// the destination has a scheme the browser will open. On the active line
    /// (markup shown for editing) this is `None`, so clicks there edit normally.
    /// Drives both following a click and the clickable pointer shape on hover.
    fn link_under_pointer(&mut self) -> Option<String> {
        let off = self.pointer_offset();
        let (a0, a1) = self.reveal_range();
        let rope = self.text.rope();
        let line = rope.byte_to_line(off);
        let ls = rope.line_to_byte(line);
        // The logical line spans `ls` up to (but not including) its trailing '\n'.
        let le = if line + 1 < rope.len_lines() {
            rope.line_to_byte(line + 1).saturating_sub(1)
        } else {
            rope.len()
        };
        // A revealed line shows its markup, so a click there should edit, not follow.
        if a0 <= le && a1 >= ls {
            return None;
        }
        let content = rope.byte_slice(ls..le);
        let rel = off - ls;
        let link = markdown::links(&content)
            .into_iter()
            .find(|l| l.text.0 <= rel && rel <= l.text.1)?;
        let dest = content[link.dest.0..link.dest.1].to_string();
        crate::browser::can_open(&dest).then_some(dest)
    }

    /// The byte offset under the pointer, mapping its window position into
    /// content space by the current scroll.
    fn pointer_offset(&mut self) -> usize {
        self.ensure_layout();
        let (x, y) = (self.pointer_x, self.pointer_y + self.scroll_y as f32);
        self.layout_cache.as_ref().map_or(0, |layout| {
            layout.offset_at(&self.fonts, self.text.rope(), x, y)
        })
    }

    /// Arm or disarm drag auto-scroll from the pointer's distance to a vertical
    /// edge: past the top scrolls up, past the bottom scrolls down, anywhere in
    /// between turns it off.
    fn update_autoscroll(&mut self) {
        let y = self.pointer_y as i32;
        let step = if y < AUTOSCROLL_MARGIN {
            -AUTOSCROLL_STEP
        } else if y > self.height as i32 - AUTOSCROLL_MARGIN {
            AUTOSCROLL_STEP
        } else {
            0
        };
        if step == 0 {
            self.stop_autoscroll();
        } else {
            self.autoscroll = step;
            // Fire the first tick promptly; later ticks follow the interval.
            self.autoscroll_at.get_or_insert_with(Instant::now);
        }
    }

    fn stop_autoscroll(&mut self) {
        self.autoscroll = 0;
        self.autoscroll_at = None;
    }

    /// While drag-selecting with the pointer held in an edge zone, scroll a step
    /// and re-extend the selection to the (stationary) pointer, on the auto-
    /// scroll interval. A no-op when not auto-scrolling.
    fn fire_autoscroll(&mut self) {
        if self.autoscroll == 0 || !self.selecting {
            self.stop_autoscroll();
            return;
        }
        let Some(at) = self.autoscroll_at else {
            return;
        };
        let now = Instant::now();
        if at > now {
            return;
        }
        self.scroll_by(self.autoscroll);
        self.place_cursor_at(self.pointer_x, self.pointer_y, true);
        self.autoscroll_at = Some(now + AUTOSCROLL_INTERVAL);
    }

    // --- Caret blink ------------------------------------------------------

    /// Toggle the caret if its blink interval has elapsed, requesting a redraw.
    /// The first call just arms the timer.
    fn fire_caret_blink(&mut self) {
        // An unfocused window shows no caret and arms no blink timer, so it never
        // wakes the loop just to toggle a caret nobody is looking at.
        if !self.focused {
            return;
        }
        let now = Instant::now();
        match self.caret_blink_at {
            None => self.caret_blink_at = Some(now + CARET_BLINK),
            Some(at) if at <= now => {
                self.caret_visible = !self.caret_visible;
                self.caret_blink_at = Some(now + CARET_BLINK);
                self.dirty = true;
            }
            Some(_) => {}
        }
    }

    /// Show the caret and restart its blink cycle, called on any activity that
    /// moves the caret, so it never blinks out mid-edit.
    fn bump_caret(&mut self) {
        self.caret_visible = true;
        self.caret_blink_at = Some(Instant::now() + CARET_BLINK);
    }

    // --- Key repeat -------------------------------------------------------

    /// Begin repeating `keycode` after the delay, if repeat is enabled and the
    /// keymap marks the key as repeatable.
    fn start_repeat(&mut self, keycode: u32) {
        if self.repeat_interval.is_some() && self.xkb.key_repeats(keycode) {
            self.repeat_key = Some(keycode);
            self.repeat_at = Some(Instant::now() + self.repeat_delay);
        }
    }

    fn stop_repeat(&mut self) {
        self.repeat_key = None;
        self.repeat_at = None;
    }

    /// Fire every repeat whose deadline has passed, re-running the held key's
    /// action (so modifiers held with it still apply).
    fn fire_repeats(&mut self) -> Result<()> {
        let (Some(keycode), Some(mut at), Some(interval)) =
            (self.repeat_key, self.repeat_at, self.repeat_interval)
        else {
            return Ok(());
        };
        let now = Instant::now();
        let mut fired = 0;
        while at <= now && fired < REPEAT_CATCHUP_CAP {
            self.apply_action(self.xkb.action(keycode))?;
            at += interval;
            fired += 1;
        }
        self.repeat_at = Some(at);
        Ok(())
    }

    /// How long to wait for input before the next repeat is due, or `None` to
    /// block indefinitely when nothing is repeating.
    fn repeat_timeout(&self) -> Option<Duration> {
        let at = self.repeat_at?;
        self.repeat_interval?;
        Some(
            at.saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1)),
        )
    }

    /// How long to block for input: the soonest of the next key-repeat, the next
    /// auto-scroll tick, and the next caret blink, or `None` when none is due.
    fn next_wake(&self) -> Option<Duration> {
        let now = Instant::now();
        let until = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        let autoscroll = (self.autoscroll != 0 && self.selecting)
            .then_some(self.autoscroll_at)
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

    // --- Clipboard --------------------------------------------------------

    fn on_data_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_device::EV_DATA_OFFER => {
                // The compositor introduces a new offer; its MIME types follow.
                self.incoming_offer = r.u32()?;
                self.incoming_text_mime = None;
            }
            wl_data_device::EV_SELECTION => {
                let offer = r.u32()?;
                if self.selection_offer != 0 && self.selection_offer != offer {
                    self.conn
                        .request(self.selection_offer, wl_data_offer::DESTROY, &[]);
                }
                self.selection_offer = offer;
                self.selection_text_mime = if offer != 0 && offer == self.incoming_offer {
                    self.incoming_text_mime.clone()
                } else {
                    None
                };
            }
            _ => {}
        }
        Ok(())
    }

    fn on_data_source(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_source::EV_SEND => {
                // A target is pasting our data: write it to the supplied fd. Every
                // MIME we advertise maps to the same bytes (text encodings, or the
                // one image type), so the requested MIME does not change what we send.
                let _mime = r.string()?;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("data_source.send arrived without its fd"))?;
                // A target that closes its read end early makes this write fail
                // (EPIPE). Serving the clipboard must not bring the editor down, so
                // report and swallow the I/O error rather than propagate it out of
                // the event loop (the protocol-parse errors above stay fatal).
                if let Err(e) = ffi::write_all(fd.as_raw_fd(), &self.clipboard_data) {
                    eprintln!("bnknotes: clipboard serve failed: {e}");
                }
            }
            wl_data_source::EV_CANCELLED => {
                // Another selection replaced ours; tear the source down.
                if self.clipboard_source != 0 {
                    self.conn
                        .request(self.clipboard_source, wl_data_source::DESTROY, &[]);
                }
                self.clipboard_source = 0;
                self.clipboard_data.clear();
                self.clipboard_mimes.clear();
            }
            _ => {}
        }
        Ok(())
    }

    /// Put the current selection on the clipboard. No-op without a selection.
    fn copy_selection(&mut self) -> Result<()> {
        if let Some(text) = self.text.selected_text() {
            self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        }
        Ok(())
    }

    /// Copy the selection, then delete it.
    fn cut_selection(&mut self) -> Result<()> {
        let Some(text) = self.text.selected_text() else {
            return Ok(());
        };
        self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        if self.text.backspace() {
            self.after_edit();
        }
        Ok(())
    }

    /// Put the selected image's file bytes on the clipboard, advertised as its real
    /// type (`image/png`/`image/jpeg`), so it can be pasted into other apps. No-op
    /// if nothing is selected or the file can no longer be read.
    fn copy_selected_image(&mut self) -> Result<()> {
        let Some(ls) = self.selected_image else {
            return Ok(());
        };
        let Some(raw) = self.image_path_at(ls) else {
            return Ok(());
        };
        if let Some((bytes, mime)) = self.images.read_source(&raw) {
            self.set_clipboard(bytes, vec![mime]);
        }
        Ok(())
    }

    /// The raw destination path of the image whose logical line starts at `ls`, or
    /// `None` if that line is no longer a standalone image.
    fn image_path_at(&self, ls: usize) -> Option<String> {
        let rope = self.text.rope();
        let line_idx = rope.byte_to_line(ls);
        let n = rope.len_lines();
        let le = if line_idx + 1 < n {
            rope.line_to_byte(line_idx + 1) - 1
        } else {
            rope.line_to_byte(line_idx + 1)
        };
        let line = rope.byte_slice(ls..le);
        let link = markdown::standalone_image(&line)?;
        Some(line[link.path.0..link.path.1].to_string())
    }

    /// Become the clipboard owner serving `data` under `mimes`, replacing any
    /// source we held.
    fn set_clipboard(&mut self, data: Vec<u8>, mimes: Vec<&'static str>) {
        let (Some(manager), true) = (self.data_device_manager, self.data_device != 0) else {
            return;
        };
        if self.clipboard_source != 0 {
            self.conn
                .request(self.clipboard_source, wl_data_source::DESTROY, &[]);
        }
        let source = self.alloc_id();
        self.conn.request(
            manager,
            wl_data_device_manager::CREATE_DATA_SOURCE,
            &[Arg::NewId(source)],
        );
        for mime in &mimes {
            self.conn
                .request(source, wl_data_source::OFFER, &[Arg::Str(mime)]);
        }
        self.conn.request(
            self.data_device,
            wl_data_device::SET_SELECTION,
            &[Arg::Object(source), Arg::Uint(self.last_serial)],
        );
        self.clipboard_source = source;
        self.clipboard_data = data;
        self.clipboard_mimes = mimes;
    }

    /// Insert the clipboard's text at the cursor.
    fn paste(&mut self) -> Result<()> {
        // If we own the clipboard, insert our own data directly. Going through the
        // compositor would deadlock: we'd block reading the pipe while also needing
        // to service our own source's send event. Only text pastes into the buffer;
        // a copied image has no text representation, so pasting it here is a no-op.
        if self.clipboard_source != 0 {
            if self.clipboard_mimes.iter().any(|m| is_text_mime(m)) {
                let text = String::from_utf8_lossy(&self.clipboard_data).into_owned();
                self.insert_clipboard(&text);
            }
            return Ok(());
        }
        let (Some(mime), true) = (self.selection_text_mime.clone(), self.selection_offer != 0)
        else {
            return Ok(());
        };
        // Receive the data through a pipe: hand the write end to the compositor,
        // close our copy, and read the read end to EOF.
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            self.selection_offer,
            wl_data_offer::RECEIVE,
            &[Arg::Str(&mime)],
            write_end.as_raw_fd(),
        )?;
        drop(write_end);
        let bytes = ffi::read_to_end(read_end.as_raw_fd())?;
        if let Ok(text) = String::from_utf8(bytes) {
            self.insert_clipboard(&text);
        }
        Ok(())
    }

    fn insert_clipboard(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.text.insert_str(text);
        self.dirty = true;
        self.scroll_caret_into_view();
    }

    /// Scroll the viewport by `delta` pixels (positive = down), clamped to the
    /// document. The hottest input path (one call per wheel event), so it reads
    /// the cached content height instead of cloning the layout.
    fn scroll_by(&mut self, delta: i32) {
        self.ensure_layout();
        let sy = scroll::clamp(
            self.scroll_y + delta,
            self.content_height,
            self.height as i32,
        );
        if sy != self.scroll_y {
            self.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// Scroll the minimum needed to keep the caret visible after a cursor move.
    fn scroll_caret_into_view(&mut self) {
        self.ensure_layout();
        let cursor = self.text.cursor();
        let Some((_, caret_top, caret_h)) = self
            .layout_cache
            .as_ref()
            .map(|l| l.caret(&self.fonts, self.text.rope(), cursor))
        else {
            return;
        };
        let sy = scroll::reveal(
            self.scroll_y,
            caret_top,
            caret_h,
            self.height as i32,
            self.content_height,
            MARGIN,
        );
        if sy != self.scroll_y {
            self.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// The active range used for live-preview marker reveal: the live selection
    /// (or the bare cursor when there is none), except while a drag is extending a
    /// selection, when it is pinned to [`Self::frozen_reveal`] so the markdown
    /// under the pointer does not flip between rendered and raw as the drag sweeps
    /// across it. The highlight itself always tracks the live selection; only the
    /// reveal is pinned.
    fn reveal_range(&self) -> (usize, usize) {
        let cursor = self.text.cursor();
        match (self.selecting, self.frozen_reveal) {
            (true, Some(range)) => range,
            _ => self.text.selection().unwrap_or((cursor, cursor)),
        }
    }

    /// Lay out the current text, wrapping to the window width less a margin on
    /// each side, and cache the result with its content height. Per-line metrics
    /// come from the font set during the build. The reveal range (see
    /// [`Self::reveal_range`]) is the active range, so a heading the caret is
    /// editing keeps its `#` markers while the rest collapse them.
    ///
    /// Memoized on `(version, width, font size, active line span, image UI)`: the
    /// geometry depends on nothing else, so a scroll or idle redraw is a no-op here
    /// and reuses the cache. The marker reveal depends only on *which logical
    /// lines* the active range touches, so the key uses the range's line span, not
    /// its raw byte offsets: a cursor move within one line then reuses the cache
    /// instead of rebuilding. Only an edit, a move onto another line, or a resize
    /// misses. Pinning the reveal during a drag keeps this key constant too.
    fn ensure_layout(&mut self) {
        let active = self.reveal_range();
        // Key on the active range's logical line span (equivalent to the
        // build's per-line `on_line` reveal test) so within-line motion is a cache
        // hit. The image UI state (a selection border, a pinned source stacked
        // above the picture) and FONT_SIZE are part of the key too, so a future
        // zoom or font change cannot serve stale geometry.
        let active_lines = {
            let rope = self.text.rope();
            (rope.byte_to_line(active.0), rope.byte_to_line(active.1))
        };
        let key = (
            self.text.version(),
            self.width,
            FONT_SIZE,
            active_lines.0,
            active_lines.1,
            self.selected_image,
            self.raw_image,
        );
        if self.layout_key != Some(key) {
            let wrap_width = self.width as i32 - 2 * MARGIN;
            // Decode any newly referenced images before laying out, so an image
            // line gets its real height on the first build (no second pass). A
            // hit is free, so unchanged images cost only a cheap rescan.
            self.scan_images();
            let layout = Layout::build_with_images(
                &self.fonts,
                self.text.rope(),
                MARGIN,
                MARGIN,
                wrap_width,
                FONT_SIZE,
                active,
                &self.images,
                self.image_ui(),
            );
            self.content_height = layout.text_height() + 2 * MARGIN;
            self.layout_cache = Some(layout);
            self.layout_key = Some(key);
        }
    }

    /// The current per-image UI state, for the layout.
    fn image_ui(&self) -> ImageUi {
        ImageUi {
            selected: self.selected_image,
            raw: self.raw_image,
        }
    }

    /// Ensure every standalone `![](path)` line in the buffer has had its image
    /// decoded and cached, so [`Self::ensure_layout`] can size the image lines.
    /// Runs only on a layout rebuild; a path already attempted is a cache hit.
    fn scan_images(&mut self) {
        let rope = self.text.rope();
        let n = rope.len_lines();
        for i in 0..n {
            let ls = rope.line_to_byte(i);
            let le = if i + 1 < n {
                rope.line_to_byte(i + 1) - 1
            } else {
                rope.line_to_byte(i + 1)
            };
            let line = rope.byte_slice(ls..le);
            if let Some(link) = markdown::standalone_image(&line) {
                let raw = line[link.path.0..link.path.1].to_string();
                self.images.ensure(&raw);
            }
        }
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

/// Whether a clipboard MIME type carries text we can paste.
fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/") || mime == "UTF8_STRING"
}

/// The per-frame inputs [`paint_document`] needs beyond the layout itself: the
/// surface size, the scroll offset, the live selection and cursor, and whether
/// the caret is on this blink. Bundled so the paint signature stays readable and
/// so the perf harness can drive a paint without a live surface.
pub(crate) struct PaintParams {
    pub width: usize,
    pub height: usize,
    pub scroll_y: i32,
    pub selection: Option<(usize, usize)>,
    pub cursor: usize,
    pub caret_visible: bool,
}

/// Paint one frame of the document into `canvas`, building the display list and
/// executing it in full. The convenience used by tests and the perf harness; the
/// live editor instead diffs the list against the buffer's last frame and repaints
/// only the changed regions (see [`State::render_frame`]).
pub(crate) fn paint_document(
    canvas: &mut Canvas,
    fonts: &Fonts,
    layout: &Layout,
    rope: &crate::rope::Rope,
    images: &ImageStore,
    p: &PaintParams,
) {
    let list = build_display_list(fonts, layout, rope, p);
    display::execute(canvas, fonts, images, &list, None);
}

/// A [`DrawCmd::Text`] for `text` from `x` along `baseline` in `face`/`color`,
/// where `end` is the pen x after the run (its drawn width). The damage bounds are
/// the run's box padded for glyph ink that overhangs the advance, so a diff or a
/// clip never shears a glyph's edge.
fn text_cmd(
    face: FaceKey,
    x: i32,
    baseline: i32,
    end: i32,
    color: u32,
    text: String,
    m: Metrics,
) -> DrawCmd {
    let size = face.size();
    let pad_x = (size as i32 / 2).max(2);
    let pad_y = (size as i32 / 4).max(1);
    let bounds = Rect {
        x: x - pad_x,
        y: baseline - m.ascent - pad_y,
        w: (end - x).max(0) + 2 * pad_x,
        h: m.ascent + m.descent + 2 * pad_y,
    };
    DrawCmd::Text {
        bounds,
        x,
        baseline,
        face,
        color,
        text,
    }
}

/// Build the frame as a display list in screen (post-scroll) pixels: background,
/// code-block panels, inline-code panels, the selection highlight, the text (run
/// by run, in each run's face and color, with markers collapsed or revealed as
/// the layout decided), and the caret, in that stacking order. Pure CPU work over
/// the layout and rope, with no Wayland or `State` dependency. Off-screen lines
/// and spans are skipped, so the list (and the cost) track the viewport, not the
/// document.
pub(crate) fn build_display_list(
    fonts: &Fonts,
    layout: &Layout,
    rope: &crate::rope::Rope,
    p: &PaintParams,
) -> DisplayList {
    let (width, scroll_y) = (p.width, p.scroll_y);
    let view_h = p.height as i32;
    // Roughly a few commands per visual line (panel, runs, decorations) plus the
    // background and caret; one upfront allocation beats growing the list mid-build.
    let mut list: DisplayList = Vec::with_capacity(layout.lines().len() * 3 + 8);

    // The background, full surface, first so a clipped repaint clears its region
    // before the rest of the list draws over it.
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: 0,
            y: 0,
            w: width as i32,
            h: view_h,
        },
        color: BACKGROUND,
    });

    // Block backgrounds paint first, beneath both the selection highlight and the
    // text: a fenced code block sits on a full-width tinted panel, one line box
    // tall so consecutive lines tile into a continuous block, and that panel must
    // sit behind the selection rather than cover it (otherwise a selection inside
    // the block would vanish). A collapsed fence row still paints, so an unfocused
    // block keeps blank top and bottom rows that read as vertical padding. The
    // block's first row rounds its top corners and its last row its bottom ones, so
    // the whole panel reads as one softly rounded block.
    let lines = layout.lines();
    let panel_w = (width as i32 - 2 * MARGIN).max(0);
    for (i, line) in lines.iter().enumerate() {
        if !line.code_block {
            continue;
        }
        let m = fonts.metrics(line.size);
        let top = line.baseline - scroll_y - m.ascent;
        let h = m.line_height;
        if top + h <= 0 || top >= view_h {
            continue;
        }
        let first = i == 0 || !lines[i - 1].code_block;
        let last = i + 1 >= lines.len() || !lines[i + 1].code_block;
        let corners = match (first, last) {
            (true, true) => render::RoundedCorners::Both,
            (true, false) => render::RoundedCorners::Top,
            (false, true) => render::RoundedCorners::Bottom,
            (false, false) => render::RoundedCorners::None,
        };
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: MARGIN,
                y: top,
                w: panel_w,
                h,
            },
            color: CODE_BG,
            radius: CODE_RADIUS,
            corners,
        });
    }

    // Inline code spans get their own tinted panel, also beneath the selection so a
    // selection over inline code shows through (the glyphs still draw on top in the
    // text loop). Block code is excluded here; its full-width panel above covers it.
    for rect in layout.code_rects(fonts, rope) {
        let y = rect.y - scroll_y;
        if y + rect.h <= 0 || y >= view_h {
            continue;
        }
        list.push(DrawCmd::Fill {
            rect: Rect { y, ..rect },
            color: CODE_BG,
        });
    }

    // Everything is laid out in content space; the viewport scroll shifts it up by
    // `scroll_y`. Off-screen rectangles are dropped from the list.
    if let Some((start, end)) = p.selection {
        for rect in layout.selection_rects(fonts, rope, start, end) {
            let y = rect.y - scroll_y;
            if y + rect.h <= 0 || y >= view_h {
                continue;
            }
            list.push(DrawCmd::Fill {
                rect: Rect { y, ..rect },
                color: SELECTION_COLOR,
            });
        }
    }

    // The text, one visual line at a time, each at its own font size, materialized
    // from the rope. A line is a sequence of inline runs: regular text, code spans
    // (drawn in the code face on a tinted panel), and the delimiter markers, which
    // are dimmed on the cursor's line and collapse to nothing (skipped) once it
    // leaves, the live-preview reveal that `layout` bakes into each run's `visible`.
    for line in lines {
        let m = fonts.metrics(line.size);
        let baseline = line.baseline - scroll_y;
        // An image line blits its decoded picture into its (taller-than-text) box,
        // centered with the layout's vertical padding, and nothing else draws on the
        // line. Cull on the full box height so a tall image straddling a viewport
        // edge still paints. A picture that has not decoded leaves the box blank.
        if let Some(img) = line.image {
            let box_top = baseline - m.ascent;
            if box_top < view_h && box_top + img.box_height() > 0 {
                let pic = Rect {
                    x: MARGIN + line.indent,
                    y: box_top + (img.box_height() - img.height) / 2,
                    w: img.width,
                    h: img.height,
                };
                // Blit the cached display-size bitmap (resampled once, not per
                // frame); a no-op at paint time if the image is not decoded.
                list.push(DrawCmd::Image {
                    rect: pic,
                    path: rope.byte_slice(img.path_start..img.path_end),
                });
                // A selected image gets an accent border (four thin edge fills) and
                // a corner toggle button: a rounded panel with a centered `</>`
                // "source" glyph, the affordance to reveal its markdown.
                if img.selected {
                    push_image_chrome(&mut list, fonts, &pic);
                }
            }
            continue;
        }
        // Skip lines fully above or below the viewport, so a frame only rasterizes
        // what is on screen.
        if baseline + m.descent <= 0 || baseline - m.ascent >= view_h {
            continue;
        }
        // A blockquote draws a bar down its gutter, on every wrapped line.
        if matches!(line.gutter, Gutter::QuoteBar { .. }) {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: MARGIN,
                    y: baseline - m.ascent,
                    w: QUOTE_BAR_WIDTH,
                    h: m.line_height,
                },
                color: QUOTE_BAR_COLOR,
            });
        }
        // A thematic break draws a horizontal line centered in its row; the raw `---`
        // is collapsed (its runs are hidden), so nothing else paints on the line.
        if line.rule {
            let mid = baseline - m.ascent + m.line_height / 2;
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: MARGIN,
                    y: mid - RULE_THICKNESS / 2,
                    w: (width as i32 - 2 * MARGIN).max(0),
                    h: RULE_THICKNESS,
                },
                color: RULE_COLOR,
            });
        }
        // Content is inset by the line's hanging indent (0 unless this is a wrapped
        // list/quote continuation).
        let heading = line.heading.is_some();
        let mut pen = MARGIN + line.indent;
        for (i, run) in line.runs.iter().enumerate() {
            if !run.visible {
                continue;
            }
            // Each run draws in its own face, so bold/italic emphasis is carried by
            // the glyphs, not color: only code and markers recolor, the rest
            // (regular and emphasized prose) is FG.
            let fk = face_key(line.size, run.kind, heading);
            let face = fonts.face_for(fk);
            let slice = rope.byte_slice(run.start..run.end);
            // The block marker (the first run on a list/quote line) gives way to its
            // gutter decoration: a bullet glyph for a list, nothing for a quote (the
            // bar above replaces it). Either keeps the source width, so text never
            // shifts.
            if i == 0 && matches!(run.kind, markdown::SpanKind::Marker) {
                let advance = render::text_advance(face, &slice).round() as i32;
                match line.gutter {
                    Gutter::Bullet => {
                        // Place the marker's leading indentation first, so a nested
                        // item's dot sits at its depth, then the dot in place of the
                        // dash. The dot stands in for the `- ` marker, so it takes
                        // the muted marker color (MARKER_COLOR), not the body FG.
                        let indent = slice.len() - slice.trim_start().len();
                        let dot_x =
                            pen + render::text_advance(face, &slice[..indent]).round() as i32;
                        let end =
                            (dot_x as f32 + render::text_advance(face, BULLET)).round() as i32;
                        list.push(text_cmd(
                            fk,
                            dot_x,
                            baseline,
                            end,
                            MARKER_COLOR,
                            BULLET.to_string(),
                            m,
                        ));
                        pen += advance;
                        continue;
                    }
                    Gutter::QuoteBar { marker: true } => {
                        pen += advance;
                        continue;
                    }
                    _ => {}
                }
            }
            let color = match run.kind {
                markdown::SpanKind::Marker => MARKER_COLOR,
                markdown::SpanKind::Text(markdown::Emphasis::Code) => CODE_COLOR,
                markdown::SpanKind::Text(markdown::Emphasis::Link) => LINK_COLOR,
                // Heading prose takes the level's color (the marker and any inline
                // code on the line keep their own); body prose is FG.
                markdown::SpanKind::Text(_) => match line.heading {
                    Some(level) => HEADING_COLORS
                        .get(level.saturating_sub(1) as usize)
                        .copied()
                        .unwrap_or(FOREGROUND),
                    None => FOREGROUND,
                },
            };
            // Code backgrounds (inline panels and the full-width block panel) are
            // emitted in the pre-pass above, beneath the selection; here we only
            // draw the glyphs over them.
            let run_x = pen;
            let end = (run_x as f32 + render::text_advance(face, &slice)).round() as i32;
            list.push(text_cmd(fk, run_x, baseline, end, color, slice, m));
            pen = end;
            // Inline decorations over the glyphs, each spanning the run's drawn
            // width: a link is underlined and struck text gets a line through its
            // middle, both in the run's own color. Thickness scales with the
            // line's size so it stays proportionate on larger headings.
            let deco = match run.kind {
                markdown::SpanKind::Text(markdown::Emphasis::Link) => {
                    Some(baseline + m.descent / 3)
                }
                markdown::SpanKind::Text(markdown::Emphasis::Strikethrough) => {
                    Some(baseline - m.ascent / 3)
                }
                _ => None,
            };
            if let Some(y) = deco {
                let thickness = (line.size as i32 / 14).max(1);
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: run_x,
                        y,
                        w: end - run_x,
                        h: thickness,
                    },
                    color,
                });
            }
        }
    }

    // The caret, drawn only on the visible half of its blink cycle.
    if p.caret_visible {
        let (cx, cy, ch) = layout.caret(fonts, rope, p.cursor);
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: cx,
                y: cy - scroll_y,
                w: CARET_WIDTH,
                h: ch,
            },
            color: CARET_COLOR,
        });
    }

    list
}

/// Append the selection chrome for a selected image at picture rect `pic` (screen
/// space): a four-edge accent border and the corner `</>` toggle button.
fn push_image_chrome(list: &mut DisplayList, fonts: &Fonts, pic: &Rect) {
    let t = IMAGE_BORDER_THICKNESS;
    for rect in [
        Rect {
            x: pic.x,
            y: pic.y,
            w: pic.w,
            h: t,
        },
        Rect {
            x: pic.x,
            y: pic.y + pic.h - t,
            w: pic.w,
            h: t,
        },
        Rect {
            x: pic.x,
            y: pic.y,
            w: t,
            h: pic.h,
        },
        Rect {
            x: pic.x + pic.w - t,
            y: pic.y,
            w: t,
            h: pic.h,
        },
    ] {
        list.push(DrawCmd::Fill {
            rect,
            color: IMAGE_BORDER_COLOR,
        });
    }

    let btn = image_button_rect(pic);
    list.push(DrawCmd::RoundRect {
        rect: btn,
        color: IMAGE_BUTTON_BG,
        radius: IMAGE_BUTTON_RADIUS,
        corners: render::RoundedCorners::Both,
    });
    // Center the glyph in both axes from the code-face metrics.
    let fk = FaceKey::Code { size: FONT_SIZE };
    let face = fonts.face_for(fk);
    let glyph = "</>";
    let advance = render::text_advance(face, glyph).round() as i32;
    let m = fonts.metrics(FONT_SIZE);
    let gx = btn.x + (btn.w - advance) / 2;
    let gbaseline = btn.y + (btn.h - (m.ascent + m.descent)) / 2 + m.ascent;
    let gend = (gx as f32 + render::text_advance(face, glyph)).round() as i32;
    list.push(text_cmd(
        fk,
        gx,
        gbaseline,
        gend,
        FOREGROUND,
        glyph.to_string(),
        m,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{Image, ImageStore};
    use crate::rope::Rope;

    fn paint_params(width: usize, height: usize) -> PaintParams {
        PaintParams {
            width,
            height,
            scroll_y: 0,
            selection: None,
            cursor: 0,
            caret_visible: false,
        }
    }

    /// Painting a *selected* image draws its accent border and its corner button
    /// panel: end-to-end proof through `paint_document` that the selection chrome
    /// reaches the canvas (the click that sets the selection is covered by the
    /// layout hit-test tests).
    #[test]
    fn a_selected_image_paints_its_border_and_button() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("![x](cat.png)");
        // An opaque gray picture, distinct from the border/button colors.
        let img = Image {
            pixels: vec![0xff_80_80_80; 60 * 40],
            width: 60,
            height: 40,
        };
        let store = ImageStore::from_decoded(vec![("cat.png".to_string(), img)]);
        let (w, h) = (800usize, 600usize);
        let layout = Layout::build_with_images(
            &fonts,
            &rope,
            MARGIN,
            MARGIN,
            w as i32 - 2 * MARGIN,
            FONT_SIZE,
            (99, 99), // cursor off the image: it renders as a picture
            &store,
            ImageUi {
                selected: Some(0),
                raw: None,
            },
        );
        let mut buf = vec![0u8; w * h * 4];
        let mut canvas = Canvas::new(&mut buf, w, h);
        paint_document(
            &mut canvas,
            &fonts,
            &layout,
            &rope,
            &store,
            &paint_params(w, h),
        );

        let has = |color: u32| {
            buf.chunks_exact(4)
                .any(|p| u32::from_ne_bytes(p.try_into().unwrap()) & 0x00ff_ffff == color)
        };
        assert!(
            has(IMAGE_BORDER_COLOR),
            "the accent selection border is painted"
        );
        assert!(has(IMAGE_BUTTON_BG), "the corner button panel is painted");
    }

    /// An idle (unselected) image paints neither border nor button, so the chrome
    /// only appears on selection.
    #[test]
    fn an_idle_image_has_no_selection_chrome() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("![x](cat.png)");
        let img = Image {
            pixels: vec![0xff_80_80_80; 60 * 40],
            width: 60,
            height: 40,
        };
        let store = ImageStore::from_decoded(vec![("cat.png".to_string(), img)]);
        let (w, h) = (800usize, 600usize);
        let layout = Layout::build_with_images(
            &fonts,
            &rope,
            MARGIN,
            MARGIN,
            w as i32 - 2 * MARGIN,
            FONT_SIZE,
            (99, 99),
            &store,
            ImageUi::default(),
        );
        let mut buf = vec![0u8; w * h * 4];
        let mut canvas = Canvas::new(&mut buf, w, h);
        paint_document(
            &mut canvas,
            &fonts,
            &layout,
            &rope,
            &store,
            &paint_params(w, h),
        );
        let border = buf
            .chunks_exact(4)
            .any(|p| u32::from_ne_bytes(p.try_into().unwrap()) & 0x00ff_ffff == IMAGE_BORDER_COLOR);
        assert!(!border, "no border without selection");
    }

    /// Repainting only the damaged regions when going from frame `a` to frame `b`
    /// must land on exactly the pixels a full repaint of `b` would: the invariant
    /// the whole damage path rests on, and the one an under-reporting diff breaks.
    fn assert_incremental_matches_full(
        fonts: &Fonts,
        images: &ImageStore,
        a: &DisplayList,
        b: &DisplayList,
        w: usize,
        h: usize,
        label: &str,
    ) {
        // Reference: b painted from scratch.
        let mut full = vec![0u8; w * h * 4];
        {
            let mut c = Canvas::new(&mut full, w, h);
            display::execute(&mut c, fonts, images, b, None);
        }
        // Incremental: a painted in full, then only damage(a, b) repainted with b.
        let mut inc = vec![0u8; w * h * 4];
        {
            let mut c = Canvas::new(&mut inc, w, h);
            display::execute(&mut c, fonts, images, a, None);
        }
        let dmg = display::damage(a, b, w as i32, h as i32);
        {
            let mut c = Canvas::new(&mut inc, w, h);
            for clip in &dmg {
                display::execute(&mut c, fonts, images, b, Some(*clip));
            }
        }
        assert!(
            inc == full,
            "incremental repaint must match a full repaint: {label}"
        );
    }

    #[test]
    fn incremental_repaint_matches_a_full_repaint() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let images = ImageStore::new(None);
        let rope =
            Rope::from_str("# Title\n\nSome *body* text with `code`.\n\nsecond paragraph here\n");
        let (w, h) = (400usize, 300usize);
        let wrap = w as i32 - 2 * MARGIN;
        let far = rope.len().saturating_sub(2); // inside the last paragraph

        // The cursor on the heading reveals its `#` (and pushes the title right);
        // far away the marker collapses and the title shifts left.
        let lay_top = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let lay_far = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (far, far));

        let frame =
            |layout: &Layout, cursor: usize, caret: bool, selection: Option<(usize, usize)>| {
                build_display_list(
                    &fonts,
                    layout,
                    &rope,
                    &PaintParams {
                        width: w,
                        height: h,
                        scroll_y: 0,
                        selection,
                        cursor,
                        caret_visible: caret,
                    },
                )
            };

        let on_top = frame(&lay_top, 0, true, None);
        let off_top = frame(&lay_top, 0, false, None);
        let sel_top = frame(&lay_top, 0, true, Some((0, 5)));
        let far_frame = frame(&lay_far, far, true, None);

        // Caret blink, a selection appearing, and a cursor move that collapses the
        // heading marker and shifts its text (both directions): each incremental
        // repaint must be pixel-identical to painting the target frame from scratch.
        assert_incremental_matches_full(&fonts, &images, &on_top, &off_top, w, h, "caret blink");
        assert_incremental_matches_full(
            &fonts,
            &images,
            &on_top,
            &sel_top,
            w,
            h,
            "selection appears",
        );
        assert_incremental_matches_full(
            &fonts,
            &images,
            &on_top,
            &far_frame,
            w,
            h,
            "cursor leaves heading",
        );
        assert_incremental_matches_full(
            &fonts,
            &images,
            &far_frame,
            &on_top,
            w,
            h,
            "cursor enters heading",
        );
    }
}
