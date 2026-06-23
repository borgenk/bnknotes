//! The editor's orchestration: it binds the compositor globals, builds the
//! xdg-shell surface stack, and runs the event loop.
//!
//! On top of the earlier single-line field, this drives a small but real
//! multi-line editor. A keyboard (via libxkbcommon) inserts text and newlines
//! and moves the cursor by grapheme cluster; Up/Down/Home/End navigate the
//! visual lines a [`Layout`] pass computes; a held Shift turns motion into a
//! selection, which a pointer can also drag out; the selection highlights behind
//! the text and Backspace or typing replaces it. Ctrl shortcuts select-all,
//! copy, cut, paste (over wl_data_device), and undo/redo; a held key repeats on a
//! timer. Content taller than the window scrolls (mouse wheel, or automatically
//! to keep the caret in view), and the window honors compositor resizes by
//! reallocating its shm buffers. Rendering double-buffers so the compositor never
//! reads a frame we are still painting.

use std::os::fd::AsRawFd;
use std::time::{Duration, Instant};

use crate::conn::{Connection, Fill};
use crate::error::{Error, Result};
use crate::ffi;
use crate::freetype::Fonts;
use crate::layout::Layout;
use crate::markdown;
use crate::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device, wl_data_device_manager,
    wl_data_offer, wl_data_source, wl_display, wl_keyboard, wl_pointer, wl_registry, wl_seat,
    wl_shm, wl_shm_pool, wl_surface, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::render::Canvas;
use crate::scroll;
use crate::shm::Pool;
use crate::text::TextBuffer;
use crate::wire::{Arg, Message, Reader};
use crate::xkb::{KeyAction, Xkb};

/// Initial window size; the compositor may resize us to whatever it likes.
const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;

/// Background, text, and selection colors (0x00RRGGBB), and the pixel size.
const BACKGROUND: u32 = 0x001e_1e2e;
const FOREGROUND: u32 = 0x00e0_e0e0;
const SELECTION_COLOR: u32 = 0x0026_4f78;
/// Dimmed color for markdown syntax markers (the `#`s of a heading), so they
/// read as syntax rather than content.
const MARKER_COLOR: u32 = 0x006c_7086;
/// Body text pixel size; headings scale up from it (see [`markdown::font_size`]).
const FONT_SIZE: u32 = 32;
/// Inset of the text from the top-left corner (both axes).
const MARGIN: i32 = 40;
/// The caret: a thin vertical bar at the cursor position.
const CARET_COLOR: u32 = 0x00ff_ffff;
const CARET_WIDTH: i32 = 2;
/// Pixels scrolled per unit of wl_pointer.axis value (wheel/touchpad).
const SCROLL_FACTOR: f32 = 2.0;

/// Fallback auto-repeat timing if the compositor never sends repeat_info.
const DEFAULT_REPEAT_DELAY_MS: u64 = 400;
const DEFAULT_REPEAT_RATE: u32 = 25;
/// Cap on repeats fired per wake-up, so a stalled frame cannot flood input.
const REPEAT_CATCHUP_CAP: u32 = 16;

/// MIME types we advertise when putting text on the clipboard, and accept when
/// pasting. The charset variant is listed first as the preferred form.
const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

pub fn run() -> Result<()> {
    State::new()?.bring_up()
}

struct State {
    conn: Connection,
    /// The font opened at every size the styles need (body and heading sizes),
    /// so each line can render at its own size.
    fonts: Fonts,
    xkb: Xkb,
    text: TextBuffer,

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

    // Pointer state: latest position (surface-local pixels) and whether a left
    // button drag is currently extending a selection.
    pointer_x: f32,
    pointer_y: f32,
    selecting: bool,
    /// Sticky x for vertical motion, so repeated Up/Down keep a column.
    goal_x: Option<f32>,
    /// How far the viewport is scrolled down the document, in pixels.
    scroll_y: i32,

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
    // wl_data_source and `clipboard_text` is what it serves. Incoming offers are
    // tracked so a paste can request the right MIME type.
    clipboard_source: u32,
    clipboard_text: String,
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
}

impl State {
    fn new() -> Result<Self> {
        let conn = Connection::connect()?;
        // Open the font at every size the styles use, so headings can render
        // larger than body text without re-opening mid-frame.
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
        Ok(Self {
            conn,
            fonts,
            xkb: Xkb::new()?,
            text: TextBuffer::new(),
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
            pointer_x: 0.0,
            pointer_y: 0.0,
            selecting: false,
            goal_x: None,
            scroll_y: 0,
            repeat_delay: Duration::from_millis(DEFAULT_REPEAT_DELAY_MS),
            repeat_interval: Some(Duration::from_secs(1) / DEFAULT_REPEAT_RATE),
            repeat_key: None,
            repeat_at: None,
            last_serial: 0,
            clipboard_source: 0,
            clipboard_text: String::new(),
            incoming_offer: 0,
            incoming_text_mime: None,
            selection_offer: 0,
            selection_text_mime: None,
            pool: None,
            pool_id: 0,
            buffers: [0, 0],
            busy: [false, false],
            frame_len: 0,
            next_id: 2, // 1 is wl_display
            free_ids: Vec::new(),
            pending_sync: None,
            pending_size: None,
            configured: false,
            closed: false,
            dirty: true,
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
        self.conn.request(
            self.toplevel,
            xdg_toplevel::SET_TITLE,
            &[Arg::Str("bnknotes")],
        );

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
            "bnknotes: type to edit; Enter, arrows/Home/End, Shift selects, click/drag; \
             Ctrl+A/C/X/V select-all/copy/cut/paste, Ctrl+Z/Y undo/redo; held keys repeat; \
             close to exit"
        );
        self.run_until(|s| s.closed)?;
        Ok(())
    }

    /// Allocate the shm pool and its two buffers at the current size.
    fn create_buffers(&mut self) -> Result<()> {
        let shm = self
            .shm
            .ok_or_else(|| Error::msg("creating buffers before wl_shm was bound"))?;
        let stride = self.width * 4;
        let frame_len = (stride * self.height) as usize;
        let pool_size = frame_len * 2;
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
            if self.configured && self.dirty && self.render_frame()? {
                self.dirty = false;
            }
            if done(self) {
                return Ok(());
            }
            self.conn.flush()?;
            let timeout = self.repeat_timeout();
            if let Fill::Bytes(0) = self.conn.fill(timeout)? {
                return Err(Error::msg("compositor closed the connection"));
            }
        }
    }

    /// Paint the current text into a free buffer and present it. Returns false
    /// if both buffers are still held by the compositor, leaving `dirty` set so
    /// the next buffer release triggers another attempt.
    fn render_frame(&mut self) -> Result<bool> {
        let Some(idx) = (0..2).find(|&i| !self.busy[i]) else {
            return Ok(false);
        };
        let frame_len = self.frame_len;
        let offset = idx * frame_len;
        let width = self.width as usize;
        let height = self.height as usize;
        let selection = self.text.selection();
        let cursor = self.text.cursor();
        let layout = self.layout();
        // Keep the scroll offset valid for the current content and window size.
        let content_height = self.content_height(&layout);
        self.scroll_y = scroll::clamp(self.scroll_y, content_height, self.height as i32);
        let scroll_y = self.scroll_y;

        {
            let Some(pool) = self.pool.as_mut() else {
                return Ok(false);
            };
            let fonts = &self.fonts;
            let text = self.text.text();
            let frame = &mut pool.data_mut()[offset..offset + frame_len];
            let mut canvas = Canvas::new(frame, width, height);
            canvas.fill(BACKGROUND);

            // Everything is laid out in content space; the viewport scroll shifts
            // it up by `scroll_y`. The canvas clips whatever falls off-screen.
            if let Some((start, end)) = selection {
                for rect in layout.selection_rects(fonts, text, start, end) {
                    canvas.fill_rect(rect.x, rect.y - scroll_y, rect.w, rect.h, SELECTION_COLOR);
                }
            }

            // The text, one visual line at a time, each at its own font size.
            // A heading whose markers are hidden (the cursor has left the line)
            // draws only the text past them, flush to the margin. When the
            // markers are shown they are dimmed so the markup reads as syntax.
            for line in layout.lines() {
                let face = fonts.face(line.size);
                let baseline = line.baseline - scroll_y;
                if line.hidden_prefix > 0 {
                    let visible = &text[line.start + line.hidden_prefix..line.end];
                    canvas.draw_text(face, MARGIN, baseline, visible, FOREGROUND);
                    continue;
                }
                let slice = &text[line.start..line.end];
                let marker = markdown::marker_len(slice);
                if marker > 0 {
                    let pen =
                        canvas.draw_text(face, MARGIN, baseline, &slice[..marker], MARKER_COLOR);
                    canvas.draw_text(face, pen, baseline, &slice[marker..], FOREGROUND);
                } else {
                    canvas.draw_text(face, MARGIN, baseline, slice, FOREGROUND);
                }
            }

            // The caret.
            let (cx, cy, ch) = layout.caret(fonts, text, cursor);
            canvas.fill_rect(cx, cy - scroll_y, CARET_WIDTH, ch, CARET_COLOR);
        }

        let buffer = self.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        self.conn.request(
            self.surface,
            wl_surface::DAMAGE,
            &[
                Arg::Int(0),
                Arg::Int(0),
                Arg::Int(self.width as i32),
                Arg::Int(self.height as i32),
            ],
        );
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.busy[idx] = true;
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
                    self.free_ids.push(id);
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
                    // A zero dimension means the compositor leaves it to us.
                    if w > 0 && h > 0 {
                        self.pending_size = Some((w, h));
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
            wl_keyboard::EV_ENTER => self.last_serial = r.u32()?,
            wl_keyboard::EV_LEAVE => self.stop_repeat(),
            _ => {}
        }
        Ok(())
    }

    fn apply_action(&mut self, action: KeyAction) -> Result<()> {
        let extend = self.xkb.shift_active();
        let changed = match action {
            KeyAction::Char(c) => {
                let mut encoded = [0u8; 4];
                self.text.insert_str(c.encode_utf8(&mut encoded));
                self.goal_x = None;
                true
            }
            KeyAction::Enter => {
                self.text.insert_str("\n");
                self.goal_x = None;
                true
            }
            KeyAction::Backspace => {
                self.goal_x = None;
                self.text.backspace()
            }
            KeyAction::Delete => {
                self.goal_x = None;
                self.text.delete_forward()
            }
            KeyAction::Left => {
                self.goal_x = None;
                self.text.move_left(extend)
            }
            KeyAction::Right => {
                self.goal_x = None;
                self.text.move_right(extend)
            }
            KeyAction::Up => self.move_vertical(-1, extend),
            KeyAction::Down => self.move_vertical(1, extend),
            KeyAction::Home => {
                self.goal_x = None;
                self.move_line_edge(false, extend)
            }
            KeyAction::End => {
                self.goal_x = None;
                self.move_line_edge(true, extend)
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
            // The clipboard actions manage their own redraw and may do I/O.
            KeyAction::Copy => {
                self.copy_selection()?;
                false
            }
            KeyAction::Cut => {
                self.cut_selection()?;
                false
            }
            KeyAction::Paste => {
                self.paste()?;
                false
            }
            KeyAction::None => false,
        };
        if changed {
            self.dirty = true;
            self.scroll_caret_into_view();
        }
        Ok(())
    }

    /// Move the cursor `dir` lines (-1 up, +1 down), keeping a sticky goal x.
    fn move_vertical(&mut self, dir: i32, extend: bool) -> bool {
        let layout = self.layout();
        let cursor = self.text.cursor();
        let (caret_x, _, _) = layout.caret(&self.fonts, self.text.text(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        self.goal_x = Some(goal);
        let target = layout.offset_vertical(&self.fonts, self.text.text(), cursor, dir, goal);
        let before_sel = self.text.selection();
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// Move the cursor to the start (`to_end = false`) or end of its line.
    fn move_line_edge(&mut self, to_end: bool, extend: bool) -> bool {
        let layout = self.layout();
        let cursor = self.text.cursor();
        let (start, end) = layout.line_bounds(cursor);
        let target = if to_end { end } else { start };
        let before_sel = self.text.selection();
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                let _serial = r.u32()?;
                let _surface = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
            }
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                self.pointer_x = r.fixed()?;
                self.pointer_y = r.fixed()?;
                if self.selecting {
                    // Drag extends the selection from the press point.
                    self.place_cursor_at(self.pointer_x, self.pointer_y, true);
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
                        self.place_cursor_at(self.pointer_x, self.pointer_y, false);
                        self.selecting = true;
                    } else {
                        self.selecting = false;
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
        let layout = self.layout();
        let offset = layout.offset_at(&self.fonts, self.text.text(), x, y + self.scroll_y as f32);
        let before = self.text.cursor();
        let before_sel = self.text.selection();
        self.text.set_cursor(offset, extend);
        if self.text.cursor() != before || self.text.selection() != before_sel {
            self.dirty = true;
        }
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
                // A target is pasting our data: write it to the supplied fd.
                let _mime = r.string()?;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("data_source.send arrived without its fd"))?;
                ffi::write_all(fd.as_raw_fd(), self.clipboard_text.as_bytes())?;
            }
            wl_data_source::EV_CANCELLED => {
                // Another selection replaced ours; tear the source down.
                if self.clipboard_source != 0 {
                    self.conn
                        .request(self.clipboard_source, wl_data_source::DESTROY, &[]);
                }
                self.clipboard_source = 0;
                self.clipboard_text.clear();
            }
            _ => {}
        }
        Ok(())
    }

    /// Put the current selection on the clipboard. No-op without a selection.
    fn copy_selection(&mut self) -> Result<()> {
        if let Some(text) = self.text.selected_text() {
            let text = text.to_string();
            self.set_clipboard(text);
        }
        Ok(())
    }

    /// Copy the selection, then delete it.
    fn cut_selection(&mut self) -> Result<()> {
        let Some(text) = self.text.selected_text() else {
            return Ok(());
        };
        let text = text.to_string();
        self.set_clipboard(text);
        if self.text.backspace() {
            self.dirty = true;
            self.scroll_caret_into_view();
        }
        Ok(())
    }

    /// Become the clipboard owner with `text`, replacing any source we held.
    fn set_clipboard(&mut self, text: String) {
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
        for mime in CLIPBOARD_MIMES {
            self.conn
                .request(source, wl_data_source::OFFER, &[Arg::Str(mime)]);
        }
        self.conn.request(
            self.data_device,
            wl_data_device::SET_SELECTION,
            &[Arg::Object(source), Arg::Uint(self.last_serial)],
        );
        self.clipboard_source = source;
        self.clipboard_text = text;
    }

    /// Insert the clipboard's text at the cursor.
    fn paste(&mut self) -> Result<()> {
        // If we own the clipboard, insert our own text directly. Going through
        // the compositor would deadlock: we'd block reading the pipe while also
        // needing to service our own source's send event.
        if self.clipboard_source != 0 {
            if !self.clipboard_text.is_empty() {
                let text = self.clipboard_text.clone();
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

    /// Total document height in content pixels: the laid-out text block plus a
    /// top and bottom margin.
    fn content_height(&self, layout: &Layout) -> i32 {
        layout.text_height() + 2 * MARGIN
    }

    /// Scroll the viewport by `delta` pixels (positive = down), clamped to the
    /// document.
    fn scroll_by(&mut self, delta: i32) {
        let layout = self.layout();
        let content_height = self.content_height(&layout);
        let sy = scroll::clamp(self.scroll_y + delta, content_height, self.height as i32);
        if sy != self.scroll_y {
            self.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// Scroll the minimum needed to keep the caret visible after a cursor move.
    fn scroll_caret_into_view(&mut self) {
        let layout = self.layout();
        let (_, caret_top, caret_h) =
            layout.caret(&self.fonts, self.text.text(), self.text.cursor());
        let content_height = self.content_height(&layout);
        let sy = scroll::reveal(
            self.scroll_y,
            caret_top,
            caret_h,
            self.height as i32,
            content_height,
            MARGIN,
        );
        if sy != self.scroll_y {
            self.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// Lay out the current text, wrapping to the window width less a margin on
    /// each side. Per-line metrics come from the font set during the build. The
    /// cursor (or selection) is passed in as the active range so a heading the
    /// caret is editing keeps its `#` markers while the rest collapse them.
    fn layout(&self) -> Layout {
        let wrap_width = self.width as i32 - 2 * MARGIN;
        let cursor = self.text.cursor();
        let active = self.text.selection().unwrap_or((cursor, cursor));
        Layout::build(
            &self.fonts,
            self.text.text(),
            MARGIN,
            MARGIN,
            wrap_width,
            FONT_SIZE,
            active,
        )
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
