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

use std::collections::BTreeSet;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::conn::{Connection, Fill};
use crate::display::{self, DisplayList, DrawCmd};
use crate::dmabuf;
use crate::error::{Error, Result};
use crate::ffi;
use crate::file::{self, FileFormat};
use crate::find::{self, FindState};
use crate::freetype::{FaceKey, FontStyle, Fonts, Metrics};
use crate::gpu;
use crate::grapheme;
use crate::image::ImageStore;
use crate::layout::{face_key, image_button_rect, Gutter, ImageUi, Layout, Rect};
use crate::markdown;
use crate::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device, wl_data_device_manager,
    wl_data_offer, wl_data_source, wl_display, wl_keyboard, wl_pointer, wl_registry, wl_seat,
    wl_surface, wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
    wp_linux_drm_syncobj_manager_v1, wp_linux_drm_syncobj_surface_v1, xdg_surface, xdg_toplevel,
    xdg_wm_base, zwp_linux_buffer_params_v1, zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use crate::scroll;
use crate::shape;
use crate::store;
use crate::text::TextBuffer;
use crate::tree;
use crate::vulkan;
use crate::wire::{Arg, Message, Reader};
use crate::xkb::{KeyAction, Xkb};

/// Initial window size; the compositor may resize us to whatever it likes.
const INIT_WIDTH: u32 = 800;
const INIT_HEIGHT: u32 = 600;
/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display (8K is 7680 wide), it just keeps a
/// bogus or hostile size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// Background, text, and selection colors (0x00RRGGBB), and the pixel size: a
/// dark palette on base hue ~212.
const BACKGROUND: u32 = 0x001c_2127;
const FOREGROUND: u32 = 0x00da_dada;
/// The accent blue (`hsl(215,75%,70%)`) pre-blended for selection: the fill is
/// opaque, so this is the accent at a set opacity, high enough to stay legible
/// over the darker code panel.
const SELECTION_COLOR: u32 = 0x0041_5776;
/// Fill behind find matches other than the current one; the current match is
/// the live selection, so a muted gold keeps the two apart at a glance.
const FIND_MATCH_COLOR: u32 = 0x005e_5128;
/// The find bar's floating panel.
const FIND_BG: u32 = 0x0031_3244;
/// An invalid regex query and its counter, drawn in the alert color.
const FIND_ERROR_COLOR: u32 = 0x00f3_8ba8;
/// The right-edge match ticks (brighter than the in-text match fill, since a
/// tick is a few pixels).
const FIND_TICK_COLOR: u32 = 0x00a8_9040;
/// Geometry of one right-edge match tick.
const TICK_WIDTH: i32 = 4;
const TICK_HEIGHT: i32 = 3;
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

/// File-tree sidebar and splitter colors and metrics.
const TREE_BG: u32 = 0x0016_1a1f;
const TREE_DIR_COLOR: u32 = 0x008c_a0b3;
const TREE_HOVER_BG: u32 = 0x0028_2c34;
const SPLITTER_COLOR: u32 = 0x003f_444b;
/// Inset for tree rows from the column edges, and the per-depth indent step.
const TREE_PAD: i32 = 6;
const TREE_INDENT: i32 = 14;
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

/// `--gpu-probe`: report what the compositor offers the dmabuf presentation
/// path (and, in later milestones, what the GPU offers), without opening a
/// window. This is the same negotiation the editor performs at startup, so a
/// successful probe means the GPU backend's protocol prerequisites hold.
pub fn gpu_probe() -> Result<()> {
    State::new(None)?.probe_dmabuf()
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
type LayoutKey = (
    u64,
    i32,
    i32,
    u32,
    usize,
    usize,
    Option<usize>,
    Option<usize>,
);

/// Client-owned explicit-sync state (`linux-drm-syncobj-v1`), negotiated over a
/// live GPU backend when the compositor advertises the manager. Two DRM syncobj
/// timelines the compositor shares: we signal `acquire` with each frame's
/// render-done fence, and it signals `release` when done reading a buffer. A
/// scratch binary syncobj carries the sync-file transfers. `point` increments
/// per present; `buffer_release[i]` is the release point last published for
/// buffer slot `i`, so that slot's next reuse waits on it. Dropping this closes
/// the render-node fd, which frees every syncobj created on it.
struct ExplicitSync {
    drm: OwnedFd,
    acquire_syncobj: u32,
    release_syncobj: u32,
    scratch: u32,
    acquire_timeline: u32,
    release_timeline: u32,
    surface: u32,
    point: u64,
    buffer_release: [Option<u64>; 2],
}

/// Which region holds keyboard focus. Editing keys go to the focused editor;
/// the file tree takes navigation keys when it is focused.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Tree,
    /// Content pane 0 (left/A) or 1 (right/B).
    Editor(usize),
}

/// Width of the grab bar between two columns: a few pixels of hit zone, with a
/// thin divider line drawn down its middle.
const SPLITTER: i32 = 6;

/// The adjustable column layout. The sidebar has a pixel width; the two content
/// panes split the remaining width by a ratio. Splitters drag these values.
struct PaneLayout {
    tree_width: i32,
    /// Pane A's share of the content area (right of the tree), clamped 0.1..0.9.
    split_ratio: f32,
    tree_visible: bool,
    second_open: bool,
}

/// The pixel rectangles the columns and their splitter bars occupy, derived
/// from the window size and the [`PaneLayout`]. A hidden column is `None`.
struct Columns {
    tree: Option<Rect>,
    /// Editor A is always present; editor B only when `second_open`.
    editors: [Option<Rect>; 2],
    /// Grab bars: `[0]` between tree and A, `[1]` between A and B.
    splitters: [Option<Rect>; 2],
}

impl PaneLayout {
    /// Lay the columns out across `width`×`height`. Editor A always takes what is
    /// left after an optional sidebar and an optional second pane; each column is
    /// full height, so content stays in its column by construction (no clip).
    fn columns(&self, width: i32, height: i32) -> Columns {
        let mut x = 0;
        let mut tree = None;
        let mut splitters = [None, None];
        if self.tree_visible {
            tree = Some(Rect {
                x,
                y: 0,
                w: self.tree_width,
                h: height,
            });
            x += self.tree_width;
            splitters[0] = Some(Rect {
                x,
                y: 0,
                w: SPLITTER,
                h: height,
            });
            x += SPLITTER;
        }
        let mut editors = [None, None];
        if self.second_open {
            let content = (width - x - SPLITTER).max(0);
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
                w: SPLITTER,
                h: height,
            });
            x += SPLITTER;
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
enum Hit {
    Splitter(usize),
    Tree(Rect),
    Editor(usize, Rect),
    None,
}

/// One content pane: an independent editor over its own buffer, carrying its own
/// file, inline images, memoized layout, scroll, find bar, and the drag/selection
/// transients a click-drag needs. Two of these live on [`State`].
struct EditorPane {
    text: TextBuffer,
    path: Option<PathBuf>,
    format: FileFormat,
    images: ImageStore,
    selected_image: Option<usize>,
    raw_image: Option<usize>,
    find: FindState,
    scroll_y: i32,
    content_height: i32,
    layout_cache: Option<Layout>,
    layout_key: Option<LayoutKey>,
    goal_x: Option<f32>,
    selecting: bool,
    frozen_reveal: Option<(usize, usize)>,
    last_click: Option<Instant>,
    last_click_pos: (f32, f32),
    click_count: u32,
    autoscroll: i32,
    autoscroll_at: Option<Instant>,
}

impl EditorPane {
    /// An empty, file-less pane (blank buffer, no images).
    fn empty() -> Self {
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
            layout_cache: None,
            layout_key: None,
            goal_x: None,
            selecting: false,
            frozen_reveal: None,
            last_click: None,
            last_click_pos: (0.0, 0.0),
            click_count: 0,
            autoscroll: 0,
            autoscroll_at: None,
        }
    }

    /// Load `path` into a fresh pane, resolving inline images against its
    /// directory. An unreadable file is an error the caller surfaces.
    fn open(path: &Path) -> Result<Self> {
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
    fn reveal_range(&self) -> (usize, usize) {
        let cursor = self.text.cursor();
        match (self.selecting, self.frozen_reveal) {
            (true, Some(range)) => range,
            _ => self.text.selection().unwrap_or((cursor, cursor)),
        }
    }

    /// The per-image UI state (selected border, pinned raw source) for the layout.
    fn image_ui(&self) -> ImageUi {
        ImageUi {
            selected: self.selected_image,
            raw: self.raw_image,
        }
    }

    /// Decode every standalone `![](path)` line so [`Self::ensure_layout`] can
    /// size the image lines. A path already attempted is a cache hit.
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

    /// Lay the text out into the column `rect`: wrap to its width less a margin
    /// each side, origin at its top-left plus margin. Memoized on `(version,
    /// column x, column width, font, active line span, image UI)`, so a scroll or
    /// idle redraw reuses the cache; only an edit, a cross-line move, or a column
    /// resize misses. Baking the column's `x` into the layout keeps content in
    /// its column with no clip.
    fn ensure_layout(&mut self, fonts: &Fonts, rect: Rect) {
        let active = self.reveal_range();
        let active_lines = {
            let rope = self.text.rope();
            (rope.byte_to_line(active.0), rope.byte_to_line(active.1))
        };
        let key = (
            self.text.version(),
            rect.x,
            rect.w,
            FONT_SIZE,
            active_lines.0,
            active_lines.1,
            self.selected_image,
            self.raw_image,
        );
        if self.layout_key != Some(key) {
            let wrap_width = rect.w - 2 * MARGIN;
            self.scan_images();
            let layout = Layout::build_with_images(
                fonts,
                self.text.rope(),
                rect.x + MARGIN,
                rect.y + MARGIN,
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

    /// Scroll by `delta` px within the column `rect`, clamped to the content.
    /// Returns whether the offset changed (so the caller can mark a repaint).
    fn scroll_by(&mut self, fonts: &Fonts, rect: Rect, delta: i32) -> bool {
        self.ensure_layout(fonts, rect);
        let sy = scroll::clamp(self.scroll_y + delta, self.content_height, rect.h);
        let moved = sy != self.scroll_y;
        self.scroll_y = sy;
        moved
    }

    /// Scroll the minimum to keep the caret visible after a cursor move; returns
    /// whether the offset changed.
    fn scroll_caret_into_view(&mut self, fonts: &Fonts, rect: Rect) -> bool {
        self.ensure_layout(fonts, rect);
        let cursor = self.text.cursor();
        let Some((_, caret_top, caret_h)) = self
            .layout_cache
            .as_ref()
            .map(|l| l.caret(fonts, self.text.rope(), cursor))
        else {
            return false;
        };
        let sy = scroll::reveal(
            self.scroll_y,
            caret_top,
            caret_h,
            rect.h,
            self.content_height,
            MARGIN,
        );
        let moved = sy != self.scroll_y;
        self.scroll_y = sy;
        moved
    }

    /// Move the cursor `dir` lines (-1 up, +1 down), keeping a sticky goal x.
    fn move_vertical(&mut self, fonts: &Fonts, rect: Rect, dir: i32, extend: bool) -> bool {
        self.ensure_layout(fonts, rect);
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        // Read the cached layout by reference; `goal_x`/`text` are disjoint
        // fields, so they can be touched while the layout borrow is live.
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (caret_x, _, _) = layout.caret(fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        let target = layout.offset_vertical(fonts, self.text.rope(), cursor, dir, goal);
        self.goal_x = Some(goal);
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// Move the cursor to the start (`to_end = false`) or end of its visual line.
    fn move_line_edge(&mut self, fonts: &Fonts, rect: Rect, to_end: bool, extend: bool) -> bool {
        self.ensure_layout(fonts, rect);
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (start, end) = layout.line_bounds(cursor);
        let target = if to_end {
            end
        } else {
            // Smart Home. Visual bounds, so it works on wrapped rows too.
            let line = self.text.rope().byte_slice(start..end);
            smart_home_target(&line, start, end, cursor)
        };
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// Move the cursor one column-height up/down, keeping the sticky goal column.
    fn move_page(&mut self, fonts: &Fonts, rect: Rect, dir: i32, extend: bool) -> bool {
        self.ensure_layout(fonts, rect);
        let cursor = self.text.cursor();
        let before_sel = self.text.selection();
        let page_lines = self.page_lines(fonts, rect);
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let (caret_x, _, _) = layout.caret(fonts, self.text.rope(), cursor);
        let goal = self.goal_x.unwrap_or(caret_x as f32);
        // Step a line at a time so each hop honors the goal column and the
        // varying per-line heights; stop early at the first or last line.
        let mut target = cursor;
        for _ in 0..page_lines {
            let next = layout.offset_vertical(fonts, self.text.rope(), target, dir, goal);
            if next == target {
                break;
            }
            target = next;
        }
        self.goal_x = Some(goal);
        self.text.set_cursor(target, extend);
        self.text.cursor() != cursor || self.text.selection() != before_sel
    }

    /// How many visual lines make up a Page Up/Down jump in this column: its
    /// height in body line-heights, less one row of overlap, at least one.
    fn page_lines(&self, fonts: &Fonts, rect: Rect) -> usize {
        let line_height = fonts.metrics(FONT_SIZE).line_height.max(1);
        (rect.h / line_height - 1).max(1) as usize
    }

    /// Content-space y for a window y inside this pane's column (the column top
    /// and this pane's scroll offset removed).
    fn content_y(&self, rect: Rect, py: f32) -> f32 {
        py - rect.y as f32 + self.scroll_y as f32
    }

    /// The byte offset under a window position within this pane's column.
    fn pointer_offset(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> usize {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        self.layout_cache
            .as_ref()
            .map_or(0, |l| l.offset_at(fonts, self.text.rope(), px, y))
    }

    /// Place the cursor (or, with `extend`, the selection's free end) at a window
    /// position within this pane's column. Returns whether it moved.
    fn place_cursor_at(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
        extend: bool,
    ) -> bool {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        let Some(layout) = self.layout_cache.as_ref() else {
            return false;
        };
        let offset = layout.offset_at(fonts, self.text.rope(), px, y);
        let before = self.text.cursor();
        let before_sel = self.text.selection();
        self.text.set_cursor(offset, extend);
        self.text.cursor() != before || self.text.selection() != before_sel
    }

    /// The destination of the openable link under a window position, if its line
    /// is rendered (markup collapsed) and the scheme is one the browser opens.
    fn link_under_pointer(
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
        let ls = rope.line_to_byte(line);
        let le = if line + 1 < rope.len_lines() {
            rope.line_to_byte(line + 1).saturating_sub(1)
        } else {
            rope.len()
        };
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

    /// The logical-line start of the rendered task checkbox under a window
    /// position, if any.
    fn checkbox_under_pointer(
        &mut self,
        fonts: &Fonts,
        rect: Rect,
        px: f32,
        py: f32,
    ) -> Option<usize> {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        self.layout_cache
            .as_ref()?
            .checkbox_at(fonts, self.text.rope(), px, y)
    }

    /// Handle a press on an image button (toggles its raw source) or picture
    /// (selects it). Returns whether it was consumed.
    fn image_press(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> bool {
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        let Some(layout) = self.layout_cache.as_ref() else {
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

    /// The pointer shape for a window position within this pane: a hand over a
    /// link/checkbox/image-button, the arrow over a picture, the I-beam elsewhere.
    fn cursor_shape_at(&mut self, fonts: &Fonts, rect: Rect, px: f32, py: f32) -> u32 {
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_POINTER, SHAPE_TEXT};
        let over_link = self.link_under_pointer(fonts, rect, px, py).is_some();
        let over_checkbox = self.checkbox_under_pointer(fonts, rect, px, py).is_some();
        self.ensure_layout(fonts, rect);
        let y = self.content_y(rect, py);
        match self.layout_cache.as_ref() {
            _ if over_link || over_checkbox => SHAPE_POINTER,
            Some(layout) if layout.image_button_at(fonts, px, y).is_some() => SHAPE_POINTER,
            Some(layout) if layout.image_at(fonts, px, y).is_some() => SHAPE_DEFAULT,
            _ => SHAPE_TEXT,
        }
    }

    /// Arm or disarm drag auto-scroll from the pointer's distance to this
    /// column's top/bottom edge.
    fn update_autoscroll(&mut self, rect: Rect, py: f32) {
        let y = py as i32 - rect.y;
        let step = if y < AUTOSCROLL_MARGIN {
            -AUTOSCROLL_STEP
        } else if y > rect.h - AUTOSCROLL_MARGIN {
            AUTOSCROLL_STEP
        } else {
            0
        };
        if step == 0 {
            self.autoscroll = 0;
            self.autoscroll_at = None;
        } else {
            self.autoscroll = step;
            self.autoscroll_at.get_or_insert_with(Instant::now);
        }
    }
}

/// The file-tree sidebar: the directory model plus its own scroll and the row
/// the pointer last hovered (for highlight). Rows come from [`tree::Tree`].
struct TreePane {
    tree: tree::Tree,
    scroll_y: i32,
    content_height: i32,
    /// The row the pointer last hovered (mouse highlight).
    hovered: Option<usize>,
    /// The keyboard-selected row, moved by Up/Down and acted on by Enter while
    /// the tree holds focus.
    selected: usize,
}

/// Build an editor pane from a saved path, restoring its caret and scroll.
/// Best-effort: an unreadable or missing file yields an empty pane (logged, not
/// fatal) so a moved-away note never stops the editor opening. `set_cursor`
/// clamps the offset into the (possibly changed) file.
fn restore_editor(path: Option<&Path>, saved: &store::PaneSession) -> EditorPane {
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
            present_list: Rc::default(),
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

    /// Bring the Vulkan backend up from the committed dmabuf feedback: a
    /// device matching the compositor's, and a modifier list their common
    /// ground. Called before `create_buffers`, which allocates the exportable
    /// GPU images. A failure here is fatal (the editor requires a GPU).
    fn init_gpu(&mut self) -> Result<()> {
        if self.dmabuf.is_none() {
            return Err(Error::msg("compositor lacks zwp_linux_dmabuf_v1 v4+"));
        }
        let fb = self
            .dmabuf_fb
            .ready()
            .ok_or_else(|| Error::msg("dmabuf feedback never committed"))?;
        let main_device = fb.main_device;
        let compositor_mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if compositor_mods.is_empty() {
            return Err(Error::msg("compositor does not accept XRGB8888 dmabufs"));
        }
        let gpu = vulkan::Gpu::new(Some(main_device))?;
        let mods = gpu.image_modifiers(&compositor_mods);
        if mods.is_empty() {
            return Err(Error::msg(
                "no common DRM modifier between compositor and device",
            ));
        }
        self.gpu_modifiers = mods;
        self.gpu = Some(gpu);
        Ok(())
    }

    /// Negotiate explicit sync (`linux-drm-syncobj-v1`) when a GPU backend is
    /// live and the compositor advertises the manager. Best-effort: on any
    /// failure the GPU path keeps the implicit-sync bridge, which is why this
    /// returns nothing and only logs under stats.
    fn init_explicit_sync(&mut self) {
        let Some(manager) = self.drm_syncobj_manager else {
            return;
        };
        let Some(minor) = self.gpu.as_ref().and_then(vulkan::Gpu::render_node) else {
            return; // No GPU backend, or the device exposes no render node.
        };
        match self.negotiate_explicit_sync(manager, minor) {
            Ok(es) => self.explicit_sync = Some(es),
            Err(e) => {
                if self.stats {
                    eprintln!("bnknotes: explicit sync unavailable, using the sync bridge: {e}");
                }
            }
        }
    }

    /// Create the two client-owned timelines plus a scratch syncobj, share the
    /// timelines with the compositor, and attach explicit sync to the surface.
    fn negotiate_explicit_sync(&mut self, manager: u32, minor: u32) -> Result<ExplicitSync> {
        let drm = ffi::open_drm_render_node(minor)?;
        let fd = drm.as_raw_fd();
        let acquire_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let release_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let scratch = ffi::drm_syncobj_create(fd, false)?;
        let acquire_timeline = self.import_timeline(manager, fd, acquire_syncobj)?;
        let release_timeline = self.import_timeline(manager, fd, release_syncobj)?;
        let surface = self.alloc_id();
        self.conn.request(
            manager,
            wp_linux_drm_syncobj_manager_v1::GET_SURFACE,
            &[Arg::NewId(surface), Arg::Object(self.surface)],
        );
        Ok(ExplicitSync {
            drm,
            acquire_syncobj,
            release_syncobj,
            scratch,
            acquire_timeline,
            release_timeline,
            surface,
            point: 0,
            buffer_release: [None, None],
        })
    }

    /// Export `syncobj` as a DRM syncobj fd and import it into the compositor as
    /// a timeline object; returns the protocol object id. The fd is sent inline,
    /// so the local `OwnedFd` can drop on return.
    fn import_timeline(&mut self, manager: u32, drm: RawFd, syncobj: u32) -> Result<u32> {
        let fd = ffi::drm_syncobj_export_fd(drm, syncobj)?;
        let id = self.alloc_id();
        self.conn.request_with_fd(
            manager,
            wp_linux_drm_syncobj_manager_v1::IMPORT_TIMELINE,
            &[Arg::NewId(id)],
            fd.as_raw_fd(),
        )?;
        Ok(id)
    }

    /// Allocate the two presentable buffers at the current size: one exported
    /// GPU image per swap slot, each wrapped in a wl_buffer via
    /// `zwp_linux_buffer_params_v1` (`create_immed` is sound because every
    /// parameter comes from the compositor's own feedback). A failure (a resize
    /// on a dying device) propagates and ends the process; there is no software
    /// path to fall back to.
    fn create_buffers(&mut self) -> Result<()> {
        let dmabuf_global = self
            .dmabuf
            .ok_or_else(|| Error::msg("gpu buffers without zwp_linux_dmabuf_v1"))?;
        let (w, h) = (self.width, self.height);
        let mut images = Vec::with_capacity(2);
        {
            let Some(gpu) = self.gpu.as_ref() else {
                return Err(Error::msg("gpu buffers without a gpu backend"));
            };
            for _ in 0..2 {
                images.push(gpu.create_image(w, h, &self.gpu_modifiers)?);
            }
        }
        for (i, img) in images.iter().enumerate() {
            let params = self.alloc_id();
            self.conn.request(
                dmabuf_global,
                zwp_linux_dmabuf_v1::CREATE_PARAMS,
                &[Arg::NewId(params)],
            );
            let modifier = img.modifier();
            self.conn.request_with_fd(
                params,
                zwp_linux_buffer_params_v1::ADD,
                &[
                    Arg::Uint(0), // plane index
                    Arg::Uint(img.offset()),
                    Arg::Uint(img.stride()),
                    Arg::Uint((modifier >> 32) as u32),
                    Arg::Uint(modifier as u32),
                ],
                img.fd(),
            )?;
            let buffer = self.alloc_id();
            self.conn.request(
                params,
                zwp_linux_buffer_params_v1::CREATE_IMMED,
                &[
                    Arg::NewId(buffer),
                    Arg::Int(w as i32),
                    Arg::Int(h as i32),
                    Arg::Uint(dmabuf::DRM_FORMAT_XRGB8888),
                    Arg::Uint(0),
                ],
            );
            self.conn
                .request(params, zwp_linux_buffer_params_v1::DESTROY, &[]);
            self.buffers[i] = buffer;
        }
        self.gpu_images = images;
        self.busy = [false, false];
        // Fresh images never carried a release point, so a resize must forget
        // the old buffers' points; their next use starts with no wait.
        if let Some(es) = self.explicit_sync.as_mut() {
            es.buffer_release = [None, None];
        }
        // Fresh images hold no known frame, so forget what was on screen: the
        // screen diff gates whether the next frame presents at all.
        self.present_list = Rc::default();
        Ok(())
    }

    /// Destroy the current buffers and their backing GPU images. The server
    /// recycles the wl_buffer ids via `delete_id` into our free list.
    fn destroy_buffers(&mut self) {
        let buffers = self.buffers;
        for b in buffers {
            if b != 0 {
                self.conn.request(b, wl_buffer::DESTROY, &[]);
            }
        }
        if let Some(gpu) = &self.gpu {
            // Nothing may be in flight while images are destroyed; resize is
            // rare enough that a full drain is the simple correct answer.
            gpu.wait_idle();
            for img in self.gpu_images.drain(..) {
                gpu.destroy_image(img);
            }
        }
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
        // A width change re-wraps and a height change shrinks the viewport; either
        // can push the caret off-screen. Re-reveal the focused pane's caret (this
        // rebuilds its layout at the new column width and re-clamps the scroll).
        self.reveal_active_caret();
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

    /// Build the current frame and present it on the GPU. Returns false if both
    /// buffers are still held by the compositor, leaving `dirty` set so the next
    /// buffer release retries.
    ///
    /// The display list is built and diffed against `present_list` (what is on
    /// screen): an identical frame early-outs (the common idle case), otherwise
    /// the changed screen regions are reported to the compositor as damage. The
    /// per-buffer diff the software path used is gone: the GPU repaints the full
    /// frame every present.
    ///
    /// A rendering failure (a device lost mid-session) propagates and ends the
    /// process; there is no software path to fall back to.
    fn render_frame(&mut self) -> Result<bool> {
        let Some(idx) = (0..2).find(|&i| !self.busy[i]) else {
            return Ok(false);
        };
        let started = self.stats.then(Instant::now);
        let (sw, sh) = (self.width as i32, self.height as i32);
        // Compose the frame from every pane (tree, editors) plus the splitters.
        let new_list = self.build_frame_list();

        // Nothing changed since the on-screen frame: nothing to present (the
        // common idle case).
        let screen_dmg = display::damage(&self.present_list, &new_list, sw, sh);
        if screen_dmg.is_empty() {
            return Ok(true);
        }

        // Render, returning the render-done fence under explicit sync (`None`
        // on the bridge path, or when the driver could not export one). Inline
        // images resolve against both panes' stores, so a picture renders in
        // whichever pane it belongs to.
        let images = [&self.editors[0].images, &self.editors[1].images];
        let render_done = {
            let Some(gpu) = self.gpu.as_mut() else {
                return Err(Error::msg("gpu frame without a gpu backend"));
            };
            let Some(img) = self.gpu_images.get_mut(idx) else {
                return Err(Error::msg("gpu frame without gpu buffers"));
            };
            let known = |key: &gpu::ImageKey| gpu.has_image(key);
            let frame = gpu::build_frame(
                &self.fonts,
                &images,
                &new_list,
                &mut self.gpu_glyphs,
                &known,
            );
            match self.explicit_sync.as_ref() {
                Some(es) => {
                    // Wait on the compositor's release of this buffer's previous
                    // frame; nothing to wait on the first time the slot is used.
                    let release_wait = match es.buffer_release[idx] {
                        Some(p) => Some(ffi::drm_syncobj_point_to_sync_file(
                            es.drm.as_raw_fd(),
                            es.release_syncobj,
                            p,
                            es.scratch,
                        )?),
                        None => None,
                    };
                    gpu.render_list_explicit(img, &frame, color_f32(BACKGROUND), release_wait)?
                }
                None => {
                    gpu.render_list(img, &frame, color_f32(BACKGROUND))?;
                    None
                }
            }
        };
        self.present_list = new_list;

        let buffer = self.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Explicit sync: publish this frame's render-done fence as the acquire
        // point and claim a release point, so the compositor waits before
        // reading and signals when done (the buffer's next reuse waits on it).
        // With no exportable fence the frame was CPU-waited: set no points.
        match (render_done, self.explicit_sync.as_mut()) {
            (Some(fence), Some(es)) => {
                es.point += 1;
                let p = es.point;
                ffi::drm_syncobj_sync_file_to_point(
                    es.drm.as_raw_fd(),
                    fence.as_raw_fd(),
                    es.acquire_syncobj,
                    p,
                    es.scratch,
                )?;
                es.buffer_release[idx] = Some(p);
                let (hi, lo) = ((p >> 32) as u32, p as u32);
                let (sync_surface, acquire_tl, release_tl) =
                    (es.surface, es.acquire_timeline, es.release_timeline);
                self.conn.request(
                    sync_surface,
                    wp_linux_drm_syncobj_surface_v1::SET_ACQUIRE_POINT,
                    &[Arg::Object(acquire_tl), Arg::Uint(hi), Arg::Uint(lo)],
                );
                self.conn.request(
                    sync_surface,
                    wp_linux_drm_syncobj_surface_v1::SET_RELEASE_POINT,
                    &[Arg::Object(release_tl), Arg::Uint(hi), Arg::Uint(lo)],
                );
                self.explicit_fence_frames += 1;
            }
            (None, Some(es)) => {
                es.buffer_release[idx] = None;
                self.explicit_cpu_wait_frames += 1;
            }
            _ => {}
        }
        // The GPU repainted everything, but the *screen* change is still just
        // the diff, so report those regions and let the compositor
        // recomposite only what moved.
        for r in &screen_dmg {
            self.conn.request(
                self.surface,
                wl_surface::DAMAGE,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
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
            let sync = if self.explicit_sync.is_some() {
                format!(
                    "explicit-sync {} fence/{} cpu-wait",
                    self.explicit_fence_frames, self.explicit_cpu_wait_frames
                )
            } else {
                "implicit-sync bridge".to_string()
            };
            eprintln!(
                "[stats] gpu frame {} {:.2}ms build+submit, {:.1}ms since last | glyphs hit {} miss {} ({:.1}% of {}) | {}",
                self.frame_count,
                started.elapsed().as_secs_f64() * 1000.0,
                gap,
                cache.hits,
                cache.misses,
                cache.hit_rate() * 100.0,
                cache.lookups(),
                sync,
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

    /// Ask the dmabuf global for its default feedback. The events accumulate in
    /// `dmabuf_fb` as they arrive; nothing waits on them.
    fn request_dmabuf_feedback(&mut self) {
        let Some(dmabuf) = self.dmabuf else {
            return;
        };
        let id = self.alloc_id();
        self.conn.request(
            dmabuf,
            zwp_linux_dmabuf_v1::GET_DEFAULT_FEEDBACK,
            &[Arg::NewId(id)],
        );
        self.dmabuf_feedback = id;
    }

    /// One event of a feedback burst; `dmabuf_fb` accumulates and commits.
    fn on_dmabuf_feedback(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            zwp_linux_dmabuf_feedback_v1::EV_FORMAT_TABLE => {
                let size = r.u32()? as usize;
                let fd = self.conn.take_fd().ok_or_else(|| {
                    Error::msg("dmabuf feedback format_table arrived without its fd")
                })?;
                if size == 0 {
                    self.dmabuf_fb.set_table(&[]);
                    return Ok(());
                }
                let ptr = ffi::mmap_ro(fd.as_raw_fd(), size)?;
                // SAFETY: ptr maps `size` readable bytes from the table fd; the
                // bytes are copied out before the mapping is released.
                let bytes = unsafe { core::slice::from_raw_parts(ptr, size) }.to_vec();
                // SAFETY: ptr/size came from mmap_ro and are not used again.
                unsafe { ffi::unmap(ptr, size) };
                self.dmabuf_fb.set_table(&bytes);
            }
            zwp_linux_dmabuf_feedback_v1::EV_MAIN_DEVICE => {
                self.dmabuf_fb.main_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_TARGET_DEVICE => {
                self.dmabuf_fb.tranche_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FORMATS => {
                self.dmabuf_fb.tranche_formats(r.array()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FLAGS => {
                self.dmabuf_fb.tranche_flags(r.u32()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_DONE => self.dmabuf_fb.tranche_done(),
            zwp_linux_dmabuf_feedback_v1::EV_DONE => self.dmabuf_fb.done(),
            _ => {}
        }
        Ok(())
    }

    /// The `--gpu-probe` body: bind the registry, fetch dmabuf feedback, and
    /// print what the GPU presentation path has to work with.
    fn probe_dmabuf(&mut self) -> Result<()> {
        self.registry = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::GET_REGISTRY,
            &[Arg::NewId(self.registry)],
        );
        self.roundtrip()?;
        if self.dmabuf.is_none() {
            println!(
                "gpu-probe: compositor lacks zwp_linux_dmabuf_v1 v{}+; \
                 the editor cannot run here (it requires the GPU path)",
                protocol::VERSION_DMABUF
            );
            return Ok(());
        }
        println!(
            "gpu-probe: zwp_linux_dmabuf_v1 bound at v{}",
            self.dmabuf_version
        );
        match self.drm_syncobj_manager {
            Some(_) => println!(
                "gpu-probe: {} present: explicit sync available",
                protocol::IFACE_DRM_SYNCOBJ
            ),
            None => println!(
                "gpu-probe: {} absent: GPU path uses the implicit-sync bridge",
                protocol::IFACE_DRM_SYNCOBJ
            ),
        }
        // The server answers get_default_feedback before the sync that follows
        // it, so one roundtrip delivers the whole feedback burst.
        self.request_dmabuf_feedback();
        self.roundtrip()?;
        let Some(fb) = self.dmabuf_fb.ready() else {
            return Err(Error::msg(
                "compositor never committed dmabuf feedback (no done event)",
            ));
        };
        println!(
            "gpu-probe: format table: {} format+modifier pairs",
            self.dmabuf_fb.table_len()
        );
        println!(
            "gpu-probe: main device: {}:{}",
            dmabuf::dev_major(fb.main_device),
            dmabuf::dev_minor(fb.main_device)
        );
        for (i, t) in fb.tranches.iter().enumerate() {
            println!(
                "gpu-probe: tranche {i}: device {}:{}, flags {:#x}, {} pairs",
                dmabuf::dev_major(t.device),
                dmabuf::dev_minor(t.device),
                t.flags,
                t.formats.len()
            );
        }
        let mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if mods.is_empty() {
            println!("gpu-probe: XRGB8888 is not offered; the GPU path needs it");
            return Ok(());
        }
        let rendered: Vec<String> = mods
            .iter()
            .map(|&m| {
                if m == dmabuf::MOD_LINEAR {
                    "linear".to_string()
                } else {
                    format!("{m:#x}")
                }
            })
            .collect();
        println!("gpu-probe: XRGB8888 modifiers: {}", rendered.join(", "));

        // The Vulkan half: can a device matching the compositor's be brought
        // up with the extensions the presentation path needs, and can it
        // actually allocate and export a presentable image? A failure here is
        // the probe's answer, not an error.
        let main_device = fb.main_device;
        let gpu = match vulkan::Gpu::new(Some(main_device)) {
            Ok(gpu) => gpu,
            Err(e) => {
                println!("gpu-probe: vulkan unavailable: {e}");
                return Ok(());
            }
        };
        println!(
            "gpu-probe: vulkan device: {} (graphics queue family {})",
            gpu.name(),
            gpu.queue_family()
        );
        let common = gpu.image_modifiers(&mods);
        if common.is_empty() {
            println!("gpu-probe: no common DRM modifier; the GPU path is unavailable");
            return Ok(());
        }
        println!(
            "gpu-probe: common modifiers: {} of {}",
            common.len(),
            mods.len()
        );
        match gpu.create_image(64, 64, &common) {
            Ok(img) => {
                println!(
                    "gpu-probe: test image exported: modifier {:#x}, stride {}, offset {}",
                    img.modifier(),
                    img.stride(),
                    img.offset()
                );
                gpu.destroy_image(img);
                println!("gpu-probe: ready (the editor would use the GPU path)");
            }
            Err(e) => println!("gpu-probe: image export failed: {e}"),
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
        // Window-level pane management, handled before any editor routing: toggle
        // the sidebar/second pane, or move focus between regions.
        match action {
            KeyAction::ToggleTree => {
                self.panes.tree_visible = !self.panes.tree_visible;
                if !self.panes.tree_visible && self.focus == Focus::Tree {
                    self.focus = Focus::Editor(self.last_editor);
                }
                self.dirty = true;
                self.save_session();
                return Ok(());
            }
            KeyAction::ToggleSecondPane => {
                self.panes.second_open = !self.panes.second_open;
                if !self.panes.second_open && self.focus == Focus::Editor(1) {
                    self.focus = Focus::Editor(0);
                    self.last_editor = 0;
                }
                self.refresh_title();
                self.dirty = true;
                self.save_session();
                return Ok(());
            }
            KeyAction::FocusTree => {
                if self.panes.tree_visible {
                    self.focus = Focus::Tree;
                    self.dirty = true;
                }
                return Ok(());
            }
            KeyAction::FocusPaneA => {
                self.focus = Focus::Editor(0);
                self.last_editor = 0;
                self.refresh_title();
                self.dirty = true;
                return Ok(());
            }
            KeyAction::FocusPaneB => {
                if self.panes.second_open {
                    self.focus = Focus::Editor(1);
                    self.last_editor = 1;
                    self.refresh_title();
                    self.dirty = true;
                }
                return Ok(());
            }
            _ => {}
        }
        // With the tree focused, keys navigate the sidebar (the pane keys above
        // still move focus out); nothing reaches an editor.
        if self.focus == Focus::Tree {
            self.tree_key(action);
            return Ok(());
        }
        // Any keyboard action but Copy drops an image selection and its pinned raw
        // source (using the keyboard means the user has left the image widget). A
        // revealed source the caret sits in stays shown via the cursor-on-line
        // reveal. Copy must see the selection to choose image-vs-text, so it is
        // exempt. Repaint so the border and button disappear immediately.
        // Keyboard input goes to the focused editor; `rect` is its column, for
        // the motion methods that need the viewport.
        let i = self.active_editor();
        let rect = self.editor_rect(i);
        if !matches!(action, KeyAction::Copy)
            && (self.editors[i].selected_image.is_some() || self.editors[i].raw_image.is_some())
        {
            self.editors[i].selected_image = None;
            self.editors[i].raw_image = None;
            self.dirty = true;
        }
        let extend = self.xkb.shift_active();
        // With the find bar open the keyboard drives the query, not the buffer,
        // so a stray shortcut cannot edit the document mid-search (replacements
        // go through the bar's own Enter handling).
        if self.editors[i].find.is_open() {
            self.handle_find_key(action);
            return Ok(());
        }
        if matches!(action, KeyAction::Find | KeyAction::Replace) {
            let text = self.editors[i].text.text();
            let cursor = self.editors[i].text.cursor();
            let target = if action == KeyAction::Find {
                self.editors[i].find.open_at(cursor, &text)
            } else {
                self.editors[i].find.open_replace_at(cursor, &text)
            };
            self.land_on_match(target);
            return Ok(());
        }
        let ed = &mut self.editors[i];
        let changed = match action {
            KeyAction::Char(c) => {
                let mut encoded = [0u8; 4];
                ed.text.insert_str(c.encode_utf8(&mut encoded));
                ed.goal_x = None;
                true
            }
            KeyAction::Enter => {
                ed.goal_x = None;
                ed.text.enter()
            }
            // On a list item, Tab nests it one level deeper and Shift+Tab one
            // shallower. Anywhere else, Tab inserts a soft tab and Shift+Tab is a
            // no-op (there is nothing to dedent).
            KeyAction::Tab => {
                ed.goal_x = None;
                if ed.text.cursor_on_list_item() {
                    ed.text.nest_list_item(extend)
                } else if extend {
                    false
                } else {
                    ed.text.insert_str(SOFT_TAB);
                    true
                }
            }
            KeyAction::Backspace => {
                ed.goal_x = None;
                ed.text.backspace()
            }
            KeyAction::Delete => {
                ed.goal_x = None;
                ed.text.delete_forward()
            }
            KeyAction::DeleteWordBack => {
                ed.goal_x = None;
                ed.text.delete_word_back()
            }
            KeyAction::DeleteWordForward => {
                ed.goal_x = None;
                ed.text.delete_word_forward()
            }
            KeyAction::DeleteLine => {
                ed.goal_x = None;
                ed.text.delete_line()
            }
            KeyAction::Left => {
                ed.goal_x = None;
                ed.text.move_left(extend)
            }
            KeyAction::Right => {
                ed.goal_x = None;
                ed.text.move_right(extend)
            }
            KeyAction::WordLeft => {
                ed.goal_x = None;
                ed.text.move_word_left(extend)
            }
            KeyAction::WordRight => {
                ed.goal_x = None;
                ed.text.move_word_right(extend)
            }
            KeyAction::Up => ed.move_vertical(&self.fonts, rect, -1, extend),
            KeyAction::Down => ed.move_vertical(&self.fonts, rect, 1, extend),
            KeyAction::PageUp => ed.move_page(&self.fonts, rect, -1, extend),
            KeyAction::PageDown => ed.move_page(&self.fonts, rect, 1, extend),
            KeyAction::Home => {
                ed.goal_x = None;
                ed.move_line_edge(&self.fonts, rect, false, extend)
            }
            KeyAction::End => {
                ed.goal_x = None;
                ed.move_line_edge(&self.fonts, rect, true, extend)
            }
            KeyAction::DocStart => {
                ed.goal_x = None;
                ed.text.move_doc_start(extend)
            }
            KeyAction::DocEnd => {
                ed.goal_x = None;
                ed.text.move_doc_end(extend)
            }
            KeyAction::SelectAll => {
                ed.text.select_all();
                true
            }
            KeyAction::Undo => {
                ed.goal_x = None;
                ed.text.undo()
            }
            KeyAction::Redo => {
                ed.goal_x = None;
                ed.text.redo()
            }
            // The clipboard actions manage their own redraw and may do I/O over a
            // pipe to another app. As with Save, an I/O failure (e.g. a peer that
            // closes the pipe early, EPIPE) must not bring the editor down and lose
            // the user's unsaved work, so it is reported and swallowed rather than
            // propagated out of the loop. With an image selected, Copy puts the
            // picture on the clipboard instead of text.
            KeyAction::Copy => {
                let result = if self.editors[i].selected_image.is_some() {
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
            // Handled before this match: Find/Replace open the bar, and while
            // it is open the bar consumes them (and Escape, to close itself).
            KeyAction::Find | KeyAction::Replace => false,
            // With no find bar, Escape collapses a selection to a bare cursor;
            // with nothing selected there is nothing to leave.
            KeyAction::Escape => {
                if self.editors[i].text.selection().is_some() {
                    let c = self.editors[i].text.cursor();
                    self.editors[i].text.set_cursor(c, false);
                    true
                } else {
                    false
                }
            }
            // Handled before this match (returned early); listed for exhaustiveness.
            KeyAction::ToggleTree
            | KeyAction::ToggleSecondPane
            | KeyAction::FocusTree
            | KeyAction::FocusPaneA
            | KeyAction::FocusPaneB
            | KeyAction::None => false,
        };
        if changed {
            self.after_edit();
        }
        Ok(())
    }

    /// One key while the find bar is open: characters and Backspace edit the
    /// query, Enter steps to the next match (Shift+Enter to the previous), and
    /// Escape or Ctrl+F closes the bar, keeping the current match selected.
    /// Anything else is ignored. Whenever the state machine yields a match, it
    /// becomes the buffer selection and is scrolled into view.
    /// One key while the find bar is open. Characters and Backspace edit the
    /// focused field (Alt+W / Alt+R toggle whole-word and regex instead), Tab
    /// moves between the query and replace fields, Ctrl+V pastes into the
    /// focused field, and Enter navigates or replaces: next match from the
    /// query field (Shift+Enter previous), replace-current-and-advance from
    /// the replace field, replace-all with Ctrl. Escape or Ctrl+F closes the
    /// bar, keeping the current match selected. Anything else is ignored.
    fn handle_find_key(&mut self, action: KeyAction) {
        let shift = self.xkb.shift_active();
        let ctrl = self.xkb.ctrl_active();
        let alt = self.xkb.alt_active();
        // The find bar belongs to the focused editor.
        let i = self.active_editor();
        let target = match action {
            KeyAction::Char(c) if alt => {
                let text = self.editors[i].text.text();
                match c.to_ascii_lowercase() {
                    'w' => self.editors[i].find.toggle_word(&text),
                    'r' => self.editors[i].find.toggle_regex(&text),
                    _ => return,
                }
            }
            KeyAction::Char(c) => {
                let text = self.editors[i].text.text();
                self.editors[i].find.push(c, &text)
            }
            KeyAction::Backspace => {
                let text = self.editors[i].text.text();
                self.editors[i].find.pop(&text)
            }
            KeyAction::Tab => {
                self.editors[i].find.toggle_focus();
                None
            }
            KeyAction::SelectAll => {
                self.editors[i].find.select_all();
                None
            }
            KeyAction::Paste => match self.clipboard_text() {
                Ok(Some(pasted)) => {
                    let text = self.editors[i].text.text();
                    self.editors[i].find.push_str(&pasted, &text)
                }
                Ok(None) => None,
                Err(e) => {
                    eprintln!("bnknotes: paste failed: {e}");
                    None
                }
            },
            KeyAction::Enter if ctrl => {
                self.replace_all_matches();
                return;
            }
            KeyAction::Enter if shift => self.editors[i].find.prev(),
            KeyAction::Enter
                if self.editors[i].find.show_replace()
                    && self.editors[i].find.focus() == find::Field::Replace =>
            {
                self.replace_current_match();
                return;
            }
            KeyAction::Enter => self.editors[i].find.next(),
            KeyAction::Replace => {
                let text = self.editors[i].text.text();
                let cursor = self.editors[i].text.cursor();
                self.editors[i].find.open_replace_at(cursor, &text)
            }
            KeyAction::Escape | KeyAction::Find => {
                self.editors[i].find.close();
                self.dirty = true;
                return;
            }
            _ => return,
        };
        self.land_on_match(target);
    }

    /// Select `target` (a match's byte range) and bring it on screen; with no
    /// target, just repaint so the bar reflects its state.
    fn land_on_match(&mut self, target: Option<(usize, usize)>) {
        if let Some((start, end)) = target {
            let i = self.active_editor();
            self.editors[i].text.select_range(start, end);
            self.bump_caret();
            self.reveal_active_caret();
        }
        self.dirty = true;
    }

    /// Replace the current match with the bar's replacement and advance to the
    /// next match, rescanning from just past the inserted text so a
    /// replacement containing the query is not immediately re-matched.
    fn replace_current_match(&mut self) {
        let i = self.active_editor();
        let Some((start, end)) = self.editors[i].find.current_match() else {
            self.dirty = true;
            return;
        };
        let with = self.editors[i].find.replacement().to_string();
        self.editors[i].text.select_range(start, end);
        self.editors[i].text.insert_str(&with);
        self.after_edit();
        let text = self.editors[i].text.text();
        let target = self.editors[i].find.refresh(&text, start + with.len());
        self.land_on_match(target);
    }

    /// Replace every match with the bar's replacement in one undo step.
    fn replace_all_matches(&mut self) {
        let i = self.active_editor();
        let ranges = self.editors[i].find.matches().to_vec();
        let with = self.editors[i].find.replacement().to_string();
        if !self.editors[i].text.replace_ranges(&ranges, &with) {
            return;
        }
        self.after_edit();
        let text = self.editors[i].text.text();
        let cursor = self.editors[i].text.cursor();
        let target = self.editors[i].find.refresh(&text, cursor);
        self.land_on_match(target);
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
                let (px, py) = (self.pointer_x, self.pointer_y);
                if let Some(s) = self.drag_splitter {
                    self.drag_splitter_to(s, px);
                } else {
                    let i = self.active_editor();
                    if matches!(self.focus, Focus::Editor(_)) && self.editors[i].selecting {
                        // Drag extends the focused editor's selection from the press
                        // point, arming auto-scroll at the column's vertical edges.
                        let rect = self.editor_rect(i);
                        if self.editors[i].place_cursor_at(&self.fonts, rect, px, py, true) {
                            self.dirty = true;
                        }
                        self.bump_caret();
                        self.editors[i].update_autoscroll(rect, py);
                    } else {
                        self.update_cursor_shape();
                        self.update_tree_hover();
                    }
                }
            }
            wl_pointer::EV_BUTTON => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let button = r.u32()?;
                let state = r.u32()?;
                if button == wl_pointer::BTN_LEFT {
                    if state == wl_pointer::BUTTON_STATE_PRESSED {
                        self.on_left_press();
                    } else {
                        // A splitter drag just ended: persist the new column sizes.
                        if self.drag_splitter.take().is_some() {
                            self.save_session();
                        }
                        let i = self.active_editor();
                        self.editors[i].selecting = false;
                        self.editors[i].autoscroll = 0;
                        self.editors[i].autoscroll_at = None;
                        // Drag over: let the reveal catch up to the final selection.
                        if self.editors[i].frozen_reveal.take().is_some() {
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
                    let delta = (value * SCROLL_FACTOR).round() as i32;
                    let (px, py) = (self.pointer_x, self.pointer_y);
                    // Scroll whichever pane the pointer is over, not just the focused one.
                    match self.hit_test(px, py) {
                        Hit::Editor(i, rect) => {
                            if self.editors[i].scroll_by(&self.fonts, rect, delta) {
                                self.dirty = true;
                            }
                        }
                        Hit::Tree(rect) => self.scroll_tree(rect, delta),
                        _ => {}
                    }
                }
            }
            _ => {} // leave, frame, axis_source, and friends
        }
        Ok(())
    }

    /// What the window position `(px, py)` lands on. Splitters win over columns
    /// since they sit in the gaps; columns are full height, so only x decides.
    fn hit_test(&self, px: f32, py: f32) -> Hit {
        let _ = py;
        let x = px as i32;
        let cols = self.columns();
        for (s, rect) in cols.splitters.iter().enumerate() {
            if let Some(r) = rect {
                if x >= r.x && x < r.x + r.w {
                    return Hit::Splitter(s);
                }
            }
        }
        if let Some(r) = cols.tree {
            if x >= r.x && x < r.x + r.w {
                return Hit::Tree(r);
            }
        }
        for (i, rect) in cols.editors.iter().enumerate() {
            if let Some(r) = rect {
                if x >= r.x && x < r.x + r.w {
                    return Hit::Editor(i, *r);
                }
            }
        }
        Hit::None
    }

    /// Drag splitter `s` so its column edge follows window x `px`: splitter 0
    /// resizes the sidebar, splitter 1 the A/B content split. Clamped to keep
    /// every column usable.
    fn drag_splitter_to(&mut self, s: usize, px: f32) {
        let x = px as i32;
        let w = self.width as i32;
        if s == 0 {
            self.panes.tree_width = x.clamp(120, (w - 2 * SPLITTER - 120).max(120));
        } else {
            let base = if self.panes.tree_visible {
                self.panes.tree_width + SPLITTER
            } else {
                0
            };
            let content = (w - base - SPLITTER).max(1) as f32;
            let a = (x - base) as f32 / content;
            self.panes.split_ratio = a.clamp(0.1, 0.9);
        }
        self.dirty = true;
    }

    /// Update the tree's hovered row from the pointer, repainting if it changed.
    fn update_tree_hover(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        let hovered = match self.hit_test(px, py) {
            Hit::Tree(rect) => self.tree_row_at(rect, py),
            _ => None,
        };
        if hovered != self.tree.hovered {
            self.tree.hovered = hovered;
            self.dirty = true;
        }
    }

    /// The tree row index at window y `py` within the tree column, if any.
    fn tree_row_at(&self, rect: Rect, py: f32) -> Option<usize> {
        let row_h = self.fonts.metrics(FONT_SIZE).line_height.max(1);
        let y = py as i32 - rect.y + self.tree.scroll_y - TREE_PAD;
        if y < 0 {
            return None;
        }
        let idx = (y / row_h) as usize;
        (idx < self.tree.tree.rows().len()).then_some(idx)
    }

    /// Scroll the tree column by `delta`, clamped, repainting if it moved.
    fn scroll_tree(&mut self, rect: Rect, delta: i32) {
        let sy = scroll::clamp(self.tree.scroll_y + delta, self.tree.content_height, rect.h);
        if sy != self.tree.scroll_y {
            self.tree.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// Open `path` into the last-focused content pane and focus it. An unreadable
    /// file is reported without disturbing the current pane.
    fn open_file(&mut self, path: &Path) {
        match EditorPane::open(path) {
            Ok(pane) => {
                let i = self.last_editor;
                self.editors[i] = pane;
                self.focus = Focus::Editor(i);
                self.refresh_title();
                self.dirty = true;
                self.save_session();
            }
            Err(e) => eprintln!("bnknotes: cannot open {}: {e}", path.display()),
        }
    }

    /// The tree column rectangle, or an empty rect when the sidebar is hidden.
    fn tree_rect(&self) -> Rect {
        self.columns().tree.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        })
    }

    /// Scroll the tree so its selected row is on screen.
    fn reveal_tree_selection(&mut self) {
        let row_h = self.fonts.metrics(FONT_SIZE).line_height.max(1);
        let rect = self.tree_rect();
        let top = TREE_PAD + self.tree.selected as i32 * row_h;
        self.tree.scroll_y = scroll::reveal(
            self.tree.scroll_y,
            top,
            row_h,
            rect.h,
            self.tree.content_height,
            TREE_PAD,
        );
    }

    /// Keyboard navigation while the tree holds focus: Up/Down move the selected
    /// row, Left/Right collapse/expand a folder, Enter toggles a folder or opens
    /// a file into the last-focused editor.
    fn tree_key(&mut self, action: KeyAction) {
        let n = self.tree.tree.rows().len();
        if n == 0 {
            return;
        }
        let last = n - 1;
        let sel = self.tree.selected.min(last);
        self.tree.selected = sel;
        match action {
            KeyAction::Up => self.tree.selected = sel.saturating_sub(1),
            KeyAction::Down => self.tree.selected = (sel + 1).min(last),
            KeyAction::Left | KeyAction::Right | KeyAction::Enter => {
                let row = &self.tree.tree.rows()[sel];
                let (is_dir, expanded, path) = (row.is_dir, row.expanded, row.path.clone());
                if matches!(action, KeyAction::Enter) && !is_dir {
                    self.open_file(&path);
                    return;
                }
                let toggle = match action {
                    KeyAction::Left => is_dir && expanded,
                    KeyAction::Right => is_dir && !expanded,
                    KeyAction::Enter => is_dir,
                    _ => false,
                };
                if !toggle {
                    return;
                }
                self.tree.tree.toggle(&path);
                self.save_session();
            }
            _ => return,
        }
        self.dirty = true;
        self.reveal_tree_selection();
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
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_EW_RESIZE};
        let (px, py) = (self.pointer_x, self.pointer_y);
        let shape = match self.hit_test(px, py) {
            Hit::Splitter(_) => SHAPE_EW_RESIZE,
            Hit::Tree(_) => SHAPE_DEFAULT,
            Hit::Editor(i, rect) => self.editors[i].cursor_shape_at(&self.fonts, rect, px, py),
            Hit::None => SHAPE_DEFAULT,
        };
        if shape != self.cursor_shape {
            self.cursor_shape = shape;
            self.set_cursor_shape(shape);
        }
    }

    /// Handle a left-button press: route it to whatever it landed on. A splitter
    /// starts a drag-resize, a tree row expands a folder or opens a file, and an
    /// editor column takes focus and then handles the click as a text press.
    fn on_left_press(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        match self.hit_test(px, py) {
            Hit::Splitter(s) => self.drag_splitter = Some(s),
            Hit::Tree(rect) => self.tree_press(rect, py),
            Hit::Editor(i, rect) => {
                self.focus = Focus::Editor(i);
                self.last_editor = i;
                self.editor_press(i, rect, px, py);
            }
            Hit::None => {}
        }
    }

    /// A press in the tree: focus the sidebar, then toggle a folder's expansion
    /// or open a file into the last-focused editor.
    fn tree_press(&mut self, rect: Rect, py: f32) {
        self.focus = Focus::Tree;
        let Some(idx) = self.tree_row_at(rect, py) else {
            self.dirty = true;
            return;
        };
        let Some(row) = self.tree.tree.rows().get(idx) else {
            return;
        };
        let path = row.path.clone();
        if row.is_dir {
            self.tree.tree.toggle(&path);
            self.dirty = true;
            self.save_session();
        } else {
            self.open_file(&path);
        }
    }

    /// A text press in editor `i`'s column: chain the click count and act on it,
    /// after handling image/checkbox/rule/link interactions. Routed from
    /// [`Self::on_left_press`] once the pane is chosen and focused.
    fn editor_press(&mut self, i: usize, rect: Rect, px: f32, py: f32) {
        let shift = self.xkb.shift_active();
        // A click takes focus back from this editor's find bar.
        if self.editors[i].find.is_open() {
            self.editors[i].find.close();
            self.dirty = true;
        }
        self.editors[i].goal_x = None;
        // Image interactions win over text.
        if self.editors[i].image_press(&self.fonts, rect, px, py) {
            self.dirty = true;
            self.bump_caret();
            return;
        }
        self.editors[i].selected_image = None;
        self.editors[i].raw_image = None;
        // A plain click on a rendered task checkbox toggles it without moving the
        // caret. Shift-click falls through so it can extend a selection.
        if !shift {
            if let Some(ls) = self.editors[i].checkbox_under_pointer(&self.fonts, rect, px, py) {
                if self.editors[i].text.toggle_task(ls) {
                    self.dirty = true;
                }
                self.bump_caret();
                return;
            }
        }
        // Click chaining for double/triple-click.
        let now = Instant::now();
        let (lx, ly) = self.editors[i].last_click_pos;
        let near = (px - lx).abs() <= MULTI_CLICK_RADIUS && (py - ly).abs() <= MULTI_CLICK_RADIUS;
        let chained = near
            && self.editors[i]
                .last_click
                .is_some_and(|t| now.saturating_duration_since(t) <= MULTI_CLICK_INTERVAL);
        self.editors[i].click_count = if chained {
            self.editors[i].click_count + 1
        } else {
            1
        };
        self.editors[i].last_click = Some(now);
        self.editors[i].last_click_pos = (px, py);
        // A plain click on a rendered horizontal rule selects the whole rule line.
        if !shift {
            self.editors[i].ensure_layout(&self.fonts, rect);
            let cy = self.editors[i].content_y(rect, py);
            let rule = self.editors[i]
                .layout_cache
                .as_ref()
                .and_then(|l| l.rule_at(&self.fonts, cy));
            if let Some((start, end)) = rule {
                self.editors[i].text.select_range(start, end);
                self.editors[i].selecting = false;
                self.dirty = true;
                self.bump_caret();
                return;
            }
        }
        // A plain, un-chained click on a rendered link follows it.
        if !chained && !shift {
            if let Some(dest) = self.editors[i].link_under_pointer(&self.fonts, rect, px, py) {
                if crate::browser::open_url(&dest).is_ok() {
                    self.editors[i].selecting = false;
                    return;
                }
            }
        }
        match self.editors[i].click_count {
            1 => {
                self.editors[i].place_cursor_at(&self.fonts, rect, px, py, shift);
                self.editors[i].selecting = true;
                // Pin marker reveal to the press point for the whole drag.
                let c = self.editors[i].text.cursor();
                let reveal = self.editors[i].text.selection().unwrap_or((c, c));
                self.editors[i].frozen_reveal = Some(reveal);
            }
            2 => {
                let offset = self.editors[i].pointer_offset(&self.fonts, rect, px, py);
                self.editors[i].text.select_word_at(offset);
                self.dirty = true;
            }
            _ => {
                let offset = self.editors[i].pointer_offset(&self.fonts, rect, px, py);
                self.editors[i].text.select_line_at(offset);
                self.dirty = true;
            }
        }
        self.bump_caret();
    }

    /// While drag-selecting with the pointer held in a column's edge zone, scroll
    /// a step and re-extend the focused editor's selection to the (stationary)
    /// pointer, on the auto-scroll interval. A no-op when not auto-scrolling.
    fn fire_autoscroll(&mut self) {
        let i = self.active_editor();
        if self.editors[i].autoscroll == 0 || !self.editors[i].selecting {
            self.editors[i].autoscroll = 0;
            self.editors[i].autoscroll_at = None;
            return;
        }
        let Some(at) = self.editors[i].autoscroll_at else {
            return;
        };
        let now = Instant::now();
        if at > now {
            return;
        }
        let rect = self.editor_rect(i);
        let step = self.editors[i].autoscroll;
        let (px, py) = (self.pointer_x, self.pointer_y);
        if self.editors[i].scroll_by(&self.fonts, rect, step) {
            self.dirty = true;
        }
        if self.editors[i].place_cursor_at(&self.fonts, rect, px, py, true) {
            self.dirty = true;
        }
        self.editors[i].autoscroll_at = Some(now + AUTOSCROLL_INTERVAL);
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

    /// Put the focused editor's selection on the clipboard. No-op without one.
    fn copy_selection(&mut self) -> Result<()> {
        let i = self.active_editor();
        if let Some(text) = self.editors[i].text.selected_text() {
            self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        }
        Ok(())
    }

    /// Copy the focused editor's selection, then delete it.
    fn cut_selection(&mut self) -> Result<()> {
        let i = self.active_editor();
        let Some(text) = self.editors[i].text.selected_text() else {
            return Ok(());
        };
        self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        if self.editors[i].text.backspace() {
            self.after_edit();
        }
        Ok(())
    }

    /// Put the selected image's file bytes on the clipboard, advertised as its real
    /// type (`image/png`/`image/jpeg`), so it can be pasted into other apps. No-op
    /// if nothing is selected or the file can no longer be read.
    fn copy_selected_image(&mut self) -> Result<()> {
        let i = self.active_editor();
        let Some(ls) = self.editors[i].selected_image else {
            return Ok(());
        };
        let Some(raw) = self.image_path_at(i, ls) else {
            return Ok(());
        };
        if let Some((bytes, mime)) = self.editors[i].images.read_source(&raw) {
            self.set_clipboard(bytes, vec![mime]);
        }
        Ok(())
    }

    /// The raw destination path of the image whose logical line starts at `ls` in
    /// editor `i`, or `None` if that line is no longer a standalone image.
    fn image_path_at(&self, i: usize, ls: usize) -> Option<String> {
        let rope = self.editors[i].text.rope();
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
        if let Some(text) = self.clipboard_text()? {
            self.insert_clipboard(&text);
        }
        Ok(())
    }

    /// The clipboard's text, if it holds any. If we own the clipboard, our own
    /// data is returned directly: going through the compositor would deadlock,
    /// blocking on the pipe while also needing to service our own source's
    /// send event. Otherwise the data is received through a pipe (hand the
    /// write end to the compositor, close our copy, read the read end to EOF).
    /// A copied image has no text representation and yields `None`.
    fn clipboard_text(&mut self) -> Result<Option<String>> {
        if self.clipboard_source != 0 {
            if self.clipboard_mimes.iter().any(|m| is_text_mime(m)) {
                return Ok(Some(
                    String::from_utf8_lossy(&self.clipboard_data).into_owned(),
                ));
            }
            return Ok(None);
        }
        let (Some(mime), true) = (self.selection_text_mime.clone(), self.selection_offer != 0)
        else {
            return Ok(None);
        };
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            self.selection_offer,
            wl_data_offer::RECEIVE,
            &[Arg::Str(&mime)],
            write_end.as_raw_fd(),
        )?;
        drop(write_end);
        let bytes = ffi::read_to_end(read_end.as_raw_fd())?;
        Ok(String::from_utf8(bytes).ok())
    }

    fn insert_clipboard(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        // Pasted text can carry CRLF or lone CR (from another app or platform);
        // the buffer only ever holds '\n' (file load normalizes the same way), so
        // normalize here too rather than let a stray '\r' into the rope.
        let text = file::normalize_newlines(text);
        let i = self.active_editor();
        self.editors[i].text.insert_str(&text);
        self.dirty = true;
        self.reveal_active_caret();
    }

    /// Scroll the viewport by `delta` pixels (positive = down), clamped to the
    /// document. The hottest input path (one call per wheel event), so it reads
    /// the cached content height instead of cloning the layout.
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

/// A 0x00RRGGBB color as the linear-float RGBA a Vulkan clear expects. The
/// bytes are the display (sRGB-encoded) values as authored, passed through as
/// the same numbers, not a color-space conversion.
fn color_f32(color: u32) -> [f32; 4] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
        1.0,
    ]
}

/// The per-frame inputs [`build_display_list`] needs beyond the layout itself:
/// the surface size, the scroll offset, the live selection and cursor, and
/// whether the caret is on this blink. Bundled so the build signature stays
/// readable and so the perf harness can drive a build without a live surface.
pub(crate) struct PaintParams<'a> {
    pub width: usize,
    pub height: usize,
    pub scroll_y: i32,
    pub selection: Option<(usize, usize)>,
    pub cursor: usize,
    pub caret_visible: bool,
    /// The find bar's paint state while it is open; `None` draws no bar and no
    /// match highlights.
    pub find: Option<FindPaint<'a>>,
}

/// What the painter needs from an open find bar, borrowed from the app's
/// [`FindState`] for the frame being built.
pub(crate) struct FindPaint<'a> {
    /// The query as typed; the bar shows the tail that fits.
    pub query: &'a str,
    /// The replacement text when the replace row is shown.
    pub replace: Option<&'a str>,
    /// Whether the replace field (rather than the query) has the keyboard.
    pub focus_replace: bool,
    /// The focused field's whole content is selected (Ctrl+A or a reopened
    /// query); drawn on a selection fill since the next edit replaces it.
    pub selected: bool,
    /// Mode flags, drawn bright when on.
    pub word: bool,
    pub regex: bool,
    /// The query is an invalid regular expression.
    pub error: bool,
    /// Byte ranges of every match, in document order.
    pub matches: &'a [(usize, usize)],
    /// Index into `matches` of the current match. It is not highlighted here
    /// because the app mirrors it into the buffer selection.
    pub current: usize,
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
/// The painter's view of an open find bar, or `None` when it is closed.
fn build_find_paint(find: &FindState) -> Option<FindPaint<'_>> {
    if !find.is_open() {
        return None;
    }
    Some(FindPaint {
        query: find.query(),
        replace: find.show_replace().then(|| find.replacement()),
        focus_replace: find.focus() == find::Field::Replace,
        selected: find.selected(),
        word: find.word(),
        regex: find.regex(),
        error: find.error(),
        matches: find.matches(),
        current: find.current(),
    })
}

/// Draw a splitter grab bar: its gap cleared to the background with a thin
/// divider line down the center. Drawn last, over any content that overhangs.
fn push_splitter(list: &mut DisplayList, rect: Rect) {
    list.push(DrawCmd::Fill {
        rect,
        color: BACKGROUND,
    });
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: rect.x + rect.w / 2,
            y: rect.y,
            w: 1,
            h: rect.h,
        },
        color: SPLITTER_COLOR,
    });
}

/// Draw the file-tree sidebar into its column `rect`: a darker background, then
/// each visible row (a folder marker plus the name), indented by depth and
/// offset by the tree's own scroll. Updates `content_height` for scroll
/// clamping. Rows do not wrap; a long name that overhangs is covered by the
/// editor's own background, which is drawn after this.
fn push_tree(
    list: &mut DisplayList,
    fonts: &Fonts,
    tree: &mut TreePane,
    rect: Rect,
    focused: bool,
) {
    list.push(DrawCmd::Fill {
        rect,
        color: TREE_BG,
    });
    let m = fonts.metrics(FONT_SIZE);
    let row_h = m.line_height.max(1);
    let rows = tree.tree.rows();
    tree.content_height = rows.len() as i32 * row_h + 2 * TREE_PAD;
    tree.scroll_y = scroll::clamp(tree.scroll_y, tree.content_height, rect.h);
    let face = FaceKey::Prose {
        size: FONT_SIZE,
        style: FontStyle::Regular,
    };
    let text_right = rect.x + rect.w - TREE_PAD;
    for (i, row) in rows.iter().enumerate() {
        let top = rect.y + TREE_PAD + i as i32 * row_h - tree.scroll_y;
        if top + row_h <= rect.y || top >= rect.y + rect.h {
            continue;
        }
        // The keyboard-selected row (while the tree is focused) reads as a
        // selection; a mouse-hovered row gets a subtler tint.
        let highlight = if focused && tree.selected == i {
            Some(SELECTION_COLOR)
        } else if tree.hovered == Some(i) {
            Some(TREE_HOVER_BG)
        } else {
            None
        };
        if let Some(color) = highlight {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: rect.x,
                    y: top,
                    w: rect.w,
                    h: row_h,
                },
                color,
            });
        }
        let x = rect.x + TREE_PAD + row.depth as i32 * TREE_INDENT;
        let baseline = top + m.ascent;
        let (marker, color) = if row.is_dir {
            (if row.expanded { "▾ " } else { "▸ " }, TREE_DIR_COLOR)
        } else {
            ("", FOREGROUND)
        };
        let label = format!("{marker}{}", row.name);
        list.push(text_cmd(face, x, baseline, text_right, color, label, m));
    }
}

pub(crate) fn build_display_list(
    fonts: &Fonts,
    layout: &Layout,
    rope: &crate::rope::Rope,
    p: &PaintParams,
) -> DisplayList {
    let (width, scroll_y) = (p.width, p.scroll_y);
    let view_h = p.height as i32;
    // The content's left edge (the column's inner margin). Text, panels, and
    // images hang off this so a pane's content sits in its own column; the
    // background clears from the column's outer edge, `MARGIN` to its left.
    let left = layout.left();
    // Roughly a few commands per visual line (panel, runs, decorations) plus the
    // background and caret; one upfront allocation beats growing the list mid-build.
    let mut list: DisplayList = Vec::with_capacity(layout.lines().len() * 3 + 8);

    // The background, full surface, first so a clipped repaint clears its region
    // before the rest of the list draws over it.
    list.push(DrawCmd::Fill {
        rect: Rect {
            x: left - MARGIN,
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
            (true, true) => display::RoundedCorners::Both,
            (true, false) => display::RoundedCorners::Top,
            (false, true) => display::RoundedCorners::Bottom,
            (false, false) => display::RoundedCorners::None,
        };
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: left,
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

    // Find matches, filled like a dimmer selection beneath the text. The current
    // match is skipped: the app mirrors it into the buffer selection, which was
    // just painted. Only matches inside the visible byte range are measured, so a
    // short query with thousands of hits does not price every frame by them.
    if let Some(find) = &p.find {
        let visible = visible_byte_range(fonts, layout, scroll_y, view_h);
        let lo = find.matches.partition_point(|&(_, end)| end <= visible.0);
        let hi = find
            .matches
            .partition_point(|&(start, _)| start < visible.1);
        for (i, &(start, end)) in find.matches.iter().enumerate().take(hi).skip(lo) {
            if i == find.current {
                continue;
            }
            for rect in layout.selection_rects(fonts, rope, start, end) {
                let y = rect.y - scroll_y;
                if y + rect.h <= 0 || y >= view_h {
                    continue;
                }
                list.push(DrawCmd::Fill {
                    rect: Rect { y, ..rect },
                    color: FIND_MATCH_COLOR,
                });
            }
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
                    x: left + line.indent,
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
                    x: left,
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
                    x: left,
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
        // A completed task's text reads dim and struck through, over the whole
        // item (wrapped continuations included), so done items scan as done.
        let done = line.task == Some(true);
        let mut pen = left + line.indent;
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
                let advance = shape::text_advance(face, &slice).round() as i32;
                match line.gutter {
                    Gutter::Bullet => {
                        // Place the marker's leading indentation first, so a nested
                        // item's dot sits at its depth, then the dot in place of the
                        // dash. The dot stands in for the `- ` marker, so it takes
                        // the muted marker color (MARKER_COLOR), not the body FG.
                        let indent = slice.len() - slice.trim_start().len();
                        let dot_x =
                            pen + shape::text_advance(face, &slice[..indent]).round() as i32;
                        let end = (dot_x as f32 + shape::text_advance(face, BULLET)).round() as i32;
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
                    // A task item's raw `- [ ]` gives way to a checkbox drawn
                    // with fills (the prose face carries no ☐/☑ glyphs), placed
                    // by the same geometry clicks hit-test against. The pen
                    // still advances by the full marker, so the text never
                    // shifts between the box and the revealed source.
                    Gutter::Checkbox { checked } => {
                        if let Some(rect) = layout.checkbox_rect(fonts, rope, line) {
                            let rect = Rect {
                                y: rect.y - scroll_y,
                                ..rect
                            };
                            push_checkbox(&mut list, rect, checked);
                        }
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
                // Every text run of a done task dims to the marker color, so
                // the whole item (its code and links included) reads muted.
                markdown::SpanKind::Text(_) if done => MARKER_COLOR,
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
            let end = (run_x as f32 + shape::text_advance(face, &slice)).round() as i32;
            list.push(text_cmd(fk, run_x, baseline, end, color, slice, m));
            pen = end;
            // Inline decorations over the glyphs, each spanning the run's drawn
            // width: a link is underlined and struck text gets a line through its
            // middle, both in the run's own color. Thickness scales with the
            // line's size so it stays proportionate on larger headings. A done
            // task strikes through all of its text instead.
            let deco = match run.kind {
                markdown::SpanKind::Text(_) if done => Some(baseline - m.ascent / 3),
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

    // The find bar and its match ticks float over everything else, so they go
    // last.
    if let Some(find) = &p.find {
        push_find_ticks(&mut list, layout, find, width as i32, view_h);
        push_find_bar(&mut list, fonts, find, width as i32);
    }

    list
}

/// Append the right-edge tick strip while find is open: one mark per match at
/// its document-proportional height, so the distribution of hits is visible at
/// a glance and survives scrolling. Ticks are deduplicated per two-pixel row,
/// which bounds the strip to O(viewport) commands however many matches exist;
/// the current match's tick is wider and bright.
fn push_find_ticks(
    list: &mut DisplayList,
    layout: &Layout,
    find: &FindPaint,
    width: i32,
    view_h: i32,
) {
    if find.matches.is_empty() || view_h <= 0 {
        return;
    }
    let lines = layout.lines();
    let content_h = (layout.text_height() + 2 * MARGIN).max(view_h) as i64;
    let mut occupied = vec![false; (view_h / 2 + 1) as usize];
    for (i, &(start, _)) in find.matches.iter().enumerate() {
        // The line containing the match's start; matches always sit inside
        // some line, but fall back to the top defensively.
        let idx = lines
            .partition_point(|l| l.start <= start)
            .saturating_sub(1);
        let Some(line) = lines.get(idx) else {
            continue;
        };
        let y = (line.baseline as i64 * (view_h - TICK_HEIGHT) as i64 / content_h) as i32;
        let bucket = (y / 2).clamp(0, view_h / 2) as usize;
        let current = i == find.current;
        if occupied[bucket] && !current {
            continue;
        }
        occupied[bucket] = true;
        let (w, color) = if current {
            (TICK_WIDTH + 3, CARET_COLOR)
        } else {
            (TICK_WIDTH, FIND_TICK_COLOR)
        };
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: width - w - 1,
                y,
                w,
                h: TICK_HEIGHT,
            },
            color,
        });
    }
}

/// The byte range spanned by the visual lines that touch the viewport, for
/// culling by byte offset before any geometry is measured. Conservative: a
/// line's visibility is judged by its text box, and the empty viewport yields
/// an empty range.
fn visible_byte_range(
    fonts: &Fonts,
    layout: &Layout,
    scroll_y: i32,
    view_h: i32,
) -> (usize, usize) {
    let mut range = (usize::MAX, 0);
    for line in layout.lines() {
        let m = fonts.metrics(line.size);
        let baseline = line.baseline - scroll_y;
        if baseline - m.ascent >= view_h || baseline + m.descent <= 0 {
            continue;
        }
        range.0 = range.0.min(line.start);
        range.1 = range.1.max(line.end);
    }
    if range.0 > range.1 {
        (0, 0)
    } else {
        range
    }
}

/// Append the find bar: a rounded panel floating at the top-right with the
/// query (its tail, when it overflows — the end is where editing happens), a
/// steady caret after it, and a match counter. Drawn from plain display-list
/// commands, so damage tracking and clipped repaints treat it like any other
/// content.
fn push_find_bar(list: &mut DisplayList, fonts: &Fonts, find: &FindPaint, width: i32) {
    let size = FONT_SIZE;
    let key = FaceKey::Prose {
        size,
        style: FontStyle::Regular,
    };
    let face = fonts.face(size, FontStyle::Regular);
    let m = fonts.metrics(size);
    let pad = size as i32 / 2;
    let row_h = m.ascent + m.descent;

    // The right-hand cluster on the query row: the mode flags (bright when
    // on) and the match counter (the alert color for a broken regex).
    let counter = if find.query.is_empty() {
        String::new()
    } else if find.error {
        "!".to_string()
    } else if find.matches.is_empty() {
        "0/0".to_string()
    } else {
        format!("{}/{}", find.current + 1, find.matches.len())
    };
    let counter_w = shape::text_advance(face, &counter).ceil() as i32;
    let flag_word_w = shape::text_advance(face, "W").ceil() as i32;
    let flag_regex_w = shape::text_advance(face, ".*").ceil() as i32;
    let gap = pad / 2;
    let tail_w = flag_word_w + gap + flag_regex_w + gap + counter_w;

    // Keep a field's display within a third of the window by dropping leading
    // clusters: the end is where editing happens. Queries are short, so the
    // remeasure loop is cheap.
    fn tail_that_fits<'t>(face: &crate::freetype::Face, text: &'t str, max: f32) -> &'t str {
        let mut shown = text;
        while shape::text_advance(face, shown) > max && !shown.is_empty() {
            shown = &shown[grapheme::next_boundary(shown, 0)..];
        }
        shown
    }
    let field_max = (width / 3).max(3 * size as i32) as f32;
    let query = tail_that_fits(face, find.query, field_max);
    let query_w = shape::text_advance(face, query).ceil() as i32;
    let replace = find.replace.map(|r| tail_that_fits(face, r, field_max));
    let replace_w = replace
        .map(|r| shape::text_advance(face, r).ceil() as i32)
        .unwrap_or(0);
    let arrow = "→ ";
    let arrow_w = shape::text_advance(face, arrow).ceil() as i32;

    let rows = 1 + replace.is_some() as i32;
    let replace_row_w = if replace.is_some() {
        arrow_w + replace_w
    } else {
        0
    };
    let field_w = query_w.max(replace_row_w) + CARET_WIDTH + 2;
    let w = (field_w + pad + tail_w + 2 * pad).max(6 * size as i32);
    let h = rows * row_h + (rows + 1) * (pad / 2);
    let x = width - w - MARGIN;
    let y = MARGIN / 2;
    list.push(DrawCmd::RoundRect {
        rect: Rect { x, y, w, h },
        color: FIND_BG,
        radius: (pad / 2).max(2),
        corners: display::RoundedCorners::Both,
    });

    // A steady caret (no blink) after the focused field's text: it marks where
    // typing goes while the bar holds the keyboard.
    let caret_after = |list: &mut DisplayList, pen: i32, baseline: i32| {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x: pen + 1,
                y: baseline - m.ascent,
                w: CARET_WIDTH,
                h: row_h,
            },
            color: CARET_COLOR,
        });
    };

    // Row one: the query, a selected query on a selection-colored fill,
    // invalid regexes in the alert color.
    let baseline = y + pad / 2 + m.ascent;
    let pen = x + pad;
    if !query.is_empty() {
        if find.selected && !find.focus_replace {
            list.push(DrawCmd::Fill {
                rect: Rect {
                    x: pen,
                    y: baseline - m.ascent,
                    w: query_w,
                    h: row_h,
                },
                color: SELECTION_COLOR,
            });
        }
        let color = if find.error {
            FIND_ERROR_COLOR
        } else {
            FOREGROUND
        };
        list.push(text_cmd(
            key,
            pen,
            baseline,
            pen + query_w,
            color,
            query.to_string(),
            m,
        ));
    }
    if !find.focus_replace {
        caret_after(list, pen + query_w, baseline);
    }

    // The flags and counter, right-aligned on the query row.
    let mut tx = x + w - pad - tail_w;
    list.push(text_cmd(
        key,
        tx,
        baseline,
        tx + flag_word_w,
        if find.word { FOREGROUND } else { MARKER_COLOR },
        "W".to_string(),
        m,
    ));
    tx += flag_word_w + gap;
    list.push(text_cmd(
        key,
        tx,
        baseline,
        tx + flag_regex_w,
        if find.regex { FOREGROUND } else { MARKER_COLOR },
        ".*".to_string(),
        m,
    ));
    tx += flag_regex_w + gap;
    if !counter.is_empty() {
        let color = if find.error {
            FIND_ERROR_COLOR
        } else {
            MARKER_COLOR
        };
        list.push(text_cmd(
            key,
            tx,
            baseline,
            tx + counter_w,
            color,
            counter,
            m,
        ));
    }

    // Row two: the replacement, prefixed with a dim arrow; selected content
    // sits on the same selection fill as a selected query.
    if let Some(replacement) = replace {
        let baseline = baseline + row_h + pad / 2;
        list.push(text_cmd(
            key,
            pen,
            baseline,
            pen + arrow_w,
            MARKER_COLOR,
            arrow.to_string(),
            m,
        ));
        if !replacement.is_empty() {
            if find.selected && find.focus_replace {
                list.push(DrawCmd::Fill {
                    rect: Rect {
                        x: pen + arrow_w,
                        y: baseline - m.ascent,
                        w: replace_w,
                        h: row_h,
                    },
                    color: SELECTION_COLOR,
                });
            }
            list.push(text_cmd(
                key,
                pen + arrow_w,
                baseline,
                pen + arrow_w + replace_w,
                FOREGROUND,
                replacement.to_string(),
                m,
            ));
        }
        if find.focus_replace {
            caret_after(list, pen + arrow_w + replace_w, baseline);
        }
    }
}

/// Append a task item's checkbox at `rect` (screen space, from
/// [`Layout::checkbox_rect`]): unchecked is an outlined rounded square (an
/// inset background fill hollows the border), checked fills with the accent
/// and carries a check mark. Built entirely from fills, since the prose face
/// has no checkbox glyphs to lean on.
fn push_checkbox(list: &mut DisplayList, rect: Rect, checked: bool) {
    let radius = (rect.w / 4).max(2);
    list.push(DrawCmd::RoundRect {
        rect,
        color: if checked { LINK_COLOR } else { MARKER_COLOR },
        radius,
        corners: display::RoundedCorners::Both,
    });
    if checked {
        push_check_mark(list, rect);
    } else {
        let t = (rect.w / 8).max(1);
        list.push(DrawCmd::RoundRect {
            rect: Rect {
                x: rect.x + t,
                y: rect.y + t,
                w: rect.w - 2 * t,
                h: rect.h - 2 * t,
            },
            color: BACKGROUND,
            radius: (radius - t).max(1),
            corners: display::RoundedCorners::Both,
        });
    }
}

/// Append a check mark inside the checked box `b`: a small tick built from
/// overlapping square fills stepping along its two arms (a short down-stroke
/// meeting a longer up-stroke), in the background color over the accent fill.
fn push_check_mark(list: &mut DisplayList, b: Rect) {
    let u = (b.w / 6).max(1);
    // The arms meet low and left of center, like a hand-drawn tick.
    let (mx, my) = (b.x + b.w * 2 / 5, b.y + b.h * 13 / 20);
    // Steps up-left (the short arm), the meeting point, then up-right (the
    // long arm); cells one pixel larger than the step so the diagonal connects.
    let cells = [
        (mx - u, my - u),
        (mx, my),
        (mx + u, my - u),
        (mx + 2 * u, my - 2 * u),
        (mx + 3 * u, my - 3 * u),
    ];
    for (x, y) in cells {
        list.push(DrawCmd::Fill {
            rect: Rect {
                x,
                y,
                w: u + 1,
                h: u + 1,
            },
            color: BACKGROUND,
        });
    }
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
        corners: display::RoundedCorners::Both,
    });
    // Center the glyph in both axes from the code-face metrics.
    let fk = FaceKey::Code { size: FONT_SIZE };
    let face = fonts.face_for(fk);
    let glyph = "</>";
    let advance = shape::text_advance(face, glyph).round() as i32;
    let m = fonts.metrics(FONT_SIZE);
    let gx = btn.x + (btn.w - advance) / 2;
    let gbaseline = btn.y + (btn.h - (m.ascent + m.descent)) / 2 + m.ascent;
    let gend = (gx as f32 + shape::text_advance(face, glyph)).round() as i32;
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
    use crate::rope::Rope;

    #[test]
    fn columns_lay_out_tree_and_two_editors_with_splitters() {
        let panes = PaneLayout {
            tree_width: 200,
            split_ratio: 0.5,
            tree_visible: true,
            second_open: true,
        };
        let cols = panes.columns(1000, 700);
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
        let cols = panes.columns(1000, 700);
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

    fn paint_params(width: usize, height: usize) -> PaintParams<'static> {
        PaintParams {
            width,
            height,
            scroll_y: 0,
            selection: None,
            cursor: 0,
            caret_visible: false,
            find: None,
        }
    }

    #[test]
    fn task_checkboxes_and_done_styling_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let src = "- [ ] open\n- [x] done\nx";
        let rope = Rope::from_str(src);
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        // Cursor on the trailing "x": both task lines render their boxes.
        let off = src.len() - 1;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (off, off));
        let list = build_display_list(&fonts, &layout, &rope, &paint_params(w, h));

        // One outlined box (marker-color shell hollowed by a background inset)
        // and one accent-filled box.
        let round = |color: u32| {
            list.iter()
                .filter(|c| matches!(c, DrawCmd::RoundRect { color: c2, .. } if *c2 == color))
                .count()
        };
        assert_eq!(round(MARKER_COLOR), 1, "the unchecked shell");
        assert_eq!(round(BACKGROUND), 1, "its hollowing inset");
        assert_eq!(round(LINK_COLOR), 1, "the checked fill");
        // The check mark: small background-color cells inside the accent box.
        let accent = list.iter().find_map(|c| match c {
            DrawCmd::RoundRect { rect, color, .. } if *color == LINK_COLOR => Some(*rect),
            _ => None,
        });
        let accent = accent.expect("the checked box");
        let cells = list
            .iter()
            .filter(|c| {
                matches!(c, DrawCmd::Fill { rect, color }
                    if *color == BACKGROUND
                        && rect.x >= accent.x
                        && rect.x + rect.w <= accent.x + accent.w
                        && rect.y >= accent.y
                        && rect.y + rect.h <= accent.y + accent.h)
            })
            .count();
        assert!(cells >= 5, "a stair of check cells, got {cells}");
        // The raw markers are substituted, not drawn: no bracket glyphs.
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            !texts.iter().any(|t| t.contains('[')),
            "no raw checkbox glyphs on rendered lines: {texts:?}"
        );
        // The done item's text dims to the marker color and gains a strike
        // fill spanning its drawn width; the open item's text stays foreground.
        let text_color = |needle: &str| {
            list.iter().find_map(|c| match c {
                DrawCmd::Text { text, color, .. } if text == needle => Some(*color),
                _ => None,
            })
        };
        assert_eq!(text_color("open"), Some(FOREGROUND));
        assert_eq!(text_color("done"), Some(MARKER_COLOR), "done text is dim");
        let done_run = list
            .iter()
            .find_map(|c| match c {
                DrawCmd::Text {
                    text, x, baseline, ..
                } if text == "done" => Some((*x, *baseline)),
                _ => None,
            })
            .expect("the done run");
        let m = fonts.metrics(FONT_SIZE);
        assert!(
            list.iter().any(|c| {
                matches!(c, DrawCmd::Fill { rect, color }
                    if *color == MARKER_COLOR
                        && rect.x == done_run.0
                        && rect.y > done_run.1 - m.ascent
                        && rect.y < done_run.1)
            }),
            "the done text is struck through"
        );
    }

    #[test]
    fn find_bar_and_match_highlights_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("one two one two one\n");
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let matches = [(0usize, 3usize), (8, 11), (16, 19)];
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            // The app mirrors the current match into the selection.
            selection: Some((8, 11)),
            cursor: 11,
            caret_visible: false,
            find: Some(FindPaint {
                query: "one",
                replace: None,
                focus_replace: false,
                selected: false,
                word: false,
                regex: false,
                error: false,
                matches: &matches,
                current: 1,
            }),
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);
        let match_fills = list
            .iter()
            .filter(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == FIND_MATCH_COLOR))
            .count();
        assert_eq!(match_fills, 2, "the two non-current matches highlight");
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::Fill { color, .. } if *color == SELECTION_COLOR)),
            "the current match is painted as the selection"
        );
        assert!(
            list.iter()
                .any(|c| matches!(c, DrawCmd::RoundRect { color, .. } if *color == FIND_BG)),
            "the bar panel is drawn"
        );
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"one"), "the query is drawn in the bar");
        assert!(texts.contains(&"2/3"), "the counter shows current/total");
    }

    #[test]
    fn the_replace_row_flags_and_ticks_reach_the_display_list() {
        let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE)).expect("a default font");
        let rope = Rope::from_str("one two one\nfiller\none more\n");
        let (w, h) = (600usize, 200usize);
        let wrap = w as i32 - 2 * MARGIN;
        let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let matches = [(0usize, 3usize), (8, 11), (19, 22)];
        let params = PaintParams {
            width: w,
            height: h,
            scroll_y: 0,
            selection: Some((0, 3)),
            cursor: 3,
            caret_visible: false,
            find: Some(FindPaint {
                query: "one",
                replace: Some("two"),
                focus_replace: true,
                selected: false,
                word: true,
                regex: false,
                error: false,
                matches: &matches,
                current: 0,
            }),
        };
        let list = build_display_list(&fonts, &layout, &rope, &params);
        let texts: Vec<&str> = list
            .iter()
            .filter_map(|c| match c {
                DrawCmd::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(texts.contains(&"→ "), "the replace row is prefixed");
        assert!(texts.contains(&"two"), "the replacement text is drawn");
        assert!(
            texts.contains(&"W") && texts.contains(&".*"),
            "both mode flags are drawn"
        );
        // The right-edge tick strip: one accent tick for the current match and
        // gold ticks for the rest (the bar caret sits well left of the edge).
        let edge = w as i32 - 10;
        let ticks = list
            .iter()
            .filter(|c| {
                matches!(c, DrawCmd::Fill { color, rect }
                    if (*color == FIND_TICK_COLOR || *color == CARET_COLOR) && rect.x >= edge)
            })
            .count();
        assert!(
            ticks >= 2,
            "expected match ticks at the right edge, got {ticks}"
        );
    }
}
