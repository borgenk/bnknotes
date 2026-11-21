//! The editor's orchestration: it binds the compositor globals, builds the
//! xdg-shell surface stack, and runs the event loop.

mod clipboard;
mod input;
mod present;

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;
use std::time::{Duration, Instant};

use self::clipboard::{CLIPBOARD_MIMES, ClipboardState, IMAGE_MIMES, is_text_mime};
use self::input::KeyRepeat;
use self::present::GpuPresentation;
use crate::ai::{self, Preview};
use crate::cli::{Mode, USAGE};
use crate::config::{self, CaretConfig, Config, IconKeys, Vault, VaultKind, vault_name};
use crate::icon;
use crate::menu::{Menu, MenuAction, MenuItem};
use crate::paint::{
    BarCtx, CaretPaint, PaintParams, RailDragView, RailView, TabStrip, build_display_list_into,
    build_find_paint, conflict_banner_layout, diff_columns, font_sizes, measure_tree,
    push_ai_preview, push_conflict_banner, push_diff_view, push_image_view, push_menu,
    push_palette, push_rail, push_splitter, push_top_bar, push_tree, top_bar_height, tree_metrics,
};
use crate::palette::{Palette, PaletteAction, PaletteRow, Rows};
use crate::pane::{
    CaretAnim, CaretBlink, CaretFrame, Columns, Focus, Hit, PaneGroup, PaneLayout, Saved, TreePane,
    restore_group,
};
use crate::platform::conn::{Connection, Fill};
use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::freetype::{FontConfig, Fonts};
use crate::platform::geom::Rect;
use crate::platform::protocol::{
    self, wl_buffer, wl_callback, wl_compositor, wl_data_device_manager, wl_data_offer, wl_display,
    wl_output, wl_registry, wl_seat, wl_surface, wp_fractional_scale_manager_v1,
    wp_fractional_scale_v1, wp_viewporter, xdg_surface, xdg_toplevel, xdg_wm_base,
};
use crate::platform::scroll::{self, Scrollbar};
use crate::platform::wire::{Arg, Message, Reader};
use crate::platform::xkb::Xkb;
use crate::render::display::{DisplayList, DrawCmd};
use crate::render::image::{Image, ImageStore, Limits};
use crate::scale::Scale;
use crate::search;
use crate::store;
use crate::theme::{BACKGROUND, CARET_WIDTH, FONT_SIZE, ICON_HOVER_ZOOM, RAIL_RADIUS};
use crate::tree;
use crate::watch::FileWatcher;

/// Upper bound on either surface dimension, applied to a compositor-supplied
/// configure size. Far beyond any real display (8K is 7680 wide), it just keeps a
/// bogus or hostile size from blowing up the buffer-size arithmetic.
const MAX_DIMENSION: u32 = 16384;

/// First object id in Wayland's server-allocated range. Client-created ids stay
/// below this; the free list only ever recycles client ids.
const SERVER_ID_BASE: u32 = 0xff00_0000;

/// How long a document sits unsaved before the autosave writes it. The clock
/// restarts on every edit, so this is time since the typing stopped, not time since
/// it started: a save lands in the pause between sentences rather than mid-word.
/// Short enough that "did that save?" never comes up, and long enough that a fast
/// typist's pauses do not each cost a write.
const AUTOSAVE_IDLE: Duration = Duration::from_secs(2);

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

pub fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    #[cfg(any(test, feature = "dev"))]
    {
        let dev_args: Vec<String> = args.iter().map(|a| a.to_string_lossy().into()).collect();
        if let Some(result) = crate::dev::run(&dev_args) {
            return report(result);
        }
    }
    let result = match Mode::parse(args) {
        Ok(Mode::Run { path, demo_diff }) => run(path, demo_diff),
        Ok(Mode::GpuProbe) => gpu_probe(),
        Err(arg) => {
            eprintln!("bnknotes: unknown argument {arg:?}; {USAGE}");
            return ExitCode::from(2);
        }
    };
    report(result)
}

fn report(result: Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("bnknotes: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: Option<PathBuf>, demo_diff: bool) -> Result<()> {
    let mut state = State::new(path)?;
    if demo_diff {
        state.open_demo_diff();
    }
    state.bring_up()
}

/// Runs the startup negotiation without opening a window, so a successful probe
/// means the GPU path's protocol prerequisites hold.
fn gpu_probe() -> Result<()> {
    State::new(None)?.probe_dmabuf()
}

/// The before and after text of the `--demo-diff` review, crafted so the diff exercises
/// every concept the view draws: unchanged lines carried through (heading, blanks, prose,
/// list items), a changed line with a word inserted, a long changed line that wraps
/// within its band, a wholly removed line (hatched on the right), a wholly added line
/// (hatched on the left), and a one-into-many replacement (a changed pair plus surplus
/// added lines). Kept as two line lists joined by newlines so the alignment reads clearly
/// from the source.
fn demo_diff_texts() -> (String, String) {
    let old = [
        "# Release checklist",
        "",
        "Run the full test suite before tagging.",
        "Update the changlog with the new version and link every entry to its pull \
         request so the release history stays easy to trace later.",
        "Remove this temporary debugging note before you cut the release.",
        "",
        "- Bump the version number",
        "- Tag the commit",
        "",
        "Deploy: build, smoke-test, then ship.",
    ]
    .join("\n");
    let new = [
        "# Release checklist",
        "",
        "Run the full test suite locally and in CI before tagging.",
        "Update the changelog with the new version and link every entry to its pull \
         request so the release history stays easy to trace later.",
        "",
        "- Bump the version number",
        "- Tag the commit",
        "- Push the tag to origin",
        "",
        "Deploy:",
        "1. build the release artifacts",
        "2. smoke-test, then ship",
    ]
    .join("\n");
    (old, new)
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

/// A picture shown over the whole window: which pane's it is (so its file resolves
/// against that pane's store, the same way the note's own copy of it does), and the
/// destination it was written with, which is what the store keys it by.
#[derive(Clone)]
struct ImageView {
    pane: usize,
    raw: String,
}

/// One AI command offered for a pane: a built-in action by its index, or the
/// free-form custom prompt. Built once by [`State::ai_commands`] and mapped into a
/// menu action or a palette action by whichever surface offers it.
#[derive(Clone, Copy)]
enum AiCommand {
    Run(usize),
    Custom,
}

/// An in-progress vault drag-to-reorder, the rail's answer to [`TabDrag`] and the
/// same shape down the y axis: which chip (by its current index, which the reorder
/// keeps up to date), the pointer y at press, the grab offset within the chip, and
/// whether the press has crossed the threshold into a drag.
///
/// Where it parts from a tab: a tab selects on press and may then be dragged, but a
/// chip only switches vault on release, and only if it never moved. Switching is not
/// a selection, it is saving both panes and rebuilding them around another vault's
/// tabs, and doing that to a chip the user meant to drag would swap the window's
/// contents out from under the drag.
#[derive(Clone, Copy)]
struct VaultDrag {
    vault: usize,
    grab_y: f32,
    grab_dy: i32,
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
    /// handed to the shader each frame (see [`crate::config::Config`]). Prose uses
    /// this; code uses [`Self::code_glyph_gamma`], baked per glyph at build time.
    glyph_gamma: f32,
    /// The gentler mask gamma for code glyphs (see [`crate::config::Config`]).
    code_glyph_gamma: f32,
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
    /// The vaults in the order the rail stacks their chips: the config's, then the
    /// temporary one this window was launched in when the config does not name it
    /// (see [`rail_vaults`]). Which one is active is not held here: it is the file
    /// tree's root, so the two can never disagree.
    vaults: Vec<Vault>,
    /// Which vault chip the pointer is over, for the rail's hover lift; `None` when
    /// off every chip.
    hovered_vault: Option<usize>,
    /// A chip being dragged to a new place in the rail, `None` when none is.
    drag_vault: Option<VaultDrag>,
    /// The chips' baked icons, keyed by [`Vault::icon_key`]: a crop of a vault
    /// picture, resampled to the chip, tinted and rounded, ready for the renderer to
    /// blit like any other image. Baked by [`Self::sync_icons`].
    icons: ImageStore,
    /// The chip side the icons in `icons` were baked for, so a display-scale change
    /// bakes them again rather than letting the GPU stretch a small one.
    icon_size: i32,
    /// The next baked icon's key. A counter, so a newly pasted icon is a key nothing
    /// has drawn: the damage diff sees a command that changed, and the renderer
    /// uploads a texture it has not got, both of which a stable key would hide.
    icon_seq: u64,
    /// The open context menu, `None` when none is. It floats over the chrome and
    /// takes every press while it is up (see [`crate::menu`]).
    menu: Option<Menu>,
    /// The open AI text-action preview, `None` when none is. While it is up it holds
    /// the keyboard (Enter accepts the diff, Escape discards it) and no editing
    /// reaches the buffer, so the captured selection stays valid until it resolves.
    ai: Option<Preview>,
    /// The pipe a finished AI job writes a byte to, polled by the event loop so a
    /// several-second call never blocks a frame (see [`crate::ai::Wake`]).
    ai_wake: ai::Wake,
    /// Whether the context menu offers AI transforms over a selection (config `ai`).
    ai_enabled: bool,
    /// The command-line backend the AI actions run through (config `ai provider`).
    ai_provider: ai::Provider,
    /// The open palette, `None` when none is. It floats over the window and takes
    /// every key while it is up (see [`crate::palette`]).
    palette: Option<Palette>,
    /// The last search's answer, kept so a query that only grew can narrow to the
    /// notes its prefix already matched rather than walking the vault again. Reset
    /// when a search palette opens; meaningless while none is.
    search: search::Results,
    /// The vault the window was rooted at before the active one, so the palette can
    /// offer going back first. Set by [`Self::switch_vault`] and so within this run
    /// only: a "where I just was" that outlived the session would be answering a
    /// question nobody asked.
    last_vault: Option<PathBuf>,
    /// A picture opened over the whole window by a double-click, `None` when none is.
    image_view: Option<ImageView>,
    /// Every vault's remembered settings (the sidebar's folder filter), kept whole
    /// so saving the active vault's choice leaves the other vaults' alone.
    vault_prefs: store::VaultPrefs,
    panes: PaneLayout,
    focus: Focus,
    /// The editor last given focus, so a file opened from the tree (which takes
    /// focus itself) lands in the content pane the user was last editing.
    last_editor: usize,
    /// What the pointer is over in a tab strip (a tab, its close glyph, or the
    /// new-tab button), for the hover highlights; `None` when off every strip.
    hovered_tab: Option<TabHover>,
    /// Each pane's horizontal tab-strip scroll, in device pixels, when its tabs
    /// overflow. The wheel drives it while the pointer is over the strip; it is
    /// clamped and nudged to keep the active tab in view each frame (see
    /// [`Self::sync_tab_scroll`]).
    tab_scroll: [i32; 2],
    /// The (active index, tab count) each strip's scroll was last reconciled for,
    /// so a tab switch or an open/close reveals the active tab while free wheeling
    /// leaves the offset alone.
    tab_anchor: [(usize, usize); 2],
    /// While a splitter is dragged, which one: 0 = tree|A edge, 1 = A|B edge.
    drag_splitter: Option<usize>,
    /// While a scrollbar thumb is dragged, whose bar it is. The grab's offset inside
    /// the thumb lives on that [`Scrollbar`]. [`Focus`] names the region here because
    /// the scrollable regions are exactly the focusable ones: the sidebar and the two
    /// content panes.
    drag_scroll: Option<Focus>,
    /// While a tab is dragged to reorder it: which pane and tab, and the grab state.
    drag_tab: Option<TabDrag>,
    /// The toplevel title last sent, so it is only re-set when it changes.
    title: String,
    /// Where the next title is formatted before the compare, so the check that runs
    /// after every edit costs no allocation on the common path (unchanged).
    title_scratch: String,

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
    /// The caret's blink clock: whether it shows this frame, when it next toggles,
    /// and how much of its run is left. Activity (`bump_caret`) resets it to shown
    /// and restarts the run. Applies to the focused editor's caret; other panes draw
    /// a dimmed, static caret.
    caret: CaretBlink,
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
    /// When the idle autosave is due, `None` when no document has unsaved edits
    /// waiting on it. Pushed back by every edit (see [`Self::poll_autosave`]), so it
    /// comes due once the typing stops.
    autosave_at: Option<Instant>,

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
    /// Watches the directories of open files, so a change made outside the editor
    /// is noticed while the window sits idle. Polled alongside the compositor
    /// socket; a wake re-checks the open files against disk.
    watcher: FileWatcher,
}

/// Carry one column's scrollbar forward a frame. A column whose content fits has
/// nothing to point at, so its bar drops out of sight at once rather than fading from
/// a document it no longer describes.
fn tick_scrollbar(bar: &mut Scrollbar, content_height: i32, view_h: i32, now: Instant) {
    if scroll::scrollable(content_height, view_h) {
        bar.tick(now);
    } else {
        bar.hide();
    }
}

/// Resolve the active vault root and any file to open, from the CLI argument, the
/// saved session vault, and the configured vaults. Precedence, never falling back
/// to the working directory: a directory argument is the vault and opens no file;
/// a file argument opens the file, rooted at the configured vault that contains it,
/// else the file's own parent; no argument restores the remembered session vault
/// only when it is still one of the configured vaults, else the first configured
/// vault (rooted there even when it does not yet exist). Requiring the remembered
/// vault to be configured keeps a stale pre-vault root (an old working directory)
/// from shadowing the configured default. Pure precedence, split out of
/// `State::new` so it is testable without a live connection.
fn resolve_vault(
    arg: Option<&Path>,
    session_root: Option<&Path>,
    vaults: &[Vault],
) -> (PathBuf, Option<PathBuf>) {
    // The configured default, used when nothing else resolves. An empty list (only
    // possible from a future misconfigured loader) degrades to the current dir.
    let default = || {
        vaults
            .first()
            .map(|v| v.path.clone())
            .unwrap_or_else(|| PathBuf::from("."))
    };
    match arg {
        Some(p) if p.is_dir() => (p.to_path_buf(), None),
        Some(p) => {
            let root = vault_containing(p, vaults).unwrap_or_else(|| {
                p.parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."))
            });
            (root, Some(p.to_path_buf()))
        }
        None => (
            session_root
                .filter(|r| vaults.iter().any(|v| v.path.as_path() == *r))
                .map(Path::to_path_buf)
                .unwrap_or_else(default),
            None,
        ),
    }
}

/// When the idle autosave is due, given what a loop pass found: `edited` is whether
/// any document changed since the last pass, `pending` whether any still holds
/// unsaved edits it has a file to write them to, and `current` the deadline standing
/// before this pass.
///
/// An edit pushes the window out to `now + AUTOSAVE_IDLE`, so a save lands once the
/// typing stops rather than during it. Nothing pending drops the deadline, which is
/// what an undo back to the saved text does, and what leaves an idle editor with no
/// timer waking the loop. A pass with no edit leaves the deadline alone to run out.
///
/// Pure, so the rules are tested without a window: the timer is the part of an
/// autosave that is easy to get subtly wrong and hard to notice.
fn autosave_deadline(
    edited: bool,
    pending: bool,
    current: Option<Instant>,
    now: Instant,
) -> Option<Instant> {
    if !pending {
        return None;
    }
    if edited {
        return Some(now + AUTOSAVE_IDLE);
    }
    current
}

/// The rail's vaults as palette rows, `active` left out: a row that goes where the
/// window already is would do nothing.
///
/// The vault the window was at before (`last`) comes first, so opening the palette
/// and pressing Enter is "back to where I just was". The rest hold the rail's order,
/// which is the order the chips were dragged into.
fn vault_rows(vaults: &[Vault], active: Option<usize>, last: Option<&Path>) -> Vec<PaletteRow> {
    let mut rows: Vec<(bool, PaletteRow)> = vaults
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != active)
        .map(|(i, vault)| {
            // A vault's name is the last component of its path, the same name the
            // chip takes its initials from.
            let name = vault.path.file_name().unwrap_or(vault.path.as_os_str());
            (
                Some(vault.path.as_path()) == last,
                PaletteRow {
                    label: format!("vault: {}", name.to_string_lossy()),
                    action: PaletteAction::SwitchVault(i),
                },
            )
        })
        .collect();
    // A stable sort by "was this the last one" leaves the rail's order standing under
    // the one row that jumps it.
    rows.sort_by_key(|(recent, _)| !recent);
    rows.into_iter().map(|(_, row)| row).collect()
}

/// A search's hits as palette rows, in the order the scan found them: the note and
/// line a match is on, then the line itself. The note is named relative to the vault,
/// which is the only part of the path that says anything the user did not choose.
///
/// The rows are the query's answer, so the palette shows them as given
/// ([`Rows::AsGiven`]) rather than ranking a hit by how its own label reads.
fn search_rows(results: &search::Results, root: &Path) -> Vec<PaletteRow> {
    results
        .hits
        .iter()
        .map(|hit| PaletteRow {
            label: format!(
                "{}:{}  {}",
                search::relative(&hit.path, root).display(),
                hit.line,
                hit.text
            ),
            action: PaletteAction::OpenMatch {
                path: hit.path.clone(),
                offset: hit.offset,
            },
        })
        .collect()
}

/// What a search's rows cost, for the field's right-hand corner: how many matches
/// there are, and whether the cap stopped the count short of them. A query too short
/// to scan says what it is waiting for; one that found nothing says that, since an
/// empty panel alone does not distinguish "no matches" from "not looking yet".
fn search_note(results: &search::Results, query: &str) -> String {
    if query.chars().count() < search::MIN_QUERY {
        return format!("type {} characters", search::MIN_QUERY);
    }
    match (results.hits.len(), results.truncated) {
        (0, _) => "no matches".to_string(),
        (1, false) => "1 line".to_string(),
        (n, false) => format!("{n} lines"),
        // The rows are the first `n` of more than `n`: say so rather than let the list
        // read as the whole answer.
        (n, true) => format!("first {n} lines"),
    }
}

/// The rail this window opens with: the vaults the config names, arranged the way
/// the rail was last left (`stored`), and after them the active `root` as a
/// temporary vault when the config does not name it (`bnknotes .` in a directory
/// that is not a vault, or a file opened outside every vault).
///
/// The two lists answer different questions, which is why both are read. The config
/// says which vaults there are; a vault only in it is new, and joins the end of the
/// rail wearing the first color no chip wears yet. The stored rail says where the
/// chips sit and what they wear; an entry only in it names a vault the config no
/// longer has, and is dropped. So dragging a chip is remembered, hand-editing the
/// config still adds and removes vaults, and neither undoes the other.
///
/// A temporary vault is always last and is never stored: it belongs to this window's
/// launch, not to the rail's arrangement.
fn rail_vaults(configured: Vec<Vault>, stored: &[store::RailVault], root: &Path) -> Vec<Vault> {
    let mut taken: Vec<u32> = Vec::new();
    let mut arranged: Vec<Vault> = Vec::new();
    // The stored order first, keeping only the chips the config still names, and only
    // one chip per vault however a hand-edited file came to name one twice.
    for chip in stored {
        let named = configured.iter().any(|v| v.path == chip.root);
        if !named || arranged.iter().any(|v| v.path == chip.root) {
            continue;
        }
        taken.push(chip.color);
        let mut vault = Vault::new(chip.root.clone(), chip.color, VaultKind::Saved);
        vault.icon = chip.icon.clone();
        arranged.push(vault);
    }
    // Then whatever the config has that the rail has not shown before.
    for vault in &configured {
        if arranged.iter().any(|v| v.path == vault.path) {
            continue;
        }
        let color = config::deal_color(&taken, arranged.len());
        taken.push(color);
        arranged.push(Vault::new(vault.path.clone(), color, VaultKind::Saved));
    }
    if !arranged.iter().any(|v| v.path == root) {
        let color = config::deal_color(&taken, arranged.len());
        arranged.push(Vault::new(root.to_path_buf(), color, VaultKind::Temporary));
    }
    arranged
}

/// Bake `icon` into the two bitmaps a chip of `side` device pixels wears: its square
/// of the picture at rest, and the same square taken [`ICON_HOVER_ZOOM`] closer in
/// for under the pointer. Both are resampled to the chip and rounded by `radius` to
/// the chip's own corners.
///
/// The picture is left as it is. A chip wearing one shows no color and no letters
/// (see the rail's painter), so there is nothing on it for a wash to keep legible,
/// and a picture worth choosing is worth seeing. The rounding is baked in because an
/// image is drawn as a plain quad: the alpha carries the corners the renderer would
/// otherwise square off. That is also why the pointer gets a bitmap of its own
/// rather than a closer look at this one: sampling inside a bitmap whose corners are
/// already rounded would cut each of them flat where the zoom crossed it.
///
/// One decode, two crops. `None` when the picture cannot be read or decoded, which
/// is a file that has moved or been deleted since it was pasted.
fn bake_icon(icon: &icon::VaultIcon, side: i32, radius: i32) -> Option<(Image, Image)> {
    let size = u32::try_from(side).ok()?;
    let bytes = std::fs::read(&icon.source).ok()?;
    let source = Image::decode(&bytes, Limits::default()).ok()?;
    let radius = u32::try_from(radius).unwrap_or(0);
    let mut rest = source.crop_scaled(icon.x, icon.y, icon.side, size)?;
    rest.round_corners(radius);
    let (hx, hy, hside) = zoomed_square(icon.x, icon.y, icon.side);
    let mut hover = source.crop_scaled(hx, hy, hside, size)?;
    hover.round_corners(radius);
    Some((rest, hover))
}

/// The square `(x, y, side)` taken [`ICON_HOVER_ZOOM`] closer in, about the same
/// middle: what a chip shows under the pointer, so the picture pushes toward the
/// pointer while the chip itself stays exactly where it was and the rail's rhythm
/// never wobbles.
fn zoomed_square(x: u32, y: u32, side: u32) -> (u32, u32, u32) {
    let zoomed = ((side as f32 / ICON_HOVER_ZOOM) as u32).max(1).min(side);
    let inset = (side - zoomed) / 2;
    (x + inset, y + inset, zoomed)
}

/// The rail's arrangement to remember: its saved chips, in order, each with the
/// color it wears. A temporary vault is left out, since it is gone by the next
/// launch and a rail full of the directories someone once opened is not an
/// arrangement worth keeping.
fn rail_session(vaults: &[Vault]) -> Vec<store::RailVault> {
    vaults
        .iter()
        .filter(|v| v.kind == VaultKind::Saved)
        .map(|v| store::RailVault {
            root: v.path.clone(),
            color: v.color,
            icon: v.icon.clone(),
        })
        .collect()
}

/// The configured vault that holds `file`, if any: the vault whose root is an
/// ancestor of the file, choosing the deepest when several nest, so a note opened
/// deep in a vault still resolves its embeds against the whole vault root.
fn vault_containing(file: &Path, vaults: &[Vault]) -> Option<PathBuf> {
    vaults
        .iter()
        .filter(|v| file.starts_with(&v.path))
        .max_by_key(|v| v.path.components().count())
        .map(|v| v.path.clone())
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
        // The user's config (the vault list; the rest is still defaults in the
        // source), or the built-in defaults when there is no file. Font selection
        // lives here too: open every size the styles use so headings render larger
        // than body text without re-opening mid-frame.
        let config = Config::load();
        // The window opens at scale 1.0 (the compositor reports the real scale
        // after the first configure), so the body size starts at the logical
        // FONT_SIZE unless BNKNOTES_FONT_SIZE pins it.
        let body_size = config_font_px(SCALE_120_UNITY);
        let fonts = Fonts::with_config(
            &config.fonts,
            &font_sizes(body_size),
            config.line_height_scale,
        )?;
        // The saved layout: window/column sizes, tree root and open folders, and
        // each pane's last file with its caret and scroll. Missing or corrupt
        // falls back to the first-run default without failing.
        let session = store::load();

        // A CLI path is resolved up front, so `bnknotes .` roots at the real current
        // directory (named after its folder) instead of the literal ".", and matches
        // a configured vault that is the same directory reached another way.
        let arg = path.as_deref().map(config::resolve_path);
        let (root, cli_file) = resolve_vault(
            arg.as_deref(),
            session.active_vault.as_deref(),
            &config.vaults,
        );
        // The rail as it was last left, plus a temporary vault for a root the config
        // does not name, so the rail shows where the window actually is.
        let vaults = rail_vaults(config.vaults, &session.rail, &root);

        // Tabs and expanded folders travel with the vault, not the window: restore
        // the resolved vault's own layout, defaulting to a first-run one (no tabs,
        // nothing expanded) for a vault seen for the first time.
        let layout = session
            .vaults
            .get(&root)
            .cloned()
            .unwrap_or_else(|| store::VaultSession::new(root.clone()));

        // Editor A: a file argument opens as a single tab; otherwise restore the
        // vault's pane A tabs. Editor B always restores from the vault (one blank
        // tab if it had none).
        let editor_a = match &cli_file {
            Some(p) => restore_group(
                &[store::PaneSession {
                    path: Some(p.to_path_buf()),
                    ..Default::default()
                }],
                0,
                Some(root.as_path()),
                0,
            ),
            None => restore_group(
                &layout.panes[0].tabs,
                layout.panes[0].active as usize,
                Some(root.as_path()),
                0,
            ),
        };
        let editor_b = restore_group(
            &layout.panes[1].tabs,
            layout.panes[1].active as usize,
            Some(root.as_path()),
            1,
        );

        let expanded: BTreeSet<PathBuf> = layout.expanded.iter().cloned().collect();
        // The folder filter belongs to the vault, not the window: each vault reopens
        // showing what it showed last, and a vault seen for the first time lists
        // every folder.
        let show_empty_dirs = session.vaults.show_empty_dirs(&root);
        let tree = TreePane {
            tree: tree::Tree::new(root, expanded, show_empty_dirs),
            scroll_y: 0,
            content_height: 0,
            scrollbar: Scrollbar::default(),
            hovered: None,
            selected: None,
            filter_hovered: false,
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
            code_glyph_gamma: config.code_glyph_coverage_gamma,
            caret_config: config.caret,
            xkb: Xkb::new()?,
            editors: [editor_a, editor_b],
            tree,
            vaults,
            hovered_vault: None,
            drag_vault: None,
            icons: ImageStore::new(None),
            icon_size: 0,
            icon_seq: 0,
            menu: None,
            ai: None,
            ai_wake: ai::Wake::new()?,
            ai_enabled: config.ai.enabled,
            ai_provider: config.ai.provider,
            palette: None,
            search: search::Results::default(),
            last_vault: None,
            image_view: None,
            vault_prefs: session.vaults,
            panes,
            focus,
            last_editor: match focus {
                Focus::Editor(i) => i,
                Focus::Tree => 0,
            },
            hovered_tab: None,
            tab_scroll: [0, 0],
            tab_anchor: [(0, 0); 2],
            drag_splitter: None,
            drag_scroll: None,
            drag_tab: None,
            title: String::new(),
            title_scratch: String::new(),
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
            caret: CaretBlink::new(),
            autosave_at: None,
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
            watcher: FileWatcher::new(),
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

    /// Which vault in the rail the file tree is rooted at. Every root reaches the
    /// rail, a directory argument the config does not name as a temporary vault, so
    /// this resolves in practice; it stays an option rather than an assumption the
    /// rail's marker would have to trust.
    fn active_vault(&self) -> Option<usize> {
        let root = self.tree.tree.root();
        self.vaults.iter().position(|v| v.path == root)
    }

    /// The current column layout for this window size, below the top bar. A diff review
    /// solos its pane (see [`Self::diff_solo`]), hiding the sidebar and the other pane so
    /// the two diff columns get the full width.
    fn columns(&self) -> Columns {
        self.panes.columns(
            self.width as i32,
            self.height as i32,
            self.top_bar_h(),
            self.scale(),
            self.diff_solo(),
        )
    }

    /// The pane a diff review has taken over, if one is open, so the layout hides
    /// everything else while it is up. The first pane showing a diff wins (only one is
    /// the norm); a normal document layout returns once the review is accepted or
    /// rejected.
    fn diff_solo(&self) -> Option<usize> {
        (0..self.editors.len()).find(|&i| self.editors[i].active_is_diff())
    }

    /// The top bar's device-pixel height at the current body size and scale.
    fn top_bar_h(&self) -> i32 {
        top_bar_height(self.fonts.metrics(self.body_size), self.scale())
    }

    /// Reconcile each pane's tab-strip scroll with the current column layout: clamp
    /// it into range, and when the active tab or the tab count has changed since the
    /// last frame, slide it just far enough to bring the active tab into view. Free
    /// wheeling leaves the anchor unchanged, so the strip stays where the user
    /// parked it until they switch or open a tab.
    fn sync_tab_scroll(&mut self, cols: &Columns) {
        let scale = self.scale();
        for g in 0..self.editors.len() {
            let Some(strip) = cols.tab_strips[g] else {
                self.tab_scroll[g] = 0;
                self.tab_anchor[g] = (0, 0);
                continue;
            };
            let count = self.editors[g].tab_count();
            let active = self.editors[g].active_index();
            let tabs: Vec<_> = self.editors[g].tab_views().collect();
            let layout = crate::paint::tab_layout(strip, &tabs, scale);
            if self.tab_anchor[g] != (active, count) {
                // The run is variable-width, so the active tab's slot is the sum of
                // the widths before it, not `active * width`.
                let (start, size) = layout.tab_span(&tabs, active);
                self.tab_scroll[g] = scroll::reveal(
                    self.tab_scroll[g],
                    start,
                    size,
                    layout.avail,
                    layout.content,
                    0,
                );
                self.tab_anchor[g] = (active, count);
            }
            self.tab_scroll[g] = scroll::clamp(self.tab_scroll[g], layout.content, layout.avail);
        }
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
        if size != self.body_size
            && let Ok(fonts) =
                Fonts::with_config(&self.font_config, &font_sizes(size), self.line_height_scale)
        {
            self.fonts = fonts;
            self.body_size = size;
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

    /// Bring the focused editor's caret on screen: open every section hiding it,
    /// then scroll the minimum to reveal it, repainting if that scrolled.
    ///
    /// Opening comes first because a caret inside a collapsed section has nowhere on
    /// screen to scroll to. Every way a caret arrives at an offset the view did not
    /// pick (an edit, a search hit, a find match, an applied revision) ends here, so
    /// this is where the "the caret is never on a line a fold hides" invariant is
    /// restored.
    fn reveal_active_caret(&mut self) {
        let i = self.active_editor();
        let rect = self.editor_rect(i);
        self.editors[i].open_folds_at_cursor(&self.fonts, rect);
        if self.editors[i].scroll_caret_into_view(&self.fonts, rect) {
            // A keyboard scroll is a scroll: show the bar moving, the way the wheel does.
            self.flash_scrollbar(Focus::Editor(i));
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
            .ok_or_else(|| Error::msg("compositor lacks wl_compositor v4+"))?;
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

        // Name the GPU device under BNKNOTES_STATS, beside the per-frame stats it
        // belongs with. `--gpu-probe` reports the same thing without a window, so a
        // quiet startup costs nothing. The backend is mandatory (bring_up exits above
        // if it is missing), so it is present.
        if self.stats
            && let Some(gpu) = &self.presentation.backend
        {
            eprintln!("bnknotes: renderer: gpu ({})", gpu.name());
        }
        self.run_until(|s| s.closed)?;
        // The last word on the way out: whatever the idle window was still holding
        // goes to disk before the documents do, so closing the window is never how
        // the last sentence is lost.
        self.save_all();
        // Remember the layout for next launch: column sizes, tree root and open
        // folders, and each pane's file with its caret and scroll.
        self.save_session();
        Ok(())
    }

    /// The current layout as a persistable session (see [`crate::store`]). The
    /// active vault's own entry is folded in by [`Self::save_session`] first, so
    /// this only reads what is already remembered.
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
            active_vault: Some(self.tree.tree.root().to_path_buf()),
            vaults: self.vault_prefs.clone(),
            rail: rail_session(&self.vaults),
        }
    }

    /// Persist the layout, reporting (not failing) if the state file cannot be
    /// written; a lost layout must never take the editor down. The active vault's
    /// live tabs and expanded folders are folded back into its entry first, leaving
    /// every other vault's alone.
    ///
    /// Only the active vault's entry reaches the file: a save merges into what is
    /// already there (see [`crate::store::save`]), which is what keeps an instance
    /// on one vault from writing its stale idea of another vault over an instance
    /// live on that one. So a vault's state has to be saved while it is still the
    /// active one, which is why [`Self::switch_vault`] saves on the way out.
    fn save_session(&mut self) {
        self.vault_prefs.set_layout(
            self.tree.tree.root(),
            self.tree.tree.expanded_dirs(),
            [self.editors[0].to_session(), self.editors[1].to_session()],
        );
        if let Err(e) = store::save(&self.current_session()) {
            eprintln!("bnknotes: could not save layout: {e}");
        }
    }

    /// Switch the window to configured vault `i`: the file tree re-roots there and
    /// both panes reopen whatever that vault last had open. Each vault's tabs and
    /// expanded folders travel with it, so leaving one and coming back restores what
    /// it was showing, within the session as well as across launches. A click on the
    /// active vault's own chip changes nothing.
    pub(super) fn switch_vault(&mut self, i: usize) {
        let Some(root) = self.vaults.get(i).map(|v| v.path.clone()) else {
            return;
        };
        if root == self.tree.tree.root() {
            return;
        }
        // Where the window is leaving, for the palette to offer coming back to.
        self.last_vault = Some(self.tree.tree.root().to_path_buf());
        // Switching replaces both panes' documents wholesale, so the outgoing vault's
        // notes are written before its tabs are let go of.
        self.save_all();
        // Save the outgoing vault while it is still the active one: a save carries
        // only the active vault's entry to disk, so its tabs have to go now. This
        // also lands them in `vault_prefs`, where switching straight back finds them.
        self.save_session();
        let layout = self
            .vault_prefs
            .get(&root)
            .cloned()
            .unwrap_or_else(|| store::VaultSession::new(root.clone()));
        let show_empty_dirs = self.vault_prefs.show_empty_dirs(&root);
        let (scale, body_size) = (self.scale(), self.body_size);
        for (g, pane) in layout.panes.iter().enumerate() {
            self.editors[g] =
                restore_group(&pane.tabs, pane.active as usize, Some(&root), g as u64);
            // A restored group starts at scale 1.0, which is the truth only at
            // startup, before the compositor has reported one. This window already
            // knows its scale, so the new tabs take it now.
            self.editors[g].set_scale(scale, body_size);
        }
        self.tree.tree = tree::Tree::new(
            root,
            layout.expanded.iter().cloned().collect(),
            show_empty_dirs,
        );
        self.tree.scroll_y = 0;
        self.tree.selected = None;
        self.tree.hovered = None;
        // Transients that name a tab or a row of the vault being left. A picture
        // opened over the window is one of them: it belongs to a note that is no
        // longer open, and the store it was drawn from goes with the pane.
        self.close_image_view();
        self.tab_scroll = [0, 0];
        self.tab_anchor = [(0, 0); 2];
        self.drag_tab = None;
        self.drag_scroll = None;
        self.hovered_tab = None;
        self.focus = Focus::Editor(0);
        self.last_editor = 0;
        self.refresh_title();
        self.save_session();
        self.dirty = true;
    }

    /// Open the context menu for whatever is at window point `(px, py)`, the
    /// right-click. A spot with nothing to offer (which is most of the window today)
    /// opens no menu, and closes any that was up.
    pub(super) fn open_menu(&mut self, px: f32, py: f32) {
        // Hit-test with any open menu out of the way: a right-click while one is up
        // is aimed at what is under it, not at the menu itself.
        self.close_menu();
        let items = self.menu_items_at(px, py);
        if !items.is_empty() {
            self.menu = Some(Menu::new(items, (px as i32, py as i32)));
        }
        self.dirty = true;
    }

    /// What a right-click at `(px, py)` offers: a rail chip's icon actions, a tab's
    /// pin toggle, or an editor pane's editing commands and (when on) AI transforms.
    /// Anywhere else offers nothing, and so opens no menu.
    fn menu_items_at(&self, px: f32, py: f32) -> Vec<MenuItem> {
        match self.hit_test(px, py) {
            Hit::Vault(i) => self.vault_menu_items(i),
            Hit::Editor(pane, _) => self.editor_menu_items(pane),
            Hit::Tab(g, i) | Hit::TabClose(g, i) => self.tab_menu_items(g, i),
            _ => Vec::new(),
        }
    }

    /// The menu a right-click on tab `i` of pane `g` offers: pin it, or unpin it if
    /// it already is. A diff review is transient and re-homes to no document, so it
    /// cannot be pinned and offers nothing.
    fn tab_menu_items(&self, g: usize, i: usize) -> Vec<MenuItem> {
        if self.editors[g].tab_is_diff(i) {
            return Vec::new();
        }
        let label = if self.editors[g].is_pinned(i) {
            "Unpin tab"
        } else {
            "Pin tab"
        };
        vec![MenuItem {
            label: label.to_string(),
            action: MenuAction::TogglePin { pane: g, tab: i },
        }]
    }

    /// The indices into [`ai::ACTIONS`] offered for pane `i` right now: all of them
    /// when there is a selection, else the ones that do not require one (they fall
    /// back to the caret's paragraph or the whole note). Empty when AI actions are
    /// off. Shared by the context menu and the AI palette so they offer the same set.
    fn available_ai_actions(&self, i: usize) -> Vec<usize> {
        if !self.ai_enabled {
            return Vec::new();
        }
        let has_selection = self.editors[i].text.selection().is_some();
        ai::ACTIONS
            .iter()
            .enumerate()
            .filter(|(_, action)| has_selection || action.scope != ai::Scope::Selection)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// The AI commands offered for pane `i`, as `(label, command)` pairs shared by the
    /// context menu and the command palette so both offer the same set. Each label is
    /// prefixed with the configured provider, so it reads as an AI action and says
    /// which backend runs it. Empty when AI actions are off.
    fn ai_commands(&self, i: usize) -> Vec<(String, AiCommand)> {
        if !self.ai_enabled {
            return Vec::new();
        }
        let provider = self.ai_provider.label();
        let mut commands: Vec<(String, AiCommand)> = self
            .available_ai_actions(i)
            .into_iter()
            .filter_map(|idx| {
                ai::action(idx)
                    .map(|action| (format!("{provider}: {}", action.label), AiCommand::Run(idx)))
            })
            .collect();
        // A free-form prompt is always available: it auto-detects selection or whole
        // note, so it needs neither a selection nor a fixed scope.
        commands.push((format!("{provider}: Custom prompt…"), AiCommand::Custom));
        commands
    }

    /// What a right-click over editor pane `i` offers: the editing commands first,
    /// then the AI commands when they are on. The editing commands do not depend on
    /// the AI feature, so a pane holding a document always has a menu, and a new
    /// editing action is another row here rather than a second pop-up.
    fn editor_menu_items(&self, i: usize) -> Vec<MenuItem> {
        // A pane with no document has nothing to format; only the AI commands (which
        // fall back to the whole note) could apply, so leave the editing rows off.
        let mut items = if self.editors[i].has_tabs() {
            Self::edit_command_items(i)
        } else {
            Vec::new()
        };
        items.extend(
            self.ai_commands(i)
                .into_iter()
                .map(|(label, command)| MenuItem {
                    label,
                    action: match command {
                        AiCommand::Run(idx) => MenuAction::Ai(idx),
                        AiCommand::Custom => MenuAction::AiCustomPrompt,
                    },
                }),
        );
        items
    }

    /// The plain editing commands the editor context menu offers on pane `i`,
    /// independent of the AI feature. Today just formatting the caret's line, or a
    /// whole selection, as a task list (the menu twin of Ctrl+L).
    fn edit_command_items(i: usize) -> Vec<MenuItem> {
        vec![MenuItem {
            label: "Toggle task list".to_string(),
            action: MenuAction::ToggleTaskList { pane: i },
        }]
    }

    /// The icon actions a right-click offers on rail vault `i`: a picture for the
    /// chip, and, for a vault the config does not name, the action that makes it
    /// permanent.
    fn vault_menu_items(&self, i: usize) -> Vec<MenuItem> {
        let mut items = Vec::new();
        if self.is_temporary_vault(i) {
            items.push(MenuItem {
                label: "Save as vault".to_string(),
                action: MenuAction::SaveVault(i),
            });
        }
        let has_icon = self.vaults.get(i).is_some_and(|v| v.icon.is_some());
        // Offered only while there is a picture to take, so the item is never one that
        // would do nothing: snip a region with whatever tool puts a picture on the
        // clipboard, then right-click the chip. It names what it will use rather than
        // what it will do: "paste" at a chip that already looks like an icon reads as
        // pasting an icon, and the picture on the clipboard is the part in question.
        if self.clipboard_offers_image() {
            items.push(MenuItem {
                label: "Use image from clipboard".to_string(),
                action: MenuAction::PasteIcon(i),
            });
        }
        if has_icon {
            items.push(MenuItem {
                label: "Clear icon".to_string(),
                action: MenuAction::ClearIcon(i),
            });
        }
        items
    }

    /// Whether rail vault `i` is one the config does not name, the only thing the
    /// menu has an offer for.
    fn is_temporary_vault(&self, i: usize) -> bool {
        self.vaults
            .get(i)
            .is_some_and(|v| v.kind == VaultKind::Temporary)
    }

    /// Open the picture pane `i` has selected over the whole window, the double-click
    /// on it. A line that is no longer a picture (the source changed under the
    /// pointer) opens nothing.
    pub(super) fn open_image_view(&mut self, i: usize) {
        let Some(ls) = self.editors[i].selected_image else {
            return;
        };
        let Some(raw) = self.image_path_at(i, ls) else {
            return;
        };
        self.image_view = Some(ImageView { pane: i, raw });
        self.dirty = true;
    }

    /// Put the full-window picture away, reporting whether one was up.
    pub(super) fn close_image_view(&mut self) -> bool {
        let was_open = self.image_view.take().is_some();
        self.dirty |= was_open;
        was_open
    }

    /// Close the menu if one is open, reporting whether there was one. The caller
    /// repaints: the panel has to leave the frame it was drawn into.
    pub(super) fn close_menu(&mut self) -> bool {
        let was_open = self.menu.take().is_some();
        self.dirty |= was_open;
        was_open
    }

    /// Open the command palette over the window, or close an open one: Ctrl+K is the
    /// way in and the way back out. It is the generic command surface: the AI actions
    /// for the focused pane first, then the vaults to switch to. A curated set, not one
    /// entry per feature.
    pub(super) fn toggle_palette(&mut self) {
        if self.close_palette() {
            return;
        }
        let pane = self.active_editor();
        let mut rows: Vec<PaletteRow> = self
            .ai_commands(pane)
            .into_iter()
            .map(|(label, command)| PaletteRow {
                label,
                action: match command {
                    AiCommand::Run(idx) => PaletteAction::RunAi(idx),
                    AiCommand::Custom => PaletteAction::AiCustomPrompt,
                },
            })
            .collect();
        rows.extend(vault_rows(
            &self.vaults,
            self.active_vault(),
            self.last_vault.as_deref(),
        ));
        // A window with nothing to offer opens nothing rather than an empty panel.
        if rows.is_empty() {
            return;
        }
        self.palette = Some(Palette::new("Run a command", Rows::Ranked, rows));
        self.dirty = true;
    }

    /// Open the palette on the active vault's contents, or close an open one:
    /// Ctrl+Shift+F. It opens with no rows, since a search with no query has no
    /// answer.
    pub(super) fn toggle_search(&mut self) {
        if self.close_palette() {
            return;
        }
        self.search = search::Results::default();
        self.palette = Some(Palette::new("Search this vault", Rows::AsGiven, Vec::new()));
        self.dirty = true;
    }

    /// Answer the search palette's query again, after it changed.
    ///
    /// The scan is synchronous: a vault of a few hundred notes reads and scans in
    /// about a frame, and a query that only grew narrows to the notes its prefix
    /// already matched, so every keystroke after the first scans a handful of files.
    pub(super) fn refresh_search(&mut self) {
        let Some(palette) = &self.palette else {
            return;
        };
        let query = palette.query().to_string();
        let root = self.tree.tree.root().to_path_buf();
        // A tab's buffer is what is on screen, so a hit in a note with unsaved edits
        // has to come from the buffer, not from the file those edits have not reached.
        let open: Vec<(PathBuf, String)> = self
            .editors
            .iter()
            .flat_map(|group| group.docs())
            .filter(|doc| doc.text.is_modified())
            .filter_map(|doc| doc.path.clone().map(|p| (p, doc.text.text())))
            .collect();
        let notes = search::OpenNotes { notes: &open };
        let narrow = if self.search.can_narrow(&query) {
            search::Narrow::Files(&self.search.files)
        } else {
            search::Narrow::Vault
        };
        let results = search::search(&root, &query, &notes, narrow);
        let rows = search_rows(&results, &root);
        let note = search_note(&results, &query);
        self.search = results;
        if let Some(palette) = &mut self.palette {
            palette.set_rows(rows);
            palette.set_note(note);
        }
        self.dirty = true;
    }

    /// Close the palette if one is open, reporting whether there was one.
    pub(super) fn close_palette(&mut self) -> bool {
        let was_open = self.palette.take().is_some();
        self.dirty |= was_open;
        was_open
    }

    /// Run a palette row's action and close the palette behind it.
    pub(super) fn run_palette_action(&mut self, action: PaletteAction) {
        self.close_palette();
        match action {
            PaletteAction::SwitchVault(i) => self.switch_vault(i),
            PaletteAction::OpenMatch { path, offset } => self.open_match(&path, offset),
            PaletteAction::RunAi(index) => self.start_ai_action(index),
            PaletteAction::AiCustomPrompt => self.open_ai_prompt(),
        }
    }

    /// Open the palette as a single free-form field for a custom AI instruction. The
    /// scope is decided when it runs: the selection if there is one, else the whole
    /// note. Reached from the command palette's "Custom prompt" row and the context
    /// menu's item.
    pub(super) fn open_ai_prompt(&mut self) {
        if !self.ai_enabled {
            return;
        }
        self.close_palette();
        self.palette = Some(Palette::query_input("Describe the change to make"));
        self.dirty = true;
    }

    /// Open `path` and select the match at `offset`, the way the find bar lands on
    /// one: the caret goes there and the view scrolls to show it.
    fn open_match(&mut self, path: &Path, offset: usize) {
        self.open_file(path);
        let i = self.active_editor();
        // The hit was measured against the note as it was scanned; the buffer it lands
        // in is that same text (an open tab was searched as its buffer), but a note
        // that changed under the scan could still be shorter, so the offset is clamped
        // rather than trusted.
        let end = self.editors[i].text.rope().len();
        let at = offset.min(end);
        self.editors[i].text.set_cursor(at, false);
        self.bump_caret();
        self.reveal_active_caret();
        self.dirty = true;
    }

    /// Run a menu item's action and close the menu behind it.
    pub(super) fn run_menu_action(&mut self, action: MenuAction) {
        self.close_menu();
        match action {
            MenuAction::SaveVault(i) => self.save_vault(i),
            MenuAction::PasteIcon(i) => self.paste_icon(i),
            MenuAction::ClearIcon(i) => self.clear_icon(i),
            MenuAction::ToggleTaskList { pane } => self.toggle_task_list_in_pane(pane),
            MenuAction::Ai(i) => self.start_ai_action(i),
            MenuAction::AiCustomPrompt => self.open_ai_prompt(),
            MenuAction::TogglePin { pane, tab } => self.toggle_pin_tab(pane, tab),
        }
    }

    /// Format pane `pane`'s caret line, or its whole selection, as a task list, the
    /// context-menu twin of Ctrl+L. Focus the pane first so the caret revealed after
    /// the edit is the one that changed, then run the after-edit bookkeeping only
    /// when the buffer actually moved (an all-blank selection is a no-op).
    fn toggle_task_list_in_pane(&mut self, pane: usize) {
        self.focus_pane(pane);
        if self.editors[pane].text.toggle_task_list() {
            self.after_edit();
        }
    }

    /// Pin or unpin tab `i` of pane `g` and remember it. A pinned tab keeps its
    /// place, wears a pin badge in its close slot, and is guarded against the click
    /// and the Ctrl+W that would otherwise close it.
    pub(super) fn toggle_pin_tab(&mut self, g: usize, i: usize) {
        self.editors[g].toggle_pin(i);
        self.save_session();
        self.dirty = true;
    }

    /// Start built-in AI action `index` on the focused pane.
    fn start_ai_action(&mut self, index: usize) {
        if let Some(action) = ai::action(index) {
            self.start_ai(action.instruction, action.label, action.scope);
        }
    }

    /// Start a free-form AI action from the prompt the user typed. The scope is the
    /// selection when there is one, else the whole note (auto-detected in
    /// [`Self::start_ai`] via [`ai::Scope::Document`]).
    fn start_ai_custom(&mut self, instruction: String) {
        if instruction.trim().is_empty() {
            return;
        }
        self.start_ai(&instruction, "Custom prompt", ai::Scope::Document);
    }

    /// Spawn the configured provider running `instruction` (titled `label`) over the
    /// focused pane's `scope` region, and put up the "thinking" preview. A selection,
    /// when present, always wins as the subject; otherwise the scope's region is used,
    /// and a selection-only action with nothing selected does nothing. Nothing reaches
    /// the buffer here; the answer lands through the diff preview once it arrives, and
    /// a spawn that fails shows the failure in its place.
    fn start_ai(&mut self, instruction: &str, label: &'static str, scope: ai::Scope) {
        let pane = self.active_editor();
        let span = {
            let text = &self.editors[pane].text;
            match text.selection() {
                Some(selection) => selection,
                None => match scope {
                    ai::Scope::Selection => return,
                    ai::Scope::Paragraph => text.paragraph_at(text.cursor()),
                    ai::Scope::Document => (0, text.rope().len()),
                },
            }
        };
        if span.0 >= span.1 {
            return;
        }
        let original = self.editors[pane].text.rope().byte_slice(span.0..span.1);
        let version = self.editors[pane].text.version();
        // The document the answer applies back to, pinned now so a tab switch while the
        // tool thinks cannot retarget the edit.
        let source = self.editors[pane].id;
        self.ai = Some(
            match ai::Job::spawn(
                self.ai_provider,
                instruction,
                &original,
                self.ai_wake.write_fd(),
            ) {
                Ok(job) => Preview::Thinking {
                    pane,
                    source,
                    span,
                    version,
                    original,
                    label,
                    job,
                },
                Err(e) => Preview::Failed {
                    label,
                    message: e.to_string(),
                },
            },
        );
        self.dirty = true;
    }

    /// A finished (or failed) AI job woke the loop: drain the wake pipe and, when the
    /// running job has its answer, open a diff tab to review it or show the failure. A
    /// spurious wake (no result yet) leaves the modal untouched.
    fn poll_ai(&mut self) {
        self.ai_wake.drain();
        if !matches!(self.ai, Some(Preview::Thinking { .. })) {
            return;
        }
        let Some(Preview::Thinking {
            pane,
            source,
            span,
            version,
            original,
            label,
            mut job,
        }) = self.ai.take()
        else {
            return;
        };
        match job.take_outcome() {
            None => {
                // Still running: put the modal back as it was.
                self.ai = Some(Preview::Thinking {
                    pane,
                    source,
                    span,
                    version,
                    original,
                    label,
                    job,
                });
            }
            Some(ai::Outcome::Revised(revised)) => {
                // The answer is a side-by-side diff in its own tab. Nothing massages
                // it: whatever the tool returned is diffed as-is against the selection.
                let view =
                    crate::pane::DiffView::new(source, span, version, original, revised, label);
                self.editors[pane].open_diff(view);
                self.focus = Focus::Editor(pane);
                self.ai = None;
                self.dirty = true;
            }
            Some(ai::Outcome::Failed(message)) => {
                self.ai = Some(Preview::Failed { label, message });
                self.dirty = true;
            }
        }
    }

    /// Accept the active diff review: splice its revision over the source span as one
    /// undo step, select the result so the change is visible, focus the source, and
    /// close the review tab. A note that changed underneath (a disk reload, or a fresh
    /// edit, while the review was up) fails the version check and is left untouched; a
    /// source whose tab was closed has nowhere to land, so the review just closes.
    pub(super) fn accept_diff(&mut self) {
        let focus = self.active_editor();
        let Some((source, span, version, revised)) = self.editors[focus].active_diff().map(|d| {
            let (s, sp, v, r) = d.apply();
            (s, sp, v, r.to_string())
        }) else {
            return;
        };
        self.dirty = true;
        // Close the review first, so focusing the source next does not fight it for the
        // active tab. The source is found by id afterwards, robust to the index shift.
        self.editors[focus].close_active();
        // Find the source wherever its tab now sits (this pane or the other), apply only
        // if it still matches the version the span was captured at, then focus it.
        for g in 0..self.editors.len() {
            let Some(doc) = self.editors[g].doc_by_id_mut(source) else {
                continue;
            };
            if doc.text.version() == version && doc.text.replace_ranges(&[span], &revised) {
                doc.text.select_range(span.0, span.0 + revised.len());
            }
            self.editors[g].focus_doc(source);
            self.focus = Focus::Editor(g);
            self.bump_caret();
            self.reveal_active_caret();
            break;
        }
    }

    /// Reject the active diff review: close its tab, leaving the source untouched.
    pub(super) fn reject_diff(&mut self) {
        let focus = self.active_editor();
        if self.editors[focus].active_is_diff() {
            self.editors[focus].close_active();
            self.dirty = true;
        }
    }

    /// Open the demonstration diff review (the `--demo-diff` flag): a crafted before and
    /// after that exercises every diff concept in one tab, so the view can be seen and
    /// tuned without running a real AI action. It targets pane A's current document, so
    /// accept (Enter) splices the revision into it and reject (Escape) just closes; the
    /// content is for looking at, not for keeping.
    fn open_demo_diff(&mut self) {
        let (old, new) = demo_diff_texts();
        let source = self.editors[0].id;
        let version = self.editors[0].text.version();
        let view = crate::pane::DiffView::new(source, (0, 0), version, old, new, "Diff demo");
        self.editors[0].open_diff(view);
        self.focus = Focus::Editor(0);
        self.dirty = true;
    }

    /// Dismiss the AI modal (a still-running job, canceled by dropping it, or a shown
    /// failure). Reports whether one was up, for the caller's repaint.
    pub(super) fn close_ai(&mut self) -> bool {
        let was_open = self.ai.take().is_some();
        self.dirty |= was_open;
        was_open
    }

    /// Put the clipboard's picture on vault `i`'s chip, the menu's "Use image from
    /// clipboard".
    ///
    /// The clipboard is the whole interface: any tool that snips a region of the
    /// screen chooses the picture and frames it, which is a better judge of what a
    /// 34px square should show than anything here could be. The chip shows the middle
    /// square of what was pasted, since that is what the snip was aimed at.
    pub(super) fn paste_icon(&mut self, i: usize) {
        let bytes = match self.clipboard_image() {
            Ok(Some(bytes)) => bytes,
            // The menu only offers this while the clipboard holds a picture, so
            // there is nothing to say when it does not: the clipboard changed under
            // the open menu, and the chip stays as it was.
            Ok(None) => return,
            Err(e) => {
                eprintln!("bnknotes: could not read the clipboard: {e}");
                return;
            }
        };
        let Some(vault) = self.vaults.get(i) else {
            return;
        };
        let pasted = match icon::paste(&vault.path.clone(), &bytes) {
            Ok(icon) => icon,
            Err(e) => {
                eprintln!("bnknotes: could not set the icon: {e}");
                return;
            }
        };
        // The picture being replaced is nobody's now: its file goes with it, so the
        // icons folder holds the icons in use and nothing else.
        let stale = self.vaults.get(i).and_then(|v| v.icon.clone());
        if let Some(vault) = self.vaults.get_mut(i) {
            vault.set_icon(Some(pasted));
        }
        if let Some(stale) = stale {
            self.drop_icon_file(i, &stale);
        }
        self.save_session();
        self.dirty = true;
    }

    /// Delete `stale`'s file, unless vault `i` still points at it (a picture pasted
    /// over itself writes the same file, and deleting it would leave a chip pointing
    /// at nothing).
    fn drop_icon_file(&self, i: usize, stale: &icon::VaultIcon) {
        let live = self
            .vaults
            .get(i)
            .and_then(|v| v.icon.as_ref())
            .is_some_and(|icon| icon.source == stale.source);
        if !live {
            icon::discard(stale);
        }
    }

    /// Take vault `i`'s picture away, the menu's "Clear icon": the chip goes back to
    /// the color and letters it had before, and the picture's file goes with it.
    pub(super) fn clear_icon(&mut self, i: usize) {
        let Some(vault) = self.vaults.get_mut(i) else {
            return;
        };
        let Some(stale) = vault.icon.clone() else {
            return;
        };
        let keys = vault.icon_keys.clone();
        vault.set_icon(None);
        self.drop_icon_bitmaps(keys);
        self.drop_icon_file(i, &stale);
        self.save_session();
        self.dirty = true;
    }

    /// Bake whatever chip icons are missing, and re-bake all of them when the chip
    /// size has changed under them (a display-scale change).
    ///
    /// Called once per frame build. In the steady state it is a walk of the chips
    /// comparing two fields, and allocates nothing: only an icon that was just rolled
    /// (or a scale that just moved) reaches the decoder.
    fn sync_icons(&mut self, side: i32) {
        if side <= 0 {
            return;
        }
        if self.icon_size != side {
            // The chips changed size, so every baked picture is the wrong one: drop
            // the keys and let the loop below bake them again at the new size.
            self.icon_size = side;
            let stale: Vec<Option<IconKeys>> =
                self.vaults.iter_mut().map(|v| v.icon_keys.take()).collect();
            for keys in stale {
                self.drop_icon_bitmaps(keys);
            }
        }
        for i in 0..self.vaults.len() {
            let Some(vault) = self.vaults.get(i) else {
                continue;
            };
            if vault.icon.is_none() || vault.icon_keys.is_some() {
                continue;
            }
            let Some(icon) = vault.icon.clone() else {
                continue;
            };
            let radius = self.scale().px(RAIL_RADIUS);
            let Some((rest, hover)) = bake_icon(&icon, side, radius) else {
                // The picture is gone or no longer decodes: the chip keeps its color,
                // and the icon goes rather than being retried every frame.
                if let Some(vault) = self.vaults.get_mut(i) {
                    vault.set_icon(None);
                }
                continue;
            };
            let keys = IconKeys {
                rest: self.next_icon_key(),
                hover: self.next_icon_key(),
            };
            self.icons.insert(keys.rest.clone(), rest);
            self.icons.insert(keys.hover.clone(), hover);
            if let Some(vault) = self.vaults.get_mut(i) {
                vault.icon_keys = Some(keys);
            }
        }
    }

    /// A key nothing has drawn under, for a bitmap about to be baked. A counter, so a
    /// picture that replaced another is a key the damage diff and the renderer both
    /// see as new, which a name derived from the vault would hide.
    fn next_icon_key(&mut self) -> String {
        self.icon_seq += 1;
        format!("vault-icon-{}", self.icon_seq)
    }

    /// Drop the bitmaps `keys` names, once nothing draws them.
    fn drop_icon_bitmaps(&mut self, keys: Option<IconKeys>) {
        let Some(keys) = keys else {
            return;
        };
        self.icons.remove(&keys.rest);
        self.icons.remove(&keys.hover);
    }

    /// Save temporary vault `i` into the config file, the rail menu's one item: its
    /// chip stops being an outline and it is in the rail at every launch from here.
    ///
    /// The chip keeps its place and its color, since a temporary vault already sits
    /// where the config will name it, last. A config that cannot be written leaves
    /// the vault temporary and says so, rather than showing a saved chip the next
    /// launch would not have.
    pub(super) fn save_vault(&mut self, i: usize) {
        if !self.is_temporary_vault(i) {
            return;
        }
        let Some(vault) = self.vaults.get_mut(i) else {
            return;
        };
        if let Err(e) = config::add_vault(&vault.path) {
            eprintln!("bnknotes: could not save vault: {e}");
            return;
        }
        vault.kind = VaultKind::Saved;
        // Now that it is a chip the rail keeps, where it sits is worth remembering.
        self.save_session();
        self.dirty = true;
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
            self.fire_scrollbar_anim();
            // After the events this pass carried are in the buffers: an edit among
            // them pushes the idle window back, and the pass the window runs out on
            // writes. Both cost a version compare per open tab when nothing is
            // pending, which is the idle case.
            let now = Instant::now();
            self.poll_autosave(now);
            self.fire_autosave(now);
            // Pace to the compositor's clock: only draw when no frame callback is
            // outstanding, so a burst of edits or held keys between refreshes
            // collapses into a single repaint instead of racing both buffers.
            if self.configured && self.dirty && self.frame_callback == 0 && self.render_frame()? {
                self.dirty = false;
            }
            if done(self) {
                return Ok(());
            }
            // Keep the file watches aligned with the open tabs (cheap when
            // unchanged), then wait on the compositor socket and the watcher fd.
            self.sync_watches();
            self.conn.flush()?;
            let timeout = self.next_wake();
            self.wait_for_events(timeout)?;
        }
    }

    /// Align the set of watched directories with the open tabs. The steady state
    /// (no tab opened or closed since last time) allocates nothing.
    fn sync_watches(&mut self) {
        if self
            .watcher
            .watches_exactly(self.editors.iter().flat_map(PaneGroup::paths))
        {
            return;
        }
        self.watcher
            .sync(self.editors.iter().flat_map(PaneGroup::paths));
    }

    /// Wait for the compositor socket or the file watcher to become ready, up to
    /// `timeout` (or indefinitely when `None`). A socket wake pulls the pending
    /// bytes; a watcher wake re-checks the open files against disk. A pure timeout
    /// returns so the caller can service its timers.
    fn wait_for_events(&mut self, timeout: Option<Duration>) -> Result<()> {
        let conn_fd = self.conn.fd();
        let ai_fd = self.ai_wake.read_fd();
        // Bit 0 is the socket, bit 1 the AI wake pipe, bit 2 the watcher (present
        // only when the watcher opened its fd); a fixed array so an idle poll does
        // not allocate.
        let mut fds = [conn_fd, ai_fd, -1];
        let n = match self.watcher.fd() {
            Some(watch_fd) => {
                fds[2] = watch_fd;
                3
            }
            None => 2,
        };
        let ready = ffi::poll_readable(&fds[..n], timeout)?;
        if ready & 0b001 != 0 {
            self.pull_socket()?;
        }
        if ready & 0b010 != 0 {
            self.poll_ai();
        }
        if n == 3 && ready & 0b100 != 0 {
            self.handle_watch_events();
        }
        Ok(())
    }

    /// Read whatever the compositor has queued. Called only when poll reports the
    /// socket readable, so the recv returns at once; a zero-length read means the
    /// compositor closed the connection.
    fn pull_socket(&mut self) -> Result<()> {
        if let Fill::Bytes(0) = self.conn.fill(Some(Duration::from_millis(1)))? {
            return Err(Error::msg("compositor closed the connection"));
        }
        Ok(())
    }

    /// Drain the file watcher and, when something changed on disk, re-check every
    /// open tab: silently reload the clean ones, flag a conflict on the edited
    /// ones, and repaint.
    fn handle_watch_events(&mut self) {
        if !self.watcher.drain() {
            return;
        }
        // A watch the kernel dropped (a directory removed then recreated) re-arms
        // on the next reconcile.
        self.sync_watches();
        let mut changed = false;
        for group in &mut self.editors {
            changed |= group.recheck_disk();
        }
        if changed {
            self.dirty = true;
            self.refresh_title();
        }
    }

    /// Compose the whole window's display list: the vault rail, the file tree, then
    /// each visible editor pane built into its column, then the splitter bars.
    /// Refreshes each pane's layout and clamps its scroll first, so the list matches
    /// the current column sizes. Left to right, so an editor's background covers any
    /// overhanging filename; splitters last so they cover the inter-column gaps.
    fn build_frame_list(&mut self) -> Rc<DisplayList> {
        let cols = self.columns();
        self.sync_tab_scroll(&cols);
        // Bake any chip picture the rail is about to ask for, at the size the chips
        // are this frame.
        let chip_side = crate::paint::rail_chip_side(cols.rail_chips, self.scale());
        self.sync_icons(chip_side);
        let now = Instant::now();
        // The frame before this one is the best estimate of this one's size, so the
        // whole window composes into a single allocation that stops growing after the
        // first few frames.
        let mut list: DisplayList = Vec::with_capacity(self.presentation.present_list.len());
        push_rail(
            &mut list,
            &BarCtx::new(&self.fonts, self.scale(), self.body_size),
            cols.rail,
            cols.rail_chips,
            &RailView {
                vaults: &self.vaults,
                active: self.active_vault(),
                hovered: self.hovered_vault,
                // A press that never crossed the threshold is a click, not a drag, so
                // it lifts nothing.
                drag: self.drag_vault.filter(|d| d.moved).map(|d| RailDragView {
                    vault: d.vault,
                    top: (self.pointer_y - d.grab_dy as f32).round() as i32,
                }),
            },
        );
        if let Some(rect) = cols.tree {
            let (scale, body_size) = (self.scale(), self.body_size);
            let m = tree_metrics(&self.fonts, body_size);
            // Measure before painting, so the tree's scroll clamp and its scrollbar
            // both see the rows this frame actually draws.
            measure_tree(&mut self.tree, rect, m, scale);
            let rows_h = self.tree_column(rect).rows.h;
            let tree_h = self.tree.content_height;
            tick_scrollbar(&mut self.tree.scrollbar, tree_h, rows_h, now);
            push_tree(&mut list, &self.fonts, &self.tree, rect, scale, body_size);
        }
        for i in 0..self.editors.len() {
            let Some(rect) = cols.editors[i] else {
                continue;
            };
            // An empty pane (no tabs open) has no content and no caret to paint, but
            // it still has a surface: without one the window shows whatever the last
            // frame left in that column, and the sidebar's long names (which are cut
            // by the pane covering them, not by the sidebar) run on into it.
            if !self.editors[i].has_tabs() {
                list.push(DrawCmd::Fill {
                    rect,
                    color: BACKGROUND,
                });
                continue;
            }
            // A diff review owns its pane's body: the two side-by-side columns with
            // their word highlights, scrolling together, in place of a document. Its
            // scroll lives on the view; the column scrollbar rides the taller side.
            if self.editors[i].active_is_diff() {
                let (scale, body_size) = (self.scale(), self.body_size);
                let dc = diff_columns(rect, scale, body_size, &self.fonts);
                let Some(dv) = self.editors[i].active_diff_mut() else {
                    continue;
                };
                dv.ensure_layouts(&self.fonts, &dc, scale, body_size);
                dv.scroll_y = scroll::clamp(dv.scroll_y, dv.content_height(), rect.h);
                let content_h = dv.content_height();
                tick_scrollbar(&mut self.editors[i].scrollbar, content_h, rect.h, now);
                let Some(dv) = self.editors[i].active_diff() else {
                    continue;
                };
                push_diff_view(
                    &mut list,
                    &self.fonts,
                    rect,
                    dv,
                    scale,
                    body_size,
                    &self.editors[i].scrollbar,
                );
                continue;
            }
            self.editors[i].ensure_layout(&self.fonts, rect);
            let ch = self.editors[i].content_height;
            self.editors[i].scroll_y = scroll::clamp(self.editors[i].scroll_y, ch, rect.h);
            tick_scrollbar(&mut self.editors[i].scrollbar, ch, rect.h, now);
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
            let hovering = over && !ed.dragging() && self.drag_splitter.is_none();
            let table_hover = if hovering {
                ed.layout_cache.as_ref().and_then(|l| {
                    l.table_button_at(&self.fonts, ed.text.rope(), px, ed.content_y(py))
                })
            } else {
                None
            };
            // The fold chevron under the pointer, recomputed the same way, so it
            // cannot go stale after a scroll or an edit moved the heading.
            let fold_hover = hovering
                .then(|| {
                    ed.layout_cache
                        .as_ref()
                        .and_then(|l| l.fold_at(&self.fonts, px, ed.content_y(py), self.body_size))
                })
                .flatten();
            let params = PaintParams {
                view: rect,
                scale: self.scale(),
                body_size: self.body_size,
                scroll_y: ed.scroll_y,
                content_height: ed.content_height,
                scrollbar: &ed.scrollbar,
                selection: ed.text.selection(),
                caret,
                find: build_find_paint(&ed.find, &tick_tops),
                table_hover,
                fold_hover,
                diff: None,
            };
            if let Some(layout) = ed.layout_cache.as_ref() {
                build_display_list_into(&mut list, &self.fonts, layout, ed.text.rope(), &params);
            }
            // The disk-conflict banner floats over the pane's top edge, so it goes
            // after the content. Only the active tab's conflict is shown here.
            if let Some(kind) = ed.conflict {
                let banner = conflict_banner_layout(rect, kind, self.body_size, &self.fonts);
                let hover = banner.button_at(self.pointer_x, self.pointer_y);
                push_conflict_banner(&mut list, &self.fonts, &banner, kind, hover, self.body_size);
            }
        }
        let scale = self.scale();
        // The column splitters, kept (Copy) so the top bar can continue their
        // divider lines up through the reserved strip.
        let splitter_rects = cols.splitters;
        for s in cols.splitters.into_iter().flatten() {
            push_splitter(&mut list, s, scale);
        }
        // The top bar: the active vault's name above the sidebar and a tab strip
        // above each pane, segmented by the continued column dividers. Drawn last;
        // it occupies the reserved strip above every column.
        let top = self.top_bar_h();
        if top > 0 {
            let bar = cols.bar;
            let views = [
                self.editors[0].tab_views().collect::<Vec<_>>(),
                self.editors[1].tab_views().collect::<Vec<_>>(),
            ];
            let strip = |g: usize| {
                cols.tab_strips[g].map(|rect| TabStrip {
                    rect,
                    tabs: &views[g],
                    active: self.editors[g].active_index(),
                    hovered: self.hovered_tab.and_then(|h| h.tab_in(g)),
                    close_hover: self.hovered_tab.and_then(|h| h.close_in(g)),
                    new_hover: self.hovered_tab == Some(TabHover::New { pane: g }),
                    scroll_x: self.tab_scroll[g],
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
            // The vault label sits above the sidebar, sharing the tree column's
            // x-span; when the sidebar is hidden there is no vault region.
            let name = vault_name(self.tree.tree.root());
            let vault = cols.tree.map(|t| {
                let region = Rect {
                    x: t.x,
                    y: 0,
                    w: t.w,
                    h: top,
                };
                (region, name)
            });
            push_top_bar(&mut list, &ctx, bar, vault, [strip(0), strip(1)], &dividers);
        }
        // A picture opened over the window covers all of it, chrome included: it is
        // the one thing being looked at, so everything else gets out of the way.
        if let Some(view) = &self.image_view
            && let Some(size) = self.editors[view.pane].images.dimensions(&view.raw)
        {
            push_image_view(
                &mut list,
                (self.width as i32, self.height as i32),
                size,
                &view.raw,
                scale,
            );
        }
        // The context menu floats over everything, so it is pushed after everything.
        if let Some(menu) = &self.menu {
            let ctx = BarCtx::new(&self.fonts, scale, self.body_size);
            push_menu(&mut list, &ctx, menu, &self.menu_layout(menu));
        }
        // The palette floats over the menu too: it takes the keyboard, and a menu
        // cannot be open under it (opening one closes any that was up).
        if let Some(palette) = &self.palette {
            let ctx = BarCtx::new(&self.fonts, scale, self.body_size);
            push_palette(&mut list, &ctx, palette, &self.palette_layout(palette));
        }
        // The AI modal (still-working or failed) floats over everything and holds the
        // keyboard until dismissed. A successful answer is not here: it opened a diff
        // tab, which the pane loop above already drew.
        if let Some(preview) = &self.ai {
            let ctx = BarCtx::new(&self.fonts, scale, self.body_size);
            push_ai_preview(
                &mut list,
                &ctx,
                preview,
                (self.width as i32, self.height as i32),
            );
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
        let visible = self.caret.visible();
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

    /// Every scrollbar in the window: the sidebar's and one per content pane.
    fn scrollbars(&self) -> impl Iterator<Item = &Scrollbar> {
        [&self.tree.scrollbar, &self.editors[0].scrollbar]
            .into_iter()
            .chain([&self.editors[1].scrollbar])
    }

    /// Whether any scrollbar is still fading or growing, so the next compositor frame
    /// should carry it on.
    fn scrollbars_animating(&self) -> bool {
        self.scrollbars().any(Scrollbar::animating)
    }

    /// When the loop should next tick the scrollbars: the soonest deadline any of them
    /// wants, whether that is the next step of a fade or the moment a lit bar's hold
    /// lapses and it starts fading.
    fn scrollbar_retry_at(&self) -> Option<Instant> {
        self.scrollbars().filter_map(Scrollbar::retry_at).min()
    }

    /// The scrollbars' counterpart to [`Self::fire_caret_anim`]: a fade step whose
    /// quantized color did not change presents nothing and arms no frame callback, so
    /// past a bar's deadline mark a repaint and let the next frame tick it on.
    fn fire_scrollbar_anim(&mut self) {
        if self.frame_callback != 0 {
            return;
        }
        if self
            .scrollbar_retry_at()
            .is_some_and(|at| at <= Instant::now())
        {
            self.dirty = true;
        }
    }

    fn handle(&mut self, msg: Message) -> Result<()> {
        let mut r = Reader::new(msg.body());

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
            // Keep the animations going: if the caret is still easing or a scrollbar
            // still fading, request the next frame now that the compositor is ready.
            if self.drawable_caret_anim().is_some_and(CaretAnim::in_flight)
                || self.scrollbars_animating()
            {
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
            // An offer lists every flavor its source can produce, so a screenshot's
            // image/png and a browser's image-plus-markup both land here; which one a
            // paste takes is decided at the paste (see `State::paste`). The first
            // image type we accept is kept unless the preferred one turns up later.
            if let Some(image) = IMAGE_MIMES.iter().find(|m| **m == mime) {
                let have_preferred =
                    self.clipboard.incoming_image_mime.as_deref() == Some(IMAGE_MIMES[0]);
                if !have_preferred {
                    self.clipboard.incoming_image_mime = Some((*image).to_string());
                }
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
            protocol::IFACE_COMPOSITOR if version >= protocol::VERSION_COMPOSITOR => {
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
            // The drm_syncobj timeline manager for explicit-sync present: the GPU
            // path waits on and signals timeline points through it, falling back to
            // the dmabuf's implicit fences where a compositor does not offer it.
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

    /// Common bookkeeping after a buffer edit: request a redraw, bring the caret on
    /// screen (which opens any section the edit wrote into), and refresh the title's
    /// modified indicator.
    fn after_edit(&mut self) {
        self.dirty = true;
        self.bump_caret();
        self.reveal_active_caret();
        self.refresh_title();
    }

    /// Save the focused document now, the Ctrl+S. The autosave already writes it
    /// seconds after the typing stops, so this is the "now, and I mean it" (and the
    /// way to force a write while the idle window is still running).
    ///
    /// A buffer with no file (a blank tab) has nowhere to go yet, and this is the
    /// one place a user asked in so many words, so it is the one place that says so.
    fn save(&mut self) -> Result<()> {
        let i = self.active_editor();
        let saved = self.editors[i].active_doc_mut().save()?;
        if saved == Saved::NoPath {
            eprintln!("bnknotes: no file to save to; start bnknotes with a path");
        }
        self.after_save(saved);
        Ok(())
    }

    /// Save every open document with unsaved edits, across both panes and all their
    /// tabs, and report whether anything reached the disk.
    ///
    /// Everything that lets a document go calls this: the idle autosave, a tab
    /// closing, a vault switching, the window shutting, the focus leaving. A
    /// background tab can hold unsaved edits (typing in one pane while the other is
    /// focused, a replace-all across tabs), and the point of an autosave is that no
    /// document is ever the one that quietly did not get written.
    fn save_all(&mut self) -> bool {
        let mut wrote = false;
        for group in &mut self.editors {
            for doc in group.docs_mut() {
                match doc.save() {
                    Ok(saved) => wrote |= saved == Saved::Written,
                    // One unwritable document must not stop the rest from being
                    // saved, nor take the editor down on the way out of a vault.
                    Err(e) => eprintln!("bnknotes: could not save: {e}"),
                }
            }
        }
        if wrote {
            self.after_save(Saved::Written);
        }
        wrote
    }

    /// Settle the window after a save: a conflict raised by one needs painting, and
    /// a write clears the title's modified marker.
    fn after_save(&mut self, saved: Saved) {
        if matches!(saved, Saved::Written | Saved::Conflict) {
            self.dirty = true;
            self.refresh_title();
        }
    }

    /// Re-time the idle autosave against what the documents look like now.
    ///
    /// Called once per loop pass, so it watches the buffers rather than the edits:
    /// every mutation reaches it (a keystroke, a paste, a replace-all, a table edit)
    /// without each having to arm the timer itself.
    fn poll_autosave(&mut self, now: Instant) {
        let mut edited = false;
        let mut pending = false;
        for group in &mut self.editors {
            for doc in group.docs_mut() {
                edited |= doc.poll_edited();
                pending |= doc.text.is_modified() && doc.path.is_some();
            }
        }
        self.autosave_at = autosave_deadline(edited, pending, self.autosave_at, now);
    }

    /// Write the pending documents once the typing has stopped for [`AUTOSAVE_IDLE`].
    fn fire_autosave(&mut self, now: Instant) {
        if self.autosave_at.is_some_and(|at| now >= at) {
            self.autosave_at = None;
            self.save_all();
        }
    }

    /// Write the toplevel title into the scratch buffer: the focused editor's file
    /// name (or just the app name for an unnamed buffer), with a leading dot when it
    /// has unsaved changes. Formatting into a buffer the state keeps means the
    /// per-keystroke refresh below compares without allocating.
    fn write_window_title(&mut self) {
        use std::fmt::Write as _;
        let pane = &self.editors[self.active_editor()];
        let mut title = std::mem::take(&mut self.title_scratch);
        title.clear();
        if pane.has_tabs() {
            if pane.text.is_modified() {
                title.push_str("• ");
            }
            match pane.path.as_ref().and_then(|p| p.file_name()) {
                Some(name) => {
                    let _ = write!(title, "{} - bnknotes", name.to_string_lossy());
                }
                None => title.push_str("bnknotes"),
            }
        } else {
            title.push_str("bnknotes");
        }
        self.title_scratch = title;
    }

    /// Send the toplevel title only if it changed, so the modified indicator and
    /// filename stay current without re-setting an identical string each frame.
    /// The title tracks the active pane, its file, and its modified flag, so every
    /// focus change routes through `focus_pane` to call this; the compare then
    /// dedupes the protocol request on the unchanged common path.
    fn refresh_title(&mut self) {
        self.write_window_title();
        if self.title_scratch == self.title {
            return;
        }
        self.conn.request(
            self.toplevel,
            xdg_toplevel::SET_TITLE,
            &[Arg::Str(&self.title_scratch)],
        );
        // The two buffers trade places, so the old title becomes next call's scratch
        // and neither is reallocated.
        std::mem::swap(&mut self.title, &mut self.title_scratch);
    }

    /// How long to block for input: the soonest of the next key-repeat, the next
    /// auto-scroll tick, the next caret blink, and the next scrollbar fade step, or
    /// `None` when none is due.
    fn next_wake(&self) -> Option<Duration> {
        let now = Instant::now();
        let until = |at: Instant| {
            at.saturating_duration_since(now)
                .max(Duration::from_millis(1))
        };
        // Auto-scroll only ever runs in the focused editor (the one being dragged).
        let ed = &self.editors[self.active_editor()];
        let autoscroll = (ed.autoscroll != 0 && ed.dragging())
            .then_some(ed.autoscroll_at)
            .flatten()
            .map(until);
        // Only wake for the caret while focused, matching `fire_caret_blink`: a
        // stray `bump_caret` off a pointer event on an unfocused window must not
        // leave a past deadline that spins the loop.
        let caret = self
            .focused
            .then(|| self.caret.wake_at())
            .flatten()
            .map(until);
        // A caret still easing needs a follow-up tick, but only when no compositor
        // frame is pending: while a callback is outstanding, the frame-done event
        // paces the animation, so adding a deadline here would just wake early.
        let caret_anim = (self.frame_callback == 0)
            .then(|| self.caret_retry_at())
            .flatten()
            .map(until);
        // A scrollbar wants the loop back either to step a fade or, once its hold
        // lapses, to start one. Same pacing rule as the caret: while a frame callback
        // is outstanding the compositor drives the animation.
        let scrollbar = (self.frame_callback == 0)
            .then(|| self.scrollbar_retry_at())
            .flatten()
            .map(until);
        // The idle autosave wakes the loop even on an unfocused window: focus leaving
        // is itself a save, but a window left dirty some other way (the pointer
        // dragging text into it, a find-replace as focus goes) must still land.
        let autosave = self.autosave_at.map(until);
        [
            self.repeat_timeout(),
            autoscroll,
            caret,
            caret_anim,
            scrollbar,
            autosave,
        ]
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
    use crate::theme::VAULT_COLORS;

    // `Focus` is not `Debug`, so compare with `==` inside `assert!`.

    fn vaults(paths: &[&str]) -> Vec<Vault> {
        crate::config::vaults(paths.iter().map(PathBuf::from))
    }

    #[test]
    fn the_demo_diff_exercises_every_concept() {
        let (old, new) = demo_diff_texts();
        let rows = crate::ai::line_diff(&old, &new);
        // An unchanged line carried through on both sides.
        assert!(
            rows.iter()
                .any(|r| !r.changed && r.old.is_some() && r.new.is_some())
        );
        // A changed pair (both sides present, tinted).
        assert!(
            rows.iter()
                .any(|r| r.changed && r.old.is_some() && r.new.is_some())
        );
        // A wholly removed line (present on the left, hatched on the right).
        assert!(
            rows.iter()
                .any(|r| r.changed && r.old.is_some() && r.new.is_none())
        );
        // A wholly added line (present on the right, hatched on the left).
        assert!(
            rows.iter()
                .any(|r| r.changed && r.old.is_none() && r.new.is_some())
        );
        // Reading each side back reproduces it, so the demo is a real diff, not a mock.
        let a: Vec<&str> = old.split('\n').collect();
        let b: Vec<&str> = new.split('\n').collect();
        let left: Vec<&str> = rows.iter().filter_map(|r| r.old).map(|i| a[i]).collect();
        let right: Vec<&str> = rows.iter().filter_map(|r| r.new).map(|j| b[j]).collect();
        assert_eq!(left, a);
        assert_eq!(right, b);
    }

    #[test]
    fn the_editor_menu_offers_the_task_list_command_per_pane() {
        // The editing command is independent of AI, names its pane, and reads as an
        // action, so a right-click over a pane always has at least this to click.
        let items = State::edit_command_items(1);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "Toggle task list");
        assert_eq!(items[0].action, MenuAction::ToggleTaskList { pane: 1 });
    }

    #[test]
    fn resolve_vault_prefers_a_directory_argument() {
        let dir = std::env::temp_dir();
        let vs = vaults(&["/vault"]);
        let (root, file) = resolve_vault(Some(&dir), None, &vs);
        assert_eq!(root, dir);
        assert_eq!(file, None);
    }

    #[test]
    fn the_hover_square_is_the_same_square_closer_in() {
        // Smaller, so the picture fills the chip more; centered on the same middle, so
        // it pushes toward the pointer rather than drifting to a corner.
        let (x, y, side) = zoomed_square(100, 200, 400);
        assert!(side < 400, "closer in");
        assert!(side > 300, "not so close the picture is lost");
        assert_eq!(x + side / 2, 100 + 400 / 2, "the same middle across");
        assert_eq!(y + side / 2, 200 + 400 / 2, "the same middle down");
    }

    #[test]
    fn the_hover_square_stays_inside_the_square_it_zooms() {
        // It is a crop of the picture, so it cannot reach past what was cropped.
        for side in [1u32, 2, 3, 34, 400, 4000] {
            let (x, y, zoomed) = zoomed_square(10, 20, side);
            assert!(zoomed >= 1, "side {side} keeps a pixel");
            assert!(zoomed <= side, "side {side} does not grow");
            assert!(x + zoomed <= 10 + side, "side {side} fits across");
            assert!(y + zoomed <= 20 + side, "side {side} fits down");
        }
    }

    #[test]
    fn autosave_waits_out_the_typing() {
        // Each edit pushes the window out, so a run of typing writes once it stops,
        // not on every keystroke.
        let now = Instant::now();
        let armed = autosave_deadline(true, true, None, now);
        assert_eq!(armed, Some(now + AUTOSAVE_IDLE));
        let later = now + Duration::from_millis(500);
        assert_eq!(
            autosave_deadline(true, true, armed, later),
            Some(later + AUTOSAVE_IDLE),
            "a second keystroke moves the deadline with it"
        );
    }

    #[test]
    fn autosave_lets_a_quiet_pass_run_the_window_down() {
        // A pass that carried no edit (a repaint, a pointer move) must not keep
        // pushing the save away, or an idle document would never be written.
        let now = Instant::now();
        let armed = Some(now + AUTOSAVE_IDLE);
        assert_eq!(
            autosave_deadline(false, true, armed, now + Duration::from_millis(500)),
            armed
        );
    }

    #[test]
    fn autosave_disarms_when_there_is_nothing_to_save() {
        // Undoing back to the saved text leaves nothing pending: the deadline goes
        // rather than firing a write of a file that is already what it was. This is
        // also what keeps an idle editor's loop asleep, since the deadline is a
        // wake-up.
        let now = Instant::now();
        let armed = Some(now + AUTOSAVE_IDLE);
        assert_eq!(autosave_deadline(true, false, armed, now), None);
        assert_eq!(autosave_deadline(false, false, armed, now), None);
    }

    #[test]
    fn autosave_stays_disarmed_with_nothing_pending() {
        // The idle case: no timer, so nothing wakes the loop.
        let now = Instant::now();
        assert_eq!(autosave_deadline(false, false, None, now), None);
    }

    /// The rail's remembered arrangement for `names` under one parent, each chip
    /// wearing the color given.
    fn stored(chips: &[(&str, u32)]) -> Vec<store::RailVault> {
        chips
            .iter()
            .map(|(name, color)| store::RailVault {
                root: PathBuf::from("/home/user").join(name),
                color: *color,
                icon: None,
            })
            .collect()
    }

    /// The rail's vault names in order, which is what the arrangement tests ask.
    fn names(rail: &[Vault]) -> Vec<String> {
        rail.iter()
            .map(|v| vault_name(&v.path).to_string())
            .collect()
    }

    /// The labels of the rows the palette would list, in order.
    fn row_labels(rows: &[PaletteRow]) -> Vec<&str> {
        rows.iter().map(|r| r.label.as_str()).collect()
    }

    /// A result of `n` hits, `truncated` or not, which is all the note reads.
    fn results(n: usize, truncated: bool) -> search::Results {
        search::Results {
            hits: (0..n)
                .map(|i| search::Hit {
                    path: PathBuf::from("a.md"),
                    offset: i,
                    line: 1,
                    heading: None,
                    text: "x".into(),
                })
                .collect(),
            truncated,
            ..search::Results::default()
        }
    }

    #[test]
    fn the_search_note_counts_the_matching_lines() {
        // A row is a line, so the count is lines: calling them matches would overcount
        // a line the query hits twice.
        assert_eq!(search_note(&results(1, false), "gamma"), "1 line");
        assert_eq!(search_note(&results(64, false), "gamma"), "64 lines");
    }

    #[test]
    fn the_search_note_says_when_the_cap_cut_the_answer_short() {
        // A capped list would otherwise read as the whole answer, which is the one
        // thing a silent cap must never do.
        assert_eq!(search_note(&results(2000, true), "the"), "first 2000 lines");
    }

    #[test]
    fn the_search_note_tells_a_short_query_what_it_is_waiting_for() {
        // An empty panel cannot say the difference between "nothing matched" and
        // "not scanning yet" on its own.
        assert_eq!(search_note(&results(0, false), "g"), "type 2 characters");
        assert_eq!(search_note(&results(0, false), ""), "type 2 characters");
        assert_eq!(search_note(&results(0, false), "gamma"), "no matches");
    }

    #[test]
    fn vault_rows_leave_out_the_vault_the_window_is_already_at() {
        let v = vaults(&["/home/user/notes", "/home/user/work", "/home/user/wiki"]);
        let rows = vault_rows(&v, Some(1), None);
        assert_eq!(row_labels(&rows), ["vault: notes", "vault: wiki"]);
        // Each row still carries its own vault's index, not its place in the list.
        assert_eq!(rows[1].action, PaletteAction::SwitchVault(2));
    }

    #[test]
    fn vault_rows_keep_the_rails_order_when_there_is_nowhere_to_go_back_to() {
        // Nothing switched yet this run, so the rail's own order is all there is.
        let v = vaults(&["/home/user/notes", "/home/user/work", "/home/user/wiki"]);
        let rows = vault_rows(&v, None, None);
        assert_eq!(
            row_labels(&rows),
            ["vault: notes", "vault: work", "vault: wiki"]
        );
    }

    #[test]
    fn the_vault_the_window_came_from_is_the_first_row() {
        // Ctrl+K then Enter goes back where you just were, whatever the rail's order.
        let v = vaults(&["/home/user/notes", "/home/user/work", "/home/user/wiki"]);
        let rows = vault_rows(&v, Some(0), Some(Path::new("/home/user/wiki")));
        assert_eq!(row_labels(&rows), ["vault: wiki", "vault: work"]);
        assert_eq!(rows[0].action, PaletteAction::SwitchVault(2));
    }

    #[test]
    fn a_rail_of_one_vault_offers_no_rows_at_all() {
        // The only vault is the active one, so there is nowhere to go and the palette
        // has nothing to open with.
        let v = vaults(&["/home/user/notes"]);
        assert!(vault_rows(&v, Some(0), None).is_empty());
    }

    #[test]
    fn a_vault_named_by_a_root_path_still_gets_a_label() {
        // A root directory has no last component to name it, so the label falls back
        // to the path rather than coming out empty.
        let v = vaults(&["/"]);
        let rows = vault_rows(&v, None, None);
        assert_eq!(row_labels(&rows), ["vault: /"]);
    }

    #[test]
    fn rail_vaults_adds_an_unconfigured_root_as_a_temporary_vault() {
        // `bnknotes .` in a directory the config does not name: the rail shows where
        // the window actually is, as a chip that says it is not written down.
        let rail = rail_vaults(
            vaults(&["/home/user/notes"]),
            &[],
            Path::new("/tmp/scratch"),
        );
        assert_eq!(rail.len(), 2);
        assert_eq!(rail[1].path, Path::new("/tmp/scratch"));
        assert_eq!(rail[1].kind, VaultKind::Temporary);
        // Last, so a saved chip never has to move to make room for a passing one.
        assert_eq!(rail[0].kind, VaultKind::Saved);
        assert_eq!(rail[0].path, Path::new("/home/user/notes"));
    }

    #[test]
    fn rail_vaults_leaves_a_configured_root_alone() {
        // Launching in a directory that is already a vault activates that chip rather
        // than adding a second one for the same place.
        let vs = vaults(&["/home/user/notes", "/home/user/work"]);
        let rail = rail_vaults(vs.clone(), &[], Path::new("/home/user/work"));
        assert_eq!(rail, vs);
    }

    #[test]
    fn a_first_run_rail_is_the_config_order() {
        // Nothing remembered yet: the config's order is the rail's, each chip dealt a
        // color of its own.
        let rail = rail_vaults(
            vaults(&["/home/user/notes", "/home/user/work"]),
            &[],
            Path::new("/home/user/notes"),
        );
        assert_eq!(names(&rail), ["notes", "work"]);
        assert_ne!(rail[0].color, rail[1].color);
    }

    #[test]
    fn a_dragged_rail_comes_back_in_the_order_it_was_left() {
        // The whole point of remembering the rail: the config still lists notes first,
        // but the chips were dragged, and the drag is what the rail shows.
        let rail = rail_vaults(
            vaults(&["/home/user/notes", "/home/user/work"]),
            &stored(&[("work", VAULT_COLORS[1]), ("notes", VAULT_COLORS[0])]),
            Path::new("/home/user/notes"),
        );
        assert_eq!(names(&rail), ["work", "notes"]);
    }

    #[test]
    fn a_chip_keeps_the_color_it_was_dealt_wherever_it_sits() {
        // What makes a drag readable: the chip that was green is still green after it
        // moves, so its color keeps saying which vault it is.
        let rail = rail_vaults(
            vaults(&["/home/user/notes", "/home/user/work"]),
            &stored(&[("work", VAULT_COLORS[1]), ("notes", VAULT_COLORS[0])]),
            Path::new("/home/user/notes"),
        );
        assert_eq!(rail[0].color, VAULT_COLORS[1], "work kept its color");
        assert_eq!(rail[1].color, VAULT_COLORS[0], "so did notes");
    }

    #[test]
    fn a_vault_added_to_the_config_joins_the_end_of_the_rail() {
        // Hand-editing the config still adds vaults, and a new one takes a color no
        // chip is wearing rather than one already spoken for.
        let rail = rail_vaults(
            vaults(&["/home/user/notes", "/home/user/work"]),
            &stored(&[("work", VAULT_COLORS[1])]),
            Path::new("/home/user/work"),
        );
        assert_eq!(names(&rail), ["work", "notes"]);
        assert_ne!(rail[1].color, VAULT_COLORS[1]);
    }

    #[test]
    fn a_vault_dropped_from_the_config_leaves_the_rail() {
        // The config says which vaults there are; a chip the config no longer names is
        // gone, however the rail remembered it.
        let rail = rail_vaults(
            vaults(&["/home/user/notes"]),
            &stored(&[("work", VAULT_COLORS[1]), ("notes", VAULT_COLORS[0])]),
            Path::new("/home/user/notes"),
        );
        assert_eq!(names(&rail), ["notes"]);
    }

    #[test]
    fn a_temporary_vault_is_not_part_of_the_remembered_rail() {
        // It belongs to this launch, not to the arrangement: a rail full of the
        // directories someone once opened is not worth keeping.
        let rail = rail_vaults(
            vaults(&["/home/user/notes"]),
            &[],
            Path::new("/tmp/scratch"),
        );
        let session = rail_session(&rail);
        assert_eq!(session.len(), 1);
        assert_eq!(session[0].root, Path::new("/home/user/notes"));
    }

    #[test]
    fn the_remembered_rail_round_trips_through_an_arrangement() {
        // Save the rail, reopen with it, and the chips are where they were left,
        // wearing what they wore. The arrangement's whole contract, in one pass.
        let configured = vaults(&["/home/user/notes", "/home/user/work"]);
        let mut rail = rail_vaults(configured.clone(), &[], Path::new("/home/user/notes"));
        rail.swap(0, 1);
        let reopened = rail_vaults(
            configured,
            &rail_session(&rail),
            Path::new("/home/user/notes"),
        );
        assert_eq!(names(&reopened), names(&rail));
        assert_eq!(reopened[0].color, rail[0].color);
        assert_eq!(reopened[1].color, rail[1].color);
    }

    #[test]
    fn resolve_vault_roots_a_file_argument_at_its_containing_vault() {
        // A note deep inside a configured vault roots at the whole vault, not just
        // the note's own folder, so its embeds resolve vault-wide.
        let vs = vaults(&["/home/user/notes"]);
        let arg = Path::new("/home/user/notes/journal/day.md");
        let (root, opened) = resolve_vault(Some(arg), None, &vs);
        assert_eq!(root, Path::new("/home/user/notes"));
        assert_eq!(opened.as_deref(), Some(arg));
    }

    #[test]
    fn resolve_vault_prefers_the_deepest_containing_vault() {
        // Nested vaults: the note belongs to the innermost one that holds it.
        let vs = vaults(&["/home/user/notes", "/home/user/notes/work"]);
        let arg = Path::new("/home/user/notes/work/plan.md");
        let (root, _) = resolve_vault(Some(arg), None, &vs);
        assert_eq!(root, Path::new("/home/user/notes/work"));
    }

    #[test]
    fn resolve_vault_roots_a_file_outside_any_vault_at_its_parent() {
        let vs = vaults(&["/home/user/notes"]);
        let arg = Path::new("/tmp/scratch/note.md");
        let (root, opened) = resolve_vault(Some(arg), None, &vs);
        assert_eq!(root, Path::new("/tmp/scratch"));
        assert_eq!(opened.as_deref(), Some(arg));
    }

    #[test]
    fn resolve_vault_restores_a_configured_session_vault() {
        // The remembered vault is one of the configured ones (not the first), so it
        // is restored rather than snapping back to the default.
        let vs = vaults(&["/home/user/notes", "/home/user/other"]);
        let (root, file) = resolve_vault(None, Some(Path::new("/home/user/other")), &vs);
        assert_eq!(root, Path::new("/home/user/other"));
        assert_eq!(file, None);
    }

    #[test]
    fn resolve_vault_ignores_an_unconfigured_session_vault() {
        // A stale pre-vault root (an old working directory, or `.`) is not a
        // configured vault, so it is dropped for the configured default even though
        // it exists on disk. This is the migration path off the cwd-based root.
        let vs = vaults(&["/home/user/notes", "/home/user/other"]);
        let stale = std::env::temp_dir();
        let (root, file) = resolve_vault(None, Some(&stale), &vs);
        assert_eq!(root, Path::new("/home/user/notes"));
        assert_eq!(file, None);
    }

    #[test]
    fn resolve_vault_with_no_argument_or_session_uses_the_first_vault() {
        let vs = vaults(&["/home/user/notes", "/home/user/other"]);
        let (root, file) = resolve_vault(None, None, &vs);
        assert_eq!(root, Path::new("/home/user/notes"));
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
