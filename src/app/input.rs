use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

use super::State;
use crate::find;
use crate::pane::{EditorPane, Focus, Hit, SPLITTER};
use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::platform::protocol::{
    wl_keyboard, wl_pointer, wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
};
use crate::platform::scroll;
use crate::platform::wire::{Arg, Reader};
use crate::platform::xkb::KeyAction;
use crate::theme::{FONT_SIZE, TREE_PAD};

/// What the Tab key inserts: spaces (a "soft tab"), not a literal `'\t'`. This
/// keeps the buffer free of tab characters and spares the renderer any tab-stop
/// logic. Rendering literal tabs found in a *loaded* file is a separate concern.
const SOFT_TAB: &str = "    ";

/// How long the caret stays shown, then hidden, while idle. Activity resets it
/// to shown, so it never blinks out mid-keystroke.
const CARET_BLINK: Duration = Duration::from_millis(500);
/// Pixels scrolled per unit of wl_pointer.axis value (wheel/touchpad). One wheel
/// detent is ~15 axis units, so this lands a notch at roughly three body lines.
const SCROLL_FACTOR: f32 = 4.0;

/// Fallback auto-repeat timing if the compositor never sends repeat_info.
const DEFAULT_REPEAT_DELAY_MS: u64 = 400;
const DEFAULT_REPEAT_RATE: u32 = 25;
/// Cap on repeats fired per wake-up, so a stalled frame cannot flood input.
const REPEAT_CATCHUP_CAP: u32 = 16;

/// Key auto-repeat timers. The compositor sends a single press; the loop refires
/// the held key on `interval` after the initial `delay`, until it is released.
pub(super) struct KeyRepeat {
    /// Delay before the first repeat.
    pub(super) delay: Duration,
    /// `None` disables repeat (the compositor reported a zero rate).
    pub(super) interval: Option<Duration>,
    /// The key currently repeating, or `None` when nothing is held.
    pub(super) key: Option<u32>,
    /// When the next repeat is due.
    pub(super) at: Option<Instant>,
}

impl KeyRepeat {
    pub(super) fn new() -> Self {
        Self {
            delay: Duration::from_millis(DEFAULT_REPEAT_DELAY_MS),
            interval: Some(Duration::from_secs(1) / DEFAULT_REPEAT_RATE),
            key: None,
            at: None,
        }
    }
}

/// Apply one pure-editor key action to the focused pane, returning whether it
/// changed the buffer (so the caller can run its after-edit bookkeeping). These
/// are the actions that need nothing but the pane, the fonts, and the viewport
/// `rect`; the window/pane, find-bar, and clipboard/save actions are routed
/// elsewhere by [`State::apply_action`]. `extend` is whether Shift is held.
fn apply_editor_action(
    ed: &mut EditorPane,
    fonts: &Fonts,
    rect: Rect,
    extend: bool,
    action: KeyAction,
) -> bool {
    let changed = match action {
        KeyAction::Char(c) => {
            let mut encoded = [0u8; 4];
            ed.text.insert_str(c.encode_utf8(&mut encoded));
            true
        }
        KeyAction::Enter => ed.text.enter(),
        // On a list item, Tab nests it one level deeper and Shift+Tab one
        // shallower. Anywhere else, Tab inserts a soft tab and Shift+Tab is a
        // no-op (there is nothing to dedent).
        KeyAction::Tab => {
            if ed.text.cursor_on_list_item() {
                ed.text.nest_list_item(extend)
            } else if extend {
                false
            } else {
                ed.text.insert_str(SOFT_TAB);
                true
            }
        }
        KeyAction::Backspace => ed.text.backspace(),
        KeyAction::Delete => ed.text.delete_forward(),
        KeyAction::DeleteWordBack => ed.text.delete_word_back(),
        KeyAction::DeleteWordForward => ed.text.delete_word_forward(),
        KeyAction::DeleteLine => ed.text.delete_line(),
        KeyAction::Left => ed.text.move_left(extend),
        KeyAction::Right => ed.text.move_right(extend),
        KeyAction::WordLeft => ed.text.move_word_left(extend),
        KeyAction::WordRight => ed.text.move_word_right(extend),
        KeyAction::Up => ed.move_vertical(fonts, rect, -1, extend),
        KeyAction::Down => ed.move_vertical(fonts, rect, 1, extend),
        KeyAction::PageUp => ed.move_page(fonts, rect, -1, extend),
        KeyAction::PageDown => ed.move_page(fonts, rect, 1, extend),
        KeyAction::Home => ed.move_line_edge(fonts, rect, false, extend),
        KeyAction::End => ed.move_line_edge(fonts, rect, true, extend),
        KeyAction::DocStart => ed.text.move_doc_start(extend),
        KeyAction::DocEnd => ed.text.move_doc_end(extend),
        KeyAction::SelectAll => {
            ed.text.select_all();
            true
        }
        KeyAction::Undo => ed.text.undo(),
        KeyAction::Redo => ed.text.redo(),
        // With no find bar, Escape collapses a selection to a bare cursor; with
        // nothing selected there is nothing to leave.
        KeyAction::Escape => {
            if ed.text.selection().is_some() {
                let c = ed.text.cursor();
                ed.text.set_cursor(c, false);
                true
            } else {
                false
            }
        }
        // Routed elsewhere by `apply_action` (window/pane, find bar, clipboard,
        // and save); listed for exhaustiveness.
        KeyAction::Copy
        | KeyAction::Cut
        | KeyAction::Paste
        | KeyAction::Save
        | KeyAction::Find
        | KeyAction::Replace
        | KeyAction::ToggleTree
        | KeyAction::ToggleSecondPane
        | KeyAction::FocusTree
        | KeyAction::FocusPaneA
        | KeyAction::FocusPaneB
        | KeyAction::None => false,
    };
    // The vertical motions read `goal_x` (the remembered column). SelectAll and
    // Escape leave it untouched (they do not move by column). Every other action
    // invalidates it, so the next vertical motion recomputes from the new column.
    if !matches!(
        action,
        KeyAction::Up
            | KeyAction::Down
            | KeyAction::PageUp
            | KeyAction::PageDown
            | KeyAction::SelectAll
            | KeyAction::Escape
    ) {
        ed.goal_x = None;
    }
    changed
}

impl State {
    pub(super) fn on_keyboard(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_keyboard::EV_KEYMAP => {
                let format = r.u32()?;
                let size = r.u32()? as usize;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("wl_keyboard.keymap arrived without its fd"))?;
                let bytes = ffi::read_mapped(fd.as_raw_fd(), size)?;
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
                } else if self.key_repeat.key == Some(keycode) {
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
                self.key_repeat.interval = if rate == 0 {
                    None
                } else {
                    Some(Duration::from_secs(1) / rate)
                };
                self.key_repeat.delay = Duration::from_millis(u64::from(delay));
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

    /// Move keyboard focus to editor pane `i`, remembering it as the last active
    /// editor, refreshing the title for its note, and marking a repaint.
    fn focus_pane(&mut self, i: usize) {
        self.focus = Focus::Editor(i);
        self.last_editor = i;
        self.refresh_title();
        self.dirty = true;
    }

    pub(super) fn apply_action(&mut self, action: KeyAction) -> Result<()> {
        // Window-level pane management, handled before any editor routing: toggle
        // the sidebar/second pane, or move focus between regions.
        match action {
            KeyAction::ToggleTree => {
                self.panes.tree_visible = !self.panes.tree_visible;
                if !self.panes.tree_visible && self.focus == Focus::Tree {
                    self.focus_pane(self.last_editor);
                }
                self.dirty = true;
                self.save_session();
                return Ok(());
            }
            KeyAction::ToggleSecondPane => {
                self.panes.second_open = !self.panes.second_open;
                if !self.panes.second_open && self.focus == Focus::Editor(1) {
                    self.focus_pane(0);
                }
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
                self.focus_pane(0);
                return Ok(());
            }
            KeyAction::FocusPaneB => {
                if self.panes.second_open {
                    self.focus_pane(1);
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
            let ed = &mut self.editors[i];
            let cursor = ed.text.cursor();
            let target = if action == KeyAction::Find {
                ed.find.open_at(cursor, &ed.text)
            } else {
                ed.find.open_replace_at(cursor, &ed.text)
            };
            self.land_on_match(target);
            return Ok(());
        }
        // The clipboard and save actions stay on `State`: they need the Wayland
        // connection and swallow-and-report their own I/O errors (a peer closing
        // the pipe early, or a disk failure, must not bring the editor down and
        // lose unsaved work). Everything else is a pure edit on the focused pane.
        let changed = match action {
            // With an image selected, Copy puts the picture on the clipboard
            // instead of text.
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
            KeyAction::Save => {
                if let Err(e) = self.save() {
                    eprintln!("bnknotes: save failed: {e}");
                }
                false
            }
            _ => apply_editor_action(&mut self.editors[i], &self.fonts, rect, extend, action),
        };
        if changed {
            self.after_edit();
        }
        Ok(())
    }

    /// One key while the find bar is open. Characters and Backspace edit the
    /// focused field (Alt+W / Alt+R toggle whole-word and regex instead), Tab
    /// moves between the query and replace fields, Ctrl+V pastes into the
    /// focused field, and Enter navigates or replaces: next match from the
    /// query field (Shift+Enter previous), replace-current-and-advance from
    /// the replace field, replace-all with Ctrl. Escape or Ctrl+F closes the
    /// bar, keeping the current match selected. Anything else is ignored.
    pub(super) fn handle_find_key(&mut self, action: KeyAction) {
        let shift = self.xkb.shift_active();
        let ctrl = self.xkb.ctrl_active();
        let alt = self.xkb.alt_active();
        // The find bar belongs to the focused editor.
        let i = self.active_editor();
        let target = match action {
            KeyAction::Char(c) if alt => {
                let ed = &mut self.editors[i];
                match c.to_ascii_lowercase() {
                    'w' => ed.find.toggle_word(&ed.text),
                    'r' => ed.find.toggle_regex(&ed.text),
                    _ => return,
                }
            }
            KeyAction::Char(c) => {
                let ed = &mut self.editors[i];
                ed.find.push(c, &ed.text)
            }
            KeyAction::Backspace => {
                let ed = &mut self.editors[i];
                ed.find.pop(&ed.text)
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
                    let ed = &mut self.editors[i];
                    ed.find.push_str(&pasted, &ed.text)
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
                let ed = &mut self.editors[i];
                let cursor = ed.text.cursor();
                ed.find.open_replace_at(cursor, &ed.text)
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
    pub(super) fn land_on_match(&mut self, target: Option<(usize, usize)>) {
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
    pub(super) fn replace_current_match(&mut self) {
        let i = self.active_editor();
        let Some((start, end)) = self.editors[i].find.current_match() else {
            self.dirty = true;
            return;
        };
        let with = self.editors[i].find.replacement().to_string();
        self.editors[i].text.select_range(start, end);
        self.editors[i].text.insert_str(&with);
        self.after_edit();
        let ed = &mut self.editors[i];
        let target = ed.find.refresh(&ed.text, start + with.len());
        self.land_on_match(target);
    }

    /// Replace every match with the bar's replacement in one undo step.
    pub(super) fn replace_all_matches(&mut self) {
        let i = self.active_editor();
        let ranges = self.editors[i].find.matches().to_vec();
        let with = self.editors[i].find.replacement().to_string();
        if !self.editors[i].text.replace_ranges(&ranges, &with) {
            return;
        }
        self.after_edit();
        let ed = &mut self.editors[i];
        let cursor = ed.text.cursor();
        let target = ed.find.refresh(&ed.text, cursor);
        self.land_on_match(target);
    }

    pub(super) fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
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
                        self.editors[i].disarm_autoscroll();
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
    pub(super) fn hit_test(&self, px: f32, py: f32) -> Hit {
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
    pub(super) fn drag_splitter_to(&mut self, s: usize, px: f32) {
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
    pub(super) fn update_tree_hover(&mut self) {
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
    pub(super) fn tree_row_at(&self, rect: Rect, py: f32) -> Option<usize> {
        let row_h = self.fonts.metrics(FONT_SIZE).line_height.max(1);
        let y = py as i32 - rect.y + self.tree.scroll_y - TREE_PAD;
        if y < 0 {
            return None;
        }
        let idx = (y / row_h) as usize;
        (idx < self.tree.tree.rows().len()).then_some(idx)
    }

    /// Scroll the tree column by `delta`, clamped, repainting if it moved.
    pub(super) fn scroll_tree(&mut self, rect: Rect, delta: i32) {
        let sy = scroll::clamp(self.tree.scroll_y + delta, self.tree.content_height, rect.h);
        if sy != self.tree.scroll_y {
            self.tree.scroll_y = sy;
            self.dirty = true;
        }
    }

    /// Open `path` into the last-focused content pane and focus it. An unreadable
    /// file is reported without disturbing the current pane.
    pub(super) fn open_file(&mut self, path: &Path) {
        match EditorPane::open(path) {
            Ok(pane) => {
                let i = self.last_editor;
                self.editors[i] = pane;
                self.focus_pane(i);
                self.save_session();
            }
            Err(e) => eprintln!("bnknotes: cannot open {}: {e}", path.display()),
        }
    }

    /// The tree column rectangle, or an empty rect when the sidebar is hidden.
    pub(super) fn tree_rect(&self) -> Rect {
        self.columns().tree.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        })
    }

    /// Scroll the tree so its selected row is on screen.
    pub(super) fn reveal_tree_selection(&mut self) {
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
    pub(super) fn tree_key(&mut self, action: KeyAction) {
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
    pub(super) fn init_cursor_shape(&mut self) {
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
    pub(super) fn set_cursor_shape(&mut self, shape: u32) {
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
    pub(super) fn update_cursor_shape(&mut self) {
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
    pub(super) fn on_left_press(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        match self.hit_test(px, py) {
            Hit::Splitter(s) => self.drag_splitter = Some(s),
            Hit::Tree(rect) => self.tree_press(rect, py),
            Hit::Editor(i, rect) => {
                self.focus_pane(i);
                self.editor_press(i, rect, px, py);
            }
            Hit::None => {}
        }
    }

    /// A press in the tree: focus the sidebar, then toggle a folder's expansion
    /// or open a file into the last-focused editor.
    pub(super) fn tree_press(&mut self, rect: Rect, py: f32) {
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
    pub(super) fn editor_press(&mut self, i: usize, rect: Rect, px: f32, py: f32) {
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
        let (click_count, chained) = self.editors[i].clicks.register(px, py, Instant::now());
        // A plain click on a rendered horizontal rule selects the whole rule line.
        if !shift {
            if let Some((start, end)) = self.editors[i].rule_under_pointer(&self.fonts, rect, py) {
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
                if crate::platform::browser::open_url(&dest).is_ok() {
                    self.editors[i].selecting = false;
                    return;
                }
            }
        }
        match click_count {
            1 => {
                self.editors[i].place_cursor_at(&self.fonts, rect, px, py, shift);
                self.editors[i].selecting = true;
                // Pin marker reveal to the press point for the whole drag.
                let c = self.editors[i].text.cursor();
                let reveal = self.editors[i].text.selection().unwrap_or((c, c));
                self.editors[i].frozen_reveal = Some(reveal);
                // Show the moved caret on the next frame, not the next blink wake.
                self.dirty = true;
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
    pub(super) fn fire_autoscroll(&mut self) {
        let i = self.active_editor();
        // A drag that ended (the selection was released) disarms; otherwise fire a
        // tick only once its deadline has passed.
        if !self.editors[i].selecting {
            self.editors[i].disarm_autoscroll();
            return;
        }
        let Some(step) = self.editors[i].autoscroll_due(Instant::now()) else {
            return;
        };
        let rect = self.editor_rect(i);
        let (px, py) = (self.pointer_x, self.pointer_y);
        if self.editors[i].scroll_by(&self.fonts, rect, step) {
            self.dirty = true;
        }
        if self.editors[i].place_cursor_at(&self.fonts, rect, px, py, true) {
            self.dirty = true;
        }
    }

    // --- Caret blink ------------------------------------------------------

    /// Toggle the caret if its blink interval has elapsed, requesting a redraw.
    /// The first call just arms the timer.
    pub(super) fn fire_caret_blink(&mut self) {
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
    pub(super) fn bump_caret(&mut self) {
        self.caret_visible = true;
        self.caret_blink_at = Some(Instant::now() + CARET_BLINK);
    }

    // --- Key repeat -------------------------------------------------------

    /// Begin repeating `keycode` after the delay, if repeat is enabled and the
    /// keymap marks the key as repeatable.
    pub(super) fn start_repeat(&mut self, keycode: u32) {
        if self.key_repeat.interval.is_some() && self.xkb.key_repeats(keycode) {
            self.key_repeat.key = Some(keycode);
            self.key_repeat.at = Some(Instant::now() + self.key_repeat.delay);
        }
    }

    pub(super) fn stop_repeat(&mut self) {
        self.key_repeat.key = None;
        self.key_repeat.at = None;
    }

    /// Fire every repeat whose deadline has passed, re-running the held key's
    /// action (so modifiers held with it still apply).
    pub(super) fn fire_repeats(&mut self) -> Result<()> {
        let (Some(keycode), Some(mut at), Some(interval)) = (
            self.key_repeat.key,
            self.key_repeat.at,
            self.key_repeat.interval,
        ) else {
            return Ok(());
        };
        let now = Instant::now();
        let mut fired = 0;
        while at <= now && fired < REPEAT_CATCHUP_CAP {
            self.apply_action(self.xkb.action(keycode))?;
            at += interval;
            fired += 1;
        }
        self.key_repeat.at = Some(at);
        Ok(())
    }

    /// How long to wait for input before the next repeat is due, or `None` to
    /// block indefinitely when nothing is repeating.
    pub(super) fn repeat_timeout(&self) -> Option<Duration> {
        let at = self.key_repeat.at?;
        self.key_repeat.interval?;
        Some(
            at.saturating_duration_since(Instant::now())
                .max(Duration::from_millis(1)),
        )
    }
}
