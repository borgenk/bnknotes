use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

use super::{State, TabDrag, TabHover, VaultDrag};
use crate::find;
use crate::keymap::{self, KeyAction};
use crate::menu::{Menu, MenuLayout};
use crate::paint::{BannerButton, conflict_banner_layout};
use crate::palette::{Palette, PaletteLayout, Rows, Submit};
use crate::pane::{CaretAnim, Columns, DiskConflict, EditorPane, Focus, Hit, ImagePress, SPLITTER};
use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::freetype::{Fonts, Metrics};
use crate::platform::geom::Rect;
use crate::platform::protocol::{
    wl_keyboard, wl_pointer, wl_seat, wp_cursor_shape_device_v1, wp_cursor_shape_manager_v1,
};
use crate::platform::scroll::{self, Scrollbar};
use crate::platform::wire::{Arg, Reader};
use crate::text::SOFT_TAB;
use crate::theme::{DRAG_THRESHOLD, SCROLL_MIN_THUMB, TREE_PAD};

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

/// Whether this action produces text, so pressing it in an empty pane should open
/// a tab to receive it. Navigation and shortcuts leave an empty pane empty.
fn opens_a_tab(action: &KeyAction) -> bool {
    matches!(
        action,
        KeyAction::Char(_) | KeyAction::Enter | KeyAction::Tab | KeyAction::Paste
    )
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
        // A typed `|` inside a table is escaped to `\|` so it cannot be mistaken
        // for a cell divider; everywhere else a character inserts verbatim.
        KeyAction::Char('|') if ed.text.cursor_in_table() => {
            ed.text.insert_str("\\|");
            true
        }
        KeyAction::Char(c) => {
            let mut encoded = [0u8; 4];
            ed.text.insert_str(c.encode_utf8(&mut encoded));
            true
        }
        // In a table cell, Shift+Enter is a line break within the cell (`<br>`) and
        // a bare Enter adds a new row; elsewhere Enter continues a list or breaks.
        KeyAction::Enter if extend && ed.text.cursor_in_table() => {
            ed.text.insert_str("<br>");
            true
        }
        // On a list item, Shift+Enter adds a continuation line indented under the
        // item's text (no bullet), so a wrapped thought stays part of the item.
        KeyAction::Enter if extend && ed.text.cursor_on_list_item() => ed.text.list_continuation(),
        KeyAction::Enter => ed.text.enter(),
        // In a table, Tab moves to the next cell and Shift+Tab to the previous.
        // Over a selection spanning lines, Tab shifts the whole block one unit
        // right and Shift+Tab one left. On a list item, Tab nests it one level
        // deeper and Shift+Tab one shallower. Anywhere else, Tab inserts a soft
        // tab and Shift+Tab outdents the line the cursor is on.
        KeyAction::Tab => {
            if ed.text.cursor_in_table() {
                ed.text.tab_cell(!extend)
            } else if ed.text.selection_spans_lines() {
                ed.text.indent_block(extend)
            } else if ed.text.cursor_on_list_item() {
                ed.text.nest_list_item(extend)
            } else if extend {
                ed.text.indent_block(true)
            } else {
                ed.text.insert_str(SOFT_TAB);
                true
            }
        }
        KeyAction::Backspace => ed.text.backspace(),
        KeyAction::Delete => ed.text.delete_forward(),
        // In a table, Ctrl+Shift+Backspace removes the current row and
        // Ctrl+Shift+Delete the current column (the shift distinguishes them from
        // the plain Ctrl word-deletes); outside a table they stay word-deletes.
        KeyAction::DeleteWordBack if extend && ed.text.cursor_in_table() => {
            ed.text.remove_table_row()
        }
        KeyAction::DeleteWordForward if extend && ed.text.cursor_in_table() => {
            ed.text.remove_table_column()
        }
        KeyAction::DeleteWordBack => ed.text.delete_word_back(),
        KeyAction::DeleteWordForward => ed.text.delete_word_forward(),
        KeyAction::DeleteLine => ed.text.delete_line(),
        // A horizontal step is a pure buffer move, so it can walk into a folded
        // section; stepping clear of one afterwards makes the arrow cross the whole
        // section, which is what the collapsed heading implies on screen.
        KeyAction::Left => ed.text.move_left(extend) | ed.step_over_folds(false, extend),
        KeyAction::Right => ed.text.move_right(extend) | ed.step_over_folds(true, extend),
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
        KeyAction::ToggleFold => ed.toggle_fold_at_cursor(fonts, rect),
        KeyAction::UnfoldAll => ed.unfold_all(),
        // Ctrl+L: convert the cursor's line, or every line the selection touches,
        // into a task item, toggling the box on lines already tasks.
        KeyAction::ToggleTaskList => ed.text.toggle_task_list(),
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
        // Routed elsewhere by `apply_action` (window/pane, the palette, find bar,
        // clipboard, and save); listed for exhaustiveness.
        KeyAction::Copy
        | KeyAction::Cut
        | KeyAction::Paste
        | KeyAction::Save
        | KeyAction::Find
        | KeyAction::Palette
        | KeyAction::SearchVault
        | KeyAction::Replace
        | KeyAction::ToggleTree
        | KeyAction::ToggleSecondPane
        | KeyAction::FocusTree
        | KeyAction::FocusPaneA
        | KeyAction::FocusPaneB
        | KeyAction::CloseTab
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
                    self.apply_action(keymap::action(&self.xkb, keycode))?;
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
                // Regained keyboard focus: show the caret and resume its blink. Snap
                // the focused pane's caret so restored focus never resumes a stale,
                // mid-flight smear.
                self.focused = true;
                if let Focus::Editor(i) = self.focus {
                    self.editors[i].request_caret_snap();
                }
                self.bump_caret();
                self.dirty = true;
            }
            wl_keyboard::EV_LEAVE => self.keyboard_left(),
            _ => {}
        }
        Ok(())
    }

    /// Keyboard focus is gone, by leave or because the keyboard itself went away.
    fn keyboard_left(&mut self) {
        self.stop_repeat();
        self.xkb.reset_held_state();
        // Lost keyboard focus: hide the caret, stop blinking, and let the loop go
        // idle (no 500ms blink wake-ups while unfocused).
        self.focused = false;
        self.caret.hide();
        self.dirty = true;
        // Turning to another window is the user done with this one for now, so it
        // does not wait out the idle window: what they typed is on disk by the time
        // they are reading it somewhere else.
        self.save_all();
    }

    /// The pointer is gone from the surface, by leave or because the pointer
    /// itself went away, so drop the mouse hover highlights; otherwise the
    /// last-hovered tree row, filter toggle, tab, or scrollbar stays lit with no
    /// pointer on it.
    fn pointer_left(&mut self) {
        let lit =
            self.tree.hovered.is_some() || self.tree.filter_hovered || self.hovered_tab.is_some();
        self.tree.hovered = None;
        self.tree.filter_hovered = false;
        self.hovered_tab = None;
        let mut bars = false;
        for region in [Focus::Tree, Focus::Editor(0), Focus::Editor(1)] {
            bars |= self.scrollbar_mut(region).set_near(false);
        }
        if lit || bars {
            self.dirty = true;
        }
    }

    /// The seat lost its keyboard: release the inert object and forget it,
    /// keeping it retired until the compositor confirms the deletion (a seat below
    /// version 3 cannot release it, so there it stays retired).
    pub(super) fn release_keyboard(&mut self) {
        let keyboard = std::mem::take(&mut self.keyboard);
        if self.seat_version >= wl_seat::VERSION_RELEASE {
            self.conn.request(keyboard, wl_keyboard::RELEASE, &[]);
        }
        self.retired.retire(keyboard, wl_keyboard::EV_KEYMAP);
        self.keyboard_left();
    }

    /// The seat lost its pointer: destroy its cursor-shape device, then release
    /// the inert object and forget it. Pointer events carry no fds, so nothing in
    /// flight for it needs taking off the fd queue.
    pub(super) fn release_pointer(&mut self) {
        if self.cursor_shape_device != 0 {
            self.conn.request(
                self.cursor_shape_device,
                wp_cursor_shape_device_v1::DESTROY,
                &[],
            );
            self.cursor_shape_device = 0;
        }
        let pointer = std::mem::take(&mut self.pointer);
        if self.seat_version >= wl_seat::VERSION_RELEASE {
            self.conn.request(pointer, wl_pointer::RELEASE, &[]);
        }
        // A new device starts with no shape set, so the next one is requested.
        self.cursor_shape = 0;
        self.pointer_left();
    }

    /// Move keyboard focus to editor pane `i`, remembering it as the last active
    /// editor, refreshing the title for its note, and marking a repaint.
    pub(super) fn focus_pane(&mut self, i: usize) {
        // Snap the caret only when focus actually crosses panes; a click within the
        // already-focused pane is a navigation move that should still smear.
        let crossed = self.focus != Focus::Editor(i);
        self.focus = Focus::Editor(i);
        self.last_editor = i;
        if crossed {
            self.editors[i].request_caret_snap();
        }
        // A pane taking focus starts a fresh blink run, so the caret that was resting
        // solid (or blinking somewhere else) marks the insertion point here.
        self.bump_caret();
        self.refresh_title();
        self.dirty = true;
    }

    pub(super) fn apply_action(&mut self, action: KeyAction) -> Result<()> {
        // A keyboard action drives this move, so any caret navigation it causes eases
        // rather than snapping. Set for every action (including non-caret ones); it is
        // only read when the caret's offset actually moves.
        self.caret_from_pointer = false;
        // The AI preview is the topmost modal: while it is up it holds the keyboard,
        // Enter to accept the diff and Escape to discard, and swallows everything else
        // so no edit reaches the buffer under it and its captured selection stays put.
        if self.ai.is_some() {
            self.handle_ai_key(action);
            return Ok(());
        }
        // A picture over the window is the topmost thing there is, and the keyboard
        // reaches it first: Escape puts it away and does no more, and any other key
        // that acts puts it away before acting, since typing under a picture that
        // covers the window would be typing blind.
        if self.image_view.is_some() && action != KeyAction::None {
            self.close_image_view();
            if action == KeyAction::Escape {
                return Ok(());
            }
        }
        // An open context menu is the pointer's, but it must not outlive the moment:
        // Escape dismisses it and does no more (that is what the key is for here),
        // and any other key that acts drops it first, rather than leaving a panel
        // floating over text that has moved on. A key that does nothing on its own (a
        // bare modifier) leaves it up.
        if self.menu.is_some() && action != KeyAction::None {
            self.close_menu();
            if action == KeyAction::Escape {
                return Ok(());
            }
        }
        // The palette holds the keyboard while it is up, the way the find bar does:
        // its query takes the keys, so a shortcut cannot toggle a pane or reach a
        // document from under it. Ctrl+K (and Escape) is how it gives them back.
        if self.palette.is_some() {
            self.handle_palette_key(action);
            return Ok(());
        }
        if action == KeyAction::Palette {
            self.toggle_palette();
            return Ok(());
        }
        if action == KeyAction::SearchVault {
            self.toggle_search();
            return Ok(());
        }
        // Window-level pane management, handled before any editor routing: toggle the
        // sidebar/second pane, or move focus between regions.
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
            KeyAction::CloseTab => {
                // The tree has no tabs of its own, so Ctrl+W there closes one in the
                // pane the user was last editing, the pane the tree opens files into.
                // A pinned tab is guarded against Ctrl+W the same as against the click,
                // so it is left alone; unpin it (its badge, or the menu) to close it.
                let g = self.last_editor;
                let active = self.editors[g].active_index();
                if self.editors[g].has_tabs() && !self.editors[g].is_pinned(active) {
                    self.close_tab(g, active);
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
        // A diff review holds this pane's keyboard: Enter keeps the change, Escape
        // discards it, the navigation keys scroll the two columns, and everything else
        // is swallowed so no edit reaches a document that is not on screen.
        if self.editors[i].active_is_diff() {
            self.handle_diff_key(action);
            return Ok(());
        }
        // Typing into an empty pane opens a fresh tab to receive the text, so a
        // just-launched blank pane becomes editable on the first keystroke.
        if !self.editors[i].has_tabs() && opens_a_tab(&action) {
            let (scale, body) = (self.scale(), self.body_size);
            self.editors[i].new_tab(scale, body);
            self.dirty = true;
        }
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
            let ed = self.editors[i].active_doc_mut();
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

    /// One key while the palette is open. Characters and Backspace edit the query,
    /// the arrows move the selection (Home and End jump to its ends), Ctrl+V pastes
    /// into the query, Enter goes where the selected row says, and Escape or Ctrl+K
    /// puts the palette away.
    /// Route a key to the open AI modal (a still-running job or a shown failure):
    /// Escape cancels or dismisses it, and every other key is swallowed so the modal
    /// keeps the keyboard. A successful answer is not a modal; it is a diff tab, keyed
    /// through [`Self::handle_diff_key`].
    pub(super) fn handle_ai_key(&mut self, action: KeyAction) {
        if action == KeyAction::Escape {
            self.close_ai();
        }
    }

    /// Route a key to the focused pane's diff review: Enter keeps the change, Escape
    /// discards it, the navigation keys scroll the two columns together, and every
    /// other key is swallowed so no edit reaches a document that is not on screen.
    pub(super) fn handle_diff_key(&mut self, action: KeyAction) {
        let i = self.active_editor();
        let page = (self.editor_rect(i).h - self.line_step()).max(self.line_step());
        match action {
            KeyAction::Enter => self.accept_diff(),
            KeyAction::Escape => self.reject_diff(),
            KeyAction::Up => self.scroll_diff(i, -self.line_step()),
            KeyAction::Down => self.scroll_diff(i, self.line_step()),
            KeyAction::PageUp => self.scroll_diff(i, -page),
            KeyAction::PageDown => self.scroll_diff(i, page),
            KeyAction::DocStart => self.scroll_diff(i, i32::MIN),
            KeyAction::DocEnd => self.scroll_diff(i, i32::MAX),
            _ => {}
        }
    }

    /// One text line's worth of scroll, for stepping a diff review with the arrows.
    fn line_step(&self) -> i32 {
        self.fonts.metrics(self.body_size).line_height
    }

    /// Scroll the focused pane's diff review by `delta` device pixels (its two sides
    /// move together), clamped to the taller side's extent. A saturating add lets a
    /// Home/End jump pass `i32::MIN`/`MAX` without overflowing before the clamp.
    fn scroll_diff(&mut self, i: usize, delta: i32) {
        let h = self.editor_rect(i).h;
        if let Some(dv) = self.editors[i].active_diff_mut() {
            let content = dv.content_height();
            dv.scroll_y = scroll::clamp(dv.scroll_y.saturating_add(delta), content, h);
            self.dirty = true;
        }
    }

    pub(super) fn handle_palette_key(&mut self, action: KeyAction) {
        // The arms that need more of the app than the palette, taken before it is
        // borrowed.
        match action {
            KeyAction::Enter => {
                // A free-form field (the custom AI prompt) takes the typed query
                // itself; every other palette takes the selected row's action.
                if self.palette.as_ref().map(Palette::submit) == Some(Submit::Query) {
                    let query = self
                        .palette
                        .as_ref()
                        .map(|p| p.query().trim().to_string())
                        .unwrap_or_default();
                    if !query.is_empty() {
                        self.close_palette();
                        self.start_ai_custom(query);
                    }
                    return;
                }
                if let Some(target) = self.palette.as_ref().and_then(Palette::activate) {
                    self.run_palette_action(target);
                }
                // Enter on a query that matches nothing leaves the palette up: an
                // empty list is a typo to fix, not a reason to start over.
                return;
            }
            KeyAction::Escape | KeyAction::Palette | KeyAction::SearchVault => {
                self.close_palette();
                return;
            }
            KeyAction::Paste => {
                let pasted = match self.clipboard_text() {
                    Ok(Some(pasted)) => pasted,
                    Ok(None) => return,
                    Err(e) => {
                        eprintln!("bnknotes: paste failed: {e}");
                        return;
                    }
                };
                if let Some(palette) = &mut self.palette {
                    palette.push_str(&pasted);
                }
                self.answer_palette_query();
                self.dirty = true;
                return;
            }
            _ => {}
        }
        // Whether the query changed, decided while the palette is borrowed and acted
        // on once it is not: answering a query needs the rest of the app.
        let typed = {
            let Some(palette) = &mut self.palette else {
                return;
            };
            match action {
                KeyAction::Char(c) => {
                    palette.push(c);
                    true
                }
                KeyAction::Backspace => {
                    palette.pop();
                    true
                }
                KeyAction::Up => {
                    palette.move_selection(-1);
                    false
                }
                KeyAction::Down => {
                    palette.move_selection(1);
                    false
                }
                KeyAction::Home | KeyAction::DocStart => {
                    palette.move_selection(i32::MIN);
                    false
                }
                KeyAction::End | KeyAction::DocEnd => {
                    palette.move_selection(i32::MAX);
                    false
                }
                _ => return,
            }
        };
        if typed {
            self.answer_palette_query();
        }
        self.dirty = true;
    }

    /// Re-answer the palette's query, for a palette whose rows are an answer rather
    /// than a list to rank. A ranked palette (the vaults) filters its own rows and has
    /// nothing to ask the app for.
    fn answer_palette_query(&mut self) {
        if self.palette.as_ref().map(Palette::mode) == Some(Rows::AsGiven) {
            self.refresh_search();
        }
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
                let ed = self.editors[i].active_doc_mut();
                match c.to_ascii_lowercase() {
                    'w' => ed.find.toggle_word(&ed.text),
                    'r' => ed.find.toggle_regex(&ed.text),
                    _ => return,
                }
            }
            KeyAction::Char(c) => {
                let ed = self.editors[i].active_doc_mut();
                ed.find.push(c, &ed.text)
            }
            KeyAction::Backspace => {
                let ed = self.editors[i].active_doc_mut();
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
                    let ed = self.editors[i].active_doc_mut();
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
                let ed = self.editors[i].active_doc_mut();
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
        let ed = self.editors[i].active_doc_mut();
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
        let ed = self.editors[i].active_doc_mut();
        let cursor = ed.text.cursor();
        let target = ed.find.refresh(&ed.text, cursor);
        self.land_on_match(target);
    }

    pub(super) fn on_pointer(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_pointer::EV_ENTER => {
                let serial = r.u32()?;
                let _surface = r.u32()?;
                // The compositor reports pointer positions in logical (surface-local)
                // pixels; scale them to the device space the layout and column rects
                // live in, so a click on a HiDPI display lands where it looks.
                let scale = self.scale();
                self.pointer_x = scale.scale_f32(r.fixed()?);
                self.pointer_y = scale.scale_f32(r.fixed()?);
                self.pointer_serial = serial;
                // The compositor resets the cursor on enter, so force a re-set by
                // invalidating the remembered shape, then pick one for the position.
                self.cursor_shape = 0;
                self.update_cursor_shape();
            }
            wl_pointer::EV_LEAVE => self.pointer_left(),
            wl_pointer::EV_MOTION => {
                let _time = r.u32()?;
                let scale = self.scale();
                self.pointer_x = scale.scale_f32(r.fixed()?);
                self.pointer_y = scale.scale_f32(r.fixed()?);
                let (px, py) = (self.pointer_x, self.pointer_y);
                if let Some(s) = self.drag_splitter {
                    self.drag_splitter_to(s, px);
                } else if self.drag_scroll.is_some() {
                    self.drag_scroll_to(py);
                } else if self.drag_tab.is_some() {
                    self.drag_tab_to(px);
                } else if self.drag_vault.is_some() {
                    self.drag_vault_to(py);
                } else {
                    let i = self.active_editor();
                    if matches!(self.focus, Focus::Editor(_)) && self.editors[i].dragging() {
                        // Drag extends the focused editor's selection from the press
                        // point, arming auto-scroll at the column's vertical edges.
                        let rect = self.editor_rect(i);
                        self.caret_from_pointer = true;
                        if self.editors[i].drag_to(&self.fonts, rect, px, py) {
                            self.dirty = true;
                        }
                        self.bump_caret();
                        self.editors[i].update_autoscroll(rect, py);
                    } else {
                        self.update_cursor_shape();
                        self.update_palette_hover();
                        self.update_menu_hover();
                        self.update_rail_hover();
                        self.update_tree_hover();
                        self.update_tab_hover();
                        self.update_scrollbar_hover();
                    }
                }
            }
            wl_pointer::EV_BUTTON => {
                self.last_serial = r.u32()?;
                let _time = r.u32()?;
                let button = r.u32()?;
                let state = r.u32()?;
                // A modal AI preview is driven by the keyboard (Enter accepts, Escape
                // discards): a press is swallowed, not acted on, so refocusing the
                // window or a stray click cannot discard the diff by accident.
                if self.ai.is_some() {
                    return Ok(());
                }
                if button == wl_pointer::BTN_RIGHT {
                    if state == wl_pointer::BUTTON_STATE_PRESSED {
                        self.open_menu(self.pointer_x, self.pointer_y);
                    }
                } else if button == wl_pointer::BTN_LEFT {
                    if state == wl_pointer::BUTTON_STATE_PRESSED {
                        self.on_left_press();
                    } else {
                        // A splitter drag just ended: persist the new column sizes.
                        if self.drag_splitter.take().is_some() {
                            self.save_session();
                        }
                        // A scrollbar drag just ended: let go of the thumb, and let the
                        // bar fade unless the pointer stayed on it.
                        if let Some(region) = self.drag_scroll.take() {
                            self.scrollbar_mut(region).release();
                            self.update_scrollbar_hover();
                            self.dirty = true;
                        }
                        // A tab reorder just ended: persist the new order and let the
                        // lifted tab settle back into its slot. A press that never
                        // grew into a drag was a plain click, already handled.
                        if let Some(drag) = self.drag_tab.take()
                            && drag.moved
                        {
                            self.save_session();
                            self.dirty = true;
                        }
                        // A chip is let go of here rather than at press: this is where
                        // a rail reorder is remembered, and where a chip that never
                        // moved finally switches vault.
                        self.release_vault();
                        let i = self.active_editor();
                        let was_selecting = self.editors[i].release_drag();
                        self.editors[i].disarm_autoscroll();
                        if was_selecting {
                            // The drag is over: image sources under the selection
                            // ends were held collapsed while dragging and can now
                            // expand, so force a redraw.
                            self.dirty = true;
                        }
                    }
                }
            }
            wl_pointer::EV_AXIS => {
                let _time = r.u32()?;
                let axis = r.u32()?;
                let value = r.fixed()?;
                // The scroll step is authored in logical pixels; scale it so a wheel
                // notch moves the same physical distance at any DPI.
                let delta = self.scale().scale_f32(value * SCROLL_FACTOR).round() as i32;
                let (px, py) = (self.pointer_x, self.pointer_y);
                let vertical = axis == wl_pointer::AXIS_VERTICAL_SCROLL;
                // Scroll whichever region the pointer is over, not just the focused
                // one. A tab strip slides horizontally on either wheel axis; the panes
                // and the tree take the vertical wheel only.
                match self.hit_test(px, py) {
                    Hit::Tab(g, _) | Hit::TabClose(g, _) | Hit::NewTab(g) | Hit::TabStrip(g) => {
                        self.scroll_tabs(g, delta);
                    }
                    Hit::Editor(i, _) if vertical && self.editors[i].active_is_diff() => {
                        self.scroll_diff(i, delta);
                        self.flash_scrollbar(Focus::Editor(i));
                    }
                    Hit::Editor(i, rect) if vertical => {
                        if self.editors[i].scroll_by(&self.fonts, rect, delta) {
                            self.flash_scrollbar(Focus::Editor(i));
                        }
                    }
                    Hit::Tree(rect) if vertical => self.scroll_tree(rect, delta),
                    _ => {}
                }
            }
            _ => {} // leave, frame, axis_source, and friends
        }
        Ok(())
    }

    /// What the window position `(px, py)` lands on. An open context menu wins over
    /// the whole window: it floats above every column, and a press off it dismisses
    /// it rather than reaching what it covers. The vault rail is next: it is the one
    /// full-height column, so the bar runs beside it rather than over it. The bar
    /// wins above every remaining column; below it, splitters win over columns (they
    /// sit in the gaps), and each column is full height, so only x decides.
    pub(super) fn hit_test(&self, px: f32, py: f32) -> Hit {
        // The palette floats above everything, the menu included, and takes every
        // press: one on a row goes there, one on the panel is the field's, and one
        // outside puts the palette away without reaching what it covers.
        if let Some(palette) = &self.palette {
            let layout = self.palette_layout(palette);
            if !layout.panel.contains(px, py) {
                return Hit::PaletteOutside;
            }
            return match layout.row_at(px, py).and_then(|vis| palette.row_pos(vis)) {
                Some(pos) => Hit::PaletteRow(pos),
                None => Hit::Palette,
            };
        }
        if let Some(menu) = &self.menu {
            return match self.menu_layout(menu).item_at(px, py) {
                Some(i) => Hit::MenuItem(i),
                None => Hit::Menu,
            };
        }
        // A picture over the window covers everything under it, so it takes every
        // press: the click that lands on what it covers is the click that puts it
        // away, and nothing beneath is clicked by accident on the way out.
        if self.image_view.is_some() {
            return Hit::ImageView;
        }
        let cols = self.columns();
        let x = px as i32;
        if x < cols.rail.x + cols.rail.w {
            return self.rail_hit(cols.rail_chips, px, py);
        }
        if (py as i32) < self.top_bar_h() {
            return self.top_bar_hit(&cols, px, py);
        }
        for (s, rect) in cols.splitters.iter().enumerate() {
            if let Some(r) = rect
                && x >= r.x
                && x < r.x + r.w
            {
                return Hit::Splitter(s);
            }
        }
        if let Some(r) = cols.tree
            && x >= r.x
            && x < r.x + r.w
        {
            let col = self.tree_column(r);
            if py as i32 >= col.footer.y {
                return if col.button.contains(px, py) {
                    Hit::TreeFilter
                } else {
                    Hit::TreeFooter
                };
            }
            return Hit::Tree(col.rows);
        }
        for (i, rect) in cols.editors.iter().enumerate() {
            if let Some(r) = rect
                && x >= r.x
                && x < r.x + r.w
            {
                return Hit::Editor(i, *r);
            }
        }
        Hit::None
    }

    /// Where `menu` is on screen this frame. Measured on demand, the way the tab
    /// strips and the conflict banner are, so the painter and the pointer read one
    /// geometry and a window resized under an open menu has no stale rectangle.
    pub(super) fn menu_layout(&self, menu: &Menu) -> MenuLayout {
        menu.layout(
            &self.fonts,
            self.body_size,
            self.scale(),
            (self.width as i32, self.height as i32),
        )
    }

    /// Where `palette` is on screen this frame, measured on demand for the reason the
    /// menu's is.
    pub(super) fn palette_layout(&self, palette: &Palette) -> PaletteLayout {
        palette.layout(
            &self.fonts,
            self.body_size,
            self.scale(),
            (self.width as i32, self.height as i32),
        )
    }

    /// What a pointer inside the vault rail lands on: a vault's chip, or the column
    /// around them, which consumes the click rather than letting it through. Chips
    /// are measured the same way the painter draws them, from the rail's `chips`
    /// region, so clicks land true.
    fn rail_hit(&self, chips: Rect, px: f32, py: f32) -> Hit {
        for (i, chip) in crate::paint::rail_buttons(chips, self.vaults.len(), self.scale()) {
            if chip.contains(px, py) {
                return Hit::Vault(i);
            }
        }
        Hit::Rail
    }

    /// What a pointer inside the top bar lands on: a tab, its close box, or the
    /// empty strip. The vault label is inert for now, so it falls to `None`. Tabs
    /// are measured the same way the painter draws them, so clicks land true.
    fn top_bar_hit(&self, cols: &Columns, px: f32, py: f32) -> Hit {
        for g in 0..cols.tab_strips.len() {
            let Some(strip) = cols.tab_strips[g] else {
                continue;
            };
            if !strip.contains(px, py) {
                continue;
            }
            let views: Vec<_> = self.editors[g].tab_views().collect();
            let scale = self.scale();
            let layout = crate::paint::tab_layout(strip, &views, scale);
            // The pinned + button wins over any tab scrolled beneath it, so it is
            // tested first; tabs count only within the viewport lane the run is
            // clipped to, matching what the painter draws.
            if crate::paint::new_tab_button_rect(strip, layout, scale).contains(px, py) {
                return Hit::NewTab(g);
            }
            let tabs = crate::paint::tab_rects(strip, &views, scale, self.tab_scroll[g]);
            if (px as i32) < layout.viewport_right(strip) {
                for (i, tab) in tabs.iter().enumerate() {
                    if tab.rect.contains(px, py) {
                        if tab.close.contains(px, py) {
                            return Hit::TabClose(g, i);
                        }
                        return Hit::Tab(g, i);
                    }
                }
            }
            return Hit::TabStrip(g);
        }
        Hit::None
    }

    /// Drag splitter `s` so its column edge follows window x `px`: splitter 0
    /// resizes the sidebar, splitter 1 the A/B content split. Clamped to keep
    /// every column usable.
    pub(super) fn drag_splitter_to(&mut self, s: usize, px: f32) {
        // The pointer, the window size, and the splitter are all device pixels here;
        // the sidebar width is persisted logical, so its edge converts device -> logical
        // on the way in and the min-width floor scales the other way. Both columns are
        // measured from the rail's right edge, where they start, not from the window's.
        let rail = self.columns().rail;
        let x = px as i32 - (rail.x + rail.w);
        let w = self.width as i32 - (rail.x + rail.w);
        let scale = self.scale();
        let splitter = scale.px(SPLITTER);
        if s == 0 {
            let min = scale.px(120);
            let device_edge = x.clamp(min, (w - 2 * splitter - min).max(min));
            self.panes.tree_width = scale.to_logical(device_edge);
        } else {
            let base = if self.panes.tree_visible {
                scale.px(self.panes.tree_width) + splitter
            } else {
                0
            };
            let content = (w - base - splitter).max(1) as f32;
            let a = (x - base) as f32 / content;
            self.panes.split_ratio = a.clamp(0.1, 0.9);
        }
        self.dirty = true;
    }

    /// Take hold of vault chip `i`, pressed at `py`, without acting on it yet: the
    /// press becomes a reorder if the pointer moves and a vault switch if it does not
    /// (see [`Self::release_vault`]).
    fn press_vault(&mut self, i: usize, py: f32) {
        let grab_dy =
            crate::paint::rail_buttons(self.columns().rail_chips, self.vaults.len(), self.scale())
                .find(|(k, _)| *k == i)
                .map(|(_, chip)| py as i32 - chip.y)
                .unwrap_or(0);
        self.drag_vault = Some(VaultDrag {
            vault: i,
            grab_y: py,
            grab_dy,
            moved: false,
        });
    }

    /// Carry a held chip to `py`, reordering the rail as it passes its neighbors.
    /// The rail is reordered as the drag goes rather than at the drop, so the chips
    /// show where the drop will land.
    pub(super) fn drag_vault_to(&mut self, py: f32) {
        let Some(mut drag) = self.drag_vault else {
            return;
        };
        if !drag.moved {
            // A press that has not traveled is still a click; a hand is never quite
            // still, so a chip must not slip out of one on the way to a switch.
            if (py - drag.grab_y).abs() < self.scale().px(DRAG_THRESHOLD) as f32 {
                return;
            }
            drag.moved = true;
            // The hover names a chip by index, and a reorder moves the chips out from
            // under every index; the chip in flight draws its own lift, so the rail
            // has no use for a stale one until the pointer is free again.
            self.hovered_vault = None;
        }
        let target = crate::paint::rail_drop_index(
            self.columns().rail_chips,
            self.vaults.len(),
            self.scale(),
            Some(drag.vault),
            py,
        );
        if target != drag.vault && target < self.vaults.len() {
            let vault = self.vaults.remove(drag.vault);
            self.vaults.insert(target, vault);
            drag.vault = target;
        }
        self.drag_vault = Some(drag);
        self.dirty = true;
    }

    /// Let go of a held chip: a chip that moved leaves the rail in its new order (and
    /// is remembered there), and one that never moved was a click, so it switches the
    /// window to its vault.
    pub(super) fn release_vault(&mut self) {
        let Some(drag) = self.drag_vault.take() else {
            return;
        };
        if drag.moved {
            self.save_session();
            // The pointer is free again, so whatever it came to rest over lights up.
            self.update_rail_hover();
            self.dirty = true;
        } else {
            self.switch_vault(drag.vault);
        }
    }

    /// Move the open palette's selection to the row the pointer is over, repainting if
    /// that changed anything. Nothing to do when no palette is open, which is the
    /// common case.
    ///
    /// The pointer moves the selection rather than lighting a row of its own: the row
    /// under the pointer is the row a click takes, which is what the selection means.
    /// Off the rows it leaves the selection where it was, so crossing the field on the
    /// way to the keyboard does not throw away what the arrows had picked.
    pub(super) fn update_palette_hover(&mut self) {
        let Some(palette) = &self.palette else {
            return;
        };
        let pos = self
            .palette_layout(palette)
            .row_at(self.pointer_x, self.pointer_y)
            .and_then(|vis| palette.row_pos(vis));
        // Re-borrowed mutably only now: the layout above reads the palette it measures.
        if let (Some(pos), Some(palette)) = (pos, &mut self.palette) {
            self.dirty |= palette.select(pos);
        }
    }

    /// Update which item of the open context menu the pointer is over, repainting if
    /// it changed. Nothing to do when no menu is open, which is the common case.
    pub(super) fn update_menu_hover(&mut self) {
        let Some(menu) = &self.menu else {
            return;
        };
        let item = self
            .menu_layout(menu)
            .item_at(self.pointer_x, self.pointer_y);
        // Re-borrowed mutably only now: the layout above reads the menu it measures.
        if let Some(menu) = &mut self.menu {
            self.dirty |= menu.set_hovered(item);
        }
    }

    /// Update which vault chip the pointer is over, repainting if it changed. The
    /// chip under the pointer lifts toward white.
    pub(super) fn update_rail_hover(&mut self) {
        let hovered = match self.hit_test(self.pointer_x, self.pointer_y) {
            Hit::Vault(i) => Some(i),
            _ => None,
        };
        if hovered != self.hovered_vault {
            self.hovered_vault = hovered;
            self.dirty = true;
        }
    }

    /// Update what the pointer is over in the sidebar, a row or the footer's filter
    /// toggle, repainting if it changed. Both light up under the pointer.
    pub(super) fn update_tree_hover(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        let hit = self.hit_test(px, py);
        let hovered = match hit {
            Hit::Tree(rect) => self.tree_row_at(rect, py),
            _ => None,
        };
        let filter = matches!(hit, Hit::TreeFilter);
        if hovered != self.tree.hovered || filter != self.tree.filter_hovered {
            self.tree.hovered = hovered;
            self.tree.filter_hovered = filter;
            self.dirty = true;
        }
    }

    /// Update what the pointer is over in a tab strip, repainting if it changed (a
    /// hovered tab shows a highlight, its close glyph and the new-tab button light
    /// up under the pointer).
    pub(super) fn update_tab_hover(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        let hovered = match self.hit_test(px, py) {
            Hit::Tab(pane, tab) => Some(TabHover::Tab {
                pane,
                tab,
                close: false,
            }),
            Hit::TabClose(pane, tab) => Some(TabHover::Tab {
                pane,
                tab,
                close: true,
            }),
            Hit::NewTab(pane) => Some(TabHover::New { pane }),
            _ => None,
        };
        if hovered != self.hovered_tab {
            self.hovered_tab = hovered;
            self.dirty = true;
        }
    }

    /// The tree row index at window y `py` within the tree column, if any.
    pub(super) fn tree_row_at(&self, rect: Rect, py: f32) -> Option<usize> {
        let row_h = crate::paint::tree_row_height(self.tree_metrics(), self.scale());
        let y = py as i32 - rect.y + self.tree.scroll_y - self.scale().px(TREE_PAD);
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
            self.flash_scrollbar(Focus::Tree);
        }
    }

    /// Open `path` as a tab in the last-focused content pane and focus it: a note
    /// already open there is re-focused rather than duplicated. An unreadable file
    /// is reported without disturbing the pane.
    pub(super) fn open_file(&mut self, path: &Path) {
        let i = self.last_editor;
        match self.editors[i].open_or_focus(path) {
            Ok(()) => {
                self.focus_pane(i);
                self.save_session();
            }
            Err(e) => eprintln!("bnknotes: cannot open {}: {e}", path.display()),
        }
    }

    /// The metrics the sidebar lays out in: its own text size, a step below the body.
    fn tree_metrics(&self) -> Metrics {
        crate::paint::tree_metrics(&self.fonts, self.body_size)
    }

    /// The sidebar column `rect` split into its rows viewport and its footer, as the
    /// painter lays it out.
    pub(super) fn tree_column(&self, rect: Rect) -> crate::paint::TreeColumn {
        crate::paint::tree_column(rect, self.tree_metrics(), self.scale())
    }

    /// The rows' viewport in the tree column (the part above the footer), or an empty
    /// rect when the sidebar is hidden. What a row index and the tree's scroll are
    /// measured against.
    pub(super) fn tree_rect(&self) -> Rect {
        let empty = Rect {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        self.columns()
            .tree
            .map_or(empty, |rect| self.tree_column(rect).rows)
    }

    /// Scroll the tree so its active row is on screen. A no-op when nothing is
    /// active.
    pub(super) fn reveal_tree_selection(&mut self) {
        let Some(sel) = self.tree.selected else {
            return;
        };
        let row_h = crate::paint::tree_row_height(self.tree_metrics(), self.scale());
        let rect = self.tree_rect();
        let tree_pad = self.scale().px(TREE_PAD);
        let top = tree_pad + sel as i32 * row_h;
        let sy = scroll::reveal(
            self.tree.scroll_y,
            top,
            row_h,
            rect.h,
            self.tree.content_height,
            tree_pad,
        );
        if sy != self.tree.scroll_y {
            self.tree.scroll_y = sy;
            self.flash_scrollbar(Focus::Tree);
        }
    }

    /// Keyboard navigation while the tree holds focus: Up/Down move the active
    /// row (starting at the first when nothing is active), Left/Right
    /// collapse/expand a folder, Enter toggles a folder or opens a file into the
    /// last-focused editor.
    pub(super) fn tree_key(&mut self, action: KeyAction) {
        let n = self.tree.tree.rows().len();
        if n == 0 {
            return;
        }
        let last = n - 1;
        let sel = self.tree.selected.unwrap_or(0).min(last);
        self.tree.selected = Some(sel);
        match action {
            KeyAction::Up => self.tree.selected = Some(sel.saturating_sub(1)),
            KeyAction::Down => self.tree.selected = Some((sel + 1).min(last)),
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
    /// selected image's button, the plain arrow over a picture or a scrollbar, the
    /// I-beam over text. Only re-requests when the shape changes, so it is cheap to call
    /// on every motion event. Uses the cached layout (no clone).
    pub(super) fn update_cursor_shape(&mut self) {
        use wp_cursor_shape_device_v1::{SHAPE_DEFAULT, SHAPE_EW_RESIZE, SHAPE_POINTER};
        let (px, py) = (self.pointer_x, self.pointer_y);
        // A scrollbar is a control, not text, so over its grab band the I-beam gives way.
        let shape = if self.on_scrollbar(px, py) {
            SHAPE_DEFAULT
        } else {
            match self.hit_test(px, py) {
                Hit::Splitter(_) => SHAPE_EW_RESIZE,
                Hit::Tree(_) => SHAPE_DEFAULT,
                // A diff review is not editable, so its body takes the plain arrow
                // rather than the text I-beam.
                Hit::Editor(i, _) if self.editors[i].active_is_diff() => SHAPE_DEFAULT,
                Hit::Editor(i, rect) => self.editors[i].cursor_shape_at(&self.fonts, rect, px, py),
                Hit::Tab(..)
                | Hit::TabClose(..)
                | Hit::NewTab(_)
                | Hit::TreeFilter
                | Hit::MenuItem(_)
                | Hit::PaletteRow(_)
                | Hit::Vault(_) => SHAPE_POINTER,
                Hit::TabStrip(_)
                | Hit::TreeFooter
                | Hit::Rail
                | Hit::Menu
                | Hit::Palette
                | Hit::PaletteOutside
                | Hit::ImageView
                | Hit::None => SHAPE_DEFAULT,
            }
        };
        if shape != self.cursor_shape {
            self.cursor_shape = shape;
            self.set_cursor_shape(shape);
        }
    }

    /// The scrollable column a window position falls in, paired with the rectangle its
    /// scrollbar is measured against (the editor's whole column; the tree's rows, above
    /// its pinned footer). `None` off every column, and for a column whose content fits,
    /// which has no bar to speak of.
    fn scroll_column_at(&self, px: f32, py: f32) -> Option<(Focus, Rect)> {
        let (region, view) = match self.hit_test(px, py) {
            Hit::Editor(i, rect) => (Focus::Editor(i), rect),
            Hit::Tree(rows) => (Focus::Tree, rows),
            _ => return None,
        };
        let (content_h, view_h) = self.scroll_extent(region, view);
        scroll::scrollable(content_h, view_h).then_some((region, view))
    }

    /// How tall the document in `region` is and how much of it the column `view` shows,
    /// the two numbers every scrollbar question (is it scrollable, how long is the
    /// thumb, where does a drag land) is answered from.
    fn scroll_extent(&self, region: Focus, view: Rect) -> (i32, i32) {
        match region {
            Focus::Tree => (self.tree.content_height, view.h),
            // A diff review's scrollable extent is its own, not the placeholder
            // document the pane derefs to while the review is active.
            Focus::Editor(i) => {
                let content = match self.editors[i].active_diff() {
                    Some(dv) => dv.content_height(),
                    None => self.editors[i].content_height,
                };
                (content, view.h)
            }
        }
    }

    /// This region's scrollbar.
    fn scrollbar_mut(&mut self, region: Focus) -> &mut Scrollbar {
        match region {
            Focus::Tree => &mut self.tree.scrollbar,
            Focus::Editor(i) => &mut self.editors[i].scrollbar,
        }
    }

    /// Light the scrollbar of the region that just scrolled, so every scroll (a wheel
    /// notch, a caret revealed by a keystroke, a drag auto-scroll) is confirmed by the
    /// bar sliding into view and then fading back out.
    pub(super) fn flash_scrollbar(&mut self, region: Focus) {
        let now = Instant::now();
        self.scrollbar_mut(region).flash(now);
        self.dirty = true;
    }

    /// Follow the pointer with the scrollbars: the bar of the column whose zone it has
    /// entered grows into a slider, every other bar shrinks back. Repaints when one
    /// changed. A live drag owns its bar and ignores this.
    pub(super) fn update_scrollbar_hover(&mut self) {
        if self.drag_scroll.is_some() {
            return;
        }
        let (px, py) = (self.pointer_x, self.pointer_y);
        let scale = self.scale();
        let near = self.scroll_column_at(px, py).filter(|(_, view)| {
            crate::paint::scroll_lane(*view, scale)
                .zone
                .contains(px, py)
        });
        let mut changed = false;
        for region in [Focus::Tree, Focus::Editor(0), Focus::Editor(1)] {
            let is_near = near.is_some_and(|(r, _)| r == region);
            changed |= self.scrollbar_mut(region).set_near(is_near);
        }
        if changed {
            self.dirty = true;
        }
    }

    /// Whether the pointer is on a scrollbar, meaning a press would take its thumb
    /// rather than reach the column beneath it.
    fn on_scrollbar(&self, px: f32, py: f32) -> bool {
        let scale = self.scale();
        self.scroll_column_at(px, py)
            .is_some_and(|(_, view)| crate::paint::scroll_lane(view, scale).grab.contains(px, py))
    }

    /// A press in a scrollbar's lane: take the thumb under the pointer, or, when the
    /// press landed on the empty track, warp the thumb to it first and take it there,
    /// so a click jumps to that spot and holding turns it into a scrub. Returns
    /// whether the press was the scrollbar's.
    fn scrollbar_press(&mut self, px: f32, py: f32) -> bool {
        if !self.on_scrollbar(px, py) {
            return false;
        }
        let scale = self.scale();
        let Some((region, view)) = self.scroll_column_at(px, py) else {
            return false;
        };
        let lane = crate::paint::scroll_lane(view, scale);
        let (content_h, view_h) = self.scroll_extent(region, view);
        let min = scale.px(SCROLL_MIN_THUMB);
        let Some(thumb) = scroll::thumb(lane.track, view_h, content_h, self.scroll_y(region), min)
        else {
            return false;
        };
        // On the thumb, the grab keeps its offset so it does not jump under the finger;
        // on the track, the thumb lands centered on the pointer.
        let grab = if py >= thumb.y as f32 && py < (thumb.y + thumb.h) as f32 {
            py as i32 - thumb.y
        } else {
            thumb.h / 2
        };
        self.scrollbar_mut(region).press(grab);
        self.drag_scroll = Some(region);
        self.drag_scroll_to(py);
        true
    }

    /// Track a scrollbar drag: put the thumb's top where the pointer holds it and scroll
    /// the column to match.
    pub(super) fn drag_scroll_to(&mut self, py: f32) {
        let Some(region) = self.drag_scroll else {
            return;
        };
        let scale = self.scale();
        let view = match region {
            Focus::Tree => self.tree_column(self.tree_rect()).rows,
            Focus::Editor(i) => self.editor_rect(i),
        };
        let lane = crate::paint::scroll_lane(view, scale);
        let (content_h, view_h) = self.scroll_extent(region, view);
        let at = self.scroll_y(region);
        let min = scale.px(SCROLL_MIN_THUMB);
        let (Some(grab), Some(thumb)) = (
            self.scrollbar_mut(region).grab(),
            scroll::thumb(lane.track, view_h, content_h, at, min),
        ) else {
            return;
        };
        let sy = scroll::scroll_at_thumb(lane.track, thumb.h, view_h, content_h, py as i32 - grab);
        if sy != at {
            match region {
                Focus::Tree => self.tree.scroll_y = sy,
                Focus::Editor(i) => match self.editors[i].active_diff_mut() {
                    Some(dv) => dv.scroll_y = sy,
                    None => self.editors[i].scroll_y = sy,
                },
            }
            self.dirty = true;
        }
    }

    /// Where the region is scrolled to right now.
    fn scroll_y(&self, region: Focus) -> i32 {
        match region {
            Focus::Tree => self.tree.scroll_y,
            Focus::Editor(i) => match self.editors[i].active_diff() {
                Some(dv) => dv.scroll_y,
                None => self.editors[i].scroll_y,
            },
        }
    }

    /// Handle a left-button press: route it to whatever it landed on. An open context
    /// menu takes the press before anything else, a scrollbar's own band included; a
    /// scrollbar takes the press before the column under it, a splitter starts a
    /// drag-resize, a tree row expands a folder or opens a file, and an editor column
    /// takes focus and then handles the click as a text press.
    pub(super) fn on_left_press(&mut self) {
        let (px, py) = (self.pointer_x, self.pointer_y);
        // A scrollbar takes a press before the column under it, but not before a
        // pop-up over it: a press while one is open belongs to the pop-up.
        if self.menu.is_none() && self.palette.is_none() && self.scrollbar_press(px, py) {
            return;
        }
        match self.hit_test(px, py) {
            Hit::PaletteRow(pos) => {
                if let Some(target) = self.palette.as_ref().and_then(|p| p.action_at(pos)) {
                    self.run_palette_action(target);
                }
            }
            // The panel's own field and padding: the press is the palette's, and
            // there is nothing there to act on.
            Hit::Palette => {}
            Hit::PaletteOutside => {
                self.close_palette();
            }
            Hit::MenuItem(i) => {
                if let Some(action) = self.menu.as_ref().and_then(|m| m.action(i)) {
                    self.run_menu_action(action);
                }
            }
            // A press anywhere else while the menu is open only dismisses it: the
            // click that closes a menu belongs to the menu, not to what is under it.
            Hit::Menu => {
                self.close_menu();
            }
            // Any press puts an opened picture away: it covers the window, so there
            // is nothing else the click could have been meant for.
            Hit::ImageView => {
                self.close_image_view();
            }
            // A chip takes hold of the press without acting on it: what it meant is
            // only known at release, when the pointer either moved (a reorder) or did
            // not (a switch).
            Hit::Vault(i) => self.press_vault(i, py),
            // The rail around its chips: a click there acts on nothing, but it must
            // not reach the column beside it.
            Hit::Rail => {}
            Hit::Splitter(s) => self.drag_splitter = Some(s),
            Hit::Tree(rect) => self.tree_press(rect, py),
            Hit::TreeFilter => self.toggle_empty_dirs(),
            // The footer beside the toggle: a click there acts on nothing, but it must
            // not reach the row scrolled beneath the strip.
            Hit::TreeFooter => {}
            Hit::Editor(i, rect) => {
                // A press in a diff review just focuses its pane (so the keyboard
                // resolves it); there is no text under it to place a caret in.
                if self.editors[i].active_is_diff() {
                    self.focus_pane(i);
                    return;
                }
                // A disk-conflict banner sits over the pane's top edge; a press on
                // it acts on its buttons rather than placing a caret beneath it.
                if self.conflict_banner_press(i, rect, px, py) {
                    return;
                }
                // A click places the caret; that jump snaps rather than smearing.
                self.caret_from_pointer = true;
                self.focus_pane(i);
                self.editor_press(i, rect, px, py);
            }
            Hit::Tab(g, i) => self.press_tab(g, i, px),
            // The pin badge stands where the close cross would: a pinned tab's slot
            // unpins rather than closes, so an accidental click cannot lose the tab.
            Hit::TabClose(g, i) => {
                if self.editors[g].is_pinned(i) {
                    self.toggle_pin_tab(g, i);
                } else {
                    self.close_tab(g, i);
                }
            }
            Hit::NewTab(g) => self.new_tab_in(g),
            // The empty part of a pane's tab strip still focuses that pane.
            Hit::TabStrip(g) => self.focus_pane(g),
            Hit::None => {}
        }
    }

    /// A press inside pane `i`: when its active tab shows a disk-conflict banner and
    /// the press lands on the banner, act on the button under it (and swallow a press
    /// anywhere else on the strip so it does not reach the text). Reports whether the
    /// click was the banner's.
    fn conflict_banner_press(&mut self, i: usize, rect: Rect, px: f32, py: f32) -> bool {
        let Some(kind) = self.editors[i].conflict else {
            return false;
        };
        let banner = conflict_banner_layout(rect, kind, self.body_size, &self.fonts);
        if !banner.contains(px, py) {
            return false;
        }
        self.focus_pane(i);
        if let Some(button) = banner.button_at(px, py) {
            self.resolve_conflict(i, kind, button);
        }
        true
    }

    /// Apply the user's choice on pane `i`'s disk-conflict banner: take the on-disk
    /// version, or keep the buffer's.
    fn resolve_conflict(&mut self, i: usize, kind: DiskConflict, button: BannerButton) {
        match (kind, button) {
            // Reload: drop the buffer's edits for the file's on-disk contents.
            (DiskConflict::Changed, BannerButton::Primary) => {
                if let Err(e) = self.editors[i].active_doc_mut().reload_from_disk() {
                    eprintln!("bnknotes: reload failed: {e}");
                }
            }
            // Keep mine: accept the on-disk version as seen and keep the edits, so a
            // later save overwrites it deliberately rather than being blocked again.
            (DiskConflict::Changed, BannerButton::Secondary) => {
                self.editors[i].active_doc_mut().sync_disk();
            }
            // Save: recreate the deleted file from the buffer. The save's success
            // clears the conflict.
            (DiskConflict::Deleted, BannerButton::Primary) => {
                self.focus_pane(i);
                if let Err(e) = self.save() {
                    eprintln!("bnknotes: {e}");
                }
            }
            // Dismiss: treat the file as gone (an unsaved buffer); a later save
            // recreates it.
            (DiskConflict::Deleted, BannerButton::Secondary) => {
                self.editors[i].active_doc_mut().sync_disk();
            }
        }
        self.dirty = true;
        self.refresh_title();
        self.save_session();
    }

    /// Switch pane `g` to tab `i` and focus that pane.
    fn select_tab(&mut self, g: usize, i: usize) {
        self.editors[g].activate(i);
        self.focus_pane(g);
        self.save_session();
    }

    /// A press on tab `i` of pane `g`: select it, and arm a reorder drag anchored at
    /// pointer x `px`. The drag only starts moving tabs once the pointer crosses the
    /// click threshold, so a plain click still just selects.
    fn press_tab(&mut self, g: usize, i: usize, px: f32) {
        self.select_tab(g, i);
        let grab_dx = self
            .tab_strip_rects(g)
            .and_then(|(_, tabs)| tabs.get(i).map(|t| (px - t.rect.x as f32) as i32))
            .unwrap_or(0);
        self.drag_tab = Some(TabDrag {
            pane: g,
            tab: i,
            grab_x: px,
            grab_dx,
            moved: false,
        });
    }

    /// Continue a tab drag as the pointer reaches x `px`: once past the click
    /// threshold, put the dragged tab where the pointer points. Within its own pane
    /// that is a reorder; over the other pane's strip the tab crosses the split,
    /// leaving its old pane and re-homing in the one under the pointer. The frame
    /// keeps repainting so the lifted tab tracks the cursor.
    fn drag_tab_to(&mut self, px: f32) {
        let Some(mut drag) = self.drag_tab else {
            return;
        };
        if !drag.moved {
            if (px - drag.grab_x).abs() < self.scale().px(DRAG_THRESHOLD) as f32 {
                return;
            }
            drag.moved = true;
        }
        let over = self.tab_target_pane(px).unwrap_or(drag.pane);
        // A diff review re-homes to no document, so it cannot cross the split; it stays
        // in its pane, where it can still be reordered.
        if over != drag.pane && !self.editors[drag.pane].tab_is_diff(drag.tab) {
            // Cross the split: detach from the source pane and insert where the
            // pointer sits in the target's strip, which then owns and focuses it.
            let idx = self
                .tab_strip_rects(over)
                .map(|(_, tabs)| crate::paint::tab_drop_index(&tabs, None, px))
                .unwrap_or(0);
            let pinned = self.editors[drag.pane].is_pinned(drag.tab);
            if let Some(doc) = self.editors[drag.pane].detach(drag.tab) {
                drag.tab = self.editors[over].insert_tab(idx, doc);
                self.editors[over].set_pinned(drag.tab, pinned);
                drag.pane = over;
                self.focus_pane(over);
            }
        } else if let Some((_, tabs)) = self.tab_strip_rects(drag.pane) {
            let target = crate::paint::tab_drop_index(&tabs, Some(drag.tab), px);
            if target != drag.tab {
                self.editors[drag.pane].move_tab(drag.tab, target);
                drag.tab = target;
            }
        }
        self.drag_tab = Some(drag);
        self.dirty = true;
    }

    /// Which pane's tab strip the pointer x `px` sits over, if any. Only panes that
    /// are actually shown have a strip, so this is `None` over the tree, a splitter,
    /// or a closed second pane, which is exactly when a tab has nowhere to cross to.
    fn tab_target_pane(&self, px: f32) -> Option<usize> {
        let strips = self.columns().tab_strips;
        (0..strips.len())
            .find(|&g| strips[g].is_some_and(|s| px >= s.x as f32 && px < (s.x + s.w) as f32))
    }

    /// Pane `g`'s tab strip rect and its per-tab geometry, measured the same way the
    /// painter lays them out, so a drag reorders against exactly what is drawn.
    fn tab_strip_rects(&self, g: usize) -> Option<(Rect, Vec<crate::paint::TabHit>)> {
        let strip = self.columns().tab_strips[g]?;
        let views: Vec<_> = self.editors[g].tab_views().collect();
        let tabs = crate::paint::tab_rects(strip, &views, self.scale(), self.tab_scroll[g]);
        Some((strip, tabs))
    }

    /// Slide pane `g`'s tab strip horizontally by `delta` device pixels, clamped to
    /// the overflow, repainting if it moved. A no-op when the tabs already fit.
    fn scroll_tabs(&mut self, g: usize, delta: i32) {
        let Some(strip) = self.columns().tab_strips[g] else {
            return;
        };
        let views: Vec<_> = self.editors[g].tab_views().collect();
        let layout = crate::paint::tab_layout(strip, &views, self.scale());
        let want = scroll::clamp(self.tab_scroll[g] + delta, layout.content, layout.avail);
        if want != self.tab_scroll[g] {
            self.tab_scroll[g] = want;
            self.dirty = true;
        }
    }

    /// Open a fresh blank tab in pane `g` and focus it (the new-tab button).
    fn new_tab_in(&mut self, g: usize) {
        let (scale, body) = (self.scale(), self.body_size);
        self.editors[g].new_tab(scale, body);
        self.focus_pane(g);
        self.save_session();
    }

    /// Close tab `i` of pane `g` (a blank tab remains if it was the last) and focus
    /// that pane on its now-active tab.
    ///
    /// Closing drops the document, so its edits go to disk first, on the same terms
    /// as everything else: nothing typed is lost by being closed, and the idle window
    /// is not a race against Ctrl+W.
    fn close_tab(&mut self, g: usize, i: usize) {
        if let Some(doc) = self.editors[g].docs_mut().nth(i)
            && let Err(e) = doc.save()
        {
            eprintln!("bnknotes: could not save the closing tab: {e}");
        }
        self.editors[g].close(i);
        self.hovered_tab = None;
        self.focus_pane(g);
        self.save_session();
    }

    /// The footer toggle: list, or hide, the folders with no notes beneath them. The
    /// choice is the vault's, so it is remembered per vault. Rebuilding the tree
    /// renumbers its rows, so the active row is followed by path across the rebuild
    /// rather than by index; a row the filter just hid stops being active.
    pub(super) fn toggle_empty_dirs(&mut self) {
        let selected = self
            .tree
            .selected
            .and_then(|i| self.tree.tree.rows().get(i))
            .map(|row| row.path.clone());
        let show = !self.tree.tree.show_empty_dirs();
        self.tree.tree.set_show_empty_dirs(show);
        self.tree.selected = selected.and_then(|path| self.tree.tree.index_of(&path));
        self.tree.hovered = None;
        self.vault_prefs
            .set_show_empty_dirs(self.tree.tree.root(), show);
        self.reveal_tree_selection();
        self.dirty = true;
        self.save_session();
    }

    /// A press in the tree: focus the sidebar, mark the clicked row active, then
    /// toggle a folder's expansion or open a file into the last-focused editor.
    pub(super) fn tree_press(&mut self, rect: Rect, py: f32) {
        self.focus = Focus::Tree;
        let Some(idx) = self.tree_row_at(rect, py) else {
            self.dirty = true;
            return;
        };
        let Some(row) = self.tree.tree.rows().get(idx) else {
            return;
        };
        self.tree.selected = Some(idx);
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
        // A click on a heading's fold chevron collapses or opens its section. The
        // chevron sits in the margin, left of the text, so this never competes with
        // placing the caret.
        if let Some(ls) = self.editors[i].fold_under_pointer(&self.fonts, rect, px, py) {
            if self.editors[i].toggle_fold_at(&self.fonts, rect, ls) {
                self.dirty = true;
                self.save_session();
            }
            self.bump_caret();
            return;
        }
        // A click on a table's add-row / add-column hover strip grows the table.
        if self.editors[i].press_table_button(&self.fonts, rect, px, py) {
            self.dirty = true;
            self.bump_caret();
            return;
        }
        // Image interactions win over text.
        match self.editors[i].image_press(&self.fonts, rect, px, py) {
            ImagePress::Button => {
                self.dirty = true;
                self.bump_caret();
                return;
            }
            ImagePress::Picture => {
                // A second click on the picture opens it over the whole window. The
                // chain is registered here rather than on the way past, since a click
                // the image path consumes never reaches the text's own counting.
                let (count, _) = self.editors[i].clicks.register(px, py, Instant::now());
                if count >= 2 {
                    self.open_image_view(i);
                }
                self.dirty = true;
                self.bump_caret();
                return;
            }
            ImagePress::Missed => {}
        }
        self.editors[i].selected_image = None;
        self.editors[i].raw_image = None;
        // A plain click on a rendered task checkbox toggles it without moving the
        // caret. Shift-click falls through so it can extend a selection.
        if !shift
            && let Some(ls) = self.editors[i].checkbox_under_pointer(&self.fonts, rect, px, py)
        {
            if self.editors[i].text.toggle_task(ls) {
                self.dirty = true;
            }
            self.bump_caret();
            return;
        }
        // Click chaining for double/triple-click.
        let (click_count, chained) = self.editors[i].clicks.register(px, py, Instant::now());
        // A plain click on a rendered horizontal rule selects the whole rule line,
        // and a drag on from it sweeps the selection out of the rule.
        if !shift && self.editors[i].press_rule(&self.fonts, rect, py) {
            self.dirty = true;
            self.bump_caret();
            return;
        }
        // A plain, un-chained click on a rendered link follows it.
        if !chained
            && !shift
            && let Some(dest) = self.editors[i].link_under_pointer(&self.fonts, rect, px, py)
            && crate::platform::browser::open_url(&dest).is_ok()
        {
            self.editors[i].end_drag();
            return;
        }
        match click_count {
            1 => {
                self.editors[i].press_at(&self.fonts, rect, px, py, shift);
                // Show the moved caret on the next frame, not the next blink wake.
                self.dirty = true;
            }
            2 => {
                self.editors[i].press_word(&self.fonts, rect, px, py);
                self.dirty = true;
            }
            _ => {
                self.editors[i].press_line(&self.fonts, rect, px, py);
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
        if !self.editors[i].dragging() {
            self.editors[i].disarm_autoscroll();
            return;
        }
        let Some(step) = self.editors[i].autoscroll_due(Instant::now()) else {
            return;
        };
        let rect = self.editor_rect(i);
        let (px, py) = (self.pointer_x, self.pointer_y);
        if self.editors[i].scroll_by(&self.fonts, rect, step) {
            self.flash_scrollbar(Focus::Editor(i));
        }
        if self.editors[i].drag_to(&self.fonts, rect, px, py) {
            self.dirty = true;
        }
    }

    // --- Caret blink ------------------------------------------------------

    /// Advance the caret's blink clock, requesting a redraw when the caret changed
    /// visibility.
    pub(super) fn fire_caret_blink(&mut self) {
        // Nothing draws a caret right now: an unfocused window, focus on the tree or
        // a diff tab, or a cursor scrolled out of the painted band. Rest the clock
        // rather than toggle a caret nobody sees, or every toggle would rebuild a
        // frame that comes out identical.
        let Some(in_flight) = self.drawable_caret_anim().map(CaretAnim::in_flight) else {
            self.caret.rest();
            return;
        };
        self.dirty |= self.caret.tick(Instant::now(), in_flight);
    }

    /// Show the caret and restart its blink run, called on any activity that
    /// moves the caret, so it never blinks out mid-edit.
    pub(super) fn bump_caret(&mut self) {
        self.caret.bump(Instant::now());
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
            self.apply_action(keymap::action(&self.xkb, keycode))?;
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
