//! The editing model: a UTF-8 string, a cursor, and an optional selection
//! anchor, all as byte offsets onto grapheme cluster boundaries.
//!
//! The buffer is one `String`; newlines are ordinary `'\n'` bytes, so multi-line
//! text needs no special structure here (a [`crate::layout`] pass turns it into
//! visual lines). The cursor and the selection anchor are byte offsets that
//! always sit on a grapheme cluster boundary, so motion and deletion act on
//! whole user-perceived characters: Backspace removes a flag or a ZWJ family
//! emoji in one stroke, never splitting it. When an anchor is set and differs
//! from the cursor, the half-open range between them is the selection; typing or
//! Backspace replaces it.
//!
//! Vertical motion (up/down) and click placement are not here: they need glyph
//! geometry, so the app computes a target offset via the layout and calls
//! [`TextBuffer::set_cursor`].
//!
//! Undo/redo is snapshot based: each edit first checkpoints the pre-edit state
//! onto the undo stack. Consecutive edits of the same kind coalesce into one
//! step, so typing a word, or a run of held-Backspace deletes, undoes in a
//! single stroke; any cursor move breaks the run.
//!
//! Each distinct content state carries a monotonic version number, captured in
//! its snapshot so undo/redo restore it too. [`TextBuffer::mark_saved`] records
//! the version that is on disk, and [`TextBuffer::is_modified`] compares against
//! it, so the buffer reads as clean again the moment an edit is undone back to
//! the saved state.

use std::collections::VecDeque;

use crate::grapheme;
use crate::markdown;
use crate::rope::Rope;

/// Cap on undo history, so a long session cannot grow the stacks without bound.
const UNDO_CAP: usize = 500;

/// The kind of edit a checkpoint groups. Consecutive edits of the same non-`None`
/// kind merge into one undo step.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Coalesce {
    None,
    Insert,
    Delete,
}

/// A restorable editing state. Cloning the `Rope` is O(1) and shares the text
/// with the live buffer, so checkpoints are cheap regardless of document size.
#[derive(Clone)]
struct Snapshot {
    text: Rope,
    cursor: usize,
    anchor: Option<usize>,
    version: u64,
}

pub struct TextBuffer {
    text: Rope,
    /// Byte offset of the cursor; always a grapheme cluster boundary.
    cursor: usize,
    /// The other end of the selection, if one is active. `None` means no
    /// selection; equal to `cursor` means a collapsed (empty) one.
    anchor: Option<usize>,
    /// Undo history, oldest at the front. A deque so trimming the oldest entry
    /// once [`UNDO_CAP`] is reached is a cheap `pop_front`, not an O(cap) shift.
    undo: VecDeque<Snapshot>,
    redo: Vec<Snapshot>,
    /// The coalescing group of the last edit; a cursor move resets it to `None`.
    group: Coalesce,
    /// Monotonic version of the current contents, bumped on every content
    /// change and restored from snapshots by undo/redo.
    version: u64,
    /// The next unused version number.
    next_version: u64,
    /// The version last written to disk, against which `is_modified` compares.
    saved_version: u64,
}

impl TextBuffer {
    /// An empty, unsaved buffer.
    pub fn new() -> Self {
        Self::from_text(String::new())
    }

    /// A buffer initialized with `text` (e.g. a file just read from disk): the
    /// cursor sits at the top, there is no undo history to walk back into, and
    /// the contents start out clean (not modified).
    pub fn from_text(text: String) -> Self {
        Self {
            text: Rope::from_str(&text),
            cursor: 0,
            anchor: None,
            undo: VecDeque::new(),
            redo: Vec::new(),
            group: Coalesce::None,
            version: 0,
            next_version: 1,
            saved_version: 0,
        }
    }

    /// The full buffer contents as an owned `String`. The rope is not
    /// contiguous, so this materializes it; used for saving and in tests, not on
    /// the per-frame path (which reads lines from [`rope`](Self::rope)).
    pub fn text(&self) -> String {
        self.text.to_string()
    }

    /// The backing rope, for the layout to read lines from without materializing
    /// the whole document.
    pub fn rope(&self) -> &Rope {
        &self.text
    }

    /// Whether the contents differ from the last point marked saved.
    pub fn is_modified(&self) -> bool {
        self.version != self.saved_version
    }

    /// The current content version. Distinct contents always have distinct
    /// versions (each edit takes a fresh number, and undo/redo restore the
    /// version along with the text), so it is a sound cache key for anything
    /// derived purely from the buffer contents, like the layout.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Record that the current contents are now on disk, so [`is_modified`]
    /// reads clean until the next change. Also breaks the coalescing group, so
    /// the next edit checkpoints the saved state and can be undone back to it.
    ///
    /// [`is_modified`]: Self::is_modified
    pub fn mark_saved(&mut self) {
        self.saved_version = self.version;
        self.group = Coalesce::None;
    }

    /// Stamp the contents with a fresh version. Called after every change so
    /// modified-tracking and undo can tell distinct states apart.
    fn touch(&mut self) {
        self.version = self.next_version;
        self.next_version += 1;
    }

    /// Byte offset of the cursor (always a grapheme cluster boundary).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The selection as an ordered half-open byte range, or `None` when there is
    /// no selection or it is collapsed onto the cursor.
    pub fn selection(&self) -> Option<(usize, usize)> {
        let anchor = self.anchor?;
        if anchor == self.cursor {
            None
        } else {
            Some((anchor.min(self.cursor), anchor.max(self.cursor)))
        }
    }

    /// The selected text, if any. Used by Copy/Cut.
    pub fn selected_text(&self) -> Option<String> {
        self.selection()
            .map(|(start, end)| self.text.byte_slice(start..end))
    }

    /// Insert `s` at the cursor, first replacing any selection, and advance past
    /// it. Clears the selection.
    pub fn insert_str(&mut self, s: &str) {
        // A single non-whitespace character coalesces with neighboring typing;
        // pastes, whitespace, and selection-replacements are their own steps.
        let group = if self.selection().is_none() && coalescable_insert(s) {
            Coalesce::Insert
        } else {
            Coalesce::None
        };
        self.checkpoint(group);
        self.delete_selection();
        self.text.insert(self.cursor, s);
        self.cursor += s.len();
        self.anchor = None;
        self.touch();
    }

    /// Replace every `[start, end)` range in `ranges` (ascending, disjoint,
    /// in-bounds — the find machinery's match list) with `with`, as a single
    /// undo step. The cursor lands at the end of the last replacement.
    /// Returns whether anything changed.
    pub fn replace_ranges(&mut self, ranges: &[(usize, usize)], with: &str) -> bool {
        if ranges.is_empty() {
            return false;
        }
        self.checkpoint(Coalesce::None);
        // Edit back to front so earlier ranges stay valid while later ones are
        // spliced; each replacement before the last shifts the last one's
        // position by the length difference.
        for &(start, end) in ranges.iter().rev() {
            self.text.remove(start..end);
            self.text.insert(start, with);
        }
        let last_start = ranges[ranges.len() - 1].0;
        let shift: isize = ranges[..ranges.len() - 1]
            .iter()
            .map(|&(s, e)| with.len() as isize - (e - s) as isize)
            .sum();
        self.cursor = (last_start as isize + shift).max(0) as usize + with.len();
        self.anchor = None;
        self.touch();
        true
    }

    /// Handle the Enter key, continuing a list or quote where one is in play.
    /// On a bullet, ordered, task, or quote line a non-empty item starts a new
    /// one with the same (or next) marker, splitting the line at the cursor; an
    /// empty item instead has its marker removed, leaving a blank line, so a
    /// second Enter ends the list. Inserting a numbered item also renumbers the
    /// same-level items after it, in the same undo step. Anywhere else, with a
    /// selection, or with the cursor before the marker, it inserts a plain
    /// newline. Always changes the buffer, so it returns `true`.
    pub fn enter(&mut self) -> bool {
        if self.selection().is_some() {
            self.insert_str("\n");
            return true;
        }
        let (ls, le) = self.line_content_bounds(self.cursor);
        let line = self.text.byte_slice(ls..le);
        let (style, marker_len) = markdown::classify(&line);
        // Continue the list only while editing the item's content; with the
        // cursor before or within the marker a plain newline is what is meant.
        if marker_len == 0 || self.cursor < ls + marker_len {
            // Auto-indent a plain newline: the new line inherits this line's
            // leading whitespace (capped at the cursor), so indented prose and
            // code keep their indent. Only for non-list lines; a continued list
            // carries its indent in the marker, and a cursor inside a marker
            // means a deliberate plain break.
            let indent = if marker_len == 0 {
                inherited_indent(&line, self.cursor - ls)
            } else {
                ""
            };
            if indent.is_empty() {
                self.insert_str("\n");
            } else {
                self.insert_str(&format!("\n{indent}"));
            }
            return true;
        }
        match markdown::next_item_marker(&line) {
            Some(_) if line[marker_len..].trim().is_empty() => self.delete_range(ls, le),
            Some(marker) => {
                self.insert_str(&format!("\n{marker}"));
                // The new number bumps every same-level item after it; the
                // renumbering edits directly (no fresh checkpoint), so it
                // shares the insertion's undo step.
                if matches!(style, markdown::LineStyle::Ordered) {
                    self.renumber_following_items();
                }
                true
            }
            None => {
                self.insert_str("\n");
                true
            }
        }
    }

    /// Renumber the ordered items that follow the cursor's line at its list
    /// level, so they continue its number in sequence. Runs after Enter inserts
    /// the next numbered item; edits the rope directly with no checkpoint, so
    /// the renumbering and the insertion undo as one step. Blank lines are
    /// skipped (a loose list is still one list) and deeper-indented items are
    /// left alone (a child list numbers itself); the first shallower or
    /// differently-marked line ends the list, matching how the layout resolves
    /// list nesting. All the edits land after the cursor's line, so the cursor
    /// never moves.
    fn renumber_following_items(&mut self) {
        let (ls, le) = self.line_content_bounds(self.cursor);
        let line = self.text.byte_slice(ls..le);
        let Some((mut number, _)) = markdown::ordered_number(&line) else {
            return;
        };
        let indent = markdown::leading_indent(&line);
        // Collect the digit-range edits first: renumbering can change byte
        // lengths ("9" -> "10"), so they apply back to front afterwards.
        let mut edits: Vec<(usize, usize, u64)> = Vec::new();
        let n = self.text.len_lines();
        for i in self.text.byte_to_line(self.cursor) + 1..n {
            let (ls, le) = self.text.line_content_range(i);
            let line = self.text.byte_slice(ls..le);
            if line.trim().is_empty() {
                continue;
            }
            let li = markdown::leading_indent(&line);
            if is_list(&line) && li > indent {
                continue; // a nested child; its numbering is its own
            }
            // A shallower line, a non-list line, or a same-level bullet/task
            // (a different list marker) all end this ordered run.
            let Some((value, (ds, de))) = markdown::ordered_number(&line) else {
                break;
            };
            if li < indent {
                break;
            }
            number += 1;
            if value != number {
                edits.push((ls + ds, ls + de, number));
            }
        }
        if edits.is_empty() {
            return;
        }
        for &(start, end, num) in edits.iter().rev() {
            self.text.remove(start..end);
            self.text.insert(start, &num.to_string());
        }
        // A fresh version: the insertion's stamp covered the pre-renumber text,
        // and distinct contents must never share a version.
        self.touch();
    }

    /// Toggle the checkbox of the task item on the line containing `offset`
    /// (`[ ]` <-> `[x]`), as one undo step. The swap is a single byte, and the
    /// cursor and selection stay untouched, so toggling never moves the caret
    /// or reveals the line. Returns whether the buffer changed (`false` off a
    /// task line).
    pub fn toggle_task(&mut self, offset: usize) -> bool {
        let (ls, le) = self.line_content_bounds(offset.min(self.text.len()));
        let line = self.text.byte_slice(ls..le);
        let (style, marker_len) = markdown::classify(&line);
        let markdown::LineStyle::Task { checked } = style else {
            return false;
        };
        // The check character sits inside the marker's trailing `[c] `.
        let at = ls + marker_len - 3;
        self.checkpoint(Coalesce::None);
        self.text.remove(at..at + 1);
        self.text.insert(at, if checked { " " } else { "x" });
        self.touch();
        true
    }

    /// The content byte range `[ls, le)` of the logical line containing `byte`,
    /// excluding any trailing `'\n'`.
    fn line_content_bounds(&self, byte: usize) -> (usize, usize) {
        self.text.line_content_range(self.text.byte_to_line(byte))
    }

    /// Whether the cursor's line is a list item (bullet or ordered, perhaps
    /// indented), so Tab nests it rather than inserting a soft tab.
    pub fn cursor_on_list_item(&self) -> bool {
        let (ls, le) = self.line_content_bounds(self.cursor);
        is_list(&self.text.byte_slice(ls..le))
    }

    /// Nest the cursor's list item one level deeper, or `outdent` it one level
    /// shallower, by adding or removing one indentation unit of leading spaces.
    /// Indenting is capped one level past the preceding item, so a nested item
    /// always has a parent above it. The cursor rides with the text so the editing
    /// position is preserved. Returns whether the buffer changed.
    pub fn nest_list_item(&mut self, outdent: bool) -> bool {
        let line_idx = self.text.byte_to_line(self.cursor);
        let (ls, le) = self.line_content_bounds(self.cursor);
        let line = self.text.byte_slice(ls..le);
        if !is_list(&line) {
            return false;
        }
        let unit = markdown::LIST_INDENT_SPACES;
        let cur_indent = markdown::leading_indent(&line);
        // The leading-space change applied to the item and its whole subtree.
        let delta: i64 = if outdent {
            let remove = cur_indent.min(unit);
            if remove == 0 {
                return false;
            }
            -(remove as i64)
        } else {
            // Cap one past the preceding item, so an item never nests without a
            // parent above it.
            let cap = self.prev_list_depth(line_idx).map_or(0, |d| d + 1);
            if cur_indent / unit >= cap {
                return false;
            }
            unit as i64
        };
        // The subtree: this line plus the following lines indented deeper than it
        // (its nested descendants), so reparenting carries them along rather than
        // orphaning a child whose parent just moved.
        let starts = self.subtree_line_starts(line_idx, cur_indent);
        self.checkpoint(Coalesce::None);
        // Edit bottom-up so each line start stays valid as lower lines change.
        for &start in starts.iter().rev() {
            if delta > 0 {
                self.text.insert(start, &" ".repeat(delta as usize));
            } else {
                self.text.remove(start..start + (-delta) as usize);
            }
        }
        // Only the cursor's own line (at `ls`) shifts it; deeper lines come after.
        self.cursor = (self.cursor as i64 + delta).max(ls as i64) as usize;
        self.anchor = None;
        self.touch();
        true
    }

    /// The nesting depth of the nearest list item above line `line_idx`, skipping
    /// blank lines; `None` if a non-list line (or the buffer top) comes first.
    fn prev_list_depth(&self, line_idx: usize) -> Option<usize> {
        let mut i = line_idx;
        while i > 0 {
            i -= 1;
            let (ls, le) = self.text.line_content_range(i);
            let line = self.text.byte_slice(ls..le);
            if line.trim().is_empty() {
                continue;
            }
            return is_list(&line)
                .then(|| markdown::leading_indent(&line) / markdown::LIST_INDENT_SPACES);
        }
        None
    }

    /// The line-start byte offsets of a list item's subtree: the line at
    /// `line_idx`, then each following line indented deeper than `cur_indent` (its
    /// descendants), stopping at the first line indented no deeper (a sibling or
    /// ancestor) or the buffer end.
    fn subtree_line_starts(&self, line_idx: usize, cur_indent: usize) -> Vec<usize> {
        let n = self.text.len_lines();
        let mut starts = vec![self.text.line_to_byte(line_idx)];
        let mut i = line_idx + 1;
        while i < n {
            let (ls, le) = self.text.line_content_range(i);
            if markdown::leading_indent(&self.text.byte_slice(ls..le)) <= cur_indent {
                break;
            }
            starts.push(ls);
            i += 1;
        }
        starts
    }

    /// Delete the byte range `[start, end)` as one undo step, leaving the cursor
    /// at `start` and no selection. The bounds must be cluster boundaries (line
    /// and marker edges, which callers pass, always are). A collapsed range is a
    /// no-op returning `false`.
    fn delete_range(&mut self, start: usize, end: usize) -> bool {
        if start >= end {
            return false;
        }
        self.checkpoint(Coalesce::None);
        self.text.remove(start..end);
        self.cursor = start;
        self.anchor = None;
        self.touch();
        true
    }

    /// Delete the selection as its own undo step if there is one, reporting
    /// whether one was consumed. The shared preamble of the delete keys: each
    /// falls through to its per-key delete only when there is no selection.
    fn take_selection(&mut self) -> bool {
        if self.selection().is_some() {
            self.checkpoint(Coalesce::None);
            self.delete_selection();
            self.touch();
            true
        } else {
            false
        }
    }

    /// Delete the selection if there is one, else the grapheme cluster before
    /// the cursor. Returns whether anything changed.
    pub fn backspace(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let Some(prev) = self.prev_grapheme() else {
            return false;
        };
        self.checkpoint(Coalesce::Delete);
        self.text.remove(prev..self.cursor);
        self.cursor = prev;
        self.touch();
        true
    }

    /// Delete the selection if there is one, else the grapheme cluster after the
    /// cursor (forward delete). Returns whether anything changed.
    pub fn delete_forward(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let Some(next) = self.next_grapheme() else {
            return false;
        };
        self.checkpoint(Coalesce::Delete);
        self.text.remove(self.cursor..next);
        self.touch();
        true
    }

    /// Delete the selection if there is one, else from the cursor back to the
    /// previous word boundary (Ctrl+Backspace). Each word delete is its own undo
    /// step. Returns whether anything changed.
    pub fn delete_word_back(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let target = prev_word_rope(&self.text, self.cursor);
        if target == self.cursor {
            return false;
        }
        self.checkpoint(Coalesce::None);
        self.text.remove(target..self.cursor);
        self.cursor = target;
        self.touch();
        true
    }

    /// Delete the selection if there is one, else from the cursor forward to the
    /// next word boundary (Ctrl+Delete). Returns whether anything changed.
    pub fn delete_word_forward(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let target = next_word_rope(&self.text, self.cursor);
        if target == self.cursor {
            return false;
        }
        self.checkpoint(Coalesce::None);
        self.text.remove(self.cursor..target);
        self.touch();
        true
    }

    /// Select the whole buffer (anchor at the start, cursor at the end).
    pub fn select_all(&mut self) {
        self.set_selection(0, self.text.len());
    }

    /// Select the word (or run of separators) at `offset`, for a double-click.
    /// The word model matches the word-motion keys: a word is a run of
    /// alphanumerics and `_`; whitespace and punctuation are separators.
    pub fn select_word_at(&mut self, offset: usize) {
        let (start, end) = word_range_rope(&self.text, offset);
        self.set_selection(start, end);
    }

    /// Select the visible content of the logical line containing `offset`, for a
    /// triple-click. Deliberately *excludes* the trailing newline: the selection
    /// then matches what is on screen (no phantom cell past the last glyph), and
    /// the cursor stays at the line's end, so a following Backspace clears the
    /// line and leaves it empty instead of eating the break and merging into the
    /// line below. Removing the line itself is the explicit Ctrl+Shift+K
    /// ([`Self::delete_line`]). The buffer only ever holds `\n` (line endings are
    /// normalized on load), so one byte is stripped.
    pub fn select_line_at(&mut self, offset: usize) {
        let line = self.text.byte_to_line(offset.min(self.text.len()));
        let start = self.text.line_to_byte(line);
        let mut end = self.text.line_to_byte(line + 1);
        if end > start && self.text.byte(end - 1) == Some(b'\n') {
            end -= 1;
        }
        self.set_selection(start, end);
    }

    /// Delete the whole logical line(s) the cursor or selection touches, newline
    /// included, as one undo step (Ctrl+Shift+K). Triple-click deliberately
    /// selects only a line's content ([`Self::select_line_at`]), so this is the
    /// gesture for removing the line itself. When the span reaches the end of the
    /// buffer with no trailing newline (the final line, or an empty last line),
    /// the preceding newline is consumed instead so no blank line is left behind.
    pub fn delete_line(&mut self) -> bool {
        let (lo, hi) = self.selection().unwrap_or((self.cursor, self.cursor));
        let first = self.text.byte_to_line(lo);
        let last = self.text.byte_to_line(hi);
        let start = self.text.line_to_byte(first);
        let end = self.text.line_to_byte(last + 1);
        // A span ending in a newline already carries its own terminator; one that
        // does not can only be the buffer's end, so eat the preceding newline to
        // avoid leaving an empty line where the deleted one was.
        let has_terminator = end > start && self.text.byte(end - 1) == Some(b'\n');
        let start = if !has_terminator && start > 0 {
            start - 1
        } else {
            start
        };
        self.delete_range(start, end)
    }

    /// Select the half-open byte range `[start, end)`, anchor at `start` and
    /// cursor at `end`, for selecting a whole rendered element on a click (a
    /// horizontal rule). `start`/`end` come from the layout's line bounds, so they
    /// already sit on cluster boundaries.
    pub fn select_range(&mut self, start: usize, end: usize) {
        self.set_selection(start, end);
    }

    /// Set the selection to the half-open range `[start, end)` with the cursor at
    /// `end`. A collapsed range (`start == end`) leaves a bare cursor, no
    /// selection. Breaks the coalescing group, like any cursor move.
    fn set_selection(&mut self, start: usize, end: usize) {
        self.group = Coalesce::None;
        self.anchor = if start == end { None } else { Some(start) };
        self.cursor = end;
    }

    /// Undo the most recent edit group. Returns whether anything changed.
    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo.pop_back() else {
            return false;
        };
        self.redo.push(self.snapshot());
        self.restore(prev);
        self.group = Coalesce::None;
        true
    }

    /// Redo the most recently undone edit group. Returns whether anything changed.
    pub fn redo(&mut self) -> bool {
        let Some(next) = self.redo.pop() else {
            return false;
        };
        self.undo.push_back(self.snapshot());
        self.restore(next);
        self.group = Coalesce::None;
        true
    }

    /// Move the cursor one grapheme cluster left. With `extend`, grow the
    /// selection; without it, a non-empty selection collapses to its left edge.
    /// Returns whether the cursor moved.
    pub fn move_left(&mut self, extend: bool) -> bool {
        self.group = Coalesce::None;
        if !extend {
            if let Some((start, _)) = self.selection() {
                self.cursor = start;
                self.anchor = None;
                return true;
            }
        }
        match self.prev_grapheme() {
            Some(prev) => {
                self.set_cursor(prev, extend);
                true
            }
            None => {
                if !extend {
                    self.anchor = None;
                }
                false
            }
        }
    }

    /// Move the cursor one grapheme cluster right. With `extend`, grow the
    /// selection; without it, a non-empty selection collapses to its right edge.
    /// Returns whether the cursor moved.
    pub fn move_right(&mut self, extend: bool) -> bool {
        self.group = Coalesce::None;
        if !extend {
            if let Some((_, end)) = self.selection() {
                self.cursor = end;
                self.anchor = None;
                return true;
            }
        }
        match self.next_grapheme() {
            Some(next) => {
                self.set_cursor(next, extend);
                true
            }
            None => {
                if !extend {
                    self.anchor = None;
                }
                false
            }
        }
    }

    /// Move the cursor one word left (Ctrl+Left), to the start of the previous
    /// word. With `extend`, grow the selection. Returns whether anything changed.
    pub fn move_word_left(&mut self, extend: bool) -> bool {
        let target = prev_word_rope(&self.text, self.cursor);
        self.set_cursor_changed(target, extend)
    }

    /// Move the cursor one word right (Ctrl+Right), to the end of the next word.
    /// With `extend`, grow the selection. Returns whether anything changed.
    pub fn move_word_right(&mut self, extend: bool) -> bool {
        let target = next_word_rope(&self.text, self.cursor);
        self.set_cursor_changed(target, extend)
    }

    /// Move the cursor to the very start of the buffer (Ctrl+Home). With
    /// `extend`, grow the selection. Returns whether anything changed.
    pub fn move_doc_start(&mut self, extend: bool) -> bool {
        self.set_cursor_changed(0, extend)
    }

    /// Move the cursor to the very end of the buffer (Ctrl+End). With `extend`,
    /// grow the selection. Returns whether anything changed.
    pub fn move_doc_end(&mut self, extend: bool) -> bool {
        self.set_cursor_changed(self.text.len(), extend)
    }

    /// Move the cursor to `target` via [`set_cursor`](Self::set_cursor),
    /// reporting whether the cursor or selection actually changed. The
    /// change-reporting form of `set_cursor`, shared by the word- and
    /// document-motion keys and the pane's window-relative motions.
    pub fn set_cursor_changed(&mut self, target: usize, extend: bool) -> bool {
        let before_cursor = self.cursor;
        let before_sel = self.selection();
        self.set_cursor(target, extend);
        self.cursor != before_cursor || self.selection() != before_sel
    }

    /// Place the cursor at `offset`, clamped into the buffer and snapped down to
    /// the nearest grapheme cluster boundary so it never lands inside a cluster.
    /// With `extend`, anchor a selection at the previous cursor (or keep the
    /// existing one); without it, clear any selection. Used by click and
    /// vertical-motion hit-testing, where the offset comes from glyph geometry.
    pub fn set_cursor(&mut self, offset: usize, extend: bool) {
        self.group = Coalesce::None;
        let offset = self.snap(offset);
        if extend {
            if self.anchor.is_none() {
                self.anchor = Some(self.cursor);
            }
            self.cursor = offset;
            if self.anchor == Some(self.cursor) {
                self.anchor = None;
            }
        } else {
            self.cursor = offset;
            self.anchor = None;
        }
    }

    /// Record the pre-edit state for undo, unless this edit coalesces with the
    /// previous one (same non-`None` group). Clears the redo stack on a real
    /// checkpoint, since new edits invalidate any undone future.
    fn checkpoint(&mut self, group: Coalesce) {
        let merge = group != Coalesce::None && group == self.group;
        if !merge {
            self.undo.push_back(self.snapshot());
            if self.undo.len() > UNDO_CAP {
                self.undo.pop_front();
            }
            self.redo.clear();
        }
        self.group = group;
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            text: self.text.clone(),
            cursor: self.cursor,
            anchor: self.anchor,
            version: self.version,
        }
    }

    fn restore(&mut self, snapshot: Snapshot) {
        self.text = snapshot.text;
        self.cursor = snapshot.cursor;
        self.anchor = snapshot.anchor;
        self.version = snapshot.version;
    }

    /// Delete the active selection, collapsing the cursor to its start. Returns
    /// whether a selection was removed.
    fn delete_selection(&mut self) -> bool {
        let Some((start, end)) = self.selection() else {
            return false;
        };
        self.text.remove(start..end);
        self.cursor = start;
        self.anchor = None;
        true
    }

    /// Snap `offset` into `[0, len]` and back to the nearest grapheme cluster
    /// boundary at or before it. A grapheme cluster never spans a `'\n'`, so the
    /// search only needs the logical line `offset` falls on, materialized from
    /// the rope rather than the whole document.
    fn snap(&self, offset: usize) -> usize {
        let len = self.text.len();
        let offset = offset.min(len);
        if offset == 0 || offset == len {
            return offset;
        }
        let line = self.text.byte_to_line(offset);
        let ls = self.text.line_to_byte(line);
        if offset == ls {
            return offset; // a line start is always a boundary
        }
        let le = self.text.line_to_byte(line + 1);
        let s = self.text.byte_slice(ls..le);
        ls + grapheme::prev_boundary(&s, offset - ls + 1)
    }

    /// Byte offset of the grapheme cluster boundary before the cursor, if any.
    /// The preceding cluster lies on the line containing `cursor - 1` (a `'\n'`
    /// is its own single-byte cluster), so only that line is materialized.
    fn prev_grapheme(&self) -> Option<usize> {
        if self.cursor == 0 {
            return None;
        }
        let line = self.text.byte_to_line(self.cursor - 1);
        let ls = self.text.line_to_byte(line);
        let le = self.text.line_to_byte(line + 1);
        let s = self.text.byte_slice(ls..le);
        Some(ls + grapheme::prev_boundary(&s, self.cursor - ls))
    }

    /// Byte offset of the grapheme cluster boundary after the cursor, if any.
    /// The cluster at the cursor lies entirely on its line, so only that line is
    /// materialized.
    fn next_grapheme(&self) -> Option<usize> {
        if self.cursor == self.text.len() {
            return None;
        }
        let line = self.text.byte_to_line(self.cursor);
        let ls = self.text.line_to_byte(line);
        let le = self.text.line_to_byte(line + 1);
        let s = self.text.byte_slice(ls..le);
        Some(ls + grapheme::next_boundary(&s, self.cursor - ls))
    }
}

/// Whether `line` is a list item (bullet, ordered, or task, perhaps indented),
/// the lines Tab nests and `enter` continues.
/// The leading whitespace of `line` (spaces and tabs), capped at `col` bytes,
/// for inheriting a line's indent onto the next line on Enter. Capping at the
/// cursor keeps a break inside the indent from over-indenting, and a break at
/// column 0 from indenting at all.
fn inherited_indent(line: &str, col: usize) -> &str {
    let ws = line
        .bytes()
        .take_while(|&b| b == b' ' || b == b'\t')
        .count();
    &line[..ws.min(col)]
}

fn is_list(line: &str) -> bool {
    matches!(
        markdown::style_of(line),
        markdown::LineStyle::Bullet
            | markdown::LineStyle::Ordered
            | markdown::LineStyle::Task { .. }
    )
}

/// Whether inserting `s` should coalesce with adjacent typing: a single
/// non-whitespace character does, so a typed word is one undo step while spaces,
/// newlines, and multi-character pastes start fresh steps.
fn coalescable_insert(s: &str) -> bool {
    let mut chars = s.chars();
    matches!((chars.next(), chars.next()), (Some(c), None) if !c.is_whitespace())
}

/// Whether a grapheme `cluster` is part of a word, classified by its first
/// scalar: letters, digits, and `_` are word characters; whitespace and
/// punctuation alike are separators. This is the simple, predictable heuristic
/// editors use for Ctrl+Arrow and Ctrl+Backspace, not full UAX #29 word
/// segmentation.
fn cluster_is_word(cluster: &str) -> bool {
    cluster.chars().next().is_some_and(grapheme::is_word_char)
}

/// The grapheme clusters of the logical line `line` as `(absolute byte start,
/// is_word)` pairs. Materializes only that one line of the rope (clusters never
/// span a `'\n'`, so a line is a self-contained unit), so the word-motion
/// scanners below read a bounded window outward from the cursor instead of the
/// whole document.
fn line_clusters(rope: &Rope, line: usize) -> Vec<(usize, bool)> {
    let ls = rope.line_to_byte(line);
    let le = rope.line_to_byte(line + 1);
    let s = rope.byte_slice(ls..le);
    grapheme::graphemes(&s)
        .map(|(off, c)| (ls + off, cluster_is_word(c)))
        .collect()
}

/// The byte offset one word forward from `pos`: skip any separators, then the
/// run of word clusters, landing at the end of that word (or the buffer end).
/// Scans the rope a logical line at a time from the cursor outward (words and
/// separator runs may cross blank lines), never materializing the whole document.
fn next_word_rope(rope: &Rope, pos: usize) -> usize {
    let len = rope.len();
    if pos >= len {
        return len;
    }
    let n_lines = rope.len_lines();
    let mut line = rope.byte_to_line(pos);
    // Two phases carried across lines: skip leading separators, then consume the
    // word run, stopping at the first separator after it.
    let mut in_word = false;
    while line < n_lines {
        for (abs, is_word) in line_clusters(rope, line) {
            if abs < pos {
                continue;
            }
            if in_word && !is_word {
                return abs;
            }
            in_word |= is_word;
        }
        line += 1;
    }
    len
}

/// The byte offset one word back from `pos`: skip any trailing separators, then
/// the run of word clusters, landing at the start of that word (or the buffer
/// start). Scans the rope backward a logical line at a time from the cursor.
fn prev_word_rope(rope: &Rope, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut line = rope.byte_to_line(pos - 1);
    let mut in_word = false;
    let mut start = 0;
    loop {
        for &(abs, is_word) in line_clusters(rope, line).iter().rev() {
            if abs >= pos {
                continue;
            }
            if in_word && !is_word {
                return start;
            }
            if is_word {
                in_word = true;
                start = abs;
            }
        }
        if line == 0 {
            return if in_word { start } else { 0 };
        }
        line -= 1;
    }
}

/// The byte range of the word (or run of separators) at `offset`, for
/// double-click selection: the maximal run of clusters sharing the clicked
/// cluster's class (word vs separator). A click at a word's trailing edge, or at
/// the buffer end, still selects the word to the left. Scans the rope outward
/// from the click instead of materializing the whole document.
fn word_range_rope(rope: &Rope, offset: usize) -> (usize, usize) {
    let len = rope.len();
    if len == 0 {
        return (0, 0);
    }
    let offset = offset.min(len);
    let (c_start, c_is_word, prev_is_word) = clicked_cluster(rope, offset);
    // A separator under the click with a word immediately to its left selects
    // that word (so a click on a word's trailing edge or the space after it still
    // grabs the word): expand the word run that ends at the click.
    if !c_is_word && prev_is_word == Some(true) {
        return (class_run_start(rope, c_start, true), c_start);
    }
    let start = class_run_start(rope, c_start, c_is_word);
    let end = class_run_end(rope, c_start, c_is_word);
    (start, end)
}

/// The cluster a double-click at byte `offset` lands on: its start offset and
/// class, plus the class of the cluster immediately to its left (for the
/// word-to-the-left preference). A click at the buffer end falls on the last
/// cluster; for a click at a line start the left neighbor is the previous line's
/// terminating `'\n'`, a separator. Materializes only the click's logical line.
fn clicked_cluster(rope: &Rope, offset: usize) -> (usize, bool, Option<bool>) {
    let len = rope.len();
    let probe = if offset >= len {
        len.saturating_sub(1)
    } else {
        offset
    };
    let line = rope.byte_to_line(probe);
    let clusters = line_clusters(rope, line);
    let idx = if offset >= len {
        clusters.len().saturating_sub(1)
    } else {
        clusters
            .iter()
            .position(|&(abs, _)| abs == offset)
            .unwrap_or(clusters.len().saturating_sub(1))
    };
    let (c_start, c_is_word) = clusters
        .get(idx)
        .copied()
        .unwrap_or((rope.line_to_byte(line), false));
    let prev_is_word = if idx > 0 {
        Some(clusters[idx - 1].1)
    } else if line > 0 {
        // The previous line's last cluster is its terminating '\n', a separator.
        Some(false)
    } else {
        None
    };
    (c_start, c_is_word, prev_is_word)
}

/// The start byte offset of the maximal run of class `word` that contains the
/// cluster beginning at `from`, scanning the rope backward line by line.
fn class_run_start(rope: &Rope, from: usize, word: bool) -> usize {
    if from == 0 {
        return 0;
    }
    let mut line = rope.byte_to_line(from - 1);
    let mut start = from;
    loop {
        for &(abs, is_word) in line_clusters(rope, line).iter().rev() {
            if abs >= from {
                continue;
            }
            if is_word != word {
                return start;
            }
            start = abs;
        }
        if line == 0 {
            return start;
        }
        line -= 1;
    }
}

/// The end byte offset (exclusive) of the maximal run of class `word` that
/// contains the cluster beginning at `from`, scanning the rope forward line by
/// line.
fn class_run_end(rope: &Rope, from: usize, word: bool) -> usize {
    let len = rope.len();
    let n_lines = rope.len_lines();
    let mut line = rope.byte_to_line(from.min(len.saturating_sub(1)));
    while line < n_lines {
        for (abs, is_word) in line_clusters(rope, line) {
            if abs < from {
                continue;
            }
            if is_word != word {
                return abs;
            }
        }
        line += 1;
    }
    len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_advances_cursor() {
        let mut b = TextBuffer::new();
        b.insert_str("hi");
        assert_eq!(b.text(), "hi");
        assert_eq!(b.cursor, 2);
    }

    #[test]
    fn insert_in_the_middle() {
        let mut b = TextBuffer::new();
        b.insert_str("ac");
        b.move_left(false); // between a and c
        b.insert_str("b");
        assert_eq!(b.text(), "abc");
        assert_eq!(b.cursor, 2);
    }

    #[test]
    fn newlines_are_just_bytes() {
        let mut b = TextBuffer::new();
        b.insert_str("a");
        b.insert_str("\n");
        b.insert_str("b");
        assert_eq!(b.text(), "a\nb");
        assert_eq!(b.cursor, 3);
    }

    // --- Enter / smart list continuation ----------------------------------

    #[test]
    fn enter_continues_a_bullet_with_the_same_marker() {
        let mut b = TextBuffer::new();
        b.insert_str("- foo");
        assert!(b.enter());
        assert_eq!(b.text(), "- foo\n- ");
        assert_eq!(b.cursor(), 8);
    }

    #[test]
    fn enter_on_an_empty_bullet_removes_the_marker() {
        let mut b = TextBuffer::new();
        b.insert_str("- ");
        assert!(b.enter());
        assert_eq!(b.text(), "", "an empty item clears its marker, no new line");
        assert_eq!(b.cursor(), 0);
    }

    #[test]
    fn enter_advances_an_ordered_item() {
        let mut b = TextBuffer::new();
        b.insert_str("1. a");
        b.enter();
        assert_eq!(b.text(), "1. a\n2. ");
        assert_eq!(b.cursor(), 8);
    }

    #[test]
    fn enter_on_an_empty_ordered_item_removes_the_marker() {
        let mut b = TextBuffer::new();
        b.insert_str("3) ");
        b.enter();
        assert_eq!(b.text(), "");
        assert_eq!(b.cursor(), 0);
    }

    #[test]
    fn enter_continues_a_blockquote() {
        let mut b = TextBuffer::new();
        b.insert_str("> q");
        b.enter();
        assert_eq!(b.text(), "> q\n> ");
        assert_eq!(b.cursor(), 6);
    }

    #[test]
    fn enter_in_the_middle_of_an_item_splits_and_prefixes() {
        let mut b = TextBuffer::new();
        b.insert_str("- ab");
        b.move_left(false); // between 'a' and 'b'
        b.enter();
        assert_eq!(b.text(), "- a\n- b");
        assert_eq!(b.cursor(), 6);
    }

    #[test]
    fn enter_before_the_marker_inserts_a_plain_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("- foo");
        b.set_cursor(0, false);
        b.enter();
        assert_eq!(b.text(), "\n- foo", "no second marker is added");
        assert_eq!(b.cursor(), 1);
    }

    #[test]
    fn enter_off_a_list_inserts_a_plain_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("plain");
        b.enter();
        assert_eq!(b.text(), "plain\n");
        assert_eq!(b.cursor(), 6);
    }

    #[test]
    fn enter_does_not_continue_a_heading() {
        let mut b = TextBuffer::new();
        b.insert_str("# h");
        b.enter();
        assert_eq!(b.text(), "# h\n");
        assert_eq!(b.cursor(), 4);
    }

    #[test]
    fn enter_inherits_the_line_indent() {
        let mut b = TextBuffer::new();
        b.insert_str("    foo"); // four-space indent, cursor at end
        b.enter();
        b.insert_str("bar");
        assert_eq!(b.text(), "    foo\n    bar");
    }

    #[test]
    fn enter_at_column_zero_adds_no_indent() {
        let mut b = TextBuffer::new();
        b.insert_str("    foo");
        b.set_selection(0, 0); // cursor before the indent
        b.enter();
        assert_eq!(b.text(), "\n    foo");
    }

    #[test]
    fn enter_within_the_indent_caps_at_the_cursor() {
        let mut b = TextBuffer::new();
        b.insert_str("    foo");
        b.set_selection(2, 2); // between the second and third indent space
        b.enter();
        // Two spaces stay on the old line; the new line inherits only two.
        assert_eq!(b.text(), "  \n    foo");
    }

    #[test]
    fn enter_replaces_a_selection_with_a_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("- foo");
        b.select_all();
        b.enter();
        assert_eq!(b.text(), "\n");
        assert_eq!(b.cursor(), 1);
    }

    #[test]
    fn enter_is_one_undo_step() {
        let mut b = TextBuffer::new();
        b.insert_str("- foo");
        b.enter();
        assert_eq!(b.text(), "- foo\n- ");
        assert!(b.undo());
        assert_eq!(b.text(), "- foo", "one undo reverts the whole continuation");
    }

    #[test]
    fn enter_continues_a_task_with_a_fresh_unchecked_box() {
        let mut b = TextBuffer::new();
        b.insert_str("- [ ] foo");
        assert!(b.enter());
        assert_eq!(b.text(), "- [ ] foo\n- [ ] ");
        assert_eq!(b.cursor(), 16);
        // A finished item's successor still starts unchecked.
        let mut b = TextBuffer::new();
        b.insert_str("- [x] done");
        b.enter();
        assert_eq!(b.text(), "- [x] done\n- [ ] ");
    }

    #[test]
    fn enter_on_an_empty_task_removes_the_marker() {
        let mut b = TextBuffer::new();
        b.insert_str("- [ ] ");
        assert!(b.enter());
        assert_eq!(b.text(), "", "an empty task clears its marker");
        assert_eq!(b.cursor(), 0);
    }

    #[test]
    fn tab_nests_a_task_item() {
        let mut b = TextBuffer::new();
        b.insert_str("- [ ] a\n- [x] b");
        assert!(b.cursor_on_list_item(), "a task line is a list item");
        assert!(b.nest_list_item(false));
        assert_eq!(b.text(), "- [ ] a\n  - [x] b");
    }

    // --- Task toggling ------------------------------------------------------

    #[test]
    fn toggle_task_flips_the_box_both_ways_without_moving_the_cursor() {
        let mut b = TextBuffer::from_text("- [ ] a\nplain".to_string());
        b.set_cursor(13, false); // cursor far away, on "plain"
        assert!(b.toggle_task(0));
        assert_eq!(b.text(), "- [x] a\nplain");
        assert_eq!(b.cursor(), 13, "the caret never moves");
        assert!(b.toggle_task(0));
        assert_eq!(b.text(), "- [ ] a\nplain");
        // An uppercase X unchecks to a space too.
        let mut b = TextBuffer::from_text("- [X] a".to_string());
        assert!(b.toggle_task(0));
        assert_eq!(b.text(), "- [ ] a");
    }

    #[test]
    fn toggle_task_works_anywhere_on_the_line_and_on_nested_items() {
        let mut b = TextBuffer::from_text("- a\n  - [ ] sub".to_string());
        // Any offset within the task's logical line finds its box.
        assert!(b.toggle_task(10));
        assert_eq!(b.text(), "- a\n  - [x] sub");
    }

    #[test]
    fn toggle_task_is_one_undo_step_and_ignores_non_tasks() {
        let mut b = TextBuffer::from_text("- [ ] a".to_string());
        assert!(b.toggle_task(0));
        assert!(b.undo());
        assert_eq!(b.text(), "- [ ] a", "one undo reverts the toggle");
        for text in ["plain", "- bullet", "1. ordered", "# h"] {
            let mut b = TextBuffer::from_text(text.to_string());
            assert!(!b.toggle_task(0), "{text:?} has no box to toggle");
            assert_eq!(b.text(), text);
        }
    }

    #[test]
    fn toggle_task_keeps_a_selection_intact() {
        let mut b = TextBuffer::from_text("- [ ] a\nplain".to_string());
        b.set_cursor(8, false);
        b.set_cursor(13, true); // select "plain"
        assert!(b.toggle_task(0));
        assert_eq!(b.selection(), Some((8, 13)), "the selection survives");
    }

    // --- Ordered renumbering on Enter ---------------------------------------

    #[test]
    fn enter_renumbers_the_following_items() {
        let mut b = TextBuffer::from_text("1. a\n2. b\n3. c".to_string());
        b.set_cursor(4, false); // end of "1. a"
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \n3. b\n4. c");
        assert_eq!(b.cursor(), 8, "cursor after the new item's marker");
    }

    #[test]
    fn enter_renumbering_is_one_undo_step() {
        let mut b = TextBuffer::from_text("1. a\n2. b".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \n3. b");
        assert!(b.undo());
        assert_eq!(
            b.text(),
            "1. a\n2. b",
            "insertion and renumbering revert together"
        );
        assert!(b.redo());
        assert_eq!(b.text(), "1. a\n2. \n3. b", "and redo replays both");
    }

    #[test]
    fn splitting_an_item_renumbers_the_rest() {
        let mut b = TextBuffer::from_text("1. ax\n2. b".to_string());
        b.set_cursor(4, false); // between "a" and "x"
        b.enter();
        assert_eq!(b.text(), "1. a\n2. x\n3. b");
    }

    #[test]
    fn renumbering_skips_children_and_blank_lines_and_carries_digits() {
        // A deeper child list keeps its own numbers; a blank line does not end
        // the list; "9" carries into "10".
        let mut b = TextBuffer::from_text("1. a\n  1. child\n\n9. b".to_string());
        b.set_cursor(4, false); // end of "1. a"
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \n  1. child\n\n3. b");
        let mut b = TextBuffer::from_text("9. a\n10. b".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "9. a\n10. \n11. b");
    }

    #[test]
    fn renumbering_stops_at_the_end_of_the_list() {
        // A same-level bullet, a shallower item, or plain text all end the run.
        let mut b = TextBuffer::from_text("1. a\n- x\n2. keep".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \n- x\n2. keep");
        let mut b = TextBuffer::from_text("1. a\nplain\n2. keep".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \nplain\n2. keep");
    }

    #[test]
    fn renumbering_keeps_each_items_own_delimiter() {
        let mut b = TextBuffer::from_text("1) a\n2) b".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1) a\n2) \n3) b", "the paren style carries");
    }

    #[test]
    fn renumbering_leaves_already_correct_numbers_alone() {
        // Nothing follows, or the following numbers already fit: no extra edits
        // (observable as: undo still reverts cleanly to the input).
        let mut b = TextBuffer::from_text("1. a".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1. a\n2. ");
        let mut b = TextBuffer::from_text("1. a\n3. b".to_string());
        b.set_cursor(4, false);
        b.enter();
        assert_eq!(b.text(), "1. a\n2. \n3. b", "a lucky gap needs no edit");
    }

    #[test]
    fn nested_ordered_items_renumber_within_their_level() {
        let mut b = TextBuffer::from_text("1. a\n  1. x\n  2. y\n2. b".to_string());
        b.set_cursor(11, false); // end of "  1. x"
        b.enter();
        assert_eq!(
            b.text(),
            "1. a\n  1. x\n  2. \n  3. y\n2. b",
            "the child level renumbers; the parent level is untouched"
        );
    }

    #[test]
    fn backspace_removes_one_cluster() {
        let mut b = TextBuffer::new();
        b.insert_str("ab");
        assert!(b.backspace());
        assert_eq!(b.text(), "a");
        assert_eq!(b.cursor, 1);
    }

    #[test]
    fn backspace_at_start_is_noop() {
        let mut b = TextBuffer::new();
        b.insert_str("x");
        b.move_left(false);
        assert!(!b.backspace());
        assert_eq!(b.text(), "x");
        assert_eq!(b.cursor, 0);
    }

    #[test]
    fn backspace_removes_a_whole_combining_cluster() {
        // 'e' + combining acute is one grapheme: Backspace removes both bytes.
        let mut b = TextBuffer::new();
        b.insert_str("xe\u{0301}");
        assert_eq!(b.text().len(), 4);
        assert!(b.backspace());
        assert_eq!(b.text(), "x");
        assert_eq!(b.cursor, 1);
    }

    #[test]
    fn backspace_removes_a_whole_zwj_emoji() {
        // A ZWJ family emoji is many scalars but one cluster.
        let mut b = TextBuffer::new();
        b.insert_str("a👨\u{200D}👩\u{200D}👧");
        assert!(b.backspace());
        assert_eq!(b.text(), "a");
        assert_eq!(b.cursor, 1);
    }

    #[test]
    fn forward_delete_removes_the_next_cluster() {
        let mut b = TextBuffer::new();
        b.insert_str("a😀b");
        b.set_cursor(1, false); // between 'a' and the emoji
        assert!(b.delete_forward());
        assert_eq!(b.text(), "ab");
        assert_eq!(b.cursor, 1);
    }

    #[test]
    fn cursor_motion_steps_whole_clusters() {
        let mut b = TextBuffer::new();
        b.insert_str("é😀z"); // 2 + 4 + 1 bytes
        assert_eq!(b.cursor, 7);
        assert!(b.move_left(false)); // before z
        assert_eq!(b.cursor, 6);
        assert!(b.move_left(false)); // before emoji
        assert_eq!(b.cursor, 2);
        assert!(b.move_left(false)); // before é
        assert_eq!(b.cursor, 0);
        assert!(!b.move_left(false));
        assert!(b.move_right(false));
        assert_eq!(b.cursor, 2);
    }

    #[test]
    fn shift_motion_builds_and_clears_a_selection() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        assert_eq!(b.cursor, 5);
        assert!(b.move_left(true)); // select "o"
        assert!(b.move_left(true)); // select "lo"
        assert_eq!(b.selection(), Some((3, 5)));
        // Extending back onto the anchor collapses the selection.
        assert!(b.move_right(true));
        assert!(b.move_right(true));
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor, 5);
    }

    #[test]
    fn plain_arrow_collapses_a_selection_to_an_edge() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        b.set_cursor(1, false);
        b.set_cursor(4, true); // select "ell"
        assert_eq!(b.selection(), Some((1, 4)));
        b.move_left(false); // collapse to the left edge
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor, 1);
    }

    #[test]
    fn typing_replaces_the_selection() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        b.set_cursor(1, false);
        b.set_cursor(4, true); // select "ell"
        b.insert_str("i");
        assert_eq!(b.text(), "hio");
        assert_eq!(b.cursor, 2);
        assert_eq!(b.selection(), None);
    }

    #[test]
    fn backspace_deletes_the_selection_only() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        b.set_cursor(1, false);
        b.set_cursor(4, true); // select "ell"
        assert!(b.backspace());
        assert_eq!(b.text(), "ho");
        assert_eq!(b.cursor, 1);
        assert_eq!(b.selection(), None);
    }

    #[test]
    fn select_all_spans_the_buffer_and_reports_selected_text() {
        let mut b = TextBuffer::new();
        assert_eq!(b.selected_text(), None);
        b.insert_str("hello");
        b.select_all();
        assert_eq!(b.selection(), Some((0, 5)));
        assert_eq!(b.selected_text().as_deref(), Some("hello"));
        // Select-all on an empty buffer selects nothing.
        let mut empty = TextBuffer::new();
        empty.select_all();
        assert_eq!(empty.selection(), None);
    }

    #[test]
    fn undo_redo_round_trips_a_coalesced_word() {
        let mut b = TextBuffer::new();
        b.insert_str("a");
        b.insert_str("b");
        b.insert_str("c"); // one coalesced "abc" group
        assert_eq!(b.text(), "abc");
        assert!(b.undo());
        assert_eq!(b.text(), "", "the whole word undoes in one step");
        assert_eq!(b.cursor, 0);
        assert!(b.redo());
        assert_eq!(b.text(), "abc");
        assert_eq!(b.cursor, 3);
        assert!(!b.redo(), "nothing left to redo");
    }

    #[test]
    fn a_cursor_move_breaks_the_undo_group() {
        let mut b = TextBuffer::new();
        b.insert_str("ab");
        b.move_left(false); // breaks coalescing
        b.insert_str("x"); // inserted between a and b -> "axb"
        assert_eq!(b.text(), "axb");
        assert!(b.undo());
        assert_eq!(b.text(), "ab", "only the post-move insert undoes");
        assert!(b.undo());
        assert_eq!(b.text(), "");
    }

    #[test]
    fn a_new_edit_clears_the_redo_stack() {
        let mut b = TextBuffer::new();
        b.insert_str("a");
        b.insert_str(" "); // whitespace breaks the group
        assert!(b.undo()); // back to "a"
        assert_eq!(b.text(), "a");
        b.insert_str("b"); // a fresh edit; redo of " " is gone
        assert_eq!(b.text(), "ab");
        assert!(!b.redo(), "redo was invalidated by the new edit");
    }

    #[test]
    fn undo_restores_a_deleted_selection() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        b.set_cursor(1, false);
        b.set_cursor(4, true); // select "ell"
        assert!(b.backspace());
        assert_eq!(b.text(), "ho");
        assert!(b.undo());
        assert_eq!(b.text(), "hello", "the selection's text comes back");
        assert_eq!(b.selection(), Some((1, 4)), "and so does the selection");
    }

    #[test]
    fn consecutive_backspaces_undo_together() {
        let mut b = TextBuffer::new();
        b.insert_str("word");
        b.set_cursor(4, false); // ensure a clean group boundary before deleting
        b.backspace();
        b.backspace(); // "wo", one coalesced delete group
        assert_eq!(b.text(), "wo");
        assert!(b.undo());
        assert_eq!(b.text(), "word", "the run of deletes restores at once");
    }

    #[test]
    fn set_cursor_snaps_into_cluster_boundaries_and_clamps() {
        let mut b = TextBuffer::new();
        b.insert_str("é😀z"); // boundaries at 0, 2, 6, 7
        b.set_cursor(0, false);
        assert_eq!(b.cursor, 0);
        b.set_cursor(6, false);
        assert_eq!(b.cursor, 6);
        // Offsets 3, 4, 5 fall inside the emoji and snap back to 2.
        for inside in [3, 4, 5] {
            b.set_cursor(inside, false);
            assert_eq!(b.cursor, 2, "offset {inside} snaps to the emoji start");
        }
        b.set_cursor(999, false);
        assert_eq!(b.cursor, 7, "past the end clamps to the buffer length");
    }

    #[test]
    fn from_text_loads_clean_with_no_history() {
        let mut b = TextBuffer::from_text("hello\nworld".to_string());
        assert_eq!(b.text(), "hello\nworld");
        assert_eq!(b.cursor, 0, "the cursor opens at the top of the file");
        assert!(!b.is_modified(), "a freshly loaded file is not modified");
        assert!(!b.undo(), "there is no history to undo into");
    }

    #[test]
    fn editing_marks_modified_and_saving_clears_it() {
        let mut b = TextBuffer::new();
        assert!(!b.is_modified(), "an untouched buffer is clean");
        b.insert_str("hi");
        assert!(b.is_modified());
        b.mark_saved();
        assert!(!b.is_modified(), "saving clears the modified flag");
        b.backspace();
        assert!(b.is_modified(), "a later edit dirties it again");
    }

    #[test]
    fn undoing_back_to_the_saved_state_is_not_modified() {
        let mut b = TextBuffer::new();
        b.insert_str("a");
        b.mark_saved(); // clean at "a"
        b.insert_str(" "); // whitespace breaks coalescing into its own step
        b.insert_str("b");
        assert_eq!(b.text(), "a b");
        assert!(b.is_modified());
        b.undo(); // remove "b"
        assert!(b.is_modified(), "still past the saved state");
        b.undo(); // remove " ", back to the saved "a"
        assert_eq!(b.text(), "a");
        assert!(
            !b.is_modified(),
            "undoing onto the saved contents reads clean"
        );
        // Redoing forward off the saved state dirties it once more.
        b.redo();
        assert!(b.is_modified());
    }

    #[test]
    fn word_motion_steps_whole_words() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar baz"); // cursor at the end (11)
        assert!(b.move_word_left(false));
        assert_eq!(b.cursor, 8, "start of baz");
        assert!(b.move_word_left(false));
        assert_eq!(b.cursor, 4, "start of bar");
        assert!(b.move_word_left(false));
        assert_eq!(b.cursor, 0, "start of foo");
        assert!(!b.move_word_left(false), "no move at the start");
        assert!(b.move_word_right(false));
        assert_eq!(b.cursor, 3, "end of foo");
        assert!(b.move_word_right(false));
        assert_eq!(b.cursor, 7, "end of bar");
        assert!(b.move_word_right(false));
        assert_eq!(b.cursor, 11, "end of baz");
        assert!(!b.move_word_right(false), "no move at the end");
    }

    #[test]
    fn word_motion_crosses_blank_lines_to_the_next_word() {
        // The bounded line-by-line scanners must still cross a '\n' (and a blank
        // line) when the separator run between two words spans them. "ab\n\ncd":
        // ab = 0..2, '\n' = 2, '\n' = 3, cd = 4..6.
        let mut b = TextBuffer::from_text("ab\n\ncd".to_string());
        b.set_cursor(2, false); // end of "ab"
        b.move_word_right(false);
        assert_eq!(b.cursor, 6, "skips both newlines to the end of cd");
        b.move_word_left(false);
        assert_eq!(b.cursor, 4, "back across the blank line to the start of cd");
        b.move_word_left(false);
        assert_eq!(b.cursor, 0, "back to the start of ab");
    }

    #[test]
    fn double_click_separator_run_spans_a_newline() {
        // A separator run a double-click selects can cross a line break. In
        // "a \n b" the space, '\n', and space form one run (bytes 1..4). Clicking
        // the '\n' (offset 2) selects the whole run; clicking the space right
        // after "a" would instead prefer the word "a" to its left.
        let mut b = TextBuffer::from_text("a \n b".to_string());
        b.select_word_at(2);
        assert_eq!(
            b.selection(),
            Some((1, 4)),
            "space, newline, space as one run"
        );
    }

    #[test]
    fn punctuation_is_a_word_separator() {
        let mut b = TextBuffer::new();
        b.insert_str("foo.bar");
        b.set_cursor(0, false);
        b.move_word_right(false);
        assert_eq!(b.cursor, 3, "stops at the dot, after foo");
        b.move_word_right(false);
        assert_eq!(b.cursor, 7, "skips the dot, lands after bar");
    }

    #[test]
    fn word_motion_lands_on_cluster_boundaries() {
        // "café x" with é as 'e' + combining acute (U+0301): the word run "café"
        // ends at byte 6 (the space), never inside the two-scalar é cluster.
        let mut b = TextBuffer::new();
        b.insert_str("cafe\u{0301} x");
        b.set_cursor(0, false);
        b.move_word_right(false);
        assert_eq!(b.cursor, 6, "lands at the space, not inside the é cluster");
    }

    #[test]
    fn shift_word_motion_builds_a_selection() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar");
        b.set_cursor(0, false);
        assert!(b.move_word_right(true));
        assert_eq!(b.selection(), Some((0, 3)), "Ctrl+Shift+Right selects foo");
    }

    #[test]
    fn delete_word_back_removes_the_previous_word() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar"); // cursor at the end
        assert!(b.delete_word_back());
        assert_eq!(b.text(), "foo ");
        assert_eq!(b.cursor, 4);
        // And it undoes in one step.
        assert!(b.undo());
        assert_eq!(b.text(), "foo bar");
    }

    #[test]
    fn delete_word_forward_removes_the_next_word() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar");
        b.set_cursor(0, false);
        assert!(b.delete_word_forward());
        assert_eq!(b.text(), " bar");
        assert_eq!(b.cursor, 0);
    }

    #[test]
    fn word_delete_takes_the_selection_when_there_is_one() {
        let mut b = TextBuffer::new();
        b.insert_str("hello world");
        b.set_cursor(0, false);
        b.set_cursor(5, true); // select "hello"
        assert!(b.delete_word_back());
        assert_eq!(b.text(), " world", "the selection goes, not a word");
    }

    #[test]
    fn word_delete_at_an_edge_is_a_noop() {
        let mut b = TextBuffer::new();
        b.insert_str("word");
        b.set_cursor(0, false);
        assert!(!b.delete_word_back(), "nothing before the start");
        b.set_cursor(b.text().len(), false);
        assert!(!b.delete_word_forward(), "nothing after the end");
        assert_eq!(b.text(), "word");
    }

    #[test]
    fn document_motion_jumps_to_the_ends() {
        let mut b = TextBuffer::new();
        b.insert_str("line one\nline two"); // cursor at the end
        assert!(b.move_doc_start(false));
        assert_eq!(b.cursor, 0);
        assert!(!b.move_doc_start(false), "already at the start");
        assert!(b.move_doc_end(false));
        assert_eq!(b.cursor, b.text().len());
        // Extending from the top selects the whole buffer.
        b.set_cursor(0, false);
        assert!(b.move_doc_end(true));
        assert_eq!(b.selection(), Some((0, b.text().len())));
    }

    #[test]
    fn double_click_selects_the_word_under_it() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar baz");
        b.select_word_at(5); // inside "bar"
        assert_eq!(b.selection(), Some((4, 7)));
        assert_eq!(b.selected_text().as_deref(), Some("bar"));
    }

    #[test]
    fn double_click_on_whitespace_selects_the_run() {
        let mut b = TextBuffer::new();
        b.insert_str("a   b");
        b.select_word_at(2); // inside the spaces
        assert_eq!(b.selected_text().as_deref(), Some("   "));
    }

    #[test]
    fn double_click_at_a_word_edge_prefers_the_word() {
        let mut b = TextBuffer::new();
        b.insert_str("foo bar");
        b.select_word_at(3); // the space starts here; the word is to the left
        assert_eq!(b.selected_text().as_deref(), Some("foo"));
    }

    #[test]
    fn double_click_at_the_buffer_end_selects_the_last_word() {
        let mut b = TextBuffer::new();
        b.insert_str("hello");
        b.select_word_at(5);
        assert_eq!(b.selected_text().as_deref(), Some("hello"));
    }

    #[test]
    fn triple_click_selects_the_line_content_without_its_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("first\nsecond\nthird");
        b.select_line_at(8); // inside "second"
        assert_eq!(b.selected_text().as_deref(), Some("second"));
        // The cursor stays on the line (at its end), not on the line below, so a
        // following Backspace clears the line rather than merging it upward.
        assert_eq!(b.cursor(), "first\nsecond".len());
    }

    #[test]
    fn triple_click_on_the_last_line_has_no_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("first\nlast");
        b.select_line_at(8); // inside "last"
        assert_eq!(b.selected_text().as_deref(), Some("last"));
    }

    #[test]
    fn delete_line_removes_a_middle_line_and_its_newline() {
        let mut b = TextBuffer::new();
        b.insert_str("first\nsecond\nthird");
        b.set_selection(8, 8); // cursor inside "second", no selection
        assert!(b.delete_line());
        assert_eq!(b.text(), "first\nthird");
        assert_eq!(b.cursor(), "first\n".len());
    }

    #[test]
    fn delete_line_on_the_last_line_leaves_no_blank_line() {
        let mut b = TextBuffer::new();
        b.insert_str("first\nlast");
        b.set_selection(8, 8); // cursor inside "last"
        assert!(b.delete_line());
        assert_eq!(b.text(), "first");
        assert_eq!(b.cursor(), "first".len());
    }

    #[test]
    fn delete_line_spans_every_line_the_selection_touches() {
        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo\nthree\nfour");
        b.set_selection(2, 9); // inside "one" through inside "three"
        assert!(b.delete_line());
        assert_eq!(b.text(), "four");
    }

    #[test]
    fn tab_nests_a_list_item_under_its_parent_and_caps_without_one() {
        let mut b = TextBuffer::new();
        b.insert_str("- a\n- b"); // cursor ends on the second item
        assert!(b.cursor_on_list_item());
        // Tab nests "- b" under "- a".
        assert!(b.nest_list_item(false));
        assert_eq!(b.text(), "- a\n  - b");
        // Tab again is capped: a depth-2 child has no depth-1 parent.
        assert!(!b.nest_list_item(false));
        assert_eq!(b.text(), "- a\n  - b");
        // Shift+Tab outdents it back to the top level.
        assert!(b.nest_list_item(true));
        assert_eq!(b.text(), "- a\n- b");
        // Shift+Tab at depth 0 has nothing to remove.
        assert!(!b.nest_list_item(true));
    }

    #[test]
    fn nesting_cascades_to_descendants() {
        let mut b = TextBuffer::new();
        b.insert_str("- a\n  - b\n    - c"); // a > b > c, each one deeper
        b.set_cursor(8, false); // on the middle item "  - b"
                                // Outdenting b carries its child c along, so c stays nested under b.
        b.nest_list_item(true);
        assert_eq!(b.text(), "- a\n- b\n  - c");
        // Indenting b again carries c back down: the subtree moves as a unit.
        b.nest_list_item(false);
        assert_eq!(b.text(), "- a\n  - b\n    - c");
    }

    #[test]
    fn tab_keeps_the_cursor_with_the_text_and_is_a_no_op_off_a_list() {
        let mut b = TextBuffer::new();
        b.insert_str("- a\n- item");
        let before = b.cursor();
        b.nest_list_item(false); // inserts two spaces at the line start
        assert_eq!(
            b.cursor(),
            before + 2,
            "the cursor rides the inserted indent"
        );
        // A non-list line is not Tab-nestable.
        b.insert_str("\n\nplain");
        assert!(!b.cursor_on_list_item());
    }

    #[test]
    fn select_range_anchors_at_start_and_puts_cursor_at_end() {
        let mut b = TextBuffer::from_text("---\nx".to_string());
        b.select_range(0, 3); // select the "---"
        assert_eq!(b.selection(), Some((0, 3)));
        assert_eq!(b.cursor(), 3, "cursor at the end of the range");
        assert_eq!(b.selected_text().as_deref(), Some("---"));
        // A collapsed range leaves a bare cursor, no selection.
        b.select_range(2, 2);
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor(), 2);
    }

    #[test]
    fn replace_ranges_is_one_undo_step() {
        let mut b = TextBuffer::from_text("aa bb aa bb aa".to_string());
        assert!(b.replace_ranges(&[(0, 2), (6, 8), (12, 14)], "XYZ"));
        assert_eq!(b.text(), "XYZ bb XYZ bb XYZ");
        assert_eq!(b.cursor(), 17, "cursor at the end of the last replacement");
        assert!(b.undo());
        assert_eq!(b.text(), "aa bb aa bb aa", "all three undone at once");
    }

    #[test]
    fn replace_ranges_handles_varying_lengths() {
        // Regex matches vary in length; the cursor must land after the last
        // replacement even when earlier splices shift it.
        let mut b = TextBuffer::from_text("a11 b2 c333".to_string());
        assert!(b.replace_ranges(&[(1, 3), (5, 6), (8, 11)], "N"));
        assert_eq!(b.text(), "aN bN cN");
        assert_eq!(b.cursor(), 8);
        assert!(!b.replace_ranges(&[], "x"), "no ranges, no change");
    }

    #[test]
    fn replace_ranges_with_empty_replacement_deletes() {
        let mut b = TextBuffer::from_text("xay".to_string());
        assert!(b.replace_ranges(&[(1, 2)], ""));
        assert_eq!(b.text(), "xy");
        assert_eq!(b.cursor(), 1);
    }
}
