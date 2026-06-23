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

use crate::grapheme;

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

/// A restorable editing state.
#[derive(Clone)]
struct Snapshot {
    text: String,
    cursor: usize,
    anchor: Option<usize>,
    version: u64,
}

pub struct TextBuffer {
    text: String,
    /// Byte offset of the cursor; always a grapheme cluster boundary.
    cursor: usize,
    /// The other end of the selection, if one is active. `None` means no
    /// selection; equal to `cursor` means a collapsed (empty) one.
    anchor: Option<usize>,
    undo: Vec<Snapshot>,
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
            text,
            cursor: 0,
            anchor: None,
            undo: Vec::new(),
            redo: Vec::new(),
            group: Coalesce::None,
            version: 0,
            next_version: 1,
            saved_version: 0,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether the contents differ from the last point marked saved.
    pub fn is_modified(&self) -> bool {
        self.version != self.saved_version
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
    pub fn selected_text(&self) -> Option<&str> {
        self.selection().map(|(start, end)| &self.text[start..end])
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
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.anchor = None;
        self.touch();
    }

    /// Delete the selection if there is one, else the grapheme cluster before
    /// the cursor. Returns whether anything changed.
    pub fn backspace(&mut self) -> bool {
        if self.selection().is_some() {
            self.checkpoint(Coalesce::None);
            self.delete_selection();
            self.touch();
            return true;
        }
        let Some(prev) = self.prev_grapheme() else {
            return false;
        };
        self.checkpoint(Coalesce::Delete);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
        self.touch();
        true
    }

    /// Delete the selection if there is one, else the grapheme cluster after the
    /// cursor (forward delete). Returns whether anything changed.
    pub fn delete_forward(&mut self) -> bool {
        if self.selection().is_some() {
            self.checkpoint(Coalesce::None);
            self.delete_selection();
            self.touch();
            return true;
        }
        let Some(next) = self.next_grapheme() else {
            return false;
        };
        self.checkpoint(Coalesce::Delete);
        self.text.replace_range(self.cursor..next, "");
        self.touch();
        true
    }

    /// Select the whole buffer (anchor at the start, cursor at the end).
    pub fn select_all(&mut self) {
        self.group = Coalesce::None;
        if self.text.is_empty() {
            self.anchor = None;
            self.cursor = 0;
        } else {
            self.anchor = Some(0);
            self.cursor = self.text.len();
        }
    }

    /// Undo the most recent edit group. Returns whether anything changed.
    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.undo.pop() else {
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
        self.undo.push(self.snapshot());
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
            self.undo.push(self.snapshot());
            if self.undo.len() > UNDO_CAP {
                self.undo.remove(0);
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
        self.text.replace_range(start..end, "");
        self.cursor = start;
        self.anchor = None;
        true
    }

    /// Snap `offset` into `[0, len]` and back to the nearest grapheme cluster
    /// boundary at or before it. `prev_boundary(offset + 1)` is the last boundary
    /// at or before `offset`, which is the cluster start to land on.
    fn snap(&self, offset: usize) -> usize {
        let offset = offset.min(self.text.len());
        if offset == 0 || offset == self.text.len() {
            return offset;
        }
        grapheme::prev_boundary(&self.text, offset + 1)
    }

    /// Byte offset of the grapheme cluster boundary before the cursor, if any.
    fn prev_grapheme(&self) -> Option<usize> {
        if self.cursor == 0 {
            None
        } else {
            Some(grapheme::prev_boundary(&self.text, self.cursor))
        }
    }

    /// Byte offset of the grapheme cluster boundary after the cursor, if any.
    fn next_grapheme(&self) -> Option<usize> {
        if self.cursor == self.text.len() {
            None
        } else {
            Some(grapheme::next_boundary(&self.text, self.cursor))
        }
    }
}

/// Whether inserting `s` should coalesce with adjacent typing: a single
/// non-whitespace character does, so a typed word is one undo step while spaces,
/// newlines, and multi-character pastes start fresh steps.
fn coalescable_insert(s: &str) -> bool {
    let mut chars = s.chars();
    matches!((chars.next(), chars.next()), (Some(c), None) if !c.is_whitespace())
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
        assert_eq!(b.selected_text(), Some("hello"));
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
}
