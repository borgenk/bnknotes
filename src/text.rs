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

use std::borrow::Cow;
use std::collections::VecDeque;

use crate::markdown;
use crate::platform::grapheme;
use crate::rope::Rope;
use crate::table;

/// Cap on undo history, so a long session cannot grow the stacks without bound.
const UNDO_CAP: usize = 500;

/// What the Tab key inserts: spaces (a "soft tab"), not a literal `'\t'`. This
/// keeps the buffer free of tab characters and spares the renderer any tab-stop
/// logic. Rendering literal tabs found in a *loaded* file is a separate concern.
pub const SOFT_TAB: &str = "    ";

/// The kind of edit a checkpoint groups. Consecutive edits of the same non-`None`
/// kind merge into one undo step.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Coalesce {
    None,
    Insert,
    Delete,
}

/// The unit a pointer drag extends its selection by, set by the press that
/// started it: a click drags by character, a double-click by word, a triple-click
/// by line. See [`TextBuffer::drag_to`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragGrain {
    Char,
    Word,
    Line,
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

    /// Replace the whole buffer with `text` freshly read from disk, and mark it
    /// clean: the contents now match the file again. Used when the editor reloads
    /// a document changed outside it. Undo history is dropped (it described the
    /// old contents), the cursor returns to the top, and the version advances
    /// (never reused) so anything keyed on the version re-derives against the new
    /// text rather than a stale cache from a coincidentally equal number.
    pub fn reload(&mut self, text: String) {
        self.text = Rope::from_str(&text);
        self.cursor = 0;
        self.anchor = None;
        self.undo.clear();
        self.redo.clear();
        self.group = Coalesce::None;
        self.touch();
        self.saved_version = self.version;
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

    /// The byte range of the paragraph containing `offset`: the run of consecutive
    /// non-blank lines around it, bounded by blank lines or the buffer's ends, with
    /// the trailing newline excluded so the surrounding blank lines survive a
    /// replace. A blank line is its own (empty) paragraph. Used to scope an AI action
    /// to "this paragraph" when there is no selection.
    pub fn paragraph_at(&self, offset: usize) -> (usize, usize) {
        let rope = &self.text;
        let lines = rope.len_lines();
        let here = rope.byte_to_line(offset.min(rope.len()));
        let blank = |line: usize| {
            let (start, end) = rope.line_content_range(line);
            rope.byte_slice(start..end).trim().is_empty()
        };
        if blank(here) {
            return rope.line_content_range(here);
        }
        let mut first = here;
        while first > 0 && !blank(first - 1) {
            first -= 1;
        }
        let mut last = here;
        while last + 1 < lines && !blank(last + 1) {
            last += 1;
        }
        (
            rope.line_content_range(first).0,
            rope.line_content_range(last).1,
        )
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

    /// Insert pasted text at the cursor as its own undo step, first replacing any
    /// selection. With the caret in a quote's content the paste stays in the quote:
    /// each line it adds carries the quote marker. Before or within the marker, or
    /// off a quote, the text goes in unchanged.
    pub fn paste_str(&mut self, s: &str) {
        // The line is read once the selection is gone, since the paste lands on what
        // is left of it.
        let replacing = self.selection().is_some();
        if replacing {
            self.checkpoint(Coalesce::None);
            self.delete_selection();
        }
        let (ls, _, line) = self.content_line_at(self.cursor);
        let quoted = match markdown::classify(&line) {
            (markdown::LineStyle::Quote, len) if self.cursor >= ls + len => {
                markdown::quote_paste(&line[..len], self.cursor == ls + len, s)
            }
            _ => None,
        };
        let s = quoted.as_deref().unwrap_or(s);
        if !replacing {
            // A pasted marker that gave way to the line's own can leave nothing to
            // insert, and an edit that changes nothing takes no undo step.
            if s.is_empty() {
                return;
            }
            self.checkpoint(Coalesce::None);
        }
        self.text.insert(self.cursor, s);
        self.cursor += s.len();
        self.anchor = None;
        self.touch();
    }

    /// Insert s on a line of its own at the cursor, which stands at an edge of a
    /// line's content: above that line when the cursor is at its start, below it
    /// when at its end. The caret lands at the end of s, on the new line, and typing
    /// on from there joins the same undo step.
    pub fn insert_line(&mut self, s: &str, above: bool) {
        let group = if coalescable_insert(s) {
            Coalesce::Insert
        } else {
            Coalesce::None
        };
        self.checkpoint(group);
        self.delete_selection();
        let at = self.cursor;
        if above {
            self.text.insert(at, "\n");
            self.text.insert(at, s);
            self.cursor = at + s.len();
        } else {
            self.text.insert(at, s);
            self.text.insert(at, "\n");
            self.cursor = at + 1 + s.len();
        }
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
        // Anywhere in a table (a cell, or the header/delimiter edge right after the
        // table is typed), Enter adds a fresh body row and drops the caret into it.
        if let Some(block) = table::block_at(&self.text, self.text.byte_to_line(self.cursor)) {
            self.insert_body_row(block, self.text.byte_to_line(self.cursor));
            return true;
        }
        let (ls, le, line) = self.content_line_at(self.cursor);
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

    /// Handle Shift+Enter on a list item: insert a continuation line indented to
    /// the item's content column, with no marker of its own, so it reads as
    /// belonging to the item above. Splits the line at the cursor like a soft
    /// break. The indent is the item's full marker width in spaces; the layout
    /// recognizes such a line and insets it under the item text with no bullet.
    /// Falls back to a plain break when the cursor is not on an item's content.
    pub fn list_continuation(&mut self) -> bool {
        let (ls, _le, line) = self.content_line_at(self.cursor);
        let (_style, marker_len) = markdown::classify(&line);
        if marker_len == 0 || self.cursor < ls + marker_len {
            return self.enter();
        }
        let indent = " ".repeat(marker_len);
        self.insert_str(&format!("\n{indent}"));
        true
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
        let (.., line) = self.content_line_at(self.cursor);
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
        let (ls, _, line) = self.content_line_at(offset.min(self.text.len()));
        let (style, marker_len) = markdown::classify(&line);
        let markdown::LineStyle::Task { checked } = style else {
            return false;
        };
        self.checkpoint(Coalesce::None);
        self.flip_task_box(ls, marker_len, checked);
        self.touch();
        true
    }

    /// Flip the checkbox byte of the task whose content line starts at `ls` with
    /// block-marker length `marker_len`: `[ ] ` gains an `x`, `[x] ` clears back
    /// to a space. The check character sits inside the marker's trailing `[c] `.
    /// Edits the rope directly with no checkpoint or version bump, so the caller
    /// groups and stamps it; the swap is one byte for one, so nothing downstream
    /// shifts.
    fn flip_task_box(&mut self, ls: usize, marker_len: usize, checked: bool) {
        let at = ls + marker_len - 3;
        self.text.remove(at..at + 1);
        self.text.insert(at, if checked { " " } else { "x" });
    }

    /// Format the line(s) the cursor or selection touches as task-list items, the
    /// Ctrl+L gesture. Each line is handled on its own: a task flips its checkbox,
    /// and any other line (body, bullet, ordered, quote, heading) becomes an
    /// unchecked `- [ ] ` item, its old block marker replaced and its indentation
    /// kept. Across a multi-line selection blank lines are left alone, so a
    /// checklist gains no empty rows, but a single targeted line converts even
    /// when blank, so Ctrl+L on an empty line opens a checkbox. The whole sweep is
    /// one undo step, and the cursor and selection ride the edits so a repeated
    /// press keeps acting on the same lines. Returns whether the buffer changed.
    pub fn toggle_task_list(&mut self) -> bool {
        let (first, last) = self.block_lines();
        let single = first == last;
        let mut changed = false;
        // Bottom-up, so a line start above the one being edited stays valid as the
        // lines below it change length.
        for line_idx in (first..=last).rev() {
            let (ls, le) = self.text.line_content_range(line_idx);
            let content = self.text.byte_slice(ls..le);
            // A blank row in a multi-line sweep is a separator, not an item; a
            // single targeted line converts even when blank.
            if !single && content.trim().is_empty() {
                continue;
            }
            let (style, marker_len) = markdown::classify(&content);
            if !changed {
                self.checkpoint(Coalesce::None);
                changed = true;
            }
            if let markdown::LineStyle::Task { checked } = style {
                self.flip_task_box(ls, marker_len, checked);
                continue;
            }
            // Replace the old block marker (empty on a body line) with `- [ ] `,
            // keeping the line's leading indentation.
            let indent = markdown::leading_indent(&content);
            let start = ls + indent;
            let cut = ls + marker_len.max(indent);
            let new = "- [ ] ";
            let delta = new.len() as i64 - (cut - start) as i64;
            self.text.remove(start..cut);
            self.text.insert(start, new);
            self.cursor = shift_past(self.cursor, start, delta);
            self.anchor = self.anchor.map(|a| shift_past(a, start, delta));
        }
        if changed {
            self.touch();
        }
        changed
    }

    /// Quote the line(s) the cursor or selection touches, or unquote them when
    /// every one holding text is already a quote, the Ctrl+Shift+Q gesture. The
    /// marker goes at the line start in front of what the line holds, so a list item
    /// or heading becomes one inside the quote, and a line already quoted is left
    /// alone. A blank line between quoted lines takes a bare > so the quote stays one
    /// block; blank lines at the edges of a sweep stay blank, and a single targeted
    /// blank line takes "> " to start a quote. Unquoting removes one marker per line.
    /// The whole sweep is one undo step, and the cursor and selection ride the
    /// edits. Returns whether the buffer changed.
    pub fn toggle_quote(&mut self) -> bool {
        let (first, last) = self.block_lines();
        let has_text = |line| !matches!(self.quote_line(line), QuoteLine::Blank);
        let top = (first..=last).find(|&line| has_text(line));
        let bottom = (first..=last).rev().find(|&line| has_text(line));
        let (top, bottom) = match top.zip(bottom) {
            Some(span) => span,
            None if first == last => (first, last),
            None => return false,
        };
        let unquote = has_text(top)
            && (top..=bottom).all(|line| !matches!(self.quote_line(line), QuoteLine::Other));
        // Every path here edits at least one line: an unquote has a quote at the top,
        // a quote has a line without one, and a lone blank line takes a marker.
        self.checkpoint(Coalesce::None);
        // Bottom-up, so a line start above the one being edited stays valid as the
        // lines below it change length.
        for line in (top..=bottom).rev() {
            let ls = self.text.line_to_byte(line);
            let delta = match (unquote, self.quote_line(line)) {
                (true, QuoteLine::Quote(len)) => {
                    self.text.remove(ls..ls + len);
                    -(len as i64)
                }
                (false, QuoteLine::Blank) if top != bottom => {
                    self.text.insert(ls, ">");
                    1
                }
                (false, QuoteLine::Blank | QuoteLine::Other) => {
                    self.text.insert(ls, "> ");
                    2
                }
                _ => continue,
            };
            self.cursor = shift_past(self.cursor, ls, delta);
            self.anchor = self.anchor.map(|a| shift_past(a, ls, delta));
        }
        self.touch();
        true
    }

    /// How a line stands for a quote toggle.
    fn quote_line(&self, line: usize) -> QuoteLine {
        let (ls, le) = self.text.line_content_range(line);
        let content = self.text.slice(ls..le).as_cow();
        match markdown::classify(&content) {
            (markdown::LineStyle::Quote, len) => QuoteLine::Quote(len),
            _ if content.trim().is_empty() => QuoteLine::Blank,
            _ => QuoteLine::Other,
        }
    }

    /// The content of the logical line containing `byte` (excluding any trailing
    /// `'\n'`), as `(content_start, content_end, text)`. The shared "resolve the
    /// line, take its bounds, read it" step behind the list and markdown line
    /// inspections. The text is borrowed out of the rope whenever the line sits in
    /// one leaf, so inspecting a long paragraph copies nothing; a caller that needs
    /// the line to outlive an edit takes `into_owned` at its own call site.
    fn content_line_at(&self, byte: usize) -> (usize, usize, Cow<'_, str>) {
        let (ls, le) = self.text.line_content_range(self.text.byte_to_line(byte));
        (ls, le, self.text.slice(ls..le).as_cow())
    }

    /// The full logical line containing `byte`, through its trailing `'\n'` (so
    /// the newline is present as its own cluster), as `(line_start, text)`. A
    /// grapheme cluster never spans a `'\n'`, so this one-line window is all the
    /// cursor-boundary scans need. Borrowed like [`Self::content_line_at`], which
    /// matters most here: every cursor move runs this.
    fn full_line_at(&self, byte: usize) -> (usize, Cow<'_, str>) {
        let line = self.text.byte_to_line(byte);
        let ls = self.text.line_to_byte(line);
        let le = self.text.line_to_byte(line + 1);
        (ls, self.text.slice(ls..le).as_cow())
    }

    /// Whether the cursor's line is a list item (bullet or ordered, perhaps
    /// indented), so Tab nests it rather than inserting a soft tab.
    pub fn cursor_on_list_item(&self) -> bool {
        let (.., line) = self.content_line_at(self.cursor);
        is_list(&line)
    }

    /// Nest the cursor's list item one level deeper, or `outdent` it one level
    /// shallower, by adding or removing one indentation unit of leading spaces.
    /// Indenting is capped one level past the preceding item, so a nested item
    /// always has a parent above it. The cursor rides with the text so the editing
    /// position is preserved. Returns whether the buffer changed.
    pub fn nest_list_item(&mut self, outdent: bool) -> bool {
        let line_idx = self.text.byte_to_line(self.cursor);
        let (ls, _, line) = self.content_line_at(self.cursor);
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

    /// Whether the selection covers more than one line, so Tab shifts the whole
    /// block instead of replacing the selection with a soft tab.
    pub fn selection_spans_lines(&self) -> bool {
        self.selection().is_some_and(|(start, end)| {
            self.text.byte_to_line(start) != self.text.byte_to_line(end)
        })
    }

    /// Indent every line of the cursor's block by one unit, or `outdent` them by
    /// one. The cursor and selection stay over the same text, so holding Tab keeps
    /// shifting the same block. Blank lines are left blank; outdenting a line with
    /// less than a full unit of indentation takes what is there. Returns whether
    /// the buffer changed.
    pub fn indent_block(&mut self, outdent: bool) -> bool {
        let (first, last) = self.block_lines();
        let unit = self.block_indent_unit(first, last);
        let mut changed = false;
        // Bottom-up, so every line start above the one being edited stays valid.
        // Each line's change also moves the cursor and anchor when they sit after
        // it, which by then includes the shifts the lines below already applied.
        for line in (first..=last).rev() {
            let (ls, le) = self.text.line_content_range(line);
            let delta = if outdent {
                let indent = markdown::leading_indent(&self.text.slice(ls..le).as_cow());
                -(indent.min(unit) as i64)
            } else if ls == le {
                0
            } else {
                unit as i64
            };
            if delta == 0 {
                continue;
            }
            if !changed {
                self.checkpoint(Coalesce::None);
                changed = true;
            }
            if delta > 0 {
                self.text.insert(ls, &" ".repeat(delta as usize));
            } else {
                self.text.remove(ls..ls + (-delta) as usize);
            }
            self.cursor = shift_past(self.cursor, ls, delta);
            self.anchor = self.anchor.map(|a| shift_past(a, ls, delta));
        }
        if changed {
            self.touch();
        }
        changed
    }

    /// The inclusive line range Tab shifts: every line the selection touches, or
    /// the cursor's own line when there is no selection.
    fn block_lines(&self) -> (usize, usize) {
        let Some((start, end)) = self.selection() else {
            let line = self.text.byte_to_line(self.cursor);
            return (line, line);
        };
        let first = self.text.byte_to_line(start);
        let mut last = self.text.byte_to_line(end);
        // A selection ending at column 0 stops on the line above: it reached that
        // line's start without covering any of it.
        if last > first && end == self.text.line_to_byte(last) {
            last -= 1;
        }
        (first, last)
    }

    /// The indentation unit for the block of lines `first..=last`: a block holding
    /// a list item shifts by one nesting level, so Tab moves it exactly one level
    /// deeper; anything else shifts by a soft tab.
    fn block_indent_unit(&self, first: usize, last: usize) -> usize {
        let nests = (first..=last).any(|line| {
            let (ls, le) = self.text.line_content_range(line);
            is_list(&self.text.slice(ls..le).as_cow())
        });
        if nests {
            markdown::LIST_INDENT_SPACES
        } else {
            SOFT_TAB.len()
        }
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

    /// Delete the byte range start..end as one undo step, leaving the cursor at
    /// start and no selection; undo restores the cursor and selection as they were
    /// before. The bounds must be cluster boundaries (line and marker edges, and
    /// the delete keys' own reach, always are). A collapsed range is a no-op
    /// returning false.
    pub fn delete_range(&mut self, start: usize, end: usize) -> bool {
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

    /// Delete the span between the cursor and `target` as one undo step under
    /// `group`, leaving the cursor at the lower end. No-op (returns `false`) when
    /// the target is the cursor. The shared body of the per-key deletes, reached
    /// only after [`take_selection`](Self::take_selection) finds no selection.
    fn delete_to(&mut self, target: usize, group: Coalesce) -> bool {
        let (lo, hi) = (self.cursor.min(target), self.cursor.max(target));
        if lo == hi {
            return false;
        }
        self.checkpoint(group);
        self.text.remove(lo..hi);
        self.cursor = lo;
        self.touch();
        true
    }

    /// Delete the selection if there is one, else the grapheme cluster before
    /// the cursor. Returns whether anything changed. At the start of a table cell,
    /// Backspace moves to the previous cell rather than deleting a structural pipe,
    /// so the table cannot be broken by a stray delete; at the first cell it is a
    /// no-op.
    pub fn backspace(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        if let Some(cell) = table::cell_at(&self.text, self.cursor)
            && self.cursor <= cell.start
        {
            return match table::prev_cell_end(&self.text, cell) {
                Some(target) => self.set_cursor_changed(target, false),
                None => false,
            };
        }
        match self.prev_grapheme() {
            Some(prev) => self.delete_to(prev, Coalesce::Delete),
            None => false,
        }
    }

    /// Whether the caret sits in a table cell, so the caller routes Tab, a typed
    /// `|`, and Shift+Enter to their in-table behavior.
    pub fn cursor_in_table(&self) -> bool {
        table::cell_at(&self.text, self.cursor).is_some()
    }

    /// Move the caret to the next cell (`forward`) or the previous cell, for Tab and
    /// Shift+Tab. At the last cell, Tab adds a new row and lands in it; at the first
    /// cell, Shift+Tab does nothing. Returns whether the buffer or caret changed.
    pub fn tab_cell(&mut self, forward: bool) -> bool {
        let Some(cell) = table::cell_at(&self.text, self.cursor) else {
            return false;
        };
        let target = if forward {
            table::next_cell_start(&self.text, cell)
        } else {
            table::prev_cell_start(&self.text, cell)
        };
        match target {
            Some(t) => self.set_cursor_changed(t, false),
            None if forward => {
                self.insert_body_row(cell.block, cell.row);
                true
            }
            None => false,
        }
    }

    /// Insert a fresh empty body row into table `block`, coming from `from_row`, and
    /// drop the caret into its first cell. From a body row the new row lands just
    /// below it; from the header or delimiter it lands just after the delimiter (the
    /// first body position), never between the header and the delimiter, which would
    /// break the table.
    fn insert_body_row(&mut self, block: table::Block, from_row: usize) {
        let cols = table::row_cells(&self.text, block.delim).len().max(1);
        let after = if from_row <= block.delim {
            block.delim
        } else {
            from_row
        };
        self.insert_row_after(after, cols);
    }

    /// Add an empty row to the bottom of the table `block` (the add-row affordance),
    /// caret into its first cell. Returns true.
    pub fn add_table_row(&mut self, block: table::Block) -> bool {
        let cols = table::row_cells(&self.text, block.delim).len().max(1);
        self.insert_row_after(block.end - 1, cols);
        true
    }

    /// Insert an empty `cols`-column row just after logical line `row`, as one undo
    /// step, and move the caret to the new row's first cell.
    fn insert_row_after(&mut self, row: usize, cols: usize) {
        let (_, ce) = self.text.line_content_range(row);
        self.checkpoint(Coalesce::None);
        self.text
            .insert(ce, &format!("\n{}", table::empty_row(cols)));
        let new_line = self.text.byte_to_line(ce + 1);
        self.cursor = table::row_cells(&self.text, new_line)
            .first()
            .map_or(ce + 1, |c| c.0);
        self.anchor = None;
        self.touch();
    }

    /// Remove the caret's table row (Ctrl+Shift+Backspace), moving the caret into
    /// the row that takes its place (or out of the table). One undo step. `false`
    /// when the caret is not in a table cell.
    pub fn remove_table_row(&mut self) -> bool {
        let Some(cell) = table::cell_at(&self.text, self.cursor) else {
            return false;
        };
        let start = self.text.line_to_byte(cell.row);
        let end = self.text.line_to_byte(cell.row + 1).min(self.text.len());
        if start >= end {
            return false;
        }
        self.checkpoint(Coalesce::None);
        self.text.remove(start..end);
        // Land in the first cell of whatever row now sits here, else at the seam.
        let start = start.min(self.text.len());
        self.cursor = table::block_at(&self.text, self.text.byte_to_line(start))
            .and_then(|_| {
                table::row_cells(&self.text, self.text.byte_to_line(start))
                    .first()
                    .copied()
            })
            .map_or_else(|| self.snap(start), |c| c.0);
        self.anchor = None;
        self.touch();
        true
    }

    /// Remove the caret's table column from every row, delimiter included
    /// (Ctrl+Shift+Delete), then reseat the caret in the same row at a clamped
    /// column. A no-op on a single-column table (its last column cannot go) or off
    /// a cell. One undo step; rows edited back to front so cuts do not shift.
    pub fn remove_table_column(&mut self) -> bool {
        let Some(cell) = table::cell_at(&self.text, self.cursor) else {
            return false;
        };
        if table::row_cells(&self.text, cell.block.delim).len() <= 1 {
            return false;
        }
        let (col, row) = (cell.col, cell.row);
        self.checkpoint(Coalesce::None);
        for r in (cell.block.header..cell.block.end).rev() {
            if let Some((s, e)) = table::column_cut(&self.text, r, col) {
                self.text.remove(s..e);
            }
        }
        let cells = table::row_cells(&self.text, row);
        let target = col.min(cells.len().saturating_sub(1));
        self.cursor = cells
            .get(target)
            .map_or_else(|| self.text.line_to_byte(row), |c| c.0);
        self.anchor = None;
        self.touch();
        true
    }

    /// Add a trailing column to every row of the table `block` (the add-column
    /// affordance): an empty cell on each header/body row and a left-aligned
    /// delimiter cell on the delimiter row. One undo step. Rows are edited back to
    /// front so earlier insertions do not shift later ones.
    pub fn add_table_column(&mut self, block: table::Block) -> bool {
        self.checkpoint(Coalesce::None);
        for row in (block.header..block.end).rev() {
            let (ls, le) = self.text.line_content_range(row);
            let ends_pipe = self.text.byte_slice(ls..le).trim_end().ends_with('|');
            let cell = if row == block.delim { " --- " } else { "  " };
            let insert = if ends_pipe {
                format!("{cell}|")
            } else {
                format!(" |{cell}|")
            };
            self.text.insert(le, &insert);
        }
        self.touch();
        true
    }

    /// Delete the selection if there is one, else the grapheme cluster after the
    /// cursor (forward delete). Returns whether anything changed.
    pub fn delete_forward(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        match self.next_grapheme() {
            Some(next) => self.delete_to(next, Coalesce::Delete),
            None => false,
        }
    }

    /// Delete the selection if there is one, else from the cursor back to the
    /// previous word boundary (Ctrl+Backspace). Each word delete is its own undo
    /// step. Returns whether anything changed.
    pub fn delete_word_back(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let target = prev_word_rope(&self.text, self.cursor);
        self.delete_to(target, Coalesce::None)
    }

    /// Delete the selection if there is one, else from the cursor forward to the
    /// next word boundary (Ctrl+Delete). Returns whether anything changed.
    pub fn delete_word_forward(&mut self) -> bool {
        if self.take_selection() {
            return true;
        }
        let target = next_word_rope(&self.text, self.cursor);
        self.delete_to(target, Coalesce::None)
    }

    /// Where a delete from the bare cursor reaches: back (Backspace) or forward
    /// (Delete), by one grapheme cluster or, with word, to the word boundary. The
    /// cursor itself at the buffer's edge.
    pub fn delete_target(&self, forward: bool, word: bool) -> usize {
        match (forward, word) {
            (false, false) => self.prev_grapheme().unwrap_or(self.cursor),
            (true, false) => self.next_grapheme().unwrap_or(self.cursor),
            (false, true) => prev_word_rope(&self.text, self.cursor),
            (true, true) => next_word_rope(&self.text, self.cursor),
        }
    }

    /// Select the whole buffer (anchor at the start, cursor at the end).
    pub fn select_all(&mut self) {
        self.set_selection(0, self.text.len());
    }

    /// Select the word (or run of separators) at `offset`, for a double-click, and
    /// return its range. The word model matches the word-motion keys: a word is a
    /// run of alphanumerics and `_`; whitespace and punctuation are separators.
    pub fn select_word_at(&mut self, offset: usize) -> (usize, usize) {
        let (start, end) = word_range_rope(&self.text, offset);
        self.set_selection(start, end);
        (start, end)
    }

    /// Select the visible content of the logical line containing `offset`, for a
    /// triple-click. Deliberately *excludes* the trailing newline: the selection
    /// then matches what is on screen (no phantom cell past the last glyph), and
    /// the cursor stays at the line's end, so a following Backspace clears the
    /// line and leaves it empty instead of eating the break and merging into the
    /// line below. Removing the line itself is the explicit Ctrl+Shift+K
    /// ([`Self::delete_line`]). The buffer only ever holds `\n` (line endings are
    /// normalized on load), so one byte is stripped.
    pub fn select_line_at(&mut self, offset: usize) -> (usize, usize) {
        let (start, end) = self.line_range_at(offset);
        self.set_selection(start, end);
        (start, end)
    }

    /// The visible content range of the logical line containing `offset`.
    fn line_range_at(&self, offset: usize) -> (usize, usize) {
        let line = self.text.byte_to_line(offset.min(self.text.len()));
        self.text.line_content_range(line)
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

    /// The selection's fixed end: its anchor, or the cursor when nothing is
    /// selected. A pointer drag extends from here.
    pub fn selection_anchor(&self) -> usize {
        self.anchor.unwrap_or(self.cursor)
    }

    /// Extend a pointer drag anchored on the byte range `[start, end)` out to
    /// `offset`, by whole units of `grain`. Past the anchor's end the selection runs
    /// from `start`, before its start it runs from `end`, and inside it the anchor
    /// stays selected whole, so what the press took as a unit (a word, a line, a
    /// horizontal rule) survives a drag in either direction. A caret anchor
    /// (`start == end`) dragged by [`DragGrain::Char`] is a plain drag from the press
    /// point. Returns whether the selection moved.
    pub fn drag_to(
        &mut self,
        (start, end): (usize, usize),
        offset: usize,
        grain: DragGrain,
    ) -> bool {
        let (lo, hi) = self.grain_range(offset, grain);
        let (fixed, free) = if hi > end {
            (start, hi)
        } else if lo < start {
            (end, lo)
        } else {
            (start, end)
        };
        let before = (self.anchor, self.cursor);
        self.set_selection(fixed, free);
        before != (self.anchor, self.cursor)
    }

    /// The range a drag's free end covers at `offset`: the caret there, the word
    /// under it, or the content of its line.
    fn grain_range(&self, offset: usize, grain: DragGrain) -> (usize, usize) {
        match grain {
            DragGrain::Char => {
                let offset = self.snap(offset);
                (offset, offset)
            }
            DragGrain::Word => word_range_rope(&self.text, offset),
            DragGrain::Line => self.line_range_at(offset),
        }
    }

    /// Pull the low end of a mouse-driven selection back to the line start when it
    /// sits in a heading, bullet, or task marker, so the marker is taken whole. A
    /// collapsed heading marker is zero-width, so a press at the left edge lands on
    /// the text start, and that position grabs it too. A bullet or task marker keeps
    /// its width under the disc or checkbox, so only a press on that side of the
    /// text grabs it, and a sweep from the label's first letter selects the label.
    pub fn include_block_marker(&mut self) {
        let Some(anchor) = self.anchor else {
            return;
        };
        if anchor == self.cursor {
            return;
        }
        let lo = anchor.min(self.cursor);
        let line = self.text.byte_to_line(lo);
        let (ls, le) = self.text.line_content_range(line);
        let (style, marker_len) = markdown::classify(&self.text.byte_slice(ls..le));
        let marker_end = ls + marker_len;
        let in_marker = match style {
            markdown::LineStyle::Heading(_) => lo <= marker_end,
            markdown::LineStyle::Bullet | markdown::LineStyle::Task { .. } => lo < marker_end,
            _ => false,
        };
        if lo <= ls || !in_marker {
            return;
        }
        if self.cursor < anchor {
            self.cursor = ls;
        } else {
            self.anchor = Some(ls);
        }
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
        self.move_cluster(true, extend)
    }

    /// Move the cursor one grapheme cluster right. With `extend`, grow the
    /// selection; without it, a non-empty selection collapses to its right edge.
    /// Returns whether the cursor moved.
    pub fn move_right(&mut self, extend: bool) -> bool {
        self.move_cluster(false, extend)
    }

    /// Move the cursor one grapheme cluster, `back` (left) or forward (right).
    /// With `extend`, grow the selection; without it, a non-empty selection
    /// collapses to the edge in the direction of travel. Returns whether the
    /// cursor moved.
    fn move_cluster(&mut self, back: bool, extend: bool) -> bool {
        self.group = Coalesce::None;
        if !extend && let Some((start, end)) = self.selection() {
            self.cursor = if back { start } else { end };
            self.anchor = None;
            return true;
        }
        // Inside a table, a step off a cell's edge jumps to the adjacent cell (over
        // the hidden pipes and padding), or leaves the table at its outer edge.
        if let Some(cell) = table::cell_at(&self.text, self.cursor) {
            if back && self.cursor <= cell.start {
                let target = table::prev_cell_end(&self.text, cell)
                    .unwrap_or_else(|| table::exit_before(&self.text, cell.block));
                return self.set_cursor_changed(target, extend);
            }
            if !back && self.cursor >= cell.end {
                let target = table::next_cell_start(&self.text, cell)
                    .unwrap_or_else(|| table::exit_after(&self.text, cell.block));
                return self.set_cursor_changed(target, extend);
            }
            // Within the cell's content: fall through to grapheme movement.
        }
        let target = if back {
            self.prev_grapheme()
        } else {
            self.next_grapheme()
        };
        match target {
            Some(t) => {
                self.set_cursor(t, extend);
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
    /// whether a selection was removed. A selection anchored in a table removes the
    /// whole rows it spans (see [`Self::selection_for_delete`]).
    fn delete_selection(&mut self) -> bool {
        let Some((start, end)) = self.selection_for_delete() else {
            return false;
        };
        self.text.remove(start..end);
        self.cursor = start;
        self.anchor = None;
        true
    }

    /// The byte range a delete of the current selection removes: the selection
    /// itself, except a selection anchored in a table cell removes the whole rows
    /// it spans, so selecting cells and pressing Delete removes those rows rather
    /// than shredding the pipes. `None` when there is no selection.
    fn selection_for_delete(&self) -> Option<(usize, usize)> {
        let (s, e) = self.selection()?;
        if table::cell_at(&self.text, s).is_none() {
            return Some((s, e));
        }
        let lo = self.text.byte_to_line(s);
        let hi = self.text.byte_to_line(e.saturating_sub(1).max(s));
        let start = self.text.line_to_byte(lo);
        let end = self.text.line_to_byte(hi + 1).min(self.text.len());
        Some((start, end))
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
        let (ls, s) = self.full_line_at(offset);
        if offset == ls {
            return offset; // a line start is always a boundary
        }
        ls + grapheme::prev_boundary(&s, offset - ls + 1)
    }

    /// Byte offset of the grapheme cluster boundary before the cursor, if any.
    /// The preceding cluster lies on the line containing `cursor - 1` (a `'\n'`
    /// is its own single-byte cluster), so only that line is materialized.
    fn prev_grapheme(&self) -> Option<usize> {
        if self.cursor == 0 {
            return None;
        }
        let (ls, s) = self.full_line_at(self.cursor - 1);
        Some(ls + grapheme::prev_boundary(&s, self.cursor - ls))
    }

    /// Byte offset of the grapheme cluster boundary after the cursor, if any.
    /// The cluster at the cursor lies entirely on its line, so only that line is
    /// materialized.
    fn next_grapheme(&self) -> Option<usize> {
        if self.cursor == self.text.len() {
            return None;
        }
        let (ls, s) = self.full_line_at(self.cursor);
        Some(ls + grapheme::next_boundary(&s, self.cursor - ls))
    }
}

/// A line as a quote toggle sees it. A bare > is a quote with no content, never
/// blank.
enum QuoteLine {
    Blank,
    /// A quote, with its marker's length.
    Quote(usize),
    Other,
}

/// Move a position along with a `delta`-byte indentation change made at line
/// start `ls`. Positions before the line hold still; one inside a run that was
/// removed lands at the line start rather than in the middle of a cluster.
fn shift_past(pos: usize, ls: usize, delta: i64) -> usize {
    if pos < ls {
        pos
    } else if delta >= 0 {
        pos + delta as usize
    } else {
        pos.saturating_sub((-delta) as usize).max(ls)
    }
}

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

/// Whether `line` is a list item (bullet, ordered, or task, perhaps indented),
/// the lines Tab nests and `enter` continues.
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

/// The `(absolute byte start, is_word)` clusters at or after `pos`, in order,
/// walking the rope a logical line at a time from the line containing `pos`. The
/// shared traversal of the forward word-motion scanners; each layers its own
/// state machine over these.
fn forward_clusters(rope: &Rope, pos: usize) -> impl Iterator<Item = (usize, bool)> + '_ {
    (rope.byte_to_line(pos.min(rope.len()))..rope.len_lines())
        .flat_map(move |line| line_clusters(rope, line))
        .filter(move |&(abs, _)| abs >= pos)
}

/// The `(absolute byte start, is_word)` clusters strictly before `pos`, in
/// reverse order, walking the rope backward a logical line at a time from the
/// line containing `pos - 1`. The shared traversal of the backward word-motion
/// scanners; each layers its own state machine over these.
fn backward_clusters(rope: &Rope, pos: usize) -> impl Iterator<Item = (usize, bool)> + '_ {
    (0..=rope.byte_to_line(pos.saturating_sub(1)))
        .rev()
        .flat_map(move |line| line_clusters(rope, line).into_iter().rev())
        .filter(move |&(abs, _)| abs < pos)
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
    // Two phases: skip leading separators, then consume the word run, stopping at
    // the first separator after it.
    let mut in_word = false;
    for (abs, is_word) in forward_clusters(rope, pos) {
        if in_word && !is_word {
            return abs;
        }
        in_word |= is_word;
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
    let mut in_word = false;
    let mut start = 0;
    for (abs, is_word) in backward_clusters(rope, pos) {
        if in_word && !is_word {
            return start;
        }
        if is_word {
            in_word = true;
            start = abs;
        }
    }
    if in_word { start } else { 0 }
}

/// The byte range of the word (or run of separators) at `offset`, for
/// double-click selection: the maximal run of clusters sharing the clicked
/// cluster's class (word vs separator), bounded by the clicked line. A click at a
/// word's trailing edge, or at the buffer end, still selects the word to the
/// left. Scans the rope outward from the click instead of materializing the whole
/// document.
fn word_range_rope(rope: &Rope, offset: usize) -> (usize, usize) {
    let len = rope.len();
    if len == 0 {
        return (0, 0);
    }
    let c = clicked_cluster(rope, offset.min(len));
    // A separator under the click with a word immediately to its left selects
    // that word (so a click on a word's trailing edge or the space after it still
    // grabs the word): expand the word run that ends at the click. A word run
    // stops at the newline on its own, so only the general case needs clamping.
    if !c.is_word && c.prev_is_word == Some(true) {
        return (class_run_start(rope, c.start, true), c.start);
    }
    // The run stays on the clicked line: a separator run is otherwise unbounded by
    // line breaks, so double-clicking a `---` rule (or any punctuation run flanked
    // by blank lines) would reach through the newlines and take the paragraphs on
    // both sides.
    let (ls, le) = c.line;
    let start = class_run_start(rope, c.start, c.is_word).max(ls);
    let end = class_run_end(rope, c.start, c.is_word).min(le);
    (start, end)
}

/// The cluster a double-click landed on, and the context a word range grows out
/// of it with.
struct Clicked {
    /// The cluster's start offset.
    start: usize,
    /// Whether the cluster is a word cluster rather than a separator.
    is_word: bool,
    /// The class of the cluster immediately to its left, `None` at the buffer
    /// start.
    prev_is_word: Option<bool>,
    /// The content range of the logical line it sits on, newline excluded.
    line: (usize, usize),
}

/// The cluster a double-click at byte `offset` lands on. A click at the buffer
/// end falls on the last cluster; for a click at a line start the left neighbor
/// is the previous line's terminating `'\n'`, a separator. Materializes only the
/// click's logical line.
fn clicked_cluster(rope: &Rope, offset: usize) -> Clicked {
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
    let (start, is_word) = clusters
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
    Clicked {
        start,
        is_word,
        prev_is_word,
        line: rope.line_content_range(line),
    }
}

/// The start byte offset of the maximal run of class `word` that contains the
/// cluster beginning at `from`, scanning the rope backward line by line.
fn class_run_start(rope: &Rope, from: usize, word: bool) -> usize {
    if from == 0 {
        return 0;
    }
    let mut start = from;
    for (abs, is_word) in backward_clusters(rope, from) {
        if is_word != word {
            return start;
        }
        start = abs;
    }
    start
}

/// The end byte offset (exclusive) of the maximal run of class `word` that
/// contains the cluster beginning at `from`, scanning the rope forward line by
/// line.
fn class_run_end(rope: &Rope, from: usize, word: bool) -> usize {
    for (abs, is_word) in forward_clusters(rope, from) {
        if is_word != word {
            return abs;
        }
    }
    rope.len()
}

#[cfg(test)]
mod tests {
    use crate::text::*;

    #[test]
    fn insert_advances_cursor() {
        let mut b = TextBuffer::new();
        b.insert_str("hi");
        assert_eq!(b.text(), "hi");
        assert_eq!(b.cursor, 2);
    }

    #[test]
    fn reload_replaces_contents_clean_at_a_fresh_version() {
        let mut b = TextBuffer::from_text("old text".to_string());
        b.insert_str(" and edits");
        assert!(b.is_modified(), "edits leave the buffer dirty");
        let dirty_version = b.version();

        b.reload("brand new from disk".to_string());
        assert_eq!(b.text(), "brand new from disk");
        assert!(!b.is_modified(), "a reload matches disk, so it reads clean");
        assert_eq!(b.cursor, 0, "the caret returns to the top");
        assert!(
            b.version() > dirty_version,
            "the version advances so caches keyed on it re-derive"
        );
        // Undo must not walk back into the pre-reload contents.
        b.undo();
        assert_eq!(
            b.text(),
            "brand new from disk",
            "reload clears undo history"
        );
    }

    // --- In-grid table editing ---------------------------------------------

    /// A buffer holding one table, with the caret at the start of `row`'s cell
    /// `col`. `row` and `col` index the drawn cells (the delimiter is line 1).
    fn table_buf(text: &str, row: usize, col: usize) -> TextBuffer {
        let mut b = TextBuffer::from_text(text.to_string());
        b.set_cursor(table::row_cells(b.rope(), row)[col].0, false);
        b
    }

    const TBL: &str = "| ab | cd |\n| - | - |\n| ef | gh |\n";

    #[test]
    fn cursor_in_table_tracks_prose_versus_cells() {
        let mut b = TextBuffer::from_text(format!("hi\n{TBL}"));
        b.set_cursor(1, false);
        assert!(!b.cursor_in_table(), "in the prose line");
        b.set_cursor(table::row_cells(b.rope(), 1)[0].0, false);
        assert!(b.cursor_in_table(), "in a header cell");
    }

    #[test]
    fn move_right_steps_within_a_cell_then_hops_to_the_next() {
        let mut b = table_buf(TBL, 0, 0); // start of "ab"
        let cells = table::row_cells(b.rope(), 0);
        b.move_right(false);
        assert_eq!(b.cursor(), cells[0].0 + 1, "one grapheme inside the cell");
        b.set_cursor(cells[0].1, false); // end of "ab"
        b.move_right(false);
        assert_eq!(b.cursor(), cells[1].0, "hopped to the next cell's start");
    }

    #[test]
    fn move_left_hops_to_the_previous_cell_and_exits_the_table() {
        let mut b = table_buf(TBL, 0, 1); // start of "cd"
        let cells = table::row_cells(b.rope(), 0);
        b.move_left(false);
        assert_eq!(b.cursor(), cells[0].1, "back to the previous cell's end");
        // Left from the very first cell leaves the table (it opens the document).
        b.set_cursor(cells[0].0, false);
        b.move_left(false);
        assert_eq!(b.cursor(), 0, "exited before the table");
    }

    #[test]
    fn tab_walks_cells_and_appends_a_row_at_the_end() {
        let mut b = table_buf(TBL, 0, 0);
        assert!(b.tab_cell(true));
        assert_eq!(b.cursor(), table::row_cells(b.rope(), 0)[1].0, "next cell");
        // From the last cell of the last row, Tab adds a fresh row.
        let last = table::row_cells(b.rope(), 2);
        b.set_cursor(last[1].0, false);
        let n = b.rope().len_lines();
        assert!(b.tab_cell(true));
        assert_eq!(b.rope().len_lines(), n + 1, "a row was appended");
        assert!(b.cursor_in_table(), "and the caret is in it");
    }

    #[test]
    fn enter_adds_a_row_below_with_the_right_column_count() {
        let mut b = table_buf(TBL, 2, 0); // in the body row
        let n = b.rope().len_lines();
        assert!(b.enter());
        assert_eq!(b.rope().len_lines(), n + 1);
        let new_line = b.rope().byte_to_line(b.cursor());
        assert_eq!(table::row_cells(b.rope(), new_line).len(), 2, "two cells");
        assert!(b.cursor_in_table());
    }

    #[test]
    fn backspace_at_a_cell_start_moves_left_and_keeps_the_structure() {
        let mut b = table_buf(TBL, 0, 1); // start of "cd"
        let before = b.text();
        let cells = table::row_cells(b.rope(), 0);
        assert!(b.backspace());
        assert_eq!(b.text(), before, "no structural byte deleted");
        assert_eq!(b.cursor(), cells[0].1, "moved to the previous cell's end");
    }

    #[test]
    fn typing_inside_a_cell_edits_only_that_cell() {
        let mut b = table_buf(TBL, 0, 0);
        b.set_cursor(table::row_cells(b.rope(), 0)[0].1, false); // end of "ab"
        b.insert_str("X");
        assert_eq!(b.text(), "| abX | cd |\n| - | - |\n| ef | gh |\n");
    }

    #[test]
    fn enter_at_the_header_edge_adds_a_body_row_not_a_break() {
        // Regression: with the caret at the end of a just-typed 1-column table's
        // header line, Enter used to insert a row *between* the header and the
        // delimiter, which reparsed "|test|" as plain text above a shifted table.
        let mut b = TextBuffer::from_text("|test|\n|-".to_string());
        b.set_cursor(6, false); // end of the header content, on the newline
        assert!(b.enter());
        assert_eq!(
            b.text(),
            "|test|\n|-\n|  |",
            "a body row is added after the delimiter, header intact"
        );
        let block = table::block_at(b.rope(), 0).expect("still one table from line 0");
        assert_eq!((block.header, block.delim, block.end), (0, 1, 3));
        assert!(b.cursor_in_table(), "caret in the new body row");
        assert_eq!(b.rope().byte_to_line(b.cursor()), 2);
    }

    #[test]
    fn enter_on_the_delimiter_row_also_adds_a_body_row() {
        // Right after typing the delimiter, the caret sits on it; Enter should still
        // grow the table rather than drop a blank line below it.
        let mut b = TextBuffer::from_text("|test|\n|-".to_string());
        b.set_cursor(b.rope().len(), false); // end of "|-"
        assert!(b.enter());
        assert_eq!(b.text(), "|test|\n|-\n|  |");
        assert_eq!(
            b.rope().byte_to_line(b.cursor()),
            2,
            "caret in the body row"
        );
    }

    #[test]
    fn shift_selecting_in_a_row_then_delete_removes_that_row() {
        let mut b = table_buf(TBL, 2, 0); // body row "ef gh" (line 2, past the delimiter)
        b.move_right(true); // Shift+Right marks the row (row-granular)
        assert!(b.backspace(), "delete consumed the selection");
        assert_eq!(b.text(), "| ab | cd |\n| - | - |\n", "the body row is gone");
    }

    #[test]
    fn selecting_across_cells_of_two_rows_deletes_both_rows() {
        let text = "| a | b |\n| - | - |\n| c | d |\n| e | f |\n";
        let mut b = TextBuffer::from_text(text.to_string());
        let r2 = table::row_cells(b.rope(), 2); // "c d"
        let r3 = table::row_cells(b.rope(), 3); // "e f"
        b.set_cursor(r2[0].0, false); // anchor in row 2
        b.set_cursor(r3[1].0, true); // extend into row 3
        assert!(b.delete_forward());
        assert_eq!(b.text(), "| a | b |\n| - | - |\n", "both body rows removed");
    }

    #[test]
    fn add_row_appends_an_empty_row_to_the_bottom() {
        let mut b = TextBuffer::from_text(TBL.to_string());
        let block = table::block_at(b.rope(), 0).unwrap();
        let n = b.rope().len_lines();
        assert!(b.add_table_row(block));
        assert_eq!(b.rope().len_lines(), n + 1);
        // The new row is the last table row and has the table's column count.
        let last = table::block_at(b.rope(), 0).unwrap().end - 1;
        assert_eq!(table::row_cells(b.rope(), last).len(), 2);
    }

    #[test]
    fn remove_row_deletes_the_carets_row() {
        let mut b = table_buf(TBL, 2, 0); // body row "ef gh"
        assert!(b.remove_table_row());
        assert_eq!(b.text(), "| ab | cd |\n| - | - |\n", "the row is gone");
    }

    #[test]
    fn remove_column_drops_it_from_every_row() {
        let text = "| a | b | c |\n| - | - | - |\n| d | e | f |\n";
        let mut b = TextBuffer::from_text(text.to_string());
        let cells = table::row_cells(b.rope(), 2); // caret in the middle column
        b.set_cursor(cells[1].0, false);
        assert!(b.remove_table_column());
        assert_eq!(
            table::row_cells(b.rope(), 0).len(),
            2,
            "header lost a column"
        );
        assert_eq!(table::row_cells(b.rope(), 2).len(), 2, "body lost a column");
        assert!(
            table::block_at(b.rope(), 0).is_some(),
            "still a valid table"
        );
        assert!(
            !b.text().contains('b') && !b.text().contains('e'),
            "the middle column's cells are gone: {:?}",
            b.text()
        );
    }

    #[test]
    fn remove_column_refuses_the_last_one() {
        let mut b = TextBuffer::from_text("| a |\n| - |\n| b |\n".to_string());
        b.set_cursor(table::row_cells(b.rope(), 0)[0].0, false);
        assert!(
            !b.remove_table_column(),
            "a one-column table keeps its column"
        );
    }

    #[test]
    fn add_column_widens_every_row_including_the_delimiter() {
        let mut b = TextBuffer::from_text(TBL.to_string());
        let block = table::block_at(b.rope(), 0).unwrap();
        assert!(b.add_table_column(block));
        // Header, delimiter, and each body row all gain a cell.
        assert_eq!(table::row_cells(b.rope(), 0).len(), 3, "header widened");
        assert_eq!(table::row_cells(b.rope(), 2).len(), 3, "body widened");
        // The table still parses (the delimiter grew a valid cell).
        assert!(table::block_at(b.rope(), 0).is_some(), "still a table");
        assert_eq!(
            table::block_at(b.rope(), 0).unwrap().end,
            block.end,
            "no rows added"
        );
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
    fn shift_enter_continues_a_bullet_with_an_indented_line() {
        let mut b = TextBuffer::new();
        b.insert_str("- foo");
        assert!(b.list_continuation());
        assert_eq!(b.text(), "- foo\n  ", "an indented continuation, no bullet");
        assert_eq!(b.cursor(), 8, "the cursor lands after the indent");
    }

    #[test]
    fn shift_enter_indents_an_ordered_continuation_under_its_number() {
        let mut b = TextBuffer::new();
        b.insert_str("1. foo");
        assert!(b.list_continuation());
        assert_eq!(b.text(), "1. foo\n   ", "three spaces, the `1. ` width");
        assert_eq!(b.cursor(), 10);
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

    // --- Ctrl+L: format line(s) as task items ------------------------------

    #[test]
    fn toggle_task_list_converts_each_block_style_and_keeps_indent() {
        // A body line gains the box; the caret rides past the inserted marker.
        let mut b = TextBuffer::from_text("groceries".to_string());
        b.set_cursor(9, false); // end of "groceries"
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] groceries");
        assert_eq!(b.cursor(), 15, "the caret follows the text past the marker");
        // A bullet, an ordered item, a quote, and a heading each shed their old
        // marker for the box, and indentation (list nesting) is preserved.
        for (before, after) in [
            ("- milk", "- [ ] milk"),
            ("+ milk", "- [ ] milk"),
            ("1. milk", "- [ ] milk"),
            ("3) milk", "- [ ] milk"),
            ("> quote", "- [ ] quote"),
            ("# head", "- [ ] head"),
            ("  - sub", "  - [ ] sub"),
            ("    1. deep", "    - [ ] deep"),
        ] {
            let mut b = TextBuffer::from_text(before.to_string());
            assert!(b.toggle_task_list(), "{before:?} converts");
            assert_eq!(b.text(), after, "{before:?} -> {after:?}");
        }
    }

    #[test]
    fn toggle_task_list_toggles_a_line_already_a_task() {
        let mut b = TextBuffer::from_text("- [ ] a".to_string());
        b.set_cursor(7, false); // end of the label
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [x] a");
        assert_eq!(b.cursor(), 7, "toggling a task never moves the caret");
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] a");
    }

    #[test]
    fn toggle_task_list_over_a_selection_handles_each_line() {
        // A mixed block: a bullet and a plain line convert, an existing task
        // flips, all in one pass over the selection.
        let mut b = TextBuffer::from_text("- one\ntwo\n- [ ] three".to_string());
        b.set_cursor(0, false);
        b.set_cursor(b.rope().len(), true); // select every line
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] one\n- [ ] two\n- [x] three");
        // One more press now flips all three (they are all tasks).
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [x] one\n- [x] two\n- [ ] three");
    }

    #[test]
    fn toggle_task_list_skips_blank_lines_in_a_multi_line_sweep() {
        // A loose list: the blank separator stays blank, only the items convert.
        let mut b = TextBuffer::from_text("a\n\nb".to_string());
        b.set_cursor(0, false);
        b.set_cursor(4, true); // select all three lines
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] a\n\n- [ ] b");
    }

    #[test]
    fn toggle_task_list_converts_a_single_blank_line() {
        // With no selection, a blank line is a deliberate "start a checkbox here".
        let mut b = TextBuffer::from_text("x\n\ny".to_string());
        b.set_cursor(2, false); // on the empty middle line
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "x\n- [ ] \ny");
        assert_eq!(b.cursor(), 8, "the caret waits at the end of the fresh box");
    }

    #[test]
    fn toggle_task_list_is_one_undo_step() {
        let mut b = TextBuffer::from_text("- one\ntwo".to_string());
        b.set_cursor(0, false);
        b.set_cursor(b.rope().len(), true);
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] one\n- [ ] two");
        assert!(b.undo());
        assert_eq!(b.text(), "- one\ntwo", "the whole sweep reverts at once");
        assert!(b.redo());
        assert_eq!(b.text(), "- [ ] one\n- [ ] two", "and redo replays it");
    }

    #[test]
    fn toggle_task_list_keeps_the_selection_over_the_converted_lines() {
        // The selection rides the edits (the anchor moves past the marker just
        // inserted on the first line, as block-indent does), so it still spans
        // both lines and a second press keeps acting on them.
        let mut b = TextBuffer::from_text("one\ntwo".to_string());
        b.set_cursor(0, false);
        b.set_cursor(7, true); // select both lines
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] one\n- [ ] two");
        let (lo, hi) = b.selection().expect("a selection survives");
        assert_eq!(
            b.rope().byte_to_line(lo),
            0,
            "still starts on the first line"
        );
        assert_eq!(b.rope().byte_to_line(hi), 1, "still ends on the second");
        assert!(b.toggle_task_list());
        assert_eq!(
            b.text(),
            "- [x] one\n- [x] two",
            "the second press flips both",
        );
    }

    #[test]
    fn toggle_task_list_on_a_long_ordered_marker_shifts_the_caret_back() {
        // A marker longer than `- [ ] ` shrinks the line; the caret, past it,
        // moves back by the difference rather than drifting into the markup.
        let mut b = TextBuffer::from_text("123456789. deep".to_string());
        b.set_cursor(15, false); // end of "deep"
        assert!(b.toggle_task_list());
        assert_eq!(b.text(), "- [ ] deep");
        assert_eq!(b.cursor(), 10, "caret at the end of the shortened line");
    }

    #[test]
    fn toggle_task_list_on_empty_multi_line_selection_is_a_no_op() {
        let mut b = TextBuffer::from_text("\n\n".to_string());
        b.set_cursor(0, false);
        b.set_cursor(2, true);
        assert!(!b.toggle_task_list(), "all-blank selection changes nothing");
        assert_eq!(b.text(), "\n\n");
    }

    // --- Ctrl+Shift+Q: quote line(s) -----------------------------------------

    #[test]
    fn toggle_quote_quotes_a_line_and_unquotes_it_again() {
        let mut b = TextBuffer::from_text("note".to_string());
        b.set_cursor(4, false);
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "> note");
        assert_eq!(b.cursor(), 6, "the caret rides past the marker");
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "note");
        assert_eq!(b.cursor(), 4);
        // A tight marker and a bare marker unquote too.
        let mut b = TextBuffer::from_text(">tight\n>".to_string());
        b.set_cursor(0, false);
        b.set_cursor(8, true);
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "tight\n");
    }

    #[test]
    fn toggle_quote_bridges_the_blank_lines_inside_a_block_only() {
        // Blank lines between paragraphs take a bare marker so the quote stays one
        // block; the blank lines the selection ran over at its edges stay blank.
        let mut b = TextBuffer::from_text("x\n\na\n\nb\n\ny".to_string());
        b.set_cursor(2, false); // the blank line under x
        b.set_cursor(8, true); // the blank line above y
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "x\n\n> a\n>\n> b\n\ny");
        assert!(b.toggle_quote(), "the selection still spans the block");
        assert_eq!(b.text(), "x\n\na\n\nb\n\ny", "and unquoting round-trips");
    }

    #[test]
    fn toggle_quote_puts_block_markers_inside_the_quote() {
        // A quote is a container: the line's own marker stays, behind the quote's.
        let mut b = TextBuffer::from_text("- item\n# head\n  1. one".to_string());
        b.set_cursor(0, false);
        b.set_cursor(b.rope().len(), true);
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "> - item\n> # head\n>   1. one");
    }

    #[test]
    fn toggle_quote_over_a_partly_quoted_block_quotes_the_rest() {
        // Unquoting needs every line with text quoted; otherwise the lines missing a
        // marker gain one and the quoted lines are not quoted twice.
        let mut b = TextBuffer::from_text("> a\nb".to_string());
        b.set_cursor(0, false);
        b.set_cursor(5, true);
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "> a\n> b");
    }

    #[test]
    fn toggle_quote_starts_a_quote_on_a_lone_blank_line() {
        let mut b = TextBuffer::from_text("x\n\ny".to_string());
        b.set_cursor(2, false);
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "x\n> \ny");
        assert_eq!(b.cursor(), 4, "the caret waits after the marker");
        // A selection of only blank lines has nothing to quote.
        let mut b = TextBuffer::from_text("\n\n".to_string());
        b.set_cursor(0, false);
        b.set_cursor(2, true);
        assert!(!b.toggle_quote());
        assert_eq!(b.text(), "\n\n");
    }

    #[test]
    fn toggle_quote_is_one_undo_step() {
        let mut b = TextBuffer::from_text("a\n\nb".to_string());
        b.set_cursor(0, false);
        b.set_cursor(4, true);
        assert!(b.toggle_quote());
        assert!(b.undo());
        assert_eq!(b.text(), "a\n\nb", "the whole sweep reverts at once");
        assert!(b.redo());
        assert_eq!(b.text(), "> a\n>\n> b");
    }

    #[test]
    fn unquoting_clamps_a_caret_inside_the_marker() {
        let mut b = TextBuffer::from_text("> a".to_string());
        b.set_cursor(1, false); // between > and the space
        assert!(b.toggle_quote());
        assert_eq!(b.text(), "a");
        assert_eq!(b.cursor(), 0, "the caret is not left past a deleted byte");
    }

    // --- Paste into a quote ---------------------------------------------------

    #[test]
    fn a_paste_on_a_quote_line_quotes_every_line_it_adds() {
        let mut b = TextBuffer::from_text("> ".to_string());
        b.set_cursor(2, false);
        b.paste_str("one\n\ntwo");
        assert_eq!(
            b.text(),
            "> one\n>\n> two",
            "a blank line takes the bare marker"
        );
        assert_eq!(b.cursor(), b.rope().len());
        // A tight marker carries over as it is.
        let mut b = TextBuffer::from_text(">".to_string());
        b.set_cursor(1, false);
        b.paste_str("a\nb");
        assert_eq!(b.text(), ">a\n>b");
    }

    #[test]
    fn a_paste_in_a_quote_keeps_the_rest_of_the_line_in_the_quote() {
        // The last pasted line joins the text after the caret, so it takes the full
        // marker even when the paste ends in a newline.
        let mut b = TextBuffer::from_text("> foobar".to_string());
        b.set_cursor(5, false); // after "foo"
        b.paste_str("x\ny");
        assert_eq!(b.text(), "> foox\n> ybar");
        assert_eq!(b.cursor(), 10, "the caret lands after the pasted y");
        let mut b = TextBuffer::from_text("> ".to_string());
        b.set_cursor(2, false);
        b.paste_str("a\n");
        assert_eq!(b.text(), "> a\n> ", "the caret waits on a fresh quote line");
        assert_eq!(b.cursor(), 6);
    }

    #[test]
    fn a_quoted_paste_is_not_quoted_twice() {
        // Right after the marker, a pasted line's own marker gives way to the one
        // the caret's line already has; later lines keep theirs.
        for (pasted, after) in [
            ("> a\n> b", "> a\n> b"),
            ("> a\nb", "> a\n> b"),
            ("> a", "> a"),
        ] {
            let mut b = TextBuffer::from_text("> ".to_string());
            b.set_cursor(2, false);
            b.paste_str(pasted);
            assert_eq!(b.text(), after, "{pasted:?}");
        }
        // Past other text the first line goes in as it is.
        let mut b = TextBuffer::from_text("> x".to_string());
        b.set_cursor(3, false);
        b.paste_str("> a");
        assert_eq!(b.text(), "> x> a");
    }

    #[test]
    fn a_paste_of_only_a_quote_marker_into_a_quote_is_no_edit() {
        // The pasted marker gives way to the line's own, leaving nothing to insert,
        // so the paste leaves no undo step and the note stays unmodified.
        let mut b = TextBuffer::from_text("> ".to_string());
        b.set_cursor(2, false);
        let version = b.version();
        b.paste_str("> ");
        assert_eq!(b.text(), "> ");
        assert_eq!(b.version(), version, "nothing changed");
        assert!(!b.undo(), "and there is no step to undo");
    }

    #[test]
    fn a_paste_before_the_quote_marker_or_off_a_quote_goes_in_unchanged() {
        let mut b = TextBuffer::from_text("> q".to_string());
        b.set_cursor(0, false);
        b.paste_str("a\n\nb");
        assert_eq!(b.text(), "a\n\nb> q");
        let mut b = TextBuffer::from_text("- item".to_string());
        b.set_cursor(6, false);
        b.paste_str("a\n\nb");
        assert_eq!(b.text(), "- itema\n\nb");
    }

    #[test]
    fn a_paste_over_a_selection_in_a_quote_is_one_undo_step() {
        let mut b = TextBuffer::from_text("> old".to_string());
        b.set_cursor(2, false);
        b.set_cursor(5, true); // select "old"
        b.paste_str("a\nb");
        assert_eq!(b.text(), "> a\n> b");
        assert!(b.undo());
        assert_eq!(b.text(), "> old", "one undo puts the selected text back");
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
    fn paragraph_at_spans_the_block_around_the_caret() {
        let mut b = TextBuffer::new();
        b.insert_str("first line\nsecond line\n\nnext para\n");
        // A caret anywhere in the first block returns both its lines, no trailing newline.
        let (s, e) = b.paragraph_at(3);
        assert_eq!(b.rope().byte_slice(s..e), "first line\nsecond line");
        // A caret in the second block returns just it, bounded by the blank line.
        let off = b.text().find("next").unwrap();
        let (s2, e2) = b.paragraph_at(off);
        assert_eq!(b.rope().byte_slice(s2..e2), "next para");
        // On the blank line between them, the empty paragraph is that line.
        let blank = b.text().find("\n\n").unwrap() + 1;
        let (s3, e3) = b.paragraph_at(blank);
        assert_eq!(s3, e3);
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
    fn selecting_a_heading_from_its_text_start_takes_the_marker() {
        // "# Title\nbody": "Title" is bytes 2..7, the content starting past "# ".
        let mut b = TextBuffer::new();
        b.insert_str("# Title\nbody");
        // Drag rightward from the heading's text start (anchor at 2, the content
        // byte a far-left click lands on).
        b.set_cursor(2, false);
        b.set_cursor(7, true);
        assert_eq!(b.selection(), Some((2, 7)));
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 7)));
        assert_eq!(b.selected_text().as_deref(), Some("# Title"));
    }

    #[test]
    fn selecting_a_heading_leftward_to_its_start_takes_the_marker() {
        let mut b = TextBuffer::new();
        b.insert_str("# Title\nbody");
        // Drag leftward: anchor past the text, cursor lands on the text start.
        b.set_cursor(7, false);
        b.set_cursor(2, true);
        assert_eq!(b.selection(), Some((2, 7)));
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 7)));
        // The cursor (the low end here) moved onto the line start.
        assert_eq!(b.cursor, 0);
    }

    #[test]
    fn deeper_heading_levels_include_their_full_marker() {
        // "## Head": marker is "## " (3 bytes), content starts at 3.
        let mut b = TextBuffer::new();
        b.insert_str("## Head");
        b.set_cursor(3, false);
        b.set_cursor(7, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 7)));
        assert_eq!(b.selected_text().as_deref(), Some("## Head"));
    }

    #[test]
    fn selecting_a_heading_from_mid_text_keeps_the_marker_out() {
        let mut b = TextBuffer::new();
        b.insert_str("# Title");
        // Low end at byte 4, inside "Title": the drag did not start at the text
        // start, so the marker stays unselected.
        b.set_cursor(4, false);
        b.set_cursor(7, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((4, 7)));
    }

    #[test]
    fn selecting_a_bullet_from_its_text_start_keeps_the_marker_out() {
        // "- item": marker is "- " (2 bytes), content "item" at 2..6. A sweep from
        // the label's first letter copies the label, not the "- " before it.
        let mut b = TextBuffer::new();
        b.insert_str("- item");
        b.set_cursor(2, false);
        b.set_cursor(6, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((2, 6)));
        assert_eq!(b.selected_text().as_deref(), Some("item"));
    }

    #[test]
    fn selecting_a_bullet_from_between_dash_and_space_takes_the_marker() {
        // Low end at byte 1, between the "-" and its space (the item's second
        // position): still inside the marker, so it pulls the dash in.
        let mut b = TextBuffer::new();
        b.insert_str("- item");
        b.set_cursor(1, false);
        b.set_cursor(6, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 6)));
    }

    #[test]
    fn selecting_a_task_item_from_its_text_start_keeps_the_marker_out() {
        // "- [ ] do": marker is "- [ ] " (6 bytes), content "do" at 6..8.
        let mut b = TextBuffer::new();
        b.insert_str("- [ ] do");
        b.set_cursor(6, false);
        b.set_cursor(8, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((6, 8)));
        assert_eq!(b.selected_text().as_deref(), Some("do"));
    }

    #[test]
    fn selecting_a_task_item_from_inside_its_marker_takes_the_whole_marker() {
        // Low end at byte 5, the space between "]" and the label: past the bullet's
        // own two bytes, so the grab has to measure the whole "- [ ] ".
        let mut b = TextBuffer::new();
        b.insert_str("- [ ] do");
        b.set_cursor(5, false);
        b.set_cursor(8, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 8)));
        assert_eq!(b.selected_text().as_deref(), Some("- [ ] do"));
    }

    #[test]
    fn selecting_a_bullet_from_mid_text_keeps_the_marker_out() {
        let mut b = TextBuffer::new();
        b.insert_str("- item");
        // Low end at byte 3, inside "item" past the marker: the marker stays out.
        b.set_cursor(3, false);
        b.set_cursor(6, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((3, 6)));
    }

    #[test]
    fn marker_inclusion_leaves_body_and_empty_selections_alone() {
        // A body line has no marker to pull in.
        let mut b = TextBuffer::new();
        b.insert_str("plain line");
        b.set_cursor(0, false);
        b.set_cursor(5, true);
        b.include_block_marker();
        assert_eq!(b.selection(), Some((0, 5)));
        // A bare cursor (no selection) is untouched.
        let mut c = TextBuffer::new();
        c.insert_str("# Title");
        c.set_cursor(2, false);
        c.include_block_marker();
        assert_eq!(c.selection(), None);
        assert_eq!(c.cursor, 2);
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
    fn double_click_separator_run_stops_at_the_line_break() {
        // A separator run a double-click selects is bounded by its line. In
        // "a \n b" the space, '\n', and space are one run by class, but clicking
        // the '\n' (offset 2) takes only what is on that line: the space before it.
        let mut b = TextBuffer::from_text("a \n b".to_string());
        b.select_word_at(2);
        assert_eq!(b.selection(), Some((1, 2)), "the space, not the run below");
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
    fn double_click_on_a_rule_takes_only_its_dashes() {
        // The reported bug: `---` is a separator run like the blank lines around
        // it, so an unbounded run swallowed both paragraphs. The click takes the
        // rule and nothing else, wherever on it it lands.
        let mut b = TextBuffer::from_text("above\n\n---\n\nbelow\n".to_string());
        for offset in 7..=10 {
            b.select_word_at(offset);
            assert_eq!(
                b.selected_text().as_deref(),
                Some("---"),
                "a click at {offset} takes the rule alone"
            );
        }
    }

    #[test]
    fn double_click_on_a_blank_line_selects_nothing() {
        // Nothing on the line to take, and the runs above and below are out of
        // reach, so the click is a plain caret.
        let mut b = TextBuffer::from_text("above\n\nbelow".to_string());
        b.select_word_at(6);
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor(), 6);
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
    fn tab_over_a_multi_line_selection_shifts_every_line_it_touches() {
        let mut b = TextBuffer::new();
        b.insert_str("- a\n- b\n- c");
        b.set_selection(2, 6); // inside "a" through inside "b", clear of "c"
        assert!(b.selection_spans_lines());
        assert!(b.indent_block(false));
        assert_eq!(b.text(), "  - a\n  - b\n- c");
        assert!(b.indent_block(true));
        assert_eq!(b.text(), "- a\n- b\n- c");
    }

    #[test]
    fn block_indent_uses_a_nesting_level_for_lists_and_a_soft_tab_elsewhere() {
        let mut b = TextBuffer::new();
        b.insert_str("- a\n  continued");
        b.set_selection(0, b.text().len());
        b.indent_block(false);
        assert_eq!(
            b.text(),
            "  - a\n    continued",
            "a block holding a list item moves exactly one nesting level"
        );

        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo");
        b.set_selection(0, b.text().len());
        b.indent_block(false);
        assert_eq!(b.text(), "    one\n    two");
    }

    #[test]
    fn block_indent_keeps_the_selection_over_the_same_text() {
        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo");
        b.set_selection(1, 5); // "ne\nt"
        b.indent_block(false);
        assert_eq!(b.text(), "    one\n    two");
        assert_eq!(b.selected_text().as_deref(), Some("ne\n    t"));
        // The selection still spans lines, so a second press shifts the same block.
        assert!(b.indent_block(false));
        assert_eq!(b.text(), "        one\n        two");
    }

    #[test]
    fn block_indent_stops_above_a_selection_ending_at_column_zero() {
        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo\nthree");
        b.set_selection(0, 8); // through the newline, onto the start of "three"
        b.indent_block(false);
        assert_eq!(b.text(), "    one\n    two\nthree");
    }

    #[test]
    fn block_indent_leaves_blank_lines_blank_and_outdents_a_partial_indent() {
        let mut b = TextBuffer::new();
        b.insert_str("one\n\ntwo");
        b.set_selection(0, b.text().len());
        b.indent_block(false);
        assert_eq!(
            b.text(),
            "    one\n\n    two",
            "no trailing space on the gap"
        );
        // Two of the four spaces are removed by hand, so the outdent takes what is
        // left of that line's indent and a full unit from the other.
        b.text.remove(0..2);
        b.set_selection(0, b.text().len());
        assert!(b.indent_block(true));
        assert_eq!(b.text(), "one\n\ntwo");
        // Nothing left to remove: no change, and no undo step for the no-op.
        let version = b.version;
        assert!(!b.indent_block(true));
        assert_eq!(b.version, version);
    }

    #[test]
    fn block_indent_is_one_undo_step_and_leaves_no_cursor_mid_indent() {
        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo");
        b.set_selection(0, b.text().len());
        b.indent_block(false);
        b.set_selection(6, 9); // both ends inside the second line's new indent
        assert!(b.indent_block(true));
        assert_eq!(b.text(), "one\ntwo");
        assert_eq!(b.cursor(), 4, "clamped to the line start, not into 'two'");
        // The whole block came back in a single undo.
        assert!(b.undo());
        assert_eq!(b.text(), "    one\n    two");
    }

    #[test]
    fn a_selection_within_one_line_is_not_a_block() {
        let mut b = TextBuffer::new();
        b.insert_str("one\ntwo");
        b.set_selection(4, 6);
        assert!(!b.selection_spans_lines());
        // A collapsed selection is a bare cursor, which spans no lines either.
        b.set_selection(4, 4);
        assert!(!b.selection_spans_lines());
    }

    #[test]
    fn shift_tab_outdents_the_cursor_line_with_no_selection() {
        let mut b = TextBuffer::new();
        b.insert_str("      indented"); // six spaces, so one unit is left over
        assert!(b.indent_block(true));
        assert_eq!(
            b.text(),
            "  indented",
            "one soft tab off, not the whole run"
        );
        assert_eq!(
            b.cursor(),
            "  indented".len(),
            "the cursor rides the outdent"
        );
        assert_eq!(b.selection(), None, "outdenting a line selects nothing");
        // The stub of an indent goes on the next press, then there is nothing left.
        assert!(b.indent_block(true));
        assert_eq!(b.text(), "indented");
        let version = b.version;
        assert!(!b.indent_block(true));
        assert_eq!(b.version, version, "a no-op outdent is not an undo step");
    }

    #[test]
    fn outdenting_a_line_clamps_a_cursor_sitting_in_its_indent() {
        let mut b = TextBuffer::new();
        b.insert_str("    indented");
        b.set_cursor(2, false); // between the indent's spaces
        assert!(b.indent_block(true));
        assert_eq!(b.text(), "indented");
        assert_eq!(b.cursor(), 0, "clamped to the line start");
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
    fn a_drag_from_a_caret_anchor_selects_from_the_press_point() {
        let mut b = TextBuffer::from_text("hello world".to_string());
        let press = (5, 5);
        // Forward and back over the press point at byte 5.
        assert!(b.drag_to(press, 11, DragGrain::Char));
        assert_eq!(b.selection(), Some((5, 11)));
        assert!(b.drag_to(press, 0, DragGrain::Char));
        assert_eq!(b.selection(), Some((0, 5)));
        // Back onto the press point: a bare caret, no selection.
        assert!(b.drag_to(press, 5, DragGrain::Char));
        assert_eq!(b.selection(), None);
        assert_eq!(b.cursor(), 5);
        // A drag that lands where it already is changes nothing.
        assert!(!b.drag_to(press, 5, DragGrain::Char));
    }

    #[test]
    fn a_drag_from_a_range_anchor_keeps_that_range_selected() {
        let mut b = TextBuffer::from_text("above\n---\nbelow".to_string());
        let rule = (6, 9); // the "---" of the thematic break
        b.select_range(rule.0, rule.1);
        // Down: the rule plus the text the drag sweeps into.
        assert!(b.drag_to(rule, 15, DragGrain::Char));
        assert_eq!(b.selection(), Some((6, 15)));
        // Up: the rule stays whole, and the selection grows from its far edge.
        assert!(b.drag_to(rule, 0, DragGrain::Char));
        assert_eq!(b.selection(), Some((0, 9)));
        // Wandering back inside the rule selects it whole again, so a jitter during
        // the click cannot cut the anchored element in half.
        assert!(b.drag_to(rule, 7, DragGrain::Char));
        assert_eq!(b.selection(), Some((6, 9)));
    }

    #[test]
    fn a_word_grained_drag_snaps_both_ends_to_whole_words() {
        let mut b = TextBuffer::from_text("alpha beta gamma delta".to_string());
        // The double-click takes "beta"; the drag then extends by whole words.
        let word = b.select_word_at(7);
        assert_eq!(word, (6, 10));
        // A pointer part-way into "gamma" takes all of it, not the letters swept.
        assert!(b.drag_to(word, 13, DragGrain::Word));
        assert_eq!(b.selected_text().as_deref(), Some("beta gamma"));
        // On into "delta".
        assert!(b.drag_to(word, 18, DragGrain::Word));
        assert_eq!(b.selected_text().as_deref(), Some("beta gamma delta"));
        // Backwards past the press: "alpha" comes in whole and the anchor word stays.
        assert!(b.drag_to(word, 2, DragGrain::Word));
        assert_eq!(b.selected_text().as_deref(), Some("alpha beta"));
        // Back onto the pressed word: just it again.
        assert!(b.drag_to(word, 8, DragGrain::Word));
        assert_eq!(b.selected_text().as_deref(), Some("beta"));
    }

    #[test]
    fn a_line_grained_drag_takes_whole_lines() {
        let mut b = TextBuffer::from_text("first\nsecond\nthird\n".to_string());
        let line = b.select_line_at(8); // inside "second"
        assert_eq!(b.selected_text().as_deref(), Some("second"));
        // Down into "third": the whole line comes, however far in the pointer sits.
        assert!(b.drag_to(line, 14, DragGrain::Line));
        assert_eq!(b.selected_text().as_deref(), Some("second\nthird"));
        // Up into "first": the pressed line stays whole.
        assert!(b.drag_to(line, 2, DragGrain::Line));
        assert_eq!(b.selected_text().as_deref(), Some("first\nsecond"));
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
