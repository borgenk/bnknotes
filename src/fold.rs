//! Collapsible heading sections: which headings are folded, and what that hides.
//!
//! A heading owns every line below it up to the next heading at the same or a
//! shallower level, so folding one hides its body and everything nested under it
//! and stops at its first sibling. The heading line itself always stays visible.
//!
//!     # Guide          owns lines 1..6, down to EOF
//!     intro            <- hidden by folding # Guide
//!     ## Setup         owns lines 3..5, stopped by ## Usage
//!     install          <- hidden by folding # Guide or ## Setup
//!     ### Notes
//!     detail           <- hidden by any of the three
//!     ## Usage         a sibling of ## Setup, so it ends that section
//!
//! Two types split the state from what it implies. [`Folds`] is the durable set,
//! the heading line numbers a document has folded; it rides the editor pane and is
//! persisted per tab beside the caret and scroll offset. [`FoldMap`] is derived: it
//! resolves those anchors against a [`LineIndex`] into the line spans actually
//! hidden and the pixel height that removes, and is rebuilt whenever the content or
//! the anchor set changes.
//!
//! Folding never touches the base geometry. The index measures a document as if
//! nothing were folded, so it stays a pure function of content and width, and the
//! map is applied as a query-time overlay on top (see [`crate::layout::Adjust`]).
//! Toggling a fold therefore costs the span scan below, not a re-measure of the
//! document.

use std::collections::BTreeSet;

use crate::layout::LineIndex;
use crate::rope::Rope;

/// Cap on remembered folds per document. Bounds the anchor set an untrusted state
/// file can ask for, and the work [`FoldMap::build`] does per rebuild.
pub const MAX_FOLDS: usize = 4096;

/// The headings a document has folded, as the logical line numbers of their
/// heading lines. Anchors survive editing (they shift with the text) and a session
/// (they are written to the state file), but they are only ever advisory: one that
/// no longer lands on a heading is dropped by [`Self::retain_headings`], and one
/// whose section is empty simply hides nothing.
///
/// `revision` changes on every mutation, so a [`FoldMap`] cache keyed on it
/// rebuilds. Equality is not derived: two sets with the same anchors but different
/// revisions are the same folds, and comparing revisions would be a trap.
#[derive(Clone, Debug, Default)]
pub struct Folds {
    lines: BTreeSet<usize>,
    revision: u64,
}

impl Folds {
    /// The folds restored from a session, capped and deduped by the set itself.
    /// Anchors are validated against the document later, by
    /// [`retain_headings`](Self::retain_headings), once its index exists.
    pub fn from_lines(lines: impl IntoIterator<Item = usize>) -> Self {
        Self {
            lines: lines.into_iter().take(MAX_FOLDS).collect(),
            revision: 0,
        }
    }

    /// The folded heading lines in ascending order, for persisting to the session.
    pub fn lines(&self) -> impl Iterator<Item = usize> + '_ {
        self.lines.iter().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Changes whenever the anchor set does. A [`FoldMap`] built from this set is
    /// current exactly while the revision it was built at still matches.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Fold or unfold the heading on `line`. Returns whether anything changed,
    /// which is false only when folding a set already at [`MAX_FOLDS`].
    pub fn toggle(&mut self, line: usize) -> bool {
        if self.lines.remove(&line) {
            self.revision += 1;
            return true;
        }
        if self.lines.len() >= MAX_FOLDS {
            return false;
        }
        self.lines.insert(line);
        self.revision += 1;
        true
    }

    /// Unfold everything. Returns whether anything was folded.
    pub fn clear(&mut self) -> bool {
        if self.lines.is_empty() {
            return false;
        }
        self.lines.clear();
        self.revision += 1;
        true
    }

    /// Shift the anchors across an edit that turned `prev` into `new`, so a fold
    /// stays on its heading when text above it grows or shrinks.
    ///
    /// The edit's untouched tail comes from the same byte-cheap rope diff
    /// [`LineIndex::update`] uses (matching prefix from the front, matching suffix
    /// from the back). An anchor whose heading line begins at or after where that
    /// tail starts sits wholly inside it, so it moved by exactly the line delta.
    /// Any other anchor keeps its line: one before the edit never moved, and the
    /// edited line itself is held for [`retain_headings`](Self::retain_headings) to
    /// judge, so editing a folded heading's own text does not pop it open while
    /// deleting the heading outright drops the fold.
    ///
    /// The start-byte test is what makes an insert or delete *directly above* a
    /// heading ride correctly: the heading's line is byte-identical in the tail even
    /// though the ambiguous boundary newline is split between the two sides of the
    /// diff, so a line-boundary guess would misplace it but its start byte does not.
    pub fn remap(&mut self, prev: &Rope, new: &Rope) {
        if self.lines.is_empty() {
            return;
        }
        let delta = new.len_lines() as isize - prev.len_lines() as isize;
        if delta == 0 {
            return;
        }
        let prefix = new.common_prefix_len(prev);
        let suffix = new.common_suffix_len(prev, new.len().min(prev.len()) - prefix);
        // Where the unchanged tail begins in the old text. A heading line starting
        // here or later is wholly inside it and rode the edit down (or up) by delta.
        let tail_start = prev.len() - suffix;
        let new_n = new.len_lines();
        let shifted: BTreeSet<usize> = self
            .lines
            .iter()
            .map(|&line| {
                if prev.line_to_byte(line) >= tail_start {
                    line.saturating_add_signed(delta)
                } else {
                    line
                }
            })
            .filter(|&line| line < new_n)
            .collect();
        if shifted != self.lines {
            self.lines = shifted;
            self.revision += 1;
        }
    }

    /// Drop every anchor that no longer lands on a heading, so a fold does not
    /// linger on a line whose `#` was deleted (or that scrolled under a code fence).
    pub fn retain_headings(&mut self, index: &LineIndex) {
        let before = self.lines.len();
        self.lines.retain(|&line| index.heading_at(line).is_some());
        if self.lines.len() != before {
            self.revision += 1;
        }
    }
}

/// One hidden run of logical lines: a folded heading's section, having swallowed
/// any nested section folded inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HiddenSpan {
    /// First hidden line, always the line after the folded heading.
    start: usize,
    /// One past the last hidden line.
    end: usize,
    /// Base pixel height of the lines it hides, removed from the document while
    /// the fold holds.
    height: i32,
    /// Height hidden by every earlier span, so a query resolves in one binary
    /// search instead of a walk.
    before: i32,
    /// The cross-line carry entering `end` (packed fence, list depth), so a
    /// windowed layout that jumps this span resumes in the state the skipped lines
    /// would have left it. Without it a fence opened inside a fold would leak.
    exit: (u16, i16),
}

/// What a document's [`Folds`] hide right now, resolved against its [`LineIndex`]:
/// the merged line spans and the height they remove. Empty when nothing is folded,
/// which is the common case and costs nothing to build or consult.
#[derive(Clone, Debug, Default)]
pub struct FoldMap {
    /// Disjoint and ascending by `start`.
    hidden: Vec<HiddenSpan>,
}

impl FoldMap {
    /// Nothing folded, for the measurement paths that want fold-free base geometry
    /// and for callers with no fold state of their own.
    pub const NONE: &'static FoldMap = &FoldMap { hidden: Vec::new() };

    /// Resolve `folds` against `index` into the spans they hide.
    ///
    /// Walks the anchors in ascending order, scanning forward from each for the
    /// heading that ends its section. An anchor already covered by the span in hand
    /// is skipped, which is what merges a nested fold into the outer one that
    /// swallows it. The scan is the only cost, so it is proportional to the lines
    /// actually hidden and is skipped entirely when nothing is folded.
    pub fn build(folds: &Folds, index: &LineIndex) -> Self {
        if folds.is_empty() {
            return Self::default();
        }
        let n = index.len();
        let mut hidden: Vec<HiddenSpan> = Vec::new();
        let mut before = 0;
        for anchor in folds.lines() {
            let Some(level) = index.heading_at(anchor) else {
                continue;
            };
            // Anchors ascend, so only the last span can cover this one.
            if hidden.last().is_some_and(|s| anchor < s.end) {
                continue;
            }
            let start = anchor + 1;
            let mut end = start;
            while end < n && !index.heading_at(end).is_some_and(|l| l <= level) {
                end += 1;
            }
            if end == start {
                // The section is empty (the next heading is a sibling, or the
                // heading is the last line), so there is nothing to hide.
                continue;
            }
            let height = index.span_height(start, end);
            hidden.push(HiddenSpan {
                start,
                end,
                height,
                before,
                exit: index.entry_packed(end),
            });
            before += height;
        }
        Self { hidden }
    }

    pub fn is_empty(&self) -> bool {
        self.hidden.is_empty()
    }

    /// Whether `line` is hidden by a fold.
    pub fn hides(&self, line: usize) -> bool {
        self.span_containing(line).is_some()
    }

    /// Whether the heading on `line` is folded, that is, whether it is the anchor
    /// of a span that hides something.
    pub fn is_folded(&self, line: usize) -> bool {
        self.hidden
            .binary_search_by_key(&(line + 1), |s| s.start)
            .is_ok()
    }

    /// The end of the span hiding `line`, with the carry state entering it, for a
    /// layout walk that needs to jump the whole run at once.
    pub fn skip_from(&self, line: usize) -> Option<(usize, (u16, i16))> {
        self.span_containing(line).map(|s| (s.end, s.exit))
    }

    /// The first visible line at or after `line`: `line` itself when it is not
    /// hidden, else the line just past the span hiding it (which may be the line
    /// count, when the fold runs to EOF).
    pub fn first_visible_at_or_after(&self, line: usize) -> usize {
        self.span_containing(line).map_or(line, |s| s.end)
    }

    /// The last visible line at or before `line`: `line` itself when it is not
    /// hidden, else the folded heading whose section hides it.
    pub fn first_visible_at_or_before(&self, line: usize) -> usize {
        self.span_containing(line)
            .map_or(line, |s| s.start.saturating_sub(1))
    }

    /// Total hidden height, removed from the document's content height.
    pub fn hidden_total(&self) -> i32 {
        self.hidden.last().map_or(0, |s| s.before + s.height)
    }

    /// Height hidden strictly above `line` by spans that end at or before it.
    /// Callers resolving a line inside a span want [`Self::collapsed_top`] instead.
    pub fn hidden_before(&self, line: usize) -> i32 {
        match self.hidden.binary_search_by(|s| s.end.cmp(&line)) {
            // A span ending exactly at `line` is entirely above it.
            Ok(i) => self.hidden[i].before + self.hidden[i].height,
            Err(0) => 0,
            Err(i) => {
                let s = self.hidden[i - 1];
                s.before + s.height
            }
        }
    }

    /// Where a hidden `line` collapses to, as the span's first line paired with the
    /// height hidden above that. Every line in a span shares one position (the
    /// span's own top), so the caller resolves them all the same way.
    pub fn collapsed_top(&self, line: usize) -> Option<(usize, i32)> {
        self.span_containing(line).map(|s| (s.start, s.before))
    }

    fn span_containing(&self, line: usize) -> Option<HiddenSpan> {
        // The last span starting at or before `line`, kept only if it reaches past it.
        let i = match self.hidden.binary_search_by(|s| s.start.cmp(&line)) {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };
        self.hidden.get(i).copied().filter(|s| line < s.end)
    }
}

/// The heading whose section holds `line`: `line` itself when it is a heading,
/// else the nearest heading above it. `None` above the document's first heading,
/// where there is no section to fold.
///
/// The nearest heading above a line always owns it: a section runs to the next
/// heading at the same or a shallower level, and there is no heading at all in
/// between.
pub fn enclosing_heading(index: &LineIndex, line: usize) -> Option<usize> {
    let last = index.len().checked_sub(1)?;
    (0..=line.min(last))
        .rev()
        .find(|&l| index.heading_at(l).is_some())
}

/// Whether a heading of `level` owns at least one line, and so has a section to
/// collapse, given the byte offset the line below it starts at. False when that
/// line is already a sibling or a shallower heading. The caller only asks about a
/// heading that has a line below it at all.
///
/// This runs for every heading of every layout build, so it reads only the marker:
/// one to six leading hashes and the byte after them. It never materializes the
/// line or forms a text slice whose fixed end could bisect a UTF-8 character.
///
/// A heading closes any open code fence by definition (a `#` inside a fence is
/// code, not a heading), so the line below classifies on its own without carrying
/// parser state into the question.
pub fn owns_section(rope: &Rope, next_start: usize, level: u8) -> bool {
    !matches!(
        heading_level_at(rope, next_start),
        Some(next_level) if next_level <= level
    )
}

/// The ATX heading level at `start`, using the same marker rule as markdown:
/// one to six hashes followed by a space, a newline, or the document end.
fn heading_level_at(rope: &Rope, start: usize) -> Option<u8> {
    let mut hashes = 0;
    while rope.byte(start.saturating_add(hashes)) == Some(b'#') {
        hashes += 1;
        if hashes > 6 {
            return None;
        }
    }
    if hashes == 0 {
        return None;
    }
    match rope.byte(start.saturating_add(hashes)) {
        None | Some(b'\n' | b' ') => Some(hashes as u8),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown;
    use crate::platform::freetype::Fonts;
    use crate::render::image::ImageStore;
    use crate::scale::Scale;

    const BODY: u32 = 32;

    /// A document's index, the input every fold resolution is measured against.
    fn index(text: &str) -> (Rope, LineIndex) {
        let rope = Rope::from_str(text);
        let fonts = Fonts::new(&markdown::all_sizes(BODY)).expect("a default font");
        let images = ImageStore::new(None);
        let idx = LineIndex::build(&fonts, &rope, 1_000_000, BODY, &images, Scale::ONE);
        (rope, idx)
    }

    /// The nested document the section rules are easiest to read against.
    const NESTED: &str = "\
# Guide
intro
## Setup
install
### Notes
detail
## Usage
run
";

    fn folded(text: &str, anchors: &[usize]) -> FoldMap {
        let (_, idx) = index(text);
        FoldMap::build(&Folds::from_lines(anchors.iter().copied()), &idx)
    }

    /// Which lines a fold map hides, as a list, so a failure names the lines.
    fn hidden_lines(map: &FoldMap, n: usize) -> Vec<usize> {
        (0..n).filter(|&l| map.hides(l)).collect()
    }

    #[test]
    fn a_section_stops_at_the_next_heading_of_the_same_level() {
        // `## Setup` owns its body and the deeper `### Notes` beneath it, and stops
        // at its sibling `## Usage`. The heading line itself stays visible.
        let map = folded(NESTED, &[2]);
        assert_eq!(hidden_lines(&map, 8), vec![3, 4, 5]);
        assert!(map.is_folded(2));
    }

    #[test]
    fn a_shallower_section_swallows_everything_under_it() {
        let map = folded(NESTED, &[0]);
        assert_eq!(hidden_lines(&map, 8), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn the_last_section_runs_to_the_end_of_the_document() {
        let map = folded(NESTED, &[6]);
        assert_eq!(hidden_lines(&map, 8), vec![7]);
    }

    #[test]
    fn a_nested_fold_merges_into_the_outer_one_that_covers_it() {
        // Folding both `# Guide` and `## Setup` hides exactly what `# Guide` alone
        // hides: the inner span is contained, not counted twice (which would
        // subtract its height from the document's geometry twice over).
        let outer = folded(NESTED, &[0]);
        let both = folded(NESTED, &[0, 2]);
        assert_eq!(hidden_lines(&both, 8), hidden_lines(&outer, 8));
        assert_eq!(both.hidden_total(), outer.hidden_total());
    }

    #[test]
    fn a_heading_with_no_lines_under_it_folds_nothing() {
        // `## Empty` is immediately followed by a sibling, so it owns no lines.
        let text = "# Top\n## Empty\n## Next\nbody\n";
        let map = folded(text, &[1]);
        assert!(hidden_lines(&map, 4).is_empty());
        // With no span there is no chevron to draw and nothing to click.
        assert!(!map.is_folded(1));
        let rope = Rope::from_str(text);
        // `owns_section` is asked about the line below, by its byte offset.
        assert!(!owns_section(&rope, rope.line_to_byte(2), 2));
        assert!(owns_section(&rope, rope.line_to_byte(3), 2));
    }

    #[test]
    fn a_bare_hash_below_a_heading_is_read_as_the_heading_it_is() {
        // `owns_section` peeks at a fixed window of bytes rather than the line, so
        // it has to stop at the newline: a lone `#` is a level-1 heading, and
        // reading on into the line below would leave it unclassified and wrongly
        // report a section to fold.
        let rope = Rope::from_str("# Top\n#\nbody\n");
        assert!(
            !owns_section(&rope, rope.line_to_byte(1), 1),
            "the `#` below is a sibling, so `# Top` owns nothing"
        );
        // A bare marker one level deeper is nested, so the section runs on through
        // it. Same short-line window, opposite answer.
        let deeper = Rope::from_str("# Top\n##\nbody\n");
        assert!(
            owns_section(&deeper, deeper.line_to_byte(1), 1),
            "a bare `##` nests under the `#` rather than ending it"
        );
    }

    #[test]
    fn a_heading_longer_than_the_peek_window_still_classifies() {
        // The window is cut short of the text, which is fine: only the marker run
        // and the byte after it decide the level.
        let rope = Rope::from_str("# Top\n###### a very long heading indeed\nbody\n");
        assert!(
            owns_section(&rope, rope.line_to_byte(1), 1),
            "an H6 below an H1 is nested, so the H1 owns it"
        );
        assert!(
            !owns_section(&rope, rope.line_to_byte(1), 6),
            "an H6 below an H6 is a sibling"
        );
    }

    #[test]
    fn a_unicode_byte_at_the_marker_peek_boundary_is_safe() {
        let body = Rope::from_str("# Top\nabcdefæ body\n");
        assert!(
            owns_section(&body, body.line_to_byte(1), 1),
            "ordinary Unicode text belongs to the section"
        );

        let sibling = Rope::from_str("# Top\n# café\n");
        assert!(
            !owns_section(&sibling, sibling.line_to_byte(1), 1),
            "Unicode heading text does not change the sibling level"
        );

        let tag = Rope::from_str("# Top\n#æ\n");
        assert!(
            owns_section(&tag, tag.line_to_byte(1), 1),
            "a hash without a separating space is not a heading"
        );
    }

    #[test]
    fn a_hash_inside_a_code_fence_is_not_a_heading() {
        // The index classifies with the fence carry in hand, so the `#` on line 2 is
        // code. It therefore neither ends `# A`'s section nor offers a fold of its
        // own, and `# A` runs on through the deeper `## B` to the end.
        let text = "# A\n```rust\n# not a heading\n```\nbody\n## B\ntail\n";
        let (_, idx) = index(text);
        assert_eq!(idx.heading_at(0), Some(1));
        assert_eq!(idx.heading_at(2), None, "a `#` inside a fence is code");
        assert_eq!(idx.heading_at(5), Some(2));
        let map = folded(text, &[0]);
        assert_eq!(hidden_lines(&map, 7), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn enclosing_heading_finds_the_section_a_line_sits_in() {
        let (_, idx) = index(NESTED);
        // A heading is its own section.
        assert_eq!(enclosing_heading(&idx, 2), Some(2));
        // A body line belongs to the nearest heading above it.
        assert_eq!(enclosing_heading(&idx, 5), Some(4));
        assert_eq!(enclosing_heading(&idx, 7), Some(6));
        // Above the first heading there is no section to fold.
        let (_, plain) = index("intro\nmore\n# Later\n");
        assert_eq!(enclosing_heading(&plain, 1), None);
    }

    #[test]
    fn hidden_height_accounts_for_every_line_in_the_span() {
        // The height a fold removes is exactly the base height of the lines it
        // hides, which is what keeps the document's content height honest.
        let (_, idx) = index(NESTED);
        let map = FoldMap::build(&Folds::from_lines([2]), &idx);
        assert_eq!(map.hidden_total(), idx.span_height(3, 6));
        // Nothing above the span is affected; everything below shifts up by it.
        assert_eq!(map.hidden_before(3), 0);
        assert_eq!(map.hidden_before(6), map.hidden_total());
    }

    #[test]
    fn an_anchor_rides_its_heading_when_lines_are_inserted_above() {
        let prev = Rope::from_str(NESTED);
        let new = Rope::from_str(&format!("added\nlines\n{NESTED}"));
        let mut folds = Folds::from_lines([2, 6]);
        folds.remap(&prev, &new);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![4, 8]);
    }

    #[test]
    fn an_anchor_rides_its_heading_when_lines_are_deleted_above() {
        let prev = Rope::from_str(&format!("added\nlines\n{NESTED}"));
        let new = Rope::from_str(NESTED);
        let mut folds = Folds::from_lines([4, 8]);
        folds.remap(&prev, &new);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![2, 6]);
    }

    #[test]
    fn a_fold_survives_a_line_inserted_directly_above_its_heading() {
        // The reported bug: inserting a blank line right above a folded heading used
        // to leave the anchor unshifted (the boundary newline is split across the
        // diff), so `retain_headings` then saw the old line gone blank and dropped
        // the fold, popping it open. The anchor must ride the heading down instead.
        let new_text = NESTED.replace("intro\n## Setup", "intro\n\n## Setup");
        let prev = Rope::from_str(NESTED);
        let new = Rope::from_str(&new_text);
        let mut folds = Folds::from_lines([2, 6]);
        folds.remap(&prev, &new);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![3, 7]);
        // And they still land on their headings, so the fold holds.
        let (_, idx) = index(&new_text);
        folds.retain_headings(&idx);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![3, 7]);
    }

    #[test]
    fn a_fold_survives_a_line_deleted_directly_above_its_heading() {
        // The mirror case: deleting the line just above a folded heading pulls it up
        // by one, and the fold rides with it rather than being dropped.
        let new_text = NESTED.replace("intro\n## Setup", "## Setup");
        let prev = Rope::from_str(NESTED);
        let new = Rope::from_str(&new_text);
        let mut folds = Folds::from_lines([2, 6]);
        folds.remap(&prev, &new);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![1, 5]);
        let (_, idx) = index(&new_text);
        folds.retain_headings(&idx);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![1, 5]);
    }

    #[test]
    fn editing_a_folded_headings_own_text_keeps_the_fold() {
        // Typing in the heading itself must not pop the section open; the anchor is
        // inside the changed range, so it holds its line and stays a heading.
        let prev = Rope::from_str(NESTED);
        let new = Rope::from_str(&NESTED.replace("## Setup", "## Setup and teardown"));
        let mut folds = Folds::from_lines([2]);
        folds.remap(&prev, &new);
        let (_, idx) = index(&NESTED.replace("## Setup", "## Setup and teardown"));
        folds.retain_headings(&idx);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![2]);
    }

    #[test]
    fn an_anchor_whose_heading_is_gone_is_dropped() {
        // The `##` is deleted, so line 2 is ordinary prose and holds no fold.
        let text = NESTED.replace("## Setup", "Setup");
        let (_, idx) = index(&text);
        let mut folds = Folds::from_lines([2, 6]);
        folds.retain_headings(&idx);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![6]);
    }

    #[test]
    fn an_anchor_past_the_end_of_a_shortened_document_is_dropped() {
        let (_, idx) = index("# Only\nbody\n");
        let mut folds = Folds::from_lines([0, 99]);
        folds.retain_headings(&idx);
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn toggling_folds_and_unfolds_and_moves_the_revision() {
        let mut folds = Folds::default();
        let start = folds.revision();
        assert!(folds.toggle(4));
        assert_eq!(folds.lines().collect::<Vec<_>>(), vec![4]);
        assert_ne!(
            folds.revision(),
            start,
            "a cache keyed on this must rebuild"
        );
        let after_fold = folds.revision();
        assert!(folds.toggle(4));
        assert!(folds.is_empty());
        assert_ne!(folds.revision(), after_fold);
        // Clearing an already-empty set changes nothing, so nothing rebuilds.
        let idle = folds.revision();
        assert!(!folds.clear());
        assert_eq!(folds.revision(), idle);
    }

    #[test]
    fn the_anchor_set_is_capped() {
        // A state file cannot make a document carry an unbounded fold set, and a
        // capped set refuses to grow rather than evicting silently.
        let mut folds = Folds::from_lines(0..MAX_FOLDS * 2);
        assert_eq!(folds.lines().count(), MAX_FOLDS);
        assert!(!folds.toggle(usize::MAX), "a full set takes no more folds");
        // Unfolding still works at the cap, and frees a slot.
        assert!(folds.toggle(0));
        assert!(folds.toggle(usize::MAX));
    }

    #[test]
    fn the_edges_of_a_span_resolve_to_the_visible_lines_around_it() {
        // How the caret escapes a section that just closed under it.
        let map = folded(NESTED, &[2]);
        assert_eq!(map.first_visible_at_or_before(4), 2, "back to the heading");
        assert_eq!(map.first_visible_at_or_after(4), 6, "forward past the span");
        // A visible line resolves to itself in both directions.
        assert_eq!(map.first_visible_at_or_before(7), 7);
        assert_eq!(map.first_visible_at_or_after(7), 7);
    }
}
