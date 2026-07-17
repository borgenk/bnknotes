//! The palette: a searchable list of places to go, opened with Ctrl+K.
//!
//! One field and a ranked list. It lists the rail's vaults today, and the rows are
//! the only thing that says what it lists, so a second source (a vault's files, a
//! search across them) is a list for the app to build rather than a widget to
//! write. Editing stays on the keys that already do it: nothing here saves, finds,
//! or types into a document.
//!
//! The app hands in the rows when it opens the palette, and they stand for as long
//! as it is up. That holds because the palette takes the keyboard while it is open,
//! the way the find bar does, so nothing it lists can move under it. Ranking is
//! pure, which is why this module tests against plain strings and never sees the
//! app.
//!
//! [`Palette::layout`] is the geometry both the painter and the pointer read, the
//! way [`crate::menu`] does it, so a row is clicked exactly where it is drawn and a
//! window resized under an open palette has no stale rectangle.

use std::path::PathBuf;

use crate::platform::freetype::Fonts;
use crate::platform::geom::Rect;
use crate::scale::Scale;
use crate::theme::{
    PALETTE_BORDER_W, PALETTE_FIELD_VPAD, PALETTE_PAD, PALETTE_ROW_VPAD, PALETTE_TOP, PALETTE_WIDTH,
};

/// Rows drawn at once. A longer list scrolls under the selection rather than
/// growing the panel down past the window.
const MAX_VISIBLE: usize = 12;

/// How a palette's rows answer the query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The rows stand for as long as the palette is open and the query ranks them
    /// (the vaults). Filtering is this module's job.
    Ranked,
    /// The rows are the query's answer, so they are shown in the order they came in
    /// and never re-ranked (the search across a vault, where the scan did the
    /// matching). Ranking them again would score a hit's own path against the query
    /// it already matched, floating a note whose name happens to contain the word
    /// above the lines that actually hold it. The app re-supplies them through
    /// [`Palette::set_rows`] as the query changes.
    AsGiven,
}

/// What a palette row does when it is activated. Like [`crate::menu::MenuAction`],
/// the action names its subject, so a row carries everything needed to act on it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaletteAction {
    /// Root the window at rail vault `i`.
    SwitchVault(usize),
    /// Open `path` and put the caret on the match at `offset`.
    OpenMatch { path: PathBuf, offset: usize },
}

/// One row: what it says, and what it does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaletteRow {
    pub label: String,
    pub action: PaletteAction,
}

/// The byte ranges where `query` occurs in `label`, ASCII case folded and never
/// overlapping, so the painter can mark what a row matched. Empty for an empty query
/// or one that does not occur, which lets the caller mark matches without a special
/// case for "nothing typed yet".
///
/// The fold is ASCII only, the same rule the search scans by, so a match aligns on a
/// character boundary (both sides are UTF-8, and only A-Z fold), and slicing `label`
/// at these ranges never splits one.
pub fn match_spans(label: &str, query: &str) -> Vec<(usize, usize)> {
    let (hay, needle) = (label.as_bytes(), query.as_bytes());
    let mut spans = Vec::new();
    if needle.is_empty() || hay.len() < needle.len() {
        return spans;
    }
    let mut at = 0;
    while at + needle.len() <= hay.len() {
        if hay[at..at + needle.len()].eq_ignore_ascii_case(needle) {
            spans.push((at, at + needle.len()));
            at += needle.len();
        } else {
            at += 1;
        }
    }
    spans
}

/// An open palette: its rows, the query typed against them, and which match Enter
/// takes. A palette with no rows to offer is never opened, so a live one always has
/// something to show.
#[derive(Clone, Debug)]
pub struct Palette {
    query: String,
    /// What the field says before anything is typed. It names what the rows are, so
    /// it comes from whoever built them.
    prompt: &'static str,
    /// Whether the query ranks the rows or the rows already answer it.
    mode: Rows,
    /// A word about the rows, drawn at the field's right the way the find bar's
    /// counter is: how many matches a search found, and whether it stopped counting.
    /// Empty when there is nothing to say, which is every ranked palette.
    note: String,
    rows: Vec<PaletteRow>,
    /// Positions into `rows`, best match first; rebuilt on every query change.
    filtered: Vec<usize>,
    /// Scratch for the ranking pass, kept across keystrokes so typing allocates
    /// nothing once the first query has grown it.
    scored: Vec<(usize, u8)>,
    /// Which of `filtered` Enter takes. The pointer moves it too, so a row is lit by
    /// one thing however it was picked, rather than a hover and a selection lighting
    /// two rows and neither saying which Enter would take.
    selected: usize,
    /// The first row drawn, kept so the selection is always among them.
    scroll: usize,
}

impl Palette {
    /// A palette over `rows`, in the order the caller wants them before a query ranks
    /// them, under a `prompt` naming what they are. `mode` says whether a query ranks
    /// the rows or is already their answer.
    pub fn new(prompt: &'static str, mode: Rows, rows: Vec<PaletteRow>) -> Self {
        let mut palette = Self {
            query: String::new(),
            prompt,
            mode,
            note: String::new(),
            filtered: Vec::with_capacity(rows.len()),
            scored: Vec::with_capacity(rows.len()),
            rows,
            selected: 0,
            scroll: 0,
        };
        palette.refilter();
        palette
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    pub fn prompt(&self) -> &'static str {
        self.prompt
    }

    pub fn mode(&self) -> Rows {
        self.mode
    }

    pub fn note(&self) -> &str {
        &self.note
    }

    /// Say something about the rows at the field's right, or nothing with an empty
    /// string. Set beside [`Self::set_rows`], by whoever knows what the rows cost.
    pub fn set_note(&mut self, note: String) {
        self.note = note;
    }

    /// Replace the rows with the query's latest answer, putting the selection back on
    /// the top one. Only an [`Rows::AsGiven`] palette has anything to replace: a
    /// ranked one's rows stand for as long as it is open.
    pub fn set_rows(&mut self, rows: Vec<PaletteRow>) {
        self.rows = rows;
        self.refilter();
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Type `c` into the query.
    pub fn push(&mut self, c: char) {
        self.query.push(c);
        self.refilter();
    }

    /// Type `s` into the query, as a paste does. Controls are dropped: the field is
    /// one line, and a pasted newline would sit in it matching nothing.
    pub fn push_str(&mut self, s: &str) {
        self.query.extend(s.chars().filter(|c| !c.is_control()));
        self.refilter();
    }

    /// Delete the query's last character.
    pub fn pop(&mut self) {
        self.query.pop();
        self.refilter();
    }

    /// Move the selection `delta` rows, clamped at both ends: nothing in this editor
    /// wraps, and the file tree's arrows already stop at the last row.
    pub fn move_selection(&mut self, delta: i32) {
        if self.filtered.is_empty() {
            return;
        }
        let last = self.filtered.len() as i64 - 1;
        self.selected = (self.selected as i64 + delta as i64).clamp(0, last) as usize;
        self.reveal();
    }

    /// Put the selection on `pos`, which is what the pointer moving onto a row does,
    /// reporting whether that changed anything (and so whether the frame has to be
    /// repainted). A position past the matches is no row, and leaves the selection be.
    pub fn select(&mut self, pos: usize) -> bool {
        if pos >= self.filtered.len() || self.selected == pos {
            return false;
        }
        self.selected = pos;
        self.reveal();
        true
    }

    /// The action of the row at `pos` in the filtered list, if there is one there.
    pub fn action_at(&self, pos: usize) -> Option<PaletteAction> {
        let &row = self.filtered.get(pos)?;
        self.rows.get(row).map(|r| r.action.clone())
    }

    /// The selected row's action, or `None` when the query matches nothing and Enter
    /// has nowhere to go.
    pub fn activate(&self) -> Option<PaletteAction> {
        self.action_at(self.selected)
    }

    /// How many rows are drawn: the matches from the scroll down, capped.
    pub fn visible_count(&self) -> usize {
        self.filtered
            .len()
            .saturating_sub(self.scroll)
            .min(MAX_VISIBLE)
    }

    /// The filtered position of the row drawn at `vis`, if one is drawn there. The
    /// painter counts rows from the first one drawn and the pointer answers in the
    /// same terms; everything else is keyed by position, which scrolling does not
    /// move.
    pub fn row_pos(&self, vis: usize) -> Option<usize> {
        let pos = self.scroll + vis;
        (pos < self.filtered.len()).then_some(pos)
    }

    /// The rows drawn, each with its position in the filtered list, which is what
    /// [`Self::action_at`] and the hover are keyed by.
    pub fn visible(&self) -> impl Iterator<Item = (usize, &PaletteRow)> {
        self.filtered
            .iter()
            .enumerate()
            .skip(self.scroll)
            .take(MAX_VISIBLE)
            .filter_map(|(pos, &row)| self.rows.get(row).map(|r| (pos, r)))
    }

    /// Scroll so the selection is among the rows drawn. Only a selection move can
    /// take it off them, and it moves one row at a time, so this steps rather than
    /// centers.
    fn reveal(&mut self) {
        if self.selected < self.scroll {
            self.scroll = self.selected;
        } else if self.selected >= self.scroll + MAX_VISIBLE {
            self.scroll = self.selected + 1 - MAX_VISIBLE;
        }
    }

    /// Rank the rows against the query, best first, and put the selection back on the
    /// top one. An empty query keeps every row in the order the caller built them, and
    /// so does every query when the rows are the answer to one already.
    fn refilter(&mut self) {
        self.filtered.clear();
        self.selected = 0;
        self.scroll = 0;
        if self.query.is_empty() || self.mode == Rows::AsGiven {
            self.filtered.extend(0..self.rows.len());
            return;
        }
        self.scored.clear();
        self.scored.extend(
            self.rows
                .iter()
                .enumerate()
                .filter_map(|(i, row)| score(&row.label, &self.query).map(|s| (i, s))),
        );
        // A stable sort leaves the caller's order standing inside each tier.
        self.scored.sort_by_key(|&(_, tier)| tier);
        self.filtered.extend(self.scored.iter().map(|&(i, _)| i));
    }

    /// Where the palette lands in a `window`-sized window: a panel below the window's
    /// top edge, centered, [`PALETTE_WIDTH`] wide and narrower in a window too small
    /// to hold that.
    pub fn layout(
        &self,
        fonts: &Fonts,
        body_size: u32,
        scale: Scale,
        window: (i32, i32),
    ) -> PaletteLayout {
        let m = fonts.metrics(body_size);
        let pad = scale.px(PALETTE_PAD);
        let text_h = m.ascent + m.descent;
        let field_h = text_h + 2 * scale.px(PALETTE_FIELD_VPAD);
        let row_h = text_h + 2 * scale.px(PALETTE_ROW_VPAD);

        // A query matching nothing draws the field alone, so the rule that divides it
        // from the rows goes too: there is nothing below it to divide from.
        let count = self.visible_count();
        let rule_h = if count == 0 {
            0
        } else {
            scale.px(PALETTE_BORDER_W).max(1)
        };
        let w = scale.px(PALETTE_WIDTH).min(window.0 - 2 * pad).max(0);
        let h = 2 * pad + field_h + rule_h + count as i32 * row_h;
        // Centered across, and pushed up out of a window too short to hold the panel
        // where it wants to sit; one too short to hold it at all pins it to the top,
        // where the field is at least reachable.
        let x = ((window.0 - w) / 2).max(0);
        let y = scale.px(PALETTE_TOP).min(window.1 - h).max(0);

        let inner_w = (w - 2 * pad).max(0);
        let field = Rect {
            x: x + pad,
            y: y + pad,
            w: inner_w,
            h: field_h,
        };
        PaletteLayout {
            panel: Rect { x, y, w, h },
            field,
            rule: Rect {
                x: field.x,
                y: field.y + field_h,
                w: inner_w,
                h: rule_h,
            },
            rows: Rect {
                x: field.x,
                y: field.y + field_h + rule_h,
                w: inner_w,
                h: row_h,
            },
            count,
            field_baseline: field.y + (field_h - text_h) / 2 + m.ascent,
            baseline_offset: (row_h - text_h) / 2 + m.ascent,
        }
    }
}

/// Where an open palette's panel, field, and rows are on screen. The rows are one
/// column of equal heights, so the layout holds the first and steps down to the rest
/// rather than carrying a rectangle per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteLayout {
    /// The panel behind everything, border and padding included.
    pub panel: Rect,
    /// The query row.
    pub field: Rect,
    /// The hairline between the field and the rows; zero-height when no row shows.
    pub rule: Rect,
    /// The baseline for the query's text, from the window's top.
    pub field_baseline: i32,
    /// The first drawn row; each one below it is `rows.h` further down.
    rows: Rect,
    count: usize,
    /// The text baseline inside a row, from the row's top.
    baseline_offset: i32,
}

impl PaletteLayout {
    /// The row drawn at `vis`, counting from the first one drawn rather than from the
    /// first match. Only [`Self::row_at`] and the painter ask, and both stay within
    /// the count.
    pub fn row_rect(&self, vis: usize) -> Rect {
        Rect {
            y: self.rows.y + vis as i32 * self.rows.h,
            ..self.rows
        }
    }

    /// The row at window point `(px, py)`, if the point is on one. A point on the
    /// field or the panel's padding is on no row, which is how a click there leaves
    /// the palette up rather than activating something.
    pub fn row_at(&self, px: f32, py: f32) -> Option<usize> {
        (0..self.count).find(|&i| self.row_rect(i).contains(px, py))
    }

    /// The baseline for row `vis`'s label: its row, centered on the text.
    pub fn baseline(&self, vis: usize) -> i32 {
        self.row_rect(vis).y + self.baseline_offset
    }
}

/// Tiered match score for `label` against a non-empty `query`, lower being the
/// better match:
///
///   0 = every query word starts a word of the label
///   1 = the query appears in the label
///   2 = the query's characters appear in the label in order
///
/// `None` when it does not match at all. Letters compare case-insensitively in ASCII
/// only, the fold [`crate::find`] uses and for its reason: a full Unicode fold
/// changes byte lengths, and nothing in a vault name or a path needs one to be typed
/// in lower case.
fn score(label: &str, query: &str) -> Option<u8> {
    // An all-whitespace query has no words, which would make the word test vacuously
    // true and rank every row first.
    let mut query_words = query.split_whitespace().peekable();
    if query_words.peek().is_some() && query_words.all(|q| words(label).any(|w| starts_with(w, q)))
    {
        return Some(0);
    }
    if contains(label, query) {
        return Some(1);
    }
    if subsequence(label, query) {
        return Some(2);
    }
    None
}

/// A label's words: the runs of letters and digits in it, which is what a query word
/// is matched against. The punctuation a label puts between them (a colon, an arrow,
/// a path separator) is not something anyone types to find it.
fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
}

/// Whether `hay` starts with `needle`, ASCII letters folded.
///
/// The comparison is byte-wise and needs no boundary check: both sides are UTF-8 and
/// the fold only touches A-Z, so equal bytes mean `hay` opens with exactly the whole
/// of `needle`, which ends where a character does.
fn starts_with(hay: &str, needle: &str) -> bool {
    let (hay, needle) = (hay.as_bytes(), needle.as_bytes());
    hay.len() >= needle.len() && hay[..needle.len()].eq_ignore_ascii_case(needle)
}

/// Whether `needle` appears anywhere in `hay`, ASCII letters folded.
fn contains(hay: &str, needle: &str) -> bool {
    let (hay, needle) = (hay.as_bytes(), needle.as_bytes());
    needle.is_empty()
        || (hay.len() >= needle.len()
            && hay
                .windows(needle.len())
                .any(|w| w.eq_ignore_ascii_case(needle)))
}

/// Whether `needle`'s characters appear in `hay` in order, ASCII letters folded.
fn subsequence(hay: &str, needle: &str) -> bool {
    let mut hay = hay.chars();
    needle
        .chars()
        .all(|n| hay.any(|h| h.eq_ignore_ascii_case(&n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A palette over `labels`, each row switching to the vault at its index, which
    /// is all the ranking and the geometry care about.
    fn palette(labels: &[&str]) -> Palette {
        Palette::new("Go to vault", Rows::Ranked, vault_rows(labels))
    }

    /// Rows switching to the vault at each label's index.
    fn vault_rows(labels: &[&str]) -> Vec<PaletteRow> {
        labels
            .iter()
            .enumerate()
            .map(|(i, l)| PaletteRow {
                label: (*l).to_string(),
                action: PaletteAction::SwitchVault(i),
            })
            .collect()
    }

    /// The labels of the rows a palette would draw, in the order it would draw them.
    fn shown(p: &Palette) -> Vec<&str> {
        p.visible().map(|(_, row)| row.label.as_str()).collect()
    }

    /// The app's real font stack at the body size: the panel is sized from real
    /// measured text, so the geometry is tested against it rather than a stand-in.
    fn test_fonts() -> Fonts {
        Fonts::with_config(
            &crate::platform::freetype::FontConfig::default(),
            &[crate::theme::FONT_SIZE],
            1.5,
        )
        .expect("open the default fonts")
    }

    /// The layout of `p` in a `window`-sized window at scale 1.
    fn layout(p: &Palette, window: (i32, i32)) -> PaletteLayout {
        p.layout(&test_fonts(), crate::theme::FONT_SIZE, Scale::ONE, window)
    }

    #[test]
    fn an_empty_query_keeps_the_order_it_was_given() {
        let p = palette(&["work", "notes", "archive"]);
        assert_eq!(shown(&p), ["work", "notes", "archive"]);
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(0)));
    }

    #[test]
    fn a_query_drops_the_rows_it_does_not_match() {
        let mut p = palette(&["work", "notes", "archive"]);
        p.push('z');
        assert!(shown(&p).is_empty());
        // Enter has nowhere to go rather than falling back to a row nobody picked.
        assert_eq!(p.activate(), None);
    }

    #[test]
    fn a_word_prefix_outranks_a_substring_and_a_substring_outranks_a_scatter() {
        // "arc" scatters through the first, sits inside a word of the second, and
        // starts a word of the third. Built worst-first, so only the ranking can put
        // them in this order.
        let mut p = palette(&["a: r: cat", "march: files", "arc: files"]);
        for c in "arc".chars() {
            p.push(c);
        }
        assert_eq!(shown(&p), ["arc: files", "march: files", "a: r: cat"]);
    }

    #[test]
    fn ranking_is_stable_inside_a_tier() {
        // Both are word-prefix matches, so the order they were built in stands.
        let mut p = palette(&["vault: work", "vault: wiki"]);
        for c in "vault".chars() {
            p.push(c);
        }
        assert_eq!(shown(&p), ["vault: work", "vault: wiki"]);
    }

    #[test]
    fn every_query_word_has_to_start_a_word_of_the_label() {
        let mut p = palette(&["vault: work", "vault: wiki"]);
        p.push_str("vault wo");
        assert_eq!(shown(&p), ["vault: work"]);
    }

    #[test]
    fn a_query_word_matches_across_the_labels_punctuation() {
        // A vault whose name is two words is found by typing them, with the colon the
        // row is formatted with in between and never typed.
        let mut p = palette(&["vault: reading notes"]);
        p.push_str("reading notes");
        assert_eq!(shown(&p).len(), 1);
    }

    #[test]
    fn letters_match_whatever_case_they_are_typed_in() {
        let mut p = palette(&["Work Notes"]);
        p.push_str("WORK");
        assert_eq!(shown(&p).len(), 1);
        p.pop();
        p.pop();
        p.pop();
        p.pop();
        p.push_str("work notes");
        assert_eq!(shown(&p).len(), 1);
    }

    #[test]
    fn a_non_ascii_label_matches_its_own_bytes_and_ranks_by_them() {
        // The fold is ASCII-only, so an accented letter matches itself and not its
        // unaccented twin. Nothing here may split a character.
        let mut p = palette(&["Ärlig", "notes"]);
        p.push('Ä');
        assert_eq!(shown(&p), ["Ärlig"]);
        p.pop();
        p.push('A');
        assert!(shown(&p).is_empty());
    }

    #[test]
    fn backspacing_back_to_nothing_shows_everything_again() {
        let mut p = palette(&["work", "notes"]);
        p.push('w');
        assert_eq!(shown(&p), ["work"]);
        p.pop();
        assert_eq!(shown(&p), ["work", "notes"]);
    }

    #[test]
    fn a_pasted_newline_does_not_reach_the_query() {
        let mut p = palette(&["work"]);
        p.push_str("wo\nrk\n");
        assert_eq!(p.query(), "work");
    }

    #[test]
    fn the_arrows_move_the_selection_and_stop_at_the_ends() {
        let mut p = palette(&["a", "b", "c"]);
        assert_eq!(p.selected(), 0);
        p.move_selection(1);
        assert_eq!(p.selected(), 1);
        p.move_selection(-5);
        assert_eq!(p.selected(), 0);
        p.move_selection(9);
        assert_eq!(p.selected(), 2);
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(2)));
    }

    #[test]
    fn the_arrows_do_nothing_when_the_query_matches_nothing() {
        let mut p = palette(&["work"]);
        p.push('z');
        p.move_selection(1);
        assert_eq!(p.activate(), None);
    }

    #[test]
    fn a_new_query_puts_the_selection_back_on_the_top_match() {
        let mut p = palette(&["work", "notes"]);
        p.move_selection(1);
        assert_eq!(p.selected(), 1);
        p.push('n');
        assert_eq!(p.selected(), 0);
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(1)));
    }

    #[test]
    fn activate_follows_the_filtered_order_not_the_build_order() {
        let mut p = palette(&["archive", "work"]);
        p.push('w');
        // One match, and it is the row that was built second.
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(1)));
    }

    #[test]
    fn a_long_list_draws_only_a_capped_window_of_rows() {
        let labels: Vec<String> = (0..40).map(|i| format!("row{i:02}")).collect();
        let p = palette(&labels.iter().map(String::as_str).collect::<Vec<_>>());
        assert_eq!(p.visible_count(), MAX_VISIBLE);
        assert_eq!(shown(&p).len(), MAX_VISIBLE);
    }

    #[test]
    fn arrowing_past_the_last_drawn_row_scrolls_it_into_view() {
        let labels: Vec<String> = (0..40).map(|i| format!("row{i:02}")).collect();
        let mut p = palette(&labels.iter().map(String::as_str).collect::<Vec<_>>());
        // Step down past the cap one row at a time, the way the arrow key does.
        for _ in 0..MAX_VISIBLE {
            p.move_selection(1);
        }
        assert_eq!(p.selected(), MAX_VISIBLE);
        // The selection is drawn, and it is the last of the drawn rows.
        let drawn: Vec<usize> = p.visible().map(|(pos, _)| pos).collect();
        assert_eq!(drawn.last(), Some(&p.selected()));
        assert_eq!(drawn.len(), MAX_VISIBLE);
        // And back up to the top brings the first rows back.
        for _ in 0..MAX_VISIBLE {
            p.move_selection(-1);
        }
        assert_eq!(p.visible().next().map(|(pos, _)| pos), Some(0));
    }

    #[test]
    fn a_row_the_scroll_has_passed_keeps_its_own_action() {
        let labels: Vec<String> = (0..40).map(|i| format!("row{i:02}")).collect();
        let mut p = palette(&labels.iter().map(String::as_str).collect::<Vec<_>>());
        for _ in 0..20 {
            p.move_selection(1);
        }
        // The drawn rows are a window into the list, not a list of their own: each
        // one still answers with the vault it was built for.
        for (pos, _) in p.visible() {
            assert_eq!(p.action_at(pos), Some(PaletteAction::SwitchVault(pos)));
        }
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(20)));
    }

    #[test]
    fn match_spans_marks_every_occurrence_case_folded() {
        // Both "editor"s are marked, in the path and in the line, whatever their case.
        let label = "Simple image Editor.md:1  a video editor";
        let spans = match_spans(label, "editor");
        let marked: Vec<&str> = spans.iter().map(|&(s, e)| &label[s..e]).collect();
        assert_eq!(marked, ["Editor", "editor"]);
    }

    #[test]
    fn match_spans_is_empty_without_a_query_or_a_match() {
        assert!(match_spans("vault: work", "").is_empty());
        assert!(match_spans("vault: work", "zzz").is_empty());
    }

    #[test]
    fn match_spans_do_not_overlap() {
        // "aa" in "aaaa" marks two runs, not three: the scan resumes past the one it
        // took, so a highlight never double-covers a byte.
        assert_eq!(match_spans("aaaa", "aa"), [(0, 2), (2, 4)]);
    }

    #[test]
    fn a_match_span_lands_on_character_boundaries() {
        // The fold is ASCII only, so a multibyte character neither matches an ASCII
        // query nor splits under one: the ranges are always safe to slice.
        let label = "æøå editor";
        let spans = match_spans(label, "editor");
        assert_eq!(spans.len(), 1);
        let (s, e) = spans[0];
        assert_eq!(&label[s..e], "editor"); // slices cleanly, no panic
    }

    #[test]
    fn rows_that_answer_the_query_are_never_re_ranked() {
        // A search's rows already matched; scoring them again would rank a hit by how
        // its own label reads rather than leaving the scan's order alone. Here the
        // second row's path holds the query and the first's does not, which a ranked
        // palette would reorder.
        let mut p = Palette::new(
            "Search this vault",
            Rows::AsGiven,
            vault_rows(&[
                "notes/a.md:12  the gamma line",
                "notes/gamma.md:3  nothing to see",
            ]),
        );
        p.push_str("gamma");
        assert_eq!(
            shown(&p),
            [
                "notes/a.md:12  the gamma line",
                "notes/gamma.md:3  nothing to see"
            ]
        );
    }

    #[test]
    fn rows_that_answer_the_query_survive_a_query_that_matches_no_label() {
        // The scan matched file contents, so a row's label need not contain the query
        // at all (a hit's line can be shown trimmed or elided). A ranked palette would
        // drop every row here.
        let mut p = Palette::new(
            "Search this vault",
            Rows::AsGiven,
            vault_rows(&["a.md:1  ..."]),
        );
        p.push_str("gamma");
        assert_eq!(shown(&p).len(), 1);
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(0)));
    }

    #[test]
    fn new_rows_replace_the_answer_and_reset_the_selection() {
        let mut p = Palette::new(
            "Search this vault",
            Rows::AsGiven,
            vault_rows(&["a", "b", "c"]),
        );
        p.move_selection(2);
        assert_eq!(p.selected(), 2);
        // The query changed, so the app hands over what it now answers.
        p.set_rows(vault_rows(&["x"]));
        assert_eq!(shown(&p), ["x"]);
        assert_eq!(p.selected(), 0);
        // And the stale selection cannot outlive the rows it pointed into.
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(0)));
    }

    #[test]
    fn an_answer_of_no_rows_leaves_nothing_to_activate() {
        let mut p = Palette::new("Search this vault", Rows::AsGiven, vault_rows(&["a"]));
        p.set_rows(Vec::new());
        assert!(shown(&p).is_empty());
        assert_eq!(p.activate(), None);
    }

    #[test]
    fn a_ranked_palette_still_ranks_after_the_mode_was_added() {
        // The vault palette's behavior is the other half of the same switch.
        let mut p = palette(&["vault: wiki", "vault: work"]);
        p.push_str("work");
        assert_eq!(shown(&p), ["vault: work"]);
    }

    #[test]
    fn the_pointer_moves_the_selection_and_reports_only_real_changes() {
        let mut p = palette(&["a", "b"]);
        assert!(p.select(1));
        assert_eq!(p.selected(), 1);
        assert!(!p.select(1));
        // Enter takes whatever the pointer last moved to, the same row it lights.
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(1)));
    }

    #[test]
    fn the_pointer_cannot_select_a_row_that_is_not_there() {
        let mut p = palette(&["work", "notes"]);
        p.push('w');
        // One match left, so the second row's position is nobody's.
        assert!(!p.select(1));
        assert_eq!(p.activate(), Some(PaletteAction::SwitchVault(0)));
    }

    #[test]
    fn the_panel_is_centered_below_the_windows_top() {
        let p = palette(&["work"]);
        let l = layout(&p, (1000, 800));
        assert_eq!(l.panel.x + l.panel.w / 2, 500);
        assert!(l.panel.y > 0);
        assert!(l.panel.y + l.panel.h < 800);
    }

    #[test]
    fn a_narrow_window_shrinks_the_panel_instead_of_overflowing_it() {
        let p = palette(&["work"]);
        let l = layout(&p, (200, 800));
        assert!(l.panel.x >= 0);
        assert!(l.panel.x + l.panel.w <= 200);
    }

    #[test]
    fn a_window_too_short_pins_the_panel_to_the_top() {
        // Nothing fits, so the clamp must not push the panel off the top and put the
        // field out of reach.
        let p = palette(&["a", "b", "c"]);
        let l = layout(&p, (1000, 20));
        assert_eq!(l.panel.y, 0);
    }

    #[test]
    fn the_field_and_the_rows_stack_inside_the_panel_without_overlapping() {
        let p = palette(&["a", "b", "c"]);
        let l = layout(&p, (1000, 800));
        assert!(l.field.y > l.panel.y);
        assert_eq!(l.field.y + l.field.h, l.rule.y);
        assert_eq!(l.rule.y + l.rule.h, l.row_rect(0).y);
        for i in 0..2 {
            let (a, b) = (l.row_rect(i), l.row_rect(i + 1));
            assert_eq!(a.y + a.h, b.y);
            assert_eq!(a.h, b.h);
        }
        let last = l.row_rect(2);
        assert!(last.y + last.h <= l.panel.y + l.panel.h);
    }

    #[test]
    fn a_query_matching_nothing_draws_the_field_alone() {
        let mut p = palette(&["work"]);
        p.push('z');
        let l = layout(&p, (1000, 800));
        // No rows, so no rule to divide the field from them, and the panel is only
        // as tall as the field it holds.
        assert_eq!(l.rule.h, 0);
        assert_eq!(l.row_at(500.0, (l.field.y + l.field.h + 4) as f32), None);
        assert!(l.panel.h < 3 * l.field.h);
    }

    #[test]
    fn row_at_finds_the_row_under_the_pointer() {
        let p = palette(&["a", "b", "c"]);
        let l = layout(&p, (1000, 800));
        for i in 0..3 {
            let r = l.row_rect(i);
            let mid = (r.x as f32 + r.w as f32 / 2.0, r.y as f32 + r.h as f32 / 2.0);
            assert_eq!(l.row_at(mid.0, mid.1), Some(i));
        }
    }

    #[test]
    fn row_at_misses_the_field_and_everything_outside_the_panel() {
        let p = palette(&["a"]);
        let l = layout(&p, (1000, 800));
        // A click in the field is not a click on a row: it leaves the palette up.
        let f = l.field;
        assert_eq!(
            l.row_at(f.x as f32 + 1.0, f.y as f32 + f.h as f32 / 2.0),
            None
        );
        assert_eq!(l.row_at(0.0, 700.0), None);
    }

    #[test]
    fn a_label_sits_on_its_own_row() {
        let p = palette(&["a", "b"]);
        let l = layout(&p, (1000, 800));
        for i in 0..2 {
            let r = l.row_rect(i);
            assert!(l.baseline(i) > r.y && l.baseline(i) < r.y + r.h);
        }
        assert!(l.field_baseline > l.field.y && l.field_baseline < l.field.y + l.field.h);
    }

    #[test]
    fn the_drawn_rows_never_outrun_the_layouts_count() {
        // The painter walks `visible` and the pointer walks `row_at`, so the two have
        // to agree on how many rows there are, scrolled or not.
        let labels: Vec<String> = (0..40).map(|i| format!("row{i:02}")).collect();
        let mut p = palette(&labels.iter().map(String::as_str).collect::<Vec<_>>());
        for _ in 0..25 {
            p.move_selection(1);
        }
        let l = layout(&p, (1000, 800));
        assert_eq!(l.count, p.visible().count());
    }
}
