//! Find in document: the pure state machine behind the editor's Ctrl+F bar,
//! including its replace field and the whole-word and regex modes.
//!
//! Literal matching is byte-exact except ASCII letters, which compare
//! case-insensitively. That asymmetry is deliberate: full Unicode case folding
//! changes byte lengths (ß to ss) and would unmoor a match's range from the
//! text it matched, while ASCII folding keeps every match exactly
//! `query.len()` bytes and never lands mid-character (a UTF-8 continuation
//! byte can equal neither an ASCII byte nor a leading byte). Regex mode
//! matches through [`crate::regex`], which folds the same way and walks
//! characters, so its ranges are boundary-safe too. Matches never overlap.
//!
//! This module owns the query, the replacement, the mode flags, and the match
//! set — and nothing else. The app glues the returned ranges to the buffer
//! (select + reveal, replace) and hands the state to the painter, so every
//! transition here is unit-testable with plain strings.

use crate::grapheme;
use crate::regex::Regex;

/// Which of the bar's fields the keyboard currently edits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Field {
    Query,
    Replace,
}

/// Cap on collected matches. A query matching a large fraction of a big document
/// would otherwise allocate an unbounded match list (the mem-report's worst case
/// is gigabytes of `(start, end)` pairs). Collection stops at this many, which is
/// far past any count a user navigates or a note contains, so it only bites a
/// pathological search; a repeated replace-all still drains the rest as the anchor
/// advances past the replaced text.
const MAX_MATCHES: usize = 100_000;

/// The find bar's state: fields, modes, and the matches of the query against
/// the text it was last scanned over.
pub struct FindState {
    open: bool,
    query: String,
    replace: String,
    /// Whether the replace row is shown (Ctrl+H); Enter in the replace field
    /// replaces the current match.
    show_replace: bool,
    focus: Field,
    /// The focused field's entire content is selected (Ctrl+A, or the kept
    /// query on reopen): the next edit replaces it wholesale, Backspace clears
    /// it, and stepping through matches accepts it.
    selected: bool,
    /// Only match at word boundaries (Alt+W). Persists across close/open.
    word: bool,
    /// Interpret the query as a regular expression (Alt+R). Persists.
    regex: bool,
    /// The query is regex mode and does not compile.
    error: bool,
    /// Byte ranges of every match, in document order.
    matches: Vec<(usize, usize)>,
    /// Index into `matches` of the current match (the one the app selects).
    current: usize,
    /// The cursor position when the bar opened (or where the last replacement
    /// ended). While the query is edited the current match stays the first one
    /// at or after this point, so refining a query narrows in place instead of
    /// jumping around the document.
    anchor: usize,
}

impl FindState {
    pub fn new() -> Self {
        Self {
            open: false,
            query: String::new(),
            replace: String::new(),
            show_replace: false,
            focus: Field::Query,
            selected: false,
            word: false,
            regex: false,
            error: false,
            matches: Vec::new(),
            current: 0,
            anchor: 0,
        }
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    /// The replacement text, shown and applied when the replace row is open.
    pub fn replacement(&self) -> &str {
        &self.replace
    }

    pub fn show_replace(&self) -> bool {
        self.show_replace
    }

    pub fn focus(&self) -> Field {
        self.focus
    }

    /// Whether the focused field's content is shown selected (the next edit
    /// replaces it).
    pub fn selected(&self) -> bool {
        self.selected
    }

    pub fn word(&self) -> bool {
        self.word
    }

    pub fn regex(&self) -> bool {
        self.regex
    }

    /// Whether the query is an invalid regular expression.
    pub fn error(&self) -> bool {
        self.error
    }

    pub fn matches(&self) -> &[(usize, usize)] {
        &self.matches
    }

    /// Index of the current match within [`Self::matches`].
    pub fn current(&self) -> usize {
        self.current
    }

    /// The current match's range, if any.
    pub fn current_match(&self) -> Option<(usize, usize)> {
        self.matches.get(self.current).copied()
    }

    /// Open the bar searching from `cursor`. The previous query is kept,
    /// selected, and searched immediately, so Enter repeats the last search
    /// and typing starts a fresh one. Returns the range the app should select.
    pub fn open_at(&mut self, cursor: usize, text: &str) -> Option<(usize, usize)> {
        self.open = true;
        self.focus = Field::Query;
        self.selected = !self.query.is_empty();
        self.anchor = cursor;
        self.rescan(text)
    }

    /// Open (or, when already open, reveal) the replace row. Focus lands on
    /// the replace field when a query is already in place — the usual flow is
    /// "search, then decide to replace".
    pub fn open_replace_at(&mut self, cursor: usize, text: &str) -> Option<(usize, usize)> {
        let was_open = self.open;
        let target = if was_open {
            self.current_match()
        } else {
            self.open_at(cursor, text)
        };
        self.show_replace = true;
        if !self.query.is_empty() {
            self.focus = Field::Replace;
            self.selected = !self.replace.is_empty();
        }
        target
    }

    /// Close the bar. The query is kept for the next open's preselection; the
    /// mode toggles persist too.
    pub fn close(&mut self) {
        self.open = false;
        self.show_replace = false;
        self.focus = Field::Query;
        self.selected = false;
        self.error = false;
        self.matches.clear();
    }

    /// Move the keyboard between the query and replace fields.
    pub fn toggle_focus(&mut self) {
        if !self.show_replace {
            return;
        }
        self.selected = false;
        self.focus = match self.focus {
            Field::Query => Field::Replace,
            Field::Replace => Field::Query,
        };
    }

    /// Append `c` to the focused field (replacing its selected content) and
    /// rescan. Returns the range the app should select.
    pub fn push(&mut self, c: char, text: &str) -> Option<(usize, usize)> {
        self.edit_focused(text, |field| field.push(c))
    }

    /// Append a pasted string to the focused field, dropping control
    /// characters (the fields are single-line). Returns the range to select.
    pub fn push_str(&mut self, s: &str, text: &str) -> Option<(usize, usize)> {
        self.edit_focused(text, |field| {
            field.extend(s.chars().filter(|c| !c.is_control()))
        })
    }

    /// Delete the focused field's last grapheme cluster (a selected field
    /// clears wholesale) and rescan. Returns the range to select.
    pub fn pop(&mut self, text: &str) -> Option<(usize, usize)> {
        if self.selected {
            self.selected = false;
            match self.focus {
                Field::Query => {
                    self.query.clear();
                    return self.rescan(text);
                }
                Field::Replace => {
                    self.replace.clear();
                    return None;
                }
            }
        }
        let field = match self.focus {
            Field::Query => &mut self.query,
            Field::Replace => &mut self.replace,
        };
        if field.is_empty() {
            return None;
        }
        let cut = grapheme::prev_boundary(field, field.len());
        field.truncate(cut);
        match self.focus {
            Field::Query => self.rescan(text),
            Field::Replace => None,
        }
    }

    /// Toggle whole-word matching and rescan.
    pub fn toggle_word(&mut self, text: &str) -> Option<(usize, usize)> {
        self.word = !self.word;
        self.rescan(text)
    }

    /// Toggle regex interpretation of the query and rescan.
    pub fn toggle_regex(&mut self, text: &str) -> Option<(usize, usize)> {
        self.regex = !self.regex;
        self.rescan(text)
    }

    /// Step to the next match, wrapping past the last. `None` when there are
    /// no matches.
    pub fn next(&mut self) -> Option<(usize, usize)> {
        if self.matches.is_empty() {
            return None;
        }
        self.selected = false;
        self.current = (self.current + 1) % self.matches.len();
        self.current_match()
    }

    /// Step to the previous match, wrapping past the first. `None` when there
    /// are no matches.
    pub fn prev(&mut self) -> Option<(usize, usize)> {
        if self.matches.is_empty() {
            return None;
        }
        self.selected = false;
        self.current = (self.current + self.matches.len() - 1) % self.matches.len();
        self.current_match()
    }

    /// Rescan after the document changed (a replacement was applied),
    /// continuing from byte `from`. Returns the range the app should select.
    pub fn refresh(&mut self, text: &str, from: usize) -> Option<(usize, usize)> {
        self.anchor = from;
        self.rescan(text)
    }

    /// Run `edit` on the focused field (replacing its selected content first)
    /// and rescan when the query changed.
    fn edit_focused(
        &mut self,
        text: &str,
        edit: impl FnOnce(&mut String),
    ) -> Option<(usize, usize)> {
        let field = match self.focus {
            Field::Query => &mut self.query,
            Field::Replace => &mut self.replace,
        };
        if self.selected {
            self.selected = false;
            field.clear();
        }
        edit(field);
        match self.focus {
            Field::Query => self.rescan(text),
            Field::Replace => None,
        }
    }

    /// Select the focused field's entire content (Ctrl+A while the bar is
    /// open), so the next edit replaces it. A no-op on an empty field.
    pub fn select_all(&mut self) {
        let field = match self.focus {
            Field::Query => &self.query,
            Field::Replace => &self.replace,
        };
        if !field.is_empty() {
            self.selected = true;
        }
    }

    /// Rebuild the match set for the current query and modes, and re-aim
    /// `current` at the first match at or after the anchor (wrapping to the
    /// first overall).
    fn rescan(&mut self, text: &str) -> Option<(usize, usize)> {
        self.error = false;
        let mut found = if self.query.is_empty() {
            Vec::new()
        } else if self.regex {
            match Regex::compile(&self.query) {
                Ok(re) => re.find_all(text, MAX_MATCHES),
                Err(_) => {
                    self.error = true;
                    Vec::new()
                }
            }
        } else {
            scan(text, &self.query, MAX_MATCHES)
        };
        if self.word {
            found.retain(|&(start, end)| on_word_boundaries(text, start, end));
        }
        self.matches = found;
        self.current = self
            .matches
            .iter()
            .position(|&(start, _)| start >= self.anchor)
            .unwrap_or(0);
        self.current_match()
    }
}

/// Byte ranges of every non-overlapping occurrence of `query` in `text`, in
/// order, comparing ASCII letters case-insensitively and everything else
/// byte-exactly. An empty query matches nothing.
fn scan(text: &str, query: &str, limit: usize) -> Vec<(usize, usize)> {
    let (t, q) = (text.as_bytes(), query.as_bytes());
    let mut out = Vec::new();
    if q.is_empty() {
        return out;
    }
    let mut i = 0;
    while i + q.len() <= t.len() {
        if t[i..i + q.len()].eq_ignore_ascii_case(q) {
            out.push((i, i + q.len()));
            if out.len() >= limit {
                break;
            }
            i += q.len();
        } else {
            i += 1;
        }
    }
    out
}

/// Whether `[start, end)` sits on word boundaries: the characters adjacent to
/// the range (if any) are not word characters. Word = alphanumeric or `_`,
/// matching the editor's word-motion heuristic.
fn on_word_boundaries(text: &str, start: usize, end: usize) -> bool {
    let is_word = crate::grapheme::is_word_char;
    text[..start]
        .chars()
        .next_back()
        .is_none_or(|c| !is_word(c))
        && text[end..].chars().next().is_none_or(|c| !is_word(c))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The module `scan` gained a match cap; these tests want every match, so
    /// this shadow passes an unbounded limit.
    fn scan(text: &str, query: &str) -> Vec<(usize, usize)> {
        super::scan(text, query, usize::MAX)
    }

    #[test]
    fn scan_finds_all_occurrences_in_order() {
        assert_eq!(scan("one two one", "one"), vec![(0, 3), (8, 11)]);
        assert_eq!(scan("one two one", "zzz"), Vec::new());
        assert_eq!(
            scan("abc", ""),
            Vec::new(),
            "an empty query matches nothing"
        );
    }

    #[test]
    fn scan_folds_ascii_case_only() {
        assert_eq!(scan("Fox FOX fox", "fox"), vec![(0, 3), (4, 7), (8, 11)]);
        // Non-ASCII compares exactly: å (2 bytes) does not fold to Å.
        assert_eq!(scan("blåbær BLÅBÆR", "blå"), vec![(0, 4)]);
    }

    #[test]
    fn scan_matches_never_overlap() {
        assert_eq!(scan("aaaa", "aa"), vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn scan_ranges_stay_on_char_boundaries() {
        let text = "aéaéa";
        for (start, end) in scan(text, "é") {
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        }
        assert_eq!(scan(text, "é").len(), 2);
    }

    #[test]
    fn scan_stops_at_the_match_limit() {
        // The cap bounds the collected list, so a query matching a huge fraction
        // of a document cannot allocate without bound.
        assert_eq!(super::scan("a a a a a", "a", 2).len(), 2);
        assert_eq!(super::scan("a a a a a", "a", usize::MAX).len(), 5);
    }

    #[test]
    fn typing_narrows_from_the_anchor() {
        let text = "cat catalog cathedral";
        let mut f = FindState::new();
        f.open_at(4, text); // cursor at the start of "catalog"
        assert_eq!(
            f.push('c', text),
            Some((4, 5)),
            "first c at/after the anchor"
        );
        f.push('a', text);
        assert_eq!(f.push('t', text), Some((4, 7)), "still anchored at catalog");
        f.push('h', text);
        assert_eq!(f.query(), "cath");
        assert_eq!(f.matches(), &[(12, 16)], "only cathedral matches");
        assert_eq!(f.current(), 0);
    }

    #[test]
    fn the_anchor_wraps_to_the_first_match() {
        let text = "alpha beta";
        let mut f = FindState::new();
        f.open_at(8, text); // past the only "alpha"
        assert_eq!(f.push('a', text), Some((9, 10)), "the a in beta");
        assert_eq!(
            f.push('l', text),
            Some((0, 2)),
            "no al after the anchor: wrap"
        );
    }

    #[test]
    fn enter_steps_and_wraps_both_ways() {
        let text = "x.x.x";
        let mut f = FindState::new();
        f.open_at(0, text);
        assert_eq!(f.push('x', text), Some((0, 1)));
        assert_eq!(f.next(), Some((2, 3)));
        assert_eq!(f.next(), Some((4, 5)));
        assert_eq!(f.next(), Some((0, 1)), "wraps forward");
        assert_eq!(f.prev(), Some((4, 5)), "wraps backward");
        assert_eq!(f.current(), 2);
    }

    #[test]
    fn backspace_pops_a_whole_grapheme_and_rescans() {
        let text = "cafe caf";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in "caf".chars() {
            f.push(c, text);
        }
        // A decomposed é is two scalars but one cluster; Backspace removes both.
        f.push('e', text);
        f.push('\u{301}', text);
        assert_eq!(
            f.matches(),
            &[] as &[(usize, usize)],
            "café is not in the text"
        );
        assert_eq!(
            f.pop(text),
            Some((0, 3)),
            "back to caf, matching both words"
        );
        assert_eq!(f.query(), "caf");
        assert_eq!(f.matches().len(), 2);
    }

    #[test]
    fn popping_to_empty_clears_matches_and_selects_nothing() {
        let mut f = FindState::new();
        f.open_at(0, "aaa");
        f.push('a', "aaa");
        assert_eq!(f.pop("aaa"), None);
        assert!(f.matches().is_empty());
        assert_eq!(f.pop("aaa"), None, "popping an empty query is a no-op");
    }

    #[test]
    fn stepping_with_no_matches_is_a_no_op() {
        let mut f = FindState::new();
        f.open_at(0, "abc");
        assert_eq!(f.next(), None);
        assert_eq!(f.prev(), None);
    }

    #[test]
    fn reopening_preselects_the_last_query_and_searches_it() {
        let text = "ab ab ab";
        let mut f = FindState::new();
        f.open_at(0, text);
        f.push('a', text);
        f.push('b', text);
        f.close();

        // Reopen: the old query is live immediately, from the new cursor.
        let landed = f.open_at(4, text);
        assert_eq!(landed, Some((6, 8)), "reopen searches the kept query");
        assert!(f.selected());
        assert_eq!(f.query(), "ab");

        // The first typed character replaces the selected query wholesale.
        assert_eq!(f.push('x', text), None);
        assert_eq!(f.query(), "x");
        assert!(!f.selected());
    }

    #[test]
    fn backspace_clears_a_selected_query() {
        let text = "hay hay";
        let mut f = FindState::new();
        f.open_at(0, text);
        f.push('h', text);
        f.close();
        f.open_at(0, text);
        assert!(f.selected());
        assert_eq!(f.pop(text), None, "backspace wipes the preselection");
        assert_eq!(f.query(), "");
        assert!(f.matches().is_empty());
    }

    #[test]
    fn stepping_keeps_a_selected_query() {
        let text = "ab ab";
        let mut f = FindState::new();
        f.open_at(0, text);
        f.push('a', text);
        f.close();
        f.open_at(0, text);
        assert_eq!(f.next(), Some((3, 4)), "Enter reuses the kept query");
        assert!(!f.selected(), "stepping accepts the query");
        assert_eq!(f.query(), "a");
    }

    #[test]
    fn paste_appends_and_strips_control_characters() {
        let text = "one two";
        let mut f = FindState::new();
        f.open_at(0, text);
        assert_eq!(f.push_str("tw\no", text), Some((4, 7)), "newline dropped");
        assert_eq!(f.query(), "two");
    }

    #[test]
    fn whole_word_filters_substring_hits() {
        let text = "cat catalog cat_x scat";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in "cat".chars() {
            f.push(c, text);
        }
        assert_eq!(f.matches().len(), 4);
        f.toggle_word(text);
        assert_eq!(f.matches(), &[(0, 3)], "only the standalone word survives");
        assert!(f.word());
        f.toggle_word(text);
        assert_eq!(f.matches().len(), 4, "toggling back restores all hits");
    }

    #[test]
    fn regex_mode_matches_and_flags_errors() {
        let text = "a1 b22 c3";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in r"\d+".chars() {
            f.push(c, text);
        }
        assert!(
            f.matches().is_empty(),
            "literal backslash-d-plus matches nothing"
        );
        let landed = f.toggle_regex(text);
        assert_eq!(landed, Some((1, 2)));
        assert_eq!(f.matches(), &[(1, 2), (4, 6), (8, 9)]);
        assert!(!f.error());

        // An invalid pattern flags the error instead of matching garbage.
        f.push('(', text);
        assert!(f.error());
        assert!(f.matches().is_empty());
        f.pop(text);
        assert!(!f.error());
    }

    #[test]
    fn replace_field_edits_do_not_rescan() {
        let text = "aa aa";
        let mut f = FindState::new();
        f.open_replace_at(0, text);
        assert!(f.show_replace());
        assert_eq!(f.focus(), Field::Query, "empty query keeps focus on it");
        f.push('a', text);
        f.toggle_focus();
        assert_eq!(f.focus(), Field::Replace);
        assert_eq!(f.push('X', text), None, "replacement edits select nothing");
        assert_eq!(f.push('Y', text), None);
        assert_eq!(f.replacement(), "XY");
        assert_eq!(f.pop(text), None);
        assert_eq!(f.replacement(), "X");
        assert_eq!(f.query(), "a", "the query is untouched");
        assert_eq!(f.matches().len(), 4);
    }

    #[test]
    fn opening_replace_with_a_kept_query_focuses_the_replacement() {
        let text = "hit hit";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in "hit".chars() {
            f.push(c, text);
        }
        f.close();
        let landed = f.open_replace_at(0, text);
        assert_eq!(landed, Some((0, 3)));
        assert_eq!(f.focus(), Field::Replace);
        assert!(
            !f.selected(),
            "an empty replacement leaves nothing selected"
        );
    }

    #[test]
    fn refresh_rescans_from_a_position() {
        let text = "aa aa aa";
        let mut f = FindState::new();
        f.open_at(0, text);
        f.push('a', text);
        f.push('a', text);
        assert_eq!(
            f.refresh(text, 4),
            Some((6, 8)),
            "first match at/after byte 4"
        );
    }

    #[test]
    fn select_all_then_typing_replaces_the_query() {
        let text = "old new old";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in "old".chars() {
            f.push(c, text);
        }
        assert_eq!(f.matches().len(), 2);
        f.select_all();
        assert!(f.selected());
        assert_eq!(f.push('n', text), Some((4, 5)), "typing starts fresh");
        assert_eq!(f.query(), "n");
        assert!(!f.selected());
    }

    #[test]
    fn select_all_then_backspace_clears_the_field() {
        let text = "aaa";
        let mut f = FindState::new();
        f.open_at(0, text);
        f.push('a', text);
        f.select_all();
        assert_eq!(f.pop(text), None, "the whole query clears at once");
        assert_eq!(f.query(), "");
        assert!(f.matches().is_empty());
    }

    #[test]
    fn select_all_works_on_the_replace_field_without_touching_the_query() {
        let text = "hit hit";
        let mut f = FindState::new();
        f.open_replace_at(0, text);
        f.push('h', text);
        f.toggle_focus();
        f.push('X', text);
        f.push('Y', text);
        f.select_all();
        assert_eq!(f.push('Z', text), None, "typing replaces the replacement");
        assert_eq!(f.replacement(), "Z");
        assert_eq!(f.query(), "h", "the query is untouched");
        assert_eq!(f.matches().len(), 2, "matches survive replacement edits");
    }

    #[test]
    fn select_all_on_an_empty_field_is_a_no_op() {
        let mut f = FindState::new();
        f.open_at(0, "abc");
        f.select_all();
        assert!(!f.selected());
        f.push('a', "abc");
        assert_eq!(f.query(), "a", "typing appended normally");
    }

    #[test]
    fn switching_fields_drops_the_selection() {
        let text = "q q";
        let mut f = FindState::new();
        f.open_replace_at(0, text);
        f.push('q', text);
        f.select_all();
        f.toggle_focus();
        assert!(!f.selected(), "Tab accepts the selection");
        f.toggle_focus();
        f.push('x', text);
        assert_eq!(f.query(), "qx", "the query was not wiped");
    }

    #[test]
    fn reopening_replace_selects_the_kept_replacement() {
        let text = "a b a";
        let mut f = FindState::new();
        f.open_replace_at(0, text);
        f.push('a', text);
        f.toggle_focus();
        f.push('b', text);
        f.close();
        f.open_replace_at(0, text);
        assert_eq!(f.focus(), Field::Replace);
        assert!(f.selected(), "the old replacement is selected");
        f.push('c', text);
        assert_eq!(f.replacement(), "c", "typing replaced it");
    }

    #[test]
    fn word_boundaries_respect_unicode_letters() {
        // æ is a letter: "cat" inside "cætcat" borders it and is not a word.
        let text = "cat æcat";
        let mut f = FindState::new();
        f.open_at(0, text);
        for c in "cat".chars() {
            f.push(c, text);
        }
        f.toggle_word(text);
        assert_eq!(f.matches(), &[(0, 3)], "æcat's cat is glued to a letter");
    }
}
