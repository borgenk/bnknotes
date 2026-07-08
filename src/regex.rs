//! A small, dependency-free regular-expression engine for find-in-document.
//!
//! Supported syntax: literal characters, `.` (any character but a newline),
//! escapes (`\d \D \w \W \s \S`, `\n \t \r`, and `\x` for any punctuation `x`),
//! character classes `[abc]`, `[a-z0-9]`, `[^...]`, groups `(...)`,
//! alternation `|`, the quantifiers `* + ?`, and the line anchors `^ $`.
//! No captures, no backreferences, no counted repetition (`{` is a literal).
//!
//! Matching is ASCII-case-insensitive, exactly like the editor's literal
//! search, and returns the *longest* match at the leftmost position. The
//! pattern compiles to a nondeterministic automaton executed breadth-first (a
//! Pike VM): every step advances all alternatives at once, so runtime is
//! text × program with no catastrophic backtracking, ever. Execution walks
//! characters, not bytes, so a match can never start or end inside a UTF-8
//! sequence.

/// A compiled pattern.
pub struct Regex {
    prog: Vec<Inst>,
}

/// One automaton instruction. `Split` tries both targets (alternation and
/// quantifiers), `Jump` is an unconditional hop, the anchors are zero-width
/// context checks, and `Accept` marks a complete match.
enum Inst {
    Class(CharClass),
    Split(usize, usize),
    Jump(usize),
    LineStart,
    LineEnd,
    Accept,
}

/// A set of character ranges, possibly negated. A literal is a single
/// one-character range; `.` is a negated newline; `\d`/`\w`/`\s` expand to the
/// obvious range lists.
#[derive(Clone)]
struct CharClass {
    negated: bool,
    ranges: Vec<(char, char)>,
}

impl CharClass {
    fn lit(c: char) -> Self {
        Self {
            negated: false,
            ranges: vec![(c, c)],
        }
    }

    /// Whether `c` matches, folding ASCII case (both the character and its
    /// swapped-case twin are tried against the ranges, then negation applies,
    /// so `[^a]` does not match `A` either).
    fn matches(&self, c: char) -> bool {
        let mut hit = self.in_ranges(c);
        if !hit && c.is_ascii_alphabetic() {
            let swapped = if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else {
                c.to_ascii_lowercase()
            };
            hit = self.in_ranges(swapped);
        }
        hit != self.negated
    }

    fn in_ranges(&self, c: char) -> bool {
        self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi)
    }
}

/// The parsed shape of a pattern, compiled to instructions after parsing.
enum Ast {
    Empty,
    Class(CharClass),
    LineStart,
    LineEnd,
    Concat(Vec<Ast>),
    Alt(Box<Ast>, Box<Ast>),
    Star(Box<Ast>),
    Plus(Box<Ast>),
    Quest(Box<Ast>),
}

/// Reusable Pike VM scratch: the two thread lists (the set live at the current
/// position and the set being built for the next character) with a dedup bitmap
/// each. [`Regex::find_all`] keeps one across every starting byte and character,
/// so scanning a long non-matching document reuses these buffers instead of
/// allocating a fresh pair per step.
struct Threads {
    /// Program counters live at the current position.
    current: Vec<usize>,
    /// Program counters reached by consuming the current character.
    next: Vec<usize>,
    /// Which counters `current` already holds, to dedup threads within a step.
    current_seen: Vec<bool>,
    /// The same dedup bitmap for `next`.
    next_seen: Vec<bool>,
}

impl Threads {
    fn new(prog_len: usize) -> Self {
        Self {
            current: Vec::new(),
            next: Vec::new(),
            current_seen: vec![false; prog_len],
            next_seen: vec![false; prog_len],
        }
    }

    /// Empty both lists and clear both dedup bitmaps for a fresh `match_at` at a
    /// new starting position.
    fn reset(&mut self) {
        self.current.clear();
        self.next.clear();
        self.current_seen.fill(false);
        self.next_seen.fill(false);
    }
}

impl Regex {
    /// Compile `pattern`, or a one-line reason it is not a valid pattern.
    pub fn compile(pattern: &str) -> Result<Self, &'static str> {
        let mut parser = Parser {
            chars: pattern.chars().collect(),
            pos: 0,
        };
        let ast = parser.alt()?;
        if parser.pos != parser.chars.len() {
            return Err("unmatched )");
        }
        let mut prog = Vec::new();
        emit(&ast, &mut prog);
        prog.push(Inst::Accept);
        Ok(Self { prog })
    }

    /// Every non-overlapping match in `text`, in order, each the longest match
    /// at its starting position. Empty matches are skipped: highlighting a
    /// zero-width range is useless in a find UI and would otherwise match at
    /// every position.
    pub fn find_all(&self, text: &str, limit: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut threads = Threads::new(self.prog.len());
        let mut i = 0;
        while i <= text.len() {
            if let Some(end) = self.match_at(text, i, &mut threads) {
                if end > i {
                    out.push((i, end));
                    if out.len() >= limit {
                        break;
                    }
                    i = end;
                    continue;
                }
            }
            match text[i..].chars().next() {
                Some(c) => i += c.len_utf8(),
                None => break,
            }
        }
        out
    }

    /// The end of the longest match starting exactly at byte `start` (a char
    /// boundary), or `None`. Runs every automaton alternative in lockstep over
    /// the text, recording the furthest position where an `Accept` was live.
    fn match_at(&self, text: &str, start: usize, threads: &mut Threads) -> Option<usize> {
        threads.reset();
        let mut best = None;
        self.add_thread(
            &mut threads.current,
            &mut threads.current_seen,
            0,
            text,
            start,
            &mut best,
        );

        let mut pos = start;
        while pos < text.len() && !threads.current.is_empty() {
            let Some(c) = text[pos..].chars().next() else {
                break;
            };
            let next_pos = pos + c.len_utf8();
            threads.next.clear();
            threads.next_seen.fill(false);
            for &pc in &threads.current {
                if let Inst::Class(class) = &self.prog[pc] {
                    if class.matches(c) {
                        self.add_thread(
                            &mut threads.next,
                            &mut threads.next_seen,
                            pc + 1,
                            text,
                            next_pos,
                            &mut best,
                        );
                    }
                }
            }
            std::mem::swap(&mut threads.current, &mut threads.next);
            std::mem::swap(&mut threads.current_seen, &mut threads.next_seen);
            pos = next_pos;
        }
        best
    }

    /// Add the thread at `pc` and everything reachable from it without
    /// consuming a character (splits, jumps, satisfied anchors), recording any
    /// reachable `Accept` into `best`. `visited` de-duplicates threads within
    /// one step, which is what bounds the VM to text × program time.
    fn add_thread(
        &self,
        list: &mut Vec<usize>,
        visited: &mut [bool],
        pc: usize,
        text: &str,
        pos: usize,
        best: &mut Option<usize>,
    ) {
        if visited[pc] {
            return;
        }
        visited[pc] = true;
        match &self.prog[pc] {
            Inst::Class(_) => list.push(pc),
            Inst::Split(a, b) => {
                self.add_thread(list, visited, *a, text, pos, best);
                self.add_thread(list, visited, *b, text, pos, best);
            }
            Inst::Jump(t) => self.add_thread(list, visited, *t, text, pos, best),
            Inst::LineStart => {
                if pos == 0 || text.as_bytes()[pos - 1] == b'\n' {
                    self.add_thread(list, visited, pc + 1, text, pos, best);
                }
            }
            Inst::LineEnd => {
                if pos == text.len() || text.as_bytes()[pos] == b'\n' {
                    self.add_thread(list, visited, pc + 1, text, pos, best);
                }
            }
            Inst::Accept => *best = Some(best.map_or(pos, |b: usize| b.max(pos))),
        }
    }
}

/// Emit `ast` onto `prog`. Quantifiers and alternation compile to the standard
/// split/jump shapes; the split's first target is the "stay in the loop"
/// branch, though match length is decided by the longest live `Accept`, not by
/// branch order.
fn emit(ast: &Ast, prog: &mut Vec<Inst>) {
    match ast {
        Ast::Empty => {}
        Ast::Class(c) => prog.push(Inst::Class(c.clone())),
        Ast::LineStart => prog.push(Inst::LineStart),
        Ast::LineEnd => prog.push(Inst::LineEnd),
        Ast::Concat(items) => {
            for item in items {
                emit(item, prog);
            }
        }
        Ast::Alt(a, b) => {
            let split = prog.len();
            prog.push(Inst::Split(0, 0));
            emit(a, prog);
            let jump = prog.len();
            prog.push(Inst::Jump(0));
            let b_start = prog.len();
            emit(b, prog);
            let after = prog.len();
            prog[split] = Inst::Split(split + 1, b_start);
            prog[jump] = Inst::Jump(after);
        }
        Ast::Star(e) => {
            let split = prog.len();
            prog.push(Inst::Split(0, 0));
            emit(e, prog);
            prog.push(Inst::Jump(split));
            let after = prog.len();
            prog[split] = Inst::Split(split + 1, after);
        }
        Ast::Plus(e) => {
            let body = prog.len();
            emit(e, prog);
            let after = prog.len() + 1;
            prog.push(Inst::Split(body, after));
        }
        Ast::Quest(e) => {
            let split = prog.len();
            prog.push(Inst::Split(0, 0));
            emit(e, prog);
            let after = prog.len();
            prog[split] = Inst::Split(split + 1, after);
        }
    }
}

/// Recursive-descent parser over the pattern's characters. Grammar:
/// `alt = concat ('|' concat)*`, `concat = repeat*`,
/// `repeat = atom ('*'|'+'|'?')*`, `atom = group | class | escape | anchor |
/// '.' | literal`.
struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn alt(&mut self) -> Result<Ast, &'static str> {
        let mut lhs = self.concat()?;
        while self.peek() == Some('|') {
            self.pos += 1;
            let rhs = self.concat()?;
            lhs = Ast::Alt(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn concat(&mut self) -> Result<Ast, &'static str> {
        let mut items = Vec::new();
        while !matches!(self.peek(), None | Some('|') | Some(')')) {
            items.push(self.repeat()?);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.remove(0),
            _ => Ast::Concat(items),
        })
    }

    fn repeat(&mut self) -> Result<Ast, &'static str> {
        let mut atom = self.atom()?;
        loop {
            let quantifier = match self.peek() {
                Some(q @ ('*' | '+' | '?')) => q,
                _ => return Ok(atom),
            };
            if matches!(atom, Ast::LineStart | Ast::LineEnd | Ast::Empty) {
                return Err("nothing to repeat");
            }
            self.pos += 1;
            atom = match quantifier {
                '*' => Ast::Star(Box::new(atom)),
                '+' => Ast::Plus(Box::new(atom)),
                _ => Ast::Quest(Box::new(atom)),
            };
        }
    }

    fn atom(&mut self) -> Result<Ast, &'static str> {
        match self.bump() {
            Some('(') => {
                let inner = self.alt()?;
                if self.bump() != Some(')') {
                    return Err("unclosed (");
                }
                Ok(inner)
            }
            Some('[') => self.class(),
            Some('.') => Ok(Ast::Class(CharClass {
                negated: true,
                ranges: vec![('\n', '\n')],
            })),
            Some('^') => Ok(Ast::LineStart),
            Some('$') => Ok(Ast::LineEnd),
            Some('\\') => Ok(Ast::Class(self.escape()?)),
            Some('*' | '+' | '?') => Err("nothing to repeat"),
            Some(c) => Ok(Ast::Class(CharClass::lit(c))),
            None => Ok(Ast::Empty),
        }
    }

    /// The class an escape denotes; the parser has already consumed the `\`.
    /// Unknown escapes are the literal character, so `\.` and `\\` just work.
    fn escape(&mut self) -> Result<CharClass, &'static str> {
        let shorthand = |negated, ranges: &[(char, char)]| CharClass {
            negated,
            ranges: ranges.to_vec(),
        };
        let digits = [('0', '9')];
        let word = [('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')];
        let space = [(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')];
        match self.bump() {
            Some('d') => Ok(shorthand(false, &digits)),
            Some('D') => Ok(shorthand(true, &digits)),
            Some('w') => Ok(shorthand(false, &word)),
            Some('W') => Ok(shorthand(true, &word)),
            Some('s') => Ok(shorthand(false, &space)),
            Some('S') => Ok(shorthand(true, &space)),
            Some('n') => Ok(CharClass::lit('\n')),
            Some('t') => Ok(CharClass::lit('\t')),
            Some('r') => Ok(CharClass::lit('\r')),
            Some(c) => Ok(CharClass::lit(c)),
            None => Err("trailing backslash"),
        }
    }

    /// A `[...]` class; the parser has already consumed the `[`. A leading `]`
    /// (after any `^`) is a literal, as is a trailing `-`.
    fn class(&mut self) -> Result<Ast, &'static str> {
        let negated = if self.peek() == Some('^') {
            self.pos += 1;
            true
        } else {
            false
        };
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let c = match self.bump() {
                None => return Err("unclosed ["),
                Some(']') if !first => break,
                Some(c) => c,
            };
            first = false;
            let lo = if c == '\\' {
                let class = self.escape()?;
                // A shorthand inside a class contributes its ranges wholesale
                // (negated shorthands inside classes are not supported).
                if class.negated {
                    return Err("negated escape inside [ ]");
                }
                if class.ranges.len() > 1 {
                    ranges.extend(class.ranges);
                    continue;
                }
                class.ranges[0].0
            } else {
                c
            };
            // `a-z` is a range unless the `-` is last (then both are literal).
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.pos += 1;
                let hi = match self.bump() {
                    None => return Err("unclosed ["),
                    Some('\\') => {
                        let class = self.escape()?;
                        if class.negated || class.ranges.len() > 1 {
                            return Err("bad range");
                        }
                        class.ranges[0].0
                    }
                    Some(hi) => hi,
                };
                if lo > hi {
                    return Err("bad range");
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        Ok(Ast::Class(CharClass { negated, ranges }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(pattern: &str, text: &str) -> Vec<(usize, usize)> {
        Regex::compile(pattern)
            .expect("pattern compiles")
            .find_all(text, usize::MAX)
    }

    #[test]
    fn literals_match_like_the_plain_search() {
        assert_eq!(all("one", "one two one"), vec![(0, 3), (8, 11)]);
        assert_eq!(all("fox", "Fox FOX"), vec![(0, 3), (4, 7)], "ASCII folded");
    }

    #[test]
    fn find_all_stops_at_the_limit() {
        let re = Regex::compile("a").expect("pattern compiles");
        assert_eq!(re.find_all("aaaa", 2).len(), 2, "capped");
        assert_eq!(re.find_all("aaaa", usize::MAX).len(), 4, "uncapped");
    }

    #[test]
    fn dot_matches_any_char_but_newline() {
        assert_eq!(all("a.c", "abc a\nc aéc"), vec![(0, 3), (8, 12)]);
    }

    #[test]
    fn escapes_and_shorthands() {
        assert_eq!(all(r"\d+", "a12 b345"), vec![(1, 3), (5, 8)]);
        assert_eq!(all(r"\w+", "so_x y!"), vec![(0, 4), (5, 6)]);
        assert_eq!(
            all(r"a\.b", "a.b axb"),
            vec![(0, 3)],
            "escaped dot is literal"
        );
        assert_eq!(all(r"\S+", "a  bc"), vec![(0, 1), (3, 5)]);
    }

    #[test]
    fn classes_ranges_and_negation() {
        assert_eq!(all("[abc]+", "cab dz"), vec![(0, 3)]);
        assert_eq!(all("[a-c0-2]+", "ab12 xz2"), vec![(0, 4), (7, 8)]);
        assert_eq!(all("[^ ]+", "ab c"), vec![(0, 2), (3, 4)]);
        assert_eq!(all("x[-]y", "x-y"), vec![(0, 3)], "lone - is literal");
        assert_eq!(all("[a-c]", "B"), vec![(0, 1)], "class folds ASCII case");
        assert_eq!(all("[^a]", "A"), Vec::new(), "negation folds too");
    }

    #[test]
    fn quantifiers_are_greedy_and_longest_wins() {
        assert_eq!(all("ab*", "abbb a"), vec![(0, 4), (5, 6)]);
        assert_eq!(all("ab+", "abbb a"), vec![(0, 4)]);
        assert_eq!(all("ab?c", "ac abc"), vec![(0, 2), (3, 6)]);
        assert_eq!(all("a|ab", "ab"), vec![(0, 2)], "longest alternative wins");
    }

    #[test]
    fn groups_and_alternation() {
        assert_eq!(all("(ab)+", "ababab x"), vec![(0, 6)]);
        assert_eq!(all("cat|dog", "a cat, a dog"), vec![(2, 5), (9, 12)]);
        assert_eq!(all("c(a|o)t", "cat cot"), vec![(0, 3), (4, 7)]);
    }

    #[test]
    fn anchors_bind_to_lines() {
        assert_eq!(all("^b", "b\nab\nba"), vec![(0, 1), (5, 6)]);
        assert_eq!(all("a$", "a\nba\nab"), vec![(0, 1), (3, 4)]);
        assert_eq!(all("^$", "a\n\nb"), Vec::new(), "empty matches are skipped");
    }

    #[test]
    fn empty_matches_never_loop_or_report() {
        assert_eq!(all("a*", "bbb"), Vec::new());
        assert_eq!(all("a*", "bab"), vec![(1, 2)]);
    }

    #[test]
    fn matches_stay_on_char_boundaries() {
        let text = "æblegrød på blåbær";
        for (start, end) in all(".+", text) {
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        }
        assert_eq!(
            all("b[^ ]+r", text),
            vec![(2, 7), (15, 23)],
            "blegr in æblegrød, then all of blåbær"
        );
    }

    #[test]
    fn invalid_patterns_are_errors_not_panics() {
        for bad in ["*a", "+", "?x", "(a", "a)", "[a", r"\", "[z-a]", "a**"] {
            match bad {
                // Chained quantifiers are accepted ((a*)* is harmless here).
                "a**" => assert!(Regex::compile(bad).is_ok()),
                _ => assert!(Regex::compile(bad).is_err(), "{bad} should not compile"),
            }
        }
    }

    #[test]
    fn braces_are_literal() {
        assert_eq!(all("a{2}", "a{2} aa"), vec![(0, 4)]);
    }

    #[test]
    fn pathological_nesting_stays_linear_enough() {
        // The classic backtracking killer: (a+)+ against a long non-matching
        // tail. A Pike VM walks it in text x program time; this returning at
        // all (instantly, in practice) is the point.
        let text = format!("{}b", "a".repeat(200));
        assert_eq!(all("(a+)+c", &text), Vec::new());
    }
}
