//! Finding a URL inside a line of text.
//!
//! A URL usually arrives with no markup around it: typed straight into a note's prose,
//! or pasted in from a command's output (`git push` prints a pull-request URL, a dev
//! server prints `http://localhost:3000`). Nothing marks those as links, so an app
//! that wants them clickable has to *find* them in the text. That is this module, and
//! nothing else: a pure `&str -> Range` scan holding no buffer and no parser state, so
//! it is exhaustively testable without a window or a GPU.
//!
//! ```text
//!   "see (https://en.wikipedia.org/wiki/Cure_(album)) for more."
//!         ╰─┬──╯
//!           └─ initiation: a scheme we can open, at a token boundary
//!              termination then walks right, carrying
//!                last_safe  the last offset known to be *inside* the link
//!                stack      the brackets the URL itself opened
//!
//!   result:      https://en.wikipedia.org/wiki/Cure_(album)
//!                  the inner `(album)` is kept: its `)` closes a `(` the URL opened
//!                  the outer `)` is dropped: it closes nothing the URL opened
//!                  the trailing `.` is dropped: only punctuation follows it
//! ```
//!
//! Those two cases are the entire reason this is not a one-line regex, and both are
//! solved the way [UTS #58][uts58] specifies: one left-to-right pass with a backup
//! pointer and a bracket stack.
//!
//! - **Trailing punctuation.** A *hard* character (whitespace, a control, anything
//!   RFC 3986 forbids) always ends the link. A *soft* one (`.`, `,`, `!`, ...) is
//!   legal inside a URL yet is also how prose punctuates, so it may not *end* one.
//!   `last_safe` gets this for free: soft characters never advance it, so a run of
//!   them at the end is simply never reached. `example.com/a.b` keeps its dot;
//!   `see example.com/a.` gives it back to the sentence.
//! - **Balanced brackets.** A closing bracket belongs to the URL only when the URL
//!   opened it, which is why `last_safe` advances only at nesting depth zero: while
//!   a bracket is open the link cannot end, and an unmatched closer ends it at once.
//!   `en.wikipedia.org/wiki/Cure_(album)` keeps its parentheses; `(see example.com/a)`
//!   gives its `)` back to the prose.
//!
//! We detect exactly the schemes [`browser::OPENABLE_SCHEMES`] lists, by scanning for
//! that same constant. A link the app decorates and then refuses to follow is a worse
//! bug than one it never decorated, and the text being scanned may be hostile (a note
//! can hold whatever was pasted into it), so `javascript:` and `data:` are never
//! even candidates.
//!
//! [uts58]: https://www.unicode.org/reports/tr58/

use std::ops::Range;

use crate::platform::browser;

/// How deep a URL may nest brackets before we stop believing it is one. UTS #58
/// allows 125; a real URL never passes two or three, and a fixed array keeps the
/// scan allocation-free, which matters because it runs on pointer motion.
const MAX_NESTING: usize = 16;

/// The URLs in `text`, left to right and non-overlapping. A scheme *inside* an
/// already-yielded URL (the `http://` in a `?to=` query string) belongs to that URL
/// and is not reported again.
pub fn urls(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    let mut at = 0;
    std::iter::from_fn(move || {
        while at < text.len() {
            match scheme_at(text, at).and_then(|len| extent(text, at, len)) {
                Some(range) => {
                    at = range.end;
                    return Some(range);
                }
                // Not a link here (or a scheme with no host, or mid-character): the
                // next byte is the next candidate.
                None => at += 1,
            }
        }
        None
    })
}

/// The length of the scheme starting exactly at `at`, or `None` when no scheme does.
///
/// The token-boundary check is what stops `xhttps://x` and `see-mailto:x` from reading
/// as links: a character that could *continue* a scheme (RFC 3986 §3.1's `ALPHA /
/// DIGIT / "+" / "-" / "."`) sitting immediately before means the match is the tail of
/// a longer word, not a scheme.
fn scheme_at(text: &str, at: usize) -> Option<usize> {
    let rest = text.get(at..)?;
    let scheme = browser::OPENABLE_SCHEMES
        .iter()
        .find(|scheme| rest.starts_with(**scheme))?;
    let preceded = text
        .get(..at)
        .and_then(|before| before.chars().next_back())
        .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    (!preceded).then_some(scheme.len())
}

/// Walk right from the scheme at `start` to the end of the URL (the UTS #58
/// termination pass), or `None` when nothing survives past the scheme itself: a bare
/// `https://` has no host, so it is a word, not a link.
fn extent(text: &str, start: usize, scheme_len: usize) -> Option<Range<usize>> {
    let body = start + scheme_len;
    // The last offset known to be inside the link. It advances only at nesting depth
    // zero and only past a character that can legally *end* a URL, so a trailing run
    // of prose punctuation, or an unclosed bracket and everything after it, is simply
    // never reached and falls out of the range.
    let mut last_safe = body;
    let mut stack = ['\0'; MAX_NESTING];
    let mut depth = 0usize;
    for (offset, c) in text.get(body..)?.char_indices() {
        let next = body + offset + c.len_utf8();
        match classify(c) {
            Class::Hard => break,
            Class::Soft => {}
            Class::Open => {
                if depth == MAX_NESTING {
                    break;
                }
                stack[depth] = c;
                depth += 1;
            }
            Class::Close(opener) => {
                // A closer the URL never opened belongs to the text around it.
                if depth == 0 || stack[depth - 1] != opener {
                    break;
                }
                depth -= 1;
                if depth == 0 {
                    last_safe = next;
                }
            }
            Class::Url => {
                if depth == 0 {
                    last_safe = next;
                }
            }
        }
    }
    (last_safe > body).then_some(start..last_safe)
}

/// What a character does to the URL being scanned.
enum Class {
    /// Ends the URL unconditionally: whitespace, a control, or an ASCII character
    /// RFC 3986 forbids in one.
    Hard,
    /// Legal inside a URL, but also how prose punctuates, so it may not end one.
    Soft,
    /// Opens a bracket pair the URL may go on to close.
    Open,
    /// Closes the pair opened by the character it carries.
    Close(char),
    /// An ordinary URL character, which may end one.
    Url,
}

/// Classify one character. The bracket and soft arms come first, so the RFC 3986
/// membership test below only has to speak for what is left.
fn classify(c: char) -> Class {
    match c {
        // `[` and `]` are not decoration: they fence an IPv6 literal host
        // (`http://[::1]:8080/`), so the same stack that balances `(album)` balances
        // those too.
        '(' | '[' => Class::Open,
        ')' => Class::Close('('),
        ']' => Class::Close('['),
        // Sentence punctuation that is legal in a URL yet usually ends the sentence
        // rather than the path: the classic `see http://x.com/a.`
        '.' | ',' | ':' | ';' | '!' | '?' | '\'' => Class::Soft,
        c if c.is_whitespace() || c.is_control() => Class::Hard,
        c if c.is_ascii_alphanumeric() => Class::Url,
        // The rest of RFC 3986's ASCII: unreserved, gen-delims, sub-delims, and the
        // `%` of a percent-escape.
        '-' | '_' | '~' | '/' | '#' | '@' | '$' | '&' | '*' | '+' | '=' | '%' => Class::Url,
        // Every other ASCII character (`"`, `<`, `>`, `\`, `^`, backtick, `{`, `|`,
        // `}`) is illegal in a URL, so it ends one. This is what makes the common ways
        // of quoting a link — `<http://x>`, `"http://x"` — come out clean with no
        // special case for them.
        c if c.is_ascii() => Class::Hard,
        // Non-ASCII: an IRI may carry it (an accented Wikipedia title), and it is
        // never prose punctuation we would have to guess about.
        _ => Class::Url,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every URL in `text`, as [`urls`] yields them.
    fn all(text: &str) -> Vec<&str> {
        urls(text).map(|r| &text[r]).collect()
    }

    #[test]
    fn a_bare_url_is_found_whole() {
        assert_eq!(all("https://example.com/a/b"), ["https://example.com/a/b"]);
    }

    #[test]
    fn trailing_sentence_punctuation_is_left_to_the_sentence() {
        // Only punctuation follows, so none of it is part of the link.
        assert_eq!(
            all("go to https://example.com/a."),
            ["https://example.com/a"]
        );
        assert_eq!(
            all("https://example.com/a, and then"),
            ["https://example.com/a"]
        );
        assert_eq!(
            all("really? https://example.com/a?!"),
            ["https://example.com/a"]
        );
        assert_eq!(all("https://example.com/a..."), ["https://example.com/a"]);
    }

    #[test]
    fn punctuation_inside_a_url_is_kept() {
        // The same characters, with something other than punctuation after them, are
        // ordinary URL bytes: a dotted path, a query string, a port, a fragment.
        assert_eq!(
            all("https://example.com/a.b.c"),
            ["https://example.com/a.b.c"]
        );
        assert_eq!(
            all("https://example.com/s?q=a,b&n=1#top"),
            ["https://example.com/s?q=a,b&n=1#top"]
        );
        // A dev server's URL has no dot in the host at all, which is why none is
        // required.
        assert_eq!(
            all("serving on http://localhost:3000/ ok"),
            ["http://localhost:3000/"]
        );
    }

    #[test]
    fn a_url_closes_the_brackets_it_opened() {
        assert_eq!(
            all("https://en.wikipedia.org/wiki/Cure_(album) rocks"),
            ["https://en.wikipedia.org/wiki/Cure_(album)"]
        );
        // Nested, and with the sentence's own period after it.
        assert_eq!(
            all("https://example.com/a(b(c)d)e."),
            ["https://example.com/a(b(c)d)e"]
        );
        // An IPv6 literal host is bracket-balanced by the same stack.
        assert_eq!(all("http://[::1]:8080/x"), ["http://[::1]:8080/x"]);
    }

    #[test]
    fn a_bracket_the_url_did_not_open_belongs_to_the_prose() {
        assert_eq!(
            all("(see https://example.com/a)"),
            ["https://example.com/a"]
        );
        // The parenthetical closes *and* the sentence ends.
        assert_eq!(all("(https://example.com/a)."), ["https://example.com/a"]);
        // A bracket the URL opened but never closed takes everything after it out of
        // the link: half a bracketed path is not a path.
        assert_eq!(all("https://example.com/a(b c"), ["https://example.com/a"]);
    }

    #[test]
    fn the_usual_ways_of_quoting_a_link_come_out_clean() {
        // `<`, `>`, and `"` cannot appear in a URL, so they end one with no special
        // case: this is RFC 3986 doing the work. The `<https://x>` form is also how
        // markdown writes an autolink, so it falls out for free.
        assert_eq!(all("<https://example.com/a>"), ["https://example.com/a"]);
        assert_eq!(all("\"https://example.com/a\""), ["https://example.com/a"]);
        assert_eq!(all("'https://example.com/a'"), ["https://example.com/a"]);
    }

    #[test]
    fn a_scheme_must_start_a_token() {
        // The tail of a longer word only looks like a scheme.
        assert!(all("xhttps://example.com").is_empty());
        assert!(all("not-a-mailto:x@y.com").is_empty());
        assert!(all("1.https://example.com").is_empty());
        // A delimiter before it is a boundary, so this one is real.
        assert_eq!(all("url=https://example.com/a"), ["https://example.com/a"]);
    }

    #[test]
    fn only_schemes_the_browser_can_open_are_detected() {
        // The detect set is the open set: never decorate what would then be refused,
        // and never make a hostile scheme reachable in the first place.
        assert!(all("javascript:alert(1)").is_empty());
        assert!(all("data:text/html,<h1>x").is_empty());
        assert!(all("ftp://example.com/a").is_empty());
        assert!(all("see example.com/a").is_empty());
        assert_eq!(all("mail me at mailto:a@b.com ok"), ["mailto:a@b.com"]);
        assert_eq!(all("file:///etc/hosts"), ["file:///etc/hosts"]);
        // Whatever is detected, the browser will in fact launch.
        for url in all("https://a.com http://b.com file:///c mailto:d@e.com") {
            assert!(browser::can_open(url), "{url} is detected but not openable");
        }
    }

    #[test]
    fn a_scheme_with_no_host_is_not_a_link() {
        // `extent` collapses to the scheme itself, so nothing is yielded.
        assert!(all("https://").is_empty());
        assert!(all("say https:// then").is_empty());
        assert!(all("mailto:").is_empty());
        assert!(all("https:// and mailto: alone").is_empty());
    }

    #[test]
    fn every_url_in_a_line_is_yielded_left_to_right() {
        assert_eq!(
            all("a https://one.com/x b http://two.com/y c"),
            ["https://one.com/x", "http://two.com/y"]
        );
        assert!(all("no links here at all").is_empty());
        assert!(all("").is_empty());
        // A scheme nested in a query string is swallowed by the URL that contains it,
        // not reported a second time on its own.
        assert_eq!(
            all("https://a.com/r?to=http://b.com/x"),
            ["https://a.com/r?to=http://b.com/x"]
        );
    }

    #[test]
    fn an_iri_keeps_its_non_ascii() {
        assert_eq!(
            all("https://de.wikipedia.org/wiki/Grüße ok"),
            ["https://de.wikipedia.org/wiki/Grüße"]
        );
        // A multi-byte character before a scheme is a boundary, and one that lands
        // mid-character never splits it.
        assert_eq!(all("så https://example.com/a"), ["https://example.com/a"]);
    }

    #[test]
    fn a_url_at_either_edge_of_the_line_is_whole() {
        assert_eq!(
            all("https://example.com/a and text"),
            ["https://example.com/a"]
        );
        assert_eq!(
            all("text and https://example.com/a"),
            ["https://example.com/a"]
        );
    }

    #[test]
    fn control_characters_end_a_url() {
        // Pasted text can run a URL up against a control byte; it must never be
        // swallowed into the link.
        assert_eq!(
            all("https://example.com/a\u{7}b"),
            ["https://example.com/a"]
        );
        assert_eq!(all("https://example.com/a\tb"), ["https://example.com/a"]);
    }

    #[test]
    fn absurd_bracket_nesting_is_refused_rather_than_trusted() {
        // Past MAX_NESTING the scan stops believing this is a URL and keeps only what
        // was safe before the nesting ran away. Bounded work on hostile input.
        let deep = format!("https://example.com/{}x", "(".repeat(MAX_NESTING + 4));
        assert_eq!(all(&deep), ["https://example.com/"]);
    }
}
