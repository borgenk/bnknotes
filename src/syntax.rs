//! Code-block syntax highlighting: a small line-local lexer shared across a set
//! of languages. A fenced block's info string ([`Lang::from_info`]) picks the
//! language; [`lex_into`] then splits each code line into colored token spans
//! (keyword, string, comment, number, plain) that flow through the ordinary run
//! pipeline as [`markdown::SpanKind::Code`].
//!
//! The lexer is line-local: it re-lexes each line from a clean state and does not
//! thread multi-line constructs across lines. Line comments, single-line strings,
//! numbers, and keywords, the bulk of what a reader scans for, all resolve on one
//! line; a block comment or string that opens without closing colors only to the
//! line end and the next line starts fresh. That keeps the cross-line carry to
//! just the language, with no lexer state to serialize through the line index.

use crate::markdown::{self, CodeToken};

/// A highlightable language, selected from a fenced block's info string. The u8
/// discriminant is packed into the line index's fence carry, so keep it stable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Lang {
    /// No highlighting: the whole line is one plain code span.
    PlainText = 0,
    Rust = 1,
    JavaScript = 2,
    Python = 3,
    Json = 4,
    Shell = 5,
    Toml = 6,
    Yaml = 7,
}

impl Lang {
    /// Recover a language from the u8 packed into the fence carry; an unknown
    /// value reads as plain text (no highlighting).
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Lang::Rust,
            2 => Lang::JavaScript,
            3 => Lang::Python,
            4 => Lang::Json,
            5 => Lang::Shell,
            6 => Lang::Toml,
            7 => Lang::Yaml,
            _ => Lang::PlainText,
        }
    }

    /// Map a fence info string (`rust`, `ts`, `py`, …) to a language. Only the
    /// first token matters; the rest (e.g. `rust,ignore`) is ignored. An unknown
    /// or empty tag is [`Lang::PlainText`], so the block renders as flat code.
    pub fn from_info(info: &str) -> Self {
        let tag = info
            .trim()
            .split(|c: char| c.is_whitespace() || c == ',')
            .next()
            .unwrap_or("");
        match tag.to_ascii_lowercase().as_str() {
            "rust" | "rs" => Lang::Rust,
            "js" | "javascript" | "jsx" | "mjs" | "cjs" | "ts" | "typescript" | "tsx" => {
                Lang::JavaScript
            }
            "py" | "python" => Lang::Python,
            "json" | "jsonc" => Lang::Json,
            "sh" | "bash" | "zsh" | "shell" | "console" => Lang::Shell,
            "toml" => Lang::Toml,
            "yaml" | "yml" => Lang::Yaml,
            _ => Lang::PlainText,
        }
    }
}

/// One language's lexical rules: what opens a comment or string, and the keyword
/// set. Numbers and identifiers are recognized generically.
struct LangSpec {
    /// Prefixes that begin a rest-of-line comment (`//`, `#`).
    line_comments: &'static [&'static str],
    /// The open/close pair of a block comment, if the language has one.
    block_comment: Option<(&'static str, &'static str)>,
    /// The quote characters that open a string literal.
    strings: &'static [char],
    /// Reserved words, colored as keywords when they stand as whole identifiers.
    keywords: &'static [&'static str],
}

/// The rules for `lang`, or `None` for [`Lang::PlainText`] (draw flat code).
fn spec(lang: Lang) -> Option<&'static LangSpec> {
    Some(match lang {
        Lang::PlainText => return None,
        Lang::Rust => &RUST,
        Lang::JavaScript => &JAVASCRIPT,
        Lang::Python => &PYTHON,
        Lang::Json => &JSON,
        Lang::Shell => &SHELL,
        Lang::Toml => &TOML,
        Lang::Yaml => &YAML,
    })
}

/// Tokenize one code `line` (line-relative) into colored spans, appending them to
/// `out` as [`markdown::SpanKind::Code`] spans that tile `[0, line.len())`. An
/// unknown language yields a single plain span, matching the pre-highlight look.
pub fn lex_into(lang: Lang, line: &str, out: &mut Vec<markdown::Span>) {
    let Some(spec) = spec(lang) else {
        if !line.is_empty() {
            push(out, 0, line.len(), CodeToken::Plain);
        }
        return;
    };
    let b = line.as_bytes();
    let n = b.len();
    let mut i = 0;
    // Start of the run of not-yet-classified bytes that become one plain span.
    let mut plain = 0;

    while i < n {
        // A line comment swallows the rest of the line.
        if spec
            .line_comments
            .iter()
            .any(|p| starts_at(b, i, p.as_bytes()))
        {
            flush(out, plain, i);
            push(out, i, n, CodeToken::Comment);
            return;
        }
        // A block comment runs to its close, or to the line end if it does not
        // close here (line-local: the next line starts fresh).
        if let Some((open, close)) = spec.block_comment {
            if starts_at(b, i, open.as_bytes()) {
                flush(out, plain, i);
                let end = find_from(b, i + open.len(), close.as_bytes())
                    .map(|p| p + close.len())
                    .unwrap_or(n);
                push(out, i, end, CodeToken::Comment);
                i = end;
                plain = i;
                continue;
            }
        }
        let c = b[i];
        // A string literal to its closing quote (or the line end if unterminated).
        if (c as char).is_ascii() && spec.strings.contains(&(c as char)) {
            flush(out, plain, i);
            let end = scan_string(b, i, c);
            push(out, i, end, CodeToken::Str);
            i = end;
            plain = i;
            continue;
        }
        // A number at a token boundary (a digit inside an identifier stays part of
        // the identifier, handled below).
        if c.is_ascii_digit() && (i == 0 || !is_ident_byte(b[i - 1])) {
            flush(out, plain, i);
            let end = scan_number(b, i);
            push(out, i, end, CodeToken::Number);
            i = end;
            plain = i;
            continue;
        }
        // An identifier; a keyword recolors, anything else stays in the plain run.
        if is_ident_start(c) {
            let start = i;
            let mut j = i + 1;
            while j < n && is_ident_byte(b[j]) {
                j += 1;
            }
            if spec.keywords.contains(&&line[start..j]) {
                flush(out, plain, start);
                push(out, start, j, CodeToken::Keyword);
                plain = j;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    flush(out, plain, n);
}

/// Append `[start, end)` as a code token span; empty ranges are dropped.
fn push(out: &mut Vec<markdown::Span>, start: usize, end: usize, token: CodeToken) {
    if end > start {
        out.push(markdown::Span {
            start,
            end,
            kind: markdown::SpanKind::Code(token),
        });
    }
}

/// Flush the pending plain run `[from, to)` as one plain code span.
fn flush(out: &mut Vec<markdown::Span>, from: usize, to: usize) {
    push(out, from, to, CodeToken::Plain);
}

/// Whether `pat` occurs in `b` at `i`.
fn starts_at(b: &[u8], i: usize, pat: &[u8]) -> bool {
    b.len() - i >= pat.len() && b[i..i + pat.len()] == *pat
}

/// The index of `pat` in `b` at or after `from`, if any.
fn find_from(b: &[u8], from: usize, pat: &[u8]) -> Option<usize> {
    if pat.is_empty() || from > b.len() {
        return None;
    }
    (from..=b.len().saturating_sub(pat.len())).find(|&i| b[i..i + pat.len()] == *pat)
}

/// The byte just past a string opened by `quote` at `open`, honoring backslash
/// escapes; the line end if the string does not close on this line.
fn scan_string(b: &[u8], open: usize, quote: u8) -> usize {
    let n = b.len();
    let mut i = open + 1;
    while i < n {
        if b[i] == b'\\' {
            i += 2;
            continue;
        }
        if b[i] == quote {
            return i + 1;
        }
        i += 1;
    }
    n
}

/// The byte just past a numeric literal starting at `start`. Generous: it takes a
/// run of digits, letters, dots, and underscores, so hex, floats, exponents, and
/// digit separators all read as one number.
fn scan_number(b: &[u8], start: usize) -> usize {
    let n = b.len();
    let mut i = start;
    while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_') {
        i += 1;
    }
    i
}

fn is_ident_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_'
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

const RUST: LangSpec = LangSpec {
    line_comments: &["//"],
    block_comment: Some(("/*", "*/")),
    // Only double quotes: a single quote in Rust also opens a lifetime (`'a`), so
    // treating it as a string start would mis-color the common case.
    strings: &['"'],
    keywords: &[
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait",
        "true", "type", "unsafe", "use", "where", "while",
    ],
};

const JAVASCRIPT: LangSpec = LangSpec {
    line_comments: &["//"],
    block_comment: Some(("/*", "*/")),
    strings: &['"', '\'', '`'],
    keywords: &[
        "abstract",
        "as",
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "from",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "interface",
        "let",
        "new",
        "null",
        "of",
        "private",
        "protected",
        "public",
        "readonly",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "type",
        "typeof",
        "undefined",
        "var",
        "void",
        "while",
        "yield",
    ],
};

const PYTHON: LangSpec = LangSpec {
    line_comments: &["#"],
    block_comment: None,
    strings: &['"', '\''],
    keywords: &[
        "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
        "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import",
        "in", "is", "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True",
        "try", "while", "with", "yield",
    ],
};

const JSON: LangSpec = LangSpec {
    line_comments: &[],
    block_comment: None,
    strings: &['"'],
    keywords: &["true", "false", "null"],
};

const SHELL: LangSpec = LangSpec {
    line_comments: &["#"],
    block_comment: None,
    strings: &['"', '\''],
    keywords: &[
        "if", "then", "elif", "else", "fi", "for", "while", "until", "do", "done", "case", "esac",
        "function", "in", "select", "return", "break", "continue", "local", "export", "readonly",
        "declare",
    ],
};

const TOML: LangSpec = LangSpec {
    line_comments: &["#"],
    block_comment: None,
    strings: &['"', '\''],
    keywords: &["true", "false"],
};

const YAML: LangSpec = LangSpec {
    line_comments: &["#"],
    block_comment: None,
    strings: &['"', '\''],
    keywords: &["true", "false", "null", "yes", "no", "on", "off"],
};

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(lang: Lang, line: &str) -> Vec<(String, CodeToken)> {
        let mut out = Vec::new();
        lex_into(lang, line, &mut out);
        out.iter()
            .map(|s| {
                let tok = match s.kind {
                    markdown::SpanKind::Code(t) => t,
                    _ => panic!("code span expected"),
                };
                (line[s.start..s.end].to_string(), tok)
            })
            .collect()
    }

    #[test]
    fn spans_tile_the_whole_line() {
        for (lang, line) in [
            (Lang::Rust, "let x = 3; // hi"),
            (Lang::Python, "def f(): return 'a'"),
            (Lang::Json, "{\"k\": 12}"),
            (Lang::PlainText, "anything at all"),
        ] {
            let mut out = Vec::new();
            lex_into(lang, line, &mut out);
            let mut at = 0;
            for s in &out {
                assert_eq!(s.start, at, "spans are contiguous and in order");
                at = s.end;
            }
            assert_eq!(at, line.len(), "spans cover the whole line");
        }
    }

    #[test]
    fn rust_keywords_strings_numbers_and_comments() {
        let t = toks(Lang::Rust, "let s = \"hi\"; // note");
        assert!(t.contains(&("let".into(), CodeToken::Keyword)));
        assert!(t.contains(&("\"hi\"".into(), CodeToken::Str)));
        assert!(t
            .iter()
            .any(|(txt, k)| txt == "// note" && *k == CodeToken::Comment));
        let n = toks(Lang::Rust, "x = 0x1F");
        assert!(n.contains(&("0x1F".into(), CodeToken::Number)));
    }

    #[test]
    fn a_single_quote_is_not_a_string_in_rust() {
        // A lifetime must not read as an unterminated string.
        let t = toks(Lang::Rust, "fn f<'a>() {}");
        assert!(
            !t.iter().any(|(_, k)| *k == CodeToken::Str),
            "no string span for a lifetime"
        );
    }

    #[test]
    fn a_digit_inside_an_identifier_is_not_a_number() {
        let t = toks(Lang::Rust, "let var1 = x2");
        assert!(!t
            .iter()
            .any(|(txt, k)| txt == "1" && *k == CodeToken::Number));
        assert!(!t.iter().any(|(_, k)| *k == CodeToken::Number));
    }

    #[test]
    fn unknown_language_is_one_plain_span() {
        let mut out = Vec::new();
        lex_into(Lang::from_info("brainfuck"), "++[>+<]", &mut out);
        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0].kind,
            markdown::SpanKind::Code(CodeToken::Plain)
        ));
    }

    #[test]
    fn info_string_aliases_resolve() {
        assert_eq!(Lang::from_info("rs"), Lang::Rust);
        assert_eq!(Lang::from_info("TypeScript"), Lang::JavaScript);
        assert_eq!(Lang::from_info("rust,ignore"), Lang::Rust);
        assert_eq!(Lang::from_info(""), Lang::PlainText);
    }
}
