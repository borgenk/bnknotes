//! Structural golden tests for markdown rendering. Each corpus document under
//! testdata/markdown is laid out through the real pipeline and reduced to a
//! stable, readable snapshot of its structure: block kind, styling, inline runs,
//! table grid, task state. The snapshot is compared against a checked-in golden.
//!
//! The snapshot records what the layout *decides*, never where it *places*
//! things. Every pixel coordinate (baselines, run x, indents, wrap points) is
//! omitted, because those come from system font metrics and the FreeType version,
//! which differ between machines. What is left, block classification, gutters,
//! heading levels, emphasis spans, syntax-token classes, table alignment, depends
//! only on the input, so a golden authored on one machine holds on any other.
//! Pixel-exact rendering is a separate concern for an image-diff tier.
//!
//! Lines wrap at a width wide enough that nothing soft-wraps, so one visual line
//! is one logical line and the snapshot reflects document structure rather than
//! the layout's metric-driven line breaks.
//!
//! Update the goldens after an intended change with `BLESS=1 cargo test golden`.

use std::fs;
use std::path::{Path, PathBuf};

use crate::layout::{Gutter, Layout, TableRow, VisualLine, NO_REVEAL};
use crate::markdown::{self, Align, CodeToken, Emphasis, SpanKind};
use crate::platform::freetype::Fonts;
use crate::rope::Rope;

/// Body text size the snapshot lays out at. Fixed so the heading sizes derived
/// from it stay stable in the golden. It is a chosen scale, independent of any
/// font's metrics.
const BODY: u32 = 32;

/// Wide enough that no line soft-wraps, so a visual line is a logical line.
const WRAP_WIDTH: i32 = 1 << 20;

/// Lay `text` out through the real pipeline and format its structure as a stable,
/// font-independent snapshot.
fn snapshot(text: &str) -> String {
    let fonts = Fonts::new(&markdown::all_sizes(BODY)).expect("a system font to lay out with");
    let rope = Rope::from_str(text);
    let layout = Layout::build(&fonts, &rope, 0, 0, WRAP_WIDTH, BODY, NO_REVEAL);

    let mut out = String::new();
    for line in layout.lines() {
        if let Some(table) = &line.table {
            out.push_str(&table_row(&rope, table));
        } else {
            out.push_str(&text_line(&rope, &layout, line));
        }
        out.push('\n');
    }
    out
}

/// One non-table visual line: its block tag, then each inline run as
/// `label"text"`, a leading `~` marking a run the live preview hides.
fn text_line(rope: &Rope, layout: &Layout, line: &VisualLine) -> String {
    let mut tag = block_tag(line);
    if line.size != BODY {
        tag.push_str(&format!("@{}", line.size));
    }
    let mut s = format!("{tag:<8} |");

    // An image line draws a picture, not runs; show its source so a broken or
    // restyled `![alt](path)` is visible in the diff.
    if line.image.is_some() {
        s.push(' ');
        s.push_str(&quote(&rope.byte_slice(line.start..line.end)));
        return s;
    }

    for run in layout.run_slice(line) {
        let hidden = if run.visible { "" } else { "~" };
        let text = quote(&rope.byte_slice(run.start..run.end));
        s.push_str(&format!(" {hidden}{}{text}", run_label(run.kind)));
    }
    s
}

/// One table row: its role, then each cell as `<align>"text"`.
fn table_row(rope: &Rope, row: &TableRow) -> String {
    let role = if row.header {
        "TABLE.head"
    } else {
        "TABLE.body"
    };
    let mut s = format!("{role} |");
    for (col, cell) in row.columns.iter().zip(&row.cells) {
        let text = quote(&rope.byte_slice(cell.start..cell.end));
        s.push_str(&format!(" {}{text} |", align_label(col.align)));
    }
    s
}

/// The block classification of a line, in precedence order: a collapsed rule and
/// an image box outrank the gutter, a heading and a code line outrank list chrome.
fn block_tag(v: &VisualLine) -> String {
    if v.rule {
        return "HR".into();
    }
    if v.image.is_some() {
        return "IMG".into();
    }
    if let Some(level) = v.heading {
        return format!("H{level}");
    }
    if v.code_block {
        return "CODE".into();
    }
    match v.gutter {
        Gutter::Bullet => "UL".into(),
        Gutter::Checkbox { checked } => if checked { "TASK[x]" } else { "TASK[ ]" }.into(),
        // A continuation carries the bar but no marker; the first line carries both.
        Gutter::QuoteBar { marker } => if marker { "QUOTE" } else { "QUOTE~" }.into(),
        // A wrapped task line keeps `task` set but drops the checkbox gutter.
        Gutter::None if v.task.is_some() => "TASK~".into(),
        Gutter::None => "P".into(),
    }
}

/// A short label for a run's kind: emphasis for prose, `c.*` for a code-block
/// token, `mark` for a delimiter the live preview may hide.
fn run_label(kind: SpanKind) -> &'static str {
    match kind {
        SpanKind::Text(Emphasis::Regular) => "t",
        SpanKind::Text(Emphasis::Bold) => "b",
        SpanKind::Text(Emphasis::Italic) => "i",
        SpanKind::Text(Emphasis::BoldItalic) => "bi",
        SpanKind::Text(Emphasis::Code) => "code",
        SpanKind::Text(Emphasis::Link) => "link",
        SpanKind::Text(Emphasis::Strikethrough) => "s",
        SpanKind::Marker => "mark",
        SpanKind::Code(CodeToken::Plain) => "c",
        SpanKind::Code(CodeToken::Keyword) => "c.kw",
        SpanKind::Code(CodeToken::Str) => "c.str",
        SpanKind::Code(CodeToken::Comment) => "c.com",
        SpanKind::Code(CodeToken::Number) => "c.num",
    }
}

fn align_label(a: Align) -> &'static str {
    match a {
        Align::Left => "L",
        Align::Center => "C",
        Align::Right => "R",
    }
}

/// Quote a run's text and escape the few characters that would blur one snapshot
/// line into the next.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The corpus directory, resolved against the crate root so the test runs from
/// any working directory.
fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/markdown")
}

/// The `.md` inputs, sorted for a stable failure order.
fn corpus_cases(dir: &Path) -> Vec<PathBuf> {
    let mut cases: Vec<PathBuf> = fs::read_dir(dir)
        .expect("testdata/markdown exists")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    cases.sort();
    cases
}

/// The 1-based line where `want` and `got` first differ, with both texts.
fn first_diff(want: &str, got: &str) -> Option<(usize, String, String)> {
    let mut w = want.lines();
    let mut g = got.lines();
    let mut n = 0;
    loop {
        n += 1;
        match (w.next(), g.next()) {
            (Some(a), Some(b)) if a == b => continue,
            (None, None) => return None,
            (a, b) => {
                return Some((
                    n,
                    a.unwrap_or("<end of golden>").to_string(),
                    b.unwrap_or("<end of output>").to_string(),
                ))
            }
        }
    }
}

#[test]
fn markdown_layout_goldens() {
    let dir = corpus_dir();
    let cases = corpus_cases(&dir);
    assert!(
        !cases.is_empty(),
        "no corpus documents in {}",
        dir.display()
    );

    let bless = std::env::var_os("BLESS").is_some();
    let mut failures = Vec::new();

    for md in cases {
        let name = md.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let input = fs::read_to_string(&md).expect("read corpus document");
        let got = snapshot(&input);
        let snap = md.with_extension("snap");

        if bless {
            fs::write(&snap, &got).expect("write golden");
            continue;
        }

        let want = match fs::read_to_string(&snap) {
            Ok(w) => w,
            Err(_) => {
                failures.push(format!(
                    "{name}: no golden yet (run BLESS=1 cargo test golden)"
                ));
                continue;
            }
        };
        if got != want {
            let detail = match first_diff(&want, &got) {
                Some((line, w, g)) => {
                    format!("{name}: line {line} differs\n  golden: {w}\n  output: {g}")
                }
                None => format!("{name}: differs"),
            };
            failures.push(detail);
        }
    }

    assert!(
        failures.is_empty(),
        "markdown golden mismatch (run BLESS=1 cargo test golden to update):\n{}",
        failures.join("\n")
    );
}
