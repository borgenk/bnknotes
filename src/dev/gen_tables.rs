//! `--gen-tables`: regenerate the Unicode property tables that `grapheme.rs` and
//! `link.rs` binary-search, from the UCD data files vendored in `ucd/`.
//!
//! ```text
//!   ucd/GraphemeBreakProperty.txt   ─┐
//!   ucd/emoji-data.txt              ─┼─▶ src/platform/grapheme_tables.rs
//!   ucd/DerivedCoreProperties.txt   ─┘   (GCB, EXTENDED_PICTOGRAPHIC, INCB)
//!   ucd/DerivedGeneralCategory.txt  ─┬─▶ src/platform/link_tables.rs
//!   ucd/BidiBrackets.txt            ─┘   (PUNCTUATION, OPENING_BRACKETS,
//!                                         CLOSING_BRACKETS)
//! ```
//!
//! The inputs are vendored and version-pinned, so a regeneration is a pure,
//! offline, deterministic function of the tree: `--gen-tables` reproduces the
//! committed file byte-for-byte, and the output is validated against the official
//! `ucd/GraphemeBreakTest.txt` conformance suite (see `grapheme.rs`). The emitted
//! source is already rustfmt-canonical, so no formatting pass is needed.
//!
//! To bump the Unicode version: `make deps` (installs the `unicode-character-database`
//! package), refresh `ucd/` from `/usr/share/unicode/`, then `--gen-tables`.

use crate::platform::error::{Error, Result};
use std::fmt::Write as _;

const GRAPHEME_TABLES: &str = "src/platform/grapheme_tables.rs";
const LINK_TABLES: &str = "src/platform/link_tables.rs";

pub fn run(_args: &[String]) -> Result<()> {
    // Every vendored file is one Unicode version; read it off the one whose header
    // carries it (emoji-data.txt's does not).
    let version = read_version(&ucd("GraphemeBreakProperty.txt"))?;
    std::fs::write(GRAPHEME_TABLES, gen_grapheme(&version)?)
        .map_err(|e| Error::msg(format!("write {GRAPHEME_TABLES}: {e}")))?;
    std::fs::write(LINK_TABLES, gen_link(&version)?)
        .map_err(|e| Error::msg(format!("write {LINK_TABLES}: {e}")))?;
    println!("wrote {GRAPHEME_TABLES} and {LINK_TABLES} from Unicode {version}");
    Ok(())
}

/// The tables link.rs reads, all non-ASCII (ASCII is classified by hand there).
/// The bracket pairs (Bidi_Paired_Bracket_Type) balance inside a URL the way ASCII
/// parentheses do. PUNCTUATION is the rest by General_Category: an opening or
/// closing mark that pairs with nothing (Ps, Pe) ends a URL, and quotation marks and
/// other punctuation (Pi, Pf, Po) may not end one.
fn gen_link(version: &str) -> Result<String> {
    let dgc = read(&ucd("DerivedGeneralCategory.txt"))?;
    let mut punctuation = Vec::new();
    for (lo, hi, f) in dgc.lines().filter_map(ucd_fields) {
        let tag = match f.first().copied() {
            Some("Ps" | "Pe") => "Punct::Unpaired",
            Some("Pi" | "Pf" | "Po") => "Punct::Prose",
            _ => continue,
        };
        let lo = lo.max(0x80);
        if lo <= hi {
            punctuation.push((lo, hi, tag));
        }
    }
    let brackets = read(&ucd("BidiBrackets.txt"))?;
    let mut opening = Vec::new();
    let mut closing = Vec::new();
    for (bracket, _, f) in brackets.lines().filter_map(ucd_fields) {
        let Some(pair) = f.first().and_then(|h| hex(h)) else {
            continue;
        };
        if bracket < 0x80 {
            continue;
        }
        match f.get(1).copied() {
            Some("o") => opening.push(bracket),
            Some("c") => closing.push((bracket, pair)),
            _ => {}
        }
    }
    opening.sort_unstable();
    closing.sort_unstable();

    let mut out = String::new();
    header(
        &mut out,
        version,
        &["ucd/DerivedGeneralCategory.txt", "ucd/BidiBrackets.txt"],
    );
    let _ = writeln!(
        out,
        "/// Every non-ASCII punctuation range, tagged with what it does to a bare URL."
    );
    emit_tagged(&mut out, "PUNCTUATION", "Punct", &merge_tagged(punctuation));
    let _ = writeln!(out);
    let _ = writeln!(out, "/// Every non-ASCII opening bracket that has a pair.");
    let _ = writeln!(out, "static OPENING_BRACKETS: &[u32] = &[");
    for bracket in &opening {
        let _ = writeln!(out, "    0x{bracket:04X},");
    }
    let _ = writeln!(out, "];");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Every non-ASCII closing bracket, with the opening bracket it closes."
    );
    let _ = writeln!(out, "static CLOSING_BRACKETS: &[(u32, u32)] = &[");
    for (close, open) in &closing {
        let _ = writeln!(out, "    (0x{close:04X}, 0x{open:04X}),");
    }
    let _ = writeln!(out, "];");
    Ok(out)
}

/// The path of a vendored UCD file.
fn ucd(name: &str) -> String {
    format!("ucd/{name}")
}

fn read(path: &str) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| Error::msg(format!("read {path}: {e}")))
}

/// The `X.Y.Z` version from a UCD file's first header line (`# GraphemeBreakProperty-18.0.0.txt`).
fn read_version(path: &str) -> Result<String> {
    let text = read(path)?;
    let first = text.lines().next().unwrap_or_default();
    parse_version(first).ok_or_else(|| Error::msg(format!("no version in {path} header")))
}

/// The first `digits.digits.digits` run in `line`, if any.
fn parse_version(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
            i += 1;
        }
        // A run like `18.0.0` in `GraphemeBreakProperty-18.0.0.txt` picks up the `.`
        // before the extension; drop any trailing dots before validating the parts.
        let token = line[start..i].trim_end_matches('.');
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() == 3
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        {
            return Some(token.to_string());
        }
    }
    None
}

/// One UCD data line's code range and the `;`-separated property fields after it
/// (comment stripped, each trimmed). `None` for a blank or comment line.
fn ucd_fields(line: &str) -> Option<(u32, u32, Vec<&str>)> {
    let content = line.split('#').next().unwrap_or_default().trim();
    if content.is_empty() {
        return None;
    }
    let mut parts = content.split(';');
    let (lo, hi) = parse_range(parts.next()?.trim())?;
    Some((lo, hi, parts.map(str::trim).collect()))
}

fn parse_range(code: &str) -> Option<(u32, u32)> {
    match code.split_once("..") {
        Some((lo, hi)) => Some((hex(lo)?, hex(hi)?)),
        None => {
            let v = hex(code)?;
            Some((v, v))
        }
    }
}

fn hex(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim(), 16).ok()
}

/// Sort 2-tuple ranges and coalesce overlapping or adjacent ones.
fn merge(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (lo, hi) in ranges {
        match out.last_mut() {
            Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// Sort value-tagged ranges by code point and merge adjacent ranges sharing a tag.
fn merge_tagged(mut ranges: Vec<(u32, u32, &'static str)>) -> Vec<(u32, u32, &'static str)> {
    ranges.sort_by_key(|r| r.0);
    let mut out: Vec<(u32, u32, &'static str)> = Vec::new();
    for (lo, hi, tag) in ranges {
        match out.last_mut() {
            Some(last) if last.2 == tag && lo <= last.1.saturating_add(1) => {
                last.1 = last.1.max(hi)
            }
            _ => out.push((lo, hi, tag)),
        }
    }
    out
}

fn gen_grapheme(version: &str) -> Result<String> {
    let gcb = gcb_ranges()?;
    let ext = extended_pictographic()?;
    let incb = incb_ranges()?;

    let mut out = String::new();
    header(
        &mut out,
        version,
        &[
            "ucd/GraphemeBreakProperty.txt",
            "ucd/emoji-data.txt",
            "ucd/DerivedCoreProperties.txt",
        ],
    );
    let _ = writeln!(
        out,
        "/// The Unicode version these tables were generated from."
    );
    let _ = writeln!(out, "#[allow(dead_code)]");
    let _ = writeln!(out, "const UNICODE_VERSION: &str = \"{version}\";");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Every assigned Grapheme_Cluster_Break range; the rest is `Other`."
    );
    emit_tagged(&mut out, "GCB_RANGES", "Gcb", &gcb);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Every Extended_Pictographic range (emoji), for rule GB11."
    );
    emit_ranges(&mut out, "EXTENDED_PICTOGRAPHIC", &ext);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "/// Every Indic_Conjunct_Break range; the rest is `None`. For rule GB9c."
    );
    emit_tagged(&mut out, "INCB_RANGES", "Incb", &incb);
    Ok(out)
}

/// The `Grapheme_Cluster_Break` ranges, each tagged with its [`Gcb`] variant.
fn gcb_ranges() -> Result<Vec<(u32, u32, &'static str)>> {
    let text = read(&ucd("GraphemeBreakProperty.txt"))?;
    let mut ranges = Vec::new();
    for (lo, hi, f) in text.lines().filter_map(ucd_fields) {
        let value = f.first().copied().unwrap_or_default();
        let variant = gcb_variant(value)
            .ok_or_else(|| Error::msg(format!("unknown Grapheme_Cluster_Break value {value:?}")))?;
        ranges.push((lo, hi, variant));
    }
    Ok(merge_tagged(ranges))
}

fn gcb_variant(value: &str) -> Option<&'static str> {
    Some(match value {
        "Control" => "Gcb::Control",
        "CR" => "Gcb::Cr",
        "LF" => "Gcb::Lf",
        "Extend" => "Gcb::Extend",
        "ZWJ" => "Gcb::Zwj",
        "Regional_Indicator" => "Gcb::RegionalIndicator",
        "Prepend" => "Gcb::Prepend",
        "SpacingMark" => "Gcb::SpacingMark",
        "L" => "Gcb::L",
        "V" => "Gcb::V",
        "T" => "Gcb::T",
        "LV" => "Gcb::Lv",
        "LVT" => "Gcb::Lvt",
        _ => return None,
    })
}

/// The `Indic_Conjunct_Break` ranges, each tagged with its [`Incb`] variant. The
/// `None` value carries no rule weight and is left out (the table default).
fn incb_ranges() -> Result<Vec<(u32, u32, &'static str)>> {
    let text = read(&ucd("DerivedCoreProperties.txt"))?;
    let mut ranges = Vec::new();
    for (lo, hi, f) in text.lines().filter_map(ucd_fields) {
        // The catalog form: `<code> ; InCB ; <value>`.
        if f.first() != Some(&"InCB") {
            continue;
        }
        let value = f.get(1).copied().unwrap_or_default();
        let variant = match value {
            "Consonant" => "Incb::Consonant",
            "Extend" => "Incb::Extend",
            "Linker" => "Incb::Linker",
            "None" => continue,
            other => {
                return Err(Error::msg(format!(
                    "unknown Indic_Conjunct_Break value {other:?}"
                )));
            }
        };
        ranges.push((lo, hi, variant));
    }
    Ok(merge_tagged(ranges))
}

fn extended_pictographic() -> Result<Vec<(u32, u32)>> {
    let text = read(&ucd("emoji-data.txt"))?;
    Ok(merge(
        text.lines()
            .filter_map(ucd_fields)
            .filter(|(_, _, f)| f.first() == Some(&"Extended_Pictographic"))
            .map(|(lo, hi, _)| (lo, hi))
            .collect(),
    ))
}

/// The `// @generated` banner, rustfmt-canonical.
fn header(out: &mut String, version: &str, sources: &[&str]) {
    let _ = writeln!(
        out,
        "// @generated by `bnknotes --gen-tables` from Unicode {version} (vendored in ucd/)."
    );
    let _ = writeln!(out, "// DO NOT EDIT.");
    let _ = writeln!(out, "//");
    let _ = writeln!(out, "// Sources:");
    for source in sources {
        let _ = writeln!(out, "//   {source}");
    }
    let _ = writeln!(out, "//");
    let _ = writeln!(
        out,
        "// Sorted, non-overlapping, contiguous-merged ranges for binary search."
    );
    let _ = writeln!(out);
}

fn emit_ranges(out: &mut String, name: &str, ranges: &[(u32, u32)]) {
    let _ = writeln!(out, "static {name}: &[(u32, u32)] = &[");
    for (lo, hi) in ranges {
        let _ = writeln!(out, "    (0x{lo:04X}, 0x{hi:04X}),");
    }
    let _ = writeln!(out, "];");
}

fn emit_tagged(out: &mut String, name: &str, ty: &str, ranges: &[(u32, u32, &'static str)]) {
    let _ = writeln!(out, "static {name}: &[(u32, u32, {ty})] = &[");
    for (lo, hi, tag) in ranges {
        let _ = writeln!(out, "    (0x{lo:04X}, 0x{hi:04X}, {tag}),");
    }
    let _ = writeln!(out, "];");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_version_from_a_ucd_header() {
        assert_eq!(
            parse_version("# GraphemeBreakProperty-18.0.0.txt").as_deref(),
            Some("18.0.0")
        );
        assert_eq!(
            parse_version("# Date: 2024, foo 1.2.3 bar").as_deref(),
            Some("1.2.3")
        );
        assert_eq!(parse_version("no version").as_deref(), None);
    }

    #[test]
    fn parses_single_and_ranged_ucd_lines() {
        // A ranged, single-property line (a trailing comment is stripped).
        assert_eq!(
            ucd_fields("0600..0605 ; Prepend # Cf"),
            Some((0x0600, 0x0605, vec!["Prepend"]))
        );
        // A single code point with a two-field catalog value (InCB).
        assert_eq!(
            ucd_fields("094D ; InCB; Linker # Mn"),
            Some((0x094D, 0x094D, vec!["InCB", "Linker"]))
        );
        // No space before the '#' (emoji-data's Extended_Pictographic).
        assert_eq!(
            ucd_fields("00A9 ; Extended_Pictographic# E0.6"),
            Some((0x00A9, 0x00A9, vec!["Extended_Pictographic"]))
        );
        assert_eq!(ucd_fields("# comment"), None);
        assert_eq!(ucd_fields(""), None);
    }

    #[test]
    fn merges_adjacent_same_tag_ranges_but_not_across_tags() {
        let merged = merge_tagged(vec![
            (0x000B, 0x000C, "Gcb::Control"),
            (0x0000, 0x0009, "Gcb::Control"),
            (0x000A, 0x000A, "Gcb::Lf"),
        ]);
        // 0..9 Control and B..C Control stay split by the LF at A between them.
        assert_eq!(
            merged,
            vec![
                (0x0000, 0x0009, "Gcb::Control"),
                (0x000A, 0x000A, "Gcb::Lf"),
                (0x000B, 0x000C, "Gcb::Control"),
            ]
        );
    }
}
