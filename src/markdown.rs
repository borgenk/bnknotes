//! The first sliver of the markdown layer: classifying a logical line's block
//! style so the editor can render markup inline rather than in a side preview.
//!
//! Markdown is rendered as you type, and this is the start of it.
//! Classification is per *logical* line (the text between two `'\n'`s).
//! [`crate::layout`] resolves the style to a pixel size and stamps it onto every
//! visual line the logical line wraps into, so a heading renders larger while
//! sharing all the cursor and selection math.
//!
//! Scope for now: ATX headings only (`# ` .. `###### `). Everything else is
//! body text. Indented headings (CommonMark allows up to three leading spaces)
//! are deliberately not recognized; notes are written at the left margin.

/// The block style of one logical line. A closed set, extended as more markup
/// lands (lists, quotes, ...).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LineStyle {
    /// Ordinary paragraph text.
    Body,
    /// An ATX heading of the given level, 1..=6.
    Heading(u8),
}

/// The ATX heading level of `line`, or `None` if it is not a heading. A heading
/// is one to six leading `'#'` followed immediately by a space or the line's
/// end, per CommonMark: `#tag` is not a heading, but `#`, `# `, and `# Title`
/// are. `line` must be a single logical line (no embedded `'\n'`).
pub fn heading_level(line: &str) -> Option<u8> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    match line.as_bytes().get(hashes) {
        // Hashes run to the end of the line: an empty heading.
        None => Some(hashes as u8),
        // A space separates the marker from the heading text.
        Some(&b' ') => Some(hashes as u8),
        // Anything else (e.g. `#tag`) is not a heading.
        Some(_) => None,
    }
}

/// The block style of `line`.
pub fn style_of(line: &str) -> LineStyle {
    match heading_level(line) {
        Some(level) => LineStyle::Heading(level),
        None => LineStyle::Body,
    }
}

/// Byte length of a heading's marker prefix: the run of `'#'` plus the single
/// separating space if one is present. `0` for a non-heading line. Used to draw
/// the markers in a distinct color so the markup reads as syntax.
pub fn marker_len(line: &str) -> usize {
    if heading_level(line).is_none() {
        return 0;
    }
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    match line.as_bytes().get(hashes) {
        Some(&b' ') => hashes + 1,
        _ => hashes,
    }
}

/// The pixel size `style` renders at, scaled from the body `base`. Headings step
/// down from H1 toward body size; H5 and H6 match body (they read as bold body
/// text in most themes, and bold is not wired up yet). Scales are in tenths to
/// stay in integer pixels without a float dependency on rounding.
pub fn font_size(style: LineStyle, base: u32) -> u32 {
    let tenths = match style {
        LineStyle::Body => return base,
        LineStyle::Heading(1) => 16,
        LineStyle::Heading(2) => 14,
        LineStyle::Heading(3) => 12,
        LineStyle::Heading(4) => 11,
        // H5, H6, and any out-of-range level fall back to body size.
        _ => 10,
    };
    (base * tenths + 5) / 10
}

/// Every distinct pixel size the styles use at body size `base`, so the font set
/// can be opened up front. Body first, then each heading size not already seen.
pub fn all_sizes(base: u32) -> Vec<u32> {
    let mut sizes = vec![font_size(LineStyle::Body, base)];
    for level in 1..=6u8 {
        let size = font_size(LineStyle::Heading(level), base);
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_each_heading_level() {
        for level in 1..=6u8 {
            let line = format!("{} heading", "#".repeat(level as usize));
            assert_eq!(heading_level(&line), Some(level));
            assert_eq!(style_of(&line), LineStyle::Heading(level));
        }
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        assert_eq!(heading_level("####### too deep"), None);
        assert_eq!(style_of("####### too deep"), LineStyle::Body);
    }

    #[test]
    fn hash_without_a_space_is_not_a_heading() {
        assert_eq!(heading_level("#tag"), None);
        assert_eq!(heading_level("##bold"), None);
    }

    #[test]
    fn bare_and_space_only_markers_are_empty_headings() {
        assert_eq!(heading_level("#"), Some(1));
        assert_eq!(heading_level("###"), Some(3));
        assert_eq!(heading_level("# "), Some(1));
    }

    #[test]
    fn leading_space_disqualifies_a_heading() {
        assert_eq!(heading_level(" # indented"), None);
    }

    #[test]
    fn plain_text_and_empty_lines_are_body() {
        assert_eq!(style_of(""), LineStyle::Body);
        assert_eq!(style_of("just text"), LineStyle::Body);
    }

    #[test]
    fn marker_len_covers_the_hashes_and_one_space() {
        assert_eq!(marker_len("# Title"), 2);
        assert_eq!(marker_len("### Deep"), 4);
        assert_eq!(marker_len("#"), 1);
        assert_eq!(marker_len("## "), 3);
        assert_eq!(marker_len("body"), 0);
    }

    #[test]
    fn font_size_steps_down_to_body() {
        let base = 32;
        assert_eq!(font_size(LineStyle::Body, base), base);
        let h1 = font_size(LineStyle::Heading(1), base);
        let h2 = font_size(LineStyle::Heading(2), base);
        let h3 = font_size(LineStyle::Heading(3), base);
        assert!(h1 > h2 && h2 > h3 && h3 > base);
        assert_eq!(font_size(LineStyle::Heading(6), base), base);
    }

    #[test]
    fn all_sizes_are_distinct_and_include_body() {
        let sizes = all_sizes(32);
        assert_eq!(sizes[0], 32, "body size first");
        let mut seen = sizes.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), sizes.len(), "no duplicate sizes");
    }
}
