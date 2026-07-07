//! Deterministic, seeded generators for the performance lab's stress corpora.
//! Each produces a document of roughly a requested byte size, built to hammer one
//! of the editor's hot paths (wrapping, grapheme segmentation, emoji shaping,
//! inline parsing). Deterministic by construction (a fixed-seed SplitMix64, no
//! `rand` crate, no clock), so a lab run is reproducible and a regression is real
//! rather than a different random document.

/// The corpora the lab knows how to generate, in the order it runs them.
pub const NAMES: &[&str] = &[
    "ascii_prose",
    "giant_line",
    "many_tiny_lines",
    "unicode",
    "emoji",
    "combining",
    "markdown",
];

/// Generate the named corpus at approximately `target_bytes`, or `None` for an
/// unknown name. The result may overshoot the target by one unit (a word, a
/// line, a block); it never undershoots.
pub fn generate(name: &str, target_bytes: usize) -> Option<String> {
    let out = match name {
        "ascii_prose" => ascii_prose(target_bytes),
        "giant_line" => giant_line(target_bytes),
        "many_tiny_lines" => many_tiny_lines(target_bytes),
        "unicode" => unicode(target_bytes),
        "emoji" => emoji(target_bytes),
        "combining" => combining(target_bytes),
        "markdown" => markdown(target_bytes),
        _ => return None,
    };
    Some(out)
}

/// A minimal SplitMix64: enough spread to shape realistic text, fully
/// deterministic from its seed.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const WORDS: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "eiusmod",
    "tempor",
    "incididunt",
    "labore",
    "magna",
    "aliqua",
    "enim",
];

/// Long prose paragraphs of common ASCII words: the throughput baseline for
/// layout and wrapping on ordinary notes.
fn ascii_prose(target: usize) -> String {
    let mut rng = Rng::new(1);
    let mut out = String::with_capacity(target + 64);
    while out.len() < target {
        let sentences = 3 + rng.below(4);
        for _ in 0..sentences {
            let words = 6 + rng.below(12);
            for w in 0..words {
                if w > 0 {
                    out.push(' ');
                }
                out.push_str(WORDS[rng.below(WORDS.len())]);
            }
            out.push_str(". ");
        }
        out.push_str("\n\n");
    }
    out
}

/// One enormous logical line with no newline: the worst case for soft-wrap and
/// the O(document) layout, which must wrap it into thousands of visual lines.
fn giant_line(target: usize) -> String {
    let mut rng = Rng::new(2);
    let mut out = String::with_capacity(target + 16);
    while out.len() < target {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(WORDS[rng.below(WORDS.len())]);
    }
    out
}

/// Hundreds of thousands of very short lines: the per-line loop and any
/// line-start index, the case windowed layout exists to fix.
fn many_tiny_lines(target: usize) -> String {
    let mut rng = Rng::new(3);
    let mut out = String::with_capacity(target + 16);
    while out.len() < target {
        out.push_str(WORDS[rng.below(WORDS.len())]);
        out.push('\n');
    }
    out
}

const SCRIPTS: &[&str] = &[
    "Съешь же ещё этих мягких французских булок да выпей чаю",
    "いろはにほへとちりぬるをわかよたれそつねならむ",
    "동해물과 백두산이 마르고 닳도록 하느님이 보우하사",
    "الأبجدية العربية تكتب من اليمين إلى اليسار دائما",
    "עברית היא שפה שמית עתיקה הנכתבת מימין לשמאל",
    "Ελληνικά γράμματα και πολυτονική ορθογραφία",
    "中文字符测试文本包含常用的汉字组合",
    "ภาษาไทยทดสอบการตัดคำและการวางสระ",
];

/// Mixed-script Unicode, including right-to-left runs: grapheme segmentation and
/// the advance cache across a wide codepoint range.
fn unicode(target: usize) -> String {
    let mut rng = Rng::new(4);
    let mut out = String::with_capacity(target + 128);
    while out.len() < target {
        out.push_str(SCRIPTS[rng.below(SCRIPTS.len())]);
        out.push('\n');
    }
    out
}

const EMOJI: &[&str] = &["😀", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇳🇴", "🧑🏿‍🚀", "❤️", "🤦🏼‍♂️", "🎉", "🌍"];

/// Emoji-dense lines, including skin-tone modifiers and ZWJ sequences: the
/// HarfBuzz cluster-shaping path and the color-strike resample cache.
fn emoji(target: usize) -> String {
    let mut rng = Rng::new(5);
    let mut out = String::with_capacity(target + 128);
    while out.len() < target {
        let count = 8 + rng.below(24);
        for _ in 0..count {
            out.push_str(EMOJI[rng.below(EMOJI.len())]);
        }
        out.push('\n');
    }
    out
}

const BASES: &[char] = &['a', 'e', 'o', 'n', 'u', 's', 'z'];
const MARKS: &[char] = &[
    '\u{0300}', '\u{0301}', '\u{0302}', '\u{0303}', '\u{0308}', '\u{0323}', '\u{0327}', '\u{0331}',
];

/// Base letters buried under stacks of combining marks (Zalgo): the UAX #29
/// grapheme-clustering worst case, where one column is many codepoints.
fn combining(target: usize) -> String {
    let mut rng = Rng::new(6);
    let mut out = String::with_capacity(target + 128);
    while out.len() < target {
        let letters = 20 + rng.below(30);
        for _ in 0..letters {
            out.push(BASES[rng.below(BASES.len())]);
            let stack = 1 + rng.below(6);
            for _ in 0..stack {
                out.push(MARKS[rng.below(MARKS.len())]);
            }
        }
        out.push('\n');
    }
    out
}

/// Every markdown construct in rotation: headings, wrapped prose with inline
/// emphasis and code and links, nested and ordered lists, quotes, fenced code,
/// rules. Exercises the block classifier and the inline span parser at once.
fn markdown(target: usize) -> String {
    let mut rng = Rng::new(7);
    let mut out = String::with_capacity(target + 256);
    let word = |rng: &mut Rng| WORDS[rng.below(WORDS.len())];
    while out.len() < target {
        let level = 1 + rng.below(3);
        for _ in 0..level {
            out.push('#');
        }
        out.push(' ');
        out.push_str(word(&mut rng));
        out.push('\n');
        out.push('\n');

        out.push_str("A paragraph with **");
        out.push_str(word(&mut rng));
        out.push_str("** and *");
        out.push_str(word(&mut rng));
        out.push_str("* and `");
        out.push_str(word(&mut rng));
        out.push_str("` and a [");
        out.push_str(word(&mut rng));
        out.push_str("](https://example.com/");
        out.push_str(word(&mut rng));
        out.push_str(") that runs on long enough to wrap across the content width.\n\n");

        for i in 0..3 {
            out.push_str("- ");
            out.push_str(word(&mut rng));
            out.push('\n');
            if i == 1 {
                out.push_str("  - ");
                out.push_str(word(&mut rng));
                out.push('\n');
            }
        }
        out.push('\n');

        for i in 1..=3 {
            out.push_str(&i.to_string());
            out.push_str(". ");
            out.push_str(word(&mut rng));
            out.push('\n');
        }
        out.push('\n');

        out.push_str("> ");
        out.push_str(word(&mut rng));
        out.push_str(" quoted text\n\n```\nfn code() { let x = 42; }\n```\n\n---\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_corpus_generates_and_meets_its_target() {
        for name in NAMES {
            let doc = generate(name, 32 * 1024).unwrap_or_else(|| panic!("{name} missing"));
            assert!(
                doc.len() >= 32 * 1024,
                "{name} undershot: {} bytes",
                doc.len()
            );
            // Every generator must emit valid UTF-8 (String guarantees it) and be
            // non-trivial, so a scenario has real work.
            assert!(doc.chars().count() > 1000, "{name} too small");
        }
    }

    #[test]
    fn unknown_corpus_is_none() {
        assert!(generate("does_not_exist", 1024).is_none());
    }

    #[test]
    fn generation_is_deterministic() {
        // Same name and size, byte-identical output: the property the whole lab
        // rests on (a regression is real, not a reroll).
        for name in NAMES {
            let a = generate(name, 16 * 1024).unwrap();
            let b = generate(name, 16 * 1024).unwrap();
            assert_eq!(a, b, "{name} is not deterministic");
        }
    }

    #[test]
    fn giant_line_has_no_newline() {
        let doc = generate("giant_line", 64 * 1024).unwrap();
        assert!(!doc.contains('\n'), "giant_line must be one logical line");
    }

    #[test]
    fn many_tiny_lines_are_actually_many() {
        let doc = generate("many_tiny_lines", 64 * 1024).unwrap();
        assert!(doc.lines().count() > 5000, "expected many short lines");
    }
}
