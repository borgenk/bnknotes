//! Font discovery through `libfontconfig`.
//!
//! This module resolves configured family names to files. Family queries apply the
//! current fontconfig configuration. All C types are opaque.

use std::ffi::{CStr, CString, OsString};
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::ptr::{self, NonNull};

use core::ffi::{c_char, c_int, c_uchar};

use crate::platform::freetype::FontStyle;

/// An opaque fontconfig pattern: a bag of properties, both the query and the answer.
#[repr(C)]
struct FcPattern {
    _opaque: [u8; 0],
}

/// An opaque fontconfig character set: the code points a font has glyphs for.
#[repr(C)]
struct FcCharSet {
    _opaque: [u8; 0],
}

/// `FcFontSet`, a public struct: an array of patterns.
#[repr(C)]
struct FcFontSet {
    nfont: c_int,
    sfont: c_int,
    fonts: *mut *mut FcPattern,
}

/// An opaque fontconfig configuration. Every call passes null, meaning "the current one",
/// which is the system's, the same one every other application uses.
#[repr(C)]
struct FcConfig {
    _opaque: [u8; 0],
}

// Property names (`fontconfig.h`). NUL-terminated because the C API takes `const char *`.
const FC_FILE: &[u8] = b"file\0";
const FC_INDEX: &[u8] = b"index\0";
const FC_FAMILY: &[u8] = b"family\0";
const FC_WEIGHT: &[u8] = b"weight\0";
const FC_SLANT: &[u8] = b"slant\0";
const FC_CHARSET: &[u8] = b"charset\0";
const FC_OUTLINE: &[u8] = b"outline\0";
const FC_COLOR: &[u8] = b"color\0";

// Fontconfig's weight and slant values from fontconfig.h.
const FC_WEIGHT_REGULAR: c_int = 80;
const FC_WEIGHT_BOLD: c_int = 200;
const FC_SLANT_ROMAN: c_int = 0;
const FC_SLANT_ITALIC: c_int = 100;

/// `FcMatchPattern` (`FcMatchKind`): substitutions to apply to a *query*, as opposed to
/// to a font. The first variant of the enum, hence 0.
const FC_MATCH_PATTERN: c_int = 0;
/// `FcResultMatch` (`FcResult`): the property was found. Also the first variant, hence 0.
const FC_RESULT_MATCH: c_int = 0;
/// `FcTrue`.
const FC_TRUE: c_int = 1;

#[link(name = "fontconfig")]
unsafe extern "C" {
    fn FcInit() -> c_int;
    fn FcPatternCreate() -> *mut FcPattern;
    fn FcPatternDestroy(pattern: *mut FcPattern);
    fn FcPatternAddString(
        pattern: *mut FcPattern,
        object: *const c_char,
        value: *const c_uchar,
    ) -> c_int;
    fn FcPatternAddInteger(pattern: *mut FcPattern, object: *const c_char, value: c_int) -> c_int;
    fn FcPatternGetString(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut *mut c_uchar,
    ) -> c_int;
    fn FcPatternGetInteger(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut c_int,
    ) -> c_int;
    fn FcConfigSubstitute(config: *mut FcConfig, pattern: *mut FcPattern, kind: c_int) -> c_int;
    fn FcDefaultSubstitute(pattern: *mut FcPattern);
    fn FcFontMatch(
        config: *mut FcConfig,
        pattern: *mut FcPattern,
        result: *mut c_int,
    ) -> *mut FcPattern;
    fn FcFontSort(
        config: *mut FcConfig,
        pattern: *mut FcPattern,
        trim: c_int,
        csp: *mut *mut FcCharSet,
        result: *mut c_int,
    ) -> *mut FcFontSet;
    fn FcFontSetDestroy(set: *mut FcFontSet);
    fn FcPatternGetCharSet(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut *mut FcCharSet,
    ) -> c_int;
    fn FcPatternGetBool(
        pattern: *const FcPattern,
        object: *const c_char,
        index: c_int,
        value: *mut c_int,
    ) -> c_int;
    fn FcCharSetHasChar(charset: *const FcCharSet, ucs4: u32) -> c_int;
}

/// A resolved font path and its FreeType face index.
///
/// The index may select a face in a collection or a named variable-font instance.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FontFile {
    pub path: PathBuf,
    pub index: i32,
}

/// A matched file and the family names fontconfig assigns to it.
pub(super) struct FamilyMatch {
    pub file: FontFile,
    pub families: Vec<String>,
}

const GENERIC_FAMILIES: [&str; 5] = ["monospace", "sans-serif", "serif", "system-ui", "emoji"];
const UNCONFIGURED_FAMILY: &str = "__bnknotes_unconfigured_family__";

/// Whether `family` names a role rather than a typeface.
fn is_generic(family: &str) -> bool {
    GENERIC_FAMILIES
        .iter()
        .any(|generic| generic.eq_ignore_ascii_case(family))
}

/// Resolve `family_name` and `style` through the current fontconfig rules.
///
/// Generic aliases accept the selected family. Named families return `None` when the
/// matched pattern does not list that name, allowing the next configured candidate to run.
pub fn font_for_family(family_name: &str, style: FontStyle) -> Option<FontFile> {
    match_family(family_name, style).map(|matched| matched.file)
}

/// Resolve a family while retaining the matched family names for style checks.
pub(super) fn match_family(family_name: &str, style: FontStyle) -> Option<FamilyMatch> {
    // SAFETY: FcInit takes no arguments and is idempotent; it returns FcFalse when the
    // configuration cannot be loaded.
    if unsafe { FcInit() } != FC_TRUE {
        return None;
    }

    let (weight, slant) = weight_and_slant(style);
    let (query, configured_families) = configured_family_query(family_name, weight, slant)?;
    // SAFETY: query is live and has already received configured substitutions.
    unsafe { FcDefaultSubstitute(query.0) };

    let mut result: c_int = 0;
    // SAFETY: query is live and substituted; a null config means the current (system)
    // one; result is a live local. The returned pattern is *owned* by us, which is what
    // the Pattern wrapper is for, unlike the borrowed patterns inside a sorted set.
    // Wrapped before the result is checked, so a pattern returned without a match is freed.
    let matched = Pattern(unsafe { FcFontMatch(ptr::null_mut(), query.0, &mut result) });
    if matched.0.is_null() || result != FC_RESULT_MATCH {
        return None;
    }

    let families = pattern_strings(matched.0, FC_FAMILY);
    if !is_generic(family_name)
        && !families
            .iter()
            .any(|name| name.eq_ignore_ascii_case(family_name))
        && !configured_alias_matches(family_name, weight, slant, &configured_families, &families)
    {
        return None;
    }

    Some(FamilyMatch {
        file: FontFile {
            path: pattern_path(matched.0, FC_FILE)?,
            // No index property means a plain single-face font: face 0.
            index: pattern_integer(matched.0, FC_INDEX).unwrap_or(0),
        },
        families,
    })
}

/// A family query after configured substitutions but before default properties.
fn configured_family_query(
    family_name: &str,
    weight: c_int,
    slant: c_int,
) -> Option<(Pattern, Vec<String>)> {
    // A family name with an interior NUL is not a family any font has.
    let family = CString::new(family_name).ok()?;
    let query = Pattern::new()?;
    // SAFETY: query is live; the object names are NUL-terminated literals and `family` is
    // a live CString that fontconfig copies out of rather than borrows.
    unsafe {
        if FcPatternAddString(query.0, FC_FAMILY.as_ptr().cast(), family.as_ptr().cast()) != FC_TRUE
        {
            return None;
        }
        FcPatternAddInteger(query.0, FC_WEIGHT.as_ptr().cast(), weight);
        FcPatternAddInteger(query.0, FC_SLANT.as_ptr().cast(), slant);
        FcConfigSubstitute(ptr::null_mut(), query.0, FC_MATCH_PATTERN);
    }
    let families = pattern_strings(query.0, FC_FAMILY);
    Some((query, families))
}

/// Whether configured family substitutions introduced the family that won the match.
fn configured_alias_matches(
    requested: &str,
    weight: c_int,
    slant: c_int,
    configured: &[String],
    matched: &[String],
) -> bool {
    let Some((_, baseline)) = configured_family_query(UNCONFIGURED_FAMILY, weight, slant) else {
        return false;
    };
    alias_targets(requested, configured, &baseline)
        .iter()
        .any(|target| {
            matched
                .iter()
                .any(|family| family.eq_ignore_ascii_case(target))
        })
}

/// Families added or reordered specifically for `requested`, excluding global defaults.
fn alias_targets<'a>(
    requested: &str,
    configured: &'a [String],
    baseline: &[String],
) -> Vec<&'a str> {
    let configured: Vec<&str> = configured
        .iter()
        .map(String::as_str)
        .filter(|family| !family.eq_ignore_ascii_case(requested))
        .collect();
    let baseline: Vec<&str> = baseline
        .iter()
        .map(String::as_str)
        .filter(|family| !family.eq_ignore_ascii_case(UNCONFIGURED_FAMILY))
        .collect();
    let mut unmatched = baseline.clone();
    let mut targets = Vec::new();
    for family in &configured {
        if let Some(index) = unmatched
            .iter()
            .position(|candidate| candidate.eq_ignore_ascii_case(family))
        {
            unmatched.remove(index);
        } else {
            targets.push(*family);
        }
    }
    let same_order = configured.len() == baseline.len()
        && configured
            .iter()
            .zip(&baseline)
            .all(|(left, right)| left.eq_ignore_ascii_case(right));
    if targets.is_empty()
        && !same_order
        && let Some(first) = configured.first()
    {
        targets.push(*first);
    }
    targets
}

/// fontconfig's fallback order for a family: the installed fonts that each add
/// coverage, best match first, as other text renderers find a font for a character
/// their chosen family lacks. The answer for a character depends only on the
/// fontconfig configuration, never on what was asked before.
pub struct FallbackFonts {
    set: NonNull<FcFontSet>,
}

impl FallbackFonts {
    /// The sorted list for family at regular weight, or None when fontconfig cannot
    /// load its configuration or sort.
    pub fn for_family(family: &str) -> Option<Self> {
        // SAFETY: FcInit takes no arguments and is idempotent.
        if unsafe { FcInit() } != FC_TRUE {
            return None;
        }
        let (weight, slant) = weight_and_slant(FontStyle::Regular);
        let (query, _) = configured_family_query(family, weight, slant)?;
        let mut result: c_int = 0;
        // SAFETY: query is live; FcDefaultSubstitute completes it before the sort. A
        // null config is the current one, a null charset pointer asks for no union,
        // and the returned set is owned here and freed on drop.
        let set = unsafe {
            FcDefaultSubstitute(query.0);
            FcFontSort(
                ptr::null_mut(),
                query.0,
                FC_TRUE,
                ptr::null_mut(),
                &mut result,
            )
        };
        let set = NonNull::new(set)?;
        let fonts = Self { set };
        (result == FC_RESULT_MATCH).then_some(fonts)
    }

    /// The position of the first font in the list that draws ch with outline
    /// glyphs, or None when no installed font does. A color font is left out: its
    /// glyphs are bitmaps or color layers, which the emoji path draws.
    pub fn covering(&self, ch: char) -> Option<usize> {
        self.patterns().iter().position(|&pattern| {
            pattern_bool(pattern, FC_OUTLINE) == Some(true)
                && pattern_bool(pattern, FC_COLOR) != Some(true)
                && pattern_has_char(pattern, ch)
        })
    }

    /// The file of the font at position i.
    pub fn file(&self, i: usize) -> Option<FontFile> {
        let pattern = *self.patterns().get(i)?;
        Some(FontFile {
            path: pattern_path(pattern, FC_FILE)?,
            index: pattern_integer(pattern, FC_INDEX).unwrap_or(0),
        })
    }

    fn patterns(&self) -> &[*mut FcPattern] {
        // SAFETY: the set came from FcFontSort and lives until drop; fonts points at
        // nfont pattern pointers the set owns.
        unsafe {
            let set = self.set.as_ref();
            match usize::try_from(set.nfont) {
                Ok(n) if !set.fonts.is_null() => std::slice::from_raw_parts(set.fonts, n),
                _ => &[],
            }
        }
    }
}

impl Drop for FallbackFonts {
    fn drop(&mut self) {
        // SAFETY: the set came from FcFontSort and is freed exactly once.
        unsafe { FcFontSetDestroy(self.set.as_ptr()) };
    }
}

/// Whether a pattern's charset has ch.
fn pattern_has_char(pattern: *const FcPattern, ch: char) -> bool {
    let mut charset: *mut FcCharSet = ptr::null_mut();
    // SAFETY: pattern is live and object a NUL-terminated literal; fontconfig writes
    // a pointer borrowed from the pattern, read before the pattern can go.
    unsafe {
        FcPatternGetCharSet(pattern, FC_CHARSET.as_ptr().cast(), 0, &mut charset) == FC_RESULT_MATCH
            && !charset.is_null()
            && FcCharSetHasChar(charset, ch as u32) == FC_TRUE
    }
}

/// A boolean property of a pattern, or None when it does not carry one.
fn pattern_bool(pattern: *const FcPattern, object: &[u8]) -> Option<bool> {
    let mut value: c_int = 0;
    // SAFETY: as pattern_integer; fontconfig writes the FcBool through the pointer.
    let result = unsafe { FcPatternGetBool(pattern, object.as_ptr().cast(), 0, &mut value) };
    (result == FC_RESULT_MATCH).then_some(value == FC_TRUE)
}

/// Map a font style to fontconfig's weight and slant scales.
fn weight_and_slant(style: FontStyle) -> (c_int, c_int) {
    match style {
        FontStyle::Regular => (FC_WEIGHT_REGULAR, FC_SLANT_ROMAN),
        FontStyle::Bold => (FC_WEIGHT_BOLD, FC_SLANT_ROMAN),
        FontStyle::Italic => (FC_WEIGHT_REGULAR, FC_SLANT_ITALIC),
        FontStyle::BoldItalic => (FC_WEIGHT_BOLD, FC_SLANT_ITALIC),
    }
}

/// Every string value of a pattern property, in fontconfig preference order.
fn pattern_strings(pattern: *const FcPattern, object: &[u8]) -> Vec<String> {
    (0..)
        .map_while(|index| pattern_string_at(pattern, object, index))
        .collect()
}

/// The `index`th value of a string property, for the properties that carry several (a
/// font's family names, one per language it declares). The pointer fontconfig hands back
/// points *into* the pattern, so it lives and dies with it: copy before the pattern is
/// freed.
fn pattern_string_at(pattern: *const FcPattern, object: &[u8], index: c_int) -> Option<String> {
    let mut value: *mut c_uchar = ptr::null_mut();
    // SAFETY: pattern is live, object is a NUL-terminated literal, and value is a live
    // local that fontconfig writes a borrowed pointer through.
    let result = unsafe { FcPatternGetString(pattern, object.as_ptr().cast(), index, &mut value) };
    if result != FC_RESULT_MATCH || value.is_null() {
        return None;
    }
    // SAFETY: on FcResultMatch, value points at a NUL-terminated UTF-8 string owned by the
    // pattern, which outlives this borrow (we copy before returning).
    let text = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
    text.to_str().ok().map(str::to_owned)
}

/// A filesystem property copied without interpreting its Unix path bytes as text.
fn pattern_path(pattern: *const FcPattern, object: &[u8]) -> Option<PathBuf> {
    let mut value: *mut c_uchar = ptr::null_mut();
    // SAFETY: pattern is live, object is a NUL-terminated literal, and value is a live
    // local that fontconfig writes a borrowed pointer through.
    let result = unsafe { FcPatternGetString(pattern, object.as_ptr().cast(), 0, &mut value) };
    if result != FC_RESULT_MATCH || value.is_null() {
        return None;
    }
    // SAFETY: on FcResultMatch, value points at a NUL-terminated byte string owned by
    // the pattern, which remains live while the bytes are copied.
    let value = unsafe { CStr::from_ptr(value.cast::<c_char>()) };
    Some(PathBuf::from(OsString::from_vec(value.to_bytes().to_vec())))
}

/// An integer property of a pattern, or `None` when it does not carry one.
fn pattern_integer(pattern: *const FcPattern, object: &[u8]) -> Option<i32> {
    let mut value: c_int = 0;
    // SAFETY: as `pattern_string_at`; fontconfig writes the integer through the pointer.
    let result = unsafe { FcPatternGetInteger(pattern, object.as_ptr().cast(), 0, &mut value) };
    (result == FC_RESULT_MATCH).then_some(value)
}

/// An owned `FcPattern`, freed on drop. Every early return in construction would otherwise
/// leak one.
struct Pattern(*mut FcPattern);

impl Pattern {
    fn new() -> Option<Self> {
        // SAFETY: takes no arguments; returns null on allocation failure.
        let pattern = unsafe { FcPatternCreate() };
        (!pattern.is_null()).then_some(Pattern(pattern))
    }
}

impl Drop for Pattern {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 was returned by FcPatternCreate and is dropped exactly once.
            unsafe { FcPatternDestroy(self.0) };
        }
    }
}

/// Silence the unused-type warning for the config handle, which only ever appears behind
/// a null pointer in the calls above.
const _: Option<&FcConfig> = None;

/// A letter from each of several scripts a Latin text family lacks: Devanagari,
/// Arabic, Hebrew, Thai, Ethiopic, Georgian, Han and Hangul.
#[cfg(test)]
pub(crate) const TEST_SCRIPTS: &str = "कبאกሀა一가";

#[cfg(test)]
mod tests {
    use crate::platform::fontconfig::*;
    use crate::platform::freetype::Face;

    #[test]
    fn fontconfig_font_set_mirror_matches_the_c_layout() {
        pin_c_layout!(FcFontSet, size 16; nfont @ 0, sfont @ 4, fonts @ 8);
    }

    #[test]
    fn the_fallback_order_finds_a_font_for_a_script_the_family_lacks() {
        let Some(fonts) = FallbackFonts::for_family("sans-serif") else {
            eprintln!("bnknotes: fontconfig cannot sort; skipping the fallback order test");
            return;
        };
        assert!(fonts.covering('a').is_some());
        assert_eq!(fonts.covering('\u{0378}'), None, "an unassigned code point");
        let Some((ch, at)) = TEST_SCRIPTS
            .chars()
            .find_map(|ch| fonts.covering(ch).map(|at| (ch, at)))
        else {
            eprintln!("bnknotes: no font for any test script; skipping the fallback order test");
            return;
        };
        let file = fonts.file(at).expect("the font's file");
        assert!(file.path.exists(), "{}", file.path.display());
        let again = FallbackFonts::for_family("sans-serif").expect("sorts again");
        assert_eq!(
            again.covering(ch).and_then(|at| again.file(at)),
            Some(file),
            "the answer depends on the configuration alone"
        );
    }

    #[test]
    fn a_color_font_is_never_the_fallback_for_text() {
        let Some(fonts) = FallbackFonts::for_family("sans-serif") else {
            return;
        };
        // Usually only the emoji font has this smiley, as color bitmaps no text
        // glyph can be drawn from.
        if let Some(at) = fonts.covering('\u{1F600}') {
            let face = Face::from_file(&fonts.file(at).expect("a file")).expect("opens");
            assert!(face.select_first_strike().is_none(), "an outline font");
        }
    }

    /// Whether fontconfig loads its configuration here; a test needing it skips when
    /// it does not.
    fn fontconfig_available() -> bool {
        // SAFETY: FcInit takes no arguments and is idempotent.
        unsafe { FcInit() == FC_TRUE }
    }

    #[test]
    fn the_system_resolves_its_generic_monospace() {
        if !fontconfig_available() {
            eprintln!("no usable fontconfig in this environment; skipping");
            return;
        }
        let mono = font_for_family("monospace", FontStyle::Regular)
            .expect("fontconfig resolves the monospace alias");
        assert!(
            mono.path.exists(),
            "the file it named must be on disk: {}",
            mono.path.display()
        );
    }

    #[test]
    fn every_style_resolves_to_a_file_that_exists() {
        if !fontconfig_available() {
            return;
        }
        for style in [
            FontStyle::Regular,
            FontStyle::Bold,
            FontStyle::Italic,
            FontStyle::BoldItalic,
        ] {
            let file =
                font_for_family("monospace", style).unwrap_or_else(|| panic!("{style:?} resolves"));
            assert!(file.path.exists(), "{style:?} named a missing file");
        }
    }

    #[test]
    fn an_uninstalled_family_is_a_miss_not_a_substitute() {
        // FcFontMatch substitutes a nearby font, but named candidates must remain misses.
        if !fontconfig_available() {
            return;
        }
        assert!(font_for_family("NoSuchFamilyExistsAnywhere", FontStyle::Regular).is_none());
    }

    #[test]
    fn a_generic_alias_takes_whatever_it_resolves_to() {
        if !fontconfig_available() {
            return;
        }
        for generic in GENERIC_FAMILIES {
            if generic == "emoji" {
                continue;
            }
            let file = font_for_family(generic, FontStyle::Regular)
                .unwrap_or_else(|| panic!("the {generic} alias resolves"));
            assert!(file.path.exists());
        }
    }

    #[test]
    fn configured_alias_targets_exclude_the_global_fallbacks() {
        let strings = |names: &[&str]| -> Vec<String> {
            names.iter().map(|name| (*name).to_string()).collect()
        };
        let baseline = strings(&[UNCONFIGURED_FAMILY, "Default Sans", "Last Resort"]);

        let missing = strings(&["Missing Family", "Default Sans", "Last Resort"]);
        assert!(alias_targets("Missing Family", &missing, &baseline).is_empty());

        let alias = strings(&[
            "Chosen Mono",
            "My Editor Font",
            "Default Sans",
            "Last Resort",
        ]);
        assert_eq!(
            alias_targets("My Editor Font", &alias, &baseline),
            ["Chosen Mono"]
        );

        let unavailable = strings(&[
            "Unavailable Target",
            "My Editor Font",
            "Default Sans",
            "Last Resort",
        ]);
        let targets = alias_targets("My Editor Font", &unavailable, &baseline);
        assert!(!targets.contains(&"Default Sans"));
    }

    #[test]
    fn a_named_family_that_is_installed_comes_back() {
        if !fontconfig_available() {
            return;
        }
        let Some(mono) = font_for_family("monospace", FontStyle::Regular) else {
            return;
        };
        let Some(name) = installed_family_name() else {
            return;
        };
        let by_name = font_for_family(&name, FontStyle::Regular)
            .unwrap_or_else(|| panic!("{name} is installed, so it resolves by name"));
        assert_eq!(by_name.path, mono.path);
    }

    /// An installed family name derived without assuming the system's font selection:
    /// the name fontconfig gives the file the monospace alias resolves to.
    fn installed_family_name() -> Option<String> {
        match_family("monospace", FontStyle::Regular)?
            .families
            .into_iter()
            .next()
    }

    #[test]
    fn a_family_name_with_an_interior_nul_is_refused() {
        assert!(font_for_family("mono\0space", FontStyle::Regular).is_none());
    }

    #[test]
    fn a_file_property_preserves_non_utf8_path_bytes() {
        use std::os::unix::ffi::OsStrExt;

        let pattern = Pattern::new().expect("a fontconfig pattern");
        let path = b"/tmp/bnknotes-font-\xff.ttf\0";
        // SAFETY: pattern is live, both byte strings are NUL-terminated, and
        // fontconfig copies the property value.
        let added =
            unsafe { FcPatternAddString(pattern.0, FC_FILE.as_ptr().cast(), path.as_ptr().cast()) };
        assert_eq!(added, FC_TRUE);

        let resolved = pattern_path(pattern.0, FC_FILE).expect("the path property");
        assert_eq!(resolved.as_os_str().as_bytes(), &path[..path.len() - 1]);
    }
}
