//! Glyph rasterization backed by the system FreeType (libfreetype.so).
//!
//! FreeType turns a TrueType/OpenType outline into an anti-aliased coverage
//! bitmap, handling hinting, scan conversion, and per-size metrics. This is the
//! thin FFI layer over it: a small extern block, one struct owning the C-side
//! handles, a Drop impl, and methods returning owned data. The font file is read
//! into a Vec the Face borrows, rather than mmapped, and coverage comes back as
//! a Vec<u8>.
//!
//! FreeType exposes glyph data as struct fields reached from the Face pointer
//! (face->glyph->bitmap, face->size->metrics), not accessors. The repr(C)
//! structs below mirror those layouts up to the last field this module reads;
//! FreeType allocates the full structs, we only read a prefix, so declaring it
//! with matching primitive types plus repr(C) padding is enough. The touched
//! fields are old and load-bearing for every FreeType consumer, so their
//! offsets have been ABI-stable for years.

use core::ffi::{c_char, c_int, c_long, c_short, c_uint, c_ulong, c_ushort, c_void};
use core::ptr::{self, NonNull};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::{Error, Result};

/// A monospace family and the four style files the editor draws emphasis with.
/// The first family whose `regular` file exists wins; a missing bold, italic, or
/// bold-italic variant falls back to the regular face at render time, so a family
/// that ships only some weights still renders (just without the slant or weight).
/// A short list of hardcoded paths keeps the editor runnable across machines
/// without crates.
struct FontFamily {
    regular: &'static str,
    bold: &'static str,
    italic: &'static str,
    bold_italic: &'static str,
}

const FONT_FAMILIES: &[FontFamily] = &[
    FontFamily {
        regular: "/usr/share/fonts/TTF/Hack-Regular.ttf",
        bold: "/usr/share/fonts/TTF/Hack-Bold.ttf",
        italic: "/usr/share/fonts/TTF/Hack-Italic.ttf",
        bold_italic: "/usr/share/fonts/TTF/Hack-BoldItalic.ttf",
    },
    FontFamily {
        regular: "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        bold: "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
        // Noto Sans Mono ships no italic; these paths simply will not exist, and
        // the lookup falls back to the regular face.
        italic: "/usr/share/fonts/noto/NotoSansMono-Italic.ttf",
        bold_italic: "/usr/share/fonts/noto/NotoSansMono-BoldItalic.ttf",
    },
    FontFamily {
        regular: "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
        bold: "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
        italic: "/usr/share/fonts/Adwaita/AdwaitaMono-Italic.ttf",
        bold_italic: "/usr/share/fonts/Adwaita/AdwaitaMono-BoldItalic.ttf",
    },
    FontFamily {
        regular: "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        bold: "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
        // DejaVu calls its slanted styles Oblique rather than Italic.
        italic: "/usr/share/fonts/TTF/DejaVuSansMono-Oblique.ttf",
        bold_italic: "/usr/share/fonts/TTF/DejaVuSansMono-BoldOblique.ttf",
    },
    FontFamily {
        regular: "/usr/share/fonts/TTF/Roboto-Regular.ttf",
        bold: "/usr/share/fonts/TTF/Roboto-Bold.ttf",
        italic: "/usr/share/fonts/TTF/Roboto-Italic.ttf",
        bold_italic: "/usr/share/fonts/TTF/Roboto-BoldItalic.ttf",
    },
];

/// The first family whose regular face is installed on this machine.
fn default_family() -> Result<&'static FontFamily> {
    FONT_FAMILIES
        .iter()
        .find(|f| std::path::Path::new(f.regular).exists())
        .ok_or_else(|| Error::msg("no usable font family found in the candidate list"))
}

/// Candidate regular faces for code: a monospace deliberately distinct from the
/// prose family, so code blocks and inline code read in their own typeface. Only
/// the regular face is ever needed (code never renders bold or italic), so this
/// is a flat list of paths rather than full families.
const CODE_FONT_FAMILIES: &[&str] = &[
    "/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/gnu-free/FreeMono.otf",
];

/// The first installed code face whose path differs from the prose family's
/// regular face, or `None` if none is available. A `None` leaves code sharing the
/// prose face: still readable, just without the visual distinction.
fn code_regular_path(prose: &FontFamily) -> Option<&'static str> {
    CODE_FONT_FAMILIES
        .iter()
        .copied()
        .find(|&p| p != prose.regular && std::path::Path::new(p).exists())
}

/// Which variant of the font family a run renders in. Body text and syntax
/// markers use [`FontStyle::Regular`]; emphasis selects a heavier or slanted
/// file. Kept distinct from [`crate::markdown::Emphasis`] so this font layer
/// stays independent of the markdown vocabulary: an inline code span is its own
/// emphasis but still renders in the regular monospace face.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FontStyle {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

// ---------------------------------------------------------------------------
// FFI surface.
// ---------------------------------------------------------------------------

type FtError = c_int;
type FtLibrary = *mut c_void;
type FtFace = *mut FtFaceRec;

/// Load the glyph and metrics only, no bitmap (FT_LOAD_DEFAULT).
const FT_LOAD_DEFAULT: i32 = 0x0;
/// Also rasterize to an 8-bit gray coverage bitmap (FT_LOAD_RENDER).
const FT_LOAD_RENDER: i32 = 1 << 2;
/// Ignore embedded bitmap strikes so an outlined glyph renders to 8-bit gray
/// rather than a mono/color strike (FT_LOAD_NO_BITMAP).
const FT_LOAD_NO_BITMAP: i32 = 1 << 3;
/// One coverage byte per pixel: the only mode whose row width in bytes equals
/// its pixel width (FT_PIXEL_MODE_GRAY).
const FT_PIXEL_MODE_GRAY: u8 = 2;

#[link(name = "freetype")]
#[allow(non_snake_case)]
extern "C" {
    fn FT_Init_FreeType(alibrary: *mut FtLibrary) -> FtError;
    fn FT_Done_FreeType(library: FtLibrary) -> FtError;
    fn FT_New_Memory_Face(
        library: FtLibrary,
        file_base: *const u8,
        file_size: c_long,
        face_index: c_long,
        aface: *mut FtFace,
    ) -> FtError;
    fn FT_Done_Face(face: FtFace) -> FtError;
    fn FT_Set_Pixel_Sizes(face: FtFace, pixel_width: c_uint, pixel_height: c_uint) -> FtError;
    fn FT_Load_Char(face: FtFace, char_code: c_ulong, load_flags: i32) -> FtError;
}

// ---------------------------------------------------------------------------
// C struct layouts (read-only mirrors of the FreeType headers). Most fields
// exist only to place the ones we read at the right offset.
// ---------------------------------------------------------------------------

#[repr(C)]
#[allow(dead_code)]
struct FtGeneric {
    data: *mut c_void,
    finalizer: *mut c_void,
}

#[repr(C)]
#[allow(dead_code)]
struct FtVector {
    x: c_long,
    y: c_long,
}

#[repr(C)]
#[allow(dead_code)]
struct FtBitmap {
    rows: c_uint,
    width: c_uint,
    pitch: c_int,
    buffer: *const u8,
    num_grays: c_ushort,
    pixel_mode: u8,
    palette_mode: u8,
    palette: *mut c_void,
}

#[repr(C)]
#[allow(dead_code)]
struct FtSizeMetrics {
    x_ppem: c_ushort,
    y_ppem: c_ushort,
    x_scale: c_long,
    y_scale: c_long,
    ascender: c_long,
    descender: c_long,
    height: c_long,
    max_advance: c_long,
}

#[repr(C)]
#[allow(dead_code)]
struct FtGlyphSlotRec {
    library: *mut c_void,
    face: *mut c_void,
    next: *mut c_void,
    glyph_index: c_uint,
    generic: FtGeneric,
    metrics: [c_long; 8],
    linear_hori_advance: c_long,
    linear_vert_advance: c_long,
    advance: FtVector,
    format: c_int,
    bitmap: FtBitmap,
    bitmap_left: c_int,
    bitmap_top: c_int,
}

#[repr(C)]
#[allow(dead_code)]
struct FtSizeRec {
    face: *mut c_void,
    generic: FtGeneric,
    metrics: FtSizeMetrics,
}

#[repr(C)]
#[allow(dead_code)]
struct FtFaceRec {
    num_faces: c_long,
    face_index: c_long,
    face_flags: c_long,
    style_flags: c_long,
    num_glyphs: c_long,
    family_name: *mut c_char,
    style_name: *mut c_char,
    num_fixed_sizes: c_int,
    available_sizes: *mut c_void,
    num_charmaps: c_int,
    charmaps: *mut c_void,
    generic: FtGeneric,
    bbox: [c_long; 4],
    units_per_em: c_ushort,
    ascender: c_short,
    descender: c_short,
    height: c_short,
    max_advance_width: c_short,
    max_advance_height: c_short,
    underline_position: c_short,
    underline_thickness: c_short,
    glyph: *mut FtGlyphSlotRec,
    size: *mut FtSizeRec,
}

// ---------------------------------------------------------------------------
// Owning wrapper.
// ---------------------------------------------------------------------------

/// A rasterized glyph, owned so no reference into FreeType memory escapes.
/// `coverage` is tightly packed top-down, `width` bytes per row, `rows` rows.
pub struct Glyph {
    /// Horizontal offset from the pen to the bitmap's left edge.
    pub left: i32,
    /// Vertical offset from the baseline up to the bitmap's top edge.
    pub top: i32,
    pub width: usize,
    pub rows: usize,
    /// Pen advance in pixels.
    pub advance: f32,
    pub coverage: Vec<u8>,
}

impl Glyph {
    fn empty() -> Self {
        Self {
            left: 0,
            top: 0,
            width: 0,
            rows: 0,
            advance: 0.0,
            coverage: Vec::new(),
        }
    }
}

/// Render-ready line metrics for a face at its current pixel size, rounded to
/// whole pixels. Ascent and descent are both positive (pixels above and below
/// the baseline); `line_height` is the baseline-to-baseline distance.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    pub ascent: i32,
    pub descent: i32,
    pub line_height: i32,
}

/// Owns the FreeType library, the face, and the font-file bytes the face
/// borrows. FT_New_Memory_Face keeps a pointer into the bytes for the face's
/// life; the Vec's heap buffer has a stable address across moves of this
/// struct, so holding both together is sound as long as the bytes are never
/// mutated or reallocated (they are not).
pub struct Face {
    library: NonNull<c_void>,
    face: NonNull<FtFaceRec>,
    _data: Vec<u8>,
    /// Rasterized glyphs cached by character. A face is opened at one fixed pixel
    /// size and never resized, so a glyph's bitmap is stable for the face's life:
    /// the first draw of a character rasterizes it through FreeType, every later
    /// frame blits the cached coverage. Without this a scroll or caret blink re-runs
    /// hinting and scan conversion for every glyph on screen, the dominant per-frame
    /// cost. Interior mutability so drawing through a shared `&Face` can fill it.
    glyphs: RefCell<HashMap<char, Glyph>>,
}

impl Face {
    /// Open the regular face of the first available font family at `pixel_size`.
    /// A test-only convenience: production goes through [`Fonts`], which opens the
    /// family's variants per size itself.
    #[cfg(test)]
    pub fn open_default(pixel_size: u32) -> Result<Self> {
        let face = Self::from_path(default_family()?.regular)?;
        face.set_pixel_size(pixel_size)?;
        Ok(face)
    }

    /// Build a face from a font file on disk.
    pub fn from_path(path: &str) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| Error::msg(format!("read font {path}: {e}")))?;

        let mut library: FtLibrary = ptr::null_mut();
        // SAFETY: alibrary points at a live local; FT writes the handle through
        // it and returns nonzero on failure.
        let err = unsafe { FT_Init_FreeType(&mut library) };
        let library = NonNull::new(library)
            .filter(|_| err == 0)
            .ok_or_else(|| Error::msg("FT_Init_FreeType failed"))?;

        let mut face: FtFace = ptr::null_mut();
        // SAFETY: library is valid; data outlives the face (it is moved into the
        // returned struct and dropped only after FT_Done_Face); aface is a live
        // local. FreeType keeps a pointer into data.
        let err = unsafe {
            FT_New_Memory_Face(
                library.as_ptr(),
                data.as_ptr(),
                data.len() as c_long,
                0,
                &mut face,
            )
        };
        let face = match NonNull::new(face).filter(|_| err == 0) {
            Some(face) => face,
            None => {
                // SAFETY: library came from FT_Init_FreeType and is freed once.
                unsafe { FT_Done_FreeType(library.as_ptr()) };
                return Err(Error::msg(format!("FT_New_Memory_Face failed for {path}")));
            }
        };

        Ok(Self {
            library,
            face,
            _data: data,
            glyphs: RefCell::new(HashMap::new()),
        })
    }

    /// Select the pixel size FreeType renders and reports metrics at; width 0
    /// matches it to the height.
    pub fn set_pixel_size(&self, size: u32) -> Result<()> {
        // SAFETY: face is a valid FT_Face; the call only reads size arguments.
        let err = unsafe { FT_Set_Pixel_Sizes(self.face.as_ptr(), 0, size) };
        if err != 0 {
            return Err(Error::msg("FT_Set_Pixel_Sizes failed"));
        }
        Ok(())
    }

    /// Run `f` over `ch`'s rasterized glyph, caching it on the face so the
    /// FreeType render only happens the first time the character is drawn at this
    /// size. The glyph is handed over by reference, never cloned: the coverage
    /// buffer stays in the cache and the caller blits straight out of it.
    pub fn with_glyph<R>(&self, ch: char, f: impl FnOnce(&Glyph) -> R) -> R {
        let mut cache = self.glyphs.borrow_mut();
        // `rasterize` reads only FreeType state, never the cache, so filling a
        // miss while the cache is borrowed cannot reenter this borrow.
        let glyph = cache.entry(ch).or_insert_with(|| self.rasterize(ch));
        f(glyph)
    }

    /// Rasterize a character at the current pixel size, copying the coverage
    /// out. A load failure or glyphless character yields an empty glyph. The
    /// render path goes through [`Self::with_glyph`], which caches the result;
    /// this is the uncached primitive behind it.
    pub fn rasterize(&self, ch: char) -> Glyph {
        // SAFETY: face is valid; FT_Load_Char renders into the face's shared
        // glyph slot. No reference into that slot is held across this call.
        let err = unsafe {
            FT_Load_Char(
                self.face.as_ptr(),
                ch as u32 as c_ulong,
                FT_LOAD_RENDER | FT_LOAD_NO_BITMAP,
            )
        };
        if err != 0 {
            return Glyph::empty();
        }
        // SAFETY: the load succeeded, so face->glyph points at the populated
        // slot; the borrow ends before any later FreeType call.
        let slot = unsafe { (*self.face.as_ptr()).glyph };
        let Some(slot) = NonNull::new(slot) else {
            return Glyph::empty();
        };
        let slot = unsafe { slot.as_ref() };
        let advance = slot.advance.x as f32 / 64.0;
        // The blit treats coverage as one byte per pixel, true only for the gray
        // render mode. Skip non-gray ink but keep the advance so layout holds.
        if slot.bitmap.pixel_mode != FT_PIXEL_MODE_GRAY {
            return Glyph {
                advance,
                ..Glyph::empty()
            };
        }
        Glyph {
            left: slot.bitmap_left,
            top: slot.bitmap_top,
            width: slot.bitmap.width as usize,
            rows: slot.bitmap.rows as usize,
            advance,
            coverage: copy_coverage(&slot.bitmap),
        }
    }

    /// Pen advance of a character at the current pixel size, in pixels. Loads
    /// metrics only (no rasterization), so it is cheap for measuring runs.
    /// Zero on a load failure.
    pub fn advance(&self, ch: char) -> f32 {
        // SAFETY: face is valid; FT_LOAD_DEFAULT loads metrics into the glyph
        // slot without rendering. The advance is read out immediately.
        let err =
            unsafe { FT_Load_Char(self.face.as_ptr(), ch as u32 as c_ulong, FT_LOAD_DEFAULT) };
        if err != 0 {
            return 0.0;
        }
        // SAFETY: the load succeeded, so face->glyph points at the slot.
        let slot = unsafe { (*self.face.as_ptr()).glyph };
        let Some(slot) = NonNull::new(slot) else {
            return 0.0;
        };
        // SAFETY: slot is valid after a successful load.
        unsafe { slot.as_ref().advance.x as f32 / 64.0 }
    }

    /// Ascent and descent at the current pixel size, in pixels. Descent is
    /// negative, matching FreeType.
    pub fn line_metrics(&self) -> (f32, f32) {
        // SAFETY: face is valid; face->size is non-null once a size is set,
        // which open_default always does. The metrics are copied out.
        let size = unsafe { (*self.face.as_ptr()).size };
        let Some(size) = NonNull::new(size) else {
            return (0.0, 0.0);
        };
        let metrics = unsafe { &size.as_ref().metrics };
        (
            metrics.ascender as f32 / 64.0,
            metrics.descender as f32 / 64.0,
        )
    }

    /// The recommended baseline-to-baseline distance at the current pixel size,
    /// in pixels: FreeType's line height, which includes any line gap the font
    /// asks for. Zero if no size is set.
    pub fn line_height(&self) -> f32 {
        // SAFETY: face is valid; face->size is non-null once a size is set.
        let size = unsafe { (*self.face.as_ptr()).size };
        let Some(size) = NonNull::new(size) else {
            return 0.0;
        };
        unsafe { size.as_ref().metrics.height as f32 / 64.0 }
    }

    /// Render-ready [`Metrics`] at the current pixel size: ascent and descent
    /// rounded up to whole pixels, and the baseline-to-baseline line height
    /// (falling back to ascent + descent if the font reports none).
    pub fn metrics(&self) -> Metrics {
        let (ascent, descent) = self.line_metrics();
        let ascent = ascent.ceil() as i32;
        let descent = (-descent).ceil() as i32;
        let line_height = {
            let h = self.line_height().ceil() as i32;
            if h > 0 {
                h
            } else {
                ascent + descent
            }
        };
        Metrics {
            ascent,
            descent,
            line_height,
        }
    }
}

impl Drop for Face {
    fn drop(&mut self) {
        // The face must be freed before the library it came from; the data Vec
        // is freed afterwards by its own Drop. SAFETY: each handle came from its
        // matching FreeType new call and is freed exactly once.
        unsafe {
            FT_Done_Face(self.face.as_ptr());
            FT_Done_FreeType(self.library.as_ptr());
        }
    }
}

/// The faces for one pixel size: the regular face (always present) plus any
/// bold/italic/bold-italic variants the family ships, the distinct code face (if
/// a separate code family is installed), and the cached line metrics. Metrics
/// come from the regular face only, so emphasis or code selects a different glyph
/// set without ever changing line height.
struct SizedFaces {
    size: u32,
    regular: Face,
    bold: Option<Face>,
    italic: Option<Face>,
    bold_italic: Option<Face>,
    /// The code family's regular face at this size, or `None` when no distinct
    /// code family is installed (code then falls back to `regular`).
    code: Option<Face>,
    metrics: Metrics,
}

/// A set of [`Face`]s at different pixel sizes and styles, so a document can mix
/// body text, larger headings, and inline emphasis. Each distinct size opens the
/// family's regular face plus whatever variant files exist (the same files,
/// re-read per size); the first size opened is the fallback for any size that was
/// not requested, and a missing variant falls back to the regular face, so a
/// lookup can never fail and never panics.
pub struct Fonts {
    sized: Vec<SizedFaces>,
}

impl Fonts {
    /// Open the default family at each of `sizes` (zeros and duplicates skipped),
    /// loading every available style per size. At least one nonzero size is
    /// required; the first opened becomes the fallback.
    pub fn new(sizes: &[u32]) -> Result<Self> {
        let family = default_family()?;
        let code_path = code_regular_path(family);
        let mut sized: Vec<SizedFaces> = Vec::new();
        for &size in sizes {
            if size == 0 || sized.iter().any(|s| s.size == size) {
                continue;
            }
            let regular = Face::from_path(family.regular)?;
            regular.set_pixel_size(size)?;
            let metrics = regular.metrics();
            sized.push(SizedFaces {
                size,
                regular,
                bold: open_variant(family.bold, size),
                italic: open_variant(family.italic, size),
                bold_italic: open_variant(family.bold_italic, size),
                code: code_path.and_then(|p| open_variant(p, size)),
                metrics,
            });
        }
        if sized.is_empty() {
            return Err(Error::msg("Fonts::new needs at least one nonzero size"));
        }
        Ok(Self { sized })
    }

    /// The entry for `size`, or the fallback (first opened) if `size` is unknown.
    fn entry(&self, size: u32) -> &SizedFaces {
        self.sized
            .iter()
            .find(|s| s.size == size)
            .unwrap_or(&self.sized[0])
    }

    /// The face for `size` in `style`, falling back to the regular face when the
    /// family lacks that variant and to the first opened size when `size` was not
    /// opened. Never fails, never panics.
    pub fn face(&self, size: u32, style: FontStyle) -> &Face {
        let entry = self.entry(size);
        match style {
            FontStyle::Regular => &entry.regular,
            FontStyle::Bold => entry.bold.as_ref().unwrap_or(&entry.regular),
            FontStyle::Italic => entry.italic.as_ref().unwrap_or(&entry.regular),
            FontStyle::BoldItalic => entry.bold_italic.as_ref().unwrap_or(&entry.regular),
        }
    }

    /// The code face for `size`: the distinct code family's regular face, or the
    /// prose regular face when no code family is installed (or `size` was not
    /// opened). Code never renders bold or italic, so there is only the one face.
    /// Never fails, never panics.
    pub fn code_face(&self, size: u32) -> &Face {
        let entry = self.entry(size);
        entry.code.as_ref().unwrap_or(&entry.regular)
    }

    /// The metrics for `size`, or the fallback's metrics if `size` was not opened.
    pub fn metrics(&self, size: u32) -> Metrics {
        self.entry(size).metrics
    }
}

/// Open a style variant at `size`, or `None` if the file is absent or fails to
/// load, leaving the caller to fall back to the regular face. Keeps a family
/// that ships only some weights from being a hard error.
fn open_variant(path: &str, size: u32) -> Option<Face> {
    if !std::path::Path::new(path).exists() {
        return None;
    }
    let face = Face::from_path(path).ok()?;
    face.set_pixel_size(size).ok()?;
    Some(face)
}

/// Copy a FreeType bitmap into a tightly packed top-down coverage buffer of
/// `width * rows` bytes, dropping inter-row pitch padding. Row y starts at
/// `buffer + y*pitch` for either pitch sign. Assumes one coverage byte per
/// pixel, so the caller must confirm FT_PIXEL_MODE_GRAY first.
fn copy_coverage(bitmap: &FtBitmap) -> Vec<u8> {
    let width = bitmap.width as usize;
    let rows = bitmap.rows as usize;
    let total = rows * width;
    if width == 0 || rows == 0 || bitmap.buffer.is_null() {
        return Vec::new();
    }
    let mut out = vec![0u8; total];
    let pitch = bitmap.pitch as isize;
    for y in 0..rows {
        // SAFETY: FreeType guarantees |pitch| >= width bytes per row, so the
        // width-byte read at buffer + y*pitch stays within that row; out is a
        // distinct buffer.
        unsafe {
            let src = bitmap.buffer.offset(y as isize * pitch);
            ptr::copy_nonoverlapping(src, out[y * width..].as_mut_ptr(), width);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open() -> Face {
        Face::open_default(16).expect("a default font should be available")
    }

    #[test]
    fn rasterizes_a_visible_glyph() {
        let face = open();
        let g = face.rasterize('A');
        assert!(g.rows > 0 && g.width > 0, "A should have ink");
        assert_eq!(g.coverage.len(), g.rows * g.width);
        assert!(
            g.coverage.iter().any(|&c| c > 0),
            "A should have inked pixels"
        );
        assert!(g.advance > 0.0);
    }

    #[test]
    fn with_glyph_matches_the_uncached_rasterization() {
        let face = open();
        let direct = face.rasterize('g');
        // First access populates the cache; the handed-over glyph must match the
        // uncached render byte for byte.
        face.with_glyph('g', |cached| {
            assert_eq!(cached.left, direct.left);
            assert_eq!(cached.top, direct.top);
            assert_eq!(cached.width, direct.width);
            assert_eq!(cached.rows, direct.rows);
            assert_eq!(cached.advance, direct.advance);
            assert_eq!(cached.coverage, direct.coverage);
        });
    }

    #[test]
    fn with_glyph_reuses_the_cached_entry() {
        let face = open();
        // A second access returns the same coverage without re-rasterizing; the
        // cache holds exactly one entry for the character.
        let first = face.with_glyph('Q', |g| g.coverage.clone());
        let second = face.with_glyph('Q', |g| g.coverage.clone());
        assert_eq!(first, second);
        assert_eq!(face.glyphs.borrow().len(), 1);
    }

    #[test]
    fn space_is_blank_but_advances() {
        let face = open();
        let g = face.rasterize(' ');
        assert_eq!(g.rows, 0);
        assert!(g.coverage.is_empty());
        assert!(g.advance > 0.0, "space still advances the pen");
    }

    #[test]
    fn line_metrics_have_expected_signs() {
        let face = open();
        let (ascent, descent) = face.line_metrics();
        assert!(ascent > 0.0);
        assert!(descent < 0.0);
    }

    #[test]
    fn line_height_covers_ascent_plus_descent() {
        let face = open();
        let (ascent, descent) = face.line_metrics();
        let height = face.line_height();
        // Baseline-to-baseline spacing is at least the ink height (ascent above
        // the baseline plus the descent below it).
        assert!(height >= ascent - descent, "height {height} too small");
    }

    #[test]
    fn metrics_are_positive_and_consistent() {
        let m = open().metrics();
        assert!(m.ascent > 0 && m.descent > 0);
        assert!(m.line_height >= m.ascent + m.descent);
    }

    #[test]
    fn fonts_scale_metrics_with_size() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        let small = fonts.metrics(16);
        let large = fonts.metrics(32);
        assert!(large.line_height > small.line_height, "32px is taller");
    }

    #[test]
    fn fonts_fall_back_for_an_unopened_size() {
        let fonts = Fonts::new(&[16]).expect("a default font");
        // An unrequested size resolves to the only opened face, not a panic.
        assert_eq!(fonts.metrics(99).line_height, fonts.metrics(16).line_height);
    }

    #[test]
    fn face_resolves_every_style_without_panicking() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        // Every (size, style) pair resolves to a usable face: a present variant,
        // or the regular face when the family lacks it, or the fallback size.
        for &size in &[16, 32, 99] {
            for style in [
                FontStyle::Regular,
                FontStyle::Bold,
                FontStyle::Italic,
                FontStyle::BoldItalic,
            ] {
                assert!(
                    fonts.face(size, style).advance('m') > 0.0,
                    "{style:?} at {size}px advances"
                );
            }
        }
    }

    #[test]
    fn fonts_reject_an_empty_size_set() {
        assert!(Fonts::new(&[]).is_err());
        assert!(Fonts::new(&[0]).is_err());
    }

    #[test]
    fn code_face_resolves_for_every_size_without_panicking() {
        let fonts = Fonts::new(&[16, 32]).expect("a default font");
        // The code face is usable at opened and unopened sizes: a distinct code
        // family if installed, else the prose regular face. Never panics.
        for &size in &[16, 32, 99] {
            assert!(
                fonts.code_face(size).advance('m') > 0.0,
                "code face at {size}px advances"
            );
        }
    }
}
