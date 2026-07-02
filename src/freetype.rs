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
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{Error, Result};
use crate::grapheme;
use crate::shape::Shaper;

/// A monospace family and the four style files the editor draws emphasis with.
/// The first family whose `regular` file exists wins; a missing bold, italic, or
/// bold-italic variant falls back to the regular face at render time, so a family
/// that ships only some weights still renders (just without the slant or weight).
/// Paths are hardcoded by design; a short list keeps the editor runnable
/// across machines without a font-discovery crate.
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FontStyle {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}

/// A self-contained description of which [`Face`] a run draws in, with no borrow
/// of [`Fonts`]. The display list stores this (rather than a `&Face`) so a built
/// frame outlives the borrow and can be diffed across frames; [`Fonts::face_for`]
/// resolves it back to a face at paint time. The two arms mirror the two families
/// [`Fonts`] holds: the prose family at a style, and the distinct code family.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FaceKey {
    Prose { size: u32, style: FontStyle },
    Code { size: u32 },
}

impl FaceKey {
    /// The pixel size this face is drawn at, carried by both arms.
    pub fn size(self) -> u32 {
        match self {
            FaceKey::Prose { size, .. } | FaceKey::Code { size } => size,
        }
    }
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
/// Load embedded color bitmaps (the CBDT strikes a color emoji font ships)
/// rather than skipping them (FT_LOAD_COLOR).
const FT_LOAD_COLOR: i32 = 1 << 20;
/// One coverage byte per pixel: the only mode whose row width in bytes equals
/// its pixel width (FT_PIXEL_MODE_GRAY).
const FT_PIXEL_MODE_GRAY: u8 = 2;
/// Four bytes per pixel, blue-green-red-alpha, premultiplied: what a CBDT color
/// strike decodes to (FT_PIXEL_MODE_BGRA).
const FT_PIXEL_MODE_BGRA: u8 = 7;

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
    fn FT_Select_Size(face: FtFace, strike_index: c_int) -> FtError;
    fn FT_Load_Char(face: FtFace, char_code: c_ulong, load_flags: i32) -> FtError;
    fn FT_Load_Glyph(face: FtFace, glyph_index: c_uint, load_flags: i32) -> FtError;
    fn FT_Get_Char_Index(face: FtFace, charcode: c_ulong) -> c_uint;
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
    /// Pen advances cached by character, the measure-only twin of `glyphs`. Layout
    /// measures every character to wrap and size runs but rasterizes only the ones
    /// on screen, and the advance is constant for a fixed-size face, so caching it
    /// turns a per-character `FT_Load_Char` (hinting plus metrics) into a map hit.
    /// Without this a full re-layout reloads every glyph's metrics through FreeType,
    /// which dominates the rebuild a selection drag triggers each frame.
    advances: RefCell<HashMap<char, f32>>,
    /// Glyph-cache hit and miss tallies, bumped on every [`Self::with_glyph`].
    /// Pure instrumentation (see [`CacheStats`]): a near-zero steady-state miss
    /// rate is the evidence that the cache is doing its job rather than thrashing.
    hits: Cell<u64>,
    misses: Cell<u64>,
    /// The pixel size this face was opened at, recorded by `set_pixel_size`.
    /// The emoji face scales its color glyphs to this so an emoji sits on the
    /// same em square as the surrounding text.
    pixel_size: Cell<u32>,
    /// The shared color emoji font, if one is installed; [`Fonts::new`] attaches
    /// it to every face it opens. `None` on the emoji face itself and in
    /// face-only tests, in which case emoji clusters take the per-character
    /// path (tofu, exactly the pre-emoji behavior).
    emoji: Option<Rc<EmojiFont>>,
}

/// A snapshot of a glyph cache's hit and miss tallies. Summing these across every
/// face turns "scrolling feels smooth" into a number: once the working set is
/// cached, steady-state frames should add hits and almost no misses.
#[derive(Clone, Copy, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    fn merge(self, other: Self) -> Self {
        Self {
            hits: self.hits + other.hits,
            misses: self.misses + other.misses,
        }
    }

    /// Total lookups (hits + misses).
    pub fn lookups(self) -> u64 {
        self.hits + self.misses
    }

    /// Fraction of lookups served from the cache, 0.0 when nothing was looked up.
    pub fn hit_rate(self) -> f32 {
        match self.lookups() {
            0 => 0.0,
            total => self.hits as f32 / total as f32,
        }
    }
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
            advances: RefCell::new(HashMap::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
            pixel_size: Cell::new(0),
            emoji: None,
        })
    }

    /// Select the pixel size FreeType renders and reports metrics at; width 0
    /// matches it to the height. Construction-only (private): the glyph and
    /// advance caches key on `char` alone, so changing a face's size after it has
    /// cached anything would serve stale-size glyphs. Every face is sized once,
    /// at open time, and never resized.
    fn set_pixel_size(&self, size: u32) -> Result<()> {
        self.pixel_size.set(size);
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
        use std::collections::hash_map::Entry;
        let mut cache = self.glyphs.borrow_mut();
        // `rasterize` reads only FreeType state, never the cache, so filling a
        // miss while the cache is borrowed cannot reenter this borrow.
        let glyph = match cache.entry(ch) {
            Entry::Occupied(e) => {
                self.hits.set(self.hits.get() + 1);
                e.into_mut()
            }
            Entry::Vacant(e) => {
                self.misses.set(self.misses.get() + 1);
                e.insert(self.rasterize(ch))
            }
        };
        f(glyph)
    }

    /// This face's cumulative glyph-cache hit and miss counts.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.get(),
            misses: self.misses.get(),
        }
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

    /// Pen advance of a character at the current pixel size, in pixels, cached on
    /// the face. The first measurement loads the glyph's metrics through FreeType;
    /// every later one is a map hit. Zero on a load failure.
    pub fn advance(&self, ch: char) -> f32 {
        // `measure_advance` reads only FreeType state, never this cache, so filling
        // a miss while the cache is borrowed cannot reenter this borrow.
        *self
            .advances
            .borrow_mut()
            .entry(ch)
            .or_insert_with(|| self.measure_advance(ch))
    }

    /// Load a character's pen advance from FreeType, metrics only (no
    /// rasterization), so measuring a run never pays for scan conversion. The
    /// uncached primitive behind [`Self::advance`]. Zero on a load failure.
    fn measure_advance(&self, ch: char) -> f32 {
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

    /// Pen advance of one grapheme cluster, in pixels: the emoji face's glyph
    /// advance when the cluster routes to color emoji, else the sum of the
    /// per-character advances (identical to measuring the characters one by
    /// one, so ASCII fast paths and this stay in exact agreement). This is the
    /// measuring twin of [`Self::with_cluster_glyph`]; the two must route the
    /// same way or the caret drifts from the pixels.
    pub fn cluster_advance(&self, cluster: &str) -> f32 {
        if let Some((emoji, px)) = self.emoji_cluster(cluster) {
            if let Some(advance) = emoji.advance(cluster, px) {
                return advance;
            }
        }
        cluster.chars().map(|c| self.advance(c)).sum()
    }

    /// Run `f` over the color glyph for an emoji cluster, or return `None` when
    /// the cluster renders through the ordinary per-character path (not an
    /// emoji, no emoji font installed, or the emoji font cannot form it).
    /// `None` tells the caller to draw the cluster's characters instead, which
    /// matches what [`Self::cluster_advance`] measured for that case.
    pub fn with_cluster_glyph<R>(
        &self,
        cluster: &str,
        f: impl FnOnce(&ColorGlyph) -> R,
    ) -> Option<R> {
        let (emoji, px) = self.emoji_cluster(cluster)?;
        emoji.with_glyph(cluster, px, f)
    }

    /// The emoji font and target pixel size for `cluster`, or `None` when the
    /// cluster should take the per-character path. The single routing decision
    /// both the measure and draw paths go through.
    fn emoji_cluster(&self, cluster: &str) -> Option<(&Rc<EmojiFont>, u32)> {
        let emoji = self.emoji.as_ref()?;
        if !wants_emoji(cluster, |c| self.has_scalar(c)) {
            return None;
        }
        Some((emoji, self.pixel_size.get()))
    }

    /// Whether this face's character map has a real glyph for `ch` (a missing
    /// character maps to glyph 0, `.notdef`, the tofu box).
    fn has_scalar(&self, ch: char) -> bool {
        // SAFETY: face is valid; the call is a read-only cmap lookup.
        unsafe { FT_Get_Char_Index(self.face.as_ptr(), ch as u32 as c_ulong) != 0 }
    }

    /// Select the face's first fixed bitmap strike and return its nominal pixel
    /// size, or `None` when the face has none (an outline font). Only the emoji
    /// font uses strikes; outline faces size with [`Self::set_pixel_size`].
    fn select_first_strike(&self) -> Option<f32> {
        // SAFETY: face is valid; num_fixed_sizes is a plain field read.
        let strikes = unsafe { (*self.face.as_ptr()).num_fixed_sizes };
        if strikes <= 0 {
            return None;
        }
        // SAFETY: strike 0 exists (checked above); the call selects it.
        let err = unsafe { FT_Select_Size(self.face.as_ptr(), 0) };
        if err != 0 {
            return None;
        }
        // SAFETY: a successful select populates face->size.
        let size = NonNull::new(unsafe { (*self.face.as_ptr()).size })?;
        let ppem = unsafe { size.as_ref().metrics.y_ppem };
        (ppem > 0).then(|| f32::from(ppem))
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

/// Whether a grapheme cluster should render as color emoji. `mono_has` reports
/// whether the text face can draw a character itself, so symbols the text font
/// covers (dagger, copyright, box drawing) stay text.
///
/// The rules, in order:
/// - a text-presentation selector (U+FE0E) anywhere forces the text path;
/// - a single scalar goes color only when it is `Extended_Pictographic` *and*
///   the text face has no glyph for it (a smiley, not a copyright sign);
/// - a multi-scalar cluster goes color when it carries an emoji-presentation
///   selector (U+FE0F) or keycap (U+20E3), starts pictographic (ZWJ sequences,
///   skin tones), or starts with a regional indicator (flags). Anything else
///   (plain combining marks) is ordinary text.
fn wants_emoji(cluster: &str, mono_has: impl Fn(char) -> bool) -> bool {
    let mut chars = cluster.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if cluster.contains('\u{FE0E}') {
        return false;
    }
    if chars.next().is_none() {
        return grapheme::is_extended_pictographic(first) && !mono_has(first);
    }
    cluster.contains('\u{FE0F}')
        || cluster.contains('\u{20E3}')
        || grapheme::is_extended_pictographic(first)
        || matches!(first, '\u{1F1E6}'..='\u{1F1FF}')
}

/// A rasterized emoji cluster at display size: straight (unpremultiplied)
/// `0xAARRGGBB` pixels, row-major, plus the same pen metrics [`Glyph`] carries.
/// Owned and cached by [`EmojiFont`]; the blitter draws straight out of it.
pub struct ColorGlyph {
    /// Horizontal offset from the pen to the bitmap's left edge.
    pub left: i32,
    /// Vertical offset from the baseline up to the bitmap's top edge.
    pub top: i32,
    pub width: usize,
    pub rows: usize,
    /// Pen advance in pixels.
    pub advance: f32,
    pub argb: Vec<u32>,
}

/// The usual install paths of the Noto color emoji font, tried in order. With none
/// of them present, the editor draws no emoji.
const EMOJI_FONTS: &[&str] = &[
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
];

/// A cache keyed by (cluster bytes, target pixel size). `None` records that
/// the font cannot form the cluster, so the miss is not re-shaped every frame.
type ClusterCache<T> = RefCell<HashMap<(Box<str>, u32), Option<T>>>;

/// The color emoji font: a bitmap-strike (CBDT) face plus a HarfBuzz shaper
/// over it. The shaper is what turns a multi-scalar cluster (ZWJ family, flag,
/// skin tone, keycap) into the single ligature glyph the font's GSUB table
/// defines; FreeType then decodes that glyph's color strike, and the result is
/// scaled to the text size and cached per (cluster, size).
///
/// Everything degrades gracefully: no installed font, an outline-only emoji
/// font, or a cluster the font cannot form all yield `None`, and the caller
/// falls back to per-character rendering.
pub struct EmojiFont {
    /// Declared before `face` deliberately: fields drop in declaration order,
    /// and the shaper's HarfBuzz font holds a reference to the FreeType face
    /// that it releases with `FT_Done_Face` on destroy. The face's own `Drop`
    /// tears down its whole FreeType library, so HarfBuzz must let go first.
    shaper: Shaper,
    face: Face,
    /// The strike's nominal pixel size; glyph bitmaps and shaped positions are
    /// scaled by `target / strike` to land on the text's em square.
    strike: f32,
    /// Rasterized clusters at display size.
    glyphs: ClusterCache<ColorGlyph>,
    /// The measure-only twin, filled by shaping alone (no strike decode), so
    /// layout can measure off-screen emoji without rasterizing them.
    advances: ClusterCache<f32>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl EmojiFont {
    /// Open the first installed emoji font, or `None` when there is none, the
    /// font has no bitmap strikes (an outline-only build), or the shaper cannot
    /// bind to it.
    pub fn open() -> Option<Self> {
        let path = EMOJI_FONTS
            .iter()
            .find(|p| std::path::Path::new(p).exists())?;
        let face = Face::from_path(path).ok()?;
        let strike = face.select_first_strike()?;
        // SAFETY: the face handle is valid and has a size selected; the shaper
        // takes its own FreeType reference, so drop order does not matter.
        let shaper = unsafe { Shaper::from_ft_face(face.face.as_ptr() as *mut c_void) }?;
        Some(Self {
            face,
            shaper,
            strike,
            glyphs: RefCell::new(HashMap::new()),
            advances: RefCell::new(HashMap::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        })
    }

    /// Pen advance of `cluster` scaled to `target` pixels, or `None` when the
    /// font cannot form the cluster. Cached; filled by shaping only, no bitmap
    /// is decoded.
    fn advance(&self, cluster: &str, target: u32) -> Option<f32> {
        *self
            .advances
            .borrow_mut()
            .entry((cluster.into(), target))
            .or_insert_with(|| {
                let glyphs = self.shaped(cluster)?;
                Some(glyphs.iter().map(|g| g.x_advance).sum::<f32>() * self.scale(target))
            })
    }

    /// Run `f` over the cached color glyph for `cluster` at `target` pixels,
    /// rasterizing (shape, decode strikes, composite, resample) on first use.
    /// `None` means the font cannot form this cluster, matching what
    /// [`Self::advance`] reported, since both derive from the same shaping.
    fn with_glyph<R>(
        &self,
        cluster: &str,
        target: u32,
        f: impl FnOnce(&ColorGlyph) -> R,
    ) -> Option<R> {
        use std::collections::hash_map::Entry;
        let mut cache = self.glyphs.borrow_mut();
        // `raster` reads only FreeType and shaper state, never this cache, so
        // filling a miss while the cache is borrowed cannot reenter this borrow.
        let glyph = match cache.entry((cluster.into(), target)) {
            Entry::Occupied(e) => {
                self.hits.set(self.hits.get() + 1);
                e.into_mut()
            }
            Entry::Vacant(e) => {
                self.misses.set(self.misses.get() + 1);
                e.insert(self.raster(cluster, target))
            }
        };
        glyph.as_ref().map(f)
    }

    /// This font's cluster-cache hit and miss counts, folded into the frame
    /// stats alongside the per-character caches.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.get(),
            misses: self.misses.get(),
        }
    }

    /// Shape `cluster` and keep the result only when the font can really form
    /// it: at least one glyph and no `.notdef`. The single success predicate
    /// behind both [`Self::advance`] and [`Self::with_glyph`], which is what
    /// keeps measuring and drawing in agreement.
    fn shaped(&self, cluster: &str) -> Option<Vec<crate::shape::ShapedGlyph>> {
        let glyphs = self.shaper.shape(cluster);
        if glyphs.is_empty() || glyphs.iter().any(|g| g.id == 0) {
            return None;
        }
        Some(glyphs)
    }

    /// How much a strike-sized measurement shrinks to land on `target` pixels.
    fn scale(&self, target: u32) -> f32 {
        target as f32 / self.strike
    }

    /// Rasterize `cluster` at `target` pixels: decode each shaped glyph's color
    /// strike, composite them along the pen into one strike-sized image, then
    /// resample once to the target size. Emoji clusters almost always shape to
    /// a single glyph, so the composite loop usually runs once.
    ///
    /// The pipeline stays *premultiplied* (as FreeType decodes it) through
    /// compositing and resampling, and un-premultiplies only the final
    /// display-size pixels. Filtering straight alpha would mix the black of
    /// transparent texels into edge colors and fringe the glyph outline dark.
    fn raster(&self, cluster: &str, target: u32) -> Option<ColorGlyph> {
        let shaped = self.shaped(cluster)?;
        let scale = self.scale(target);

        // Decode and place each glyph's bitmap in strike space. x/y are the
        // bitmap's left and top edges relative to the pen origin and baseline.
        let mut placed: Vec<(i32, i32, StrikeBitmap)> = Vec::new();
        let mut pen = 0.0f32;
        for g in &shaped {
            if let Some(bitmap) = self.load_strike(g.id) {
                let x = (pen + g.x_offset).round() as i32 + bitmap.left;
                let y = bitmap.top + g.y_offset.round() as i32;
                placed.push((x, y, bitmap));
            }
            pen += g.x_advance;
        }
        let advance = pen * scale;
        if placed.is_empty() {
            // Shaped but inkless; keep the advance so layout stays consistent.
            return Some(ColorGlyph {
                left: 0,
                top: 0,
                width: 0,
                rows: 0,
                advance,
                argb: Vec::new(),
            });
        }

        // The union box of every placed bitmap, still in strike space.
        let x0 = placed.iter().map(|&(x, _, _)| x).min().unwrap_or(0);
        let x1 = placed
            .iter()
            .map(|&(x, _, ref b)| x + b.width as i32)
            .max()
            .unwrap_or(0);
        let top = placed.iter().map(|&(_, y, _)| y).max().unwrap_or(0);
        let bottom = placed
            .iter()
            .map(|&(_, y, ref b)| y - b.rows as i32)
            .min()
            .unwrap_or(0);
        let (w, h) = ((x1 - x0).max(0) as usize, (top - bottom).max(0) as usize);
        let mut canvas = vec![0u32; w * h];
        for (x, y, bitmap) in &placed {
            for row in 0..bitmap.rows {
                let dst_row = (top - y) as usize + row;
                for col in 0..bitmap.width {
                    let dst_col = (x - x0) as usize + col;
                    let src = bitmap.argb[row * bitmap.width + col];
                    let dst = &mut canvas[dst_row * w + dst_col];
                    *dst = premultiplied_over(src, *dst);
                }
            }
        }

        let dst_w = ((w as f32) * scale).round().max(1.0) as i32;
        let dst_h = ((h as f32) * scale).round().max(1.0) as i32;
        let argb: Vec<u32> =
            crate::render::resample_argb(&canvas, w as u32, h as u32, dst_w, dst_h)
                .into_iter()
                .map(straight_from_premultiplied)
                .collect();
        Some(ColorGlyph {
            left: (x0 as f32 * scale).round() as i32,
            top: (top as f32 * scale).round() as i32,
            width: dst_w as usize,
            rows: dst_h as usize,
            advance,
            argb,
        })
    }

    /// Decode one glyph's color strike into premultiplied ARGB. `None` for a
    /// load failure or a non-BGRA result (an inkless or malformed glyph).
    fn load_strike(&self, glyph_id: u32) -> Option<StrikeBitmap> {
        // SAFETY: face is valid with a strike selected; FT_LOAD_COLOR decodes
        // the CBDT strike into the face's shared glyph slot, read out below
        // before any later FreeType call.
        let err = unsafe {
            FT_Load_Glyph(
                self.face.face.as_ptr(),
                glyph_id as c_uint,
                FT_LOAD_RENDER | FT_LOAD_COLOR,
            )
        };
        if err != 0 {
            return None;
        }
        // SAFETY: the load succeeded, so face->glyph points at the populated slot.
        let slot = unsafe { (*self.face.face.as_ptr()).glyph };
        let slot = NonNull::new(slot)?;
        let slot = unsafe { slot.as_ref() };
        if slot.bitmap.pixel_mode != FT_PIXEL_MODE_BGRA {
            return None;
        }
        Some(StrikeBitmap {
            left: slot.bitmap_left,
            top: slot.bitmap_top,
            width: slot.bitmap.width as usize,
            rows: slot.bitmap.rows as usize,
            argb: copy_bgra(&slot.bitmap),
        })
    }
}

/// One glyph's decoded color strike in strike space: premultiplied ARGB plus
/// its placement relative to the pen and baseline.
struct StrikeBitmap {
    left: i32,
    top: i32,
    width: usize,
    rows: usize,
    argb: Vec<u32>,
}

/// Premultiplied source-over composite of `src` on `dst`, both premultiplied
/// `0xAARRGGBB`: every channel, alpha included, is `s + d * (255 - sa) / 255`.
/// Used only when a shaped cluster places more than one glyph; the common
/// single-glyph case writes onto transparent pixels where this is a plain copy.
fn premultiplied_over(src: u32, dst: u32) -> u32 {
    let sa = src >> 24;
    if sa == 255 || dst == 0 {
        return src;
    }
    if sa == 0 {
        return dst;
    }
    let inv = 255 - sa;
    let ch = |shift: u32| ((src >> shift) & 0xff) + ((dst >> shift) & 0xff) * inv / 255;
    (ch(24) << 24) | (ch(16) << 16) | (ch(8) << 8) | ch(0)
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

impl SizedFaces {
    /// Every face this size actually opened: the regular face plus whichever
    /// variants and code face exist. Used to roll up cache stats across faces.
    fn faces(&self) -> impl Iterator<Item = &Face> {
        std::iter::once(&self.regular)
            .chain(self.bold.as_ref())
            .chain(self.italic.as_ref())
            .chain(self.bold_italic.as_ref())
            .chain(self.code.as_ref())
    }
}

/// A set of [`Face`]s at different pixel sizes and styles, so a document can mix
/// body text, larger headings, and inline emphasis. Each distinct size opens the
/// family's regular face plus whatever variant files exist (the same files,
/// re-read per size); the first size opened is the fallback for any size that was
/// not requested, and a missing variant falls back to the regular face, so a
/// lookup can never fail and never panics.
pub struct Fonts {
    sized: Vec<SizedFaces>,
    /// The color emoji font, opened once and shared by every face, or `None`
    /// when no emoji font is installed (emoji then render as tofu, as before).
    emoji: Option<Rc<EmojiFont>>,
}

impl Fonts {
    /// Open the default family at each of `sizes` (zeros and duplicates skipped),
    /// loading every available style per size. At least one nonzero size is
    /// required; the first opened becomes the fallback.
    pub fn new(sizes: &[u32]) -> Result<Self> {
        let family = default_family()?;
        let code_path = code_regular_path(family);
        let emoji = EmojiFont::open().map(Rc::new);
        let mut sized: Vec<SizedFaces> = Vec::new();
        for &size in sizes {
            if size == 0 || sized.iter().any(|s| s.size == size) {
                continue;
            }
            let mut regular = Face::from_path(family.regular)?;
            regular.set_pixel_size(size)?;
            regular.emoji = emoji.clone();
            let metrics = regular.metrics();
            sized.push(SizedFaces {
                size,
                regular,
                bold: open_variant(family.bold, size, emoji.as_ref()),
                italic: open_variant(family.italic, size, emoji.as_ref()),
                bold_italic: open_variant(family.bold_italic, size, emoji.as_ref()),
                code: code_path.and_then(|p| open_variant(p, size, emoji.as_ref())),
                metrics,
            });
        }
        if sized.is_empty() {
            return Err(Error::msg("Fonts::new needs at least one nonzero size"));
        }
        Ok(Self { sized, emoji })
    }

    /// The entry for `size`, or the fallback (first opened) if `size` is unknown.
    fn entry(&self, size: u32) -> &SizedFaces {
        // `sized` is never empty (Fonts::new errors otherwise), so the `[0]`
        // fallback for an unknown size cannot panic.
        debug_assert!(!self.sized.is_empty());
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

    /// Resolve a [`FaceKey`] to its face, the inverse of the key a run records.
    /// Routes to [`Self::face`] or [`Self::code_face`], so it inherits their
    /// fallbacks and never panics.
    pub fn face_for(&self, key: FaceKey) -> &Face {
        match key {
            FaceKey::Prose { size, style } => self.face(size, style),
            FaceKey::Code { size } => self.code_face(size),
        }
    }

    /// The metrics for `size`, or the fallback's metrics if `size` was not opened.
    pub fn metrics(&self, size: u32) -> Metrics {
        self.entry(size).metrics
    }

    /// Glyph-cache hits and misses summed over every open face (the emoji
    /// font's cluster cache included), for frame-loop instrumentation.
    pub fn cache_stats(&self) -> CacheStats {
        let emoji = self
            .emoji
            .as_ref()
            .map(|e| e.cache_stats())
            .unwrap_or_default();
        self.sized
            .iter()
            .flat_map(SizedFaces::faces)
            .map(Face::cache_stats)
            .fold(emoji, CacheStats::merge)
    }
}

/// Open a style variant at `size`, or `None` if the file is absent or fails to
/// load, leaving the caller to fall back to the regular face. Keeps a family
/// that ships only some weights from being a hard error.
fn open_variant(path: &str, size: u32, emoji: Option<&Rc<EmojiFont>>) -> Option<Face> {
    if !std::path::Path::new(path).exists() {
        return None;
    }
    let mut face = Face::from_path(path).ok()?;
    face.set_pixel_size(size).ok()?;
    face.emoji = emoji.cloned();
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

/// Copy a BGRA color strike into tightly packed `0xAARRGGBB` pixels, keeping
/// FreeType's premultiplied alpha as-is (a pure byte reorder). Compositing and
/// resampling stay premultiplied; [`straight_from_premultiplied`] converts the
/// final display-size pixels for the straight-alpha blitter. The caller must
/// confirm FT_PIXEL_MODE_BGRA first.
fn copy_bgra(bitmap: &FtBitmap) -> Vec<u32> {
    let width = bitmap.width as usize;
    let rows = bitmap.rows as usize;
    if width == 0 || rows == 0 || bitmap.buffer.is_null() {
        return Vec::new();
    }
    let mut out = vec![0u32; width * rows];
    let pitch = bitmap.pitch as isize;
    for y in 0..rows {
        for x in 0..width {
            // SAFETY: FreeType guarantees |pitch| >= width*4 bytes per BGRA row,
            // so the 4-byte read at buffer + y*pitch + x*4 stays within the row;
            // out is a distinct buffer.
            let [b, g, r, a] = unsafe {
                let p = bitmap.buffer.offset(y as isize * pitch + (x as isize) * 4);
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            };
            out[y * width + x] = u32::from_be_bytes([a, r, g, b]);
        }
    }
    out
}

/// Straight-alpha `0xAARRGGBB` from a premultiplied pixel, for the blitter,
/// which blends straight colors by coverage. Fully transparent maps to zero.
fn straight_from_premultiplied(px: u32) -> u32 {
    let a = px >> 24;
    if a == 0 {
        return 0;
    }
    let un = |shift: u32| (((px >> shift) & 0xff) * 255 / a).min(255) << shift;
    (a << 24) | un(16) | un(8) | un(0)
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
    fn cache_stats_count_misses_then_hits() {
        let face = open();
        // A fresh face has touched nothing.
        assert_eq!(face.cache_stats().lookups(), 0);
        // First touch of a character is a miss (it rasterizes), every repeat a hit.
        face.with_glyph('z', |_| {});
        face.with_glyph('z', |_| {});
        face.with_glyph('z', |_| {});
        // A different character is another miss.
        face.with_glyph('w', |_| {});
        let stats = face.cache_stats();
        assert_eq!(stats.misses, 2, "two distinct characters, two misses");
        assert_eq!(stats.hits, 2, "two repeat lookups of 'z'");
        assert!((stats.hit_rate() - 0.5).abs() < 1e-6, "2 hits of 4 lookups");
    }

    #[test]
    fn fonts_cache_stats_sum_across_faces() {
        let fonts = Fonts::new(&[16, 24]).expect("a default font should be available");
        assert_eq!(fonts.cache_stats().lookups(), 0);
        // Draw through two different sizes (hence different faces); the rolled-up
        // tally must see both faces' lookups.
        fonts.face(16, FontStyle::Regular).with_glyph('a', |_| {});
        fonts.face(24, FontStyle::Regular).with_glyph('b', |_| {});
        fonts.face(16, FontStyle::Regular).with_glyph('a', |_| {});
        let stats = fonts.cache_stats();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn advance_is_cached_and_matches_the_uncached_measure() {
        let face = open();
        // A repeat lookup must return the same width and populate exactly one entry.
        let direct = face.measure_advance('M');
        let first = face.advance('M');
        let second = face.advance('M');
        assert_eq!(first, direct, "cached advance matches the raw measure");
        assert_eq!(first, second);
        assert!(first > 0.0);
        assert_eq!(face.advances.borrow().len(), 1, "one cached character");
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

    fn emoji() -> EmojiFont {
        EmojiFont::open().expect("a color emoji font should be installed")
    }

    #[test]
    fn wants_emoji_routes_clusters_correctly() {
        let mono_lacks = |_: char| false;
        let mono_has = |_: char| true;
        // Plain text and combining marks never route to emoji.
        assert!(!wants_emoji("a", mono_lacks));
        assert!(
            !wants_emoji("e\u{301}", mono_lacks),
            "combining mark is text"
        );
        // A lone pictographic scalar goes color only when the text face lacks it.
        assert!(wants_emoji("😀", mono_lacks));
        assert!(!wants_emoji("😀", mono_has), "the text face covers it");
        // Presentation selectors win in both directions.
        assert!(wants_emoji("☺\u{fe0f}", mono_has), "emoji presentation");
        assert!(!wants_emoji("☺\u{fe0e}", mono_lacks), "text presentation");
        // Flags, skin tones, ZWJ sequences, and keycaps are emoji outright.
        assert!(wants_emoji("🇳🇴", mono_has));
        assert!(wants_emoji("👍🏽", mono_has));
        assert!(wants_emoji("👨\u{200d}👩\u{200d}👧", mono_has));
        assert!(wants_emoji("1\u{fe0f}\u{20e3}", mono_has));
    }

    #[test]
    fn emoji_font_measures_and_rasters_a_smiley() {
        let font = emoji();
        let advance = font.advance("😀", 32).expect("the smiley should shape");
        assert!(advance > 0.0);
        let (width, rows, has_ink, glyph_advance) = font
            .with_glyph("😀", 32, |g| {
                let ink = g.argb.iter().any(|&p| p >> 24 != 0);
                (g.width, g.rows, ink, g.advance)
            })
            .expect("the smiley should raster");
        assert!(width > 0 && rows > 0, "the smiley has pixels");
        assert!(has_ink, "the smiley has visible ink");
        assert!(
            rows <= 40,
            "scaled to the 32px em square, not the strike ({rows} rows)"
        );
        assert_eq!(glyph_advance, advance, "measure and raster agree");
    }

    #[test]
    fn zwj_and_flag_clusters_ligate_to_one_glyph() {
        let font = emoji();
        let single = font.advance("👨", 32).expect("the man emoji shapes");
        let family = font
            .advance("👨\u{200d}👩\u{200d}👧", 32)
            .expect("the family should ligate");
        assert!(
            family < single * 2.0,
            "a family is one ligature, not three glyphs ({family} vs {single})"
        );
        let flag = font.advance("🇳🇴", 32).expect("the flag should ligate");
        assert!(
            flag < single * 2.0,
            "a flag is one glyph, not two letter symbols"
        );
    }

    #[test]
    fn emoji_cluster_cache_hits_after_the_first_raster() {
        let font = emoji();
        font.with_glyph("🎉", 32, |_| ()).expect("party popper");
        font.with_glyph("🎉", 32, |_| ())
            .expect("party popper again");
        let stats = font.cache_stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
    }

    #[test]
    fn premultiplied_edges_resample_without_dark_fringe() {
        // Half yellow, half transparent, averaged to one pixel in premultiplied
        // space and converted to straight alpha afterwards: the color must stay
        // pure yellow at half coverage. Filtering straight alpha instead would
        // mix the transparent texel's black into the color and darken the edge,
        // which is exactly the fringe this pipeline order exists to avoid.
        let src = [0xff_ff_ff_00, 0x0000_0000];
        let avg = crate::render::resample_argb(&src, 2, 1, 1, 1)[0];
        let straight = straight_from_premultiplied(avg);
        assert_eq!(straight >> 24, 0x80, "half coverage");
        assert_eq!(straight & 0x00ff_ffff, 0x00ff_ff00, "still pure yellow");
    }

    #[test]
    fn premultiplied_over_keeps_translucent_sources_undarkened() {
        let half_yellow = 0x80_80_80_00; // premultiplied: a=128, r=g=128, b=0
        assert_eq!(
            premultiplied_over(half_yellow, 0),
            half_yellow,
            "compositing onto transparent is a plain copy"
        );
        let opaque = 0xff_10_20_30;
        assert_eq!(premultiplied_over(opaque, half_yellow), opaque);
        assert_eq!(premultiplied_over(0, half_yellow), half_yellow);
        // A half-alpha source over an opaque background accumulates both.
        let out = premultiplied_over(half_yellow, 0xff_00_00_ff);
        assert_eq!(out >> 24, 0xff, "alpha saturates over an opaque ground");
        assert_eq!((out >> 16) & 0xff, 0x80, "red comes from the source");
        assert_eq!(out & 0xff, 0x7f, "blue is the ground's remainder");
    }

    #[test]
    fn straight_from_premultiplied_recovers_color_and_maps_clear_to_zero() {
        assert_eq!(straight_from_premultiplied(0), 0);
        assert_eq!(straight_from_premultiplied(0x80_80_00_40), 0x80_ff_00_7f);
        let opaque = 0xff_12_34_56;
        assert_eq!(
            straight_from_premultiplied(opaque),
            opaque,
            "opaque is identity"
        );
    }

    #[test]
    fn an_unformable_cluster_answers_none() {
        let font = emoji();
        // A letter with a combining mark is no emoji ligature; the emoji font
        // has no glyphs for it and must say so rather than render garbage.
        // (Routing filters this out anyway; the font still answers safely.)
        assert!(font.advance("e\u{301}", 32).is_none());
        assert!(font.with_glyph("e\u{301}", 32, |_| ()).is_none());
    }

    #[test]
    fn faces_route_emoji_clusters_and_keep_ascii_on_the_char_path() {
        let fonts = Fonts::new(&[32]).expect("a default font");
        let face = fonts.face(32, FontStyle::Regular);
        let advance = face.cluster_advance("😀");
        assert!(advance > 0.0);
        assert_eq!(
            face.with_cluster_glyph("😀", |g| g.advance),
            Some(advance),
            "draw and measure route identically"
        );
        assert!(
            face.with_cluster_glyph("a", |_| ()).is_none(),
            "ASCII never routes to the emoji path"
        );
        assert_eq!(
            face.cluster_advance("a"),
            face.advance('a'),
            "a simple cluster measures exactly like its character"
        );
    }
}
