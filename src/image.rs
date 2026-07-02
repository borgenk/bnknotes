//! Decoding user-provided image bytes into a flat ARGB8888 buffer for inline
//! display. This is the single seam over the external `zune` decoders: the rest
//! of the editor speaks only [`Image`] and never imports a decoder crate, so the
//! backend can be swapped by rewriting this one file.
//!
//! Image decoders parse hostile, attacker-controlled bytes, so this is the one
//! place the editor leans on memory-safe third-party code rather than an
//! in-tree decoder. [`Limits`] guards against decompression bombs: an image's declared
//! dimensions are read from its header and checked before any pixel buffer is
//! allocated. The decoders' own built-in dimension cap (16384 square) sits above
//! that as a hard backstop, so an absurd size is refused even sooner.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use zune_core::options::DecoderOptions;

use crate::error::{Error, Result};

/// A decoded image: ARGB8888 pixels (`0xAARRGGBB`, the byte order of the editor's
/// xRGB8888 shm buffer) plus dimensions. One heap allocation, sampled by the
/// renderer without further copying.
pub struct Image {
    pub pixels: Vec<u32>,
    pub width: u32,
    pub height: u32,
}

/// Caps applied before decoding to bound work and memory on hostile or malformed
/// input. An image whose declared dimensions exceed these is rejected before its
/// pixel buffer is allocated.
#[derive(Clone, Copy)]
pub struct Limits {
    pub max_width: u32,
    pub max_height: u32,
}

impl Default for Limits {
    fn default() -> Self {
        // 8K square. Comfortably covers any sane inline image; anything larger is
        // far more likely a decompression bomb or a mistake than a real picture.
        Self {
            max_width: 8192,
            max_height: 8192,
        }
    }
}

impl Image {
    /// Decode `bytes`, sniffing the format from its leading magic number (the
    /// file extension is not trusted). PNG and JPEG only for now; any other
    /// signature is rejected. `limits` bounds the dimensions that are accepted.
    pub fn decode(bytes: &[u8], limits: Limits) -> Result<Image> {
        match Format::sniff(bytes) {
            Some(Format::Png) => decode_png(bytes, limits),
            Some(Format::Jpeg) => decode_jpeg(bytes, limits),
            None => Err(Error::msg("unsupported or unrecognized image format")),
        }
    }
}

/// The image formats the editor decodes, identified by their leading bytes.
/// Sniffing the content rather than trusting the path's extension is both more
/// robust (a mislabeled file still decodes) and the safer default.
enum Format {
    Png,
    Jpeg,
}

impl Format {
    fn sniff(bytes: &[u8]) -> Option<Format> {
        const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF];
        if bytes.starts_with(PNG) {
            Some(Format::Png)
        } else if bytes.starts_with(JPEG) {
            Some(Format::Jpeg)
        } else {
            None
        }
    }
}

/// Reject a zero or over-limit image up front, so [`pack`] and the decoders only
/// ever see dimensions we are willing to allocate a buffer for.
fn check_dimensions(w: usize, h: usize, limits: Limits) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(Error::msg("image has a zero dimension"));
    }
    if w > limits.max_width as usize || h > limits.max_height as usize {
        return Err(Error::msg(format!(
            "image {w}x{h} exceeds the {}x{} limit",
            limits.max_width, limits.max_height
        )));
    }
    Ok(())
}

fn decode_png(bytes: &[u8], limits: Limits) -> Result<Image> {
    use zune_core::result::DecodingResult;
    use zune_png::PngDecoder;

    // `add_alpha` keeps palette/RGB transparency (tRNS) by emitting an alpha
    // channel rather than dropping it; `pack` then handles whatever channel
    // count comes out.
    let opts = DecoderOptions::default().png_set_add_alpha_channel(true);
    let mut decoder = PngDecoder::new_with_options(Cursor::new(bytes), opts);
    decoder
        .decode_headers()
        .map_err(|_| Error::msg("invalid png"))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| Error::msg("png has no dimensions"))?;
    check_dimensions(w, h, limits)?;
    let pixels = match decoder
        .decode()
        .map_err(|_| Error::msg("png decode failed"))?
    {
        DecodingResult::U8(px) => px,
        // Down-convert 16-bit channels to 8-bit; the editor renders at 8 bits.
        DecodingResult::U16(px) => px.iter().map(|&v| (v >> 8) as u8).collect(),
        _ => return Err(Error::msg("unsupported png sample type")),
    };
    pack(pixels, w as u32, h as u32)
}

fn decode_jpeg(bytes: &[u8], limits: Limits) -> Result<Image> {
    use zune_core::colorspace::ColorSpace;
    use zune_jpeg::JpegDecoder;

    // Ask zune for RGBA so it does the YCbCr/CMYK -> RGB conversion (the gnarly,
    // security-sensitive part) internally and we always pack a 4-channel buffer.
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGBA);
    let mut decoder = JpegDecoder::new_with_options(Cursor::new(bytes), opts);
    decoder
        .decode_headers()
        .map_err(|_| Error::msg("invalid jpeg"))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| Error::msg("jpeg has no dimensions"))?;
    check_dimensions(w, h, limits)?;
    let pixels = decoder
        .decode()
        .map_err(|_| Error::msg("jpeg decode failed"))?;
    pack(pixels, w as u32, h as u32)
}

/// Fold decoded bytes into ARGB8888 `u32`s, inferring the channel count from the
/// buffer length so grayscale (1), grayscale+alpha (2), RGB (3), and RGBA (4) all
/// convert correctly. The single normalization point: every decoder funnels here,
/// so the rest of the editor only ever meets straight-alpha ARGB.
fn pack(bytes: Vec<u8>, w: u32, h: u32) -> Result<Image> {
    let count = (w as usize) * (h as usize);
    if count == 0 {
        return Err(Error::msg("image has a zero dimension"));
    }
    if bytes.is_empty() || !bytes.len().is_multiple_of(count) {
        return Err(Error::msg("decoded image size does not match dimensions"));
    }
    let channels = bytes.len() / count;
    let mut pixels = Vec::with_capacity(count);
    match channels {
        4 => pixels.extend(bytes.chunks_exact(4).map(|p| argb(p[3], p[0], p[1], p[2]))),
        3 => pixels.extend(bytes.chunks_exact(3).map(|p| argb(255, p[0], p[1], p[2]))),
        2 => pixels.extend(bytes.chunks_exact(2).map(|p| argb(p[1], p[0], p[0], p[0]))),
        1 => pixels.extend(bytes.iter().map(|&g| argb(255, g, g, g))),
        _ => return Err(Error::msg("unexpected channel count")),
    }
    Ok(Image {
        pixels,
        width: w,
        height: h,
    })
}

const fn argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// A cache of decoded inline images, keyed by the destination path exactly as
/// written in the document. Each referenced image is read and decoded at most
/// once; a repeated path or a redraw is a cache hit. Relative paths resolve
/// against the document's directory. A path that fails to read or decode is kept
/// as a negative entry so a broken link is not retried on every frame.
///
/// The store owns the decoded pixels; [`crate::layout`] borrows it to size image
/// lines, and the painter borrows it to blit them.
pub struct ImageStore {
    /// The document's directory, that relative image paths resolve against.
    /// `None` for an unsaved buffer, where only absolute paths resolve.
    dir: Option<PathBuf>,
    /// Decoded images by raw path. A `None` value is a negative entry: the path
    /// was tried and could not be read or decoded.
    cache: HashMap<String, Option<Image>>,
    /// Display-size bitmaps by (raw path, width, height): the decoded image
    /// resampled to the size it is drawn at. Bilinear resampling is expensive, so
    /// it runs once per (image, size) and every frame just blits the cached bitmap.
    /// Interior-mutable so the painter (which holds `&self`) can fill it lazily,
    /// the same pattern the glyph cache uses.
    scaled: RefCell<HashMap<(String, u32, u32), Vec<u32>>>,
    limits: Limits,
}

impl ImageStore {
    /// An empty store resolving relative paths against `dir` (or only absolute
    /// paths when `dir` is `None`). The caches allocate nothing until used.
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            cache: HashMap::new(),
            scaled: RefCell::new(HashMap::new()),
            limits: Limits::default(),
        }
    }

    /// Decode `raw_path` unless it has already been attempted, caching the result
    /// (a decoded image or a negative entry). Cheap to call every layout: a hit
    /// returns at once, so only a newly referenced path does any I/O.
    pub fn ensure(&mut self, raw_path: &str) {
        if self.cache.contains_key(raw_path) {
            return;
        }
        let decoded = self.load(raw_path);
        self.cache.insert(raw_path.to_string(), decoded);
    }

    fn load(&self, raw_path: &str) -> Option<Image> {
        let bytes = std::fs::read(self.resolve(raw_path)).ok()?;
        Image::decode(&bytes, self.limits).ok()
    }

    /// Resolve `raw_path` to a filesystem path: an absolute path as-is, a relative
    /// one against the document's directory (or as-is when the buffer is unsaved).
    fn resolve(&self, raw_path: &str) -> PathBuf {
        let path = Path::new(raw_path);
        match &self.dir {
            Some(dir) if path.is_relative() => dir.join(path),
            _ => path.to_path_buf(),
        }
    }

    /// The decoded image for `raw_path`, if it decoded successfully.
    pub fn get(&self, raw_path: &str) -> Option<&Image> {
        self.cache.get(raw_path).and_then(Option::as_ref)
    }

    /// The intrinsic pixel size of `raw_path`'s decoded image, if available.
    pub fn dimensions(&self, raw_path: &str) -> Option<(u32, u32)> {
        self.get(raw_path).map(|img| (img.width, img.height))
    }

    /// The raw bytes of `raw_path`'s source file and the MIME type of its format,
    /// for putting the image on the clipboard. These are the file as-is (lossless),
    /// not a re-encode of the decoded pixels, so no encoder is needed. `None` if the
    /// file can no longer be read or is not a recognized image.
    pub fn read_source(&self, raw_path: &str) -> Option<(Vec<u8>, &'static str)> {
        let bytes = std::fs::read(self.resolve(raw_path)).ok()?;
        let mime = match Format::sniff(&bytes)? {
            Format::Png => "image/png",
            Format::Jpeg => "image/jpeg",
        };
        Some((bytes, mime))
    }

    /// Call `f` with `raw_path`'s decoded image resampled to `w` x `h`, caching the
    /// resampled bitmap so the expensive bilinear pass runs once per (image, size)
    /// rather than every frame. The bitmap is borrowed for the duration of `f` (the
    /// painter blits it there). Returns `None` if the image is not decoded.
    pub fn with_scaled<R>(
        &self,
        raw_path: &str,
        w: u32,
        h: u32,
        f: impl FnOnce(&[u32]) -> R,
    ) -> Option<R> {
        let key = (raw_path.to_string(), w, h);
        if !self.scaled.borrow().contains_key(&key) {
            let src = self.get(raw_path)?;
            let bitmap = crate::render::resample_argb(
                &src.pixels,
                src.width,
                src.height,
                w as i32,
                h as i32,
            );
            self.scaled.borrow_mut().insert(key.clone(), bitmap);
        }
        let cache = self.scaled.borrow();
        cache.get(&key).map(|bitmap| f(bitmap))
    }

    /// A store pre-populated with already-decoded images, bypassing the filesystem:
    /// used by tests and the performance harness.
    pub(crate) fn from_decoded(items: Vec<(String, Image)>) -> Self {
        let cache = items.into_iter().map(|(k, v)| (k, Some(v))).collect();
        Self {
            dir: None,
            cache,
            scaled: RefCell::new(HashMap::new()),
            limits: Limits::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real, tiny fixtures generated offline (no external sources): a 2x2 RGBA PNG
    // and a 16x16 solid-red JPEG. See src/testdata.
    const TINY_PNG: &[u8] = include_bytes!("testdata/tiny.png");
    const TINY_JPEG: &[u8] = include_bytes!("testdata/tiny.jpg");

    #[test]
    fn sniffs_png_and_jpeg_and_rejects_others() {
        assert!(matches!(Format::sniff(TINY_PNG), Some(Format::Png)));
        assert!(matches!(Format::sniff(TINY_JPEG), Some(Format::Jpeg)));
        assert!(Format::sniff(b"GIF89a....").is_none());
        assert!(Format::sniff(b"").is_none());
        assert!(
            Format::sniff(&[0x89]).is_none(),
            "a truncated magic is not png"
        );
    }

    #[test]
    fn decodes_png_to_argb_preserving_alpha() {
        let img = Image::decode(TINY_PNG, Limits::default()).expect("decode png");
        assert_eq!((img.width, img.height), (2, 2));
        // The exact pixels the fixture was built from: red(opaque), green(half),
        // blue(opaque), transparent. Alpha survives the round trip.
        assert_eq!(img.pixels[0], argb(255, 255, 0, 0));
        assert_eq!(img.pixels[1], argb(128, 0, 255, 0));
        assert_eq!(img.pixels[2], argb(255, 0, 0, 255));
        assert_eq!(img.pixels[3], argb(0, 0, 0, 0));
    }

    #[test]
    fn decodes_jpeg_to_opaque_argb() {
        let img = Image::decode(TINY_JPEG, Limits::default()).expect("decode jpeg");
        assert_eq!((img.width, img.height), (16, 16));
        // JPEG is lossy, so assert the solid red reads back reddish and opaque
        // rather than an exact value.
        let px = img.pixels[0];
        let (a, r, g, b) = (px >> 24, (px >> 16) & 0xff, (px >> 8) & 0xff, px & 0xff);
        assert_eq!(a, 255, "jpeg has no alpha, so it packs opaque");
        assert!(r > 200, "red channel high, got {r}");
        assert!(g < 60 && b < 60, "green/blue low, got {g}/{b}");
    }

    #[test]
    fn rejects_image_over_the_limit() {
        // The 2x2 fixture exceeds a 1x1 cap, and is refused before allocation.
        let limits = Limits {
            max_width: 1,
            max_height: 1,
        };
        match Image::decode(TINY_PNG, limits) {
            Ok(_) => panic!("over-limit png should be rejected"),
            Err(e) => assert!(e.to_string().contains("exceeds"), "got: {e}"),
        }
    }

    #[test]
    fn rejects_unrecognized_bytes() {
        assert!(Image::decode(b"not an image at all", Limits::default()).is_err());
    }

    #[test]
    fn pack_infers_channels_from_length() {
        // RGB -> opaque ARGB.
        assert_eq!(
            pack(vec![255, 0, 0], 1, 1).unwrap().pixels,
            vec![argb(255, 255, 0, 0)]
        );
        // Grayscale -> opaque gray, replicated across channels.
        assert_eq!(
            pack(vec![128, 64], 2, 1).unwrap().pixels,
            vec![argb(255, 128, 128, 128), argb(255, 64, 64, 64)]
        );
        // Grayscale+alpha -> gray with its alpha.
        assert_eq!(
            pack(vec![200, 100], 1, 1).unwrap().pixels,
            vec![argb(100, 200, 200, 200)]
        );
        // RGBA -> ARGB, alpha preserved.
        assert_eq!(
            pack(vec![10, 20, 30, 40], 1, 1).unwrap().pixels,
            vec![argb(40, 10, 20, 30)]
        );
    }

    #[test]
    fn pack_rejects_size_mismatch_and_zero_dims() {
        assert!(
            pack(vec![1, 2, 3, 4, 5], 2, 1).is_err(),
            "5 bytes is not a whole channel count over 2 px"
        );
        assert!(pack(vec![], 1, 1).is_err(), "no bytes");
        assert!(pack(vec![1, 2, 3], 0, 1).is_err(), "zero width");
    }

    /// A fresh, empty temp directory unique to `tag` (one tag per test, so
    /// parallel tests never collide). Cleaned first in case a prior run crashed.
    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-imgstore-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn store_decodes_and_caches_a_relative_path() {
        let dir = temp_dir("relative");
        std::fs::write(dir.join("cat.png"), TINY_PNG).unwrap();
        let mut store = ImageStore::new(Some(dir.clone()));
        store.ensure("cat.png");
        assert_eq!(store.dimensions("cat.png"), Some((2, 2)));
        assert_eq!(store.get("cat.png").map(|i| i.pixels.len()), Some(4));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_resolves_an_absolute_path_ignoring_dir() {
        let dir = temp_dir("absolute");
        let file = dir.join("pic.png");
        std::fs::write(&file, TINY_PNG).unwrap();
        // No base dir set, but an absolute path still resolves.
        let mut store = ImageStore::new(None);
        let raw = file.to_str().unwrap().to_string();
        store.ensure(&raw);
        assert_eq!(store.dimensions(&raw), Some((2, 2)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_negative_caches_a_missing_file() {
        let dir = temp_dir("missing");
        let mut store = ImageStore::new(Some(dir.clone()));
        store.ensure("nope.png");
        assert!(store.get("nope.png").is_none());
        assert!(store.dimensions("nope.png").is_none());
        // The negative entry is recorded, so the path is not retried.
        assert!(store.cache.contains_key("nope.png"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_rejects_a_non_image_file() {
        let dir = temp_dir("garbage");
        std::fs::write(dir.join("note.txt"), b"not an image").unwrap();
        let mut store = ImageStore::new(Some(dir.clone()));
        store.ensure("note.txt");
        assert!(store.get("note.txt").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
