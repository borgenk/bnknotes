//! Decoding user-provided image bytes into a flat ARGB8888 buffer for inline
//! display. This is the single seam over the external decoders (`zune` for
//! PNG/JPEG, `image-webp` for WebP): the rest of the editor speaks only
//! [`Image`] and never imports a decoder crate, so a backend can be swapped by
//! rewriting this one file.
//!
//! Image decoders parse hostile, attacker-controlled bytes, so this is the one
//! place the editor leans on memory-safe third-party code rather than an
//! in-tree decoder. [`Limits`] guards against decompression bombs: an image's declared
//! dimensions are read from its header and checked before any pixel buffer is
//! allocated. The zune decoders' own built-in dimension cap (16384 square) sits
//! above that as a hard backstop, so an absurd size is refused even sooner.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use zune_core::options::DecoderOptions;

use crate::platform::error::{Error, Result};

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
    /// file extension is not trusted). PNG, JPEG, and WebP; any other signature
    /// is rejected. `limits` bounds the dimensions that are accepted.
    pub fn decode(bytes: &[u8], limits: Limits) -> Result<Image> {
        match Format::sniff(bytes) {
            Some(Format::Png) => decode_png(bytes, limits),
            Some(Format::Jpeg) => decode_jpeg(bytes, limits),
            Some(Format::Webp) => decode_webp(bytes, limits),
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
    Webp,
}

impl Format {
    fn sniff(bytes: &[u8]) -> Option<Format> {
        const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF];
        if bytes.starts_with(PNG) {
            Some(Format::Png)
        } else if bytes.starts_with(JPEG) {
            Some(Format::Jpeg)
        } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            // WebP is a RIFF container: "RIFF" <u32 chunk size> "WEBP". Both
            // anchors are checked so a different RIFF payload (wav, avi) is not
            // taken for an image.
            Some(Format::Webp)
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

fn decode_webp(bytes: &[u8], limits: Limits) -> Result<Image> {
    use image_webp::WebPDecoder;

    // `new` parses the RIFF header, so dimensions are known (and checked) before
    // any pixel buffer is sized.
    let mut decoder =
        WebPDecoder::new(Cursor::new(bytes)).map_err(|_| Error::msg("invalid webp"))?;
    let (w, h) = decoder.dimensions();
    check_dimensions(w as usize, h as usize, limits)?;
    // The decoder reports the exact buffer it needs: w*h*3 (RGB) or w*h*4 (RGBA),
    // which `pack` then reads back as a channel count. `None` only on a dimension
    // overflow, already ruled out by check_dimensions.
    let size = decoder
        .output_buffer_size()
        .ok_or_else(|| Error::msg("webp dimensions overflow"))?;
    let mut buf = vec![0u8; size];
    // Decodes the first frame; an animated webp shows that frame as a still.
    decoder
        .read_image(&mut buf)
        .map_err(|_| Error::msg("webp decode failed"))?;
    pack(buf, w, h)
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
/// The store owns the decoded pixels; the layout layer borrows it to size image
/// lines, and the painter borrows it to blit them.
pub struct ImageStore {
    /// The document's directory, that relative image paths resolve against.
    /// `None` for an unsaved buffer, where only absolute paths resolve.
    dir: Option<PathBuf>,
    /// The vault root, the base for attachment lookups. A relative path
    /// missing beside the note is looked for here and under its attachment folders,
    /// so a bare embed name like `Pasted image ….png` resolves. `None` disables the
    /// vault fallback (the buffer has no vault).
    vault: Option<PathBuf>,
    /// Decoded images by raw path. A `None` value is a negative entry: the path
    /// was tried and could not be read or decoded.
    cache: HashMap<String, Option<Image>>,
    limits: Limits,
}

/// Attachment subfolders tried under the vault root when a relative embed name is
/// not beside the note: the folder names notes apps commonly default to. The vault
/// root itself is tried before these.
const ATTACHMENT_DIRS: [&str; 3] = ["attachments", "assets", "@attachment"];

impl ImageStore {
    /// An empty store resolving relative paths against `dir` (or only absolute
    /// paths when `dir` is `None`). The cache allocates nothing until used.
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            vault: None,
            cache: HashMap::new(),
            limits: Limits::default(),
        }
    }

    /// Point the store at the vault root, the base for attachment lookups,
    /// so a relative embed name resolves against the vault and its attachment
    /// folders as well as the note's directory. Clears the cache when the root
    /// actually changes, since a name that failed to resolve before may now be
    /// found (and vice versa).
    pub fn set_vault(&mut self, vault: Option<PathBuf>) {
        if self.vault != vault {
            self.vault = vault;
            self.cache.clear();
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
        self.candidates(raw_path).into_iter().find_map(|cand| {
            let bytes = std::fs::read(&cand).ok()?;
            Image::decode(&bytes, self.limits).ok()
        })
    }

    /// The filesystem paths to try for `raw_path`, in order. An absolute path is
    /// used as-is. A relative path resolves first against the note's own directory,
    /// then (when a vault is set) against the vault root and its attachment
    /// subfolders, so a bare `Pasted image ….png` is found wherever the vault filed
    /// it. Duplicate bases (a note that sits at the vault root) collapse to one try.
    fn candidates(&self, raw_path: &str) -> Vec<PathBuf> {
        let path = Path::new(raw_path);
        if path.is_absolute() {
            return vec![path.to_path_buf()];
        }
        let mut bases: Vec<PathBuf> = Vec::new();
        if let Some(dir) = &self.dir {
            bases.push(dir.clone());
        }
        if let Some(vault) = &self.vault {
            bases.push(vault.clone());
            bases.extend(ATTACHMENT_DIRS.iter().map(|sub| vault.join(sub)));
        }
        if bases.is_empty() {
            // Unsaved buffer with no vault: use the path as given.
            return vec![path.to_path_buf()];
        }
        let mut out: Vec<PathBuf> = Vec::with_capacity(bases.len());
        for base in bases {
            let candidate = base.join(path);
            if !out.contains(&candidate) {
                out.push(candidate);
            }
        }
        out
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
        self.candidates(raw_path).into_iter().find_map(|cand| {
            let bytes = std::fs::read(&cand).ok()?;
            let mime = match Format::sniff(&bytes)? {
                Format::Png => "image/png",
                Format::Jpeg => "image/jpeg",
                Format::Webp => "image/webp",
            };
            Some((bytes, mime))
        })
    }

    /// A store pre-populated with already-decoded images, bypassing the filesystem:
    /// used by tests and the performance harness.
    pub(crate) fn from_decoded(items: Vec<(String, Image)>) -> Self {
        let cache = items.into_iter().map(|(k, v)| (k, Some(v))).collect();
        Self {
            dir: None,
            vault: None,
            cache,
            limits: Limits::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real, tiny fixtures generated offline (no external sources): a 2x2 RGBA PNG,
    // a 16x16 solid-red JPEG, and the same 2x2 RGBA image as a lossless (VP8L)
    // WebP. See src/render/testdata.
    const TINY_PNG: &[u8] = include_bytes!("testdata/tiny.png");
    const TINY_JPEG: &[u8] = include_bytes!("testdata/tiny.jpg");
    const TINY_WEBP: &[u8] = include_bytes!("testdata/tiny.webp");

    #[test]
    fn sniffs_png_jpeg_webp_and_rejects_others() {
        assert!(matches!(Format::sniff(TINY_PNG), Some(Format::Png)));
        assert!(matches!(Format::sniff(TINY_JPEG), Some(Format::Jpeg)));
        assert!(matches!(Format::sniff(TINY_WEBP), Some(Format::Webp)));
        assert!(Format::sniff(b"GIF89a....").is_none());
        // A RIFF container that is not WebP (a wav file) shares the leading magic
        // but not the form tag, so it is not mistaken for an image.
        assert!(Format::sniff(b"RIFF\0\0\0\0WAVEfmt ").is_none());
        assert!(Format::sniff(b"").is_none());
        assert!(
            Format::sniff(&[0x89]).is_none(),
            "a truncated magic is not png"
        );
    }

    #[test]
    fn decodes_lossless_webp_preserving_alpha() {
        let img = Image::decode(TINY_WEBP, Limits::default()).expect("decode webp");
        assert_eq!((img.width, img.height), (2, 2));
        // The same 2x2 fixture as the png, encoded lossless (VP8L), so the exact
        // pixels survive: red(opaque), green(half), blue(opaque), transparent.
        assert_eq!(img.pixels[0], argb(255, 255, 0, 0));
        assert_eq!(img.pixels[1], argb(128, 0, 255, 0));
        assert_eq!(img.pixels[2], argb(255, 0, 0, 255));
        assert_eq!(img.pixels[3], argb(0, 0, 0, 0));
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

    #[test]
    fn store_reads_source_bytes_and_webp_mime_for_clipboard() {
        let dir = temp_dir("read-source-webp");
        std::fs::write(dir.join("pic.webp"), TINY_WEBP).unwrap();
        let store = ImageStore::new(Some(dir.clone()));
        // The clipboard copies the file as-is (no re-encode), so the bytes come
        // back verbatim alongside the sniffed MIME.
        let (bytes, mime) = store.read_source("pic.webp").expect("read source");
        assert_eq!(bytes, TINY_WEBP);
        assert_eq!(mime, "image/webp");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_resolves_a_bare_name_from_the_vault_root() {
        let vault = temp_dir("vault-root");
        let notes = vault.join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        // The picture sits at the vault root, not beside the note (a paste filed at
        // the default attachment location).
        std::fs::write(vault.join("Pasted image 1.png"), TINY_PNG).unwrap();
        let mut store = ImageStore::new(Some(notes));
        store.set_vault(Some(vault.clone()));
        store.ensure("Pasted image 1.png");
        assert_eq!(store.dimensions("Pasted image 1.png"), Some((2, 2)));
        let _ = std::fs::remove_dir_all(&vault);
    }

    #[test]
    fn store_resolves_a_bare_name_from_a_vault_attachment_folder() {
        let vault = temp_dir("vault-attach");
        std::fs::create_dir_all(vault.join("attachments")).unwrap();
        std::fs::create_dir_all(vault.join("@attachment")).unwrap();
        std::fs::write(vault.join("attachments/cat.png"), TINY_PNG).unwrap();
        std::fs::write(vault.join("@attachment/dog.png"), TINY_PNG).unwrap();
        // Only the attachment folders hold these; the note's directory does not.
        let mut store = ImageStore::new(Some(vault.join("notes")));
        store.set_vault(Some(vault.clone()));
        store.ensure("cat.png");
        store.ensure("dog.png");
        assert_eq!(store.dimensions("cat.png"), Some((2, 2)));
        assert_eq!(store.dimensions("dog.png"), Some((2, 2)));
        let _ = std::fs::remove_dir_all(&vault);
    }

    #[test]
    fn setting_the_vault_reattempts_a_previously_missing_name() {
        let vault = temp_dir("vault-late");
        let notes = vault.join("notes");
        std::fs::create_dir_all(&notes).unwrap();
        std::fs::write(vault.join("cat.png"), TINY_PNG).unwrap();
        let mut store = ImageStore::new(Some(notes));
        // Without a vault the name is not beside the note: a negative entry.
        store.ensure("cat.png");
        assert!(store.get("cat.png").is_none());
        // Learning the vault clears the negative cache; the name now resolves.
        store.set_vault(Some(vault.clone()));
        store.ensure("cat.png");
        assert_eq!(store.dimensions("cat.png"), Some((2, 2)));
        let _ = std::fs::remove_dir_all(&vault);
    }
}
