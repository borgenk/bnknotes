//! The GPU backend's CPU half: turning a display list into GPU-consumable
//! data. This module is deliberately free of Vulkan: it packs glyph rasters
//! into atlas mirrors and emits vertices and draw batches as plain values, so
//! the geometry and packing logic is unit-testable without a device.
//! `vulkan.rs` consumes the output ([`FrameData`]) verbatim.
//!
//! ```text
//!   &[DrawCmd] ─┬─ Fill/RoundRect ──────────────▶ quads (solid / SDF corners)
//!               ├─ Text ─▶ FreeType rasters ─▶ atlas slots ─▶ textured quads
//!               └─ Image ─▶ resample cache ──▶ per-image texture batches
//! ```
//!
//! Text traversal matches how layout measured each run (the ASCII fast path,
//! then grapheme clusters routed to the emoji or per-character path), so the
//! pixels the GPU composites land exactly where the caret and wrapping expect.

use std::collections::HashMap;

use crate::platform::emoji::wants_emoji;
use crate::platform::freetype::{CodeInk, Face, FaceKey, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::display::DrawCmd;
use crate::render::display::RoundedCorners;
use crate::render::image::ImageStore;

/// Vertex modes, matching shaders/quad.frag.
pub const MODE_SOLID: u32 = 0;
pub const MODE_GLYPH: u32 = 1;
pub const MODE_EMOJI: u32 = 2;
pub const MODE_IMAGE: u32 = 3;
pub const MODE_ROUND: u32 = 4;
pub const MODE_CHECK: u32 = 5;
pub const MODE_CHEVRON: u32 = 6;
pub const MODE_HATCH: u32 = 7;
pub const MODE_PIN: u32 = 8;

/// Atlases start here and double when full, up to [`ATLAS_MAX`]; growth wipes
/// the atlas (sources are cached in `Fonts`, so re-inserting is cheap) and the
/// frame build restarts so no stale coordinates survive.
const ATLAS_START: u32 = 1024;
const ATLAS_MAX: u32 = 8192;

/// One vertex, laid out exactly as the pipeline's vertex input expects
/// (`repr(C)`: pos, uv, color, extra, mode). Six of these make a quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub extra: [f32; 4],
    pub mode: u32,
}

/// A contiguous vertex range drawn with one descriptor binding: `image` names
/// the texture for [`MODE_IMAGE`] quads, `None` for everything else (the
/// atlases are always bound).
#[derive(Debug, PartialEq, Eq)]
pub struct Batch {
    pub start: u32,
    pub count: u32,
    pub image: Option<ImageId>,
}

/// One uploaded image texture, interned by [`GlyphCache`] per destination path and
/// native size. A frame's quads carry the id rather than the path, so batching an
/// image costs no string and binding one hashes four bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImageId(u32);

/// An image bitmap the renderer does not hold yet: its id, its native size, and
/// straight ARGB words (uploadable as B8G8R8A8 bytes verbatim).
pub struct ImageUpload {
    pub id: ImageId,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
}

/// A rectangle of an atlas that changed and needs a texture upload, with its
/// bytes extracted row-contiguous from the mirror.
pub struct AtlasUpload {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub bytes: Vec<u8>,
}

/// Everything the Vulkan renderer needs to draw one frame. Owned across frames by
/// the presentation and refilled by [`build_frame`], so a steady-state frame reuses
/// the vertex and batch capacity it grew instead of regrowing it from zero.
#[derive(Default)]
pub struct FrameData {
    pub vertices: Vec<Vertex>,
    pub batches: Vec<Batch>,
    /// Atlas geometry and generation, so the renderer can (re)create textures.
    pub glyph_atlas: AtlasInfo,
    pub emoji_atlas: AtlasInfo,
    /// Dirty-region uploads for the two atlases, if any.
    pub glyph_upload: Option<AtlasUpload>,
    pub emoji_upload: Option<AtlasUpload>,
    /// Native-size image bitmaps the renderer does not have yet.
    pub image_uploads: Vec<ImageUpload>,
}

impl FrameData {
    /// Empty the per-frame buffers, keeping their capacity for the refill.
    fn clear(&mut self) {
        self.vertices.clear();
        self.batches.clear();
        self.image_uploads.clear();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtlasInfo {
    pub width: u32,
    pub height: u32,
    /// Bumped whenever the atlas is wiped and regrown; the renderer recreates
    /// the texture (and re-uploads in full) when it changes.
    pub generation: u32,
}

/// Where a raster landed in an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

/// A glyph's atlas placement plus its pen metrics, cached beside the slot so a
/// steady-state frame emits the quad without re-rasterizing the glyph. `slot` is
/// `None` for an inkless glyph (a space, or an emoji that shaped to no ink) or
/// one the atlas could not fit; the pen still advances by `advance`, so the
/// caret and the glyphs after it land where layout measured.
#[derive(Clone, Copy)]
struct PackedGlyph {
    slot: Option<Slot>,
    left: i32,
    top: i32,
    advance: f32,
}

/// A shelf-packing texture atlas with a CPU mirror. Rasters are packed onto
/// horizontal shelves (rows share a height class); the mirror holds the texel
/// bytes so the GPU texture can be (re)filled at any time, and a dirty
/// rectangle accumulates what changed since the last upload.
pub struct Atlas {
    width: u32,
    height: u32,
    /// Bytes per texel: 1 (R8 coverage) or 4 (BGRA emoji).
    bpp: u32,
    pixels: Vec<u8>,
    /// Shelves as (y, height, x_used).
    shelves: Vec<(u32, u32, u32)>,
    next_y: u32,
    generation: u32,
    /// Union of slots written since the last [`Atlas::take_upload`].
    dirty: Option<(u32, u32, u32, u32)>,
}

impl Atlas {
    pub fn new(bpp: u32) -> Self {
        Self {
            width: ATLAS_START,
            height: ATLAS_START,
            bpp,
            pixels: vec![0; (ATLAS_START * ATLAS_START * bpp) as usize],
            shelves: Vec::new(),
            next_y: 0,
            generation: 0,
            dirty: None,
        }
    }

    pub fn info(&self) -> AtlasInfo {
        AtlasInfo {
            width: self.width,
            height: self.height,
            generation: self.generation,
        }
    }

    /// Reserve a `w` x `h` slot. `None` means the atlas is full even after
    /// growing to its cap (the caller skips the raster; a pathological case).
    /// Growing wipes all previous slots and bumps the generation, which the
    /// frame builder answers by restarting the build.
    fn reserve(&mut self, w: u32, h: u32) -> Option<Slot> {
        if w == 0 || h == 0 || w > ATLAS_MAX || h > ATLAS_MAX {
            return None;
        }
        loop {
            // A shelf fits if the raster is no taller than the shelf and the
            // remaining width holds it. Shelves are created at the raster's
            // height, so height classes cluster and waste stays bounded.
            if let Some(shelf) = self
                .shelves
                .iter_mut()
                .find(|(_, sh, used)| h <= *sh && used + w <= self.width)
            {
                let slot = Slot {
                    x: shelf.2,
                    y: shelf.0,
                    w,
                    h,
                };
                shelf.2 += w;
                return Some(slot);
            }
            if self.next_y + h <= self.height && w <= self.width {
                self.shelves.push((self.next_y, h, w));
                let slot = Slot {
                    x: 0,
                    y: self.next_y,
                    w,
                    h,
                };
                self.next_y += h;
                return Some(slot);
            }
            if !self.grow() {
                return None;
            }
        }
    }

    /// Double the atlas (square), wiping every slot. Returns false at the cap.
    fn grow(&mut self) -> bool {
        let next = (self.width * 2).min(ATLAS_MAX);
        if next == self.width {
            return false;
        }
        self.width = next;
        self.height = next;
        self.pixels = vec![0; (next * next * self.bpp) as usize];
        self.shelves.clear();
        self.next_y = 0;
        self.generation += 1;
        self.dirty = None;
        true
    }

    /// Write a raster's rows into the mirror at `slot` and extend the dirty
    /// rectangle over it. `src` is `w * h * bpp` bytes, row-major, tight.
    fn write(&mut self, slot: Slot, src: &[u8]) {
        let row_bytes = (slot.w * self.bpp) as usize;
        for row in 0..slot.h {
            let src_start = row as usize * row_bytes;
            let dst_start = (((slot.y + row) * self.width + slot.x) * self.bpp) as usize;
            self.pixels[dst_start..dst_start + row_bytes]
                .copy_from_slice(&src[src_start..src_start + row_bytes]);
        }
        let rect = (slot.x, slot.y, slot.x + slot.w, slot.y + slot.h);
        self.dirty = Some(match self.dirty {
            None => rect,
            Some((x0, y0, x1, y1)) => (
                x0.min(rect.0),
                y0.min(rect.1),
                x1.max(rect.2),
                y1.max(rect.3),
            ),
        });
    }

    /// Reserve a slot for a `w` x `h` raster and write `src` into it, returning
    /// the slot; `None` for a zero-area raster or a full atlas (the caller
    /// records the empty placement so the pen still advances). `src` is
    /// `w * h * bpp` bytes, row-major, tight.
    fn pack(&mut self, w: u32, h: u32, src: &[u8]) -> Option<Slot> {
        if w == 0 || h == 0 {
            return None;
        }
        let slot = self.reserve(w, h)?;
        self.write(slot, src);
        Some(slot)
    }

    /// The bytes of the dirty region (row-contiguous), clearing the flag.
    /// A texture recreated after a grow needs no separate full upload: the
    /// grow wiped the mirror, so everything in it is inside the dirty region.
    pub fn take_upload(&mut self) -> Option<AtlasUpload> {
        let (x0, y0, x1, y1) = self.dirty.take()?;
        let (w, h) = (x1 - x0, y1 - y0);
        let row_bytes = (w * self.bpp) as usize;
        let mut bytes = Vec::with_capacity(row_bytes * h as usize);
        for row in y0..y1 {
            let start = ((row * self.width + x0) * self.bpp) as usize;
            bytes.extend_from_slice(&self.pixels[start..start + row_bytes]);
        }
        Some(AtlasUpload {
            x: x0,
            y: y0,
            w,
            h,
            bytes,
        })
    }
}

/// The persistent glyph caches the batcher packs into: slots by scalar and by
/// cluster, over the two atlases. Owned by the app next to the display list;
/// survives across frames so steady-state frames insert nothing.
pub struct GlyphCache {
    pub glyphs: Atlas,
    pub emoji: Atlas,
    scalar_slots: HashMap<(FaceKey, char), PackedGlyph>,
    /// Nested so lookups borrow the cluster as `&str` (no per-frame `String`).
    /// A `None` value records a cluster that is not color emoji, so a
    /// steady-state frame takes the per-character path without re-consulting
    /// the `Face`.
    cluster_slots: HashMap<FaceKey, HashMap<String, Option<PackedGlyph>>>,
    /// Image texture ids, by native size then by path: the same nesting as
    /// `cluster_slots` and for the same reason, so a frame's image lookup borrows
    /// the path instead of building a key.
    image_ids: HashMap<(u32, u32), HashMap<String, ImageId>>,
    next_image_id: u32,
    stats: CacheStats,
}

/// Slot lookups for glyphs and emoji clusters since the cache was made. Once the
/// visible text is packed, a steady frame adds hits and no misses.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    pub fn lookups(self) -> u64 {
        self.hits + self.misses
    }

    /// 0.0 when nothing was looked up.
    pub fn hit_rate(self) -> f32 {
        match self.lookups() {
            0 => 0.0,
            total => self.hits as f32 / total as f32,
        }
    }
}

impl GlyphCache {
    pub fn new() -> Self {
        Self {
            glyphs: Atlas::new(1),
            emoji: Atlas::new(4),
            scalar_slots: HashMap::new(),
            cluster_slots: HashMap::new(),
            image_ids: HashMap::new(),
            next_image_id: 0,
            stats: CacheStats::default(),
        }
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// The id of `path`'s texture at its native size, minting one the first time
    /// the pair is seen. Ids are permanent: the renderer's own texture map is what
    /// says whether an id is uploaded right now, so an evicted image comes back
    /// under the id it had.
    fn image_id(&mut self, path: &str, width: u32, height: u32) -> ImageId {
        let by_path = self.image_ids.entry((width, height)).or_default();
        if let Some(&id) = by_path.get(path) {
            return id;
        }
        let id = ImageId(self.next_image_id);
        self.next_image_id += 1;
        by_path.insert(path.to_string(), id);
        id
    }

    /// Drop every slot record (after an atlas grew and wiped its content).
    fn reset_for(&mut self, glyphs_gen: u32, emoji_gen: u32) {
        if self.glyphs.generation != glyphs_gen {
            self.scalar_slots.clear();
        }
        if self.emoji.generation != emoji_gen {
            self.cluster_slots.clear();
        }
    }
}

/// Build one frame's GPU data from the display list into `out`, whose buffers are
/// cleared and refilled (the caller keeps it across frames, so a steady-state frame
/// allocates nothing). `known_images` is the set of image textures the renderer
/// already holds; missing ones get their bitmaps attached to the result. Restarts
/// internally if an atlas grows mid-build, so the vertices always reference current
/// atlas coordinates.
pub fn build_frame(
    fonts: &Fonts,
    stores: &[&ImageStore],
    list: &[DrawCmd],
    cache: &mut GlyphCache,
    known_images: &dyn Fn(ImageId) -> bool,
    code_glyph_gamma: f32,
    out: &mut FrameData,
) {
    loop {
        let glyphs_gen = cache.glyphs.generation;
        let emoji_gen = cache.emoji.generation;
        out.clear();
        {
            let mut b = Batcher {
                fonts,
                stores,
                cache,
                vertices: &mut out.vertices,
                batches: &mut out.batches,
                image_uploads: &mut out.image_uploads,
                known_images,
                code_glyph_gamma,
            };
            for cmd in list {
                b.command(cmd);
            }
        }
        // An atlas grew (and wiped) during the build: earlier quads reference
        // dead coordinates. Clear the stale slot records and rebuild; the
        // second pass fits by construction (or skips at the cap).
        if cache.glyphs.generation != glyphs_gen || cache.emoji.generation != emoji_gen {
            cache.reset_for(glyphs_gen, emoji_gen);
            continue;
        }
        out.glyph_atlas = cache.glyphs.info();
        out.emoji_atlas = cache.emoji.info();
        out.glyph_upload = cache.glyphs.take_upload();
        out.emoji_upload = cache.emoji.take_upload();
        return;
    }
}

struct Batcher<'a> {
    fonts: &'a Fonts,
    stores: &'a [&'a ImageStore],
    cache: &'a mut GlyphCache,
    vertices: &'a mut Vec<Vertex>,
    batches: &'a mut Vec<Batch>,
    image_uploads: &'a mut Vec<ImageUpload>,
    known_images: &'a dyn Fn(ImageId) -> bool,
    /// Per-glyph mask gamma for code runs, baked into the vertex `extra.x` so the
    /// shader thins them less than the prose gamma it carries as a push constant.
    /// Zero in the vertex means "use the push default"; a code run writes this.
    code_glyph_gamma: f32,
}

impl Batcher<'_> {
    fn command(&mut self, cmd: &DrawCmd) {
        match cmd {
            DrawCmd::Fill { rect, color } => {
                self.quad(*rect, MODE_SOLID, *color, [0.0; 2], [0.0; 4]);
            }
            DrawCmd::RoundRect {
                rect,
                color,
                radius,
                corners,
            } => self.round_rect(*rect, *color, *radius, *corners),
            DrawCmd::CheckMark { rect, color } => {
                // The tick is intrinsic to the mode: uv carries the pixel
                // position within the box (like the rounded-rect mode) and extra
                // carries the box size the shader's distance field needs.
                let extra = [rect.w as f32, rect.h as f32, 0.0, 0.0];
                self.quad(*rect, MODE_CHECK, *color, [0.0; 2], extra);
            }
            DrawCmd::Chevron { rect, color, down } => {
                // Same intrinsic-shape path as the check mark: the box size in
                // extra.xy feeds the distance field, and extra.z carries the
                // facing (0 right for collapsed, 1 down for expanded).
                let extra = [rect.w as f32, rect.h as f32, f32::from(*down), 0.0];
                self.quad(*rect, MODE_CHEVRON, *color, [0.0; 2], extra);
            }
            DrawCmd::Pin { rect, color } => {
                // Same intrinsic-shape path as the check mark: the box size in
                // extra.xy feeds the distance field the pin marker is inked from.
                let extra = [rect.w as f32, rect.h as f32, 0.0, 0.0];
                self.quad(*rect, MODE_PIN, *color, [0.0; 2], extra);
            }
            DrawCmd::Hatch {
                rect,
                color,
                period,
                stroke,
            } => {
                // The stripe pattern is intrinsic to the mode: extra carries the diagonal
                // spacing (period) and the stroke half-width the shader fades coverage
                // over. The uv is seeded at the quad's absolute screen position (not the
                // usual box-local zero), so the diagonals are computed in one global
                // screen-space grid and line up seamlessly across separate hatch bands
                // rather than each restarting its own phase.
                let extra = [*period as f32, *stroke as f32 * 0.5, 0.0, 0.0];
                self.quad(
                    *rect,
                    MODE_HATCH,
                    *color,
                    [rect.x as f32, rect.y as f32],
                    extra,
                );
            }
            DrawCmd::Text {
                x,
                baseline,
                face,
                color,
                text,
                clip_x,
                ..
            } => self.text(*face, *x, *baseline, text, *color, *clip_x),
            DrawCmd::Image { rect, path } => self.image(*rect, path),
        }
    }

    /// Emit one axis-aligned quad (two triangles) into `image`'s texture, with
    /// uv running from `[u0, v0]` at the top-left to `[u1, v1]` at the
    /// bottom-right (`uv` is `[u0, v0, u1, v1]`). The atlases and fills pass
    /// texel coordinates (uv advancing 1:1 with pixels); an image passes
    /// normalized `0..1` corners so the sampler scales its native texture across
    /// the quad.
    fn quad_uv(
        &mut self,
        rect: Rect,
        mode: u32,
        color: u32,
        uv: [f32; 4],
        extra: [f32; 4],
        image: Option<ImageId>,
    ) {
        if rect.w <= 0 || rect.h <= 0 {
            return;
        }
        let (x0, y0) = (rect.x as f32, rect.y as f32);
        let (x1, y1) = ((rect.x + rect.w) as f32, (rect.y + rect.h) as f32);
        let c = color_f32(color);
        let v = |x: f32, y: f32, u: f32, vv: f32| Vertex {
            pos: [x, y],
            uv: [u, vv],
            color: c,
            extra,
            mode,
        };
        let [u0, v0, u1, v1] = uv;
        self.vertices.extend_from_slice(&[
            v(x0, y0, u0, v0),
            v(x1, y0, u1, v0),
            v(x0, y1, u0, v1),
            v(x1, y0, u1, v0),
            v(x1, y1, u1, v1),
            v(x0, y1, u0, v1),
        ]);
        self.extend_batch(image, 6);
    }

    /// Emit one axis-aligned quad with texel UVs: `uv0` is the coordinate at the
    /// quad's top-left and uv advances 1:1 with pixels. The atlas-bound path (no
    /// per-image texture), for fills, glyphs, emoji, and rounded corners.
    fn quad(&mut self, rect: Rect, mode: u32, color: u32, uv0: [f32; 2], extra: [f32; 4]) {
        let uv = [
            uv0[0],
            uv0[1],
            uv0[0] + rect.w as f32,
            uv0[1] + rect.h as f32,
        ];
        self.quad_uv(rect, mode, color, uv, extra, None);
    }

    /// Append `count` vertices to the current batch, or open a new one when
    /// the bound image texture would change.
    fn extend_batch(&mut self, image: Option<ImageId>, count: u32) {
        let start = self.vertices.len() as u32 - count;
        if let Some(last) = self.batches.last_mut()
            && last.image == image
        {
            last.count += count;
            return;
        }
        self.batches.push(Batch {
            start,
            count,
            image,
        });
    }

    fn round_rect(&mut self, rect: Rect, color: u32, radius: i32, corners: RoundedCorners) {
        // Clamp the radius to half the shorter side so opposite corners never
        // overlap; the shader's coverage function assumes this bound holds.
        let r = radius.clamp(0, (rect.w / 2).min(rect.h / 2).max(0));
        let mask = match corners {
            RoundedCorners::None => 0u32,
            RoundedCorners::Top => 1,
            RoundedCorners::Bottom => 2,
            RoundedCorners::Both => 3,
        };
        if r == 0 || mask == 0 {
            self.quad(rect, MODE_SOLID, color, [0.0; 2], [0.0; 4]);
            return;
        }
        // uv carries the position within the rect (pixels); extra carries the
        // geometry the shader's coverage function needs.
        let extra = [rect.w as f32, rect.h as f32, r as f32, mask as f32];
        self.quad(rect, MODE_ROUND, color, [0.0; 2], extra);
    }

    /// Emit a run's glyph quads: the ASCII fast path, then grapheme clusters
    /// routed to the emoji glyph or per-character drawing. The routing mirrors
    /// `shape::text_advance`, so the quads land where layout measured.
    fn text(
        &mut self,
        face_key: FaceKey,
        x: i32,
        baseline: i32,
        text: &str,
        color: u32,
        clip_x: Option<[i32; 2]>,
    ) {
        let face = self.fonts.face_for(face_key);
        // A block of code thins less than prose (its small stems read as faint under
        // the prose gamma), so it overrides the push-constant gamma per glyph. Zero
        // means "use the push default": prose, and inline code, which sits at the
        // prose size inside a line of prose and thins with it.
        let gamma = match face_key {
            FaceKey::Code {
                ink: CodeInk::Block,
                ..
            } => self.code_glyph_gamma,
            FaceKey::Code { .. } | FaceKey::Prose { .. } => 0.0,
        };
        // The ink class picked the gamma; the raster is the same either way, so the
        // glyph cache sees one key per face and size.
        let face_key = face_key.raster();
        let mut pen = x as f32;
        if text.is_ascii() {
            for ch in text.chars() {
                pen = self.scalar(face, face_key, ch, pen, baseline, color, gamma, clip_x);
            }
            return;
        }
        for (_, cluster) in grapheme::graphemes(text) {
            match self.packed_cluster(face, face_key, cluster) {
                Some(packed) => {
                    // Emoji carry their own color and ignore the mask gamma.
                    self.emit_glyph(&packed, pen, baseline, MODE_EMOJI, 0, 0.0, clip_x);
                    pen += packed.advance;
                }
                None => {
                    for ch in cluster.chars() {
                        pen = self.scalar(face, face_key, ch, pen, baseline, color, gamma, clip_x);
                    }
                }
            }
        }
    }

    /// One character's glyph quad (or nothing for empty rasters like spaces),
    /// returning the advanced pen. Its placement is cached beside the atlas
    /// slot, so a steady-state frame emits the quad without re-rasterizing the
    /// glyph.
    #[allow(clippy::too_many_arguments)]
    fn scalar(
        &mut self,
        face: &Face,
        face_key: FaceKey,
        ch: char,
        pen: f32,
        baseline: i32,
        color: u32,
        gamma: f32,
        clip_x: Option<[i32; 2]>,
    ) -> f32 {
        let packed = self.packed_scalar(face, face_key, ch);
        self.emit_glyph(&packed, pen, baseline, MODE_GLYPH, color, gamma, clip_x);
        pen + packed.advance
    }

    /// The cached placement for a scalar glyph. On the first sight of this
    /// (face, character) the glyph is rasterized once and its coverage written
    /// straight into the atlas; later frames read the cached placement.
    fn packed_scalar(&mut self, face: &Face, face_key: FaceKey, ch: char) -> PackedGlyph {
        if let Some(&packed) = self.cache.scalar_slots.get(&(face_key, ch)) {
            self.cache.stats.hits += 1;
            return packed;
        }
        self.cache.stats.misses += 1;
        // Rasterize from the fallback face when the keyed face lacks the glyph, so a
        // scalar the prose font has no glyph for (an arrow, a maths sign) draws
        // instead of a tofu box. The advance comes from the same face, since
        // `Face::advance` resolves the scalar the way `glyph_face` does, so the pen
        // moves by the width of the glyph that was actually inked.
        let g = self.fonts.glyph_face(face_key, ch).rasterize(ch);
        let atlas = &mut self.cache.glyphs;
        let packed = PackedGlyph {
            slot: atlas.pack(g.width as u32, g.rows as u32, &g.coverage),
            left: g.left,
            top: g.top,
            advance: face.advance(ch),
        };
        self.cache.scalar_slots.insert((face_key, ch), packed);
        packed
    }

    /// The cached placement for an emoji cluster, or `None` when the cluster is
    /// not color emoji and the caller should draw its characters instead. The
    /// raster is packed into the color atlas on first sight and the decision
    /// cached, so a steady-state frame neither re-shapes nor touches the `Face`.
    /// Only clusters that draw as emoji count in the stats; the rest count on the
    /// per-character path.
    ///
    /// A cluster that would not be emoji even where the text face lacks its glyph
    /// (a letter, an accented letter, a box-drawing character) returns None before
    /// the lookup, so plain text never enters the map.
    fn packed_cluster(
        &mut self,
        face: &Face,
        face_key: FaceKey,
        cluster: &str,
    ) -> Option<PackedGlyph> {
        if !wants_emoji(cluster, |_| false) {
            return None;
        }
        if let Some(&packed) = self
            .cache
            .cluster_slots
            .get(&face_key)
            .and_then(|by| by.get(cluster))
        {
            if packed.is_some() {
                self.cache.stats.hits += 1;
            }
            return packed;
        }
        let atlas = &mut self.cache.emoji;
        let packed = face.rasterize_cluster(cluster).map(|g| PackedGlyph {
            slot: pack_argb(atlas, g.width as u32, g.rows as u32, &g.argb),
            left: g.left,
            top: g.top,
            advance: g.advance,
        });
        if packed.is_some() {
            self.cache.stats.misses += 1;
        }
        self.cache
            .cluster_slots
            .entry(face_key)
            .or_default()
            .insert(cluster.to_string(), packed);
        packed
    }

    /// Emit a cached glyph's textured quad. An inkless or unplaced glyph (slot
    /// `None`) draws nothing; the caller advances the pen regardless.
    #[allow(clippy::too_many_arguments)]
    fn emit_glyph(
        &mut self,
        packed: &PackedGlyph,
        pen: f32,
        baseline: i32,
        mode: u32,
        color: u32,
        gamma: f32,
        clip_x: Option<[i32; 2]>,
    ) {
        let Some(slot) = packed.slot else {
            return;
        };
        let rect = Rect {
            x: pen.round() as i32 + packed.left,
            y: baseline - packed.top,
            w: slot.w as i32,
            h: slot.h as i32,
        };
        // extra.x carries the per-glyph mask gamma override (0 = use the push
        // default); the shader reads it only for MODE_GLYPH.
        let extra = [gamma, 0.0, 0.0, 0.0];
        // A horizontal scissor keeps the glyph's atlas UVs aligned to the pixels it
        // still covers, so a run slides under a fixed edge without shearing. uv
        // advances 1:1 with pixels from the slot's top-left, so a clipped left edge
        // shifts u0 by the same amount the quad's x moved.
        if let Some([left, right]) = clip_x {
            let x0 = rect.x.max(left);
            let x1 = (rect.x + rect.w).min(right);
            if x1 <= x0 {
                return;
            }
            let u0 = slot.x as f32 + (x0 - rect.x) as f32;
            let u1 = slot.x as f32 + (x1 - rect.x) as f32;
            let clipped = Rect {
                x: x0,
                y: rect.y,
                w: x1 - x0,
                h: rect.h,
            };
            let v0 = slot.y as f32;
            self.quad_uv(
                clipped,
                mode,
                color,
                [u0, v0, u1, v0 + rect.h as f32],
                extra,
                None,
            );
            return;
        }
        self.quad(rect, mode, color, [slot.x as f32, slot.y as f32], extra);
    }

    /// An image quad over its own texture, attaching the bitmap when the
    /// renderer does not have it yet. A not-yet-decoded image draws nothing,
    /// like the CPU path.
    fn image(&mut self, rect: Rect, path: &str) {
        if rect.w <= 0 || rect.h <= 0 {
            return;
        }
        // Upload the image at its native size, keyed by that size (constant per
        // image), and let the GPU sampler scale it to the quad. A resize reuses
        // the same texture, so no CPU resample or re-upload happens per drag.
        // Resolve against whichever pane's store holds it.
        let Some(img) = self.stores.iter().find_map(|s| s.get(path)) else {
            return;
        };
        let id = self.cache.image_id(path, img.width, img.height);
        if !(self.known_images)(id) && !self.image_uploads.iter().any(|u| u.id == id) {
            self.image_uploads.push(ImageUpload {
                id,
                width: img.width,
                height: img.height,
                pixels: img.pixels.clone(),
            });
        }
        // Normalized UVs so texture() samples the whole native texture across the
        // display rect (quad.frag mode 3, which ignores the vertex color).
        self.quad_uv(
            rect,
            MODE_IMAGE,
            0,
            [0.0, 0.0, 1.0, 1.0],
            [0.0; 4],
            Some(id),
        );
    }
}

/// Pack an emoji cluster's straight-`0xAARRGGBB` raster into the color atlas.
/// `None` for a zero-area or malformed glyph or a full atlas; the placement is
/// recorded as `None` so the pen advances without drawing.
fn pack_argb(atlas: &mut Atlas, w: u32, h: u32, argb: &[u32]) -> Option<Slot> {
    if w == 0 || h == 0 || argb.len() < (w * h) as usize {
        return None;
    }
    // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim.
    let bytes: Vec<u8> = argb.iter().flat_map(|px| px.to_ne_bytes()).collect();
    atlas.pack(w, h, &bytes)
}

/// A 0x00RRGGBB display-list color as straight-alpha RGBA floats.
fn color_f32(color: u32) -> [f32; 4] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use crate::render::gpu::*;

    /// Build one frame into a fresh buffer. The app keeps one across frames; a test
    /// wants each build on its own, so this hands back what it produced.
    fn built(
        fonts: &Fonts,
        stores: &[&ImageStore],
        list: &[DrawCmd],
        cache: &mut GlyphCache,
        known_images: &dyn Fn(ImageId) -> bool,
        code_glyph_gamma: f32,
    ) -> FrameData {
        let mut out = FrameData::default();
        build_frame(
            fonts,
            stores,
            list,
            cache,
            known_images,
            code_glyph_gamma,
            &mut out,
        );
        out
    }

    #[test]
    fn atlas_packs_onto_shelves_and_tracks_dirty() {
        let mut a = Atlas::new(1);
        let s1 = a.reserve(10, 8).expect("fits");
        let s2 = a.reserve(6, 8).expect("fits on the same shelf");
        assert_eq!((s1.x, s1.y), (0, 0));
        assert_eq!((s2.x, s2.y), (10, 0), "same shelf, packed rightward");
        let s3 = a.reserve(10, 20).expect("taller: a new shelf");
        assert_eq!(s3.y, 8, "new shelf below the first");
        a.write(s1, &[7u8; 80]);
        let up = a.take_upload().expect("dirty after write");
        assert_eq!((up.x, up.y, up.w, up.h), (0, 0, 10, 8));
        assert!(up.bytes.iter().all(|&b| b == 7));
        assert!(a.take_upload().is_none(), "dirty cleared");
    }

    #[test]
    fn atlas_grows_by_wiping_and_bumping_generation() {
        let mut a = Atlas::new(1);
        let g0 = a.generation;
        // A raster wider than the atlas forces growth.
        let s = a.reserve(ATLAS_START + 1, 4).expect("grows to fit");
        assert!(a.generation > g0);
        assert_eq!(a.width, ATLAS_START * 2);
        assert_eq!((s.x, s.y), (0, 0), "the grown atlas starts empty");
        // Beyond the cap is a clean None.
        assert!(a.reserve(ATLAS_MAX + 1, 1).is_none());
    }

    #[test]
    fn fill_becomes_one_solid_quad_batch() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Fill {
            rect: Rect {
                x: 1,
                y: 2,
                w: 10,
                h: 20,
            },
            color: 0x00ff_0000,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(f.vertices.len(), 6);
        assert_eq!(f.batches.len(), 1);
        assert_eq!(
            f.batches[0],
            Batch {
                start: 0,
                count: 6,
                image: None
            }
        );
        assert!(f.vertices.iter().all(|v| v.mode == MODE_SOLID));
        assert_eq!(f.vertices[0].pos, [1.0, 2.0]);
        assert_eq!(f.vertices[4].pos, [11.0, 22.0]);
        assert_eq!(f.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn block_code_glyphs_carry_the_gamma_override_prose_and_inline_code_do_not() {
        // One size opens both the prose and code faces at 16.
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 20,
        };
        let text = |face| DrawCmd::Text {
            bounds,
            x: 0,
            baseline: 16,
            face,
            color: 0x00ff_ffff,
            text: "xy".to_string(),
            clip_x: None,
        };
        const CODE_GAMMA: f32 = 1.3;
        let prose = text(FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        });
        let inline = text(FaceKey::Code {
            size: 16,
            ink: CodeInk::Inline,
        });
        let block = text(FaceKey::Code {
            size: 16,
            ink: CodeInk::Block,
        });
        let overrides = |list: &[DrawCmd], cache: &mut GlyphCache| {
            let f = built(&fonts, &[&images], list, cache, &|_| false, CODE_GAMMA);
            let g: Vec<f32> = f
                .vertices
                .iter()
                .filter(|v| v.mode == MODE_GLYPH)
                .map(|v| v.extra[0])
                .collect();
            assert!(!g.is_empty(), "some glyph quads were emitted");
            g
        };
        // A block of code carries the per-run override; prose and inline code pass
        // zero and defer to the shader's push default.
        assert!(
            overrides(&[block], &mut cache)
                .iter()
                .all(|&g| g == CODE_GAMMA),
            "block code glyphs carry the per-run gamma override"
        );
        assert!(
            overrides(&[prose, inline], &mut cache)
                .iter()
                .all(|&g| g == 0.0),
            "prose and inline code defer to the push default (extra.x == 0)"
        );
    }

    #[test]
    fn the_two_code_inks_share_one_atlas_slot() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let text = |ink| DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Code { size: 16, ink },
            color: 0x00ff_ffff,
            text: "xy".to_string(),
            clip_x: None,
        };
        // The ink class only picks a gamma, so the same glyphs at the same size are
        // rasterized once however they are inked.
        built(
            &fonts,
            &[&images],
            &[text(CodeInk::Inline)],
            &mut cache,
            &|_| false,
            1.3,
        );
        let packed = cache.scalar_slots.len();
        built(
            &fonts,
            &[&images],
            &[text(CodeInk::Block)],
            &mut cache,
            &|_| false,
            1.3,
        );
        assert_eq!(
            cache.scalar_slots.len(),
            packed,
            "the other ink reuses the rasters"
        );
    }

    #[test]
    fn text_reuses_atlas_slots_across_frames() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            text: "abcabc".to_string(),
            clip_x: None,
        }];
        let f1 = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert!(f1.glyph_upload.is_some(), "first frame uploads new glyphs");
        assert!(
            !f1.vertices.is_empty(),
            "glyph quads were emitted ({} vertices)",
            f1.vertices.len()
        );
        // Three distinct letters miss once each; their repeats hit.
        assert_eq!(cache.stats(), CacheStats { hits: 3, misses: 3 });
        let f2 = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert!(f2.glyph_upload.is_none(), "steady state uploads nothing");
        assert_eq!(f1.vertices, f2.vertices, "same list, same geometry");
        assert_eq!(cache.stats(), CacheStats { hits: 9, misses: 3 });
    }

    #[test]
    fn glyph_quads_share_one_batch_with_fills() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![
            DrawCmd::Fill {
                rect: Rect {
                    x: 0,
                    y: 0,
                    w: 5,
                    h: 5,
                },
                color: 0,
            },
            DrawCmd::Text {
                bounds: Rect {
                    x: 0,
                    y: 0,
                    w: 100,
                    h: 20,
                },
                x: 0,
                baseline: 16,
                face: FaceKey::Prose {
                    size: 16,
                    style: crate::platform::freetype::FontStyle::Regular,
                },
                color: 0x00ff_ffff,
                text: "hi".to_string(),
                clip_x: None,
            },
        ];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(
            f.batches.len(),
            1,
            "no image, so one batch: {:?}",
            f.batches
        );
        assert_eq!(f.batches[0].count, f.vertices.len() as u32);
    }

    #[test]
    fn round_rect_emits_sdf_quad_with_geometry_in_extra() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let rect = Rect {
            x: 4,
            y: 6,
            w: 20,
            h: 10,
        };
        let list = vec![DrawCmd::RoundRect {
            rect,
            color: 0x0011_2233,
            radius: 4,
            corners: RoundedCorners::Top,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(f.vertices.len(), 6);
        let v = &f.vertices[0];
        assert_eq!(v.mode, MODE_ROUND);
        assert_eq!(v.extra, [20.0, 10.0, 4.0, 1.0], "w, h, r, corner mask");
        assert_eq!(v.uv, [0.0, 0.0], "uv is rect-local pixels");
        assert_eq!(f.vertices[4].uv, [20.0, 10.0]);
    }

    #[test]
    fn chevron_emits_sdf_quad_with_facing_in_extra() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let rect = Rect {
            x: 4,
            y: 6,
            w: 12,
            h: 12,
        };
        let list = vec![DrawCmd::Chevron {
            rect,
            color: 0x0011_2233,
            down: true,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(f.vertices.len(), 6);
        let v = &f.vertices[0];
        assert_eq!(v.mode, MODE_CHEVRON);
        assert_eq!(v.extra, [12.0, 12.0, 1.0, 0.0], "w, h, down=1, unused");
    }

    #[test]
    fn zero_radius_round_rect_degrades_to_solid() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::RoundRect {
            rect: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            color: 0,
            radius: 0,
            corners: RoundedCorners::Both,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert!(f.vertices.iter().all(|v| v.mode == MODE_SOLID));
    }

    #[test]
    fn image_splits_its_own_batch_and_uploads_once() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/images");
        let mut images = ImageStore::new(Some(dir));
        images.ensure("tiny.png");
        let mut cache = GlyphCache::new();
        let rect = Rect {
            x: 0,
            y: 10,
            w: 8,
            h: 8,
        };
        let fill = DrawCmd::Fill {
            rect: Rect {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            color: 0,
        };
        let list = vec![
            fill.clone(),
            DrawCmd::Image {
                rect,
                path: "tiny.png".to_string(),
            },
            fill,
        ];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(f.batches.len(), 3, "solid / image / solid: {:?}", f.batches);
        // The texture is uploaded at the image's native 2x2 size (the GPU scales
        // it to the 8x8 quad), under the id the batch carries.
        let id = f.batches[1]
            .image
            .expect("the image batch names its texture");
        assert_eq!(f.image_uploads.len(), 1);
        assert_eq!(f.image_uploads[0].id, id);
        assert_eq!(
            (f.image_uploads[0].width, f.image_uploads[0].height),
            (2, 2)
        );
        assert_eq!(f.image_uploads[0].pixels.len(), 4, "2x2 native bitmap");
        // A renderer that already has the texture gets no upload, and the same path
        // batches under the same id: no per-frame key is built for it.
        let f2 = built(&fonts, &[&images], &list, &mut cache, &|k| k == id, 1.0);
        assert!(f2.image_uploads.is_empty());
        assert_eq!(f2.batches.len(), 3, "batches still split around the image");
        assert_eq!(f2.batches[1].image, Some(id), "the id is stable per path");
    }

    #[test]
    fn two_sizes_of_one_path_get_their_own_ids() {
        // The id interns (path, native size), so a store that re-decodes a path at a
        // different size cannot draw the old texture under the new geometry.
        let fonts = Fonts::new(&[16]).expect("default font");
        let rect = Rect {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
        };
        let list = vec![DrawCmd::Image {
            rect,
            path: "tiny.png".to_string(),
        }];
        let mut cache = GlyphCache::new();
        let small = ImageStore::from_decoded(vec![(
            "tiny.png".to_string(),
            crate::render::image::Image {
                pixels: vec![0xffff_ffff; 4],
                width: 2,
                height: 2,
            },
        )]);
        let large = ImageStore::from_decoded(vec![(
            "tiny.png".to_string(),
            crate::render::image::Image {
                pixels: vec![0xffff_ffff; 16],
                width: 4,
                height: 4,
            },
        )]);
        let a = built(&fonts, &[&small], &list, &mut cache, &|_| false, 1.0);
        let b = built(&fonts, &[&large], &list, &mut cache, &|_| false, 1.0);
        assert_ne!(a.batches[0].image, b.batches[0].image);
        assert_eq!(
            (b.image_uploads[0].width, b.image_uploads[0].height),
            (4, 4)
        );
    }

    #[test]
    fn a_clipped_text_run_keeps_every_glyph_quad_inside_the_scissor() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let prose = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let bounds = Rect {
            x: 0,
            y: 0,
            w: 300,
            h: 20,
        };
        let run = |clip_x| {
            vec![DrawCmd::Text {
                bounds,
                x: 0,
                baseline: 16,
                face: prose,
                color: 0x00ff_ffff,
                text: "MMMMMMMMMM".to_string(),
                clip_x,
            }]
        };
        let mut cache = GlyphCache::new();
        let full = built(&fonts, &[&images], &run(None), &mut cache, &|_| false, 1.0);
        let clipped = built(
            &fonts,
            &[&images],
            &run(Some([20, 60])),
            &mut cache,
            &|_| false,
            1.0,
        );
        // The whole run spills past the scissor window; the clipped run never does,
        // and it still emits the glyphs that straddle or sit inside the window.
        assert!(
            full.vertices
                .iter()
                .any(|v| v.pos[0] < 20.0 || v.pos[0] > 60.0),
            "the unclipped run reaches outside [20, 60]"
        );
        assert!(
            !clipped.vertices.is_empty(),
            "some glyph quads survive the clip"
        );
        assert!(
            clipped
                .vertices
                .iter()
                .all(|v| v.pos[0] >= 20.0 && v.pos[0] <= 60.0),
            "every clipped glyph quad stays inside [20, 60]"
        );
    }

    #[test]
    fn missing_image_draws_nothing_like_the_cpu_path() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Image {
            rect: Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8,
            },
            path: "nope.png".to_string(),
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert!(f.vertices.is_empty());
        assert!(f.batches.is_empty());
        assert!(f.image_uploads.is_empty());
    }

    /// Text the emoji font never draws stays out of the cluster map, so a note of
    /// distinct letters cannot grow it without bound.
    #[test]
    fn plain_text_leaves_the_cluster_cache_alone() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let text: String = ('α'..='ω')
            .chain("café naïve Ærøskøbing ├──".chars())
            .collect();
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 2000,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            text: format!("{text} 😀"),
            clip_x: None,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        let clusters: usize = cache.cluster_slots.values().map(HashMap::len).sum();
        assert!(
            clusters <= 1,
            "only the emoji is recorded, not {clusters} clusters"
        );
        if crate::platform::emoji::EmojiFont::open().is_some() {
            assert!(
                f.vertices.iter().any(|v| v.mode == MODE_EMOJI),
                "the emoji still draws in color"
            );
        }
    }

    #[test]
    fn emoji_lands_in_the_color_atlas() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let list = vec![DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 100,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            text: "😀".to_string(),
            clip_x: None,
        }];
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        // With an emoji font installed the cluster takes MODE_EMOJI; without
        // one it renders as per-character glyphs. Either way it must not
        // crash, and with color output there must be an emoji upload.
        if f.vertices.iter().any(|v| v.mode == MODE_EMOJI) {
            assert!(f.emoji_upload.is_some(), "the color raster was uploaded");
        }
        // The cluster, or its one scalar without an emoji font, misses once and
        // then hits.
        built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1 });
    }
}
