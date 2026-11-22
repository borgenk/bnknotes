//! The GPU backend's CPU half: turning a display list into GPU-consumable
//! data. This module is deliberately free of Vulkan: it packs glyph rasters
//! into atlas slots and emits vertices, draw batches and the rasters to upload
//! as plain values, so the geometry and packing logic is unit-testable without
//! a device.
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

use crate::platform::emoji::wants_emoji;
use crate::platform::freetype::{CodeInk, Face, FaceKey, Fonts};
use crate::platform::geom::Rect;
use crate::platform::grapheme;
use crate::render::atlas::{AtlasInfo, AtlasUploads};
use crate::render::display::DrawCmd;
use crate::render::display::RoundedCorners;
use crate::render::glyph_cache::{GlyphCache, ImageId, PackedGlyph};
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

/// An image bitmap the renderer does not hold yet: its id, its native size, and
/// straight ARGB words (uploadable as B8G8R8A8 bytes verbatim).
pub struct ImageUpload {
    pub id: ImageId,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u32>,
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
    /// The rasters each atlas packed for this frame.
    pub glyph_uploads: AtlasUploads,
    pub emoji_uploads: AtlasUploads,
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

/// Build one frame's GPU data from the display list into `out`, whose buffers are
/// cleared and refilled (the caller keeps it across frames, so a steady-state frame
/// allocates nothing). `known_images` is the set of image textures the renderer
/// already holds; missing ones get their bitmaps attached to the result. Restarts
/// internally if an atlas grows or is wiped mid-build, so the vertices always
/// reference current atlas coordinates.
pub fn build_frame(
    fonts: &Fonts,
    stores: &[&ImageStore],
    list: &[DrawCmd],
    cache: &mut GlyphCache,
    known_images: &dyn Fn(ImageId) -> bool,
    code_glyph_gamma: f32,
    out: &mut FrameData,
) {
    cache.glyphs.allow_reclaim();
    cache.emoji.allow_reclaim();
    loop {
        let glyphs_gen = cache.glyphs.generation();
        let emoji_gen = cache.emoji.generation();
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
        // An atlas grew, or was wiped at its cap, during the build: earlier quads
        // reference dead coordinates, so rebuild. A wipe happens once per build,
        // so a frame whose own rasters overflow the cap draws what fits.
        if cache.glyphs.generation() != glyphs_gen || cache.emoji.generation() != emoji_gen {
            cache.reset_for(glyphs_gen, emoji_gen);
            continue;
        }
        out.glyph_atlas = cache.glyphs.info();
        out.emoji_atlas = cache.emoji.info();
        cache.glyphs.take_uploads(&mut out.glyph_uploads);
        cache.emoji.take_uploads(&mut out.emoji_uploads);
        return;
    }
}

/// What every glyph of one text run is drawn with.
struct Run<'f> {
    face: &'f Face,
    /// The raster key, shared by both code inks.
    key: FaceKey,
    /// The key's ASCII table in the glyph cache.
    ascii: usize,
    baseline: i32,
    color: u32,
    gamma: f32,
    clip_x: Option<[i32; 2]>,
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
        let key = face_key.raster();
        let run = Run {
            face,
            key,
            ascii: self.cache.ascii_table(key),
            baseline,
            color,
            gamma,
            clip_x,
        };
        let mut pen = x as f32;
        if text.is_ascii() {
            for ch in text.chars() {
                pen = self.scalar(&run, ch, pen);
            }
            return;
        }
        for (_, cluster) in grapheme::graphemes(text) {
            match self.packed_cluster(face, key, cluster) {
                Some(packed) => {
                    // Emoji carry their own color and ignore the mask gamma.
                    self.emit_glyph(&packed, pen, baseline, MODE_EMOJI, 0, 0.0, clip_x);
                    pen += packed.advance;
                }
                None => {
                    for ch in cluster.chars() {
                        pen = self.scalar(&run, ch, pen);
                    }
                }
            }
        }
    }

    /// One character's glyph quad (or nothing for empty rasters like spaces),
    /// returning the advanced pen. Its placement is cached beside the atlas
    /// slot, so a steady-state frame emits the quad without re-rasterizing the
    /// glyph.
    fn scalar(&mut self, run: &Run, ch: char, pen: f32) -> f32 {
        let packed = self.packed_scalar(run, ch);
        self.emit_glyph(
            &packed,
            pen,
            run.baseline,
            MODE_GLYPH,
            run.color,
            run.gamma,
            run.clip_x,
        );
        pen + packed.advance
    }

    /// The cached placement for a scalar glyph. On the first sight of this
    /// (face, character) the glyph is rasterized once and its coverage written
    /// straight into the atlas; later frames read the cached placement.
    fn packed_scalar(&mut self, run: &Run, ch: char) -> PackedGlyph {
        if let Some(packed) = self.cache.scalar_slot(run.ascii, run.key, ch) {
            self.cache.stats.hits += 1;
            return packed;
        }
        self.cache.stats.misses += 1;
        // Rasterize from the fallback face when the keyed face lacks the glyph, so a
        // scalar the prose font has no glyph for (an arrow, a maths sign) draws
        // instead of a tofu box. The advance comes from the same face, since
        // `Face::advance` resolves the scalar the way `glyph_face` does, so the pen
        // moves by the width of the glyph that was actually inked.
        let g = self.fonts.glyph_face(run.key, ch).rasterize(ch);
        let placed =
            self.cache
                .glyphs
                .pack(g.width as u32, g.rows as u32, g.coverage.iter().copied());
        // A raster the full atlas could not take stays unrecorded, so a later
        // frame packs it again.
        let fitted = placed.is_ok();
        let packed = PackedGlyph {
            slot: placed.unwrap_or_default(),
            left: g.left,
            top: g.top,
            advance: run.face.advance(ch),
        };
        if fitted {
            self.cache.record_scalar(run.ascii, run.key, ch, packed);
        }
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
        let Some(g) = face.rasterize_cluster(cluster) else {
            self.cache.record_cluster(face_key, cluster, None);
            return None;
        };
        self.cache.stats.misses += 1;
        // ARGB words are B,G,R,A bytes in memory: B8G8R8A8 verbatim.
        let placed = self.cache.emoji.pack(
            g.width as u32,
            g.rows as u32,
            g.argb.iter().flat_map(|px| px.to_ne_bytes()),
        );
        let fitted = placed.is_ok();
        let packed = PackedGlyph {
            slot: placed.unwrap_or_default(),
            left: g.left,
            top: g.top,
            advance: g.advance,
        };
        if fitted {
            self.cache.record_cluster(face_key, cluster, Some(packed));
        }
        Some(packed)
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
    use std::collections::HashMap;

    use crate::render::atlas::ATLAS_MAX;
    use crate::render::glyph_cache::{CacheStats, MAX_SLOT_RECORDS};
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

    /// A run of regular prose at 16 px.
    fn prose_text(text: &str) -> DrawCmd {
        DrawCmd::Text {
            bounds: Rect {
                x: 0,
                y: 0,
                w: 200,
                h: 20,
            },
            x: 0,
            baseline: 16,
            face: FaceKey::Prose {
                size: 16,
                style: crate::platform::freetype::FontStyle::Regular,
            },
            color: 0x00ff_ffff,
            text: text.to_string(),
            clip_x: None,
        }
    }

    #[test]
    fn text_still_draws_once_the_atlas_fills_at_its_cap() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        // Earlier frames filled the glyph atlas to its cap.
        cache.glyphs.reserve(ATLAS_MAX, ATLAS_MAX).unwrap();
        let list = vec![prose_text("abc")];
        let glyph_quads =
            |f: &FrameData| f.vertices.iter().filter(|v| v.mode == MODE_GLYPH).count() / 6;

        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(glyph_quads(&f), 3, "the full atlas makes room");
        assert_eq!(f.glyph_uploads.regions.len(), 3);
        let f = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(glyph_quads(&f), 3, "and keeps them");
        assert!(f.glyph_uploads.regions.is_empty());
    }

    #[test]
    fn the_slot_maps_start_over_at_their_cap() {
        let fonts = Fonts::new(&[16]).expect("default font");
        let images = ImageStore::new(None);
        let mut cache = GlyphCache::new();
        let key = FaceKey::Prose {
            size: 16,
            style: crate::platform::freetype::FontStyle::Regular,
        };
        let inkless = PackedGlyph {
            slot: None,
            left: 0,
            top: 0,
            advance: 4.0,
        };
        // Records that take no atlas room, as many as either map holds.
        for i in 0..MAX_SLOT_RECORDS as u32 {
            let ch = char::from_u32(0x2_0000 + i).expect("a scalar value");
            cache.scalar_slots.insert((key, ch), inkless);
            cache
                .cluster_slots
                .entry(key)
                .or_default()
                .insert(format!("{i}"), None);
        }
        cache.cluster_records = MAX_SLOT_RECORDS;
        let (glyphs, emoji) = (cache.glyphs.generation(), cache.emoji.generation());

        built(
            &fonts,
            &[&images],
            &[prose_text("é😀")],
            &mut cache,
            &|_| false,
            1.0,
        );
        assert_eq!(
            cache.glyphs.generation(),
            glyphs + 1,
            "the cap wipes the atlas"
        );
        assert_eq!(cache.emoji.generation(), emoji + 1);
        // Without an emoji font the smiley draws per character, so it is a
        // scalar record too.
        let drawn = ['é', '😀'];
        assert!(
            !cache.scalar_slots.is_empty()
                && cache.scalar_slots.keys().all(|(_, ch)| drawn.contains(ch)),
            "only this frame's glyphs"
        );
        assert_eq!(cache.cluster_records, 1, "only this frame's cluster");
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
        assert_eq!(
            f1.glyph_uploads.regions.len(),
            3,
            "one upload per new glyph"
        );
        assert!(
            !f1.vertices.is_empty(),
            "glyph quads were emitted ({} vertices)",
            f1.vertices.len()
        );
        // Three distinct letters miss once each; their repeats hit.
        assert_eq!(cache.stats(), CacheStats { hits: 3, misses: 3 });
        let f2 = built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert!(
            f2.glyph_uploads.regions.is_empty(),
            "steady state uploads nothing"
        );
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
            assert_eq!(
                f.emoji_uploads.regions.len(),
                1,
                "the color raster was uploaded"
            );
        }
        // The cluster, or its one scalar without an emoji font, misses once and
        // then hits.
        built(&fonts, &[&images], &list, &mut cache, &|_| false, 1.0);
        assert_eq!(cache.stats(), CacheStats { hits: 1, misses: 1 });
    }
}
