//! Software rendering into an xRGB8888 pixel buffer. A `Canvas` borrows the shm
//! pool bytes and offers the few primitives the editor needs: a solid fill and
//! per-pixel coverage blending for anti-aliased glyphs.

use crate::freetype::Face;

/// Which corners [`Canvas::fill_round_rect`] rounds. A fenced code block's panel
/// is drawn row by row: its first row rounds the top, its last row the bottom, a
/// single-row block both, and the rows between none.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RoundedCorners {
    None,
    Top,
    Bottom,
    Both,
}

impl RoundedCorners {
    fn top(self) -> bool {
        matches!(self, Self::Top | Self::Both)
    }
    fn bottom(self) -> bool {
        matches!(self, Self::Bottom | Self::Both)
    }
}

pub struct Canvas<'a> {
    buf: &'a mut [u8],
    width: usize,
    height: usize,
    /// An optional clip rectangle `(x, y, w, h)` in pixels. When set, every
    /// primitive is confined to it (intersected with the surface): the damage path
    /// paints one changed region at a time without disturbing the rest of the
    /// buffer. `None` means clip only to the surface. Set via [`Self::set_clip`].
    clip: Option<(i32, i32, i32, i32)>,
}

impl<'a> Canvas<'a> {
    pub fn new(buf: &'a mut [u8], width: usize, height: usize) -> Self {
        Self {
            buf,
            width,
            height,
            clip: None,
        }
    }

    /// Confine subsequent drawing to `clip` (`(x, y, w, h)` in pixels), or clear
    /// the restriction with `None`. The clip is always intersected with the
    /// surface, so an oversized or negative rectangle can never write out of
    /// bounds. The damage executor sets this per changed region.
    pub fn set_clip(&mut self, clip: Option<(i32, i32, i32, i32)>) {
        self.clip = clip;
    }

    /// Whether pixel `(x, y)` is inside the active clip (always true when no clip
    /// is set). Surface bounds are checked separately by the caller.
    fn in_clip(&self, x: i32, y: i32) -> bool {
        match self.clip {
            None => true,
            Some((cx, cy, cw, ch)) => x >= cx && y >= cy && x < cx + cw && y < cy + ch,
        }
    }

    /// Fill an opaque rectangle by writing pixels directly (no per-pixel blend),
    /// clipped to the surface and the active clip. This is the fast path for the
    /// editor's solid fills (background, selection, caret, panels), and it is how
    /// a damage region is cleared before the frame is repainted over it: row runs
    /// are written with one `copy` each rather than a blend per pixel.
    pub fn fill_solid(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let (x0, y0, x1, y1) = self.clamp_rect(x, y, w, h);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let bytes = color.to_ne_bytes();
        let row_run = (x1 - x0) as usize * 4;
        for row in y0..y1 {
            let start = (row as usize * self.width + x0 as usize) * 4;
            let dst = &mut self.buf[start..start + row_run];
            for pixel in dst.chunks_exact_mut(4) {
                pixel.copy_from_slice(&bytes);
            }
        }
    }

    /// The half-open pixel rectangle `[x0, x1) x [y0, y1)` of `(x, y, w, h)` after
    /// clamping to the surface and (if any) the clip. An empty result has
    /// `x0 >= x1` or `y0 >= y1`.
    fn clamp_rect(&self, x: i32, y: i32, w: i32, h: i32) -> (i32, i32, i32, i32) {
        let (mut x0, mut y0) = (x, y);
        let (mut x1, mut y1) = (x + w.max(0), y + h.max(0));
        if let Some((cx, cy, cw, ch)) = self.clip {
            x0 = x0.max(cx);
            y0 = y0.max(cy);
            x1 = x1.min(cx + cw);
            y1 = y1.min(cy + ch);
        }
        x0 = x0.clamp(0, self.width as i32);
        y0 = y0.clamp(0, self.height as i32);
        x1 = x1.clamp(0, self.width as i32);
        y1 = y1.clamp(0, self.height as i32);
        (x0, y0, x1, y1)
    }

    /// Blend `color` over the pixel at (x, y) with `coverage` as the alpha
    /// (0 = transparent, 255 = opaque). Pixels outside the surface or the active
    /// clip are skipped. This is the bounds-checked entry; the hot loops
    /// ([`Self::draw_text`], [`Self::blit_argb`]) clip their span once against
    /// [`Self::clamp_rect`] and then call [`Self::blend_at`] with no per-pixel
    /// guard.
    pub fn blend(&mut self, x: i32, y: i32, color: u32, coverage: u8) {
        if x < 0 || y < 0 || !self.in_clip(x, y) {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height {
            return;
        }
        self.blend_at((y * self.width + x) * 4, color, coverage);
    }

    /// Alpha-blend `color`/`coverage` into the pixel at byte index `idx`, with no
    /// bounds or clip check: the caller guarantees `idx` is in range (the span
    /// loops clip first). This is the inner write the [`blend`](Self::blend)
    /// guards funnel into.
    #[inline]
    fn blend_at(&mut self, idx: usize, color: u32, coverage: u8) {
        let dst = &mut self.buf[idx..idx + 4];
        // Full coverage replaces the pixel outright (`blend_over` at 255 is exact),
        // skipping the read-modify-write. Glyph interiors and opaque image pixels
        // dominate real blits, so this is the common case, not the corner case.
        // The mask keeps the written bytes identical to `blend_over`'s, which
        // clears the unused x byte whatever the caller passed.
        if coverage == 255 {
            dst.copy_from_slice(&(color & 0x00ff_ffff).to_ne_bytes());
            return;
        }
        let bg = u32::from_ne_bytes([dst[0], dst[1], dst[2], dst[3]]);
        dst.copy_from_slice(&blend_over(bg, color, coverage).to_ne_bytes());
    }

    /// Fill a rectangle whose top and/or bottom corners are rounded with `radius`,
    /// anti-aliased. A fenced code block's panel is drawn one row at a time, so the
    /// block's first row rounds its top corners and its last row its bottom ones,
    /// and a multi-row block reads as one rounded panel. Pixels outside a rounded
    /// corner are left untouched, so the background shows through the curve. With
    /// no rounded corner (or zero radius) this is a plain [`Self::fill_solid`].
    ///
    /// Only the `radius`-sized corner cells blend per pixel; everything else is
    /// solid row runs, so a panel costs about the same as a plain fill.
    // Positional geometry args, like `fill_solid`; a struct would not read clearer.
    #[allow(clippy::too_many_arguments)]
    pub fn fill_round_rect(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        color: u32,
        radius: i32,
        corners: RoundedCorners,
    ) {
        // `.max(0)` keeps the clamp's upper bound non-negative: a degenerate
        // negative `w`/`h` would otherwise make `clamp(0, negative)` panic
        // (min > max). Not reachable from the sole caller, but cheap to make safe.
        let r = radius.clamp(0, (w / 2).min(h / 2).max(0));
        let (round_top, round_bottom) = (corners.top(), corners.bottom());
        if r == 0 || (!round_top && !round_bottom) {
            self.fill_solid(x, y, w, h, color);
            return;
        }
        // The rows clear of both corner bands are one solid block.
        let top_band = if round_top { r } else { 0 };
        let bottom_band = if round_bottom { r } else { 0 };
        self.fill_solid(x, y + top_band, w, h - top_band - bottom_band, color);
        // Within a corner band, only the two r-wide corner cells need per-pixel
        // coverage; the span between them is a solid run. `r <= w/2` and
        // `r <= h/2` keep the cells and bands from overlapping.
        for dy in (0..top_band).chain(h - bottom_band..h) {
            self.fill_solid(x + r, y + dy, w - 2 * r, 1, color);
            for dx in (0..r).chain(w - r..w) {
                let coverage = corner_coverage(dx, dy, w, h, r, round_top, round_bottom);
                if coverage > 0 {
                    self.blend(x + dx, y + dy, color, coverage);
                }
            }
        }
    }

    /// Lay out and draw `text` at `pen_x`, baseline `baseline`, in `color`.
    /// Glyphs are rasterized on demand; the pen advances by each glyph's
    /// advance. Returns the pen x after the last glyph.
    ///
    /// Each glyph's bitmap is clipped once to the visible region (surface
    /// intersected with any active clip) and the in-range pixels are blended with
    /// no per-pixel bounds check, so a screenful of glyphs costs millions fewer
    /// branches than guarding every pixel.
    pub fn draw_text(
        &mut self,
        face: &Face,
        pen_x: i32,
        baseline: i32,
        text: &str,
        color: u32,
    ) -> i32 {
        let (cx0, cy0, cx1, cy1) = self.clamp_rect(0, 0, self.width as i32, self.height as i32);
        let width = self.width;
        let mut pen = pen_x as f32;
        for ch in text.chars() {
            face.with_glyph(ch, |glyph| {
                let origin_x = pen.round() as i32 + glyph.left;
                let origin_y = baseline - glyph.top;
                // The glyph's bitmap rectangle, clipped to the visible region.
                let gx0 = origin_x.max(cx0);
                let gy0 = origin_y.max(cy0);
                let gx1 = (origin_x + glyph.width as i32).min(cx1);
                let gy1 = (origin_y + glyph.rows as i32).min(cy1);
                for y in gy0..gy1 {
                    let cov_row = (y - origin_y) as usize * glyph.width;
                    let px_row = y as usize * width;
                    for x in gx0..gx1 {
                        let coverage = glyph.coverage[cov_row + (x - origin_x) as usize];
                        if coverage > 0 {
                            self.blend_at((px_row + x as usize) * 4, color, coverage);
                        }
                    }
                }
                pen += glyph.advance;
            });
        }
        pen.round() as i32
    }

    /// Blit `src` (ARGB8888 `0xAARRGGBB`, exactly `w` x `h`, row-major) into the
    /// rectangle at `(x, y)`, alpha-blended over what is already there. No scaling:
    /// `src` is the display-size bitmap already (the image store caches the result
    /// of [`resample_argb`]), so a redraw is a cheap per-pixel blend rather than a
    /// resample every frame. A fully transparent pixel is skipped; pixels outside
    /// the canvas are clipped by [`Self::blend`].
    pub fn blit_argb(&mut self, x: i32, y: i32, w: i32, h: i32, src: &[u32]) {
        // Guard against a buffer smaller than its claimed size (no OOB indexing).
        if w <= 0 || h <= 0 || src.len() < (w * h) as usize {
            return;
        }
        // Clip the destination rectangle to the visible region once, then blend the
        // in-range pixels with no per-pixel bounds check.
        let (cx0, cy0, cx1, cy1) = self.clamp_rect(0, 0, self.width as i32, self.height as i32);
        let width = self.width;
        let bx0 = x.max(cx0);
        let by0 = y.max(cy0);
        let bx1 = (x + w).min(cx1);
        let by1 = (y + h).min(cy1);
        for py in by0..by1 {
            let src_row = ((py - y) * w) as usize;
            let px_row = py as usize * width;
            for px in bx0..bx1 {
                let argb = src[src_row + (px - x) as usize];
                let alpha = (argb >> 24) as u8;
                if alpha != 0 {
                    self.blend_at((px_row + px as usize) * 4, argb & 0x00ff_ffff, alpha);
                }
            }
        }
    }
}

/// Bilinearly resample ARGB `src` (`src_w` x `src_h`) into a fresh `dst_w` x
/// `dst_h` buffer. The image store caches the result per (image, display size), so
/// this runs once when an image first appears (or is resized) rather than on every
/// frame; [`Canvas::blit_argb`] then draws the cached bitmap cheaply. Returns an
/// empty buffer for degenerate or malformed input.
pub(crate) fn resample_argb(
    src: &[u32],
    src_w: u32,
    src_h: u32,
    dst_w: i32,
    dst_h: i32,
) -> Vec<u32> {
    let (sw, sh) = (src_w as i32, src_h as i32);
    if dst_w <= 0 || dst_h <= 0 || sw <= 0 || sh <= 0 || src.len() < (sw * sh) as usize {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((dst_w * dst_h) as usize);
    for dy in 0..dst_h {
        // Map each destination pixel center back into source space; the -0.5
        // offsets sample at source pixel centers, so a 1:1 copy is exact.
        let sy = (dy as f32 + 0.5) * sh as f32 / dst_h as f32 - 0.5;
        for dx in 0..dst_w {
            let sx = (dx as f32 + 0.5) * sw as f32 / dst_w as f32 - 0.5;
            out.push(sample_bilinear(src, sw, sh, sx, sy));
        }
    }
    out
}

/// Bilinearly sample the ARGB pixels `src` (`w` x `h`, row-major) at fractional
/// `(fx, fy)`, clamping to the edges. Each channel, alpha included, is
/// interpolated independently across the four nearest texels. Heavy downscaling
/// aliases (this skips source pixels rather than averaging them); an area filter
/// would be the polish if that ever shows.
fn sample_bilinear(src: &[u32], w: i32, h: i32, fx: f32, fy: f32) -> u32 {
    let (x0f, y0f) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0f, fy - y0f);
    let (xi, yi) = (x0f as i32, y0f as i32);
    // Clamp each corner independently, so a coordinate past an edge folds both
    // texels onto the edge pixel (no wrap, no out-of-range index).
    let x0 = xi.clamp(0, w - 1);
    let x1 = (xi + 1).clamp(0, w - 1);
    let y0 = yi.clamp(0, h - 1);
    let y1 = (yi + 1).clamp(0, h - 1);
    let at = |x: i32, y: i32| src[(y * w + x) as usize];
    let (c00, c10, c01, c11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    let mut out = 0u32;
    for shift in [0, 8, 16, 24] {
        let ch = |c: u32| ((c >> shift) & 0xff) as f32;
        let top = ch(c00) + (ch(c10) - ch(c00)) * tx;
        let bot = ch(c01) + (ch(c11) - ch(c01)) * tx;
        let v = (top + (bot - top) * ty).round().clamp(0.0, 255.0) as u32;
        out |= v << shift;
    }
    out
}

/// Total pen advance of `text` at the face's current pixel size, in pixels.
/// Measures with metrics only (no rasterization), matching how `draw_text`
/// advances the pen, so the caret lands where the next glyph would be drawn.
/// [`crate::layout`] uses this to place the caret and hit-test clicks.
pub fn text_advance(face: &Face, text: &str) -> f32 {
    text.chars().map(|c| face.advance(c)).sum()
}

/// Coverage (0..=255) of pixel `(dx, dy)` within a `w`x`h` rounded rectangle
/// whose top and/or bottom corners have radius `r`. Interior and straight-edge
/// pixels are fully covered; a pixel inside an active corner square is shaded by
/// how far its center falls within the corner's quarter-circle (centered `r` in
/// from each rounded edge), which anti-aliases the curve.
fn corner_coverage(
    dx: i32,
    dy: i32,
    w: i32,
    h: i32,
    r: i32,
    round_top: bool,
    round_bottom: bool,
) -> u8 {
    let rf = r as f32;
    // The center of the corner circle this pixel belongs to, if any; otherwise it
    // is interior or on a straight edge and fully covered.
    let (cx, cy) = if round_top && dx < r && dy < r {
        (rf, rf)
    } else if round_top && dx >= w - r && dy < r {
        ((w - r) as f32, rf)
    } else if round_bottom && dx < r && dy >= h - r {
        (rf, (h - r) as f32)
    } else if round_bottom && dx >= w - r && dy >= h - r {
        ((w - r) as f32, (h - r) as f32)
    } else {
        return 255;
    };
    let ex = dx as f32 + 0.5 - cx;
    let ey = dy as f32 + 0.5 - cy;
    let dist = (ex * ex + ey * ey).sqrt();
    // Coverage falls from 1 to 0 across the one-pixel band straddling the radius.
    let coverage = (rf - dist + 0.5).clamp(0.0, 1.0);
    (coverage * 255.0).round() as u8
}

/// Alpha-composite `fg` over `bg` with coverage `a` (0..=255), per channel.
fn blend_over(bg: u32, fg: u32, a: u8) -> u32 {
    let a = u32::from(a);
    let inv = 255 - a;
    let mix = |s: u32, d: u32| ((s * a + d * inv + 127) / 255) & 0xff;
    let r = mix((fg >> 16) & 0xff, (bg >> 16) & 0xff);
    let g = mix((fg >> 8) & 0xff, (bg >> 8) & 0xff);
    let b = mix(fg & 0xff, bg & 0xff);
    (r << 16) | (g << 8) | b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_coverage_takes_foreground() {
        assert_eq!(blend_over(0x00_00_00_00, 0x00_ab_cd_ef, 255), 0x00_ab_cd_ef);
    }

    #[test]
    fn zero_coverage_keeps_background() {
        assert_eq!(blend_over(0x00_12_34_56, 0x00_ff_ff_ff, 0), 0x00_12_34_56);
    }

    #[test]
    fn half_coverage_is_midpoint() {
        // 0 and 255 at alpha 128 round to 128 (within a unit).
        let out = blend_over(0x00_00_00_00, 0x00_ff_ff_ff, 128);
        for shift in [16, 8, 0] {
            let c = (out >> shift) & 0xff;
            assert!((127..=129).contains(&c), "channel {c} near midpoint");
        }
    }

    #[test]
    fn fill_solid_writes_every_pixel() {
        let mut buf = vec![0u8; 4 * 4]; // 2x2
        let mut canvas = Canvas::new(&mut buf, 2, 2);
        canvas.fill_solid(0, 0, 2, 2, 0x00_10_20_30);
        for px in buf.chunks_exact(4) {
            assert_eq!(u32::from_ne_bytes(px.try_into().unwrap()), 0x00_10_20_30);
        }
    }

    #[test]
    fn fill_solid_clips_to_surface_and_clip_rect() {
        let mut buf = vec![0u8; 4 * 9]; // 3x3
        let mut canvas = Canvas::new(&mut buf, 3, 3);
        // A clip of the center column lets only that column through, even though the
        // fill asks for the whole surface.
        canvas.set_clip(Some((1, 0, 1, 3)));
        canvas.fill_solid(0, 0, 3, 3, 0x00_ff_ff_ff);
        canvas.set_clip(None);
        let at = |x: usize, y: usize| {
            let i = (y * 3 + x) * 4;
            u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap()) & 0x00ff_ffff
        };
        assert_eq!(at(0, 1), 0, "outside the clip column");
        assert_eq!(at(1, 1), 0x00ff_ffff, "inside the clip column");
        assert_eq!(at(2, 1), 0, "outside the clip column");
    }

    #[test]
    fn blend_clips_out_of_bounds() {
        let mut buf = vec![0u8; 4]; // 1x1
        let mut canvas = Canvas::new(&mut buf, 1, 1);
        canvas.blend(-1, 0, 0x00_ff_ff_ff, 255);
        canvas.blend(5, 5, 0x00_ff_ff_ff, 255);
        assert_eq!(buf, vec![0u8; 4], "out-of-bounds writes are ignored");
    }

    #[test]
    fn round_rect_matches_a_per_pixel_coverage_reference() {
        // The row-run fast path must write exactly what blending `corner_coverage`
        // over every pixel would, for every corner combination and thin shapes.
        let color = 0x00_ab_cd_ef;
        for corners in [
            RoundedCorners::None,
            RoundedCorners::Top,
            RoundedCorners::Bottom,
            RoundedCorners::Both,
        ] {
            for (w, h, r) in [(12, 9, 3), (8, 8, 4), (10, 2, 3), (2, 10, 5)] {
                let mut fast = vec![0u8; (w * h) as usize * 4];
                Canvas::new(&mut fast, w as usize, h as usize)
                    .fill_round_rect(0, 0, w, h, color, r, corners);
                let mut reference = vec![0u8; (w * h) as usize * 4];
                let mut canvas = Canvas::new(&mut reference, w as usize, h as usize);
                let rr = r.clamp(0, (w / 2).min(h / 2).max(0));
                for dy in 0..h {
                    for dx in 0..w {
                        let cov = if rr == 0 || corners == RoundedCorners::None {
                            255
                        } else {
                            corner_coverage(dx, dy, w, h, rr, corners.top(), corners.bottom())
                        };
                        if cov > 0 {
                            canvas.blend(dx, dy, color, cov);
                        }
                    }
                }
                assert_eq!(
                    fast, reference,
                    "row-run path diverges for {corners:?} at {w}x{h} r{r}"
                );
            }
        }
    }

    #[test]
    fn round_rect_carves_only_the_requested_corners() {
        let white = 0x00_ff_ff_ff;
        let mut buf = vec![0u8; 4 * 100]; // 10x10, background 0
        let mut canvas = Canvas::new(&mut buf, 10, 10);
        // Round only the top corners.
        canvas.fill_round_rect(0, 0, 10, 10, white, 3, RoundedCorners::Top);
        let px = |x: usize, y: usize| {
            let i = (y * 10 + x) * 4;
            u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap())
        };
        assert_eq!(px(5, 5), white, "the interior is fully filled");
        assert_eq!(px(0, 0), 0, "the top-left corner is carved to background");
        assert_eq!(px(9, 0), 0, "the top-right corner is carved");
        assert_eq!(px(0, 9), white, "the bottom corners are left square");
        assert_eq!(px(9, 9), white, "the bottom-right corner is left square");
    }

    #[test]
    fn round_rect_with_no_rounded_corner_is_a_plain_fill() {
        let white = 0x00_ff_ff_ff;
        let mut buf = vec![0u8; 4 * 9]; // 3x3
        let mut canvas = Canvas::new(&mut buf, 3, 3);
        canvas.fill_round_rect(0, 0, 3, 3, white, 1, RoundedCorners::None);
        assert!(
            buf.chunks_exact(4)
                .all(|p| u32::from_ne_bytes(p.try_into().unwrap()) == white),
            "no rounded corner fills every pixel"
        );
    }

    /// Read back canvas pixel `(x, y)` as `0x00RRGGBB` (the x byte ignored).
    fn px(buf: &[u8], w: usize, x: usize, y: usize) -> u32 {
        let i = (y * w + x) * 4;
        u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap()) & 0x00ff_ffff
    }

    #[test]
    fn resample_upscales_a_single_pixel_to_a_solid_block() {
        // 1x1 red, scaled to 3x3, is nine reds (the cache stores this, blitted later).
        let scaled = resample_argb(&[0xff_ff_00_00], 1, 1, 3, 3);
        assert_eq!(scaled.len(), 9);
        assert!(scaled.iter().all(|&p| p == 0xff_ff_00_00), "all red");
    }

    #[test]
    fn resample_is_exact_at_one_to_one() {
        let src = [0xff_11_22_33, 0xff_44_55_66, 0xff_77_88_99, 0xff_aa_bb_cc];
        let out = resample_argb(&src, 2, 2, 2, 2);
        assert_eq!(out, src, "a 1:1 resample copies the pixels exactly");
    }

    #[test]
    fn resample_rejects_malformed_input() {
        // Claims 2x2 but carries one pixel: empty result rather than reading OOB.
        assert!(resample_argb(&[0xff_ff_00_00], 2, 2, 4, 4).is_empty());
        assert!(
            resample_argb(&[0xff_ff_00_00], 1, 1, 0, 4).is_empty(),
            "zero dst"
        );
    }

    #[test]
    fn blit_argb_copies_alpha_blending_over_the_background() {
        let mut buf = vec![0u8; 4 * 4]; // 2x2, black
        let mut canvas = Canvas::new(&mut buf, 2, 2);
        // Opaque red at (0,0), ~50% white at (1,1); the other two transparent.
        let src = [0xff_ff_00_00, 0x00_00_00_00, 0x00_00_00_00, 0x80_ff_ff_ff];
        canvas.blit_argb(0, 0, 2, 2, &src);
        assert_eq!(px(&buf, 2, 0, 0), 0x00ff_0000, "opaque red copied");
        assert_eq!(px(&buf, 2, 1, 0), 0, "transparent pixel skipped");
        for shift in [16, 8, 0] {
            let c = (px(&buf, 2, 1, 1) >> shift) & 0xff;
            assert!((120..=136).contains(&c), "half white over black, got {c}");
        }
    }

    #[test]
    fn blit_argb_clips_outside_the_canvas() {
        let mut buf = vec![0u8; 4 * 4]; // 2x2, black
        let mut canvas = Canvas::new(&mut buf, 2, 2);
        // A 2x2 source anchored at (1,1): only its top-left pixel lands in bounds.
        let src = [0xff_ff_00_00, 0xff_ff_00_00, 0xff_ff_00_00, 0xff_ff_00_00];
        canvas.blit_argb(1, 1, 2, 2, &src);
        assert_eq!(px(&buf, 2, 0, 0), 0, "untouched");
        assert_eq!(px(&buf, 2, 1, 1), 0x00ff_0000, "the one in-bounds pixel");
    }

    #[test]
    fn blit_argb_ignores_a_short_buffer() {
        let mut buf = vec![0u8; 4];
        let mut canvas = Canvas::new(&mut buf, 1, 1);
        canvas.fill_solid(0, 0, 1, 1, 0x00_aa_bb_cc);
        // Claims 2x2 but carries one pixel: refused rather than reading OOB.
        canvas.blit_argb(0, 0, 2, 2, &[0xff_ff_00_00]);
        assert_eq!(px(&buf, 1, 0, 0), 0x00_aa_bb_cc, "left untouched");
    }

    #[test]
    fn text_advance_is_zero_empty_and_grows_with_length() {
        let face = Face::open_default(32).expect("a default font");
        assert_eq!(text_advance(&face, ""), 0.0);
        let one = text_advance(&face, "m");
        let two = text_advance(&face, "mm");
        assert!(one > 0.0);
        assert!(two > one, "two glyphs advance further than one");
    }
}
