//! Software rendering into an xRGB8888 pixel buffer. A `Canvas` borrows the shm
//! pool bytes and offers the few primitives the editor needs: a solid fill and
//! per-pixel coverage blending for anti-aliased glyphs.

use crate::freetype::Face;

/// Which corners [`Canvas::fill_round_rect`] rounds. A fenced code block's panel
/// is drawn row by row: its first row rounds the top, its last row the bottom, a
/// single-row block both, and the rows between none.
#[derive(Clone, Copy, PartialEq, Eq)]
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
}

impl<'a> Canvas<'a> {
    pub fn new(buf: &'a mut [u8], width: usize, height: usize) -> Self {
        Self { buf, width, height }
    }

    /// Paint every pixel with `color` (0x00RRGGBB).
    pub fn fill(&mut self, color: u32) {
        let bytes = color.to_ne_bytes();
        for pixel in self.buf.chunks_exact_mut(4) {
            pixel.copy_from_slice(&bytes);
        }
    }

    /// Blend `color` over the pixel at (x, y) with `coverage` as the alpha
    /// (0 = transparent, 255 = opaque). Out-of-bounds pixels are skipped.
    pub fn blend(&mut self, x: i32, y: i32, color: u32, coverage: u8) {
        if x < 0 || y < 0 {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) * 4;
        let dst = &mut self.buf[idx..idx + 4];
        let bg = u32::from_ne_bytes([dst[0], dst[1], dst[2], dst[3]]);
        dst.copy_from_slice(&blend_over(bg, color, coverage).to_ne_bytes());
    }

    /// Fill an axis-aligned rectangle with an opaque `color`. Used for the
    /// caret. Pixels outside the canvas are clipped by `blend`.
    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        for dy in 0..h {
            for dx in 0..w {
                self.blend(x + dx, y + dy, color, 255);
            }
        }
    }

    /// Fill a rectangle whose top and/or bottom corners are rounded with `radius`,
    /// anti-aliased. A fenced code block's panel is drawn one row at a time, so the
    /// block's first row rounds its top corners and its last row its bottom ones,
    /// and a multi-row block reads as one rounded panel. Pixels outside a rounded
    /// corner are left untouched, so the background shows through the curve. With
    /// no rounded corner (or zero radius) this is a plain [`Self::fill_rect`].
    // Positional geometry args, like `fill_rect`; a struct would not read clearer.
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
        let r = radius.clamp(0, (w / 2).min(h / 2));
        let (round_top, round_bottom) = (corners.top(), corners.bottom());
        if r == 0 || (!round_top && !round_bottom) {
            self.fill_rect(x, y, w, h, color);
            return;
        }
        for dy in 0..h {
            for dx in 0..w {
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
    pub fn draw_text(
        &mut self,
        face: &Face,
        pen_x: i32,
        baseline: i32,
        text: &str,
        color: u32,
    ) -> i32 {
        let mut pen = pen_x as f32;
        for ch in text.chars() {
            face.with_glyph(ch, |glyph| {
                let origin_x = pen.round() as i32 + glyph.left;
                let origin_y = baseline - glyph.top;
                for row in 0..glyph.rows {
                    for col in 0..glyph.width {
                        let coverage = glyph.coverage[row * glyph.width + col];
                        if coverage > 0 {
                            self.blend(
                                origin_x + col as i32,
                                origin_y + row as i32,
                                color,
                                coverage,
                            );
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
        for dy in 0..h {
            let row = (dy * w) as usize;
            for dx in 0..w {
                let argb = src[row + dx as usize];
                let alpha = (argb >> 24) as u8;
                if alpha != 0 {
                    self.blend(x + dx, y + dy, argb & 0x00ff_ffff, alpha);
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
    fn fill_writes_every_pixel() {
        let mut buf = vec![0u8; 4 * 4]; // 2x2
        let mut canvas = Canvas::new(&mut buf, 2, 2);
        canvas.fill(0x00_10_20_30);
        for px in buf.chunks_exact(4) {
            assert_eq!(u32::from_ne_bytes(px.try_into().unwrap()), 0x00_10_20_30);
        }
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
    fn fill_rect_paints_only_its_region_and_clips() {
        let mut buf = vec![0u8; 4 * 9]; // 3x3
        let mut canvas = Canvas::new(&mut buf, 3, 3);
        // A 2x2 rect at (1,1) reaches one pixel past the edge and must clip.
        canvas.fill_rect(1, 1, 2, 2, 0x00_ff_ff_ff);
        let px = |x: usize, y: usize| {
            let i = (y * 3 + x) * 4;
            u32::from_ne_bytes(buf[i..i + 4].try_into().unwrap())
        };
        assert_eq!(px(0, 0), 0, "outside the rect");
        assert_eq!(px(1, 1), 0x00_ff_ff_ff, "inside");
        assert_eq!(px(2, 2), 0x00_ff_ff_ff, "inside, at the corner");
        assert_eq!(px(0, 1), 0, "left of the rect");
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
        canvas.fill(0x00_aa_bb_cc);
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
