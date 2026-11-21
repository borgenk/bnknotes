//! Display-scale arithmetic: turning the editor's logical (96 DPI) geometry into
//! the device pixels the GPU buffer is drawn at.
//!
//! Scale is carried as 120ths, the unit the fractional-scale protocol reports
//! (`round(scale * 120)`): 120 is 1.0, 180 is 1.5, 240 is 2.0. A device length is
//! `round(logical * factor_120 / 120)`. At [`Scale::ONE`] every conversion is the
//! identity, so the scale-1.0 path draws the exact pixels the unscaled editor did.
//!
//! A [`Scale`] is a value threaded into the layout and paint passes so a constant
//! authored in logical pixels (a margin, a caret width) lands at the right device
//! size. The compositor tracking that produces the current scale lives in the app
//! layer (the `Scaling` controller on the editor state), not here.

/// A display scale as a 120ths factor. See the module docs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Scale {
    factor_120: u32,
}

impl Scale {
    /// Scale 1.0: the identity. Every conversion returns its input unchanged.
    pub const ONE: Scale = Scale { factor_120: 120 };

    /// Adopt a raw 120ths factor (from the compositor's fractional scale, or an
    /// integer output scale times 120), guarded against zero so a stray report can
    /// never collapse every length to nothing.
    pub fn from_120(factor_120: u32) -> Self {
        Self {
            factor_120: factor_120.max(1),
        }
    }

    /// The scale as a 120ths factor (120 = 1.0), for the cache keys and the buffer
    /// arithmetic that need the raw value.
    pub fn factor_120(self) -> u32 {
        self.factor_120
    }

    /// A logical length in device pixels, rounded to nearest. The `+ 60` is half of
    /// 120 (round-half-up); `i64` math so a large window times a large scale cannot
    /// overflow before the divide. Exact identity at [`Scale::ONE`]. `logical` is a
    /// size or an offset, so it is non-negative in practice.
    pub fn px(self, logical: i32) -> i32 {
        ((logical as i64 * self.factor_120 as i64 + 60) / 120) as i32
    }

    /// A logical font size in device pixels, rounded to nearest and floored at 1 so
    /// a face is never opened at zero. Identity at [`Scale::ONE`].
    pub fn font_px(self, logical: u32) -> u32 {
        (((logical as u64 * self.factor_120 as u64 + 60) / 120) as u32).max(1)
    }

    /// A logical pointer coordinate in device space, as a float (the pointer
    /// reports fractional positions). Identity at [`Scale::ONE`].
    pub fn scale_f32(self, logical: f32) -> f32 {
        logical * self.factor_120 as f32 / 120.0
    }

    /// A device length back in logical pixels, rounded to nearest: the inverse of
    /// [`Self::px`], for a device measurement (a dragged column edge) that is
    /// persisted in the scale-free logical space. Identity at [`Scale::ONE`].
    pub fn to_logical(self, device: i32) -> i32 {
        let f = self.factor_120.max(1) as i64;
        ((device as i64 * 120 + f / 2) / f) as i32
    }
}

#[cfg(test)]
mod tests {
    use crate::scale::*;

    #[test]
    fn scale_one_is_the_identity() {
        let s = Scale::ONE;
        assert_eq!(s.factor_120(), 120);
        for logical in [0, 1, 14, 40, 260, 16384] {
            assert_eq!(s.px(logical), logical, "px is identity at unity");
        }
        assert_eq!(s.font_px(14), 14);
        assert_eq!(s.scale_f32(123.5), 123.5);
    }

    #[test]
    fn px_rounds_to_nearest() {
        assert_eq!(Scale::from_120(120).px(100), 100); // 1.0x is identity
        assert_eq!(Scale::from_120(240).px(100), 200); // 2.0x
        assert_eq!(Scale::from_120(180).px(100), 150); // 1.5x
        assert_eq!(Scale::from_120(150).px(100), 125); // 1.25x
        assert_eq!(Scale::from_120(150).px(101), 126); // 126.25 -> 126 (nearest)
        // No overflow at the extremes (i64 math, then narrowed).
        assert_eq!(Scale::from_120(240).px(16384), 32768);
    }

    #[test]
    fn font_px_scales_with_the_display() {
        assert_eq!(Scale::from_120(120).font_px(14), 14); // 1.0x -> 14
        assert_eq!(Scale::from_120(240).font_px(14), 28); // 2.0x -> 28
        assert_eq!(Scale::from_120(180).font_px(14), 21); // 1.5x -> 21
        assert_eq!(Scale::from_120(150).font_px(14), 18); // 1.25x -> 17.5 -> 18
        // Same physical size two ways: 14px at 2x equals 28px at 1x.
        assert_eq!(
            Scale::from_120(240).font_px(14),
            Scale::from_120(120).font_px(28)
        );
    }

    #[test]
    fn to_logical_inverts_px() {
        // Round-trips at the common scales (nearest-rounding, so exact for these).
        for &f in &[120u32, 150, 180, 240] {
            let s = Scale::from_120(f);
            assert_eq!(s.to_logical(s.px(260)), 260, "tree width survives at {f}");
        }
        assert_eq!(Scale::ONE.to_logical(260), 260);
    }

    #[test]
    fn a_bogus_factor_never_collapses_to_zero() {
        // A zero report is guarded to 1; a tiny factor still floors the font at 1px
        // rather than opening a zero-size face.
        assert_eq!(Scale::from_120(0).factor_120(), 1);
        assert_eq!(Scale::from_120(1).font_px(14), 1);
    }
}
