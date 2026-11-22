//! Shelf-packed texture atlases for glyph rasters: where each raster lands, how
//! an atlas grows or is wiped at its cap, and the rasters queued for upload.

/// Atlases start here and double when full, up to [`ATLAS_MAX`]; growth wipes
/// the atlas (sources are cached in `Fonts`, so re-inserting is cheap) and the
/// frame build restarts so no stale coordinates survive.
const ATLAS_START: u32 = 1024;
pub(super) const ATLAS_MAX: u32 = 8192;

/// The rasters packed into an atlas since the last frame, for the renderer to
/// copy into the atlas texture.
#[derive(Default)]
pub struct AtlasUploads {
    pub regions: Vec<UploadRegion>,
    /// Every region's texels, tight rows, at the region's offset.
    pub bytes: Vec<u8>,
}

impl AtlasUploads {
    fn clear(&mut self) {
        self.regions.clear();
        self.bytes.clear();
    }
}

/// One packed raster: its slot in the atlas and where its texels start in
/// AtlasUploads::bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UploadRegion {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub offset: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtlasInfo {
    pub width: u32,
    pub height: u32,
    /// Bumped whenever the atlas is wiped, as it grows or is reclaimed; the
    /// renderer recreates the texture when it changes.
    pub generation: u32,
}

/// Where a raster landed in an atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Slot {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) w: u32,
    pub(super) h: u32,
}

/// A shelf-packing texture atlas. Rasters are packed onto horizontal shelves
/// (rows share a height class), and each one is queued for upload as it is
/// packed. The texels live only in the GPU texture: a slot is written once and
/// read only inside its own bounds (the shader fetches texels at integer
/// coordinates), so the atlas keeps no copy.
pub struct Atlas {
    width: u32,
    height: u32,
    /// Bytes per texel: 1 (R8 coverage) or 4 (BGRA emoji).
    bpp: u32,
    /// Shelves as (y, height, x_used).
    shelves: Vec<(u32, u32, u32)>,
    next_y: u32,
    generation: u32,
    pending: AtlasUploads,
    /// Whether a full atlas at ATLAS_MAX may be wiped to make room. Once per
    /// frame build, so a frame whose own rasters overflow the cap cannot loop.
    may_reclaim: bool,
}

/// The atlas is at its cap and full, and this frame build already wiped it once.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct AtlasFull;

impl Atlas {
    pub fn new(bpp: u32) -> Self {
        Self {
            width: ATLAS_START,
            height: ATLAS_START,
            bpp,
            shelves: Vec::new(),
            next_y: 0,
            generation: 0,
            pending: AtlasUploads::default(),
            may_reclaim: true,
        }
    }

    pub(super) fn generation(&self) -> u32 {
        self.generation
    }

    /// Let the next frame build wipe this atlas once if it fills at its cap.
    pub(super) fn allow_reclaim(&mut self) {
        self.may_reclaim = true;
    }

    pub fn info(&self) -> AtlasInfo {
        AtlasInfo {
            width: self.width,
            height: self.height,
            generation: self.generation,
        }
    }

    /// Reserve a w x h slot. Growing, or wiping a full atlas at its cap, forgets
    /// every previous slot and bumps the generation, which the frame builder
    /// answers by restarting the build. A raster larger than the cap gets no slot.
    pub(super) fn reserve(&mut self, w: u32, h: u32) -> Result<Option<Slot>, AtlasFull> {
        if w == 0 || h == 0 || w > ATLAS_MAX || h > ATLAS_MAX {
            return Ok(None);
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
                return Ok(Some(slot));
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
                return Ok(Some(slot));
            }
            if !self.grow() && !self.reclaim() {
                return Err(AtlasFull);
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
        self.wipe();
        true
    }

    /// Wipe the atlas so the slots this frame draws can be packed again, when it
    /// is full at its cap or its slot map is. Returns false once this frame build
    /// has already done it.
    pub(super) fn reclaim(&mut self) -> bool {
        if !self.may_reclaim {
            return false;
        }
        self.may_reclaim = false;
        self.wipe();
        true
    }

    /// Forget every slot and start a new texture generation.
    fn wipe(&mut self) {
        self.clear();
        self.generation += 1;
    }

    /// Forget every slot and every queued upload, keeping the generation.
    pub(super) fn clear(&mut self) {
        self.shelves.clear();
        self.next_y = 0;
        self.pending.clear();
    }

    /// Reserve a slot for a w x h raster and queue src, its texels in tight rows,
    /// for upload. Ok(None) for a zero-area raster, which takes no slot. A src of
    /// the wrong length is cut or zero-filled to w * h texels.
    pub(super) fn pack(
        &mut self,
        w: u32,
        h: u32,
        src: impl IntoIterator<Item = u8>,
    ) -> Result<Option<Slot>, AtlasFull> {
        if w == 0 || h == 0 {
            return Ok(None);
        }
        let Some(slot) = self.reserve(w, h)? else {
            return Ok(None);
        };
        let offset = self.pending.bytes.len();
        let len = (w * h * self.bpp) as usize;
        self.pending.bytes.extend(src.into_iter().take(len));
        self.pending.bytes.resize(offset + len, 0);
        self.pending.regions.push(UploadRegion {
            x: slot.x,
            y: slot.y,
            w,
            h,
            offset,
        });
        Ok(Some(slot))
    }

    /// Hand the queued uploads to into, taking back its consumed buffers so their
    /// capacity is reused.
    pub(super) fn take_uploads(&mut self, into: &mut AtlasUploads) {
        std::mem::swap(into, &mut self.pending);
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::render::atlas::*;

    #[test]
    fn atlas_packs_onto_shelves_and_queues_each_raster() {
        let mut a = Atlas::new(1);
        let s1 = a.pack(10, 8, [7u8; 80]).unwrap().expect("fits");
        let s2 = a
            .pack(6, 8, [8u8; 48])
            .unwrap()
            .expect("fits on the same shelf");
        assert_eq!((s1.x, s1.y), (0, 0));
        assert_eq!((s2.x, s2.y), (10, 0), "same shelf, packed rightward");
        let s3 = a
            .pack(10, 20, [9u8; 200])
            .unwrap()
            .expect("taller: a new shelf");
        assert_eq!(s3.y, 8, "new shelf below the first");

        // Each raster uploads on its own, not as the box spanning all three.
        let mut up = AtlasUploads::default();
        a.take_uploads(&mut up);
        let regions: Vec<_> = up.regions.iter().map(|r| (r.x, r.y, r.w, r.h)).collect();
        assert_eq!(regions, [(0, 0, 10, 8), (10, 0, 6, 8), (0, 8, 10, 20)]);
        assert_eq!(up.bytes.len(), 80 + 48 + 200);
        let last = up.regions[2];
        assert!(up.bytes[last.offset..].iter().all(|&b| b == 9));

        a.take_uploads(&mut up);
        assert!(up.regions.is_empty() && up.bytes.is_empty(), "taken once");
    }

    #[test]
    fn atlas_grows_by_wiping_and_bumping_generation() {
        let mut a = Atlas::new(1);
        let g0 = a.generation;
        a.pack(4, 4, [1u8; 16]).unwrap();
        // A raster wider than the atlas forces growth.
        let s = a
            .reserve(ATLAS_START + 1, 4)
            .unwrap()
            .expect("grows to fit");
        assert!(a.generation > g0);
        assert_eq!(a.width, ATLAS_START * 2);
        assert_eq!((s.x, s.y), (0, 0), "the grown atlas starts empty");
        assert!(
            a.pending.regions.is_empty(),
            "the wiped raster is not uploaded"
        );
        // Beyond the cap is no slot at all.
        assert_eq!(a.reserve(ATLAS_MAX + 1, 1), Ok(None));
    }

    #[test]
    fn a_full_atlas_at_its_cap_is_wiped_once_per_build() {
        let mut a = Atlas::new(1);
        a.reserve(ATLAS_MAX, ATLAS_MAX)
            .unwrap()
            .expect("grows to the cap");
        let full = a.generation;
        let slot = a.reserve(4, 4).unwrap().expect("a wipe makes room");
        assert_eq!((slot.x, slot.y), (0, 0));
        assert_eq!(a.generation, full + 1, "a wiped atlas is a new texture");

        a.reserve(ATLAS_MAX, ATLAS_MAX - 4)
            .unwrap()
            .expect("fills the rest");
        assert_eq!(
            a.reserve(8, 8),
            Err(AtlasFull),
            "a second wipe in one build would loop"
        );
    }
}
