//! The caches the frame builder packs into: glyph and emoji-cluster placements
//! over the two atlases, and the ids of uploaded image textures.

use std::collections::HashMap;

use crate::platform::freetype::FaceKey;
use crate::render::atlas::{Atlas, Slot};

/// Most records either slot map holds. Inkless glyphs and clusters that are not
/// emoji take no atlas room, so a full atlas alone would not bound the maps; at
/// this many the atlas is wiped as if full, and its map starts over with it.
pub(super) const MAX_SLOT_RECORDS: usize = 1 << 16;

/// One uploaded image texture, interned by [`GlyphCache`] per destination path and
/// native size. A frame's quads carry the id rather than the path, so batching an
/// image costs no string and binding one hashes four bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImageId(u32);

/// One face's ASCII placements, indexed by the character's byte.
type AsciiSlots = [Option<PackedGlyph>; 128];

/// A glyph's atlas placement plus its pen metrics, cached beside the slot so a
/// steady-state frame emits the quad without re-rasterizing the glyph. `slot` is
/// `None` for an inkless glyph (a space, or an emoji that shaped to no ink) or
/// one the atlas could not fit; the pen still advances by `advance`, so the
/// caret and the glyphs after it land where layout measured.
#[derive(Clone, Copy)]
pub(super) struct PackedGlyph {
    pub(super) slot: Option<Slot>,
    pub(super) left: i32,
    pub(super) top: i32,
    pub(super) advance: f32,
}

/// The persistent glyph caches the batcher packs into: slots by scalar and by
/// cluster, over the two atlases. Owned by the app next to the display list;
/// survives across frames so steady-state frames insert nothing.
pub struct GlyphCache {
    pub glyphs: Atlas,
    pub emoji: Atlas,
    /// ASCII placements, one table per raster key. A text run finds its table
    /// once and indexes it per character, where scalar_slots, which holds every
    /// other character, hashes each lookup. The tables are few (one per face and
    /// size in use) and fixed in size, so they need no cap.
    ascii_slots: Vec<AsciiSlots>,
    ascii_tables: HashMap<FaceKey, usize>,
    pub(super) scalar_slots: HashMap<(FaceKey, char), PackedGlyph>,
    /// Nested so lookups borrow the cluster as `&str` (no per-frame `String`).
    /// A `None` value records a cluster that is not color emoji, so a
    /// steady-state frame takes the per-character path without re-consulting
    /// the `Face`.
    pub(super) cluster_slots: HashMap<FaceKey, HashMap<String, Option<PackedGlyph>>>,
    /// Records across every map in cluster_slots.
    pub(super) cluster_records: usize,
    /// Image texture ids, by native size then by path: the same nesting as
    /// `cluster_slots` and for the same reason, so a frame's image lookup borrows
    /// the path instead of building a key.
    image_ids: HashMap<(u32, u32), HashMap<String, ImageId>>,
    next_image_id: u32,
    pub(super) stats: CacheStats,
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
            ascii_slots: Vec::new(),
            ascii_tables: HashMap::new(),
            scalar_slots: HashMap::new(),
            cluster_slots: HashMap::new(),
            cluster_records: 0,
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
    pub(super) fn image_id(&mut self, path: &str, width: u32, height: u32) -> ImageId {
        let by_path = self.image_ids.entry((width, height)).or_default();
        if let Some(&id) = by_path.get(path) {
            return id;
        }
        let id = ImageId(self.next_image_id);
        self.next_image_id += 1;
        by_path.insert(path.to_string(), id);
        id
    }

    /// The index of key's ASCII table, adding an empty one on first use.
    pub(super) fn ascii_table(&mut self, key: FaceKey) -> usize {
        if let Some(&table) = self.ascii_tables.get(&key) {
            return table;
        }
        let table = self.ascii_slots.len();
        self.ascii_slots.push([None; 128]);
        self.ascii_tables.insert(key, table);
        table
    }

    /// A scalar's recorded placement: an ASCII character's from the run's table,
    /// any other from the map.
    pub(super) fn scalar_slot(&self, ascii: usize, key: FaceKey, ch: char) -> Option<PackedGlyph> {
        match self.ascii_slots.get(ascii).and_then(|t| t.get(ch as usize)) {
            Some(slot) => *slot,
            None => self.scalar_slots.get(&(key, ch)).copied(),
        }
    }

    /// Record a scalar's placement. At the map's record cap the glyph atlas is
    /// wiped instead, which restarts the build with an empty map; when this build
    /// has wiped it already, the placement goes unrecorded and a later frame
    /// retries.
    pub(super) fn record_scalar(
        &mut self,
        ascii: usize,
        key: FaceKey,
        ch: char,
        packed: PackedGlyph,
    ) {
        let table = self.ascii_slots.get_mut(ascii);
        if let Some(slot) = table.and_then(|t| t.get_mut(ch as usize)) {
            *slot = Some(packed);
        } else if self.scalar_slots.len() < MAX_SLOT_RECORDS {
            self.scalar_slots.insert((key, ch), packed);
        } else {
            self.glyphs.reclaim();
        }
    }

    /// Record a cluster's placement, or None for a cluster that does not draw as
    /// emoji, under the same cap as record_scalar.
    pub(super) fn record_cluster(
        &mut self,
        key: FaceKey,
        cluster: &str,
        packed: Option<PackedGlyph>,
    ) {
        if self.cluster_records < MAX_SLOT_RECORDS {
            self.cluster_slots
                .entry(key)
                .or_default()
                .insert(cluster.to_string(), packed);
            self.cluster_records += 1;
        } else {
            self.emoji.reclaim();
        }
    }

    /// Forget an atlas that started a new generation mid-build: its slot records,
    /// and what the build packed into it before restarting, which nothing refers to.
    pub(super) fn reset_for(&mut self, glyphs_gen: u32, emoji_gen: u32) {
        if self.glyphs.generation() != glyphs_gen {
            self.ascii_slots.clear();
            self.ascii_tables.clear();
            self.scalar_slots.clear();
            self.glyphs.clear();
        }
        if self.emoji.generation() != emoji_gen {
            self.cluster_slots.clear();
            self.cluster_records = 0;
            self.emoji.clear();
        }
    }
}
