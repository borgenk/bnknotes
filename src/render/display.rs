//! The frame as data: a flat list of drawing primitives the painter emits and a
//! backend consumes. This is the display list, and two things hang off it:
//!
//! - **Damage tracking**: because a frame is now a value, two frames can be
//!   compared. [`Damage`] diffs the previous frame against the new one and
//!   returns just the screen rectangles that changed, so an idle caret blink
//!   presents a caret-sized damaged region instead of the whole surface, and a
//!   keystroke reports the edited line rather than the screen.
//! - **A backend-agnostic frame**: the list is turned into pixels by
//!   the GPU batcher ([`crate::render::gpu::build_frame`]), which consumes exactly these
//!   commands. Nothing here is Vulkan-specific.
//!
//! Commands are in *screen* space (already shifted by the scroll), so scrolling,
//! which moves every command, naturally diffs as "everything changed" and falls
//! back to a full repaint.
//!
//! ```text
//!   layout + params ──build_display_list──▶ DisplayList ──┬─ gpu::build_frame ─▶ vertices
//!                                                         └─ Damage::diff ─────▶ [Rect]
//! ```

use std::hash::{BuildHasher, RandomState};

use crate::platform::freetype::FaceKey;
use crate::platform::geom::Rect;

/// A frame, as the ordered primitives that draw it. Stacking is the list order:
/// background first, then panels, selection, glyphs, and the caret last.
pub type DisplayList = Vec<DrawCmd>;

/// Beyond this many distinct changed regions, repainting the whole surface is
/// cheaper than tracking and clipping each one.
const MAX_DAMAGE_RECTS: usize = 16;

/// Past this many changed commands the damage is the whole surface, unmerged.
/// Merging costs up to the cube of the count, so the cap bounds it.
const MAX_MERGED_RECTS: usize = 4 * MAX_DAMAGE_RECTS;

/// Which corners a rounded rectangle rounds; the rest stay square. A shape
/// stacked from rows (a fenced code block's panel, a selection over several lines)
/// rounds only the corners on its outline, so the rows join without a notch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RoundedCorners {
    pub top_left: bool,
    pub top_right: bool,
    pub bottom_left: bool,
    pub bottom_right: bool,
}

impl RoundedCorners {
    pub const ALL: Self = Self::sides(true, true);

    /// Round the top pair, the bottom pair, both, or neither.
    pub const fn sides(top: bool, bottom: bool) -> Self {
        Self {
            top_left: top,
            top_right: top,
            bottom_left: bottom,
            bottom_right: bottom,
        }
    }
}

/// A corner of a box, top-left first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// One drawing primitive in screen (post-scroll) pixels. The set is deliberately
/// small and low level so a future GPU backend can execute the same list; the
/// painter decomposes higher-level chrome (an image's selection border, a button)
/// into these. Every variant is `Eq + Hash` so [`Damage`] can diff two frames by
/// value, which is why text runs carry an owned [`String`] and a borrow-free
/// [`FaceKey`] rather than a `&str`/`&Face` tied to the rope and fonts.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum DrawCmd {
    /// A solid, opaque rectangle (background, caret, quote bar, rule).
    Fill { rect: Rect, color: u32 },
    /// A solid rectangle with anti-aliased rounded corners (a code panel, a
    /// highlight row, a button).
    RoundRect {
        rect: Rect,
        color: u32,
        radius: i32,
        corners: RoundedCorners,
    },
    /// An inside corner of a shape stacked from rounded rows, where one row reaches
    /// past the row above or below it. The square rect sits outside both rows with
    /// `corner` touching them, and is filled outside the quarter circle centered on
    /// the opposite corner, so the rows meet in a curve.
    Fillet {
        rect: Rect,
        color: u32,
        corner: Corner,
    },
    /// A task item's check mark, inked from a distance field over `rect` (the
    /// checked box). The tick is intrinsic to the shape, so the command carries
    /// only its box and color; the GPU draws the smooth diagonals.
    CheckMark { rect: Rect, color: u32 },
    /// A file-tree disclosure chevron, inked from a distance field over `rect`.
    /// Points right when `down` is false (a collapsed folder) and down when true
    /// (expanded). Like the check mark, the shape is intrinsic, so the command
    /// carries only its box, color, and direction.
    Chevron { rect: Rect, color: u32, down: bool },
    /// A pin marker, inked from a distance field over `rect`: a round head with a
    /// needle tapering to a point below it, the badge a pinned tab wears where its
    /// close cross would sit. Like the check mark and chevron, the shape is
    /// intrinsic, so the command carries only its box and color.
    Pin { rect: Rect, color: u32 },
    /// A run of text drawn from `x` along `baseline`. `bounds` is the run's screen
    /// rectangle padded for glyph ink that overhangs the advance box, so the
    /// damage diff and the clip test never clip a glyph's edge. `clip_x`, when set,
    /// is a window x-range `[left, right)` each glyph is scissored to (adjusting its
    /// quad and atlas UVs), so a run can slide under a fixed edge, as a scrolled tab
    /// strip does; `None` draws the whole run.
    Text {
        bounds: Rect,
        x: i32,
        baseline: i32,
        face: FaceKey,
        color: u32,
        text: String,
        clip_x: Option<[i32; 2]>,
    },
    /// A cached, display-size image bitmap blitted into `rect`. `path` is the
    /// store key (the raw markdown destination); a not-yet-decoded image draws
    /// nothing.
    Image { rect: Rect, path: String },
    /// Diagonal hatching over `rect`, in `color`: parallel 45-degree strokes
    /// `stroke` device pixels wide repeating every `period` pixels. The diff view
    /// fills the empty half of a one-sided change with it, so a line present on one
    /// side reads as absent (not blank) on the other. The pattern is intrinsic to
    /// the shader mode, so the command carries only its box, color, and spacing.
    Hatch {
        rect: Rect,
        color: u32,
        period: i32,
        stroke: i32,
    },
}

impl DrawCmd {
    /// The screen rectangle this command can touch: a region intersecting it must
    /// be repainted, and the command itself is skipped when it misses the clip.
    fn bounds(&self) -> Rect {
        match self {
            DrawCmd::Fill { rect, .. }
            | DrawCmd::RoundRect { rect, .. }
            | DrawCmd::Fillet { rect, .. }
            | DrawCmd::CheckMark { rect, .. }
            | DrawCmd::Chevron { rect, .. }
            | DrawCmd::Pin { rect, .. }
            | DrawCmd::Hatch { rect, .. }
            | DrawCmd::Image { rect, .. } => *rect,
            DrawCmd::Text { bounds, .. } => *bounds,
        }
    }
}

/// The screen regions that differ between the previous frame `old` and the new
/// frame `new`, clamped to the `width` x `height` surface and free of overlaps.
///
/// A multiset difference does the work: a command appearing an unequal number of
/// times in the two frames contributes its bounds, so an identical command (same
/// geometry, face, color, text) cancels and costs nothing, while a moved or
/// restyled run shows up as a removal plus an addition and damages both spots.
/// A frame that changed most of the surface (a scroll, a resize, a large edit)
/// returns a single full-surface rectangle, since repainting whole then beats
/// clipping many regions. An empty result means the two frames are identical.
/// The screen regions that changed between two frames. It keeps its buffers
/// across frames, so a frame that changed allocates nothing once they have
/// grown, a caret blink included.
#[derive(Default)]
pub struct Damage {
    hasher: RandomState,
    /// The changed commands as (hash, index): an index below the old span's
    /// length names an old command, the rest a new one.
    keys: Vec<(u64, usize)>,
    /// Which of a hash group's new commands an old one already cancelled.
    matched: Vec<bool>,
    rects: Vec<Rect>,
}

impl Damage {
    /// Diff old against new on a width x height surface: the regions to repaint,
    /// none when the frames draw the same. A command present in both frames
    /// cancels wherever it sits in the lists, so only what appeared, vanished
    /// or moved is damage.
    pub fn diff(&mut self, old: &[DrawCmd], new: &[DrawCmd], width: i32, height: i32) -> &[Rect] {
        // Dropping an equal pair never changes the difference, so strip the
        // common prefix and suffix first. Lists are built in document order, so
        // a blink or a small edit leaves a handful of commands to hash; equality
        // bails on the first differing field, cheaper than hashing.
        let prefix = old.iter().zip(new).take_while(|(a, b)| a == b).count();
        let (old, new) = (&old[prefix..], &new[prefix..]);
        let suffix = old
            .iter()
            .rev()
            .zip(new.iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let (old, new) = (&old[..old.len() - suffix], &new[..new.len() - suffix]);

        let Damage {
            hasher,
            keys,
            matched,
            rects,
        } = self;
        let surface = Rect {
            x: 0,
            y: 0,
            w: width.max(0),
            h: height.max(0),
        };
        let mut push = |c: &DrawCmd| rects.extend(intersection(c.bounds(), surface));

        // Sorting by hash puts equal commands side by side, old ones first in
        // each group. Within a group each old command cancels one equal new one,
        // and whatever is left over changed.
        keys.clear();
        keys.extend(old.iter().map(|c| hasher.hash_one(c)).zip(0..));
        keys.extend(new.iter().map(|c| hasher.hash_one(c)).zip(old.len()..));
        keys.sort_unstable();
        for group in keys.chunk_by(|a, b| a.0 == b.0) {
            let (olds, news) = group.split_at(group.partition_point(|&(_, i)| i < old.len()));
            let new_at = |i: usize| new.get(i - old.len());
            matched.clear();
            matched.resize(news.len(), false);
            for &(_, i) in olds {
                let Some(gone) = old.get(i) else { continue };
                let twin = news
                    .iter()
                    .zip(matched.iter_mut())
                    .find(|((_, j), done)| !**done && new_at(*j) == Some(gone));
                match twin {
                    Some((_, done)) => *done = true,
                    None => push(gone),
                }
            }
            for (&(_, j), done) in news.iter().zip(matched.iter()) {
                if let (false, Some(came)) = (*done, new_at(j)) {
                    push(came);
                }
            }
        }

        let surface_area = surface.w as i64 * surface.h as i64;
        if rects.len() <= MAX_MERGED_RECTS {
            coalesce(rects);
        }
        let area: i64 = rects.iter().map(|r| r.w as i64 * r.h as i64).sum();
        if rects.len() > MAX_DAMAGE_RECTS || area * 2 > surface_area {
            rects.clear();
            if surface_area > 0 {
                rects.push(surface);
            }
        }
        rects
    }

    /// The regions the last diff found.
    pub fn rects(&self) -> &[Rect] {
        &self.rects
    }
}

/// Merge any two overlapping rectangles into their bounding box, repeated until
/// none overlap, so a region is never cleared and repainted twice. The damage set
/// is tiny in practice, so the quadratic scan never matters.
fn coalesce(rects: &mut Vec<Rect>) {
    let mut merged = true;
    while merged {
        merged = false;
        'scan: for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                if intersects(rects[i], rects[j]) {
                    rects[i] = union(rects[i], rects[j]);
                    rects.remove(j);
                    merged = true;
                    break 'scan;
                }
            }
        }
    }
}

/// Whether two rectangles share any interior pixel (touching edges do not count):
/// exactly when their [`intersection`] is non-empty.
fn intersects(a: Rect, b: Rect) -> bool {
    intersection(a, b).is_some()
}

/// The smallest rectangle covering both `a` and `b`.
fn union(a: Rect, b: Rect) -> Rect {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 = (a.x + a.w).max(b.x + b.w);
    let y1 = (a.y + a.h).max(b.y + b.h);
    Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    }
}

/// The overlap of `a` and `b`, or `None` when they are disjoint.
fn intersection(a: Rect, b: Rect) -> Option<Rect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    (x0 < x1 && y0 < y1).then_some(Rect {
        x: x0,
        y: y0,
        w: x1 - x0,
        h: y1 - y0,
    })
}

#[cfg(test)]
mod tests {
    use crate::render::display::*;

    /// One diff on fresh buffers.
    fn damage(old: &[DrawCmd], new: &[DrawCmd], width: i32, height: i32) -> Vec<Rect> {
        Damage::default().diff(old, new, width, height).to_vec()
    }

    fn fill(x: i32, y: i32, w: i32, h: i32, color: u32) -> DrawCmd {
        DrawCmd::Fill {
            rect: Rect { x, y, w, h },
            color,
        }
    }

    const W: i32 = 800;
    const H: i32 = 600;

    #[test]
    fn identical_frames_have_no_damage() {
        let frame = vec![fill(0, 0, W, H, 0x010101), fill(10, 10, 5, 5, 0xffffff)];
        assert!(damage(&frame, &frame, W, H).is_empty());
    }

    #[test]
    fn a_caret_toggle_damages_only_the_caret() {
        // The blink case: the frames differ by one small Fill (the caret).
        let bg = fill(0, 0, W, H, 0x010101);
        let caret = fill(100, 50, 2, 18, 0xffffff);
        let off = vec![bg.clone()];
        let on = vec![bg, caret];
        assert_eq!(
            damage(&off, &on, W, H),
            vec![Rect {
                x: 100,
                y: 50,
                w: 2,
                h: 18
            }],
            "only the caret rectangle is damaged"
        );
    }

    #[test]
    fn a_moved_command_damages_both_positions() {
        // A run that shifts (an edit before it) appears as a remove plus an add;
        // the two non-overlapping spots both need repainting.
        let bg = fill(0, 0, W, H, 0x010101);
        let before = vec![bg.clone(), fill(10, 10, 4, 4, 0xaa)];
        let after = vec![bg, fill(500, 400, 4, 4, 0xaa)];
        let mut d = damage(&before, &after, W, H);
        d.sort_by_key(|r| r.x);
        assert_eq!(
            d,
            vec![
                Rect {
                    x: 10,
                    y: 10,
                    w: 4,
                    h: 4
                },
                Rect {
                    x: 500,
                    y: 400,
                    w: 4,
                    h: 4
                },
            ]
        );
    }

    #[test]
    fn a_repeated_command_cancels_one_copy_at_a_time() {
        // Two identical fills where one remains: the copy that vanished still
        // needs repainting. The changed ends keep both copies out of the common
        // prefix and suffix.
        let bg = fill(0, 0, W, H, 0x010101);
        let dot = fill(40, 40, 6, 6, 0xffffff);
        let (first, last) = ((0, 0, 1, 1), (300, 300, 2, 2));
        let cmd = |(x, y, w, h), color| fill(x, y, w, h, color);
        let old = vec![
            bg.clone(),
            cmd(first, 1),
            dot.clone(),
            dot.clone(),
            cmd(last, 1),
        ];
        let new = vec![bg, cmd(first, 2), dot, cmd(last, 2)];
        let mut d = damage(&old, &new, W, H);
        d.sort_by_key(|r| r.x);
        let rect = |(x, y, w, h)| Rect { x, y, w, h };
        assert_eq!(d, vec![rect(first), rect((40, 40, 6, 6)), rect(last)]);
    }

    #[test]
    fn a_command_that_only_moved_in_the_list_is_no_damage() {
        // The same commands in another order draw the same pixels where they do
        // not overlap.
        let bg = fill(0, 0, W, H, 0x010101);
        let (a, b, c) = (
            fill(10, 10, 4, 4, 1),
            fill(100, 10, 4, 4, 2),
            fill(200, 10, 4, 4, 3),
        );
        let old = vec![bg.clone(), a.clone(), b.clone(), c.clone()];
        let new = vec![bg, c, b, a];
        assert!(damage(&old, &new, W, H).is_empty());
    }

    #[test]
    fn overlapping_changes_coalesce_into_one_region() {
        let bg = fill(0, 0, W, H, 0x010101);
        let old = vec![bg.clone()];
        let new = vec![bg, fill(10, 10, 20, 20, 0x01), fill(20, 20, 20, 20, 0x02)];
        // The two added fills overlap, so the damage is their bounding box.
        assert_eq!(
            damage(&old, &new, W, H),
            vec![Rect {
                x: 10,
                y: 10,
                w: 30,
                h: 30
            }]
        );
    }

    #[test]
    fn a_large_change_falls_back_to_a_full_repaint() {
        // Two big disjoint fills cover more than half the surface, so the diff
        // collapses to one whole-surface rectangle rather than tracking regions.
        let old: Vec<DrawCmd> = Vec::new();
        let new = vec![fill(0, 0, W, H / 2 + 10, 0x01), fill(0, H / 2, W, H, 0x02)];
        assert_eq!(
            damage(&old, &new, W, H),
            vec![Rect {
                x: 0,
                y: 0,
                w: W,
                h: H
            }]
        );
    }

    #[test]
    fn damage_is_clamped_to_the_surface() {
        // A command partly off-screen damages only its on-surface part; one fully
        // off-screen contributes nothing.
        let on = fill(-5, -5, 20, 20, 0x01);
        let off = fill(W + 100, 0, 10, 10, 0x02);
        let new = vec![on, off];
        assert_eq!(
            damage(&[], &new, W, H),
            vec![Rect {
                x: 0,
                y: 0,
                w: 15,
                h: 15
            }]
        );
    }
}
