use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::app::State;
use crate::platform::conn::Connection;
use crate::platform::dmabuf;
use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::protocol::{
    self, wl_buffer, wl_display, wl_surface, wp_linux_drm_syncobj_manager_v1,
    wp_linux_drm_syncobj_surface_v1, wp_viewport, xdg_surface, zwp_linux_buffer_params_v1,
    zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use crate::platform::wire::{Arg, Reader};
use crate::render::display::{self, DisplayList};
use crate::render::glyph_cache::{self, GlyphCache};
use crate::render::gpu;
use crate::render::vulkan;
use crate::theme::BACKGROUND;

/// How soon a frame waiting for a free buffer under explicit sync looks again. The
/// wait doubles with each miss up to SLOT_RETRY_MAX, so a compositor that holds
/// both buffers for long costs few wakeups.
const SLOT_RETRY: Duration = Duration::from_millis(1);
const SLOT_RETRY_MAX: Duration = Duration::from_millis(32);

/// Client-owned explicit-sync state (linux-drm-syncobj-v1), negotiated over a
/// live GPU backend when the compositor advertises the manager. DRM syncobj
/// timelines the compositor shares: the client signals acquire with each frame's
/// render-done fence, and the compositor signals a buffer slot's release timeline
/// when done reading that buffer. A scratch binary syncobj carries the sync-file
/// transfers, and a binary syncobj created signaled is the acquire fence of a
/// frame the CPU already waited out. point increments per present and numbers
/// both of a frame's points. Dropping this closes the render-node fd, which frees
/// every syncobj created on it.
pub(super) struct ExplicitSync {
    drm: OwnedFd,
    acquire_syncobj: u32,
    scratch: u32,
    signaled: u32,
    acquire_timeline: u32,
    release: [ReleaseTimeline; 2],
    surface: u32,
    point: u64,
}

/// One buffer slot's release timeline. Each slot has its own because signaling
/// a timeline point signals every earlier point too, and a compositor may release
/// buffers in a different order than they were committed.
#[derive(Default)]
struct ReleaseTimeline {
    syncobj: u32,
    /// The protocol object the compositor imported syncobj as.
    timeline: u32,
    /// The point last published for the slot's buffer, which its next reuse
    /// waits on. None before the slot's first frame and after a resize.
    point: Option<u64>,
}

impl ExplicitSync {
    /// Create the client's syncobjs on drm. Their protocol objects stay zero until
    /// State::negotiate_explicit_sync shares them with the compositor.
    fn new(drm: OwnedFd) -> Result<Self> {
        let fd = drm.as_raw_fd();
        let mut release = [ReleaseTimeline::default(), ReleaseTimeline::default()];
        for slot in &mut release {
            slot.syncobj = ffi::drm_syncobj_create(fd, false)?;
        }
        Ok(Self {
            acquire_syncobj: ffi::drm_syncobj_create(fd, false)?,
            scratch: ffi::drm_syncobj_create(fd, false)?,
            signaled: ffi::drm_syncobj_create(fd, true)?,
            drm,
            acquire_timeline: 0,
            release,
            surface: 0,
            point: 0,
        })
    }

    /// The sync file to wait on before rendering into buffer idx: the
    /// compositor's release of that buffer's previous frame, or None the first
    /// time the slot is used (nothing has read it yet). free_slot picks a buffer
    /// only once its point holds a fence, so the wait for one, up to the kernel's
    /// bound, is a safety net rather than the usual path.
    fn release_fence(&self, idx: usize) -> Result<Option<OwnedFd>> {
        let slot = self
            .release
            .get(idx)
            .ok_or_else(|| Error::msg(format!("no buffer slot {idx}")))?;
        match slot.point {
            Some(p) => Ok(Some(ffi::drm_syncobj_point_to_sync_file(
                self.drm.as_raw_fd(),
                slot.syncobj,
                p,
                self.scratch,
                true,
            )?)),
            None => Ok(None),
        }
    }

    /// Put this frame's acquire fence at a fresh point and claim the same point
    /// on buffer idx's release timeline, returning the point and that timeline's
    /// protocol object. A fence of None is a frame the CPU already waited out,
    /// whose acquire point goes in already signaled.
    fn claim(&mut self, fence: Option<&OwnedFd>, idx: usize) -> Result<(u64, u32)> {
        let slot = self
            .release
            .get_mut(idx)
            .ok_or_else(|| Error::msg(format!("no buffer slot {idx}")))?;
        self.point += 1;
        let p = self.point;
        let drm = self.drm.as_raw_fd();
        match fence {
            Some(fence) => ffi::drm_syncobj_sync_file_to_point(
                drm,
                fence.as_raw_fd(),
                self.acquire_syncobj,
                p,
                self.scratch,
            )?,
            None => ffi::drm_syncobj_signal_point(drm, self.acquire_syncobj, p, self.signaled)?,
        }
        slot.point = Some(p);
        Ok((p, slot.timeline))
    }

    /// Claim buffer idx's points and set both on the surface. Every commit that
    /// attaches a buffer needs both: one without them is the no_acquire_point or
    /// no_release_point protocol error, which closes the connection.
    fn publish(
        &mut self,
        conn: &mut Connection,
        fence: Option<&OwnedFd>,
        idx: usize,
    ) -> Result<()> {
        let (p, release_timeline) = self.claim(fence, idx)?;
        let (hi, lo) = ((p >> 32) as u32, p as u32);
        conn.request(
            self.surface,
            wp_linux_drm_syncobj_surface_v1::SET_ACQUIRE_POINT,
            &[
                Arg::Object(self.acquire_timeline),
                Arg::Uint(hi),
                Arg::Uint(lo),
            ],
        );
        conn.request(
            self.surface,
            wp_linux_drm_syncobj_surface_v1::SET_RELEASE_POINT,
            &[Arg::Object(release_timeline), Arg::Uint(hi), Arg::Uint(lo)],
        );
        Ok(())
    }

    /// The buffer to render the next frame into, or None while both are still the
    /// compositor's. Under explicit sync wl_buffer.release is undefined, so a
    /// buffer is free once it was never committed or the compositor has put a
    /// fence on its release point, which the render waits on. The buffer committed
    /// longest ago is tried first, so the one on screen is the last choice.
    fn free_slot(&self) -> Result<Option<usize>> {
        let [a, b] = &self.release;
        let mut order = [(a.point, 0, a.syncobj), (b.point, 1, b.syncobj)];
        order.sort_unstable();
        for (point, idx, syncobj) in order {
            let free = match point {
                None => true,
                Some(p) => ffi::drm_syncobj_point_available(self.drm.as_raw_fd(), syncobj, p)?,
            };
            if free {
                return Ok(Some(idx));
            }
        }
        Ok(None)
    }

    /// Forget every buffer's published release point (the buffers they named are
    /// gone, e.g. after a resize), so the next use of each slot waits on nothing.
    fn forget_buffers(&mut self) {
        for slot in &mut self.release {
            slot.point = None;
        }
    }

    /// The syncobj behind the release timeline the compositor knows as timeline.
    #[cfg(test)]
    fn release_syncobj_of(&self, timeline: u32) -> Option<u32> {
        self.release
            .iter()
            .find(|slot| slot.timeline == timeline)
            .map(|slot| slot.syncobj)
    }
}

/// GPU/dmabuf presentation resources owned by the app: protocol globals and
/// feedback, the Vulkan backend, the two exported swap buffers, explicit-sync
/// state, and the small counters used by the stats line.
pub(super) struct GpuPresentation {
    pub(super) dmabuf: Option<u32>,
    pub(super) dmabuf_version: u32,
    pub(super) feedback_id: u32,
    pub(super) feedback: dmabuf::FeedbackState,
    pub(super) backend: Option<vulkan::Gpu>,
    pub(super) images: Vec<vulkan::GpuImage>,
    pub(super) modifiers: Vec<u64>,
    pub(super) glyphs: GlyphCache,
    /// The frame the batcher fills, kept across frames so its vertex and batch
    /// buffers are refilled rather than reallocated every present.
    pub(super) frame: gpu::FrameData,
    pub(super) syncobj_manager: Option<u32>,
    pub(super) explicit_sync: Option<ExplicitSync>,
    pub(super) buffers: [u32; 2],
    /// Buffers the compositor holds, until its wl_buffer.release. Read only
    /// without explicit sync, where that event is defined.
    pub(super) busy: [bool; 2],
    /// When to look for a free buffer again, set while a frame waits for one
    /// under explicit sync: no event says when a release point gets its fence.
    pub(super) slot_retry_at: Option<Instant>,
    /// The wait before the next look, growing while no buffer frees.
    slot_retry: Duration,
    pub(super) buffer_size: (u32, u32),
    pub(super) present_list: Rc<DisplayList>,
    /// The regions the frame being presented changed.
    pub(super) damage: display::Damage,
    pub(super) frame_count: u64,
    pub(super) last_present: Option<Instant>,
    pub(super) explicit_fence_frames: u64,
    pub(super) explicit_cpu_wait_frames: u64,
}

impl GpuPresentation {
    pub(super) fn new() -> Self {
        Self {
            dmabuf: None,
            dmabuf_version: 0,
            feedback_id: 0,
            feedback: dmabuf::FeedbackState::default(),
            backend: None,
            images: Vec::new(),
            modifiers: Vec::new(),
            glyphs: GlyphCache::new(),
            frame: gpu::FrameData::default(),
            syncobj_manager: None,
            explicit_sync: None,
            buffers: [0, 0],
            busy: [false, false],
            slot_retry_at: None,
            slot_retry: SLOT_RETRY,
            buffer_size: (0, 0),
            present_list: Rc::default(),
            damage: display::Damage::default(),
            frame_count: 0,
            last_present: None,
            explicit_fence_frames: 0,
            explicit_cpu_wait_frames: 0,
        }
    }
}

impl State {
    /// Bring the Vulkan backend up from the committed dmabuf feedback: a
    /// device matching the compositor's, and a modifier list their common
    /// ground. Called before `create_buffers`, which allocates the exportable
    /// GPU images. A failure here is fatal (the editor requires a GPU).
    pub(super) fn init_gpu(&mut self) -> Result<()> {
        if self.presentation.dmabuf.is_none() {
            return Err(Error::msg("compositor lacks zwp_linux_dmabuf_v1 v4+"));
        }
        let fb = self
            .presentation
            .feedback
            .ready()
            .ok_or_else(|| Error::msg("dmabuf feedback never committed"))?;
        let main_device = fb.main_device;
        let compositor_mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if compositor_mods.is_empty() {
            return Err(Error::msg("compositor does not accept XRGB8888 dmabufs"));
        }
        let gpu = vulkan::Gpu::new(main_device)?;
        let mods = gpu.image_modifiers(&compositor_mods);
        if mods.is_empty() {
            return Err(Error::msg(
                "no common DRM modifier between compositor and device",
            ));
        }
        self.presentation.modifiers = mods;
        self.presentation.backend = Some(gpu);
        Ok(())
    }

    /// Negotiate explicit sync (`linux-drm-syncobj-v1`) when a GPU backend is
    /// live and the compositor advertises the manager. Best-effort: on any
    /// failure the GPU path keeps the implicit-sync bridge, which is why this
    /// returns nothing and only logs under stats.
    pub(super) fn init_explicit_sync(&mut self) {
        let Some(manager) = self.presentation.syncobj_manager else {
            return;
        };
        let Some(minor) = self
            .presentation
            .backend
            .as_ref()
            .and_then(vulkan::Gpu::render_node)
        else {
            return; // No GPU backend, or the device exposes no render node.
        };
        match self.negotiate_explicit_sync(manager, minor) {
            Ok(es) => self.presentation.explicit_sync = Some(es),
            Err(e) => {
                if self.stats {
                    eprintln!("bnknotes: explicit sync unavailable, using the sync bridge: {e}");
                }
            }
        }
    }

    /// Create the client-owned syncobjs, share the timelines with the compositor,
    /// and attach explicit sync to the surface.
    pub(super) fn negotiate_explicit_sync(
        &mut self,
        manager: u32,
        minor: u32,
    ) -> Result<ExplicitSync> {
        let mut es = ExplicitSync::new(ffi::open_drm_render_node(minor)?)?;
        let fd = es.drm.as_raw_fd();
        es.acquire_timeline = self.import_timeline(manager, fd, es.acquire_syncobj)?;
        for slot in &mut es.release {
            slot.timeline = self.import_timeline(manager, fd, slot.syncobj)?;
        }
        es.surface = self.alloc_id();
        self.conn.request(
            manager,
            wp_linux_drm_syncobj_manager_v1::GET_SURFACE,
            &[Arg::NewId(es.surface), Arg::Object(self.surface)],
        );
        Ok(es)
    }

    /// Export `syncobj` as a DRM syncobj fd and import it into the compositor as
    /// a timeline object; returns the protocol object id. The fd is sent inline,
    /// so the local `OwnedFd` can drop on return.
    pub(super) fn import_timeline(
        &mut self,
        manager: u32,
        drm: RawFd,
        syncobj: u32,
    ) -> Result<u32> {
        let fd = ffi::drm_syncobj_export_fd(drm, syncobj)?;
        let id = self.alloc_id();
        self.conn.request_with_fd(
            manager,
            wp_linux_drm_syncobj_manager_v1::IMPORT_TIMELINE,
            &[Arg::NewId(id)],
            fd.as_raw_fd(),
        )?;
        Ok(id)
    }

    /// Allocate the two presentable buffers at the current size: one exported
    /// GPU image per swap slot, each wrapped in a wl_buffer via
    /// zwp_linux_buffer_params_v1 (create_immed is sound because every parameter
    /// comes from the compositor's own feedback). On a resize the images are
    /// rebuilt in place and keep their submission state. A failure (a resize on
    /// a dying device) propagates and ends the process; there is no software
    /// path to fall back to.
    pub(super) fn create_buffers(&mut self) -> Result<()> {
        let dmabuf_global = self
            .presentation
            .dmabuf
            .ok_or_else(|| Error::msg("gpu buffers without zwp_linux_dmabuf_v1"))?;
        let (w, h) = (self.width, self.height);
        self.destroy_buffers();
        {
            let Some(gpu) = self.presentation.backend.as_ref() else {
                return Err(Error::msg("gpu buffers without a gpu backend"));
            };
            let modifiers = &self.presentation.modifiers;
            let images = &mut self.presentation.images;
            if images.is_empty() {
                for _ in 0..2 {
                    images.push(gpu.create_image(w, h, modifiers)?);
                }
            } else {
                // Nothing may be in flight while the images are rebuilt; resize
                // is rare enough that a full drain is the simple correct answer.
                gpu.wait_idle();
                for img in images.iter_mut() {
                    gpu.resize_image(img, w, h, modifiers)?;
                }
            }
        }
        for i in 0..self.presentation.images.len() {
            let Some(img) = self.presentation.images.get(i) else {
                break;
            };
            let (fd, offset, stride, modifier) =
                (img.fd(), img.offset(), img.stride(), img.modifier());
            let params = self.alloc_id();
            self.conn.request(
                dmabuf_global,
                zwp_linux_dmabuf_v1::CREATE_PARAMS,
                &[Arg::NewId(params)],
            );
            self.conn.request_with_fd(
                params,
                zwp_linux_buffer_params_v1::ADD,
                &[
                    Arg::Uint(0), // plane index
                    Arg::Uint(offset),
                    Arg::Uint(stride),
                    Arg::Uint((modifier >> 32) as u32),
                    Arg::Uint(modifier as u32),
                ],
                fd,
            )?;
            let buffer = self.alloc_id();
            self.conn.request(
                params,
                zwp_linux_buffer_params_v1::CREATE_IMMED,
                &[
                    Arg::NewId(buffer),
                    Arg::Int(w as i32),
                    Arg::Int(h as i32),
                    Arg::Uint(dmabuf::DRM_FORMAT_XRGB8888),
                    Arg::Uint(0),
                ],
            );
            self.conn
                .request(params, zwp_linux_buffer_params_v1::DESTROY, &[]);
            self.presentation.buffers[i] = buffer;
        }
        self.presentation.buffer_size = (self.width, self.height);
        self.presentation.busy = [false, false];
        // New images never carried a release point, so a resize must forget the
        // old buffers' points; their next use starts with no wait.
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.forget_buffers();
        }
        // New images hold no known frame, so forget what was on screen: the
        // screen diff gates whether the next frame presents at all.
        self.presentation.present_list = Rc::default();
        Ok(())
    }

    /// Destroy the current wl_buffers; the GPU images stay for create_buffers to
    /// resize. The server recycles the ids via delete_id into the free list.
    fn destroy_buffers(&mut self) {
        let buffers = self.presentation.buffers;
        for b in buffers {
            if b != 0 {
                self.conn.request(b, wl_buffer::DESTROY, &[]);
            }
        }
        self.presentation.buffers = [0, 0];
        self.presentation.busy = [false, false];
    }

    /// Record a new surface size `(w, h)`. The buffers are NOT reallocated here:
    /// an interactive resize fires many configures, and reallocating exportable
    /// dmabuf images on each one is heavy churn (a full GPU drain plus an
    /// export/import round-trip), enough to upset a compositor's buffer path.
    /// `render_frame` reallocates lazily when `buffer_size` no longer matches, so
    /// the churn collapses to one reallocation per painted frame. A no-op if the
    /// size is unchanged.
    pub(super) fn resize_to(&mut self, w: u32, h: u32) {
        if (w, h) == (self.width, self.height) {
            return;
        }
        self.width = w;
        self.height = h;
        // A width change re-wraps and a height change shrinks the viewport; either
        // can push the caret off-screen. Re-reveal the focused pane's caret (this
        // rebuilds its layout at the new column width and re-clamps the scroll).
        self.reveal_active_caret();
        self.dirty = true;
    }

    /// Declare the surface's scale state, then ack configure `serial`. Sending the
    /// geometry with the ack and the matching buffer gives the compositor one
    /// atomic, consistent size for the frame, so an anchored (left/top) resize edge
    /// does not drift. Caller commits after this.
    pub(super) fn ack_configure(&mut self, serial: u32) {
        self.declare_surface_scale();
        self.conn.request(
            self.xdg_surface,
            xdg_surface::ACK_CONFIGURE,
            &[Arg::Uint(serial)],
        );
    }

    /// (Re)declare the surface's logical geometry and how the device-pixel buffer
    /// maps onto it: a viewport destination on the fractional path, or an integer
    /// buffer scale on the fallback. The geometry is the whole logical surface (no
    /// client-side decorations), the double-buffered state the compositor anchors a
    /// resize's fixed edge to. Idempotent surface state; marks the scale synced.
    pub(super) fn declare_surface_scale(&mut self) {
        let (lw, lh) = (self.scaling.logical.0 as i32, self.scaling.logical.1 as i32);
        self.conn.request(
            self.xdg_surface,
            xdg_surface::SET_WINDOW_GEOMETRY,
            &[Arg::Int(0), Arg::Int(0), Arg::Int(lw), Arg::Int(lh)],
        );
        if self.scaling.is_fractional() {
            self.conn.request(
                self.scaling.viewport,
                wp_viewport::SET_DESTINATION,
                &[Arg::Int(lw), Arg::Int(lh)],
            );
        } else {
            let n = (self.scaling.factor_120 / 120).max(1) as i32;
            self.conn
                .request(self.surface, wl_surface::SET_BUFFER_SCALE, &[Arg::Int(n)]);
        }
        self.scaling.synced = true;
    }

    /// Build the current frame and present it on the GPU. Returns false if both
    /// buffers are still held by the compositor, leaving dirty set so the frame is
    /// tried again: at the next wl_buffer.release, or under explicit sync at
    /// slot_retry_at.
    ///
    /// The display list is built and diffed against `present_list` (what is on
    /// screen): an identical frame early-outs (the common idle case), otherwise
    /// the changed screen regions are reported to the compositor as damage. The
    /// per-buffer diff the software path used is gone: the GPU repaints the full
    /// frame every present.
    ///
    /// A rendering failure (a device lost mid-session) propagates and ends the
    /// process; there is no software path to fall back to.
    pub(super) fn render_frame(&mut self) -> Result<bool> {
        // Reallocate the buffers if the surface was resized since they were made.
        // Deferring this from every configure to the first frame after them
        // collapses an interactive resize's dmabuf churn to one reallocation per
        // painted frame (frame-callback paced), instead of one per configure.
        if self.presentation.buffer_size != (self.width, self.height) {
            self.create_buffers()?;
        }
        let free = match self.presentation.explicit_sync.as_ref() {
            Some(es) => es.free_slot()?,
            None => (0..2).find(|&i| !self.presentation.busy[i]),
        };
        self.presentation.slot_retry_at = match free {
            None if self.presentation.explicit_sync.is_some() => {
                let wait = self.presentation.slot_retry;
                self.presentation.slot_retry = (wait * 2).min(SLOT_RETRY_MAX);
                Some(Instant::now() + wait)
            }
            _ => {
                self.presentation.slot_retry = SLOT_RETRY;
                None
            }
        };
        let Some(idx) = free else {
            return Ok(false);
        };
        let started = self.stats.then(Instant::now);
        let (sw, sh) = (self.width as i32, self.height as i32);
        // Compose the frame from every pane (tree, editors) plus the splitters.
        let new_list = self.build_frame_list();

        // Nothing changed since the on-screen frame: nothing to present (the
        // common idle case).
        let changed = !self
            .presentation
            .damage
            .diff(&self.presentation.present_list, &new_list, sw, sh)
            .is_empty();
        // Acknowledge the latest configure paired with this commit, so the buffer
        // the compositor sees is always the one sized to the configure it just
        // acked; that is what keeps an anchored (left/top) resize edge from
        // jumping. Taken here so it rides the same flush as the commit below.
        let ack = self.pending_configure.take();
        if !changed {
            // Nothing new to draw, but a pending configure must still be acked so
            // the compositor can finalize a state-only change (e.g. a maximize
            // that altered no pixels); a bare commit applies it against the
            // current, already correctly sized buffer.
            if let Some(serial) = ack {
                self.ack_configure(serial);
                self.conn.request(self.surface, wl_surface::COMMIT, &[]);
            }
            return Ok(true);
        }
        if let Some(serial) = ack {
            self.ack_configure(serial);
        }
        // A scale change can arrive without a configure (a monitor move), so make
        // sure the geometry + viewport/buffer-scale ride this commit too.
        if !self.scaling.synced {
            self.declare_surface_scale();
        }

        // Render, returning the render-done fence under explicit sync (`None`
        // on the bridge path, or when the driver could not export one). Inline
        // images resolve against both panes' stores, so a picture renders in
        // whichever pane it belongs to.
        // The chips' baked icons resolve here too: an icon is a picture the editor
        // made rather than read, but the renderer draws it like any other.
        let images = [
            &self.editors[0].images,
            &self.editors[1].images,
            &self.icons,
        ];
        let glyph_gamma = self.glyph_gamma;
        let render_done = {
            let Some(gpu) = self.presentation.backend.as_mut() else {
                return Err(Error::msg("gpu frame without a gpu backend"));
            };
            let Some(img) = self.presentation.images.get_mut(idx) else {
                return Err(Error::msg("gpu frame without gpu buffers"));
            };
            let known = |id: glyph_cache::ImageId| gpu.has_image(id);
            // Built into the frame buffer the presentation keeps, so the vertices and
            // batches reuse last frame's capacity rather than regrowing from zero.
            let frame = &mut self.presentation.frame;
            gpu::build_frame(
                &self.fonts,
                &images,
                &new_list,
                &mut self.presentation.glyphs,
                &known,
                self.code_glyph_gamma,
                frame,
            );
            match self.presentation.explicit_sync.as_ref() {
                Some(es) => {
                    let release_wait = es.release_fence(idx)?;
                    gpu.render_list_explicit(
                        img,
                        frame,
                        color_f32(BACKGROUND),
                        glyph_gamma,
                        release_wait,
                    )?
                }
                None => {
                    gpu.render_list(img, frame, color_f32(BACKGROUND), glyph_gamma)?;
                    None
                }
            }
        };
        self.presentation.present_list = new_list;

        let buffer = self.presentation.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Explicit sync: set this frame's acquire point and claim a release point,
        // so the compositor waits before reading and signals when done (the
        // buffer's next reuse waits on it). A frame with no exportable fence was
        // CPU-waited, and its acquire point goes out already signaled.
        if let Some(es) = self.presentation.explicit_sync.as_mut() {
            es.publish(&mut self.conn, render_done.as_ref(), idx)?;
            if render_done.is_some() {
                self.presentation.explicit_fence_frames += 1;
            } else {
                self.presentation.explicit_cpu_wait_frames += 1;
            }
        }
        // The GPU repainted everything, but the screen change is still just the
        // diff, so report those regions and let the compositor recomposite only
        // what moved. The diff is in buffer pixels, hence damage_buffer.
        for r in self.presentation.damage.rects() {
            self.conn.request(
                self.surface,
                wl_surface::DAMAGE_BUFFER,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
        let callback = self.alloc_id();
        self.conn
            .request(self.surface, wl_surface::FRAME, &[Arg::NewId(callback)]);
        self.frame_callback = callback;
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.presentation.busy[idx] = true;

        if let Some(started) = started {
            self.log_frame_stats(started);
            self.presentation.last_present = Some(started);
        }
        self.presentation.frame_count += 1;
        Ok(true)
    }

    /// Print the per-frame stats line under `BNKNOTES_STATS`: the build+submit
    /// time, the gap since the last present, the glyph-cache hit rate, and which
    /// sync strategy (explicit points or the implicit bridge) carried the frame.
    fn log_frame_stats(&self, started: Instant) {
        let cache = self.presentation.glyphs.stats();
        let gap = self
            .presentation
            .last_present
            .map(|t| started.duration_since(t).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let sync = if self.presentation.explicit_sync.is_some() {
            format!(
                "explicit-sync {} fence/{} cpu-wait",
                self.presentation.explicit_fence_frames, self.presentation.explicit_cpu_wait_frames
            )
        } else {
            "implicit-sync bridge".to_string()
        };
        eprintln!(
            "[stats] gpu frame {} {:.2}ms build+submit, {:.1}ms since last | glyphs hit {} miss {} ({:.1}% of {}) | {}",
            self.presentation.frame_count,
            started.elapsed().as_secs_f64() * 1000.0,
            gap,
            cache.hits,
            cache.misses,
            cache.hit_rate() * 100.0,
            cache.lookups(),
            sync,
        );
    }

    /// Ask the dmabuf global for its default feedback. The events accumulate in
    /// `dmabuf_fb` as they arrive; nothing waits on them.
    pub(super) fn request_dmabuf_feedback(&mut self) {
        let Some(dmabuf) = self.presentation.dmabuf else {
            return;
        };
        let id = self.alloc_id();
        self.conn.request(
            dmabuf,
            zwp_linux_dmabuf_v1::GET_DEFAULT_FEEDBACK,
            &[Arg::NewId(id)],
        );
        self.presentation.feedback_id = id;
    }

    /// One event of a feedback burst; `dmabuf_fb` accumulates and commits.
    pub(super) fn on_dmabuf_feedback(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            zwp_linux_dmabuf_feedback_v1::EV_FORMAT_TABLE => {
                let size = r.u32()? as usize;
                let fd = self.conn.take_fd().ok_or_else(|| {
                    Error::msg("dmabuf feedback format_table arrived without its fd")
                })?;
                if size == 0 {
                    self.presentation.feedback.set_table(&[]);
                    return Ok(());
                }
                let bytes = ffi::read_mapped(fd.as_raw_fd(), size)?;
                self.presentation.feedback.set_table(&bytes);
            }
            zwp_linux_dmabuf_feedback_v1::EV_MAIN_DEVICE => {
                self.presentation.feedback.main_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_TARGET_DEVICE => {
                self.presentation.feedback.tranche_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FORMATS => {
                self.presentation.feedback.tranche_formats(r.array()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FLAGS => {
                self.presentation.feedback.tranche_flags(r.u32()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_DONE => {
                self.presentation.feedback.tranche_done()
            }
            zwp_linux_dmabuf_feedback_v1::EV_DONE => self.presentation.feedback.done(),
            _ => {}
        }
        Ok(())
    }

    /// The `--gpu-probe` body: bind the registry, fetch dmabuf feedback, and
    /// print what the GPU presentation path has to work with.
    pub(super) fn probe_dmabuf(&mut self) -> Result<()> {
        self.registry = self.alloc_id();
        self.conn.request(
            protocol::WL_DISPLAY,
            wl_display::GET_REGISTRY,
            &[Arg::NewId(self.registry)],
        );
        self.roundtrip()?;
        if self.presentation.dmabuf.is_none() {
            println!(
                "gpu-probe: compositor lacks zwp_linux_dmabuf_v1 v{}+; \
                 the editor cannot run here (it requires the GPU path)",
                protocol::VERSION_DMABUF
            );
            return Ok(());
        }
        println!(
            "gpu-probe: zwp_linux_dmabuf_v1 bound at v{}",
            self.presentation.dmabuf_version
        );
        match self.presentation.syncobj_manager {
            Some(_) => println!(
                "gpu-probe: {} present: explicit sync available",
                protocol::IFACE_DRM_SYNCOBJ
            ),
            None => println!(
                "gpu-probe: {} absent: GPU path uses the implicit-sync bridge",
                protocol::IFACE_DRM_SYNCOBJ
            ),
        }
        // The server answers get_default_feedback before the sync that follows
        // it, so one roundtrip delivers the whole feedback burst.
        self.request_dmabuf_feedback();
        self.roundtrip()?;
        let Some(fb) = self.presentation.feedback.ready() else {
            return Err(Error::msg(
                "compositor never committed dmabuf feedback (no done event)",
            ));
        };
        println!(
            "gpu-probe: format table: {} format+modifier pairs",
            self.presentation.feedback.table_len()
        );
        println!(
            "gpu-probe: main device: {}:{}",
            dmabuf::dev_major(fb.main_device),
            dmabuf::dev_minor(fb.main_device)
        );
        for (i, t) in fb.tranches.iter().enumerate() {
            println!(
                "gpu-probe: tranche {i}: device {}:{}, flags {:#x}, {} pairs",
                dmabuf::dev_major(t.device),
                dmabuf::dev_minor(t.device),
                t.flags,
                t.formats.len()
            );
        }
        let mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if mods.is_empty() {
            println!("gpu-probe: XRGB8888 is not offered; the GPU path needs it");
            return Ok(());
        }
        let rendered: Vec<String> = mods
            .iter()
            .map(|&m| {
                if m == dmabuf::MOD_LINEAR {
                    "linear".to_string()
                } else {
                    format!("{m:#x}")
                }
            })
            .collect();
        println!("gpu-probe: XRGB8888 modifiers: {}", rendered.join(", "));

        // The Vulkan half: can a device matching the compositor's be brought
        // up with the extensions the presentation path needs, and can it
        // actually allocate and export a presentable image? A failure here is
        // the probe's answer, not an error.
        let main_device = fb.main_device;
        let gpu = match vulkan::Gpu::new(main_device) {
            Ok(gpu) => gpu,
            Err(e) => {
                println!("gpu-probe: vulkan unavailable: {e}");
                return Ok(());
            }
        };
        println!(
            "gpu-probe: vulkan device: {} (graphics queue family {})",
            gpu.name(),
            gpu.queue_family()
        );
        let common = gpu.image_modifiers(&mods);
        if common.is_empty() {
            println!("gpu-probe: no common DRM modifier; the GPU path is unavailable");
            return Ok(());
        }
        println!(
            "gpu-probe: common modifiers: {} of {}",
            common.len(),
            mods.len()
        );
        match gpu.create_image(64, 64, &common) {
            Ok(img) => {
                println!(
                    "gpu-probe: test image exported: modifier {:#x}, stride {}, offset {}",
                    img.modifier(),
                    img.stride(),
                    img.offset()
                );
                gpu.destroy_image(img);
                println!("gpu-probe: ready (the editor would use the GPU path)");
            }
            Err(e) => println!("gpu-probe: image export failed: {e}"),
        }
        Ok(())
    }
}

/// A 0x00RRGGBB color as the linear-float RGBA a Vulkan clear expects. The
/// authored bytes are sRGB-encoded; the color attachment is an sRGB view, so
/// the clear value is taken as linear and re-encoded on store. Converting here
/// makes the cleared background land on the same bytes as the authored color.
fn color_f32(color: u32) -> [f32; 4] {
    let srgb_to_linear = |c: u32| {
        let c = c as f32 / 255.0;
        if c > 0.04045 {
            ((c + 0.055) / 1.055).powf(2.4)
        } else {
            c / 12.92
        }
    };
    [
        srgb_to_linear((color >> 16) & 0xff),
        srgb_to_linear((color >> 8) & 0xff),
        srgb_to_linear(color & 0xff),
        1.0,
    ]
}

#[cfg(test)]
mod tests {
    use crate::app::present::*;

    /// A frame the CPU waited out still gets an acquire point, already signaled,
    /// and claims its buffer's release point.
    #[test]
    fn a_cpu_waited_frame_publishes_a_signaled_acquire_point() {
        let Some(drm) = ffi::open_any_render_node() else {
            eprintln!("bnknotes: no DRM render node; skipping the acquire-point test");
            return;
        };
        let mut es = ExplicitSync::new(drm).expect("create the syncobjs");
        let (point, _) = es.claim(None, 0).expect("claim a CPU-waited frame");
        let acquire = ffi::drm_syncobj_point_to_sync_file(
            es.drm.as_raw_fd(),
            es.acquire_syncobj,
            point,
            es.scratch,
            false,
        )
        .expect("the acquire point holds a fence");
        assert!(
            ffi::fd_is_signaled(acquire.as_raw_fd()),
            "the compositor must not wait on a frame that is already done",
        );
        assert!(
            !signaled_now(&es, 0),
            "the buffer stays held until the compositor signals its release point",
        );
    }

    /// Whether buffer idx's release point holds a signaled fence right now.
    /// release_fence waits for a fence to be attached, which a test of a point
    /// nobody signals would sit out in full.
    fn signaled_now(es: &ExplicitSync, idx: usize) -> bool {
        let slot = &es.release[idx];
        slot.point.is_some_and(|point| {
            let drm = es.drm.as_raw_fd();
            matches!(
                ffi::drm_syncobj_point_to_sync_file(drm, slot.syncobj, point, es.scratch, false),
                Ok(f) if ffi::fd_is_signaled(f.as_raw_fd())
            )
        })
    }

    /// Frames go into buffer 0 and then buffer 1, and the compositor finishes
    /// with buffer 1 first, which the protocol allows. Buffer 0 is still being
    /// read, so it must not read as released until its own point signals.
    #[test]
    fn a_buffer_released_out_of_order_does_not_release_the_other() {
        let Some(drm) = ffi::open_any_render_node() else {
            eprintln!("bnknotes: no DRM render node; skipping the release-order test");
            return;
        };
        let mut es = ExplicitSync::new(drm).expect("create the syncobjs");
        es.release[0].timeline = 10;
        es.release[1].timeline = 11;
        let commits = [
            es.claim(None, 0).expect("frame into buffer 0"),
            es.claim(None, 1).expect("frame into buffer 1"),
        ];
        // What the compositor does once it is done with a commit's buffer: signal
        // the point the commit named, on the timeline it named.
        let release = |es: &ExplicitSync, (point, timeline): (u64, u32)| {
            let syncobj = es.release_syncobj_of(timeline).expect("a known timeline");
            ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), syncobj, point, es.signaled)
                .expect("signal the release point");
        };
        let released = |es: &ExplicitSync, idx: usize| matches!(es.release_fence(idx), Ok(Some(f)) if ffi::fd_is_signaled(f.as_raw_fd()));

        release(&es, commits[1]);
        assert!(released(&es, 1), "buffer 1 reads as released");
        assert!(!signaled_now(&es, 0), "buffer 0 is still being read");
        release(&es, commits[0]);
        assert!(
            released(&es, 0),
            "buffer 0 reads as released once its own point signals"
        );
    }

    #[test]
    fn a_buffer_is_free_once_its_release_point_holds_a_fence() {
        let Some(drm) = ffi::open_any_render_node() else {
            eprintln!("bnknotes: no DRM render node; skipping the free-buffer test");
            return;
        };
        let mut es = ExplicitSync::new(drm).expect("create the syncobjs");
        es.release[0].timeline = 10;
        es.release[1].timeline = 11;
        // What the compositor does once it is done with a commit's buffer.
        let release = |es: &ExplicitSync, (point, timeline): (u64, u32)| {
            let syncobj = es.release_syncobj_of(timeline).expect("a known timeline");
            ffi::drm_syncobj_signal_point(es.drm.as_raw_fd(), syncobj, point, es.signaled)
                .expect("signal the release point");
        };
        let free = |es: &ExplicitSync| es.free_slot().expect("query the release points");

        assert_eq!(free(&es), Some(0), "unused buffers are free");
        let first = es.claim(None, 0).expect("frame into buffer 0");
        assert_eq!(free(&es), Some(1), "a committed buffer is the compositor's");
        let second = es.claim(None, 1).expect("frame into buffer 1");
        assert_eq!(free(&es), None, "no release event frees either");

        release(&es, first);
        assert_eq!(free(&es), Some(0), "its release point now holds a fence");
        release(&es, second);
        assert_eq!(free(&es), Some(0), "the buffer committed longest ago first");
        let third = es.claim(None, 0).expect("frame into buffer 0 again");
        assert_eq!(free(&es), Some(1), "the other one, still released");
        release(&es, third);
        assert_eq!(
            free(&es),
            Some(1),
            "both released, and buffer 1 went to the screen first"
        );
    }

    /// The store-side sRGB encode the attachment applies, inverse of the decode
    /// in color_f32.
    fn linear_to_srgb(c: f32) -> f32 {
        if c > 0.003_130_8 {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * c
        }
    }

    #[test]
    fn clear_color_decode_then_store_encode_reproduces_the_authored_bytes() {
        for color in [
            0x0000_0000,
            0x00ff_ffff,
            0x0080_8080,
            0x0012_3456,
            BACKGROUND,
        ] {
            let lin = color_f32(color);
            let enc = |c: f32| (linear_to_srgb(c) * 255.0).round() as u32;
            let got = (enc(lin[0]) << 16) | (enc(lin[1]) << 8) | enc(lin[2]);
            assert_eq!(
                got, color,
                "0x{color:08x} must survive decode then re-encode"
            );
            assert_eq!(lin[3], 1.0, "the clear is opaque");
        }
    }
}
