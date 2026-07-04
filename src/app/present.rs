use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::rc::Rc;
use std::time::Instant;

use super::State;
use crate::display;
use crate::dmabuf;
use crate::error::{Error, Result};
use crate::ffi;
use crate::gpu;
use crate::protocol::{
    self, wl_buffer, wl_display, wl_surface, wp_linux_drm_syncobj_manager_v1,
    wp_linux_drm_syncobj_surface_v1, xdg_surface, zwp_linux_buffer_params_v1,
    zwp_linux_dmabuf_feedback_v1, zwp_linux_dmabuf_v1,
};
use crate::theme::BACKGROUND;
use crate::vulkan;
use crate::wire::{Arg, Reader};

/// Client-owned explicit-sync state (`linux-drm-syncobj-v1`), negotiated over a
/// live GPU backend when the compositor advertises the manager. Two DRM syncobj
/// timelines the compositor shares: we signal `acquire` with each frame's
/// render-done fence, and it signals `release` when done reading a buffer. A
/// scratch binary syncobj carries the sync-file transfers. `point` increments
/// per present; `buffer_release[i]` is the release point last published for
/// buffer slot `i`, so that slot's next reuse waits on it. Dropping this closes
/// the render-node fd, which frees every syncobj created on it.
pub(super) struct ExplicitSync {
    drm: OwnedFd,
    acquire_syncobj: u32,
    release_syncobj: u32,
    scratch: u32,
    acquire_timeline: u32,
    release_timeline: u32,
    surface: u32,
    point: u64,
    buffer_release: [Option<u64>; 2],
}

impl State {
    /// Bring the Vulkan backend up from the committed dmabuf feedback: a
    /// device matching the compositor's, and a modifier list their common
    /// ground. Called before `create_buffers`, which allocates the exportable
    /// GPU images. A failure here is fatal (the editor requires a GPU).
    pub(super) fn init_gpu(&mut self) -> Result<()> {
        if self.dmabuf.is_none() {
            return Err(Error::msg("compositor lacks zwp_linux_dmabuf_v1 v4+"));
        }
        let fb = self
            .dmabuf_fb
            .ready()
            .ok_or_else(|| Error::msg("dmabuf feedback never committed"))?;
        let main_device = fb.main_device;
        let compositor_mods = fb.modifiers_for(dmabuf::DRM_FORMAT_XRGB8888);
        if compositor_mods.is_empty() {
            return Err(Error::msg("compositor does not accept XRGB8888 dmabufs"));
        }
        let gpu = vulkan::Gpu::new(Some(main_device))?;
        let mods = gpu.image_modifiers(&compositor_mods);
        if mods.is_empty() {
            return Err(Error::msg(
                "no common DRM modifier between compositor and device",
            ));
        }
        self.gpu_modifiers = mods;
        self.gpu = Some(gpu);
        Ok(())
    }

    /// Negotiate explicit sync (`linux-drm-syncobj-v1`) when a GPU backend is
    /// live and the compositor advertises the manager. Best-effort: on any
    /// failure the GPU path keeps the implicit-sync bridge, which is why this
    /// returns nothing and only logs under stats.
    pub(super) fn init_explicit_sync(&mut self) {
        let Some(manager) = self.drm_syncobj_manager else {
            return;
        };
        let Some(minor) = self.gpu.as_ref().and_then(vulkan::Gpu::render_node) else {
            return; // No GPU backend, or the device exposes no render node.
        };
        match self.negotiate_explicit_sync(manager, minor) {
            Ok(es) => self.explicit_sync = Some(es),
            Err(e) => {
                if self.stats {
                    eprintln!("bnknotes: explicit sync unavailable, using the sync bridge: {e}");
                }
            }
        }
    }

    /// Create the two client-owned timelines plus a scratch syncobj, share the
    /// timelines with the compositor, and attach explicit sync to the surface.
    pub(super) fn negotiate_explicit_sync(
        &mut self,
        manager: u32,
        minor: u32,
    ) -> Result<ExplicitSync> {
        let drm = ffi::open_drm_render_node(minor)?;
        let fd = drm.as_raw_fd();
        let acquire_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let release_syncobj = ffi::drm_syncobj_create(fd, false)?;
        let scratch = ffi::drm_syncobj_create(fd, false)?;
        let acquire_timeline = self.import_timeline(manager, fd, acquire_syncobj)?;
        let release_timeline = self.import_timeline(manager, fd, release_syncobj)?;
        let surface = self.alloc_id();
        self.conn.request(
            manager,
            wp_linux_drm_syncobj_manager_v1::GET_SURFACE,
            &[Arg::NewId(surface), Arg::Object(self.surface)],
        );
        Ok(ExplicitSync {
            drm,
            acquire_syncobj,
            release_syncobj,
            scratch,
            acquire_timeline,
            release_timeline,
            surface,
            point: 0,
            buffer_release: [None, None],
        })
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
    /// `zwp_linux_buffer_params_v1` (`create_immed` is sound because every
    /// parameter comes from the compositor's own feedback). A failure (a resize
    /// on a dying device) propagates and ends the process; there is no software
    /// path to fall back to.
    pub(super) fn create_buffers(&mut self) -> Result<()> {
        let dmabuf_global = self
            .dmabuf
            .ok_or_else(|| Error::msg("gpu buffers without zwp_linux_dmabuf_v1"))?;
        let (w, h) = (self.width, self.height);
        let mut images = Vec::with_capacity(2);
        {
            let Some(gpu) = self.gpu.as_ref() else {
                return Err(Error::msg("gpu buffers without a gpu backend"));
            };
            for _ in 0..2 {
                images.push(gpu.create_image(w, h, &self.gpu_modifiers)?);
            }
        }
        for (i, img) in images.iter().enumerate() {
            let params = self.alloc_id();
            self.conn.request(
                dmabuf_global,
                zwp_linux_dmabuf_v1::CREATE_PARAMS,
                &[Arg::NewId(params)],
            );
            let modifier = img.modifier();
            self.conn.request_with_fd(
                params,
                zwp_linux_buffer_params_v1::ADD,
                &[
                    Arg::Uint(0), // plane index
                    Arg::Uint(img.offset()),
                    Arg::Uint(img.stride()),
                    Arg::Uint((modifier >> 32) as u32),
                    Arg::Uint(modifier as u32),
                ],
                img.fd(),
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
            self.buffers[i] = buffer;
        }
        self.gpu_images = images;
        self.buffer_size = (self.width, self.height);
        self.busy = [false, false];
        // Fresh images never carried a release point, so a resize must forget
        // the old buffers' points; their next use starts with no wait.
        if let Some(es) = self.explicit_sync.as_mut() {
            es.buffer_release = [None, None];
        }
        // Fresh images hold no known frame, so forget what was on screen: the
        // screen diff gates whether the next frame presents at all.
        self.present_list = Rc::default();
        Ok(())
    }

    /// Destroy the current buffers and their backing GPU images. The server
    /// recycles the wl_buffer ids via `delete_id` into our free list.
    pub(super) fn destroy_buffers(&mut self) {
        let buffers = self.buffers;
        for b in buffers {
            if b != 0 {
                self.conn.request(b, wl_buffer::DESTROY, &[]);
            }
        }
        if let Some(gpu) = &self.gpu {
            // Nothing may be in flight while images are destroyed; resize is
            // rare enough that a full drain is the simple correct answer.
            gpu.wait_idle();
            for img in self.gpu_images.drain(..) {
                gpu.destroy_image(img);
            }
        }
        self.buffers = [0, 0];
        self.busy = [false, false];
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

    /// Declare the window geometry, then ack configure `serial`. The geometry is
    /// the whole surface (no client-side decorations or shadows), and being
    /// double-buffered surface state it is what the compositor anchors a resize's
    /// fixed edge to; sending it in the same commit as the ack and the matching
    /// buffer gives the compositor one atomic, consistent size for the frame, so
    /// the anchored edge does not drift. Caller commits after this.
    pub(super) fn ack_configure(&mut self, serial: u32) {
        let (w, h) = (self.width as i32, self.height as i32);
        self.conn.request(
            self.xdg_surface,
            xdg_surface::SET_WINDOW_GEOMETRY,
            &[Arg::Int(0), Arg::Int(0), Arg::Int(w), Arg::Int(h)],
        );
        self.conn.request(
            self.xdg_surface,
            xdg_surface::ACK_CONFIGURE,
            &[Arg::Uint(serial)],
        );
    }

    /// Build the current frame and present it on the GPU. Returns false if both
    /// buffers are still held by the compositor, leaving `dirty` set so the next
    /// buffer release retries.
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
        if self.buffer_size != (self.width, self.height) {
            self.destroy_buffers();
            self.create_buffers()?;
        }
        let Some(idx) = (0..2).find(|&i| !self.busy[i]) else {
            return Ok(false);
        };
        let started = self.stats.then(Instant::now);
        let (sw, sh) = (self.width as i32, self.height as i32);
        // Compose the frame from every pane (tree, editors) plus the splitters.
        let new_list = self.build_frame_list();

        // Nothing changed since the on-screen frame: nothing to present (the
        // common idle case).
        let screen_dmg = display::damage(&self.present_list, &new_list, sw, sh);
        // Acknowledge the latest configure paired with this commit, so the buffer
        // the compositor sees is always the one sized to the configure it just
        // acked; that is what keeps an anchored (left/top) resize edge from
        // jumping. Taken here so it rides the same flush as the commit below.
        let ack = self.pending_configure.take();
        if screen_dmg.is_empty() {
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

        // Render, returning the render-done fence under explicit sync (`None`
        // on the bridge path, or when the driver could not export one). Inline
        // images resolve against both panes' stores, so a picture renders in
        // whichever pane it belongs to.
        let images = [&self.editors[0].images, &self.editors[1].images];
        let render_done = {
            let Some(gpu) = self.gpu.as_mut() else {
                return Err(Error::msg("gpu frame without a gpu backend"));
            };
            let Some(img) = self.gpu_images.get_mut(idx) else {
                return Err(Error::msg("gpu frame without gpu buffers"));
            };
            let known = |key: &gpu::ImageKey| gpu.has_image(key);
            let frame = gpu::build_frame(
                &self.fonts,
                &images,
                &new_list,
                &mut self.gpu_glyphs,
                &known,
            );
            match self.explicit_sync.as_ref() {
                Some(es) => {
                    // Wait on the compositor's release of this buffer's previous
                    // frame; nothing to wait on the first time the slot is used.
                    let release_wait = match es.buffer_release[idx] {
                        Some(p) => Some(ffi::drm_syncobj_point_to_sync_file(
                            es.drm.as_raw_fd(),
                            es.release_syncobj,
                            p,
                            es.scratch,
                        )?),
                        None => None,
                    };
                    gpu.render_list_explicit(img, &frame, color_f32(BACKGROUND), release_wait)?
                }
                None => {
                    gpu.render_list(img, &frame, color_f32(BACKGROUND))?;
                    None
                }
            }
        };
        self.present_list = new_list;

        let buffer = self.buffers[idx];
        self.conn.request(
            self.surface,
            wl_surface::ATTACH,
            &[Arg::Object(buffer), Arg::Int(0), Arg::Int(0)],
        );
        // Explicit sync: publish this frame's render-done fence as the acquire
        // point and claim a release point, so the compositor waits before
        // reading and signals when done (the buffer's next reuse waits on it).
        // With no exportable fence the frame was CPU-waited: set no points.
        match (render_done, self.explicit_sync.as_mut()) {
            (Some(fence), Some(es)) => {
                es.point += 1;
                let p = es.point;
                ffi::drm_syncobj_sync_file_to_point(
                    es.drm.as_raw_fd(),
                    fence.as_raw_fd(),
                    es.acquire_syncobj,
                    p,
                    es.scratch,
                )?;
                es.buffer_release[idx] = Some(p);
                let (hi, lo) = ((p >> 32) as u32, p as u32);
                let (sync_surface, acquire_tl, release_tl) =
                    (es.surface, es.acquire_timeline, es.release_timeline);
                self.conn.request(
                    sync_surface,
                    wp_linux_drm_syncobj_surface_v1::SET_ACQUIRE_POINT,
                    &[Arg::Object(acquire_tl), Arg::Uint(hi), Arg::Uint(lo)],
                );
                self.conn.request(
                    sync_surface,
                    wp_linux_drm_syncobj_surface_v1::SET_RELEASE_POINT,
                    &[Arg::Object(release_tl), Arg::Uint(hi), Arg::Uint(lo)],
                );
                self.explicit_fence_frames += 1;
            }
            (None, Some(es)) => {
                es.buffer_release[idx] = None;
                self.explicit_cpu_wait_frames += 1;
            }
            _ => {}
        }
        // The GPU repainted everything, but the *screen* change is still just
        // the diff, so report those regions and let the compositor
        // recomposite only what moved.
        for r in &screen_dmg {
            self.conn.request(
                self.surface,
                wl_surface::DAMAGE,
                &[Arg::Int(r.x), Arg::Int(r.y), Arg::Int(r.w), Arg::Int(r.h)],
            );
        }
        let callback = self.alloc_id();
        self.conn
            .request(self.surface, wl_surface::FRAME, &[Arg::NewId(callback)]);
        self.frame_callback = callback;
        self.conn.request(self.surface, wl_surface::COMMIT, &[]);
        self.busy[idx] = true;

        if let Some(started) = started {
            let cache = self.fonts.cache_stats();
            let gap = self
                .last_present
                .map(|t| started.duration_since(t).as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let sync = if self.explicit_sync.is_some() {
                format!(
                    "explicit-sync {} fence/{} cpu-wait",
                    self.explicit_fence_frames, self.explicit_cpu_wait_frames
                )
            } else {
                "implicit-sync bridge".to_string()
            };
            eprintln!(
                "[stats] gpu frame {} {:.2}ms build+submit, {:.1}ms since last | glyphs hit {} miss {} ({:.1}% of {}) | {}",
                self.frame_count,
                started.elapsed().as_secs_f64() * 1000.0,
                gap,
                cache.hits,
                cache.misses,
                cache.hit_rate() * 100.0,
                cache.lookups(),
                sync,
            );
            self.last_present = Some(started);
        }
        self.frame_count += 1;
        Ok(true)
    }

    /// Ask the dmabuf global for its default feedback. The events accumulate in
    /// `dmabuf_fb` as they arrive; nothing waits on them.
    pub(super) fn request_dmabuf_feedback(&mut self) {
        let Some(dmabuf) = self.dmabuf else {
            return;
        };
        let id = self.alloc_id();
        self.conn.request(
            dmabuf,
            zwp_linux_dmabuf_v1::GET_DEFAULT_FEEDBACK,
            &[Arg::NewId(id)],
        );
        self.dmabuf_feedback = id;
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
                    self.dmabuf_fb.set_table(&[]);
                    return Ok(());
                }
                let ptr = ffi::mmap_ro(fd.as_raw_fd(), size)?;
                // SAFETY: ptr maps `size` readable bytes from the table fd; the
                // bytes are copied out before the mapping is released.
                let bytes = unsafe { core::slice::from_raw_parts(ptr, size) }.to_vec();
                // SAFETY: ptr/size came from mmap_ro and are not used again.
                unsafe { ffi::unmap(ptr, size) };
                self.dmabuf_fb.set_table(&bytes);
            }
            zwp_linux_dmabuf_feedback_v1::EV_MAIN_DEVICE => {
                self.dmabuf_fb.main_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_TARGET_DEVICE => {
                self.dmabuf_fb.tranche_device(r.array()?)?;
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FORMATS => {
                self.dmabuf_fb.tranche_formats(r.array()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_FLAGS => {
                self.dmabuf_fb.tranche_flags(r.u32()?);
            }
            zwp_linux_dmabuf_feedback_v1::EV_TRANCHE_DONE => self.dmabuf_fb.tranche_done(),
            zwp_linux_dmabuf_feedback_v1::EV_DONE => self.dmabuf_fb.done(),
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
        if self.dmabuf.is_none() {
            println!(
                "gpu-probe: compositor lacks zwp_linux_dmabuf_v1 v{}+; \
                 the editor cannot run here (it requires the GPU path)",
                protocol::VERSION_DMABUF
            );
            return Ok(());
        }
        println!(
            "gpu-probe: zwp_linux_dmabuf_v1 bound at v{}",
            self.dmabuf_version
        );
        match self.drm_syncobj_manager {
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
        let Some(fb) = self.dmabuf_fb.ready() else {
            return Err(Error::msg(
                "compositor never committed dmabuf feedback (no done event)",
            ));
        };
        println!(
            "gpu-probe: format table: {} format+modifier pairs",
            self.dmabuf_fb.table_len()
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
        let gpu = match vulkan::Gpu::new(Some(main_device)) {
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
/// bytes are the display (sRGB-encoded) values as authored, passed through as
/// the same numbers, not a color-space conversion.
fn color_f32(color: u32) -> [f32; 4] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
        1.0,
    ]
}
