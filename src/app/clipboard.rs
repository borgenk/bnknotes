use std::os::fd::AsRawFd;

use super::State;
use crate::error::{Error, Result};
use crate::ffi;
use crate::file;
use crate::markdown;
use crate::protocol::{wl_data_device, wl_data_device_manager, wl_data_offer, wl_data_source};
use crate::wire::{Arg, Reader};

/// MIME types we advertise when putting text on the clipboard, and accept when
/// pasting. The charset variant is listed first as the preferred form.
pub(super) const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

/// Clipboard state. When we own the selection, `source` is our live
/// `wl_data_source`; `data` is the bytes it serves and `mimes` the types it
/// advertises (text for a text selection, `image/png`/`image/jpeg` for a copied
/// image). Incoming offers are tracked so a paste can request the right MIME type.
pub(super) struct ClipboardState {
    pub(super) source: u32,
    pub(super) data: Vec<u8>,
    pub(super) mimes: Vec<&'static str>,
    pub(super) incoming_offer: u32,
    pub(super) incoming_text_mime: Option<String>,
    pub(super) selection_offer: u32,
    pub(super) selection_text_mime: Option<String>,
}

impl ClipboardState {
    pub(super) fn new() -> Self {
        Self {
            source: 0,
            data: Vec::new(),
            mimes: Vec::new(),
            incoming_offer: 0,
            incoming_text_mime: None,
            selection_offer: 0,
            selection_text_mime: None,
        }
    }
}

impl State {
    pub(super) fn on_data_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_device::EV_DATA_OFFER => {
                // The compositor introduces a new offer; its MIME types follow.
                self.clipboard.incoming_offer = r.u32()?;
                self.clipboard.incoming_text_mime = None;
            }
            wl_data_device::EV_SELECTION => {
                let offer = r.u32()?;
                if self.clipboard.selection_offer != 0 && self.clipboard.selection_offer != offer {
                    self.conn
                        .request(self.clipboard.selection_offer, wl_data_offer::DESTROY, &[]);
                }
                self.clipboard.selection_offer = offer;
                self.clipboard.selection_text_mime =
                    if offer != 0 && offer == self.clipboard.incoming_offer {
                        self.clipboard.incoming_text_mime.clone()
                    } else {
                        None
                    };
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn on_data_source(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_source::EV_SEND => {
                // A target is pasting our data: write it to the supplied fd. Every
                // MIME we advertise maps to the same bytes (text encodings, or the
                // one image type), so the requested MIME does not change what we send.
                let _mime = r.string()?;
                let fd = self
                    .conn
                    .take_fd()
                    .ok_or_else(|| Error::msg("data_source.send arrived without its fd"))?;
                // A target that closes its read end early makes this write fail
                // (EPIPE). Serving the clipboard must not bring the editor down, so
                // report and swallow the I/O error rather than propagate it out of
                // the event loop (the protocol-parse errors above stay fatal).
                if let Err(e) = ffi::write_all(fd.as_raw_fd(), &self.clipboard.data) {
                    eprintln!("bnknotes: clipboard serve failed: {e}");
                }
            }
            wl_data_source::EV_CANCELLED => {
                // Another selection replaced ours; tear the source down.
                if self.clipboard.source != 0 {
                    self.conn
                        .request(self.clipboard.source, wl_data_source::DESTROY, &[]);
                }
                self.clipboard.source = 0;
                self.clipboard.data.clear();
                self.clipboard.mimes.clear();
            }
            _ => {}
        }
        Ok(())
    }

    /// Put the focused editor's selection on the clipboard. No-op without one.
    pub(super) fn copy_selection(&mut self) -> Result<()> {
        let i = self.active_editor();
        if let Some(text) = self.editors[i].text.selected_text() {
            self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        }
        Ok(())
    }

    /// Copy the focused editor's selection, then delete it.
    pub(super) fn cut_selection(&mut self) -> Result<()> {
        let i = self.active_editor();
        let Some(text) = self.editors[i].text.selected_text() else {
            return Ok(());
        };
        self.set_clipboard(text.into_bytes(), CLIPBOARD_MIMES.to_vec());
        if self.editors[i].text.backspace() {
            self.after_edit();
        }
        Ok(())
    }

    /// Put the selected image's file bytes on the clipboard, advertised as its real
    /// type (`image/png`/`image/jpeg`), so it can be pasted into other apps. No-op
    /// if nothing is selected or the file can no longer be read.
    pub(super) fn copy_selected_image(&mut self) -> Result<()> {
        let i = self.active_editor();
        let Some(ls) = self.editors[i].selected_image else {
            return Ok(());
        };
        let Some(raw) = self.image_path_at(i, ls) else {
            return Ok(());
        };
        if let Some((bytes, mime)) = self.editors[i].images.read_source(&raw) {
            self.set_clipboard(bytes, vec![mime]);
        }
        Ok(())
    }

    /// The raw destination path of the image whose logical line starts at `ls` in
    /// editor `i`, or `None` if that line is no longer a standalone image.
    pub(super) fn image_path_at(&self, i: usize, ls: usize) -> Option<String> {
        let rope = self.editors[i].text.rope();
        let (_, le) = rope.line_content_range(rope.byte_to_line(ls));
        let line = rope.slice(ls..le).as_cow();
        let link = markdown::standalone_image(&line)?;
        Some(line[link.path.0..link.path.1].to_string())
    }

    /// Become the clipboard owner serving `data` under `mimes`, replacing any
    /// source we held.
    pub(super) fn set_clipboard(&mut self, data: Vec<u8>, mimes: Vec<&'static str>) {
        let (Some(manager), true) = (self.data_device_manager, self.data_device != 0) else {
            return;
        };
        if self.clipboard.source != 0 {
            self.conn
                .request(self.clipboard.source, wl_data_source::DESTROY, &[]);
        }
        let source = self.alloc_id();
        self.conn.request(
            manager,
            wl_data_device_manager::CREATE_DATA_SOURCE,
            &[Arg::NewId(source)],
        );
        for mime in &mimes {
            self.conn
                .request(source, wl_data_source::OFFER, &[Arg::Str(mime)]);
        }
        self.conn.request(
            self.data_device,
            wl_data_device::SET_SELECTION,
            &[Arg::Object(source), Arg::Uint(self.last_serial)],
        );
        self.clipboard.source = source;
        self.clipboard.data = data;
        self.clipboard.mimes = mimes;
    }

    /// Insert the clipboard's text at the cursor.
    pub(super) fn paste(&mut self) -> Result<()> {
        if let Some(text) = self.clipboard_text()? {
            self.insert_clipboard(&text);
        }
        Ok(())
    }

    /// The clipboard's text, if it holds any. If we own the clipboard, our own
    /// data is returned directly: going through the compositor would deadlock,
    /// blocking on the pipe while also needing to service our own source's
    /// send event. Otherwise the data is received through a pipe (hand the
    /// write end to the compositor, close our copy, read the read end to EOF).
    /// A copied image has no text representation and yields `None`.
    pub(super) fn clipboard_text(&mut self) -> Result<Option<String>> {
        if self.clipboard.source != 0 {
            if self.clipboard.mimes.iter().any(|m| is_text_mime(m)) {
                return Ok(Some(
                    String::from_utf8_lossy(&self.clipboard.data).into_owned(),
                ));
            }
            return Ok(None);
        }
        let (Some(mime), true) = (
            self.clipboard.selection_text_mime.clone(),
            self.clipboard.selection_offer != 0,
        ) else {
            return Ok(None);
        };
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            self.clipboard.selection_offer,
            wl_data_offer::RECEIVE,
            &[Arg::Str(&mime)],
            write_end.as_raw_fd(),
        )?;
        drop(write_end);
        let bytes = ffi::read_to_end(read_end.as_raw_fd())?;
        Ok(String::from_utf8(bytes).ok())
    }

    pub(super) fn insert_clipboard(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        // Pasted text can carry CRLF or lone CR (from another app or platform);
        // the buffer only ever holds '\n' (file load normalizes the same way), so
        // normalize here too rather than let a stray '\r' into the rope.
        let text = file::normalize_newlines(text);
        let i = self.active_editor();
        self.editors[i].text.insert_str(&text);
        self.dirty = true;
        self.reveal_active_caret();
    }
}

/// Whether a clipboard MIME type carries text we can paste.
pub(super) fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/") || mime == "UTF8_STRING"
}
