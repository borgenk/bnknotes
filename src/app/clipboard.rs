use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::app::State;
use crate::file;
use crate::markdown;
use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::protocol::{
    wl_data_device, wl_data_device_manager, wl_data_offer, wl_data_source,
};
use crate::platform::wire::{Arg, Reader};
use crate::render::image;

/// How pasted text meets the line it lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PasteMode {
    /// Fitted to the caret's line: on a quote line every pasted line joins the
    /// quote (Ctrl+V).
    Fit,
    /// Exactly as copied (Ctrl+Shift+V).
    Verbatim,
}

/// MIME types we advertise when putting text on the clipboard, and accept when
/// pasting. The charset variant is listed first as the preferred form.
pub(super) const CLIPBOARD_MIMES: &[&str] = &["text/plain;charset=utf-8", "text/plain"];

/// The image types a paste accepts, in preference order: what the editor's decoder
/// reads, since a picture it cannot draw is one it should not file in the vault.
/// PNG first, the lossless form a screenshot tool and a browser both offer.
pub(super) const IMAGE_MIMES: &[&str] = &["image/png", "image/jpeg"];

/// Largest clipboard payload a paste will take. Another app decides what it puts on
/// the clipboard and how big it is, so the read is bounded; this is the same ceiling
/// the editor opens a file at, since a paste lands in the same rope (and, for an
/// image, the same vault) that a file would.
const MAX_PASTE_BYTES: usize = 64 * 1024 * 1024;

/// How long a paste waits for the clipboard owner to send more, and for the whole
/// payload. The read runs on the event loop, so an owner that accepts and never
/// writes would otherwise freeze the editor.
const PASTE_IDLE: Duration = Duration::from_secs(3);
const PASTE_TOTAL: Duration = Duration::from_secs(30);

/// How long quitting waits for transfers of our selection still in flight.
const SERVE_DRAIN: Duration = Duration::from_secs(2);

/// Transfers of our selection allowed in flight at once. Each is a thread blocked
/// on a target's pipe, so a target that never reads holds one until it closes.
const MAX_SERVES: usize = 8;

/// Clipboard state. When we own the selection, `source` is our live
/// `wl_data_source`; `data` is the bytes it serves and `mimes` the types it
/// advertises (text for a text selection, `image/png`/`image/jpeg` for a copied
/// image). Incoming offers are tracked so a paste can request the right MIME type.
pub(super) struct ClipboardState {
    pub(super) source: u32,
    /// Our selection's bytes, shared with the threads serving it, so a transfer
    /// keeps the selection it started with.
    pub(super) data: Arc<[u8]>,
    /// The transfers of our selection now in flight.
    serving: Arc<AtomicUsize>,
    pub(super) mimes: Vec<&'static str>,
    pub(super) incoming_offer: u32,
    pub(super) incoming_text_mime: Option<String>,
    /// The image type this offer carries, if it carries one we decode.
    pub(super) incoming_image_mime: Option<String>,
    pub(super) selection_offer: u32,
    pub(super) selection_text_mime: Option<String>,
    pub(super) selection_image_mime: Option<String>,
}

impl ClipboardState {
    pub(super) fn new() -> Self {
        Self {
            source: 0,
            data: Arc::default(),
            serving: Arc::default(),
            mimes: Vec::new(),
            incoming_offer: 0,
            incoming_text_mime: None,
            incoming_image_mime: None,
            selection_offer: 0,
            selection_text_mime: None,
            selection_image_mime: None,
        }
    }
}

impl State {
    /// Ask the selection's offer for mime and read what its source sends, to the
    /// end. The read waits on the source, so the request has to have gone first.
    fn receive_offer(&mut self, mime: &str) -> Result<Vec<u8>> {
        let (read_end, write_end) = ffi::pipe()?;
        self.conn.request_with_fd(
            self.clipboard.selection_offer,
            wl_data_offer::RECEIVE,
            &[Arg::Str(mime)],
            write_end.as_raw_fd(),
        )?;
        drop(write_end);
        if !self.conn.flush_within(PASTE_IDLE)? {
            return Err(Error::msg(
                "the compositor is not reading, so the paste could not be asked for",
            ));
        }
        ffi::read_to_end(
            read_end.as_raw_fd(),
            MAX_PASTE_BYTES,
            PASTE_IDLE,
            PASTE_TOTAL,
        )
    }

    /// Create the data device that drives the clipboard, once both the seat and
    /// the data device manager are bound. Without a manager editing still works,
    /// just with no copy and paste.
    pub(super) fn bind_data_device(&mut self) {
        let (Some(manager), Some(seat), 0) =
            (self.data_device_manager, self.seat, self.data_device)
        else {
            return;
        };
        let device = self.alloc_id();
        self.conn.request(
            manager,
            wl_data_device_manager::GET_DATA_DEVICE,
            &[Arg::NewId(device), Arg::Object(seat)],
        );
        self.data_device = device;
    }

    /// Let go of the data device with its seat, and of the selection it offered.
    /// Our own selection stays, for pasting within the editor.
    pub(super) fn release_data_device(&mut self) {
        if self.clipboard.selection_offer != 0 {
            self.conn
                .request(self.clipboard.selection_offer, wl_data_offer::DESTROY, &[]);
        }
        self.clipboard.selection_offer = 0;
        self.clipboard.selection_text_mime = None;
        self.clipboard.selection_image_mime = None;
        self.clipboard.incoming_offer = 0;
        self.clipboard.incoming_text_mime = None;
        self.clipboard.incoming_image_mime = None;
        let device = std::mem::take(&mut self.data_device);
        if device != 0 && self.data_device_version >= wl_data_device::VERSION_RELEASE {
            self.conn.request(device, wl_data_device::RELEASE, &[]);
        }
    }

    pub(super) fn on_data_device(&mut self, opcode: u16, r: &mut Reader) -> Result<()> {
        match opcode {
            wl_data_device::EV_DATA_OFFER => {
                // The compositor introduces a new offer; its MIME types follow.
                self.clipboard.incoming_offer = r.u32()?;
                self.clipboard.incoming_text_mime = None;
                self.clipboard.incoming_image_mime = None;
            }
            wl_data_device::EV_SELECTION => {
                let offer = r.u32()?;
                if self.clipboard.selection_offer != 0 && self.clipboard.selection_offer != offer {
                    self.conn
                        .request(self.clipboard.selection_offer, wl_data_offer::DESTROY, &[]);
                }
                self.clipboard.selection_offer = offer;
                let current = offer != 0 && offer == self.clipboard.incoming_offer;
                self.clipboard.selection_text_mime = current
                    .then(|| self.clipboard.incoming_text_mime.clone())
                    .flatten();
                self.clipboard.selection_image_mime = current
                    .then(|| self.clipboard.incoming_image_mime.clone())
                    .flatten();
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
                self.serve_selection(fd);
            }
            wl_data_source::EV_CANCELLED => {
                // Another selection replaced ours; tear the source down.
                self.destroy_source();
                self.clipboard.data = Arc::default();
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

    /// Put the selected image on the clipboard. No-op if nothing is selected.
    pub(super) fn copy_selected_image(&mut self) -> Result<()> {
        let i = self.active_editor();
        if let Some(ls) = self.editors[i].selected_picture() {
            self.copy_image(i, ls);
        }
        Ok(())
    }

    /// Put the file bytes of the image whose logical line starts at `ls` in editor
    /// `i` on the clipboard, advertised as its real type (`image/png`/`image/jpeg`),
    /// so it can be pasted into other apps. No-op if that line is no longer an
    /// image or the file can no longer be read.
    pub(super) fn copy_image(&mut self, i: usize, ls: usize) {
        let Some(raw) = self.image_path_at(i, ls) else {
            return;
        };
        if let Some((bytes, mime)) = self.editors[i].images.read_source(&raw) {
            self.set_clipboard(bytes, vec![mime]);
        }
    }

    /// Put the path of that same image's file on the clipboard as text, absolute
    /// and with symlinks resolved, however the note names it. No-op under the same
    /// conditions as [`Self::copy_image`].
    pub(super) fn copy_image_path(&mut self, i: usize, ls: usize) {
        let Some(raw) = self.image_path_at(i, ls) else {
            return;
        };
        if let Some(path) = self.editors[i].images.source_path(&raw) {
            self.set_clipboard(path.into_os_string().into_vec(), CLIPBOARD_MIMES.to_vec());
        }
    }

    /// The raw destination path of the image whose logical line starts at `ls` in
    /// editor `i`, or `None` if that line is no longer a standalone image. An offset
    /// held across an edit (an open menu's) can land mid-line, which names no image.
    pub(super) fn image_path_at(&self, i: usize, ls: usize) -> Option<String> {
        let rope = self.editors[i].text.rope();
        let (start, le) = rope.line_content_range(rope.byte_to_line(ls));
        if start != ls {
            return None;
        }
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
        self.destroy_source();
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
        self.clipboard.data = data.into();
        self.clipboard.mimes = mimes;
    }

    /// Insert the clipboard at the cursor: its text, fitted to the line or exactly
    /// as copied by the paste mode, or an image filed into the vault and embedded.
    ///
    /// Which one is not simply "an image if there is one". A source offers every
    /// flavor it can produce, so the flavors together say what was copied better
    /// than any one does: plain text alongside an image means text was copied and the
    /// image is a rendering of it (a spreadsheet cell, a formatted quote), while an
    /// image alongside only markup means the image was copied and the markup is a tag
    /// pointing at it (a browser's "Copy Image"). So plain text wins over an image,
    /// and an image wins over rich text alone.
    pub(super) fn paste(&mut self, mode: PasteMode) -> Result<()> {
        if self.clipboard_offers_image() && !self.clipboard_offers_plain_text() {
            // A paste that cannot be filed says so and stops, rather than quietly
            // pasting an <img> tag's markup instead of the picture it names.
            if let Err(e) = self.paste_image() {
                eprintln!("bnknotes: could not paste the image: {e}");
            }
            return Ok(());
        }
        if let Some(text) = self.clipboard_text()? {
            self.insert_clipboard(&text, mode);
        }
        Ok(())
    }

    /// Whether the clipboard holds an image this can decode.
    pub(super) fn clipboard_offers_image(&self) -> bool {
        if self.clipboard.source != 0 {
            return self.clipboard.mimes.iter().any(|m| IMAGE_MIMES.contains(m));
        }
        self.clipboard.selection_image_mime.is_some() && self.clipboard.selection_offer != 0
    }

    /// Whether the clipboard holds plain text, as opposed to the rich text a source
    /// offers alongside an image to describe it.
    fn clipboard_offers_plain_text(&self) -> bool {
        if self.clipboard.source != 0 {
            return self.clipboard.mimes.iter().any(|m| is_plain_text_mime(m));
        }
        self.clipboard
            .selection_text_mime
            .as_deref()
            .is_some_and(is_plain_text_mime)
    }

    /// File the clipboard's image in the vault's attachment folder and embed it at
    /// the cursor.
    fn paste_image(&mut self) -> Result<()> {
        let Some(bytes) = self.clipboard_image()? else {
            return Ok(());
        };
        // Decode before writing: the vault gets a picture the editor can draw back,
        // or nothing. This also runs the bytes past the dimension limits, so a
        // hostile or absurd image is refused here rather than at every later layout.
        image::Image::decode(&bytes, image::Limits::default())?;
        let name = self.write_attachment(&bytes)?;
        self.insert_embed(&name);
        Ok(())
    }

    /// The clipboard's image bytes, if it holds an image. Mirrors
    /// [`Self::clipboard_text`], including why our own selection is answered from our
    /// own copy rather than through the compositor: receiving would block on a pipe
    /// only we can fill.
    pub(super) fn clipboard_image(&mut self) -> Result<Option<Vec<u8>>> {
        if self.clipboard.source != 0 {
            if self.clipboard.mimes.iter().any(|m| IMAGE_MIMES.contains(m)) {
                return Ok(Some(self.clipboard.data.to_vec()));
            }
            return Ok(None);
        }
        let (Some(mime), true) = (
            self.clipboard.selection_image_mime.clone(),
            self.clipboard.selection_offer != 0,
        ) else {
            return Ok(None);
        };
        Ok(Some(self.receive_offer(&mime)?))
    }

    /// Write `bytes` into the active vault's attachment folder, under a fresh name,
    /// and report that name for the embed to point at.
    ///
    /// The name is `Pasted image <local timestamp>.<ext>`, the usual name for a pasted
    /// attachment, and the folder is whichever attachment folder the vault already
    /// keeps (see [`image::attachment_dir`]), so the file lands beside the vault's
    /// other pasted images.
    fn write_attachment(&mut self, bytes: &[u8]) -> Result<String> {
        let ext = image::extension(bytes)
            .ok_or_else(|| Error::msg("the clipboard's image is not a PNG or a JPEG"))?;
        let dir = image::attachment_dir(self.tree.tree.root());
        std::fs::create_dir_all(&dir)?;
        let stamp = timestamp()?;
        let (name, path) = free_attachment_name(&dir, &stamp, ext)
            .ok_or_else(|| Error::msg("no free name for the pasted image"))?;
        file::save_atomic(&path, bytes)?;
        Ok(name)
    }

    /// Insert an embed of `name` at the cursor, on a line of its own: a picture only
    /// renders when nothing shares its line, so this opens a line for it when the
    /// cursor is mid-text and leaves the caret on the next one, ready to keep typing
    /// under the image.
    fn insert_embed(&mut self, name: &str) {
        let i = self.active_editor();
        let lead = if self.active_line_is_blank_before_cursor(i) {
            ""
        } else {
            "\n"
        };
        self.editors[i]
            .text
            .insert_str(&format!("{lead}![[{name}]]\n"));
        self.after_edit();
    }

    /// Whether the cursor in editor `i` has nothing but blanks before it on its line,
    /// so an embed can start where it stands rather than on a new line.
    fn active_line_is_blank_before_cursor(&self, i: usize) -> bool {
        let text = &self.editors[i].text;
        let cursor = text.cursor();
        let rope = text.rope();
        let start = rope.line_to_byte(rope.byte_to_line(cursor));
        // Borrowed: the run before a cursor is one line's head, and a single-leaf
        // range hands back a `&str` without copying it.
        rope.slice(start..cursor)
            .as_cow()
            .chars()
            .all(char::is_whitespace)
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
        let bytes = self.receive_offer(&mime)?;
        // Text that is not UTF-8 pastes lossily, as our own selection does.
        Ok(Some(String::from_utf8(bytes).unwrap_or_else(|e| {
            String::from_utf8_lossy(e.as_bytes()).into_owned()
        })))
    }

    /// Destroy our data source, if there is one. A send already on its way for it
    /// still carries an fd, so the source stays known as retired until the
    /// compositor confirms the deletion.
    fn destroy_source(&mut self) {
        let source = std::mem::take(&mut self.clipboard.source);
        if source != 0 {
            self.conn.request(source, wl_data_source::DESTROY, &[]);
            self.retired.retire(source, wl_data_source::EV_SEND);
        }
    }

    /// Give transfers of our selection still in flight a bounded time to finish,
    /// since exiting would cut them short and hand the target a truncated
    /// selection.
    pub(super) fn finish_serving(&self) {
        let deadline = std::time::Instant::now() + SERVE_DRAIN;
        while self.clipboard.serving.load(Ordering::Acquire) > 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn serve_selection(&mut self, fd: OwnedFd) {
        serve(
            Arc::clone(&self.clipboard.data),
            fd,
            &self.clipboard.serving,
        );
    }

    fn insert_clipboard(&mut self, text: &str, mode: PasteMode) {
        if text.is_empty() {
            return;
        }
        // Pasted text can carry CRLF or lone CR (from another app or platform);
        // the buffer only ever holds '\n' (file load normalizes the same way), so
        // normalize here too rather than let a stray '\r' into the rope.
        let text = file::normalize_newlines(text);
        let i = self.active_editor();
        let rect = self.editor_rect(i);
        match mode {
            PasteMode::Fit => self.editors[i].paste_text(&self.fonts, rect, &text),
            PasteMode::Verbatim => self.editors[i].insert_text(&self.fonts, rect, &text),
        }
        // A paste is an edit like any other: the same settling every keystroke gets,
        // so the title's modified marker is not left reading clean over pasted text.
        self.after_edit();
    }
}

/// Whether a clipboard MIME type carries text we can paste.
pub(super) fn is_text_mime(mime: &str) -> bool {
    mime.starts_with("text/") || mime == "UTF8_STRING"
}

/// Whether a clipboard MIME type carries text as text, rather than as markup
/// describing something else. What separates a copied paragraph from the `<img>` tag
/// a browser offers beside a copied image.
fn is_plain_text_mime(mime: &str) -> bool {
    mime.starts_with("text/plain") || mime == "UTF8_STRING"
}

/// The local wall-clock stamp a pasted image is named with: `YYYYMMDDHHMMSS`, to the
/// second. Local rather than UTC because the name is read by a person looking for
/// what they pasted this morning.
fn timestamp() -> Result<String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| Error::msg("the system clock is before 1970"))?
        .as_secs();
    let now = i64::try_from(secs)
        .ok()
        .and_then(ffi::local_time)
        .ok_or_else(|| Error::msg("cannot read the local time"))?;
    Ok(format!(
        "{:04}{:02}{:02}{:02}{:02}{:02}",
        now.year, now.month, now.day, now.hour, now.minute, now.second
    ))
}

/// A name no file in `dir` has yet, and the path to write it at: `Pasted image
/// <stamp>.<ext>`, then the same with a counter. The stamp only counts to the
/// second, so a counter is what a second paste inside one second needs; past
/// [`MAX_ATTACHMENT_TRIES`] of them, `None`.
fn free_attachment_name(dir: &Path, stamp: &str, ext: &str) -> Option<(String, PathBuf)> {
    (0..MAX_ATTACHMENT_TRIES)
        .map(|n| match n {
            0 => format!("Pasted image {stamp}.{ext}"),
            _ => format!("Pasted image {stamp} {n}.{ext}"),
        })
        .map(|name| {
            let path = dir.join(&name);
            (name, path)
        })
        .find(|(_, path)| !path.exists())
}

/// How many same-second names a paste tries before giving up. A second paste inside
/// one second takes the first counter; a hundred of them is not a person pasting.
const MAX_ATTACHMENT_TRIES: u32 = 100;

/// Write data to a target's pipe on a worker thread, so a target that reads
/// slowly, or not at all, cannot stall the event loop. Past MAX_SERVES transfers in
/// flight a request is refused by closing its pipe, which the target reads as an
/// empty selection.
fn serve(data: Arc<[u8]>, fd: OwnedFd, serving: &Arc<AtomicUsize>) {
    let Some(slot) = ServeSlot::take(serving) else {
        eprintln!("bnknotes: clipboard serve refused: {MAX_SERVES} already in flight");
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("clipboard-serve".into())
        .spawn(move || {
            // A target that closes its read end early makes the write fail
            // (EPIPE), which is reported and otherwise harmless. The pipe closes
            // when fd drops, which is the target's end of data.
            if let Err(e) = ffi::write_all(fd.as_raw_fd(), &data) {
                eprintln!("bnknotes: clipboard serve failed: {e}");
            }
            drop(slot);
        });
    if let Err(e) = spawned {
        eprintln!("bnknotes: clipboard serve failed: {e}");
    }
}

/// One transfer's place in the in-flight count, given back when it drops: when the
/// transfer ends, or with the thread's closure if the thread never starts.
struct ServeSlot(Arc<AtomicUsize>);

impl ServeSlot {
    fn take(serving: &Arc<AtomicUsize>) -> Option<Self> {
        serving
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_SERVES).then_some(n + 1)
            })
            .ok()
            .map(|_| Self(Arc::clone(serving)))
    }
}

impl Drop for ServeSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use crate::app::clipboard::*;

    /// A copied image runs to megabytes, past a pipe's 64 KiB buffer, and the
    /// target may not read for a while. Serving must not wait for it, and the
    /// target must still get every byte and then the end of data.
    #[test]
    fn serving_a_large_selection_returns_at_once_and_delivers_it_whole() {
        let data: Arc<[u8]> = (0..1 << 20).map(|i| i as u8).collect();
        let (read_end, write_end) = ffi::pipe().expect("pipe");
        let serving = Arc::default();
        let started = std::time::Instant::now();
        serve(Arc::clone(&data), write_end, &serving);
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "serving does not block"
        );

        let (idle, total) = (Duration::from_secs(10), Duration::from_secs(30));
        let got = ffi::read_to_end(read_end.as_raw_fd(), 2 << 20, idle, total)
            .expect("the whole selection, then end of data");
        assert_eq!(&got[..], &data[..]);
    }

    #[test]
    fn transfers_past_the_cap_are_refused_until_one_ends() {
        let serving = Arc::default();
        let slots: Vec<_> = (0..MAX_SERVES)
            .map(|_| ServeSlot::take(&serving).expect("a free slot"))
            .collect();
        assert!(ServeSlot::take(&serving).is_none(), "the cap holds");
        drop(slots);
        assert!(
            ServeSlot::take(&serving).is_some(),
            "an ended transfer frees its slot"
        );
    }

    #[test]
    fn plain_text_is_told_from_the_markup_beside_an_image() {
        // What decides whether a paste takes the picture or the words: a source
        // offering plain text was copied as text, one offering only markup beside an
        // image was not.
        assert!(is_plain_text_mime("text/plain"));
        assert!(is_plain_text_mime("text/plain;charset=utf-8"));
        assert!(is_plain_text_mime("UTF8_STRING"));
        assert!(!is_plain_text_mime("text/html"));
        assert!(!is_plain_text_mime("image/png"));
        // Every plain flavor is still a text flavor.
        assert!(CLIPBOARD_MIMES.iter().all(|m| is_plain_text_mime(m)));
        assert!(CLIPBOARD_MIMES.iter().all(|m| is_text_mime(m)));
    }

    #[test]
    fn rich_text_is_text_but_not_plain_text() {
        // text/html still pastes as text when it is all there is, so it must stay a
        // text mime while losing to an image.
        assert!(is_text_mime("text/html"));
        assert!(!is_plain_text_mime("text/html"));
    }

    /// A scratch attachment folder, unique per test name.
    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-paste-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn an_attachment_is_named_pasted_image_and_a_stamp() {
        let dir = tmp_dir("name");
        let (name, path) =
            free_attachment_name(&dir, "20260717041530", "png").expect("a free name");
        assert_eq!(name, "Pasted image 20260717041530.png");
        assert_eq!(path, dir.join(&name));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_paste_in_the_same_second_takes_a_counter() {
        // The stamp only counts to the second, so two quick pastes would otherwise
        // write the same name twice and the first picture would be gone.
        let dir = tmp_dir("counter");
        std::fs::write(dir.join("Pasted image 20260717041530.png"), b"first").expect("write");
        let (name, _) = free_attachment_name(&dir, "20260717041530", "png").expect("a free name");
        assert_eq!(name, "Pasted image 20260717041530 1.png");

        std::fs::write(dir.join(&name), b"second").expect("write");
        let (name, _) = free_attachment_name(&dir, "20260717041530", "png").expect("a free name");
        assert_eq!(name, "Pasted image 20260717041530 2.png");
        // The first paste is still there under its own name.
        assert_eq!(
            std::fs::read(dir.join("Pasted image 20260717041530.png")).expect("read"),
            b"first"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_name_carries_the_images_own_extension() {
        let dir = tmp_dir("ext");
        let (name, _) = free_attachment_name(&dir, "20260717041530", "jpg").expect("a free name");
        assert!(name.ends_with(".jpg"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_timestamp_is_fourteen_digits() {
        // YYYYMMDDHHMMSS, so attachments sort by when they were pasted.
        let stamp = timestamp().expect("the local time");
        assert_eq!(stamp.len(), 14);
        assert!(stamp.chars().all(|c| c.is_ascii_digit()));
        // A plausible year, zero-padded rather than truncated.
        let year: i32 = stamp[..4].parse().expect("a year");
        assert!((2000..2200).contains(&year), "year was {year}");
    }
}
