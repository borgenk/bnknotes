//! The Wayland connection: owns the socket, buffers outgoing requests, and
//! parses framed messages out of the incoming byte stream.

use std::collections::VecDeque;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::platform::error::{Error, Result};
use crate::platform::ffi;
use crate::platform::protocol::{WL_DISPLAY, wl_display};
use crate::platform::wire::{self, Arg, Message, Reader};

/// The outcome of a [`Connection::fill`]: bytes arrived (0 means the peer
/// closed), or the read timed out with the socket idle.
pub enum Fill {
    Bytes(usize),
    TimedOut,
}

/// The fixed 8-byte header at the front of every Wayland message.
struct FrameHeader {
    object: u32,
    opcode: u16,
    /// Total message length in bytes, this header included.
    size: usize,
}

/// Parse the [`FrameHeader`] from the front of `buf`. The caller checks `buf`
/// holds the 8-byte header before trusting `size`; a shorter buffer reads as
/// zeroes rather than panicking.
fn frame_header(buf: &[u8]) -> FrameHeader {
    let word_at = |i: usize| {
        u32::from_ne_bytes(
            buf.get(i..i + 4)
                .and_then(|b| b.try_into().ok())
                .unwrap_or([0; 4]),
        )
    };
    let word = word_at(4);
    FrameHeader {
        object: word_at(0),
        opcode: (word & 0xffff) as u16,
        size: (word >> 16) as usize,
    }
}

/// The message at the front of buf, as its header and body, or None until all of
/// it has arrived.
fn frame(buf: &[u8]) -> Result<Option<(FrameHeader, &[u8])>> {
    if buf.len() < 8 {
        return Ok(None);
    }
    let header = frame_header(buf);
    if header.size < 8 {
        return Err(Error::msg("malformed wayland message: size < 8"));
    }
    Ok(buf.get(8..header.size).map(|body| (header, body)))
}

/// The error a wl_display.error event's body reports.
pub fn protocol_error(body: &[u8]) -> Error {
    let mut r = Reader::new(body);
    match (r.u32(), r.u32(), r.string()) {
        (Ok(object), Ok(code), Ok(message)) => Error::msg(format!(
            "wayland protocol error: object {object} code {code}: {message}"
        )),
        _ => Error::msg("malformed wl_display.error event"),
    }
}

pub struct Connection {
    stream: UnixStream,
    out: Vec<u8>,
    in_buf: Vec<u8>,
    /// How many bytes at the front of `in_buf` have already been consumed by
    /// [`Connection::next_message`]. Each parsed message advances this cursor
    /// (cheap) instead of draining the front (O(remaining) per message, O(N^2)
    /// over a burst); the consumed prefix is compacted away once per [`Self::fill`].
    in_pos: usize,
    fds: VecDeque<OwnedFd>,
}

impl Connection {
    /// Connect to `$XDG_RUNTIME_DIR/$WAYLAND_DISPLAY` (default `wayland-0`). An
    /// absolute `WAYLAND_DISPLAY` is used as-is.
    pub fn connect() -> Result<Self> {
        let runtime = std::env::var_os("XDG_RUNTIME_DIR")
            .ok_or_else(|| Error::msg("XDG_RUNTIME_DIR is not set"))?;
        let display = std::env::var_os("WAYLAND_DISPLAY")
            .unwrap_or_else(|| std::ffi::OsString::from("wayland-0"));
        let path = if Path::new(&display).is_absolute() {
            PathBuf::from(display)
        } else {
            Path::new(&runtime).join(display)
        };
        let stream = UnixStream::connect(&path)
            .map_err(|e| Error::msg(format!("connect to {}: {e}", path.display())))?;
        Ok(Self {
            stream,
            out: Vec::new(),
            in_buf: Vec::new(),
            in_pos: 0,
            fds: VecDeque::new(),
        })
    }

    /// The socket's raw fd, so an event loop can `poll` it alongside other fds.
    /// Read-only: the connection keeps ownership and does all the actual I/O.
    pub fn fd(&self) -> RawFd {
        self.stream.as_raw_fd()
    }

    /// Queue a request for the next flush.
    pub fn request(&mut self, object: u32, opcode: u16, args: &[Arg]) {
        wire::encode(&mut self.out, object, opcode, args);
    }

    /// Send everything queued so far, then send one request carrying `fd` as
    /// ancillary data. Flushing first keeps the fd attached to the request that
    /// references it, with no earlier bytes riding along.
    pub fn request_with_fd(
        &mut self,
        object: u32,
        opcode: u16,
        args: &[Arg],
        fd: RawFd,
    ) -> Result<()> {
        self.flush()?;
        let mut msg = Vec::new();
        wire::encode(&mut msg, object, opcode, args);
        if let Err(e) = ffi::send_with_fds(self.stream.as_raw_fd(), &msg, &[fd]) {
            return Err(self.diagnose_send_error(e));
        }
        Ok(())
    }

    /// Write all queued requests to the socket.
    pub fn flush(&mut self) -> Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        if let Err(e) = ffi::send_with_fds(self.stream.as_raw_fd(), &self.out, &[]) {
            return Err(self.diagnose_send_error(e));
        }
        self.out.clear();
        Ok(())
    }

    /// A send failed (typically `EPIPE`): the compositor closed the socket. On a
    /// protocol violation it sends `wl_display.error` *then* closes, and because
    /// we were writing we never read it, so the raw errno only says "peer gone".
    /// Drain whatever it sent before disconnecting and surface that error, which
    /// is the real reason. Best-effort: on failure fall back to the send error.
    fn diagnose_send_error(&mut self, send_err: Error) -> Error {
        let _ = self
            .stream
            .set_read_timeout(Some(Duration::from_millis(200)));
        let mut buf = [0u8; 4096];
        loop {
            let mut fds: Vec<OwnedFd> = Vec::new();
            match ffi::recv_with_fds(self.stream.as_raw_fd(), &mut buf, &mut fds) {
                Ok(Some(n)) if n > 0 => self.in_buf.extend_from_slice(&buf[..n]),
                // 0 bytes = peer closed, None = idle/timeout, Err = reset: stop.
                _ => break,
            }
        }
        // Look without consuming: the send error may not be fatal, and then every
        // buffered event is still to be handled.
        let mut rest = self.in_buf.get(self.in_pos..).unwrap_or_default();
        while let Ok(Some((header, body))) = frame(rest) {
            if header.object == WL_DISPLAY && header.opcode == wl_display::EV_ERROR {
                return protocol_error(body);
            }
            rest = rest.get(header.size..).unwrap_or_default();
        }
        Error::msg(format!(
            "{send_err}; the compositor closed the connection without sending a protocol error"
        ))
    }

    /// Wait up to `timeout` (or forever if `None`) for more bytes from the
    /// compositor. Any fds received as ancillary data (the keyboard keymap, a
    /// clipboard transfer fd) are queued for [`Self::take_fd`]. A `TimedOut` result
    /// lets the caller service key repeats while the socket is idle.
    pub fn fill(&mut self, timeout: Option<Duration>) -> Result<Fill> {
        // Drop the messages consumed since the last fill in one shift, rather
        // than draining the front per message in `next_message`.
        self.compact();
        self.stream
            .set_read_timeout(timeout)
            .map_err(|e| Error::msg(format!("set_read_timeout: {e}")))?;
        let mut buf = [0u8; 4096];
        let mut fds: Vec<OwnedFd> = Vec::new();
        match ffi::recv_with_fds(self.stream.as_raw_fd(), &mut buf, &mut fds)? {
            Some(n) => {
                self.in_buf.extend_from_slice(&buf[..n]);
                self.fds.extend(fds);
                Ok(Fill::Bytes(n))
            }
            None => Ok(Fill::TimedOut),
        }
    }

    /// Pop the oldest received fd. Events carrying an fd are processed in
    /// order, so the front of the queue belongs to the next such event.
    pub fn take_fd(&mut self) -> Option<OwnedFd> {
        self.fds.pop_front()
    }

    /// Pull one fully buffered message off the front of the stream, if present.
    pub fn next_message(&mut self) -> Result<Option<Message>> {
        let buf = self.in_buf.get(self.in_pos..).unwrap_or_default();
        let Some((header, body)) = frame(buf)? else {
            return Ok(None);
        };
        let message = Message::new(header.object, header.opcode, body);
        self.in_pos += header.size;
        Ok(Some(message))
    }

    /// Discard the already-consumed prefix of the input buffer in a single shift.
    /// Called once per [`Connection::fill`], so a burst of messages costs one
    /// compaction instead of one front-drain per message.
    fn compact(&mut self) {
        if self.in_pos == 0 {
            return;
        }
        self.in_buf.drain(..self.in_pos);
        self.in_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::platform::conn::*;

    fn connected() -> (Connection, UnixStream) {
        let (ours, theirs) = UnixStream::pair().expect("socket pair");
        let conn = Connection {
            stream: ours,
            out: Vec::new(),
            in_buf: Vec::new(),
            in_pos: 0,
            fds: VecDeque::new(),
        };
        (conn, theirs)
    }

    fn encoded(object: u32, opcode: u16, args: &[Arg]) -> Vec<u8> {
        let mut out = Vec::new();
        wire::encode(&mut out, object, opcode, args);
        out
    }

    #[test]
    fn messages_come_out_whole_and_in_order() {
        let (mut conn, _peer) = connected();
        let third = encoded(7, 1, &[Arg::Uint(9)]);
        conn.in_buf.extend(encoded(5, 2, &[Arg::Uint(7)]));
        conn.in_buf.extend(encoded(6, 0, &[]));
        conn.in_buf.extend_from_slice(&third[..10]);

        let m = conn.next_message().unwrap().expect("first");
        assert_eq!(
            (m.object, m.opcode, m.body()),
            (5, 2, &7u32.to_ne_bytes()[..])
        );
        let m = conn.next_message().unwrap().expect("second");
        assert_eq!((m.object, m.opcode, m.body().len()), (6, 0, 0));
        assert!(
            conn.next_message().unwrap().is_none(),
            "the third has not all arrived"
        );

        conn.in_buf.extend_from_slice(&third[10..]);
        let m = conn.next_message().unwrap().expect("third");
        assert_eq!(
            (m.object, m.opcode, m.body()),
            (7, 1, &9u32.to_ne_bytes()[..])
        );
        assert!(conn.next_message().unwrap().is_none());
    }

    #[test]
    fn a_size_below_the_header_is_refused() {
        let (mut conn, _peer) = connected();
        conn.in_buf.extend(5u32.to_ne_bytes());
        conn.in_buf.extend((4u32 << 16).to_ne_bytes());
        assert!(conn.next_message().is_err());
    }

    #[test]
    fn a_failed_send_reports_the_protocol_error_and_keeps_the_events() {
        let (mut conn, mut peer) = connected();
        peer.write_all(&encoded(9, 3, &[])).unwrap();
        peer.write_all(&encoded(
            WL_DISPLAY,
            wl_display::EV_ERROR,
            &[Arg::Uint(9), Arg::Uint(2), Arg::Str("bad buffer")],
        ))
        .unwrap();
        drop(peer);

        let e = conn.diagnose_send_error(Error::msg("send failed"));
        assert_eq!(
            e.to_string(),
            "wayland protocol error: object 9 code 2: bad buffer"
        );
        let m = conn.next_message().unwrap().expect("the event before it");
        assert_eq!((m.object, m.opcode), (9, 3));
    }
}
