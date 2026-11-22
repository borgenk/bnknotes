//! The Wayland connection: owns the socket, buffers outgoing requests, and
//! parses framed messages out of the incoming byte stream.

use std::collections::VecDeque;
use std::os::fd::{AsRawFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

/// Most bytes of requests the socket has not taken. Only a compositor that stops
/// reading fills it; the request that does not fit is lost, so every flush after
/// it fails and the editor ends rather than send a stream with a gap in it.
const OUT_CAP: usize = 64 * 1024;

pub struct Connection {
    stream: UnixStream,
    /// Requests not yet written. The bytes before out_pos have gone; the rest wait
    /// for the socket to take them, whole and in order.
    out: Vec<u8>,
    out_pos: usize,
    /// Fds waiting to go, each with the offset in out of the request that carries
    /// it. An fd goes with the first byte of its request, so a write starts at one
    /// fd's request and stops short of the next one's.
    out_fds: VecDeque<(usize, OwnedFd)>,
    /// Why the stream lost a request (the queue was full, or a send failed), after
    /// which every flush fails: what follows would reach the compositor with a gap.
    failed: Option<&'static str>,
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
            out_pos: 0,
            out_fds: VecDeque::new(),
            failed: None,
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
        self.queue(object, opcode, args);
    }

    /// Queue a request carrying fd and start sending. The queue holds its own copy
    /// of fd until the request goes, so the caller may close theirs on return.
    pub fn request_with_fd(
        &mut self,
        object: u32,
        opcode: u16,
        args: &[Arg],
        fd: RawFd,
    ) -> Result<()> {
        // SAFETY: the caller's fd is open for this call; it is only duplicated.
        let copy = unsafe { BorrowedFd::borrow_raw(fd) }
            .try_clone_to_owned()
            .map_err(|e| Error::msg(format!("duplicate an fd to send: {e}")))?;
        if let Some(at) = self.queue(object, opcode, args) {
            self.out_fds.push_back((at, copy));
        }
        self.flush()
    }

    /// Append a request to the queue, returning its offset, or None once the queue
    /// is full.
    fn queue(&mut self, object: u32, opcode: u16, args: &[Arg]) -> Option<usize> {
        if self.failed.is_some() {
            return None;
        }
        self.compact_out();
        let at = self.out.len();
        wire::encode(&mut self.out, object, opcode, args);
        if self.out.len() > OUT_CAP {
            self.out.truncate(at);
            self.failed = Some("wayland send buffer full: the compositor stopped reading");
            return None;
        }
        Some(at)
    }

    /// Whether requests are waiting for the socket to take them.
    pub fn has_pending_output(&self) -> bool {
        self.out_pos < self.out.len()
    }

    /// Write as much of the queue as the socket takes without waiting. What it
    /// cannot take stays queued, and the next flush picks up where this one
    /// stopped; the event loop flushes again once the socket has room.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(why) = self.failed {
            return Err(Error::msg(why));
        }
        while self.out_pos < self.out.len() {
            let pos = self.out_pos;
            let riding = (self.out_fds.front())
                .filter(|(at, _)| *at == pos)
                .map(|(_, fd)| fd.as_raw_fd());
            let end = (self.out_fds.iter())
                .map(|(at, _)| *at)
                .find(|&at| at > pos)
                .unwrap_or(self.out.len());
            let bytes = self.out.get(pos..end).unwrap_or_default();
            match ffi::send_some(self.stream.as_raw_fd(), bytes, riding.as_slice()) {
                Ok(Some(n)) if n > 0 => {
                    self.out_pos += n;
                    // The fd went with the first byte; the queue's copy closes.
                    if riding.is_some() {
                        self.out_fds.pop_front();
                    }
                }
                Ok(_) => return Ok(()),
                Err(e) => {
                    // What was queued is lost, perhaps a request cut short, so
                    // nothing more may follow it.
                    self.out.clear();
                    self.out_pos = 0;
                    self.out_fds.clear();
                    self.failed = Some("an earlier wayland send failed");
                    return Err(self.diagnose_send_error(e));
                }
            }
        }
        self.out.clear();
        self.out_pos = 0;
        Ok(())
    }

    /// Flush, waiting up to limit for the socket to take everything queued: for a
    /// request whose answer the caller waits on next. Whether all of it went.
    pub fn flush_within(&mut self, limit: Duration) -> Result<bool> {
        let deadline = Instant::now() + limit;
        loop {
            self.flush()?;
            if !self.has_pending_output() {
                return Ok(true);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || !ffi::wait_writable(self.fd(), left) {
                return Ok(false);
            }
        }
    }

    /// Move the unsent tail of the queue to the front, reclaiming the room the
    /// sent part took.
    fn compact_out(&mut self) {
        let pos = self.out_pos;
        if pos == 0 {
            return;
        }
        self.out.drain(..pos);
        for (at, _) in &mut self.out_fds {
            *at -= pos;
        }
        self.out_pos = 0;
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
    use std::io::{Read, Write};

    use crate::platform::conn::*;

    /// Queue numbered requests one at a time, flushing after each, until a flush
    /// leaves some queued: the socket is full. Returns the bytes queued.
    fn fill_socket(conn: &mut Connection) -> Vec<u8> {
        let mut expected = Vec::new();
        // A kilobyte each, so the socket fills quickly.
        let pad = "x".repeat(1000);
        for i in 0u32.. {
            let args = [Arg::Uint(i), Arg::Str(&pad)];
            conn.request(7, 1, &args);
            wire::encode(&mut expected, 7, 1, &args);
            conn.flush().expect("a flush that does not wait");
            if conn.has_pending_output() {
                break;
            }
            assert!(i < 100_000, "the socket never filled");
        }
        expected
    }

    /// Read what the peer has now, without waiting and up to limit bytes in all,
    /// with any fds it carried and the byte offset each chunk started at.
    fn read_now(
        peer: &UnixStream,
        limit: usize,
        got: &mut Vec<u8>,
        fds: &mut Vec<(usize, OwnedFd)>,
    ) {
        peer.set_nonblocking(true).expect("non-blocking peer");
        let mut buf = [0u8; 65536];
        loop {
            let room = limit.saturating_sub(got.len()).min(buf.len());
            if room == 0 {
                return;
            }
            let mut arrived = Vec::new();
            let start = got.len();
            match ffi::recv_with_fds(peer.as_raw_fd(), &mut buf[..room], &mut arrived) {
                Ok(Some(n)) if n > 0 => {
                    got.extend_from_slice(&buf[..n]);
                    fds.extend(arrived.into_iter().map(|fd| (start, fd)));
                }
                _ => return,
            }
        }
    }

    #[test]
    fn a_full_socket_keeps_the_rest_queued_and_sends_it_in_order() {
        let (mut conn, peer) = connected();
        let expected = fill_socket(&mut conn);
        let (mut got, mut fds) = (Vec::new(), Vec::new());
        while conn.has_pending_output() {
            read_now(&peer, usize::MAX, &mut got, &mut fds);
            conn.flush().expect("flush the rest");
        }
        read_now(&peer, usize::MAX, &mut got, &mut fds);
        assert_eq!(got.len(), expected.len());
        assert!(got == expected, "every request, whole and in order");
    }

    #[test]
    fn an_fd_goes_with_its_own_request_behind_a_backlog() {
        let (mut conn, peer) = connected();
        let mut expected = fill_socket(&mut conn);
        let (read_end, write_end) = ffi::pipe().expect("a pipe");
        let at = expected.len();
        conn.request_with_fd(9, 2, &[Arg::Uint(42)], write_end.as_raw_fd())
            .expect("queued behind the backlog");
        wire::encode(&mut expected, 9, 2, &[Arg::Uint(42)]);
        let its_end = expected.len();
        conn.request(9, 3, &[]);
        wire::encode(&mut expected, 9, 3, &[]);
        // The queue holds its own copy, so the caller's can close at once.
        drop(write_end);

        // Read exactly to the end of the fd's own request: the fd must be here by
        // then, not with a later one.
        let (mut got, mut fds) = (Vec::new(), Vec::new());
        while got.len() < its_end {
            read_now(&peer, its_end, &mut got, &mut fds);
            conn.flush().expect("flush the rest");
        }
        let [(start, fd)] = <[_; 1]>::try_from(fds).expect("the fd came with its request");
        assert!(start <= at, "with its request's first byte or before");
        let mut rest = Vec::new();
        while conn.has_pending_output() {
            read_now(&peer, usize::MAX, &mut got, &mut rest);
            conn.flush().expect("flush the rest");
        }
        read_now(&peer, usize::MAX, &mut got, &mut rest);
        assert!(rest.is_empty(), "and no other fd");
        assert!(got == expected, "every request, whole and in order");

        // It is the pipe's write end, and with the peer's copy closed the pipe
        // ends: the queue let go of its own once sent.
        let mut sent = std::fs::File::from(fd);
        sent.write_all(b"hi")
            .expect("write through the received fd");
        drop(sent);
        let mut text = String::new();
        std::fs::File::from(read_end)
            .read_to_string(&mut text)
            .expect("read to the end");
        assert_eq!(text, "hi");
    }

    #[test]
    fn a_flush_within_a_limit_gives_up_on_a_socket_with_no_room() {
        let (mut conn, peer) = connected();
        fill_socket(&mut conn);
        let limit = Duration::from_millis(20);
        let started = Instant::now();
        assert!(!conn.flush_within(limit).expect("no error, just no room"));
        assert!(started.elapsed() >= limit, "it waited out the limit");
        let (mut got, mut fds) = (Vec::new(), Vec::new());
        read_now(&peer, usize::MAX, &mut got, &mut fds);
        assert!(
            conn.flush_within(Duration::from_secs(5)).expect("flushes"),
            "a peer that reads makes room"
        );
    }

    #[test]
    fn a_queue_past_its_cap_fails_every_flush_after() {
        let (mut conn, peer) = connected();
        fill_socket(&mut conn);
        let pad = "x".repeat(1000);
        for i in 0..(OUT_CAP / 1000 + 2) as u32 {
            conn.request(7, 1, &[Arg::Uint(i), Arg::Str(&pad)]);
        }
        let err = conn.flush().expect_err("a request was dropped");
        assert!(err.to_string().contains("stopped reading"), "{err}");
        let (mut got, mut fds) = (Vec::new(), Vec::new());
        read_now(&peer, usize::MAX, &mut got, &mut fds);
        assert!(
            conn.flush().is_err(),
            "the stream has a gap, so it stays failed"
        );
        assert!(conn.request_with_fd(9, 2, &[], peer.as_raw_fd()).is_err());
    }

    fn connected() -> (Connection, UnixStream) {
        let (ours, theirs) = UnixStream::pair().expect("socket pair");
        let conn = Connection {
            stream: ours,
            out: Vec::new(),
            out_pos: 0,
            out_fds: VecDeque::new(),
            failed: None,
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
