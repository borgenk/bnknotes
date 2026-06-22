//! The Wayland connection: owns the socket, buffers outgoing requests, and
//! parses framed messages out of the incoming byte stream.

use std::collections::VecDeque;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::ffi;
use crate::wire::{self, Arg, Message};

/// The outcome of a [`Connection::fill`]: bytes arrived (0 means the peer
/// closed), or the read timed out with the socket idle.
pub enum Fill {
    Bytes(usize),
    TimedOut,
}

pub struct Connection {
    stream: UnixStream,
    out: Vec<u8>,
    in_buf: Vec<u8>,
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
            fds: VecDeque::new(),
        })
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
        ffi::send_with_fds(self.stream.as_raw_fd(), &msg, &[fd])
    }

    /// Write all queued requests to the socket.
    pub fn flush(&mut self) -> Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        ffi::send_with_fds(self.stream.as_raw_fd(), &self.out, &[])?;
        self.out.clear();
        Ok(())
    }

    /// Wait up to `timeout` (or forever if `None`) for more bytes from the
    /// compositor. Any fds received as ancillary data (the keyboard keymap, a
    /// clipboard transfer fd) are queued for [`take_fd`]. A `TimedOut` result
    /// lets the caller service key repeats while the socket is idle.
    pub fn fill(&mut self, timeout: Option<Duration>) -> Result<Fill> {
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
        if self.in_buf.len() < 8 {
            return Ok(None);
        }
        let object = u32::from_ne_bytes([
            self.in_buf[0],
            self.in_buf[1],
            self.in_buf[2],
            self.in_buf[3],
        ]);
        let word = u32::from_ne_bytes([
            self.in_buf[4],
            self.in_buf[5],
            self.in_buf[6],
            self.in_buf[7],
        ]);
        let size = (word >> 16) as usize;
        let opcode = (word & 0xffff) as u16;
        if size < 8 {
            return Err(Error::msg("malformed wayland message: size < 8"));
        }
        if self.in_buf.len() < size {
            return Ok(None);
        }
        let body = self.in_buf[8..size].to_vec();
        self.in_buf.drain(0..size);
        Ok(Some(Message {
            object,
            opcode,
            body,
        }))
    }
}
