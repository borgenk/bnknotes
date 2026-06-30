//! A shared-memory pool: an anonymous memfd, mmapped writable, that the
//! compositor reads pixels from. One pool holds both double-buffered frames
//! (two full-window buffers, side by side); the editor paints whichever frame
//! the compositor has released while the other is being scanned out.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use crate::error::Result;
use crate::ffi;

pub struct Pool {
    fd: OwnedFd,
    ptr: *mut u8,
    len: usize,
}

impl Pool {
    /// Allocate and map a pool of `len` bytes.
    pub fn new(len: usize) -> Result<Self> {
        let fd = ffi::memfd(len)?;
        let ptr = ffi::mmap_shared(fd.as_raw_fd(), len)?;
        Ok(Self { fd, ptr, len })
    }

    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// The mapped bytes, for the renderer to write pixels into.
    pub fn data_mut(&mut self) -> &mut [u8] {
        // SAFETY: ptr maps exactly len writable bytes for the life of self, and
        // the borrow ties the slice to that lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // SAFETY: ptr/len describe our live mapping; it is not used after this.
        // The fd is closed afterwards by OwnedFd's own Drop.
        unsafe { ffi::unmap(self.ptr, self.len) };
    }
}
