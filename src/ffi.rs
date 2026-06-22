//! Unsafe calls into libc, kept in one module with their safe wrappers. std
//! links libc, so these symbols resolve without a `#[link]` attribute; we
//! declare only the handful we need.
//!
//! Two jobs live here:
//!   1. shm backing memory: `memfd_create` + `ftruncate` + `mmap`.
//!   2. fd-passing on the Wayland socket: `sendmsg`/`recvmsg` with SCM_RIGHTS
//!      ancillary data, which is the only way to hand the compositor the shm
//!      pool fd (and, later, to receive the keyboard keymap fd).
//!
//! Struct layouts (`msghdr`, `iovec`, `cmsghdr`) mirror the Linux x86_64/arm64
//! ABI; this targets Linux/Wayland and nothing else.

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::mem;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// Constants (Linux generic ABI).
// ---------------------------------------------------------------------------

const PROT_READ: c_int = 0x1;
const PROT_WRITE: c_int = 0x2;
const MAP_SHARED: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x2;
const MFD_CLOEXEC: c_uint = 0x1;

const SOL_SOCKET: c_int = 1;
const SCM_RIGHTS: c_int = 1;
const MSG_NOSIGNAL: c_int = 0x4000;
const MSG_CMSG_CLOEXEC: c_int = 0x4000_0000;

/// Open the pipe with both ends close-on-exec.
const O_CLOEXEC: c_int = 0x8_0000;

const EINTR: c_int = 4;
/// A timed-out or would-block read (EAGAIN == EWOULDBLOCK on Linux).
const EAGAIN: c_int = 11;

// ---------------------------------------------------------------------------
// C struct layouts.
// ---------------------------------------------------------------------------

#[repr(C)]
struct IoVec {
    iov_base: *mut c_void,
    iov_len: usize,
}

#[repr(C)]
struct MsgHdr {
    msg_name: *mut c_void,
    msg_namelen: c_uint,
    // repr(C) inserts 4 bytes of padding here to 8-align the pointer below,
    // matching the C layout.
    msg_iov: *mut IoVec,
    msg_iovlen: usize,
    msg_control: *mut c_void,
    msg_controllen: usize,
    msg_flags: c_int,
}

#[repr(C)]
struct CmsgHdr {
    cmsg_len: usize,
    cmsg_level: c_int,
    cmsg_type: c_int,
}

// ---------------------------------------------------------------------------
// FFI declarations.
// ---------------------------------------------------------------------------

extern "C" {
    fn memfd_create(name: *const c_char, flags: c_uint) -> c_int;
    fn ftruncate(fd: c_int, length: i64) -> c_int;
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn sendmsg(sockfd: c_int, msg: *const MsgHdr, flags: c_int) -> isize;
    fn recvmsg(sockfd: c_int, msg: *mut MsgHdr, flags: c_int) -> isize;
    fn pipe2(pipefd: *mut c_int, flags: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn __errno_location() -> *mut c_int;
}

fn errno() -> c_int {
    // SAFETY: glibc/musl both expose a valid thread-local errno here.
    unsafe { *__errno_location() }
}

// ---------------------------------------------------------------------------
// shm backing memory.
// ---------------------------------------------------------------------------

/// Create an anonymous, CLOEXEC memory file of `len` bytes. The caller mmaps it
/// and hands the fd to the compositor via `create_pool`.
pub fn memfd(len: usize) -> Result<OwnedFd> {
    let name = b"bnknotes\0";
    // SAFETY: name is a valid NUL-terminated C string; the call only reads it.
    let raw = unsafe { memfd_create(name.as_ptr() as *const c_char, MFD_CLOEXEC) };
    if raw < 0 {
        return Err(Error::msg(format!(
            "memfd_create failed: errno {}",
            errno()
        )));
    }
    // SAFETY: raw is a fresh, owned fd from memfd_create.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    // SAFETY: fd is valid; ftruncate only resizes the backing file.
    let rc = unsafe { ftruncate(fd.as_raw_fd(), len as i64) };
    if rc < 0 {
        return Err(Error::msg(format!("ftruncate failed: errno {}", errno())));
    }
    Ok(fd)
}

/// Map `len` bytes of `fd` shared and writable. Returns the base pointer.
pub fn mmap_shared(fd: RawFd, len: usize) -> Result<*mut u8> {
    // SAFETY: fd is a valid shm fd sized to at least len; a fresh anonymous
    // mapping is requested (addr null).
    let p = unsafe {
        mmap(
            core::ptr::null_mut(),
            len,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd,
            0,
        )
    };
    if p as isize == -1 {
        return Err(Error::msg(format!("mmap failed: errno {}", errno())));
    }
    Ok(p as *mut u8)
}

/// Map `len` bytes of `fd` private and read-only. Used for the keyboard keymap
/// the compositor sends over wl_keyboard.keymap.
pub fn mmap_ro(fd: RawFd, len: usize) -> Result<*mut u8> {
    // SAFETY: a fresh anonymous read-only private mapping of a received fd is
    // sound; failure is checked below.
    let p = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
    if p as isize == -1 {
        return Err(Error::msg(format!("mmap (ro) failed: errno {}", errno())));
    }
    Ok(p as *mut u8)
}

/// Unmap a mapping previously returned by [`mmap_shared`] or [`mmap_ro`].
///
/// # Safety
/// `ptr`/`len` must come from a live mapping that is not used again afterwards.
pub unsafe fn unmap(ptr: *mut u8, len: usize) {
    // SAFETY: forwarded contract from the caller.
    unsafe {
        munmap(ptr as *mut c_void, len);
    }
}

// ---------------------------------------------------------------------------
// SCM_RIGHTS ancillary-data plumbing.
// ---------------------------------------------------------------------------

const CMSG_CAP: usize = 256;

/// A control-message buffer, over-aligned so a `cmsghdr` can be written at its
/// start. 256 bytes holds well over a dozen fds, far more than we ever pass.
#[repr(C, align(8))]
struct CmsgBuf([u8; CMSG_CAP]);

impl CmsgBuf {
    fn zeroed() -> Self {
        CmsgBuf([0; CMSG_CAP])
    }
}

const fn cmsg_align(len: usize) -> usize {
    let a = mem::size_of::<usize>();
    (len + a - 1) & !(a - 1)
}

const fn cmsg_space(data_len: usize) -> usize {
    cmsg_align(mem::size_of::<CmsgHdr>()) + cmsg_align(data_len)
}

const fn cmsg_len(data_len: usize) -> usize {
    cmsg_align(mem::size_of::<CmsgHdr>()) + data_len
}

/// Send `bytes` on `sock`, attaching `fds` as SCM_RIGHTS ancillary data on the
/// first segment. Handles short writes (the kernel may not take everything at
/// once) and EINTR. The fds ride only on the first `sendmsg`, which is where
/// the bytes that reference them begin.
pub fn send_with_fds(sock: RawFd, bytes: &[u8], fds: &[RawFd]) -> Result<()> {
    let data_len = mem::size_of_val(fds);
    if cmsg_space(data_len) > CMSG_CAP {
        return Err(Error::msg("too many fds for one control message"));
    }

    let mut sent = 0usize;
    let mut first = true;
    while sent < bytes.len() {
        let iov = IoVec {
            iov_base: bytes[sent..].as_ptr() as *mut c_void,
            iov_len: bytes.len() - sent,
        };

        let mut cbuf = CmsgBuf::zeroed();
        let (control, controllen) = if first && !fds.is_empty() {
            let hdr = CmsgHdr {
                cmsg_len: cmsg_len(data_len),
                cmsg_level: SOL_SOCKET,
                cmsg_type: SCM_RIGHTS,
            };
            // SAFETY: cbuf is 8-aligned and large enough (checked above); we
            // write the header then each fd into the data area.
            unsafe {
                core::ptr::write(cbuf.0.as_mut_ptr() as *mut CmsgHdr, hdr);
                let data = cbuf
                    .0
                    .as_mut_ptr()
                    .add(cmsg_align(mem::size_of::<CmsgHdr>()));
                for (i, &fd) in fds.iter().enumerate() {
                    core::ptr::write_unaligned(
                        data.add(i * mem::size_of::<RawFd>()) as *mut RawFd,
                        fd,
                    );
                }
            }
            (cbuf.0.as_mut_ptr() as *mut c_void, cmsg_space(data_len))
        } else {
            (core::ptr::null_mut(), 0)
        };

        let msg = MsgHdr {
            msg_name: core::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &iov as *const IoVec as *mut IoVec,
            msg_iovlen: 1,
            msg_control: control,
            msg_controllen: controllen,
            msg_flags: 0,
        };

        // SAFETY: msg points at a fully initialized header whose iov/control
        // buffers outlive the call.
        let n = unsafe { sendmsg(sock, &msg, MSG_NOSIGNAL) };
        if n < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            return Err(Error::msg(format!("sendmsg failed: errno {e}")));
        }
        sent += n as usize;
        first = false;
    }
    Ok(())
}

/// Receive into `buf`, appending any fds carried as SCM_RIGHTS ancillary data
/// to `out`. Returns `Some(n)` data bytes (0 means the peer closed), or `None`
/// if the socket's read timeout elapsed with no data (EAGAIN), which the caller
/// uses to drive key-repeat timing.
pub fn recv_with_fds(sock: RawFd, buf: &mut [u8], out: &mut Vec<OwnedFd>) -> Result<Option<usize>> {
    loop {
        let iov = IoVec {
            iov_base: buf.as_mut_ptr() as *mut c_void,
            iov_len: buf.len(),
        };
        let mut cbuf = CmsgBuf::zeroed();
        let mut msg = MsgHdr {
            msg_name: core::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &iov as *const IoVec as *mut IoVec,
            msg_iovlen: 1,
            msg_control: cbuf.0.as_mut_ptr() as *mut c_void,
            msg_controllen: CMSG_CAP,
            msg_flags: 0,
        };

        // SAFETY: msg points at an initialized header with live iov/control
        // buffers; recvmsg fills them and updates msg_controllen.
        let n = unsafe { recvmsg(sock, &mut msg, MSG_CMSG_CLOEXEC) };
        if n < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            if e == EAGAIN {
                return Ok(None);
            }
            return Err(Error::msg(format!("recvmsg failed: errno {e}")));
        }
        parse_cmsgs(&cbuf, msg.msg_controllen, out);
        return Ok(Some(n as usize));
    }
}

/// Create a close-on-exec pipe, returning `(read_end, write_end)`. Used to
/// transfer clipboard data to and from the compositor.
pub fn pipe() -> Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0 as c_int; 2];
    // SAFETY: fds is a live 2-element array the call fills in.
    let rc = unsafe { pipe2(fds.as_mut_ptr(), O_CLOEXEC) };
    if rc < 0 {
        return Err(Error::msg(format!("pipe2 failed: errno {}", errno())));
    }
    // SAFETY: both fds are fresh and owned.
    let read_end = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write_end = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read_end, write_end))
}

/// Write every byte of `bytes` to `fd`, retrying short writes and EINTR.
pub fn write_all(fd: RawFd, bytes: &[u8]) -> Result<()> {
    let mut sent = 0usize;
    while sent < bytes.len() {
        // SAFETY: the slice from `sent` is valid for `len - sent` bytes.
        let n = unsafe {
            write(
                fd,
                bytes[sent..].as_ptr() as *const c_void,
                bytes.len() - sent,
            )
        };
        if n < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            return Err(Error::msg(format!("write failed: errno {e}")));
        }
        if n == 0 {
            break;
        }
        sent += n as usize;
    }
    Ok(())
}

/// Read `fd` to end-of-file, returning all bytes. Used to pull a paste payload
/// out of the pipe the compositor writes into.
pub fn read_to_end(fd: RawFd) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        // SAFETY: buf is a live writable array of buf.len() bytes.
        let n = unsafe { read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            return Err(Error::msg(format!("read failed: errno {e}")));
        }
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n as usize]);
    }
    Ok(out)
}

/// Walk the control buffer and collect every fd from SCM_RIGHTS messages.
fn parse_cmsgs(cbuf: &CmsgBuf, controllen: usize, out: &mut Vec<OwnedFd>) {
    let hdr_sz = mem::size_of::<CmsgHdr>();
    let base = cbuf.0.as_ptr();
    let mut off = 0usize;
    while off + hdr_sz <= controllen {
        // SAFETY: off + hdr_sz is within the control buffer.
        let hdr: CmsgHdr = unsafe { core::ptr::read_unaligned(base.add(off) as *const CmsgHdr) };
        let len = hdr.cmsg_len;
        if len < hdr_sz || off + len > controllen {
            break;
        }
        if hdr.cmsg_level == SOL_SOCKET && hdr.cmsg_type == SCM_RIGHTS {
            let data_off = off + cmsg_align(hdr_sz);
            let data_len = len - cmsg_align(hdr_sz);
            for i in 0..(data_len / mem::size_of::<RawFd>()) {
                // SAFETY: data_off + i*4 stays within the cmsg data region.
                let fd = unsafe {
                    core::ptr::read_unaligned(
                        base.add(data_off + i * mem::size_of::<RawFd>()) as *const RawFd
                    )
                };
                // SAFETY: the kernel just handed us this fd via SCM_RIGHTS.
                out.push(unsafe { OwnedFd::from_raw_fd(fd) });
            }
        }
        let adv = cmsg_align(len);
        if adv == 0 {
            break;
        }
        off += adv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmsg_math_matches_one_fd() {
        // On a 64-bit ABI: aligned header is 16, CMSG_LEN(4)=20, CMSG_SPACE(4)=24.
        assert_eq!(cmsg_align(mem::size_of::<CmsgHdr>()), 16);
        assert_eq!(cmsg_len(4), 20);
        assert_eq!(cmsg_space(4), 24);
    }

    #[test]
    fn memfd_maps_and_is_writable() {
        let len = 4096;
        let fd = memfd(len).expect("memfd");
        let ptr = mmap_shared(fd.as_raw_fd(), len).expect("mmap");
        // SAFETY: ptr maps len writable bytes.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        slice[0] = 0xAB;
        slice[len - 1] = 0xCD;
        assert_eq!(slice[0], 0xAB);
        assert_eq!(slice[len - 1], 0xCD);
        // SAFETY: ptr/len came from mmap_shared and are not used afterwards.
        unsafe { unmap(ptr, len) };
    }
}
