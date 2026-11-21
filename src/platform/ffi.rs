//! Unsafe calls into libc, kept in one module with their safe wrappers. std
//! links libc, so these symbols resolve without a `#[link]` attribute; we
//! declare only the handful we need.
//!
//! Five jobs live here:
//!   1. mapping the keyboard keymap fd the compositor sends: `mmap`/`munmap`.
//!   2. fd-passing on the Wayland socket: `sendmsg`/`recvmsg` with SCM_RIGHTS
//!      ancillary data, which is how the compositor hands us the keyboard
//!      keymap fd (and how we pass dmabuf and syncobj fds back).
//!   3. runtime library loading (`dlopen`/`dlsym`) for the Vulkan loader, so a
//!      machine without Vulkan reports a clean error instead of failing to link
//!      (a `#[link]` would make the loader a hard startup dependency).
//!   4. the dma-buf sync-file ioctls, the implicit-sync bridge between the
//!      GPU backend's explicit fences and the fences other dmabuf users (the
//!      compositor) observe. Kernel dma-buf core, driver-independent.
//!   5. local wall-clock time (`localtime_r`), for the timestamp a pasted image's
//!      filename carries: the zone rules are the system's, not ours to model.
//!
//! Struct layouts (`msghdr`, `iovec`, `cmsghdr`) mirror the Linux x86_64/arm64
//! ABI; this targets Linux/Wayland and nothing else.

use core::ffi::{CStr, c_char, c_int, c_long, c_short, c_uint, c_ulong, c_void};
use std::mem;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::time::Duration;

use crate::platform::error::{Error, Result};

// ---------------------------------------------------------------------------
// Constants (Linux generic ABI).
// ---------------------------------------------------------------------------

const PROT_READ: c_int = 0x1;
const MAP_PRIVATE: c_int = 0x2;

/// `lseek` whence: from the end of the file, i.e. seek(0) yields its size.
const SEEK_END: c_int = 2;

const SOL_SOCKET: c_int = 1;
const SCM_RIGHTS: c_int = 1;
const MSG_NOSIGNAL: c_int = 0x4000;
const MSG_CMSG_CLOEXEC: c_int = 0x4000_0000;
/// Set in `msg_flags` by `recvmsg` when the control buffer was too small to hold
/// all the ancillary data, so some passed fds were dropped.
const MSG_CTRUNC: c_int = 0x8;

/// Open the pipe with both ends close-on-exec.
const O_CLOEXEC: c_int = 0x8_0000;
/// Open a pipe end non-blocking, so a read that finds it empty returns rather than
/// waiting (the wake pipe, drained after poll reports it readable).
const O_NONBLOCK: c_int = 0x800;

const EINTR: c_int = 4;
/// A timed-out or would-block read (EAGAIN == EWOULDBLOCK on Linux).
const EAGAIN: c_int = 11;

/// Resolve all symbols at load time, so a broken library fails at `dlopen`
/// rather than at first call.
const RTLD_NOW: c_int = 2;

/// The ioctl or its flags are not supported (pre-6.0 kernel, or an exporter
/// without sync-file support); the sync bridge then degrades to CPU waits.
const ENOTTY: c_int = 25;
const EINVAL: c_int = 22;

/// `struct dma_buf_export_sync_file` / `struct dma_buf_import_sync_file`:
/// identical layout, flags plus a sync-file fd (out for export, in for import).
#[repr(C)]
struct DmaBufSyncFile {
    flags: u32,
    fd: c_int,
}

/// `DMA_BUF_SYNC_WRITE`: for export, "every fence a writer must wait on"
/// (readers and writers); for import, "this fence is a write" (readers and
/// writers must wait on it). The GPU backend is always the writer, so this is
/// the only flag it uses.
const DMA_BUF_SYNC_WRITE: u32 = 2;

/// `_IOWR('b', 2, struct dma_buf_export_sync_file)`.
const DMA_BUF_IOCTL_EXPORT_SYNC_FILE: c_ulong = 0xc008_6202;
/// `_IOW('b', 3, struct dma_buf_import_sync_file)`.
const DMA_BUF_IOCTL_IMPORT_SYNC_FILE: c_ulong = 0x4008_6203;

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

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    fn lseek(fd: c_int, offset: i64, whence: c_int) -> i64;
    fn sendmsg(sockfd: c_int, msg: *const MsgHdr, flags: c_int) -> isize;
    fn recvmsg(sockfd: c_int, msg: *mut MsgHdr, flags: c_int) -> isize;
    fn pipe2(pipefd: *mut c_int, flags: c_int) -> c_int;
    fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn __errno_location() -> *mut c_int;
    // In libc proper since glibc 2.34 (earlier glibc kept them in libdl, which
    // this dependency-free build does not link; the target systems are newer).
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    fn localtime_r(time: *const i64, result: *mut Tm) -> *mut Tm;
}

/// `struct tm`, the Linux/glibc layout. Only the civil fields below are read; the
/// rest are here to size the struct correctly, since `localtime_r` writes all of it.
#[repr(C)]
struct Tm {
    tm_sec: c_int,
    tm_min: c_int,
    tm_hour: c_int,
    tm_mday: c_int,
    tm_mon: c_int,
    tm_year: c_int,
    tm_wday: c_int,
    tm_yday: c_int,
    tm_isdst: c_int,
    tm_gmtoff: c_long,
    tm_zone: *const c_char,
}

/// A wall-clock date and time, broken into the parts a person reads. Fields are as
/// written rather than as C stores them: a full year, and months and days from 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalTime {
    pub year: i32,
    /// 1..=12.
    pub month: u32,
    /// 1..=31.
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

/// The local wall-clock time for a unix timestamp: the zone and its daylight rules
/// come from the system (`TZ`, else `/etc/localtime`), which is what makes a
/// timestamp written into a filename read as the time the user was at.
///
/// `None` when the conversion fails, which for a plausible timestamp it does not.
pub fn local_time(unix_seconds: i64) -> Option<LocalTime> {
    let mut tm = mem::MaybeUninit::<Tm>::uninit();
    // SAFETY: `localtime_r` fills the caller's `struct tm` and returns it, or null
    // on failure; the timestamp is passed by pointer and only read. On success the
    // struct is initialized, so reading it is defined.
    let tm = unsafe {
        if localtime_r(&unix_seconds, tm.as_mut_ptr()).is_null() {
            return None;
        }
        tm.assume_init()
    };
    Some(LocalTime {
        // C counts years from 1900 and months from 0; a reader does not.
        year: tm.tm_year.checked_add(1900)?,
        month: u32::try_from(tm.tm_mon.checked_add(1)?).ok()?,
        day: u32::try_from(tm.tm_mday).ok()?,
        hour: u32::try_from(tm.tm_hour).ok()?,
        minute: u32::try_from(tm.tm_min).ok()?,
        second: u32::try_from(tm.tm_sec).ok()?,
    })
}

fn errno() -> c_int {
    // SAFETY: glibc/musl both expose a valid thread-local errno here.
    unsafe { *__errno_location() }
}

/// `Ok(())` when the syscall return `rc` is zero, else an error naming `what`
/// with the current errno. The shared success-or-errno tail of the ioctl
/// wrappers below (the FFI analog of vulkan's `check`).
fn ok_or_errno(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::msg(format!("{what} failed: errno {}", errno())))
    }
}

// ---------------------------------------------------------------------------
// Waiting on several fds at once (the compositor socket + the inotify fd).
// ---------------------------------------------------------------------------

// Data to read, plus the error and hang-up conditions. The latter two also make
// a following read return at once (with the error or EOF), so the caller treats
// them as "ready, go find out" rather than a reason to keep waiting.
const POLLIN: c_short = 0x0001;
const POLLERR: c_short = 0x0008;
const POLLHUP: c_short = 0x0010;

#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: c_int,
    events: c_short,
    revents: c_short,
}

/// Block until one of `fds` is readable (or errored / hung up), or `timeout`
/// elapses; `None` waits indefinitely. Returns a bitmask whose bit `i` is set
/// when `fds[i]` is ready, so the caller drains exactly the ready ones. An
/// interrupting signal returns an empty mask (bit none set), which the event
/// loop treats as a timeout: it services its timers and polls again.
pub fn poll_readable(fds: &[RawFd], timeout: Option<Duration>) -> Result<u32> {
    assert!(fds.len() <= 32, "poll_readable handles at most 32 fds");
    let mut pollfds = [PollFd {
        fd: -1,
        events: 0,
        revents: 0,
    }; 32];
    for (slot, &fd) in pollfds.iter_mut().zip(fds) {
        slot.fd = fd;
        slot.events = POLLIN;
    }
    // poll's timeout is milliseconds, -1 for an infinite wait; clamp the
    // conversion so a very long duration cannot wrap the c_int.
    let ms = match timeout {
        None => -1,
        Some(d) => d.as_millis().min(c_int::MAX as u128) as c_int,
    };
    // SAFETY: pollfds is a valid array of fds.len() entries; poll reads events
    // and writes revents on the first nfds of them.
    let n = unsafe { poll(pollfds.as_mut_ptr(), fds.len() as c_ulong, ms) };
    if n < 0 {
        if errno() == EINTR {
            return Ok(0);
        }
        return Err(Error::msg(format!("poll failed: errno {}", errno())));
    }
    let mut ready = 0u32;
    for (i, slot) in pollfds.iter().take(fds.len()).enumerate() {
        if slot.revents & (POLLIN | POLLERR | POLLHUP) != 0 {
            ready |= 1 << i;
        }
    }
    Ok(ready)
}

// ---------------------------------------------------------------------------
// Memory mapping (the keyboard keymap fd).
// ---------------------------------------------------------------------------

/// Read `len` bytes a compositor sent over an fd (the keyboard keymap, the dmabuf
/// format table) into an owned `Vec`: map the fd read-only, copy the bytes out,
/// and release the mapping. The one region of `unsafe` needed to read a raw
/// mapping lives here, so callers deal only in `Vec<u8>`; `len` must be non-zero
/// (`mmap` rejects a zero length). Errors if the mapping fails.
///
/// `len` is always the *sender's* claim about a file we did not create, and mmap
/// will happily map past the end of a short one: the mapping succeeds, and the
/// first read of a page beyond the file's last raises SIGBUS, which no `Result`
/// can catch and no compositor is obliged to spare us. So the claim is measured
/// against the file before a byte of it is mapped.
pub fn read_mapped(fd: RawFd, len: usize) -> Result<Vec<u8>> {
    let size = file_size(fd)?;
    if len as u64 > size {
        return Err(Error::msg(format!(
            "sender claimed {len} bytes over an fd holding {size}"
        )));
    }
    let ptr = mmap_ro(fd, len)?;
    // SAFETY: mmap_ro returned a live mapping of `len` readable bytes; the slice
    // is copied into an owned Vec and the mapping is released before returning, so
    // the raw pointer never escapes this function.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec();
    // SAFETY: ptr/len came from mmap_ro just above and are not used again.
    unsafe { unmap(ptr, len) };
    Ok(bytes)
}

/// The size of the file behind `fd`, via a seek to its end. The fds this layer is
/// handed (memfds, shm segments) are seekable, and the offset is not used again:
/// [`mmap_ro`] maps from an explicit offset of 0.
fn file_size(fd: RawFd) -> Result<u64> {
    // SAFETY: `fd` is a live descriptor owned by the caller for this call; lseek
    // touches no memory.
    let end = unsafe { lseek(fd, 0, SEEK_END) };
    if end < 0 {
        return Err(Error::msg(format!("lseek failed: errno {}", errno())));
    }
    Ok(end as u64)
}

/// Map `len` bytes of `fd` private and read-only.
fn mmap_ro(fd: RawFd, len: usize) -> Result<*mut u8> {
    // SAFETY: a fresh anonymous read-only private mapping of a received fd is
    // sound; failure is checked below.
    let p = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
    if p as isize == -1 {
        return Err(Error::msg(format!("mmap (ro) failed: errno {}", errno())));
    }
    Ok(p as *mut u8)
}

/// Unmap a mapping previously returned by [`mmap_ro`].
///
/// # Safety
/// `ptr`/`len` must come from a live mapping that is not used again afterwards.
unsafe fn unmap(ptr: *mut u8, len: usize) {
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
    // The fds ride on the first segment's sendmsg; with no payload the send loop
    // below never runs and they would be dropped. Wayland always pairs an fd
    // with the (non-empty) request that references it, so assert rather than leak.
    debug_assert!(
        fds.is_empty() || !bytes.is_empty(),
        "send_with_fds called with fds but no bytes; the fds would be dropped"
    );
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
        // A full control buffer means the kernel truncated the ancillary data and
        // some passed fds were lost. CMSG_CAP holds far more than Wayland's 1-2
        // fds, so this is unreachable; surface it rather than proceed with a
        // half-received message if it ever fires.
        if msg.msg_flags & MSG_CTRUNC != 0 {
            return Err(Error::msg(
                "recvmsg truncated ancillary data (too many fds)",
            ));
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

/// Create a close-on-exec, non-blocking pipe, returning `(read_end, write_end)`.
/// A worker thread writes a byte to the write end to wake the event loop, which
/// polls the read end and drains it; the non-blocking read end lets [`drain`] read
/// once without risking a block on a spurious wake.
pub fn wake_pipe() -> Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0 as c_int; 2];
    // SAFETY: fds is a live 2-element array the call fills in.
    let rc = unsafe { pipe2(fds.as_mut_ptr(), O_CLOEXEC | O_NONBLOCK) };
    if rc < 0 {
        return Err(Error::msg(format!("pipe2 failed: errno {}", errno())));
    }
    // SAFETY: both fds are fresh and owned.
    let read_end = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let write_end = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    Ok((read_end, write_end))
}

/// Discard whatever is readable on `fd` in one read, for a wake pipe going from an
/// edge back to empty. The fd is non-blocking, so a read finding it already empty
/// (EAGAIN) is not an error.
pub fn drain(fd: RawFd) -> Result<()> {
    let mut buf = [0u8; 64];
    loop {
        // SAFETY: buf is a live writable array of buf.len() bytes.
        let n = unsafe { read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n < 0 {
            let e = errno();
            if e == EINTR {
                continue;
            }
            if e == EAGAIN {
                return Ok(());
            }
            return Err(Error::msg(format!("drain failed: errno {e}")));
        }
        // A short read leaves the pipe empty; a full one may have more queued.
        if (n as usize) < buf.len() {
            return Ok(());
        }
    }
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

/// Read `fd` to end-of-file, returning all bytes, or an error once they would pass
/// `cap`. Used to pull a paste payload out of the pipe the compositor writes into:
/// the payload is another app's, of a size it chooses, so the cap is the caller's
/// statement of what it is willing to hold.
pub fn read_to_end(fd: RawFd, cap: usize) -> Result<Vec<u8>> {
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
        // What the far end sends is its choice, not ours, so the read is bounded:
        // refuse past the cap rather than grow a buffer until the editor dies.
        if out.len() + n as usize > cap {
            return Err(Error::msg(format!("more than {cap} bytes to read")));
        }
        out.extend_from_slice(&buf[..n as usize]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// dma-buf sync-file bridging.
// ---------------------------------------------------------------------------

/// Export the fences a writer of `dmabuf` must wait on (the compositor's
/// in-flight reads and writes) as a sync file. `Ok(None)` means the kernel or
/// exporter does not support the ioctl, so the caller should wait on the CPU
/// instead. An already-idle dmabuf yields an already-signaled sync file, which
/// is fine to wait on.
pub fn dmabuf_export_sync_file(dmabuf: RawFd) -> Result<Option<OwnedFd>> {
    let mut arg = DmaBufSyncFile {
        flags: DMA_BUF_SYNC_WRITE,
        fd: -1,
    };
    loop {
        // SAFETY: arg is a live, correctly laid out dma_buf_export_sync_file;
        // the kernel fills in fd on success.
        let rc = unsafe { ioctl(dmabuf, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &mut arg) };
        if rc == 0 {
            // SAFETY: the kernel just returned a fresh sync-file fd we own.
            return Ok(Some(unsafe { OwnedFd::from_raw_fd(arg.fd) }));
        }
        let e = errno();
        if e == EINTR {
            continue;
        }
        if e == ENOTTY || e == EINVAL {
            return Ok(None);
        }
        return Err(Error::msg(format!(
            "DMA_BUF_IOCTL_EXPORT_SYNC_FILE failed: errno {e}"
        )));
    }
}

/// Attach `sync` (a sync-file fd) to `dmabuf` as a write fence, so every other
/// user of the dmabuf (the compositor) waits for it before reading. `Ok(false)`
/// means unsupported; the caller should fall back to a CPU wait before letting
/// anyone see the buffer.
pub fn dmabuf_import_sync_file(dmabuf: RawFd, sync: RawFd) -> Result<bool> {
    let arg = DmaBufSyncFile {
        flags: DMA_BUF_SYNC_WRITE,
        fd: sync,
    };
    loop {
        // SAFETY: arg is a live, correctly laid out dma_buf_import_sync_file
        // holding a valid sync-file fd; the kernel only reads it.
        let rc = unsafe { ioctl(dmabuf, DMA_BUF_IOCTL_IMPORT_SYNC_FILE, &arg) };
        if rc == 0 {
            return Ok(true);
        }
        let e = errno();
        if e == EINTR {
            continue;
        }
        if e == ENOTTY || e == EINVAL {
            return Ok(false);
        }
        return Err(Error::msg(format!(
            "DMA_BUF_IOCTL_IMPORT_SYNC_FILE failed: errno {e}"
        )));
    }
}

// ---------------------------------------------------------------------------
// DRM sync objects (explicit sync, wp_linux_drm_syncobj_v1).
// ---------------------------------------------------------------------------

/// `struct drm_syncobj_create`: the kernel writes `handle`.
#[repr(C)]
struct DrmSyncobjCreate {
    handle: u32,
    flags: u32,
}

/// `struct drm_syncobj_handle`: trades a syncobj `handle` for a shareable `fd`
/// (export) or the reverse. The kernel's `drm.h` carries the timeline `point`
/// field, so the struct is 24 bytes, not the classic 16; [`drm_iowr`] takes the
/// size from here so the ioctl encoding can never drift from the ABI.
#[repr(C)]
struct DrmSyncobjHandle {
    handle: u32,
    flags: u32,
    fd: c_int,
    pad: u32,
    point: u64,
}

/// `struct drm_syncobj_transfer`: copy the fence at (`src_handle`, `src_point`)
/// to (`dst_handle`, `dst_point`), moving between timeline points and the
/// point-0 of a binary syncobj.
#[repr(C)]
struct DrmSyncobjTransfer {
    src_handle: u32,
    dst_handle: u32,
    src_point: u64,
    dst_point: u64,
    flags: u32,
    pad: u32,
}

/// `DRM_SYNCOBJ_CREATE_SIGNALED`: start the object already signaled.
const DRM_SYNCOBJ_CREATE_SIGNALED: u32 = 1;
/// `DRM_SYNCOBJ_{HANDLE_TO_FD_FLAGS_EXPORT,FD_TO_HANDLE_FLAGS_IMPORT}_SYNC_FILE`:
/// read/write a sync file at the binary syncobj's point 0 rather than moving the
/// syncobj object itself. Both directions use the same flag value.
const DRM_SYNCOBJ_SYNC_FILE: u32 = 1;

/// `DRM_IOWR('d', nr, struct)`: the DRM ioctl request encoding. Deriving the
/// size from `size_of` means a struct edit updates the ioctl number with it.
const fn drm_iowr(nr: u32, size: usize) -> c_ulong {
    const DIR_READ_WRITE: c_ulong = 3 << 30;
    const DRM_TYPE: c_ulong = 0x64 << 8; // 'd'
    DIR_READ_WRITE | ((size as c_ulong) << 16) | DRM_TYPE | nr as c_ulong
}

const DRM_IOCTL_SYNCOBJ_CREATE: c_ulong = drm_iowr(0xBF, mem::size_of::<DrmSyncobjCreate>());
const DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD: c_ulong = drm_iowr(0xC1, mem::size_of::<DrmSyncobjHandle>());
const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: c_ulong = drm_iowr(0xC2, mem::size_of::<DrmSyncobjHandle>());
const DRM_IOCTL_SYNCOBJ_TRANSFER: c_ulong = drm_iowr(0xCC, mem::size_of::<DrmSyncobjTransfer>());

/// Open the GPU's DRM render node (`/dev/dri/renderD<minor>`) read-write, for
/// creating the sync objects that back explicit sync. Render nodes need no DRM
/// master, so a plain client can open one.
pub fn open_drm_render_node(minor: u32) -> Result<OwnedFd> {
    let path = format!("/dev/dri/renderD{minor}");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| Error::msg(format!("open {path}: {e}")))?;
    Ok(OwnedFd::from(file))
}

/// Create a DRM sync object on `drm` (a render-node fd); returns its handle.
/// `signaled` starts it already signaled.
pub fn drm_syncobj_create(drm: RawFd, signaled: bool) -> Result<u32> {
    let mut arg = DrmSyncobjCreate {
        handle: 0,
        flags: if signaled {
            DRM_SYNCOBJ_CREATE_SIGNALED
        } else {
            0
        },
    };
    // SAFETY: arg is a live, correctly laid out drm_syncobj_create; the kernel
    // writes handle on success.
    ok_or_errno(
        unsafe { ioctl(drm, DRM_IOCTL_SYNCOBJ_CREATE, &mut arg) },
        "DRM_IOCTL_SYNCOBJ_CREATE",
    )?;
    Ok(arg.handle)
}

/// Import `sync_file` as the fence at `handle`'s point 0 (a binary syncobj).
fn drm_syncobj_import_sync_file(drm: RawFd, handle: u32, sync_file: RawFd) -> Result<()> {
    let mut arg = DrmSyncobjHandle {
        handle,
        flags: DRM_SYNCOBJ_SYNC_FILE,
        fd: sync_file,
        pad: 0,
        point: 0,
    };
    // SAFETY: arg is a live drm_syncobj_handle; the kernel reads handle and fd.
    ok_or_errno(
        unsafe { ioctl(drm, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &mut arg) },
        "DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE (import sync file)",
    )
}

/// Export the fence at `handle`'s point 0 (a binary syncobj) as a sync file.
fn drm_syncobj_export_sync_file(drm: RawFd, handle: u32) -> Result<OwnedFd> {
    let mut arg = DrmSyncobjHandle {
        handle,
        flags: DRM_SYNCOBJ_SYNC_FILE,
        fd: -1,
        pad: 0,
        point: 0,
    };
    // SAFETY: arg is a live drm_syncobj_handle; the kernel writes fd on success.
    ok_or_errno(
        unsafe { ioctl(drm, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &mut arg) },
        "DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD (export sync file)",
    )?;
    // SAFETY: the kernel handed us ownership of this sync-file fd.
    Ok(unsafe { OwnedFd::from_raw_fd(arg.fd) })
}

/// Copy the fence at (`src`, `src_point`) to (`dst`, `dst_point`).
fn drm_syncobj_transfer(
    drm: RawFd,
    src: u32,
    src_point: u64,
    dst: u32,
    dst_point: u64,
) -> Result<()> {
    let arg = DrmSyncobjTransfer {
        src_handle: src,
        dst_handle: dst,
        src_point,
        dst_point,
        flags: 0,
        pad: 0,
    };
    // SAFETY: arg is a live drm_syncobj_transfer; the kernel only reads it.
    ok_or_errno(
        unsafe { ioctl(drm, DRM_IOCTL_SYNCOBJ_TRANSFER, &arg) },
        "DRM_IOCTL_SYNCOBJ_TRANSFER",
    )
}

/// Export sync object `handle` as a shareable DRM syncobj fd (the object itself,
/// not a sync file), to hand a whole timeline to the compositor over
/// `wp_linux_drm_syncobj_manager_v1.import_timeline`.
pub fn drm_syncobj_export_fd(drm: RawFd, handle: u32) -> Result<OwnedFd> {
    let mut arg = DrmSyncobjHandle {
        handle,
        flags: 0,
        fd: -1,
        pad: 0,
        point: 0,
    };
    // SAFETY: arg is a live, correctly laid out drm_syncobj_handle; the kernel
    // writes fd on success.
    ok_or_errno(
        unsafe { ioctl(drm, DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &mut arg) },
        "DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD",
    )?;
    // SAFETY: the kernel handed us ownership of this syncobj fd.
    Ok(unsafe { OwnedFd::from_raw_fd(arg.fd) })
}

/// Publish `sync_file` (a render-done fence) at timeline `point` of `timeline`,
/// via the caller-owned binary `scratch` syncobj: import into scratch point 0,
/// then transfer to the timeline point. The compositor then waits on this point
/// as the frame's acquire fence.
pub fn drm_syncobj_sync_file_to_point(
    drm: RawFd,
    sync_file: RawFd,
    timeline: u32,
    point: u64,
    scratch: u32,
) -> Result<()> {
    drm_syncobj_import_sync_file(drm, scratch, sync_file)?;
    drm_syncobj_transfer(drm, scratch, 0, timeline, point)
}

/// Export timeline `point` of `timeline` as a sync file, via the caller-owned
/// binary `scratch` syncobj: transfer the point to scratch point 0, then export
/// it. Used to turn the compositor's release point into a fence the next render
/// into that buffer waits on.
pub fn drm_syncobj_point_to_sync_file(
    drm: RawFd,
    timeline: u32,
    point: u64,
    scratch: u32,
) -> Result<OwnedFd> {
    drm_syncobj_transfer(drm, timeline, point, scratch, 0)?;
    drm_syncobj_export_sync_file(drm, scratch)
}

// ---------------------------------------------------------------------------
// Runtime library loading.
// ---------------------------------------------------------------------------

/// A library opened with `dlopen`, closed on drop. Symbols looked up from it
/// are raw pointers; the caller owns the transmute into the right function
/// type and must not outlive the library.
pub struct DynLib {
    handle: *mut c_void,
}

impl DynLib {
    /// Open `name` (a soname like `libvulkan.so.1`), resolving all symbols now.
    /// A missing or unloadable library is a clean `Err`, which is how an
    /// optional backend discovers it is unavailable.
    pub fn open(name: &CStr) -> Result<Self> {
        // SAFETY: name is a valid NUL-terminated string; dlopen only reads it.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(Error::msg(format!(
                "dlopen {} failed",
                name.to_string_lossy()
            )));
        }
        Ok(Self { handle })
    }

    /// Look up `symbol`, or `None` if the library does not export it.
    pub fn sym(&self, symbol: &CStr) -> Option<*mut c_void> {
        // SAFETY: handle is a live dlopen handle; symbol is NUL-terminated.
        let p = unsafe { dlsym(self.handle, symbol.as_ptr()) };
        (!p.is_null()).then_some(p)
    }
}

impl Drop for DynLib {
    fn drop(&mut self) {
        // SAFETY: handle came from dlopen and is closed exactly once.
        unsafe { dlclose(self.handle) };
    }
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
    use core::ffi::c_short;
    use std::os::fd::AsRawFd;

    #[test]
    fn cmsg_math_matches_one_fd() {
        // On a 64-bit ABI: aligned header is 16, CMSG_LEN(4)=20, CMSG_SPACE(4)=24.
        assert_eq!(cmsg_align(mem::size_of::<CmsgHdr>()), 16);
        assert_eq!(cmsg_len(4), 20);
        assert_eq!(cmsg_space(4), 24);
    }

    #[test]
    fn ok_or_errno_maps_zero_to_ok_and_nonzero_to_a_named_error() {
        assert!(ok_or_errno(0, "DRM_IOCTL_X").is_ok());
        let err = ok_or_errno(-1, "DRM_IOCTL_X").unwrap_err();
        assert!(err.to_string().contains("DRM_IOCTL_X failed: errno"));
    }

    /// The length handed to `read_mapped` is the compositor's claim about a file
    /// the compositor owns. A claim larger than the file must be refused: mapping
    /// it succeeds, and reading the bytes past the end raises SIGBUS, killing the
    /// process where no `Result` can intervene.
    #[test]
    fn read_mapped_refuses_a_length_past_the_end_of_the_file() {
        let path = std::env::temp_dir().join(format!("bnknotes-mapped-{}", std::process::id()));
        std::fs::write(&path, b"keymap").expect("write fixture");
        let file = std::fs::File::open(&path).expect("open fixture");
        let fd = file.as_raw_fd();

        assert_eq!(read_mapped(fd, 6).expect("exact length"), b"keymap");
        assert_eq!(read_mapped(fd, 3).expect("short length"), b"key");
        let err = read_mapped(fd, 4096).expect_err("a length past the end must be refused");
        assert!(err.to_string().contains("holding 6"), "got: {err}");

        std::fs::remove_file(&path).ok();
    }

    // The DRM syncobj ioctls talk straight to the kernel, so their request
    // numbers must match `DRM_IOWR(0xNN, struct)` exactly. These are computed
    // from the struct sizes; pin them against the values a C compiler emits
    // from the kernel's `drm.h`, where drm_syncobj_handle carries the timeline
    // `point` field (24 bytes).
    #[test]
    fn drm_syncobj_ioctl_numbers_match_the_abi() {
        assert_eq!(mem::size_of::<DrmSyncobjCreate>(), 8);
        assert_eq!(mem::size_of::<DrmSyncobjHandle>(), 24);
        assert_eq!(mem::size_of::<DrmSyncobjTransfer>(), 32);
        assert_eq!(DRM_IOCTL_SYNCOBJ_CREATE, 0xC008_64BF);
        assert_eq!(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, 0xC018_64C1);
        assert_eq!(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, 0xC018_64C2);
        assert_eq!(DRM_IOCTL_SYNCOBJ_TRANSFER, 0xC020_64CC);
    }

    // Kernel readiness check for the round-trip test below. A sync file's fd
    // becomes POLLIN-readable exactly when its fence has signaled, so a
    // zero-timeout poll reads the fence state without waiting. Test-only, so the
    // binding stays here rather than widening the module's production surface.
    const POLLIN: c_short = 0x0001;

    #[repr(C)]
    struct PollFd {
        fd: c_int,
        events: c_short,
        revents: c_short,
    }

    unsafe extern "C" {
        fn poll(fds: *mut PollFd, nfds: c_ulong, timeout: c_int) -> c_int;
    }

    /// Whether `fd` (a sync file) currently carries a signaled fence.
    fn fd_is_signaled(fd: RawFd) -> bool {
        let mut pfd = PollFd {
            fd,
            events: POLLIN,
            revents: 0,
        };
        // SAFETY: one live PollFd; poll reads events and writes revents only.
        let n = unsafe { poll(&mut pfd, 1, 0) };
        n > 0 && (pfd.revents & POLLIN) != 0
    }

    /// Open the first usable DRM render node, or `None` when the machine has
    /// none (a GPU-less box), so the round-trip test skips rather than fails.
    /// Drives the production `open_drm_render_node` on the way.
    fn open_any_render_node() -> Option<OwnedFd> {
        for entry in std::fs::read_dir("/dev/dri").ok()?.flatten() {
            let minor = entry
                .file_name()
                .to_str()
                .and_then(|n| n.strip_prefix("renderD"))
                .and_then(|n| n.parse::<u32>().ok());
            if let Some(minor) = minor
                && let Ok(fd) = open_drm_render_node(minor)
            {
                return Some(fd);
            }
        }
        None
    }

    // A real round-trip against the kernel's DRM syncobj interface: push a
    // signaled fence into a timeline point and pull it back out, the exact
    // plumbing the present path uses to publish a frame's acquire fence and
    // later read a buffer's release fence. No compositor, no GPU submission.
    // Skips on a machine with no render node, where there is nothing to drive.
    #[test]
    fn drm_syncobj_round_trips_a_fence_through_a_timeline_point() {
        let Some(drm) = open_any_render_node() else {
            let app = env!("CARGO_PKG_NAME");
            eprintln!("{app}: no DRM render node; skipping syncobj round-trip");
            return;
        };
        let fd = drm.as_raw_fd();

        // A syncobj born signaled must export an already-signaled sync file;
        // this also covers create(signaled) and export_sync_file on their own.
        let src = drm_syncobj_create(fd, true).expect("create signaled syncobj");
        let signaled = drm_syncobj_export_sync_file(fd, src).expect("export sync file");
        assert!(
            fd_is_signaled(signaled.as_raw_fd()),
            "a syncobj created signaled must export a signaled fence",
        );

        // Publish that fence at a timeline point, then read the point back as a
        // fresh sync file; it must survive both transfers still signaled. This
        // exercises import_sync_file, transfer both directions (binary point 0
        // to a timeline point and back), and export_sync_file.
        let timeline = drm_syncobj_create(fd, false).expect("create timeline syncobj");
        let scratch = drm_syncobj_create(fd, false).expect("create scratch syncobj");
        const POINT: u64 = 42;
        drm_syncobj_sync_file_to_point(fd, signaled.as_raw_fd(), timeline, POINT, scratch)
            .expect("publish fence to timeline point");
        let round_tripped = drm_syncobj_point_to_sync_file(fd, timeline, POINT, scratch)
            .expect("read timeline point back as a sync file");
        assert!(
            fd_is_signaled(round_tripped.as_raw_fd()),
            "the fence must stay signaled across the timeline round-trip",
        );
        // Dropping `drm` closes the render node, freeing every syncobj on it.
    }
}
