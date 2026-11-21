//! A thin, safe wrapper over Linux inotify: watch directories, read the events
//! the kernel queues, and hand each one back as a borrowed record.
//!
//! The editor watches the *directory* that holds an open file, not the file
//! itself. A modern save is an atomic rename (write a temp, rename it over the
//! target), which swaps the target's inode; a watch on the old inode goes silent
//! the moment the file is replaced. Watching the parent and filtering by name
//! survives the swap, and catches creates, deletes, and moves the same way.
//!
//! The fd is non-blocking and pollable, so the caller waits on it in the same
//! poll as the compositor socket and drains it with [`Inotify::for_each_event`]
//! only when it signals readable.

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use crate::platform::error::{Error, Result};

// The subset of event bits the watcher acts on. A directory watch reports these
// for the entries inside it (CLOSE_WRITE, the moves, CREATE, DELETE) and for the
// watched directory itself (DELETE_SELF, MOVE_SELF). IGNORED means the kernel
// dropped the watch (the directory went away); Q_OVERFLOW means events were lost,
// so the caller must re-check everything from scratch.
const IN_CLOSE_WRITE: u32 = 0x0000_0008;
const IN_MOVED_FROM: u32 = 0x0000_0040;
const IN_MOVED_TO: u32 = 0x0000_0080;
const IN_CREATE: u32 = 0x0000_0100;
const IN_DELETE: u32 = 0x0000_0200;
const IN_DELETE_SELF: u32 = 0x0000_0400;
const IN_MOVE_SELF: u32 = 0x0000_0800;

pub const IN_IGNORED: u32 = 0x0000_8000;

/// The mask a directory watch is armed with: any change to the set of names in
/// the directory, plus the directory's own disappearance. IN_MODIFY is left out
/// on purpose, CLOSE_WRITE and MOVED_TO are the two signals a completed write or
/// atomic-rename save produce, and reacting per keystroke-flush would be noise.
pub const DIR_MASK: u32 = IN_CLOSE_WRITE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF;

// O_NONBLOCK | O_CLOEXEC, spelled as inotify_init1 wants them: the fd never
// blocks a drain and does not leak across an exec.
const IN_NONBLOCK: c_int = 0o4000;
const IN_CLOEXEC: c_int = 0o2000000;

unsafe extern "C" {
    fn inotify_init1(flags: c_int) -> c_int;
    fn inotify_add_watch(fd: c_int, pathname: *const c_char, mask: u32) -> c_int;
    fn inotify_rm_watch(fd: c_int, wd: c_int) -> c_int;
    fn read(fd: c_int, buf: *mut c_void, count: usize) -> isize;
}

/// A watch descriptor: the kernel's handle for one `add_watch`, used to match an
/// event back to the directory it came from and to remove the watch later.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Wd(c_int);

/// One inotify event: which watch fired and the bitmask of what happened.
pub struct Event {
    pub wd: Wd,
    pub mask: u32,
}

/// An open inotify instance. Owns the fd (closed on drop) and nothing else; the
/// mapping from watch descriptor back to a directory lives in the caller.
pub struct Inotify {
    fd: OwnedFd,
}

impl Inotify {
    /// Open a fresh inotify instance. Fails only if the process is out of fds or
    /// inotify instances, in which case the caller runs without live watching.
    pub fn new() -> Result<Self> {
        // SAFETY: a plain syscall with no pointer arguments.
        let raw = unsafe { inotify_init1(IN_NONBLOCK | IN_CLOEXEC) };
        if raw < 0 {
            return Err(Error::msg(format!(
                "inotify_init1 failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        // SAFETY: raw is a fresh, owned fd this call just created.
        let fd = unsafe { OwnedFd::from_raw_fd(raw) };
        Ok(Self { fd })
    }

    /// The raw fd, so an event loop can poll it alongside the compositor socket.
    pub fn fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Watch `dir` for [`DIR_MASK`]. Re-adding an already watched directory just
    /// returns its existing descriptor, so the caller can be idempotent.
    pub fn add_dir(&self, dir: &Path) -> Result<Wd> {
        let c = CString::new(dir.as_os_str().as_bytes())
            .map_err(|_| Error::msg("watch path contains an interior NUL byte"))?;
        // SAFETY: c is a valid NUL-terminated string that outlives the call.
        let wd = unsafe { inotify_add_watch(self.fd.as_raw_fd(), c.as_ptr(), DIR_MASK) };
        if wd < 0 {
            return Err(Error::msg(format!(
                "inotify_add_watch({}) failed: {}",
                dir.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Wd(wd))
    }

    /// Stop watching a directory. A watch the kernel already dropped (the
    /// directory went away, so the descriptor is stale) is not an error here.
    pub fn rm(&self, wd: Wd) {
        // SAFETY: a plain syscall; a stale descriptor just returns -1 (EINVAL),
        // which is the state we wanted anyway, so the result is ignored.
        unsafe {
            inotify_rm_watch(self.fd.as_raw_fd(), wd.0);
        }
    }

    /// Drain every queued event into `f`, using `buf` as reusable scratch. Reads
    /// until the fd would block, so a burst is consumed in one drain. The kernel
    /// never splits an event across reads, so each read yields whole records.
    pub fn for_each_event<F: FnMut(Event)>(&self, buf: &mut [u8], mut f: F) -> Result<()> {
        loop {
            // SAFETY: buf is a valid, writable slice of the given length; read
            // writes at most that many bytes and returns the count.
            let n = unsafe {
                read(
                    self.fd.as_raw_fd(),
                    buf.as_mut_ptr().cast::<c_void>(),
                    buf.len(),
                )
            };
            if n < 0 {
                let e = std::io::Error::last_os_error();
                match e.kind() {
                    // Queue empty: the drain is done.
                    std::io::ErrorKind::WouldBlock => return Ok(()),
                    // A signal interrupted the read; retry it.
                    std::io::ErrorKind::Interrupted => continue,
                    _ => return Err(Error::msg(format!("inotify read failed: {e}"))),
                }
            }
            if n == 0 {
                return Ok(());
            }
            parse_events(&buf[..n as usize], &mut f);
        }
    }
}

/// Walk one read's worth of packed inotify_event records. Each record is a fixed
/// 16-byte header (wd, mask, cookie, name length) followed by the NUL-padded
/// name; the next record begins right after it.
fn parse_events<F: FnMut(Event)>(bytes: &[u8], f: &mut F) {
    const HEADER: usize = 16;
    let mut i = 0;
    while i + HEADER <= bytes.len() {
        let wd = c_int::from_ne_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        let mask = u32::from_ne_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]);
        let len = u32::from_ne_bytes([bytes[i + 12], bytes[i + 13], bytes[i + 14], bytes[i + 15]])
            as usize;
        let name_end = i + HEADER + len;
        if name_end > bytes.len() {
            break;
        }
        f(Event { wd: Wd(wd), mask });
        i = name_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A unique scratch directory under the system temp dir, per test name.
    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-inotify-{}-{}", std::process::id(), name));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Read events until at least one arrives or the attempt budget runs out.
    /// inotify delivery is asynchronous, so a single drain may see an empty queue;
    /// a short bounded poll avoids both a hang and a flake.
    fn drain_until_event(ino: &Inotify) -> Vec<u32> {
        let mut buf = [0u8; 4096];
        let mut out = Vec::new();
        for _ in 0..200 {
            ino.for_each_event(&mut buf, |e| {
                out.push(e.mask);
            })
            .unwrap();
            if !out.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        out
    }

    #[test]
    fn reports_a_file_created_in_a_watched_dir() {
        let dir = tmp_dir("create");
        let ino = Inotify::new().unwrap();
        ino.add_dir(&dir).unwrap();

        fs::write(dir.join("note.md"), b"hi").unwrap();

        let events = drain_until_event(&ino);
        assert!(
            events.iter().any(|mask| mask & IN_CREATE != 0),
            "expected a create event, got {events:?}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reports_an_atomic_rename_over_the_target() {
        // The save path bnknotes uses: write a sibling temp, rename it over the
        // target. A directory watch must see the MOVED_TO for the final name even
        // though the target's inode changed.
        let dir = tmp_dir("rename");
        let target = dir.join("note.md");
        fs::write(&target, b"old").unwrap();

        let ino = Inotify::new().unwrap();
        ino.add_dir(&dir).unwrap();

        let tmp = dir.join(".note.md.tmp");
        fs::write(&tmp, b"new").unwrap();
        fs::rename(&tmp, &target).unwrap();

        // Only the rename's target receives a MOVED_TO: the temp file is created and
        // written, then moved away.
        let events = drain_until_event(&ino);
        assert!(
            events.iter().any(|mask| mask & IN_MOVED_TO != 0),
            "expected MOVED_TO for the target, got {events:?}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_removed_watch_stops_reporting() {
        let dir = tmp_dir("rm");
        let ino = Inotify::new().unwrap();
        let wd = ino.add_dir(&dir).unwrap();
        ino.rm(wd);

        // Drain the IN_IGNORED the removal itself queues, then confirm a later
        // change produces nothing.
        let mut buf = [0u8; 4096];
        ino.for_each_event(&mut buf, |_| {}).unwrap();

        fs::write(dir.join("after.md"), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut masks = Vec::new();
        ino.for_each_event(&mut buf, |e| masks.push(e.mask))
            .unwrap();
        assert!(
            !masks.iter().any(|mask| mask & !IN_IGNORED != 0),
            "removed watch still reported {masks:?}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_walks_two_packed_records() {
        // Two records back to back: header is wd(4) mask(4) cookie(4) len(4), then
        // the NUL-padded name. Hand-pack them and confirm the parser splits both.
        fn record(out: &mut Vec<u8>, wd: i32, mask: u32, name: &str) {
            let padded = (name.len() + 1 + 3) & !3; // NUL then align to 4
            out.extend_from_slice(&wd.to_ne_bytes());
            out.extend_from_slice(&mask.to_ne_bytes());
            out.extend_from_slice(&0u32.to_ne_bytes()); // cookie
            out.extend_from_slice(&(padded as u32).to_ne_bytes());
            let start = out.len();
            out.extend_from_slice(name.as_bytes());
            out.resize(start + padded, 0);
        }
        let mut bytes = Vec::new();
        record(&mut bytes, 1, IN_CREATE, "alpha.md");
        record(&mut bytes, 2, IN_DELETE, "beta");

        let mut seen = Vec::new();
        parse_events(&bytes, &mut |e| seen.push((e.wd.0, e.mask)));
        assert_eq!(seen, vec![(1, IN_CREATE), (2, IN_DELETE)]);
    }
}
