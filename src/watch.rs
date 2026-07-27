//! Watching open files for changes made outside the editor.
//!
//! The editor watches the *directory* each open file sits in, not the file
//! itself, so an atomic-rename save (the standard write-temp-then-rename) does
//! not slip past the watch when it swaps the target's inode. Several open tabs in
//! one directory share a single watch, reference-counted so it drops only when
//! the last of them closes.
//!
//! This layer answers one question cheaply: did anything happen that warrants
//! re-checking the open files against disk? It does not decide what changed; that
//! is [`crate::pane::EditorPane::recheck_disk`], which re-stats and reloads or
//! flags a conflict. inotify is best-effort (it can drop events under load, and
//! does not reach every filesystem), so the re-stat is the source of truth and
//! this is only the nudge that triggers it.

use std::collections::HashMap;
use std::os::fd::RawFd;
use std::path::{Path, PathBuf};

use crate::platform::inotify::{Inotify, Wd, IN_IGNORED};

/// One directory watch plus a count of how many open files live in it, so the
/// watch is removed only when the last file in that directory closes.
struct DirWatch {
    wd: Wd,
    files: usize,
}

pub(crate) struct FileWatcher {
    /// The inotify instance, or `None` when the kernel would not grant one. The
    /// watcher then degrades to a no-op and the editor leans on the save-time
    /// guard alone (which is stat-based and needs no watch).
    inotify: Option<Inotify>,
    /// Watched directories, keyed by their canonical path, each with its
    /// descriptor and how many open files it holds.
    dirs: HashMap<PathBuf, DirWatch>,
    /// The file paths currently watched (through their parent directories), so a
    /// [`Self::sync`] can diff the new desired set against it. Each carries the last
    /// [`Self::watches_exactly`] pass that yielded it, which is how that compare
    /// counts a path once when two panes hold it.
    files: HashMap<PathBuf, u64>,
    /// Bumped by every [`Self::watches_exactly`], so the stamps of the pass before
    /// it read as stale rather than as this pass's.
    pass: u64,
    /// Reusable read buffer, so draining a burst of events allocates nothing.
    buf: Vec<u8>,
}

impl FileWatcher {
    /// Open a watcher. A kernel that refuses an inotify instance is not fatal:
    /// the watcher becomes a no-op and live detection is simply off.
    pub(crate) fn new() -> Self {
        let inotify = match Inotify::new() {
            Ok(i) => Some(i),
            Err(e) => {
                eprintln!("bnknotes: file watching disabled: {e}");
                None
            }
        };
        Self {
            inotify,
            dirs: HashMap::new(),
            files: HashMap::new(),
            pass: 0,
            buf: vec![0u8; 8192],
        }
    }

    /// The inotify fd to poll alongside the compositor socket, or `None` when
    /// watching is off (nothing to wait on).
    pub(crate) fn fd(&self) -> Option<RawFd> {
        self.inotify.as_ref().map(Inotify::fd)
    }

    /// Whether the given open-file paths are exactly the set already watched, so
    /// the caller can skip a reconcile (and its allocation) in the steady state.
    /// A no-op watcher (no inotify) always reports in sync, so nothing churns.
    ///
    /// The paths arrive per tab, so one note open in two panes is yielded twice.
    /// Stamping each entry with this pass and counting it only the first time is
    /// what makes the compare see a set rather than a multiset: a raw count would
    /// never match, and the reconcile would then run on every loop pass forever.
    pub(crate) fn watches_exactly<'a, I: Iterator<Item = &'a Path>>(&mut self, paths: I) -> bool {
        if self.inotify.is_none() {
            return true;
        }
        self.pass += 1;
        let pass = self.pass;
        let mut distinct = 0;
        for p in paths {
            let Some(seen) = self.files.get_mut(p) else {
                return false;
            };
            if *seen != pass {
                *seen = pass;
                distinct += 1;
            }
        }
        distinct == self.files.len()
    }

    /// Reconcile the watched directories with the current set of open file paths:
    /// start watching the directory of a newly opened file, and stop watching one
    /// no directory of an open file needs any more. Idempotent, so it is safe to
    /// call on every open, close, or reload.
    pub(crate) fn sync<'a, I: Iterator<Item = &'a Path>>(&mut self, paths: I) {
        if self.inotify.is_none() {
            return;
        }
        let desired: HashMap<PathBuf, u64> = paths.map(|p| (p.to_path_buf(), 0)).collect();
        let added: Vec<PathBuf> = desired
            .keys()
            .filter(|f| !self.files.contains_key(*f))
            .cloned()
            .collect();
        let removed: Vec<PathBuf> = self
            .files
            .keys()
            .filter(|f| !desired.contains_key(*f))
            .cloned()
            .collect();
        for f in removed {
            self.unwatch_file(&f);
        }
        for f in added {
            self.watch_file(&f);
        }
        self.files = desired;
    }

    /// Drain every queued event, returning whether any arrived (so the caller
    /// re-checks the open files against disk). A dropped watch (its directory
    /// went away) is forgotten here so the next [`Self::sync`] re-arms it.
    pub(crate) fn drain(&mut self) -> bool {
        let Some(inotify) = self.inotify.as_ref() else {
            return false;
        };
        let mut woke = false;
        let mut dropped: Vec<Wd> = Vec::new();
        let result = inotify.for_each_event(&mut self.buf, |e| {
            if e.mask & IN_IGNORED != 0 {
                dropped.push(e.wd);
            }
            // A bare IGNORED is this watcher's own rm_watch confirming a drop, not
            // a reason to re-check. Every other event (a write, move, create,
            // delete, self-delete, or a queue overflow) is.
            if e.mask & !IN_IGNORED != 0 {
                woke = true;
            }
        });
        if let Err(e) = result {
            eprintln!("bnknotes: inotify drain failed: {e}");
        }
        // A watch the kernel dropped leaves a stale descriptor. Forget the
        // directory and the files under it so the next sync watches it afresh.
        for wd in dropped {
            self.forget_watch(wd);
        }
        woke
    }

    /// Begin watching `file`'s parent directory, sharing an existing watch when
    /// one already covers that directory.
    fn watch_file(&mut self, file: &Path) {
        let Some(inotify) = self.inotify.as_ref() else {
            return;
        };
        let dir = canonical_parent(file);
        if let Some(dw) = self.dirs.get_mut(&dir) {
            dw.files += 1;
            return;
        }
        match inotify.add_dir(&dir) {
            Ok(wd) => {
                self.dirs.insert(dir, DirWatch { wd, files: 1 });
            }
            Err(e) => eprintln!("bnknotes: cannot watch {}: {e}", dir.display()),
        }
    }

    /// Drop one reference to `file`'s parent directory's watch, removing the watch
    /// when the last open file there is gone.
    fn unwatch_file(&mut self, file: &Path) {
        let dir = canonical_parent(file);
        let Some(dw) = self.dirs.get_mut(&dir) else {
            return;
        };
        dw.files -= 1;
        if dw.files == 0 {
            if let (Some(inotify), Some(dw)) = (self.inotify.as_ref(), self.dirs.remove(&dir)) {
                inotify.rm(dw.wd);
            }
        }
    }

    /// Forget a watch the kernel already dropped, along with the files it covered,
    /// so a later sync re-adds them.
    fn forget_watch(&mut self, wd: Wd) {
        let Some(dir) = self
            .dirs
            .iter()
            .find(|(_, dw)| dw.wd == wd)
            .map(|(p, _)| p.clone())
        else {
            return;
        };
        self.dirs.remove(&dir);
        self.files.retain(|f, _| canonical_parent(f) != dir);
    }
}

/// The directory holding `file`, canonicalized so two spellings of the same
/// directory (or a symlink to it) share one watch. Falls back to the literal
/// parent (or the current directory for a bare name) when it cannot be resolved,
/// e.g. a file in a directory that does not exist yet.
fn canonical_parent(file: &Path) -> PathBuf {
    let parent = match file.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    std::fs::canonicalize(&parent).unwrap_or(parent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("bnknotes-watch-{}-{}", std::process::id(), name));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Drain until an event is seen or the budget runs out (inotify delivery is
    /// asynchronous, so a single drain can miss a just-made change).
    fn drained(w: &mut FileWatcher) -> bool {
        for _ in 0..200 {
            if w.drain() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }

    #[test]
    fn watches_an_open_files_directory_and_reports_a_change() {
        let dir = tmp_dir("open");
        let file = dir.join("note.md");
        fs::write(&file, b"x").unwrap();

        let mut w = FileWatcher::new();
        w.sync([file.as_path()].into_iter());

        fs::write(&file, b"changed").unwrap();
        assert!(
            drained(&mut w),
            "a change to a watched file must wake the watcher"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_closed_file_stops_being_watched() {
        let dir = tmp_dir("close");
        let file = dir.join("note.md");
        fs::write(&file, b"x").unwrap();

        let mut w = FileWatcher::new();
        w.sync([file.as_path()].into_iter());
        // Close every tab: the directory watch is dropped.
        w.sync(std::iter::empty());
        assert!(w.dirs.is_empty(), "no directory should remain watched");

        fs::write(&file, b"changed").unwrap();
        // Nothing is watched, so no event arrives; one drain is enough.
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(
            !w.drain(),
            "a change to an unwatched file must not wake the watcher"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn one_file_open_twice_still_reads_as_in_sync() {
        let dir = tmp_dir("twice");
        let file = dir.join("note.md");
        fs::write(&file, b"x").unwrap();

        let mut w = FileWatcher::new();
        w.sync([file.as_path()].into_iter());
        // The same note open in both panes yields its path once per tab. The fast
        // path must still hold, or every loop pass would re-run the reconcile.
        assert!(
            w.watches_exactly([file.as_path(), file.as_path()].into_iter()),
            "a path yielded twice must count once against the watched set"
        );
        assert!(
            w.watches_exactly([file.as_path(), file.as_path()].into_iter()),
            "the compare must hold on every later pass too"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_changed_open_set_breaks_the_fast_path() {
        let dir = tmp_dir("changed");
        let (a, b) = (dir.join("a.md"), dir.join("b.md"));
        fs::write(&a, b"a").unwrap();
        fs::write(&b, b"b").unwrap();

        let mut w = FileWatcher::new();
        w.sync([a.as_path()].into_iter());
        assert!(
            !w.watches_exactly([a.as_path(), b.as_path()].into_iter()),
            "a newly opened file must fall through to the reconcile"
        );
        assert!(
            !w.watches_exactly(std::iter::empty()),
            "a closed file must fall through to the reconcile"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tabs_in_one_directory_share_a_single_watch() {
        let dir = tmp_dir("share");
        let a = dir.join("a.md");
        let b = dir.join("b.md");
        fs::write(&a, b"a").unwrap();
        fs::write(&b, b"b").unwrap();

        let mut w = FileWatcher::new();
        w.sync([a.as_path(), b.as_path()].into_iter());
        assert_eq!(w.dirs.len(), 1, "two files in a directory share one watch");

        // Closing one tab keeps the watch alive for the other.
        w.sync([b.as_path()].into_iter());
        assert_eq!(w.dirs.len(), 1, "the watch survives while a file remains");
        fs::remove_dir_all(&dir).ok();
    }
}
