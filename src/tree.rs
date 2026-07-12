//! The file-tree sidebar's model: a directory rooted at some folder plus the
//! set of expanded sub-directories, flattened on demand into the visible rows
//! the pane renders and hit-tests. Directories are read lazily (only expanded
//! ones are walked) and each level is sorted directories-first, then
//! case-insensitively by name. Only Markdown files are listed, shown without
//! their .md suffix; hidden entries (dotfiles and dot-folders like .git) are skipped.
//! A folder holding no note anywhere beneath it can be hidden too, the sidebar
//! footer's toggle. Pure filesystem + data; the pane owns the pixels.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// How deep [`has_notes`] descends looking for a note. A vault nests a handful of
/// levels; this only bounds the stack on a pathological tree.
const MAX_SCAN_DEPTH: usize = 32;

/// One visible row in the flattened tree: enough to draw it (name, indent,
/// folder-vs-file, open-vs-closed) and to act on a click (its path).
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub path: PathBuf,
    pub name: String,
    pub depth: usize,
    pub is_dir: bool,
    pub expanded: bool,
}

/// A directory tree rooted at `root`, remembering which folders are open. The
/// flattened `rows` are rebuilt whenever the expansion set or the folder filter
/// changes; the pane reads them each frame.
pub struct Tree {
    root: PathBuf,
    expanded: BTreeSet<PathBuf>,
    show_empty_dirs: bool,
    rows: Vec<Row>,
}

/// What a walk of the tree keeps in hand at every level: which folders are open,
/// and whether a folder with no note beneath it is listed at all.
struct View<'a> {
    expanded: &'a BTreeSet<PathBuf>,
    show_empty_dirs: bool,
}

impl Tree {
    /// Build a tree at `root` with `expanded` folders already open and the folder
    /// filter in `show_empty_dirs` (both as restored from the session). Reads the
    /// visible folders immediately.
    pub fn new(root: PathBuf, expanded: BTreeSet<PathBuf>, show_empty_dirs: bool) -> Self {
        let mut tree = Self {
            root,
            expanded,
            show_empty_dirs,
            rows: Vec::new(),
        };
        tree.rebuild();
        tree
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// The row showing `path`, if it is currently visible. Lets a caller hold on to
    /// a row across a rebuild that shifts the indices under it.
    pub fn index_of(&self, path: &Path) -> Option<usize> {
        self.rows.iter().position(|r| r.path == path)
    }

    /// The open folders, for persisting to the session file.
    pub fn expanded_dirs(&self) -> Vec<PathBuf> {
        self.expanded.iter().cloned().collect()
    }

    /// Whether folders with no note beneath them are listed.
    pub fn show_empty_dirs(&self) -> bool {
        self.show_empty_dirs
    }

    /// List (or hide) the folders that hold no note, and re-flatten.
    pub fn set_show_empty_dirs(&mut self, show: bool) {
        if self.show_empty_dirs != show {
            self.show_empty_dirs = show;
            self.rebuild();
        }
    }

    /// Toggle a folder open/closed and re-flatten. A no-op on files.
    pub fn toggle(&mut self, path: &Path) {
        if !self.expanded.remove(path) {
            self.expanded.insert(path.to_path_buf());
        }
        self.rebuild();
    }

    /// Re-read the open folders from disk and re-flatten, picking up files that
    /// appeared or vanished since the last build.
    fn rebuild(&mut self) {
        let view = View {
            expanded: &self.expanded,
            show_empty_dirs: self.show_empty_dirs,
        };
        let mut rows = Vec::new();
        // The root itself is not a row (it is the pane); its children are depth 0.
        walk(&self.root, 0, &view, &mut rows);
        self.rows = rows;
    }
}

fn walk(dir: &Path, depth: usize, view: &View, out: &mut Vec<Row>) {
    for (path, name, is_dir) in read_dir_sorted(dir) {
        // A folder with no note beneath it is dropped whole when the filter is on,
        // children and all, so its subtree cannot come back through an open folder.
        if is_dir && !view.show_empty_dirs && !has_notes(&path, 0) {
            continue;
        }
        let is_expanded = is_dir && view.expanded.contains(&path);
        out.push(Row {
            path: path.clone(),
            name,
            depth,
            is_dir,
            expanded: is_expanded,
        });
        if is_expanded {
            walk(&path, depth + 1, view, out);
        }
    }
}

/// Whether `dir` holds a note anywhere beneath it, so the tree lists it under the
/// folder filter. A folder whose own notes sit in a sub-folder still counts, so the
/// filter never hides the path to a note; the scan stops at the first one found.
/// Hidden entries are skipped, as they are in the tree itself.
fn has_notes(dir: &Path, depth: usize) -> bool {
    if depth >= MAX_SCAN_DEPTH {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let raw = entry.file_name();
        if raw.to_str().is_none_or(is_hidden) {
            continue;
        }
        let path = entry.path();
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if has_notes(&path, depth + 1) {
                return true;
            }
        } else if is_markdown(&path) {
            return true;
        }
    }
    false
}

/// Read one directory into `(path, name, is_dir)`, folders first, then
/// case-insensitively by name. Only folders and Markdown files are kept; a file's
/// name drops the .md suffix, while its path keeps it. Hidden entries are skipped.
/// An unreadable directory (permissions, a race) yields nothing rather than
/// failing, so one bad folder cannot blank the tree.
fn read_dir_sorted(dir: &Path) -> Vec<(PathBuf, String, bool)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut items: Vec<(PathBuf, String, bool)> = entries
        .flatten()
        .filter_map(|e| {
            let raw = e.file_name();
            let raw = raw.to_str()?;
            if is_hidden(raw) {
                return None;
            }
            let path = e.path();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir {
                Some((path, raw.to_string(), true))
            } else if is_markdown(&path) {
                let name = path.file_stem().and_then(|s| s.to_str())?.to_string();
                Some((path, name, false))
            } else {
                None
            }
        })
        .collect();
    // Directories (is_dir = true) sort ahead of files, then by lowercased name.
    items.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| a.1.to_lowercase().cmp(&b.1.to_lowercase()))
    });
    items
}

/// Whether the tree lists this file: a Markdown document, matched on the .md
/// extension case-insensitively.
fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

/// A dotfile or dot-directory (.git): never listed, never scanned.
fn is_hidden(name: &str) -> bool {
    name.starts_with('.')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway directory tree under the temp dir, removed on drop.
    struct Fixture(PathBuf);

    impl Fixture {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("bnknotes_tree_{}_{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("zeta")).expect("mkdir zeta");
            std::fs::create_dir_all(dir.join("alpha/nested")).expect("mkdir alpha/nested");
            std::fs::create_dir_all(dir.join(".git")).expect("mkdir .git");
            std::fs::write(dir.join("readme.md"), b"r").expect("write readme");
            std::fs::write(dir.join("notes.txt"), b"n").expect("write notes");
            std::fs::write(dir.join(".secret.md"), b"s").expect("write secret");
            std::fs::write(dir.join("alpha/a.md"), b"a").expect("write a");
            std::fs::write(dir.join("alpha/nested/deep.md"), b"d").expect("write deep");
            Self(dir)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The tree as the app first shows it: every folder listed, none expanded.
    fn tree_at(root: &Path) -> Tree {
        Tree::new(root.to_path_buf(), BTreeSet::new(), true)
    }

    /// The names of the visible rows, in order.
    fn names(tree: &Tree) -> Vec<&str> {
        tree.rows().iter().map(|r| r.name.as_str()).collect()
    }

    #[test]
    fn lists_folders_before_files_each_sorted() {
        let fx = Fixture::new("sort");
        let tree = tree_at(&fx.0);
        // Folders (alpha, zeta) before the file (readme, .md dropped); collapsed,
        // so no children. The non-Markdown and hidden entries are absent.
        assert_eq!(names(&tree), ["alpha", "zeta", "readme"]);
    }

    #[test]
    fn hides_non_markdown_and_dotfiles() {
        let fx = Fixture::new("filter");
        // A Markdown file's row keeps the full path but shows without the suffix.
        std::fs::write(fx.0.join("Loud.MD"), b"u").expect("write Loud");
        let tree = tree_at(&fx.0);
        let rows: Vec<(&str, &std::path::Path)> = tree
            .rows()
            .iter()
            .map(|r| (r.name.as_str(), r.path.as_path()))
            .collect();
        // notes.txt, .secret.md, and .git are all gone; Loud.MD matches case-insensitively.
        assert_eq!(
            rows,
            [
                ("alpha", fx.0.join("alpha").as_path()),
                ("zeta", fx.0.join("zeta").as_path()),
                ("Loud", fx.0.join("Loud.MD").as_path()),
                ("readme", fx.0.join("readme.md").as_path()),
            ]
        );
    }

    #[test]
    fn expanding_a_folder_reveals_its_children_indented() {
        let fx = Fixture::new("expand");
        let mut tree = tree_at(&fx.0);
        tree.toggle(&fx.0.join("alpha"));
        let rows: Vec<(&str, usize, bool)> = tree
            .rows()
            .iter()
            .map(|r| (r.name.as_str(), r.depth, r.expanded))
            .collect();
        assert_eq!(
            rows,
            [
                ("alpha", 0, true),
                ("nested", 1, false),
                ("a", 1, false),
                ("zeta", 0, false),
                ("readme", 0, false),
            ]
        );
    }

    #[test]
    fn toggle_collapses_and_persists_the_expanded_set() {
        let fx = Fixture::new("toggle");
        let alpha = fx.0.join("alpha");
        let mut tree = tree_at(&fx.0);
        tree.toggle(&alpha);
        assert_eq!(tree.expanded_dirs(), vec![alpha.clone()]);
        tree.toggle(&alpha);
        assert!(tree.expanded_dirs().is_empty());
        assert_eq!(tree.rows().iter().filter(|r| r.depth > 0).count(), 0);
    }

    #[test]
    fn nested_expansion_restored_from_the_session() {
        let fx = Fixture::new("restore");
        let expanded: BTreeSet<PathBuf> = [fx.0.join("alpha"), fx.0.join("alpha/nested")]
            .into_iter()
            .collect();
        let tree = Tree::new(fx.0.clone(), expanded, true);
        let deep = tree.rows().iter().any(|r| r.name == "deep" && r.depth == 2);
        assert!(
            deep,
            "restored expansion should reveal the deeply nested file"
        );
    }

    #[test]
    fn index_of_locates_a_visible_row() {
        let fx = Fixture::new("index");
        let tree = tree_at(&fx.0);
        assert_eq!(tree.index_of(&fx.0.join("readme.md")), Some(2));
        // A file inside a collapsed folder has no row.
        assert_eq!(tree.index_of(&fx.0.join("alpha/a.md")), None);
    }

    #[test]
    fn the_filter_hides_a_folder_with_no_notes() {
        let fx = Fixture::new("empty");
        // zeta holds nothing; a folder of non-notes is just as empty to the tree.
        std::fs::create_dir_all(fx.0.join("assets")).expect("mkdir assets");
        std::fs::write(fx.0.join("assets/pic.png"), b"p").expect("write pic");
        let tree = Tree::new(fx.0.clone(), BTreeSet::new(), false);
        assert_eq!(names(&tree), ["alpha", "readme"]);
    }

    #[test]
    fn the_filter_keeps_a_folder_whose_notes_are_nested() {
        let fx = Fixture::new("nested");
        // outer holds no note itself, only a folder that (deeply) does; hiding it
        // would hide the path to that note.
        std::fs::create_dir_all(fx.0.join("outer/inner")).expect("mkdir outer/inner");
        std::fs::write(fx.0.join("outer/inner/note.md"), b"n").expect("write note");
        let expanded: BTreeSet<PathBuf> = [fx.0.join("outer")].into_iter().collect();
        let tree = Tree::new(fx.0.clone(), expanded, false);
        assert_eq!(names(&tree), ["alpha", "outer", "inner", "readme"]);
    }

    #[test]
    fn the_filter_hides_an_empty_folder_nested_in_a_kept_one() {
        let fx = Fixture::new("deep_empty");
        // alpha stays (it has notes), but the drafts folder inside it has none.
        std::fs::create_dir_all(fx.0.join("alpha/drafts")).expect("mkdir drafts");
        let expanded: BTreeSet<PathBuf> = [fx.0.join("alpha")].into_iter().collect();
        let tree = Tree::new(fx.0.clone(), expanded, false);
        assert_eq!(names(&tree), ["alpha", "nested", "a", "readme"]);
    }

    #[test]
    fn toggling_the_filter_relists_the_empty_folders() {
        let fx = Fixture::new("relist");
        let mut tree = tree_at(&fx.0);
        assert!(tree.show_empty_dirs());
        tree.set_show_empty_dirs(false);
        assert!(!tree.show_empty_dirs());
        assert_eq!(names(&tree), ["alpha", "readme"]);
        tree.set_show_empty_dirs(true);
        assert_eq!(names(&tree), ["alpha", "zeta", "readme"]);
    }

    #[test]
    fn a_hidden_note_does_not_keep_its_folder() {
        let fx = Fixture::new("hidden_note");
        // The tree never lists a dotfile, so a folder holding only one is empty as
        // far as the filter is concerned, as is one holding only a dot-directory.
        std::fs::create_dir_all(fx.0.join("stash/.trash")).expect("mkdir stash/.trash");
        std::fs::write(fx.0.join("stash/.draft.md"), b"d").expect("write draft");
        std::fs::write(fx.0.join("stash/.trash/gone.md"), b"g").expect("write gone");
        let tree = Tree::new(fx.0.clone(), BTreeSet::new(), false);
        assert_eq!(names(&tree), ["alpha", "readme"]);
    }
}
