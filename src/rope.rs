//! A persistent rope: a balanced binary tree of UTF-8 text chunks.
//!
//! The editor's buffer was one `String`, where every insert or delete memmoves
//! the tail and every undo snapshot clones the whole text. A rope replaces that:
//! edits, byte indexing, and line lookup are O(log n), and the tree is
//! *persistent*, its nodes are shared through `Rc`, so cloning a rope (what an
//! undo snapshot does) is O(1) and shares every untouched chunk with the
//! original. An edit copies only the O(log n) nodes on the path from the root to
//! the touched leaf; the rest stay shared between the old and new versions.
//!
//! Leaves hold up to [`MAX_LEAF`] bytes; every node caches its subtree's byte
//! and newline counts, so a byte offset or a line number finds its leaf without
//! scanning. Edits split and concat subtrees. The tree is kept roughly balanced
//! by rebuilding a subtree from its leaves when its height outgrows its size
//! (Boehm's Fibonacci criterion); ordinary scattered editing keeps it balanced
//! without ever triggering that, and the rebuild only bounds the pathological
//! case of many edits hammering one position.
//!
//! ```text
//!            (branch: bytes=9, nl=1)
//!            /                      \
//!   (leaf "ab\n")              (branch)
//!                              /        \
//!                       (leaf "cde")  (leaf "fgh")
//! ```
//!
//! Byte offsets passed in are assumed to fall on `char` boundaries (the editor
//! only ever uses grapheme-cluster boundaries), exactly as `String` slicing
//! assumes today.

use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::rc::Rc;

/// Maximum bytes in a leaf chunk. Large enough that a note lives in a handful of
/// leaves, small enough that splitting a leaf copies little.
const MAX_LEAF: usize = 1024;

/// One tree node: its cached subtree totals plus the leaf text or two children.
struct Inner {
    /// Total bytes in this subtree.
    bytes: usize,
    /// Total `'\n'` count in this subtree, for line lookup.
    newlines: usize,
    /// Tree height (a leaf is 1), for the balance check.
    height: u32,
    kind: Kind,
}

enum Kind {
    Leaf(String),
    Branch(Rc<Inner>, Rc<Inner>),
}

/// A rope. Cloning is O(1): it bumps the root's refcount and shares the tree.
#[derive(Clone)]
pub struct Rope {
    root: Rc<Inner>,
}

impl Rope {
    /// An empty rope.
    pub fn new() -> Self {
        Rope {
            root: leaf(String::new()),
        }
    }

    /// A rope holding `s`.
    pub fn from_str(s: &str) -> Self {
        Rope { root: build(s) }
    }

    /// Total length in bytes.
    pub fn len(&self) -> usize {
        self.root.bytes
    }

    /// The number of lines: one more than the newline count, so text with no
    /// trailing newline still has a final line, and the empty rope has one line.
    pub fn len_lines(&self) -> usize {
        self.root.newlines + 1
    }

    /// Insert `s` at byte offset `at` (clamped into the rope, assumed to be on a
    /// `char` boundary).
    pub fn insert(&mut self, at: usize, s: &str) {
        if s.is_empty() {
            return;
        }
        let at = at.min(self.root.bytes);
        let (l, r) = split(&self.root, at);
        let joined = concat(concat(l, build(s)), r);
        self.root = balance(joined);
    }

    /// Remove the byte range `range` (clamped into the rope, assumed to be on
    /// `char` boundaries).
    pub fn remove(&mut self, range: Range<usize>) {
        let start = range.start.min(self.root.bytes);
        let end = range.end.min(self.root.bytes);
        if start >= end {
            return;
        }
        let (l, rest) = split(&self.root, start);
        let (_, r) = split(&rest, end - start);
        self.root = balance(concat(l, r));
    }

    /// A borrowed view of the bytes in `range` (assumed `char`-aligned), without
    /// copying. The bounds are clamped into the rope. See [`RopeSlice`] for how
    /// to borrow a contiguous `&str` (single-leaf ranges) or materialize the
    /// rest; this is the allocation-free entry point layout reads lines through.
    pub fn slice(&self, range: Range<usize>) -> RopeSlice<'_> {
        let start = range.start.min(self.root.bytes);
        let end = range.end.min(self.root.bytes).max(start);
        RopeSlice {
            root: &self.root,
            start,
            end,
        }
    }

    /// Copy out the bytes in `range` as an owned `String` (assumed
    /// `char`-aligned). Prefer [`slice`](Self::slice) where a borrow suffices;
    /// this is the shorthand for the cases that genuinely need owned bytes
    /// (clipboard text, a stored image path).
    pub fn byte_slice(&self, range: Range<usize>) -> String {
        let slice = self.slice(range);
        let mut out = String::with_capacity(slice.end - slice.start);
        slice.append_to(&mut out);
        out
    }

    /// The single byte at offset `at`, or `None` past the end. Reads one byte
    /// without materializing a slice, for the "is this a newline?" style checks.
    pub fn byte(&self, at: usize) -> Option<u8> {
        if at >= self.root.bytes {
            return None;
        }
        Some(byte_at(&self.root, at))
    }

    /// The byte offset where line `line` begins (line 0 at byte 0). `line` at or
    /// past [`len_lines`](Self::len_lines) returns the end of the rope.
    pub fn line_to_byte(&self, line: usize) -> usize {
        if line == 0 {
            return 0;
        }
        if line >= self.len_lines() {
            return self.root.bytes;
        }
        line_start(&self.root, line)
    }

    /// The line index containing byte offset `byte`: the count of newlines
    /// strictly before it.
    pub fn byte_to_line(&self, byte: usize) -> usize {
        let byte = byte.min(self.root.bytes);
        newlines_before(&self.root, byte)
    }

    /// The number of leading bytes this rope shares with `other`. Walks both
    /// leaf-byte streams from the front (no allocation, at byte-comparison speed),
    /// so diffing an edited document against its previous version is cheap; the
    /// shared-root fast path makes an untouched document O(1).
    pub fn common_prefix_len(&self, other: &Rope) -> usize {
        if Rc::ptr_eq(&self.root, &other.root) {
            return self.root.bytes;
        }
        let mut ia = Leaves::forward(&self.root);
        let mut ib = Leaves::forward(&other.root);
        let (mut a, mut b): (&[u8], &[u8]) = (&[], &[]);
        let mut matched = 0;
        loop {
            if a.is_empty() {
                match ia.next() {
                    Some(s) => a = s.as_bytes(),
                    None => return matched,
                }
            }
            if b.is_empty() {
                match ib.next() {
                    Some(s) => b = s.as_bytes(),
                    None => return matched,
                }
            }
            let n = a.len().min(b.len());
            let mut i = 0;
            while i < n && a[i] == b[i] {
                i += 1;
            }
            matched += i;
            if i < n {
                return matched;
            }
            a = &a[n..];
            b = &b[n..];
        }
    }

    /// The number of trailing bytes this rope shares with `other`, up to `cap`
    /// bytes (so a caller can keep the suffix from overlapping a known common
    /// prefix). Walks both leaf-byte streams from the back.
    pub fn common_suffix_len(&self, other: &Rope, cap: usize) -> usize {
        let mut ia = Leaves::backward(&self.root);
        let mut ib = Leaves::backward(&other.root);
        let (mut a, mut b): (&[u8], &[u8]) = (&[], &[]);
        let mut matched = 0;
        while matched < cap {
            if a.is_empty() {
                match ia.next() {
                    Some(s) => a = s.as_bytes(),
                    None => return matched,
                }
            }
            if b.is_empty() {
                match ib.next() {
                    Some(s) => b = s.as_bytes(),
                    None => return matched,
                }
            }
            let n = a.len().min(b.len()).min(cap - matched);
            let mut i = 0;
            while i < n && a[a.len() - 1 - i] == b[b.len() - 1 - i] {
                i += 1;
            }
            matched += i;
            if i < n {
                return matched;
            }
            a = &a[..a.len() - n];
            b = &b[..b.len() - n];
        }
        matched
    }
}

/// A borrowed, read-only view into a byte range of a [`Rope`]. It holds no copy
/// of the text: just a reference to the rope's tree and a `[start, end)` byte
/// range, so producing one is O(1) and allocation-free.
///
/// Because the rope stores text in separate leaf chunks, a slice is contiguous
/// in memory only when it falls inside a single leaf. [`as_str`](Self::as_str)
/// hands back the borrow in that case (the common one for a single line) and
/// `None` when the range straddles a chunk boundary; [`as_cow`](Self::as_cow)
/// wraps that decision up, borrowing when it can and copying only when it must.
/// [`chunks`](Self::chunks) walks the pieces in order without allocating.
#[derive(Clone, Copy)]
pub struct RopeSlice<'a> {
    root: &'a Rc<Inner>,
    start: usize,
    end: usize,
}

impl<'a> RopeSlice<'a> {
    /// The slice as a contiguous `&str`, or `None` when it spans more than one
    /// leaf. A single logical line usually lives in one leaf, so layout takes
    /// the borrow and never allocates for it.
    pub fn as_str(&self) -> Option<&'a str> {
        single_leaf(self.root, self.start, self.end)
    }

    /// The slice as a `Cow<str>`: borrowed when it is contiguous (no
    /// allocation), owned only when it straddles a chunk boundary. `Cow<str>`
    /// derefs and indexes like `&str`, so callers use it in place of an owned
    /// line without caring which case they got.
    pub fn as_cow(&self) -> Cow<'a, str> {
        match self.as_str() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(self.to_string()),
        }
    }

    /// Iterate the slice's text as the contiguous `&str` pieces it spans, in
    /// order, without allocating. One piece when the slice is single-leaf.
    pub fn chunks(&self) -> Chunks<'a> {
        Chunks {
            stack: vec![(self.root, 0)],
            start: self.start,
            end: self.end,
        }
    }

    /// Append the slice's bytes to `out`, reusing the caller's buffer instead of
    /// allocating a fresh `String`.
    pub fn append_to(&self, out: &mut String) {
        if self.start < self.end {
            append_range(self.root, self.start, self.end, out);
        }
    }
}

impl fmt::Display for RopeSlice<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in self.chunks() {
            f.write_str(chunk)?;
        }
        Ok(())
    }
}

/// An iterator over the contiguous `&str` pieces of a [`RopeSlice`], left to
/// right, trimmed to the slice's byte range. Subtrees outside the range are
/// pruned, so it visits only the leaves the slice actually touches.
pub struct Chunks<'a> {
    /// Each entry is a subtree and its absolute start offset within the rope.
    stack: Vec<(&'a Rc<Inner>, usize)>,
    start: usize,
    end: usize,
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        while let Some((node, off)) = self.stack.pop() {
            // Skip a subtree that lies wholly before or after the range.
            if off + node.bytes <= self.start || off >= self.end {
                continue;
            }
            match &node.kind {
                Kind::Leaf(s) => {
                    let lo = self.start.saturating_sub(off);
                    let hi = (self.end - off).min(s.len());
                    if lo < hi {
                        return Some(&s[lo..hi]);
                    }
                }
                Kind::Branch(l, r) => {
                    // Push right then left so the left child pops (yields) first.
                    self.stack.push((r, off + l.bytes));
                    self.stack.push((l, off));
                }
            }
        }
        None
    }
}

/// The bytes `[start, end)` of `node` as one `&str`, or `None` if they span more
/// than one leaf. `start`/`end` are relative to `node`.
fn single_leaf(node: &Inner, start: usize, end: usize) -> Option<&str> {
    match &node.kind {
        Kind::Leaf(s) => Some(&s[start..end]),
        Kind::Branch(l, r) => {
            let lb = l.bytes;
            if end <= lb {
                single_leaf(l, start, end)
            } else if start >= lb {
                single_leaf(r, start - lb, end - lb)
            } else {
                None
            }
        }
    }
}

/// The byte at offset `at` within `node` (`at < node.bytes`, guaranteed by the
/// descent, so the leaf index is in bounds).
fn byte_at(node: &Inner, at: usize) -> u8 {
    match &node.kind {
        Kind::Leaf(s) => s.as_bytes()[at],
        Kind::Branch(l, r) => {
            if at < l.bytes {
                byte_at(l, at)
            } else {
                byte_at(r, at - l.bytes)
            }
        }
    }
}

/// A lazy left-to-right or right-to-left iterator over a subtree's leaf strings,
/// for streaming byte comparison without materializing the text.
struct Leaves<'a> {
    stack: Vec<&'a Rc<Inner>>,
    /// When true, yield leaves right-to-left (the stack pushes the left child last
    /// so the right child pops first).
    reverse: bool,
}

impl<'a> Leaves<'a> {
    fn forward(root: &'a Rc<Inner>) -> Self {
        Leaves {
            stack: vec![root],
            reverse: false,
        }
    }

    fn backward(root: &'a Rc<Inner>) -> Self {
        Leaves {
            stack: vec![root],
            reverse: true,
        }
    }
}

impl<'a> Iterator for Leaves<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        while let Some(node) = self.stack.pop() {
            match &node.kind {
                Kind::Leaf(s) => return Some(s),
                Kind::Branch(l, r) => {
                    // Push so the near child pops first: left-first for forward,
                    // right-first for backward.
                    if self.reverse {
                        self.stack.push(l);
                        self.stack.push(r);
                    } else {
                        self.stack.push(r);
                        self.stack.push(l);
                    }
                }
            }
        }
        None
    }
}

impl Default for Rope {
    fn default() -> Self {
        Rope::new()
    }
}

impl fmt::Display for Rope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_node(&self.root, f)
    }
}

// --- Node construction -------------------------------------------------------

fn leaf(s: String) -> Rc<Inner> {
    Rc::new(Inner {
        bytes: s.len(),
        newlines: bytecount_nl(&s),
        height: 1,
        kind: Kind::Leaf(s),
    })
}

fn branch(l: Rc<Inner>, r: Rc<Inner>) -> Rc<Inner> {
    Rc::new(Inner {
        bytes: l.bytes + r.bytes,
        newlines: l.newlines + r.newlines,
        height: 1 + l.height.max(r.height),
        kind: Kind::Branch(l, r),
    })
}

/// Join two subtrees. Adjacent small leaves merge into one, so an edit does not
/// fragment the tree into many tiny chunks; otherwise a plain branch.
fn concat(l: Rc<Inner>, r: Rc<Inner>) -> Rc<Inner> {
    if l.bytes == 0 {
        return r;
    }
    if r.bytes == 0 {
        return l;
    }
    if let (Kind::Leaf(a), Kind::Leaf(b)) = (&l.kind, &r.kind) {
        if a.len() + b.len() <= MAX_LEAF {
            let mut s = String::with_capacity(a.len() + b.len());
            s.push_str(a);
            s.push_str(b);
            return leaf(s);
        }
    }
    branch(l, r)
}

/// Split `node` into the subtrees holding `[0, at)` and `[at, len)`. `at` is a
/// byte offset on a `char` boundary.
fn split(node: &Rc<Inner>, at: usize) -> (Rc<Inner>, Rc<Inner>) {
    match &node.kind {
        Kind::Leaf(s) => {
            debug_assert!(
                at <= s.len() && s.is_char_boundary(at),
                "split off a char boundary"
            );
            (leaf(s[..at].to_string()), leaf(s[at..].to_string()))
        }
        Kind::Branch(l, r) => {
            let lb = l.bytes;
            if at < lb {
                let (ll, lr) = split(l, at);
                (ll, concat(lr, r.clone()))
            } else if at > lb {
                let (rl, rr) = split(r, at - lb);
                (concat(l.clone(), rl), rr)
            } else {
                (l.clone(), r.clone())
            }
        }
    }
}

/// Build a balanced subtree from `s`, chunking it into `char`-aligned leaves of
/// at most [`MAX_LEAF`] bytes.
fn build(s: &str) -> Rc<Inner> {
    if s.len() <= MAX_LEAF {
        return leaf(s.to_string());
    }
    let mut leaves = Vec::new();
    let mut start = 0;
    while start < s.len() {
        let mut end = (start + MAX_LEAF).min(s.len());
        while end > start && !s.is_char_boundary(end) {
            end -= 1;
        }
        leaves.push(leaf(s[start..end].to_string()));
        start = end;
    }
    combine(&leaves)
}

/// Combine ordered leaves into a balanced tree by recursive halving.
fn combine(leaves: &[Rc<Inner>]) -> Rc<Inner> {
    match leaves.len() {
        0 => leaf(String::new()),
        1 => leaves[0].clone(),
        n => {
            let mid = n / 2;
            branch(combine(&leaves[..mid]), combine(&leaves[mid..]))
        }
    }
}

// --- Balancing ---------------------------------------------------------------

/// Keep the tree from degenerating: a subtree tall relative to its size is
/// rebuilt balanced from its leaves. Shallow trees are always left alone.
fn balance(node: Rc<Inner>) -> Rc<Inner> {
    if node.height <= 2 || node.bytes as u64 >= fib(node.height + 2) {
        node
    } else {
        let mut leaves = Vec::new();
        collect_leaves(&node, &mut leaves);
        combine(&leaves)
    }
}

fn collect_leaves(node: &Rc<Inner>, out: &mut Vec<Rc<Inner>>) {
    match &node.kind {
        Kind::Leaf(_) => out.push(node.clone()),
        Kind::Branch(l, r) => {
            collect_leaves(l, out);
            collect_leaves(r, out);
        }
    }
}

/// The `n`-th Fibonacci number (`fib(0) = 0`, `fib(1) = 1`), saturating so a very
/// tall tree never overflows the comparison.
fn fib(n: u32) -> u64 {
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 0..n {
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    a
}

// --- Traversal ---------------------------------------------------------------

fn write_node(node: &Inner, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match &node.kind {
        Kind::Leaf(s) => f.write_str(s),
        Kind::Branch(l, r) => {
            write_node(l, f)?;
            write_node(r, f)
        }
    }
}

/// Append the bytes of `node` in `[start, end)` to `out`. Bounds are within the
/// node and `char`-aligned (leaf boundaries fall on `char` boundaries).
fn append_range(node: &Inner, start: usize, end: usize, out: &mut String) {
    match &node.kind {
        Kind::Leaf(s) => {
            debug_assert!(
                start <= end && end <= s.len(),
                "append_range within the leaf"
            );
            out.push_str(&s[start..end]);
        }
        Kind::Branch(l, r) => {
            let lb = l.bytes;
            if start < lb {
                append_range(l, start, end.min(lb), out);
            }
            if end > lb {
                append_range(r, start.saturating_sub(lb), end - lb, out);
            }
        }
    }
}

/// The byte offset just after the `line`-th newline (the start of line `line`),
/// for `1 <= line <= newlines`.
fn line_start(node: &Inner, line: usize) -> usize {
    match &node.kind {
        Kind::Leaf(s) => {
            let mut count = 0;
            for (i, b) in s.bytes().enumerate() {
                if b == b'\n' {
                    count += 1;
                    if count == line {
                        return i + 1;
                    }
                }
            }
            s.len()
        }
        Kind::Branch(l, r) => {
            if line <= l.newlines {
                line_start(l, line)
            } else {
                l.bytes + line_start(r, line - l.newlines)
            }
        }
    }
}

/// The number of newlines strictly before byte offset `byte` (within `node`).
fn newlines_before(node: &Inner, byte: usize) -> usize {
    match &node.kind {
        Kind::Leaf(s) => {
            debug_assert!(byte <= s.len(), "newlines_before within the leaf");
            s.as_bytes()[..byte].iter().filter(|&&b| b == b'\n').count()
        }
        Kind::Branch(l, r) => {
            if byte <= l.bytes {
                newlines_before(l, byte)
            } else {
                l.newlines + newlines_before(r, byte - l.bytes)
            }
        }
    }
}

fn bytecount_nl(s: &str) -> usize {
    s.bytes().filter(|&b| b == b'\n').count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rope_is_one_empty_line() {
        let r = Rope::new();
        assert_eq!(r.len(), 0);
        assert_eq!(r.len_lines(), 1);
        assert_eq!(r.to_string(), "");
    }

    #[test]
    fn from_str_round_trips_and_counts_lines() {
        let r = Rope::from_str("a\nbc\n\nd");
        assert_eq!(r.to_string(), "a\nbc\n\nd");
        assert_eq!(r.len(), 7);
        assert_eq!(r.len_lines(), 4, "three newlines -> four lines");
    }

    #[test]
    fn common_prefix_and_suffix_match_a_naive_scan() {
        fn naive_prefix(a: &str, b: &str) -> usize {
            a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
        }
        fn naive_suffix(a: &str, b: &str, cap: usize) -> usize {
            a.bytes()
                .rev()
                .zip(b.bytes().rev())
                .take_while(|(x, y)| x == y)
                .count()
                .min(cap)
        }
        // A base long enough to span many leaves, edited several ways so the ropes
        // share structure at shifted offsets.
        let base = "The quick brown fox jumps over the lazy dog. ".repeat(2000);
        let cases = [
            base.clone(),
            format!("X{base}"),
            format!("{base}Y"),
            {
                let mut s = base.clone();
                s.insert(s.len() / 2, 'Z');
                s
            },
            {
                let mut s = base.clone();
                s.replace_range(100..200, "");
                s
            },
            format!("{base}日本語"),
            String::new(),
        ];
        for a in &cases {
            for b in &cases {
                let (ra, rb) = (Rope::from_str(a), Rope::from_str(b));
                let p = ra.common_prefix_len(&rb);
                assert_eq!(p, naive_prefix(a, b), "prefix");
                let cap = a.len().min(b.len()) - p;
                assert_eq!(
                    ra.common_suffix_len(&rb, cap),
                    naive_suffix(a, b, cap),
                    "suffix"
                );
            }
        }
    }

    #[test]
    fn common_prefix_of_a_shared_clone_is_the_whole_length() {
        let mut r = Rope::from_str(&"lorem ipsum dolor ".repeat(500));
        // A clone shares the root, so the fast path returns the full length.
        let before = r.clone();
        assert_eq!(before.common_prefix_len(&before.clone()), before.len());
        // After a one-byte insert, only that byte is outside the shared prefix and
        // suffix; the scan finds them across the still-shared leaves.
        r.insert(r.len() / 2, "!");
        let p = r.common_prefix_len(&before);
        let s = r.common_suffix_len(&before, r.len().min(before.len()) - p);
        assert_eq!(
            p + s,
            before.len(),
            "one inserted byte splits prefix and suffix"
        );
    }

    #[test]
    fn insert_at_start_middle_and_end() {
        let mut r = Rope::from_str("ac");
        r.insert(1, "b");
        assert_eq!(r.to_string(), "abc");
        r.insert(0, "X");
        assert_eq!(r.to_string(), "Xabc");
        r.insert(r.len(), "Z");
        assert_eq!(r.to_string(), "XabcZ");
    }

    #[test]
    fn remove_ranges() {
        let mut r = Rope::from_str("hello world");
        r.remove(5..11); // " world"
        assert_eq!(r.to_string(), "hello");
        r.remove(0..r.len());
        assert_eq!(r.to_string(), "");
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn out_of_range_edits_clamp_and_do_not_panic() {
        let mut r = Rope::from_str("abc");
        r.insert(999, "Z"); // clamps to the end
        assert_eq!(r.to_string(), "abcZ");
        r.remove(2..999); // clamps to the end
        assert_eq!(r.to_string(), "ab");
        r.remove(5..9); // wholly past the end: a no-op
        assert_eq!(r.to_string(), "ab");
    }

    #[test]
    fn byte_slice_reads_subranges() {
        let r = Rope::from_str("hello world");
        assert_eq!(r.byte_slice(0..5), "hello");
        assert_eq!(r.byte_slice(6..11), "world");
        assert_eq!(r.byte_slice(0..r.len()), "hello world");
        assert_eq!(r.byte_slice(3..3), "");
    }

    #[test]
    fn slice_borrows_within_a_leaf_and_copies_across_one() {
        // A short rope lives in a single leaf: every slice is contiguous, so
        // as_str borrows and as_cow is Borrowed (no allocation).
        let r = Rope::from_str("hello world");
        assert_eq!(r.slice(0..5).as_str(), Some("hello"));
        assert!(matches!(r.slice(6..11).as_cow(), Cow::Borrowed("world")));

        // A rope past MAX_LEAF spans several leaves. A slice inside one leaf
        // still borrows; one crossing a leaf boundary has no contiguous &str,
        // so as_str is None and as_cow copies, but both round-trip the bytes.
        let big = "abcdefghij".repeat(400); // 4000 bytes, > MAX_LEAF
        let r = Rope::from_str(&big);
        let mut saw_borrow = false;
        let mut saw_copy = false;
        for start in (0..big.len()).step_by(97) {
            let end = (start + 300).min(big.len());
            let slice = r.slice(start..end);
            assert_eq!(slice.to_string(), big[start..end], "slice bytes");
            assert_eq!(slice.as_cow(), big[start..end], "as_cow bytes");
            match slice.as_str() {
                Some(s) => {
                    saw_borrow = true;
                    assert_eq!(s, &big[start..end]);
                }
                None => saw_copy = true,
            }
        }
        assert!(saw_borrow, "some slices fall inside a single leaf");
        assert!(saw_copy, "some slices straddle a leaf boundary");
    }

    #[test]
    fn chunks_concatenate_to_the_slice() {
        let big = "The quick brown fox. ".repeat(300);
        let r = Rope::from_str(&big);
        let slice = r.slice(50..2500);
        let joined: String = slice.chunks().collect();
        assert_eq!(joined, big[50..2500]);
        // An empty slice yields no chunks.
        assert_eq!(r.slice(10..10).chunks().count(), 0);
    }

    #[test]
    fn byte_reads_single_bytes_and_bounds() {
        let r = Rope::from_str("a\nb");
        assert_eq!(r.byte(0), Some(b'a'));
        assert_eq!(r.byte(1), Some(b'\n'));
        assert_eq!(r.byte(2), Some(b'b'));
        assert_eq!(r.byte(3), None, "past the end is None");
        // Reads the right byte deep inside a multi-leaf rope.
        let big: String = (0..4000).map(|i| (b'0' + (i % 10) as u8) as char).collect();
        let r = Rope::from_str(&big);
        for at in [0, 999, 1024, 2048, 3999] {
            assert_eq!(r.byte(at), Some(big.as_bytes()[at]));
        }
    }

    #[test]
    fn line_and_byte_lookups_agree() {
        // Lines: "a"(0..1) "\n" "bb"(2..4) "\n" ""(5) "\n" "c"(6..7)
        let r = Rope::from_str("a\nbb\n\nc");
        assert_eq!(r.line_to_byte(0), 0);
        assert_eq!(r.line_to_byte(1), 2);
        assert_eq!(r.line_to_byte(2), 5);
        assert_eq!(r.line_to_byte(3), 6);
        // Past the last line clamps to the end.
        assert_eq!(r.line_to_byte(4), r.len());
        assert_eq!(r.line_to_byte(99), r.len());
        // byte_to_line is the inverse: newlines strictly before the offset.
        assert_eq!(r.byte_to_line(0), 0);
        assert_eq!(r.byte_to_line(1), 0);
        assert_eq!(r.byte_to_line(2), 1);
        assert_eq!(r.byte_to_line(5), 2);
        assert_eq!(r.byte_to_line(6), 3);
    }

    #[test]
    fn clone_is_independent_after_edits() {
        // Persistence: a clone (an undo snapshot) is unaffected by later edits.
        let original = Rope::from_str("snapshot");
        let mut edited = original.clone();
        edited.insert(0, "X"); // "Xsnapshot"
        edited.remove(4..8); // drop "psho" -> "Xsnat"
        assert_eq!(original.to_string(), "snapshot", "the clone is untouched");
        assert_eq!(edited.to_string(), "Xsnat");
    }

    #[test]
    fn handles_text_far_larger_than_a_leaf() {
        // Force many leaves and branching, then round-trip and edit across them.
        let big: String = (0..5000)
            .map(|i| if i % 7 == 0 { '\n' } else { 'x' })
            .collect();
        let mut r = Rope::from_str(&big);
        assert_eq!(r.to_string(), big);
        assert_eq!(r.len(), big.len());
        assert_eq!(r.len_lines(), big.matches('\n').count() + 1);
        // An edit deep inside still matches a String doing the same.
        let mut oracle = big.clone();
        r.insert(2500, "INSERTED");
        oracle.insert_str(2500, "INSERTED");
        assert_eq!(r.to_string(), oracle);
    }

    #[test]
    fn multibyte_text_round_trips() {
        let s = "héllo 😀 wörld\n日本語";
        let r = Rope::from_str(s);
        assert_eq!(r.to_string(), s);
        assert_eq!(r.len(), s.len());
        // Slice on char boundaries: take the "héllo" prefix (h + é(2 bytes)+llo).
        assert_eq!(r.byte_slice(0..6), "héllo");
    }

    #[test]
    fn matches_a_string_oracle_under_many_edits() {
        // The strongest test: drive a Rope and a String through the same
        // deterministic stream of edits and assert they stay byte-identical.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rope = Rope::new();
        let mut oracle = String::new();
        let samples = ["a", "b", "\n", "héllo", "😀", "  ", "x\ny", "word "];

        for _ in 0..4000 {
            // Insert when empty or two times out of three; otherwise remove.
            if oracle.is_empty() || !next(&mut state).is_multiple_of(3) {
                let s = samples[next(&mut state) as usize % samples.len()];
                let at = boundary(&oracle, next(&mut state) as usize % (oracle.len() + 1));
                rope.insert(at, s);
                oracle.insert_str(at, s);
            } else {
                let a = boundary(&oracle, next(&mut state) as usize % (oracle.len() + 1));
                let b = boundary(&oracle, next(&mut state) as usize % (oracle.len() + 1));
                let (lo, hi) = (a.min(b), a.max(b));
                rope.remove(lo..hi);
                oracle.replace_range(lo..hi, "");
            }
            assert_eq!(rope.len(), oracle.len());
            assert_eq!(rope.len_lines(), oracle.matches('\n').count() + 1);
        }
        assert_eq!(rope.to_string(), oracle);

        // Line and slice queries also agree with the oracle's own computation.
        for line in 0..rope.len_lines() {
            let start = nth_line_start(&oracle, line);
            assert_eq!(rope.line_to_byte(line), start, "line {line} start");
        }
    }

    /// A small deterministic PRNG (an LCG); `Math.random` is unavailable and we
    /// want a reproducible edit stream anyway.
    fn next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 16
    }

    /// Snap `i` down to the nearest `char` boundary at or before it.
    fn boundary(s: &str, i: usize) -> usize {
        let mut i = i.min(s.len());
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    /// The byte offset where `line` starts, computed independently of the rope.
    fn nth_line_start(s: &str, line: usize) -> usize {
        if line == 0 {
            return 0;
        }
        let mut seen = 0;
        for (i, b) in s.bytes().enumerate() {
            if b == b'\n' {
                seen += 1;
                if seen == line {
                    return i + 1;
                }
            }
        }
        s.len()
    }
}
