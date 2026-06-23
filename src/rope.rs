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

    /// Copy out the bytes in `range` as a `String` (assumed `char`-aligned). The
    /// rope is not contiguous, so callers materialize the (short) slice they
    /// need, e.g. a single visual line.
    pub fn byte_slice(&self, range: Range<usize>) -> String {
        let start = range.start.min(self.root.bytes);
        let end = range.end.min(self.root.bytes);
        let mut out = String::with_capacity(end.saturating_sub(start));
        if start < end {
            append_range(&self.root, start, end, &mut out);
        }
        out
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
        Kind::Leaf(s) => (leaf(s[..at].to_string()), leaf(s[at..].to_string())),
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
        Kind::Leaf(s) => out.push_str(&s[start..end]),
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
        Kind::Leaf(s) => s.as_bytes()[..byte].iter().filter(|&&b| b == b'\n').count(),
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
