//! Branchless byte-scan primitives, written so the compiler autovectorizes them
//! (SSE2/AVX2 on x86-64, NEON on aarch64) with no intrinsics and no dependency.
//! They back the rope's newline counting and line finding, and newline
//! normalization on load, the hottest whole-buffer scans in a large-file open.

/// Count the occurrences of `needle` in `hay`.
///
/// Summing `(b == needle) as usize` is a branchless reduction the compiler turns
/// into vector compares plus a horizontal add, rather than the data-dependent
/// conditional increment a `filter().count()` compiles to. On a multi-megabyte
/// buffer that is several times faster.
pub fn count_byte(hay: &[u8], needle: u8) -> usize {
    hay.iter().map(|&b| usize::from(b == needle)).sum()
}

/// The byte index of the first `needle` in `hay`, or `None`. A byte-equality
/// `position` vectorizes to a memchr-style scan.
pub fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    hay.iter().position(|&b| b == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_byte_matches_a_naive_count() {
        let cases: [&[u8]; 5] = [b"", b"aaa", b"a\nb\nc\n", b"no newlines", &[b'\n'; 1000]];
        for hay in cases {
            let want = hay.iter().filter(|&&b| b == b'\n').count();
            assert_eq!(count_byte(hay, b'\n'), want, "count in {hay:?}");
        }
        // A buffer long enough to span many vector widths, newlines scattered.
        let big: Vec<u8> = (0..10_000u32)
            .map(|i| if i % 7 == 0 { b'\n' } else { b'x' })
            .collect();
        assert_eq!(
            count_byte(&big, b'\n'),
            big.iter().filter(|&&b| b == b'\n').count()
        );
    }

    #[test]
    fn find_byte_matches_position() {
        assert_eq!(find_byte(b"abc\ndef", b'\n'), Some(3));
        assert_eq!(find_byte(b"\nfirst", b'\n'), Some(0));
        assert_eq!(find_byte(b"no newline", b'\n'), None);
        assert_eq!(find_byte(b"", b'\n'), None);
        // Match past a vector-width boundary is still found at the right index.
        let mut buf = vec![b'x'; 5000];
        buf[4097] = b'\n';
        assert_eq!(find_byte(&buf, b'\n'), Some(4097));
    }
}
