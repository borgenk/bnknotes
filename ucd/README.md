# Vendored Unicode Character Database

The UCD data files the editor generates its grapheme tables from, pinned to one
Unicode version and committed so a build (and a table regeneration) never touches
the network. Everything here is verbatim from the Unicode Consortium's public
files.

**Version: 18.0.0**

| File | Feeds | Property used |
| --- | --- | --- |
| `GraphemeBreakProperty.txt` | `src/platform/grapheme_tables.rs` | Grapheme_Cluster_Break |
| `emoji-data.txt` | `src/platform/grapheme_tables.rs` | Extended_Pictographic (rule GB11) |
| `DerivedCoreProperties.txt` | `src/platform/grapheme_tables.rs` | Indic_Conjunct_Break (rule GB9c) |
| `GraphemeBreakTest.txt` | `grapheme.rs` test | the official UAX #29 conformance suite |

## Refreshing on a Unicode bump

These came from the Arch `unicode-character-database` package
(`/usr/share/unicode/`), which tracks the latest Unicode. To move to a new version:

1. `make deps` (installs `unicode-character-database`), or fetch the files from
   `https://www.unicode.org/Public/<version>/ucd/`.
2. Copy the four files above into this directory (from `/usr/share/unicode/`, note
   the `auxiliary/` and `emoji/` subpaths for the grapheme and emoji files).
3. `make tables`, which regenerates `src/platform/grapheme_tables.rs`. It is
   deterministic and offline: rerunning it reproduces the committed file byte for
   byte.
4. `cargo test`, the grapheme conformance test over `GraphemeBreakTest.txt` must
   pass. Update the case-count note in that test if the suite size changed.
