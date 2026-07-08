//! Two performance harnesses for the editor, sharing one embedded-fixture and
//! FFI-font setup.
//!
//! **Tier 1, the gate** (`bnknotes --perf`): a handful of hot-path microbenchmarks
//! (full re-layout, pointer hit-testing, glyph rasterization, frame batching,
//! image resample/reflow/upload), timed CPU-only (no Wayland, no GPU submission)
//! against a committed baseline in `perf/baseline.txt`. It writes `perf/latest.txt`
//! and exits nonzero if any scenario regressed past [`REGRESSION_RATIO`], so a
//! script or CI can gate on it. `--save` accepts the current numbers as the new
//! baseline. Built with `--features perf-alloc` (what `make perf` does) it also
//! reports allocations per operation, a deterministic signal that gates too.
//!
//! **Tier 2, the lab** (`bnknotes --perf-lab`): whole-pipeline scenarios over large
//! generated stress corpora (see [`crate::corpus`]), reporting throughput,
//! CPU-seconds, resident memory, and allocations, and appending each run to
//! `perf/history.jsonl` keyed by the git commit. This is for finding work and
//! learning, not gating; it needs no GPU either, but it is not deterministic in
//! wall-clock the way the gate's counters are.
//!
//! Always build `--release`: a debug build's numbers are meaningless and not
//! comparable to a release baseline.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::alloc;
use crate::corpus;
use crate::error::{Error, Result};
use crate::find::FindState;
use crate::freetype::Fonts;
use crate::image::{Image, ImageStore};
use crate::layout::{ImageUi, Layout, LayoutCtx, LineIndex};
use crate::markdown;
use crate::paint::{build_display_list, PaintParams};
use crate::rope::Rope;
use crate::sys;
use crate::text::TextBuffer;
use crate::theme::{FONT_SIZE, MARGIN};

/// Where the committed baseline lives, where each run records its numbers, and
/// where the lab appends its time series.
const BASELINE_PATH: &str = "perf/baseline.txt";
const LATEST_PATH: &str = "perf/latest.txt";
const HISTORY_PATH: &str = "perf/history.jsonl";

/// A scenario counts as a regression when it is at least this much slower (or,
/// for allocations, this much larger) than the baseline. Wide enough to absorb
/// run-to-run timing noise, tight enough to catch the order-of-magnitude
/// regressions this harness exists to prevent.
const REGRESSION_RATIO: f64 = 1.10;

/// The surface the harness paints into, matching the app's initial window.
const WIDTH: usize = 800;
const HEIGHT: usize = 600;

/// Repetitions per lab scenario. The macro scenarios are expensive (a full
/// layout over megabytes), so a handful of rounds is enough to take the minimum.
const LAB_REPS: u32 = 5;

/// The default and maximum corpus size for the lab, in mebibytes.
const LAB_DEFAULT_MIB: usize = 4;
const LAB_MAX_MIB: usize = 512;

/// One measured Tier 1 scenario: its best time and, when measured, its allocation
/// count per operation (0 with the counting allocator absent).
struct Sample {
    name: &'static str,
    ns: u64,
    allocs: u64,
}

/// A baseline entry: the recorded time, and allocations when the baseline was
/// saved with the counting allocator on.
#[derive(Clone, Copy)]
struct Base {
    ns: u64,
    allocs: Option<u64>,
}

// ---------------------------------------------------------------------------
// Tier 1: the regression gate.
// ---------------------------------------------------------------------------

/// Run the gate: measure, write `latest`, then compare to (or establish) the
/// baseline. Returns an error when a scenario regressed and `--save` was not
/// given, so the process exit code signals the regression to a script or CI.
pub fn run(args: &[String]) -> Result<()> {
    let save = args.iter().any(|a| a == "--save");
    eprintln!("bnknotes perf: measuring (build --release for meaningful numbers)...");
    if !alloc::ENABLED {
        eprintln!(
            "  allocations unmeasured; build --features perf-alloc (or `make perf`) to gate them"
        );
    }
    let samples = measure()?;
    write_samples(LATEST_PATH, &samples)?;

    let Some(baseline) = load_baseline(BASELINE_PATH)? else {
        write_samples(BASELINE_PATH, &samples)?;
        print_table(&samples, &HashMap::new());
        println!("\nNo baseline existed; wrote {BASELINE_PATH}. Commit it to compare future runs.");
        return Ok(());
    };

    let regressed = print_table(&samples, &baseline);
    if save {
        write_samples(BASELINE_PATH, &samples)?;
        println!("\nUpdated baseline at {BASELINE_PATH}.");
        return Ok(());
    }
    if regressed.is_empty() {
        println!(
            "\nNo regressions over {:.0}% vs baseline. Latest written to {LATEST_PATH}.",
            (REGRESSION_RATIO - 1.0) * 100.0
        );
        return Ok(());
    }
    Err(Error::msg(format!(
        "perf regression over {:.0}% in: {}. Re-run with --save to accept.",
        (REGRESSION_RATIO - 1.0) * 100.0,
        regressed.join(", ")
    )))
}

/// The environment every perf mode boots from: the bundled faces at all
/// body/heading sizes, the content wrap width (the window less its margins), and
/// an empty image store (the perf corpora carry no images, but build/paint need
/// one to borrow).
fn boot() -> Result<(Fonts, i32, ImageStore)> {
    let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
    let wrap = WIDTH as i32 - 2 * MARGIN;
    Ok((fonts, wrap, ImageStore::new(None)))
}

/// Measure every scenario against the embedded fixture and a warm font set.
fn measure() -> Result<Vec<Sample>> {
    let (fonts, wrap, images) = boot()?;
    let text = fixture();
    let rope = Rope::from_str(&text);

    // Build once up front: this warms the per-character advance cache, so the
    // timed `layout_build` below measures the steady re-layout cost (the per-frame
    // cost while editing), not the one-time cold measure of every glyph.
    let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));

    let mut out = Vec::new();

    // Full re-layout of the document. Guards the advance cache above all: if glyph
    // measuring went uncached again, this scenario balloons roughly 75x.
    out.push(sample("layout_build", 50, || {
        Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0))
            .lines()
            .len()
    }));

    // Windowed re-layout: build only a viewport-sized band, the cost now paid on a
    // cursor move or a reveal change instead of a full `layout_build`. Bounded by
    // the window, not the document, so it should sit far below `layout_build` and
    // stay flat as documents grow. The index is warmed first (its build is the
    // per-edit cost, guarded separately below).
    let index = LineIndex::build(&fonts, &rope, wrap, FONT_SIZE, &images);
    let band = 30.min(index.len());
    let window_ctx = LayoutCtx::new(
        &fonts,
        &rope,
        wrap,
        FONT_SIZE,
        (0, 0),
        &images,
        ImageUi::default(),
    );
    out.push(sample("layout_window", 500, || {
        Layout::build_window(&window_ctx, MARGIN, MARGIN, &index, 0, band)
            .lines()
            .len()
    }));

    // Incremental index update after a one-character edit: the per-keystroke cost
    // that replaces a full `layout_build`. It re-measures only the edited line and
    // reuses the rest, so it stays flat as documents grow. (The clone per iteration
    // is measurement scaffolding, since `update` mutates in place.)
    let edited = edited_middle(&text);
    let update_ctx = LayoutCtx::for_index(&fonts, &edited, wrap, FONT_SIZE, &images);
    out.push(sample("index_update", 500, || {
        let mut idx = index.clone();
        idx.update(&update_ctx, &rope);
        idx.len()
    }));

    // Hit-test the pointer against the layout: the per-motion cost of a drag.
    out.push(sample("offset_at", 2000, || {
        layout.offset_at(&fonts, &rope, (WIDTH / 2) as f32, (HEIGHT / 2) as f32)
    }));

    // A decoded image the editor sizes and uploads to the GPU, reused by the image
    // scenarios below (reflow at a fresh width, and the GPU texture upload).
    let image_rope = Rope::from_str("# Perf\n\n![pic](perf.png)\n\nbody text under the image\n");
    let image_store = ImageStore::from_decoded(vec![(
        "perf.png".to_string(),
        Image {
            pixels: vec![0xff_30_60_90; 400 * 300],
            width: 400,
            height: 300,
        },
    )]);
    // Cursor on the heading (active `(0, 0)`): the image renders as a picture.
    let image_ctx = LayoutCtx::new(
        &fonts,
        &image_rope,
        wrap,
        FONT_SIZE,
        (0, 0),
        &image_store,
        ImageUi::default(),
    );
    let image_layout = Layout::build_with_images(&image_ctx, MARGIN, MARGIN);

    // Raw glyph rasterization of printable ASCII (the uncached FreeType path), so a
    // regression in the rasterizer itself is visible independent of the cache. The
    // GPU path still rasterizes every glyph once, into its texture atlas.
    let face = fonts.face(FONT_SIZE, crate::freetype::FontStyle::Regular);
    out.push(sample("glyph_raster_ascii", 20, || {
        let mut sum = 0usize;
        for c in 0x20u8..0x7f {
            sum += face.rasterize(c as char).coverage.len();
        }
        sum
    }));

    // A full text frame's display list, batched below into GPU vertices.
    let frame_list = build_display_list(&fonts, &layout, &rope, &params(0, None));

    // The GPU path's CPU-side frame cost: batch the display list into vertices
    // (glyph lookups against warm atlases, quad emission, batch splitting).
    // This runs even without a GPU, so the harness can hold it; the GPU-side
    // frame time is only observable via BNKNOTES_STATS on a live run.
    let mut glyph_cache = crate::gpu::GlyphCache::new();
    // Warm the atlases so the steady state (no uploads) is what is measured.
    let _ = crate::gpu::build_frame(&fonts, &[&images], &frame_list, &mut glyph_cache, &|_| true);
    out.push(sample("gpu_batch_frame", 500, || {
        crate::gpu::build_frame(&fonts, &[&images], &frame_list, &mut glyph_cache, &|_| true)
            .vertices
            .len()
    }));

    // Image resampling: the area/bilinear pass that scales a source bitmap to a
    // display size. It is cached per (source, size), so it runs off the frame path;
    // this guards the resampler itself (now used for color-emoji glyph strikes).
    let native: Vec<u32> = vec![0xff_30_60_90; 400 * 300];
    out.push(sample("image_resample", 200, || {
        crate::pixel::resample_argb(&native, 400, 300, 200, 150).len()
    }));

    // Re-laying-out a document containing an image at a fresh width, as a splitter
    // or window resize does on every drag event (the layout build has no cache).
    let reflow_ctx = LayoutCtx::new(
        &fonts,
        &image_rope,
        wrap - 40,
        FONT_SIZE,
        (0, 0),
        &image_store,
        ImageUi::default(),
    );
    out.push(sample("layout_reflow_image", 200, || {
        Layout::build_with_images(&reflow_ctx, MARGIN, MARGIN).text_height()
    }));

    // The GPU frame build for an image the renderer does not yet hold: it uploads
    // the native texture (`known_images` = false forces it each call). After the
    // resize fix this is paid once per image, not once per drag event, because the
    // key is the native size rather than the changing display size.
    let image_list = build_display_list(&fonts, &image_layout, &image_rope, &params(0, None));
    out.push(sample("image_upload_gpu", 200, || {
        crate::gpu::build_frame(
            &fonts,
            &[&image_store],
            &image_list,
            &mut glyph_cache,
            &|_| false,
        )
        .image_uploads
        .len()
    }));

    Ok(out)
}

/// Paint parameters for a frame at `scroll_y` with an optional `selection`.
fn params(scroll_y: i32, selection: Option<(usize, usize)>) -> PaintParams<'static> {
    PaintParams {
        width: WIDTH,
        height: HEIGHT,
        scroll_y,
        selection,
        cursor: 0,
        caret_visible: true,
        find: None,
    }
}

/// Measure one Tier 1 scenario into a [`Sample`]: allocations from one clean
/// (untimed) run, then the best time over a few rounds.
fn sample<T>(name: &'static str, iters: u64, f: impl FnMut() -> T) -> Sample {
    let (ns, allocs) = bench(iters, f);
    Sample { name, ns, allocs }
}

/// The best (minimum) ns per call over `rounds`, each timing `inner` calls, plus
/// the allocations one call makes. The minimum is the least noisy time estimator
/// for a micro-benchmark: the round least disturbed by the scheduler. The
/// allocation count is taken from a single warmed run, untimed (after a reset so a
/// first-call lazy init is not charged as steady state), so counting never skews
/// the time. `black_box` stops the optimizer eliding the work as dead. Shared by
/// the Tier 1 [`bench`] (many `inner` calls per round) and the Tier 2 [`lab`]
/// (one whole-document call per round).
fn best_of<T>(rounds: u32, inner: u64, mut f: impl FnMut() -> T) -> (u64, u64) {
    black_box(f());
    alloc::reset();
    black_box(f());
    let allocs = alloc::snapshot().allocs;

    let mut best = u64::MAX;
    for _ in 0..rounds.max(1) {
        let start = Instant::now();
        for _ in 0..inner.max(1) {
            black_box(f());
        }
        best = best.min((start.elapsed().as_nanos() as u64) / inner.max(1));
    }
    (best, allocs)
}

/// Time a Tier 1 scenario: warm the cache, then the best per-call ns over five
/// rounds of `iters` calls each.
fn bench<T>(iters: u64, mut f: impl FnMut() -> T) -> (u64, u64) {
    for _ in 0..iters.min(16) {
        black_box(f());
    }
    best_of(5, iters, f)
}

/// How a measured number compares to its baseline: the one classification behind
/// both the time and allocation columns and the regression gate, so all three
/// agree by construction.
#[derive(Clone, Copy)]
enum Delta {
    /// At least [`REGRESSION_RATIO`] worse than the baseline.
    Regressed,
    /// At least [`REGRESSION_RATIO`] better than the baseline.
    Improved,
    /// Within the noise band either way.
    Flat,
}

impl Delta {
    /// Classify `current` against `base` by the same ratio the gate uses.
    fn classify(current: u64, base: u64) -> Delta {
        if is_regression(current, base) {
            Delta::Regressed
        } else if base as f64 / current.max(1) as f64 >= REGRESSION_RATIO {
            Delta::Improved
        } else {
            Delta::Flat
        }
    }

    /// The word printed beside the percentage, empty when flat.
    fn label(self) -> &'static str {
        match self {
            Delta::Regressed => "REGRESSION",
            Delta::Improved => "improved",
            Delta::Flat => "",
        }
    }

    fn regressed(self) -> bool {
        matches!(self, Delta::Regressed)
    }
}

/// Print a current-vs-baseline table and return the names that regressed. Time is
/// always compared; allocations are compared only when both the run and the
/// baseline measured them (the `perf-alloc` feature was on for each), so an
/// unmeasured run never reads as a spurious improvement to zero.
fn print_table(samples: &[Sample], baseline: &HashMap<String, Base>) -> Vec<&'static str> {
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    println!(
        "\n{:<20} {:>11} {:>11} {:>8}   {:>10} {:>10} {:>8}",
        "metric", "ns", "base ns", "Δ", "allocs", "base", "Δ"
    );
    let mut regressed = Vec::new();
    for s in sorted {
        let base = baseline.get(s.name);

        // Time column: always compared when a baseline row exists.
        let ns_delta = base.map(|b| Delta::classify(s.ns, b.ns));
        let (base_ns_col, ns_flag) = match (base, ns_delta) {
            (Some(b), Some(delta)) => (
                b.ns.to_string(),
                format!("{:+.1}% {}", ratio_pct(s.ns, b.ns), delta.label()),
            ),
            _ => ("(new)".to_string(), String::new()),
        };

        // Allocation columns: shown when this run measured them; compared only when
        // the baseline did too.
        let base_allocs = base.and_then(|b| b.allocs);
        let alloc_delta = base_allocs.map(|b| Delta::classify(s.allocs, b));
        let (alloc_cur, alloc_base, alloc_col) = if alloc::ENABLED {
            let col = match (base_allocs, alloc_delta) {
                (Some(b), Some(Delta::Flat)) => format!("{:+.1}%", ratio_pct(s.allocs, b)),
                (Some(b), Some(delta)) => {
                    format!("{:+.1}% {}", ratio_pct(s.allocs, b), delta.label())
                }
                _ => String::new(),
            };
            (
                s.allocs.to_string(),
                base_allocs.map(|b| b.to_string()).unwrap_or_default(),
                col,
            )
        } else {
            ("-".to_string(), "-".to_string(), String::new())
        };

        // One scenario regresses if either its time or (when measured) its
        // allocations did; both classifications are known here, so it is pushed
        // once with no membership check.
        let time_regressed = ns_delta.is_some_and(Delta::regressed);
        let alloc_regressed = alloc::ENABLED && alloc_delta.is_some_and(Delta::regressed);
        if time_regressed || alloc_regressed {
            regressed.push(s.name);
        }

        println!(
            "{:<20} {:>11} {:>11} {:>8}   {:>10} {:>10} {:>8}",
            s.name, s.ns, base_ns_col, ns_flag, alloc_cur, alloc_base, alloc_col
        );
    }
    regressed
}

/// Percentage change of `current` over `base` (positive is a regression).
fn ratio_pct(current: u64, base: u64) -> f64 {
    (current as f64 / base.max(1) as f64 - 1.0) * 100.0
}

/// Whether `current` is worse than `base` by at least [`REGRESSION_RATIO`].
fn is_regression(current: u64, base: u64) -> bool {
    current as f64 >= base.max(1) as f64 * REGRESSION_RATIO
}

/// Write samples to `path` (creating its directory), sorted by name for a stable,
/// diff-friendly file. The allocation column is written only when this run
/// measured it, so a non-`perf-alloc` run never records zeros a later comparison
/// would misread.
fn write_samples(path: &str, samples: &[Sample]) -> Result<()> {
    ensure_parent(path)?;
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    let mut text =
        String::from("# bnknotes perf results: name, ns/op, allocs/op (lower is better)\n");
    for s in sorted {
        if alloc::ENABLED {
            let _ = writeln!(text, "{}\t{}\t{}", s.name, s.ns, s.allocs);
        } else {
            let _ = writeln!(text, "{}\t{}", s.name, s.ns);
        }
    }
    std::fs::write(path, text).map_err(|e| Error::msg(format!("write {path}: {e}")))
}

/// Load a baseline file into a name -> entry map, or `None` if it does not exist.
fn load_baseline(path: &str) -> Result<Option<HashMap<String, Base>>> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_baseline(&text).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::msg(format!("read {path}: {e}"))),
    }
}

/// Parse the `name<whitespace>ns[<whitespace>allocs]` baseline format, skipping
/// blank and `#` lines. The allocation column is optional so an older two-column
/// baseline still loads (its allocations simply read as unmeasured).
fn parse_baseline(text: &str) -> Result<HashMap<String, Base>> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(name), Some(ns)) = (it.next(), it.next()) else {
            return Err(Error::msg(format!("malformed perf line: {line:?}")));
        };
        let ns = ns
            .parse::<u64>()
            .map_err(|_| Error::msg(format!("malformed perf number: {ns:?}")))?;
        let allocs = match it.next() {
            Some(a) => Some(
                a.parse::<u64>()
                    .map_err(|_| Error::msg(format!("malformed alloc number: {a:?}")))?,
            ),
            None => None,
        };
        out.insert(name.to_string(), Base { ns, allocs });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tier 2: the lab.
// ---------------------------------------------------------------------------

/// One measured lab scenario over a corpus.
struct LabResult {
    name: &'static str,
    ns: u64,
    allocs: u64,
    /// Whether a throughput number is meaningful: true for the whole-document
    /// passes, false for the viewport-only paint (its cost is not the doc size).
    doc_wide: bool,
}

/// Run the lab: generate each selected corpus, drive whole-pipeline scenarios over
/// it, print a per-corpus table, and append the numbers to `perf/history.jsonl`.
pub fn run_lab(args: &[String]) -> Result<()> {
    let size_mib = parse_size(args)
        .unwrap_or(LAB_DEFAULT_MIB)
        .clamp(1, LAB_MAX_MIB);
    let target = size_mib * 1024 * 1024;
    let selected = select_corpora(args);

    let (fonts, wrap, images) = boot()?;
    eprintln!(
        "bnknotes perf-lab: {} MiB/corpus, {} corpus/corpora{}",
        size_mib,
        selected.len(),
        if alloc::ENABLED {
            ""
        } else {
            " (allocs unmeasured; build --features perf-alloc)"
        }
    );

    let commit = git_head();
    let ts = unix_seconds();
    let mut history: Vec<String> = Vec::new();

    for &name in selected.iter() {
        let text = corpus::generate(name, target)
            .ok_or_else(|| Error::msg(format!("unknown corpus: {name}")))?;
        let rope = Rope::from_str(&text);
        let bytes = text.len();
        let lines = rope.len_lines();

        // Warm the caches the real editor warms before it reaches steady state.
        let warm = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
        let frame = build_display_list(&fonts, &warm, &rope, &params(0, None));
        let mut glyphs = crate::gpu::GlyphCache::new();
        let _ = crate::gpu::build_frame(&fonts, &[&images], &frame, &mut glyphs, &|_| true);

        // Windowed layout + incremental index: the per-cursor-move and per-keystroke
        // costs that replace a full `layout` rebuild. A warm index, a viewport band,
        // and a one-character edit (at a char boundary near the middle) set them up.
        let index = LineIndex::build(&fonts, &rope, wrap, FONT_SIZE, &images);
        let band = 30.min(index.len());
        let edited = edited_middle(&text);
        let window_ctx = LayoutCtx::new(
            &fonts,
            &rope,
            wrap,
            FONT_SIZE,
            (0, 0),
            &images,
            ImageUi::default(),
        );
        let update_ctx = LayoutCtx::for_index(&fonts, &edited, wrap, FONT_SIZE, &images);

        let cpu_before = sys::cpu_time();

        let find_buf = TextBuffer::from_text(text.clone());
        let results = [
            lab("rope_build", LAB_REPS, true, || Rope::from_str(&text).len()),
            lab("layout", LAB_REPS, true, || {
                Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0))
                    .lines()
                    .len()
            }),
            lab("paint", LAB_REPS, false, || {
                build_display_list(&fonts, &warm, &rope, &params(0, None)).len()
            }),
            lab("find", LAB_REPS, true, || {
                let mut f = FindState::new();
                f.open_at(0, &find_buf);
                f.push_str("the", &find_buf);
                f.matches().len()
            }),
            lab("window", LAB_REPS, false, || {
                Layout::build_window(&window_ctx, MARGIN, MARGIN, &index, 0, band)
                    .lines()
                    .len()
            }),
            lab("index_upd", LAB_REPS, false, || {
                let mut idx = index.clone();
                idx.update(&update_ctx, &rope);
                idx.len()
            }),
        ];

        let cpu = sys::cpu_time().saturating_sub(cpu_before);
        let rss = sys::current_rss_bytes();
        let peak = sys::peak_rss_bytes();

        print_lab(
            name,
            bytes,
            lines,
            &results,
            cpu.as_nanos() as u64,
            rss,
            peak,
        );
        collect_history(
            &mut history,
            &commit,
            ts,
            name,
            bytes,
            &results,
            cpu.as_nanos() as u64,
            rss,
        );
    }

    append_history(&history)?;
    println!(
        "\nAppended {} rows to {HISTORY_PATH} (commit {commit}).",
        history.len()
    );
    Ok(())
}

/// The corpora to run: any named on the command line, else all of them.
fn select_corpora(args: &[String]) -> Vec<&'static str> {
    let named: Vec<&'static str> = corpus::NAMES
        .iter()
        .copied()
        .filter(|n| args.iter().any(|a| a == n))
        .collect();
    if named.is_empty() {
        corpus::NAMES.to_vec()
    } else {
        named
    }
}

/// Parse `--size <MiB>` or `--size=<MiB>` from the arguments.
fn parse_size(args: &[String]) -> Option<usize> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--size" {
            return it.next().and_then(|v| v.parse().ok());
        }
        if let Some(v) = a.strip_prefix("--size=") {
            return v.parse().ok();
        }
    }
    None
}

/// Measure one lab scenario: the best whole-document time over [`LAB_REPS`] rounds
/// (one call each), with allocations from a warmed, untimed run.
fn lab<T>(name: &'static str, reps: u32, doc_wide: bool, f: impl FnMut() -> T) -> LabResult {
    let (ns, allocs) = best_of(reps, 1, f);
    LabResult {
        name,
        ns,
        allocs,
        doc_wide,
    }
}

/// Throughput in megabytes per second for a whole-document pass.
fn mbps(bytes: usize, ns: u64) -> f64 {
    if ns == 0 {
        return 0.0;
    }
    bytes as f64 / (ns as f64 / 1e9) / 1e6
}

fn print_lab(
    name: &str,
    bytes: usize,
    lines: usize,
    results: &[LabResult],
    cpu_ns: u64,
    rss: u64,
    peak: u64,
) {
    println!(
        "\ncorpus {name}  ({}, {lines} lines)",
        human_bytes(bytes as u64)
    );
    println!(
        "  {:<12} {:>12} {:>14} {:>12}",
        "scenario", "time", "throughput", "allocs"
    );
    for r in results {
        let throughput = if r.doc_wide {
            format!("{:.0} MB/s", mbps(bytes, r.ns))
        } else {
            "-".to_string()
        };
        let allocs = if alloc::ENABLED {
            r.allocs.to_string()
        } else {
            "-".to_string()
        };
        println!(
            "  {:<12} {:>12} {:>14} {:>12}",
            r.name,
            human_time(r.ns),
            throughput,
            allocs
        );
    }
    println!(
        "  cpu {} user+sys   rss {} (peak {})",
        human_time(cpu_ns),
        human_bytes(rss),
        human_bytes(peak)
    );
}

/// Turn a run's numbers into JSONL rows for the history log.
#[allow(clippy::too_many_arguments)]
fn collect_history(
    out: &mut Vec<String>,
    commit: &str,
    ts: u64,
    corpus: &str,
    bytes: usize,
    results: &[LabResult],
    cpu_ns: u64,
    rss: u64,
) {
    for r in results {
        out.push(json_row(commit, ts, corpus, r.name, "time_ns", r.ns as f64));
        if alloc::ENABLED {
            out.push(json_row(
                commit,
                ts,
                corpus,
                r.name,
                "allocs",
                r.allocs as f64,
            ));
        }
        if r.doc_wide {
            out.push(json_row(
                commit,
                ts,
                corpus,
                r.name,
                "mbps",
                mbps(bytes, r.ns),
            ));
        }
    }
    out.push(json_row(
        commit,
        ts,
        corpus,
        "corpus",
        "cpu_ns",
        cpu_ns as f64,
    ));
    out.push(json_row(
        commit,
        ts,
        corpus,
        "corpus",
        "rss_bytes",
        rss as f64,
    ));
}

/// One JSON object per line. Values are machine identifiers or numbers, so the
/// writer emits them without escaping.
fn json_row(
    commit: &str,
    ts: u64,
    corpus: &str,
    scenario: &str,
    metric: &str,
    value: f64,
) -> String {
    format!(
        "{{\"commit\":\"{commit}\",\"ts\":{ts},\"corpus\":\"{corpus}\",\"scenario\":\"{scenario}\",\"metric\":\"{metric}\",\"value\":{value}}}"
    )
}

/// Append the run's rows to the history log, creating it (and its directory) if
/// needed. A no-op for an empty run.
fn append_history(rows: &[String]) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    ensure_parent(HISTORY_PATH)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(HISTORY_PATH)
        .map_err(|e| Error::msg(format!("open {HISTORY_PATH}: {e}")))?;
    for row in rows {
        writeln!(file, "{row}").map_err(|e| Error::msg(format!("write {HISTORY_PATH}: {e}")))?;
    }
    Ok(())
}

/// The current commit, short form, or `"unknown"` outside a git tree. Read-only.
fn git_head() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

/// The command-line argument immediately following `flag`, if `flag` is present
/// and has a following token. Folds the harness's flag-then-value scans.
fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).map(String::as_str)
}

/// Run `f`, returning its result paired with the wall-clock nanoseconds it took.
/// Folds the load bench's repeated `Instant::now()` / `elapsed` stage timing.
fn timed<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed().as_nanos() as u64)
}

/// Create `path`'s parent directory if it has one.
fn ensure_parent(path: &str) -> Result<()> {
    if let Some(dir) = Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::msg(format!("create {}: {e}", dir.display())))?;
    }
    Ok(())
}

/// Human-readable byte count.
fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

/// Human-readable duration from nanoseconds.
fn human_time(ns: u64) -> String {
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.1} us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.1} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    }
}

/// Clone `text` and insert one character near its middle, returning the edited
/// rope. Models the single-keystroke edit the incremental-index benches measure;
/// the insertion point is advanced to the next char boundary so a multibyte
/// corpus cannot panic `String::insert`.
fn edited_middle(text: &str) -> Rope {
    let mut t = text.to_string();
    let mut at = t.len() / 2;
    while at < t.len() && !t.is_char_boundary(at) {
        at += 1;
    }
    t.insert(at, 'x');
    Rope::from_str(&t)
}

/// A few hundred lines of representative markdown: headings, wrapping prose with
/// inline emphasis and code, nested lists, a blockquote, a fenced code block, and
/// a rule. Embedded so the baseline never drifts with an external file, and tall
/// enough that scrolling has somewhere to go.
fn fixture() -> String {
    const BLOCK: &str = r#"# Heading one

A paragraph with **bold**, *italic*, and `inline code` that runs long enough to
soft-wrap across the content width when laid out at the body size, so the wrap
path and run measuring are exercised on real prose, not one short line.

## Heading two

- first list item
- second item with `code` inside it
  - a nested item that is also long enough to wrap onto a second visual line under
    its parent, exercising the hanging-indent continuation geometry
- third item

> A blockquote that likewise wraps because it is comfortably longer than the
> available content width, so the quote bar is drawn down several wrapped lines.

```
fn rendered_as_code() {
    let x = 42; // inside a fenced block, not parsed as markdown
}
```

---

"#;
    BLOCK.repeat(13)
}

// ---------------------------------------------------------------------------
// Load bench: the real file-open pipeline, timed stage by stage.
// ---------------------------------------------------------------------------

/// Measure opening `path` the way the editor does, stage by stage: read the
/// bytes off disk, decode them ([`crate::file::decode`]: BOM strip, UTF-8
/// validation, newline normalization), build the rope, build the line-metrics
/// index (the O(document) cost the first frame pays), and lay out the opening
/// viewport. This is the whole synchronous open path minus the window system,
/// so a large-file stall can be pinned to a stage. Invoked as
/// `--load-bench <path>`.
pub fn run_load_bench(args: &[String]) -> Result<()> {
    let path =
        arg_after(args, "--load-bench").ok_or_else(|| Error::msg("usage: --load-bench <path>"))?;

    let (fonts, wrap, images) = boot()?;

    eprintln!("bnknotes load-bench: {path}\n");
    println!("  {:<14}{:>10}{:>17}", "stage", "time", "throughput");

    let (bytes, read_ns) = timed(|| std::fs::read(path));
    let bytes = bytes.map_err(|e| Error::msg(format!("read {path}: {e}")))?;
    let n = bytes.len();
    report_stage("read (disk)", read_ns, n);

    let (decoded, decode_ns) = timed(|| crate::file::decode(&bytes));
    let (text, _format) = decoded?;
    report_stage("decode", decode_ns, n);

    let (rope, rope_ns) = timed(|| Rope::from_str(&text));
    report_stage("rope build", rope_ns, n);

    // The image pre-scan the editor runs before building the index: one walk over
    // every line looking for a standalone `![](...)`. Cheap per line (no glyph
    // work), but O(document), so it counts toward a large-file open.
    let lines = rope.len_lines();
    let (image_lines, scan_ns) = timed(|| {
        let mut image_lines = 0usize;
        for (ls, le, _) in rope.lines() {
            if markdown::standalone_image(&rope.slice(ls..le).as_cow()).is_some() {
                image_lines += 1;
            }
        }
        image_lines
    });
    report_stage("scan images", scan_ns, n);

    let (_index_seq, index_seq_ns) =
        timed(|| LineIndex::build(&fonts, &rope, wrap, FONT_SIZE, &images));
    report_stage("index (1 core)", index_seq_ns, n);

    // An optional positional arg after the path overrides the worker count, so the
    // memory/speed knee can be swept.
    let threads = arg_after(args, path)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        });
    let (index, index_ns) =
        timed(|| LineIndex::build_parallel(&fonts, &rope, wrap, FONT_SIZE, &images, threads));
    report_stage(&format!("index ({threads} cores)"), index_ns, n);

    let band = 60.min(index.len());
    let window_ctx = LayoutCtx::new(
        &fonts,
        &rope,
        wrap,
        FONT_SIZE,
        (0, 0),
        &images,
        ImageUi::default(),
    );
    let (window, window_ns) =
        timed(|| Layout::build_window(&window_ctx, MARGIN, MARGIN, &index, 0, band));
    report_stage("first window", window_ns, n);

    let total = read_ns + decode_ns + rope_ns + scan_ns + index_ns + window_ns;
    println!("  {:<14}{:>10}", "TOTAL", human_time(total));
    println!(
        "\n  {} on disk, {lines} lines ({image_lines} image), {} visual rows in the first window, RSS {}",
        human_bytes(n as u64),
        window.lines().len(),
        human_bytes(sys::current_rss_bytes()),
    );
    Ok(())
}

/// Print one load-bench stage: its wall-clock time and throughput over `bytes`.
fn report_stage(label: &str, ns: u64, bytes: usize) {
    let secs = ns as f64 / 1e9;
    let mbps = if secs > 0.0 {
        bytes as f64 / secs / 1e6
    } else {
        0.0
    };
    println!("  {label:<14}{:>10}{:>12.0} MB/s", human_time(ns), mbps);
}

// ---------------------------------------------------------------------------
// Memory report: the retained footprint of an open document, by component.
// ---------------------------------------------------------------------------

/// Open `path` and report the memory it retains, broken down by the structures
/// that scale with the file, so caps can be set against real numbers. Rope and
/// index heap are computed exactly; RSS shows what the process actually holds,
/// including the transient decode peak. Invoked as `--mem-report <path>`.
pub fn run_mem_report(args: &[String]) -> Result<()> {
    let path =
        arg_after(args, "--mem-report").ok_or_else(|| Error::msg("usage: --mem-report <path>"))?;

    let (fonts, wrap, images) = boot()?;

    // Build the document as an open does, then free the transient decode buffers
    // so the current RSS reflects the steady state.
    let bytes = std::fs::read(path).map_err(|e| Error::msg(format!("read {path}: {e}")))?;
    let file_bytes = bytes.len();
    let (text, _format) = crate::file::decode(&bytes)?;
    drop(bytes);
    let rope = Rope::from_str(&text);
    drop(text);
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let index = LineIndex::build_parallel(&fonts, &rope, wrap, FONT_SIZE, &images, threads);

    let peak = sys::peak_rss_bytes();
    let rope_heap = rope.heap_bytes();
    let index_heap = index.heap_bytes();
    let lines = rope.len_lines();
    let retained = rope_heap + index_heap;
    // The one uncapped, file-variable structure: a query matching every byte would
    // hold that many (start, end) pairs.
    let find_worst = rope.len() * std::mem::size_of::<(usize, usize)>();

    eprintln!("bnknotes mem-report: {path}\n");
    println!("  {:<16}{:>12}   note", "component", "heap");
    report_mem("rope", rope_heap as u64, "document text + tree");
    report_mem(
        "line index",
        index_heap as u64,
        &format!("16 B x {lines} lines"),
    );
    report_mem(
        "= retained",
        retained as u64,
        "steady state, scales with the file",
    );
    println!();
    report_mem(
        "peak on open",
        peak,
        "incl. ~2x-decode + per-worker font sets, freed after",
    );
    println!(
        "\n  {} on disk, {lines} lines\n  find worst case: every byte a match = {} of (start,end) pairs, UNBOUNDED",
        human_bytes(file_bytes as u64),
        human_bytes(find_worst as u64),
    );
    Ok(())
}

/// Print one memory-report line: a component's heap and a note.
fn report_mem(label: &str, bytes: u64, note: &str) {
    println!("  {label:<16}{:>12}   {note}", human_bytes(bytes));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_baseline_reads_pairs_and_skips_comments() {
        let text = "# a comment\n\nlayout_build\t1234\ngpu_batch_frame   5678\n";
        let map = parse_baseline(text).expect("valid baseline");
        assert_eq!(map.get("layout_build").map(|b| b.ns), Some(1234));
        assert_eq!(map.get("gpu_batch_frame").map(|b| b.ns), Some(5678));
        assert_eq!(map.get("layout_build").and_then(|b| b.allocs), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn parse_baseline_reads_the_optional_alloc_column() {
        let map = parse_baseline("layout_build\t1234\t42\n").expect("valid");
        let base = map.get("layout_build").expect("present");
        assert_eq!(base.ns, 1234);
        assert_eq!(base.allocs, Some(42));
    }

    #[test]
    fn parse_baseline_rejects_a_bad_number() {
        assert!(parse_baseline("layout_build\tnot_a_number\n").is_err());
        assert!(parse_baseline("layout_build\t10\tnope\n").is_err());
    }

    #[test]
    fn regression_uses_the_ratio_threshold() {
        // 10% threshold: 1099/1000 is fine, 1100/1000 trips it.
        assert!(!is_regression(1099, 1000));
        assert!(is_regression(1100, 1000));
        assert!(!is_regression(900, 1000), "faster is never a regression");
    }

    #[test]
    fn delta_classifies_around_the_ratio() {
        // At/over the ratio worse is a regression; at/over it better is an
        // improvement; the band between is flat. Same threshold as the gate.
        assert!(Delta::classify(1100, 1000).regressed());
        assert_eq!(Delta::classify(1100, 1000).label(), "REGRESSION");
        assert!(!Delta::classify(1050, 1000).regressed());
        assert_eq!(Delta::classify(1050, 1000).label(), "");
        assert_eq!(Delta::classify(900, 1000).label(), "improved");
        assert!(!Delta::classify(900, 1000).regressed());
    }

    #[test]
    fn arg_after_reads_the_following_token() {
        let args: Vec<String> = ["--load-bench", "notes.md", "4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(arg_after(&args, "--load-bench"), Some("notes.md"));
        // The token after the path is the next positional (the worker count).
        assert_eq!(arg_after(&args, "notes.md"), Some("4"));
        // Absent flag, or a flag with nothing after it, is None.
        assert_eq!(arg_after(&args, "--missing"), None);
        assert_eq!(arg_after(&args, "4"), None);
    }

    #[test]
    fn parse_size_reads_both_spellings() {
        assert_eq!(parse_size(&["--size".into(), "16".into()]), Some(16));
        assert_eq!(parse_size(&["--size=32".into()]), Some(32));
        assert_eq!(parse_size(&["--perf-lab".into()]), None);
    }

    #[test]
    fn select_corpora_defaults_to_all_then_filters() {
        assert_eq!(
            select_corpora(&["--perf-lab".into()]).len(),
            corpus::NAMES.len()
        );
        let one = select_corpora(&["--perf-lab".into(), "emoji".into()]);
        assert_eq!(one, vec!["emoji"]);
    }

    #[test]
    fn throughput_is_bytes_over_time() {
        // 1 MB in 1 ms is 1000 MB/s.
        assert!((mbps(1_000_000, 1_000_000) - 1000.0).abs() < 1e-6);
        assert_eq!(mbps(1000, 0), 0.0);
    }

    #[test]
    fn json_row_is_well_formed() {
        let row = json_row("abc123", 42, "emoji", "layout", "time_ns", 1500.0);
        assert!(row.starts_with('{') && row.ends_with('}'));
        assert!(row.contains("\"commit\":\"abc123\""));
        assert!(row.contains("\"corpus\":\"emoji\""));
        assert!(row.contains("\"value\":1500"));
    }

    #[test]
    fn human_helpers_are_readable() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KiB");
        assert_eq!(human_time(500), "500 ns");
        assert_eq!(human_time(1_500_000), "1.5 ms");
    }

    #[test]
    fn fixture_is_tall_enough_to_scroll() {
        // Many lines, so the layout and frame-batch scenarios have real work.
        assert!(fixture().lines().count() > 200);
    }
}
