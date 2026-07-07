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
use crate::layout::{ImageUi, Layout};
use crate::markdown;
use crate::paint::{build_display_list, PaintParams};
use crate::rope::Rope;
use crate::sys;
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

/// Measure every scenario against the embedded fixture and a warm font set.
fn measure() -> Result<Vec<Sample>> {
    let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
    let text = fixture();
    let rope = Rope::from_str(&text);
    let wrap = WIDTH as i32 - 2 * MARGIN;

    // Build once up front: this warms the per-character advance cache, so the
    // timed `layout_build` below measures the steady re-layout cost (the per-frame
    // cost while editing), not the one-time cold measure of every glyph.
    let layout = Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0));
    // An empty store satisfies build_frame's signature; the fixture has no images.
    let images = ImageStore::new(None);

    let mut out = Vec::new();

    // Full re-layout of the document. Guards the advance cache above all: if glyph
    // measuring went uncached again, this scenario balloons roughly 75x.
    out.push(sample("layout_build", 50, || {
        Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0))
            .lines()
            .len()
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
    let image_layout = Layout::build_with_images(
        &fonts,
        &image_rope,
        MARGIN,
        MARGIN,
        wrap,
        FONT_SIZE,
        (0, 0), // cursor on the heading: the image renders as a picture
        &image_store,
        ImageUi::default(),
    );

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
        crate::freetype::resample_argb(&native, 400, 300, 200, 150).len()
    }));

    // Re-laying-out a document containing an image at a fresh width, as a splitter
    // or window resize does on every drag event (the layout build has no cache).
    out.push(sample("layout_reflow_image", 200, || {
        Layout::build_with_images(
            &fonts,
            &image_rope,
            MARGIN,
            MARGIN,
            wrap - 40,
            FONT_SIZE,
            (0, 0),
            &image_store,
            ImageUi::default(),
        )
        .text_height()
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

/// Time `f` and return the best (minimum) ns per call plus the allocations one
/// call makes. The minimum is the least noisy time estimator for a
/// micro-benchmark: it is the run least disturbed by the scheduler. The allocation
/// count is taken from a single warmed run, untimed, so counting never skews the
/// time. `black_box` stops the optimizer eliding the work as dead.
fn bench<T>(iters: u64, mut f: impl FnMut() -> T) -> (u64, u64) {
    const ROUNDS: u32 = 5;
    // One warmup before counting, so a first-call lazy init is not charged as
    // steady-state allocation.
    black_box(f());
    alloc::reset();
    black_box(f());
    let allocs = alloc::snapshot().allocs;

    for _ in 0..iters.min(16) {
        black_box(f());
    }
    let mut best = u64::MAX;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        best = best.min((start.elapsed().as_nanos() as u64) / iters.max(1));
    }
    (best, allocs)
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
        let (base_ns_col, ns_flag) = match base {
            Some(b) => {
                let ratio = s.ns as f64 / b.ns.max(1) as f64;
                let flag = if is_regression(s.ns, b.ns) {
                    regressed.push(s.name);
                    "REGRESSION"
                } else if (b.ns as f64 / s.ns.max(1) as f64) >= REGRESSION_RATIO {
                    "improved"
                } else {
                    ""
                };
                (
                    format!("{}", b.ns),
                    format!("{:+.1}% {}", (ratio - 1.0) * 100.0, flag),
                )
            }
            None => ("(new)".to_string(), String::new()),
        };

        // Allocation columns: shown when this run measured them; compared only when
        // the baseline did too.
        let (alloc_cur, alloc_base, alloc_delta) = if alloc::ENABLED {
            let base_allocs = base.and_then(|b| b.allocs);
            let delta = match base_allocs {
                Some(b) => {
                    if is_regression(s.allocs, b) {
                        if !regressed.contains(&s.name) {
                            regressed.push(s.name);
                        }
                        format!("{:+.1}% REGRESSION", ratio_pct(s.allocs, b))
                    } else if b as f64 / s.allocs.max(1) as f64 >= REGRESSION_RATIO {
                        format!("{:+.1}% improved", ratio_pct(s.allocs, b))
                    } else {
                        format!("{:+.1}%", ratio_pct(s.allocs, b))
                    }
                }
                None => String::new(),
            };
            (
                format!("{}", s.allocs),
                base_allocs.map(|b| b.to_string()).unwrap_or_default(),
                delta,
            )
        } else {
            ("-".to_string(), "-".to_string(), String::new())
        };

        println!(
            "{:<20} {:>11} {:>11} {:>8}   {:>10} {:>10} {:>8}",
            s.name, s.ns, base_ns_col, ns_flag, alloc_cur, alloc_base, alloc_delta
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

    let fonts = Fonts::new(&markdown::all_sizes(FONT_SIZE))?;
    let wrap = WIDTH as i32 - 2 * MARGIN;
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
        let images = ImageStore::new(None);
        let frame = build_display_list(&fonts, &warm, &rope, &params(0, None));
        let mut glyphs = crate::gpu::GlyphCache::new();
        let _ = crate::gpu::build_frame(&fonts, &[&images], &frame, &mut glyphs, &|_| true);

        let cpu_before = sys::cpu_time();

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
                f.open_at(0, &text);
                f.push_str("the", &text);
                f.matches().len()
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

/// Measure one lab scenario: allocations from a warmed, untimed run, then the
/// best time over [`LAB_REPS`] rounds.
fn lab<T>(name: &'static str, reps: u32, doc_wide: bool, mut f: impl FnMut() -> T) -> LabResult {
    black_box(f());
    alloc::reset();
    black_box(f());
    let allocs = alloc::snapshot().allocs;
    let mut best = u64::MAX;
    for _ in 0..reps.max(1) {
        let start = Instant::now();
        black_box(f());
        best = best.min(start.elapsed().as_nanos() as u64);
    }
    LabResult {
        name,
        ns: best,
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
