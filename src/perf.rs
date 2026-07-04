//! A small, deterministic performance harness for the editor's hot paths: full
//! re-layout, pointer hit-testing, painting a frame, scrolling, highlighting, and
//! glyph rasterization. It runs CPU-only (no Wayland, no display), times each
//! scenario, and writes the results to a file so a later run can diff against a
//! committed baseline and flag regressions.
//!
//! Run `bnknotes --perf` to measure and compare against `perf/baseline.txt`
//! (writing `perf/latest.txt`); the process exits nonzero if any scenario
//! regressed past [`REGRESSION_RATIO`]. Run `bnknotes --perf --save` to accept
//! the current numbers as the new baseline. Always build `--release`: a debug
//! build's numbers are meaningless and not comparable to a release baseline.
//!
//! The fixture is embedded (not read from a file), so the baseline is stable and
//! never drifts because some sample document changed.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use crate::app::{build_display_list, paint_document, PaintParams, FONT_SIZE, MARGIN};
use crate::display;
use crate::error::{Error, Result};
use crate::freetype::Fonts;
use crate::image::{Image, ImageStore};
use crate::layout::{ImageUi, Layout};
use crate::markdown;
use crate::render::Canvas;
use crate::rope::Rope;

/// Where the committed baseline lives and where each run records its numbers.
const BASELINE_PATH: &str = "perf/baseline.txt";
const LATEST_PATH: &str = "perf/latest.txt";

/// A scenario counts as a regression when it is at least this much slower than the
/// baseline. Wide enough to absorb run-to-run timing noise, tight enough to catch
/// the order-of-magnitude regressions this harness exists to prevent.
const REGRESSION_RATIO: f64 = 1.10;

/// The surface the harness paints into, matching the app's initial window.
const WIDTH: usize = 800;
const HEIGHT: usize = 600;

/// One measured scenario and its best time, in nanoseconds per operation.
struct Sample {
    name: &'static str,
    ns: u64,
}

/// Run the harness: measure, write `latest`, then compare to (or establish) the
/// baseline. Returns an error when a scenario regressed and `--save` was not
/// given, so the process exit code signals the regression to a script or CI.
pub fn run(args: &[String]) -> Result<()> {
    let save = args.iter().any(|a| a == "--save");
    eprintln!("bnknotes perf: measuring (build --release for meaningful numbers)...");
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
    let content_h = layout.text_height() + 2 * MARGIN;
    let mut buf = vec![0u8; WIDTH * HEIGHT * 4];
    // The fixture has no images; an empty store satisfies the painter's signature.
    let images = ImageStore::new(None);

    let mut out = Vec::new();

    // Full re-layout of the document. Guards the advance cache above all: if glyph
    // measuring went uncached again, this scenario balloons roughly 75x.
    out.push(Sample {
        name: "layout_build",
        ns: bench(50, || {
            Layout::build(&fonts, &rope, MARGIN, MARGIN, wrap, FONT_SIZE, (0, 0))
                .lines()
                .len()
        }),
    });

    // Hit-test the pointer against the layout: the per-motion cost of a drag.
    out.push(Sample {
        name: "offset_at",
        ns: bench(2000, || {
            layout.offset_at(&fonts, &rope, (WIDTH / 2) as f32, (HEIGHT / 2) as f32)
        }),
    });

    // Paint one frame at the top of the document: steady-state rendering.
    out.push(Sample {
        name: "paint_frame",
        ns: bench(100, || {
            let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
            paint_document(
                &mut canvas,
                &fonts,
                &layout,
                &rope,
                &images,
                &params(0, None),
            );
            buf[0]
        }),
    });

    // Scroll sweep: paint at successive offsets through the whole document, one
    // line height apart, so a per-line pathology in paint would show up here.
    let line_h = fonts.metrics(FONT_SIZE).line_height.max(1);
    let max_scroll = (content_h - HEIGHT as i32).max(0);
    let offsets: Vec<i32> = (0..=max_scroll).step_by(line_h as usize).collect();
    let mut k = 0usize;
    out.push(Sample {
        name: "scroll_sweep",
        ns: bench(200, || {
            let sy = offsets[k % offsets.len()];
            k += 1;
            let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
            paint_document(
                &mut canvas,
                &fonts,
                &layout,
                &rope,
                &images,
                &params(sy, None),
            );
            buf[0]
        }),
    });

    // Highlight sweep: a full drag-select frame. Each step hit-tests the pointer
    // and then paints a selection that grows line by line across the document, so
    // both the per-motion hit-test and the selection-rect geometry are measured.
    let n_lines = rope.len_lines().max(1);
    let mut k = 0usize;
    out.push(Sample {
        name: "highlight_sweep",
        ns: bench(200, || {
            let _ = layout.offset_at(&fonts, &rope, (WIDTH / 2) as f32, (HEIGHT / 2) as f32);
            let end = rope.line_to_byte(k % n_lines);
            k += 1;
            let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
            paint_document(
                &mut canvas,
                &fonts,
                &layout,
                &rope,
                &images,
                &params(0, Some((0, end))),
            );
            buf[0]
        }),
    });

    // Paint a frame containing a decoded image, to guard the per-frame image blit.
    // Bilinear resampling is cached at display size, so the steady cost is a plain
    // blit of the cached bitmap; the warm-up paint below fills that cache first. A
    // regression here would mean the resample leaked back into the per-frame path.
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
    {
        let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
        paint_document(
            &mut canvas,
            &fonts,
            &image_layout,
            &image_rope,
            &image_store,
            &params(0, None),
        );
    }
    out.push(Sample {
        name: "image_paint",
        ns: bench(100, || {
            let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
            paint_document(
                &mut canvas,
                &fonts,
                &image_layout,
                &image_rope,
                &image_store,
                &params(0, None),
            );
            buf[0]
        }),
    });

    // Raw glyph rasterization of printable ASCII (the uncached FreeType path), so a
    // regression in the rasterizer itself is visible independent of the cache.
    let face = fonts.face(FONT_SIZE, crate::freetype::FontStyle::Regular);
    out.push(Sample {
        name: "glyph_raster_ascii",
        ns: bench(20, || {
            let mut sum = 0usize;
            for c in 0x20u8..0x7f {
                sum += face.rasterize(c as char).coverage.len();
            }
            sum
        }),
    });

    // The incremental damage path: an idle caret blink. Only the caret toggles
    // between frames, so the diff against the on-screen frame yields a caret-sized
    // region and the repaint touches a sliver of the buffer, not the whole surface.
    // This is what the display list and damage tracking buy, so it should be a
    // fraction of `paint_frame` above; a regression here means a blink frame stopped
    // being incremental (e.g. the diff started reporting the whole surface, or the
    // build itself got expensive).
    let caret_on = build_display_list(&fonts, &layout, &rope, &params(0, None));
    let mut caret_off = params(0, None);
    caret_off.caret_visible = false;
    out.push(Sample {
        name: "damage_blink",
        ns: bench(2000, || {
            let frame = build_display_list(&fonts, &layout, &rope, &caret_off);
            let dmg = display::damage(&caret_on, &frame, WIDTH as i32, HEIGHT as i32);
            let mut canvas = Canvas::new(&mut buf, WIDTH, HEIGHT);
            for clip in &dmg {
                display::execute(&mut canvas, &fonts, &[&images], &frame, Some(*clip));
            }
            dmg.len()
        }),
    });

    // The GPU path's CPU-side frame cost: batch the display list into vertices
    // (glyph lookups against warm atlases, quad emission, batch splitting).
    // This runs even without a GPU, so the harness can hold it; the GPU-side
    // frame time is only observable via BNKNOTES_STATS on a live run.
    let mut glyph_cache = crate::gpu::GlyphCache::new();
    // Warm the atlases so the steady state (no uploads) is what is measured.
    let _ = crate::gpu::build_frame(&fonts, &[&images], &caret_on, &mut glyph_cache, &|_| true);
    out.push(Sample {
        name: "gpu_batch_frame",
        ns: bench(500, || {
            crate::gpu::build_frame(&fonts, &[&images], &caret_on, &mut glyph_cache, &|_| true)
                .vertices
                .len()
        }),
    });

    // Image resampling: the bilinear downscale the software path runs whenever an
    // image's display size changes (every drag event of a resize). The GPU path
    // avoids it entirely (native texture, hardware scaling); this guards the CPU
    // resampler and the software fallback's resize cost.
    let native: Vec<u32> = vec![0xff_30_60_90; 400 * 300];
    out.push(Sample {
        name: "image_resample",
        ns: bench(200, || {
            crate::render::resample_argb(&native, 400, 300, 200, 150).len()
        }),
    });

    // Re-laying-out a document containing an image at a fresh width, as a splitter
    // or window resize does on every drag event (the layout build has no cache).
    out.push(Sample {
        name: "layout_reflow_image",
        ns: bench(200, || {
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
        }),
    });

    // The GPU frame build for an image the renderer does not yet hold: it uploads
    // the native texture (`known_images` = false forces it each call). After the
    // resize fix this is paid once per image, not once per drag event, because the
    // key is the native size rather than the changing display size.
    let image_list = build_display_list(&fonts, &image_layout, &image_rope, &params(0, None));
    out.push(Sample {
        name: "image_upload_gpu",
        ns: bench(200, || {
            crate::gpu::build_frame(
                &fonts,
                &[&image_store],
                &image_list,
                &mut glyph_cache,
                &|_| false,
            )
            .image_uploads
            .len()
        }),
    });

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

/// Time `f` and return the best (minimum) ns per call over a few rounds. The
/// minimum is the least noisy estimator for a micro-benchmark: it is the run least
/// disturbed by the scheduler and other processes. `black_box` stops the optimizer
/// eliding the work as dead.
fn bench<T>(iters: u64, mut f: impl FnMut() -> T) -> u64 {
    const ROUNDS: u32 = 5;
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
    best
}

/// Print a current-vs-baseline table and return the names that regressed. A
/// metric absent from the baseline is shown as new; one present is flagged
/// `REGRESSION` past the ratio and `improved` below its reciprocal.
fn print_table(samples: &[Sample], baseline: &HashMap<String, u64>) -> Vec<&'static str> {
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    println!(
        "\n{:<20} {:>12} {:>12} {:>10}",
        "metric", "current", "baseline", "delta"
    );
    let mut regressed = Vec::new();
    for s in sorted {
        match baseline.get(s.name) {
            Some(&base) => {
                let ratio = s.ns as f64 / base.max(1) as f64;
                let flag = if is_regression(s.ns, base) {
                    regressed.push(s.name);
                    "  REGRESSION"
                } else if (base as f64 / s.ns.max(1) as f64) >= REGRESSION_RATIO {
                    "  improved"
                } else {
                    ""
                };
                println!(
                    "{:<20} {:>10}ns {:>10}ns {:>+9.1}%{}",
                    s.name,
                    s.ns,
                    base,
                    (ratio - 1.0) * 100.0,
                    flag
                );
            }
            None => println!("{:<20} {:>10}ns {:>12} {:>10}", s.name, s.ns, "(new)", ""),
        }
    }
    regressed
}

/// Whether `current` is slower than `base` by at least [`REGRESSION_RATIO`].
fn is_regression(current: u64, base: u64) -> bool {
    current as f64 >= base.max(1) as f64 * REGRESSION_RATIO
}

/// Write samples to `path` (creating its directory), sorted by name for a stable,
/// diff-friendly file.
fn write_samples(path: &str, samples: &[Sample]) -> Result<()> {
    if let Some(dir) = Path::new(path)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir)
            .map_err(|e| Error::msg(format!("create {}: {e}", dir.display())))?;
    }
    let mut sorted: Vec<&Sample> = samples.iter().collect();
    sorted.sort_by_key(|s| s.name);
    let mut text = String::from("# bnknotes perf results, ns per operation (lower is better)\n");
    for s in sorted {
        let _ = writeln!(text, "{}\t{}", s.name, s.ns);
    }
    std::fs::write(path, text).map_err(|e| Error::msg(format!("write {path}: {e}")))
}

/// Load a baseline file into a name -> ns map, or `None` if it does not exist.
fn load_baseline(path: &str) -> Result<Option<HashMap<String, u64>>> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_baseline(&text).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::msg(format!("read {path}: {e}"))),
    }
}

/// Parse the `name<whitespace>ns` baseline format, skipping blank and `#` lines.
fn parse_baseline(text: &str) -> Result<HashMap<String, u64>> {
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
        out.insert(name.to_string(), ns);
    }
    Ok(out)
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
        let text = "# a comment\n\nlayout_build\t1234\npaint_frame   5678\n";
        let map = parse_baseline(text).expect("valid baseline");
        assert_eq!(map.get("layout_build"), Some(&1234));
        assert_eq!(map.get("paint_frame"), Some(&5678));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn parse_baseline_rejects_a_bad_number() {
        assert!(parse_baseline("layout_build\tnot_a_number\n").is_err());
    }

    #[test]
    fn regression_uses_the_ratio_threshold() {
        // 10% threshold: 1099/1000 is fine, 1100/1000 trips it.
        assert!(!is_regression(1099, 1000));
        assert!(is_regression(1100, 1000));
        assert!(!is_regression(900, 1000), "faster is never a regression");
    }

    #[test]
    fn fixture_is_tall_enough_to_scroll() {
        // Many lines, so the scroll sweep and paint have real work.
        assert!(fixture().lines().count() > 200);
    }
}
