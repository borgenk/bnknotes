//! `--gpu-parity`: render fixture display lists on both backends and compare
//! pixels. The CPU path is the reference; the GPU path must reproduce it
//! within a small blend-rounding tolerance (the software blitter blends in
//! integer arithmetic, the GPU in float, so ±2 per channel is the expected
//! noise floor and anything past it is a real divergence).
//!
//! Runs offline: no compositor, no window. It needs a Vulkan device (any),
//! so it is a local `make gpu-parity` target rather than part of `make test`.

use crate::display::{self, DisplayList, DrawCmd};
use crate::error::{Error, Result};
use crate::freetype::{FaceKey, FontStyle, Fonts};
use crate::gpu;
use crate::image::ImageStore;
use crate::layout::Rect;
use crate::render::{Canvas, RoundedCorners};
use crate::vulkan;

const W: u32 = 800;
const H: u32 = 600;
/// Per-channel tolerance: integer vs float blend rounding.
const TOLERANCE: u8 = 2;
const BACKGROUND: u32 = 0x001c_2127;

pub fn run() -> Result<()> {
    let fonts = Fonts::new(&[14, 28])?;
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/testdata");
    let mut images = ImageStore::new(Some(dir));
    images.ensure("tiny.png");
    images.ensure("tiny.jpg");

    let mut gpu = vulkan::Gpu::new(None)?;
    println!("gpu-parity: device {}", gpu.name());
    let modifiers = gpu.own_modifiers();
    let mut img = gpu.create_image(W, H, &modifiers)?;
    let mut cache = gpu::GlyphCache::new();

    let mut failures = 0;
    for (name, list) in fixtures(&images) {
        // CPU reference.
        let mut cpu = vec![0u8; (W * H * 4) as usize];
        {
            let mut canvas = Canvas::new(&mut cpu, W as usize, H as usize);
            canvas.fill_solid(0, 0, W as i32, H as i32, BACKGROUND);
            display::execute(&mut canvas, &fonts, &[&images], &list, None);
        }

        // GPU rendition of the same list.
        let known = |key: &gpu::ImageKey| gpu.has_image(key);
        let frame = gpu::build_frame(&fonts, &[&images], &list, &mut cache, &known);
        gpu.render_list(&mut img, &frame, color_f32(BACKGROUND))?;
        let gpu_pixels = gpu.read_pixels(&mut img)?;

        match compare(&cpu, &gpu_pixels) {
            Ok(max) => println!("gpu-parity: {name}: ok (max channel diff {max})"),
            Err(report) => {
                failures += 1;
                println!("gpu-parity: {name}: FAIL {report}");
            }
        }
    }

    gpu.wait_idle();
    gpu.destroy_image(img);
    if failures > 0 {
        return Err(Error::msg(format!("{failures} fixture(s) diverged")));
    }
    println!("gpu-parity: all fixtures within ±{TOLERANCE}");
    Ok(())
}

/// Compare B,G,R channels (byte 3 is X on the CPU, alpha on the GPU; neither
/// is displayed). Returns the max diff, or a report of the worst offender.
fn compare(cpu: &[u8], gpu: &[u8]) -> std::result::Result<u8, String> {
    if cpu.len() != gpu.len() {
        return Err(format!("length mismatch {} vs {}", cpu.len(), gpu.len()));
    }
    let mut max = 0u8;
    let mut beyond = 0usize;
    let mut first: Option<(usize, usize)> = None;
    for (i, (c, g)) in cpu.chunks_exact(4).zip(gpu.chunks_exact(4)).enumerate() {
        for ch in 0..3 {
            let d = c[ch].abs_diff(g[ch]);
            if d > max {
                max = d;
            }
            if d > TOLERANCE {
                beyond += 1;
                if first.is_none() {
                    first = Some((i % W as usize, i / W as usize));
                }
            }
        }
    }
    match first {
        None => Ok(max),
        Some((x, y)) => Err(format!(
            "{beyond} channel(s) beyond ±{TOLERANCE}, max {max}, first at ({x}, {y})"
        )),
    }
}

/// The fixture display lists: each exercises one family of primitives, plus a
/// combined "everything" frame close to a real editor screen.
fn fixtures(images: &ImageStore) -> Vec<(&'static str, DisplayList)> {
    let body = FaceKey::Prose {
        size: 14,
        style: FontStyle::Regular,
    };
    let bold = FaceKey::Prose {
        size: 14,
        style: FontStyle::Bold,
    };
    let italic = FaceKey::Prose {
        size: 14,
        style: FontStyle::Italic,
    };
    let heading = FaceKey::Prose {
        size: 28,
        style: FontStyle::Bold,
    };
    let code = FaceKey::Code { size: 14 };
    let rect = |x, y, w, h| Rect { x, y, w, h };
    let text = |x: i32, baseline: i32, face: FaceKey, color: u32, s: &str| DrawCmd::Text {
        bounds: rect(x - 4, baseline - 34, 700, 44),
        x,
        baseline,
        face,
        color,
        text: s.to_string(),
    };

    let fills = vec![
        DrawCmd::Fill {
            rect: rect(10, 10, 200, 100),
            color: 0x0041_5776,
        },
        DrawCmd::Fill {
            rect: rect(150, 60, 200, 100), // overlaps the first
            color: 0x005e_5128,
        },
        DrawCmd::Fill {
            rect: rect(-20, 550, 900, 100), // clips at the edges
            color: 0x00f3_8ba8,
        },
        DrawCmd::Fill {
            rect: rect(400, 200, 2, 18), // a caret
            color: 0x00ff_ffff,
        },
    ];

    let rounds = vec![
        DrawCmd::RoundRect {
            rect: rect(20, 20, 160, 90),
            color: 0x0028_2c34,
            radius: 8,
            corners: RoundedCorners::Both,
        },
        DrawCmd::RoundRect {
            rect: rect(220, 20, 160, 90),
            color: 0x0031_3244,
            radius: 12,
            corners: RoundedCorners::Top,
        },
        DrawCmd::RoundRect {
            rect: rect(420, 20, 160, 90),
            color: 0x005e_5128,
            radius: 12,
            corners: RoundedCorners::Bottom,
        },
        DrawCmd::RoundRect {
            rect: rect(620, 20, 60, 8), // thin: radius clamps
            color: 0x00f3_8ba8,
            radius: 10,
            corners: RoundedCorners::Both,
        },
    ];

    let texts = vec![
        text(40, 60, body, 0x00da_dada, "The quick brown fox 0123456789"),
        text(40, 100, bold, 0x00da_dada, "bold text over the background"),
        text(
            40,
            140,
            italic,
            0x005c_9cf6,
            "italic and colored, with clip",
        ),
        text(40, 200, heading, 0x002e_80f2, "A heading at 28px"),
        text(40, 240, code, 0x00f3_8ba8, "let x = code_font(42);"),
        text(40, 290, body, 0x00da_dada, "unicode: åøæ é → ∑ 漢字"),
        text(40, 340, body, 0x00da_dada, "emoji: 😀 🚀 mixed with text"),
        // Overlapping a fill, so AA blends against a non-background color.
        DrawCmd::Fill {
            rect: rect(30, 360, 300, 40),
            color: 0x0041_5776,
        },
        text(40, 390, body, 0x00da_dada, "text over a selection fill"),
    ];

    let mut image_list = vec![DrawCmd::Fill {
        rect: rect(0, 0, W as i32, H as i32),
        color: BACKGROUND,
    }];
    // Skip image fixtures gracefully if decoding failed (no store entry).
    if images.dimensions("tiny.png").is_some() {
        image_list.push(DrawCmd::Image {
            rect: rect(50, 50, 64, 64),
            path: "tiny.png".to_string(),
        });
        image_list.push(DrawCmd::Image {
            rect: rect(150, 50, 33, 47), // non-square resample
            path: "tiny.png".to_string(),
        });
    }
    if images.dimensions("tiny.jpg").is_some() {
        image_list.push(DrawCmd::Image {
            rect: rect(250, 50, 80, 40),
            path: "tiny.jpg".to_string(),
        });
    }

    let mut everything = fills.clone();
    everything.extend(rounds.clone());
    everything.extend(texts.clone());
    everything.extend(image_list.clone().into_iter().skip(1));

    vec![
        ("fills", fills),
        ("round-rects", rounds),
        ("text", texts),
        ("images", image_list),
        ("everything", everything),
    ]
}

/// A 0x00RRGGBB color as RGBA floats (alpha 1), the clear-value form.
fn color_f32(color: u32) -> [f32; 4] {
    [
        ((color >> 16) & 0xff) as f32 / 255.0,
        ((color >> 8) & 0xff) as f32 / 255.0,
        (color & 0xff) as f32 / 255.0,
        1.0,
    ]
}
