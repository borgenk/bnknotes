// The one fragment shader. Modes (src/gpu.rs): 0 solid fill, 1 glyph coverage
// from the R8 atlas, 2 emoji from the BGRA atlas, 3 an image texture,
// 4 rounded-rect coverage. Texture reads are texelFetch at integer coords so a
// 1:1 quad reproduces its source texels exactly (no sampler filtering between
// this path and the software blitter's pixels).
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec4 v_extra;
layout(location = 3) flat in uint v_mode;

layout(set = 0, binding = 0) uniform sampler2D glyph_atlas;
layout(set = 0, binding = 1) uniform sampler2D emoji_atlas;
layout(set = 1, binding = 0) uniform sampler2D image_tex;

layout(location = 0) out vec4 out_color;

// Replicates render.rs corner_coverage: full coverage everywhere except
// inside an active corner cell, where the pixel center's distance to the
// corner circle fades coverage across the one-pixel band at the radius.
// v_uv already sits at the fragment (pixel) center, the CPU's dx + 0.5.
float round_coverage(vec2 p, vec2 size, float r, uint mask) {
    bool top = (mask & 1u) != 0u;
    bool bottom = (mask & 2u) != 0u;
    vec2 c;
    if (top && p.x < r && p.y < r) {
        c = vec2(r, r);
    } else if (top && p.x >= size.x - r && p.y < r) {
        c = vec2(size.x - r, r);
    } else if (bottom && p.x < r && p.y >= size.y - r) {
        c = vec2(r, size.y - r);
    } else if (bottom && p.x >= size.x - r && p.y >= size.y - r) {
        c = vec2(size.x - r, size.y - r);
    } else {
        return 1.0;
    }
    return clamp(r - length(p - c) + 0.5, 0.0, 1.0);
}

void main() {
    if (v_mode == 0u) {
        out_color = v_color;
    } else if (v_mode == 1u) {
        float cov = texelFetch(glyph_atlas, ivec2(v_uv), 0).r;
        out_color = vec4(v_color.rgb, v_color.a * cov);
    } else if (v_mode == 2u) {
        out_color = texelFetch(emoji_atlas, ivec2(v_uv), 0);
    } else if (v_mode == 3u) {
        out_color = texelFetch(image_tex, ivec2(v_uv), 0);
    } else {
        float cov = round_coverage(v_uv, v_extra.xy, v_extra.z, uint(v_extra.w));
        out_color = vec4(v_color.rgb, v_color.a * cov);
    }
}
