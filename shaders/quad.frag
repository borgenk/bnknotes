// The one fragment shader. Modes (src/gpu.rs): 0 solid fill, 1 glyph coverage
// from the R8 atlas, 2 emoji from the BGRA atlas, 3 an image texture,
// 4 rounded-rect coverage, 5 a task check mark. Glyph and emoji reads are texelFetch at integer
// coords so a 1:1 atlas quad reproduces its source texels exactly. An image is
// an sRGB texture sampled with texture() and normalized UVs, so the GPU decodes
// and scales its native-size texture to the display quad in linear space,
// keeping resize cheap and smooth.
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec4 v_extra;
layout(location = 3) flat in uint v_mode;

layout(set = 0, binding = 0) uniform sampler2D glyph_atlas;
layout(set = 0, binding = 1) uniform sampler2D emoji_atlas;
layout(set = 1, binding = 0) uniform sampler2D image_tex;

layout(location = 0) out vec4 out_color;

// Shared with the vertex stage (which reads viewport). The fragment stage reads
// only the glyph mask gamma; both members sit at the offsets the CPU packs.
layout(push_constant) uniform Push {
    vec2 viewport;
    float glyph_coverage_gamma;
} push;

// sRGB to linear. Vertex colors and the emoji atlas carry sRGB-encoded bytes
// and are decoded here; the image texture is an sRGB format the sampler decodes
// on its own. The color attachment is an sRGB view, so blending runs in linear
// space and the hardware re-encodes on store, leaving opaque fills byte-exact
// while anti-aliased edges blend gamma-correct.
vec3 srgb_to_linear(vec3 c) {
    bvec3 hi = greaterThan(c, vec3(0.04045));
    vec3 lo = c / 12.92;
    vec3 hs = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hs, hi);
}

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

// Distance from point p to the segment ab.
float seg_dist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// A check mark filling the box `size` (v_extra.xy): a two-segment tick in the
// box's normalized coordinates, inked by the anti-aliased distance to it. The
// stroke half-width scales with the box (with a floor so it never sub-pixels
// away) and coverage fades over the one-pixel band at that width, so the
// diagonals read smooth at any size instead of the staircase a fill leaves.
float check_coverage(vec2 p, vec2 size) {
    vec2 a = vec2(0.22, 0.52) * size;
    vec2 b = vec2(0.42, 0.71) * size;
    vec2 c = vec2(0.80, 0.30) * size;
    float d = min(seg_dist(p, a, b), seg_dist(p, b, c));
    float half_w = max(size.x * 0.065, 0.75);
    return clamp(half_w - d + 0.5, 0.0, 1.0);
}

void main() {
    if (v_mode == 0u) {
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a);
    } else if (v_mode == 1u) {
        // Mask gamma thins the anti-aliased edges: linear-light compositing is
        // physically correct but renders light-on-dark text heavier than the
        // gamma-space stacks beside it, so raising coverage to this power lowers
        // partial-coverage alpha back to a matching weight. Solid pixels
        // (coverage 1) are unchanged, so color fills keep full strength.
        float cov = pow(texelFetch(glyph_atlas, ivec2(v_uv), 0).r, push.glyph_coverage_gamma);
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    } else if (v_mode == 2u) {
        vec4 e = texelFetch(emoji_atlas, ivec2(v_uv), 0);
        out_color = vec4(srgb_to_linear(e.rgb), e.a);
    } else if (v_mode == 3u) {
        // Image: the sRGB texture is decoded and scaled to the quad in linear
        // space by the sampler, so its result is already linear.
        out_color = texture(image_tex, v_uv);
    } else if (v_mode == 4u) {
        float cov = round_coverage(v_uv, v_extra.xy, v_extra.z, uint(v_extra.w));
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    } else {
        float cov = check_coverage(v_uv, v_extra.xy);
        out_color = vec4(srgb_to_linear(v_color.rgb), v_color.a * cov);
    }
}
