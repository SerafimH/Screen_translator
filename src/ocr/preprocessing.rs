//! Pure image preprocessing functions for OCR (no Win32 or OS dependencies).
//!
//! Provides filters for extracting meme-style text (light text with dark outlines),
//! inverting image buffers, and scaling/contrast-enhancing frames within engine bounds.

use crate::types::luma_from_bgr;

/// Maximum image dimension supported by Windows WinRT OcrEngine (2600x2600).
pub const MAX_OCR_DIMENSION: u32 = 2600;

/// Minimum image dimension supported by Windows WinRT OcrEngine (40x40).
pub const MIN_OCR_DIMENSION: u32 = 40;

/// Extracts a black-and-white mask for light text with dark outlines (typical meme/Impact font).
///
/// A pixel is classified as text if:
/// 1. It is very light (`luma >= 165`) and low-saturation (`max(RGB) - min(RGB) <= 65`).
/// 2. It has dark outline pixels (`luma <= 85`) within a small neighborhood (radius 1..7 px).
///
/// Returns a BGRA8 buffer where detected text pixels are pure black (0, 0, 0, 255)
/// and all background pixels are pure white (255, 255, 255, 255).
pub fn extract_outlined_text_mask(bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    if width == 0 || height == 0 || bgra.len() < pixel_count * 4 {
        return Vec::new();
    }

    let w = width as i32;
    let h = height as i32;

    // Precompute luma and candidate categories for all pixels
    let mut is_dark = vec![false; pixel_count];
    let mut is_light_candidate = vec![false; pixel_count];

    for y in 0..h {
        let row_offset = y as usize * width as usize;
        for x in 0..w {
            let idx = (row_offset + x as usize) * 4;
            let luma = luma_from_bgr(bgra[idx], bgra[idx + 1], bgra[idx + 2]);
            let max_c = bgra[idx].max(bgra[idx + 1]).max(bgra[idx + 2]);
            let min_c = bgra[idx].min(bgra[idx + 1]).min(bgra[idx + 2]);
            let sat = max_c - min_c;

            let p_idx = row_offset + x as usize;
            if luma <= 85 {
                is_dark[p_idx] = true;
            }
            if luma >= 170 && sat <= 60 {
                is_light_candidate[p_idx] = true;
            }
        }
    }

    const RADIUS: usize = 6;
    let w_usize = width as usize;
    let h_usize = height as usize;

    // 1. Separable horizontal morphological dilation of is_dark in O(W * H)
    let mut dark_h = vec![false; pixel_count];
    for y in 0..h_usize {
        let row_offset = y * w_usize;
        let mut count = 0usize;
        let init_end = RADIUS.min(w_usize.saturating_sub(1));
        for i in 0..=init_end {
            if is_dark[row_offset + i] {
                count += 1;
            }
        }
        for x in 0..w_usize {
            dark_h[row_offset + x] = count > 0;
            let enter_x = x + 1 + RADIUS;
            if enter_x < w_usize && is_dark[row_offset + enter_x] {
                count += 1;
            }
            if x >= RADIUS && is_dark[row_offset + (x - RADIUS)] {
                count -= 1;
            }
        }
    }

    // 2. Separable vertical morphological dilation of dark_h in O(W * H)
    let mut dark_dilated = vec![false; pixel_count];
    for x in 0..w_usize {
        let mut count = 0usize;
        let init_end = RADIUS.min(h_usize.saturating_sub(1));
        for y in 0..=init_end {
            if dark_h[y * w_usize + x] {
                count += 1;
            }
        }
        for y in 0..h_usize {
            dark_dilated[y * w_usize + x] = count > 0;
            let enter_y = y + 1 + RADIUS;
            if enter_y < h_usize && dark_h[enter_y * w_usize + x] {
                count += 1;
            }
            if y >= RADIUS && dark_h[(y - RADIUS) * w_usize + x] {
                count -= 1;
            }
        }
    }

    // Default entire output to pure white background
    let mut result = vec![255u8; pixel_count * 4];

    for i in 0..pixel_count {
        if is_light_candidate[i] && dark_dilated[i] {
            let out_idx = i * 4;
            result[out_idx] = 0; // B
            result[out_idx + 1] = 0; // G
            result[out_idx + 2] = 0; // R
            result[out_idx + 3] = 255;
        }
    }

    result
}

/// Inverts the colors of a BGRA8 buffer, leaving Alpha as 255.
pub fn invert_bgra(bgra: &[u8]) -> Vec<u8> {
    let mut inverted = bgra.to_vec();
    for chunk in inverted.as_chunks_mut::<4>().0 {
        chunk[0] = 255 - chunk[0];
        chunk[1] = 255 - chunk[1];
        chunk[2] = 255 - chunk[2];
        chunk[3] = 255;
    }
    inverted
}

/// Upscales an image buffer by `requested_scale` (clamped so output does not exceed `MAX_OCR_DIMENSION`),
/// and applies contrast enhancement (S-curve contrast boost).
///
/// Returns `(scaled_bgra, new_width, new_height, applied_scale)`.
pub fn upscale_and_enhance_contrast(
    bgra: &[u8],
    width: u32,
    height: u32,
    requested_scale: f32,
) -> (Vec<u8>, u32, u32, f32) {
    if width == 0 || height == 0 || bgra.len() < (width * height * 4) as usize {
        return (Vec::new(), 0, 0, 1.0);
    }

    // Ensure image does not exceed engine limit (2600x2600)
    let max_side = width.max(height) as f32;
    let max_allowable_scale = MAX_OCR_DIMENSION as f32 / max_side;
    let scale = requested_scale.min(max_allowable_scale).max(1.0);

    let new_w = ((width as f32 * scale).round() as u32).clamp(MIN_OCR_DIMENSION, MAX_OCR_DIMENSION);
    let new_h =
        ((height as f32 * scale).round() as u32).clamp(MIN_OCR_DIMENSION, MAX_OCR_DIMENSION);

    let mut dst = vec![0u8; (new_w * new_h * 4) as usize];
    let x_ratio = (width as f32 - 1.0) / (new_w as f32 - 1.0).max(1.0);
    let y_ratio = (height as f32 - 1.0) / (new_h as f32 - 1.0).max(1.0);

    // Bilinear interpolation
    for dy in 0..new_h {
        let sy = dy as f32 * y_ratio;
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(height - 1);
        let wy = sy - y0 as f32;
        let wy0 = 1.0 - wy;

        let row0 = (y0 as usize * width as usize) * 4;
        let row1 = (y1 as usize * width as usize) * 4;
        let out_row = (dy as usize * new_w as usize) * 4;

        for dx in 0..new_w {
            let sx = dx as f32 * x_ratio;
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let wx = sx - x0 as f32;
            let wx0 = 1.0 - wx;

            let col0 = x0 as usize * 4;
            let col1 = x1 as usize * 4;
            let out_col = out_row + dx as usize * 4;

            for c in 0..3 {
                let v00 = bgra[row0 + col0 + c] as f32;
                let v01 = bgra[row0 + col1 + c] as f32;
                let v10 = bgra[row1 + col0 + c] as f32;
                let v11 = bgra[row1 + col1 + c] as f32;

                let top = v00 * wx0 + v01 * wx;
                let bottom = v10 * wx0 + v11 * wx;
                let val = top * wy0 + bottom * wy;

                // Contrast enhancement: stretch values away from 128 (factor 1.35)
                let enhanced = ((val - 128.0) * 1.35 + 128.0).clamp(0.0, 255.0).round() as u8;
                dst[out_col + c] = enhanced;
            }
            dst[out_col + 3] = 255;
        }
    }

    (dst, new_w, new_h, scale)
}

/// Converts a BGRA8 buffer to a 3-channel RGB buffer with forced alpha=255.
/// If pixel alpha < 255, it composites the color over a pure white background.
pub fn flatten_bgra_to_rgb(bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    if width == 0 || height == 0 || bgra.len() < pixel_count * 4 {
        return Vec::new();
    }
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in bgra.as_chunks::<4>().0.iter().take(pixel_count) {
        let b = chunk[0] as u32;
        let g = chunk[1] as u32;
        let r = chunk[2] as u32;
        let a = chunk[3] as u32;
        if a == 255 {
            rgb.push(r as u8);
            rgb.push(g as u8);
            rgb.push(b as u8);
        } else {
            let cr = ((r * a + 255 * (255 - a)) / 255) as u8;
            let cg = ((g * a + 255 * (255 - a)) / 255) as u8;
            let cb = ((b * a + 255 * (255 - a)) / 255) as u8;
            rgb.push(cr);
            rgb.push(cg);
            rgb.push(cb);
        }
    }
    rgb
}

/// Adds `pad` pixels of padding around an RGB image using the detected background color
/// averaged from the outermost perimeter pixels.
pub fn add_rgb_padding_with_edge_color(
    rgb: &[u8],
    width: u32,
    height: u32,
    pad: u32,
) -> (Vec<u8>, u32, u32) {
    if width == 0 || height == 0 || rgb.len() < (width * height * 3) as usize {
        return (Vec::new(), 0, 0);
    }
    if pad == 0 {
        return (rgb.to_vec(), width, height);
    }

    let w = width as usize;
    let h = height as usize;

    // Sample border pixels to determine edge background color
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut count = 0u64;

    for x in 0..w {
        let top_idx = x * 3;
        sum_r += rgb[top_idx] as u64;
        sum_g += rgb[top_idx + 1] as u64;
        sum_b += rgb[top_idx + 2] as u64;

        let bot_idx = ((h - 1) * w + x) * 3;
        sum_r += rgb[bot_idx] as u64;
        sum_g += rgb[bot_idx + 1] as u64;
        sum_b += rgb[bot_idx + 2] as u64;
        count += 2;
    }
    for y in 1..h.saturating_sub(1) {
        let left_idx = (y * w) * 3;
        sum_r += rgb[left_idx] as u64;
        sum_g += rgb[left_idx + 1] as u64;
        sum_b += rgb[left_idx + 2] as u64;

        let right_idx = (y * w + w - 1) * 3;
        sum_r += rgb[right_idx] as u64;
        sum_g += rgb[right_idx + 1] as u64;
        sum_b += rgb[right_idx + 2] as u64;
        count += 2;
    }

    let pad_r = if count > 0 { (sum_r / count) as u8 } else { 255 };
    let pad_g = if count > 0 { (sum_g / count) as u8 } else { 255 };
    let pad_b = if count > 0 { (sum_b / count) as u8 } else { 255 };

    let out_w = width + pad * 2;
    let out_h = height + pad * 2;
    let out_w_usize = out_w as usize;
    let pad_usize = pad as usize;

    let mut out = vec![0u8; (out_w * out_h * 3) as usize];
    for chunk in out.as_chunks_mut::<3>().0 {
        chunk[0] = pad_r;
        chunk[1] = pad_g;
        chunk[2] = pad_b;
    }

    for y in 0..h {
        let src_row = y * w * 3;
        let dst_row = ((y + pad_usize) * out_w_usize + pad_usize) * 3;
        out[dst_row..dst_row + w * 3].copy_from_slice(&rgb[src_row..src_row + w * 3]);
    }

    (out, out_w, out_h)
}

/// Computes the negative of the min(R, G, B) channel: `255 - min(R, G, B)`.
/// This transforms white text (R,G,B all high) on colored background (at least one channel low)
/// into dark text on a light background.
pub fn min_rgb_channel_negative(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    if width == 0 || height == 0 || rgb.len() < pixel_count * 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pixel_count * 3);
    for chunk in rgb.as_chunks::<3>().0.iter().take(pixel_count) {
        let min_c = chunk[0].min(chunk[1]).min(chunk[2]);
        let val = 255 - min_c;
        out.push(val);
        out.push(val);
        out.push(val);
    }
    out
}

/// Extracts bright pixels where `min(R, G, B) >= threshold`, turning them pure black (0,0,0)
/// and all other pixels pure white (255,255,255).
pub fn bright_pixel_mask(rgb: &[u8], width: u32, height: u32, threshold: u8) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    if width == 0 || height == 0 || rgb.len() < pixel_count * 3 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(pixel_count * 3);
    for chunk in rgb.as_chunks::<3>().0.iter().take(pixel_count) {
        let min_c = chunk[0].min(chunk[1]).min(chunk[2]);
        let val = if min_c >= threshold { 0u8 } else { 255u8 };
        out.push(val);
        out.push(val);
        out.push(val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_outlined_text_mask_synthetic_gradient() {
        let width = 60;
        let height = 30;
        let mut bgra = vec![0u8; (width * height * 4) as usize];

        // 1. Fill with a colorful background gradient (e.g. bright orange/yellow to green)
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                bgra[idx] = 30; // Blue
                bgra[idx + 1] = (x * 4).min(200) as u8; // Green
                bgra[idx + 2] = (200 - y * 4).max(50) as u8; // Red
                bgra[idx + 3] = 255;
            }
        }

        // 2. Draw black outline around a letter box (x: 20..40, y: 10..20)
        for y in 9..=21 {
            for x in 19..=41 {
                if y == 9 || y == 21 || x == 19 || x == 41 {
                    let idx = ((y * width + x) * 4) as usize;
                    bgra[idx] = 10;
                    bgra[idx + 1] = 10;
                    bgra[idx + 2] = 10;
                    bgra[idx + 3] = 255;
                }
            }
        }

        // 3. Draw white text interior inside the box (x: 22..38, y: 12..18)
        for y in 12..=18 {
            for x in 22..=38 {
                let idx = ((y * width + x) * 4) as usize;
                bgra[idx] = 250;
                bgra[idx + 1] = 250;
                bgra[idx + 2] = 250;
                bgra[idx + 3] = 255;
            }
        }

        let mask = extract_outlined_text_mask(&bgra, width, height);
        assert_eq!(mask.len(), (width * height * 4) as usize);

        // Text pixel inside the white box must become black [0, 0, 0, 255]
        let text_idx = ((15 * width + 30) * 4) as usize;
        assert_eq!(
            &mask[text_idx..text_idx + 4],
            &[0, 0, 0, 255],
            "Interior text must be black"
        );

        // Background pixel outside (e.g. at (5, 5)) must be pure white [255, 255, 255, 255]
        let bg_idx = ((5 * width + 5) * 4) as usize;
        assert_eq!(
            &mask[bg_idx..bg_idx + 4],
            &[255, 255, 255, 255],
            "Background must be white"
        );
    }

    #[test]
    fn test_invert_bgra() {
        let input = vec![10, 20, 30, 255, 200, 220, 240, 255];
        let inv = invert_bgra(&input);
        assert_eq!(inv, vec![245, 235, 225, 255, 55, 35, 15, 255]);
    }

    #[test]
    fn test_upscale_and_enhance_respects_max_dimension() {
        let large_w = 2000;
        let large_h = 1500;
        let bgra = vec![128u8; (large_w * large_h * 4) as usize];

        // Requesting 2.0x would result in 4000 width, which exceeds 2600.
        // It must be clamped so new width <= MAX_OCR_DIMENSION.
        let (_, new_w, new_h, effective_scale) =
            upscale_and_enhance_contrast(&bgra, large_w, large_h, 2.0);

        assert!(
            new_w <= MAX_OCR_DIMENSION,
            "Width must not exceed engine limit"
        );
        assert!(
            new_h <= MAX_OCR_DIMENSION,
            "Height must not exceed engine limit"
        );
        assert!(effective_scale <= (MAX_OCR_DIMENSION as f32 / large_w as f32) + 0.01);
    }

    #[test]
    fn test_flatten_bgra_to_rgb() {
        // BGRA: B=10, G=20, R=30, A=255 -> RGB: R=30, G=20, B=10
        let bgra = vec![10, 20, 30, 255];
        let rgb = flatten_bgra_to_rgb(&bgra, 1, 1);
        assert_eq!(rgb, vec![30, 20, 10]);

        // Semi-transparent: B=0, G=0, R=0, A=128 composited on white (255)
        // (0*128 + 255*127)/255 = 127
        let bgra_trans = vec![0, 0, 0, 128];
        let rgb_trans = flatten_bgra_to_rgb(&bgra_trans, 1, 1);
        assert_eq!(rgb_trans, vec![127, 127, 127]);
    }

    #[test]
    fn test_add_rgb_padding_with_edge_color() {
        // 2x2 image, edge color is blue: [0, 0, 200]
        let rgb = vec![
            0, 0, 200,   0, 0, 200,
            0, 0, 200,   0, 0, 200,
        ];
        let (padded, out_w, out_h) = add_rgb_padding_with_edge_color(&rgb, 2, 2, 1);
        assert_eq!(out_w, 4);
        assert_eq!(out_h, 4);
        assert_eq!(padded.len(), 4 * 4 * 3);
        // All pixels in padded image should match the edge color [0, 0, 200]
        for chunk in padded.as_chunks::<3>().0 {
            assert_eq!(chunk, &[0, 0, 200]);
        }
    }

    #[test]
    fn test_min_rgb_channel_negative() {
        // White text: [255, 255, 255] -> min=255 -> val=0 (black)
        // Orange background: [255, 128, 0] -> min=0 -> val=255 (white)
        let rgb = vec![255, 255, 255, 255, 128, 0];
        let out = min_rgb_channel_negative(&rgb, 2, 1);
        assert_eq!(out, vec![0, 0, 0, 255, 255, 255]);
    }

    #[test]
    fn test_bright_pixel_mask() {
        // Very bright pixel: [240, 230, 225] -> min=225 >= 225 -> black [0, 0, 0]
        // Darker pixel: [200, 200, 200] -> min=200 < 225 -> white [255, 255, 255]
        let rgb = vec![240, 230, 225, 200, 200, 200];
        let out = bright_pixel_mask(&rgb, 2, 1, 225);
        assert_eq!(out, vec![0, 0, 0, 255, 255, 255]);
    }
}
