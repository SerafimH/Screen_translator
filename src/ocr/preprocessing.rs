//! Pure image preprocessing functions for OCR (no Win32 or OS dependencies).
//!
//! Provides filters for extracting meme-style text (light text with dark outlines),
//! inverting image buffers, and scaling/contrast-enhancing frames within engine bounds.

use crate::types::luma_from_bgr;

/// Maximum image dimension supported by Windows WinRT OcrEngine (2600x2600).
pub const MAX_OCR_DIMENSION: u32 = 2600;

/// Minimum image dimension supported by Windows WinRT OcrEngine (40x40).
pub const MIN_OCR_DIMENSION: u32 = 40;

/// Gives tightly cropped lines enough background for the recognizer without
/// stretching their glyphs. Large images are borrowed by the caller unchanged.
pub fn pad_small_ocr_image(bgra: &[u8], width: u32, height: u32) -> Option<(Vec<u8>, u32, u32, u32)> {
    if width == 0 || height == 0 || width.min(height) >= MIN_OCR_DIMENSION
        || bgra.len() != width as usize * height as usize * 4 {
        return None;
    }
    let pad = (MIN_OCR_DIMENSION - width.min(height)).div_ceil(2).max(12);
    let padded_w = width.checked_add(pad * 2)?;
    let padded_h = height.checked_add(pad * 2)?;
    let mut histograms = [[0usize; 256]; 3];
    let mut samples = 0usize;
    for y in 0..height {
        for x in 0..width {
            if x != 0 && y != 0 && x + 1 != width && y + 1 != height { continue; }
            let pixel = ((y * width + x) * 4) as usize;
            for c in 0..3 { histograms[c][bgra[pixel + c] as usize] += 1; }
            samples += 1;
        }
    }
    let mut background = [0u8, 0, 0, 255];
    for c in 0..3 {
        let mut count = 0;
        for (value, &frequency) in histograms[c].iter().enumerate() {
            count += frequency;
            if count > samples / 2 { background[c] = value as u8; break; }
        }
    }
    let mut padded = vec![0u8; padded_w as usize * padded_h as usize * 4];
    for pixel in padded.chunks_exact_mut(4) { pixel.copy_from_slice(&background); }
    for y in 0..height as usize {
        let source = y * width as usize * 4;
        let target = ((y + pad as usize) * padded_w as usize + pad as usize) * 4;
        padded[target..target + width as usize * 4].copy_from_slice(&bgra[source..source + width as usize * 4]);
    }
    Some((padded, padded_w, padded_h, pad))
}

/// Fits the whole image, including any padding, within the engine limit.
pub fn ocr_dimensions(width: u32, height: u32, max_dimension: u32) -> (u32, u32) {
    let scale = (max_dimension.max(1) as f64 / width.max(height).max(1) as f64).min(1.0);
    (
        ((width as f64 * scale).floor() as u32).max(1),
        ((height as f64 * scale).floor() as u32).max(1),
    )
}

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

    let new_w = ((width as f32 * scale).round() as u32).clamp(1, MAX_OCR_DIMENSION);
    let new_h =
        ((height as f32 * scale).round() as u32).clamp(1, MAX_OCR_DIMENSION);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_line_padding_preserves_pixels_and_uses_background_not_text_color() {
        let mut pixels = [32u8, 48, 64, 255].repeat(20 * 6);
        // A glyph touching the upper edge must not turn the whole margin black.
        pixels[..4].copy_from_slice(&[0, 0, 0, 255]);
        let (padded, width, height, pad) = pad_small_ocr_image(&pixels, 20, 6).unwrap();
        assert_eq!((width, height, pad), (54, 40, 17));
        assert_eq!(&padded[..4], &[32, 48, 64, 255]);
        for y in 0..6usize {
            let start = ((y + pad as usize) * width as usize + pad as usize) * 4;
            assert_eq!(&padded[start..start + 80], &pixels[y * 80..(y + 1) * 80]);
        }
        assert!(pad_small_ocr_image(&[255; 40 * 40 * 4], 40, 40).is_none());
        assert!(pad_small_ocr_image(&[], 20, 6).is_none());
    }

    #[test]
    fn tiny_line_upscale_does_not_stretch_height_independently() {
        let (_, width, height, scale) = upscale_and_enhance_contrast(&[128; 100 * 8 * 4], 100, 8, 2.0);
        assert_eq!((width, height, scale), (200, 16, 2.0));
    }

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
    fn test_ocr_dimensions_include_padding_and_preserve_aspect_ratio() {
        for (width, height) in [(2608, 1488), (3888, 2208), (2208, 3888)] {
            let (w, h) = ocr_dimensions(width, height, MAX_OCR_DIMENSION);
            assert!(w <= MAX_OCR_DIMENSION && h <= MAX_OCR_DIMENSION);
            assert!((width as f64 / height as f64 - w as f64 / h as f64).abs() < 0.002);
        }
        assert_eq!(ocr_dimensions(1000, 360, MAX_OCR_DIMENSION), (1000, 360));
    }

}
