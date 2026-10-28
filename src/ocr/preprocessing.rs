//! Pure image preprocessing functions for OCR (no Win32 or OS dependencies).
//!
//! Provides filters for extracting meme-style text (light text with dark outlines),
//! inverting image buffers, and scaling/contrast-enhancing frames within engine bounds.

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
            let b = bgra[idx] as f32;
            let g = bgra[idx + 1] as f32;
            let r = bgra[idx + 2] as f32;

            let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            let max_c = bgra[idx].max(bgra[idx + 1]).max(bgra[idx + 2]);
            let min_c = bgra[idx].min(bgra[idx + 1]).min(bgra[idx + 2]);
            let sat = max_c - min_c;

            let p_idx = row_offset + x as usize;
            if luma <= 85 {
                is_dark[p_idx] = true;
            }
            if luma >= 165 && sat <= 65 {
                is_light_candidate[p_idx] = true;
            }
        }
    }

    // Default entire output to pure white background
    let mut result = vec![255u8; pixel_count * 4];

    // Search offsets within circular radius of 7 pixels (Chebyshev bounds: dx in -7..=7, dy in -7..=7)
    const RADIUS: i32 = 7;
    const RADIUS_SQ: i32 = RADIUS * RADIUS;

    for y in 0..h {
        let row_offset = y as usize * width as usize;
        for x in 0..w {
            let p_idx = row_offset + x as usize;
            if !is_light_candidate[p_idx] {
                continue;
            }

            // Check if there is at least one dark outline pixel within radius
            let mut has_dark_outline = false;
            let y_min = (y - RADIUS).max(0);
            let y_max = (y + RADIUS).min(h - 1);
            let x_min = (x - RADIUS).max(0);
            let x_max = (x + RADIUS).min(w - 1);

            'outer: for ny in y_min..=y_max {
                let dy = ny - y;
                let ny_row = ny as usize * width as usize;
                for nx in x_min..=x_max {
                    let dx = nx - x;
                    if dx * dx + dy * dy <= RADIUS_SQ && is_dark[ny_row + nx as usize] {
                        has_dark_outline = true;
                        break 'outer;
                    }
                }
            }

            if has_dark_outline {
                let out_idx = p_idx * 4;
                result[out_idx] = 0; // B
                result[out_idx + 1] = 0; // G
                result[out_idx + 2] = 0; // R
                result[out_idx + 3] = 255;
            }
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
}
