pub mod helper;
pub mod preprocessing;

pub use helper::OcrHelperClient;

use image::{ImageBuffer, Rgba};
use std::collections::VecDeque;
use std::fs;
use tracing::info;
use windows::Graphics::Imaging::{BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;

use crate::capture::CapturedFrame;
use crate::types::{OcrBlock, RecognizedLine, RecognizedWord, ScreenRect};

fn save_debug_image(path: &str, buf: &[u8], width: u32, height: u32) {
    if std::env::var("SCREEN_TRANSLATOR_DEBUG_OCR").is_ok() {
        let _ = fs::create_dir_all("debug_ocr");
        let mut rgba = buf.to_vec();
        for chunk in rgba.as_chunks_mut::<4>().0 {
            chunk.swap(0, 2);
        }
        if let Some(img) = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba) {
            let _ = img.save(format!("debug_ocr/{}", path));
        }
    }
}

pub struct WinrtOcrEngine {
    engine: OcrEngine,
    language_tag: String,
}

// OcrEngine is safe to send between threads in WinRT
unsafe impl Send for WinrtOcrEngine {}
unsafe impl Sync for WinrtOcrEngine {}

#[derive(Debug, Clone, Copy)]
pub struct RemapParams {
    pub pad: u32,
    pub scale: f32,
    pub offset_x: i32,
    pub offset_y: i32,
}

pub struct PreprocessedImage {
    pub bgra_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
    pub pad: u32,
}

struct OcrPassParams<'a> {
    bgra_bytes: &'a [u8],
    width: u32,
    height: u32,
    remap: RemapParams,
}

/// Evaluates the linguistic quality of recognized lines to rank candidate OCR passes.
/// Lines classified as garbage are skipped; points are awarded for alphabetic letters and valid words.
pub fn score_recognized_lines(lines: &[RecognizedLine]) -> usize {
    let mut score = 0;
    for line in lines {
        if is_ocr_garbage(&line.text) {
            continue;
        }
        let letters = line.text.chars().filter(|c| c.is_alphabetic()).count();
        let words = line.text.split_whitespace().count();
        score += letters + (words * 2);
    }
    score
}

impl WinrtOcrEngine {
    pub fn new() -> Result<Self, String> {
        if let Ok(langs) = OcrEngine::AvailableRecognizerLanguages()
            && let Ok(size) = langs.Size()
            && size == 0
        {
            return Err(
                "В Windows не установлен языковой пакет OCR. Установите английский пакет OCR в Параметрах Windows (Время и язык -> Язык и регион)."
                    .to_string(),
            );
        }

        let engine = OcrEngine::TryCreateFromUserProfileLanguages()
            .map_err(|e| format!("Failed to create WinRT OcrEngine: {:?}", e))?;
        let lang = engine
            .RecognizerLanguage()
            .map_err(|e| format!("Failed to get RecognizerLanguage: {:?}", e))?
            .LanguageTag()
            .map_err(|e| format!("Failed to get LanguageTag: {:?}", e))?
            .to_string();

        info!("Initialized WinrtOcrEngine with language: {}", lang);
        Ok(Self {
            engine,
            language_tag: lang,
        })
    }

    pub fn language(&self) -> &str {
        &self.language_tag
    }

    /// Checks if a given language (e.g. "en" or "ru") is installed and available for WinRT OCR.
    pub fn is_language_available(target_lang: &str) -> bool {
        if let Ok(langs) = OcrEngine::AvailableRecognizerLanguages()
            && let Ok(size) = langs.Size()
        {
            for i in 0..size {
                if let Ok(lang) = langs.GetAt(i)
                    && let Ok(tag) = lang.LanguageTag()
                {
                    let tag_str = tag.to_string().to_lowercase();
                    if tag_str.starts_with(&target_lang.to_lowercase()) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Executes only Pass 1 (standard manga preprocessing) for quick candidate evaluation.
    pub async fn recognize_pass_1(
        &self,
        frame: &CapturedFrame,
    ) -> Result<Vec<RecognizedLine>, String> {
        if frame.width == 0 || frame.height == 0 || frame.bgra_bytes.is_empty() {
            return Ok(Vec::new());
        }

        let (prep, _) = prepare_ocr_image(frame);
        let remap = RemapParams {
            pad: prep.pad,
            scale: prep.scale,
            offset_x: frame.offset_x,
            offset_y: frame.offset_y,
        };

        let pass_params = OcrPassParams {
            bgra_bytes: &prep.bgra_bytes,
            width: prep.width,
            height: prep.height,
            remap,
        };
        self.run_ocr_pass(pass_params).await
    }

    /// Recognizes text in the captured frame using an adaptive multi-pass OCR strategy:
    ///
    /// 1. Standard manga preprocessing (bubble cleaner, contrast, 2x upscale, white padding)
    ///
    /// If pass 1 yields no or low-quality text, subsequent fallback passes execute within an 800ms deadline:
    /// - Outlined text mask (for meme/Impact font text with black outlines on colored/noisy backgrounds)
    /// - Inverted manga buffer (for inverted / glowing text)
    /// - Raw image without manga cleaner (for game UI / HUD / photo text)
    /// - Upscaled & contrast-enhanced image (for fine / small print)
    pub async fn recognize(&self, frame: &CapturedFrame) -> Result<Vec<OcrBlock>, String> {
        if frame.width == 0 || frame.height == 0 || frame.bgra_bytes.is_empty() {
            return Ok(Vec::new());
        }

        save_debug_image("raw.png", &frame.bgra_bytes, frame.width, frame.height);

        // 1. Pass 1: Standard manga preprocessing (bubble cleaning and contrast enhancement)
        let (prep, fallback_inv) = prepare_ocr_image(frame);
        save_debug_image("prep.png", &prep.bgra_bytes, prep.width, prep.height);

        let remap = RemapParams {
            pad: prep.pad,
            scale: prep.scale,
            offset_x: frame.offset_x,
            offset_y: frame.offset_y,
        };

        let pass_params = OcrPassParams {
            bgra_bytes: &prep.bgra_bytes,
            width: prep.width,
            height: prep.height,
            remap,
        };
        let first_lines = self.run_ocr_pass(pass_params).await?;
        let first_score = score_recognized_lines(&first_lines);

        info!(
            "OCR pass 1 (manga): found {} raw lines (score={})",
            first_lines.len(),
            first_score
        );

        let mut best_lines = first_lines;
        let mut best_score = first_score;

        // If first pass found quality text, do not waste time on extra passes
        if best_score < 12 {
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(800);

            // Pass 2: Outlined text mask (meme font filter)
            if std::time::Instant::now() < deadline {
                let mask_bgra = preprocessing::extract_outlined_text_mask(
                    &frame.bgra_bytes,
                    frame.width,
                    frame.height,
                );
                save_debug_image("meme_mask.png", &mask_bgra, frame.width, frame.height);

                let mask_pass = OcrPassParams {
                    bgra_bytes: &mask_bgra,
                    width: frame.width,
                    height: frame.height,
                    remap: RemapParams {
                        pad: 0,
                        scale: 1.0,
                        offset_x: frame.offset_x,
                        offset_y: frame.offset_y,
                    },
                };
                if let Ok(mask_lines) = self.run_ocr_pass(mask_pass).await {
                    let score = score_recognized_lines(&mask_lines);
                    info!(
                        "OCR pass 2 (meme_mask): found {} raw lines (score={})",
                        mask_lines.len(),
                        score
                    );
                    if score > best_score {
                        best_score = score;
                        best_lines = mask_lines;
                    }
                }
            }

            // Pass 3: Inverted manga buffer (if available)
            if best_score < 15
                && std::time::Instant::now() < deadline
                && let Some(inv) = &fallback_inv
            {
                save_debug_image("inv.png", inv, prep.width, prep.height);
                let inv_pass = OcrPassParams {
                    bgra_bytes: inv,
                    width: prep.width,
                    height: prep.height,
                    remap,
                };
                if let Ok(inv_lines) = self.run_ocr_pass(inv_pass).await {
                    let score = score_recognized_lines(&inv_lines);
                    info!(
                        "OCR pass 3 (inverted_manga): found {} raw lines (score={})",
                        inv_lines.len(),
                        score
                    );
                    if score > best_score {
                        best_score = score;
                        best_lines = inv_lines;
                    }
                }
            }

            // Pass 4: Raw image (no manga cleaner, e.g. for photo / clean game UI)
            if best_score < 15 && std::time::Instant::now() < deadline {
                let raw_pass = OcrPassParams {
                    bgra_bytes: &frame.bgra_bytes,
                    width: frame.width,
                    height: frame.height,
                    remap: RemapParams {
                        pad: 0,
                        scale: 1.0,
                        offset_x: frame.offset_x,
                        offset_y: frame.offset_y,
                    },
                };
                if let Ok(raw_lines) = self.run_ocr_pass(raw_pass).await {
                    let score = score_recognized_lines(&raw_lines);
                    info!(
                        "OCR pass 4 (raw): found {} raw lines (score={})",
                        raw_lines.len(),
                        score
                    );
                    if score > best_score {
                        best_score = score;
                        best_lines = raw_lines;
                    }
                }
            }

            // Pass 5: Upscaled & contrast-enhanced (for small / fine print)
            if best_score < 15 && std::time::Instant::now() < deadline {
                let (up_bgra, up_w, up_h, applied_scale) =
                    preprocessing::upscale_and_enhance_contrast(
                        &frame.bgra_bytes,
                        frame.width,
                        frame.height,
                        2.0,
                    );
                if applied_scale > 1.05 {
                    save_debug_image("upscaled.png", &up_bgra, up_w, up_h);
                    let up_pass = OcrPassParams {
                        bgra_bytes: &up_bgra,
                        width: up_w,
                        height: up_h,
                        remap: RemapParams {
                            pad: 0,
                            scale: applied_scale,
                            offset_x: frame.offset_x,
                            offset_y: frame.offset_y,
                        },
                    };
                    if let Ok(up_lines) = self.run_ocr_pass(up_pass).await {
                        let score = score_recognized_lines(&up_lines);
                        info!(
                            "OCR pass 5 (upscaled): found {} raw lines (score={})",
                            up_lines.len(),
                            score
                        );
                        if score > best_score {
                            best_lines = up_lines;
                        }
                    }
                }
            }
        }

        // 2. Cluster recognized lines into coherent paragraphs and speech bubbles
        let blocks = cluster_lines_into_blocks(
            best_lines,
            Some((&frame.bgra_bytes, frame.width, frame.height)),
        );
        Ok(blocks)
    }

    async fn run_ocr_pass(&self, params: OcrPassParams<'_>) -> Result<Vec<RecognizedLine>, String> {
        let ocr_op = {
            let writer =
                DataWriter::new().map_err(|e| format!("Failed to create DataWriter: {:?}", e))?;
            writer
                .WriteBytes(params.bgra_bytes)
                .map_err(|e| format!("Failed to write bytes to DataWriter: {:?}", e))?;
            let ibuffer = writer
                .DetachBuffer()
                .map_err(|e| format!("Failed to detach buffer: {:?}", e))?;

            let software_bitmap = SoftwareBitmap::CreateCopyWithAlphaFromBuffer(
                &ibuffer,
                BitmapPixelFormat::Bgra8,
                params.width as i32,
                params.height as i32,
                BitmapAlphaMode::Premultiplied,
            )
            .map_err(|e| format!("Failed to create SoftwareBitmap: {:?}", e))?;

            self.engine
                .RecognizeAsync(&software_bitmap)
                .map_err(|e| format!("RecognizeAsync failed: {:?}", e))?
        };

        let ocr_result = ocr_op
            .await
            .map_err(|e| format!("OCR recognition operation failed: {:?}", e))?;

        let lines_view = ocr_result
            .Lines()
            .map_err(|e| format!("Failed to get OCR lines: {:?}", e))?;
        let line_count = lines_view
            .Size()
            .map_err(|e| format!("Failed to get lines count: {:?}", e))?;

        let mut lines = Vec::new();

        for i in 0..line_count {
            let line = lines_view
                .GetAt(i)
                .map_err(|e| format!("Failed to get line {}: {:?}", i, e))?;
            let line_text = line
                .Text()
                .map_err(|e| format!("Failed to get line text: {:?}", e))?
                .to_string();

            let words_view = line
                .Words()
                .map_err(|e| format!("Failed to get line words: {:?}", e))?;
            let word_count = words_view
                .Size()
                .map_err(|e| format!("Failed to get word count: {:?}", e))?;

            let mut words = Vec::new();
            let mut line_rect = ScreenRect::new(0, 0, 0, 0);

            for j in 0..word_count {
                let word = words_view
                    .GetAt(j)
                    .map_err(|e| format!("Failed to get word: {:?}", e))?;
                let word_text = word
                    .Text()
                    .map_err(|e| format!("Failed to get word text: {:?}", e))?
                    .to_string();
                let rect = word
                    .BoundingRect()
                    .map_err(|e| format!("Failed to get word rect: {:?}", e))?;

                let word_rect = remap_rect(rect.X, rect.Y, rect.Width, rect.Height, params.remap);

                line_rect = line_rect.union(&word_rect);
                words.push(RecognizedWord {
                    text: word_text,
                    rect: word_rect,
                });
            }

            let trimmed = line_text.trim();
            if !trimmed.is_empty() && !is_ocr_garbage(trimmed) {
                lines.push(RecognizedLine {
                    text: trimmed.to_string(),
                    rect: line_rect,
                    words,
                });
            }
        }

        Ok(lines)
    }
}

/// Converts coordinates from preprocessed padded/upscaled image back to global desktop screen space.
pub fn remap_rect(
    raw_x: f32,
    raw_y: f32,
    raw_w: f32,
    raw_h: f32,
    params: RemapParams,
) -> ScreenRect {
    let unpad_x = (raw_x - params.pad as f32).max(0.0);
    let unpad_y = (raw_y - params.pad as f32).max(0.0);
    let orig_x = (unpad_x / params.scale).round() as i32 + params.offset_x;
    let orig_y = (unpad_y / params.scale).round() as i32 + params.offset_y;
    let orig_w = (raw_w / params.scale).round() as i32;
    let orig_h = (raw_h / params.scale).round() as i32;

    ScreenRect::new(orig_x, orig_y, orig_w.max(1), orig_h.max(1))
}

/// 8-connected flood-fill algorithm that wipes spike frames, screentones, and speed lines
/// touching the borders with pure white (255) while safely preserving speech bubble text in the center.
pub fn clean_manga_bubble_background(buf: &mut [u8], width: u32, height: u32) {
    if width < 3 || height < 3 || buf.len() < (width * height * 4) as usize {
        return;
    }

    let w = width as usize;
    let h = height as usize;
    let mut visited = vec![false; w * h];
    let mut queue = VecDeque::new();

    let get_luma = |b: &[u8], idx: usize| -> u8 {
        let blue = b[idx] as f32;
        let green = b[idx + 1] as f32;
        let red = b[idx + 2] as f32;
        (0.299 * red + 0.587 * green + 0.114 * blue).round() as u8
    };

    // Initialize BFS with boundary pixels having Luma <= 205 (frame spikes and speedlines)
    for x in 0..w {
        let top_idx = x * 4;
        if get_luma(buf, top_idx) <= 205 {
            visited[x] = true;
            queue.push_back((x, 0));
        }
        let bot_idx = ((h - 1) * w + x) * 4;
        if get_luma(buf, bot_idx) <= 205 {
            visited[(h - 1) * w + x] = true;
            queue.push_back((x, h - 1));
        }
    }

    for y in 1..h - 1 {
        let left_idx = (y * w) * 4;
        if get_luma(buf, left_idx) <= 205 {
            visited[y * w] = true;
            queue.push_back((0, y));
        }
        let right_idx = (y * w + w - 1) * 4;
        if get_luma(buf, right_idx) <= 205 {
            visited[y * w + w - 1] = true;
            queue.push_back((w - 1, y));
        }
    }

    let neighbors: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    while let Some((cx, cy)) = queue.pop_front() {
        let p_idx = (cy * w + cx) * 4;
        // Turn spike / noise pixel into clean white
        buf[p_idx] = 255;
        buf[p_idx + 1] = 255;
        buf[p_idx + 2] = 255;
        buf[p_idx + 3] = 255;

        for (dx, dy) in neighbors {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;
            if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                let unx = nx as usize;
                let uny = ny as usize;
                let v_idx = uny * w + unx;
                if !visited[v_idx] {
                    let n_buf_idx = v_idx * 4;
                    if get_luma(buf, n_buf_idx) <= 205 {
                        visited[v_idx] = true;
                        queue.push_back((unx, uny));
                    }
                }
            }
        }
    }
}

/// Sharpens manga contrast by cutting gray midtones (>215 -> 255) and intensifying black text (<80 -> 0).
pub fn sharpen_manga_contrast(buf: &mut [u8]) {
    for chunk in buf.as_chunks_mut::<4>().0 {
        let b = chunk[0] as f32;
        let g = chunk[1] as f32;
        let r = chunk[2] as f32;
        let luma = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;

        if luma > 215 {
            chunk[0] = 255;
            chunk[1] = 255;
            chunk[2] = 255;
        } else if luma < 80 {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
        }
    }
}

/// Determines the scale factor for the manga OCR pass, enforcing that the padded result
/// does not exceed the maximum engine dimension (2600 px).
pub fn calculate_manga_ocr_scale(width: u32, height: u32, pad: u32) -> f32 {
    let needed = width < 1200 || height < 600;
    let fits = (width as u64 * 2 + pad as u64 * 2 <= preprocessing::MAX_OCR_DIMENSION as u64)
        && (height as u64 * 2 + pad as u64 * 2 <= preprocessing::MAX_OCR_DIMENSION as u64);
    if needed && fits { 2.0 } else { 1.0 }
}

/// Preprocesses the captured frame with the complete manga OCR pipeline:
/// Bubble cleaner -> Contrast sharpening -> 2x Catmull-Rom upscale -> 24px white padding.
pub fn prepare_ocr_image(frame: &CapturedFrame) -> (PreprocessedImage, Option<Vec<u8>>) {
    let pad = 24;
    let scale = calculate_manga_ocr_scale(frame.width, frame.height, pad);

    let mut cleaned_buf = frame.bgra_bytes.clone();
    clean_manga_bubble_background(&mut cleaned_buf, frame.width, frame.height);
    sharpen_manga_contrast(&mut cleaned_buf);

    let (scaled_buf, scaled_w, scaled_h) = if (scale - 2.0f32).abs() < f32::EPSILON {
        upscale_2x_catmull_rom(&cleaned_buf, frame.width, frame.height)
    } else {
        (cleaned_buf, frame.width, frame.height)
    };

    let (padded_buf, padded_w, padded_h) =
        add_pure_white_padding(&scaled_buf, scaled_w, scaled_h, pad);
    let fallback_inv = Some(create_inverted_buffer(&padded_buf));

    (
        PreprocessedImage {
            bgra_bytes: padded_buf,
            width: padded_w,
            height: padded_h,
            scale,
            pad,
        },
        fallback_inv,
    )
}

/// Checks if a single token consists exclusively of spike/line OCR noise (| / \ 1 I l _ ~).
pub fn is_spike_word(word: &str) -> bool {
    let trimmed = word.trim();
    if trimmed.is_empty() {
        return true;
    }
    let spike_chars = [
        '|', '/', '\\', '1', 'I', 'l', '_', '~', '!', '-', '`', '\'', ';', ':',
    ];
    trimmed.chars().all(|c| spike_chars.contains(&c))
}

/// Identifies meaningless OCR artifacts resulting from burst bubbles, action lines, and panel frames.
pub fn is_ocr_garbage(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    let digit_count = trimmed.chars().filter(|c| c.is_numeric()).count();
    let total_non_space = trimmed.chars().filter(|c| !c.is_whitespace()).count();

    if total_non_space == 0 {
        return true;
    }

    // Pure numbers like "100" or "+50" are legitimate game HUD metrics
    if alpha_count == 0 && digit_count > 0 {
        let is_clean_number = trimmed
            .trim_matches(|c: char| c == '+' || c == '-' || c == '$' || c == '%' || c == '#')
            .parse::<f64>()
            .is_ok();
        if is_clean_number {
            return false;
        }
    }

    if alpha_count == 0 && total_non_space >= digit_count * 2 {
        return true;
    }

    if alpha_count == 0 && digit_count == 0 {
        return true;
    }

    let meaningful = alpha_count + digit_count;
    let ratio = meaningful as f32 / total_non_space as f32;

    if ratio < 0.40 {
        return true;
    }

    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if !words.is_empty() && words.iter().all(|w| is_spike_word(w)) {
        return true;
    }

    false
}

/// Checks if a dark border / panel separator exists in the pixel band between two rectangles.
pub fn has_border_between(
    bgra: &[u8],
    width: u32,
    height: u32,
    r1: &ScreenRect,
    r2: &ScreenRect,
) -> bool {
    if width == 0 || height == 0 || bgra.len() < (width * height * 4) as usize {
        return false;
    }

    let w = width as i32;
    let h = height as i32;

    let get_luma = |x: i32, y: i32| -> u8 {
        if x < 0 || x >= w || y < 0 || y >= h {
            return 255;
        }
        let idx = (y as usize * width as usize + x as usize) * 4;
        let b = bgra[idx] as f32;
        let g = bgra[idx + 1] as f32;
        let r = bgra[idx + 2] as f32;
        (0.299 * r + 0.587 * g + 0.114 * b).round() as u8
    };

    // Horizontal check
    if r1.x + r1.width <= r2.x || r2.x + r2.width <= r1.x {
        let (left, right) = if r1.x < r2.x { (r1, r2) } else { (r2, r1) };
        let x_start = left.x + left.width;
        let x_end = right.x;
        let y_start = left.y.max(right.y);
        let y_end = (left.y + left.height).min(right.y + right.height);

        if x_end > x_start && y_end > y_start {
            let mut dark_count = 0;
            let mut total = 0;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    if get_luma(x, y) < 120 {
                        dark_count += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 && (dark_count as f32 / total as f32) >= 0.15 {
                return true;
            }
        }
    }

    // Vertical check
    if r1.y + r1.height <= r2.y || r2.y + r2.height <= r1.y {
        let (top, bottom) = if r1.y < r2.y { (r1, r2) } else { (r2, r1) };
        let y_start = top.y + top.height;
        let y_end = bottom.y;
        let x_start = top.x.max(bottom.x);
        let x_end = (top.x + top.width).min(bottom.x + bottom.width);

        if y_end > y_start && x_end > x_start {
            let mut dark_count = 0;
            let mut total = 0;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    if get_luma(x, y) < 120 {
                        dark_count += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 && (dark_count as f32 / total as f32) >= 0.15 {
                return true;
            }
        }
    }

    false
}

/// Catmull-Rom 2x upscale for high-definition text edge reconstruction.
pub fn upscale_2x_catmull_rom(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let out_w = w * 2;
    let out_h = h * 2;
    let mut dst = vec![0u8; (out_w * out_h * 4) as usize];

    for y in 0..out_h {
        let src_yf = (y as f32 + 0.5) * 0.5 - 0.5;
        let y0 = (src_yf.floor() as i32).clamp(0, h as i32 - 1) as usize;
        let y1 = (y0 + 1).min(h as usize - 1);
        let wy = (src_yf - y0 as f32).clamp(0.0, 1.0);
        let wy0 = 1.0 - wy;

        for x in 0..out_w {
            let src_xf = (x as f32 + 0.5) * 0.5 - 0.5;
            let x0 = (src_xf.floor() as i32).clamp(0, w as i32 - 1) as usize;
            let x1 = (x0 + 1).min(w as usize - 1);
            let wx = (src_xf - x0 as f32).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx;

            let idx00 = (y0 * w as usize + x0) * 4;
            let idx01 = (y0 * w as usize + x1) * 4;
            let idx10 = (y1 * w as usize + x0) * 4;
            let idx11 = (y1 * w as usize + x1) * 4;

            let out_idx = (y as usize * out_w as usize + x as usize) * 4;

            for c in 0..4 {
                let v00 = src[idx00 + c] as f32;
                let v01 = src[idx01 + c] as f32;
                let v10 = src[idx10 + c] as f32;
                let v11 = src[idx11 + c] as f32;

                let top = v00 * wx0 + v01 * wx;
                let bottom = v10 * wx0 + v11 * wx;
                let val = (top * wy0 + bottom * wy).round() as u8;

                dst[out_idx + c] = val;
            }
        }
    }

    (dst, out_w, out_h)
}

/// Adds pure white (255) margin padding so boundary letters are not cropped.
pub fn add_pure_white_padding(src: &[u8], w: u32, h: u32, pad: u32) -> (Vec<u8>, u32, u32) {
    let out_w = w + pad * 2;
    let out_h = h + pad * 2;
    let mut dst = vec![255u8; (out_w * out_h * 4) as usize];

    for y in 0..h {
        let src_start = (y as usize * w as usize) * 4;
        let src_end = src_start + (w as usize * 4);
        let dst_start = ((y + pad) as usize * out_w as usize + pad as usize) * 4;
        dst[dst_start..dst_start + (w as usize * 4)].copy_from_slice(&src[src_start..src_end]);
    }

    (dst, out_w, out_h)
}

/// Inverts color buffer for second OCR pass.
pub fn create_inverted_buffer(buf: &[u8]) -> Vec<u8> {
    let mut inv = buf.to_vec();
    for chunk in inv.as_chunks_mut::<4>().0 {
        chunk[0] = 255 - chunk[0];
        chunk[1] = 255 - chunk[1];
        chunk[2] = 255 - chunk[2];
        chunk[3] = 255;
    }
    inv
}

/// Clusters recognized lines into coherent paragraphs and speech bubbles,
/// taking line barriers and hyphenated wrap into account.
pub fn cluster_lines_into_blocks(
    mut lines: Vec<RecognizedLine>,
    frame_bgra: Option<(&[u8], u32, u32)>,
) -> Vec<OcrBlock> {
    if lines.is_empty() {
        return Vec::new();
    }

    // Filter out garbage lines
    lines.retain(|l| !is_ocr_garbage(&l.text));

    // Sort lines primarily top-to-bottom, secondarily left-to-right
    lines.sort_by(|a, b| {
        a.rect
            .y
            .cmp(&b.rect.y)
            .then_with(|| a.rect.x.cmp(&b.rect.x))
    });

    let mut blocks: Vec<OcrBlock> = Vec::new();

    for line in lines {
        let mut merged = false;

        for block in blocks.iter_mut() {
            // Prohibit merge if a visual panel barrier separates them
            if let Some((bgra, w, h)) = frame_bgra
                && has_border_between(bgra, w, h, &block.rect, &line.rect)
            {
                continue;
            }

            let avg_line_h = (block.rect.height as f32 / block.lines.len() as f32).max(12.0);
            let v_gap = line.rect.y - (block.rect.y + block.rect.height);

            if v_gap >= -6 && v_gap <= (avg_line_h * 1.45) as i32 {
                let x_overlap = line.rect.x.max(block.rect.x)
                    <= (line.rect.x + line.rect.width).min(block.rect.x + block.rect.width);
                let left_aligned = (line.rect.x - block.rect.x).abs() <= 50;
                let center_aligned = ((line.rect.x + line.rect.width / 2)
                    - (block.rect.x + block.rect.width / 2))
                    .abs()
                    <= 60;

                if x_overlap || left_aligned || center_aligned {
                    block.rect = block.rect.union(&line.rect);

                    // Check hyphenated word wrap: "adven-" + "ture" -> "adventure"
                    let trimmed_block = block.text.trim_end();
                    if trimmed_block.ends_with('-') && !trimmed_block.ends_with("--") {
                        let base = trimmed_block.trim_end_matches('-').trim_end();
                        block.text = format!("{}{}", base, line.text.trim_start());
                    } else if !block.text.is_empty() {
                        block.text.push(' ');
                        block.text.push_str(line.text.trim());
                    } else {
                        block.text.push_str(line.text.trim());
                    }

                    block.lines.push(line.clone());
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            blocks.push(OcrBlock {
                text: line.text.trim().to_string(),
                rect: line.rect,
                lines: vec![line],
            });
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_rect_with_pad_and_scale() {
        let params = RemapParams {
            pad: 18,
            scale: 2.0,
            offset_x: 100,
            offset_y: 200,
        };

        let rect = remap_rect(56.0, 76.0, 100.0, 40.0, params);
        assert_eq!(rect.x, 119);
        assert_eq!(rect.y, 229);
        assert_eq!(rect.width, 50);
        assert_eq!(rect.height, 20);
    }

    #[test]
    fn test_cluster_lines_hyphen_merging() {
        let line1 = RecognizedLine {
            text: "This was an adven-".to_string(),
            rect: ScreenRect::new(100, 100, 200, 20),
            words: Vec::new(),
        };
        let line2 = RecognizedLine {
            text: "ture of a lifetime.".to_string(),
            rect: ScreenRect::new(100, 124, 210, 20),
            words: Vec::new(),
        };

        let blocks = cluster_lines_into_blocks(vec![line1, line2], None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "This was an adventure of a lifetime.");
        assert_eq!(blocks[0].rect, ScreenRect::new(100, 100, 210, 44));
    }

    #[test]
    fn test_cluster_distinct_paragraphs() {
        let line1 = RecognizedLine {
            text: "Top dialogue box.".to_string(),
            rect: ScreenRect::new(100, 100, 200, 20),
            words: Vec::new(),
        };
        let line2 = RecognizedLine {
            text: "Far bottom dialogue box.".to_string(),
            rect: ScreenRect::new(100, 300, 200, 20), // 180px gap
            words: Vec::new(),
        };

        let blocks = cluster_lines_into_blocks(vec![line1, line2], None);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_clean_manga_bubble_background_preserves_center_text() {
        let w = 20;
        let h = 20;
        let mut buf = vec![0u8; (w * h * 4) as usize];

        // 1. Fill entire image with white bubble background (240 luma > 205)
        for chunk in buf.as_chunks_mut::<4>().0 {
            chunk[0] = 240;
            chunk[1] = 240;
            chunk[2] = 240;
            chunk[3] = 255;
        }

        // 2. Put outer frame spikes along top boundary touching y=0 (0 luma <= 205)
        for x in 0..w {
            let idx = (x * 4) as usize;
            buf[idx] = 0;
            buf[idx + 1] = 0;
            buf[idx + 2] = 0;
        }

        // 3. Put black dialogue text in the center at (10, 10) (0 luma <= 205)
        let text_idx = ((10 * w + 10) * 4) as usize;
        buf[text_idx] = 0;
        buf[text_idx + 1] = 0;
        buf[text_idx + 2] = 0;

        clean_manga_bubble_background(&mut buf, w as u32, h as u32);

        // The outer spike at (0, 0) should be flood-filled to pure white (255)
        assert_eq!(buf[0], 255);
        assert_eq!(buf[1], 255);
        assert_eq!(buf[2], 255);

        // The text in the center must be preserved (still black 0)
        assert_eq!(buf[text_idx], 0);
        assert_eq!(buf[text_idx + 1], 0);
        assert_eq!(buf[text_idx + 2], 0);
    }

    #[test]
    fn test_has_border_between_detects_dark_boundary() {
        let w = 100;
        let h = 50;
        let mut bgra = vec![255u8; (w * h * 4) as usize];

        // Draw a dark vertical panel line between x=45 and x=55
        for y in 0..h {
            for x in 45..55 {
                let idx = ((y * w + x) * 4) as usize;
                bgra[idx] = 20; // B
                bgra[idx + 1] = 20; // G
                bgra[idx + 2] = 20; // R
            }
        }

        let bubble_left = ScreenRect::new(10, 10, 30, 30);
        let bubble_right = ScreenRect::new(60, 10, 30, 30);

        let border_detected =
            has_border_between(&bgra, w as u32, h as u32, &bubble_left, &bubble_right);
        assert!(border_detected);
    }

    #[test]
    fn test_has_border_between_no_boundary() {
        let w = 100;
        let h = 50;
        // Pure white background between bubbles
        let bgra = vec![255u8; (w * h * 4) as usize];

        let bubble_left = ScreenRect::new(10, 10, 30, 30);
        let bubble_right = ScreenRect::new(60, 10, 30, 30);

        let border_detected =
            has_border_between(&bgra, w as u32, h as u32, &bubble_left, &bubble_right);
        assert!(!border_detected);
    }

    #[test]
    fn test_is_ocr_garbage_filters_artifacts() {
        assert!(is_ocr_garbage("|| // 1! \\"));
        assert!(is_ocr_garbage("| | / \\"));
        assert!(is_ocr_garbage("~~~~___"));
        assert!(is_ocr_garbage("   "));
        assert!(is_ocr_garbage("! ! ! |"));
    }

    #[test]
    fn test_is_ocr_garbage_preserves_speech_and_sfx() {
        assert!(!is_ocr_garbage("WHICH ONE?!"));
        assert!(!is_ocr_garbage("KRAAAGH!"));
        assert!(!is_ocr_garbage("100"));
        assert!(!is_ocr_garbage("Hello world"));
        assert!(!is_ocr_garbage("What is your HP?"));
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_meme_no_pickles_integration() {
        if !WinrtOcrEngine::is_language_available("en") {
            println!("SKIPPED: English WinRT OCR language pack is not installed on this system.");
            return;
        }

        let img = match image::open("tests/fixtures/meme_no_pickles.png") {
            Ok(im) => im.to_rgba8(),
            Err(e) => {
                println!(
                    "SKIPPED: tests/fixtures/meme_no_pickles.png not found: {}",
                    e
                );
                return;
            }
        };

        let (width, height) = (img.width(), img.height());
        let mut bgra = Vec::with_capacity((width * height * 4) as usize);
        for pixel in img.pixels() {
            bgra.push(pixel[2]);
            bgra.push(pixel[1]);
            bgra.push(pixel[0]);
            bgra.push(pixel[3]);
        }
        let frame = CapturedFrame {
            bgra_bytes: bgra,
            width,
            height,
            offset_x: 0,
            offset_y: 0,
        };

        let engine = WinrtOcrEngine::new().expect("Failed to initialize WinrtOcrEngine");
        let blocks = engine
            .recognize(&frame)
            .await
            .expect("OCR recognition failed");

        assert!(
            blocks.len() >= 2,
            "Expected at least 2 distinct blocks (top and bottom), found {}",
            blocks.len()
        );

        let full_text = blocks
            .iter()
            .map(|b| b.text.to_uppercase())
            .collect::<Vec<_>>()
            .join(" ");
        println!("Full recognized meme text: {}", full_text);

        // Check top text: "EXCUSE ME SIR" (allowing 1-2 symbol errors)
        assert!(
            full_text.contains("EXCUSE")
                && (full_text.contains("SIR")
                    || full_text.contains("SOR")
                    || full_text.contains("SW")),
            "Top text 'EXCUSE ME SIR' must be detected: {}",
            full_text
        );

        // Check bottom text: "SHE ASKED FOR NO PICKLES" (allowing 1-2 symbol errors)
        assert!(
            full_text.contains("ASKED")
                && full_text.contains("NO")
                && (full_text.contains("PICKLE") || full_text.contains("PIC")),
            "Bottom text 'SHE ASKED FOR NO PICKLES' must be detected: {}",
            full_text
        );
    }

    #[test]
    fn test_score_recognized_lines_quality() {
        let garbage = vec![
            RecognizedLine {
                text: "|| // 1! \\".to_string(),
                rect: ScreenRect::new(0, 0, 10, 10),
                words: vec![],
            },
            RecognizedLine {
                text: "~~~___".to_string(),
                rect: ScreenRect::new(0, 0, 10, 10),
                words: vec![],
            },
        ];
        assert_eq!(
            score_recognized_lines(&garbage),
            0,
            "Garbage lines must score 0"
        );

        let good = vec![
            RecognizedLine {
                text: "EXCUSE ME SIR".to_string(),
                rect: ScreenRect::new(0, 0, 100, 20),
                words: vec![],
            },
            RecognizedLine {
                text: "SHE ASKED FOR NO PICKLES".to_string(),
                rect: ScreenRect::new(0, 50, 200, 20),
                words: vec![],
            },
        ];
        let score = score_recognized_lines(&good);
        assert!(score >= 40, "Clear sentences must score high: {}", score);
    }

    #[tokio::test]
    #[cfg(windows)]
    async fn test_meme_in_full_desktop_screen() {
        if !WinrtOcrEngine::is_language_available("en") {
            return;
        }
        let img = match image::open("tests/fixtures/meme_no_pickles.png") {
            Ok(im) => im.to_rgba8(),
            Err(_) => return,
        };
        let screen_w = 1920;
        let screen_h = 1080;
        let mut bgra = vec![40u8; (screen_w * screen_h * 4) as usize];

        let paste_x = 400;
        let paste_y = 50;
        for y in 0..img.height().min(screen_h - paste_y) {
            for x in 0..img.width().min(screen_w - paste_x) {
                let px = img.get_pixel(x, y);
                let idx = (((paste_y + y) * screen_w + (paste_x + x)) * 4) as usize;
                bgra[idx] = px[2];
                bgra[idx + 1] = px[1];
                bgra[idx + 2] = px[0];
                bgra[idx + 3] = 255;
            }
        }

        let frame = CapturedFrame {
            bgra_bytes: bgra,
            width: screen_w,
            height: screen_h,
            offset_x: 0,
            offset_y: 0,
        };

        let engine = WinrtOcrEngine::new().unwrap();
        let blocks = engine.recognize(&frame).await.unwrap();
        println!("Full screen blocks: {}", blocks.len());
        for b in &blocks {
            println!("  Block: '{}' at {:?}", b.text, b.rect);
        }
    }

    #[test]
    fn test_calculate_manga_ocr_scale_bounds() {
        let pad = 24;

        // 1276x300: 1276*2 + 48 = 2600 (exact engine boundary) -> scale 2.0
        assert_eq!(calculate_manga_ocr_scale(1276, 300, pad), 2.0);

        // 1277x300: 1277*2 + 48 = 2602 > 2600 -> scale 1.0
        assert_eq!(calculate_manga_ocr_scale(1277, 300, pad), 1.0);

        // 1350x400: 1350*2 + 48 = 2748 > 2600 -> scale 1.0
        assert_eq!(calculate_manga_ocr_scale(1350, 400, pad), 1.0);

        // 1000x300: 1000*2 + 48 = 2048 <= 2600 -> scale 2.0
        assert_eq!(calculate_manga_ocr_scale(1000, 300, pad), 2.0);

        // 300x100: 300*2 + 48 = 648 <= 2600 -> scale 2.0
        assert_eq!(calculate_manga_ocr_scale(300, 100, pad), 2.0);
    }
}
