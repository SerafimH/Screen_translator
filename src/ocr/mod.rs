pub mod helper;
pub mod preprocessing;

pub use helper::OcrHelperClient;

use image::{ImageBuffer, Rgba};
use std::collections::VecDeque;
use std::fs;
use tracing::{info, warn};
use windows::Graphics::Imaging::{BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::DataWriter;

use crate::capture::CapturedFrame;
use crate::types::{OcrBlock, RecognizedLine, RecognizedWord, ScreenRect, luma_from_bgr};

/// Merge regional candidates without dropping text found only by one pass.
pub fn merge_recognized_lines(lines: &mut Vec<RecognizedLine>, candidates: Vec<RecognizedLine>) {
    for candidate in candidates {
        if is_ocr_garbage(&candidate.text) { continue; }
        // A rotated helper box is often taller than WinRT's tight word union.
        // Suppress every enclosed native reading of that same baseline, rather
        // than keeping multiple corrupt duplicates beside one confident line.
        if lines.iter().any(|line| strong_line_covers_unscored(line, &candidate)) {
            continue;
        }
        let replacement_index = lines.iter().position(|line| strong_line_covers_unscored(&candidate, line));
        lines.retain(|line| !strong_line_covers_unscored(&candidate, line));
        let candidate_text = candidate.text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let duplicate = lines.iter_mut().find(|line| {
            let a = line.rect;
            let b = candidate.rect;
            let overlap = ((a.x as i64 + a.width as i64).min(b.x as i64 + b.width as i64)
                - a.x.max(b.x) as i64).max(0)
                * ((a.y as i64 + a.height as i64).min(b.y as i64 + b.height as i64)
                    - a.y.max(b.y) as i64).max(0);
            let area_a = (a.width as i64 * a.height as i64).max(1);
            let area_b = (b.width as i64 * b.height as i64).max(1);
            let existing_text = line.text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
            // A clipped tile edge is a duplicate only when its text is contained in
            // the other result. Otherwise keep both partial lines: either may have
            // words that the other tile did not capture.
            let contained_text = contains_word_sequence(&existing_text, &candidate_text)
                || contains_word_sequence(&candidate_text, &existing_text);
            overlap as f64 / area_a.min(area_b) as f64 >= 0.65
                && (contained_text || overlap as f64 / area_a.max(area_b) as f64 >= 0.85)
        });
        if let Some(existing) = duplicate {
            let existing_text = existing.text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
            let candidate_is_fuller = word_tokens(&candidate_text).len() > word_tokens(&existing_text).len()
                && contains_word_sequence(&candidate_text, &existing_text)
                && covers_additional_region(candidate.rect, existing.rect);
            let existing_is_fuller = word_tokens(&existing_text).len() > word_tokens(&candidate_text).len()
                && contains_word_sequence(&existing_text, &candidate_text)
                && covers_additional_region(existing.rect, candidate.rect);
            // Completeness wins over the confidence of a clipped fragment. For
            // competing full readings WinRT still defaults to confidence 0.85.
            if candidate_is_fuller || (!existing_is_fuller
                && candidate.confidence.unwrap_or(0.85) > existing.confidence.unwrap_or(0.85)) {
                *existing = candidate;
            }
        } else if let Some(index) = replacement_index {
            lines.insert(index.min(lines.len()), candidate);
        } else { lines.push(candidate); }
    }
}

fn word_tokens(text: &str) -> Vec<&str> {
    text.split_whitespace().map(|word| word.trim_matches(|ch: char|
        matches!(ch, '.' | ',' | '!' | '?' | ':' | ';' | '"' | '“' | '”' | '«' | '»'
                      | '(' | ')' | '[' | ']' | '{' | '}')))
        .filter(|word| !word.is_empty()).collect()
}

fn contains_word_sequence(full: &str, fragment: &str) -> bool {
    let full = word_tokens(full);
    let fragment = word_tokens(fragment);
    !fragment.is_empty() && full.windows(fragment.len()).any(|words| words == fragment)
}

fn covers_additional_region(full: ScreenRect, fragment: ScreenRect) -> bool {
    !full.is_empty() && !fragment.is_empty()
        && fragment.x as i64 >= full.x as i64 - 2
        && fragment.y as i64 >= full.y as i64 - 2
        && fragment.x as i64 + fragment.width as i64 <= full.x as i64 + full.width as i64 + 2
        && fragment.y as i64 + fragment.height as i64 <= full.y as i64 + full.height as i64 + 2
        && full.width as f64 * full.height as f64 > fragment.width as f64 * fragment.height as f64 * 1.03
}

fn strong_line_covers_unscored(strong: &RecognizedLine, unscored: &RecognizedLine) -> bool {
    if !strong.confidence.is_some_and(|score| score >= 0.95) || unscored.confidence.is_some() {
        return false;
    }
    let a = strong.rect;
    let b = unscored.rect;
    if a.is_empty() || b.is_empty() { return false; }
    let same_baseline = (2 * a.y as i64 + a.height as i64
        - 2 * b.y as i64 - b.height as i64).abs() <= a.height.max(b.height) as i64 / 2;
    same_baseline && a.height as f32 <= b.height as f32 * 1.6
        && b.x as i64 >= a.x as i64 - 2 && b.y as i64 >= a.y as i64 - 2
        && b.x as i64 + b.width as i64 <= a.x as i64 + a.width as i64 + 2
        && b.y as i64 + b.height as i64 <= a.y as i64 + a.height as i64 + 2
}

pub fn ocr_tiles(width: u32, height: u32) -> Vec<ScreenRect> {
    let mut tiles = Vec::new();
    let side = if width.max(height) > 2000 { 1200 } else { width.max(height).max(1) };
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            tiles.push(ScreenRect::new(x as i32, y as i32, side.min(width - x) as i32, side.min(height - y) as i32));
            if x + side >= width { break; }
            x += side - 160;
        }
        if y + side >= height { break; }
        y += side - 160;
    }
    tiles
}

/// Crop uses frame-local coordinates and retains the desktop offset.
pub fn crop_frame(frame: &CapturedFrame, rect: ScreenRect) -> Result<CapturedFrame, String> {
    if rect.is_empty() || rect.x < 0 || rect.y < 0
        || rect.x as u64 + rect.width as u64 > frame.width as u64
        || rect.y as u64 + rect.height as u64 > frame.height as u64
        || frame.bgra_bytes.len() != frame.width as usize * frame.height as usize * 4
    { return Err("Invalid OCR crop".to_string()); }
    let mut bytes = Vec::with_capacity(rect.width as usize * rect.height as usize * 4);
    for y in rect.y..rect.y + rect.height {
        let start = (y as usize * frame.width as usize + rect.x as usize) * 4;
        bytes.extend_from_slice(&frame.bgra_bytes[start..start + rect.width as usize * 4]);
    }
    Ok(CapturedFrame { bgra_bytes: bytes, width: rect.width as u32, height: rect.height as u32,
        offset_x: frame.offset_x + rect.x, offset_y: frame.offset_y + rect.y })
}

struct CancelOnDrop<F: FnMut()>(F);
impl<F: FnMut()> Drop for CancelOnDrop<F> {
    fn drop(&mut self) { (self.0)(); }
}

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
}

// SAFETY: The underlying WinRT runtime class `Windows.Media.Ocr.OcrEngine` has Agile
// marshaling behavior (implements IAgileObject in Windows Metadata), for which the `windows`
// crate explicitly provides `unsafe impl Send for OcrEngine {}` and `unsafe impl Sync for OcrEngine {}`
// (see `windows` v0.62.2, `src/Windows/Media/Ocr/mod.rs:153-154`). Since all fields of `WinrtOcrEngine`
// (`engine: OcrEngine`) are `Send + Sync`, implementing Send and Sync is sound.
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
    pub fn new(target_lang: &str) -> Result<Self, String> {
        if let Ok(langs) = OcrEngine::AvailableRecognizerLanguages()
            && let Ok(size) = langs.Size()
            && size == 0
        {
            return Err(
                "В Windows не установлен языковой пакет OCR. Установите английский пакет OCR в Параметрах Windows (Время и язык -> Язык и регион)."
                    .to_string(),
            );
        }

        let mut matched_lang = None;
        let target_lower = target_lang.trim().to_lowercase();
        if !target_lower.is_empty()
            && let Ok(langs) = OcrEngine::AvailableRecognizerLanguages()
                && let Ok(size) = langs.Size()
            {
                for i in 0..size {
                    if let Ok(lang) = langs.GetAt(i)
                        && let Ok(tag) = lang.LanguageTag()
                    {
                        let tag_str = tag.to_string().to_lowercase();
                        if tag_str == target_lower || tag_str.starts_with(&target_lower) {
                            matched_lang = Some(lang);
                            break;
                        }
                    }
                }
            }

        let engine = if let Some(lang) = matched_lang {
            OcrEngine::TryCreateFromLanguage(&lang).map_err(|e| {
                format!(
                    "Failed to create WinRT OcrEngine for '{}': {:?}",
                    target_lang, e
                )
            })?
        } else {
            if !target_lower.is_empty() {
                warn!(
                    "Requested OCR language '{}' is not installed in Windows; falling back to user profile language.",
                    target_lang
                );
            }
            OcrEngine::TryCreateFromUserProfileLanguages()
                .map_err(|e| format!("Failed to create WinRT OcrEngine: {:?}", e))?
        };
        let lang = engine
            .RecognizerLanguage()
            .map_err(|e| format!("Failed to get RecognizerLanguage: {:?}", e))?
            .LanguageTag()
            .map_err(|e| format!("Failed to get LanguageTag: {:?}", e))?
            .to_string();

        info!("Initialized WinrtOcrEngine with language: {}", lang);
        Ok(Self { engine })
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

        let frame_clone = frame.clone();
        let (prep, _) = tokio::task::spawn_blocking(move || prepare_ocr_image(&frame_clone))
            .await
            .map_err(|e| format!("Prepare OCR image task join error: {:?}", e))?;

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

    /// Reads untouched pixels separately, preserving an independent crop candidate.
    pub async fn recognize_raw(&self, frame: &CapturedFrame) -> Result<Vec<RecognizedLine>, String> {
        self.run_ocr_pass(OcrPassParams {
            bgra_bytes: &frame.bgra_bytes, width: frame.width, height: frame.height,
            remap: RemapParams { pad: 0, scale: 1.0, offset_x: frame.offset_x, offset_y: frame.offset_y },
        }).await
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
        let frame_clone = frame.clone();
        let (prep, fallback_inv) = tokio::task::spawn_blocking(move || prepare_ocr_image(&frame_clone))
            .await
            .map_err(|e| format!("Prepare OCR image task join error: {:?}", e))?;
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
        // Always compare the untouched image: manga cleanup can erase game text.
        if let Ok(raw_lines) = self.run_ocr_pass(OcrPassParams {
            bgra_bytes: &frame.bgra_bytes, width: frame.width, height: frame.height,
            remap: RemapParams { pad: 0, scale: 1.0, offset_x: frame.offset_x, offset_y: frame.offset_y },
        }).await {
            merge_recognized_lines(&mut best_lines, raw_lines);
        }
        let mut best_score = first_score;

        // If first pass found quality text, do not waste time on extra passes
        if best_score < 25 {
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(800);

            // Pass 2: Outlined text mask (meme font filter)
            if tokio::time::Instant::now() < deadline {
                let bytes = frame.bgra_bytes.clone();
                let fw = frame.width;
                let fh = frame.height;
                let mask_bgra = tokio::task::spawn_blocking(move || {
                    preprocessing::extract_outlined_text_mask(&bytes, fw, fh)
                })
                .await
                .unwrap_or_default();
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
                if let Ok(Ok(mask_lines)) = tokio::time::timeout_at(deadline, self.run_ocr_pass(mask_pass)).await {
                    let score = score_recognized_lines(&mask_lines);
                    info!(
                        "OCR pass 2 (meme_mask): found {} raw lines (score={})",
                        mask_lines.len(),
                        score
                    );
                    best_score = best_score.max(score);
                    merge_recognized_lines(&mut best_lines, mask_lines);
                }
            }

            // Pass 3: Inverted manga buffer (if available)
            if best_score < 15
                && tokio::time::Instant::now() < deadline
                && let Some(inv) = &fallback_inv
            {
                save_debug_image("inv.png", inv, prep.width, prep.height);
                let inv_pass = OcrPassParams {
                    bgra_bytes: inv,
                    width: prep.width,
                    height: prep.height,
                    remap,
                };
                if let Ok(Ok(inv_lines)) = tokio::time::timeout_at(deadline, self.run_ocr_pass(inv_pass)).await {
                    let score = score_recognized_lines(&inv_lines);
                    info!(
                        "OCR pass 3 (inverted_manga): found {} raw lines (score={})",
                        inv_lines.len(),
                        score
                    );
                    best_score = best_score.max(score);
                    merge_recognized_lines(&mut best_lines, inv_lines);
                }
            }

            // Pass 5: Upscaled & contrast-enhanced (for small / fine print)
            if best_score < 15 && tokio::time::Instant::now() < deadline {
                let bytes = frame.bgra_bytes.clone();
                let fw = frame.width;
                let fh = frame.height;
                let (up_bgra, up_w, up_h, applied_scale) = tokio::task::spawn_blocking(move || {
                    preprocessing::upscale_and_enhance_contrast(&bytes, fw, fh, 2.0)
                })
                .await
                .unwrap_or_else(|_| (Vec::new(), 0, 0, 1.0));
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
                    if let Ok(Ok(up_lines)) = tokio::time::timeout_at(deadline, self.run_ocr_pass(up_pass)).await {
                        let score = score_recognized_lines(&up_lines);
                        info!(
                            "OCR pass 5 (upscaled): found {} raw lines (score={})",
                            up_lines.len(),
                            score
                        );
                        merge_recognized_lines(&mut best_lines, up_lines);
                    }
                }
            }
        }

        // 2. Cluster recognized lines into coherent paragraphs and speech bubbles
        let blocks = cluster_lines_into_blocks(
            best_lines,
            Some((
                &frame.bgra_bytes,
                frame.width,
                frame.height,
                frame.offset_x,
                frame.offset_y,
            )),
        );
        Ok(blocks)
    }

    async fn run_ocr_pass(&self, params: OcrPassParams<'_>) -> Result<Vec<RecognizedLine>, String> {
        if params.width == 0 || params.height == 0 { return Ok(Vec::new()); }
        if params.bgra_bytes.len() != params.width as usize * params.height as usize * 4 {
            return Err("Invalid OCR image buffer".to_string());
        }
        let padded = preprocessing::pad_small_ocr_image(params.bgra_bytes, params.width, params.height);
        let (input, input_w, input_h, extra_pad) = padded.as_ref()
            .map(|(bytes, width, height, pad)| (bytes.as_slice(), *width, *height, *pad))
            .unwrap_or((params.bgra_bytes, params.width, params.height, 0));
        let (width, height) = preprocessing::ocr_dimensions(
            input_w, input_h,
            OcrEngine::MaxImageDimension().unwrap_or(preprocessing::MAX_OCR_DIMENSION),
        );
        let resized = if (width, height) != (input_w, input_h) {
            let img = ImageBuffer::<Rgba<u8>, _>::from_raw(input_w, input_h, input)
                .ok_or_else(|| "Invalid OCR image buffer".to_string())?;
            Some(image::imageops::resize(&img, width, height, image::imageops::FilterType::CatmullRom))
        } else {
            None
        };
        let bytes = resized.as_ref().map(|img| img.as_raw().as_slice()).unwrap_or(input);
        let scale_x = input_w as f32 / width as f32;
        let scale_y = input_h as f32 / height as f32;
        let ocr_op = {
            let writer =
                DataWriter::new().map_err(|e| format!("Failed to create DataWriter: {:?}", e))?;
            writer
                .WriteBytes(bytes)
                .map_err(|e| format!("Failed to write bytes to DataWriter: {:?}", e))?;
            let ibuffer = writer
                .DetachBuffer()
                .map_err(|e| format!("Failed to detach buffer: {:?}", e))?;

            let software_bitmap = SoftwareBitmap::CreateCopyWithAlphaFromBuffer(
                &ibuffer,
                BitmapPixelFormat::Bgra8,
                width as i32,
                height as i32,
                BitmapAlphaMode::Premultiplied,
            )
            .map_err(|e| format!("Failed to create SoftwareBitmap: {:?}", e))?;

            self.engine
                .RecognizeAsync(&software_bitmap)
                .map_err(|e| format!("RecognizeAsync failed: {:?}", e))?
        };

        let cancel_op = ocr_op.clone();
        let _cancel = CancelOnDrop(move || { let _ = cancel_op.Cancel(); });
        let ocr_result = tokio::time::timeout(std::time::Duration::from_secs(2), async { ocr_op.await })
            .await.map_err(|_| "WinRT OCR timed out after 2 seconds".to_string())?
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

                let word_rect = remap_rect(rect.X * scale_x - extra_pad as f32,
                    rect.Y * scale_y - extra_pad as f32,
                    rect.Width * scale_x, rect.Height * scale_y, params.remap);

                line_rect = line_rect.union(&word_rect);
                words.push(RecognizedWord {
                    confidence: None,
                    is_estimated: false,
                    text: word_text,
                    rect: word_rect,
                });
            }

            let trimmed = line_text.trim();
            if !trimmed.is_empty() && !is_ocr_garbage(trimmed) {
                lines.push(RecognizedLine {
                    confidence: None,
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

    let get_luma = |b: &[u8], idx: usize| -> u8 { luma_from_bgr(b[idx], b[idx + 1], b[idx + 2]) };

    // Check perimeter brightness: only manga speech bubbles on bright backgrounds should be cleaned.
    // Dark game screens or dark UI boxes must never be flooded with white.
    let mut light_border_count = 0usize;
    let total_border = 2 * w + 2 * (h.saturating_sub(2));

    for x in 0..w {
        if get_luma(buf, x * 4) > 205 {
            light_border_count += 1;
        }
        if get_luma(buf, ((h - 1) * w + x) * 4) > 205 {
            light_border_count += 1;
        }
    }
    for y in 1..h - 1 {
        if get_luma(buf, (y * w) * 4) > 205 {
            light_border_count += 1;
        }
        if get_luma(buf, (y * w + w - 1) * 4) > 205 {
            light_border_count += 1;
        }
    }

    if light_border_count * 2 < total_border {
        return;
    }

    let mut visited = vec![false; w * h];
    let mut queue = VecDeque::new();

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
        let luma = luma_from_bgr(chunk[0], chunk[1], chunk[2]);

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
    let fallback_inv = Some(preprocessing::invert_bgra(&padded_buf));

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

fn is_english_first_person_word(text: &str) -> bool {
    let Some(word) = crate::types::cursor_word_spelling(text) else { return false; };
    word == "I" || matches!(word.to_ascii_lowercase().as_str(), "i'll" | "i'm" | "i'd" | "i've")
}

/// Checks if a single token consists exclusively of spike/line OCR noise (| / \ 1 I l _ ~).
pub fn is_spike_word(word: &str) -> bool {
    let trimmed = word.trim();
    if trimmed.is_empty() {
        return true;
    }
    // A vertical stroke may also be the English pronoun. Preserve validated
    // first-person words, without admitting arbitrary runs such as IIII/llll.
    if is_english_first_person_word(trimmed) { return false; }
    let spike_chars = [
        '|', '/', '\\', '1', 'I', 'l', '_', '~', '!', '-', '`', '\'', ';', ':',
    ];
    trimmed.chars().all(|c| spike_chars.contains(&c))
}

fn has_letter_run(text: &str, min_len: usize) -> bool {
    let mut count = 0;
    for c in text.chars() {
        if c.is_alphabetic() {
            count += 1;
            if count >= min_len {
                return true;
            }
        } else {
            count = 0;
        }
    }
    false
}

/// Identifies meaningless OCR artifacts resulting from burst bubbles, action lines, and panel frames.
pub fn is_ocr_garbage(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }
    if is_english_first_person_word(trimmed) { return false; }

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

    if ratio < 0.25 {
        return true;
    }
    if ratio < 0.50 && !has_letter_run(trimmed, 2) {
        return true;
    }

    // Harsh noise symbols (@, #, $, *, ~, |, \, •) typical of title page OCR garbage (e.g. "I@II#O*$", "•PA")
    let harsh_symbols = trimmed
        .chars()
        .filter(|c| matches!(c, '@' | '#' | '$' | '*' | '~' | '|' | '\\' | '^' | '•' | '·' | '●' | '○'))
        .count();
    if harsh_symbols >= 1
        && (trimmed.chars().any(|c| matches!(c, '•' | '·' | '●' | '○')) || harsh_symbols >= 2)
        && (harsh_symbols as f32 / total_non_space as f32) >= 0.25
    {
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
    offset_x: i32,
    offset_y: i32,
    r1: &ScreenRect,
    r2: &ScreenRect,
) -> bool {
    let r1_local = r1.to_local(offset_x, offset_y);
    let r2_local = r2.to_local(offset_x, offset_y);
    let r1 = &r1_local;
    let r2 = &r2_local;

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
        luma_from_bgr(bgra[idx], bgra[idx + 1], bgra[idx + 2])
    };

    let sample_bg = |r: &ScreenRect| -> f32 {
        let pts = [
            (r.x, r.y - 2),
            (r.x + r.width / 2, r.y - 2),
            ((r.x + r.width - 1).max(r.x), r.y - 2),
            (r.x, r.y + r.height + 1),
            (r.x + r.width / 2, r.y + r.height + 1),
            ((r.x + r.width - 1).max(r.x), r.y + r.height + 1),
            (r.x - 2, r.y + r.height / 2),
            (r.x + r.width + 1, r.y + r.height / 2),
        ];
        let mut sum = 0u32;
        let mut count = 0u32;
        for &(x, y) in &pts {
            if x >= 0 && x < w && y >= 0 && y < h {
                sum += get_luma(x, y) as u32;
                count += 1;
            }
        }
        if count > 0 {
            sum as f32 / count as f32
        } else {
            255.0
        }
    };

    let bg1 = sample_bg(r1);
    let bg2 = sample_bg(r2);

    let y_gap = if r1.y < r2.y {
        r2.y - (r1.y + r1.height)
    } else {
        r1.y - (r2.y + r2.height)
    };

    if (bg1 - bg2).abs() > 80.0 && y_gap > 8 {
        return true;
    }

    let bg_luma = (bg1 + bg2) / 2.0;

    let is_separator = |l: u8| -> bool {
        if bg_luma >= 150.0 {
            (bg_luma - l as f32) >= 50.0 && l < 120
        } else {
            (l as f32 - bg_luma) >= 50.0 && l > 150
        }
    };

    // Horizontal check
    if r1.x + r1.width <= r2.x || r2.x + r2.width <= r1.x {
        let (left, right) = if r1.x < r2.x { (r1, r2) } else { (r2, r1) };
        let x_start = left.x + left.width;
        let x_end = right.x;
        let y_start = left.y.max(right.y);
        let y_end = (left.y + left.height).min(right.y + right.height);

        if x_end > x_start && y_end > y_start {
            let mut sep_count = 0;
            let mut total = 0;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    if is_separator(get_luma(x, y)) {
                        sep_count += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 && (sep_count as f32 / total as f32) >= 0.25 {
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
            // Very small gaps (< 8px) between horizontally overlapping lines are line breaks, not panel borders
            if (y_end - y_start) <= 8 {
                return false;
            }
            let mut sep_count = 0;
            let mut total = 0;
            for y in y_start..y_end {
                for x in x_start..x_end {
                    if is_separator(get_luma(x, y)) {
                        sep_count += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 && (sep_count as f32 / total as f32) >= 0.35 {
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
    if w == 0 || h == 0 || src.len() < (w * h * 4) as usize {
        return (Vec::new(), 0, 0);
    }

    if let Some(img) = ImageBuffer::<Rgba<u8>, &[u8]>::from_raw(w, h, &src[..(w * h * 4) as usize])
    {
        let resized =
            image::imageops::resize(&img, out_w, out_h, image::imageops::FilterType::CatmullRom);
        (resized.into_raw(), out_w, out_h)
    } else {
        (Vec::new(), 0, 0)
    }
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

/// Saves the captured frame, translation result, and helper response or engine description.
pub fn dump_ocr_artifacts(frame: &CapturedFrame, raw_json: &str, result_text: &str) {
    match write_ocr_artifacts(std::path::Path::new("debug/ocr"), frame, raw_json, result_text) {
        Ok(dir) => info!("Saved OCR debug dump to {}", dir.display()),
        Err(e) => warn!("Failed to save OCR debug dump: {}", e),
    }
}

fn write_ocr_artifacts(
    root: &std::path::Path,
    frame: &CapturedFrame,
    raw_json: &str,
    result_text: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(root)?;
    let root = root.canonicalize()?;
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let dir = root.join(ts.to_string());
    fs::create_dir(&dir)?;

    // Only numeric, real directories created by this dumper are eligible for retention cleanup.
    let mut runs = Vec::new();
    for entry in fs::read_dir(&root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && let Some(stamp) = entry.file_name().to_str().and_then(|name| name.parse::<u128>().ok())
        {
            runs.push((stamp, entry.path()));
        }
    }
    runs.sort_unstable_by_key(|(stamp, _)| *stamp);
    for (_, old) in runs.iter().take(runs.len().saturating_sub(20)) {
        if old.parent() == Some(root.as_path()) {
            fs::remove_dir_all(old)?;
        }
    }

    let mut rgba = frame.bgra_bytes.clone();
    for pixel in rgba.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(frame.width, frame.height, rgba)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid captured frame"))?;
    img.save(dir.join("frame.png"))?;
    if let Ok(response) = serde_json::from_str::<serde_json::Value>(raw_json) {
        fs::write(dir.join("recognition.json"), serde_json::to_string_pretty(&response)?)?;
    } else {
        fs::write(dir.join("engine.txt"), raw_json)?;
    }
    fs::write(dir.join("result.txt"), result_text)?;
    Ok(dir)
}

/// Sorts lines into natural 2D reading order:
/// Clusters lines vertically by row (Y-axis bands), and within each row sorts horizontally (X-axis, left to right).
pub fn sort_lines_reading_order(lines: &mut Vec<RecognizedLine>) {
    if lines.len() <= 1 {
        return;
    }
    lines.sort_by_key(|l| l.rect.y);

    let mut rows: Vec<Vec<RecognizedLine>> = Vec::new();
    for line in lines.drain(..) {
        let mut target_row = None;
        for (r_idx, row) in rows.iter().enumerate() {
            let row_y = row[0].rect.y;
            let row_h = row[0].rect.height.max(1);
            let cy_row = row_y + row_h / 2;
            let cy_line = line.rect.y + line.rect.height / 2;
            // Overlap or vertical centers within half line height
            if (cy_row - cy_line).abs() <= (row_h / 2).max(10) {
                target_row = Some(r_idx);
                break;
            }
        }
        if let Some(r_idx) = target_row {
            rows[r_idx].push(line);
        } else {
            rows.push(vec![line]);
        }
    }
    for row in rows.iter_mut() {
        row.sort_by_key(|l| l.rect.x);
    }
    rows.sort_by_key(|row| row[0].rect.y);
    for row in rows {
        lines.extend(row);
    }
}

/// Clusters recognized lines into coherent paragraphs and speech bubbles,
/// taking line barriers and hyphenated wrap into account.
pub fn cluster_lines_into_blocks(
    mut lines: Vec<RecognizedLine>,
    frame_bgra: Option<(&[u8], u32, u32, i32, i32)>,
) -> Vec<OcrBlock> {
    if lines.is_empty() {
        return Vec::new();
    }

    // Filter out garbage lines
    lines.retain(|l| !is_ocr_garbage(&l.text));

    // Sort lines into reading order: clusters by Y, sorted by X within each cluster
    sort_lines_reading_order(&mut lines);

    let mut blocks: Vec<OcrBlock> = Vec::new();

    for line in lines {
        let mut merged = false;

        for block in blocks.iter_mut() {
            // Prohibit merge if a visual panel barrier separates them
            if let Some((bgra, w, h, off_x, off_y)) = frame_bgra
                && has_border_between(bgra, w, h, off_x, off_y, &block.rect, &line.rect)
            {
                continue;
            }

            let avg_line_h = (block.rect.height as f32 / block.lines.len() as f32).max(12.0);
            let v_gap = line.rect.y - (block.rect.y + block.rect.height);
            // Slanted/handwritten line boxes can overlap vertically even when
            // the baselines are distinct. Keep one utterance together without
            // joining two fragments on the same row.
            let previous = block.lines.last().expect("blocks contain at least one line");
            let min_h = previous.rect.height.min(line.rect.height).max(1);
            let center_distance = (line.rect.y + line.rect.height / 2)
                - (previous.rect.y + previous.rect.height / 2);
            let overlapping_baselines = v_gap >= -(min_h as f32 * 0.4) as i32
                && center_distance >= min_h / 2;

            if (v_gap >= -6 || overlapping_baselines) && v_gap <= (avg_line_h * 1.45) as i32 {
                // Require overlap of at least 30% of the narrower adjacent
                // line. Fixed alignment tolerances joined short diagonal
                // labels; the block's union could also bridge separate columns.
                let overlap = ((line.rect.x as i64 + line.rect.width as i64)
                    .min(previous.rect.x as i64 + previous.rect.width as i64)
                    - line.rect.x.max(previous.rect.x) as i64).max(0);
                let min_width = line.rect.width.min(previous.rect.width).max(1) as i64;

                if overlap * 10 >= min_width * 3 {
                    block.rect = block.rect.union(&line.rect);

                    // A visible hyphen may belong to a compound or stutter.
                    // Only an explicit soft hyphen confirms a discretionary
                    // break; preserve hard hyphens when joining wrapped text.
                    let trimmed_block = block.text.trim_end();
                    let soft_base = trimmed_block.strip_suffix('\u{00ad}');
                    if let Some(base) = soft_base.or_else(|| trimmed_block.strip_suffix('-'))
                        && base.chars().next_back().is_some_and(char::is_alphabetic)
                        && line.text.trim_start().chars().next().is_some_and(char::is_alphabetic)
                    {
                        let prefix = if soft_base.is_some() { base } else { trimmed_block };
                        block.text = format!("{}{}", prefix, line.text.trim_start());
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
    fn test_merge_keeps_missing_regions_and_uses_confidence() {
        let line = |text: &str, x, confidence| RecognizedLine {
            confidence, text: text.to_string(), rect: ScreenRect::new(x, 10, 100, 20), words: vec![],
        };
        let mut lines = vec![line("Native", 0, None)];
        merge_recognized_lines(&mut lines, vec![line("Uncertain", 0, Some(0.6)), line("Missing dialogue", 200, Some(0.95))]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "Native");
        merge_recognized_lines(&mut lines, vec![line("Corrected", 0, Some(0.99))]);
        assert_eq!(lines[0].text, "Corrected");
        assert_eq!(lines[1].text, "Missing dialogue");
        assert_eq!(lines[0].confidence, Some(0.99));
    }

    #[test]
    fn merge_does_not_expand_a_short_word_based_on_substring_or_extra_characters() {
        let line = |text: &str, rect, confidence| RecognizedLine {
            text: text.into(), rect, confidence: Some(confidence), words: vec![],
        };
        let rect = ScreenRect::new(10, 10, 100, 20);
        for (short, longer) in [("IN", "BEGINNER"), ("HE", "THE"), ("GO", "GOLD")] {
            for (first, second) in [(line(short, rect, 0.99), line(longer, rect, 0.91)),
                                   (line(longer, rect, 0.91), line(short, rect, 0.99))] {
                let mut lines = vec![first];
                merge_recognized_lines(&mut lines, vec![second]);
                assert_eq!(lines.len(), 1);
                assert_eq!(lines[0].text, short);
            }
        }
        let mut lines = vec![line("OPEN THE DOOR", rect, 0.99)];
        merge_recognized_lines(&mut lines, vec![line("DO NOT OPEN THE DOOR", rect, 0.8)]);
        assert_eq!(lines[0].text, "OPEN THE DOOR", "extra text without additional image coverage is not proof");
    }

    #[test]
    fn test_merge_handwritten_line_removes_all_enclosed_native_duplicates_in_either_order() {
        let line = |text: &str, rect, confidence| RecognizedLine {
            text: text.into(), rect, confidence, words: vec![],
        };
        let native = vec![
            line("finA 044?!", ScreenRect::new(82, 133, 196, 54), None),
            line("firlA our !", ScreenRect::new(87, 125, 195, 56), None),
        ];
        let helper = line("find out?!", ScreenRect::new(74, 126, 214, 65), Some(0.98));
        for (mut lines, candidates) in [(native.clone(), vec![helper.clone()]),
                                       (vec![helper.clone()], native)] {
            merge_recognized_lines(&mut lines, candidates);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0].text, "find out?!");
            let neighbor = line("Other panel", ScreenRect::new(300, 126, 200, 65), None);
            let lower = line("Next line", ScreenRect::new(82, 192, 196, 54), None);
            merge_recognized_lines(&mut lines, vec![neighbor, lower]);
            assert_eq!(lines.len(), 3, "uncovered text must remain");
        }
    }

    #[test]
    fn test_merge_preserves_full_line_over_clipped_tile_in_either_order() {
        let line = |text: &str, width, confidence| RecognizedLine {
            text: text.to_string(), rect: ScreenRect::new(1100, 20, width, 20),
            confidence, words: vec![],
        };
        let clipped = line("PLEASE", 100, Some(0.99));
        let full = line("PLEASE FOLLOW INSTRUCTIONS", 300, None);
        for (first, next) in [(clipped.clone(), full.clone()), (full.clone(), clipped)] {
            let mut lines = vec![first];
            merge_recognized_lines(&mut lines, vec![next]);
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0].text, full.text);
            assert_eq!(lines[0].rect, full.rect);
        }
        let mut native_lines = vec![line("PLEASE", 100, None)];
        merge_recognized_lines(&mut native_lines, vec![full]);
        assert_eq!(native_lines[0].text, "PLEASE FOLLOW INSTRUCTIONS");
    }

    #[test]
    fn test_merge_keeps_unique_words_in_overlapping_tile_fragments() {
        let line = |text: &str, x| RecognizedLine {
            text: text.to_string(), rect: ScreenRect::new(x, 20, 400, 20),
            confidence: None, words: vec![],
        };
        let mut lines = vec![line("PLEASE FOLLOW THE", 1000)];
        merge_recognized_lines(&mut lines, vec![line("FOLLOW THE INSTRUCTIONS", 1100)]);
        assert_eq!(lines.len(), 2);
        assert!(lines.iter().any(|line| line.text.contains("PLEASE")));
        assert!(lines.iter().any(|line| line.text.contains("INSTRUCTIONS")));
    }

    #[test]
    fn test_tiles_cover_4k_and_crop_preserves_pixels_and_offsets() {
        let tiles = ocr_tiles(3840, 2160);
        assert!(tiles.len() > 1);
        let mut covered = vec![false; 3840 * 2160];
        for tile in &tiles {
            assert!(tile.width <= 1200 && tile.height <= 1200);
            for y in tile.y..tile.y + tile.height {
                for x in tile.x..tile.x + tile.width { covered[y as usize * 3840 + x as usize] = true; }
            }
        }
        assert!(covered.into_iter().all(|pixel| pixel));
        let frame = CapturedFrame { bgra_bytes: (0..48).collect(), width: 4, height: 3, offset_x: -100, offset_y: 200 };
        let cropped = crop_frame(&frame, ScreenRect::new(1, 1, 2, 2)).unwrap();
        assert_eq!((cropped.offset_x, cropped.offset_y), (-99, 201));
        assert_eq!(cropped.bgra_bytes, [&frame.bgra_bytes[20..28], &frame.bgra_bytes[36..44]].concat());
        assert!(crop_frame(&frame, ScreenRect::new(3, 0, 2, 1)).is_err());
    }

    #[tokio::test]
    async fn test_cancel_guard_runs_when_wait_times_out_or_task_is_aborted() {
        use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
        let cancelled = Arc::new(AtomicBool::new(false));
        let signal = cancelled.clone();
        let task = async move {
            let _guard = CancelOnDrop(move || { signal.store(true, Ordering::SeqCst); });
            std::future::pending::<()>().await;
        };
        assert!(tokio::time::timeout(std::time::Duration::from_millis(5), task).await.is_err());
        assert!(cancelled.load(Ordering::SeqCst));
        cancelled.store(false, Ordering::SeqCst);
        let signal = cancelled.clone();
        let handle = tokio::spawn(async move {
            let _guard = CancelOnDrop(move || { signal.store(true, Ordering::SeqCst); });
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;
        handle.abort();
        let _ = handle.await;
        assert!(cancelled.load(Ordering::SeqCst));
    }

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
            confidence: None,
            text: "This was an adven-".to_string(),
            rect: ScreenRect::new(100, 100, 200, 20),
            words: Vec::new(),
        };
        let line2 = RecognizedLine {
            confidence: None,
            text: "ture of a lifetime.".to_string(),
            rect: ScreenRect::new(100, 124, 210, 20),
            words: Vec::new(),
        };

        let blocks = cluster_lines_into_blocks(vec![line1, line2], None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "This was an adven-ture of a lifetime.");
        assert_eq!(blocks[0].rect, ScreenRect::new(100, 100, 210, 44));
    }

    #[test]
    fn test_cluster_lines_preserves_dashes_and_numeric_ranges() {
        for (first, second, expected) in [
            ("Wait -", "come here", "Wait - come here"),
            ("Damage: 10-", "20 points", "Damage: 10- 20 points"),
            ("Option-", "2 is ready", "Option- 2 is ready"),
            ("Wait--", "come here", "Wait-- come here"),
            ("Это приклю-", "чение.", "Это приклю-чение."),
            ("D-", "Did he find out?!", "D-Did he find out?!"),
            ("well-", "known", "well-known"),
            ("This was an adven\u{00ad}", "ture.", "This was an adventure."),
            ("Это приклю\u{00ad}", "чение.", "Это приключение."),
        ] {
            let lines = [first, second].into_iter().enumerate().map(|(i, text)| {
                RecognizedLine {
                    confidence: None,
                    text: text.to_string(),
                    rect: ScreenRect::new(100, 100 + i as i32 * 24, 200, 20),
                    words: Vec::new(),
                }
            }).collect();
            let blocks = cluster_lines_into_blocks(lines, None);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].text, expected);
        }
    }

    #[test]
    fn test_cluster_distinct_paragraphs() {
        let line1 = RecognizedLine {
            confidence: None,
            text: "Top dialogue box.".to_string(),
            rect: ScreenRect::new(100, 100, 200, 20),
            words: Vec::new(),
        };
        let line2 = RecognizedLine {
            confidence: None,
            text: "Far bottom dialogue box.".to_string(),
            rect: ScreenRect::new(100, 300, 200, 20), // 180px gap
            words: Vec::new(),
        };

        let blocks = cluster_lines_into_blocks(vec![line1, line2], None);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_cluster_handwritten_overlapping_boxes_keeps_complete_utterance() {
        let line = |text: &str, rect| RecognizedLine {
            text: text.into(), rect, confidence: Some(0.99), words: vec![],
        };
        let blocks = cluster_lines_into_blocks(vec![
            line("D-Did he", ScreenRect::new(70, 72, 225, 75)),
            line("find out?!", ScreenRect::new(74, 126, 214, 65)),
        ], None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "D-Did he find out?!");

        let same_row = cluster_lines_into_blocks(vec![
            line("Left label", ScreenRect::new(10, 20, 100, 60)),
            line("Right label", ScreenRect::new(120, 25, 110, 60)),
        ], None);
        assert_eq!(same_row.len(), 2, "same-row labels must remain separate");
        let other_panel = cluster_lines_into_blocks(vec![
            line("D-Did he", ScreenRect::new(70, 72, 225, 75)),
            line("Other bubble", ScreenRect::new(500, 126, 214, 65)),
        ], None);
        assert_eq!(other_panel.len(), 2, "nearby panels must remain separate");
    }

    #[test]
    fn test_cluster_keeps_diagonal_and_barely_overlapping_labels_separate() {
        for (upper_x, lower_x) in [(20, 65), (65, 20), (20, 50), (20, 45), (45, 20)] {
            for (offset_x, offset_y) in [(0, 0), (-300, 500)] {
                let upper = RecognizedLine {
                    text: "YES".into(), confidence: Some(0.99), words: vec![],
                    rect: ScreenRect::new(offset_x + upper_x, offset_y + 20, 30, 20),
                };
                let lower = RecognizedLine {
                    text: "NO".into(), confidence: Some(0.99), words: vec![],
                    rect: ScreenRect::new(offset_x + lower_x, offset_y + 48, 30, 20),
                };
                let pixels = vec![255; 120 * 100 * 4];
                for reversed in [false, true] {
                    let lines = if reversed { vec![lower.clone(), upper.clone()] }
                        else { vec![upper.clone(), lower.clone()] };
                    let blocks = cluster_lines_into_blocks(lines,
                        Some((&pixels, 120, 100, offset_x, offset_y)));
                    assert_eq!(blocks.len(), 2, "labels at {upper_x}/{lower_x} must stay separate");
                    assert_eq!(blocks[0].text, "YES");
                    assert_eq!(blocks[0].rect, upper.rect);
                    assert_eq!(blocks[1].text, "NO");
                    assert_eq!(blocks[1].rect, lower.rect);
                }
            }
        }
    }

    #[test]
    fn test_cluster_matches_the_previous_line_without_bridging_through_block_bounds() {
        let line = |text: &str, x, y, width| RecognizedLine {
            text: text.into(), rect: ScreenRect::new(x, y, width, 20),
            confidence: None, words: vec![],
        };
        let blocks = cluster_lines_into_blocks(vec![
            line("A long dialogue line", 10, 20, 180),
            line("continues", 10, 48, 30),
            line("Other label", 160, 76, 30),
        ], None);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "A long dialogue line continues");
        assert_eq!(blocks[0].lines.len(), 2);
        assert_eq!(blocks[1].text, "Other label");

        // Centred and indented paragraph lines still overlap the narrower line.
        for middle_x in [20, 90, 140] {
            let blocks = cluster_lines_into_blocks(vec![
                line("A long dialogue line", 20, 20, 180),
                line("continues", middle_x, 48, 40),
                line("and finishes here", 30, 76, 160),
            ], None);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].text, "A long dialogue line continues and finishes here");
            assert_eq!(blocks[0].lines.len(), 3);
        }
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
    fn test_clean_manga_bubble_background_skips_dark_screen() {
        let w = 20;
        let h = 20;
        let mut buf = vec![30u8; (w * h * 4) as usize];
        // Put white dialogue text in the center
        let text_idx = ((10 * w + 10) * 4) as usize;
        buf[text_idx] = 255;
        buf[text_idx + 1] = 255;
        buf[text_idx + 2] = 255;

        clean_manga_bubble_background(&mut buf, w as u32, h as u32);

        // Dark background should remain untouched (not turned white 255)
        assert_eq!(buf[0], 30);
        assert_eq!(buf[text_idx], 255);
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
            has_border_between(&bgra, w as u32, h as u32, 0, 0, &bubble_left, &bubble_right);
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
            has_border_between(&bgra, w as u32, h as u32, 0, 0, &bubble_left, &bubble_right);
        assert!(!border_detected);
    }

    #[test]
    fn test_has_border_between_dark_background_no_false_positive() {
        let w = 100;
        let h = 50;
        // Dark background (e.g. dark RPG dialogue box or night scene)
        let bgra = vec![30u8; (w * h * 4) as usize];

        let line_top = ScreenRect::new(10, 5, 80, 15);
        let line_bottom = ScreenRect::new(10, 25, 80, 15);

        let border_detected =
            has_border_between(&bgra, w as u32, h as u32, 0, 0, &line_top, &line_bottom);
        assert!(
            !border_detected,
            "Dialogue lines on dark background should NOT be treated as separated by panel border"
        );
    }

    #[test]
    fn test_has_border_between_respects_non_zero_frame_offset() {
        let w = 100;
        let h = 50;
        let mut bgra = vec![255u8; (w * h * 4) as usize];

        // Draw a dark vertical panel line between local x=45 and x=55
        for y in 0..h {
            for x in 45..55 {
                let idx = ((y * w + x) * 4) as usize;
                bgra[idx] = 20;
                bgra[idx + 1] = 20;
                bgra[idx + 2] = 20;
            }
        }

        let off_x = 500;
        let off_y = 300;
        // Global screen rectangles
        let bubble_left = ScreenRect::new(510, 310, 30, 30);
        let bubble_right = ScreenRect::new(560, 310, 30, 30);

        let border_detected = has_border_between(
            &bgra,
            w as u32,
            h as u32,
            off_x,
            off_y,
            &bubble_left,
            &bubble_right,
        );
        assert!(
            border_detected,
            "Must detect border when rectangles have global coordinates with offset"
        );
    }

    #[test]
    fn test_has_border_between_dark_letters_inside_rect_do_not_skew_bg() {
        let w = 100;
        let h = 60;
        // White bubble background
        let mut bgra = vec![250u8; (w * h * 4) as usize];

        // Line 1: put dense dark letters (0 luma) in its center
        for y in 12..18 {
            for x in 30..50 {
                let idx = ((y * w + x) * 4) as usize;
                bgra[idx] = 10;
                bgra[idx + 1] = 10;
                bgra[idx + 2] = 10;
            }
        }

        let line_top = ScreenRect::new(10, 10, 80, 15);
        let line_bottom = ScreenRect::new(10, 30, 80, 15);

        let border_detected =
            has_border_between(&bgra, w as u32, h as u32, 0, 0, &line_top, &line_bottom);
        assert!(
            !border_detected,
            "Dark letter ink inside line rect must not skew background luma and cause false border detection"
        );
    }

    #[test]
    fn test_is_ocr_garbage_filters_artifacts() {
        assert!(is_ocr_garbage("|| // 1! \\"));
        assert!(is_ocr_garbage("| | / \\"));
        assert!(is_ocr_garbage("~~~~___"));
        assert!(is_ocr_garbage("   "));
        assert!(is_ocr_garbage("! ! ! |"));
        assert!(is_ocr_garbage("I@II#O*$"));
        assert!(is_ocr_garbage("!?!?a"));
        assert!(is_ocr_garbage("-=-a-=-"));
    }

    #[test]
    fn test_is_ocr_garbage_preserves_speech_and_sfx() {
        assert!(!is_ocr_garbage("WHICH ONE?!"));
        assert!(!is_ocr_garbage("KRAAAGH!"));
        assert!(!is_ocr_garbage("100"));
        assert!(!is_ocr_garbage("Hello world"));
        assert!(!is_ocr_garbage("What is your HP?"));
        assert!(!is_ocr_garbage("(No!?)"));
        assert!(!is_ocr_garbage("Ah..."));
        assert!(!is_ocr_garbage("Oh..."));
        assert!(!is_ocr_garbage("No..."));
        assert!(!is_ocr_garbage("Wait..."));
        assert!(!is_ocr_garbage("Go!"));
    }

    #[test]
    fn test_first_person_words_survive_noise_filter_and_line_grouping() {
        for word in ["I", "I!", "I...", "I'll", "I’ll", "Iʼll", "I'm", "I’d", "I've"] {
            assert!(!is_spike_word(word), "valid English word rejected: {word}");
            assert!(!is_ocr_garbage(word), "valid English line rejected: {word}");
            let lines = [word, "protect you."].into_iter().enumerate().map(|(i, text)| {
                RecognizedLine {
                    text: text.into(), confidence: Some(0.99), words: vec![],
                    rect: ScreenRect::new(100, 100 + i as i32 * 24, 150, 20),
                }
            }).collect();
            let blocks = cluster_lines_into_blocks(lines, None);
            assert_eq!(blocks.len(), 1);
            assert_eq!(blocks[0].text, format!("{word} protect you."));
        }
        for noise in ["llll", "IIII", "||", "|I|", "I@II#O*$"] {
            assert!(is_ocr_garbage(noise), "spike noise accepted: {noise}");
        }
        assert!(is_spike_word("11"));
        assert!(!is_ocr_garbage("11"), "pure HUD numbers remain legitimate");
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

        let engine = WinrtOcrEngine::new("en").expect("Failed to initialize WinrtOcrEngine");
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
                confidence: None,
                text: "|| // 1! \\".to_string(),
                rect: ScreenRect::new(0, 0, 10, 10),
                words: vec![],
            },
            RecognizedLine {
                confidence: None,
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
                confidence: None,
                text: "EXCUSE ME SIR".to_string(),
                rect: ScreenRect::new(0, 0, 100, 20),
                words: vec![],
            },
            RecognizedLine {
                confidence: None,
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

        let engine = WinrtOcrEngine::new("en").unwrap();
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

    #[test]
    fn test_upscale_2x_catmull_rom_dimensions_and_solid() {
        let w = 8u32;
        let h = 6u32;
        let solid_pixel = [120u8, 200u8, 50u8, 255u8];
        let mut src = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            src.extend_from_slice(&solid_pixel);
        }

        let (dst, out_w, out_h) = upscale_2x_catmull_rom(&src, w, h);
        assert_eq!(out_w, w * 2);
        assert_eq!(out_h, h * 2);
        assert_eq!(dst.len(), (out_w * out_h * 4) as usize);

        for chunk in dst.as_chunks::<4>().0 {
            assert_eq!(chunk, &solid_pixel);
        }
    }

    #[test]
    fn test_upscale_2x_catmull_rom_differs_from_bilinear_on_sharp_edge() {
        let w = 8u32;
        let h = 8u32;
        let mut src = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let val = if x < 4 { 0u8 } else { 255u8 };
                src[idx] = val;
                src[idx + 1] = val;
                src[idx + 2] = val;
                src[idx + 3] = 255;
            }
        }

        let (dst_catmull, out_w, out_h) = upscale_2x_catmull_rom(&src, w, h);
        assert_eq!(out_w, 16);
        assert_eq!(out_h, 16);

        let img = ImageBuffer::<Rgba<u8>, &[u8]>::from_raw(w, h, &src).unwrap();
        let dst_bilinear =
            image::imageops::resize(&img, out_w, out_h, image::imageops::FilterType::Triangle)
                .into_raw();

        assert_ne!(
            dst_catmull, dst_bilinear,
            "Catmull-Rom must differ from bilinear interpolation on sharp edges"
        );
    }

    #[tokio::test]
    async fn test_winrt_pass_accepts_large_padded_frame() {
        if !WinrtOcrEngine::is_language_available("en") { return; }
        let engine = WinrtOcrEngine::new("en").unwrap();
        let frame = CapturedFrame {
            bgra_bytes: vec![255; 3840 * 100 * 4], width: 3840, height: 100,
            offset_x: 300, offset_y: 400,
        };
        engine.recognize_pass_1(&frame).await.expect("Oversized frame must be resized before OCR");
    }

    #[tokio::test]
    async fn tightly_cropped_english_line_is_read_without_distorting_screen_coordinates() {
        if !WinrtOcrEngine::is_language_available("en") { return; }
        let rgba = image::open("tests/fixtures/ocr/small_subtitle.png").unwrap().to_rgba8();
        let (width, height) = rgba.dimensions();
        let mut bytes = rgba.into_raw();
        for pixel in bytes.chunks_exact_mut(4) { pixel.swap(0, 2); }
        let frame = CapturedFrame { bgra_bytes: bytes, width, height, offset_x: -300, offset_y: 500 };
        // 17px-high crop returned no raw WinRT lines before background padding.
        let crop = crop_frame(&frame, ScreenRect::new(208, 82, 226, 17)).unwrap();
        let engine = WinrtOcrEngine::new("en").unwrap();
        let lines = engine.recognize_raw(&crop).await.unwrap();
        let text = lines.iter().map(|line| line.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(text.contains("Please wait for the next mission"), "Unexpected crop reading: {text}");
        let words: Vec<_> = lines.iter().flat_map(|line| &line.words).collect();
        let word = words.iter().find(|word| word.text.eq_ignore_ascii_case("wait")).unwrap();
        assert!(word.rect.contains(273 - 300, 89 + 500), "Padding shifted the cursor word: {:?}", word.rect);
        assert!(words.iter().all(|word| word.rect.x >= crop.offset_x && word.rect.y >= crop.offset_y));
    }

    #[test]
    fn test_debug_artifacts_are_labelled_and_bounded() {
        let temp_dir = std::env::temp_dir();
        let root = temp_dir.join(format!("screen_translator_dump_test_{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        assert_eq!(root.parent(), Some(temp_dir.as_path()));
        fs::create_dir(&root).unwrap();
        for stamp in 0..20 { fs::create_dir(root.join(stamp.to_string())).unwrap(); }
        fs::create_dir(root.join("keep_me")).unwrap();
        let frame = CapturedFrame {
            bgra_bytes: vec![0, 0, 255, 255], width: 1, height: 1, offset_x: 0, offset_y: 0,
        };
        let first = write_ocr_artifacts(&root, &frame, "WinRT OCR", "result").unwrap();
        assert!(first.join("frame.png").exists());
        assert_eq!(fs::read_to_string(first.join("engine.txt")).unwrap(), "WinRT OCR");
        assert!(!first.join("raw.json").exists() && !first.join("input.png").exists());
        let second = write_ocr_artifacts(&root, &frame, r#"{"lines":[]}"#, "result").unwrap();
        let response: serde_json::Value = serde_json::from_slice(&fs::read(second.join("recognition.json")).unwrap()).unwrap();
        assert!(response["lines"].is_array());
        assert!(!second.join("engine.txt").exists());
        assert!(root.join("keep_me").is_dir());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 21);
        assert!(!root.join("0").exists() && !root.join("1").exists());
        fs::remove_dir_all(&root).unwrap();
    }

}
