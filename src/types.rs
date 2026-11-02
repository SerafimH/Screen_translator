use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::PathBuf;
use tracing::{info, warn};

/// Computes standard Rec. 601 luma (Y) from 8-bit BGR color components.
///
/// Formula: `round(0.299 * R + 0.587 * G + 0.114 * B) as u8`
#[inline(always)]
pub fn luma_from_bgr(b: u8, g: u8, r: u8) -> u8 {
    ((r as u32 * 19595 + g as u32 * 38470 + b as u32 * 7471 + 32768) >> 16) as u8
}

/// Represents a 2D integer rectangle on screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScreenRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl ScreenRect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn from_points(x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        let x = x1.min(x2);
        let y = y1.min(y2);
        let width = (x1 - x2).abs();
        let height = (y1 - y2).abs();
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    pub fn union(&self, other: &ScreenRect) -> ScreenRect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x1 = self.x.min(other.x);
        let y1 = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);
        ScreenRect {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    /// Converts from global screen coordinates to local window coordinates given a virtual origin.
    pub fn to_local(&self, origin_x: i32, origin_y: i32) -> Self {
        Self {
            x: self.x - origin_x,
            y: self.y - origin_y,
            width: self.width,
            height: self.height,
        }
    }
}

/// A word recognized by the OCR engine with its exact bounding box.
#[derive(Debug, Clone)]
pub struct RecognizedWord {
    pub text: String,
    pub rect: ScreenRect,
}

/// A line of text containing multiple words.
#[derive(Debug, Clone)]
pub struct RecognizedLine {
    pub text: String,
    pub rect: ScreenRect,
    pub words: Vec<RecognizedWord>,
}

/// A grouped paragraph / bubble block ready for translation.
#[derive(Debug, Clone)]
pub struct OcrBlock {
    pub text: String,
    pub rect: ScreenRect,
    pub lines: Vec<RecognizedLine>,
}

/// A completed translation block ready to be rendered onto the overlay.
#[derive(Debug, Clone)]
pub struct TranslationBlock {
    pub original_text: String,
    pub translated_text: String,
    pub rect: ScreenRect,
}

/// The detected text target (word and/or sentence) located directly under or near the mouse cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextUnderCursor {
    pub word_text: Option<String>,
    pub word_rect: Option<ScreenRect>,
    pub sentence_text: String,
    pub sentence_rect: ScreenRect,
}

/// Collects line indices that form a coherent vertical speech bubble or paragraph
/// above and below the selected line index:
/// - h = height of the selected line;
/// - candidate line is valid if vertical gap <= 0.8 * h and horizontal overlap >= 30% of the narrower width;
/// - among matching candidates, picks the closest by center;
/// - up to 6 lines in each direction.
/// Returns indices of lines sorted top-to-bottom.
pub fn collect_context_lines(lines: &[RecognizedLine], selected_idx: usize) -> Vec<usize> {
    if selected_idx >= lines.len() {
        return Vec::new();
    }

    let h = lines[selected_idx].rect.height as f32;
    if h <= 0.0 {
        return vec![selected_idx];
    }

    let mut up_indices = Vec::new();
    let mut curr_idx = selected_idx;

    // Идём вверх (не более 6 строк)
    for _ in 0..6 {
        let curr = &lines[curr_idx];
        let curr_cy = curr.rect.y as f32 + curr.rect.height as f32 / 2.0;
        let curr_cx = curr.rect.x as f32 + curr.rect.width as f32 / 2.0;

        let mut best_cand: Option<(usize, f32)> = None;

        for (i, cand) in lines.iter().enumerate() {
            if i == curr_idx || i == selected_idx || up_indices.contains(&i) {
                continue;
            }
            let cand_cy = cand.rect.y as f32 + cand.rect.height as f32 / 2.0;
            // Кандидат должен быть выше текущей строки
            if cand_cy >= curr_cy {
                continue;
            }

            // Вертикальный зазор между кандидатом (сверху) и текущей строкой:
            // gap = curr.y - (cand.y + cand.height)
            let gap = (curr.rect.y - (cand.rect.y + cand.rect.height)) as f32;
            if gap > 0.8 * h {
                continue;
            }

            // Горизонтальное перекрытие >= 30% ширины более узкой из двух строк
            let x1 = cand.rect.x.max(curr.rect.x);
            let x2 = (cand.rect.x + cand.rect.width).min(curr.rect.x + curr.rect.width);
            let h_overlap = (x2 - x1).max(0) as f32;
            let min_w = cand.rect.width.min(curr.rect.width) as f32;
            if min_w <= 0.0 || h_overlap < 0.3 * min_w {
                continue;
            }

            // Ближайший по центру
            let cand_cx = cand.rect.x as f32 + cand.rect.width as f32 / 2.0;
            let dx = cand_cx - curr_cx;
            let dy = cand_cy - curr_cy;
            let dist_sq = dx * dx + dy * dy;

            if let Some((_, best_dist)) = best_cand {
                if dist_sq < best_dist {
                    best_cand = Some((i, dist_sq));
                }
            } else {
                best_cand = Some((i, dist_sq));
            }
        }

        if let Some((cand_i, _)) = best_cand {
            up_indices.push(cand_i);
            curr_idx = cand_i;
        } else {
            break;
        }
    }

    let mut down_indices = Vec::new();
    curr_idx = selected_idx;

    // Идём вниз (не более 6 строк)
    for _ in 0..6 {
        let curr = &lines[curr_idx];
        let curr_cy = curr.rect.y as f32 + curr.rect.height as f32 / 2.0;
        let curr_cx = curr.rect.x as f32 + curr.rect.width as f32 / 2.0;

        let mut best_cand: Option<(usize, f32)> = None;

        for (i, cand) in lines.iter().enumerate() {
            if i == curr_idx
                || i == selected_idx
                || up_indices.contains(&i)
                || down_indices.contains(&i)
            {
                continue;
            }
            let cand_cy = cand.rect.y as f32 + cand.rect.height as f32 / 2.0;
            // Кандидат должен быть ниже текущей строки
            if cand_cy <= curr_cy {
                continue;
            }

            // Вертикальный зазор между текущей строкой и кандидатом (снизу):
            // gap = cand.y - (curr.y + curr.height)
            let gap = (cand.rect.y - (curr.rect.y + curr.rect.height)) as f32;
            if gap > 0.8 * h {
                continue;
            }

            // Горизонтальное перекрытие >= 30% ширины более узкой из двух строк
            let x1 = cand.rect.x.max(curr.rect.x);
            let x2 = (cand.rect.x + cand.rect.width).min(curr.rect.x + curr.rect.width);
            let h_overlap = (x2 - x1).max(0) as f32;
            let min_w = cand.rect.width.min(curr.rect.width) as f32;
            if min_w <= 0.0 || h_overlap < 0.3 * min_w {
                continue;
            }

            // Ближайший по центру
            let cand_cx = cand.rect.x as f32 + cand.rect.width as f32 / 2.0;
            let dx = cand_cx - curr_cx;
            let dy = cand_cy - curr_cy;
            let dist_sq = dx * dx + dy * dy;

            if let Some((_, best_dist)) = best_cand {
                if dist_sq < best_dist {
                    best_cand = Some((i, dist_sq));
                }
            } else {
                best_cand = Some((i, dist_sq));
            }
        }

        if let Some((cand_i, _)) = best_cand {
            down_indices.push(cand_i);
            curr_idx = cand_i;
        } else {
            break;
        }
    }

    let mut all_indices = up_indices;
    all_indices.push(selected_idx);
    all_indices.extend(down_indices);
    all_indices.sort_by_key(|&i| (lines[i].rect.y, lines[i].rect.x));
    all_indices
}

/// Identifies the word or sentence under the cursor given recognized OCR blocks.
pub fn find_text_under_cursor(blocks: &[OcrBlock], px: i32, py: i32) -> Option<TextUnderCursor> {
    if blocks.is_empty() {
        return None;
    }

    let all_lines: Vec<RecognizedLine> = blocks.iter().flat_map(|b| b.lines.clone()).collect();
    if all_lines.is_empty() {
        return None;
    }

    let dist_to_rect = |r: &ScreenRect, x: i32, y: i32| -> i64 {
        let dx = if x < r.x {
            (r.x - x) as i64
        } else if x >= r.x + r.width {
            (x - (r.x + r.width - 1)) as i64
        } else {
            0
        };
        let dy = if y < r.y {
            (r.y - y) as i64
        } else if y >= r.y + r.height {
            (y - (r.y + r.height - 1)) as i64
        } else {
            0
        };
        ((dx * dx + dy * dy) as f64).sqrt().round() as i64
    };

    // Максимальный радиус поиска текста вокруг курсора (в пикселях).
    const MAX_DISTANCE: i64 = 12;

    let candidate_score = |r: &ScreenRect| -> (i64, i64, i64) {
        let d = dist_to_rect(r, px, py);
        let dy_center = (2 * r.y as i64 + r.height as i64 - 2 * py as i64).abs();
        let dx_center = (2 * r.x as i64 + r.width as i64 - 2 * px as i64).abs();
        (d, dy_center, dx_center)
    };

    let mut best_line_match: Option<(usize, (i64, i64, i64))> = None;

    for (idx, line) in all_lines.iter().enumerate() {
        let l_dist = dist_to_rect(&line.rect, px, py);
        if l_dist <= MAX_DISTANCE {
            let l_score = candidate_score(&line.rect);
            if let Some((_, best_score)) = best_line_match {
                if l_score < best_score {
                    best_line_match = Some((idx, l_score));
                }
            } else {
                best_line_match = Some((idx, l_score));
            }
        }
    }

    let (best_idx, _) = best_line_match?;
    let line = &all_lines[best_idx];

    let mut best_word_match: Option<(&RecognizedWord, (i64, i64, i64))> = None;
    for word in &line.words {
        let w_dist = dist_to_rect(&word.rect, px, py);
        if w_dist <= MAX_DISTANCE {
            let w_score = candidate_score(&word.rect);
            if let Some((_, best_score)) = best_word_match {
                if w_score < best_score {
                    best_word_match = Some((word, w_score));
                }
            } else {
                best_word_match = Some((word, w_score));
            }
        }
    }

    let context_indices = collect_context_lines(&all_lines, best_idx);
    let s_text = context_indices
        .iter()
        .map(|&i| all_lines[i].text.trim())
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if s_text.is_empty() {
        return None;
    }

    let s_rect = context_indices
        .iter()
        .fold(line.rect, |acc, &i| acc.union(&all_lines[i].rect));

    let (w_text_opt, w_rect_opt) = if let Some((word, _)) = best_word_match {
        let w_text = word.text.trim();
        if !w_text.is_empty() {
            (Some(w_text.to_string()), Some(word.rect))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Some(TextUnderCursor {
        word_text: w_text_opt,
        word_rect: w_rect_opt,
        sentence_text: s_text,
        sentence_rect: s_rect,
    })
}

/// Status of the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Normal,
    SelectingRoi,
}

/// Supported translation backend providers
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TranslationProvider {
    #[default]
    #[serde(rename = "google")]
    Google,
    #[serde(rename = "deepl")]
    DeepL,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "openai_compatible", alias = "openai")]
    OpenAICompatible,
}

impl TranslationProvider {
    pub fn next(&self) -> Self {
        match self {
            Self::Google => Self::DeepL,
            Self::DeepL => Self::Gemini,
            Self::Gemini => Self::OpenAICompatible,
            Self::OpenAICompatible => Self::Google,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Google => "Google Translate (Бесплатно)",
            Self::DeepL => "DeepL API",
            Self::Gemini => "Gemini 3.8 Flash",
            Self::OpenAICompatible => "OpenAI / Groq",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::DeepL => "deepl",
            Self::Gemini => "gemini",
            Self::OpenAICompatible => "openai_compatible",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        let lower = s.trim().to_lowercase();
        if lower.contains("deepl") {
            Self::DeepL
        } else if lower.contains("gemini") {
            Self::Gemini
        } else if lower.contains("openai") || lower.contains("groq") {
            Self::OpenAICompatible
        } else if lower.contains("google") {
            Self::Google
        } else {
            warn!(
                "Unknown translation provider '{}', falling back to Google",
                s
            );
            Self::Google
        }
    }
}

impl fmt::Display for TranslationProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl fmt::Debug for TranslationProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// OCR Engine mode selection
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OcrMode {
    #[default]
    #[serde(rename = "winrt")]
    WinRt,
    #[serde(rename = "helper")]
    Helper,
    #[serde(rename = "auto")]
    Auto,
}

impl OcrMode {
    pub fn next(&self) -> Self {
        match self {
            Self::WinRt => Self::Helper,
            Self::Helper => Self::Auto,
            Self::Auto => Self::WinRt,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::WinRt => "WinRT OCR (Windows)",
            Self::Helper => "RapidOCR (Помощник)",
            Self::Auto => "Авто (WinRT -> RapidOCR)",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WinRt => "winrt",
            Self::Helper => "helper",
            Self::Auto => "auto",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        let lower = s.trim().to_lowercase();
        if lower.contains("helper") || lower.contains("rapid") || lower.contains("python") {
            Self::Helper
        } else if lower.contains("auto") {
            Self::Auto
        } else if lower.contains("winrt") {
            Self::WinRt
        } else {
            warn!("Unknown OCR engine '{}', falling back to WinRT", s);
            Self::WinRt
        }
    }
}

impl fmt::Display for OcrMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl fmt::Debug for OcrMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Configuration settings for Screen Translator
#[derive(Clone, Serialize, Deserialize)]
#[serde(from = "AppConfigRaw")]
pub struct AppConfig {
    pub provider: TranslationProvider,
    pub ocr_engine: OcrMode,
    pub python_path: String,
    pub helper_timeout_ms: u64,
    pub helper_min_score: f32,
    pub ocr_lang: String,
    pub source_lang: String,
    pub target_lang: String,
    pub context_history_size: usize,
    pub vision_fallback: bool,
    pub deepl_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub openai_model: String,
    pub openai_vision_model: Option<String>,
    pub auto_mode_interval_ms: u64,
    pub font_size: i32,
    pub font_family: String,
    pub overlay_bg_alpha: u8,
}

#[derive(Deserialize)]
#[serde(default)]
struct AppConfigRaw {
    pub provider: TranslationProvider,
    pub ocr_engine: OcrMode,
    pub python_path: String,
    pub helper_timeout_ms: u64,
    pub helper_min_score: f32,
    pub ocr_lang: String,
    pub source_lang: String,
    pub target_lang: String,
    pub context_history_size: usize,
    pub use_vision_ocr: Option<bool>,
    pub vision_fallback: Option<bool>,
    pub deepl_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub openai_model: String,
    pub openai_vision_model: Option<String>,
    pub auto_mode_interval_ms: u64,
    pub font_size: i32,
    pub font_family: String,
    pub overlay_bg_alpha: u8,
}

impl Default for AppConfigRaw {
    fn default() -> Self {
        let def = AppConfig::default();
        Self {
            provider: def.provider,
            ocr_engine: def.ocr_engine,
            python_path: def.python_path,
            helper_timeout_ms: def.helper_timeout_ms,
            helper_min_score: def.helper_min_score,
            ocr_lang: def.ocr_lang,
            source_lang: def.source_lang,
            target_lang: def.target_lang,
            context_history_size: def.context_history_size,
            use_vision_ocr: None,
            vision_fallback: None,
            deepl_api_key: def.deepl_api_key,
            gemini_api_key: def.gemini_api_key,
            openai_api_key: def.openai_api_key,
            openai_base_url: def.openai_base_url,
            openai_model: def.openai_model,
            openai_vision_model: def.openai_vision_model,
            auto_mode_interval_ms: def.auto_mode_interval_ms,
            font_size: def.font_size,
            font_family: def.font_family,
            overlay_bg_alpha: def.overlay_bg_alpha,
        }
    }
}

impl From<AppConfigRaw> for AppConfig {
    fn from(raw: AppConfigRaw) -> Self {
        let vision_fallback =
            resolve_vision_fallback(raw.vision_fallback, raw.use_vision_ocr, false);
        let deepl_api_key = raw.deepl_api_key.and_then(|k| {
            let trimmed = k.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with("your-")
                && trimmed != "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx"
            {
                Some(trimmed.to_string())
            } else {
                None
            }
        });
        let gemini_api_key = raw.gemini_api_key.and_then(|k| {
            let trimmed = k.trim();
            if !trimmed.is_empty() && !trimmed.starts_with("your-") && trimmed != "AIzaSy..." {
                Some(trimmed.to_string())
            } else {
                None
            }
        });
        let openai_api_key = raw.openai_api_key.and_then(|k| {
            let trimmed = k.trim();
            if !trimmed.is_empty() && !trimmed.starts_with("your-") && trimmed != "gsk_..." {
                Some(trimmed.to_string())
            } else {
                None
            }
        });
        Self {
            provider: raw.provider,
            ocr_engine: raw.ocr_engine,
            python_path: raw.python_path,
            helper_timeout_ms: raw.helper_timeout_ms,
            helper_min_score: raw.helper_min_score,
            ocr_lang: raw.ocr_lang,
            source_lang: raw.source_lang,
            target_lang: raw.target_lang,
            context_history_size: raw.context_history_size,
            vision_fallback,
            deepl_api_key,
            gemini_api_key,
            openai_api_key,
            openai_base_url: raw.openai_base_url,
            openai_model: raw.openai_model,
            openai_vision_model: raw.openai_vision_model,
            auto_mode_interval_ms: raw.auto_mode_interval_ms.max(100),
            font_size: raw.font_size,
            font_family: raw.font_family,
            overlay_bg_alpha: raw.overlay_bg_alpha,
        }
    }
}

/// Merges legacy `use_vision_ocr` and canonical `vision_fallback` settings.
/// Canonical `vision_fallback` has priority over legacy `use_vision_ocr`.
/// If both are set and their values conflict, a warning is logged and canonical `vision_fallback` takes precedence.
pub fn resolve_vision_fallback(
    canonical_vision_fallback: Option<bool>,
    legacy_use_vision_ocr: Option<bool>,
    default_value: bool,
) -> bool {
    match (canonical_vision_fallback, legacy_use_vision_ocr) {
        (Some(canonical), Some(legacy)) => {
            if canonical != legacy {
                warn!(
                    "Conflicting config options: canonical 'vision_fallback = {}' conflicts with legacy 'use_vision_ocr = {}'. Using canonical 'vision_fallback'.",
                    canonical, legacy
                );
            }
            canonical
        }
        (Some(canonical), None) => canonical,
        (None, Some(legacy)) => legacy,
        (None, None) => default_value,
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            provider: TranslationProvider::Google,
            ocr_engine: OcrMode::WinRt,
            python_path: "ocr_helper/.venv/Scripts/python.exe".to_string(),
            helper_timeout_ms: 5000,
            helper_min_score: 0.5,
            ocr_lang: "en".to_string(),
            source_lang: "en".to_string(),
            target_lang: "ru".to_string(),
            context_history_size: 5,
            vision_fallback: false,
            deepl_api_key: None,
            gemini_api_key: None,
            openai_api_key: None,
            openai_base_url: "https://api.groq.com/openai/v1".to_string(),
            openai_model: "openai/gpt-oss-120b".to_string(),
            openai_vision_model: None,
            auto_mode_interval_ms: 1200,
            font_size: 16,
            font_family: "Segoe UI".to_string(),
            overlay_bg_alpha: 220,
        }
    }
}

impl fmt::Debug for AppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AppConfig")
            .field("provider", &self.provider)
            .field("ocr_engine", &self.ocr_engine)
            .field("python_path", &self.python_path)
            .field("helper_timeout_ms", &self.helper_timeout_ms)
            .field("helper_min_score", &self.helper_min_score)
            .field("ocr_lang", &self.ocr_lang)
            .field("source_lang", &self.source_lang)
            .field("target_lang", &self.target_lang)
            .field("context_history_size", &self.context_history_size)
            .field("vision_fallback", &self.vision_fallback)
            .field(
                "deepl_api_key",
                &self.deepl_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "gemini_api_key",
                &self.gemini_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field("openai_base_url", &self.openai_base_url)
            .field("openai_model", &self.openai_model)
            .field("openai_vision_model", &self.openai_vision_model)
            .field("auto_mode_interval_ms", &self.auto_mode_interval_ms)
            .field("font_size", &self.font_size)
            .field("font_family", &self.font_family)
            .field("overlay_bg_alpha", &self.overlay_bg_alpha)
            .finish()
    }
}

impl AppConfig {
    /// Loads configuration from `config.toml`, falling back to defaults for missing fields or files.
    pub fn load_or_default() -> Self {
        let mut target_dir = PathBuf::from(".");
        let mut candidates = vec![
            PathBuf::from("config.toml"),
            PathBuf::from("../config.toml"),
        ];
        if let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            target_dir = exe_dir.to_path_buf();
            candidates.push(exe_dir.join("config.toml"));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join("config.toml"));
            }
        }
        for path in &candidates {
            if path.exists()
                && let Ok(content) = fs::read_to_string(path)
            {
                return Self::from_toml_str(&content);
            }
        }
        let def = Self::default();
        if let Ok(toml_str) = toml::to_string_pretty(&def) {
            let write_path = target_dir.join("config.toml");
            if let Err(e) = fs::write(&write_path, toml_str) {
                warn!("Failed to create default {}: {e}", write_path.display());
            } else {
                info!("Created default {}", write_path.display());
            }
        }
        def
    }

    /// Parses a TOML string into `AppConfig`, falling back to defaults on parse error.
    pub fn from_toml_str(content: &str) -> Self {
        match toml::from_str::<AppConfigRaw>(content) {
            Ok(raw) => Self::from(raw),
            Err(e) => {
                warn!("Failed to parse config TOML: {e}, falling back to defaults");
                Self::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_points_all_quadrants() {
        // Drag top-left to bottom-right
        let r1 = ScreenRect::from_points(100, 200, 500, 600);
        assert_eq!(r1, ScreenRect::new(100, 200, 400, 400));

        // Drag bottom-right to top-left (inverted)
        let r2 = ScreenRect::from_points(500, 600, 100, 200);
        assert_eq!(r2, ScreenRect::new(100, 200, 400, 400));

        // Drag top-right to bottom-left
        let r3 = ScreenRect::from_points(500, 200, 100, 600);
        assert_eq!(r3, ScreenRect::new(100, 200, 400, 400));
    }

    #[test]
    fn test_to_local_virtual_screen_conversion() {
        // Multi-monitor setup with secondary monitor to the left at x = -1920
        let origin_x = -1920;
        let origin_y = 0;

        let global_rect = ScreenRect::new(-1000, 150, 400, 300);
        let local_rect = global_rect.to_local(origin_x, origin_y);
        assert_eq!(local_rect, ScreenRect::new(920, 150, 400, 300));
    }

    #[test]
    fn test_contains_and_empty() {
        // rect x=100, y=100, width=200, height=200 → valid pixels: [100,299] x [100,299]
        let r = ScreenRect::new(100, 100, 200, 200);
        assert!(!r.is_empty());
        assert!(r.contains(100, 100)); // top-left corner — inside
        assert!(r.contains(200, 200)); // center — inside
        assert!(r.contains(299, 299)); // last valid pixel — inside
        assert!(!r.contains(300, 100)); // right edge (exclusive) — outside
        assert!(!r.contains(100, 300)); // bottom edge (exclusive) — outside
        assert!(!r.contains(99, 100)); // left of rect — outside
        assert!(!r.contains(301, 200)); // right of rect — outside

        let empty = ScreenRect::new(100, 100, 0, 10);
        assert!(empty.is_empty());
    }

    #[test]
    fn bug_1_contains_excludes_right_and_bottom_edge() {
        // BUG-1 regression: point exactly at x+width or y+height must be OUTSIDE.
        let r = ScreenRect::new(10, 20, 100, 50);
        // Right boundary: x + width = 110 → must NOT be contained
        assert!(
            !r.contains(110, 30),
            "x == x+width должен быть вне прямоугольника"
        );
        // Bottom boundary: y + height = 70 → must NOT be contained
        assert!(
            !r.contains(15, 70),
            "y == y+height должен быть вне прямоугольника"
        );
        // Last valid pixels
        assert!(r.contains(109, 30), "x == x+width-1 должен быть внутри");
        assert!(r.contains(15, 69), "y == y+height-1 должен быть внутри");
        // Corner cases: zero-sized rect contains nothing
        let zero = ScreenRect::new(5, 5, 0, 0);
        assert!(zero.is_empty());
        assert!(!zero.contains(5, 5));
    }

    #[test]
    fn test_union() {
        let r1 = ScreenRect::new(10, 10, 50, 50);
        let r2 = ScreenRect::new(40, 40, 60, 60);
        let u = r1.union(&r2);
        assert_eq!(u, ScreenRect::new(10, 10, 90, 90));
    }

    #[test]
    fn test_app_config_custom_debug_redacts_keys() {
        let cfg = AppConfig {
            gemini_api_key: Some("AIzaSySecRet1234567890".to_string()),
            deepl_api_key: Some("abcdef-1234-5678:fx".to_string()),
            openai_api_key: Some("gsk_topsecrettoken".to_string()),
            ..Default::default()
        };

        let debug_str = format!("{:?}", cfg);
        assert!(!debug_str.contains("AIzaSySecRet1234567890"));
        assert!(!debug_str.contains("abcdef-1234-5678:fx"));
        assert!(!debug_str.contains("gsk_topsecrettoken"));
        assert!(debug_str.contains("[REDACTED]"));
    }

    #[test]
    fn test_med_01_auto_mode_interval_bounds() {
        // config with 0 yields 100
        let toml_zero = "auto_mode_interval_ms = 0\n";
        let cfg_zero = AppConfig::from_toml_str(toml_zero);
        assert_eq!(cfg_zero.auto_mode_interval_ms, 100);

        // empty, negative, and garbage values do not panic and keep valid default >= 100
        let toml_empty = "auto_mode_interval_ms = \n";
        let cfg_empty = AppConfig::from_toml_str(toml_empty);
        assert!(cfg_empty.auto_mode_interval_ms >= 100);

        let toml_neg = "auto_mode_interval_ms = -50\n";
        let cfg_neg = AppConfig::from_toml_str(toml_neg);
        assert!(cfg_neg.auto_mode_interval_ms >= 100);

        let toml_garbage = "auto_mode_interval_ms = not_a_number\n";
        let cfg_garbage = AppConfig::from_toml_str(toml_garbage);
        assert!(cfg_garbage.auto_mode_interval_ms >= 100);
    }

    #[test]
    fn test_ocr_mode_cycle() {
        assert_eq!(OcrMode::WinRt.next(), OcrMode::Helper);
        assert_eq!(OcrMode::Helper.next(), OcrMode::Auto);
        assert_eq!(OcrMode::Auto.next(), OcrMode::WinRt);
    }

    #[test]
    fn test_ocr_helper_config_keys_parsing() {
        let toml = r#"
            ocr_engine = "helper"
            python_path = "custom/python.exe"
            helper_timeout_ms = 8000
            helper_min_score = 0.65
            ocr_lang = "en"
        "#;
        let cfg = AppConfig::from_toml_str(toml);
        assert_eq!(cfg.ocr_engine, OcrMode::Helper);
        assert_eq!(cfg.python_path, "custom/python.exe");
        assert_eq!(cfg.helper_timeout_ms, 8000);
        assert_eq!(cfg.helper_min_score, 0.65);
        assert_eq!(cfg.ocr_lang, "en");
    }

    #[test]
    fn test_resolve_vision_fallback_all_combinations() {
        // 1. Only canonical vision_fallback
        assert!(resolve_vision_fallback(Some(true), None, false));
        assert!(!resolve_vision_fallback(Some(false), None, true));

        // 2. Only legacy use_vision_ocr
        assert!(resolve_vision_fallback(None, Some(true), false));
        assert!(!resolve_vision_fallback(None, Some(false), true));

        // 3. Both specified and identical
        assert!(resolve_vision_fallback(Some(true), Some(true), false));
        assert!(!resolve_vision_fallback(Some(false), Some(false), true));

        // 4. Both specified and conflicting (canonical has priority)
        assert!(resolve_vision_fallback(Some(true), Some(false), false));
        assert!(!resolve_vision_fallback(Some(false), Some(true), true));

        // 5. Neither specified (fallback to default)
        assert!(!resolve_vision_fallback(None, None, false));
        assert!(resolve_vision_fallback(None, None, true));
    }

    #[test]
    fn test_toml_vision_fallback_combinations_and_order_independence() {
        // 1. Only vision_fallback = true
        let cfg = AppConfig::from_toml_str("vision_fallback = true\n");
        assert!(cfg.vision_fallback);

        // 2. Only legacy use_vision_ocr = true
        let cfg = AppConfig::from_toml_str("use_vision_ocr = true\n");
        assert!(cfg.vision_fallback);

        // 3. Both identical (true)
        let cfg = AppConfig::from_toml_str("vision_fallback = true\nuse_vision_ocr = true\n");
        assert!(cfg.vision_fallback);

        // 4. Both conflicting: canonical true, legacy false (order 1)
        let cfg1 = AppConfig::from_toml_str("vision_fallback = true\nuse_vision_ocr = false\n");
        assert!(cfg1.vision_fallback);

        // 5. Both conflicting: canonical true, legacy false (order 2 - legacy first)
        let cfg2 = AppConfig::from_toml_str("use_vision_ocr = false\nvision_fallback = true\n");
        assert!(cfg2.vision_fallback);
        assert_eq!(cfg1.vision_fallback, cfg2.vision_fallback);

        // 6. Both conflicting: canonical false, legacy true (order 1)
        let cfg3 = AppConfig::from_toml_str("vision_fallback = false\nuse_vision_ocr = true\n");
        assert!(!cfg3.vision_fallback);

        // 7. Both conflicting: canonical false, legacy true (order 2)
        let cfg4 = AppConfig::from_toml_str("use_vision_ocr = true\nvision_fallback = false\n");
        assert!(!cfg4.vision_fallback);
        assert_eq!(cfg3.vision_fallback, cfg4.vision_fallback);

        // 8. Neither specified -> default false
        let cfg_none = AppConfig::from_toml_str("");
        assert!(!cfg_none.vision_fallback);

        // Serde JSON deserialization with both fields present (must not fail duplicate field)
        let json_both = r#"{"vision_fallback": true, "use_vision_ocr": false}"#;
        let cfg_serde: AppConfig = serde_json::from_str(json_both).unwrap();
        assert!(cfg_serde.vision_fallback);
    }

    #[test]
    fn test_toml_hash_in_quotes_and_inline_comments() {
        let toml = r#"
            python_path = "C:/env #1/python.exe"
            openai_base_url = "https://api.example.com/v1#fragment" # trailing comment
            auto_mode_interval_ms = 500 # interval comment
        "#;
        let cfg = AppConfig::from_toml_str(toml);
        assert_eq!(cfg.python_path, "C:/env #1/python.exe");
        assert_eq!(cfg.openai_base_url, "https://api.example.com/v1#fragment");
        assert_eq!(cfg.auto_mode_interval_ms, 500);
    }

    #[test]
    fn test_toml_placeholder_keys_filtered_and_valid_preserved() {
        let toml_placeholders = r#"
            deepl_api_key = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx"
            gemini_api_key = "AIzaSy..."
            openai_api_key = "gsk_..."
        "#;
        let cfg_placeholders = AppConfig::from_toml_str(toml_placeholders);
        assert_eq!(cfg_placeholders.deepl_api_key, None);
        assert_eq!(cfg_placeholders.gemini_api_key, None);
        assert_eq!(cfg_placeholders.openai_api_key, None);

        let toml_readme_placeholders = r#"
            deepl_api_key = "your-deepl-key:fx"
            gemini_api_key = "your-gemini-key"
            openai_api_key = "your-openai-or-groq-key"
        "#;
        let cfg_readme = AppConfig::from_toml_str(toml_readme_placeholders);
        assert_eq!(cfg_readme.deepl_api_key, None);
        assert_eq!(cfg_readme.gemini_api_key, None);
        assert_eq!(cfg_readme.openai_api_key, None);

        let toml_valid = r#"
            deepl_api_key = "real-deepl-key:fx"
            gemini_api_key = "AIzaSyValidGeminiKey123"
            openai_api_key = "gsk_valid_openai_token_456"
        "#;
        let cfg_valid = AppConfig::from_toml_str(toml_valid);
        assert_eq!(
            cfg_valid.deepl_api_key.as_deref(),
            Some("real-deepl-key:fx")
        );
        assert_eq!(
            cfg_valid.gemini_api_key.as_deref(),
            Some("AIzaSyValidGeminiKey123")
        );
        assert_eq!(
            cfg_valid.openai_api_key.as_deref(),
            Some("gsk_valid_openai_token_456")
        );

        let toml_trimmed = r#"
            deepl_api_key = "  real-deepl-key:fx  "
            gemini_api_key = "  AIzaSyValidGeminiKey123\t"
            openai_api_key = " gsk_valid_openai_token_456 \n"
        "#;
        let cfg_trimmed = AppConfig::from_toml_str(toml_trimmed);
        assert_eq!(
            cfg_trimmed.deepl_api_key.as_deref(),
            Some("real-deepl-key:fx")
        );
        assert_eq!(
            cfg_trimmed.gemini_api_key.as_deref(),
            Some("AIzaSyValidGeminiKey123")
        );
        assert_eq!(
            cfg_trimmed.openai_api_key.as_deref(),
            Some("gsk_valid_openai_token_456")
        );
    }

    #[test]
    fn test_luma_from_bgr() {
        assert_eq!(luma_from_bgr(0, 0, 0), 0);
        assert_eq!(luma_from_bgr(255, 255, 255), 255);
        // Pure red: 0.299 * 255 = 76.245 -> 76
        assert_eq!(luma_from_bgr(0, 0, 255), 76);
        // Pure green: 0.587 * 255 = 149.685 -> 150
        assert_eq!(luma_from_bgr(0, 255, 0), 150);
        // Pure blue: 0.114 * 255 = 29.07 -> 29
        assert_eq!(luma_from_bgr(255, 0, 0), 29);
    }

    #[test]
    fn test_from_str_loose_known_and_fallback() {
        assert_eq!(
            TranslationProvider::from_str_loose("deepl"),
            TranslationProvider::DeepL
        );
        assert_eq!(
            TranslationProvider::from_str_loose("gemini"),
            TranslationProvider::Gemini
        );
        assert_eq!(
            TranslationProvider::from_str_loose("openai"),
            TranslationProvider::OpenAICompatible
        );
        assert_eq!(
            TranslationProvider::from_str_loose("groq"),
            TranslationProvider::OpenAICompatible
        );
        assert_eq!(
            TranslationProvider::from_str_loose("google"),
            TranslationProvider::Google
        );
        // Unknown fallback to Google with warning
        assert_eq!(
            TranslationProvider::from_str_loose("unknown_provider"),
            TranslationProvider::Google
        );

        assert_eq!(OcrMode::from_str_loose("winrt"), OcrMode::WinRt);
        assert_eq!(OcrMode::from_str_loose("helper"), OcrMode::Helper);
        assert_eq!(OcrMode::from_str_loose("rapid"), OcrMode::Helper);
        assert_eq!(OcrMode::from_str_loose("python"), OcrMode::Helper);
        assert_eq!(OcrMode::from_str_loose("auto"), OcrMode::Auto);
        // Unknown fallback to WinRt with warning
        assert_eq!(OcrMode::from_str_loose("unknown_engine"), OcrMode::WinRt);
    }

    #[test]
    fn test_find_text_under_cursor_word_and_sentence() {
        let block = OcrBlock {
            text: "Hello world".to_string(),
            rect: ScreenRect::new(100, 100, 200, 30),
            lines: vec![RecognizedLine {
                text: "Hello world".to_string(),
                rect: ScreenRect::new(100, 100, 200, 30),
                words: vec![
                    RecognizedWord {
                        text: "Hello".to_string(),
                        rect: ScreenRect::new(100, 100, 90, 30),
                    },
                    RecognizedWord {
                        text: "world".to_string(),
                        rect: ScreenRect::new(200, 100, 100, 30),
                    },
                ],
            }],
        };
        let blocks = vec![block];

        // Cursor inside "Hello"
        let target1 = find_text_under_cursor(&blocks, 120, 115).expect("Target must be found");
        assert_eq!(target1.word_text.as_deref(), Some("Hello"));
        assert_eq!(target1.sentence_text, "Hello world");
        assert_eq!(target1.word_rect, Some(ScreenRect::new(100, 100, 90, 30)));
        assert_eq!(target1.sentence_rect, ScreenRect::new(100, 100, 200, 30));

        // Cursor inside "world"
        let target2 = find_text_under_cursor(&blocks, 250, 115).expect("Target must be found");
        assert_eq!(target2.word_text.as_deref(), Some("world"));
        assert_eq!(target2.sentence_text, "Hello world");

        // Cursor in margin (near line/words within 40px)
        let target_near =
            find_text_under_cursor(&blocks, 100, 90).expect("Near target must be found");
        assert_eq!(target_near.word_text.as_deref(), Some("Hello"));

        // Cursor far outside
        let target_miss = find_text_under_cursor(&blocks, 500, 500);
        assert!(target_miss.is_none());

        // Extreme multi-monitor coordinates (|dx| > 46341) must not overflow i32 or panic
        let target_overflow = find_text_under_cursor(&blocks, 60000, 60000);
        assert!(target_overflow.is_none());
    }

    #[test]
    fn test_find_text_under_cursor_selection_rules() {
        // Case 1: две строки с перекрывающимися рамками и курсор в верхней части — выбирается верхняя;
        // в нижней части — выбирается нижняя.
        let block_overlap = OcrBlock {
            text: "Top line\nBottom line".to_string(),
            rect: ScreenRect::new(100, 100, 200, 50),
            lines: vec![
                RecognizedLine {
                    text: "Top line".to_string(),
                    rect: ScreenRect::new(100, 100, 200, 30), // Y: 100..130, center Y = 115
                    words: vec![
                        RecognizedWord {
                            text: "Top".to_string(),
                            rect: ScreenRect::new(100, 100, 90, 30),
                        },
                        RecognizedWord {
                            text: "line".to_string(),
                            rect: ScreenRect::new(200, 100, 100, 30),
                        },
                    ],
                },
                RecognizedLine {
                    text: "Bottom line".to_string(),
                    rect: ScreenRect::new(100, 120, 200, 30), // Y: 120..150, center Y = 135
                    words: vec![
                        RecognizedWord {
                            text: "Bottom".to_string(),
                            rect: ScreenRect::new(100, 120, 90, 30),
                        },
                        RecognizedWord {
                            text: "line".to_string(),
                            rect: ScreenRect::new(200, 120, 100, 30),
                        },
                    ],
                },
            ],
        };
        // Курсор в области перекрытия строк (120..130), в верхней части (Y = 122).
        // Расстояние до обеих строк = 0. До центра верхней: |115 - 122| = 7. До центра нижней: |135 - 122| = 13.
        let target_top = find_text_under_cursor(&[block_overlap.clone()], 120, 122)
            .expect("Top line must be selected");
        assert_eq!(target_top.word_text.as_deref(), Some("Top"));
        assert_eq!(target_top.sentence_text, "Top line Bottom line");

        // Курсор в области перекрытия, но в нижней части (Y = 128).
        // До центра верхней: 13. До центра нижней: 7.
        let target_bottom = find_text_under_cursor(&[block_overlap], 120, 128)
            .expect("Bottom line must be selected");
        assert_eq!(target_bottom.word_text.as_deref(), Some("Bottom"));
        assert_eq!(target_bottom.sentence_text, "Top line Bottom line");

        let single_word_block = OcrBlock {
            text: "Target".to_string(),
            rect: ScreenRect::new(100, 100, 100, 30),
            lines: vec![RecognizedLine {
                text: "Target".to_string(),
                rect: ScreenRect::new(100, 100, 100, 30),
                words: vec![RecognizedWord {
                    text: "Target".to_string(),
                    rect: ScreenRect::new(100, 100, 100, 30),
                }],
            }],
        };

        // Case 2: курсор в 8 px от слова — выбирается это слово
        let target_8px = find_text_under_cursor(&[single_word_block.clone()], 100, 92)
            .expect("Word must be selected within 8px");
        assert_eq!(target_8px.word_text.as_deref(), Some("Target"));

        // Case 3: курсор в 30 px от всего — None
        let target_30px = find_text_under_cursor(&[single_word_block.clone()], 100, 70);
        assert!(target_30px.is_none(), "Cursor at 30px must return None");

        // Case 4: курсор внутри рамки слова — выбирается оно
        let target_inside = find_text_under_cursor(&[single_word_block], 120, 115)
            .expect("Word must be selected when cursor is inside");
        assert_eq!(target_inside.word_text.as_deref(), Some("Target"));
    }

    #[test]
    fn test_collect_context_lines() {
        let h = 20;
        // 1. Три строки друг под другом (зазор между ними 5px <= 0.8 * 20 = 16)
        let line0 = RecognizedLine {
            text: "Line 1".to_string(),
            rect: ScreenRect::new(100, 100, 200, h),
            words: Vec::new(),
        };
        let line1 = RecognizedLine {
            text: "Line 2".to_string(),
            rect: ScreenRect::new(100, 125, 200, h),
            words: Vec::new(),
        };
        let line2 = RecognizedLine {
            text: "Line 3".to_string(),
            rect: ScreenRect::new(100, 150, 200, h),
            words: Vec::new(),
        };

        let lines_three = vec![line0.clone(), line1.clone(), line2.clone()];
        // Выбираем среднюю строку (index 1)
        let collected = collect_context_lines(&lines_three, 1);
        assert_eq!(collected, vec![0, 1, 2]);

        // 2. Строка далеко справа (не входит)
        let line_far_right = RecognizedLine {
            text: "Far Right".to_string(),
            rect: ScreenRect::new(800, 125, 200, h),
            words: Vec::new(),
        };
        let lines_with_right = vec![line0.clone(), line1.clone(), line_far_right];
        let collected_right = collect_context_lines(&lines_with_right, 1);
        assert_eq!(collected_right, vec![0, 1]); // line_far_right не вошла

        // 3. Строка ниже с зазором 3 * h (не входит)
        let line_far_below = RecognizedLine {
            text: "Far Below".to_string(),
            rect: ScreenRect::new(100, 100 + h + 3 * h, 200, h), // gap = 3 * h = 60 > 16
            words: Vec::new(),
        };
        let lines_with_gap = vec![line0.clone(), line_far_below];
        let collected_gap = collect_context_lines(&lines_with_gap, 0);
        assert_eq!(collected_gap, vec![0]); // line_far_below не вошла

        // 4. Выбранная строка одна (возвращается она сама)
        let collected_single = collect_context_lines(&[line0], 0);
        assert_eq!(collected_single, vec![0]);
    }

    #[test]
    fn test_load_or_default_returns_valid_config() {
        let config = AppConfig::load_or_default();
        assert!(!config.source_lang.is_empty());
        assert!(!config.target_lang.is_empty());
        assert!(config.auto_mode_interval_ms >= 100);
    }

    #[test]
    fn test_app_config_default_roundtrip_toml() {
        let def = AppConfig::default();
        let toml_str = toml::to_string_pretty(&def).expect("Serialization to TOML must succeed");
        let parsed = AppConfig::from_toml_str(&toml_str);
        assert_eq!(parsed.provider, def.provider);
        assert_eq!(parsed.ocr_engine, def.ocr_engine);
        assert_eq!(parsed.ocr_lang, def.ocr_lang);
        assert_eq!(parsed.source_lang, def.source_lang);
        assert_eq!(parsed.target_lang, def.target_lang);
    }
}
