use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;
use tracing::warn;

/// Computes standard Rec. 601 luma (Y) from 8-bit BGR color components.
///
/// Formula: `round(0.299 * R + 0.587 * G + 0.114 * B) as u8`
#[inline(always)]
pub fn luma_from_bgr(b: u8, g: u8, r: u8) -> u8 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8
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

    /// Converts from local window coordinates to global screen coordinates given a virtual origin.
    pub fn to_global(&self, origin_x: i32, origin_y: i32) -> Self {
        Self {
            x: self.x + origin_x,
            y: self.y + origin_y,
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

/// Identifies the word or sentence under the cursor given recognized OCR blocks.
pub fn find_text_under_cursor(blocks: &[OcrBlock], px: i32, py: i32) -> Option<TextUnderCursor> {
    if blocks.is_empty() {
        return None;
    }

    let dist_to_rect = |r: &ScreenRect, x: i32, y: i32| -> i64 {
        let dx = if x < r.x {
            r.x - x
        } else if x >= r.x + r.width {
            x - (r.x + r.width - 1)
        } else {
            0
        };
        let dy = if y < r.y {
            r.y - y
        } else if y >= r.y + r.height {
            y - (r.y + r.height - 1)
        } else {
            0
        };
        ((dx * dx + dy * dy) as f64).sqrt().round() as i64
    };

    const MAX_DISTANCE: i64 = 40;

    let mut best_word_match: Option<(&RecognizedWord, &RecognizedLine, i64)> = None;
    let mut best_line_match: Option<(&RecognizedLine, i64)> = None;

    for block in blocks {
        for line in &block.lines {
            let l_dist = dist_to_rect(&line.rect, px, py);
            if l_dist <= MAX_DISTANCE {
                if let Some((_, best_d)) = best_line_match {
                    if l_dist < best_d {
                        best_line_match = Some((line, l_dist));
                    }
                } else {
                    best_line_match = Some((line, l_dist));
                }
            }

            for word in &line.words {
                let w_dist = dist_to_rect(&word.rect, px, py);
                if w_dist <= MAX_DISTANCE {
                    if let Some((_, _, best_d)) = best_word_match {
                        if w_dist < best_d {
                            best_word_match = Some((word, line, w_dist));
                        }
                    } else {
                        best_word_match = Some((word, line, w_dist));
                    }
                }
            }
        }
    }

    if let Some((word, line, _)) = best_word_match {
        let w_text = word.text.trim();
        let s_text = line.text.trim();
        Some(TextUnderCursor {
            word_text: if !w_text.is_empty() {
                Some(w_text.to_string())
            } else {
                None
            },
            word_rect: Some(word.rect),
            sentence_text: s_text.to_string(),
            sentence_rect: line.rect,
        })
    } else if let Some((line, _)) = best_line_match {
        let s_text = line.text.trim();
        if s_text.is_empty() {
            return None;
        }
        Some(TextUnderCursor {
            word_text: None,
            word_rect: None,
            sentence_text: s_text.to_string(),
            sentence_rect: line.rect,
        })
    } else {
        None
    }
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
        let deepl_api_key = raw.deepl_api_key.filter(|k| {
            let trimmed = k.trim();
            !trimmed.is_empty() && trimmed != "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx"
        });
        let gemini_api_key = raw.gemini_api_key.filter(|k| {
            let trimmed = k.trim();
            !trimmed.is_empty() && trimmed != "AIzaSy..."
        });
        let openai_api_key = raw.openai_api_key.filter(|k| {
            let trimmed = k.trim();
            !trimmed.is_empty() && trimmed != "gsk_..."
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
            openai_model: "meta-llama/llama-4-scout-17b-16e-instruct".to_string(),
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
        let candidates = ["config.toml", "../config.toml"];
        for path in &candidates {
            if Path::new(path).exists()
                && let Ok(content) = fs::read_to_string(path)
            {
                return Self::from_toml_str(&content);
            }
        }
        Self::default()
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
    fn test_local_global_virtual_screen_conversion() {
        // Multi-monitor setup with secondary monitor to the left at x = -1920
        let origin_x = -1920;
        let origin_y = 0;

        let global_rect = ScreenRect::new(-1000, 150, 400, 300);
        let local_rect = global_rect.to_local(origin_x, origin_y);
        assert_eq!(local_rect, ScreenRect::new(920, 150, 400, 300));

        let restored_global = local_rect.to_global(origin_x, origin_y);
        assert_eq!(restored_global, global_rect);
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
        let target_near = find_text_under_cursor(&blocks, 100, 90).expect("Near target must be found");
        assert_eq!(target_near.word_text.as_deref(), Some("Hello"));

        // Cursor far outside
        let target_miss = find_text_under_cursor(&blocks, 500, 500);
        assert!(target_miss.is_none());
    }
}
