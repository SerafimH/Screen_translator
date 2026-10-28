use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

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
            Self::Gemini => "Gemini 2.5 Flash",
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
        } else {
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

/// Configuration settings for Screen Translator
#[derive(Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub provider: TranslationProvider,
    pub source_lang: String,
    pub target_lang: String,
    pub context_history_size: usize,
    pub use_vision_ocr: bool,
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

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            provider: TranslationProvider::Google,
            source_lang: "en".to_string(),
            target_lang: "ru".to_string(),
            context_history_size: 5,
            use_vision_ocr: true,
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
            .field("source_lang", &self.source_lang)
            .field("target_lang", &self.target_lang)
            .field("context_history_size", &self.context_history_size)
            .field("use_vision_ocr", &self.use_vision_ocr)
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

    /// Parses a simple TOML string into `AppConfig`.
    pub fn from_toml_str(content: &str) -> Self {
        let mut cfg = Self::default();

        for raw_line in content.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Split line at first '='
            let Some((key_part, val_part)) = line.split_once('=') else {
                continue;
            };

            let key = key_part.trim();
            // Strip trailing comments
            let raw_val = match val_part.split_once('#') {
                Some((before_comment, _)) => before_comment.trim(),
                None => val_part.trim(),
            };

            let val_unquoted = raw_val.trim_matches('"').trim_matches('\'').trim();

            match key {
                "provider" => cfg.provider = TranslationProvider::from_str_loose(val_unquoted),
                "source_lang" => cfg.source_lang = val_unquoted.to_string(),
                "target_lang" => cfg.target_lang = val_unquoted.to_string(),
                "context_history_size" => {
                    if let Ok(n) = val_unquoted.parse() {
                        cfg.context_history_size = n;
                    }
                }
                "use_vision_ocr" => {
                    if let Ok(b) = val_unquoted.parse() {
                        cfg.use_vision_ocr = b;
                        cfg.vision_fallback = b;
                    }
                }
                "vision_fallback" => {
                    if let Ok(b) = val_unquoted.parse() {
                        cfg.vision_fallback = b;
                        cfg.use_vision_ocr = b;
                    }
                }
                "deepl_api_key" => {
                    if !val_unquoted.is_empty()
                        && val_unquoted != "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx"
                    {
                        cfg.deepl_api_key = Some(val_unquoted.to_string());
                    }
                }
                "gemini_api_key" => {
                    if !val_unquoted.is_empty() && val_unquoted != "AIzaSy..." {
                        cfg.gemini_api_key = Some(val_unquoted.to_string());
                    }
                }
                "openai_api_key" => {
                    if !val_unquoted.is_empty() && val_unquoted != "gsk_..." {
                        cfg.openai_api_key = Some(val_unquoted.to_string());
                    }
                }
                "openai_base_url" => cfg.openai_base_url = val_unquoted.to_string(),
                "openai_model" => cfg.openai_model = val_unquoted.to_string(),
                "auto_mode_interval_ms" => {
                    if let Ok(n) = val_unquoted.parse::<u64>() {
                        cfg.auto_mode_interval_ms = n.max(100);
                    }
                }
                "font_size" => {
                    if let Ok(n) = val_unquoted.parse() {
                        cfg.font_size = n;
                    }
                }
                "font_family" => cfg.font_family = val_unquoted.to_string(),
                "overlay_bg_alpha" => {
                    if let Ok(n) = val_unquoted.parse() {
                        cfg.overlay_bg_alpha = n;
                    }
                }
                _ => {}
            }
        }

        cfg
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
}
