use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tracing::{error, warn};

use crate::capture::CapturedFrame;
use crate::types::{AppConfig, OcrBlock, ScreenRect, TranslationBlock, TranslationProvider};

pub const GEMINI_MODEL: &str = "gemini-3.8-flash";

/// Character Lorebook / Glossary entry for game terms, character names, and statistics.
#[derive(Debug, Clone)]
pub struct GlossaryEntry {
    pub original: String,
    pub replacement: String,
    pub notes: Option<String>,
}

pub struct ContextManager {
    history: Mutex<VecDeque<(String, String)>>, // (Original, Translated)
    glossary: Mutex<HashMap<String, GlossaryEntry>>,
    max_history: usize,
}

impl ContextManager {
    pub fn new(max_history: usize) -> Self {
        let mgr = Self {
            history: Mutex::new(VecDeque::with_capacity(max_history + 1)),
            glossary: Mutex::new(HashMap::new()),
            max_history,
        };

        // Standard gaming, RPG, and visual novel lorebook glossary
        mgr.add_glossary_term("HP", "HP", Some("Health points".to_string()));
        mgr.add_glossary_term("MP", "MP", Some("Mana points".to_string()));
        mgr.add_glossary_term("EXP", "Опыт", Some("Experience points".to_string()));
        mgr.add_glossary_term("Lv", "Ур.", Some("Level".to_string()));
        mgr.add_glossary_term("ATK", "АТК", Some("Attack power".to_string()));
        mgr.add_glossary_term("DEF", "ЗАЩ", Some("Defense power".to_string()));
        mgr.add_glossary_term("NPC", "NPC", Some("Non-player character".to_string()));

        mgr
    }

    pub fn add_glossary_term(&self, original: &str, replacement: &str, notes: Option<String>) {
        let mut g = self.glossary.lock().unwrap_or_else(|e| e.into_inner());
        g.insert(
            original.to_lowercase(),
            GlossaryEntry {
                original: original.to_string(),
                replacement: replacement.to_string(),
                notes,
            },
        );
    }

    pub fn record_translation(&self, original: String, translated: String) {
        if self.max_history == 0 {
            return;
        }
        let mut h = self.history.lock().unwrap_or_else(|e| e.into_inner());
        if h.len() >= self.max_history {
            h.pop_front();
        }
        h.push_back((original, translated));
    }

    /// Pre-processes text before sending it to the translation backend.
    /// Returns the borrowed &str without unnecessary heap allocation.
    pub fn preprocess<'a>(&self, text: &'a str) -> &'a str {
        // shortcut: glossary substitution runs in postprocess; pre-translation substitution deferred
        text
    }

    pub fn postprocess(&self, translated: &str) -> String {
        let g = self.glossary.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = translated.to_string();
        let mut lower_result = result.to_lowercase();

        let mut entries: Vec<&GlossaryEntry> = g.values().collect();
        // Sort descending by original term length so compound terms (e.g. "MAX HP")
        // are replaced before shorter sub-terms (e.g. "HP").
        entries.sort_by_key(|a| std::cmp::Reverse(a.original.len()));

        for entry in entries {
            let lower_key = entry.original.to_lowercase();
            if lower_result.contains(&lower_key)
                && let Ok(re) = regex_simple_replace(&result, &entry.original, &entry.replacement)
            {
                result = re;
                lower_result = result.to_lowercase();
            }
        }

        result
    }

    pub fn get_recent_history(&self) -> Vec<(String, String)> {
        self.history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub fn format_history_context(&self) -> String {
        let h = self.history.lock().unwrap_or_else(|e| e.into_inner());
        h.iter()
            .map(|(orig, _)| orig.as_str())
            .collect::<Vec<_>>()
            .join(". ")
    }
}

/// Matches `find_lower` against `source` starting at byte offset `start_idx` (case-insensitively).
///
/// # Contract
/// - `find_lower` must already be in lowercase.
/// - `start_idx` must be on a valid UTF-8 character boundary in `source`.
/// - Returns `Some(match_len)` with the length of the match in bytes of the original `source` string,
///   or `None` if no prefix of `&source[start_idx..]` case-folds to `find_lower`.
///
/// # Example
/// ```ignore
/// let s = "Hello world";
/// assert_eq!(match_find_at(s, 0, "hello"), Some(5)); // "Hello" is 5 bytes in source
/// assert_eq!(match_find_at(s, 6, "world"), Some(5)); // "world" is 5 bytes in source
/// assert_eq!(match_find_at(s, 0, "world"), None);
/// ```
fn match_find_at(source: &str, start_idx: usize, find_lower: &str) -> Option<usize> {
    let tail = &source[start_idx..];
    let mut lower_buf = String::new();
    for (offset, ch) in tail.char_indices() {
        for lc in ch.to_lowercase() {
            lower_buf.push(lc);
        }
        if lower_buf == find_lower {
            // Check word boundaries for terms starting/ending with alphanumeric characters
            let find_first_alpha = find_lower
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric());
            if find_first_alpha && start_idx > 0
                && let Some(prev_ch) = source[..start_idx].chars().next_back()
                    && prev_ch.is_alphanumeric() {
                        return None;
                    }

            let match_end = start_idx + offset + ch.len_utf8();
            let find_last_alpha = find_lower
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
            if find_last_alpha && match_end < source.len()
                && let Some(next_ch) = source[match_end..].chars().next()
                    && next_ch.is_alphanumeric() {
                        return None;
                    }

            return Some(offset + ch.len_utf8());
        }
        if lower_buf.len() >= find_lower.len() {
            return None;
        }
    }
    None
}

fn regex_simple_replace(source: &str, find: &str, replace: &str) -> Result<String, ()> {
    if find.is_empty() {
        return Ok(source.to_string());
    }

    let find_lower = find.to_lowercase();
    let mut out = String::with_capacity(source.len());
    let mut last_idx = 0;

    let mut char_indices = source.char_indices().peekable();

    while let Some(&(start_idx, _)) = char_indices.peek() {
        if start_idx < last_idx {
            char_indices.next();
            continue;
        }

        if let Some(match_len) = match_find_at(source, start_idx, &find_lower) {
            let match_end = start_idx + match_len;
            out.push_str(&source[last_idx..start_idx]);
            out.push_str(replace);
            last_idx = match_end;
            while let Some(&(idx, _)) = char_indices.peek() {
                if idx < match_end {
                    char_indices.next();
                } else {
                    break;
                }
            }
        } else {
            char_indices.next();
        }
    }

    out.push_str(&source[last_idx..]);
    Ok(out)
}

struct CacheEntry {
    value: String,
    last_access: u64,
}

struct CacheInner {
    map: HashMap<String, CacheEntry>,
    clock: u64,
}

/// Thread-safe in-memory LRU cache for translated phrases.
pub struct TranslationCache {
    inner: Mutex<CacheInner>,
    max_size: usize,
}

impl TranslationCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                map: HashMap::with_capacity(max_size),
                clock: 0,
            }),
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.map.contains_key(key) {
            inner.clock = inner.clock.wrapping_add(1);
            let clock = inner.clock;
            let entry = inner.map.get_mut(key).unwrap();
            entry.last_access = clock;
            Some(entry.value.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, key: String, val: String) {
        if self.max_size == 0 {
            return;
        }

        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.clock = inner.clock.wrapping_add(1);
        let access = inner.clock;

        if let Some(entry) = inner.map.get_mut(&key) {
            entry.value = val;
            entry.last_access = access;
            return;
        }

        while inner.map.len() >= self.max_size {
            if let Some(oldest_key) = inner
                .map
                .iter()
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(k, _)| k.clone())
            {
                inner.map.remove(&oldest_key);
            } else {
                break;
            }
        }

        inner.map.insert(
            key,
            CacheEntry {
                value: val,
                last_access: access,
            },
        );
    }
}

/// Standard Base64 encoder for binary buffers (e.g. PNG images).
pub fn encode_base64(bytes: &[u8]) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARSET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Automatically redacts API keys, bearer tokens, and query parameters from logs and error strings.
pub fn redact_secrets(input: &str) -> String {
    let mut out = input.to_string();

    // Redact ?key=... and &key=...
    let mut search_from = 0;
    while let Some(pos) = out[search_from..].find("key=") {
        let actual_pos = search_from + pos;
        let val_start = actual_pos + 4;
        let val_end = out[val_start..]
            .find(|c: char| c == '&' || c.is_whitespace() || c == '"' || c == '\'')
            .map(|offset| val_start + offset)
            .unwrap_or(out.len());
        out.replace_range(val_start..val_end, "[REDACTED]");
        search_from = val_start + "[REDACTED]".len();
    }

    // Redact Bearer tokens
    search_from = 0;
    while let Some(pos) = out[search_from..].find("Bearer ") {
        let actual_pos = search_from + pos;
        let val_start = actual_pos + 7;
        let val_end = out[val_start..]
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .map(|offset| val_start + offset)
            .unwrap_or(out.len());
        out.replace_range(val_start..val_end, "[REDACTED]");
        search_from = val_start + "[REDACTED]".len();
    }

    // Redact DeepL-Auth-Key
    search_from = 0;
    while let Some(pos) = out[search_from..].find("DeepL-Auth-Key ") {
        let actual_pos = search_from + pos;
        let val_start = actual_pos + 15;
        let val_end = out[val_start..]
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .map(|offset| val_start + offset)
            .unwrap_or(out.len());
        out.replace_range(val_start..val_end, "[REDACTED]");
        search_from = val_start + "[REDACTED]".len();
    }

    // Redact sk-... and gsk_... and AIzaSy... tokens
    let prefixes = ["sk-", "gsk_", "AIzaSy"];
    for prefix in &prefixes {
        search_from = 0;
        while let Some(pos) = out[search_from..].find(prefix) {
            let actual_pos = search_from + pos;
            let val_start = actual_pos;
            let val_end = out[val_start..]
                .find(|c: char| {
                    c.is_whitespace() || c == '"' || c == '\'' || c == ',' || c == ')' || c == '.'
                })
                .map(|offset| val_start + offset)
                .unwrap_or(out.len());
            if val_end > val_start + prefix.len() {
                out.replace_range(val_start..val_end, "[REDACTED]");
                search_from = val_start + "[REDACTED]".len();
            } else {
                search_from = val_start + prefix.len();
            }
        }
    }

    out
}

/// Cleans boundary noise characters and internal OCR letter-number confusions.
pub fn clean_ocr_noise(text: &str) -> String {
    let mut s = text.trim().to_string();

    let noise_chars = ['|', '_', '~', '\\', '/', '[', ']', '{', '}', '`'];
    s = s
        .trim_matches(|c: char| c.is_whitespace() || noise_chars.contains(&c))
        .to_string();

    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    let mut out_chars = Vec::with_capacity(len);

    for i in 0..len {
        let ch = chars[i];
        let prev_alpha = if i > 0 {
            chars[i - 1].is_alphabetic()
        } else {
            false
        };
        let next_alpha = if i + 1 < len {
            chars[i + 1].is_alphabetic()
        } else {
            false
        };

        let prev_non_zero = chars[..i].iter().rposition(|&c| c != '0');
        let next_non_zero = chars[i + 1..]
            .iter()
            .position(|&c| c != '0')
            .map(|p| i + 1 + p);
        let is_letter_o = match (prev_non_zero, next_non_zero) {
            (Some(pi), Some(ni)) => chars[pi].is_alphabetic() && chars[ni].is_alphabetic(),
            _ => false,
        };

        if ch == '0' && is_letter_o {
            let ni = next_non_zero.unwrap();
            let is_upper = chars[ni].is_uppercase();
            out_chars.push(if is_upper { 'O' } else { 'o' });
        } else if ch == '|' && (prev_alpha || next_alpha || i == 0) {
            out_chars.push('I');
        } else if ch == '1' && prev_alpha && next_alpha {
            out_chars.push('l');
        } else {
            out_chars.push(ch);
        }
    }

    let cleaned: String = out_chars.into_iter().collect();

    let mut out = String::new();
    let mut last_was_space = false;
    for c in cleaned.chars() {
        if c.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(c);
            last_was_space = false;
        }
    }

    out.trim().to_string()
}

/// Uses the same text acceptance rules as OCR, including short dialogue with punctuation.
pub fn is_translatable_text(text: &str) -> bool {
    !crate::ocr::is_ocr_garbage(text)
}

/// Checks whether a single token is a valid word for dictionary lookup (only letters, apostrophes, and hyphens).
/// Tokens containing foreign characters, symbols, or numbers (e.g. "•PA", "4EASE", "I@II#O*$") are rejected.
pub fn is_valid_dictionary_word(word: &str) -> bool {
    crate::types::cursor_word_token(word).is_some()
}

/// Converts comic/game ALL CAPS text into readable, natural sentence casing.
pub fn normalize_all_caps(text: &str) -> String {
    let alpha_chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha_chars.is_empty() {
        return text.to_string();
    }

    let upper_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
    let lower_count = alpha_chars.iter().filter(|c| c.is_lowercase()).count();
    let words: Vec<&str> = text.split_whitespace().collect();

    if upper_count >= 4 && lower_count == 0 && words.len() >= 2 {
        let mut result = String::with_capacity(text.len());
        let mut capitalize_next = true;

        for ch in text.chars() {
            if ch.is_alphabetic() {
                if capitalize_next {
                    result.extend(ch.to_uppercase());
                    capitalize_next = false;
                } else {
                    result.extend(ch.to_lowercase());
                }
            } else {
                result.push(ch);
                if ch == '.' || ch == '!' || ch == '?' {
                    capitalize_next = true;
                }
            }
        }

        let mut words_fixed = Vec::new();
        for word in result.split(' ') {
            let lower = word.to_lowercase();
            let core = word
                .trim_matches(|c: char| !c.is_alphabetic())
                .to_lowercase();
            if lower == "i" {
                words_fixed.push("I".to_string());
            } else if let Some(rest) = lower.strip_prefix("i'") {
                words_fixed.push(format!("I'{}", rest));
            } else if matches!(
                core.as_str(),
                "hp" | "mp" | "exp" | "lv" | "atk" | "def" | "npc"
            ) {
                words_fixed.push(word.to_uppercase());
            } else {
                words_fixed.push(word.to_string());
            }
        }

        words_fixed.join(" ")
    } else {
        text.to_string()
    }
}

/// Cleans awkward spaces before Russian punctuation.
pub fn clean_russian_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
        if ch == ' ' {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            if last_space && matches!(ch, ',' | '.' | '!' | '?' | ':' | ';') {
                out.pop();
            }
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
}

/// Strips HTML tags (e.g. <b>, </b>) and unescapes basic HTML entities.
pub fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        if c == '<' {
            in_tag = true;
        } else if c == '>' {
            in_tag = false;
        } else if !in_tag {
            out.push(c);
        }
    }
    out.replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        // Decode ampersands last so escaped entity names remain literal text.
        .replace("&amp;", "&")
}

/// Rich translation details for a standalone word, including dictionary synonyms and usage examples.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WordDetails {
    pub primary: String,
    pub synonyms: Vec<String>,
    pub examples: Vec<String>,
}

/// Parses normalized vision box coordinate array JSON into `TranslationBlock` instances.
pub fn parse_vision_blocks_json_with_box_2d(
    json_str: &str,
    roi: &ScreenRect,
) -> Result<Vec<TranslationBlock>, String> {
    let mut clean_json = json_str.trim();

    // Strip markdown fences
    if clean_json.starts_with("```") {
        if let Some(first_newline) = clean_json.find('\n') {
            clean_json = &clean_json[first_newline + 1..];
        }
        if let Some(last_fence) = clean_json.rfind("```") {
            clean_json = &clean_json[..last_fence];
        }
        clean_json = clean_json.trim();
    }

    // Find bounding JSON array
    let start_idx = clean_json
        .find('[')
        .ok_or_else(|| "No JSON array found in vision response".to_string())?;
    let end_idx = clean_json
        .rfind(']')
        .ok_or_else(|| "No closing bracket found in vision response".to_string())?;
    let array_slice = &clean_json[start_idx..=end_idx];

    let items: Vec<serde_json::Value> = serde_json::from_str(array_slice)
        .map_err(|e| format!("Failed to parse vision blocks JSON: {:?}", e))?;

    let mut blocks = Vec::new();

    for item in items {
        let orig = item
            .get("original")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        let trans = item
            .get("translation")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        if orig.is_empty() && trans.is_empty() {
            continue;
        }

        let box_coords = item
            .get("box_2d")
            .and_then(|v| v.as_array())
            .and_then(|arr| {
                if arr.len() == 4 {
                    let ymin = arr[0].as_f64()?;
                    let xmin = arr[1].as_f64()?;
                    let ymax = arr[2].as_f64()?;
                    let xmax = arr[3].as_f64()?;
                    Some((ymin, xmin, ymax, xmax))
                } else {
                    None
                }
            });

        let rect = if let Some((ymin, xmin, ymax, xmax)) = box_coords {
            let local_x = ((xmin * roi.width as f64) / 1000.0).round() as i32;
            let local_y = ((ymin * roi.height as f64) / 1000.0).round() as i32;
            let local_w = (((xmax - xmin) * roi.width as f64) / 1000.0).round() as i32;
            let local_h = (((ymax - ymin) * roi.height as f64) / 1000.0).round() as i32;

            ScreenRect::new(
                roi.x + local_x,
                roi.y + local_y,
                local_w.max(1),
                local_h.max(1),
            )
        } else {
            *roi
        };

        blocks.push(TranslationBlock {
            original_text: orig.to_string(),
            translated_text: trans.to_string(),
            rect,
        });
    }

    Ok(blocks)
}

/// Converts raw BGRA buffer into PNG format bytes in memory.
pub fn bgra_to_png(bgra: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || bgra.len() < (width * height * 4) as usize {
        return Err("Invalid buffer dimensions for PNG encoding".to_string());
    }

    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for (src, dst) in bgra
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgba.as_chunks_mut::<4>().0.iter_mut())
    {
        dst[0] = src[2]; // R
        dst[1] = src[1]; // G
        dst[2] = src[0]; // B
        dst[3] = src[3]; // A
    }

    let max_dim = width.max(height);
    let (target_w, target_h, final_rgba) = if max_dim > 1920 {
        let scale = 1920.0 / max_dim as f64;
        let new_w = ((width as f64 * scale).round() as u32).max(1);
        let new_h = ((height as f64 * scale).round() as u32).max(1);
        let img_buf = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| "Failed to construct RgbaImage from buffer".to_string())?;
        let resized = image::imageops::resize(
            &img_buf,
            new_w,
            new_h,
            image::imageops::FilterType::Triangle,
        );
        (new_w, new_h, resized.into_raw())
    } else {
        (width, height, rgba)
    };

    let mut png_bytes = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
    image::ImageEncoder::write_image(
        encoder,
        &final_rgba,
        target_w,
        target_h,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| format!("PNG encoding failed: {:?}", e))?;

    Ok(png_bytes)
}

/// Normalizes target language code for DeepL API (e.g. deprecated "EN" -> "EN-US").
pub fn deepl_target_lang(target_lang: &str) -> String {
    let upper = target_lang.to_uppercase();
    if upper == "EN" {
        "EN-US".to_string()
    } else if upper == "PT" {
        "PT-PT".to_string()
    } else {
        upper
    }
}

/// Percent-encodes a string for safe inclusion in URL query components (RFC 3986).
pub fn urlencode(s: &str) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX_CHARS[(b >> 4) as usize] as char);
            out.push(HEX_CHARS[(b & 0x0F) as usize] as char);
        }
    }
    out
}

fn is_groq_url(base_url: &str) -> bool {
    reqwest::Url::parse(base_url).ok().and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| host == "groq.com" || host.ends_with(".groq.com"))
}

/// Groq requires an explicit current vision model, independent of the text model.
pub fn resolve_openai_vision_model<'a>(
    explicit_model: Option<&'a str>,
    base_url: &str,
    text_model: &'a str,
) -> Result<&'a str, String> {
    let explicit_model = explicit_model.map(str::trim).filter(|model| !model.is_empty());
    if is_groq_url(base_url) {
        let model = explicit_model.ok_or_else(||
            "Groq Vision: задайте openai_vision_model в config.toml (отдельную модель с поддержкой изображений)".to_string())?;
        if matches!(model, "llama-3.2-11b-vision-preview" | "llama-3.2-90b-vision-preview") {
            return Err("Groq Vision: указанная модель снята с обслуживания; обновите openai_vision_model".to_string());
        }
        return Ok(model);
    }
    if let Some(explicit) = explicit_model {
        Ok(explicit)
    } else if reqwest::Url::parse(base_url).ok().and_then(|url| url.host_str().map(str::to_owned))
        .is_some_and(|host| host == "api.openai.com") && !text_model.contains("gpt-4") {
        Ok("gpt-4o-mini")
    } else {
        Ok(text_model)
    }
}

/// Parses the JSON response body from DeepL translation API for batch requests.
pub fn parse_deepl_batch_response(body: &str, expected: usize) -> Result<Vec<String>, String> {
    let json: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| redact_secrets(&format!("DeepL response error: {:?}", e)))?;

    if let Some(err) = json.get("message").and_then(|v| v.as_str()) {
        return Err(redact_secrets(&format!("DeepL error: {}", err)));
    }

    let translations = json
        .get("translations")
        .and_then(|t| t.as_array())
        .ok_or_else(|| "Invalid DeepL response structure".to_string())?;

    if expected > 0 && translations.len() != expected {
        return Err(format!(
            "DeepL response count mismatch: expected {}, got {}",
            expected,
            translations.len()
        ));
    }

    let mut results = Vec::with_capacity(translations.len());
    for item in translations {
        if let Some(txt) = item.get("text").and_then(|t| t.as_str()) {
            results.push(txt.to_string());
        } else {
            return Err("Missing text in DeepL translation item".to_string());
        }
    }

    Ok(results)
}

/// Combines cached translations with newly batch-translated strings according to original indices.
pub fn assemble_batch_translations(
    cached_results: &[Option<String>],
    uncached_indices: &[usize],
    batch_translations: &[String],
) -> Result<Vec<String>, String> {
    if uncached_indices.len() != batch_translations.len() {
        return Err(format!(
            "Batch translation count mismatch: expected {}, got {}",
            uncached_indices.len(),
            batch_translations.len()
        ));
    }

    let mut results = vec![String::new(); cached_results.len()];
    for (i, cached) in cached_results.iter().enumerate() {
        if let Some(c) = cached {
            results[i] = c.clone();
        }
    }

    for (&idx, trans) in uncached_indices.iter().zip(batch_translations.iter()) {
        if idx >= results.len() {
            return Err(format!(
                "Index {} out of bounds for result length {}",
                idx,
                results.len()
            ));
        }
        results[idx] = trans.clone();
    }

    Ok(results)
}

#[derive(Default)]
struct RetryState {
    failures: u32,
    until: Option<Instant>,
    permanent: bool,
    message: Option<String>,
}

impl RetryState {
    fn ready(&self) -> bool {
        !self.permanent && self.until.is_none_or(|until| Instant::now() >= until)
    }

    fn display_message(&self, label: &str) -> Option<String> {
        let message = self.message.as_ref()?;
        if self.permanent {
            Some(format!("{label} остановлен: проверьте настройки и нажмите Alt+T. {message}"))
        } else if let Some(remaining) = self.until.and_then(|until| until.checked_duration_since(Instant::now())) {
            Some(format!("{label}: повтор через {} с. {message}", remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0)))
        } else { Some(message.clone()) }
    }

    fn fail(&mut self, message: String, status: Option<u16>, retry_after: Option<Duration>) {
        if self.permanent { return; }
        self.failures = self.failures.saturating_add(1);
        self.permanent = status.is_some_and(|code| (400..500).contains(&code)
            && !matches!(code, 408 | 425 | 429));
        let backoff = Duration::from_secs(5 * (1u64 << self.failures.saturating_sub(1).min(4))).min(Duration::from_secs(60));
        let delay = retry_after.unwrap_or(backoff).max(backoff);
        if let Some(deadline) = Instant::now().checked_add(delay) {
            // Concurrent failures may extend a server's Retry-After, never shorten it.
            self.until = Some(self.until.map_or(deadline, |existing| existing.max(deadline)));
        } else {
            // An unreasonable Retry-After must never turn an overflow into an immediate retry.
            self.permanent = true;
        }
        self.message = Some(message);
    }
}

fn retry_after_delay(value: &str, now: SystemTime) -> Option<Duration> {
    value.trim().parse::<u64>().ok().map(Duration::from_secs)
        .or_else(|| httpdate::parse_http_date(value).ok().map(|date| date.duration_since(now).unwrap_or_default()))
}

pub struct TranslationBatchReport {
    pub blocks: Vec<TranslationBlock>,
    pub errors: Vec<String>,
}

#[derive(Clone, Copy)]
enum RequestRoute { Text, Vision }

#[derive(Clone)]
pub struct TranslatorService {
    client: reqwest::Client,
    context: Arc<ContextManager>,
    cache: Arc<TranslationCache>,
    retry: Arc<Mutex<RetryState>>,
    vision_retry: Arc<Mutex<RetryState>>,
    validated_vision_models: Arc<tokio::sync::Mutex<HashSet<String>>>,
}

impl Default for TranslatorService {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslatorService {
    pub fn new() -> Self {
        Self::with_history_size(20)
    }

    pub fn with_history_size(max_history: usize) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(std::time::Duration::from_millis(6000))
            .build()
            .unwrap_or_default();

        Self {
            client,
            context: Arc::new(ContextManager::new(max_history)),
            cache: Arc::new(TranslationCache::new(500)),
            retry: Arc::new(Mutex::new(RetryState::default())),
            vision_retry: Arc::new(Mutex::new(RetryState::default())),
            validated_vision_models: Arc::new(tokio::sync::Mutex::new(HashSet::new())),
        }
    }

    pub fn context(&self) -> &ContextManager {
        &self.context
    }

    pub fn cache(&self) -> &TranslationCache {
        &self.cache
    }

    pub fn automatic_retry_ready(&self) -> bool {
        self.retry.lock().unwrap_or_else(|e| e.into_inner()).ready()
    }

    pub fn retry_message(&self) -> Option<String> {
        let text = self.retry.lock().unwrap_or_else(|e| e.into_inner()).display_message("Автоповтор");
        text.or_else(|| self.vision_retry.lock().unwrap_or_else(|e| e.into_inner()).display_message("Vision"))
    }

    pub fn reset_retry(&self) {
        *self.retry.lock().unwrap_or_else(|e| e.into_inner()) = RetryState::default();
        *self.vision_retry.lock().unwrap_or_else(|e| e.into_inner()) = RetryState::default();
    }

    fn retry_state(&self, route: RequestRoute) -> &Mutex<RetryState> {
        match route { RequestRoute::Text => &self.retry, RequestRoute::Vision => &self.vision_retry }
    }

    fn check_retry(&self, route: RequestRoute) -> Result<(), String> {
        let state = self.retry_state(route).lock().unwrap_or_else(|e| e.into_inner());
        if state.ready() { Ok(()) } else { Err(state.message.clone().unwrap_or_else(|| "Повторные запросы временно приостановлены".to_string())) }
    }

    fn configuration_error(&self, message: String) -> String {
        let message = redact_secrets(&message);
        self.vision_retry.lock().unwrap_or_else(|e| e.into_inner()).fail(message.clone(), Some(400), None);
        message
    }

    fn translation_error(&self, message: String) -> String {
        self.route_error(message, RequestRoute::Text)
    }

    fn vision_error(&self, message: String) -> String {
        self.route_error(message, RequestRoute::Vision)
    }

    fn route_error(&self, message: String, route: RequestRoute) -> String {
        let message = redact_secrets(&message);
        let mut state = self.retry_state(route).lock().unwrap_or_else(|e| e.into_inner());
        if state.ready() { state.fail(message.clone(), None, None); }
        message
    }

    async fn send_request(&self, request: reqwest::RequestBuilder, service: &str) -> Result<reqwest::Response, String> {
        let route = if service.contains("Vision") || service == "Groq models" { RequestRoute::Vision } else { RequestRoute::Text };
        self.check_retry(route)?;
        let response = request.send().await.map_err(|error| {
            let message = redact_secrets(&format!("{service}: {error}"));
            self.retry_state(route).lock().unwrap_or_else(|e| e.into_inner()).fail(message.clone(),
                if error.is_builder() { Some(400) } else { None }, None);
            message
        })?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = response.headers().get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()).and_then(|value| retry_after_delay(value, SystemTime::now()));
            let body = response.text().await.unwrap_or_default();
            let message = redact_secrets(&format!("{service} HTTP {status}: {body}"));
            self.retry_state(route).lock().unwrap_or_else(|e| e.into_inner()).fail(message.clone(), Some(status.as_u16()), retry_after);
            return Err(message);
        }
        let mut state = self.retry_state(route).lock().unwrap_or_else(|e| e.into_inner());
        // A concurrent success must not erase a new error/cooldown from another request.
        if state.ready() { *state = RetryState::default(); }
        Ok(response)
    }

    fn cache_key(&self, purpose: &str, text: &str, source: &str, target: &str, config: &AppConfig) -> String {
        let (backend, model, contextual) = match config.provider {
            TranslationProvider::DeepL if config.deepl_api_key.is_some() => ("deepl".to_string(), "", true),
            TranslationProvider::Gemini if config.gemini_api_key.is_some() => ("gemini".to_string(), GEMINI_MODEL, true),
            TranslationProvider::OpenAICompatible if config.openai_api_key.is_some() =>
                (config.openai_base_url.trim_end_matches('/').to_string(), config.openai_model.as_str(), true),
            _ => ("google".to_string(), "", false),
        };
        let history = contextual.then(|| self.context.get_recent_history());
        // A JSON tuple keeps text, model, endpoint, and history boundaries unambiguous.
        serde_json::json!([purpose, backend, model, source, target, text, history]).to_string()
    }

    async fn validate_groq_vision_model(&self, base_url: &str, api_key: &str, model: &str) -> Result<(), String> {
        let key = serde_json::json!([base_url.trim_end_matches('/'), model, api_key]).to_string();
        let mut validated = self.validated_vision_models.lock().await;
        if validated.contains(&key) { return Ok(()); }
        let response = self.send_request(self.client.get(format!("{}/models", base_url.trim_end_matches('/')))
            .bearer_auth(api_key), "Groq models").await?;
        let body: serde_json::Value = response.json().await
            .map_err(|e| self.vision_error(format!("Groq models: {e}")))?;
        if !body.get("data").and_then(|data| data.as_array()).is_some_and(|models|
            models.iter().any(|item| item.get("id").and_then(|id| id.as_str()) == Some(model))) {
            return Err(self.configuration_error(format!("Groq Vision: модель {model} отсутствует в списке доступных моделей; обновите openai_vision_model")));
        }
        validated.insert(key);
        Ok(())
    }

    /// Translates raw text using the configured provider.
    pub async fn translate_provider_raw(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        config: &AppConfig,
    ) -> Result<String, String> {
        let clean = text.trim();
        if clean.is_empty() {
            return Ok(String::new());
        }

        match config.provider {
            TranslationProvider::Google => {
                self.translate_google(clean, source_lang, target_lang).await
            }
            TranslationProvider::DeepL => {
                if let Some(key) = &config.deepl_api_key {
                    let ctx = self.context.format_history_context();
                    self.translate_deepl(clean, source_lang, target_lang, key, &ctx)
                        .await
                } else {
                    warn!("DeepL API key not provided, falling back to Google");
                    self.translate_google(clean, source_lang, target_lang).await
                }
            }
            TranslationProvider::Gemini => {
                if let Some(key) = &config.gemini_api_key {
                    let ctx = self.context.format_history_context();
                    self.translate_gemini(clean, source_lang, target_lang, key, &ctx)
                        .await
                } else {
                    warn!("Gemini API key not provided, falling back to Google");
                    self.translate_google(clean, source_lang, target_lang).await
                }
            }
            TranslationProvider::OpenAICompatible => {
                if let Some(key) = &config.openai_api_key {
                    let ctx = self.context.format_history_context();
                    self.translate_openai(
                        clean,
                        source_lang,
                        target_lang,
                        &config.openai_base_url,
                        key,
                        &config.openai_model,
                        &ctx,
                    )
                    .await
                } else {
                    warn!("OpenAI/Groq API key not provided, falling back to Google");
                    self.translate_google(clean, source_lang, target_lang).await
                }
            }
        }
    }

    pub async fn translate_google(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<String, String> {
        let url = format!(
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&dj=1",
            urlencode(source_lang),
            urlencode(target_lang)
        );
        let body = format!("q={}", urlencode(text));

        let request = self.client
            .post(&url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(body);
        let resp = self.send_request(request, "Google").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("JSON parsing error: {:?}", e)))?;

        let mut translated_pieces = Vec::new();
        if let Some(sentences) = json.get("sentences").and_then(|v| v.as_array()) {
            for s in sentences {
                if let Some(piece) = s.get("trans").and_then(|v| v.as_str()) {
                    translated_pieces.push(piece);
                }
            }
        }

        if translated_pieces.is_empty() {
            return Err("No translated text received from Google".to_string());
        }

        Ok(translated_pieces.join(""))
    }

    pub async fn translate_deepl(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        api_key: &str,
        context: &str,
    ) -> Result<String, String> {
        if !is_translatable_text(text) {
            warn!("translate_deepl: skipping OCR garbage: {:?}", text);
            return Ok(text.to_string());
        }
        let mut results = self
            .translate_deepl_batch(&[text], source_lang, target_lang, api_key, context)
            .await?;
        results
            .pop()
            .ok_or_else(|| "Missing translation in DeepL response".to_string())
    }

    pub async fn translate_deepl_batch(
        &self,
        texts: &[&str],
        source_lang: &str,
        target_lang: &str,
        api_key: &str,
        context: &str,
    ) -> Result<Vec<String>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Filter out OCR garbage before calling DeepL API
        let mut translatable_indices = Vec::new();
        let mut translatable_texts = Vec::new();

        for (i, &t) in texts.iter().enumerate() {
            if is_translatable_text(t) {
                translatable_indices.push(i);
                translatable_texts.push(t);
            } else {
                warn!("DeepL batch filter: skipping OCR garbage: {:?}", t);
            }
        }

        if translatable_texts.is_empty() {
            return Ok(texts.iter().map(|s| s.to_string()).collect());
        }

        let endpoint = if api_key.ends_with(":fx") {
            "https://api-free.deepl.com/v2/translate"
        } else {
            "https://api.deepl.com/v2/translate"
        };

        let mut body = serde_json::json!({
            "text": translatable_texts,
            "target_lang": deepl_target_lang(target_lang),
        });

        if !source_lang.is_empty() && source_lang != "auto" {
            body["source_lang"] = serde_json::json!(source_lang.to_uppercase());
        }

        if !context.is_empty() {
            body["context"] = serde_json::json!(context);
        }

        let request = self.client
            .post(endpoint)
            .header("Authorization", format!("DeepL-Auth-Key {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = self.send_request(request, "DeepL").await?;

        let body_str = resp
            .text()
            .await
            .map_err(|e| redact_secrets(&format!("DeepL response error: {:?}", e)))?;

        let translations = parse_deepl_batch_response(&body_str, translatable_texts.len())?;
        let mut result = texts.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        for (k, trans) in translations.into_iter().enumerate() {
            let orig_idx = translatable_indices[k];
            result[orig_idx] = trans;
        }
        Ok(result)
    }

    pub async fn translate_gemini(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        api_key: &str,
        context: &str,
    ) -> Result<String, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            GEMINI_MODEL, api_key
        );

        let mut system_instruction = format!(
            "You are an expert game, manga, and visual novel translator. Translate the text from {} to {}. \
            Preserve emotion, tone, and slang. Output ONLY the translated text without explanations or quotes.",
            source_lang, target_lang
        );
        if !context.is_empty() {
            system_instruction.push_str("\nRecent dialogue context for reference:\n");
            system_instruction.push_str(context);
        }

        let body = serde_json::json!({
            "contents": [{
                "parts": [{ "text": text }]
            }],
            "system_instruction": {
                "parts": [{ "text": system_instruction }]
            }
        });

        let request = self.client
            .post(&url)
            .json(&body);
        let resp = self.send_request(request, "Gemini").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini response error: {:?}", e)))?;

        if let Some(err_msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(redact_secrets(&format!("Gemini API error: {}", err_msg)));
        }

        let candidate = json
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|cnt| cnt.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.get("text"))
            .and_then(|txt| txt.as_str())
            .ok_or_else(|| redact_secrets(&format!("Invalid Gemini structure: {:?}", json)))?;

        Ok(candidate.trim().to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn translate_openai(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        base_url: &str,
        api_key: &str,
        model: &str,
        context: &str,
    ) -> Result<String, String> {
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let mut system_prompt = format!(
            "You are an expert game and manga translator. Translate from {} to {}. Output ONLY translation.",
            source_lang, target_lang
        );
        if !context.is_empty() {
            system_prompt.push_str("\nRecent dialogue context for reference:\n");
            system_prompt.push_str(context);
        }

        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": text }
            ],
            "temperature": 0.3
        });

        let request = self.client
            .post(&endpoint)
            .timeout(std::time::Duration::from_secs(15))
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body);
        let resp = self.send_request(request, "OpenAI").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI response error: {:?}", e)))?;

        if let Some(err_msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(redact_secrets(&format!("OpenAI API error: {}", err_msg)));
        }

        let text_res = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|txt| txt.as_str())
            .ok_or_else(|| redact_secrets(&format!("Invalid OpenAI structure: {:?}", json)))?;

        Ok(text_res.trim().to_string())
    }

    /// Vision OCR Fallback: directly sends PNG frame to multimodal vision model to recognize stylized manga/SFX.
    pub async fn translate_image_direct_vision(
        &self,
        frame: &CapturedFrame,
        config: &AppConfig,
    ) -> Result<Vec<TranslationBlock>, String> {
        // Optional Vision failures must not suspend capture or local OCR/text translation.
        self.check_retry(RequestRoute::Vision)?;
        let png_bytes = bgra_to_png(&frame.bgra_bytes, frame.width, frame.height)?;
        let roi = ScreenRect::new(
            frame.offset_x,
            frame.offset_y,
            frame.width as i32,
            frame.height as i32,
        );

        let result = if config.provider == TranslationProvider::Gemini
            && let Some(key) = &config.gemini_api_key
        {
            self.translate_gemini_vision(&png_bytes, &config.target_lang, key, &roi).await
        } else if config.provider == TranslationProvider::OpenAICompatible
            && let Some(key) = &config.openai_api_key
        {
            self.translate_configured_openai_vision(&png_bytes, config, key, &roi).await
        } else if let Some(key) = &config.gemini_api_key {
            self.translate_gemini_vision(&png_bytes, &config.target_lang, key, &roi).await
        } else if let Some(key) = &config.openai_api_key {
            self.translate_configured_openai_vision(&png_bytes, config, key, &roi).await
        } else {
            Err(self.configuration_error("Vision OCR fallback requires a configured Gemini or OpenAI/Groq API key".to_string()))
        };
        result.map_err(|message| self.vision_error(message))
    }

    async fn translate_configured_openai_vision(
        &self, png: &[u8], config: &AppConfig, key: &str, roi: &ScreenRect,
    ) -> Result<Vec<TranslationBlock>, String> {
        let model = resolve_openai_vision_model(config.openai_vision_model.as_deref(),
            &config.openai_base_url, &config.openai_model)
            .map_err(|message| self.configuration_error(message))?;
        if is_groq_url(&config.openai_base_url) {
            self.validate_groq_vision_model(&config.openai_base_url, key, model).await?;
        }
        self.translate_openai_vision(png, &config.target_lang, &config.openai_base_url, key, model, roi).await
    }

    async fn translate_gemini_vision(
        &self,
        png_bytes: &[u8],
        target_lang: &str,
        api_key: &str,
        roi: &ScreenRect,
    ) -> Result<Vec<TranslationBlock>, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            GEMINI_MODEL, api_key
        );

        let base64_image = encode_base64(png_bytes);

        let prompt = format!(
            "Detect all speech bubbles, dialogue boxes, sound effects (SFX), and stylized game/manga text in this image. \
            Translate each text item to {}. \
            Return ONLY a valid JSON array of objects. Do not add any markdown explanation. Format: \
            [{{\"box_2d\": [ymin, xmin, ymax, xmax], \"original\": \"original text\", \"translation\": \"translated text\"}}]. \
            Coordinates must be normalized integers between 0 and 1000.",
            target_lang
        );

        let body = serde_json::json!({
            "contents": [{
                "parts": [
                    { "text": prompt },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }]
        });

        let request = self.client
            .post(&url)
            .timeout(std::time::Duration::from_secs(15))
            .json(&body);
        let resp = self.send_request(request, "Gemini Vision").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini Vision response error: {:?}", e)))?;

        if let Some(err_msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(redact_secrets(&format!("Gemini Vision API error: {}", err_msg)));
        }

        let raw_text = json
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("content"))
            .and_then(|cnt| cnt.get("parts"))
            .and_then(|p| p.as_array())
            .and_then(|arr| arr.first())
            .and_then(|p| p.get("text"))
            .and_then(|txt| txt.as_str())
            .ok_or_else(|| {
                redact_secrets(&format!("Invalid Gemini Vision response: {:?}", json))
            })?;

        parse_vision_blocks_json_with_box_2d(raw_text, roi)
    }

    async fn translate_openai_vision(
        &self,
        png_bytes: &[u8],
        target_lang: &str,
        base_url: &str,
        api_key: &str,
        model: &str,
        roi: &ScreenRect,
    ) -> Result<Vec<TranslationBlock>, String> {
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let base64_image = encode_base64(png_bytes);

        let prompt = format!(
            "Detect all text bubbles and SFX in this image. Translate to {}. \
            Return ONLY a JSON array with format: [{{\"box_2d\": [ymin, xmin, ymax, xmax], \"original\": \"...\", \"translation\": \"...\"}}]. \
            Coordinates must be normalized 0..1000.",
            target_lang
        );

        let data_url = format!("data:image/png;base64,{}", base64_image);

        let body = serde_json::json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        { "type": "image_url", "image_url": { "url": data_url } }
                    ]
                }
            ],
            "temperature": 0.2
        });

        let request = self.client
            .post(&endpoint)
            .timeout(std::time::Duration::from_secs(15))
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body);
        let resp = self.send_request(request, "OpenAI Vision").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI Vision response error: {:?}", e)))?;

        if let Some(err_msg) = json
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Err(redact_secrets(&format!("OpenAI Vision API error: {}", err_msg)));
        }

        let raw_text = json
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|txt| txt.as_str())
            .ok_or_else(|| {
                redact_secrets(&format!("Invalid OpenAI Vision response: {:?}", json))
            })?;

        parse_vision_blocks_json_with_box_2d(raw_text, roi)
    }

    /// Translates a single text string from source to target language with caching and normalization.
    pub async fn translate_text(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        config: &AppConfig,
    ) -> Result<String, String> {
        if !is_translatable_text(text) {
            warn!("translate_text: skipping OCR garbage: {:?}", text);
            return Ok(text.to_string());
        }
        let cleaned = clean_ocr_noise(text);
        if !is_translatable_text(&cleaned) {
            warn!("translate_text: skipping noisy cleaned text: {:?}", cleaned);
            return Ok(text.to_string());
        }
        let normalized = normalize_all_caps(&cleaned);
        self.translate_normalized_text(&normalized, source_lang, target_lang, config, true)
            .await
    }

    async fn translate_normalized_text(
        &self,
        normalized: &str,
        source_lang: &str,
        target_lang: &str,
        config: &AppConfig,
        record_history: bool,
    ) -> Result<String, String> {
        if normalized.is_empty() {
            return Ok(String::new());
        }

        let cache_key = self.cache_key("translation", normalized, source_lang, target_lang, config);
        if let Some(cached) = self.cache.get(&cache_key) {
            if record_history {
                self.context.record_translation(normalized.to_string(), cached.clone());
            }
            return Ok(cached);
        }

        let preprocessed = self.context.preprocess(normalized);
        let raw = self
            .translate_provider_raw(preprocessed, source_lang, target_lang, config)
            .await.map_err(|message| self.translation_error(message))?;
        let clean_ru = clean_russian_punctuation(&raw);
        let final_text = self.context.postprocess(&clean_ru);

        self.cache.insert(cache_key, final_text.clone());
        if record_history {
            self.context
                .record_translation(normalized.to_string(), final_text.clone());
        }

        Ok(final_text)
    }

    /// Looks up rich translation details for a word, utilizing Google dictionary API when available with caching.
    pub async fn lookup_word_details(
        &self,
        word: &str,
        source_lang: &str,
        target_lang: &str,
        config: &AppConfig,
    ) -> WordDetails {
        if !is_valid_dictionary_word(word) {
            warn!("lookup_word_details: skipping non-dictionary token: {:?}", word);
            return WordDetails {
                primary: word.to_string(),
                synonyms: Vec::new(),
                examples: Vec::new(),
            };
        }

        let w = crate::types::cursor_word_spelling(word).expect("dictionary word was validated");
        let clean = clean_ocr_noise(&w);
        let trimmed_punct = clean.trim().trim_matches(|c: char| !c.is_alphanumeric());
        let normalized = if !trimmed_punct.is_empty() {
            trimmed_punct
        } else {
            clean.trim()
        };
        if normalized.is_empty() {
            return WordDetails::default();
        }

        // English acronyms can differ in meaning from their lowercase spelling: US / us.
        let cache_key = self.cache_key("dictionary", normalized, source_lang, target_lang, config);

        if let Some(cached_json) = self.cache.get(&cache_key)
            && let Ok(details) = serde_json::from_str::<WordDetails>(&cached_json) {
                return details;
            }

        let mut details = if config.provider == TranslationProvider::Google {
            self.lookup_google_dict(normalized, source_lang, target_lang)
                .await
                .unwrap_or_default()
        } else {
            WordDetails::default()
        };

        let mut is_success = !details.primary.is_empty();

        if details.primary.is_empty() {
            match self
                .translate_normalized_text(&normalize_all_caps(normalized), source_lang, target_lang, config, false)
                .await
            {
                Ok(fallback_trans) => {
                    details.primary = fallback_trans;
                    is_success = true;
                }
                Err(e) => {
                    warn!(
                        "Translation fallback failed for word '{}': {}",
                        normalized, e
                    );
                    details.primary = normalized.to_string();
                    is_success = false;
                }
            }
        }

        if is_success && let Ok(serialized) = serde_json::to_string(&details) {
            self.cache.insert(cache_key, serialized);
        }

        details
    }

    /// Queries Google Translate GTX API with dictionary (dt=bd) and examples (dt=ex/dt=md).
    pub async fn lookup_google_dict(
        &self,
        word: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<WordDetails, String> {
        let url = format!(
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&dt=bd&dt=ex&dt=md&dj=1&q={}",
            urlencode(source_lang),
            urlencode(target_lang),
            urlencode(word)
        );

        let request = self.client
            .get(&url);
        let resp = self.send_request(request, "Google Dict").await?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("JSON parsing error: {:?}", e)))?;

        let mut primary = String::new();
        if let Some(sentences) = json.get("sentences").and_then(|v| v.as_array()) {
            for s in sentences {
                if let Some(piece) = s.get("trans").and_then(|v| v.as_str()) {
                    primary.push_str(piece);
                }
            }
        }

        let mut synonyms = Vec::new();
        if let Some(dict) = json.get("dict").and_then(|v| v.as_array()) {
            for group in dict {
                if let Some(terms) = group.get("terms").and_then(|v| v.as_array()) {
                    for t in terms {
                        if let Some(term_str) = t.as_str() {
                            let trimmed = term_str.trim();
                            if !trimmed.is_empty()
                                && trimmed.to_lowercase() != primary.trim().to_lowercase()
                                && !synonyms.contains(&trimmed.to_string())
                            {
                                synonyms.push(trimmed.to_string());
                                if synonyms.len() >= 4 {
                                    break;
                                }
                            }
                        }
                    }
                }
                if synonyms.len() >= 4 {
                    break;
                }
            }
        }

        let mut examples = Vec::new();
        if let Some(definitions) = json.get("definitions").and_then(|v| v.as_array()) {
            for def_group in definitions {
                if let Some(entry) = def_group.get("entry").and_then(|v| v.as_array()) {
                    for e in entry {
                        if let Some(ex) = e.get("example").and_then(|v| v.as_str()) {
                            let clean_ex = strip_html_tags(ex).trim().to_string();
                            if !clean_ex.is_empty() && !examples.contains(&clean_ex) {
                                examples.push(clean_ex);
                                if examples.len() >= 2 {
                                    break;
                                }
                            }
                        }
                    }
                }
                if examples.len() >= 2 {
                    break;
                }
            }
        }

        if examples.len() < 2
            && let Some(ex_obj) = json
                .get("examples")
                .and_then(|v| v.get("example"))
                .and_then(|v| v.as_array())
            {
                for item in ex_obj {
                    if let Some(txt) = item.get("text").and_then(|v| v.as_str()) {
                        let clean_ex = strip_html_tags(txt).trim().to_string();
                        if !clean_ex.is_empty() && !examples.contains(&clean_ex) {
                            examples.push(clean_ex);
                            if examples.len() >= 2 {
                                break;
                            }
                        }
                    }
                }
            }

        if primary.is_empty() {
            return Err("No translation found".to_string());
        }

        Ok(WordDetails {
            primary,
            synonyms,
            examples,
        })
    }

    /// Compatibility entry point; detailed errors are available to the HUD through the report.
    pub async fn translate_blocks(
        &self, blocks: &[OcrBlock], config: &AppConfig,
    ) -> (Vec<TranslationBlock>, bool) {
        let report = self.translate_blocks_report(blocks, config).await;
        (report.blocks, report.errors.is_empty())
    }

    pub async fn translate_blocks_report(
        &self, blocks: &[OcrBlock], config: &AppConfig,
    ) -> TranslationBatchReport {
        if blocks.is_empty() {
            return TranslationBatchReport { blocks: Vec::new(), errors: Vec::new() };
        }
        let cleaned: Vec<String> = blocks.iter().map(|b| normalize_all_caps(&clean_ocr_noise(&b.text))).collect();
        let keys: Vec<String> = cleaned.iter().map(|text|
            self.cache_key("translation", text, &config.source_lang, &config.target_lang, config)).collect();
        let mut values: Vec<Option<String>> = cleaned.iter().zip(&keys).zip(blocks).map(|((text, key), block)| {
            if is_translatable_text(text) { self.cache.get(key) } else { Some(block.text.clone()) }
        }).collect();
        let uncached: Vec<usize> = values.iter().enumerate().filter_map(|(i, value)| value.is_none().then_some(i)).collect();
        let texts: Vec<&str> = uncached.iter().map(|&i| cleaned[i].as_str()).collect();
        let mut errors = Vec::new();
        let batch = if texts.len() > 1 && config.provider == TranslationProvider::Google {
            Some(self.translate_provider_raw(&texts.join("\n[---]\n"), &config.source_lang, &config.target_lang, config)
                .await.map(|value| value.split("[---]").map(str::to_string).collect::<Vec<_>>()))
        } else if texts.len() > 1 && config.provider == TranslationProvider::DeepL
            && let Some(key) = &config.deepl_api_key {
            Some(self.translate_deepl_batch(&texts, &config.source_lang, &config.target_lang, key,
                &self.context.format_history_context()).await)
        } else { None };
        let batch_succeeded = if let Some(Ok(translated)) = batch.as_ref()
            && translated.len() == uncached.len() {
            for (&i, text) in uncached.iter().zip(translated) {
                let value = self.context.postprocess(&clean_russian_punctuation(text.trim()));
                self.cache.insert(keys[i].clone(), value.clone());
                values[i] = Some(value);
            }
            true
        } else { false };
        if !batch_succeeded && !uncached.is_empty() {
            let mut tasks = tokio::task::JoinSet::new();
            let semaphore = Arc::new(tokio::sync::Semaphore::new(3));
            for i in uncached {
                let service = self.clone();
                let text = cleaned[i].clone();
                let cfg = config.clone();
                let semaphore = semaphore.clone();
                tasks.spawn(async move {
                    let _permit = semaphore.acquire().await.ok();
                    (i, service.translate_normalized_text(&text, &cfg.source_lang, &cfg.target_lang, &cfg, false).await)
                });
            }
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok((i, Ok(value))) => values[i] = Some(value),
                    Ok((i, Err(message))) => {
                        // The retry gate returns the original HTTP/network cause; do not hide it behind the source text.
                        let message = self.translation_error(message);
                        error!("Failed to translate block '{}': {}", blocks[i].text, message);
                        errors.push(message);
                    }
                    Err(error) => errors.push(self.translation_error(format!("Translation task failed: {error}"))),
                }
            }
        }
        let mut translated_blocks = Vec::with_capacity(blocks.len());
        for (i, block) in blocks.iter().enumerate() {
            if let Some(value) = &values[i] {
                if is_translatable_text(&cleaned[i]) {
                    // Record actual dialogue order, including cached repetitions.
                    self.context.record_translation(cleaned[i].clone(), value.clone());
                }
            }
            translated_blocks.push(TranslationBlock {
                original_text: block.text.clone(),
                translated_text: values[i].take().unwrap_or_else(|| format!("[Err: {}]", block.text)),
                rect: block.rect,
            });
        }
        TranslationBatchReport { blocks: translated_blocks, errors }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    async fn mock_responses(responses: Vec<(u16, String, String)>) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, headers, body) in responses {
                let (mut stream, _) = tokio::time::timeout(Duration::from_secs(5), listener.accept()).await.unwrap().unwrap();
                let mut request = Vec::new();
                loop {
                    let mut buffer = [0u8; 4096];
                    let read = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buffer)).await.unwrap().unwrap();
                    assert!(read > 0);
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                        let header = String::from_utf8_lossy(&request[..end]);
                        let length: usize = header.lines().filter_map(|line| line.split_once(':'))
                            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                            .map(|(_, value)| value.trim().parse().unwrap()).unwrap_or(0);
                        if request.len() >= end + 4 + length { break; }
                    }
                }
                requests.push(String::from_utf8(request).unwrap());
                let response = format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}", body.len());
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        (address, task)
    }

    fn openai_config(base_url: String) -> AppConfig {
        AppConfig {
            provider: TranslationProvider::OpenAICompatible,
            openai_api_key: Some("dummy-test-key".into()),
            openai_base_url: base_url, openai_model: "mock-text-model".into(),
            ..AppConfig::default()
        }
    }

    fn completion_response(text: &str) -> String {
        serde_json::json!({"choices":[{"message":{"content":text}}]}).to_string()
    }

    #[tokio::test]
    async fn dictionary_preserves_case_sensitive_english_meanings() {
        let (base_url, server) = mock_responses(vec![
            (200, String::new(), completion_response("США")),
            (200, String::new(), completion_response("нас")),
        ]).await;
        let config = openai_config(base_url);
        let service = TranslatorService::new();
        assert_eq!(service.lookup_word_details("US", "en", "ru", &config).await.primary, "США");
        assert_eq!(service.lookup_word_details("us", "en", "ru", &config).await.primary, "нас");
        assert_eq!(service.lookup_word_details("US", "en", "ru", &config).await.primary, "США");
        assert_eq!(server.await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn dictionary_uses_canonical_english_punctuation_without_changing_name_case() {
        for (observed, expected) in [("state‑of‑the‑art", "state-of-the-art"),
            ("OʼNeill", "O'Neill"), ("don‘t", "don't")] {
            let (base_url, server) = mock_responses(vec![
                (200, String::new(), completion_response("meaning")),
            ]).await;
            let config = openai_config(base_url);
            let service = TranslatorService::new();
            assert_eq!(service.lookup_word_details(observed, "en", "ru", &config).await.primary, "meaning");
            assert_eq!(service.lookup_word_details(expected, "en", "ru", &config).await.primary, "meaning");
            let requests = server.await.unwrap();
            assert_eq!(requests.len(), 1);
            let body: serde_json::Value = serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
            assert_eq!(body["messages"][1]["content"], expected);
        }
    }

    #[tokio::test]
    async fn test_contextual_cache_does_not_reuse_a_different_dialogue_context() {
        let (base_url, server) = mock_responses(vec![
            (200, String::new(), completion_response("первый контекст")),
            (200, String::new(), completion_response("другой контекст")),
        ]).await;
        let config = openai_config(base_url);
        let service = TranslatorService::new();
        let first_key = service.cache_key("translation", "Hello", "en", "ru", &config);
        assert_eq!(service.translate_text("Hello", "en", "ru", &config).await.unwrap(), "первый контекст");
        service.context.record_translation("A woman is speaking".into(), "Говорит женщина".into());
        assert_ne!(first_key, service.cache_key("translation", "Hello", "en", "ru", &config));
        assert_eq!(service.translate_text("Hello", "en", "ru", &config).await.unwrap(), "другой контекст");
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].contains("A woman is speaking"));
        let mut other_model = config.clone();
        other_model.openai_model = "different-model".into();
        assert_ne!(service.cache_key("translation", "Hello", "en", "ru", &config),
            service.cache_key("translation", "Hello", "en", "ru", &other_model));
    }

    #[tokio::test]
    async fn test_cached_repeated_dialogue_updates_history() {
        let service = TranslatorService::new();
        let config = AppConfig::default();
        service.cache.insert(service.cache_key("translation", "Hello", "en", "ru", &config), "Привет".into());
        for _ in 0..2 {
            assert_eq!(service.translate_text("Hello", "en", "ru", &config).await.unwrap(), "Привет");
        }
        assert_eq!(service.context.get_recent_history().len(), 2);
        assert!(service.context.get_recent_history().iter().all(|(original, _)| original == "Hello"));
    }

    #[tokio::test]
    async fn test_dictionary_fallback_does_not_enter_dialogue_history() {
        let (base_url, server) = mock_responses(vec![(200, String::new(), completion_response("я"))]).await;
        let config = openai_config(base_url);
        let service = TranslatorService::new();
        assert_eq!(service.lookup_word_details("I", "en", "ru", &config).await.primary, "я");
        assert!(service.context.get_recent_history().is_empty());
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_http_configuration_error_is_reported_and_blocks_retries_until_reset() {
        let (base_url, server) = mock_responses(vec![(401, String::new(), "{\"error\":\"invalid API key\"}".into())]).await;
        let config = openai_config(base_url);
        let service = TranslatorService::new();
        let blocks = vec![OcrBlock { text: "Hello world".into(), rect: ScreenRect::new(0, 0, 100, 20), lines: vec![] }];
        let report = service.translate_blocks_report(&blocks, &config).await;
        assert_eq!(report.errors.len(), 1);
        assert!(report.errors[0].contains("HTTP 401"));
        assert!(!service.automatic_retry_ready());
        assert!(service.retry_message().unwrap().contains("Alt+T"));
        assert_eq!(server.await.unwrap().len(), 1);
        let next = service.translate_blocks_report(&blocks, &config).await;
        assert_eq!(next.errors, report.errors, "A gated retry must preserve the HTTP cause");
        service.reset_retry();
        assert!(service.automatic_retry_ready());
        assert!(service.retry_message().is_none());
    }

    #[tokio::test]
    async fn test_rate_limit_honors_retry_after_and_network_backoff_is_bounded() {
        let (base_url, server) = mock_responses(vec![(429, "Retry-After: 120\r\n".into(), "busy".into())]).await;
        let config = openai_config(base_url);
        let service = TranslatorService::new();
        assert!(service.translate_text("Hello world", "en", "ru", &config).await.unwrap_err().contains("429"));
        assert!(!service.automatic_retry_ready());
        assert!(service.retry_message().unwrap().contains("повтор через"));
        assert!(service.retry.lock().unwrap().until.unwrap().saturating_duration_since(Instant::now()) > Duration::from_secs(119));
        server.await.unwrap();
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        let header = httpdate::fmt_http_date(now + Duration::from_secs(90));
        assert_eq!(retry_after_delay(&header, now), Some(Duration::from_secs(90)));
        assert_eq!(retry_after_delay("invalid", now), None);
        let mut state = RetryState::default();
        for expected in [5, 10, 20, 40, 60, 60] {
            state.fail("network".into(), None, None);
            let remaining = state.until.unwrap().saturating_duration_since(Instant::now());
            assert!(remaining <= Duration::from_secs(expected));
            assert!(remaining > Duration::from_secs(expected - 1));
        }
        state.fail("invalid key".into(), Some(403), None);
        state.fail("network".into(), None, None);
        assert!(state.permanent, "Concurrent transient errors must not erase a configuration failure");
    }

    #[test]
    fn test_concurrent_transient_failures_cannot_shorten_retry_after() {
        let mut state = RetryState::default();
        state.fail("rate limited".into(), Some(429), Some(Duration::from_secs(120)));
        let server_deadline = state.until.unwrap();
        state.fail("unavailable".into(), Some(503), None);
        assert!(state.until.unwrap() >= server_deadline);
        state.fail("network failed".into(), None, None);
        assert!(state.until.unwrap() >= server_deadline);
        assert!(!state.ready());
    }

    #[tokio::test]
    async fn test_groq_model_validation_is_cached_and_missing_models_are_rejected() {
        let (base_url, server) = mock_responses(vec![(200, String::new(), "{\"data\":[{\"id\":\"available-vision\"}]}".into())]).await;
        let service = TranslatorService::new();
        service.validate_groq_vision_model(&base_url, "key", "available-vision").await.unwrap();
        service.validate_groq_vision_model(&base_url, "key", "available-vision").await.unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("GET /models "));
        let (base_url, server) = mock_responses(vec![(200, String::new(), "{\"data\":[]}".into())]).await;
        let service = TranslatorService::new();
        assert!(service.validate_groq_vision_model(&base_url, "key", "missing-vision").await.unwrap_err().contains("missing-vision"));
        assert!(service.automatic_retry_ready());
        assert!(!service.vision_retry.lock().unwrap().ready());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn test_optional_vision_failure_does_not_pause_local_ocr_or_text_translation() {
        let service = TranslatorService::new();
        let mut config = AppConfig {
            provider: TranslationProvider::Google, vision_fallback: true,
            openai_api_key: Some("dummy-groq-key".into()),
            openai_vision_model: None,
            ..AppConfig::default()
        };
        let frame = CapturedFrame { bgra_bytes: vec![255; 16], width: 2, height: 2, offset_x: 0, offset_y: 0 };
        let first = service.translate_image_direct_vision(&frame, &config).await.unwrap_err();
        assert!(first.contains("openai_vision_model"));
        assert!(service.automatic_retry_ready(), "Auto capture and local OCR must continue");
        assert!(!service.vision_retry.lock().unwrap().ready());
        service.cache.insert(service.cache_key("translation", "Readable dialogue", "en", "ru", &config), "Читаемая реплика".into());
        assert_eq!(service.translate_text("Readable dialogue", "en", "ru", &config).await.unwrap(), "Читаемая реплика");

        // Even if another request could reach a server now, the Vision-specific gate suppresses it.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        config.openai_base_url = format!("http://{}", listener.local_addr().unwrap());
        config.openai_vision_model = Some("mock-vision".into());
        assert_eq!(service.translate_image_direct_vision(&frame, &config).await.unwrap_err(), first);
        assert!(tokio::time::timeout(Duration::from_millis(30), listener.accept()).await.is_err());

        let (base_url, server) = mock_responses(vec![(200, String::new(), completion_response("Текстовый сервис работает"))]).await;
        let text_config = openai_config(base_url);
        assert_eq!(service.translate_text("Fresh dialogue", "en", "ru", &text_config).await.unwrap(), "Текстовый сервис работает");
        assert_eq!(server.await.unwrap().len(), 1);
        assert!(!service.vision_retry.lock().unwrap().ready(), "Text success must not erase Vision recovery state");
        service.reset_retry();
        assert!(service.automatic_retry_ready());
        assert!(service.vision_retry.lock().unwrap().ready());
    }

    #[test]
    fn test_clean_ocr_noise() {
        let dirty = " | _ The ancient sword ~ | ";
        assert_eq!(clean_ocr_noise(dirty), "The ancient sword");

        let ocr_errors = "TH|S W0RLD";
        assert_eq!(clean_ocr_noise(ocr_errors), "THIS WORLD");

        let lower_errors = "c0me p1ayer";
        assert_eq!(clean_ocr_noise(lower_errors), "come player");

        // Game stats: digits next to letters must NOT convert '0' into 'O'/'o'
        assert_eq!(clean_ocr_noise("100HP 50G 10s 50m"), "100HP 50G 10s 50m");

        // Consecutive zeros in words must convert to 'O'/'o'
        assert_eq!(clean_ocr_noise("B00M!"), "BOOM!");
        assert_eq!(clean_ocr_noise("f00l"), "fool");
        assert_eq!(clean_ocr_noise("S00N"), "SOON");

        // Title-case words with scanned zeros must become lowercase inside
        assert_eq!(clean_ocr_noise("S00n"), "Soon");
        assert_eq!(clean_ocr_noise("C0me"), "Come");
    }

    #[test]
    fn test_normalize_all_caps() {
        let caps = "THE ANCIENT SWORD OF LIGHT! CAN YOU FIND IT? I MUST GO.";
        let normal = normalize_all_caps(caps);
        assert_eq!(
            normal,
            "The ancient sword of light! Can you find it? I must go."
        );

        // Keep mixed case unchanged
        let mixed = "Already Normal Sentence.";
        assert_eq!(normalize_all_caps(mixed), "Already Normal Sentence.");

        // Stat abbreviations with trailing or wrapping punctuation
        let stats = "YOUR HP: 100! (MP) RESTORED. CURRENT LV. 5 EXP +50.";
        assert_eq!(
            normalize_all_caps(stats),
            "Your HP: 100! (MP) restored. Current LV. 5 EXP +50."
        );
    }

    #[test]
    fn test_clean_russian_punctuation() {
        let messy = "Привет , как дела ? Это отлично !";
        assert_eq!(
            clean_russian_punctuation(messy),
            "Привет, как дела? Это отлично!"
        );
    }

    #[test]
    fn test_clean_russian_punctuation_multiple_spaces() {
        assert_eq!(
            clean_russian_punctuation("  Привет   ,  мир  !  Как дела   ?  Хорошо  .  Да  :  точно  ;  "),
            "Привет, мир! Как дела? Хорошо. Да: точно;"
        );
        assert_eq!(clean_russian_punctuation("Привет, мир!"), "Привет, мир!");
        assert_eq!(clean_russian_punctuation("   "), "");
    }

    #[test]
    fn test_translation_cache() {
        let cache = TranslationCache::new(5);
        cache.insert("key1".to_string(), "val1".to_string());
        assert_eq!(cache.get("key1"), Some("val1".to_string()));
        assert_eq!(cache.get("key2"), None);
    }

    #[test]
    fn test_translation_cache_lru_eviction() {
        let cache = TranslationCache::new(3);
        cache.insert("k1".to_string(), "v1".to_string());
        cache.insert("k2".to_string(), "v2".to_string());
        cache.insert("k3".to_string(), "v3".to_string());

        // Access k1 so it becomes more recently used than k2 and k3
        assert_eq!(cache.get("k1"), Some("v1".to_string()));

        // Insert k4: should evict k2 (least recently used)
        cache.insert("k4".to_string(), "v4".to_string());

        assert_eq!(cache.get("k2"), None, "k2 should have been evicted by LRU");
        assert_eq!(
            cache.get("k1"),
            Some("v1".to_string()),
            "k1 was protected by get"
        );
        assert_eq!(cache.get("k3"), Some("v3".to_string()));
        assert_eq!(cache.get("k4"), Some("v4".to_string()));

        // Now access k3
        assert_eq!(cache.get("k3"), Some("v3".to_string()));

        // At this point:
        // k1 last_access = 4
        // k4 last_access = 5
        // k3 last_access = 6
        // Inserting k5 should evict k1 (least recently used among k1, k4, k3)
        cache.insert("k5".to_string(), "v5".to_string());

        assert_eq!(cache.get("k1"), None, "k1 should have been evicted");
        assert_eq!(cache.get("k3"), Some("v3".to_string()));
        assert_eq!(cache.get("k4"), Some("v4".to_string()));
        assert_eq!(cache.get("k5"), Some("v5".to_string()));
    }

    #[test]
    fn test_redact_secrets_in_urls_and_headers() {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}?key=AIzaSyD_SecretKey1234567890&alt=json",
            GEMINI_MODEL
        );
        let redacted_url = redact_secrets(&url);
        assert!(!redacted_url.contains("AIzaSyD_SecretKey1234567890"));
        assert!(redacted_url.contains("key=[REDACTED]"));

        let header = "Authorization: Bearer sk-proj-1234567890abcdef";
        let redacted_header = redact_secrets(header);
        assert!(!redacted_header.contains("sk-proj-1234567890abcdef"));
        assert!(redacted_header.contains("Bearer [REDACTED]"));

        let deepl_header = "Authorization: DeepL-Auth-Key 12345678-abcd-1234:fx";
        let redacted_deepl = redact_secrets(deepl_header);
        assert!(!redacted_deepl.contains("12345678-abcd-1234:fx"));
        assert!(redacted_deepl.contains("DeepL-Auth-Key [REDACTED]"));
    }

    #[test]
    fn test_redact_secrets_various_api_tokens() {
        let text = "Error connecting with token gsk_9876543210zyxwvutsrq and key AIzaSyTestKey.";
        let redacted = redact_secrets(text);
        assert!(!redacted.contains("gsk_9876543210zyxwvutsrq"));
        assert!(!redacted.contains("AIzaSyTestKey"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn test_parse_vision_blocks_json_with_box_2d() {
        let json = r#"[
            {
                "box_2d": [100, 200, 300, 600],
                "original": "KRAAAGH!",
                "translation": "КРАААГХ!"
            }
        ]"#;

        let roi = ScreenRect::new(500, 300, 1000, 500);
        let blocks = parse_vision_blocks_json_with_box_2d(json, &roi).unwrap();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].original_text, "KRAAAGH!");
        assert_eq!(blocks[0].translated_text, "КРАААГХ!");

        // local_x = (200 * 1000) / 1000 = 200 -> screen_x = 500 + 200 = 700
        // local_y = (100 * 500) / 1000 = 50 -> screen_y = 300 + 50 = 350
        // local_w = ((600 - 200) * 1000) / 1000 = 400
        // local_h = ((300 - 100) * 500) / 1000 = 100
        assert_eq!(blocks[0].rect.x, 700);
        assert_eq!(blocks[0].rect.y, 350);
        assert_eq!(blocks[0].rect.width, 400);
        assert_eq!(blocks[0].rect.height, 100);
    }

    #[test]
    fn test_parse_vision_blocks_json_with_markdown_fences() {
        let json_with_fences = r#"```json
        [
            {
                "box_2d": [0, 0, 500, 500],
                "original": "WHICH ONE?!",
                "translation": "КОТОРЫЙ ИЗ НИХ?!"
            }
        ]
        ```"#;

        let roi = ScreenRect::new(0, 0, 800, 600);
        let blocks = parse_vision_blocks_json_with_box_2d(json_with_fences, &roi).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].original_text, "WHICH ONE?!");
        assert_eq!(blocks[0].translated_text, "КОТОРЫЙ ИЗ НИХ?!");
        assert_eq!(blocks[0].rect.width, 400);
        assert_eq!(blocks[0].rect.height, 300);
    }

    #[test]
    fn test_translation_provider_cycle() {
        let p = TranslationProvider::Google;
        let p2 = p.next();
        assert_eq!(p2, TranslationProvider::DeepL);
        let p3 = p2.next();
        assert_eq!(p3, TranslationProvider::Gemini);
        let p4 = p3.next();
        assert_eq!(p4, TranslationProvider::OpenAICompatible);
        let p5 = p4.next();
        assert_eq!(p5, TranslationProvider::Google);
    }

    #[test]
    fn bug_6_regex_simple_replace_utf8_safety() {
        // German capital sharp S "ẞ" (3 bytes) becomes lowercase "ß" (2 bytes)
        let res = regex_simple_replace("ẞ AB", "AB", "CD");
        assert_eq!(res.unwrap(), "ẞ CD");

        // Turkish "İ" (2 bytes in UTF-8, lowercase 'i' is 1 byte)
        let res2 = regex_simple_replace("İ test", "test", "passed");
        assert_eq!(res2.unwrap(), "İ passed");

        // Russian, Japanese, emoji
        let res3 = regex_simple_replace("Привет МИР 🌍!", "мир", "Вселенная");
        assert_eq!(res3.unwrap(), "Привет Вселенная 🌍!");

        let res4 = regex_simple_replace("ゲーム OVER", "over", "終了");
        assert_eq!(res4.unwrap(), "ゲーム 終了");

        // Word boundary checks: substring inside unrelated word must NOT match
        assert_eq!(
            regex_simple_replace("Champion Simple Defeat Default Silver", "MP", "XX").unwrap(),
            "Champion Simple Defeat Default Silver"
        );
        assert_eq!(
            regex_simple_replace("Champion Simple Defeat Default Silver", "DEF", "ЗАЩ").unwrap(),
            "Champion Simple Defeat Default Silver"
        );
        assert_eq!(
            regex_simple_replace("Champion Simple Defeat Default Silver", "Lv", "Ур.").unwrap(),
            "Champion Simple Defeat Default Silver"
        );
        assert_eq!(
            regex_simple_replace("100 MP and Simple 50 mp!", "mp", "Мана").unwrap(),
            "100 Мана and Simple 50 Мана!"
        );

        // Edge cases: empty strings, repeated matches, case insensitive
        assert_eq!(regex_simple_replace("", "test", "replacement").unwrap(), "");
        assert_eq!(
            regex_simple_replace("hello", "", "replacement").unwrap(),
            "hello"
        );
        assert_eq!(
            regex_simple_replace("HP and hp and Hp", "hp", "Очки Жизни").unwrap(),
            "Очки Жизни and Очки Жизни and Очки Жизни"
        );
        assert_eq!(regex_simple_replace("a", "A", "b").unwrap(), "b");
        assert_eq!(
            regex_simple_replace("hello", "xyz", "123").unwrap(),
            "hello"
        );
        assert_eq!(
            regex_simple_replace("❤️ and 💙", "❤️", "💖").unwrap(),
            "💖 and 💙"
        );
    }

    #[test]
    fn test_game_lorebook_glossary_replacement() {
        let ctx = ContextManager::new(5);
        let translated = "Ваш герой восстановил 50 hp и получил 100 exp!";
        let postprocessed = ctx.postprocess(translated);
        assert!(postprocessed.contains("HP"));
        assert!(postprocessed.contains("Опыт"));

        // Compound term must precede shorter sub-term
        ctx.add_glossary_term("MAX HP", "Максимальное здоровье", None);
        let compound = ctx.postprocess("Текущие HP и MAX HP героя");
        assert_eq!(compound, "Текущие HP и Максимальное здоровье героя");
    }

    #[test]
    fn test_context_manager_record_translation_history() {
        let ctx = ContextManager::new(2);
        ctx.record_translation("one".to_string(), "один".to_string());
        ctx.record_translation("two".to_string(), "два".to_string());
        assert_eq!(ctx.get_recent_history().len(), 2);

        ctx.record_translation("three".to_string(), "три".to_string());
        let hist = ctx.get_recent_history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], ("two".to_string(), "два".to_string()));
        assert_eq!(hist[1], ("three".to_string(), "три".to_string()));

        // Zero max_history must stay empty
        let zero_ctx = ContextManager::new(0);
        zero_ctx.record_translation("hello".to_string(), "привет".to_string());
        assert!(zero_ctx.get_recent_history().is_empty());
    }

    #[test]
    fn test_encode_base64_known_vectors_and_padding() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
        assert_eq!(encode_base64(b"foob"), "Zm9vYg==");
        assert_eq!(encode_base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(encode_base64(b"Man"), "TWFu");
        assert_eq!(encode_base64(b"Ma"), "TWE=");
        assert_eq!(encode_base64(b"M"), "TQ==");
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(
            strip_html_tags("let me see <b>now</b>, &quot;yes&quot; &amp; &lt;ok&gt;"),
            "let me see now, \"yes\" & <ok>"
        );
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }

    #[test]
    fn test_strip_html_tags_decodes_entities_once() {
        assert_eq!(
            strip_html_tags("<b>&amp;lt;word&amp;gt;</b> &amp;quot; &amp;#39; &amp;amp;"),
            "&lt;word&gt; &quot; &#39; &amp;"
        );
    }

    #[test]
    fn test_deepl_target_lang_normalization() {
        assert_eq!(deepl_target_lang("en"), "EN-US");
        assert_eq!(deepl_target_lang("EN"), "EN-US");
        assert_eq!(deepl_target_lang("en-gb"), "EN-GB");
        assert_eq!(deepl_target_lang("pt"), "PT-PT");
        assert_eq!(deepl_target_lang("PT"), "PT-PT");
        assert_eq!(deepl_target_lang("pt-br"), "PT-BR");
        assert_eq!(deepl_target_lang("ru"), "RU");
        assert_eq!(deepl_target_lang("JA"), "JA");
    }

    #[test]
    fn test_urlencode() {
        assert_eq!(urlencode("hello world"), "hello%20world");
        assert_eq!(urlencode("a-b_c.d~e"), "a-b_c.d~e");
        assert_eq!(urlencode("Привет"), "%D0%9F%D1%80%D0%B8%D0%B2%D0%B5%D1%82");
        assert_eq!(
            urlencode("1 + 1 = 2 & ok?"),
            "1%20%2B%201%20%3D%202%20%26%20ok%3F"
        );
    }

    #[test]
    fn test_resolve_openai_vision_model() {
        for model in [None, Some(""), Some(" "), Some("llama-3.2-11b-vision-preview"), Some("llama-3.2-90b-vision-preview")] {
            assert!(resolve_openai_vision_model(model, "https://api.groq.com/openai/v1", "text-model").is_err());
        }
        assert_eq!(resolve_openai_vision_model(Some("available-vision"), "https://api.groq.com/openai/v1", "text-model").unwrap(), "available-vision");
        assert_eq!(resolve_openai_vision_model(None, "https://api.openai.com/v1", "gpt-3.5-turbo").unwrap(), "gpt-4o-mini");
        assert_eq!(resolve_openai_vision_model(None, "https://api.openai.com/v1", "gpt-4o").unwrap(), "gpt-4o");
        assert_eq!(resolve_openai_vision_model(None, "http://localhost:11434/v1", "llava:latest").unwrap(), "llava:latest");
        assert!(!is_groq_url("https://groq.com.invalid/v1"));
    }

    #[tokio::test]
    async fn test_lookup_word_details_failure_does_not_pollute_cache() {
        let service = TranslatorService::new();
        let config = AppConfig {
            provider: TranslationProvider::OpenAICompatible,
            openai_api_key: Some("dummy_key".to_string()),
            openai_base_url: "http://127.0.0.1:9".to_string(),
            ..Default::default()
        };

        let details = service
            .lookup_word_details("untranslatedword123", "en", "ru", &config)
            .await;
        assert_eq!(details.primary, "untranslatedword123");

        // The failed translation must NOT be stored in cache
        let cache_key = "dict:openai_compatible:en:ru:untranslatedword123";
        assert_eq!(service.cache.get(cache_key), None);
    }

    #[test]
    fn test_api_error_response_parsing() {
        let gemini_err: serde_json::Value = serde_json::json!({
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT"
            }
        });
        let msg = gemini_err
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str());
        assert_eq!(msg, Some("API key not valid. Please pass a valid API key."));

        let openai_err: serde_json::Value = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        let msg_oai = openai_err
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str());
        assert_eq!(msg_oai, Some("Incorrect API key provided"));
    }

    #[tokio::test]
    async fn test_translate_blocks_preserves_history_order() {
        let service = TranslatorService::new();
        let config = AppConfig {
            provider: TranslationProvider::OpenAICompatible,
            source_lang: "en".to_string(),
            target_lang: "ru".to_string(),
            openai_api_key: Some("dummy_key".to_string()),
            openai_base_url: "http://127.0.0.1:9".to_string(),
            ..Default::default()
        };

        // Prepopulate cache for blocks 0 and 2; block 1 is omitted to trigger translation processing
        service.cache.insert(
            service.cache_key("translation", "First block", "en", "ru", &config),
            "Первый блок".to_string(),
        );
        service.cache.insert(
            service.cache_key("translation", "Third block", "en", "ru", &config),
            "Третий блок".to_string(),
        );

        let blocks = vec![
            OcrBlock {
                text: "First block".to_string(),
                rect: ScreenRect::new(0, 0, 100, 20),
                lines: vec![],
            },
            OcrBlock {
                text: "Second block".to_string(),
                rect: ScreenRect::new(0, 30, 100, 20),
                lines: vec![],
            },
            OcrBlock {
                text: "Third block".to_string(),
                rect: ScreenRect::new(0, 60, 100, 20),
                lines: vec![],
            },
            OcrBlock {
                text: "I@II#O*$".to_string(),
                rect: ScreenRect::new(0, 90, 100, 20),
                lines: vec![],
            },
        ];

        let (results, all_succeeded) = service.translate_blocks(&blocks, &config).await;
        assert!(!all_succeeded);
        assert_eq!(results.len(), 4);
        assert_eq!(results[3].translated_text, "I@II#O*$");
        assert_eq!(results[3].rect, blocks[3].rect);
        assert_eq!(service.cache.get("openai_compatible:en:ru:I@II#O*$"), None);
        assert_eq!(results[0].translated_text, "Первый блок");
        assert!(results[1].translated_text.starts_with("[Err:"));
        assert_eq!(results[2].translated_text, "Третий блок");

        // History must record successful blocks in original order and exclude the failed block
        let history = service.context.get_recent_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0], ("First block".to_string(), "Первый блок".to_string()));
        assert_eq!(history[1], ("Third block".to_string(), "Третий блок".to_string()));

        // Contextual results must be cached under the new history snapshot. Repetitions still enter history.
        service.cache.insert(service.cache_key("translation", "First block", "en", "ru", &config), "?????? ????".into());
        service.cache.insert(service.cache_key("translation", "Third block", "en", "ru", &config), "?????? ????".into());
        service.cache.insert(
            service.cache_key("translation", "Second block", "en", "ru", &config),
            "Второй блок".to_string(),
        );
        let (cached_results, all_succeeded) = service.translate_blocks(&blocks, &config).await;
        assert!(all_succeeded);
        assert_eq!(cached_results[1].translated_text, "Второй блок");
        assert_eq!(service.context.get_recent_history().len(), 5);
    }

    #[test]
    fn test_bgra_to_png_downscaling() {
        let width = 2000;
        let height = 1000;
        let bgra = vec![255u8; (width * height * 4) as usize];
        let png = bgra_to_png(&bgra, width, height).expect("png encoding should succeed");
        let decoder = image::ImageReader::new(std::io::Cursor::new(png))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap();
        assert_eq!(decoder, (1920, 960));
    }

    #[test]
    fn test_parse_deepl_batch_response() {
        // 1. Normal response with N translations
        let json_normal = r#"{"translations": [{"text": "Hello"}, {"text": "World"}]}"#;
        let normal = parse_deepl_batch_response(json_normal, 2).expect("should parse 2 items");
        assert_eq!(normal, vec!["Hello", "World"]);

        // 2. Cyrillic text
        let json_ru = r#"{"translations": [{"text": "Привет"}, {"text": "Мир"}]}"#;
        let ru = parse_deepl_batch_response(json_ru, 2).expect("should parse cyrillic");
        assert_eq!(ru, vec!["Привет", "Мир"]);

        // 3. Fewer translations than expected
        let err_fewer = parse_deepl_batch_response(json_normal, 3);
        assert!(err_fewer.is_err());
        assert!(err_fewer.unwrap_err().contains("count mismatch"));

        // 4. More translations than expected
        let err_more = parse_deepl_batch_response(json_normal, 1);
        assert!(err_more.is_err());
        assert!(err_more.unwrap_err().contains("count mismatch"));

        // 5. Empty list
        let json_empty = r#"{"translations": []}"#;
        let err_empty = parse_deepl_batch_response(json_empty, 2);
        assert!(err_empty.is_err());
        assert_eq!(
            parse_deepl_batch_response(json_empty, 0).unwrap(),
            Vec::<String>::new()
        );

        // 6. Invalid JSON
        let err_invalid = parse_deepl_batch_response("{ not valid json", 1);
        assert!(err_invalid.is_err());

        // 7. API error message
        let json_err = r#"{"message": "Quota exceeded"}"#;
        let err_api = parse_deepl_batch_response(json_err, 1);
        assert!(err_api.is_err());
        assert!(err_api.unwrap_err().contains("Quota exceeded"));
    }

    #[test]
    fn test_assemble_batch_translations() {
        // 1. All in cache (empty uncached)
        let cached_all = vec![Some("First".to_string()), Some("Second".to_string())];
        let res_all = assemble_batch_translations(&cached_all, &[], &[]).unwrap();
        assert_eq!(res_all, vec!["First", "Second"]);

        // 2. Partial in cache (middle block translated)
        let cached_partial = vec![
            Some("Первый".to_string()),
            None,
            Some("Третий".to_string()),
        ];
        let uncached_idx = vec![1];
        let new_trans = vec!["Второй".to_string()];
        let res_partial =
            assemble_batch_translations(&cached_partial, &uncached_idx, &new_trans).unwrap();
        assert_eq!(res_partial, vec!["Первый", "Второй", "Третий"]);

        // 3. None in cache (all translated)
        let cached_none = vec![None, None];
        let uncached_all = vec![0, 1];
        let new_all = vec!["Блок 1".to_string(), "Блок 2".to_string()];
        let res_none = assemble_batch_translations(&cached_none, &uncached_all, &new_all).unwrap();
        assert_eq!(res_none, vec!["Блок 1", "Блок 2"]);

        // 4. Mismatch between uncached indices and translations count
        let err = assemble_batch_translations(&cached_partial, &uncached_idx, &[]);
        assert!(err.is_err());
    }

    #[test]
    fn test_is_translatable_text_and_dictionary_word() {
        // Translatable speech / UI
        assert!(is_translatable_text("PLEASE FOLLOW INSTRUCTIONS!"));
        assert!(is_translatable_text("LET'S POST IT ON UTUBE SHORTS!"));
        assert!(is_translatable_text("HAS ENTERED A NEW ERA."));
        assert!(is_translatable_text("+50"));
        assert!(is_translatable_text("100"));
        for text in ["No...", "Ah...", "(No!?)"] {
            assert!(is_translatable_text(text), "Valid dialogue rejected: {text}");
        }

        // Garbage / title artifacts / symbols
        assert!(!is_translatable_text("I@II#O*$"));
        assert!(!is_translatable_text("|| // 1! \\"));
        assert!(!is_translatable_text("   "));
        assert!(!is_translatable_text("•PA"));

        // Dictionary word validation
        assert!(is_valid_dictionary_word("INSTRUCTIONS"));
        assert!(is_valid_dictionary_word("LET'S"));
        assert!(is_valid_dictionary_word("don't"));
        assert!(is_valid_dictionary_word("state-of-the-art"));
        assert!(is_valid_dictionary_word("PLEASE!"));
        assert!(is_valid_dictionary_word("café"));
        assert!(is_valid_dictionary_word("Привет"));
        assert!(is_valid_dictionary_word("well-known"));

        // Invalid dictionary tokens
        assert!(!is_valid_dictionary_word("•PA"));
        assert!(!is_valid_dictionary_word("I@II#O*$"));
        assert!(!is_valid_dictionary_word("4EASE"));
        assert!(!is_valid_dictionary_word("126"));
        assert!(!is_valid_dictionary_word(""));
        assert!(!is_valid_dictionary_word("..."));
        assert!(!is_valid_dictionary_word("1ONG"));
        assert!(!is_valid_dictionary_word("@@"));
    }

    #[tokio::test]
    async fn test_dictionary_single_letter_bypasses_ocr_artifact_filter() {
        let service = TranslatorService::new();
        let config = AppConfig { provider: TranslationProvider::OpenAICompatible, ..Default::default() };
        service.cache.insert(service.cache_key("translation", "I", "en", "ru", &config), "pronoun".to_string());
        let details = service.lookup_word_details("I", "en", "ru", &config).await;
        assert_eq!(details.primary, "pronoun");
    }

}
