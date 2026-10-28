use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tracing::{error, warn};

use crate::capture::CapturedFrame;
use crate::types::{AppConfig, OcrBlock, ScreenRect, TranslationBlock, TranslationProvider};

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
        mgr.add_glossary_term("LV", "Ур.", Some("Level".to_string()));
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
        let mut h = self.history.lock().unwrap_or_else(|e| e.into_inner());
        if h.len() >= self.max_history {
            h.pop_front();
        }
        h.push_back((original, translated));
    }

    pub fn preprocess(&self, text: &str) -> String {
        text.to_string()
    }

    pub fn postprocess(&self, translated: &str) -> String {
        let g = self.glossary.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = translated.to_string();

        for (lower_key, entry) in g.iter() {
            if result.to_lowercase().contains(lower_key)
                && let Ok(re) = regex_simple_replace(&result, &entry.original, &entry.replacement)
            {
                result = re;
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

fn match_find_at(source: &str, start_idx: usize, find_lower: &str) -> Option<usize> {
    let tail = &source[start_idx..];
    let mut lower_buf = String::new();
    for (offset, ch) in tail.char_indices() {
        for lc in ch.to_lowercase() {
            lower_buf.push(lc);
        }
        if lower_buf == find_lower {
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

/// Thread-safe in-memory cache for translated phrases.
pub struct TranslationCache {
    entries: Mutex<HashMap<String, String>>,
    max_size: usize,
}

impl TranslationCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::with_capacity(max_size)),
            max_size,
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let cache = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        cache.get(key).cloned()
    }

    pub fn insert(&self, key: String, val: String) {
        let mut cache = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= self.max_size {
            let keys_to_remove: Vec<String> = cache.keys().take(50).cloned().collect();
            for k in keys_to_remove {
                cache.remove(&k);
            }
        }
        cache.insert(key, val);
    }
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

        if ch == '0' && (prev_alpha || next_alpha) {
            let is_upper = (prev_alpha && chars[i - 1].is_uppercase())
                || (next_alpha && chars[i + 1].is_uppercase());
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
            if lower == "i" {
                words_fixed.push("I".to_string());
            } else if let Some(rest) = lower.strip_prefix("i'") {
                words_fixed.push(format!("I'{}", rest));
            } else if lower == "hp"
                || lower == "mp"
                || lower == "exp"
                || lower == "lv"
                || lower == "atk"
                || lower == "def"
                || lower == "npc"
            {
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
    let mut s = text.to_string();
    let bad_pairs = [
        (" ,", ","),
        (" .", "."),
        (" !", "!"),
        (" ?", "?"),
        (" :", ":"),
        (" ;", ";"),
    ];
    for (bad, good) in bad_pairs {
        while s.contains(bad) {
            s = s.replace(bad, good);
        }
    }

    let mut out = String::new();
    let mut last_space = false;
    for ch in s.chars() {
        if ch == ' ' {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out.trim().to_string()
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

    let mut png_bytes = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png_bytes);
    image::ImageEncoder::write_image(
        encoder,
        &rgba,
        width,
        height,
        image::ExtendedColorType::Rgba8,
    )
    .map_err(|e| format!("PNG encoding failed: {:?}", e))?;

    Ok(png_bytes)
}

#[derive(Clone)]
pub struct TranslatorService {
    client: reqwest::Client,
    context: Arc<ContextManager>,
    cache: Arc<TranslationCache>,
}

impl Default for TranslatorService {
    fn default() -> Self {
        Self::new()
    }
}

impl TranslatorService {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(6000))
            .build()
            .unwrap_or_default();

        Self {
            client,
            context: Arc::new(ContextManager::new(20)),
            cache: Arc::new(TranslationCache::new(500)),
        }
    }

    pub fn context(&self) -> &ContextManager {
        &self.context
    }

    pub fn cache(&self) -> &TranslationCache {
        &self.cache
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
                    self.translate_gemini(clean, source_lang, target_lang, key)
                        .await
                } else {
                    warn!("Gemini API key not provided, falling back to Google");
                    self.translate_google(clean, source_lang, target_lang).await
                }
            }
            TranslationProvider::OpenAICompatible => {
                if let Some(key) = &config.openai_api_key {
                    self.translate_openai(
                        clean,
                        source_lang,
                        target_lang,
                        &config.openai_base_url,
                        key,
                        &config.openai_model,
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
            "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&dj=1&q={}",
            source_lang,
            target_lang,
            urlencoding::encode(text)
        );

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("HTTP request error: {:?}", e)))?;

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
        let endpoint = if api_key.ends_with(":fx") {
            "https://api-free.deepl.com/v2/translate"
        } else {
            "https://api.deepl.com/v2/translate"
        };

        let mut body = serde_json::json!({
            "text": [text],
            "target_lang": target_lang.to_uppercase(),
        });

        if !source_lang.is_empty() && source_lang != "auto" {
            body["source_lang"] = serde_json::json!(source_lang.to_uppercase());
        }

        if !context.is_empty() {
            body["context"] = serde_json::json!(context);
        }

        let resp = self
            .client
            .post(endpoint)
            .header("Authorization", format!("DeepL-Auth-Key {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("DeepL network error: {:?}", e)))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("DeepL response error: {:?}", e)))?;

        if let Some(err) = json.get("message").and_then(|v| v.as_str()) {
            return Err(redact_secrets(&format!("DeepL error: {}", err)));
        }

        let trans = json
            .get("translations")
            .and_then(|t| t.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("text"))
            .and_then(|txt| txt.as_str())
            .ok_or_else(|| "Invalid DeepL response structure".to_string())?;

        Ok(trans.to_string())
    }

    pub async fn translate_gemini(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        api_key: &str,
    ) -> Result<String, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            api_key
        );

        let system_instruction = format!(
            "You are an expert game, manga, and visual novel translator. Translate the text from {} to {}. \
            Preserve emotion, tone, and slang. Output ONLY the translated text without explanations or quotes.",
            source_lang, target_lang
        );

        let body = serde_json::json!({
            "contents": [{
                "parts": [{ "text": text }]
            }],
            "system_instruction": {
                "parts": [{ "text": system_instruction }]
            }
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini network error: {:?}", e)))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini response error: {:?}", e)))?;

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

    pub async fn translate_openai(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        base_url: &str,
        api_key: &str,
        model: &str,
    ) -> Result<String, String> {
        let endpoint = format!("{}/chat/completions", base_url.trim_end_matches('/'));

        let system_prompt = format!(
            "You are an expert game and manga translator. Translate from {} to {}. Output ONLY translation.",
            source_lang, target_lang
        );

        let body = serde_json::json!({
            "model": model,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": text }
            ],
            "temperature": 0.3
        });

        let resp = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI network error: {:?}", e)))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI response error: {:?}", e)))?;

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
        let png_bytes = bgra_to_png(&frame.bgra_bytes, frame.width, frame.height)?;
        let roi = ScreenRect::new(
            frame.offset_x,
            frame.offset_y,
            frame.width as i32,
            frame.height as i32,
        );

        match config.provider {
            TranslationProvider::Gemini => {
                let Some(key) = &config.gemini_api_key else {
                    return Err("Gemini API key is required for Vision OCR".to_string());
                };
                self.translate_gemini_vision(&png_bytes, &config.target_lang, key, &roi)
                    .await
            }
            TranslationProvider::OpenAICompatible => {
                let Some(key) = &config.openai_api_key else {
                    return Err("OpenAI/Groq API key is required for Vision OCR".to_string());
                };
                self.translate_openai_vision(
                    &png_bytes,
                    &config.target_lang,
                    &config.openai_base_url,
                    key,
                    &config.openai_model,
                    &roi,
                )
                .await
            }
            _ => Err(
                "Direct vision translation is supported on Gemini and OpenAI providers".to_string(),
            ),
        }
    }

    async fn translate_gemini_vision(
        &self,
        png_bytes: &[u8],
        target_lang: &str,
        api_key: &str,
        roi: &ScreenRect,
    ) -> Result<Vec<TranslationBlock>, String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            api_key
        );

        let base64_image = {
            const CHARSET: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut out = String::new();
            for chunk in png_bytes.chunks(3) {
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
        };

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

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini Vision network error: {:?}", e)))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("Gemini Vision response error: {:?}", e)))?;

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

        let base64_image = {
            const CHARSET: &[u8] =
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            let mut out = String::new();
            for chunk in png_bytes.chunks(3) {
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
        };

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

        let resp = self
            .client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI Vision network error: {:?}", e)))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| redact_secrets(&format!("OpenAI Vision response error: {:?}", e)))?;

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
        let cleaned = clean_ocr_noise(text);
        let normalized = normalize_all_caps(&cleaned);

        if normalized.is_empty() {
            return Ok(String::new());
        }

        let cache_key = format!(
            "{}:{}:{}:{}",
            config.provider.as_str(),
            source_lang,
            target_lang,
            normalized
        );
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        let preprocessed = self.context.preprocess(&normalized);
        let raw = self
            .translate_provider_raw(&preprocessed, source_lang, target_lang, config)
            .await?;
        let clean_ru = clean_russian_punctuation(&raw);
        let final_text = self.context.postprocess(&clean_ru);

        self.cache.insert(cache_key, final_text.clone());
        self.context
            .record_translation(normalized, final_text.clone());

        Ok(final_text)
    }

    /// Translates a batch of OCR blocks using batching or parallel JoinSet dispatch.
    pub async fn translate_blocks(
        &self,
        blocks: &[OcrBlock],
        config: &AppConfig,
    ) -> Vec<TranslationBlock> {
        if blocks.is_empty() {
            return Vec::new();
        }

        let cleaned_texts: Vec<String> = blocks
            .iter()
            .map(|b| {
                let cleaned = clean_ocr_noise(&b.text);
                normalize_all_caps(&cleaned)
            })
            .collect();

        // 1. Check in-memory cache
        let mut all_cached = true;
        let mut cached_results = Vec::with_capacity(blocks.len());

        for text in &cleaned_texts {
            let cache_key = format!(
                "{}:{}:{}:{}",
                config.provider.as_str(),
                config.source_lang,
                config.target_lang,
                text
            );
            if let Some(trans) = self.cache.get(&cache_key) {
                cached_results.push(Some(trans));
            } else {
                cached_results.push(None);
                all_cached = false;
            }
        }

        if all_cached {
            return blocks
                .iter()
                .zip(cached_results)
                .map(|(b, trans)| TranslationBlock {
                    original_text: b.text.clone(),
                    translated_text: trans.unwrap_or_else(|| b.text.clone()),
                    rect: b.rect,
                })
                .collect();
        }

        // 2. Context-aware batch translation when multiple dialogue bubbles are present
        if blocks.len() > 1 && config.provider == TranslationProvider::Google {
            let delimiter = "\n[---]\n";
            let combined = cleaned_texts.join(delimiter);

            if let Ok(batch_translated) = self
                .translate_provider_raw(&combined, &config.source_lang, &config.target_lang, config)
                .await
            {
                let parts: Vec<&str> = batch_translated.split("[---]").collect();
                if parts.len() == blocks.len() {
                    let mut results = Vec::with_capacity(blocks.len());
                    for (i, part) in parts.iter().enumerate() {
                        let clean_part = clean_russian_punctuation(part.trim());
                        let final_text = self.context.postprocess(&clean_part);
                        let key = format!(
                            "{}:{}:{}:{}",
                            config.provider.as_str(),
                            config.source_lang,
                            config.target_lang,
                            cleaned_texts[i]
                        );
                        self.cache.insert(key, final_text.clone());

                        results.push(TranslationBlock {
                            original_text: blocks[i].text.clone(),
                            translated_text: final_text,
                            rect: blocks[i].rect,
                        });
                    }
                    return results;
                }
            }
        }

        // 3. Fallback: parallel per-block translation
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, block) in blocks.iter().enumerate() {
            let this = self.clone();
            let orig_text = block.text.clone();
            let clean_text = cleaned_texts[idx].clone();
            let rect = block.rect;
            let cfg = config.clone();

            join_set.spawn(async move {
                let res = this
                    .translate_text(&clean_text, &cfg.source_lang, &cfg.target_lang, &cfg)
                    .await;
                (idx, orig_text, rect, res)
            });
        }

        let mut collected: Vec<(usize, TranslationBlock)> = Vec::with_capacity(blocks.len());

        while let Some(res) = join_set.join_next().await {
            match res {
                Ok((idx, orig_text, rect, trans_res)) => {
                    let translated_text = match trans_res {
                        Ok(t) => t,
                        Err(e) => {
                            error!("Failed to translate block '{}': {}", orig_text, e);
                            format!("[Err: {}]", orig_text)
                        }
                    };
                    collected.push((
                        idx,
                        TranslationBlock {
                            original_text: orig_text,
                            translated_text,
                            rect,
                        },
                    ));
                }
                Err(e) => {
                    error!("Join error during block translation: {:?}", e);
                }
            }
        }

        collected.sort_by_key(|(idx, _)| *idx);
        collected.into_iter().map(|(_, block)| block).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_ocr_noise() {
        let dirty = " | _ The ancient sword ~ | ";
        assert_eq!(clean_ocr_noise(dirty), "The ancient sword");

        let ocr_errors = "TH|S W0RLD";
        assert_eq!(clean_ocr_noise(ocr_errors), "THIS WORLD");

        let lower_errors = "c0me p1ayer";
        assert_eq!(clean_ocr_noise(lower_errors), "come player");
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
    fn test_translation_cache() {
        let cache = TranslationCache::new(5);
        cache.insert("key1".to_string(), "val1".to_string());
        assert_eq!(cache.get("key1"), Some("val1".to_string()));
        assert_eq!(cache.get("key2"), None);
    }

    #[test]
    fn test_redact_secrets_in_urls_and_headers() {
        let url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash?key=AIzaSyD_SecretKey1234567890&alt=json";
        let redacted_url = redact_secrets(url);
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
        let res = regex_simple_replace("ẞAB", "AB", "CD");
        assert_eq!(res.unwrap(), "ẞCD");

        // Turkish "İ" (2 bytes in UTF-8, lowercase 'i' is 1 byte)
        let res2 = regex_simple_replace("İ test", "test", "passed");
        assert_eq!(res2.unwrap(), "İ passed");

        // Russian, Japanese, emoji
        let res3 = regex_simple_replace("Привет МИР 🌍!", "мир", "Вселенная");
        assert_eq!(res3.unwrap(), "Привет Вселенная 🌍!");

        let res4 = regex_simple_replace("ゲーム OVER", "over", "終了");
        assert_eq!(res4.unwrap(), "ゲーム 終了");

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
    }
}
