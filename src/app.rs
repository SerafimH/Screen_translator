use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};
use windows::Win32::Foundation::POINT;

use crate::capture::{CapturedFrame, FrameSignature, ScreenCapture};
use crate::ocr::{
    OcrHelperClient, WinrtOcrEngine, cluster_lines_into_blocks, is_ocr_garbage,
    merge_recognized_lines, crop_frame, ocr_tiles,
};
use crate::overlay::{AppEvent, OverlayHandle, OverlayManager};
use crate::translator::TranslatorService;
use crate::types::{
    AppConfig, OcrBlock, OcrMode, ScreenRect, TextUnderCursor, TranslationBlock,
    cursor_word_key, cursor_lookup_token, find_text_under_cursor,
};
#[cfg(test)]
use crate::types::cursor_word_token;

/// RAII Guard that ensures `is_processing` is unconditionally reset to false upon completion or error.
#[derive(Debug)]
pub struct ProcessingGuard {
    flag: Arc<AtomicBool>,
}

impl ProcessingGuard {
    pub fn try_acquire(flag: &Arc<AtomicBool>) -> Option<Self> {
        if flag
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            Some(Self { flag: flag.clone() })
        } else {
            None
        }
    }

    /// Tries to acquire the guard, waiting for a pending task cancellation/drop to complete
    /// up to `timeout` before giving up.
    pub async fn acquire_or_wait(
        flag: &Arc<AtomicBool>,
        timeout: std::time::Duration,
    ) -> Option<Self> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(guard) = Self::try_acquire(flag) {
                return Some(guard);
            }
            if Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

impl Drop for ProcessingGuard {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct AppState {
    config: Arc<Mutex<AppConfig>>,
    overlay: OverlayHandle,
    ocr: Option<Arc<WinrtOcrEngine>>,
    helper_ocr: Arc<OcrHelperClient>,
    translator: Arc<TranslatorService>,
    active_roi: Arc<Mutex<Option<ScreenRect>>>,
    auto_mode: Arc<AtomicBool>,
    is_processing: Arc<AtomicBool>,
    last_signature: Arc<Mutex<Option<FrameSignature>>>,
    last_scene_content: Arc<Mutex<Vec<(String, ScreenRect)>>>,
    current_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

/// Checks whether RapidOCR helper returned valid non-garbage dialogue blocks.
pub fn is_helper_result_valid(blocks: &[OcrBlock]) -> bool {
    !blocks.is_empty() && blocks.iter().any(|b| !is_ocr_garbage(&b.text))
}

pub struct App {
    state: AppState,
    event_rx: UnboundedReceiver<AppEvent>,
}

impl App {
    pub fn new(config: AppConfig) -> Result<Self, String> {
        let (overlay, event_rx) = OverlayManager::spawn(&config)
            .map_err(|e| format!("Failed to spawn overlay: {:?}", e))?;

        let ocr = match WinrtOcrEngine::new(&config.ocr_lang) {
            Ok(engine) => Some(Arc::new(engine)),
            Err(e) => {
                if config.ocr_engine == OcrMode::Helper || config.ocr_engine == OcrMode::Auto {
                    warn!(
                        "WinRT OCR engine initialization skipped: {}. (Using RapidOCR helper).",
                        e
                    );
                    None
                } else {
                    return Err(format!("Failed to init OCR engine: {}", e));
                }
            }
        };
        let config_arc = Arc::new(Mutex::new(config.clone()));
        let helper_ocr = OcrHelperClient::new(config_arc.clone());
        let translator = Arc::new(TranslatorService::with_history_size(
            config.context_history_size,
        ));

        overlay.set_provider_name(config.provider.display_name().to_string());
        overlay.set_ocr_engine_name(config.ocr_engine.display_name().to_string());

        let state = AppState {
            config: config_arc,
            overlay,
            ocr,
            helper_ocr,
            translator,
            active_roi: Arc::new(Mutex::new(None)),
            auto_mode: Arc::new(AtomicBool::new(false)),
            is_processing: Arc::new(AtomicBool::new(false)),
            last_signature: Arc::new(Mutex::new(None)),
            last_scene_content: Arc::new(Mutex::new(Vec::new())),
            current_task: Arc::new(Mutex::new(None)),
        };

        Ok(Self { state, event_rx })
    }

    pub async fn run(mut self) {
        info!("Screen Translator application running.");
        info!(
            "Hotkeys: Alt+T (Translate), Alt+S (Select ROI), Alt+A (Auto), Alt+P (Engine), Alt+O (OCR Mode), Alt+C (Clear), Alt+Q (Quit)"
        );

        // Spawn background Dialogue Watcher (Auto-Mode)
        let auto_state = self.state.clone();
        tokio::spawn(async move {
            let interval_ms = {
                let cfg = auto_state.config.lock().unwrap_or_else(|e| e.into_inner());
                cfg.auto_mode_interval_ms.max(100)
            };
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(interval_ms));
            let mut last_scan = None;

            loop {
                interval.tick().await;

                if !auto_state.auto_mode.load(Ordering::Relaxed) {
                    continue;
                }

                if auto_state.is_processing.load(Ordering::Relaxed) {
                    continue;
                }
                if !auto_state.translator.automatic_retry_ready() {
                    continue;
                }

                let current_roi = *auto_state
                    .active_roi
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());

                let frame_res = tokio::task::spawn_blocking(move || {
                    let frame = ScreenCapture::capture_primary(current_roi)?;
                    let sig = FrameSignature::from_bgra(&frame.bgra_bytes, frame.width, frame.height);
                    Ok::<(CapturedFrame, FrameSignature), String>((frame, sig))
                })
                .await;

                let (_frame, mut sig) = match frame_res {
                    Ok(Ok(pair)) => pair,
                    _ => continue,
                };

                {
                    let last_sig_lock = auto_state
                        .last_signature
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(prev) = last_sig_lock.as_ref() {
                        sig.inherit_text_regions(prev);
                    }
                    if !auto_frame_needs_ocr(&sig, last_sig_lock.as_ref(), last_scan, Instant::now()) {
                        continue;
                    }
                }

                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                let stable = tokio::task::spawn_blocking(move || ScreenCapture::capture_primary(current_roi)).await;
                let Ok(Ok(stable)) = stable else { continue; };
                let stable_sig = FrameSignature::from_bgra(&stable.bgra_bytes, stable.width, stable.height);
                if stable_sig.has_changed(&sig)
                    || !auto_state.auto_mode.load(Ordering::Relaxed)
                    || auto_state.is_processing.load(Ordering::Relaxed)
                    || !auto_state.translator.automatic_retry_ready()
                    || *auto_state.active_roi.lock().unwrap_or_else(|e| e.into_inner()) != current_roi
                { continue; }
                last_scan = Some(Instant::now());
                auto_state.dispatch_translation_task(Some(stable));
            }
        });

        // Event listener loop
        while let Some(event) = self.event_rx.recv().await {
            match event {
                AppEvent::TriggerTranslate => {
                    crate::capture::reset_winrt_timeout_count();
                    self.state.translator.reset_retry();
                    self.state.dispatch_translation_task(None);
                }

                AppEvent::RoiSelected(rect) => {
                    info!("ROI selected: {:?}", rect);
                    self.state.translator.reset_retry();
                    *self
                        .state
                        .active_roi
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(rect);
                    self.state.overlay.set_active_roi(Some(rect));
                    self.state.dispatch_translation_task(None);
                }

                AppEvent::ToggleAutoMode => {
                    let new_val = !self.state.auto_mode.load(Ordering::Relaxed);
                    self.state.auto_mode.store(new_val, Ordering::Relaxed);
                    if new_val {
                        *self
                            .state
                            .last_signature
                            .lock()
                            .unwrap_or_else(|e| e.into_inner()) = None;
                    }
                    self.state.overlay.set_auto_mode(new_val);
                    info!("Auto-translate mode set to: {}", new_val);
                }

                AppEvent::CycleProvider => {
                    self.state.translator.reset_retry();
                    // Abort previous in-flight task on provider switch
                    if let Some(prev) = self
                        .state
                        .current_task
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    {
                        prev.abort();
                    }

                    // Reset FrameSignature so dialogue watcher immediately processes new frame if needed
                    *self
                        .state
                        .last_signature
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;

                    let new_provider = {
                        let mut cfg = self.state.config.lock().unwrap_or_else(|e| e.into_inner());
                        cfg.provider = cfg.provider.next();
                        cfg.provider
                    };

                    self.state
                        .overlay
                        .set_provider_name(new_provider.display_name().to_string());
                    let status = format!("Движок: {}", new_provider.display_name());
                    info!("{}", status);
                    self.state.overlay.set_loading(false, Some(status));
                }

                AppEvent::CycleOcrMode => {
                    self.state.translator.reset_retry();
                    crate::capture::reset_winrt_timeout_count();
                    // Abort previous in-flight task on OCR mode switch
                    if let Some(prev) = self
                        .state
                        .current_task
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    {
                        prev.abort();
                    }

                    // Reset FrameSignature so dialogue watcher immediately processes new frame if needed
                    *self
                        .state
                        .last_signature
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;

                    // Clear overlay cards without resetting active ROI
                    self.state.overlay.clear_blocks();

                    let new_mode = {
                        let mut cfg = self.state.config.lock().unwrap_or_else(|e| e.into_inner());
                        cfg.ocr_engine = cfg.ocr_engine.next();
                        cfg.ocr_engine
                    };

                    self.state
                        .overlay
                        .set_ocr_engine_name(new_mode.display_name().to_string());
                    let status = format!("Режим OCR: {}", new_mode.display_name());
                    info!("{}", status);
                    self.state.overlay.set_loading(false, Some(status));
                }

                AppEvent::Clear => {
                    if let Some(prev) = self
                        .state
                        .current_task
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    {
                        prev.abort();
                    }
                    *self
                        .state
                        .active_roi
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;
                    self.state.overlay.clear();
                    info!("Overlay and ROI cleared.");
                }

                AppEvent::ResetRoi => {
                    if let Some(task) = self.state.current_task.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        task.abort();
                    }
                    *self.state.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    *self
                        .state
                        .active_roi
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;
                    self.state.overlay.set_active_roi(None);
                    self.state.overlay.set_loading(false, Some("Область сброшена (полный экран)".to_string()));
                    info!("ROI reset to full screen.");
                }

                AppEvent::TranslateUnderCursor(pt) => {
                    self.state.translator.reset_retry();
                    self.state.dispatch_cursor_translation_task(pt);
                }

                AppEvent::SaveCard(card) => {
                    let result = tokio::task::spawn_blocking(move || crate::cards::save_card(&card)).await;
                    let status = match result {
                        Ok(Ok(path)) => format!("Карточка добавлена: {}", path.display()),
                        Ok(Err(e)) => format!("Не удалось сохранить карточку: {}", e),
                        Err(e) => format!("Ошибка сохранения карточки: {}", e),
                    };
                    self.state.overlay.set_loading(false, Some(status));
                }

                AppEvent::ShiftReleased => {
                    if let Some(prev) = self
                        .state
                        .current_task
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    {
                        prev.abort();
                    }
                    self.state.overlay.clear_cursor_card();
                }

                AppEvent::Quit => {
                    info!("Quitting application.");
                    break;
                }
            }
        }
    }
}

/// Cancels any currently running task in `task_slot` by calling `abort()` and
/// awaiting its termination inside the newly spawned task, ensuring that all resources
/// (like `ProcessingGuard`) are fully dropped before `fut` starts.
pub fn spawn_cancellable_task<Fut>(task_slot: &Arc<Mutex<Option<JoinHandle<()>>>>, fut: Fut)
where
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    // Keep cancellation and replacement atomic with Clear/provider switches.
    // Tokio never polls a spawned future synchronously, so the new task may
    // safely access this mutex after spawn returns and the guard is released.
    let mut slot_lock = task_slot.lock().unwrap_or_else(|e| e.into_inner());
    let prev = slot_lock.take();
    #[cfg(test)]
    tests::after_dispatch_slot_taken();
    if let Some(prev) = &prev {
        prev.abort();
    }

    *slot_lock = Some(tokio::spawn(async move {
        if let Some(prev) = prev {
            let _ = prev.await;
        }
        fut.await;
    }));
}

pub const CURSOR_CAPTURE_WIDTH: i32 = 1000;
pub const CURSOR_CAPTURE_HEIGHT: i32 = 800;

pub fn cursor_capture_roi(
    pt: POINT,
    active_roi: Option<ScreenRect>,
    min_x: i32,
    min_y: i32,
    max_x: i32,
    max_y: i32,
) -> ScreenRect {
    if let Some(roi) = active_roi.filter(|r| r.contains(pt.x, pt.y)) {
        roi
    } else {
        let cap_w = CURSOR_CAPTURE_WIDTH.min((max_x - min_x).max(1));
        let cap_h = CURSOR_CAPTURE_HEIGHT.min((max_y - min_y).max(1));

        let cap_x = if max_x > min_x + cap_w {
            (pt.x - cap_w / 2).clamp(min_x, max_x - cap_w)
        } else {
            min_x
        };

        let cap_y = if max_y > min_y + cap_h {
            (pt.y - cap_h / 2).clamp(min_y, max_y - cap_h)
        } else {
            min_y
        };

        ScreenRect::new(cap_x, cap_y, cap_w, cap_h)
    }
}

fn is_ordinal(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    let digits = if let Some(d) = lower.strip_suffix("st") {
        d
    } else if let Some(d) = lower.strip_suffix("nd") {
        d
    } else if let Some(d) = lower.strip_suffix("rd") {
        d
    } else if let Some(d) = lower.strip_suffix("th") {
        d
    } else {
        return false;
    };
    !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

/// Checks whether an OCR-recognized word is suspicious (likely OCR artifact/glitch):
/// - Returns true if it contains characters outside letters, ['’\-] and valid digits (e.g. •PA, I@II#O*$).
/// - Returns true if there is a lowercase -> uppercase transition inside the word (e.g. rrOTtcTION, PFOTtCTION).
/// - Returns true if length >= 3 and it contains both digits and letters, excluding ordinals (e.g. 1ONG, but 2nd / 3D are not suspicious).
pub fn is_suspicious_word(word: &str) -> bool {
    let raw = word.trim();
    if raw.is_empty() {
        return false;
    }

    // (а) Содержит символы вне латиницы/букв, апострофов, дефиса и цифр (например, «•PA», «I@II#O*$»)
    for c in raw.chars() {
        if !c.is_alphabetic() && c != '\'' && c != '’' && c != '-' && !c.is_ascii_digit() {
            return true;
        }
    }

    let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return true;
    }

    // (б) Переход «строчная буква -> заглавная» внутри слова
    let mut prev_is_lower = false;
    for c in trimmed.chars() {
        if c.is_lowercase() {
            prev_is_lower = true;
        } else if c.is_uppercase() {
            if prev_is_lower {
                return true;
            }
        } else {
            prev_is_lower = false;
        }
    }

    // (в) Длина >= 3 и содержит и цифры, и буквы (кроме порядковых числительных)
    if trimmed.chars().count() >= 3 {
        let has_digit = trimmed.chars().any(|c| c.is_ascii_digit());
        let has_alpha = trimmed.chars().any(|c| c.is_alphabetic());
        if has_digit && has_alpha && !is_ordinal(trimmed) {
            return true;
        }
    }

    false
}

impl AppState {
    fn dispatch_cursor_translation_task(&self, pt: POINT) {
        let state = self.clone();
        spawn_cancellable_task(&self.current_task, async move {
            state.perform_cursor_translation(pt).await;
        });
    }

    async fn perform_cursor_translation(&self, pt: POINT) {
        let Some(_guard) = ProcessingGuard::acquire_or_wait(
            &self.is_processing,
            std::time::Duration::from_millis(500),
        )
        .await
        else {
            info!("Translation already in progress, skipping cursor Shift trigger.");
            return;
        };

        let start_time = Instant::now();
        let cfg = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // If an active ROI was manually selected and contains the cursor, capture that ROI;
        // Otherwise capture enough vertical context for a complete manhwa
        // paragraph (1000x800), clamped to the current monitor's bounds.
        let active = *self.active_roi.lock().unwrap_or_else(|e| e.into_inner());
        let (vx, vy, vw, vh, primary_w, primary_h) = unsafe {
            use windows::Win32::UI::WindowsAndMessaging::{
                GetSystemMetrics, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN,
                SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
            };
            let vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            let vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
            let pw = GetSystemMetrics(SM_CXSCREEN);
            let ph = GetSystemMetrics(SM_CYSCREEN);
            if vw > 0 && vh > 0 {
                (
                    GetSystemMetrics(SM_XVIRTUALSCREEN),
                    GetSystemMetrics(SM_YVIRTUALSCREEN),
                    vw,
                    vh,
                    pw,
                    ph,
                )
            } else {
                (0, 0, pw, ph, pw, ph)
            }
        };

        // If cursor is within primary monitor bounds, keep ROI on primary monitor [0, primary_w / primary_h]
        // so fast WinRT D3D11 capture is preserved instead of spilling into secondary displays
        let (min_x, min_y, max_x, max_y) =
            if pt.x >= 0 && pt.x < primary_w && pt.y >= 0 && pt.y < primary_h {
                (0, 0, primary_w, primary_h)
            } else {
                (vx, vy, vx + vw, vy + vh)
            };

        let capture_roi = cursor_capture_roi(pt, active, min_x, min_y, max_x, max_y);

        self.overlay
            .set_loading(true, Some("Поиск текста под курсором...".to_string()));

        let capture_res =
            tokio::task::spawn_blocking(move || ScreenCapture::capture_primary(Some(capture_roi)))
                .await;

        let frame = match capture_res {
            Ok(Ok(f)) => f,
            Ok(Err(e)) => {
                warn!("Capture failed for cursor translation: {}", e);
                self.overlay
                    .set_loading(false, Some(format!("Ошибка захвата: {}", e)));
                return;
            }
            Err(e) => {
                warn!("Capture thread failed: {:?}", e);
                self.overlay
                    .set_loading(false, Some("Ошибка потока захвата".to_string()));
                return;
            }
        };

        let (blocks, raw_json) = match self.perform_ocr(&frame, &cfg).await {
            Ok(res) => res,
            Err(e) => {
                warn!("OCR error during cursor translation: {}", e);
                self.overlay
                    .set_loading(false, Some(format!("Ошибка OCR: {}", e)));
                return;
            }
        };

        if !crate::overlay::is_shift_held() {
            info!("Shift was released before OCR finished, aborting cursor translation.");
            self.overlay.set_loading(false, Some("Готов".to_string()));
            return;
        }

        let initial = find_text_under_cursor(&blocks, pt.x, pt.y);
        let Some(target) = self.verify_cursor_target(&frame, &cfg, initial, pt).await else {
            self.overlay.clear_cursor_card();
            self.overlay.set_loading(false, Some("Не удалось уверенно распознать слово под курсором".to_string()));
            return;
        };
        if !cursor_is_still_on_target(&target) {
            self.overlay.clear_cursor_card();
            return;
        }
        let Some(word) = target.word_text.as_ref() else { return; };
        let details_fut = self.translator.lookup_word_details(word, &cfg.source_lang, &cfg.target_lang, &cfg);
        let context = (target.sentence_text != *word && cursor_context_is_clean(&target.sentence_text))
            .then_some(target.sentence_text.as_str());
        let sentence_fut = async {
            if let Some(context) = context {
                self.translator.translate_context_in_scene(context, &blocks, &cfg).await.ok()
            } else { None }
        };
        let (details, sentence_translation) = tokio::join!(details_fut, sentence_fut);
        let mut card_text = format!("Слово: {} → {}", word, details.primary);
        if !details.synonyms.is_empty() {
            card_text.push_str(&format!(" ({})", details.synonyms.join(", ")));
        }
        if !details.examples.is_empty() {
            card_text.push_str("\n\nПримеры:");
            for example in &details.examples { card_text.push_str(&format!("\n• {}", example)); }
        }
        if let (Some(context), Some(translation)) = (context, sentence_translation) {
            card_text.push_str(&format!("\n\nВ контексте предложения:\n«{}»\n→ «{}»", context, translation));
        }
        let base_rect = target.word_rect.unwrap_or(target.line_rect);
        let translation_block = TranslationBlock {
            original_text: if let Some(context) = context { format!("{} ({})", word, context) } else { word.clone() },
            translated_text: card_text,
            rect: ScreenRect::new(base_rect.x, base_rect.y, base_rect.width.max(420), base_rect.height),
        };

        if cfg.dump_ocr {
            let res_text = format!(
                "Word: {:?}\nSentence: {}\nResult: {}\n",
                target.word_text, target.sentence_text, translation_block.translated_text
            );
            crate::ocr::dump_ocr_artifacts(
                &frame,
                &raw_json,
                &res_text,
            );
        }

        if !cursor_is_still_on_target(&target) {
            info!("Cursor left the verified word or Shift was released, skipping card display.");
            self.overlay.set_loading(false, Some("Готов".to_string()));
            return;
        }
        if !cursor_image_is_current(&frame, &target).await || !cursor_is_still_on_target(&target) {
            self.overlay.clear_cursor_card();
            self.overlay.set_loading(false, Some("Текст под курсором изменился. Повторите Shift.".to_string()));
            return;
        }

        let elapsed = start_time.elapsed().as_millis();
        let status = format!("Перевод под курсором за {} мс", elapsed);
        info!("{}", status);
        self.overlay
            .update_cursor_card(Some(translation_block), Some(status));
    }

    async fn verify_cursor_target(
        &self, frame: &CapturedFrame, cfg: &AppConfig,
        initial: Option<TextUnderCursor>, pt: POINT,
    ) -> Option<TextUnderCursor> {
        let initial = initial.or_else(|| cursor_search_hint(frame, pt))?;
        let crop_rect = cursor_line_crop(frame, initial.line_rect)?;
        let crop = crop_frame(frame, crop_rect).ok()?;
        let mut readings = Vec::new();
        let mut helper_readings = Vec::new();
        if let Some(native) = &self.ocr {
            if let Ok(lines) = native.recognize_raw(&crop).await {
                readings.push(find_text_under_cursor(&cluster_lines_into_blocks(lines, None), pt.x, pt.y));
            }
            if let Ok(lines) = native.recognize_pass_1(&crop).await {
                readings.push(find_text_under_cursor(&cluster_lines_into_blocks(lines, None), pt.x, pt.y));
            }
        }
        if self.helper_ocr.is_ready() {
            if let Ok(blocks) = self.helper_ocr.recognize(&crop, &cfg.ocr_lang).await {
                helper_readings.push(find_text_under_cursor(&blocks, pt.x, pt.y));
            }
        }
        if let Some(target) = choose_cursor_target_from_readings(
            &initial, &readings, &helper_readings, cfg.helper_min_score,
        ) {
            return Some(target);
        }
        // A missing coarse reading needs two crop readings. This also works when
        // WinRT is unavailable. Two strong helper crops can also resolve an
        // unscored WinRT error on handwriting; helper conflicts remain rejected.
        let valid_count = readings.iter().chain(&helper_readings).flatten()
            .filter(|reading| reliable_cursor_token(reading, cfg.helper_min_score).is_some()).count();
        let strong_helper = helper_readings.iter().flatten().any(|reading|
            reliable_cursor_token(reading, cfg.helper_min_score.max(0.95)).is_some()
                && reading.word_confidence.is_some());
        if (valid_count == 1 || strong_helper) && self.helper_ocr.is_ready() {
            let second_rect = cursor_line_crop_with_padding(frame, initial.line_rect, 20)?;
            if second_rect != crop_rect {
                let second_crop = crop_frame(frame, second_rect).ok()?;
                if let Ok(blocks) = self.helper_ocr.recognize(&second_crop, &cfg.ocr_lang).await {
                    helper_readings.push(find_text_under_cursor(&blocks, pt.x, pt.y));
                }
            }
        }
        let confirmed = choose_cursor_target_from_readings(&initial, &readings, &helper_readings, cfg.helper_min_score);
        if confirmed.is_some() { return confirmed; }
        // A numeric label can be damaged by neighbouring letters in a line
        // crop. Confirm its actual pixels independently with two word crops.
        if initial.raw_word.as_ref().is_some_and(|word| word.chars().any(|ch| ch.is_ascii_digit()))
            && let Some(rect) = initial.word_rect {
            let mut words = Vec::new();
            for padding in [4, 8] {
                let crop = crop_frame(frame, cursor_line_crop_with_padding(frame, rect, padding)?).ok()?;
                let blocks = self.helper_ocr.recognize(&crop, &cfg.ocr_lang).await.ok()?;
                words.push(find_text_under_cursor(&blocks, pt.x, pt.y));
            }
            return choose_cursor_target_from_readings(&initial, &[], &words, cfg.helper_min_score);
        }
        None
    }

    fn dispatch_translation_task(&self, prefetched_frame: Option<CapturedFrame>) {
        let state = self.clone();
        spawn_cancellable_task(&self.current_task, async move {
            state.perform_translation(prefetched_frame).await;
        });
    }

    async fn perform_translation(&self, prefetched_frame: Option<CapturedFrame>) {
        let Some(_guard) = ProcessingGuard::acquire_or_wait(
            &self.is_processing,
            std::time::Duration::from_millis(500),
        )
        .await
        else {
            info!("Translation already in progress, skipping trigger.");
            return;
        };

        let start_time = Instant::now();
        let roi = *self.active_roi.lock().unwrap_or_else(|e| e.into_inner());
        let cfg = self
            .config
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        let automatic = prefetched_frame.is_some();
        let frame = match prefetched_frame {
            Some(f) => f,
            None => {
                self.overlay
                    .set_loading(true, Some("Захват экрана (D3D11 GPU)...".to_string()));
                info!("Capturing frame (ROI: {:?})...", roi);

                let capture_res =
                    tokio::task::spawn_blocking(move || ScreenCapture::capture_primary(roi)).await;

                match capture_res {
                    Ok(Ok(f)) => f,
                    Ok(Err(e)) => {
                        error!("Capture failed: {}", e);
                        self.overlay
                            .update_blocks(Vec::new(), Some(format!("Ошибка захвата: {}", e)));
                        return;
                    }
                    Err(join_err) => {
                        error!("Capture join error: {:?}", join_err);
                        self.overlay.update_blocks(
                            Vec::new(),
                            Some("Ошибка фонового потока захвата".to_string()),
                        );
                        return;
                    }
                }
            }
        };

        info!(
            "Captured {}x{} frame. Running OCR (mode: {:?})...",
            frame.width, frame.height, cfg.ocr_engine
        );
        let mut sig = FrameSignature::from_bgra(&frame.bgra_bytes, frame.width, frame.height);
        let scene_changed = self.last_signature.lock().unwrap_or_else(|e| e.into_inner())
            .as_ref().is_none_or(|previous| sig.has_changed(previous));

        let (blocks, raw_json) = match self.perform_ocr(&frame, &cfg).await {
            Ok(res) => res,
            Err(e) => {
                error!("OCR error: {}", e);
                *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = None;
                if cfg.vision_fallback && (cfg.gemini_api_key.is_some() || cfg.openai_api_key.is_some()) {
                    (Vec::new(), format!("Local OCR failed: {}", e))
                } else {
                    self.overlay.update_blocks(Vec::new(), Some(format!("Ошибка OCR: {}", e)));
                    return;
                }
            }
        };

        let has_text = !blocks.is_empty() && blocks.iter().any(|b| !is_ocr_garbage(&b.text));
        sig.set_text_regions(blocks.iter().flat_map(|block| block.lines.iter().map(|line| line.rect)),
                             frame.offset_x, frame.offset_y);

        let has_vision_support = cfg.gemini_api_key.is_some() || cfg.openai_api_key.is_some();
        let mut vision_error = None;

        // Step 4: Vision fallback if all local passes found nothing and vision_fallback is enabled with API key
        if !has_text && cfg.vision_fallback && has_vision_support {
            info!("Local OCR found no text; triggering Vision OCR fallback...");
            self.overlay.set_loading(
                true,
                Some("Локальный OCR не нашёл текст, пробую через ИИ...".to_string()),
            );

            match self
                .translator
                .translate_image_direct_vision(&frame, &cfg)
                .await
            {
                Ok(vision_blocks) => {
                    if !vision_blocks.is_empty() {
                        sig.set_text_regions(vision_blocks.iter().map(|block| block.rect),
                                             frame.offset_x, frame.offset_y);
                        let elapsed = start_time.elapsed().as_millis();
                        let status = format!(
                            "Vision OCR: {} блоков за {} мс",
                            vision_blocks.len(),
                            elapsed
                        );
                        info!("{}", status);
                        *self.last_scene_content.lock().unwrap_or_else(|e| e.into_inner()) = vision_blocks.iter()
                            .map(|block| (block.original_text.clone(), block.rect)).collect();
                        *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);
                        self.overlay.update_blocks(vision_blocks, Some(status));
                        return;
                    }
                }
                Err(err) => {
                    warn!("Vision OCR fallback failed: {}", err);
                    vision_error = Some(err);
                }
            }
        }

        if !has_text {
            let elapsed = start_time.elapsed().as_millis();
            let status = if let Some(error) = vision_error.as_deref() {
                let reason = self.translator.retry_message().unwrap_or_else(|| error.to_string());
                let reason = crate::translator::redact_secrets(&reason).chars().take(240).collect::<String>();
                format!("Ошибка Vision OCR: {reason}")
            } else if !cfg.vision_fallback {
                format!(
                    "Текст не найден ({} мс). Для сложных картинок включите vision_fallback = true в config.toml",
                    elapsed
                )
            } else {
                format!("Текст не найден ({} мс)", elapsed)
            };
            *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = if vision_error.is_some() { None } else { Some(sig) };
            self.last_scene_content.lock().unwrap_or_else(|e| e.into_inner()).clear();
            self.overlay.update_blocks(Vec::new(), Some(status));
            return;
        }

        let valid_blocks: Vec<OcrBlock> = blocks
            .into_iter()
            .filter(|b| !is_ocr_garbage(&b.text))
            .collect();
        let content = valid_blocks.iter().map(|block| (block.text.clone(), block.rect)).collect::<Vec<_>>();
        if automatic && !scene_changed
            && same_scene_content(&content, &self.last_scene_content.lock().unwrap_or_else(|e| e.into_inner())) {
            // Periodic OCR can observe background animation without a new
            // utterance. Refresh detection, keeping dialogue history untouched.
            *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);
            self.overlay.set_loading(false, Some("Текст не изменился".to_string()));
            return;
        }

        self.overlay.set_loading(
            true,
            Some(format!("Перевод {} блоков текста...", valid_blocks.len())),
        );

        let report = self.translator.translate_blocks_report(&valid_blocks, &cfg).await;
        let all_succeeded = report.errors.is_empty();
        let translations = report.blocks;

        if cfg.dump_ocr {
            let res_text = translations
                .iter()
                .map(|b| format!("ORIG: {}\nTRANS: {}\n", b.original_text, b.translated_text))
                .collect::<Vec<_>>()
                .join("\n---\n");
            crate::ocr::dump_ocr_artifacts(
                &frame,
                &raw_json,
                &res_text,
            );
        }

        let elapsed = start_time.elapsed().as_millis();
        let status = translation_status(translations.len(), elapsed, &report.errors,
            self.translator.retry_message().as_deref());
        info!("{}", status);

        *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = if all_succeeded { Some(sig) } else { None };
        if all_succeeded {
            *self.last_scene_content.lock().unwrap_or_else(|e| e.into_inner()) = content;
        }

        self.overlay.update_blocks(translations, Some(status));
    }

    async fn perform_ocr(
        &self, frame: &CapturedFrame, cfg: &AppConfig,
    ) -> Result<(Vec<OcrBlock>, String), String> {
        let tiles = ocr_tiles(frame.width, frame.height);
        if tiles.len() <= 1 { return self.perform_ocr_region(frame, cfg).await; }
        let mut lines = Vec::new();
        let mut raw = Vec::new();
        for rect in tiles {
            let tile = crop_frame(frame, rect)?;
            let (blocks, response) = self.perform_ocr_region(&tile, cfg).await?;
            merge_recognized_lines(&mut lines, blocks.into_iter().flat_map(|b| b.lines).collect());
            raw.push(response);
        }
        Ok((cluster_lines_into_blocks(lines, Some((&frame.bgra_bytes, frame.width,
            frame.height, frame.offset_x, frame.offset_y))), raw.join("\n")))
    }

    async fn perform_ocr_region(
        &self,
        frame: &CapturedFrame,
        cfg: &AppConfig,
    ) -> Result<(Vec<OcrBlock>, String), String> {
        match cfg.ocr_engine {
            OcrMode::WinRt => {
                self.overlay.set_loading(
                    true,
                    Some("Распознавание текста (WinRT OCR)...".to_string()),
                );
                if let Some(ocr) = &self.ocr {
                    let blocks = ocr.recognize(frame).await?;
                    Ok((blocks, "WinRT OCR".to_string()))
                } else {
                    Err("Системный WinRT OCR недоступен. Установите языковой пакет в Windows или выберите RapidOCR.".to_string())
                }
            }
            OcrMode::Helper => {
                self.overlay
                    .set_loading(true, Some("Распознавание текста (RapidOCR)...".to_string()));
                match self.helper_ocr.recognize_with_raw(frame, &cfg.ocr_lang).await {
                    Ok((blocks, raw_json)) => Ok((blocks, raw_json)),
                    Err(err) => {
                        warn!("RapidOCR helper failed: {}. Falling back to WinRT...", err);
                        if let Some(ocr) = &self.ocr {
                            self.overlay.set_loading(
                                true,
                                Some("RapidOCR недоступен, откат на WinRT OCR...".to_string()),
                            );
                            let blocks = ocr.recognize(frame).await?;
                            Ok((blocks, format!("WinRT Fallback (RapidOCR error: {})", err)))
                        } else {
                            Err(format!(
                                "RapidOCR helper failed: {}. WinRT OCR также недоступен.",
                                err
                            ))
                        }
                    }
                }
            }
            OcrMode::Auto => {
                self.overlay.set_loading(true, Some("Auto: сравнение локальных OCR...".to_string()));
                let mut lines = Vec::new();
                let mut native_error = None;
                if let Some(ocr) = &self.ocr {
                    // Compare independent raw pixels with RapidOCR. Aggressive
                    // native mask passes can mistake portrait hatching for text.
                    match ocr.recognize_raw(frame).await {
                        Ok(native_lines) => lines.extend(native_lines),
                        Err(e) => native_error = Some(e),
                    }
                }
                match self.helper_ocr.recognize_raw_lines_with_raw(frame, &cfg.ocr_lang).await {
                    Ok((helper_lines, raw)) => {
                        let native_lines = lines;
                        let mut lines = helper_lines;
                        // Native OCR may localise text RapidOCR missed, but an
                        // unscored portrait artefact is not an independent line.
                        // Use unmatched native locations as crop hints and keep
                        // the independently recognised crop, not the guess.
                        for native in native_lines.into_iter().filter(|native| !native.rect.is_empty()).take(8) {
                            let overlaps = lines.iter().any(|line| {
                                let a = native.rect; let b = line.rect;
                                a.x < b.x + b.width && b.x < a.x + a.width
                                    && a.y < b.y + b.height && b.y < a.y + a.height
                            });
                            if overlaps { continue; }
                            if let Some(rect) = cursor_line_crop_with_padding(frame, native.rect, 8)
                                && let Ok(crop) = crop_frame(frame, rect)
                                && let Ok(candidates) = self.helper_ocr.recognize_raw_lines(&crop, &cfg.ocr_lang).await {
                                merge_recognized_lines(&mut lines, candidates);
                            }
                        }
                        Ok((cluster_lines_into_blocks(lines, Some((&frame.bgra_bytes, frame.width,
                            frame.height, frame.offset_x, frame.offset_y))), raw))
                    }
                    Err(e) if lines.is_empty() => Err(native_error.unwrap_or(e)),
                    Err(e) => {
                        warn!("RapidOCR unavailable: {}", e);
                        Ok((cluster_lines_into_blocks(lines, Some((&frame.bgra_bytes, frame.width,
                            frame.height, frame.offset_x, frame.offset_y))), "WinRT fallback".to_string()))
                    }
                }
            }
        }
    }
}

const AUTO_CONTROL_SCAN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

fn auto_frame_needs_ocr(
    current: &FrameSignature, previous: Option<&FrameSignature>,
    last_scan: Option<Instant>, now: Instant,
) -> bool {
    let Some(previous) = previous else { return true; };
    current.has_changed(previous) || (!current.same_pixels_as(previous)
        && last_scan.is_none_or(|last| now.saturating_duration_since(last) >= AUTO_CONTROL_SCAN_INTERVAL))
}

fn same_scene_content(current: &[(String, ScreenRect)], previous: &[(String, ScreenRect)]) -> bool {
    current.len() == previous.len() && current.iter().zip(previous).all(|((text, rect), (old_text, old_rect))|
        text == old_text && [(rect.x, old_rect.x), (rect.y, old_rect.y),
                            (rect.width, old_rect.width), (rect.height, old_rect.height)]
            .iter().all(|&(a, b)| (a as i64 - b as i64).abs() <= 2))
}

async fn cursor_image_is_current(frame: &CapturedFrame, target: &TextUnderCursor) -> bool {
    let Some(rect) = cursor_line_crop_with_padding(frame, target.sentence_rect, 3) else { return false; };
    let Ok(original) = crop_frame(frame, rect) else { return false; };
    let roi = ScreenRect::new(original.offset_x, original.offset_y, original.width as i32, original.height as i32);
    let fresh = tokio::task::spawn_blocking(move || ScreenCapture::capture_primary(Some(roi))).await;
    match fresh {
        Ok(Ok(fresh)) => cursor_snapshots_match(&original, &fresh, target.word_rect),
        _ => false,
    }
}

fn cursor_snapshots_match(original: &CapturedFrame, fresh: &CapturedFrame, word_rect: Option<ScreenRect>) -> bool {
    if !cursor_pixels_match(original, fresh) { return false; }
    let Some(word_rect) = word_rect else { return false; };
    let Some(old_crop) = cursor_line_crop_with_padding(original, word_rect, 1) else { return false; };
    let Some(new_crop) = cursor_line_crop_with_padding(fresh, word_rect, 1) else { return false; };
    match (crop_frame(original, old_crop), crop_frame(fresh, new_crop)) {
        (Ok(old_word), Ok(new_word)) => cursor_pixels_match(&old_word, &new_word),
        _ => false,
    }
}

fn cursor_pixels_match(original: &CapturedFrame, fresh: &CapturedFrame) -> bool {
    if (original.width, original.height, original.offset_x, original.offset_y)
        != (fresh.width, fresh.height, fresh.offset_x, fresh.offset_y)
        || original.width == 0 || original.height == 0 {
        return false;
    }
    let pixels = original.width as usize * original.height as usize;
    if original.bgra_bytes.len() != pixels * 4 || fresh.bgra_bytes.len() != pixels * 4 { return false; }
    let mut sums = [[0i64; 3]; 2];
    let mut minimum = 255u8;
    let mut maximum = 0u8;
    for (old, new) in original.bgra_bytes.chunks_exact(4).zip(fresh.bgra_bytes.chunks_exact(4)) {
        for channel in 0..3 {
            sums[0][channel] += old[channel] as i64;
            sums[1][channel] += new[channel] as i64;
            minimum = minimum.min(old[channel]);
            maximum = maximum.max(old[channel]);
        }
    }
    let shift = std::array::from_fn::<_, 3, _>(|channel| (sums[1][channel] - sums[0][channel]) / pixels as i64);
    let tolerance = if maximum - minimum <= 24 { 1 } else { 12 };
    let changed = original.bgra_bytes.chunks_exact(4).zip(fresh.bgra_bytes.chunks_exact(4))
        .filter(|(old, new)| (0..3).any(|channel|
            (new[channel] as i64 - old[channel] as i64 - shift[channel]).abs() > tolerance)).count();
    changed <= (pixels / 200).max(3)
}

/// A line-sized crop retains complete words while excluding unrelated paragraphs.
fn cursor_search_hint(frame: &CapturedFrame, pt: POINT) -> Option<TextUnderCursor> {
    let x = pt.x as i64 - frame.offset_x as i64;
    let y = pt.y as i64 - frame.offset_y as i64;
    if x < 0 || y < 0 || x >= frame.width as i64 || y >= frame.height as i64 { return None; }
    Some(TextUnderCursor {
        raw_word: None, word_confidence: None, word_is_estimated: true,
        line_rect: ScreenRect::new(pt.x.saturating_sub(172), pt.y.saturating_sub(40), 344, 80),
        word_text: None, word_rect: None, sentence_text: String::new(),
        sentence_rect: ScreenRect::new(pt.x, pt.y, 0, 0),
    })
}

fn cursor_line_crop(frame: &CapturedFrame, line: ScreenRect) -> Option<ScreenRect> {
    cursor_line_crop_with_padding(frame, line, 8)
}

fn cursor_line_crop_with_padding(frame: &CapturedFrame, line: ScreenRect, padding: i64) -> Option<ScreenRect> {
    let left = (line.x as i64 - frame.offset_x as i64 - padding).clamp(0, frame.width as i64);
    let top = (line.y as i64 - frame.offset_y as i64 - padding).clamp(0, frame.height as i64);
    let right = (line.x as i64 + line.width as i64 - frame.offset_x as i64 + padding).clamp(0, frame.width as i64);
    let bottom = (line.y as i64 + line.height as i64 - frame.offset_y as i64 + padding).clamp(0, frame.height as i64);
    if right <= left || bottom <= top { return None; }
    Some(ScreenRect::new(left as i32, top as i32, (right-left) as i32, (bottom-top) as i32))
}

fn reliable_cursor_token(target: &TextUnderCursor, min_score: f32) -> Option<&str> {
    if target.word_confidence.is_some_and(|score| !score.is_finite() || score > 1.0 || score < min_score.max(0.8)) {
        return None;
    }
    cursor_lookup_token(target.raw_word.as_deref()?)
}

/// Agreement is required; a score cannot resolve conflicting crop readings.
fn choose_verified_cursor_target(
    initial: &TextUnderCursor, crop_readings: &[Option<TextUnderCursor>], min_score: f32,
) -> Option<TextUnderCursor> {
    let valid: Vec<(&TextUnderCursor, &str)> = crop_readings.iter().flatten()
        .filter_map(|reading| reliable_cursor_token(reading, min_score).map(|token| (reading, token))).collect();
    let (_, token) = *valid.first()?;
    let normalized = cursor_word_key(token)?;
    if valid.iter().any(|(_, other)| cursor_word_key(other).as_ref() != Some(&normalized)) { return None; }
    let initial_agrees = reliable_cursor_token(initial, min_score)
        .is_some_and(|original| cursor_word_key(original).as_ref() == Some(&normalized));
    if valid.len() + usize::from(initial_agrees) < 2 { return None; }
    let real = valid.iter().find(|(reading, _)| !reading.word_is_estimated)
        .map(|(reading, _)| *reading)
        .or_else(|| (initial_agrees && !initial.word_is_estimated).then_some(initial))?;
    let mut accepted = real.clone();
    // Case-insensitive agreement confirms letters, not a different meaning of
    // US/us or Polish/polish. Keep the original spelling when it was confirmed.
    accepted.word_text = Some(if initial_agrees {
        reliable_cursor_token(initial, min_score)?.to_string()
    } else { token.to_string() });
    if initial_agrees && cursor_context_is_clean(&initial.sentence_text) {
        accepted.sentence_text = initial.sentence_text.clone();
        accepted.sentence_rect = initial.sentence_rect;
    } else if !cursor_context_is_clean(&accepted.sentence_text) {
        accepted.sentence_text = token.to_string();
    }
    Some(accepted)
}

fn choose_cursor_target_from_readings(
    initial: &TextUnderCursor, native: &[Option<TextUnderCursor>],
    helper: &[Option<TextUnderCursor>], min_score: f32,
) -> Option<TextUnderCursor> {
    let all = native.iter().chain(helper).cloned().collect::<Vec<_>>();
    choose_verified_cursor_target(initial, &all, min_score).or_else(|| {
        // Native OCR has no confidence and can read handwritten "find" as
        // "finA". Override that reading only after two distinct helper crops
        // agree with real geometry and >=95% word confidence in both crops.
        if native.iter().flatten().all(|target| target.word_confidence.is_none())
            && helper.len() == 2 && helper.iter().all(|reading| reading.as_ref().is_some_and(|target|
            !target.word_is_estimated && target.word_confidence.is_some()
                && reliable_cursor_token(target, min_score.max(0.95)).is_some())) {
            choose_verified_cursor_target(initial, helper, min_score.max(0.95))
        } else { None }
    })
}

fn cursor_context_is_clean(text: &str) -> bool {
    // A sentence is not a dictionary token: numbers and separated punctuation
    // are ordinary context, while OCR corruption such as @#* stays forbidden.
    text.chars().any(char::is_alphanumeric) && text.chars().all(|ch| {
        ch.is_alphanumeric() || ch.is_whitespace() || matches!(ch,
            '.' | ',' | '!' | '?' | ':' | ';' | '…' | '"' | '\'' | 'ʼ'
            | '‘' | '’' | '“' | '”' | '«' | '»' | '(' | ')' | '[' | ']' | '{' | '}'
            | '-' | '‐' | '‑' | '—' | '–' | '/' | '%' | '+' | '=' | '&'
            | '。' | '、' | '！' | '？' | '，' | '：' | '；'
            | '「' | '」' | '『' | '』' | '（' | '）')
    })
}

fn cursor_is_still_on_target(target: &TextUnderCursor) -> bool {
    if !crate::overlay::is_shift_held() { return false; }
    let mut pt = POINT::default();
    if unsafe { windows::Win32::UI::WindowsAndMessaging::GetCursorPos(&mut pt) }.is_err() { return false; }
    target.word_rect.is_some_and(|rect| cursor_point_on_word(rect, pt))
}

fn cursor_point_on_word(rect: ScreenRect, pt: POINT) -> bool {
    !rect.is_empty() && pt.x as i64 >= rect.x as i64 - 2 && pt.y as i64 >= rect.y as i64 - 2
        && (pt.x as i64) < rect.x as i64 + rect.width as i64 + 2
        && (pt.y as i64) < rect.y as i64 + rect.height as i64 + 2
}

fn translation_status(block_count: usize, elapsed_ms: u128, errors: &[String], retry: Option<&str>) -> String {
    if errors.is_empty() {
        format!("Переведено {block_count} блоков за {elapsed_ms} мс")
    } else {
        let reason = retry.or_else(|| errors.first().map(String::as_str)).unwrap_or("Неизвестная ошибка");
        let reason = crate::translator::redact_secrets(reason).chars().take(240).collect::<String>();
        format!("Ошибок перевода: {} из {block_count}. {reason}", errors.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    std::thread_local! {
        static DISPATCH_SLOT_TAKEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const {
            std::cell::RefCell::new(None)
        };
    }

    pub(super) fn after_dispatch_slot_taken() {
        let hook = DISPATCH_SLOT_TAKEN.with(|hook| hook.borrow_mut().take());
        if let Some(hook) = hook { hook(); }
    }

    #[tokio::test]
    async fn cancellation_cannot_miss_a_task_during_dispatch() {
        let slot = Arc::new(Mutex::new(None));
        let committed = Arc::new(AtomicBool::new(false));
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel::<()>();
        let (paused_tx, paused_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let dispatcher_slot = slot.clone();
        let dispatcher_commit = committed.clone();
        let runtime = tokio::runtime::Handle::current();
        let dispatcher = std::thread::spawn(move || {
            let _entered = runtime.enter();
            DISPATCH_SLOT_TAKEN.with(|hook| {
                *hook.borrow_mut() = Some(Box::new(move || {
                    paused_tx.send(()).unwrap();
                    resume_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                }));
            });
            spawn_cancellable_task(&dispatcher_slot, async move {
                if finish_rx.await.is_ok() {
                    dispatcher_commit.store(true, Ordering::SeqCst);
                }
            });
        });
        paused_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        // Clear/provider-switch cancellation must wait until the replacement
        // handle is installed, rather than observe a temporarily empty slot.
        let slot_locked = matches!(slot.try_lock(), Err(std::sync::TryLockError::WouldBlock));
        resume_tx.send(()).unwrap();
        dispatcher.join().unwrap();
        let task = slot.lock().unwrap().take().unwrap();
        task.abort();
        let _ = finish_tx.send(());
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(!committed.load(Ordering::SeqCst));
        assert!(slot_locked, "dispatch exposed an empty task slot to cancellation");
    }

    #[test]
    fn auto_control_scan_recovers_changes_that_fine_and_coarse_detection_miss() {
        let load = |name| {
            let image = image::open(format!("tests/fixtures/capture/dialogue_{name}.png")).unwrap().to_rgba8();
            let (width, height) = image.dimensions();
            let mut pixels = image.into_raw();
            for pixel in pixels.chunks_exact_mut(4) {
                for channel in &mut pixel[..3] { *channel = 202 + (*channel as u16 * 3 / 255) as u8; }
            }
            FrameSignature::from_bgra(&pixels, width, height)
        };
        let mut yes = load("yes");
        yes.set_text_regions([ScreenRect::new(790, 645, 100, 65)], 0, 0);
        let no = load("no");
        assert!(!no.has_changed(&yes), "this regression deliberately uses faint lettering");
        let start = Instant::now();
        assert!(!auto_frame_needs_ocr(&no, Some(&yes), Some(start), start + std::time::Duration::from_secs(1)));
        assert!(auto_frame_needs_ocr(&no, Some(&yes), Some(start), start + AUTO_CONTROL_SCAN_INTERVAL));
        assert!(!auto_frame_needs_ocr(&yes, Some(&yes), Some(start), start + AUTO_CONTROL_SCAN_INTERVAL));
        let mut blank = FrameSignature::from_bgra(&vec![205u8; 1920 * 1080 * 4], 1920, 1080);
        blank.set_text_regions([], 0, 0);
        assert!(!yes.has_changed(&blank));
        assert!(auto_frame_needs_ocr(&yes, Some(&blank), Some(start), start + AUTO_CONTROL_SCAN_INTERVAL));
    }

    #[test]
    fn cursor_snapshot_rejects_changed_word_even_when_cursor_coordinates_stay_the_same() {
        for (background, ink) in [(255u8, 0u8), (205, 202)] {
            let mut original = CapturedFrame {
                width: 50, height: 20, offset_x: -900, offset_y: 200,
                bgra_bytes: vec![background; 50 * 20 * 4],
            };
            for y in 5..15 {
                for x in 10..15 {
                    original.bgra_bytes[(y * 50 + x) * 4..(y * 50 + x) * 4 + 3].fill(ink);
                }
            }
            assert!(cursor_pixels_match(&original, &original.clone()));
            let mut changed = original.clone();
            for y in 5..15 {
                changed.bgra_bytes[(y * 50 + 20) * 4..(y * 50 + 20) * 4 + 3].fill(ink);
            }
            assert!(!cursor_pixels_match(&original, &changed));
            let mut illumination = original.clone();
            for pixel in illumination.bgra_bytes.chunks_exact_mut(4) {
                for channel in &mut pixel[..3] { *channel = channel.saturating_sub(2); }
                pixel[3] = 0; // capture backends need not agree about alpha
            }
            assert!(cursor_pixels_match(&original, &illumination));
            illumination.offset_x += 1;
            assert!(!cursor_pixels_match(&original, &illumination));
        }
    }

    #[test]
    fn cursor_word_change_cannot_hide_in_a_large_unchanged_paragraph() {
        let mut original = CapturedFrame {
            width: 600, height: 200, offset_x: -900, offset_y: 200,
            bgra_bytes: vec![255; 600 * 200 * 4],
        };
        for y in 20..30 {
            for x in 20..30 {
                original.bgra_bytes[(y * 600 + x) * 4..(y * 600 + x) * 4 + 3].fill(0);
            }
        }
        let mut fresh = original.clone();
        for y in 20..30 {
            fresh.bgra_bytes[(y * 600 + 35) * 4..(y * 600 + 35) * 4 + 3].fill(0);
        }
        assert!(cursor_pixels_match(&original, &fresh), "a paragraph-wide ratio alone misses a short word");
        let word = Some(ScreenRect::new(-885, 215, 30, 25));
        assert!(cursor_snapshots_match(&original, &original.clone(), word));
        assert!(!cursor_snapshots_match(&original, &fresh, word));
    }

    fn cursor_reading(raw: &str, confidence: Option<f32>, estimated: bool) -> TextUnderCursor {
        let rect = ScreenRect::new(100, 100, 80, 20);
        TextUnderCursor {
            raw_word: Some(raw.to_string()), word_confidence: confidence, word_is_estimated: estimated,
            line_rect: rect, word_text: cursor_word_token(raw).map(str::to_string), word_rect: Some(rect),
            sentence_text: raw.to_string(), sentence_rect: rect,
        }
    }

    #[test]
    fn cursor_verification_requires_real_geometry_and_consistent_rereading() {
        let initial = cursor_reading("Hello", None, false);
        assert!(choose_verified_cursor_target(&initial, &[], 0.5).is_none());
        let accepted = choose_verified_cursor_target(&initial, &[Some(cursor_reading("HELLO", Some(0.99), false))], 0.5).unwrap();
        assert_eq!(accepted.word_text.as_deref(), Some("Hello"));
        assert!(choose_verified_cursor_target(&initial, &[
            Some(cursor_reading("Hello", None, false)), Some(cursor_reading("Help", Some(0.99), false)),
        ], 0.5).is_none());
        let estimated = cursor_reading("Hello", Some(0.99), true);
        assert!(choose_verified_cursor_target(&estimated, &[Some(estimated.clone())], 0.5).is_none());
        assert!(choose_verified_cursor_target(&estimated, &[Some(initial.clone())], 0.5).is_some());
        assert!(cursor_point_on_word(accepted.word_rect.unwrap(), POINT { x: 120, y: 110 }));
        assert!(!cursor_point_on_word(accepted.word_rect.unwrap(), POINT { x: 250, y: 110 }));
    }

    #[test]
    fn cursor_native_conflict_needs_two_strong_agreeing_helper_crops() {
        let initial = cursor_reading("find", Some(0.99), false);
        let native = [Some(cursor_reading("finA", None, false))];
        let helper = Some(cursor_reading("find", Some(0.99), false));
        assert!(choose_cursor_target_from_readings(&initial, &native, &[helper.clone()], 0.5).is_none());
        let accepted = choose_cursor_target_from_readings(
            &initial, &native, &[helper.clone(), helper.clone()], 0.5,
        ).unwrap();
        assert_eq!(accepted.word_text.as_deref(), Some("find"));
        for other in [Some(cursor_reading("from", Some(0.99), false)),
                      Some(cursor_reading("find", Some(0.90), false)),
                      Some(cursor_reading("find", Some(0.99), true)), None] {
            assert!(choose_cursor_target_from_readings(&initial, &native, &[helper.clone(), other], 0.5).is_none());
        }
        assert!(choose_cursor_target_from_readings(&initial, &native, &[helper.clone(), helper], 1.0).is_none());
    }

    #[test]
    fn english_punctuation_variants_do_not_create_false_ocr_conflicts() {
        for (observed, reread) in [("don't", "don’t"), ("state-of-the-art", "state‑of‑the‑art"),
            ("O'Neill", "OʼNeill")] {
            let initial = cursor_reading(observed, None, false);
            let crop = cursor_reading(reread, Some(0.99), false);
            let accepted = choose_verified_cursor_target(&initial, &[Some(crop)], 0.5)
                .expect("equivalent typography should confirm the observed English word");
            assert_eq!(accepted.word_text.as_deref(), Some(observed));
        }
        let initial = cursor_reading("form", None, false);
        assert!(choose_verified_cursor_target(&initial,
            &[Some(cursor_reading("from", Some(0.99), false))], 0.5).is_none());
    }

    #[test]
    fn cursor_confirmation_preserves_case_sensitive_word_and_context() {
        for (original, reread, context) in [
            ("US", "us", "I live in the US."),
            ("us", "US", "Come with us."),
            ("Polish", "polish", "I speak Polish."),
            ("iPhone", "IPHONE", "My iPhone works."),
        ] {
            let mut initial = cursor_reading(original, Some(0.99), false);
            initial.sentence_text = context.to_string();
            let accepted = choose_verified_cursor_target(&initial,
                &[Some(cursor_reading(reread, Some(0.99), false))], 0.5).unwrap();
            assert_eq!(accepted.word_text.as_deref(), Some(original));
            assert_eq!(accepted.sentence_text, context);
        }
    }

    #[test]
    fn cursor_context_accepts_numbers_and_sentence_punctuation_without_accepting_corruption() {
        for context in ["I have 100 HP.", "Hello , world !", "HP:100 / 200 (50%).",
            "Level 10: HP + 20% & MP = 50."] {
            assert!(cursor_context_is_clean(context), "Valid context rejected: {context}");
            let mut initial = cursor_reading("HP", None, false);
            initial.sentence_text = context.to_string();
            let accepted = choose_verified_cursor_target(&initial,
                &[Some(cursor_reading("HP", Some(0.99), false))], 0.5).unwrap();
            assert_eq!(accepted.sentence_text, context);
            assert_ne!(accepted.sentence_text, accepted.word_text.unwrap());
        }
        for context in ["", "   ", "... !?", "*#($#", "Read Jdlk;@#k4 now"] {
            assert!(!cursor_context_is_clean(context), "Corrupt context accepted: {context}");
        }
    }

    #[test]
    fn cursor_verification_rejects_artifacts_and_low_word_confidence() {
        for raw in ["*#($#", "Jdlk;@#k4", "*#Jdlk;@#"] {
            let bad = cursor_reading(raw, Some(0.99), false);
            assert!(choose_verified_cursor_target(&bad, &[Some(bad.clone())], 0.5).is_none());
        }
        let good = cursor_reading("Hello", Some(0.99), false);
        for confidence in [0.01, 1.1, f32::NAN, f32::INFINITY] {
            let bad = cursor_reading("Hello", Some(confidence), false);
            assert!(choose_verified_cursor_target(&good, &[Some(bad)], 0.5).is_none());
        }
        assert!(choose_verified_cursor_target(&good, &[Some(cursor_reading("Hello", Some(0.85), false))], 0.9).is_none());
    }

    #[test]
    fn cursor_verification_can_recover_bad_initial_text_without_exposing_it() {
        let mut bad = cursor_reading("Jdlk;@#k4", Some(0.99), false);
        bad.sentence_text = "Please Jdlk;@#k4 now".to_string();
        for name in ["iPhone", "McDonald", "don't", "state-of-the-art"] {
            let corrected = cursor_reading(name, Some(0.99), false);
            assert!(choose_verified_cursor_target(&bad, &[Some(corrected.clone())], 0.5).is_none());
            let accepted = choose_verified_cursor_target(&bad, &[Some(corrected.clone()), Some(corrected)], 0.5).unwrap();
            assert_eq!(accepted.word_text.as_deref(), Some(name));
            assert_eq!(accepted.sentence_text, name);
            assert!(!accepted.sentence_text.contains('@'));
        }
        let mut initial = cursor_reading("qzxvbnm", Some(0.99), false);
        initial.sentence_text = "PLEASE qzxvbnm NOW".to_string();
        let corrected = cursor_reading("FOLLOW", Some(0.99), false);
        let accepted = choose_verified_cursor_target(&initial, &[Some(corrected.clone()), Some(corrected)], 0.5).unwrap();
        assert_eq!(accepted.sentence_text, "FOLLOW");
    }

    #[test]
    fn cursor_crop_clamps_locally_and_preserves_negative_desktop_offsets() {
        let frame = CapturedFrame { width: 100, height: 50, offset_x: -500, offset_y: -200,
            bgra_bytes: vec![255; 100 * 50 * 4] };
        let rect = cursor_line_crop(&frame, ScreenRect::new(-498, -199, 40, 20)).unwrap();
        assert_eq!(rect, ScreenRect::new(0, 0, 50, 29));
        let crop = crop_frame(&frame, rect).unwrap();
        assert_eq!((crop.offset_x, crop.offset_y, crop.width, crop.height), (-500, -200, 50, 29));
        assert!(cursor_line_crop(&frame, ScreenRect::new(500, 500, 10, 10)).is_none());
        let center = POINT { x: -450, y: -175 };
        let hint = cursor_search_hint(&frame, center).unwrap();
        assert_eq!(cursor_line_crop(&frame, hint.line_rect), Some(ScreenRect::new(0, 0, 100, 50)));
        assert!(cursor_search_hint(&frame, POINT { x: 0, y: 0 }).is_none());
        let reading = cursor_reading("Hello", None, false);
        assert!(choose_verified_cursor_target(&hint, &[Some(reading.clone())], 0.5).is_none());
        assert!(choose_verified_cursor_target(&hint, &[Some(reading.clone()), Some(reading)], 0.5).is_some());
    }

    #[tokio::test]
    async fn cursor_crop_recognition_accepts_real_fixture_word_and_rejects_background() {
        use crate::overlay::OverlaySharedState;
        use windows::Win32::Foundation::HWND;
        let rgba = image::open("tests/fixtures/ocr/small_subtitle.png").unwrap().to_rgba8();
        let (width, height) = rgba.dimensions();
        let mut bgra = rgba.into_raw();
        for pixel in bgra.chunks_exact_mut(4) { pixel.swap(0, 2); }
        let frame = CapturedFrame { width, height, bgra_bytes: bgra, offset_x: 0, offset_y: 0 };
        let config = Arc::new(Mutex::new(AppConfig {
            ocr_engine: OcrMode::Helper, helper_timeout_ms: 15_000, ..AppConfig::default()
        }));
        let helper = OcrHelperClient::new(config.clone());
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            while !helper.is_ready() { tokio::time::sleep(std::time::Duration::from_millis(50)).await; }
        }).await.expect("fixture helper must initialize within its startup budget");
        let blocks = helper.recognize(&frame, "en").await.expect("fixture OCR helper must work");
        let state = AppState {
            config: config.clone(), overlay: OverlayHandle { hwnd: HWND(std::ptr::null_mut()),
                shared_state: Arc::new(Mutex::new(OverlaySharedState::default())) },
            ocr: None, helper_ocr: helper, translator: Arc::new(TranslatorService::new()),
            active_roi: Arc::new(Mutex::new(None)), auto_mode: Arc::new(AtomicBool::new(false)),
            is_processing: Arc::new(AtomicBool::new(false)), last_signature: Arc::new(Mutex::new(None)),
            last_scene_content: Arc::new(Mutex::new(Vec::new())),
            current_task: Arc::new(Mutex::new(None)),
        };
        let cfg = config.lock().unwrap().clone();
        let pt = POINT { x: 273, y: 89 };
        let initial = find_text_under_cursor(&blocks, pt.x, pt.y);
        let accepted = state.verify_cursor_target(&frame, &cfg, initial, pt).await
            .expect("a real, correctly reread word should be accepted");
        assert_eq!(accepted.word_text.unwrap().to_lowercase(), "wait");
        // Without WinRT and without a coarse candidate, two distinct helper
        // crops must be able to confirm the word while retaining real geometry.
        let hint = cursor_search_hint(&frame, pt).unwrap();
        assert_ne!(cursor_line_crop(&frame, hint.line_rect),
            cursor_line_crop_with_padding(&frame, hint.line_rect, 20));
        let recovered = state.verify_cursor_target(&frame, &cfg, None, pt).await
            .expect("helper-only fallback must recover a missed coarse word");
        assert_eq!(recovered.word_text.unwrap().to_lowercase(), "wait");
        assert!(!recovered.word_is_estimated);
        let background = POINT { x: 50, y: 40 };
        assert!(state.verify_cursor_target(&frame, &cfg,
            find_text_under_cursor(&blocks, background.x, background.y), background).await.is_none());
    }

    #[test]
    fn failed_translation_status_shows_count_cause_and_hides_keys() {
        let errors = vec!["OpenAI HTTP 401: Bearer sk-audit-secret".to_string()];
        let status = translation_status(3, 42, &errors, None);
        assert!(status.contains("1 из 3"));
        assert!(status.contains("401"));
        assert!(!status.contains("sk-audit-secret"));
        assert!(!status.contains("Переведено"));
        assert_eq!(translation_status(3, 42, &[], None), "Переведено 3 блоков за 42 мс");
    }

    #[tokio::test]
    async fn supplied_manhwa_auto_roi_and_shift_preserve_complete_dialogue() {
        use crate::overlay::OverlaySharedState;
        use windows::Win32::Foundation::HWND;
        let config = Arc::new(Mutex::new(AppConfig {
            ocr_engine: OcrMode::Auto, helper_timeout_ms: 15_000, ..AppConfig::default()
        }));
        let helper = OcrHelperClient::new(config.clone());
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            while !helper.is_ready() { tokio::time::sleep(std::time::Duration::from_millis(50)).await; }
        }).await.expect("manhwa helper must initialize");
        let state = AppState {
            config: config.clone(), overlay: OverlayHandle { hwnd: HWND(std::ptr::null_mut()),
                shared_state: Arc::new(Mutex::new(OverlaySharedState::default())) },
            ocr: WinrtOcrEngine::new("en").ok().map(Arc::new), helper_ocr: helper,
            translator: Arc::new(TranslatorService::new()), active_roi: Arc::new(Mutex::new(None)),
            auto_mode: Arc::new(AtomicBool::new(false)), is_processing: Arc::new(AtomicBool::new(false)),
            last_signature: Arc::new(Mutex::new(None)), last_scene_content: Arc::new(Mutex::new(vec![])),
            current_task: Arc::new(Mutex::new(None)),
        };
        let cfg = config.lock().unwrap().clone();
        let cases: serde_json::Value = serde_json::from_str(include_str!("../tests/fixtures/ocr/corpus.json")).unwrap();
        let normalize = |text: &str| text.replace(['’', '‘', 'ʼ'], "'").replace('…', "...")
            .split_whitespace().collect::<Vec<_>>().join(" ").to_uppercase();
        let mut failures = Vec::new();
        let mut checked = 0;
        for case in cases.as_array().unwrap().iter().filter(|case| case["file"].as_str().unwrap().starts_with("manhwa_")) {
            let file = case["file"].as_str().unwrap();
            let image = image::open(format!("tests/fixtures/ocr/{file}")).unwrap().to_rgba8();
            let (width, height) = image.dimensions();
            let mut pixels = image.into_raw();
            for pixel in pixels.chunks_exact_mut(4) { pixel.swap(0, 2); }
            let frame = CapturedFrame { width, height, bgra_bytes: pixels, offset_x: -400, offset_y: 200 };
            let (blocks, _) = state.perform_ocr(&frame, &cfg).await.unwrap();
            let expected = case["blocks"].as_array().unwrap().iter().map(|text| text.as_str().unwrap()).collect::<Vec<_>>();
            if blocks.len() != expected.len() {
                failures.push(format!("{file}: {} paragraphs instead of {}: {:?}", blocks.len(), expected.len(), blocks.iter().map(|b| &b.text).collect::<Vec<_>>()));
            }
            for (block, expected) in blocks.iter().zip(expected) {
                if normalize(&block.text) != normalize(expected) { failures.push(format!("{file} Auto: {} != {expected}", block.text)); }
                let rect = cursor_line_crop_with_padding(&frame, block.rect, 30).unwrap();
                let roi = crop_frame(&frame, rect).unwrap();
                let (roi_blocks, _) = state.perform_ocr(&roi, &cfg).await.unwrap();
                let roi_text = roi_blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>().join(" ");
                if normalize(&roi_text) != normalize(expected) { failures.push(format!("{file} ROI {:?}: {roi_text} != {expected}", rect)); }
                let words = block.lines.iter().flat_map(|line| &line.words).collect::<Vec<_>>();
                let mut selected = vec![0, words.len().saturating_sub(1)];
                selected.extend(words.iter().enumerate().filter_map(|(i, word)|
                    word.text.chars().any(|ch| ch.is_ascii_digit()).then_some(i)));
                selected.sort_unstable(); selected.dedup();
                for index in selected {
                    let Some(word) = words.get(index) else { failures.push(format!("{file}: no cursor words")); continue; };
                    let pt = POINT { x: word.rect.x + word.rect.width / 2, y: word.rect.y + word.rect.height / 2 };
                    let capture = cursor_capture_roi(pt, None, frame.offset_x - 100, frame.offset_y - 100,
                        frame.offset_x + 1920, frame.offset_y + 1080);
                    if !capture.contains(block.rect.x, block.rect.y)
                        || !capture.contains(block.rect.x + block.rect.width - 1, block.rect.y + block.rect.height - 1) {
                        failures.push(format!("{file} Shift capture clips the paragraph at {}", word.text));
                    }
                    let initial = find_text_under_cursor(&blocks, pt.x, pt.y);
                    let confirmed = state.verify_cursor_target(&frame, &cfg, initial, pt).await;
                    if !confirmed.as_ref().is_some_and(|target| normalize(&target.sentence_text) == normalize(expected)) {
                        failures.push(format!("{file} Shift {}: {:?}", word.text, confirmed));
                    }
                }
            }
            checked += 1;
        }
        assert_eq!(checked, 17);
        std::fs::write("target/manhwa-pipeline-errors.json", serde_json::to_string_pretty(&failures).unwrap()).unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[tokio::test]
    async fn handwritten_black_panel_preserves_sentence_and_confirms_each_cursor_word() {
        use crate::overlay::OverlaySharedState;
        use windows::Win32::Foundation::HWND;
        let rgba = image::open("tests/fixtures/ocr/manga_handwritten_black.png").unwrap().to_rgba8();
        let (width, height) = rgba.dimensions();
        let mut bgra = rgba.into_raw();
        for pixel in bgra.chunks_exact_mut(4) { pixel.swap(0, 2); }
        let frame = CapturedFrame { width, height, bgra_bytes: bgra, offset_x: 0, offset_y: 0 };
        let cfg = AppConfig { ocr_engine: OcrMode::Auto, helper_timeout_ms: 15_000,
                              ..AppConfig::default() };
        let config = Arc::new(Mutex::new(cfg.clone()));
        let helper = OcrHelperClient::new(config.clone());
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            while !helper.is_ready() { tokio::time::sleep(std::time::Duration::from_millis(50)).await; }
        }).await.expect("fixture helper must initialize");
        let state = AppState {
            config, overlay: OverlayHandle { hwnd: HWND(std::ptr::null_mut()),
                shared_state: Arc::new(Mutex::new(OverlaySharedState::default())) },
            ocr: WinrtOcrEngine::new("en").ok().map(Arc::new), helper_ocr: helper,
            translator: Arc::new(TranslatorService::new()),
            active_roi: Arc::new(Mutex::new(None)), auto_mode: Arc::new(AtomicBool::new(false)),
            is_processing: Arc::new(AtomicBool::new(false)), last_signature: Arc::new(Mutex::new(None)),
            last_scene_content: Arc::new(Mutex::new(Vec::new())),
            current_task: Arc::new(Mutex::new(None)),
        };
        let (blocks, _) = state.perform_ocr(&frame, &cfg).await.unwrap();
        for (word, x, y) in [("D-Did", 145, 108), ("he", 259, 112),
                             ("find", 121, 156), ("out", 225, 162)] {
            let initial = find_text_under_cursor(&blocks, x, y);
            let confirmed = state.verify_cursor_target(&frame, &cfg, initial, POINT { x, y }).await;
            assert_eq!(confirmed.as_ref().and_then(|t| t.word_text.as_deref()), Some(word),
                       "cursor reading failed for {word}: {confirmed:?}");
            assert_eq!(confirmed.unwrap().sentence_text, "D-Did he find out?!");
        }
        assert_eq!(blocks.len(), 1, "one utterance must be translated together: {blocks:?}");
        assert_eq!(blocks[0].text, "D-Did he find out?!");
        state.translator.context().record_translation("Previous real dialogue".into(), "Предыдущая реплика".into());
        let history = state.translator.context().get_recent_history();
        let mut signature = FrameSignature::from_bgra(&frame.bgra_bytes, width, height);
        signature.set_text_regions(blocks.iter().flat_map(|block| block.lines.iter().map(|line| line.rect)), 0, 0);
        *state.last_signature.lock().unwrap() = Some(signature);
        *state.last_scene_content.lock().unwrap() = blocks.iter().map(|block| (block.text.clone(), block.rect)).collect();
        state.perform_translation(Some(frame.clone())).await;
        assert_eq!(state.translator.context().get_recent_history(), history,
                   "a control OCR scan of unchanged dialogue must not enter history again");
    }

    #[tokio::test]
    async fn test_local_ocr_error_reaches_configured_vision() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use crate::overlay::OverlaySharedState;
        use crate::types::TranslationProvider;
        use windows::Win32::Foundation::HWND;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = vec![0; 8192];
            let count = stream.read(&mut bytes).await.unwrap();
            assert!(String::from_utf8_lossy(&bytes[..count]).contains("/chat/completions"));
            let content = r#"[{"original":"Hello","translation":"Привет","box_2d":[0,0,1000,1000]}]"#;
            let body = serde_json::json!({"choices":[{"message":{"content":content}}]}).to_string();
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let config = Arc::new(Mutex::new(AppConfig {
            ocr_engine: OcrMode::WinRt, vision_fallback: true,
            provider: TranslationProvider::OpenAICompatible,
            openai_api_key: Some("test-key".to_string()),
            openai_base_url: format!("http://{}", address),
            openai_vision_model: Some("test-vision".to_string()),
            ..AppConfig::default()
        }));
        let shared = Arc::new(Mutex::new(OverlaySharedState::default()));
        let state = AppState {
            config: config.clone(), overlay: OverlayHandle { hwnd: HWND(std::ptr::null_mut()), shared_state: shared.clone() },
            ocr: None, helper_ocr: OcrHelperClient::new(config), translator: Arc::new(TranslatorService::new()),
            active_roi: Arc::new(Mutex::new(None)), auto_mode: Arc::new(AtomicBool::new(false)),
            is_processing: Arc::new(AtomicBool::new(false)), last_signature: Arc::new(Mutex::new(None)),
            last_scene_content: Arc::new(Mutex::new(Vec::new())),
            current_task: Arc::new(Mutex::new(None)),
        };
        let frame = CapturedFrame { width: 40, height: 40, bgra_bytes: vec![255; 40 * 40 * 4], offset_x: 0, offset_y: 0 };
        tokio::time::timeout(std::time::Duration::from_secs(3), state.perform_translation(Some(frame))).await.unwrap();
        server.await.unwrap();
        assert_eq!(shared.lock().unwrap().current_blocks[0].translated_text, "Привет");
        assert!(!state.is_processing.load(Ordering::SeqCst));
    }

    #[test]
    fn bug_9_overlay_clear_empties_blocks_and_resets_state() {
        use crate::overlay::{OverlayHandle, OverlaySharedState};
        use crate::types::{ScreenRect, TranslationBlock};
        use windows::Win32::Foundation::HWND;

        let shared_state = Arc::new(Mutex::new(OverlaySharedState {
            current_blocks: vec![TranslationBlock {
                original_text: "Hello".to_string(),
                translated_text: "Привет".to_string(),
                rect: ScreenRect::new(10, 10, 100, 50),
            }],
            cursor_card: None,
            status_msg: Some("Translating...".to_string()),
            is_loading: true,
            active_roi: Some(ScreenRect::new(10, 10, 100, 50)),
            auto_mode: false,
            provider_name: "Test".to_string(),
            ocr_engine_name: "RapidOCR".to_string(),
        }));

        let handle = OverlayHandle {
            hwnd: HWND(core::ptr::null_mut()),
            shared_state: shared_state.clone(),
        };

        // Before clear, blocks are present, active_roi is set, and loading is true
        assert_eq!(shared_state.lock().unwrap().current_blocks.len(), 1);
        assert!(shared_state.lock().unwrap().active_roi.is_some());
        assert!(shared_state.lock().unwrap().is_loading);

        // When clear() is called (as required by BUG-9 on AppEvent::Clear):
        handle.clear();

        // After clear, blocks must be empty, active_roi must be reset, loading must be false, status set to "Очищено"
        let state = shared_state.lock().unwrap();
        assert!(
            state.current_blocks.is_empty(),
            "Cards must be cleared from overlay"
        );
        assert!(
            state.active_roi.is_none(),
            "Active ROI must be cleared back to full screen"
        );
        assert!(!state.is_loading, "is_loading must be false after clear");
        assert_eq!(state.status_msg.as_deref(), Some("Очищено"));
    }

    #[test]
    fn test_cycle_ocr_mode_preserves_roi_and_resets_signature() {
        use crate::capture::FrameSignature;
        use crate::overlay::{OverlayHandle, OverlaySharedState};
        use crate::types::{ScreenRect, TranslationBlock};
        use windows::Win32::Foundation::HWND;

        let shared_state = Arc::new(Mutex::new(OverlaySharedState {
            current_blocks: vec![TranslationBlock {
                original_text: "Text".to_string(),
                translated_text: "Текст".to_string(),
                rect: ScreenRect::new(20, 20, 150, 60),
            }],
            cursor_card: None,
            status_msg: Some("Ready".to_string()),
            is_loading: false,
            active_roi: Some(ScreenRect::new(10, 10, 200, 100)),
            auto_mode: true,
            provider_name: "Google Translate".to_string(),
            ocr_engine_name: "RapidOCR".to_string(),
        }));

        let overlay = OverlayHandle {
            hwnd: HWND(core::ptr::null_mut()),
            shared_state: shared_state.clone(),
        };

        let last_sig = Arc::new(Mutex::new(Some(FrameSignature::default())));

        // Cycle OCR mode logic: reset signature, clear blocks (without wiping ROI), update mode name
        *last_sig.lock().unwrap() = None;
        overlay.clear_blocks();
        overlay.set_ocr_engine_name("WinRT".to_string());

        assert_eq!(*last_sig.lock().unwrap(), None);

        let state = shared_state.lock().unwrap();
        assert!(state.current_blocks.is_empty(), "Cards must be cleared");
        assert_eq!(
            state.active_roi,
            Some(ScreenRect::new(10, 10, 200, 100)),
            "Active ROI must NOT be cleared when cycling OCR mode"
        );
        assert_eq!(state.ocr_engine_name, "WinRT");
    }

    #[test]
    fn test_clear_cursor_card_preserves_current_blocks() {
        use crate::overlay::{OverlayHandle, OverlaySharedState};
        use crate::types::{ScreenRect, TranslationBlock};
        use windows::Win32::Foundation::HWND;

        let shared_state = Arc::new(Mutex::new(OverlaySharedState {
            current_blocks: vec![TranslationBlock {
                original_text: "Background".to_string(),
                translated_text: "Фон".to_string(),
                rect: ScreenRect::new(0, 0, 100, 50),
            }],
            cursor_card: Some(TranslationBlock {
                original_text: "Word".to_string(),
                translated_text: "Слово".to_string(),
                rect: ScreenRect::new(10, 10, 80, 40),
            }),
            status_msg: Some("Translating under cursor".to_string()),
            is_loading: false,
            active_roi: None,
            auto_mode: false,
            provider_name: "Google Translate".to_string(),
            ocr_engine_name: "RapidOCR".to_string(),
        }));

        let overlay = OverlayHandle {
            hwnd: HWND(core::ptr::null_mut()),
            shared_state: shared_state.clone(),
        };

        overlay.clear_cursor_card();

        let state = shared_state.lock().unwrap();
        assert_eq!(state.current_blocks.len(), 1);
        assert_eq!(state.current_blocks[0].translated_text, "Фон");
        assert!(state.cursor_card.is_none());
        assert_eq!(state.status_msg.as_deref(), Some("Готов"));
    }

    #[test]
    fn test_processing_guard_raii_safety() {
        let flag = Arc::new(AtomicBool::new(false));

        {
            let guard = ProcessingGuard::try_acquire(&flag);
            assert!(guard.is_some());
            assert!(flag.load(Ordering::SeqCst));

            // Subsequent acquire fails while guard is active
            let second_guard = ProcessingGuard::try_acquire(&flag);
            assert!(second_guard.is_none());
        }

        // When guard drops, flag must be reset to false immediately
        assert!(!flag.load(Ordering::SeqCst));

        // Now can acquire again
        let third_guard = ProcessingGuard::try_acquire(&flag);
        assert!(third_guard.is_some());
        assert!(flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_high_02_cancellable_task_waits_for_old_task_to_release_guard() {
        let flag = Arc::new(AtomicBool::new(false));
        let task_slot: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));

        let flag_clone = flag.clone();
        let old_task_started = Arc::new(tokio::sync::Notify::new());
        let old_task_started_clone = old_task_started.clone();

        struct SlowGuard {
            _guard: ProcessingGuard,
        }
        impl Drop for SlowGuard {
            fn drop(&mut self) {
                // Simulates teardown latency (e.g. closing network stream/D3D textures)
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        }

        // Start old task that acquires the ProcessingGuard with teardown latency
        let old_handle = tokio::spawn(async move {
            let guard = ProcessingGuard::try_acquire(&flag_clone).expect("old task acquires guard");
            let slow_guard = SlowGuard { _guard: guard };
            old_task_started_clone.notify_one();
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            drop(slow_guard);
        });

        *task_slot.lock().unwrap() = Some(old_handle);

        // Wait until old task has actually acquired the guard and is sleeping
        old_task_started.notified().await;
        assert!(
            flag.load(Ordering::SeqCst),
            "Flag must be true while old task runs"
        );

        // Flag to record whether new task executed successfully
        let new_task_ran = Arc::new(AtomicBool::new(false));
        let new_task_ran_clone = new_task_ran.clone();
        let flag_for_new = flag.clone();

        // New dispatch logic (spawn_cancellable_task awaits prev.abort()):
        spawn_cancellable_task(&task_slot, async move {
            if let Some(_guard) = ProcessingGuard::acquire_or_wait(
                &flag_for_new,
                std::time::Duration::from_millis(500),
            )
            .await
            {
                new_task_ran_clone.store(true, Ordering::SeqCst);
            }
        });

        // Await the new task
        let new_handle = task_slot.lock().unwrap().take().unwrap();
        new_handle.await.unwrap();

        assert!(
            new_task_ran.load(Ordering::SeqCst),
            "New task must have acquired guard and executed after cancelling old task"
        );
        assert!(
            !flag.load(Ordering::SeqCst),
            "Flag must be released upon task completion"
        );
    }

    #[tokio::test]
    async fn test_high_02_rapid_dispatches_do_not_accumulate_or_deadlock() {
        let flag = Arc::new(AtomicBool::new(false));
        let task_slot: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));
        let completed_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Rapidly dispatch 20 tasks
        for _ in 0..20 {
            let flag_clone = flag.clone();
            let counter_clone = completed_counter.clone();
            spawn_cancellable_task(&task_slot, async move {
                if let Some(_guard) = ProcessingGuard::acquire_or_wait(
                    &flag_clone,
                    std::time::Duration::from_millis(500),
                )
                .await
                {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            });
        }

        // Await the final task
        let final_handle = task_slot.lock().unwrap().take().unwrap();
        final_handle.await.unwrap();

        // Exactly one (or very few if early ones ran briefly) executed, not all 20
        let count = completed_counter.load(Ordering::SeqCst);
        assert!(count >= 1, "At least the final task must execute");
        assert!(
            count < 20,
            "Tasks must not queue up sequentially without cancellation"
        );
        assert!(
            !flag.load(Ordering::SeqCst),
            "Flag must be false at the end"
        );
    }

    #[tokio::test]
    async fn test_spawn_cancellable_task_fast_finish_and_reentrancy() {
        let task_slot: Arc<Mutex<Option<JoinHandle<()>>>> = Arc::new(Mutex::new(None));
        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();
        let slot_clone = task_slot.clone();

        // Task that immediately accesses task_slot to ensure no deadlock during execution
        spawn_cancellable_task(&task_slot, async move {
            let is_some = slot_clone.lock().unwrap().is_some();
            let _ = is_some;
            executed_clone.store(true, Ordering::SeqCst);
        });

        let handle = task_slot.lock().unwrap().take();
        assert!(handle.is_some(), "Handle must be stored in slot");
        handle.unwrap().await.unwrap();
        assert!(executed.load(Ordering::SeqCst));
    }

    #[test]
    fn cursor_capture_keeps_a_tall_manhwa_paragraph_from_its_first_and_last_lines() {
        let paragraph = ScreenRect::new(154, 348, 488, 279);
        for pt in [POINT { x: 300, y: 370 }, POINT { x: 400, y: 605 }] {
            let capture = cursor_capture_roi(pt, None, 0, 0, 1920, 1080);
            assert!(capture.contains(paragraph.x, paragraph.y));
            assert!(capture.contains(paragraph.x + paragraph.width - 1,
                paragraph.y + paragraph.height - 1), "cursor capture truncated the utterance: {capture:?}");
        }
    }

    #[test]
    fn test_cursor_capture_roi_multi_monitor_bounds() {
        // Primary monitor: pt=(500, 300), min=(0, 0), max=(1920, 1080)
        let pt_primary = POINT { x: 500, y: 300 };
        let roi_primary = cursor_capture_roi(pt_primary, None, 0, 0, 1920, 1080);
        assert_eq!(roi_primary.x, 0);
        assert_eq!(roi_primary.y, 0);
        assert_eq!(roi_primary.width, 1000);
        assert_eq!(roi_primary.height, 800);

        // Right edge of primary monitor: pt=(1850, 300) -> must clamp cap_x to 1920 - 1000 = 920
        let pt_right = POINT { x: 1850, y: 300 };
        let roi_right = cursor_capture_roi(pt_right, None, 0, 0, 1920, 1080);
        assert_eq!(roi_right.x, 920);
        assert_eq!(roi_right.y, 0);
        assert_eq!(roi_right.x + roi_right.width, 1920);

        // Secondary monitor to the left: pt=(-1000, 200), min=(-1920, 0), max=(0, 1080)
        let pt_left = POINT { x: -1000, y: 200 };
        let roi_left = cursor_capture_roi(pt_left, None, -1920, 0, 0, 1080);
        assert_eq!(roi_left.x, -1500);
        assert_eq!(roi_left.y, 0);

        // Secondary monitor to the top-left: pt=(-1800, -500), min=(-1920, -1080), max=(0, 0)
        let pt_topleft = POINT { x: -1800, y: -500 };
        let roi_topleft = cursor_capture_roi(pt_topleft, None, -1920, -1080, 0, 0);
        assert_eq!(roi_topleft.x, -1920); // clamped to min_x
        assert_eq!(roi_topleft.y, -900);

        // If manual active ROI contains pt, it takes precedence
        let active = ScreenRect::new(-1200, 100, 400, 300);
        let roi_manual = cursor_capture_roi(pt_left, Some(active), -1920, 0, 0, 1080);
        assert_eq!(roi_manual, active);
    }

    #[tokio::test]
    async fn test_auto_mode_signature_not_overwritten_when_busy() {
        let is_processing = Arc::new(AtomicBool::new(true)); // Worker is currently busy
        let last_signature: Arc<Mutex<Option<FrameSignature>>> = Arc::new(Mutex::new(None));

        // When perform_translation is called while busy, acquire_or_wait returns None
        // and last_signature must remain None (not prematurely updated).
        let guard_opt = ProcessingGuard::acquire_or_wait(&is_processing, std::time::Duration::from_millis(50)).await;
        assert!(guard_opt.is_none());
        assert_eq!(*last_signature.lock().unwrap(), None);
    }

    #[test]
    fn test_auto_ocr_fallback_predicates() {
        // RapidOCR returns valid dialogue text
        let valid_block = OcrBlock {
            text: "Hello, traveler!".to_string(),
            rect: ScreenRect::new(10, 10, 100, 30),
            lines: Vec::new(),
        };
        assert!(is_helper_result_valid(&[valid_block]));

        // RapidOCR returns empty blocks
        assert!(!is_helper_result_valid(&[]));

        // RapidOCR returns only noise/garbage
        let garbage_block = OcrBlock {
            text: "...".to_string(),
            rect: ScreenRect::new(10, 10, 30, 15),
            lines: Vec::new(),
        };
        assert!(!is_helper_result_valid(&[garbage_block]));
    }

    #[test]
    fn test_is_suspicious_word() {
        // Подозрительные слова (артефакты OCR)
        assert!(is_suspicious_word("rrOTtcTION"));
        assert!(is_suspicious_word("PFOTtCTION"));
        assert!(is_suspicious_word("1ONG"));
        assert!(is_suspicious_word("4EASE"));
        assert!(is_suspicious_word("•PA"));
        assert!(is_suspicious_word("I@II#O*$"));

        // Корректные слова (не должны считаться подозрительными)
        assert!(!is_suspicious_word("PROTECTION"));
        assert!(!is_suspicious_word("Hello"));
        assert!(!is_suspicious_word("I"));
        assert!(!is_suspicious_word("of"));
        assert!(!is_suspicious_word("GUNGNIR'S"));
        assert!(!is_suspicious_word("2nd"));
        assert!(!is_suspicious_word("3D"));
        assert!(!is_suspicious_word("café"));
        assert!(!is_suspicious_word("naïve"));
        assert!(!is_suspicious_word("Привет"));
    }

    #[tokio::test]
    async fn test_cancelling_unpolled_replacement_also_cancels_original() {
        let completed = Arc::new(AtomicBool::new(false));
        let marker = completed.clone();
        let slot = Arc::new(Mutex::new(Some(tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            marker.store(true, Ordering::SeqCst);
        }))));
        tokio::task::yield_now().await;
        spawn_cancellable_task(&slot, async {});
        let replacement = slot.lock().unwrap().take().unwrap();
        replacement.abort();
        let _ = replacement.await;
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        assert!(!completed.load(Ordering::SeqCst));
    }

}
