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
    score_recognized_lines,
};
use crate::overlay::{AppEvent, OverlayHandle, OverlayManager};
use crate::translator::TranslatorService;
use crate::types::{
    AppConfig, OcrBlock, OcrMode, ScreenRect, TranslationBlock, find_text_under_cursor,
};

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
    current_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

/// Checks whether WinRT Pass 1 yielded confident OCR text.
pub fn is_pass1_confident(pass1_score: usize, pass1_lines_count: usize) -> bool {
    pass1_score >= 12 && pass1_lines_count > 0
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

            loop {
                interval.tick().await;

                if !auto_state.auto_mode.load(Ordering::Relaxed) {
                    continue;
                }

                if auto_state.is_processing.load(Ordering::Relaxed) {
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

                let (frame, sig) = match frame_res {
                    Ok(Ok(pair)) => pair,
                    _ => continue,
                };

                {
                    let last_sig_lock = auto_state
                        .last_signature
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(prev) = last_sig_lock.as_ref()
                        && !sig.has_changed(prev)
                    {
                        continue;
                    }
                }

                let task_state = auto_state.clone();
                task_state.dispatch_translation_task(Some(frame));
            }
        });

        // Event listener loop
        while let Some(event) = self.event_rx.recv().await {
            match event {
                AppEvent::TriggerTranslate => {
                    self.state.dispatch_translation_task(None);
                }

                AppEvent::RoiSelected(rect) => {
                    info!("ROI selected: {:?}", rect);
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
                    *self
                        .state
                        .active_roi
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = None;
                    self.state.overlay.set_active_roi(None);
                    info!("ROI reset to full screen.");
                }

                AppEvent::TranslateUnderCursor(pt) => {
                    self.state.dispatch_cursor_translation_task(pt);
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
                    self.state.overlay.clear_blocks();
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
    let prev = {
        let mut slot_lock = task_slot.lock().unwrap_or_else(|e| e.into_inner());
        slot_lock.take()
    };

    let handle = tokio::spawn(async move {
        if let Some(prev) = prev {
            prev.abort();
            let _ = prev.await;
        }
        fut.await;
    });

    let mut slot_lock = task_slot.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(newer) = slot_lock.take() {
        handle.abort();
        *slot_lock = Some(newer);
    } else {
        *slot_lock = Some(handle);
    }
}

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
        let cap_w = 800.min((max_x - min_x).max(1));
        let cap_h = 240.min((max_y - min_y).max(1));

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
        // otherwise, capture a focused region (800x240) centered on the cursor clamped to virtual screen coordinates.
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

        let blocks = match self.perform_ocr(&frame, &cfg).await {
            Ok(b) => b,
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

        let Some(target) = find_text_under_cursor(&blocks, pt.x, pt.y) else {
            info!("No text found under cursor at ({}, {})", pt.x, pt.y);
            self.overlay.set_loading(false, Some("Готов".to_string()));
            return;
        };

        let is_multi_word = target
            .word_text
            .as_ref()
            .map(|w| w.to_lowercase() != target.sentence_text.to_lowercase())
            .unwrap_or(false)
            && target.sentence_text.split_whitespace().count() > 1;

        let translation_block = if is_multi_word {
            let word = target.word_text.as_ref().unwrap();
            let word_fut =
                self.translator
                    .lookup_word_details(word, &cfg.source_lang, &cfg.target_lang, &cfg);
            let sentence_fut = self.translator.translate_text(
                &target.sentence_text,
                &cfg.source_lang,
                &cfg.target_lang,
                &cfg,
            );
            let (details, sentence_res) = tokio::join!(word_fut, sentence_fut);

            let sentence_trans = sentence_res.unwrap_or_else(|_| target.sentence_text.clone());

            let mut card_text = format!("Слово: {} → {}", word, details.primary);
            if !details.synonyms.is_empty() {
                card_text.push_str(&format!(" ({})", details.synonyms.join(", ")));
            }

            if !details.examples.is_empty() {
                card_text.push_str("\n\nПримеры:");
                for ex in &details.examples {
                    card_text.push_str(&format!("\n• {}", ex));
                }
            }

            card_text.push_str("\n\nВ контексте предложения:\n«");
            card_text.push_str(&target.sentence_text);
            card_text.push_str("»\n→ «");
            card_text.push_str(&sentence_trans);
            card_text.push('»');

            let card_w = target.sentence_rect.width.max(420);
            TranslationBlock {
                original_text: format!("{} ({})", word, target.sentence_text),
                translated_text: card_text,
                rect: ScreenRect::new(
                    target.sentence_rect.x,
                    target.sentence_rect.y,
                    card_w,
                    target.sentence_rect.height,
                ),
            }
        } else {
            let word_or_line = target.word_text.as_ref().unwrap_or(&target.sentence_text);

            let details = self
                .translator
                .lookup_word_details(word_or_line, &cfg.source_lang, &cfg.target_lang, &cfg)
                .await;

            let mut card_text = format!("Слово: {} → {}", word_or_line, details.primary);
            if !details.synonyms.is_empty() {
                card_text.push_str(&format!(" ({})", details.synonyms.join(", ")));
            }
            if !details.examples.is_empty() {
                card_text.push_str("\n\nПримеры:");
                for ex in &details.examples {
                    card_text.push_str(&format!("\n• {}", ex));
                }
            }

            let card_w = target
                .word_rect
                .map(|r| r.width)
                .unwrap_or(target.sentence_rect.width)
                .max(360);
            let base_rect = target.word_rect.unwrap_or(target.sentence_rect);

            TranslationBlock {
                original_text: word_or_line.clone(),
                translated_text: card_text,
                rect: ScreenRect::new(base_rect.x, base_rect.y, card_w, base_rect.height),
            }
        };

        if !crate::overlay::is_shift_held() {
            info!("Shift was released before translation finished, skipping card display.");
            self.overlay.set_loading(false, Some("Готов".to_string()));
            return;
        }

        let elapsed = start_time.elapsed().as_millis();
        let status = format!("Перевод под курсором за {} мс", elapsed);
        info!("{}", status);
        self.overlay
            .update_blocks(vec![translation_block], Some(status));
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
        let sig = FrameSignature::from_bgra(&frame.bgra_bytes, frame.width, frame.height);

        let blocks = match self.perform_ocr(&frame, &cfg).await {
            Ok(b) => b,
            Err(e) => {
                error!("OCR error: {}", e);
                *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);
                self.overlay
                    .update_blocks(Vec::new(), Some(format!("Ошибка OCR: {}", e)));
                return;
            }
        };

        let has_text = !blocks.is_empty() && blocks.iter().any(|b| !is_ocr_garbage(&b.text));

        let has_vision_support = cfg.gemini_api_key.is_some() || cfg.openai_api_key.is_some();

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
                        let elapsed = start_time.elapsed().as_millis();
                        let status = format!(
                            "Vision OCR: {} блоков за {} мс",
                            vision_blocks.len(),
                            elapsed
                        );
                        info!("{}", status);
                        *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);
                        self.overlay.update_blocks(vision_blocks, Some(status));
                        return;
                    }
                }
                Err(err) => {
                    warn!("Vision OCR fallback failed: {}", err);
                }
            }
        }

        if !has_text {
            let elapsed = start_time.elapsed().as_millis();
            let status = if !cfg.vision_fallback {
                format!(
                    "Текст не найден ({} мс). Для сложных картинок включите vision_fallback = true в config.toml",
                    elapsed
                )
            } else {
                format!("Текст не найден ({} мс)", elapsed)
            };
            *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);
            self.overlay.update_blocks(Vec::new(), Some(status));
            return;
        }

        let valid_blocks: Vec<OcrBlock> = blocks
            .into_iter()
            .filter(|b| !is_ocr_garbage(&b.text))
            .collect();

        self.overlay.set_loading(
            true,
            Some(format!("Перевод {} блоков текста...", valid_blocks.len())),
        );

        let translations = self.translator.translate_blocks(&valid_blocks, &cfg).await;

        let elapsed = start_time.elapsed().as_millis();
        let status = format!("Переведено {} блоков за {} мс", translations.len(), elapsed);
        info!("{}", status);

        *self.last_signature.lock().unwrap_or_else(|e| e.into_inner()) = Some(sig);

        self.overlay.update_blocks(translations, Some(status));
    }

    async fn perform_ocr(
        &self,
        frame: &CapturedFrame,
        cfg: &AppConfig,
    ) -> Result<Vec<OcrBlock>, String> {
        match cfg.ocr_engine {
            OcrMode::WinRt => {
                self.overlay.set_loading(
                    true,
                    Some("Распознавание текста (WinRT OCR)...".to_string()),
                );
                if let Some(ocr) = &self.ocr {
                    ocr.recognize(frame).await
                } else {
                    Err("Системный WinRT OCR недоступен. Установите языковой пакет в Windows или выберите RapidOCR.".to_string())
                }
            }
            OcrMode::Helper => {
                self.overlay
                    .set_loading(true, Some("Распознавание текста (RapidOCR)...".to_string()));
                match self.helper_ocr.recognize(frame, &cfg.ocr_lang).await {
                    Ok(blocks) => Ok(blocks),
                    Err(err) => {
                        warn!("RapidOCR helper failed: {}. Falling back to WinRT...", err);
                        if let Some(ocr) = &self.ocr {
                            self.overlay.set_loading(
                                true,
                                Some("RapidOCR недоступен, откат на WinRT OCR...".to_string()),
                            );
                            ocr.recognize(frame).await
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
                let Some(ocr) = &self.ocr else {
                    // WinRT is not available, directly use RapidOCR helper
                    return self.helper_ocr.recognize(frame, &cfg.ocr_lang).await;
                };

                self.overlay.set_loading(
                    true,
                    Some("Распознавание (Auto: WinRT Pass 1)...".to_string()),
                );
                let pass1_lines = ocr.recognize_pass_1(frame).await.unwrap_or_default();
                let pass1_score = score_recognized_lines(&pass1_lines);

                if is_pass1_confident(pass1_score, pass1_lines.len()) {
                    info!(
                        "Auto mode: WinRT Pass 1 succeeded with high score ({})",
                        pass1_score
                    );
                    let blocks = cluster_lines_into_blocks(
                        pass1_lines,
                        Some((
                            &frame.bgra_bytes,
                            frame.width,
                            frame.height,
                            frame.offset_x,
                            frame.offset_y,
                        )),
                    );
                    Ok(blocks)
                } else {
                    info!(
                        "Auto mode: WinRT Pass 1 score is low ({}), trying RapidOCR helper...",
                        pass1_score
                    );
                    self.overlay
                        .set_loading(true, Some("Auto: пробую RapidOCR помощник...".to_string()));
                    match self.helper_ocr.recognize(frame, &cfg.ocr_lang).await {
                        Ok(blocks) if is_helper_result_valid(&blocks) => {
                            info!(
                                "Auto mode: RapidOCR helper succeeded with {} blocks",
                                blocks.len()
                            );
                            Ok(blocks)
                        }
                        _ => {
                            info!(
                                "Auto mode: RapidOCR yielded no text, falling back to full WinRT multi-pass..."
                            );
                            ocr.recognize(frame).await
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_cursor_capture_roi_multi_monitor_bounds() {
        // Primary monitor: pt=(500, 300), min=(0, 0), max=(1920, 1080)
        let pt_primary = POINT { x: 500, y: 300 };
        let roi_primary = cursor_capture_roi(pt_primary, None, 0, 0, 1920, 1080);
        assert_eq!(roi_primary.x, 100);
        assert_eq!(roi_primary.y, 180);
        assert_eq!(roi_primary.width, 800);
        assert_eq!(roi_primary.height, 240);

        // Right edge of primary monitor: pt=(1850, 300) -> must clamp cap_x to 1920 - 800 = 1120
        let pt_right = POINT { x: 1850, y: 300 };
        let roi_right = cursor_capture_roi(pt_right, None, 0, 0, 1920, 1080);
        assert_eq!(roi_right.x, 1120);
        assert_eq!(roi_right.y, 180);
        assert_eq!(roi_right.x + roi_right.width, 1920);

        // Secondary monitor to the left: pt=(-1000, 200), min=(-1920, 0), max=(0, 1080)
        let pt_left = POINT { x: -1000, y: 200 };
        let roi_left = cursor_capture_roi(pt_left, None, -1920, 0, 0, 1080);
        assert_eq!(roi_left.x, -1400);
        assert_eq!(roi_left.y, 80);

        // Secondary monitor to the top-left: pt=(-1800, -500), min=(-1920, -1080), max=(0, 0)
        let pt_topleft = POINT { x: -1800, y: -500 };
        let roi_topleft = cursor_capture_roi(pt_topleft, None, -1920, -1080, 0, 0);
        assert_eq!(roi_topleft.x, -1920); // clamped to min_x
        assert_eq!(roi_topleft.y, -620);

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
        // High score Pass 1 with lines
        assert!(is_pass1_confident(15, 2));
        // Exact threshold boundary: score 12 with lines
        assert!(is_pass1_confident(12, 1));
        // Low score Pass 1 (< 12)
        assert!(!is_pass1_confident(11, 2));
        // Zero lines with high score cannot be accepted
        assert!(!is_pass1_confident(25, 0));

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
}
