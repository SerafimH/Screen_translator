use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::capture::{FrameSignature, ScreenCapture};
use crate::ocr::{WinrtOcrEngine, is_ocr_garbage};
use crate::overlay::{AppEvent, OverlayHandle, OverlayManager};
use crate::translator::TranslatorService;
use crate::types::{AppConfig, ScreenRect, TranslationProvider};

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
    ocr: Arc<WinrtOcrEngine>,
    translator: Arc<TranslatorService>,
    active_roi: Arc<Mutex<Option<ScreenRect>>>,
    auto_mode: Arc<AtomicBool>,
    is_processing: Arc<AtomicBool>,
    current_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

pub struct App {
    state: AppState,
    event_rx: UnboundedReceiver<AppEvent>,
}

impl App {
    pub fn new(config: AppConfig) -> Result<Self, String> {
        let (overlay, event_rx) =
            OverlayManager::spawn().map_err(|e| format!("Failed to spawn overlay: {:?}", e))?;

        let ocr = Arc::new(
            WinrtOcrEngine::new().map_err(|e| format!("Failed to init OCR engine: {}", e))?,
        );
        let translator = Arc::new(TranslatorService::new());

        overlay.set_provider_name(config.provider.display_name().to_string());

        let state = AppState {
            config: Arc::new(Mutex::new(config)),
            overlay,
            ocr,
            translator,
            active_roi: Arc::new(Mutex::new(None)),
            auto_mode: Arc::new(AtomicBool::new(false)),
            is_processing: Arc::new(AtomicBool::new(false)),
            current_task: Arc::new(Mutex::new(None)),
        };

        Ok(Self { state, event_rx })
    }

    pub async fn run(mut self) {
        info!("Screen Translator application running.");
        info!(
            "Hotkeys: Alt+T (Translate), Alt+S (Select ROI), Alt+A (Auto), Alt+P (Engine), Alt+C (Clear), Alt+Q (Quit)"
        );

        // Spawn background Dialogue Watcher (Auto-Mode)
        let auto_state = self.state.clone();
        tokio::spawn(async move {
            let mut last_signature = FrameSignature::default();
            let interval_ms = {
                let cfg = auto_state.config.lock().unwrap_or_else(|e| e.into_inner());
                cfg.auto_mode_interval_ms
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
                    ScreenCapture::capture_primary(current_roi)
                })
                .await;

                let frame = match frame_res {
                    Ok(Ok(f)) => f,
                    _ => continue,
                };

                let sig = FrameSignature::from_bgra(&frame.bgra_bytes, frame.width, frame.height);
                if !sig.has_changed(&last_signature) {
                    continue;
                }
                last_signature = sig;

                let task_state = auto_state.clone();
                task_state.dispatch_translation_task();
            }
        });

        // Event listener loop
        while let Some(event) = self.event_rx.recv().await {
            match event {
                AppEvent::TriggerTranslate => {
                    self.state.dispatch_translation_task();
                }

                AppEvent::RoiSelected(rect) => {
                    info!("ROI selected: {:?}", rect);
                    *self
                        .state
                        .active_roi
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(rect);
                    self.state.overlay.set_active_roi(Some(rect));
                    self.state.dispatch_translation_task();
                }

                AppEvent::ToggleAutoMode => {
                    let new_val = !self.state.auto_mode.load(Ordering::Relaxed);
                    self.state.auto_mode.store(new_val, Ordering::Relaxed);
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
                    self.state.overlay.clear();
                    info!("Overlay cleared.");
                }

                AppEvent::Quit => {
                    info!("Quitting application.");
                    break;
                }
            }
        }
    }
}

impl AppState {
    fn dispatch_translation_task(&self) {
        // Cancel any pending/outdated translation task
        if let Some(prev) = self
            .current_task
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            prev.abort();
        }

        let state = self.clone();
        let handle = tokio::spawn(async move {
            state.perform_translation().await;
        });

        *self.current_task.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
    }

    async fn perform_translation(&self) {
        let Some(_guard) = ProcessingGuard::try_acquire(&self.is_processing) else {
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

        self.overlay
            .set_loading(true, Some("Захват экрана (D3D11 GPU)...".to_string()));
        info!("Capturing frame (ROI: {:?})...", roi);

        let capture_res =
            tokio::task::spawn_blocking(move || ScreenCapture::capture_primary(roi)).await;

        let frame = match capture_res {
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
        };

        self.overlay.set_loading(
            true,
            Some("Распознавание текста (WinRT OCR)...".to_string()),
        );
        info!(
            "Captured {}x{} frame. Running OCR...",
            frame.width, frame.height
        );

        let blocks = match self.ocr.recognize(&frame).await {
            Ok(b) => b,
            Err(e) => {
                error!("OCR error: {}", e);
                self.overlay
                    .update_blocks(Vec::new(), Some(format!("Ошибка OCR: {}", e)));
                return;
            }
        };

        let has_text = !blocks.is_empty() && blocks.iter().any(|b| !is_ocr_garbage(&b.text));

        // Direct Vision OCR fallback: if WinRT OCR detected nothing or pure garbage and vision is enabled
        if !has_text
            && cfg.use_vision_ocr
            && (cfg.provider == TranslationProvider::Gemini
                || cfg.provider == TranslationProvider::OpenAICompatible)
        {
            info!("WinRT OCR returned no text; triggering Direct Vision OCR fallback...");
            self.overlay.set_loading(
                true,
                Some("Активация нейросетевого Vision OCR...".to_string()),
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
                        self.overlay.update_blocks(vision_blocks, Some(status));
                        return;
                    }
                }
                Err(err) => {
                    warn!("Vision OCR fallback failed: {}", err);
                }
            }
        }

        if blocks.is_empty() {
            let elapsed = start_time.elapsed().as_millis();
            self.overlay.update_blocks(
                Vec::new(),
                Some(format!("Текст не обнаружен ({} мс)", elapsed)),
            );
            return;
        }

        self.overlay.set_loading(
            true,
            Some(format!("Перевод {} блоков текста...", blocks.len())),
        );

        let translations = self.translator.translate_blocks(&blocks, &cfg).await;

        let elapsed = start_time.elapsed().as_millis();
        let status = format!("Переведено {} блоков за {} мс", translations.len(), elapsed);
        info!("{}", status);

        self.overlay.update_blocks(translations, Some(status));
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
            active_roi: None,
            auto_mode: false,
            provider_name: "Test".to_string(),
        }));

        let handle = OverlayHandle {
            hwnd: HWND(core::ptr::null_mut()),
            shared_state: shared_state.clone(),
        };

        // Before clear, blocks are present and loading is true
        assert_eq!(shared_state.lock().unwrap().current_blocks.len(), 1);
        assert!(shared_state.lock().unwrap().is_loading);

        // When clear() is called (as required by BUG-9 on AppEvent::Clear):
        handle.clear();

        // After clear, blocks must be empty, loading must be false, status set to "Очищено"
        let state = shared_state.lock().unwrap();
        assert!(
            state.current_blocks.is_empty(),
            "Cards must be cleared from overlay"
        );
        assert!(!state.is_loading, "is_loading must be false after clear");
        assert_eq!(state.status_msg.as_deref(), Some("Очищено"));
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
}
