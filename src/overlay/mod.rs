#![allow(unsafe_op_in_unsafe_fn)]

pub mod renderer;

use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tracing::{error, info, warn};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    HOT_KEY_MODIFIERS, MOD_ALT, MOD_NOREPEAT, RegisterHotKey, ReleaseCapture, SetCapture, SetFocus,
    UnregisterHotKey,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, FindWindowW, GWL_EXSTYLE,
    GWLP_USERDATA, GetCursorPos, GetForegroundWindow, GetMessageW, GetSystemMetrics,
    GetWindowLongPtrW, GetWindowThreadProcessId, HWND_TOPMOST, IDC_CROSS, LoadCursorW, MSG,
    PostMessageW, PostQuitMessage, RegisterClassW, SET_WINDOW_POS_FLAGS, SM_CXSCREEN,
    SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
    SMTO_ABORTIFHUNG, SW_SHOW, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_SHOWWINDOW,
    SendMessageTimeoutW, SetCursor, SetForegroundWindow, SetWindowDisplayAffinity,
    SetWindowLongPtrW, SetWindowPos, ShowWindow, TranslateMessage, WDA_EXCLUDEFROMCAPTURE, WM_APP,
    WM_CLOSE, WM_DESTROY, WM_HOTKEY, WM_KEYDOWN, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE,
    WM_NCDESTROY, WM_NCHITTEST, WM_SETCURSOR, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE,
    WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};
use windows::core::{Result, w};

use crate::overlay::renderer::{OverlayRenderer, RenderScene};
use crate::types::{AppMode, ScreenRect, TranslationBlock};

pub const HOTKEY_TRANSLATE: i32 = 101;
pub const HOTKEY_SELECT_ROI: i32 = 102;
pub const HOTKEY_TOGGLE_AUTO: i32 = 103;
pub const HOTKEY_CLEAR: i32 = 104;
pub const HOTKEY_QUIT: i32 = 105;
pub const HOTKEY_CYCLE_PROVIDER: i32 = 106;

pub const WM_APP_ENTER_SELECTION: u32 = WM_APP + 1;
pub const WM_APP_UPDATE_STATE: u32 = WM_APP + 2;
pub const WM_APP_CLEAR: u32 = WM_APP + 3;

const HTCLIENT: isize = 1;

#[derive(Debug, Clone)]
pub enum AppEvent {
    TriggerTranslate,
    RoiSelected(ScreenRect),
    ToggleAutoMode,
    CycleProvider,
    Clear,
    Quit,
}

#[derive(Debug, Clone)]
pub struct OverlaySharedState {
    pub current_blocks: Vec<TranslationBlock>,
    pub status_msg: Option<String>,
    pub is_loading: bool,
    pub active_roi: Option<ScreenRect>,
    pub auto_mode: bool,
    pub provider_name: String,
}

impl Default for OverlaySharedState {
    fn default() -> Self {
        Self {
            current_blocks: Vec::new(),
            status_msg: Some("Готов - нажмите Alt+T для перевода".to_string()),
            is_loading: false,
            active_roi: None,
            auto_mode: false,
            provider_name: "Google Translate".to_string(),
        }
    }
}

pub struct OverlayHandle {
    pub(crate) hwnd: HWND,
    pub(crate) shared_state: Arc<Mutex<OverlaySharedState>>,
}

unsafe impl Send for OverlayHandle {}
unsafe impl Sync for OverlayHandle {}

impl Clone for OverlayHandle {
    fn clone(&self) -> Self {
        Self {
            hwnd: self.hwnd,
            shared_state: self.shared_state.clone(),
        }
    }
}

impl OverlayHandle {
    pub fn update_blocks(&self, blocks: Vec<TranslationBlock>, status: Option<String>) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.current_blocks = blocks;
            s.status_msg = status;
            s.is_loading = false;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn set_loading(&self, loading: bool, status: Option<String>) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.is_loading = loading;
            if let Some(msg) = status {
                s.status_msg = Some(msg);
            }
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn set_active_roi(&self, roi: Option<ScreenRect>) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.active_roi = roi;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn set_auto_mode(&self, enabled: bool) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.auto_mode = enabled;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn set_provider_name(&self, name: String) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.provider_name = name;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn start_roi_selection(&self) {
        unsafe {
            let _ = PostMessageW(
                Some(self.hwnd),
                WM_APP_ENTER_SELECTION,
                WPARAM(0),
                LPARAM(0),
            );
        }
    }

    pub fn clear(&self) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.current_blocks.clear();
            s.status_msg = Some("Очищено".to_string());
            s.is_loading = false;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_CLEAR, WPARAM(0), LPARAM(0));
        }
    }
}

struct WindowThreadState {
    hwnd: HWND,
    renderer: OverlayRenderer,
    mode: AppMode,
    virtual_x: i32,
    virtual_y: i32,
    virtual_w: i32,
    virtual_h: i32,
    // Selection drag state
    is_dragging: bool,
    drag_start: POINT,
    drag_current: POINT,
    // Shared state from worker
    shared_state: Arc<Mutex<OverlaySharedState>>,
    // Channel to send events to App
    event_tx: UnboundedSender<AppEvent>,
}

pub struct OverlayManager;

impl OverlayManager {
    /// Launches the Win32 transparent overlay window on a dedicated message pump thread.
    /// Closes any previous instance window and applies WS_EX_LAYERED + WDA_EXCLUDEFROMCAPTURE.
    pub fn spawn() -> Result<(OverlayHandle, UnboundedReceiver<AppEvent>)> {
        let (event_tx, event_rx) = unbounded_channel();
        let (init_tx, init_rx) =
            std::sync::mpsc::channel::<std::result::Result<isize, windows::core::Error>>();

        let shared_state = Arc::new(Mutex::new(OverlaySharedState::default()));
        let shared_for_thread = shared_state.clone();

        std::thread::spawn(move || {
            unsafe {
                let hinstance = match GetModuleHandleW(None) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                let class_name = w!("GameScreenTranslatorClass");

                // 1. Close any previous instance window to avoid ERROR_HOTKEY_ALREADY_REGISTERED
                if let Ok(old_hwnd) = FindWindowW(class_name, None)
                    && !old_hwnd.0.is_null()
                {
                    warn!(
                        "Found existing Screen Translator window {:?}, closing old instance...",
                        old_hwnd
                    );
                    let _ = SendMessageTimeoutW(
                        old_hwnd,
                        WM_CLOSE,
                        WPARAM(0),
                        LPARAM(0),
                        SMTO_ABORTIFHUNG,
                        500,
                        None,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }

                // Query full multi-monitor virtual screen bounds
                let mut virtual_x = GetSystemMetrics(SM_XVIRTUALSCREEN);
                let mut virtual_y = GetSystemMetrics(SM_YVIRTUALSCREEN);
                let mut virtual_w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
                let mut virtual_h = GetSystemMetrics(SM_CYVIRTUALSCREEN);

                if virtual_w <= 0 || virtual_h <= 0 {
                    virtual_x = 0;
                    virtual_y = 0;
                    virtual_w = GetSystemMetrics(SM_CXSCREEN);
                    virtual_h = GetSystemMetrics(SM_CYSCREEN);
                }

                let cross_cursor = LoadCursorW(None, IDC_CROSS).unwrap_or_default();

                let wc = WNDCLASSW {
                    lpfnWndProc: Some(overlay_wndproc),
                    hInstance: hinstance.into(),
                    lpszClassName: class_name,
                    hCursor: cross_cursor,
                    ..Default::default()
                };
                RegisterClassW(&wc);

                // Initial styles: WS_POPUP | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE
                let ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE;
                let hwnd = match CreateWindowExW(
                    ex_style,
                    class_name,
                    w!("GameScreenTranslatorOverlay"),
                    WS_POPUP,
                    virtual_x,
                    virtual_y,
                    virtual_w,
                    virtual_h,
                    None,
                    None,
                    Some(hinstance.into()),
                    None,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                // Exclude window from screen capture APIs (physical invisibility for DWM compositor)
                if let Err(e) = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) {
                    error!("Failed to set WDA_EXCLUDEFROMCAPTURE: {:?}", e);
                } else {
                    info!(
                        "WDA_EXCLUDEFROMCAPTURE applied successfully (overlay invisible to capture)"
                    );
                }

                // Register global hotkeys
                let mod_flags = HOT_KEY_MODIFIERS(MOD_ALT.0 | MOD_NOREPEAT.0);
                let mut hotkey_warning: Option<String> = None;
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_TRANSLATE, mod_flags, 0x54) {
                    error!("Failed to register Alt+T (Translate): {:?}", e);
                    hotkey_warning =
                        Some("Предупреждение: Alt+T занят другой программой".to_string());
                }
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_SELECT_ROI, mod_flags, 0x53) {
                    error!("Failed to register Alt+S (Select ROI): {:?}", e);
                    if hotkey_warning.is_none() {
                        hotkey_warning =
                            Some("Предупреждение: Alt+S занят другой программой".to_string());
                    }
                }
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_TOGGLE_AUTO, mod_flags, 0x41) {
                    error!("Failed to register Alt+A (Toggle Auto): {:?}", e);
                }
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_CYCLE_PROVIDER, mod_flags, 0x50) {
                    error!("Failed to register Alt+P (Cycle Provider): {:?}", e);
                }
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_CLEAR, mod_flags, 0x43) {
                    error!("Failed to register Alt+C (Clear): {:?}", e);
                }
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_QUIT, mod_flags, 0x51) {
                    error!("Failed to register Alt+Q (Quit): {:?}", e);
                }

                if let Some(msg) = hotkey_warning {
                    let mut s = shared_for_thread.lock().unwrap_or_else(|e| e.into_inner());
                    s.status_msg = Some(msg);
                }

                let renderer =
                    match OverlayRenderer::new(virtual_x, virtual_y, virtual_w, virtual_h) {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = init_tx.send(Err(e));
                            return;
                        }
                    };

                let state = Box::new(WindowThreadState {
                    hwnd,
                    renderer,
                    mode: AppMode::Normal,
                    virtual_x,
                    virtual_y,
                    virtual_w,
                    virtual_h,
                    is_dragging: false,
                    drag_start: POINT { x: 0, y: 0 },
                    drag_current: POINT { x: 0, y: 0 },
                    shared_state: shared_for_thread,
                    event_tx,
                });

                let state_ptr = Box::into_raw(state);
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr as isize);

                let _ = ShowWindow(hwnd, SW_SHOW);

                // Initial render of HUD bar
                let state_mut = &mut *state_ptr;
                render_current_scene(state_mut);

                let _ = init_tx.send(Ok(hwnd.0 as isize));

                // Win32 message pump
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }

                // Hotkeys and state_ptr are safely released in WM_DESTROY and WM_NCDESTROY
            }
        });

        let hwnd_raw = init_rx.recv().map_err(|_| {
            windows::core::Error::new(windows::core::HRESULT(-1), "Thread crashed")
        })??;

        let hwnd = HWND(hwnd_raw as *mut _);

        Ok((OverlayHandle { hwnd, shared_state }, event_rx))
    }
}

unsafe fn render_current_scene(state: &mut WindowThreadState) {
    let shared = state
        .shared_state
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    let sel_rect = if state.is_dragging {
        Some(ScreenRect::from_points(
            state.drag_start.x,
            state.drag_start.y,
            state.drag_current.x,
            state.drag_current.y,
        ))
    } else {
        None
    };

    let scene = RenderScene {
        blocks: &shared.current_blocks,
        active_roi: shared.active_roi,
        auto_mode: shared.auto_mode,
        provider_name: &shared.provider_name,
        mode: state.mode,
        selection_rect: sel_rect,
        status_msg: shared.status_msg.as_deref(),
        is_loading: shared.is_loading,
    };

    if let Err(e) = state.renderer.render(state.hwnd, &scene) {
        error!("Direct2D/LayeredWindow render error: {:?}", e);
    }
}

unsafe extern "system" fn overlay_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let state_ptr = GetWindowLongPtrW(hwnd, windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA)
        as *mut WindowThreadState;

    if state_ptr.is_null() {
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }

    let state = &mut *state_ptr;

    match msg {
        WM_NCHITTEST if state.mode == AppMode::SelectingRoi => LRESULT(HTCLIENT),

        WM_APP_UPDATE_STATE => {
            render_current_scene(state);
            LRESULT(0)
        }

        WM_APP_ENTER_SELECTION => {
            enter_selection_mode(state);
            LRESULT(0)
        }

        WM_APP_CLEAR => {
            render_current_scene(state);
            LRESULT(0)
        }

        WM_HOTKEY => {
            let id = wparam.0 as i32;
            match id {
                HOTKEY_TRANSLATE => {
                    let _ = state.event_tx.send(AppEvent::TriggerTranslate);
                }
                HOTKEY_SELECT_ROI => {
                    enter_selection_mode(state);
                }
                HOTKEY_TOGGLE_AUTO => {
                    let _ = state.event_tx.send(AppEvent::ToggleAutoMode);
                }
                HOTKEY_CYCLE_PROVIDER => {
                    let _ = state.event_tx.send(AppEvent::CycleProvider);
                }
                HOTKEY_CLEAR => {
                    let _ = state.event_tx.send(AppEvent::Clear);
                    {
                        let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                        s.current_blocks.clear();
                        s.status_msg = Some("Очищено".to_string());
                        s.is_loading = false;
                    }
                    render_current_scene(state);
                }
                HOTKEY_QUIT => {
                    let _ = state.event_tx.send(AppEvent::Quit);
                    let _ = DestroyWindow(hwnd);
                }
                _ => {}
            }
            LRESULT(0)
        }

        WM_SETCURSOR if state.mode == AppMode::SelectingRoi => {
            if let Ok(cursor) = LoadCursorW(None, IDC_CROSS) {
                SetCursor(Some(cursor));
                return LRESULT(1);
            }
            DefWindowProcW(hwnd, msg, wparam, lparam)
        }

        WM_LBUTTONDOWN if state.mode == AppMode::SelectingRoi => {
            let mut pt = POINT::default();
            let _ = GetCursorPos(&mut pt);
            state.is_dragging = true;
            state.drag_start = pt;
            state.drag_current = pt;
            SetCapture(hwnd);

            render_current_scene(state);
            LRESULT(0)
        }

        WM_MOUSEMOVE if state.mode == AppMode::SelectingRoi && state.is_dragging => {
            let mut pt = POINT::default();
            let _ = GetCursorPos(&mut pt);
            state.drag_current = pt;

            render_current_scene(state);
            LRESULT(0)
        }

        WM_LBUTTONUP if state.mode == AppMode::SelectingRoi && state.is_dragging => {
            let _ = ReleaseCapture();
            state.is_dragging = false;
            let mut pt = POINT::default();
            let _ = GetCursorPos(&mut pt);
            state.drag_current = pt;

            let rect = ScreenRect::from_points(
                state.drag_start.x,
                state.drag_start.y,
                state.drag_current.x,
                state.drag_current.y,
            );

            state.mode = AppMode::Normal;

            let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            let new_style =
                ex_style | (WS_EX_TRANSPARENT.0 as isize) | (WS_EX_NOACTIVATE.0 as isize);
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_style);
            let _ = SetWindowPos(
                hwnd,
                Some(HWND_TOPMOST),
                state.virtual_x,
                state.virtual_y,
                state.virtual_w,
                state.virtual_h,
                SET_WINDOW_POS_FLAGS(SWP_FRAMECHANGED.0 | SWP_NOACTIVATE.0),
            );

            if rect.width > 20 && rect.height > 20 {
                {
                    let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                    s.active_roi = Some(rect);
                    s.status_msg = Some(format!("Область выбрана: {}x{}", rect.width, rect.height));
                }
            } else {
                let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                s.status_msg = Some("Выделение слишком мало".to_string());
            }

            render_current_scene(state);

            if rect.width > 20 && rect.height > 20 {
                let _ = state.event_tx.send(AppEvent::RoiSelected(rect));
            }

            LRESULT(0)
        }

        WM_KEYDOWN if state.mode == AppMode::SelectingRoi && wparam.0 == 0x1B => {
            let _ = ReleaseCapture();
            state.is_dragging = false;
            state.mode = AppMode::Normal;

            let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            let new_style =
                ex_style | (WS_EX_TRANSPARENT.0 as isize) | (WS_EX_NOACTIVATE.0 as isize);
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_style);
            let _ = SetWindowPos(
                hwnd,
                Some(HWND_TOPMOST),
                state.virtual_x,
                state.virtual_y,
                state.virtual_w,
                state.virtual_h,
                SET_WINDOW_POS_FLAGS(SWP_FRAMECHANGED.0 | SWP_NOACTIVATE.0),
            );

            {
                let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                s.status_msg = Some("Выделение отменено".to_string());
            }
            render_current_scene(state);
            LRESULT(0)
        }

        WM_CLOSE => {
            let _ = DestroyWindow(hwnd);
            LRESULT(0)
        }

        WM_DESTROY => {
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_TRANSLATE);
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_SELECT_ROI);
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_TOGGLE_AUTO);
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_CYCLE_PROVIDER);
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_CLEAR);
            let _ = UnregisterHotKey(Some(hwnd), HOTKEY_QUIT);
            PostQuitMessage(0);
            LRESULT(0)
        }

        WM_NCDESTROY => {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            drop(Box::from_raw(state_ptr));
            DefWindowProcW(hwnd, msg, wparam, lparam)
        }

        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}

unsafe fn enter_selection_mode(state: &mut WindowThreadState) {
    state.mode = AppMode::SelectingRoi;
    state.is_dragging = false;

    render_current_scene(state);

    let ex_style = GetWindowLongPtrW(state.hwnd, GWL_EXSTYLE);
    let new_style = ex_style & !(WS_EX_TRANSPARENT.0 as isize) & !(WS_EX_NOACTIVATE.0 as isize);
    SetWindowLongPtrW(state.hwnd, GWL_EXSTYLE, new_style);

    let _ = SetWindowPos(
        state.hwnd,
        Some(HWND_TOPMOST),
        state.virtual_x,
        state.virtual_y,
        state.virtual_w,
        state.virtual_h,
        SET_WINDOW_POS_FLAGS(SWP_FRAMECHANGED.0 | SWP_SHOWWINDOW.0),
    );

    let fg_hwnd = GetForegroundWindow();
    let current_thread_id = GetCurrentThreadId();
    let fg_thread_id = if !fg_hwnd.0.is_null() {
        GetWindowThreadProcessId(fg_hwnd, None)
    } else {
        0
    };

    if fg_thread_id != 0 && fg_thread_id != current_thread_id {
        let _ = AttachThreadInput(current_thread_id, fg_thread_id, true);
        let _ = SetForegroundWindow(state.hwnd);
        let _ = SetFocus(Some(state.hwnd));
        if let Ok(cursor) = LoadCursorW(None, IDC_CROSS) {
            SetCursor(Some(cursor));
        }
        let _ = AttachThreadInput(current_thread_id, fg_thread_id, false);
    } else {
        let _ = SetForegroundWindow(state.hwnd);
        let _ = SetFocus(Some(state.hwnd));
        if let Ok(cursor) = LoadCursorW(None, IDC_CROSS) {
            SetCursor(Some(cursor));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::SendMessageW;

    #[test]
    #[cfg(windows)]
    fn bug_7_wm_close_destroys_window_and_frees_state_once() {
        // Spawn overlay window thread
        let (overlay, _rx) = OverlayManager::spawn().expect("Failed to spawn overlay");
        assert!(!overlay.hwnd.0.is_null(), "Overlay HWND must be valid");

        // Send WM_CLOSE to window.
        // BUG-7 fix ensures WM_CLOSE calls DestroyWindow(hwnd),
        // and state memory is freed exactly once in WM_NCDESTROY without double free.
        unsafe {
            let _ = SendMessageW(overlay.hwnd, WM_CLOSE, None, None);
        }

        // Give the message pump thread brief time to finish and exit cleanly
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
}
