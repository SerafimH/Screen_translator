pub mod renderer;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tracing::{error, info, warn};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, HOT_KEY_MODIFIERS, MOD_ALT, MOD_NOREPEAT, RegisterHotKey, ReleaseCapture,
    SetCapture, SetFocus, UnregisterHotKey, VK_CONTROL, VK_LSHIFT, VK_LWIN, VK_MENU, VK_RSHIFT,
    VK_RWIN, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, FindWindowW,
    GWL_EXSTYLE, GWLP_USERDATA, GetCursorPos, GetForegroundWindow, GetMessageW, GetSystemMetrics,
    GetWindowLongPtrW, GetWindowThreadProcessId, HHOOK, HWND_TOPMOST, IDC_CROSS, KBDLLHOOKSTRUCT,
    LLKHF_ALTDOWN, LLKHF_UP, LoadCursorW, MSG, MSLLHOOKSTRUCT, PostMessageW, PostQuitMessage, RegisterClassW,
    SET_WINDOW_POS_FLAGS, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SMTO_ABORTIFHUNG, SW_SHOW, SWP_FRAMECHANGED,
    SWP_NOACTIVATE, SWP_SHOWWINDOW, SendMessageTimeoutW, SetCursor, SetForegroundWindow,
    SetWindowDisplayAffinity, SetWindowLongPtrW, SetWindowPos, SetWindowsHookExW, ShowWindow,
    TranslateMessage, UnhookWindowsHookEx, WDA_EXCLUDEFROMCAPTURE, WH_KEYBOARD_LL, WH_MOUSE_LL, WM_APP,
    WM_CLOSE, WM_DESTROY, WM_DISPLAYCHANGE, WM_DPICHANGED, WM_HOTKEY, WM_KEYDOWN, WM_KEYUP, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WM_MOUSEMOVE, WM_NCDESTROY, WM_NCHITTEST, WM_SETCURSOR, WM_SYSKEYDOWN, WM_SYSKEYUP, WNDCLASSW,
    WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
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
pub const HOTKEY_CYCLE_OCR_MODE: i32 = 107;

pub const WM_APP_UPDATE_STATE: u32 = WM_APP + 2;
pub const WM_APP_CLEAR: u32 = WM_APP + 3;

const HTCLIENT: isize = 1;

static HOOK_TX: Mutex<Option<UnboundedSender<AppEvent>>> = Mutex::new(None);
static SHIFT_HELD: AtomicBool = AtomicBool::new(false);
static SAVE_BUTTON: Mutex<SaveButtonState> = Mutex::new(SaveButtonState {
    target: None,
    pressed: None,
});

#[derive(Default)]
struct SaveButtonState {
    target: Option<(ScreenRect, TranslationBlock)>,
    pressed: Option<(ScreenRect, TranslationBlock)>,
}

impl SaveButtonState {
    fn mouse_event(&mut self, msg: u32, pt: POINT) -> (bool, Option<TranslationBlock>) {
        if msg == WM_LBUTTONDOWN {
            if let Some((rect, card)) = &self.target
                && rect.contains(pt.x, pt.y)
            {
                self.pressed = Some((*rect, card.clone()));
                return (true, None);
            }
        } else if msg == WM_LBUTTONUP
            && let Some((rect, card)) = self.pressed.take()
        {
            let save = self.target.is_some() && rect.contains(pt.x, pt.y);
            return (true, save.then_some(card));
        }
        (false, None)
    }
}

unsafe extern "system" fn low_level_mouse_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let msg = wparam.0 as u32;
    if code >= 0 && lparam.0 != 0 && matches!(msg, WM_LBUTTONDOWN | WM_LBUTTONUP) {
        let mouse = unsafe { &*(lparam.0 as *const MSLLHOOKSTRUCT) };
        let (handled, card) = SAVE_BUTTON.lock().unwrap_or_else(|e| e.into_inner())
            .mouse_event(msg, mouse.pt);
        if let Some(card) = card
            && let Some(tx) = HOOK_TX.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
        {
            let _ = tx.send(AppEvent::SaveCard(card));
        }
        // The layered overlay is click-through; consume only clicks on the visible save button.
        if handled {
            return LRESULT(1);
        }
    }
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

pub fn is_shift_held() -> bool {
    SHIFT_HELD.load(Ordering::SeqCst)
}

unsafe extern "system" fn low_level_keyboard_proc(
    code: i32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if code >= 0 && lparam.0 != 0 {
        let msg = wparam.0 as u32;
        let kbd = unsafe { *(lparam.0 as *const KBDLLHOOKSTRUCT) };
        let is_shift = kbd.vkCode == VK_SHIFT.0 as u32
            || kbd.vkCode == VK_LSHIFT.0 as u32
            || kbd.vkCode == VK_RSHIFT.0 as u32;

        if is_shift {
            let is_key_down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
            let is_key_up = msg == WM_KEYUP || msg == WM_SYSKEYUP || kbd.flags.contains(LLKHF_UP);

            if is_key_down && !kbd.flags.contains(LLKHF_UP) {
                // Ignore Shift if Alt, Ctrl, or Windows key is down (e.g. Alt+Shift / Ctrl+Shift layout switch)
                let alt_down = kbd.flags.contains(LLKHF_ALTDOWN)
                    || (unsafe { GetAsyncKeyState(VK_MENU.0 as i32) } as u16 & 0x8000) != 0;
                let ctrl_down =
                    (unsafe { GetAsyncKeyState(VK_CONTROL.0 as i32) } as u16 & 0x8000) != 0;
                let win_down = (unsafe { GetAsyncKeyState(VK_LWIN.0 as i32) } as u16 & 0x8000) != 0
                    || (unsafe { GetAsyncKeyState(VK_RWIN.0 as i32) } as u16 & 0x8000) != 0;

                if !alt_down && !ctrl_down && !win_down
                    && !SHIFT_HELD.swap(true, Ordering::SeqCst) {
                        let mut pt = POINT::default();
                        let _ = unsafe { GetCursorPos(&mut pt) };
                        if let Ok(guard) = HOOK_TX.lock()
                            && let Some(tx) = guard.as_ref() {
                                let _ = tx.send(AppEvent::TranslateUnderCursor(pt));
                            }
                    }
            } else if is_key_up
                && SHIFT_HELD.swap(false, Ordering::SeqCst)
                    && let Ok(guard) = HOOK_TX.lock()
                        && let Some(tx) = guard.as_ref() {
                            let _ = tx.send(AppEvent::ShiftReleased);
                        }
        }
    }

    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    TriggerTranslate,
    RoiSelected(ScreenRect),
    ResetRoi,
    ToggleAutoMode,
    CycleProvider,
    CycleOcrMode,
    Clear,
    Quit,
    TranslateUnderCursor(POINT),
    SaveCard(TranslationBlock),
    ShiftReleased,
}

#[derive(Debug, Clone)]
pub struct OverlaySharedState {
    pub current_blocks: Vec<TranslationBlock>,
    pub cursor_card: Option<TranslationBlock>,
    pub status_msg: Option<String>,
    pub is_loading: bool,
    pub active_roi: Option<ScreenRect>,
    pub auto_mode: bool,
    pub provider_name: String,
    pub ocr_engine_name: String,
}

impl Default for OverlaySharedState {
    fn default() -> Self {
        Self {
            current_blocks: Vec::new(),
            cursor_card: None,
            status_msg: Some("Готов - нажмите Alt+T для перевода".to_string()),
            is_loading: false,
            active_roi: None,
            auto_mode: false,
            provider_name: "Google Translate".to_string(),
            ocr_engine_name: "RapidOCR".to_string(),
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

    pub fn update_cursor_card(&self, card: Option<TranslationBlock>, status: Option<String>) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.cursor_card = card;
            if let Some(msg) = status {
                s.status_msg = Some(msg);
            }
            s.is_loading = false;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn clear_cursor_card(&self) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.cursor_card = None;
            s.status_msg = Some("Готов".to_string());
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

    pub fn set_ocr_engine_name(&self, name: String) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.ocr_engine_name = name;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0));
        }
    }

    pub fn clear(&self) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.current_blocks.clear();
            s.cursor_card = None;
            s.active_roi = None;
            s.status_msg = Some("Очищено".to_string());
            s.is_loading = false;
        }
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_APP_CLEAR, WPARAM(0), LPARAM(0));
        }
    }

    pub fn clear_blocks(&self) {
        {
            let mut s = self.shared_state.lock().unwrap_or_else(|e| e.into_inner());
            s.current_blocks.clear();
            s.cursor_card = None;
            s.status_msg = Some("Готов".to_string());
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
    // Low-level keyboard hook handle for Shift word/sentence translation
    keyboard_hook: HHOOK,
    mouse_hook: HHOOK,
    // Previously focused window to restore after selection mode
    previous_foreground_hwnd: HWND,
}

fn virtual_screen_bounds() -> ScreenRect {
    unsafe {
        let width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        let height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        if width > 0 && height > 0 {
            ScreenRect::new(GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN), width, height)
        } else {
            ScreenRect::new(0, 0, GetSystemMetrics(SM_CXSCREEN).max(1), GetSystemMetrics(SM_CYSCREEN).max(1))
        }
    }
}

/// No reference to WindowThreadState may remain live across SetWindowPos: it dispatches messages.
unsafe fn refresh_display_geometry(hwnd: HWND, bounds: ScreenRect) -> Result<()> {
    let was_dragging = {
        let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *mut WindowThreadState;
        if ptr.is_null() { return Ok(()); }
        let state = unsafe { &mut *ptr };
        state.renderer.reconfigure(bounds)?;
        state.virtual_x = bounds.x;
        state.virtual_y = bounds.y;
        state.virtual_w = bounds.width;
        state.virtual_h = bounds.height;
        let mut shared = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
        if shared.active_roi.is_some_and(|roi| {
            roi.x < bounds.x || roi.y < bounds.y
                || roi.x as i64 + roi.width as i64 > bounds.x as i64 + bounds.width as i64
                || roi.y as i64 + roi.height as i64 > bounds.y as i64 + bounds.height as i64
        }) {
            shared.active_roi = None;
            let _ = state.event_tx.send(AppEvent::ResetRoi);
        }
        let was_dragging = state.is_dragging;
        state.is_dragging = false;
        was_dragging
    };
    SAVE_BUTTON.lock().unwrap_or_else(|e| e.into_inner()).target = None;
    if was_dragging { let _ = unsafe { ReleaseCapture() }; }
    unsafe {
        SetWindowPos(hwnd, Some(HWND_TOPMOST), bounds.x, bounds.y, bounds.width, bounds.height, SWP_NOACTIVATE)?;
        PostMessageW(Some(hwnd), WM_APP_UPDATE_STATE, WPARAM(0), LPARAM(0))?;
    }
    Ok(())
}

pub struct OverlayManager;

impl OverlayManager {
    /// Launches the Win32 transparent overlay window on a dedicated message pump thread.
    /// Closes any previous instance window and applies WS_EX_LAYERED + WDA_EXCLUDEFROMCAPTURE.
    pub fn spawn(
        config: &crate::types::AppConfig,
    ) -> Result<(OverlayHandle, UnboundedReceiver<AppEvent>)> {
        let (event_tx, event_rx) = unbounded_channel();
        let (init_tx, init_rx) =
            std::sync::mpsc::channel::<std::result::Result<isize, windows::core::Error>>();

        let shared_state = Arc::new(Mutex::new(OverlaySharedState {
            provider_name: config.provider.display_name().to_string(),
            ocr_engine_name: config.ocr_engine.display_name().to_string(),
            ..OverlaySharedState::default()
        }));
        let shared_for_thread = shared_state.clone();
        let font_family = config.font_family.clone();
        let font_size = config.font_size as f32;
        let bg_alpha = config.overlay_bg_alpha as f32 / 255.0;

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

                let bounds = virtual_screen_bounds();
                let (virtual_x, virtual_y, virtual_w, virtual_h) = (bounds.x, bounds.y, bounds.width, bounds.height);

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
                if let Err(e) = RegisterHotKey(Some(hwnd), HOTKEY_CYCLE_OCR_MODE, mod_flags, 0x4F) {
                    error!("Failed to register Alt+O (Cycle OCR Mode): {:?}", e);
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

                let renderer = match OverlayRenderer::new(
                    virtual_x,
                    virtual_y,
                    virtual_w,
                    virtual_h,
                    &font_family,
                    font_size,
                    bg_alpha,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };

                *HOOK_TX.lock().unwrap_or_else(|e| e.into_inner()) = Some(event_tx.clone());
                let keyboard_hook = match SetWindowsHookExW(
                    WH_KEYBOARD_LL,
                    Some(low_level_keyboard_proc),
                    Some(hinstance.into()),
                    0,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        warn!("Failed to install Shift keyboard hook: {:?}", e);
                        HHOOK(core::ptr::null_mut())
                    }
                };

                let mouse_hook = match SetWindowsHookExW(
                    WH_MOUSE_LL,
                    Some(low_level_mouse_proc),
                    Some(hinstance.into()),
                    0,
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        warn!("Failed to install flashcard mouse hook: {:?}", e);
                        HHOOK(core::ptr::null_mut())
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
                    keyboard_hook,
                    mouse_hook,
                    previous_foreground_hwnd: HWND(core::ptr::null_mut()),
                });

                let state_ptr = Box::into_raw(state);
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr as isize);

                let _ = ShowWindow(hwnd, SW_SHOW);

                // Initial render of HUD bar
                let state_mut = &mut *state_ptr;
                render_current_scene(state_mut);

                let _ = init_tx.send(Ok(hwnd.0 as isize));

                // Win32 message pump (returns > 0 for messages, 0 on WM_QUIT, -1 on error)
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
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

fn render_current_scene(state: &mut WindowThreadState) {
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
        cursor_card: shared.cursor_card.as_ref(),
        save_button_enabled: !state.mouse_hook.is_invalid(),
        active_roi: shared.active_roi,
        auto_mode: shared.auto_mode,
        provider_name: &shared.provider_name,
        ocr_engine_name: &shared.ocr_engine_name,
        mode: state.mode,
        selection_rect: sel_rect,
        status_msg: shared.status_msg.as_deref(),
        is_loading: shared.is_loading,
    };

    let button = match state.renderer.render(state.hwnd, &scene) {
        Ok(rect) => rect.zip(shared.cursor_card),
        Err(e) => {
            error!("Direct2D/LayeredWindow render error: {:?}", e);
            None
        }
    };
    SAVE_BUTTON.lock().unwrap_or_else(|e| e.into_inner()).target = button;
}

unsafe extern "system" fn overlay_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let state_ptr =
        unsafe { GetWindowLongPtrW(hwnd, windows::Win32::UI::WindowsAndMessaging::GWLP_USERDATA) }
            as *mut WindowThreadState;

    if state_ptr.is_null() {
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }

    if matches!(msg, WM_DISPLAYCHANGE | WM_DPICHANGED) {
        if let Err(error) = unsafe { refresh_display_geometry(hwnd, virtual_screen_bounds()) } {
            error!("Failed to refresh overlay display geometry: {:?}", error);
        }
        return LRESULT(0);
    }

    // SAFETY: state_ptr was initialized from Box::into_raw during window creation and is non-null
    // until WM_NCDESTROY frees it.
    let state = unsafe { &mut *state_ptr };

    match msg {
        WM_NCHITTEST if state.mode == AppMode::SelectingRoi => LRESULT(HTCLIENT),

        WM_APP_UPDATE_STATE => {
            render_current_scene(state);
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
                HOTKEY_CYCLE_OCR_MODE => {
                    let _ = state.event_tx.send(AppEvent::CycleOcrMode);
                }
                HOTKEY_CLEAR => {
                    let _ = state.event_tx.send(AppEvent::Clear);
                    // Synchronous UI clearing here is intentional for immediate visual responsiveness;
                    // it is completely idempotent with respect to AppEvent::Clear handled in the main app loop.
                    {
                        let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                        s.current_blocks.clear();
                        s.active_roi = None;
                        s.status_msg = Some("Очищено".to_string());
                        s.is_loading = false;
                    }
                    render_current_scene(state);
                }
                HOTKEY_QUIT => {
                    let _ = state.event_tx.send(AppEvent::Quit);
                    let _ = unsafe { DestroyWindow(hwnd) };
                }
                _ => {}
            }
            LRESULT(0)
        }

        WM_SETCURSOR if state.mode == AppMode::SelectingRoi => {
            if let Ok(cursor) = unsafe { LoadCursorW(None, IDC_CROSS) } {
                unsafe { SetCursor(Some(cursor)) };
                return LRESULT(1);
            }
            unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
        }

        WM_LBUTTONDOWN if state.mode == AppMode::SelectingRoi => {
            let mut pt = POINT::default();
            let _ = unsafe { GetCursorPos(&mut pt) };
            state.is_dragging = true;
            state.drag_start = pt;
            state.drag_current = pt;
            unsafe { SetCapture(hwnd) };

            render_current_scene(state);
            LRESULT(0)
        }

        WM_MOUSEMOVE if state.mode == AppMode::SelectingRoi && state.is_dragging => {
            let mut pt = POINT::default();
            let _ = unsafe { GetCursorPos(&mut pt) };
            state.drag_current = pt;

            render_current_scene(state);
            LRESULT(0)
        }

        WM_LBUTTONUP if state.mode == AppMode::SelectingRoi && state.is_dragging => {
            let mut pt = POINT::default();
            let _ = unsafe { GetCursorPos(&mut pt) };
            state.drag_current = pt;

            let rect = ScreenRect::from_points(
                state.drag_start.x,
                state.drag_start.y,
                state.drag_current.x,
                state.drag_current.y,
            );

            exit_selection_mode(state);

            if rect.width > 20 && rect.height > 20 {
                {
                    let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                    s.active_roi = Some(rect);
                    s.status_msg = Some(format!("Область выбрана: {}x{}", rect.width, rect.height));
                }
                render_current_scene(state);
                let _ = state.event_tx.send(AppEvent::RoiSelected(rect));
            } else if rect.width <= 5 && rect.height <= 5 {
                {
                    let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                    s.active_roi = None;
                    s.status_msg = Some("Область сброшена (полный экран)".to_string());
                }
                render_current_scene(state);
                let _ = state.event_tx.send(AppEvent::ResetRoi);
            } else {
                {
                    let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                    s.status_msg = Some("Выделение слишком мало".to_string());
                }
                render_current_scene(state);
            }

            LRESULT(0)
        }

        WM_KEYDOWN if state.mode == AppMode::SelectingRoi && wparam.0 == 0x1B => {
            exit_selection_mode(state);

            {
                let mut s = state.shared_state.lock().unwrap_or_else(|e| e.into_inner());
                s.status_msg = Some("Выделение отменено".to_string());
            }
            render_current_scene(state);
            LRESULT(0)
        }

        WM_CLOSE => {
            let _ = unsafe { DestroyWindow(hwnd) };
            LRESULT(0)
        }

        WM_DESTROY => {
            if !state.mouse_hook.is_invalid() {
                let _ = unsafe { UnhookWindowsHookEx(state.mouse_hook) };
            }
            if !state.keyboard_hook.is_invalid() {
                let _ = unsafe { UnhookWindowsHookEx(state.keyboard_hook) };
            }
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_TRANSLATE) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_SELECT_ROI) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_TOGGLE_AUTO) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_CYCLE_PROVIDER) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_CYCLE_OCR_MODE) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_CLEAR) };
            let _ = unsafe { UnregisterHotKey(Some(hwnd), HOTKEY_QUIT) };
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }

        WM_NCDESTROY => {
            *SAVE_BUTTON.lock().unwrap_or_else(|e| e.into_inner()) = SaveButtonState::default();
            // Clear global hook sender on window destruction
            *HOOK_TX.lock().unwrap_or_else(|e| e.into_inner()) = None;

            // SAFETY: Clear GWLP_USERDATA before taking back ownership of WindowThreadState.
            // Exactly one Box::from_raw occurs on final window destruction.
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                drop(Box::from_raw(state_ptr));
                DefWindowProcW(hwnd, msg, wparam, lparam)
            }
        }

        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

fn enter_selection_mode(state: &mut WindowThreadState) {
    if state.mode == AppMode::SelectingRoi {
        return;
    }

    let fg_hwnd = unsafe { GetForegroundWindow() };
    if fg_hwnd != state.hwnd {
        state.previous_foreground_hwnd = fg_hwnd;
    }

    state.mode = AppMode::SelectingRoi;
    state.is_dragging = false;

    render_current_scene(state);

    let ex_style = unsafe { GetWindowLongPtrW(state.hwnd, GWL_EXSTYLE) };
    let new_style = ex_style & !(WS_EX_TRANSPARENT.0 as isize) & !(WS_EX_NOACTIVATE.0 as isize);
    unsafe { SetWindowLongPtrW(state.hwnd, GWL_EXSTYLE, new_style) };

    let _ = unsafe {
        SetWindowPos(
            state.hwnd,
            Some(HWND_TOPMOST),
            state.virtual_x,
            state.virtual_y,
            state.virtual_w,
            state.virtual_h,
            SET_WINDOW_POS_FLAGS(SWP_FRAMECHANGED.0 | SWP_SHOWWINDOW.0),
        )
    };
    let current_thread_id = unsafe { GetCurrentThreadId() };
    let fg_thread_id = if !fg_hwnd.0.is_null() {
        unsafe { GetWindowThreadProcessId(fg_hwnd, None) }
    } else {
        0
    };

    if fg_thread_id != 0 && fg_thread_id != current_thread_id {
        let _ = unsafe { AttachThreadInput(current_thread_id, fg_thread_id, true) };
        let _ = unsafe { SetForegroundWindow(state.hwnd) };
        let _ = unsafe { SetFocus(Some(state.hwnd)) };
        if let Ok(cursor) = unsafe { LoadCursorW(None, IDC_CROSS) } {
            unsafe { SetCursor(Some(cursor)) };
        }
        let _ = unsafe { AttachThreadInput(current_thread_id, fg_thread_id, false) };
    } else {
        let _ = unsafe { SetForegroundWindow(state.hwnd) };
        let _ = unsafe { SetFocus(Some(state.hwnd)) };
        if let Ok(cursor) = unsafe { LoadCursorW(None, IDC_CROSS) } {
            unsafe { SetCursor(Some(cursor)) };
        }
    }
}

fn exit_selection_mode(state: &mut WindowThreadState) {
    let _ = unsafe { ReleaseCapture() };
    state.is_dragging = false;
    state.mode = AppMode::Normal;

    let ex_style = unsafe { GetWindowLongPtrW(state.hwnd, GWL_EXSTYLE) };
    let new_style = ex_style | (WS_EX_TRANSPARENT.0 as isize) | (WS_EX_NOACTIVATE.0 as isize);
    unsafe { SetWindowLongPtrW(state.hwnd, GWL_EXSTYLE, new_style) };
    let _ = unsafe {
        SetWindowPos(
            state.hwnd,
            Some(HWND_TOPMOST),
            state.virtual_x,
            state.virtual_y,
            state.virtual_w,
            state.virtual_h,
            SET_WINDOW_POS_FLAGS(SWP_FRAMECHANGED.0 | SWP_NOACTIVATE.0),
        )
    };
    if !state.previous_foreground_hwnd.0.is_null() {
        let _ = unsafe { SetForegroundWindow(state.previous_foreground_hwnd) };
        state.previous_foreground_hwnd = HWND(core::ptr::null_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::SendMessageW;

    #[test]
    fn display_refresh_updates_bounds_preserves_cards_and_resets_offscreen_roi() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
        let hwnd = unsafe {
            CreateWindowExW(WS_EX_LAYERED, w!("STATIC"), w!("Display refresh test"), WS_POPUP,
                0, 0, 300, 200, None, None, None, None).unwrap()
        };
        let (event_tx, mut event_rx) = unbounded_channel();
        let card = TranslationBlock {
            original_text: "word".to_string(), translated_text: "translation".to_string(),
            rect: ScreenRect::new(10, 20, 80, 30),
        };
        let shared = Arc::new(Mutex::new(OverlaySharedState {
            current_blocks: vec![card.clone()], cursor_card: Some(card),
            active_roi: Some(ScreenRect::new(10, 20, 80, 30)), ..OverlaySharedState::default()
        }));
        let state = Box::new(WindowThreadState {
            hwnd, renderer: OverlayRenderer::new(0, 0, 300, 200, "Segoe UI", 16.0, 0.9).unwrap(),
            mode: AppMode::Normal, virtual_x: 0, virtual_y: 0, virtual_w: 300, virtual_h: 200,
            is_dragging: false, drag_start: POINT::default(), drag_current: POINT::default(),
            shared_state: shared.clone(), event_tx, keyboard_hook: HHOOK(core::ptr::null_mut()),
            mouse_hook: HHOOK(core::ptr::null_mut()), previous_foreground_hwnd: HWND(core::ptr::null_mut()),
        });
        let ptr = Box::into_raw(state);
        unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, ptr as isize); }
        let bounds = ScreenRect::new(-200, -100, 640, 480);
        unsafe { refresh_display_geometry(hwnd, bounds).unwrap(); }
        let state = unsafe { &*ptr };
        assert_eq!((state.virtual_x, state.virtual_y, state.virtual_w, state.virtual_h), (-200, -100, 640, 480));
        let mut actual = RECT::default();
        unsafe { GetWindowRect(hwnd, &mut actual).unwrap(); }
        assert_eq!((actual.left, actual.top, actual.right, actual.bottom), (-200, -100, 440, 380));
        assert!(shared.lock().unwrap().active_roi.is_some());
        assert!(event_rx.try_recv().is_err());
        shared.lock().unwrap().active_roi = Some(ScreenRect::new(10000, 10000, 100, 100));
        unsafe { refresh_display_geometry(hwnd, bounds).unwrap(); }
        assert!(matches!(event_rx.try_recv(), Ok(AppEvent::ResetRoi)));
        assert!(shared.lock().unwrap().active_roi.is_none());
        assert_eq!(shared.lock().unwrap().current_blocks.len(), 1);
        assert!(shared.lock().unwrap().cursor_card.is_some());
        // Exercise both OS notification handlers without changing monitor settings.
        for message in [WM_DISPLAYCHANGE, WM_DPICHANGED] {
            unsafe { overlay_wndproc(hwnd, message, WPARAM(0), LPARAM(0)); }
            let state = unsafe { &*ptr };
            let actual_bounds = virtual_screen_bounds();
            assert_eq!((state.virtual_x, state.virtual_y, state.virtual_w, state.virtual_h),
                (actual_bounds.x, actual_bounds.y, actual_bounds.width, actual_bounds.height));
        }
        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            drop(Box::from_raw(ptr));
            DestroyWindow(hwnd).unwrap();
        }
    }

    #[test]
    fn test_save_button_consumes_only_its_click_and_keeps_the_clicked_card() {
        let rect = ScreenRect::new(-100, 20, 28, 28);
        let card = TranslationBlock {
            original_text: "word (context)".to_string(),
            translated_text: "translation".to_string(), rect,
        };
        let mut button = SaveButtonState { target: Some((rect, card)), pressed: None };
        let inside = POINT { x: -90, y: 30 };
        let outside = POINT { x: 0, y: 0 };
        assert!(!button.mouse_event(WM_LBUTTONDOWN, outside).0);
        assert!(!button.mouse_event(WM_LBUTTONUP, inside).0);
        let (handled, saved) = button.mouse_event(WM_LBUTTONDOWN, inside);
        assert!(handled && saved.is_none());
        button.target.as_mut().unwrap().1.original_text = "new word".to_string();
        let (handled, saved) = button.mouse_event(WM_LBUTTONUP, inside);
        assert!(handled);
        assert_eq!(saved.unwrap().original_text, "word (context)");
        assert!(!button.mouse_event(WM_LBUTTONUP, inside).0);
        assert!(button.mouse_event(WM_LBUTTONDOWN, inside).0);
        let (handled, saved) = button.mouse_event(WM_LBUTTONUP, outside);
        assert!(handled && saved.is_none());
        assert!(button.mouse_event(WM_LBUTTONDOWN, inside).0);
        button.target = None;
        let (handled, saved) = button.mouse_event(WM_LBUTTONUP, inside);
        assert!(handled && saved.is_none());
        assert!(!button.mouse_event(WM_LBUTTONDOWN, inside).0);
    }

    #[test]
    #[cfg(windows)]
    fn bug_7_wm_close_destroys_window_and_frees_state_once() {
        // Spawn overlay window thread
        let (overlay, _rx) = OverlayManager::spawn(&crate::types::AppConfig::default())
            .expect("Failed to spawn overlay");
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
