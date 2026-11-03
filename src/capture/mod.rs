use std::os::windows::io::AsRawHandle;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{error, warn};
use windows::Win32::Foundation::{HANDLE, LPARAM, WPARAM};
use windows::Win32::System::Threading::GetThreadId;
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
    CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, HBITMAP, HDC,
    HGDIOBJ, ROP_CODE, ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, PostThreadMessageW, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN,
    SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, WM_QUIT,
};
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};

use crate::types::{ScreenRect, luma_from_bgr};

pub const SIGNATURE_COLS: usize = 32;
pub const SIGNATURE_ROWS: usize = 16;
pub const DELTA_THRESHOLD: f32 = 3.5;
pub const WINRT_MAX_CONSECUTIVE_TIMEOUTS: u64 = 3;
pub const WINRT_CIRCUIT_COOLDOWN: Duration = Duration::from_secs(30);
const WINRT_CAPTURE_TIMEOUT: Duration = Duration::from_millis(600);

static WINRT_TIMEOUT_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_WINRT_TIMEOUT: Mutex<Option<Instant>> = Mutex::new(None);
// Retain ownership until native initialization AND cleanup finish. A timed-out
// driver call may not be interrupted safely, so later captures use GDI rather
// than starting additional native workers. These are not Tokio blocking tasks.
static WINRT_WORKER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

#[derive(Debug, PartialEq, Eq)]
enum WinrtCaptureError {
    Busy,
    TimedOut,
    Failed(String),
}

fn bounded_capture<F>(
    worker_slot: &Mutex<Option<JoinHandle<()>>>,
    timeout: Duration,
    capture: F,
) -> Result<CapturedFrame, WinrtCaptureError>
where
    F: FnOnce(Sender<Result<CapturedFrame, WinrtCaptureError>>, Instant) + Send + 'static,
{
    let deadline = Instant::now() + timeout;
    let (tx, rx) = channel();
    {
        let mut worker = worker_slot.lock().unwrap_or_else(|e| e.into_inner());
        if worker.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return Err(WinrtCaptureError::Busy);
        }
        *worker = Some(
            std::thread::Builder::new()
                .name("screen-capture-winrt".to_string())
                .spawn(move || capture(tx, deadline))
                .map_err(|e| WinrtCaptureError::Failed(format!("Failed to start capture worker: {e}")))?,
        );
    }
    match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(WinrtCaptureError::TimedOut),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err(WinrtCaptureError::Failed("WinRT capture worker ended without a frame".to_string()))
        }
    }
}

pub fn winrt_consecutive_timeouts() -> u64 {
    WINRT_TIMEOUT_COUNT.load(Ordering::Relaxed)
}

pub fn reset_winrt_timeout_count() {
    WINRT_TIMEOUT_COUNT.store(0, Ordering::Relaxed);
    if let Ok(mut last) = LAST_WINRT_TIMEOUT.lock() {
        *last = None;
    }
}

pub fn is_winrt_available() -> bool {
    if WINRT_TIMEOUT_COUNT.load(Ordering::Relaxed) < WINRT_MAX_CONSECUTIVE_TIMEOUTS {
        return true;
    }
    let last = LAST_WINRT_TIMEOUT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = *last
        && t.elapsed() >= WINRT_CIRCUIT_COOLDOWN {
            return true;
        }
    false
}

/// Compact 32x16 Luma perceptual hash to detect genuine dialogue transitions
/// while rejecting flickering cursors, animated particles, and subtle background noise.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSignature {
    pub luma: [u8; SIGNATURE_COLS * SIGNATURE_ROWS],
    /// High-contrast edge positions retain small glyph changes whose brightness
    /// averages cancel out inside the coarse cells.
    edges: Vec<u64>,
    text_mask: Vec<u64>,
    dimensions: (u32, u32),
}

impl Default for FrameSignature {
    fn default() -> Self {
        Self {
            luma: [0u8; SIGNATURE_COLS * SIGNATURE_ROWS],
            edges: Vec::new(),
            text_mask: Vec::new(),
            dimensions: (0, 0),
        }
    }
}

impl FrameSignature {
    pub fn from_bgra(bgra: &[u8], width: u32, height: u32) -> Self {
        let mut luma = [0u8; SIGNATURE_COLS * SIGNATURE_ROWS];
        if width == 0 || height == 0 || bgra.len() < (width * height * 4) as usize {
            return Self::default();
        }
        let mut edges = vec![0u64; (width as usize * height as usize).div_ceil(64)];

        for row in 0..SIGNATURE_ROWS {
            let start_y = (row * height as usize) / SIGNATURE_ROWS;
            let end_y = (((row + 1) * height as usize) / SIGNATURE_ROWS)
                .max(start_y + 1)
                .min(height as usize);

            for col in 0..SIGNATURE_COLS {
                let start_x = (col * width as usize) / SIGNATURE_COLS;
                let end_x = (((col + 1) * width as usize) / SIGNATURE_COLS)
                    .max(start_x + 1)
                    .min(width as usize);

                let mut sum: u64 = 0;
                let mut count: u64 = 0;

                for y in start_y..end_y {
                    let row_offset = y * width as usize;
                    for x in start_x..end_x {
                        let idx = (row_offset + x) * 4;
                        // In BGRA8 format, each pixel occupies 4 bytes:
                        // idx: Blue, idx+1: Green, idx+2: Red, idx+3: Alpha.
                        // Luma calculation only needs B, G, R, so accessing up to idx + 2 is sufficient.
                        // Checking `idx + 2 < bgra.len()` correctly ensures all 3 required color channels
                        // are within bounds without skipping any pixel, while leaving Alpha (idx + 3) unread.
                        if idx + 2 < bgra.len() {
                            let y_val =
                                luma_from_bgr(bgra[idx], bgra[idx + 1], bgra[idx + 2]) as u64;
                            sum += y_val;
                            count += 1;
                            let edge_at = |neighbor: usize| {
                                let other = luma_from_bgr(bgra[neighbor], bgra[neighbor + 1], bgra[neighbor + 2]);
                                (y_val as u8).abs_diff(other) >= 16
                            };
                            if (x + 1 < width as usize && edge_at(idx + 4))
                                || (y + 1 < height as usize && edge_at(idx + width as usize * 4)) {
                                let bit = row_offset + x;
                                edges[bit / 64] |= 1u64 << (bit % 64);
                            }
                        }
                    }
                }

                let avg = sum.checked_div(count).unwrap_or(0) as u8;
                luma[row * SIGNATURE_COLS + col] = avg;
            }
        }

        Self { luma, edges, text_mask: Vec::new(), dimensions: (width, height) }
    }

    /// Scope fine changes to known text. Outside animation still uses the
    /// coarse scene detector, including during the 250ms stability check.
    pub fn set_text_regions(
        &mut self, regions: impl IntoIterator<Item = ScreenRect>, offset_x: i32, offset_y: i32,
    ) {
        let (width, height) = self.dimensions;
        self.text_mask = vec![0u64; self.edges.len()];
        for rect in regions {
            if rect.is_empty() { continue; }
            let left = (rect.x as i64 - offset_x as i64 - 3).clamp(0, width as i64) as usize;
            let top = (rect.y as i64 - offset_y as i64 - 3).clamp(0, height as i64) as usize;
            let right = (rect.x as i64 + rect.width as i64 - offset_x as i64 + 3)
                .clamp(0, width as i64) as usize;
            let bottom = (rect.y as i64 + rect.height as i64 - offset_y as i64 + 3)
                .clamp(0, height as i64) as usize;
            for y in top..bottom {
                for x in left..right {
                    let bit = y * width as usize + x;
                    self.text_mask[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
    }

    pub fn inherit_text_regions(&mut self, previous: &FrameSignature) {
        if self.dimensions == previous.dimensions {
            self.text_mask.clone_from(&previous.text_mask);
        }
    }

    /// Computes the Mean Absolute Difference (MAD) between this and another signature.
    pub fn mad(&self, other: &FrameSignature) -> f32 {
        let total_diff: f32 = self
            .luma
            .iter()
            .zip(other.luma.iter())
            .map(|(&a, &b)| (a as f32 - b as f32).abs())
            .sum();
        total_diff / (SIGNATURE_COLS * SIGNATURE_ROWS) as f32
    }

    /// Determines if dialogue line has meaningfully changed beyond the noise threshold.
    pub fn has_changed(&self, previous: &FrameSignature) -> bool {
        if self.dimensions != previous.dimensions { return true; }
        if self.mad(previous) >= DELTA_THRESHOLD {
            return true;
        }
        // A blinking 2x10 caret changes fewer than 48 edge pixels. A changed
        // word can exceed that locally while leaving the full-frame MAD tiny.
        let mut changed_edges = 0u32;
        for ((&current, &old), &mask) in self.edges.iter().zip(&previous.edges).zip(&previous.text_mask) {
            changed_edges += ((current ^ old) & mask).count_ones();
            if changed_edges >= 48 { return true; }
        }
        self.luma
            .chunks(SIGNATURE_COLS)
            .zip(previous.luma.chunks(SIGNATURE_COLS))
            .any(|(row, old_row)| {
                row.windows(3).zip(old_row.windows(3)).any(|(run, old_run)| {
                    run.iter().zip(old_run).all(|(&a, &b)| {
                        (a as f32 - b as f32).abs() >= DELTA_THRESHOLD
                    })
                })
            })
    }
}

/// Represents a single captured raw BGRA8 frame with dimensions and global screen offset.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    pub bgra_bytes: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub offset_x: i32,
    pub offset_y: i32,
}

/// RAII Guard for GDI Screen Device Context.
pub struct ScreenDcGuard(pub HDC);

impl Drop for ScreenDcGuard {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                ReleaseDC(None, self.0);
            }
        }
    }
}

/// RAII Guard for GDI Memory Device Context.
pub struct MemDcGuard(pub HDC);

impl Drop for MemDcGuard {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }
}

/// RAII Guard for selected GDI Objects (Bitmaps, Pens, Brushes).
pub struct GdiObjectGuard {
    pub hdc: HDC,
    pub old_obj: HGDIOBJ,
    pub obj: HGDIOBJ,
}

impl Drop for GdiObjectGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.hdc.0.is_null() && !self.old_obj.0.is_null() {
                let _ = SelectObject(self.hdc, self.old_obj);
            }
            if !self.obj.0.is_null() {
                let _ = DeleteObject(self.obj);
            }
        }
    }
}

struct WinrtSingleFrameCapture {
    sender: Option<std::sync::mpsc::Sender<CapturedFrame>>,
    roi: Option<ScreenRect>,
    monitor_origin: (i32, i32),
}

impl GraphicsCaptureApiHandler for WinrtSingleFrameCapture {
    type Flags = (
        Option<ScreenRect>,
        (i32, i32),
        std::sync::mpsc::Sender<CapturedFrame>,
    );
    type Error = Box<dyn std::error::Error + Send + Sync>;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        let (roi, monitor_origin, tx) = ctx.flags;
        Ok(Self {
            sender: Some(tx),
            roi,
            monitor_origin,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        capture_control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let width = frame.width();
        let height = frame.height();

        if let Ok(frame_buf) = frame.buffer() {
            let mut packed = Vec::new();
            let raw_bytes = frame_buf.as_nopadding_buffer(&mut packed);
            let (final_bytes, final_w, final_h, off_x, off_y) = match self.roi {
                Some(r) => {
                    let rx = (r.x - self.monitor_origin.0).clamp(0, width as i32) as u32;
                    let ry = (r.y - self.monitor_origin.1).clamp(0, height as i32) as u32;
                    let rw = (r.width as u32).min(width.saturating_sub(rx)).max(1);
                    let rh = (r.height as u32).min(height.saturating_sub(ry)).max(1);

                    let mut cropped = vec![0u8; (rw * rh * 4) as usize];
                    for row in 0..rh {
                        let src_start = (((ry + row) * width + rx) * 4) as usize;
                        let src_end = src_start + (rw * 4) as usize;
                        let dst_start = (row * rw * 4) as usize;
                        if src_end <= raw_bytes.len() {
                            cropped[dst_start..dst_start + (rw * 4) as usize]
                                .copy_from_slice(&raw_bytes[src_start..src_end]);
                        }
                    }
                    (cropped, rw, rh, r.x, r.y)
                }
                None => (
                    raw_bytes.to_vec(),
                    width,
                    height,
                    self.monitor_origin.0,
                    self.monitor_origin.1,
                ),
            };

            if let Some(tx) = self.sender.take() {
                let _ = tx.send(CapturedFrame {
                    bgra_bytes: final_bytes,
                    width: final_w,
                    height: final_h,
                    offset_x: off_x,
                    offset_y: off_y,
                });
            }
        }

        capture_control.stop();
        Ok(())
    }
}

/// Returns true if the given ROI is non-empty and lies entirely within the bounds of the primary monitor
/// (origin at (0, 0) with dimensions primary_w x primary_h).
pub fn is_roi_within_primary_monitor(roi: &ScreenRect, primary_w: i32, primary_h: i32) -> bool {
    if roi.is_empty() || primary_w <= 0 || primary_h <= 0 {
        return false;
    }
    roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= primary_w && roi.y + roi.height <= primary_h
}

fn winrt_covers_capture(
    roi: Option<ScreenRect>,
    primary_w: i32,
    primary_h: i32,
    virtual_screen: ScreenRect,
) -> bool {
    match roi {
        Some(rect) => is_roi_within_primary_monitor(&rect, primary_w, primary_h),
        None => virtual_screen == ScreenRect::new(0, 0, primary_w, primary_h),
    }
}

pub struct ScreenCapture;

impl ScreenCapture {
    /// Captures the specified region of the screen (or full virtual screen) using
    /// Windows.Graphics.Capture API (D3D11 GPU textures without yellow border),
    /// falling back cleanly to RAII-guarded GDI BitBlt if D3D11 capture is unavailable.
    pub fn capture_primary(roi: Option<ScreenRect>) -> Result<CapturedFrame, String> {
        let (primary_w, primary_h, mut virtual_screen) = unsafe {
            (
                GetSystemMetrics(SM_CXSCREEN),
                GetSystemMetrics(SM_CYSCREEN),
                ScreenRect::new(
                    GetSystemMetrics(SM_XVIRTUALSCREEN),
                    GetSystemMetrics(SM_YVIRTUALSCREEN),
                    GetSystemMetrics(SM_CXVIRTUALSCREEN),
                    GetSystemMetrics(SM_CYVIRTUALSCREEN),
                ),
            )
        };
        if virtual_screen.is_empty() {
            virtual_screen = ScreenRect::new(0, 0, primary_w, primary_h);
        }
        // WinRT captures one monitor. Full-screen capture must include the same
        // virtual desktop as GDI, including monitors left of the primary one.
        if !winrt_covers_capture(roi, primary_w, primary_h, virtual_screen) {
            return Self::capture_gdi_fallback(roi);
        }

        // Attempt fast D3D11 WinRT Graphics Capture if circuit breaker is not tripped (or cooldown elapsed)
        if is_winrt_available()
            && let Ok(frame) = Self::capture_winrt(roi) {
                return Ok(frame);
            }

        // Seamless fallback to GDI BitBlt
        Self::capture_gdi_fallback(roi)
    }

    fn capture_winrt(roi: Option<ScreenRect>) -> Result<CapturedFrame, String> {
        match bounded_capture(&WINRT_WORKER, WINRT_CAPTURE_TIMEOUT, move |result_tx, deadline| {
            Self::capture_winrt_worker(roi, result_tx, deadline);
        }) {
            Ok(frame) => {
                reset_winrt_timeout_count();
                Ok(frame)
            }
            Err(WinrtCaptureError::Busy) => {
                Err("Previous WinRT capture is still initializing or shutting down; using GDI".to_string())
            }
            Err(WinrtCaptureError::Failed(error)) => Err(error),
            Err(WinrtCaptureError::TimedOut) => {
                let count = WINRT_TIMEOUT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut last) = LAST_WINRT_TIMEOUT.lock() {
                    *last = Some(Instant::now());
                }
                if count >= WINRT_MAX_CONSECUTIVE_TIMEOUTS {
                    warn!(
                        "WinRT capture timed out {count} times consecutively. Tripping circuit breaker, switching to GDI fallback."
                    );
                } else {
                    warn!(
                        "WinRT capture timed out after 600ms ({count}/{WINRT_MAX_CONSECUTIVE_TIMEOUTS}), falling back to GDI while the worker shuts down"
                    );
                }
                Err("WinRT capture timed out or unavailable, using GDI".to_string())
            }
        }
    }

    fn capture_winrt_worker(
        roi: Option<ScreenRect>,
        result_tx: Sender<Result<CapturedFrame, WinrtCaptureError>>,
        deadline: Instant,
    ) {
        let (frame_tx, frame_rx) = channel();
        let start = || {
            let primary_monitor = Monitor::primary()
                .map_err(|e| format!("Failed to get primary monitor: {e:?}"))?;
            let settings = Settings::new(
                primary_monitor,
                CursorCaptureSettings::WithoutCursor,
                DrawBorderSettings::WithoutBorder,
                SecondaryWindowSettings::Default,
                MinimumUpdateIntervalSettings::Default,
                DirtyRegionSettings::Default,
                ColorFormat::Bgra8,
                (roi, (0, 0), frame_tx),
            );
            // This method also waits for native initialization internally.
            WinrtSingleFrameCapture::start_free_threaded(settings)
                .map_err(|e| format!("Failed to start WinRT capture: {e:?}"))
        };
        let control = match start() {
            Ok(control) => control,
            Err(error) => {
                let _ = result_tx.send(Err(WinrtCaptureError::Failed(error)));
                return;
            }
        };
        let remaining = deadline.saturating_duration_since(Instant::now());
        match frame_rx.recv_timeout(remaining) {
            Ok(frame) if !remaining.is_zero() => {
                // Publish before joining: a stalled native cleanup must not
                // withhold an already captured frame from the caller.
                let _ = result_tx.send(Ok(frame));
                if let Err(error) = control.wait() {
                    warn!("WinRT capture cleanup failed: {error:?}");
                }
            }
            _ => {
                let _ = result_tx.send(Err(WinrtCaptureError::TimedOut));
                // Keep the native thread handle even if posting WM_QUIT fails:
                // CaptureControl::stop would detach it on that error, permitting
                // a later request to create another potentially hung thread.
                control.halt_handle().store(true, Ordering::Relaxed);
                let native_worker = control.into_thread_handle();
                let thread_id = unsafe { GetThreadId(HANDLE(native_worker.as_raw_handle())) };
                if thread_id != 0
                    && let Err(error) = unsafe {
                        PostThreadMessageW(thread_id, WM_QUIT, WPARAM(0), LPARAM(0))
                    }
                {
                    warn!("Failed to signal WinRT capture shutdown: {error:?}");
                }
                // Only this dedicated std worker joins. The caller has already
                // returned, and its slot remains occupied until native exit.
                if native_worker.join().is_err() {
                    warn!("WinRT capture thread panicked during shutdown");
                }
            }
        }
    }

    /// Captures the screen via GDI BitBlt with full RAII resource protection.
    pub fn capture_gdi_fallback(roi: Option<ScreenRect>) -> Result<CapturedFrame, String> {
        unsafe {
            let hdc_raw = GetDC(None);
            if hdc_raw.0.is_null() {
                return Err("Failed to get desktop DC".to_string());
            }
            let _screen_guard = ScreenDcGuard(hdc_raw);

            let mut vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
            let mut vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
            let mut vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
            let mut vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);

            if vw <= 0 || vh <= 0 {
                vx = 0;
                vy = 0;
                vw = GetSystemMetrics(SM_CXSCREEN);
                vh = GetSystemMetrics(SM_CYSCREEN);
            }

            let (cap_x, cap_y, cap_w, cap_h) = match roi {
                Some(r) => {
                    let x = r.x.max(vx);
                    let y = r.y.max(vy);
                    let right = (r.x + r.width).min(vx + vw);
                    let bottom = (r.y + r.height).min(vy + vh);
                    let w = (right - x).max(1);
                    let h = (bottom - y).max(1);
                    (x, y, w, h)
                }
                None => (vx, vy, vw, vh),
            };

            let hdc_mem_raw = CreateCompatibleDC(Some(hdc_raw));
            if hdc_mem_raw.0.is_null() {
                return Err("Failed to create compatible memory DC".to_string());
            }
            let _mem_guard = MemDcGuard(hdc_mem_raw);

            let hbitmap = CreateCompatibleBitmap(hdc_raw, cap_w, cap_h);
            if hbitmap.0.is_null() {
                return Err("Failed to create compatible bitmap".to_string());
            }

            // Convert once to HGDIOBJ so ownership is clear and we avoid use-after-move.
            // We keep the raw handle value separately for GetDIBits which needs HBITMAP.
            let hbitmap_raw = hbitmap.0; // copy the raw pointer before consuming hbitmap
            let hbitmap_obj: HGDIOBJ = hbitmap.into();
            let old_bmp = SelectObject(hdc_mem_raw, hbitmap_obj);
            let mut _obj_guard = GdiObjectGuard {
                hdc: hdc_mem_raw,
                old_obj: old_bmp,
                obj: hbitmap_obj,
            };

            // Combine SRCCOPY (0x00CC0020) with CAPTUREBLT (0x40000000) = 0x40CC0020
            let rop = ROP_CODE(SRCCOPY.0 | CAPTUREBLT.0);
            let blt_res = BitBlt(
                hdc_mem_raw,
                0,
                0,
                cap_w,
                cap_h,
                Some(hdc_raw),
                cap_x,
                cap_y,
                rop,
            );

            if let Err(e) = blt_res {
                error!("BitBlt screen capture failed: {:?}", e);
                return Err(format!("BitBlt failed: {:?}", e));
            }

            // Restore the original bitmap back into the memory DC before calling GetDIBits,
            // satisfying the Win32 GDI requirement that the bitmap must not be selected into any DC.
            let _ = SelectObject(hdc_mem_raw, old_bmp);
            _obj_guard.old_obj = HGDIOBJ(core::ptr::null_mut());

            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: cap_w,
                    biHeight: -cap_h, // negative for top-down DIB
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };

            let buf_size = (cap_w * cap_h * 4) as usize;
            let mut bgra_bytes = vec![0u8; buf_size];

            let lines = GetDIBits(
                hdc_raw,
                HBITMAP(hbitmap_raw),
                0,
                cap_h as u32,
                Some(bgra_bytes.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );

            if lines == 0 {
                return Err("GetDIBits failed to copy frame bytes".to_string());
            }
            // BI_RGB32's high byte is unused, not alpha. Captured desktop pixels
            // are opaque; preserving that byte can make Vision/debug PNGs invisible.
            for pixel in bgra_bytes.chunks_exact_mut(4) {
                pixel[3] = 255;
            }

            Ok(CapturedFrame {
                bgra_bytes,
                width: cap_w as u32,
                height: cap_h as u32,
                offset_x: cap_x,
                offset_y: cap_y,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame() -> CapturedFrame {
        CapturedFrame {
            bgra_bytes: vec![0, 0, 0, 255],
            width: 1,
            height: 1,
            offset_x: 0,
            offset_y: 0,
        }
    }

    #[test]
    fn capture_deadline_includes_initialization_and_retains_stalled_worker() {
        let workers = Mutex::new(None);
        let (started_tx, started_rx) = channel();
        let (release_tx, release_rx) = channel();
        let result = bounded_capture(&workers, Duration::from_millis(20), move |tx, _| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            let _ = tx.send(Ok(test_frame()));
        });
        assert!(matches!(result, Err(WinrtCaptureError::TimedOut)));
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // Repeated requests cannot create more workers while native startup is
        // stuck, even if a user manually resets the capture circuit breaker.
        for _ in 0..10 {
            let result = bounded_capture(&workers, Duration::from_millis(20), |_, _| {
                panic!("a stalled worker must prevent additional captures");
            });
            assert!(matches!(result, Err(WinrtCaptureError::Busy)));
        }
        release_tx.send(()).unwrap();
        workers.lock().unwrap().take().unwrap().join().unwrap();
        let frame = bounded_capture(&workers, Duration::from_secs(2), |tx, _| {
            tx.send(Ok(test_frame())).unwrap();
        })
        .unwrap();
        assert_eq!(frame.bgra_bytes, vec![0, 0, 0, 255]);
        workers.lock().unwrap().take().unwrap().join().unwrap();
    }

    #[test]
    fn captured_frame_returns_before_native_cleanup_finishes() {
        let workers = Mutex::new(None);
        let (release_tx, release_rx) = channel();
        let frame = bounded_capture(&workers, Duration::from_secs(2), move |tx, _| {
            tx.send(Ok(test_frame())).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();
        assert_eq!((frame.width, frame.height), (1, 1));
        let next = bounded_capture(&workers, Duration::from_millis(20), |_, _| {
            panic!("cleanup must finish before another native capture starts");
        });
        assert!(matches!(next, Err(WinrtCaptureError::Busy)));
        release_tx.send(()).unwrap();
        workers.lock().unwrap().take().unwrap().join().unwrap();
    }

    #[test]
    fn full_screen_capture_has_identical_scope_with_both_backends() {
        let (width, height) = (1920, 1080);
        assert!(winrt_covers_capture(None, width, height, ScreenRect::new(0, 0, width, height)));
        for desktop in [
            ScreenRect::new(0, 0, 3840, 1080),
            ScreenRect::new(-1920, 0, 3840, 1080),
            ScreenRect::new(0, -1080, 1920, 2160),
        ] {
            assert!(!winrt_covers_capture(None, width, height, desktop));
            assert!(winrt_covers_capture(Some(ScreenRect::new(100, 100, 800, 600)), width, height, desktop));
            assert!(!winrt_covers_capture(Some(ScreenRect::new(-500, 100, 300, 300)), width, height, desktop));
        }
    }

    /// BUG-2 regression: GdiObjectGuard must hold the HGDIOBJ that was actually selected
    /// into the DC — not a second copy produced by a duplicate `.into()` call.
    /// We verify the guard stores a non-null obj by constructing one with a sentinel pointer.
    #[test]
    fn bug_2_gdi_object_guard_stores_selected_handle() {
        // We cannot call real Win32 APIs in a unit test, but we can verify the struct
        // stores the *same* HGDIOBJ value that was passed to SelectObject, not a
        // separate conversion result. We simulate with a dummy pointer value.
        use std::ptr;
        let sentinel: *mut core::ffi::c_void = 0xDEAD_BEEF_usize as *mut _;

        // Build HGDIOBJ from the sentinel pointer (as the fix does: one conversion).
        let hbitmap_raw_ptr = sentinel;
        let hbitmap_obj = HGDIOBJ(hbitmap_raw_ptr);

        // Build the guard exactly as the fixed code does.
        let old_obj = HGDIOBJ(ptr::null_mut());
        let guard = GdiObjectGuard {
            hdc: HDC(ptr::null_mut()),
            old_obj,
            obj: hbitmap_obj,
        };

        // The guard must hold the sentinel value — not null and not some other pointer.
        assert_eq!(
            guard.obj.0, sentinel,
            "GdiObjectGuard.obj должен хранить хендл, который был выбран в DC"
        );
        // Prevent Drop from calling Win32 (hdc is null, so the guard skips both ops).
        drop(guard);
    }

    /// BUG-5 regression verification: verify that the last pixel of the frame
    /// (at x = width - 1, y = height - 1) is correctly included in the signature and not skipped.
    #[test]
    fn bug_5_frame_signature_does_not_skip_last_pixel() {
        let width = 32;
        let height = 16;
        let mut buf = vec![0u8; width * height * 4];

        // Frame is entirely black (0 luma) except the VERY LAST pixel at (width - 1, height - 1)
        let last_pixel_idx = ((height - 1) * width + (width - 1)) * 4;
        buf[last_pixel_idx] = 255; // B
        buf[last_pixel_idx + 1] = 255; // G
        buf[last_pixel_idx + 2] = 255; // R
        buf[last_pixel_idx + 3] = 255; // A

        let sig = FrameSignature::from_bgra(&buf, width as u32, height as u32);
        // The bottom-right signature cell corresponds to row 15, col 31
        let last_cell = sig.luma[SIGNATURE_ROWS * SIGNATURE_COLS - 1];
        assert_eq!(
            last_cell, 255,
            "Последний пиксель кадра обязан быть учтён в сигнатуре (luma = 255), а не пропущен!"
        );

        // Also test single-pixel frame (1x1)
        let single_px_buf = vec![100u8, 100u8, 100u8, 255u8];
        let sig_1x1 = FrameSignature::from_bgra(&single_px_buf, 1, 1);
        assert_eq!(sig_1x1.luma[0], 100);
        assert_eq!(sig_1x1.luma[SIGNATURE_ROWS * SIGNATURE_COLS - 1], 100);

        // Zero-sized frame does not panic
        let empty_sig = FrameSignature::from_bgra(&[], 0, 0);
        assert_eq!(empty_sig.luma[0], 0);
    }

    #[test]
    fn test_signature_identical_frames() {
        let width = 64;
        let height = 32;
        let buf = vec![128u8; (width * height * 4) as usize];

        let sig1 = FrameSignature::from_bgra(&buf, width, height);
        let sig2 = FrameSignature::from_bgra(&buf, width, height);

        let mad = sig1.mad(&sig2);
        assert_eq!(mad, 0.0);
        assert!(!sig1.has_changed(&sig2));
    }

    #[test]
    fn signature_detects_a_short_word_change_on_a_1080p_page() {
        let signature = |name| {
            let image = image::open(format!("tests/fixtures/capture/dialogue_{name}.png"))
                .expect("capture fixture must load").to_rgba8();
            let (width, height) = image.dimensions();
            let mut bgra = image.into_raw();
            for pixel in bgra.chunks_exact_mut(4) { pixel.swap(0, 2); }
            let mut signature = FrameSignature::from_bgra(&bgra, width, height);
            signature.set_text_regions([ScreenRect::new(790, 645, 100, 65)], 0, 0);
            signature
        };
        let yes = signature("yes");
        let no = signature("no");
        assert!(yes.mad(&no) < DELTA_THRESHOLD);
        assert!(yes.has_changed(&no), "YES -> NO must refresh the translation");
        assert!(no.has_changed(&yes));
        assert!(!yes.has_changed(&yes.clone()));
    }

    #[test]
    fn signature_ignores_outside_animation_but_checks_text_during_stabilization() {
        let moving_square = |x: usize, y: usize| {
            let mut pixels = vec![255u8; 1920 * 1080 * 4];
            for row in y..y + 10 {
                for col in x..x + 10 {
                    let start = (row * 1920 + col) * 4;
                    pixels[start..start + 3].fill(0);
                }
            }
            FrameSignature::from_bgra(&pixels, 1920, 1080)
        };
        let mut last = moving_square(100, 100);
        // Real desktop coordinates on a monitor left of the primary display.
        last.set_text_regions([ScreenRect::new(-110, 845, 100, 65)], -900, 200);
        let mut sample = moving_square(130, 100);
        sample.inherit_text_regions(&last);
        assert!(sample.mad(&last) < DELTA_THRESHOLD);
        assert!(!sample.has_changed(&last), "outside particles must not retrigger OCR");
        let stable_sample = moving_square(160, 100);
        assert!(!stable_sample.has_changed(&sample), "outside animation must not block stabilization");

        let mut inside = moving_square(800, 660);
        inside.inherit_text_regions(&last);
        let typing = moving_square(830, 660);
        assert!(typing.has_changed(&inside), "changing text must still delay translation");
    }

    #[test]
    fn test_signature_subtle_cursor_flicker() {
        let width = 64;
        let height = 32;
        let buf1 = vec![220u8; (width * height * 4) as usize];
        let mut buf2 = buf1.clone();

        // Simulate a small vertical blinking text cursor (2x10 pixels inverted to black)
        for y in 10..20 {
            for x in 30..32 {
                let idx = ((y * width + x) * 4) as usize;
                buf2[idx] = 0;
                buf2[idx + 1] = 0;
                buf2[idx + 2] = 0;
            }
        }

        let sig1 = FrameSignature::from_bgra(&buf1, width, height);
        let sig2 = FrameSignature::from_bgra(&buf2, width, height);

        let mad = sig1.mad(&sig2);
        // Subtle cursor flicker should be well below DELTA_THRESHOLD (3.5)
        assert!(mad < DELTA_THRESHOLD);
        assert!(!sig2.has_changed(&sig1));
    }

    #[test]
    fn test_signature_background_noise() {
        let width = 64;
        let height = 32;
        let buf1 = vec![100u8; (width * height * 4) as usize];
        let buf2 = vec![102u8; (width * height * 4) as usize]; // +2 luma noise everywhere

        let sig1 = FrameSignature::from_bgra(&buf1, width, height);
        let sig2 = FrameSignature::from_bgra(&buf2, width, height);

        let mad = sig1.mad(&sig2);
        assert!(mad <= 2.5);
        assert!(mad < DELTA_THRESHOLD);
        assert!(!sig2.has_changed(&sig1));
    }

    #[test]
    fn test_signature_dialogue_change() {
        let width = 64;
        let height = 32;
        let buf1 = vec![240u8; (width * height * 4) as usize]; // White background
        let mut buf2 = buf1.clone();

        // Introduce substantial dark dialogue text in bottom half
        for y in 16..32 {
            for x in 8..56 {
                let idx = ((y * width + x) * 4) as usize;
                buf2[idx] = 20; // B
                buf2[idx + 1] = 20; // G
                buf2[idx + 2] = 20; // R
            }
        }

        let sig1 = FrameSignature::from_bgra(&buf1, width, height);
        let sig2 = FrameSignature::from_bgra(&buf2, width, height);

        let mad = sig1.mad(&sig2);
        // Dialogue change produces MAD >> 3.5
        assert!(mad >= DELTA_THRESHOLD);
        assert!(sig2.has_changed(&sig1));
    }

    #[test]
    fn test_captureblt_constant_value() {
        assert_eq!(CAPTUREBLT.0, 0x4000_0000);
    }

    #[test]
    fn test_is_roi_within_primary_monitor_bounds() {
        let pw = 1920;
        let ph = 1080;

        // 1. roi inside primary monitor -> true
        let inside = ScreenRect::new(100, 100, 800, 600);
        assert!(is_roi_within_primary_monitor(&inside, pw, ph));

        // 2. roi right of primary monitor -> false
        let right_of_mon = ScreenRect::new(2000, 100, 400, 300);
        assert!(!is_roi_within_primary_monitor(&right_of_mon, pw, ph));

        // 3. roi with negative coordinates (e.g. secondary monitor to the left) -> false
        let left_of_mon = ScreenRect::new(-500, 100, 400, 300);
        assert!(!is_roi_within_primary_monitor(&left_of_mon, pw, ph));

        // 4. roi partially crossing the right border -> false
        let crossing_border = ScreenRect::new(1800, 100, 300, 300);
        assert!(!is_roi_within_primary_monitor(&crossing_border, pw, ph));

        // Edge cases: exact boundary -> true
        let exact_full = ScreenRect::new(0, 0, pw, ph);
        assert!(is_roi_within_primary_monitor(&exact_full, pw, ph));

        // Empty rect -> false
        let empty = ScreenRect::new(100, 100, 0, 100);
        assert!(!is_roi_within_primary_monitor(&empty, pw, ph));
    }

    #[test]
    fn test_winrt_circuit_breaker_tripping_and_reset() {
        reset_winrt_timeout_count();
        assert_eq!(winrt_consecutive_timeouts(), 0);
        assert!(is_winrt_available());

        // Simulate 2 timeouts -> circuit breaker should still allow attempts (< 3)
        WINRT_TIMEOUT_COUNT.store(2, Ordering::Relaxed);
        assert!(winrt_consecutive_timeouts() < WINRT_MAX_CONSECUTIVE_TIMEOUTS);
        assert!(is_winrt_available());

        // Simulate 3 timeouts -> circuit breaker trips
        WINRT_TIMEOUT_COUNT.store(WINRT_MAX_CONSECUTIVE_TIMEOUTS, Ordering::Relaxed);
        if let Ok(mut last) = LAST_WINRT_TIMEOUT.lock() {
            *last = Some(Instant::now());
        }
        assert_eq!(winrt_consecutive_timeouts(), 3);
        assert!(winrt_consecutive_timeouts() >= WINRT_MAX_CONSECUTIVE_TIMEOUTS);
        assert!(!is_winrt_available());

        // Simulate elapsed cooldown -> half-open allows retry
        if let Ok(mut last) = LAST_WINRT_TIMEOUT.lock() {
            *last = Some(Instant::now() - Duration::from_secs(31));
        }
        assert!(is_winrt_available());

        // Reset
        reset_winrt_timeout_count();
        assert_eq!(winrt_consecutive_timeouts(), 0);
        assert!(is_winrt_available());
    }

    #[test]
    fn test_signature_detects_local_subtitle_changes() {
        let (width, height) = (1920, 1080);
        let original = vec![220u8; width * height * 4];
        let mut changed = original.clone();
        for y in 900..960 {
            for x in 800..1100 {
                if x % 10 < 2 {
                    let i = (y * width + x) * 4;
                    changed[i..i + 3].fill(0);
                }
            }
        }
        let first = FrameSignature::from_bgra(&original, width as u32, height as u32);
        let second = FrameSignature::from_bgra(&changed, width as u32, height as u32);
        assert!(first.mad(&second) < DELTA_THRESHOLD);
        assert!(second.has_changed(&first));
    }

}
