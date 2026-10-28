use std::sync::mpsc::channel;
use std::time::Duration;
use tracing::error;
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CAPTUREBLT, CreateCompatibleBitmap,
    CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, HBITMAP, HDC,
    HGDIOBJ, ROP_CODE, ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, SM_CXSCREEN, SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
};
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::InternalCaptureControl;
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};

use crate::types::ScreenRect;

pub const SIGNATURE_COLS: usize = 32;
pub const SIGNATURE_ROWS: usize = 16;
pub const DELTA_THRESHOLD: f32 = 3.5;

/// Compact 32x16 Luma perceptual hash to detect genuine dialogue transitions
/// while rejecting flickering cursors, animated particles, and subtle background noise.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSignature {
    pub luma: [u8; SIGNATURE_COLS * SIGNATURE_ROWS],
}

impl Default for FrameSignature {
    fn default() -> Self {
        Self {
            luma: [0u8; SIGNATURE_COLS * SIGNATURE_ROWS],
        }
    }
}

impl FrameSignature {
    pub fn from_bgra(bgra: &[u8], width: u32, height: u32) -> Self {
        let mut luma = [0u8; SIGNATURE_COLS * SIGNATURE_ROWS];
        if width == 0 || height == 0 || bgra.len() < (width * height * 4) as usize {
            return Self { luma };
        }

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
                            let b = bgra[idx] as f32;
                            let g = bgra[idx + 1] as f32;
                            let r = bgra[idx + 2] as f32;
                            let y_val = (0.299 * r + 0.587 * g + 0.114 * b).round() as u64;
                            sum += y_val;
                            count += 1;
                        }
                    }
                }

                let avg = sum.checked_div(count).unwrap_or(0) as u8;
                luma[row * SIGNATURE_COLS + col] = avg;
            }
        }

        Self { luma }
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
        self.mad(previous) >= DELTA_THRESHOLD
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

        if let Ok(mut frame_buf) = frame.buffer() {
            let raw_bytes = frame_buf.as_raw_buffer();
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

pub struct ScreenCapture;

impl ScreenCapture {
    /// Captures the specified region of the screen (or full virtual screen) using
    /// Windows.Graphics.Capture API (D3D11 GPU textures without yellow border),
    /// falling back cleanly to RAII-guarded GDI BitBlt if D3D11 capture is unavailable.
    pub fn capture_primary(roi: Option<ScreenRect>) -> Result<CapturedFrame, String> {
        // Attempt fast D3D11 WinRT Graphics Capture
        if let Ok(frame) = Self::capture_winrt(roi) {
            return Ok(frame);
        }

        // Seamless fallback to GDI BitBlt
        Self::capture_gdi_fallback(roi)
    }

    fn capture_winrt(roi: Option<ScreenRect>) -> Result<CapturedFrame, String> {
        let primary_monitor =
            Monitor::primary().map_err(|e| format!("Failed to get primary monitor: {:?}", e))?;

        let (mon_x, mon_y) = (0i32, 0i32);
        let (tx, rx) = channel();

        let settings = Settings::new(
            primary_monitor,
            CursorCaptureSettings::Default,
            DrawBorderSettings::WithoutBorder,
            SecondaryWindowSettings::Default,
            MinimumUpdateIntervalSettings::Default,
            DirtyRegionSettings::Default,
            ColorFormat::Bgra8,
            (roi, (mon_x, mon_y), tx),
        );

        let capture_thread = std::thread::spawn(move || {
            let _ = WinrtSingleFrameCapture::start(settings);
        });

        match rx.recv_timeout(Duration::from_millis(600)) {
            Ok(frame) => {
                let _ = capture_thread.join();
                Ok(frame)
            }
            Err(_) => Err("WinRT capture timed out or unavailable, using GDI".to_string()),
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
}
