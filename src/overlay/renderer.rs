use std::ffi::c_void;
use windows::Win32::Foundation::{COLORREF, D2DERR_RECREATE_TARGET, E_FAIL, HWND, POINT, RECT, SIZE};
use windows::Win32::Graphics::Direct2D::Common::{
    D2D_RECT_F, D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_PIXEL_FORMAT,
};
use windows::Win32::Graphics::Direct2D::{
    D2D1_DRAW_TEXT_OPTIONS_CLIP, D2D1_DRAW_TEXT_OPTIONS_NONE, D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1_RENDER_TARGET_PROPERTIES,
    D2D1_RENDER_TARGET_TYPE_DEFAULT, D2D1_RENDER_TARGET_USAGE_NONE, D2D1_ROUNDED_RECT,
    D2D1CreateFactory, ID2D1DCRenderTarget, ID2D1Factory, ID2D1SolidColorBrush,
};
use windows::Win32::Graphics::DirectWrite::{
    DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL,
    DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_MEASURING_MODE_NATURAL,
    DWRITE_PARAGRAPH_ALIGNMENT_NEAR, DWRITE_TEXT_ALIGNMENT_LEADING, DWRITE_TEXT_METRICS,
    DWRITE_WORD_WRAPPING_WRAP, DWriteCreateFactory, IDWriteFactory, IDWriteTextFormat,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;
use windows::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION,
    CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC,
    GetMonitorInfoW, HDC, HGDIOBJ, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromPoint, SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, SM_CXSCREEN, ULW_ALPHA, UpdateLayeredWindow,
};
use windows::core::{Result, w};

use crate::capture::ScreenDcGuard;
use crate::types::{AppMode, ScreenRect, TranslationBlock};

pub struct RenderScene<'a> {
    pub blocks: &'a [TranslationBlock],
    pub cursor_card: Option<&'a TranslationBlock>,
    pub save_button_enabled: bool,
    pub active_roi: Option<ScreenRect>,
    pub auto_mode: bool,
    pub provider_name: &'a str,
    pub ocr_engine_name: &'a str,
    pub mode: AppMode,
    pub selection_rect: Option<ScreenRect>,
    pub status_msg: Option<&'a str>,
    pub is_loading: bool,
}

fn card_monitor_bounds(source: ScreenRect, desktop: ScreenRect) -> ScreenRect {
    let mut info = MONITORINFO {
        cbSize: core::mem::size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    unsafe {
        // Cursor cards expand the word's width for layout. Their original
        // position, rather than that expanded rectangle, anchors the monitor.
        let monitor = MonitorFromPoint(POINT { x: source.x, y: source.y }, MONITOR_DEFAULTTONEAREST);
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return desktop;
        }
    }
    let left = info.rcMonitor.left.max(desktop.x);
    let top = info.rcMonitor.top.max(desktop.y);
    let right = info.rcMonitor.right.min(desktop.x + desktop.width);
    let bottom = info.rcMonitor.bottom.min(desktop.y + desktop.height);
    if right <= left || bottom <= top {
        return desktop;
    }
    ScreenRect::new(left, top, right - left, bottom - top)
}

fn card_layout_width(bounds: ScreenRect, source_width: i32, button_space: f32) -> f32 {
    let max_width = (bounds.width as f32 - 40.0 - button_space).max(1.0);
    (source_width as f32).max(180.0).min(max_width)
}

fn card_position(
    bounds: ScreenRect, source: ScreenRect, card_w: f32, card_h: f32, button_space: f32,
) -> (f32, f32) {
    let left = (source.x as f32 - 8.0).max(bounds.x as f32 + 4.0)
        .min((bounds.x as f32 + bounds.width as f32 - card_w - button_space - 4.0)
            .max(bounds.x as f32 + 4.0));
    let top = (source.y as f32 - 8.0).max(bounds.y as f32 + 4.0)
        .min((bounds.y as f32 + bounds.height as f32 - card_h - 4.0)
            .max(bounds.y as f32 + 4.0));
    (left, top)
}

pub struct MemDcGuard(pub HDC);

impl MemDcGuard {
    pub fn take(&mut self) -> HDC {
        let h = self.0;
        self.0 = HDC(core::ptr::null_mut());
        h
    }
}

impl Drop for MemDcGuard {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }
}

pub struct BitmapGuard(pub HGDIOBJ);

impl BitmapGuard {
    pub fn take(&mut self) -> HGDIOBJ {
        let h = self.0;
        self.0 = HGDIOBJ(core::ptr::null_mut());
        h
    }
}

impl Drop for BitmapGuard {
    fn drop(&mut self) {
        if !self.0.0.is_null() {
            unsafe {
                let _ = DeleteObject(self.0);
            }
        }
    }
}

// Restore the previous object before BitmapGuard attempts to delete the selected bitmap.
struct SelectedBitmapGuard {
    dc: HDC,
    previous: HGDIOBJ,
}

impl Drop for SelectedBitmapGuard {
    fn drop(&mut self) {
        if !self.dc.is_invalid() {
            unsafe { SelectObject(self.dc, self.previous); }
        }
    }
}

pub struct D2dDrawGuard<'a> {
    rt: &'a ID2D1DCRenderTarget,
    active: bool,
}

impl<'a> D2dDrawGuard<'a> {
    pub fn new(rt: &'a ID2D1DCRenderTarget) -> Self {
        unsafe {
            rt.BeginDraw();
        }
        Self { rt, active: true }
    }

    pub fn finish(mut self) -> Result<()> {
        self.active = false;
        unsafe { self.rt.EndDraw(None, None) }
    }
}

impl<'a> Drop for D2dDrawGuard<'a> {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                let _ = self.rt.EndDraw(None, None);
            }
        }
    }
}

pub struct OverlayRenderer {
    origin_x: i32,
    origin_y: i32,
    width: i32,
    height: i32,
    font_family: String,
    font_size: f32,
    overlay_bg_alpha: f32,
    #[cfg(test)]
    next_render_error: Option<windows::core::HRESULT>,
    hdc_mem: HDC,
    hbitmap: HGDIOBJ,
    old_bitmap: HGDIOBJ,
    _bits_ptr: *mut c_void,
    dc_rt: ID2D1DCRenderTarget,
    dwrite_factory: IDWriteFactory,
    text_format_card: IDWriteTextFormat,
    text_format_hud: IDWriteTextFormat,
    brush_card_bg: ID2D1SolidColorBrush,
    brush_card_border: ID2D1SolidColorBrush,
    brush_text: ID2D1SolidColorBrush,
    brush_hud_bg: ID2D1SolidColorBrush,
    brush_hud_border: ID2D1SolidColorBrush,
    brush_hud_text: ID2D1SolidColorBrush,
    brush_hud_active: ID2D1SolidColorBrush,
    brush_hud_loading: ID2D1SolidColorBrush,
    brush_selection_veil: ID2D1SolidColorBrush,
    brush_selection_border: ID2D1SolidColorBrush,
    brush_roi_active: ID2D1SolidColorBrush,
}

impl OverlayRenderer {
    pub fn new(
        origin_x: i32,
        origin_y: i32,
        width: i32,
        height: i32,
        font_family: &str,
        font_size: f32,
        overlay_bg_alpha: f32,
    ) -> Result<Self> {
        if width <= 0 || height <= 0 {
            return Err(windows::core::Error::from_hresult(E_FAIL));
        }
        unsafe {
            // 1. Create 32-bit ARGB DIB Section and memory DC.
            //    hdc_screen, hdc_mem and hbitmap are protected by RAII guards so they are
            //    released on any early-return caused by a `?` operator during Direct2D/DirectWrite init.
            let hdc_screen = GetDC(None);
            if hdc_screen.0.is_null() {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            let _screen_guard = ScreenDcGuard(hdc_screen);

            let hdc_mem_raw = CreateCompatibleDC(Some(hdc_screen));
            if hdc_mem_raw.0.is_null() {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            let mut mem_guard = MemDcGuard(hdc_mem_raw);

            let bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width,
                    biHeight: -height, // top-down
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut bits_ptr = core::ptr::null_mut();
            let hbitmap = CreateDIBSection(
                Some(hdc_screen),
                &bmi,
                DIB_RGB_COLORS,
                &mut bits_ptr,
                None,
                0,
            )?;
            let mut bitmap_guard = BitmapGuard(hbitmap.into());

            let old_bitmap = SelectObject(hdc_mem_raw, bitmap_guard.0);
            if old_bitmap.is_invalid() {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            let mut selection_guard = SelectedBitmapGuard { dc: hdc_mem_raw, previous: old_bitmap };
            // hdc_screen can now be released early, before Direct2D initialization.
            drop(_screen_guard);

            // 2. Create Direct2D and DirectWrite Factories
            let d2d_factory: ID2D1Factory =
                D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
            let dwrite_factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;

            // 3. Create Direct2D DC Render Target with Premultiplied Alpha
            let pixel_format = D2D1_PIXEL_FORMAT {
                format: DXGI_FORMAT_B8G8R8A8_UNORM,
                alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED,
            };

            let rt_props = D2D1_RENDER_TARGET_PROPERTIES {
                r#type: D2D1_RENDER_TARGET_TYPE_DEFAULT,
                pixelFormat: pixel_format,
                dpiX: 96.0,
                dpiY: 96.0,
                usage: D2D1_RENDER_TARGET_USAGE_NONE,
                minLevel: windows::Win32::Graphics::Direct2D::D2D1_FEATURE_LEVEL_DEFAULT,
            };

            let dc_rt = d2d_factory.CreateDCRenderTarget(&rt_props)?;

            // 4. Create Text Formats
            let family_wide: Vec<u16> = font_family
                .encode_utf16()
                .chain(core::iter::once(0))
                .collect();
            let family_pcwstr = windows::core::PCWSTR(family_wide.as_ptr());
            let card_font_size = font_size.max(8.0);
            let hud_font_size = (card_font_size * 0.8).clamp(10.0, 16.0);

            let text_format_card = dwrite_factory.CreateTextFormat(
                family_pcwstr,
                None,
                DWRITE_FONT_WEIGHT_SEMI_BOLD,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                card_font_size,
                w!("ru-ru"),
            )?;
            let _ = text_format_card.SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
            let _ = text_format_card.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
            let _ = text_format_card.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);

            let text_format_hud = dwrite_factory.CreateTextFormat(
                family_pcwstr,
                None,
                DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                hud_font_size,
                w!("ru-ru"),
            )?;

            // 5. Create Solid Brushes
            let bg_alpha = overlay_bg_alpha.clamp(0.05, 1.0);
            let brush_card_bg = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.08,
                    g: 0.09,
                    b: 0.13,
                    a: bg_alpha,
                },
                None,
            )?;
            let brush_card_border = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.35,
                    g: 0.65,
                    b: 1.0,
                    a: 0.95,
                },
                None,
            )?;
            let brush_text = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.96,
                    g: 0.97,
                    b: 1.0,
                    a: 1.0,
                },
                None,
            )?;

            let brush_hud_bg = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.05,
                    g: 0.06,
                    b: 0.09,
                    a: bg_alpha,
                },
                None,
            )?;
            let brush_hud_border = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.25,
                    g: 0.35,
                    b: 0.50,
                    a: 0.85,
                },
                None,
            )?;
            let brush_hud_text = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.85,
                    g: 0.88,
                    b: 0.95,
                    a: 1.0,
                },
                None,
            )?;
            let brush_hud_active = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.20,
                    g: 0.90,
                    b: 0.55,
                    a: 1.0,
                },
                None,
            )?;
            let brush_hud_loading = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 1.0,
                    g: 0.75,
                    b: 0.20,
                    a: 1.0,
                },
                None,
            )?;

            let brush_selection_veil = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 0.35,
                },
                None,
            )?;
            let brush_selection_border = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 1.0,
                    g: 0.85,
                    b: 0.20,
                    a: 1.0,
                },
                None,
            )?;
            let brush_roi_active = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.30,
                    g: 0.75,
                    b: 1.0,
                    a: 0.85,
                },
                None,
            )?;

            // Construction succeeded: take ownership of hdc_mem and hbitmap so their
            // guards do not delete them on exit. OverlayRenderer::drop will delete them.
            let hdc_mem = mem_guard.take();
            let hbitmap = bitmap_guard.take();
            selection_guard.dc = HDC(core::ptr::null_mut());

            Ok(Self {
                origin_x,
                origin_y,
                width,
                height,
                font_family: font_family.to_string(),
                font_size,
                overlay_bg_alpha,
                #[cfg(test)]
                next_render_error: None,
                hdc_mem,
                hbitmap,
                old_bitmap,
                _bits_ptr: bits_ptr,
                dc_rt,
                dwrite_factory,
                text_format_card,
                text_format_hud,
                brush_card_bg,
                brush_card_border,
                brush_text,
                brush_hud_bg,
                brush_hud_border,
                brush_hud_text,
                brush_hud_active,
                brush_hud_loading,
                brush_selection_veil,
                brush_selection_border,
                brush_roi_active,
            })
        }
    }

    /// Replace size-dependent and device-dependent resources only after allocation succeeds.
    pub fn reconfigure(&mut self, bounds: ScreenRect) -> Result<()> {
        let replacement = Self::new(bounds.x, bounds.y, bounds.width, bounds.height,
            &self.font_family, self.font_size, self.overlay_bg_alpha)?;
        *self = replacement;
        Ok(())
    }

    /// Renders the complete scene (translation cards, HUD bar, ROI selection) and flushes to DWM.
    pub fn render(&mut self, hwnd: HWND, scene: &RenderScene) -> Result<Option<ScreenRect>> {
        match self.render_once(hwnd, scene) {
            Err(error) if error.code() == D2DERR_RECREATE_TARGET => {
                self.reconfigure(ScreenRect::new(self.origin_x, self.origin_y, self.width, self.height))?;
                self.render_once(hwnd, scene)
            }
            result => result,
        }
    }

    fn render_once(&mut self, hwnd: HWND, scene: &RenderScene) -> Result<Option<ScreenRect>> {
        #[cfg(test)]
        if let Some(error) = self.next_render_error.take() {
            return Err(windows::core::Error::from_hresult(error));
        }
        let mut save_button = None;
        unsafe {
            let bind_rect = RECT {
                left: 0,
                top: 0,
                right: self.width,
                bottom: self.height,
            };

            self.dc_rt.BindDC(self.hdc_mem, &bind_rect)?;
            let draw_guard = D2dDrawGuard::new(&self.dc_rt);

            // Clear to transparent premultiplied alpha
            self.dc_rt.Clear(Some(&D2D1_COLOR_F {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            }));

            // 1. If in interactive Selection mode: draw dim veil and live selection box
            if scene.mode == AppMode::SelectingRoi {
                let full_rect = D2D_RECT_F {
                    left: 0.0,
                    top: 0.0,
                    right: self.width as f32,
                    bottom: self.height as f32,
                };
                self.dc_rt
                    .FillRectangle(&full_rect, &self.brush_selection_veil);

                if let Some(rect) = scene.selection_rect.filter(|r| !r.is_empty()) {
                    let d2d_rect = D2D_RECT_F {
                        left: (rect.x - self.origin_x) as f32,
                        top: (rect.y - self.origin_y) as f32,
                        right: (rect.x - self.origin_x + rect.width) as f32,
                        bottom: (rect.y - self.origin_y + rect.height) as f32,
                    };
                    self.dc_rt
                        .DrawRectangle(&d2d_rect, &self.brush_selection_border, 2.5, None);

                    let dim_text = format!("{} x {}", rect.width, rect.height);
                    let utf16: Vec<u16> = dim_text.encode_utf16().collect();
                    let text_rect = D2D_RECT_F {
                        left: d2d_rect.left,
                        top: (d2d_rect.top - 22.0).max(0.0),
                        right: d2d_rect.left + 150.0,
                        bottom: d2d_rect.top,
                    };
                    self.dc_rt.DrawText(
                        &utf16,
                        &self.text_format_hud,
                        &text_rect,
                        &self.brush_selection_border,
                        D2D1_DRAW_TEXT_OPTIONS_NONE,
                        DWRITE_MEASURING_MODE_NATURAL,
                    );
                }
            } else {
                // 2. Normal mode: if active ROI exists, draw subtle outline
                if let Some(roi) = scene.active_roi.filter(|r| !r.is_empty()) {
                    let d2d_roi = D2D_RECT_F {
                        left: (roi.x - self.origin_x) as f32,
                        top: (roi.y - self.origin_y) as f32,
                        right: (roi.x - self.origin_x + roi.width) as f32,
                        bottom: (roi.y - self.origin_y + roi.height) as f32,
                    };
                    self.dc_rt
                        .DrawRectangle(&d2d_roi, &self.brush_roi_active, 1.5, None);
                }

                // 3. Draw Translation Blocks as floating rounded cards with DirectWrite TextLayout
                for (block, is_cursor_card) in scene.blocks.iter().map(|b| (b, false))
                    .chain(scene.cursor_card.map(|b| (b, true)))
                {
                    let show_button = is_cursor_card && scene.save_button_enabled;
                    let button_space = if show_button { 34.0 } else { 0.0 };
                    let pad = 8.0;
                    let bounds = card_monitor_bounds(block.rect,
                        ScreenRect::new(self.origin_x, self.origin_y, self.width, self.height))
                        .to_local(self.origin_x, self.origin_y);
                    let min_width = 140.0_f32.min((bounds.width as f32 - button_space - 8.0).max(1.0));
                    let target_layout_w = card_layout_width(bounds, block.rect.width, button_space);
                    let max_layout_h = (bounds.height as f32 - 40.0).max(1.0);

                    let utf16: Vec<u16> = block.translated_text.encode_utf16().collect();
                    let text_layout = self.dwrite_factory.CreateTextLayout(
                        &utf16,
                        &self.text_format_card,
                        target_layout_w,
                        max_layout_h,
                    )?;

                    let mut metrics = DWRITE_TEXT_METRICS::default();
                    text_layout.GetMetrics(&mut metrics)?;

                    let card_w =
                        (metrics.widthIncludingTrailingWhitespace + pad * 2.0).max(min_width);
                    // DirectWrite reports the full text height even when it
                    // exceeds the layout. Bound the plate and clip its text.
                    let card_h = (metrics.height.min(max_layout_h) + pad * 2.0).max(30.0);

                    let (left, top) = card_position(bounds,
                        block.rect.to_local(self.origin_x, self.origin_y), card_w, card_h, button_space);
                    let right = left + card_w;
                    let bottom = top + card_h;

                    let card_rect = D2D_RECT_F {
                        left,
                        top,
                        right,
                        bottom,
                    };

                    let rounded_rect = D2D1_ROUNDED_RECT {
                        rect: card_rect,
                        radiusX: 7.0,
                        radiusY: 7.0,
                    };

                    // Draw dark rounded plate
                    self.dc_rt
                        .FillRoundedRectangle(&rounded_rect, &self.brush_card_bg);
                    // Draw neon accent border
                    self.dc_rt.DrawRoundedRectangle(
                        &rounded_rect,
                        &self.brush_card_border,
                        1.2,
                        None,
                    );

                    // Draw text inside card
                    let text_pt = windows_numerics::Vector2 {
                        X: left + pad,
                        Y: top + pad,
                    };
                    self.dc_rt.DrawTextLayout(
                        text_pt,
                        &text_layout,
                        &self.brush_text,
                        D2D1_DRAW_TEXT_OPTIONS_CLIP,
                    );

                    if show_button {
                        let button = D2D_RECT_F {
                            left: right + 6.0,
                            top,
                            right: right + 34.0,
                            bottom: top + 28.0,
                        };
                        let rounded = D2D1_ROUNDED_RECT { rect: button, radiusX: 5.0, radiusY: 5.0 };
                        self.dc_rt.FillRoundedRectangle(&rounded, &self.brush_card_bg);
                        self.dc_rt.DrawRoundedRectangle(&rounded, &self.brush_card_border, 1.2, None);
                        let label_rect = D2D_RECT_F {
                            left: button.left + 8.0,
                            top: button.top + 4.0,
                            ..button
                        };
                        self.dc_rt.DrawText(
                            &['+' as u16], &self.text_format_hud, &label_rect,
                            &self.brush_text, D2D1_DRAW_TEXT_OPTIONS_NONE, DWRITE_MEASURING_MODE_NATURAL,
                        );
                        save_button = Some(ScreenRect::new(
                            self.origin_x + button.left.floor() as i32,
                            self.origin_y + button.top.floor() as i32,
                            28, 28,
                        ));
                    }
                }
            }

            // 4. Draw HUD status badge
            self.draw_hud(scene)?;

            if let Err(e) = draw_guard.finish() {
                tracing::error!("Direct2D EndDraw error: {:?}", e);
                return Err(e);
            }

            // 5. UpdateLayeredWindow to present frame
            let hdc_screen = GetDC(None);
            if hdc_screen.0.is_null() {
                return Err(windows::core::Error::from_hresult(E_FAIL));
            }
            let _screen_guard = ScreenDcGuard(hdc_screen);

            let pt_dst = POINT {
                x: self.origin_x,
                y: self.origin_y,
            };
            let pt_src = POINT { x: 0, y: 0 };
            let size = SIZE {
                cx: self.width,
                cy: self.height,
            };
            let blend = BLENDFUNCTION {
                BlendOp: AC_SRC_OVER as u8,
                BlendFlags: 0,
                SourceConstantAlpha: 255,
                AlphaFormat: AC_SRC_ALPHA as u8,
            };

            let ulw_res = UpdateLayeredWindow(
                hwnd,
                Some(hdc_screen),
                Some(&pt_dst),
                Some(&size),
                Some(self.hdc_mem),
                Some(&pt_src),
                COLORREF(0),
                Some(&blend),
                ULW_ALPHA,
            );

            if let Err(e) = ulw_res {
                tracing::error!("UpdateLayeredWindow error: {:?}", e);
                return Err(e);
            }
        }

        Ok(save_button)
    }

    fn draw_hud(&self, scene: &RenderScene<'_>) -> Result<()> {
        unsafe {
            // Center HUD on the primary monitor rather than the whole virtual desktop span
            let primary_w = GetSystemMetrics(SM_CXSCREEN).max(320);
            let hud_w = (primary_w as f32 - 40.0).clamp(300.0, 1100.0);
            let hud_h = 36.0;
            let primary_local_x = -self.origin_x as f32;
            let hud_x = primary_local_x + (primary_w as f32 - hud_w) / 2.0;
            let hud_y = (-self.origin_y as f32) + 12.0;

            let rect = D2D_RECT_F {
                left: hud_x,
                top: hud_y,
                right: hud_x + hud_w,
                bottom: hud_y + hud_h,
            };

            let rounded_hud = D2D1_ROUNDED_RECT {
                rect,
                radiusX: 18.0,
                radiusY: 18.0,
            };

            self.dc_rt
                .FillRoundedRectangle(&rounded_hud, &self.brush_hud_bg);
            let border_brush = if scene.is_loading {
                &self.brush_hud_loading
            } else {
                &self.brush_hud_border
            };
            self.dc_rt
                .DrawRoundedRectangle(&rounded_hud, border_brush, 1.2, None);

            let auto_str = if scene.auto_mode {
                "[AUTO: ON]"
            } else {
                "[AUTO: OFF]"
            };
            let loading_prefix = if scene.is_loading {
                "[ЗАГРУЗКА ⏳] "
            } else {
                ""
            };
            let status_str = scene.status_msg.unwrap_or("Готов");

            let hud_text = match scene.mode {
                AppMode::SelectingRoi => {
                    " [РЕЖИМ ВЫДЕЛЕНИЯ] Зажмите ЛКМ и выделите область на экране | Esc: Отмена "
                        .to_string()
                }
                AppMode::Normal => {
                    format!(
                        " ScreenTranslator | {}{} | [{} | {}] | Alt+T: Перевод | Alt+S: Выделение | Alt+A: {} | Alt+P: Движок | Alt+O: OCR | Alt+C: Скрыть | Alt+Q: Выход",
                        loading_prefix, status_str, scene.provider_name, scene.ocr_engine_name, auto_str
                    )
                }
            };

            let utf16: Vec<u16> = hud_text.encode_utf16().collect();
            let text_rect = D2D_RECT_F {
                left: rect.left + 16.0,
                top: rect.top + 9.0,
                right: rect.right - 16.0,
                bottom: rect.bottom - 4.0,
            };

            let brush = if scene.is_loading {
                &self.brush_hud_loading
            } else if scene.auto_mode {
                &self.brush_hud_active
            } else {
                &self.brush_hud_text
            };

            self.dc_rt.DrawText(
                &utf16,
                &self.text_format_hud,
                &text_rect,
                brush,
                D2D1_DRAW_TEXT_OPTIONS_NONE,
                DWRITE_MEASURING_MODE_NATURAL,
            );
        }

        Ok(())
    }
}

impl Drop for OverlayRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = SelectObject(self.hdc_mem, self.old_bitmap);
            let _ = DeleteDC(self.hdc_mem);
            let _ = DeleteObject(self.hbitmap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_selection_does_not_follow_the_expanded_card_width() {
        use windows::Win32::Graphics::Gdi::MonitorFromPoint;
        let mut info = MONITORINFO {
            cbSize: core::mem::size_of::<MONITORINFO>() as u32, ..Default::default()
        };
        unsafe {
            let monitor = MonitorFromPoint(POINT { x: 0, y: 0 }, MONITOR_DEFAULTTONEAREST);
            assert!(GetMonitorInfoW(monitor, &mut info).as_bool());
        }
        let desktop = ScreenRect::new(-32768, -32768, 65536, 65536);
        let source = ScreenRect::new(info.rcMonitor.right - 20, info.rcMonitor.top + 20, 20, 20);
        let expected = ScreenRect::new(info.rcMonitor.left, info.rcMonitor.top,
            info.rcMonitor.right - info.rcMonitor.left, info.rcMonitor.bottom - info.rcMonitor.top);
        for width in [20, 420, 10000] {
            assert_eq!(card_monitor_bounds(ScreenRect { width, ..source }, desktop), expected);
        }
    }

    #[test]
    fn long_card_uses_real_text_metrics_and_clips_pixels_to_available_height() {
        use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, WS_EX_LAYERED, WS_POPUP};
        let hwnd = unsafe { CreateWindowExW(WS_EX_LAYERED, w!("STATIC"), w!("Long card test"),
            WS_POPUP, 0, 0, 640, 240, None, None, None, None).unwrap() };
        let mut renderer = OverlayRenderer::new(0, 0, 640, 240, "Segoe UI", 18.0, 0.9).unwrap();
        let card = TranslationBlock {
            original_text: "paragraph".into(),
            translated_text: "word sentence paragraph ".repeat(300),
            rect: ScreenRect::new(40, 180, 420, 20),
        };
        let text = card.translated_text.encode_utf16().collect::<Vec<_>>();
        let mut metrics = DWRITE_TEXT_METRICS::default();
        unsafe {
            let layout = renderer.dwrite_factory.CreateTextLayout(&text, &renderer.text_format_card, 420.0, 200.0).unwrap();
            layout.GetMetrics(&mut metrics).unwrap();
        }
        assert!(metrics.height > 240.0, "exercise DirectWrite overflow, not a short card");
        let scene = RenderScene {
            blocks: &[], cursor_card: Some(&card), save_button_enabled: true,
            active_roi: None, auto_mode: false, provider_name: "Google", ocr_engine_name: "WinRT",
            mode: AppMode::Normal, selection_rect: None, status_msg: None, is_loading: false,
        };
        let result = renderer.render(hwnd, &scene);
        unsafe { DestroyWindow(hwnd).unwrap(); }
        let button = result.unwrap().unwrap();
        assert!(button.y >= 4 && button.y + button.height <= 236);
        let pixels = unsafe { std::slice::from_raw_parts(renderer._bits_ptr.cast::<u8>(), 640 * 240 * 4) };
        assert!(pixels.chunks_exact(4).skip(237 * 640).all(|pixel| pixel[3] == 0),
            "card background and text must leave the bottom margin transparent; first pixel: {:?}",
            pixels.chunks_exact(4).enumerate().skip(237 * 640).find(|(_, pixel)| pixel[3] != 0)
                .map(|(index, pixel)| (index % 640, index / 640, pixel)));
    }

    #[test]
    fn cards_fit_their_monitor_in_an_uneven_virtual_desktop() {
        // The right monitor extends below the primary; that gap cannot hold a card.
        for monitor in [ScreenRect::new(0, 0, 1920, 1080), ScreenRect::new(-1280, -200, 1280, 1024)] {
            let source = ScreenRect::new(monitor.x + monitor.width - 30,
                monitor.y + monitor.height - 10, 4000, 20);
            let button_space = 34.0;
            let layout_w = card_layout_width(monitor, source.width, button_space);
            let card_w = layout_w + 16.0;
            let card_h = 200.0;
            let (left, top) = card_position(monitor, source, card_w, card_h, button_space);
            assert!(left >= monitor.x as f32 + 4.0);
            assert!(top >= monitor.y as f32 + 4.0);
            assert!(left + card_w + button_space <= (monitor.x + monitor.width) as f32 - 4.0);
            assert!(top + card_h <= (monitor.y + monitor.height) as f32 - 4.0);
            assert!(layout_w < source.width as f32);
        }
    }

    #[test]
    fn selected_bitmap_is_restored_before_failure_cleanup() {
        use windows::Win32::Graphics::Gdi::{CreateCompatibleBitmap, GetCurrentObject, OBJ_BITMAP};
        unsafe {
            let screen = GetDC(None);
            let _screen_guard = ScreenDcGuard(screen);
            let memory = CreateCompatibleDC(Some(screen));
            let _memory_guard = MemDcGuard(memory);
            let bitmap = CreateCompatibleBitmap(screen, 20, 20);
            let mut bitmap_guard = BitmapGuard(bitmap.into());
            let previous = SelectObject(memory, bitmap_guard.0);
            {
                let _selection_guard = SelectedBitmapGuard { dc: memory, previous };
                assert_eq!(GetCurrentObject(memory, OBJ_BITMAP), bitmap_guard.0);
                // Leaving this scope models an early return during Direct2D initialization.
            }
            assert_eq!(GetCurrentObject(memory, OBJ_BITMAP), previous);
            assert!(DeleteObject(bitmap_guard.take()).as_bool());
        }
    }

    #[test]
    fn renderer_reconfigures_geometry_and_recovers_a_lost_target() {
        use windows::Win32::UI::WindowsAndMessaging::{CreateWindowExW, DestroyWindow, WS_EX_LAYERED, WS_POPUP};
        let hwnd = unsafe {
            CreateWindowExW(WS_EX_LAYERED, w!("STATIC"), w!("Renderer recovery test"),
                WS_POPUP, 0, 0, 200, 150, None, None, None, None).unwrap()
        };
        let mut renderer = OverlayRenderer::new(0, 0, 200, 150, "Segoe UI", 18.0, 0.8).unwrap();
        let old_bitmap = renderer.hbitmap;
        renderer.reconfigure(ScreenRect::new(-400, -200, 640, 480)).unwrap();
        assert_ne!(renderer.hbitmap, old_bitmap);
        assert_eq!((renderer.origin_x, renderer.origin_y, renderer.width, renderer.height), (-400, -200, 640, 480));
        assert_eq!((renderer.font_family.as_str(), renderer.font_size, renderer.overlay_bg_alpha), ("Segoe UI", 18.0, 0.8));
        let card = TranslationBlock {
            original_text: "word".to_string(), translated_text: "translation".to_string(),
            rect: ScreenRect::new(100, 220, 420, 20),
        };
        let scene = RenderScene {
            blocks: &[], cursor_card: Some(&card), save_button_enabled: true,
            active_roi: None, auto_mode: false, provider_name: "Google", ocr_engine_name: "WinRT",
            mode: AppMode::Normal, selection_rect: None, status_msg: None, is_loading: false,
        };
        let before_loss = renderer.hbitmap;
        renderer.next_render_error = Some(D2DERR_RECREATE_TARGET);
        let button = renderer.render(hwnd, &scene).unwrap().unwrap();
        assert_ne!(renderer.hbitmap, before_loss);
        assert!(button.x >= -400 && button.x + button.width <= 240);
        assert!(button.y >= -200 && button.y + button.height <= 280);
        renderer.next_render_error = Some(E_FAIL);
        assert_eq!(renderer.render(hwnd, &scene).unwrap_err().code(), E_FAIL);
        assert!(renderer.reconfigure(ScreenRect::new(0, 0, 0, 100)).is_err());
        assert_eq!((renderer.width, renderer.height), (640, 480));
        unsafe { DestroyWindow(hwnd).unwrap(); }
    }

    #[test]
    fn bug_3_raii_guard_defuse_and_take() {
        let dummy_raw: *mut core::ffi::c_void = 0xBAAD_F00D_usize as *mut _;
        let mut mem_guard = MemDcGuard(HDC(dummy_raw));

        // When taken, the handle must be transferred out and inner handle must become null
        let extracted = mem_guard.take();
        assert_eq!(extracted.0, dummy_raw);
        assert!(
            mem_guard.0.0.is_null(),
            "Guard must be defused after take()"
        );

        // Drop on defused guard should safely do nothing because handle is null
        drop(mem_guard);

        let mut bmp_guard = BitmapGuard(HGDIOBJ(dummy_raw));
        let bmp_extracted = bmp_guard.take();
        assert_eq!(bmp_extracted.0, dummy_raw);
        assert!(
            bmp_guard.0.0.is_null(),
            "BitmapGuard must be defused after take()"
        );
        drop(bmp_guard);
    }

    #[test]
    #[cfg(windows)]
    fn bug_3_smoke_overlay_renderer_lifecycle() {
        // Smoke test verifying that OverlayRenderer::new allocates DC and Direct2D resources,
        // and its Drop cleanly releases all GDI/D2D resources without crashing or leaking.
        let renderer_res = OverlayRenderer::new(0, 0, 200, 150, "Segoe UI", 15.5, 0.92);
        assert!(
            renderer_res.is_ok(),
            "OverlayRenderer::new should succeed in desktop environment"
        );
        let renderer = renderer_res.unwrap();
        assert!(!renderer.hdc_mem.0.is_null());
        assert!(!renderer.hbitmap.0.is_null());
        drop(renderer);
    }

    #[test]
    fn test_save_button_rendering_and_screen_bounds() {
        use windows::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, WS_EX_LAYERED, WS_POPUP,
        };
        let hwnd = unsafe {
            CreateWindowExW(WS_EX_LAYERED, w!("STATIC"), w!("Flashcard render test"),
                WS_POPUP, -800, -200, 800, 600, None, None, None, None).unwrap()
        };
        let mut renderer = OverlayRenderer::new(-800, -200, 800, 600, "Segoe UI", 16.0, 0.92).unwrap();
        let card = TranslationBlock {
            original_text: "word (context)".to_string(),
            translated_text: "Word translation\nContext sentence and translation".to_string(),
            rect: ScreenRect::new(-40, 380, 420, 30),
        };
        let mut scene = RenderScene {
            blocks: &[], cursor_card: Some(&card), save_button_enabled: true,
            active_roi: None, auto_mode: false, provider_name: "Google", ocr_engine_name: "WinRT",
            mode: AppMode::Normal, selection_rect: None, status_msg: None, is_loading: false,
        };
        let button = renderer.render(hwnd, &scene).unwrap().expect("Cursor card must have a save button");
        assert_eq!((button.width, button.height), (28, 28));
        assert!(button.x >= -800 && button.x + button.width <= 0);
        assert!(button.y >= -200 && button.y + button.height <= 400);
        scene.save_button_enabled = false;
        assert!(renderer.render(hwnd, &scene).unwrap().is_none());
        scene.save_button_enabled = true;
        scene.mode = AppMode::SelectingRoi;
        assert!(renderer.render(hwnd, &scene).unwrap().is_none());
        scene.mode = AppMode::Normal;
        scene.cursor_card = None;
        assert!(renderer.render(hwnd, &scene).unwrap().is_none());
        unsafe { DestroyWindow(hwnd).unwrap(); }
    }
}
