#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::c_void;
use windows::Win32::Foundation::{COLORREF, E_FAIL, HWND, POINT, RECT, SIZE};
use windows::Win32::Graphics::Direct2D::Common::{
    D2D_RECT_F, D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_PIXEL_FORMAT,
};
use windows::Win32::Graphics::Direct2D::{
    D2D1_DRAW_TEXT_OPTIONS_NONE, D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1_RENDER_TARGET_PROPERTIES,
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
    CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, HDC,
    HGDIOBJ, ReleaseDC, SelectObject,
};
use windows::Win32::UI::WindowsAndMessaging::{ULW_ALPHA, UpdateLayeredWindow};
use windows::core::{Result, w};

use crate::types::{AppMode, ScreenRect, TranslationBlock};

pub struct RenderScene<'a> {
    pub blocks: &'a [TranslationBlock],
    pub active_roi: Option<ScreenRect>,
    pub auto_mode: bool,
    pub provider_name: &'a str,
    pub mode: AppMode,
    pub selection_rect: Option<ScreenRect>,
    pub status_msg: Option<&'a str>,
    pub is_loading: bool,
}

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

pub struct OverlayRenderer {
    origin_x: i32,
    origin_y: i32,
    width: i32,
    height: i32,
    hdc_mem: HDC,
    hbitmap: HGDIOBJ,
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
    pub fn new(origin_x: i32, origin_y: i32, width: i32, height: i32) -> Result<Self> {
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

            SelectObject(hdc_mem_raw, bitmap_guard.0);
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
            let text_format_card = dwrite_factory.CreateTextFormat(
                w!("Segoe UI"),
                None,
                DWRITE_FONT_WEIGHT_SEMI_BOLD,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                15.5,
                w!("ru-ru"),
            )?;
            let _ = text_format_card.SetWordWrapping(DWRITE_WORD_WRAPPING_WRAP);
            let _ = text_format_card.SetTextAlignment(DWRITE_TEXT_ALIGNMENT_LEADING);
            let _ = text_format_card.SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_NEAR);

            let text_format_hud = dwrite_factory.CreateTextFormat(
                w!("Segoe UI"),
                None,
                DWRITE_FONT_WEIGHT_NORMAL,
                DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STRETCH_NORMAL,
                12.5,
                w!("ru-ru"),
            )?;

            // 5. Create Solid Brushes
            let brush_card_bg = dc_rt.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: 0.08,
                    g: 0.09,
                    b: 0.13,
                    a: 0.92,
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
                    a: 0.92,
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

            Ok(Self {
                origin_x,
                origin_y,
                width,
                height,
                hdc_mem,
                hbitmap,
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

    /// Renders the complete scene (translation cards, HUD bar, ROI selection) and flushes to DWM.
    pub fn render(&mut self, hwnd: HWND, scene: &RenderScene) -> Result<()> {
        unsafe {
            let bind_rect = RECT {
                left: 0,
                top: 0,
                right: self.width,
                bottom: self.height,
            };

            self.dc_rt.BindDC(self.hdc_mem, &bind_rect)?;
            self.dc_rt.BeginDraw();

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
                for block in scene.blocks {
                    let pad = 8.0;
                    let min_width = 140.0;
                    let max_layout_w = (self.width as f32 - 40.0).max(200.0);
                    let target_layout_w = (block.rect.width as f32).max(180.0).min(max_layout_w);
                    let max_layout_h = (self.height as f32 - 40.0).max(200.0);

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
                    let card_h = (metrics.height + pad * 2.0).max(30.0);

                    let block_screen_x = (block.rect.x - self.origin_x) as f32;
                    let block_screen_y = (block.rect.y - self.origin_y) as f32;

                    let left = (block_screen_x - pad)
                        .max(4.0)
                        .min((self.width as f32 - card_w - 4.0).max(4.0));
                    let top = (block_screen_y - pad)
                        .max(4.0)
                        .min((self.height as f32 - card_h - 4.0).max(4.0));
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
                        D2D1_DRAW_TEXT_OPTIONS_NONE,
                    );
                }
            }

            // 4. Draw HUD status badge
            self.draw_hud(scene)?;

            if let Err(e) = self.dc_rt.EndDraw(None, None) {
                tracing::error!("Direct2D EndDraw error: {:?}", e);
                return Err(e);
            }

            // 5. UpdateLayeredWindow to present frame
            let hdc_screen = GetDC(None);
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

            ReleaseDC(None, hdc_screen);

            if let Err(e) = ulw_res {
                tracing::error!("UpdateLayeredWindow error: {:?}", e);
                return Err(e);
            }
        }

        Ok(())
    }

    fn draw_hud(&self, scene: &RenderScene<'_>) -> Result<()> {
        unsafe {
            let hud_w = 920.0;
            let hud_h = 36.0;
            let hud_x = ((self.width as f32) - hud_w) / 2.0;
            let hud_y = 12.0;

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
                        " ScreenTranslator [{}] | Alt+T: Перевод | Alt+S: Выделение | Alt+A: {} | Alt+P: Движок | Alt+C: Скрыть | Alt+Q: Выход | {}{}",
                        scene.provider_name, auto_str, loading_prefix, status_str
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
            let _ = DeleteObject(self.hbitmap);
            let _ = DeleteDC(self.hdc_mem);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let renderer_res = OverlayRenderer::new(0, 0, 200, 150);
        assert!(
            renderer_res.is_ok(),
            "OverlayRenderer::new should succeed in desktop environment"
        );
        let renderer = renderer_res.unwrap();
        assert!(!renderer.hdc_mem.0.is_null());
        assert!(!renderer.hbitmap.0.is_null());
        drop(renderer);
    }
}
