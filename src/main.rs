pub mod app;
pub mod capture;
pub mod ocr;
pub mod overlay;
pub mod translator;
pub mod types;

use app::App;
use types::AppConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Enable Per-Monitor V2 DPI Awareness for exact 1:1 pixel coordinate mapping
    unsafe {
        use windows::Win32::UI::HiDpi::{
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
        };
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    }

    println!("============================================================");
    println!("     WINDOWS-NATIVE GAME & SCREEN TRANSLATOR (RUST)         ");
    println!("============================================================");
    println!("  Target OS:       Windows 10 / 11 (Exclusive Native)");
    println!("  Capture Engine:  Windows.Graphics.Capture API (GPU D3D11)");
    println!("  Overlay Engine:  Direct2D / Win32 (WDA_EXCLUDEFROMCAPTURE)");
    println!("  OCR Engine:      Windows.Media.Ocr (Zero-weight WinRT)");
    println!("  Translation:     Multi-provider MT + Lorebook + Vision");
    println!("------------------------------------------------------------");
    println!("  CONTROLS & SHORTCUTS:");
    println!("    [Alt + T]  Translate Screen / Active Dialogue Box");
    println!("    [Alt + S]  Interactive Selection (Drag box on screen)");
    println!("    [Alt + A]  Toggle Auto-Translate Mode (Dialogue Watcher)");
    println!("    [Alt + P]  Cycle Engine (Google -> DeepL -> Gemini -> Groq)");
    println!("    [Alt + O]  Cycle OCR Engine (WinRT -> RapidOCR -> Auto)");
    println!("    [Alt + C]  Clear Translated Text Cards");
    println!("    [Alt + Q]  Quit Application");
    println!("============================================================");

    let config = AppConfig::load_or_default();
    println!("  Active Engine:   {}", config.provider.display_name());
    println!("  OCR Mode:        {}", config.ocr_engine.display_name());
    println!(
        "  Languages:       {} -> {}",
        config.source_lang, config.target_lang
    );
    println!("============================================================");

    let app = match App::new(config) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to initialize Screen Translator: {}", e);
            std::process::exit(1);
        }
    };

    app.run().await;

    println!("Screen Translator shut down cleanly.");
    Ok(())
}
