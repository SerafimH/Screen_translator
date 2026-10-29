use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::{error, info, warn};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};

use crate::capture::CapturedFrame;
use crate::ocr::cluster_lines_into_blocks;
use crate::types::{AppConfig, OcrBlock, RecognizedLine, RecognizedWord, ScreenRect};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// RAII Guard for Windows Job Object.
/// Ensures all assigned child processes (Python helper) are killed when this object is closed
/// or when the Screen_translator process terminates unexpectedly.
pub struct JobObjectGuard(pub HANDLE);

unsafe impl Send for JobObjectGuard {}
unsafe impl Sync for JobObjectGuard {}

impl Drop for JobObjectGuard {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

pub fn create_kill_on_close_job() -> Result<JobObjectGuard, String> {
    unsafe {
        let job = CreateJobObjectW(None, None)
            .map_err(|e| format!("CreateJobObjectW failed: {:?}", e))?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .map_err(|e| format!("SetInformationJobObject failed: {:?}", e))?;
        Ok(JobObjectGuard(job))
    }
}

pub fn assign_process_to_job(job: &JobObjectGuard, process: HANDLE) -> Result<(), String> {
    unsafe {
        AssignProcessToJobObject(job.0, process)
            .map_err(|e| format!("AssignProcessToJobObject failed: {:?}", e))
    }
}

/// Strips ANSI terminal escape sequences (colors, cursor codes) from string.
pub fn strip_ansi_codes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' || c == 'K' || c == 'H' || c == 'J' {
                in_escape = false;
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Resolves a relative or absolute path, checking current working directory,
/// executable directory, and parent project directories.
pub fn resolve_helper_path(path_str: &str) -> Option<std::path::PathBuf> {
    let p = Path::new(path_str);
    if p.is_absolute() && p.exists() {
        return Some(p.to_path_buf());
    }

    // 1. Check relative to current working directory
    if p.exists() {
        return Some(p.to_path_buf());
    }

    // 2. Check relative to executable directory and its ancestors
    if let Ok(current_exe) = std::env::current_exe()
        && let Some(exe_dir) = current_exe.parent()
    {
        let candidate = exe_dir.join(p);
        if candidate.exists() {
            return Some(candidate);
        }
        if let Some(parent1) = exe_dir.parent() {
            let candidate1 = parent1.join(p);
            if candidate1.exists() {
                return Some(candidate1);
            }
            if let Some(parent2) = parent1.parent() {
                let candidate2 = parent2.join(p);
                if candidate2.exists() {
                    return Some(candidate2);
                }
            }
        }
    }

    // 3. Check common relative fallbacks
    for prefix in &["../", "../../", "../../../"] {
        let candidate = Path::new(prefix).join(p);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

/// Converts BGRA8 frame pixels to raw RGB bytes (R, G, B order).
pub fn bgra_to_rgb(bgra: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in bgra.as_chunks::<4>().0.iter().take(pixel_count) {
        rgb.push(chunk[2]); // R
        rgb.push(chunk[1]); // G
        rgb.push(chunk[0]); // B
    }
    rgb
}

#[derive(Debug, serde::Serialize)]
struct HelperOcrRequest<'a> {
    id: u64,
    width: u32,
    height: u32,
    lang: &'a str,
}

#[derive(Debug, serde::Deserialize)]
struct HelperLineJson {
    text: String,
    #[serde(rename = "box")]
    bbox: [[f32; 2]; 4],
    score: f32,
}

#[derive(Debug, serde::Deserialize)]
struct HelperResponseJson {
    id: Option<u64>,
    ok: bool,
    lines: Option<Vec<HelperLineJson>>,
    ms: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ReadyEventJson {
    event: Option<String>,
}

struct ProcessSession {
    _child: Child,
    stdin: ChildStdin,
    stdout_lines: Lines<BufReader<ChildStdout>>,
    _job: JobObjectGuard,
}

pub struct OcrHelperClient {
    session: tokio::sync::Mutex<Option<ProcessSession>>,
    config: Arc<Mutex<AppConfig>>,
    is_ready: Arc<AtomicBool>,
    is_available: Arc<AtomicBool>,
    has_warned: Arc<AtomicBool>,
    restart_attempted: Arc<AtomicBool>,
    next_id: AtomicU64,
}

impl OcrHelperClient {
    /// Spawns the python helper process in the background, waits for ready event asynchronously
    /// without blocking the caller or UI thread, and binds the process to a Windows Job Object.
    pub fn new(config: Arc<Mutex<AppConfig>>) -> Arc<Self> {
        let client = Arc::new(Self {
            session: tokio::sync::Mutex::new(None),
            config,
            is_ready: Arc::new(AtomicBool::new(false)),
            is_available: Arc::new(AtomicBool::new(false)),
            has_warned: Arc::new(AtomicBool::new(false)),
            restart_attempted: Arc::new(AtomicBool::new(false)),
            next_id: AtomicU64::new(1),
        });

        let client_clone = client.clone();
        tokio::spawn(async move {
            client_clone.init_session().await;
        });

        client
    }

    /// Checks if the helper is ready and available for OCR requests.
    pub fn is_ready(&self) -> bool {
        self.is_ready.load(Ordering::Relaxed) && self.is_available.load(Ordering::Relaxed)
    }

    async fn init_session(&self) {
        let (raw_python_path, raw_server_path) = {
            let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
            let py = cfg.python_path.clone();
            let srv = "ocr_helper/server.py".to_string();
            (py, srv)
        };

        let python_path = match resolve_helper_path(&raw_python_path) {
            Some(p) => p,
            None => {
                if !self.has_warned.swap(true, Ordering::Relaxed) {
                    warn!(
                        "Python interpreter for OCR helper not found at '{}'. Run ocr_helper/setup.ps1 first. Falling back to WinRT.",
                        raw_python_path
                    );
                }
                self.is_available.store(false, Ordering::SeqCst);
                self.is_ready.store(false, Ordering::SeqCst);
                return;
            }
        };

        let server_path = match resolve_helper_path(&raw_server_path) {
            Some(p) => p,
            None => {
                if !self.has_warned.swap(true, Ordering::Relaxed) {
                    warn!(
                        "OCR helper script not found at '{}'. Falling back to WinRT.",
                        raw_server_path
                    );
                }
                self.is_available.store(false, Ordering::SeqCst);
                self.is_ready.store(false, Ordering::SeqCst);
                return;
            }
        };

        info!(
            "Spawning OCR helper process: '{}' '{}'...",
            python_path.display(),
            server_path.display()
        );

        let job = match create_kill_on_close_job() {
            Ok(j) => j,
            Err(e) => {
                error!("Failed to create Job Object for OCR helper: {}", e);
                self.is_available.store(false, Ordering::SeqCst);
                return;
            }
        };

        let mut cmd = Command::new(&python_path);
        cmd.arg(&server_path);
        if let Some(parent) = server_path.parent()
            && let Some(root) = parent.parent()
        {
            cmd.current_dir(root);
        }
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.stdin(std::process::Stdio::piped());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                if !self.has_warned.swap(true, Ordering::Relaxed) {
                    warn!("Failed to spawn OCR helper process: {}", e);
                }
                self.is_available.store(false, Ordering::SeqCst);
                return;
            }
        };

        // Bind process to Windows Job Object
        if let Some(raw_handle) = child.raw_handle() {
            let handle = HANDLE(raw_handle as *mut _);
            if let Err(e) = assign_process_to_job(&job, handle) {
                error!("Failed to assign OCR helper to Job Object: {}", e);
            } else {
                info!("OCR helper successfully bound to KILL_ON_JOB_CLOSE Job Object.");
            }
        }

        let stdin = match child.stdin.take() {
            Some(s) => s,
            None => {
                error!("Failed to open stdin for OCR helper");
                self.is_available.store(false, Ordering::SeqCst);
                return;
            }
        };

        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                error!("Failed to open stdout for OCR helper");
                self.is_available.store(false, Ordering::SeqCst);
                return;
            }
        };

        if let Some(stderr) = child.stderr.take() {
            let mut stderr_reader = BufReader::new(stderr).lines();
            tokio::spawn(async move {
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    let clean = strip_ansi_codes(&line);
                    if clean.contains("[ERROR]") {
                        tracing::error!("[ocr_helper] {}", clean);
                    } else if clean.contains("[WARNING]") || clean.contains("[WARN]") {
                        tracing::warn!("[ocr_helper] {}", clean);
                    } else {
                        tracing::info!("[ocr_helper] {}", clean);
                    }
                }
            });
        }

        let mut stdout_lines = BufReader::new(stdout).lines();

        // Wait for {"event": "ready"} with 30s timeout without blocking UI
        info!("Waiting for OCR helper ready event (up to 30s)...");
        let ready_res =
            tokio::time::timeout(Duration::from_secs(30), stdout_lines.next_line()).await;

        match ready_res {
            Ok(Ok(Some(line))) => {
                if let Ok(evt) = serde_json::from_str::<ReadyEventJson>(&line)
                    && evt.event.as_deref() == Some("ready")
                {
                    info!("OCR helper reported ready event: {}", line);
                    self.is_ready.store(true, Ordering::SeqCst);
                    self.is_available.store(true, Ordering::SeqCst);
                    self.has_warned.store(false, Ordering::SeqCst);
                } else {
                    warn!("Unexpected first line from OCR helper: {}", line);
                    self.is_available.store(false, Ordering::SeqCst);
                }
            }
            Ok(Ok(None)) => {
                warn!("OCR helper process stdout closed before ready event.");
                self.is_available.store(false, Ordering::SeqCst);
            }
            Ok(Err(e)) => {
                warn!("IO error waiting for OCR helper ready event: {}", e);
                self.is_available.store(false, Ordering::SeqCst);
            }
            Err(_) => {
                warn!("Timed out waiting 30s for OCR helper ready event.");
                self.is_available.store(false, Ordering::SeqCst);
            }
        }

        if self.is_available.load(Ordering::SeqCst) {
            let mut sess_lock = self.session.lock().await;
            *sess_lock = Some(ProcessSession {
                _child: child,
                stdin,
                stdout_lines,
                _job: job,
            });
        }
    }

    /// Recognizes text in raw frame using the Python helper process.
    /// If the process is dead, times out, or fails: marks helper unavailable, attempts one restart,
    /// and returns an Err so the caller can fall back cleanly to WinRT.
    pub async fn recognize(
        &self,
        frame: &CapturedFrame,
        lang: &str,
    ) -> Result<Vec<OcrBlock>, String> {
        let lines = self.recognize_raw_lines(frame, lang).await?;
        let blocks =
            cluster_lines_into_blocks(lines, Some((&frame.bgra_bytes, frame.width, frame.height)));
        Ok(blocks)
    }

    pub async fn recognize_raw_lines(
        &self,
        frame: &CapturedFrame,
        lang: &str,
    ) -> Result<Vec<RecognizedLine>, String> {
        if frame.width == 0 || frame.height == 0 || frame.bgra_bytes.is_empty() {
            return Ok(Vec::new());
        }

        if !self.is_ready() {
            // If unavailable and restart has not been attempted yet, attempt one restart
            if !self.restart_attempted.swap(true, Ordering::SeqCst) {
                info!("Attempting one restart of OCR helper process...");
                self.init_session().await;
            }

            if !self.is_ready() {
                if !self.has_warned.swap(true, Ordering::Relaxed) {
                    warn!("OCR helper is currently unavailable. Falling back to WinRT.");
                }
                return Err("OCR helper is unavailable".to_string());
            }
        }

        let (timeout_ms, min_score) = {
            let cfg = self.config.lock().unwrap_or_else(|e| e.into_inner());
            (cfg.helper_timeout_ms.max(500), cfg.helper_min_score)
        };

        let req_id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let req_header = HelperOcrRequest {
            id: req_id,
            width: frame.width,
            height: frame.height,
            lang,
        };

        let mut header_bytes = serde_json::to_vec(&req_header)
            .map_err(|e| format!("Failed to serialize OCR request: {}", e))?;
        header_bytes.push(b'\n');

        let raw_rgb = bgra_to_rgb(&frame.bgra_bytes, frame.width, frame.height);

        let mut session_guard = self.session.lock().await;
        let session = match session_guard.as_mut() {
            Some(s) => s,
            None => {
                self.is_available.store(false, Ordering::SeqCst);
                return Err("OCR helper session is None".to_string());
            }
        };

        // Write request header + raw RGB bytes
        let write_res = async {
            session.stdin.write_all(&header_bytes).await?;
            session.stdin.write_all(&raw_rgb).await?;
            session.stdin.flush().await?;
            tokio::io::Result::Ok(())
        };

        if let Err(e) = tokio::time::timeout(Duration::from_millis(timeout_ms), write_res).await {
            self.handle_failure("write timeout or error", session_guard);
            return Err(format!("Failed to write to OCR helper: {:?}", e));
        }

        // Read single line JSON response
        let read_res = tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            session.stdout_lines.next_line(),
        )
        .await;

        let response_line = match read_res {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                self.handle_failure("stdout closed unexpectedly", session_guard);
                return Err("OCR helper closed stdout unexpectedly".to_string());
            }
            Ok(Err(e)) => {
                self.handle_failure("IO error reading stdout", session_guard);
                return Err(format!("IO error reading from OCR helper: {}", e));
            }
            Err(_) => {
                self.handle_failure("request timed out", session_guard);
                return Err(format!(
                    "OCR helper request #{} timed out after {} ms",
                    req_id, timeout_ms
                ));
            }
        };

        let resp: HelperResponseJson = serde_json::from_str(&response_line)
            .map_err(|e| format!("Failed to parse OCR helper response: {}", e))?;

        if let Some(resp_id) = resp.id
            && resp_id != req_id
        {
            return Err(format!(
                "OCR helper response ID mismatch: expected {}, got {}",
                req_id, resp_id
            ));
        }

        if !resp.ok {
            let err_msg = resp
                .error
                .unwrap_or_else(|| "Unknown helper error".to_string());
            return Err(format!("OCR helper reported error: {}", err_msg));
        }

        let raw_lines = resp.lines.unwrap_or_default();
        if let Some(ms) = resp.ms {
            info!(
                "OCR helper recognized {} lines in {:.1} ms",
                raw_lines.len(),
                ms
            );
        }

        // Convert helper lines into RecognizedLine in desktop screen space
        let mut recognized = Vec::with_capacity(raw_lines.len());
        for line in raw_lines {
            if line.score < min_score {
                continue;
            }

            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            for pt in &line.bbox {
                min_x = min_x.min(pt[0]);
                max_x = max_x.max(pt[0]);
                min_y = min_y.min(pt[1]);
                max_y = max_y.max(pt[1]);
            }

            let local_x = min_x.round() as i32;
            let local_y = min_y.round() as i32;
            let w = (max_x - min_x).round().max(1.0) as i32;
            let h = (max_y - min_y).round().max(1.0) as i32;

            let rect = ScreenRect::new(local_x + frame.offset_x, local_y + frame.offset_y, w, h);

            let trimmed = line.text.trim();
            if trimmed.is_empty() {
                continue;
            }

            let words = trimmed
                .split_whitespace()
                .map(|w| RecognizedWord {
                    text: w.to_string(),
                    rect,
                })
                .collect();

            recognized.push(RecognizedLine {
                text: trimmed.to_string(),
                rect,
                words,
            });
        }

        Ok(recognized)
    }

    fn handle_failure(
        &self,
        reason: &str,
        mut session_guard: tokio::sync::MutexGuard<'_, Option<ProcessSession>>,
    ) {
        if !self.has_warned.swap(true, Ordering::Relaxed) {
            warn!(
                "OCR helper process failure ({}). Marking unavailable.",
                reason
            );
        }
        self.is_available.store(false, Ordering::SeqCst);
        self.is_ready.store(false, Ordering::SeqCst);
        *session_guard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bgra_to_rgb_conversion() {
        // Pixel 0: B=10, G=20, R=30, A=255 -> RGB should be [30, 20, 10]
        // Pixel 1: B=50, G=60, R=70, A=255 -> RGB should be [70, 60, 50]
        let bgra = vec![10, 20, 30, 255, 50, 60, 70, 255];
        let rgb = bgra_to_rgb(&bgra, 2, 1);
        assert_eq!(rgb, vec![30, 20, 10, 70, 60, 50]);
    }

    #[test]
    fn test_helper_line_bbox_to_screen_rect() {
        let bbox = [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]];
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for pt in &bbox {
            min_x = min_x.min(pt[0]);
            max_x = max_x.max(pt[0]);
            min_y = min_y.min(pt[1]);
            max_y = max_y.max(pt[1]);
        }
        let offset_x = 500;
        let offset_y = 300;
        let rect = ScreenRect::new(
            min_x.round() as i32 + offset_x,
            min_y.round() as i32 + offset_y,
            (max_x - min_x).round() as i32,
            (max_y - min_y).round() as i32,
        );
        assert_eq!(rect.x, 510);
        assert_eq!(rect.y, 320);
        assert_eq!(rect.width, 90);
        assert_eq!(rect.height, 30);
    }

    #[test]
    fn test_kill_on_close_job_object_creation() {
        let job = create_kill_on_close_job();
        assert!(job.is_ok(), "Job Object creation must succeed on Windows");
        let guard = job.unwrap();
        assert!(!guard.0.is_invalid());
    }

    #[test]
    fn test_resolve_helper_path_finds_existing_files() {
        let py = resolve_helper_path("ocr_helper/server.py");
        assert!(py.is_some(), "Must resolve ocr_helper/server.py");
    }

    #[test]
    fn test_strip_ansi_codes() {
        let raw = "\x1b[32m[INFO] Hello World\x1b[0m";
        let clean = strip_ansi_codes(raw);
        assert_eq!(clean, "[INFO] Hello World");
    }
}
