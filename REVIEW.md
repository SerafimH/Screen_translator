# Code Review: Screen Translator

> **Дата ревью:** 2026-10-29  
> **Результат тестов:** ✅ 64/64 проходят (`cargo test --no-fail-fast`)  
> **Честная оговорка:** если что-то понял не до конца — написано явно.

---

## Общая архитектура

```
main.rs (AppConfig::load_or_default)
  └─ App::new
       ├─ OverlayManager::spawn()   → Win32 message pump (отдельный thread)
       ├─ WinrtOcrEngine::new()     → WinRT OCR (5-pass adaptive)
       ├─ OcrHelperClient::new()    → Python RapidOCR subprocess
       └─ TranslatorService::new()  → reqwest HTTP client + cache + context

App::run (tokio event loop)
  ├─ background: auto-mode watcher (tokio::spawn interval)
  └─ foreground: AppEvent receiver
       └─ dispatch_translation_task → spawn_cancellable_task
            ├─ ScreenCapture::capture_primary (WinRT GPU → GDI fallback)
            ├─ perform_ocr (WinRT 5-pass / Helper / Auto)
            └─ TranslatorService::translate_blocks
                 └─ OverlayHandle::update_blocks → Direct2D render
```

Разделение чёткое: overlay живёт в своём Win32-потоке, вся бизнес-логика — в tokio runtime.

---

## Сильные стороны

### 1. RAII везде, где нужно

| Guard | Файл | Защищает |
|---|---|---|
| `ProcessingGuard` | `app.rs` | AtomicBool флаг — сброс в Drop |
| `ScreenDcGuard` | `capture/mod.rs` | `ReleaseDC` |
| `MemDcGuard` | `capture/mod.rs` | `DeleteDC` |
| `GdiObjectGuard` | `capture/mod.rs` | `SelectObject` + `DeleteObject` |
| `JobObjectGuard` | `ocr/helper.rs` | `CloseHandle` для Win32 Job Object |

Благодаря этому нет ручных цепочек `if err { DeleteDC(); return Err(...) }`.

### 2. `spawn_cancellable_task` — нетривиально и правильно

```rust
// app.rs:280-304
pub fn spawn_cancellable_task<Fut>(task_slot: &Arc<Mutex<Option<JoinHandle<()>>>>, fut: Fut)
```

Логика: берёт предыдущий handle → abort() → await его завершения → запускает новую задачу.  
Важная деталь: await старого handle гарантирует, что `ProcessingGuard` предыдущей задачи дропнется **до** того как новая попытается его захватить. Без этого была бы гонка.  
Дополнительно защита от "кто-то параллельно вставил более новый handle" — abort себя если slot уже занят.

Тест `test_high_02_cancellable_task_waits_for_old_task_to_release_guard` отлично это покрывает, включая симуляцию 30ms teardown-задержки через `SlowGuard`.

### 3. Безопасность API ключей

- `redact_secrets()` в `translator/mod.rs` — фильтрует ключи из URL-параметров (`key=`), Bearer-токенов, DeepL-Auth-Key, префиксов `sk-`/`gsk_`/`AIzaSy` из ошибок и логов.
- `impl Debug for AppConfig` — маскирует все три поля ключей через `.map(|_| "[REDACTED]")`.
- Python-процесс привязан к Win32 Job Object с `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` — при крэше основного процесса Python-зомби не остаётся.

### 4. OCR Pipeline — продуманная стратегия

5 проходов с дедлайном 800ms и системой очков (`score_recognized_lines`):

```
Pass 1: manga preprocessing (bubble clean + contrast + 2x Catmull-Rom + white padding)
Pass 2: outlined text mask (мем-шрифты с тёмным контуром — Impact, Arial Black)
Pass 3: inverted buffer (светлый текст на тёмном фоне)
Pass 4: raw frame (game UI, чистый текст без anime-фона)
Pass 5: upscale + contrast-enhance (мелкий текст)
```

Если `score >= 12` после Pass 1 — остальные не запускаются. Это разумный trade-off точность/скорость.

`cluster_lines_into_blocks` с hyphen-merging (`adven-` + `ture` → `adventure`) и `has_border_between` для детекции границ манга-панелей — хорошее решение.

### 5. Тест-культура

64 теста. Особенно ценны:

- `bug_5_frame_signature_does_not_skip_last_pixel` — пиксель `(width-1, height-1)` должен попадать в подпись.
- `test_high_02_rapid_dispatches_do_not_accumulate_or_deadlock` — 20 задач подряд, должна выполниться только последняя (или несколько первых), а не все 20 в очереди.
- `test_toml_vision_fallback_combinations_and_order_independence` — 8 комбинаций `vision_fallback`/`use_vision_ocr`, включая serde JSON-десериализацию.
- `bug_6_regex_simple_replace_utf8_safety` — UTF-8 байтовые смещения при case-insensitive замене (`ẞ` → `ss`, турецкий `İ`, emoji).

---

## Проблемы

### 🔴 HIGH-1: Ручной TOML-парсер — конкретный баг с `#` в значении

**Файл:** `src/types.rs`, функция `from_toml_str` (~строки 460–563)

Парсер строит логику на `split_once('#')` для удаления inline-комментариев:

```rust
let raw_val = match val_part.split_once('#') {
    Some((before_comment, _)) => before_comment.trim(),
    None => val_part.trim(),
};
```

**Баг:** если `#` встречается внутри строкового значения — значение обрезается:

```toml
# Это сломается:
python_path = "C:/path/to #venv/python.exe"
# Парсер прочитает: "C:/path/to " — путь обрезан, файл не найдётся

# И это:
openai_base_url = "https://api.example.com/v1#beta"
# Прочитает: "https://api.example.com/v1" — URL сломан
```

Настоящий TOML-парсер понимает, что `#` внутри кавычек — не комментарий.

**Также не поддерживается:**
- Многострочные строки (`"""..."""`)
- Массивы (`key = ["a", "b"]`)
- Вложенные таблицы (`[section]`)

**Решение:** добавить `toml = "0.8"` в `Cargo.toml` и заменить ~100 строк парсера:

```toml
# Cargo.toml
[dependencies]
toml = "0.8"
```

```rust
// types.rs — заменить всю from_toml_str:
pub fn from_toml_str(content: &str) -> Self {
    toml::from_str::<AppConfigRaw>(content)
        .map(Into::into)
        .unwrap_or_else(|e| {
            warn!("Failed to parse config.toml: {}. Using defaults.", e);
            Self::default()
        })
}
```

`AppConfigRaw` уже реализует `serde::Deserialize` — это работает без изменений структуры.

---

### 🔴 HIGH-2: Утечка потоков в `capture_winrt` при таймауте

**Файл:** `src/capture/mod.rs`, функция `capture_winrt`, строки 296–306

```rust
let capture_thread = std::thread::spawn(move || {
    let _ = WinrtSingleFrameCapture::start(settings);
});

match rx.recv_timeout(Duration::from_millis(600)) {
    Ok(frame) => {
        let _ = capture_thread.join(); // ← join только при успехе
        Ok(frame)
    }
    Err(_) => Err("WinRT capture timed out or unavailable, using GDI".to_string()),
    // ↑ При таймауте: capture_thread НЕ join()-ится, продолжает жить в фоне.
}
```

`std::thread::JoinHandle` при drop просто detach-ится — поток продолжает выполняться. Если WinRT-захват зависает на 5+ секунд, при авто-режиме (интервал 1200ms) за несколько минут накопится N живых потоков.

**Исправление (минимальное):** сохранить handle и дать ему "догнать" после GDI fallback:

```rust
Err(_) => {
    // Не ждём — WinRT завис, но и не теряем handle
    // Поток завершится сам когда WinRT API вернёт управление.
    // Добавить комментарий что это намеренно и ожидаемо редко.
    drop(capture_thread); // явный detach с комментарием
    Err("WinRT capture timed out or unavailable, using GDI".to_string())
}
```

Или завести счётчик/статистику сколько раз WinRT падает по таймауту.

---

### 🟡 MED-1: `#[allow(unsafe_op_in_unsafe_fn)]` на весь файл

**Файл:** `src/overlay/mod.rs`, строка 1

```rust
#![allow(unsafe_op_in_unsafe_fn)]
```

Rust Edition 2024 требует явных `unsafe { }` блоков внутри `unsafe fn`. Этот атрибут отключает проверку **на весь файл** (~680 строк). Рекомендуется убрать атрибут и добавить явные `unsafe { }` блоки там, где они нужны — это локализует опасный код и упрощает аудит.

---

### 🟡 MED-2: `ContextManager::preprocess` — заглушка без реализации

**Файл:** `src/translator/mod.rs`, строки 65–67

```rust
pub fn preprocess(&self, text: &str) -> String {
    text.to_string() // ← no-op
}
```

Метод вызывается в `translate_text`. Выглядит как незаконченная фича. Варианты:
1. Удалить и вызывать текст напрямую.
2. Реализовать (например, применять glossary до перевода — сейчас он применяется только после).
3. Оставить с `// TODO: apply pre-translation glossary substitution`.

---

### 🟡 MED-3: Google Translate — неофициальный эндпоинт

**Файл:** `src/translator/mod.rs`, строки 696–701

```rust
let url = format!(
    "https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&dj=1&q={}",
    ...
);
```

`client=gtx` — это не задокументированный API. Google его не гарантирует. На практике работает годами, но может сломаться:
- IP rate-limiting без предупреждения
- Captcha-блокировки
- Изменение формата ответа

Это нормально для home-use утилиты, но пользователи должны знать об этом риске. Стоит добавить предупреждение в `config.toml` или README.

---

### 🟡 MED-4: Дублирование Luma-формулы в 5 местах

Формула `(0.299 * r + 0.587 * g + 0.114 * b).round() as u8` повторяется:

| Место | Файл |
|---|---|
| `clean_manga_bubble_background` | `ocr/mod.rs:461` |
| `has_border_between` | `ocr/mod.rs:674` |
| `sharpen_manga_contrast` | `ocr/mod.rs:535` |
| `FrameSignature::from_bgra` | `capture/mod.rs:78` |
| `extract_outlined_text_mask` | `ocr/preprocessing.rs:41` |

**Решение:** вынести в одно место:

```rust
// ocr/mod.rs или отдельный image_utils.rs
#[inline(always)]
pub fn luma_from_bgr(b: u8, g: u8, r: u8) -> u8 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8
}
```

---

### 🟢 LOW-1: `GEMINI_MODEL = "gemini-3.8-flash"` — модель не существует

**Файл:** `src/translator/mod.rs:8`

```rust
pub const GEMINI_MODEL: &str = "gemini-3.8-flash";
```

Также продублировано в `config.toml`:
```toml
# Рекомендуется: gemini-3.8-flash с прямой поддержкой Vision
```

> ⚠️ **Модель `gemini-3.8-flash` не существует в Gemini API.** Запросы вернут ошибку `404 MODEL_NOT_FOUND`.

Актуальные имена (проверить на https://ai.google.dev/gemini-api/docs/models):
- `gemini-2.5-flash` (рекомендуется)
- `gemini-2.0-flash`
- `gemini-1.5-flash`

Вероятно, это опечатка `2.8` → `3.8` или `1.5` → `3.8`.

---

### 🟢 LOW-2: `from_str_loose` — молчаливый fallback при опечатке

**Файл:** `src/types.rs`, строки 167–178

```rust
pub fn from_str_loose(s: &str) -> Self {
    // ...
    } else {
        Self::Google // ← при любой неизвестной строке
    }
}
```

Если в конфиге `provider = "goggle"` (опечатка) — молча переключается на Google без предупреждения. Аналогично для `OcrMode::from_str_loose`.

**Исправление:**
```rust
} else {
    warn!("Unknown provider value '{}', falling back to Google Translate", s);
    Self::Google
}
```

---

### 🟢 LOW-3: Дублирование логики clear в `overlay_wndproc`

**Файл:** `src/overlay/mod.rs`, строки 474–483

```rust
HOTKEY_CLEAR => {
    let _ = state.event_tx.send(AppEvent::Clear); // → App::run вызовет handle.clear()
    {
        // И тут же дублируем то, что делает handle.clear():
        let mut s = state.shared_state.lock()...;
        s.current_blocks.clear();
        s.status_msg = Some("Очищено".to_string());
        s.is_loading = false;
    }
    render_current_scene(state);
}
```

Это намеренно (немедленный отклик UI до обработки события в App), но без комментария выглядит как баг или рассинхрон. Добавить:

```rust
// Немедленно очищаем UI для отклика без ожидания round-trip через event loop.
// App::run также вызовет handle.clear() при обработке AppEvent::Clear — это идемпотентно.
```

---

### 🟢 LOW-4: `match_find_at` — неочевидная семантика, нет документации

**Файл:** `src/translator/mod.rs`, строки 102–117

Функция написана корректно (UTF-8 safe, case-insensitive через `ch.to_lowercase()`), но семантика неочевидна. Проверяет: "начинается ли `source[start_idx..]` с lowercase(find)?" — и возвращает длину совпавшей подстроки в байтах исходной строки. Это работает для `regex_simple_replace` который перебирает каждую char-позицию.

Стоит добавить doc-comment с примером:

```rust
/// Returns the byte length of the match in `source[start_idx..]` if `source[start_idx..]`
/// starts with the case-folded version of `find_lower`, or `None` otherwise.
/// Note: `find_lower` must already be lowercased.
```

---

## Что не удалось полностью разобрать

### ❓ `unsafe impl Send/Sync for WinrtOcrEngine`

```rust
// ocr/mod.rs:36-37
// OcrEngine is safe to send between threads in WinRT
unsafe impl Send for WinrtOcrEngine {}
unsafe impl Sync for WinrtOcrEngine {}
```

COM-объекты в Windows могут быть STA (Single Threaded Apartment) или MTA. Если `OcrEngine` — STA-объект, вызовы с другого потока без маршалинга — это UB. `Sync` означает "можно вызывать `&self` методы из нескольких потоков одновременно".

Я не нашёл в документации Microsoft явного указания на threading model для `Windows.Media.Ocr.OcrEngine`. Судя по тому что код работает (64 теста проходят, приложение функционирует) — либо WinRT автоматически маршалирует, либо OcrEngine действительно thread-safe. **Но стоит добавить ссылку на источник.**

### ❓ `WinrtSingleFrameCapture` — отсутствие `on_closed`

`GraphicsCaptureApiHandler` трейт из `windows-capture`. В `WinrtSingleFrameCapture` реализованы только `new` и `on_frame_arrived`. Если трейт требует `on_closed` — возможно там есть дефолтная реализация (no-op). Не проверял исходники `windows-capture`.

### ❓ `overlay/renderer.rs` — Direct2D / LayeredWindow

Файл не разбирал детально. Это нетривиальный Low-level код с D2D1, DXGI, UpdateLayeredWindowIndirect. Требует отдельного погружения в WinAPI документацию.

---

## Итоговая таблица

| # | Серьёзность | Описание | Файл |
|---|---|---|---|
| 1 | 🔴 HIGH | Ручной TOML-парсер: баг с `#` в значении | `types.rs:460` |
| 2 | 🔴 HIGH | Утечка потоков при WinRT таймауте | `capture/mod.rs:296` |
| 3 | 🟡 MED | `#[allow(unsafe_op_in_unsafe_fn)]` на весь файл | `overlay/mod.rs:1` |
| 4 | 🟡 MED | `preprocess()` — заглушка без реализации | `translator/mod.rs:65` |
| 5 | 🟡 MED | Google Translate неофициальный API | `translator/mod.rs:696` |
| 6 | 🟡 MED | Luma-формула продублирована 5 раз | несколько файлов |
| 7 | 🟢 LOW | `GEMINI_MODEL = "gemini-3.8-flash"` — не существует | `translator/mod.rs:8` |
| 8 | 🟢 LOW | Молчаливый fallback при опечатке в `provider` | `types.rs:175` |
| 9 | 🟢 LOW | Дублирование логики clear без комментария | `overlay/mod.rs:474` |
| 10 | 🟢 LOW | `match_find_at` — нет документации | `translator/mod.rs:102` |
| ❓ | Неизвестно | `unsafe impl Sync for WinrtOcrEngine` — обоснование? | `ocr/mod.rs:37` |
