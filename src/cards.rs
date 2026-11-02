use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::types::TranslationBlock;

pub fn save_card(card: &TranslationBlock) -> io::Result<PathBuf> {
    let path = std::env::current_exe()?.with_file_name("cards.txt");
    append_card(&path, card)?;
    Ok(path)
}

fn append_card(path: &Path, card: &TranslationBlock) -> io::Result<()> {
    if card.original_text.trim().is_empty() || card.translated_text.trim().is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Empty flashcard"));
    }
    let entry = format!("{}\n{}\n\n", card.original_text, card.translated_text);
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(entry.as_bytes())?;
    file.sync_data()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ScreenRect;

    #[test]
    fn test_cards_append_utf8_and_preserve_existing_content() {
        let path = std::env::temp_dir().join(format!("screen_translator_cards_{}.txt",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::write(&path, "existing\n").unwrap();
        let card = TranslationBlock {
            original_text: "word (This is a word.)".to_string(),
            translated_text: "Слово: word → слово\nВ контексте предложения: This is a word.".to_string(),
            rect: ScreenRect::new(0, 0, 1, 1),
        };
        append_card(&path, &card).unwrap();
        append_card(&path, &card).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.starts_with("existing\n"));
        assert_eq!(saved.matches("word → слово").count(), 2);
        assert_eq!(saved.matches("This is a word.").count(), 4);
        assert!(append_card(&path.join("missing.txt"), &card).is_err());
        let empty = TranslationBlock { original_text: String::new(), ..card };
        assert!(append_card(&path, &empty).is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), saved);
        std::fs::remove_file(path).unwrap();
    }
}
