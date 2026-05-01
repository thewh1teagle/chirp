use crate::Result;

pub fn phonemize(text: &str, language: &str) -> Result<Vec<String>> {
    Ok(espeak_rs::text_to_phonemes(text, language, Some('^'))?)
}
