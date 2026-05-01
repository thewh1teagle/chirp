use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use utoipa::ToSchema;

mod kokoro;
mod qwen;

pub use kokoro::{KokoroRuntime, kokoro_language};
pub use qwen::{QwenRuntime, qwen_language_id};

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Language {
    pub name: String,
    pub id: i32,
}

pub trait Runtime: Send {
    fn languages(&self) -> &[Language];
    fn voices(&self) -> Option<Vec<String>>;
    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice_or_reference: Option<&Path>,
        output_path: &Path,
        language: &str,
    ) -> Result<()>;
}

pub enum RuntimeParams {
    Qwen {
        model_path: PathBuf,
        codec_path: PathBuf,
        max_tokens: i32,
        temperature: f32,
        top_k: i32,
    },
    Kokoro {
        model_path: PathBuf,
        voices_path: PathBuf,
        voice: Option<String>,
        language: Option<String>,
        speed: f32,
    },
}

pub fn language_display_name(language: &str) -> String {
    match language.trim().to_lowercase().as_str() {
        "en-us" => "American English".into(),
        "en" | "en-gb" => "British English".into(),
        "es" => "Spanish".into(),
        "fr" | "fr-fr" => "French".into(),
        "ja" => "Japanese".into(),
        "hi" => "Hindi".into(),
        "it" => "Italian".into(),
        "pt-br" => "Brazilian Portuguese".into(),
        "pt" => "Portuguese".into(),
        other => title_case_language(other),
    }
}

pub fn language_code_alias(language: &str) -> String {
    match language.trim().to_lowercase().as_str() {
        "american english" | "american" => "en-us".into(),
        "british english" | "british" | "en-gb" => "en".into(),
        "spanish" => "es".into(),
        "french" => "fr".into(),
        "japanese" => "ja".into(),
        "hindi" => "hi".into(),
        "italian" => "it".into(),
        "brazilian portuguese" | "portuguese" => "pt-br".into(),
        other => other.into(),
    }
}

fn title_case_language(language: &str) -> String {
    language
        .split(['-', '_', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
