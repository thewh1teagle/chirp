use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::runtime::Language;

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub loaded: bool,
    pub model: String,
    pub runtime: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelsResponse {
    pub loaded: bool,
    pub runtime: String,
    pub model: String,
    pub path: String,
    pub codec: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LanguagesResponse {
    pub languages: Vec<String>,
    pub items: Vec<Language>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VoicesResponse {
    pub runtime: String,
    pub voices: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LoadResponse {
    pub status: String,
    pub model: String,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct LoadBody {
    #[serde(default)]
    pub runtime: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub codec_path: String,
    #[serde(default)]
    pub qwen: QwenLoadBody,
    #[serde(default)]
    pub kokoro: KokoroLoadBody,
    #[serde(default)]
    pub max_tokens: i32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: i32,
}

impl Default for LoadBody {
    fn default() -> Self {
        Self {
            runtime: String::new(),
            model_path: String::new(),
            codec_path: String::new(),
            qwen: QwenLoadBody::default(),
            kokoro: KokoroLoadBody::default(),
            max_tokens: 0,
            temperature: 0.0,
            top_k: 0,
        }
    }
}

#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct QwenLoadBody {
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub codec_path: String,
    #[serde(default)]
    pub max_tokens: i32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: i32,
}

#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct KokoroLoadBody {
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub voices_path: String,
    #[serde(default)]
    pub espeak_data_path: String,
    #[serde(default)]
    pub voice: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub speed: f32,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct SpeechBody {
    pub input: String,
    #[serde(default)]
    pub voice_reference: String,
    #[serde(default)]
    pub voice: String,
    #[serde(default)]
    pub response_format: String,
    #[serde(default)]
    pub language: String,
}
