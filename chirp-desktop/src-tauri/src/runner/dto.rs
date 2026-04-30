use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ReadySignal {
    pub status: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct RunnerInfo {
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub runtime: Option<String>,
    pub model_path: String,
    pub codec_path: String,
    pub voices_path: Option<String>,
    pub espeak_data_path: Option<String>,
    pub voice: Option<String>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    pub voice_reference: Option<String>,
    pub voice: Option<String>,
    pub output_path: Option<String>,
    pub language: Option<String>,
}
