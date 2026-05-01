use serde::Serialize;
use utoipa::ToSchema;

const QWEN_MODELS_TAG: &str = "chirp-models-v0.1.3";
const QWEN_MODEL_FILE: &str = "qwen3-tts-model.gguf";
const QWEN_CODEC_FILE: &str = "qwen3-tts-codec.gguf";
const QWEN_MODEL_BASE_URL: &str = "https://huggingface.co/thewh1teagle/qwen3-tts-gguf/resolve/main";
const KOKORO_MODELS_TAG: &str = "kokoro-v1.0";
const KOKORO_MODEL_DIR: &str = "chirp-kokoro-models-kokoro-v1.0";
const KOKORO_BUNDLE_URL: &str = "https://huggingface.co/thewh1teagle/chirp-kokoro-models/resolve/main/chirp-kokoro-models-kokoro-v1.0.tar.gz";
const VOICES_CATALOG_URL: &str = "https://raw.githubusercontent.com/thewh1teagle/chirp/main/chirp-desktop/src/assets/voices.json";

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSourceFile {
    name: &'static str,
    url: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSource {
    id: &'static str,
    name: &'static str,
    version: &'static str,
    #[serde(skip_serializing_if = "is_false")]
    recommended: bool,
    size: &'static str,
    description: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    files: Vec<ModelSourceFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    archive_url: Option<&'static str>,
    directory: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSourcesResponse {
    runtimes: Vec<ModelSource>,
    voices_url: &'static str,
    default_paths: Vec<&'static str>,
}

pub fn model_sources() -> ModelSourcesResponse {
    ModelSourcesResponse {
        runtimes: vec![
            ModelSource {
                id: "qwen",
                name: "Qwen",
                version: QWEN_MODELS_TAG,
                recommended: false,
                size: "~900 MB",
                description: "Voice cloning, multilingual synthesis, best quality on supported GPU hardware.",
                files: vec![
                    ModelSourceFile {
                        name: QWEN_MODEL_FILE,
                        url: format!("{QWEN_MODEL_BASE_URL}/{QWEN_MODEL_FILE}"),
                    },
                    ModelSourceFile {
                        name: QWEN_CODEC_FILE,
                        url: format!("{QWEN_MODEL_BASE_URL}/{QWEN_CODEC_FILE}"),
                    },
                ],
                archive_url: None,
                directory: "chirp-models-q5_0",
            },
            ModelSource {
                id: "kokoro",
                name: "Kokoro",
                version: KOKORO_MODELS_TAG,
                recommended: true,
                size: "~336 MB",
                description: "Fast local multi-voice speech with a lighter model bundle.",
                files: Vec::new(),
                archive_url: Some(KOKORO_BUNDLE_URL),
                directory: KOKORO_MODEL_DIR,
            },
        ],
        voices_url: VOICES_CATALOG_URL,
        default_paths: vec![
            "macOS: ~/Library/Application Support/com.thewh1teagle.chirp/models",
            "Windows: %LOCALAPPDATA%\\com.thewh1teagle.chirp\\models",
            "Linux: ~/.local/share/com.thewh1teagle.chirp/models",
        ],
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
