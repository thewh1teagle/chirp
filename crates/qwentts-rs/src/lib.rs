use std::path::{Path, PathBuf};

pub mod ar;
pub mod audio;
pub mod codec;
pub mod error;
pub mod ggml_runtime;
pub mod speaker;
pub mod text;

pub use error::{Error, Result};
pub use ggml_runtime::gguf::{GgufModel, TensorInfo};

use ar::ArTensorMap;
use codec::CodecTensorMap;
use ggml_runtime::GgmlWeights;
use speaker::SpeakerEncoder;
use text::tokenizer::QwenTokenizer;

#[derive(Debug, Clone)]
pub struct QwenTtsConfig {
    pub model_path: PathBuf,
    pub codec_path: Option<PathBuf>,
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
}

impl QwenTtsConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            codec_path: None,
            max_tokens: 8192,
            temperature: 0.9,
            top_k: 50,
            repetition_penalty: 1.05,
        }
    }

    pub fn codec_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.codec_path = Some(path.into());
        self
    }
}

pub struct QwenTts {
    model: GgufModel,
    codec: Option<GgufModel>,
    tokenizer: QwenTokenizer,
    ar_map: ArTensorMap,
    ar_weights: GgmlWeights,
    speaker_encoder: Option<SpeakerEncoder>,
    codec_map: CodecTensorMap,
    codec_weights: GgmlWeights,
    config: QwenTtsConfig,
}

#[derive(Debug, Clone)]
pub struct AudioSamples {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
    pub language_id: Option<i32>,
    pub mode: SynthesisMode,
}

#[derive(Debug, Clone)]
pub enum SynthesisMode {
    /// Default synthesis path used by the existing base model flow.
    Plain,
    /// Base-model voice cloning from a reference WAV.
    VoiceClone { ref_wav_path: PathBuf },
    /// VoiceDesign models prepend a user instruction prompt and do not use a speaker embedding.
    VoiceDesign { instruct: String },
}

impl SynthesizeRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language_id: None,
            mode: SynthesisMode::Plain,
        }
    }

    pub fn voice_clone(text: impl Into<String>, ref_wav_path: impl Into<PathBuf>) -> Self {
        Self {
            text: text.into(),
            language_id: None,
            mode: SynthesisMode::VoiceClone {
                ref_wav_path: ref_wav_path.into(),
            },
        }
    }

    pub fn voice_design(text: impl Into<String>, instruct: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language_id: None,
            mode: SynthesisMode::VoiceDesign {
                instruct: instruct.into(),
            },
        }
    }
}

impl QwenTts {
    pub fn load(config: QwenTtsConfig) -> Result<Self> {
        let model = GgufModel::open(&config.model_path)?;
        let codec = config
            .codec_path
            .as_ref()
            .map(GgufModel::open)
            .transpose()?;
        let codec_model = codec.as_ref().ok_or(Error::MissingCodec)?;
        let tokenizer = QwenTokenizer::from_gguf(&model)?;
        let ar_map = ArTensorMap::load(&model)?;
        let ar_weights = GgmlWeights::load_ar(&model, &ar_map)?;
        let speaker_encoder = match SpeakerEncoder::load(&model) {
            Ok(encoder) => Some(encoder),
            // VoiceDesign checkpoints omit the speaker encoder; clone mode checks this explicitly.
            Err(Error::MissingTensor(name)) if name.starts_with("spk_enc.") => None,
            Err(err) => return Err(err),
        };
        let codec_map = CodecTensorMap::load(codec_model)?;
        let codec_weights = GgmlWeights::load_codec(codec_model, &codec_map)?;
        Ok(Self {
            model,
            codec,
            tokenizer,
            ar_map,
            ar_weights,
            speaker_encoder,
            codec_map,
            codec_weights,
            config,
        })
    }

    pub fn model(&self) -> &GgufModel {
        &self.model
    }

    pub fn codec(&self) -> Option<&GgufModel> {
        self.codec.as_ref()
    }

    pub fn config(&self) -> &QwenTtsConfig {
        &self.config
    }

    pub fn model_path(&self) -> &Path {
        &self.config.model_path
    }

    pub fn language_id(&self, language: &str) -> Option<i32> {
        self.ar_map
            .config
            .languages
            .iter()
            .find(|item| item.name.eq_ignore_ascii_case(language))
            .map(|item| item.id)
    }

    pub fn synthesize(&mut self, request: SynthesizeRequest) -> Result<AudioSamples> {
        let token_ids = self
            .tokenizer
            .encode_tts_text(&request.text)?
            .ids
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();
        let (instruct_ids, speaker) = match request.mode {
            SynthesisMode::Plain => (
                None,
                Some(vec![0.0; self.ar_map.config.hidden_size as usize]),
            ),
            SynthesisMode::VoiceClone { ref_wav_path } => {
                let speaker = self
                    .speaker_encoder
                    .as_ref()
                    .ok_or_else(|| {
                        Error::ModelConfig(
                            "reference WAV synthesis requires a model with speaker encoder tensors"
                                .into(),
                        )
                    })?
                    .extract(ref_wav_path)?;
                (None, Some(speaker))
            }
            SynthesisMode::VoiceDesign { instruct } => {
                let instruct_ids = if instruct.is_empty() {
                    None
                } else {
                    Some(
                        self.tokenizer
                            .encode_instruct_text(&instruct)?
                            .ids
                            .into_iter()
                            .map(|id| id as i32)
                            .collect::<Vec<_>>(),
                    )
                };
                (instruct_ids, None)
            }
        };
        let max_tokens = self.config.max_tokens.max(0) as usize;
        let codes = self.ar_weights.generate_codes(
            &token_ids,
            instruct_ids.as_deref(),
            speaker.as_deref(),
            max_tokens,
            ggml_runtime::GenerateOptions {
                language_id: request.language_id,
                repetition_penalty: self.config.repetition_penalty,
                temperature: self.config.temperature,
                top_k: self.config.top_k,
            },
            &self.ar_map.config,
        )?;

        let n_frames = codes.len() / self.codec_map.config.n_codebooks as usize;
        let samples = if n_frames == 0 {
            Vec::new()
        } else {
            self.codec_weights
                .decode_codec_codes(&codes, n_frames, &self.codec_map.config)?
        };
        Ok(AudioSamples {
            sample_rate: self.codec_map.config.sample_rate,
            samples,
        })
    }
}
