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
    speaker_encoder: SpeakerEncoder,
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
    pub ref_wav_path: Option<PathBuf>,
    pub language_id: Option<i32>,
}

impl SynthesizeRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ref_wav_path: None,
            language_id: None,
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
        let speaker_encoder = SpeakerEncoder::load(&model)?;
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

    pub fn synthesize(&mut self, request: SynthesizeRequest) -> Result<AudioSamples> {
        let token_ids = self
            .tokenizer
            .encode_tts_text(&request.text)?
            .ids
            .into_iter()
            .map(|id| id as i32)
            .collect::<Vec<_>>();

        let speaker = if let Some(path) = request.ref_wav_path {
            self.speaker_encoder.extract(path)?
        } else {
            vec![0.0; self.ar_map.config.hidden_size as usize]
        };
        let max_tokens = self.config.max_tokens.max(0) as usize;
        let codes = self.ar_weights.generate_codes(
            &token_ids,
            Some(&speaker),
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
