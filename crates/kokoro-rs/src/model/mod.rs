use std::path::PathBuf;

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::{
    phoneme::{chunk_text, espeak_to_misaki, pack_misaki_sentences, phonemize, tokenize_phonemes},
    voice::{load_voice, VoiceData},
    AudioSamples, Error, Result,
};

const SAMPLE_RATE: u32 = 24_000;
const MAX_PHONEME_LENGTH: usize = 510;
const STYLE_DIMS: usize = 256;
const INTER_BATCH_SILENCE_SECONDS: f32 = 0.12;

#[derive(Debug, Clone)]
pub struct KokoroConfig {
    pub model_path: PathBuf,
    pub voices_path: PathBuf,
    pub voice: String,
    pub language: String,
    pub speed: f32,
}

impl KokoroConfig {
    pub fn new(model_path: impl Into<PathBuf>, voices_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            voices_path: voices_path.into(),
            voice: "af_heart".into(),
            language: "en-us".into(),
            speed: 1.0,
        }
    }

    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }
}

#[derive(Debug, Clone)]
pub struct SynthesizeRequest {
    pub text: String,
}

impl SynthesizeRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

pub struct Kokoro {
    session: Session,
    voice: VoiceData,
    language: String,
    speed: f32,
}

impl Kokoro {
    pub fn load(config: KokoroConfig) -> Result<Self> {
        let voice = load_voice(&config.voices_path, &config.voice)?;
        if voice.dims != STYLE_DIMS {
            return Err(Error::InvalidVoice {
                voice: config.voice,
                reason: format!("expected {STYLE_DIMS} dims, got {}", voice.dims),
            });
        }

        let session = Session::builder()?
            .with_intra_threads(1)
            .map_err(|err| Error::Message(err.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|err| Error::Message(err.to_string()))?
            .commit_from_file(&config.model_path)?;

        Ok(Self {
            session,
            voice,
            language: config.language,
            speed: if config.speed > 0.0 { config.speed } else { 1.0 },
        })
    }

    pub fn synthesize(&mut self, request: SynthesizeRequest) -> Result<AudioSamples> {
        let text_chunks = chunk_text(&request.text);
        let mut sentences = Vec::new();
        let british = self.language == "en";

        for chunk in text_chunks {
            for phonemes in phonemize(&chunk, &self.language)? {
                let misaki = espeak_to_misaki(&phonemes, british);
                if !misaki.is_empty() {
                    sentences.push(misaki);
                }
            }
        }

        let max_tokens = MAX_PHONEME_LENGTH.min(self.voice.rows.saturating_sub(1));
        let batches = pack_misaki_sentences(&sentences, max_tokens);
        let silence_len = (SAMPLE_RATE as f32 * INTER_BATCH_SILENCE_SECONDS).round() as usize;
        let mut samples = Vec::new();

        for batch in batches {
            let phoneme_tokens = tokenize_phonemes(&batch);
            if phoneme_tokens.is_empty() {
                continue;
            }
            if !samples.is_empty() {
                samples.extend(std::iter::repeat(0.0).take(silence_len));
            }
            samples.extend(self.infer(&phoneme_tokens)?);
        }

        if samples.is_empty() {
            return Err(Error::NoAudio);
        }

        Ok(AudioSamples {
            sample_rate: SAMPLE_RATE,
            samples,
        })
    }

    fn infer(&mut self, phoneme_tokens: &[i64]) -> Result<Vec<f32>> {
        let mut tokens = Vec::with_capacity(phoneme_tokens.len() + 2);
        tokens.push(0);
        tokens.extend_from_slice(phoneme_tokens);
        tokens.push(0);

        let style = self.voice.style_for_token_count(phoneme_tokens.len()).to_vec();
        let outputs = self.session.run(ort::inputs! {
            "tokens" => Tensor::<i64>::from_array(([1, tokens.len()], tokens))?,
            "style" => Tensor::<f32>::from_array(([1, self.voice.dims], style))?,
            "speed" => Tensor::<f32>::from_array(([1], vec![self.speed]))?,
        })?;
        let audio = outputs["audio"].try_extract_array::<f32>()?;
        Ok(audio.iter().copied().collect())
    }
}
