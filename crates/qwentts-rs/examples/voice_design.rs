use std::path::PathBuf;

use clap::Parser;
use qwentts_rs::audio::wav::write_mono_i16_wav;
use qwentts_rs::{QwenTts, QwenTtsConfig, Result, SynthesizeRequest};

const WHISPER_TEXT: &str =
    "Please keep your voice low. The lights are off, and I only need a whisper.";
const WHISPER_PROMPT: &str = "Whisper the line very quietly in English, with soft breath, close-mic intimacy, minimal pitch movement, and no projected speaking voice.";

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value = "models/qwen3-tts-1.7b-voicedesign-f16.gguf")]
    model: PathBuf,

    #[arg(long, default_value = "models/qwen3-tts-tokenizer-q5_0.gguf")]
    codec: PathBuf,

    #[arg(long, default_value = WHISPER_TEXT)]
    text: String,

    #[arg(long, default_value = WHISPER_PROMPT)]
    prompt: String,

    #[arg(long, default_value = "English")]
    language: String,

    #[arg(
        long,
        short,
        default_value = "crates/qwentts-rs/examples/qwen3_tts_voice_design_whisper_rs.wav"
    )]
    output: PathBuf,

    #[arg(long, default_value_t = 240)]
    max_tokens: i32,

    #[arg(long, default_value_t = 0.9)]
    temperature: f32,

    #[arg(long, default_value_t = 50)]
    top_k: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut config = QwenTtsConfig::new(args.model).codec_path(args.codec);
    config.max_tokens = args.max_tokens;
    config.temperature = args.temperature;
    config.top_k = args.top_k;

    let mut tts = QwenTts::load(config)?;
    let language_id = tts.language_id(&args.language);

    let mut request = SynthesizeRequest::voice_design(args.text, args.prompt);
    request.language_id = language_id;

    let audio = tts.synthesize(request)?;
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_mono_i16_wav(&args.output, &audio)?;
    println!(
        "Wrote {} at {} Hz",
        args.output.display(),
        audio.sample_rate
    );

    Ok(())
}
