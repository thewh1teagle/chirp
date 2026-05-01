use std::path::PathBuf;

use clap::Parser;
use qwentts_rs::audio::wav::write_mono_i16_wav;
use qwentts_rs::{QwenTts, QwenTtsConfig, Result, SynthesizeRequest};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    codec: PathBuf,

    #[arg(long)]
    text: String,

    #[arg(long, short)]
    output: PathBuf,

    #[arg(long)]
    ref_wav: Option<PathBuf>,

    #[arg(long, default_value_t = 180)]
    max_tokens: i32,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1)]
    top_k: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut config = QwenTtsConfig::new(args.model).codec_path(args.codec);
    config.max_tokens = args.max_tokens;
    config.temperature = args.temperature;
    config.top_k = args.top_k;

    let mut request = SynthesizeRequest::new(args.text);
    request.ref_wav_path = args.ref_wav;

    let mut tts = QwenTts::load(config)?;
    let audio = tts.synthesize(request)?;
    write_mono_i16_wav(args.output, &audio)?;

    Ok(())
}
