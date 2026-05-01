use std::path::PathBuf;

use clap::Parser;
use kokoro_rs::{write_mono_i16_wav, Kokoro, KokoroConfig, SynthesizeRequest};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    voices: PathBuf,
    #[arg(long, default_value = "af_heart")]
    voice: String,
    #[arg(long, default_value = "en-us")]
    language: String,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long)]
    text: String,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> kokoro_rs::Result<()> {
    let args = Args::parse();
    let config = KokoroConfig::new(args.model, args.voices)
        .voice(args.voice)
        .language(args.language)
        .speed(args.speed);
    let mut kokoro = Kokoro::load(config)?;
    let audio = kokoro.synthesize(SynthesizeRequest::new(args.text))?;
    write_mono_i16_wav(args.output, &audio)?;
    Ok(())
}
