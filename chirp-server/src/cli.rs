use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{ArgAction, Parser, Subcommand};

use crate::{
    parent::watch_parent,
    runtime::{qwen_language_id, QwenRuntime, Runtime, RuntimeParams},
    server::{listen_and_serve, LoadParams, Server},
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const COMMIT: &str = match option_env!("CHIRP_COMMIT") {
    Some(commit) => commit,
    None => "dev",
};

#[derive(Debug, Parser)]
#[command(name = "chirp-server", version = VERSION, about = "Chirp local TTS server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, short, default_value_t = 0)]
        port: u16,
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        codec: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        max_tokens: i32,
        #[arg(long, default_value_t = 0.9)]
        temperature: f32,
        #[arg(long, default_value_t = 50)]
        top_k: i32,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        exit_with_parent: bool,
    },
    Speak {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        codec: PathBuf,
        #[arg(long)]
        text: String,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long = "ref")]
        ref_wav: Option<PathBuf>,
        #[arg(long, short)]
        output: PathBuf,
        #[arg(long, default_value = "auto")]
        language: String,
        #[arg(long, default_value_t = 0)]
        max_tokens: i32,
        #[arg(long, default_value_t = 0.9)]
        temperature: f32,
        #[arg(long, default_value_t = 50)]
        top_k: i32,
    },
    Languages {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        codec: PathBuf,
    },
}

pub async fn run() -> Result<()> {
    match Args::parse().command {
        Command::Serve {
            host,
            port,
            model,
            codec,
            max_tokens,
            temperature,
            top_k,
            exit_with_parent,
        } => {
            if exit_with_parent {
                tokio::spawn(watch_parent());
            }
            let server = Server::new(VERSION.into(), COMMIT.into());
            if model.is_some() || codec.is_some() {
                let (Some(model), Some(codec)) = (model, codec) else {
                    bail!("--model and --codec must be provided together");
                };
                server
                    .load_model(LoadParams {
                        runtime: "qwen".into(),
                        qwen: RuntimeParams::Qwen {
                            model_path: model,
                            codec_path: codec,
                            max_tokens,
                            temperature,
                            top_k,
                        },
                        kokoro: None,
                    })
                    .await?;
            }
            listen_and_serve(&host, port, server).await
        }
        Command::Speak {
            model,
            codec,
            text,
            reference,
            ref_wav,
            output,
            language,
            max_tokens,
            temperature,
            top_k,
        } => {
            let mut runtime = QwenRuntime::load(model, codec, max_tokens, temperature, top_k)?;
            qwen_language_id(runtime.languages(), &language)?;
            runtime.synthesize_to_file(&text, reference.or(ref_wav).as_deref(), &output, &language)?;
            println!("{}", serde_json::json!({ "output": output }));
            Ok(())
        }
        Command::Languages { model, codec } => {
            let runtime = QwenRuntime::load(model, codec, 0, 0.9, 50)?;
            let languages = runtime.languages();
            let names = std::iter::once("auto".to_string())
                .chain(languages.iter().map(|language| language.name.clone()))
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::json!({
                    "languages": names,
                    "items": languages,
                })
            );
            Ok(())
        }
    }
}
