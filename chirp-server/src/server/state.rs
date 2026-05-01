use std::{path::PathBuf, sync::Arc};

use anyhow::{Result, bail};
use tokio::sync::Mutex;

use crate::runtime::{KokoroRuntime, QwenRuntime, Runtime, RuntimeParams};

pub type SharedServer = Arc<Server>;

pub struct LoadParams {
    pub runtime: String,
    pub qwen: RuntimeParams,
    pub kokoro: Option<RuntimeParams>,
}

pub struct Server {
    pub(crate) inner: Mutex<ServerState>,
    pub(crate) version: String,
    pub(crate) commit: String,
}

pub(crate) struct ServerState {
    pub(crate) ctx: Option<Box<dyn Runtime>>,
    pub(crate) runtime: String,
    pub(crate) model_name: String,
    pub(crate) model_path: String,
    pub(crate) codec_path: String,
}

impl Server {
    pub fn new(version: String, commit: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(ServerState {
                ctx: None,
                runtime: String::new(),
                model_name: String::new(),
                model_path: String::new(),
                codec_path: String::new(),
            }),
            version,
            commit,
        })
    }

    pub async fn load_model(&self, params: LoadParams) -> Result<()> {
        let mut inner = self.inner.lock().await;
        let runtime = if params.runtime.is_empty() {
            "qwen".to_string()
        } else {
            params.runtime
        };
        let (ctx, model_path, codec_path): (Box<dyn Runtime>, PathBuf, PathBuf) =
            match runtime.as_str() {
                "qwen" => match params.qwen {
                    RuntimeParams::Qwen {
                        model_path,
                        codec_path,
                        max_tokens,
                        temperature,
                        top_k,
                    } => (
                        Box::new(QwenRuntime::load(
                            model_path.clone(),
                            codec_path.clone(),
                            max_tokens,
                            temperature,
                            top_k,
                        )?),
                        model_path,
                        codec_path,
                    ),
                    _ => bail!("invalid qwen runtime params"),
                },
                "kokoro" => match params.kokoro {
                    Some(RuntimeParams::Kokoro {
                        model_path,
                        voices_path,
                        voice,
                        language,
                        speed,
                    }) => (
                        Box::new(KokoroRuntime::load(
                            model_path.clone(),
                            voices_path.clone(),
                            voice,
                            language,
                            speed,
                        )?),
                        model_path,
                        voices_path,
                    ),
                    _ => bail!("invalid kokoro runtime params"),
                },
                other => bail!("unsupported runtime {other:?}"),
            };
        inner.ctx = Some(ctx);
        inner.runtime = runtime;
        inner.model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .into();
        inner.model_path = model_path.display().to_string();
        inner.codec_path = codec_path.display().to_string();
        Ok(())
    }

    pub(crate) async fn unload_model(&self) {
        let mut inner = self.inner.lock().await;
        inner.ctx = None;
        inner.runtime.clear();
        inner.model_name.clear();
        inner.model_path.clear();
        inner.codec_path.clear();
    }
}
