use futures_util::StreamExt;
use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::{Emitter, Manager};
use tokio::io::AsyncWriteExt;

const MODELS_TAG: &str = "chirp-models-v0.1.3";
const MODEL_DIR: &str = "chirp-models-q5_0";
const MODEL_FILE: &str = "qwen3-tts-model.gguf";
const CODEC_FILE: &str = "qwen3-tts-codec.gguf";
const MODEL_BASE_URL: &str = "https://huggingface.co/thewh1teagle/qwen3-tts-gguf/resolve/main";

#[derive(Debug, Serialize)]
pub struct ModelBundle {
    pub installed: bool,
    pub model_path: String,
    pub codec_path: String,
    pub model_dir: String,
    pub version: String,
    pub url: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelDownloadProgress {
    downloaded: u64,
    total: Option<u64>,
    progress: Option<f64>,
    stage: &'static str,
}

#[tauri::command]
pub async fn get_model_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    model_bundle(&app)
}

#[tauri::command]
pub async fn download_model_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    let bundle = model_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }

    let models_root = models_root(&app)?;
    tokio::fs::create_dir_all(&models_root)
        .await
        .map_err(|err| format!("failed to create {}: {err}", models_root.display()))?;
    let dir = model_dir(&app)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let mut downloaded = 0_u64;
    let model_total = remote_content_length(&client, &model_file_url(MODEL_FILE)).await;
    let codec_total = remote_content_length(&client, &model_file_url(CODEC_FILE)).await;
    let total = match (model_total, codec_total) {
        (Some(model), Some(codec)) => Some(model + codec),
        _ => None,
    };

    download_model_file(
        &app,
        &client,
        &model_file_url(MODEL_FILE),
        &dir.join(MODEL_FILE),
        &mut downloaded,
        total,
    )
    .await?;
    download_model_file(
        &app,
        &client,
        &model_file_url(CODEC_FILE),
        &dir.join(CODEC_FILE),
        &mut downloaded,
        total,
    )
    .await?;

    let bundle = model_bundle(&app)?;
    if !bundle.installed {
        return Err("model files downloaded, but expected GGUF files were not found".to_string());
    }
    Ok(bundle)
}

pub fn model_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let dir = model_dir(app)?;
    let model_path = dir.join(MODEL_FILE);
    let codec_path = dir.join(CODEC_FILE);
    Ok(ModelBundle {
        installed: model_path.exists() && codec_path.exists(),
        model_path: path_string(&model_path),
        codec_path: path_string(&codec_path),
        model_dir: path_string(&dir),
        version: MODELS_TAG.to_string(),
        url: MODEL_BASE_URL.to_string(),
    })
}

fn models_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_local_data_dir()
        .map_err(|err| format!("failed to resolve app data dir: {err}"))?
        .join("models");
    Ok(dir)
}

fn model_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(models_root(app)?.join(MODEL_DIR))
}

async fn remote_content_length(client: &reqwest::Client, url: &str) -> Option<u64> {
    client
        .head(url)
        .send()
        .await
        .ok()
        .filter(|response| response.status().is_success())
        .and_then(|response| response_content_length(&response))
}

fn response_content_length(response: &reqwest::Response) -> Option<u64> {
    response.content_length().or_else(|| {
        response
            .headers()
            .get("x-linked-size")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
    })
}

async fn download_model_file(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    downloaded: &mut u64,
    total: Option<u64>,
) -> Result<(), String> {
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.gguf")
    ));
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("failed to download model file {url}: {err}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "model file download failed for {url}: {}",
            response.status(),
        ));
    }

    let fallback_file_total = response_content_length(&response);
    emit_progress(
        app,
        ModelDownloadProgress {
            downloaded: *downloaded,
            total,
            progress: total
                .filter(|total| *total > 0)
                .map(|total| *downloaded as f64 / total as f64),
            stage: "downloading",
        },
    );

    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| format!("failed to create {}: {err}", part.display()))?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|err| format!("failed to read model download from {url}: {err}"))?;
        *downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .await
            .map_err(|err| format!("failed to write {}: {err}", part.display()))?;
        let progress_total = total.or(fallback_file_total);
        emit_progress(
            app,
            ModelDownloadProgress {
                downloaded: *downloaded,
                total: progress_total,
                progress: progress_total
                    .filter(|total| *total > 0)
                    .map(|total| *downloaded as f64 / total as f64),
                stage: "downloading",
            },
        );
    }
    file.flush()
        .await
        .map_err(|err| format!("failed to flush {}: {err}", part.display()))?;
    tokio::fs::rename(&part, dest).await.map_err(|err| {
        format!(
            "failed to move {} to {}: {err}",
            part.display(),
            dest.display()
        )
    })
}

fn model_file_url(file_name: &str) -> String {
    format!("{MODEL_BASE_URL}/{file_name}")
}

fn emit_progress(app: &tauri::AppHandle, payload: ModelDownloadProgress) {
    let _ = app.emit("model_download_progress", payload);
}

fn path_string(path: &Path) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}
