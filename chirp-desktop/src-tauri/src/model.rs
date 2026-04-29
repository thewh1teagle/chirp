use flate2::read::GzDecoder;
use futures_util::StreamExt;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tar::Archive;
use tauri::{Emitter, Manager};
use tokio::io::AsyncWriteExt;

const MODELS_TAG: &str = "chirp-models-v0.1.3";
const MODEL_DIR: &str = "chirp-models-q5_0";
const MODEL_FILE: &str = "qwen3-tts-model.gguf";
const CODEC_FILE: &str = "qwen3-tts-codec.gguf";
const MODEL_BASE_URL: &str = "https://huggingface.co/thewh1teagle/qwen3-tts-gguf/resolve/main";
const KOKORO_MODELS_TAG: &str = "kokoro-v1.0";
const KOKORO_MODEL_DIR: &str = "chirp-kokoro-models-kokoro-v1.0";
const KOKORO_MODEL_FILE: &str = "kokoro-v1.0.onnx";
const KOKORO_VOICES_FILE: &str = "voices-v1.0.bin";
const KOKORO_ESPEAK_DIR: &str = "espeak-ng-data";
const KOKORO_BUNDLE_URL: &str = "https://huggingface.co/thewh1teagle/chirp-kokoro-models/resolve/main/chirp-kokoro-models-kokoro-v1.0.tar.gz";

#[derive(Debug, Clone, Serialize)]
pub struct ModelBundle {
    pub installed: bool,
    pub runtime: String,
    pub model_path: String,
    pub codec_path: String,
    pub voices_path: Option<String>,
    pub espeak_data_path: Option<String>,
    pub model_dir: String,
    pub version: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSourceFile {
    pub name: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSource {
    pub id: String,
    pub name: String,
    pub version: String,
    pub recommended: bool,
    pub size: String,
    pub description: String,
    pub files: Vec<ModelSourceFile>,
    pub archive_url: Option<String>,
    pub directory: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSources {
    pub runtimes: Vec<ModelSource>,
    pub voices_url: String,
    pub default_paths: Vec<String>,
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
    model_bundle_for_runtime(&app, "qwen")
}

#[tauri::command]
pub async fn get_model_bundle_for_runtime(
    app: tauri::AppHandle,
    runtime: String,
) -> Result<ModelBundle, String> {
    model_bundle_for_runtime(&app, &runtime)
}

#[tauri::command]
pub fn get_model_sources() -> ModelSources {
    model_sources()
}

#[tauri::command]
pub async fn download_model_bundle(
    app: tauri::AppHandle,
    runtime: Option<String>,
) -> Result<ModelBundle, String> {
    let runtime = runtime.unwrap_or_else(|| "qwen".to_string());
    if runtime == "kokoro" {
        return download_kokoro_bundle(app).await;
    }
    let bundle = qwen_bundle(&app)?;
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
    let source = runtime_source("qwen").ok_or_else(|| "missing qwen source".to_string())?;
    let model_url = source
        .files
        .iter()
        .find(|file| file.name == MODEL_FILE)
        .map(|file| file.url.clone())
        .ok_or_else(|| "missing qwen model source URL".to_string())?;
    let codec_url = source
        .files
        .iter()
        .find(|file| file.name == CODEC_FILE)
        .map(|file| file.url.clone())
        .ok_or_else(|| "missing qwen codec source URL".to_string())?;
    let model_total = remote_content_length(&client, &model_url).await;
    let codec_total = remote_content_length(&client, &codec_url).await;
    let total = match (model_total, codec_total) {
        (Some(model), Some(codec)) => Some(model + codec),
        _ => None,
    };

    download_model_file(
        &app,
        &client,
        &model_url,
        &dir.join(MODEL_FILE),
        &mut downloaded,
        total,
    )
    .await?;
    download_model_file(
        &app,
        &client,
        &codec_url,
        &dir.join(CODEC_FILE),
        &mut downloaded,
        total,
    )
    .await?;

    let bundle = qwen_bundle(&app)?;
    if !bundle.installed {
        return Err("model files downloaded, but expected GGUF files were not found".to_string());
    }
    Ok(bundle)
}

pub fn model_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let qwen = qwen_bundle(app)?;
    if qwen.installed {
        return Ok(qwen);
    }
    let kokoro = kokoro_bundle(app)?;
    if kokoro.installed {
        return Ok(kokoro);
    }
    Ok(qwen)
}

pub fn model_bundle_for_runtime(
    app: &tauri::AppHandle,
    runtime: &str,
) -> Result<ModelBundle, String> {
    match runtime {
        "kokoro" => kokoro_bundle(app),
        _ => qwen_bundle(app),
    }
}

fn qwen_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("qwen").ok_or_else(|| "missing qwen source".to_string())?;
    let dir = model_dir(app)?;
    let model_path = dir.join(MODEL_FILE);
    let codec_path = dir.join(CODEC_FILE);
    Ok(ModelBundle {
        installed: model_path.exists() && codec_path.exists(),
        runtime: "qwen".to_string(),
        model_path: path_string(&model_path),
        codec_path: path_string(&codec_path),
        voices_path: None,
        espeak_data_path: None,
        model_dir: path_string(&dir),
        version: source.version,
        url: MODEL_BASE_URL.to_string(),
    })
}

fn kokoro_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("kokoro").ok_or_else(|| "missing kokoro source".to_string())?;
    let dir = models_root(app)?.join(KOKORO_MODEL_DIR);
    let model_path = dir.join(KOKORO_MODEL_FILE);
    let voices_path = dir.join(KOKORO_VOICES_FILE);
    let espeak_data_path = dir.join(KOKORO_ESPEAK_DIR);
    Ok(ModelBundle {
        installed: model_path.exists() && voices_path.exists() && espeak_data_path.is_dir(),
        runtime: "kokoro".to_string(),
        model_path: path_string(&model_path),
        codec_path: String::new(),
        voices_path: Some(path_string(&voices_path)),
        espeak_data_path: Some(path_string(&espeak_data_path)),
        model_dir: path_string(&dir),
        version: source.version,
        url: source.archive_url.unwrap_or_else(|| KOKORO_BUNDLE_URL.to_string()),
    })
}

fn model_sources() -> ModelSources {
    ModelSources {
        runtimes: vec![
            ModelSource {
                id: "qwen".to_string(),
                name: "Qwen".to_string(),
                version: MODELS_TAG.to_string(),
                recommended: false,
                size: "~900 MB".to_string(),
                description: "Voice cloning, multilingual synthesis, best quality on supported GPU hardware.".to_string(),
                files: vec![
                    ModelSourceFile {
                        name: MODEL_FILE.to_string(),
                        url: model_file_url(MODEL_FILE),
                    },
                    ModelSourceFile {
                        name: CODEC_FILE.to_string(),
                        url: model_file_url(CODEC_FILE),
                    },
                ],
                archive_url: None,
                directory: MODEL_DIR.to_string(),
            },
            ModelSource {
                id: "kokoro".to_string(),
                name: "Kokoro".to_string(),
                version: KOKORO_MODELS_TAG.to_string(),
                recommended: true,
                size: "~336 MB".to_string(),
                description: "Fast local multi-voice speech with a lighter model bundle.".to_string(),
                files: Vec::new(),
                archive_url: Some(KOKORO_BUNDLE_URL.to_string()),
                directory: KOKORO_MODEL_DIR.to_string(),
            },
        ],
        voices_url: "https://raw.githubusercontent.com/thewh1teagle/chirp/main/chirp-desktop/src/assets/voices.json".to_string(),
        default_paths: vec![
            "macOS: ~/Library/Application Support/com.thewh1teagle.chirp/models".to_string(),
            "Windows: %LOCALAPPDATA%\\com.thewh1teagle.chirp\\models".to_string(),
            "Linux: ~/.local/share/com.thewh1teagle.chirp/models".to_string(),
        ],
    }
}

fn runtime_source(runtime: &str) -> Option<ModelSource> {
    model_sources()
        .runtimes
        .into_iter()
        .find(|source| source.id == runtime)
}

async fn download_kokoro_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    let bundle = kokoro_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }
    let source = runtime_source("kokoro").ok_or_else(|| "missing kokoro source".to_string())?;
    let archive_url = source
        .archive_url
        .ok_or_else(|| "missing kokoro archive source URL".to_string())?;
    let models_root = models_root(&app)?;
    tokio::fs::create_dir_all(&models_root)
        .await
        .map_err(|err| format!("failed to create {}: {err}", models_root.display()))?;
    let archive_path = models_root.join("chirp-kokoro-models-kokoro-v1.0.tar.gz.part");
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let mut downloaded = 0_u64;
    let total = remote_content_length(&client, &archive_url).await;
    download_model_file(
        &app,
        &client,
        &archive_url,
        &archive_path,
        &mut downloaded,
        total,
    )
    .await?;

    emit_progress(
        &app,
        ModelDownloadProgress {
            downloaded,
            total,
            progress: Some(1.0),
            stage: "extracting",
        },
    );
    let root = models_root.clone();
    let archive = archive_path.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let file = fs::File::open(&archive)
            .map_err(|err| format!("failed to open {}: {err}", archive.display()))?;
        let decoder = GzDecoder::new(file);
        let mut tar = Archive::new(decoder);
        tar.unpack(&root)
            .map_err(|err| format!("failed to extract Kokoro model bundle: {err}"))?;
        let _ = fs::remove_file(&archive);
        Ok(())
    })
    .await
    .map_err(|err| format!("failed to join extraction task: {err}"))??;
    let bundle = kokoro_bundle(&app)?;
    if !bundle.installed {
        return Err("Kokoro bundle extracted, but expected files were not found".to_string());
    }
    Ok(bundle)
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
