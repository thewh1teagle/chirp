use futures_util::StreamExt;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tauri::Manager;
use tokio::io::AsyncWriteExt;

const MODELS_TAG: &str = "chirp-models-v0.1.1";
const MODEL_ARCHIVE: &str = "chirp-models-q5_0.tar.gz";
const MODEL_DIR: &str = "chirp-models-q5_0";
const MODEL_FILE: &str = "qwen3-tts-model.gguf";
const CODEC_FILE: &str = "qwen3-tts-codec.gguf";
const MODEL_URL: &str =
    "https://github.com/thewh1teagle/chirp/releases/download/chirp-models-v0.1.1/chirp-models-q5_0.tar.gz";

#[derive(Debug, Serialize)]
pub struct ModelBundle {
    pub installed: bool,
    pub model_path: String,
    pub codec_path: String,
    pub model_dir: String,
    pub version: String,
    pub url: String,
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

    let archive_path = models_root.join(MODEL_ARCHIVE);
    download_archive(&archive_path).await?;
    extract_archive(archive_path.clone(), models_root.clone()).await?;
    let _ = tokio::fs::remove_file(&archive_path).await;

    let bundle = model_bundle(&app)?;
    if !bundle.installed {
        return Err("model archive extracted, but expected GGUF files were not found".to_string());
    }
    Ok(bundle)
}

fn model_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let dir = model_dir(app)?;
    let model_path = dir.join(MODEL_FILE);
    let codec_path = dir.join(CODEC_FILE);
    Ok(ModelBundle {
        installed: model_path.exists() && codec_path.exists(),
        model_path: path_string(&model_path),
        codec_path: path_string(&codec_path),
        model_dir: path_string(&dir),
        version: MODELS_TAG.to_string(),
        url: MODEL_URL.to_string(),
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

async fn download_archive(dest: &Path) -> Result<(), String> {
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(MODEL_ARCHIVE)
    ));
    let response = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?
        .get(MODEL_URL)
        .send()
        .await
        .map_err(|err| format!("failed to download model bundle: {err}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "model bundle download failed: {}",
            response.status()
        ));
    }

    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| format!("failed to create {}: {err}", part.display()))?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| format!("failed to read model download: {err}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|err| format!("failed to write {}: {err}", part.display()))?;
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

async fn extract_archive(archive_path: PathBuf, dest: PathBuf) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let file = fs::File::open(&archive_path)
            .map_err(|err| format!("failed to open {}: {err}", archive_path.display()))?;
        let decoder = flate2::read::GzDecoder::new(file);
        let mut archive = tar::Archive::new(decoder);
        for entry in archive
            .entries()
            .map_err(|err| format!("failed to read model archive: {err}"))?
        {
            let mut entry =
                entry.map_err(|err| format!("failed to read model archive entry: {err}"))?;
            let entry_path = entry
                .path()
                .map_err(|err| format!("failed to read archive path: {err}"))?;
            let out_path = safe_join(&dest, &entry_path)?;
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
            }
            entry
                .unpack(&out_path)
                .map_err(|err| format!("failed to extract {}: {err}", out_path.display()))?;
        }
        Ok(())
    })
    .await
    .map_err(|err| format!("failed to join extraction task: {err}"))?
}

fn safe_join(base: &Path, relative: &Path) -> Result<PathBuf, String> {
    let mut out = base.to_path_buf();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => out.push(part),
            std::path::Component::CurDir => {}
            _ => return Err("model archive contains an unsafe path".to_string()),
        }
    }
    Ok(out)
}

fn path_string(path: &Path) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}
