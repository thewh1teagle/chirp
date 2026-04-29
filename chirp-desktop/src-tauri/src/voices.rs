use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::Manager;

#[derive(Debug, Deserialize)]
pub struct DownloadVoiceRequest {
    pub id: String,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct DownloadedVoice {
    pub path: String,
}

#[tauri::command]
pub async fn download_voice(
    app: tauri::AppHandle,
    request: DownloadVoiceRequest,
) -> Result<DownloadedVoice, String> {
    let id = sanitize_voice_id(&request.id)?;
    if request.url.trim().is_empty() {
        return Err("voice URL is empty".to_string());
    }

    let voices_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|err| format!("failed to resolve app data dir: {err}"))?
        .join("voices");
    tokio::fs::create_dir_all(&voices_dir)
        .await
        .map_err(|err| format!("failed to create {}: {err}", voices_dir.display()))?;

    let dest = voices_dir.join(format!("{id}.wav"));
    if dest.exists() {
        return Ok(DownloadedVoice {
            path: path_string(&dest),
        });
    }

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let bytes = client
        .get(&request.url)
        .send()
        .await
        .map_err(|err| format!("failed to download voice: {err}"))?
        .error_for_status()
        .map_err(|err| format!("failed to download voice: {err}"))?
        .bytes()
        .await
        .map_err(|err| format!("failed to read voice response: {err}"))?;

    tokio::fs::write(&dest, bytes)
        .await
        .map_err(|err| format!("failed to write {}: {err}", dest.display()))?;
    Ok(DownloadedVoice {
        path: path_string(&dest),
    })
}

fn sanitize_voice_id(id: &str) -> Result<String, String> {
    let value = id.trim();
    if value.is_empty() {
        return Err("voice id is empty".to_string());
    }
    if value
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        Ok(value.to_string())
    } else {
        Err(format!("invalid voice id: {id}"))
    }
}

fn path_string(path: &PathBuf) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}
