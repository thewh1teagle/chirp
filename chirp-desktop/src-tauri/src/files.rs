use std::{panic, path::PathBuf};

#[tauri::command]
pub async fn reveal_path(path: PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("path does not exist: {}", path.display()));
    }

    tauri::async_runtime::spawn_blocking(move || {
        panic::catch_unwind(|| {
            showfile::show_path_in_file_manager(path);
        })
        .map_err(|_| "failed to reveal path in file manager".to_string())
    })
    .await
    .map_err(|err| format!("failed to join reveal task: {err}"))?
}
