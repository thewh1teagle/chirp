use serde::{Deserialize, Serialize};
use std::{
    env,
    io::BufRead,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tauri::{Manager, State};

use crate::analytics;
use crate::model;

pub struct RunnerState {
    pub process: Mutex<Option<RunnerProcess>>,
}

pub struct RunnerProcess {
    port: u16,
    child: Child,
    client: reqwest::Client,
    stderr_buf: Arc<Mutex<String>>,
}

#[derive(Debug, Deserialize)]
struct ReadySignal {
    status: String,
    port: u16,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    code: String,
    message: String,
}

#[derive(Debug, Serialize)]
pub struct RunnerInfo {
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub runtime: Option<String>,
    pub model_path: String,
    pub codec_path: String,
    pub voices_path: Option<String>,
    pub espeak_data_path: Option<String>,
    pub voice: Option<String>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    pub voice_reference: Option<String>,
    pub voice: Option<String>,
    pub output_path: Option<String>,
    pub language: Option<String>,
}

impl RunnerProcess {
    fn spawn(app: &tauri::AppHandle, binary_path: &Path) -> Result<Self, String> {
        let mut cmd = Command::new(binary_path);
        cmd.args(["serve", "--host", "127.0.0.1", "--port", "0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Ok(bundle) = model::model_bundle(app) {
            if bundle.installed {
                cmd.env("CHIRP_RUNTIME", &bundle.runtime);
                cmd.env("CHIRP_MODEL_PATH", &bundle.model_path);
                if bundle.runtime == "kokoro" {
                    cmd.env("CHIRP_KOKORO_MODEL_PATH", &bundle.model_path);
                }
                if !bundle.codec_path.is_empty() {
                    cmd.env("CHIRP_CODEC_PATH", &bundle.codec_path);
                }
                if let Some(path) = &bundle.voices_path {
                    cmd.env("CHIRP_KOKORO_VOICES_PATH", path);
                }
                if let Some(path) = &bundle.espeak_data_path {
                    cmd.env("CHIRP_ESPEAK_DATA_PATH", path);
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "failed to spawn Chirp runner at {}: {err}",
                binary_path.display()
            )
        })?;

        let mut stderr = child.stderr.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture Chirp runner stdout".to_string())?;
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();

        if let Err(err) = reader.read_line(&mut line) {
            let stderr_output = read_first_stderr_line(stderr.take());
            kill_child(&mut child);
            return Err(format_runner_start_error(
                "failed to read ready signal",
                &err.to_string(),
                &stderr_output,
            ));
        }

        let signal: ReadySignal = match serde_json::from_str(line.trim()) {
            Ok(signal) => signal,
            Err(err) => {
                let stderr_output = read_first_stderr_line(stderr.take());
                kill_child(&mut child);
                return Err(format_runner_start_error(
                    "failed to parse ready signal",
                    &err.to_string(),
                    &stderr_output,
                ));
            }
        };
        if signal.status != "ready" {
            kill_child(&mut child);
            return Err(format!("unexpected Chirp runner status: {}", signal.status));
        }

        let client = match reqwest::Client::builder().no_proxy().build() {
            Ok(client) => client,
            Err(err) => {
                kill_child(&mut child);
                return Err(format!("failed to build HTTP client: {err}"));
            }
        };

        std::thread::spawn(move || {
            let mut buf = String::new();
            while reader.read_line(&mut buf).unwrap_or(0) > 0 {
                buf.clear();
            }
        });

        let stderr_buf = Arc::new(Mutex::new(String::new()));
        if let Some(stderr) = stderr {
            let buf = stderr_buf.clone();
            std::thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if let Ok(mut stored) = buf.lock() {
                        if stored.len() < 8192 {
                            stored.push_str(&line);
                        }
                    }
                    line.clear();
                }
            });
        }

        Ok(Self {
            port: signal.port,
            child,
            client,
            stderr_buf,
        })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn recent_stderr(&self) -> String {
        self.stderr_buf
            .lock()
            .map(|buf| buf.trim().to_string())
            .unwrap_or_default()
    }

    fn kill(&mut self) {
        kill_child(&mut self.child);
    }
}

impl Drop for RunnerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

#[tauri::command]
pub async fn start_runner(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<RunnerInfo, String> {
    ensure_runner(&app, &state)
        .map(|base_url| RunnerInfo { base_url })
        .map_err(|err| {
            track_err(
                &app,
                analytics::events::ERROR_RUNNER_START_FAILED,
                err,
                "start_runner",
            )
        })
}

#[tauri::command]
pub async fn stop_runner(state: State<'_, RunnerState>) -> Result<(), String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(mut process) = guard.take() {
        process.kill();
    }
    Ok(())
}

pub fn stop_managed_runner(app: &tauri::AppHandle) {
    let state = app.state::<RunnerState>();
    match state.process.lock() {
        Ok(mut guard) => {
            if let Some(mut process) = guard.take() {
                process.kill();
            }
        }
        Err(_) => eprintln!("runner state lock poisoned during app shutdown"),
    };
}

#[tauri::command]
pub async fn get_runner_url(state: State<'_, RunnerState>) -> Result<Option<String>, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(process) = guard.as_mut() {
        if process.is_alive() {
            return Ok(Some(process.base_url()));
        }
    }
    *guard = None;
    Ok(None)
}

#[tauri::command]
pub async fn load_model(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: LoadModelRequest,
) -> Result<serde_json::Value, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let runtime = request.runtime.unwrap_or_else(|| {
        if request
            .voices_path
            .as_deref()
            .is_some_and(|path| !path.is_empty())
            || request.codec_path.is_empty()
        {
            "kokoro".to_string()
        } else {
            "qwen".to_string()
        }
    });
    let body = if runtime == "kokoro" {
        serde_json::json!({
            "runtime": "kokoro",
            "kokoro": {
                "model_path": request.model_path,
                "voices_path": request.voices_path.unwrap_or_default(),
                "espeak_data_path": request.espeak_data_path.unwrap_or_default(),
                "voice": request.voice.unwrap_or_else(|| "af_heart".to_string()),
                "language": "auto",
            },
        })
    } else {
        let mut body = serde_json::json!({
            "runtime": "qwen",
            "qwen": {
                "model_path": request.model_path,
                "codec_path": request.codec_path,
            },
        });
        if let Some(value) = request.max_tokens {
            body["qwen"]["max_tokens"] = serde_json::json!(value);
        }
        if let Some(value) = request.temperature {
            body["qwen"]["temperature"] = serde_json::json!(value);
        }
        if let Some(value) = request.top_k {
            body["qwen"]["top_k"] = serde_json::json!(value);
        }
        body
    };

    let response = client
        .post(format!("{base_url}/v1/models/load"))
        .json(&body)
        .send()
        .await
        .map_err(|err| {
            track_runner_err(
                &app,
                analytics::events::ERROR_MODEL_LOAD_FAILED,
                format!("failed to send model load request: {err}"),
                "load_model",
                &runtime,
            )
        })?;
    json_response(response).await.map_err(|err| {
        track_runner_err(
            &app,
            analytics::events::ERROR_MODEL_LOAD_FAILED,
            err,
            "load_model",
            &runtime,
        )
    })
}

#[tauri::command]
pub async fn get_languages(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let response = client
        .get(format!("{base_url}/v1/languages"))
        .send()
        .await
        .map_err(|err| {
            track_err(
                &app,
                analytics::events::ERROR_RUNNER_REQUEST_FAILED,
                format!("failed to send languages request: {err}"),
                "get_languages",
            )
        })?;
    if !response.status().is_success() {
        let err = response_error(response).await;
        return Err(track_err(
            &app,
            analytics::events::ERROR_RUNNER_REQUEST_FAILED,
            err,
            "get_languages",
        ));
    }
    let body = response.json::<LanguagesResponse>().await.map_err(|err| {
        track_err(
            &app,
            analytics::events::ERROR_RUNNER_REQUEST_FAILED,
            format!("failed to parse languages response: {err}"),
            "get_languages",
        )
    })?;
    Ok(body.languages)
}

#[tauri::command]
pub async fn synthesize(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: SpeechRequest,
) -> Result<String, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let output_path = request.output_path.unwrap_or_else(default_output_path);
    let language = request.language.unwrap_or_else(|| "auto".to_string());
    let voice = request.voice.unwrap_or_default();
    let has_voice_reference = request
        .voice_reference
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let body = serde_json::json!({
        "input": request.input,
        "voice_reference": request.voice_reference.unwrap_or_default(),
        "voice": voice,
        "response_format": "wav",
        "language": language,
    });
    let props = || {
        serde_json::json!({
            "operation": "synthesize",
            "voice": body["voice"].as_str().unwrap_or_default(),
            "language": body["language"].as_str().unwrap_or("auto"),
            "has_voice_reference": has_voice_reference,
        })
    };

    let response = client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
        .await
        .map_err(|err| {
            analytics::track_error(
                &app,
                analytics::events::ERROR_SYNTHESIS_FAILED,
                format!("failed to send speech request: {err}"),
                props(),
            )
        })?;
    if !response.status().is_success() {
        let err = response_error(response).await;
        return Err(analytics::track_error(
            &app,
            analytics::events::ERROR_SYNTHESIS_FAILED,
            err,
            props(),
        ));
    }

    let bytes = response.bytes().await.map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_SYNTHESIS_FAILED,
            format!("failed to read speech response: {err}"),
            props(),
        )
    })?;
    tauri::async_runtime::spawn_blocking({
        let output_path = output_path.clone();
        move || {
            std::fs::write(&output_path, bytes)
                .map_err(|err| format!("failed to write {output_path}: {err}"))
        }
    })
    .await
    .map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_SYNTHESIS_FAILED,
            format!("failed to join file write task: {err}"),
            props(),
        )
    })?
    .map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_SYNTHESIS_FAILED,
            err,
            props(),
        )
    })?;
    analytics::track_event_handle_with_props(
        &app,
        analytics::events::SYNTHESIS_COMPLETED,
        Some(props()),
    );
    Ok(output_path)
}

#[tauri::command]
pub async fn copy_audio_file(
    app: tauri::AppHandle,
    source_path: String,
    destination_path: String,
) -> Result<(), String> {
    if source_path.trim().is_empty() {
        let err = "source audio path is empty".to_string();
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }
    if destination_path.trim().is_empty() {
        let err = "destination audio path is empty".to_string();
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }

    let source = PathBuf::from(source_path);
    if !source.is_file() {
        let err = format!("source audio file does not exist: {}", source.display());
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }

    let destination = PathBuf::from(destination_path);
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|err| {
            let message = format!(
                "failed to create destination folder {}: {err}",
                parent.display()
            );
            track_err(
                &app,
                analytics::events::ERROR_FILE_OPERATION_FAILED,
                message,
                "copy_audio_file",
            )
        })?;
    }

    tokio::fs::copy(&source, &destination)
        .await
        .map_err(|err| {
            let message = format!(
                "failed to copy audio from {} to {}: {err}",
                source.display(),
                destination.display()
            );
            track_err(
                &app,
                analytics::events::ERROR_FILE_OPERATION_FAILED,
                message,
                "copy_audio_file",
            )
        })?;
    Ok(())
}

fn ensure_runner(app: &tauri::AppHandle, state: &State<'_, RunnerState>) -> Result<String, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(process) = guard.as_mut() {
        if process.is_alive() {
            return Ok(process.base_url());
        }
        let stderr = process.recent_stderr();
        if !stderr.is_empty() {
            eprintln!("Chirp runner exited; recent stderr: {stderr}");
        }
        *guard = None;
    }

    let binary_path = resolve_runner_binary(app)?;
    let process = RunnerProcess::spawn(app, &binary_path)?;
    let base_url = process.base_url();
    *guard = Some(process);
    Ok(base_url)
}

fn runner_client(
    app: &tauri::AppHandle,
    state: &State<'_, RunnerState>,
) -> Result<(reqwest::Client, String), String> {
    let base_url = ensure_runner(app, state)?;
    let guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    let process = guard
        .as_ref()
        .ok_or_else(|| "runner process missing after start".to_string())?;
    Ok((process.client.clone(), base_url))
}

fn track_err(app: &tauri::AppHandle, event: &str, error: String, operation: &str) -> String {
    analytics::track_error(
        app,
        event,
        error,
        serde_json::json!({"operation": operation}),
    )
}

fn track_runner_err(
    app: &tauri::AppHandle,
    event: &str,
    error: String,
    operation: &str,
    runtime: &str,
) -> String {
    analytics::track_error(
        app,
        event,
        error,
        serde_json::json!({"operation": operation, "runtime": runtime}),
    )
}

fn resolve_runner_binary(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let binary_name = runner_binary_name();

    if let Ok(resource_dir) = app.path().resource_dir() {
        let path = resource_dir.join(binary_name);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let path = exe_dir.join(binary_name);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        for base in [
            "/usr/lib/chirp",
            "/usr/lib/chirp/binaries",
            "/opt/chirp",
            "/opt/chirp/binaries",
        ] {
            let path = PathBuf::from(base).join(binary_name);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    if let Some(path) = find_in_path(binary_name) {
        return Ok(path);
    }

    Err("Chirp runner sidecar not found".to_string())
}

fn runner_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "chirp-runner.exe"
    } else {
        "chirp-runner"
    }
}

fn find_in_path(binary_name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|dir| dir.join(binary_name))
        .find(|path| path.exists() && path.is_file())
}

fn read_first_stderr_line(stderr: Option<impl std::io::Read>) -> String {
    let Some(stderr) = stderr else {
        return String::new();
    };
    let mut reader = std::io::BufReader::new(stderr);
    let mut buf = String::new();
    let _ = reader.read_line(&mut buf);
    buf.truncate(4096);
    buf
}

fn kill_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn format_runner_start_error(context: &str, error: &str, stderr: &str) -> String {
    if stderr.trim().is_empty() {
        format!("{context}: {error}")
    } else {
        format!("{context}: {error}\n\nrunner stderr: {}", stderr.trim())
    }
}

async fn json_response(response: reqwest::Response) -> Result<serde_json::Value, String> {
    if !response.status().is_success() {
        return Err(response_error(response).await);
    }
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|err| format!("failed to parse JSON response: {err}"))
}

async fn response_error(response: reqwest::Response) -> String {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if let Ok(parsed) = serde_json::from_str::<ErrorResponse>(&text) {
        format!(
            "runner request failed ({status}, {}): {}",
            parsed.error.code, parsed.error.message
        )
    } else if text.trim().is_empty() {
        format!("runner request failed ({status})")
    } else {
        format!("runner request failed ({status}): {text}")
    }
}

fn default_output_path() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    env::temp_dir()
        .join(format!("chirp-speech-{millis}.wav"))
        .as_os_str()
        .to_string_lossy()
        .into_owned()
}
