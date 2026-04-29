use tauri::{App, AppHandle};
use tauri_plugin_aptabase::EventTracker;

pub const APTABASE_APP_KEY: &str = match option_env!("APTABASE_APP_KEY") {
    Some(value) => value,
    None => "",
};

pub const APTABASE_BASE_URL: &str = match option_env!("APTABASE_BASE_URL") {
    Some(value) => value,
    None => "",
};

pub mod events {
    pub const APP_STARTED: &str = "app_started";
    pub const MODEL_DOWNLOAD_STARTED: &str = "model_download_started";
    pub const MODEL_DOWNLOAD_SUCCEEDED: &str = "model_download_succeeded";
    pub const MODEL_DOWNLOAD_FAILED: &str = "model_download_failed";
    pub const RUNNER_START_FAILED: &str = "runner_start_failed";
    pub const SPEECH_STARTED: &str = "speech_started";
    pub const SPEECH_SUCCEEDED: &str = "speech_succeeded";
    pub const SPEECH_FAILED: &str = "speech_failed";
}

pub fn is_configured() -> bool {
    !APTABASE_APP_KEY.is_empty() && !APTABASE_BASE_URL.is_empty()
}

pub fn track_event(app: &App, event_name: &str) {
    track_event_handle_with_props(app.handle(), event_name, None);
}

pub fn track_event_handle(app: &AppHandle, event_name: &str) {
    track_event_handle_with_props(app, event_name, None);
}

pub fn track_event_handle_with_props(
    app: &AppHandle,
    event_name: &str,
    props: Option<serde_json::Value>,
) {
    if !is_configured() {
        return;
    }
    if let Err(error) = app.track_event(event_name, props) {
        eprintln!("analytics track_event failed for {event_name}: {error}");
    }
}
