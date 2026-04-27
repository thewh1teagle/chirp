mod model;
mod runner;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(runner::RunnerState {
            process: std::sync::Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            model::get_model_bundle,
            model::download_model_bundle,
            runner::start_runner,
            runner::stop_runner,
            runner::get_runner_url,
            runner::load_model,
            runner::get_languages,
            runner::synthesize,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
