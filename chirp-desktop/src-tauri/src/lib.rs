mod files;
mod model;
mod runner;
mod voices;

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(runner::RunnerState {
            process: std::sync::Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            files::reveal_path,
            model::get_model_bundle,
            model::download_model_bundle,
            runner::start_runner,
            runner::stop_runner,
            runner::get_runner_url,
            runner::load_model,
            runner::get_languages,
            runner::synthesize,
            runner::copy_audio_file,
            voices::download_voice,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| match event {
        tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
            runner::stop_managed_runner(app);
        }
        _ => {}
    });
}
