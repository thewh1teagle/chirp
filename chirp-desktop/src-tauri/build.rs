fn main() {
    println!("cargo:rerun-if-changed=../../.env");
    for key in ["APTABASE_APP_KEY", "APTABASE_BASE_URL"] {
        if let Ok(value) = std::env::var(key) {
            println!("cargo:rustc-env={}={}", key, value.trim());
        }
    }
    if let Ok(raw) = std::fs::read_to_string("../../.env") {
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                if key == "APTABASE_APP_KEY" || key == "APTABASE_BASE_URL" {
                    println!("cargo:rustc-env={}={}", key, value.trim());
                }
            }
        }
    }
    tauri_build::build()
}
