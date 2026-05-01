use std::time::Duration;

pub async fn watch_parent() {
    let parent = std::os::unix::process::parent_id();
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if std::os::unix::process::parent_id() != parent {
            std::process::exit(0);
        }
    }
}
