use std::path::Path;
use std::process::{Command, Output};

fn run_config(root: &Path) -> Output {
    let config = root.join("config.toml");
    std::fs::write(
        &config,
        r#"[paths]
game_dir = "game"
cache_dir = "cache"
data_dir = "data"
test_titles = ["test-title"]
"#,
    )
    .unwrap();
    std::fs::create_dir(root.join("game")).unwrap();
    std::fs::create_dir(root.join("test-title")).unwrap();

    Command::new(env!("CARGO_BIN_EXE_victory"))
        .arg("config")
        .current_dir(root)
        .env("VICTORY_CONFIG", config)
        .env_remove("VICTORY_GAME_DIR")
        .env_remove("VICTORY_CACHE_DIR")
        .env_remove("VICTORY_DATA_DIR")
        .env_remove("VICTORY_TEST_TITLES")
        .output()
        .unwrap()
}

#[test]
fn config_succeeds_with_paths_outside_a_source_tree() {
    let root = tempfile::tempdir().unwrap();
    let output = run_config(root.path());
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("game_dir    : game"));
    assert!(stdout.contains("cache_dir   : cache"));
    assert!(stdout.contains("data_dir    : data"));
    assert!(stdout.contains("test_title  : test-title"));
    assert!(!stdout.contains("POLICY VIOLATION"));
}

#[test]
fn config_reports_all_path_violations_and_exits_unsuccessfully() {
    let root = tempfile::tempdir().unwrap();
    for name in ["cache", "data"] {
        let dir = root.path().join(name);
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join(".victory-root"), "").unwrap();
    }

    let output = run_config(root.path());
    assert!(!output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("cache_dir   : cache"));
    assert!(stdout.contains("data_dir    : data"));
    assert!(stdout.contains("POLICY VIOLATION: refusing to use cache for game-derived data"));
    assert!(stdout.contains("POLICY VIOLATION: refusing to use data for game-derived data"));
}
