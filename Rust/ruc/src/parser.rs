use anyhow::Result;
use std::fs;
use std::path::Path;

pub fn parse_str(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path)?;

    let mut lines = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        if line.contains("-->") {
            continue;
        }

        if line.parse::<u32>().is_ok() {
            continue;
        }

        lines.push(line.to_string());
    }

    Ok(lines)
}


pub fn load_season_text(season_dir: &Path) -> Result<Vec<String>> {
    let mut subtitles = Vec::new();

    for entry in fs::read_dir(season_dir)? {
        let path = entry?.path();

        if path.extension().and_then(|s| s.to_str()) == Some("srt") {
            subtitles.extend(parse_str(&path)?);
        }
    }

    Ok(subtitles)
}
