use anyhow::{Context, Result};
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};
use zip::ZipArchive;

#[derive(Debug)]
pub struct SeasonStats {
    pub name: String,
    pub path: PathBuf,
    pub total_files: usize,
    pub subtitle_files: usize,
    pub total_size_bytes: u64,
}

pub fn extract_archive(zip_path: &Path) -> Result<SeasonStats> {
    let file = File::open(zip_path)
        .with_context(|| format!("Failed to open {}", zip_path.display()))?;

    let mut archive = ZipArchive::new(file)?;

    let season_name = zip_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let output_dir = zip_path.with_extension("");

    fs::create_dir_all(&output_dir)?;

    let mut total_files = 0;
    let mut subtitle_files = 0;
    let mut total_size = 0;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;

        let outpath = output_dir.join(entry.mangled_name());

        if entry.is_dir() {
            fs::create_dir_all(&outpath)?;
            continue;
        }

        if let Some(parent) = outpath.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut outfile = File::create(&outpath)?;

        io::copy(&mut entry, &mut outfile)?;

        total_files += 1;
        total_size += entry.size();

        if outpath
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("srt"))
            .unwrap_or(false)
        {
            subtitle_files += 1;
        }
    }

    Ok(SeasonStats {
        name: season_name,
        path: output_dir,
        total_files,
        subtitle_files,
        total_size_bytes: total_size,
    })
}
