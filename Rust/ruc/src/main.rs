use anyhow::{Result};
use clap::Parser;
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};
use std::collections::HashMap;

use lindera::dictionary::load_dictionary;
use std::borrow::Cow;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::LinderaResult;

pub mod archive;
pub mod parser;


#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Folder containing season zip archives
    path: PathBuf,
    /// Season you want to analyze
    season: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.path.is_dir() {
        anyhow::bail!("{} is not a directory", args.path.display());        
    }
    
    let mut seasons = Vec::new();

    
    for entry in fs::read_dir(&args.path)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|e| e.to_str()) != Some("zip") {
            continue;
        }

        println!("Processing {}", path.display());

        let stats = archive::extract_archive(&path)?;
        seasons.push(stats);
    }

    println!();
    println!("=== Summary ===");
    seasons.sort_by(|a, b| a.name.cmp(&b.name));
    
    for season in &seasons {
        println!(
            "{} | episode: {} | files: {} | {:.2} MB",
            season.name,
            season.subtitle_files,
            season.total_files,
            season.total_size_bytes as f64 / 1024.0 / 1024.0
        )
    }

    let season_index = args.season as usize - 1;
    let season = seasons
        .get(season_index)
        .ok_or_else(|| anyhow::anyhow!("Season {} not found", args.season))?;
        
    println!("Analyzing {}", season.name);
    
    let subtitles = parser::load_season_text(&season.path)?;
    
    let dictionary = load_dictionary("embedded://ipadic")?;
    let segmenter = Segmenter::new(Mode::Normal, dictionary, None);

    let frequencies = count_tokens(&segmenter, &subtitles)?;

    let mut words: Vec<_> = frequencies.into_iter().collect();

    words.sort_by(|a, b| b.1.cmp(&a.1));

    for (word, count) in words.iter().take(50) {
        println!("{:>5} {}", count, word);
    }

    Ok(())
}

pub fn count_tokens(
    segmenter: &Segmenter,
    lines: &[String],
) -> LinderaResult<HashMap<String, usize>> {
    let mut freq = HashMap::new();

    for line in lines {
        let tokens = segmenter.segment(Cow::Borrowed(line))?;

        for token in tokens {
            *freq.entry(token.surface.to_string()).or_insert(0) += 1;
        }
    }

    Ok(freq)
}
