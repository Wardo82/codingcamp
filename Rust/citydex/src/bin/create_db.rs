use anyhow::Result;
use clap::Parser;
use reqwest;
use rusqlite;
use serde::Deserialize;
use std::{fs, path::PathBuf};

use citydex::city::City;
use citydex::db::city::*;

fn parse_city(line: &str) -> Result<City> {
    let fields: Vec<&str> = line.split('\t').collect();

    Ok(City {
        geoname_id: fields[0].parse()?,
        name: fields[1].to_string(),
        latitude: fields[4].parse()?,
        longitude: fields[5].parse()?,
        country_code: fields[8].to_string(),
        country: fields[8].to_string(),
        population: fields[14].parse()?,
        timezone: fields[17].to_string(),
        image: "".to_string(),
    })
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the .txt file
    path: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.path.exists() {
        anyhow::bail!("{} Does not exists", args.path.display());
    }

    let conn = rusqlite::Connection::open("cities.db")?;

    conn.execute(
        "
    CREATE TABLE IF NOT EXISTS cities (
        geoname_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        country_code TEXT NOT NULL,
        population INTEGER NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        timezone TEXT
    )
    ",
        [],
    )?;

    let content = fs::read_to_string(args.path)?;

    for line in content.lines() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        let city = parse_city(line)?;

        conn.execute(
            "
        INSERT OR REPLACE INTO cities
        (
            geoname_id,
            name,
            country_code,
            population,
            latitude,
            longitude,
            timezone
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ",
            (
                city.geoname_id,
                city.name,
                city.country_code,
                city.population,
                city.latitude,
                city.longitude,
                city.timezone,
            ),
        )?;
    }

    Ok(())
}
