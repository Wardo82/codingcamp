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

#[derive(Debug, Deserialize)]
struct SparqlResponse {
    results: Results,
}

#[derive(Debug, Deserialize)]
struct Results {
    bindings: Vec<Binding>,
}

#[derive(Debug, Deserialize)]
struct Binding {
    item: Value,
}

#[derive(Debug, Deserialize)]
struct Value {
    value: String,
}

#[derive(Debug, Deserialize)]
struct WikipediaSummary {
    extract: String,
}

fn wikipedia_summary(city: &str) -> Result<String> {
    let client = reqwest::blocking::Client::new();

    let response: WikipediaSummary = client
        .get(format!(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
            city
        ))
        .header(
            reqwest::header::USER_AGENT,
            "city-explorer/0.1 (https://example.com)",
        )
        .send()?
        .json()?;

    Ok(response.extract)
}

fn wikidata_qid(geoname_id: i64) -> Result<String> {
    let query = format!(
        r#"
        SELECT ?item WHERE {{
            ?item wdt:P1566 "{}"
        }}
        "#,
        geoname_id
    );

    let client = reqwest::blocking::Client::new();

    let response: SparqlResponse = client
        .get("https://query.wikidata.org/sparql")
        .header(
            reqwest::header::USER_AGENT,
            "city-explorer/0.1 (https://example.com)",
        )
        .query(&[("query", query.as_str()), ("format", "json")])
        .send()?
        .json()?;

    let uri = &response.results.bindings[0].item.value;
    Ok(uri.rsplit('/').next().unwrap().to_string())
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Name of city
    city: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let conn = rusqlite::Connection::open("cities.db")?;

    let city = get_city(&conn, &args.city)?;
    let qid = wikidata_qid(city.geoname_id)?;
    // let title = wikidata_title(qid);
    let summary = wikipedia_summary(&city.name)?;
    println!("\n=== City Report ===\n");
    println!("Name       : {}", city.name);
    println!("Country    : {}", city.country_code);
    println!("Population : {}", city.population);
    println!("Latitude   : {}", city.latitude);
    println!("Longitude  : {}", city.longitude);
    println!("Timezone   : {}", city.timezone);
    println!("GeoNamesId : {}", city.geoname_id);
    println!("WikidataId : {}", qid);
    println!("Extract    : {}", summary);

    Ok(())
}
