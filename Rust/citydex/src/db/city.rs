use anyhow::Result;
use rusqlite;
use serde::Deserialize;
use std::collections::HashMap;

use crate::city::City;

#[derive(Debug, Deserialize)]
struct EntityResponse {
    entities: HashMap<String, Entity>,
}

#[derive(Debug, Deserialize)]
struct Entity {
    sitelinks: SiteLinks,
}

#[derive(Debug, Deserialize)]
struct SiteLinks {
    enwiki: WikiSite,
}

#[derive(Debug, Deserialize)]
struct WikiSite {
    title: String,
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

pub fn get_city(name: &str) -> anyhow::Result<City> {
    // Rusqlite has a convenient query_row method for fetching a single row.
    let db = rusqlite::Connection::open("./assets/db/cities.db").expect("Failed to open database");

    let city = db.query_row(
        "
            SELECT
                geoname_id,
                name,
                country_code,
                population,
                latitude,
                longitude,
                timezone
            FROM cities
            WHERE name = ?1
            ",
        [name],
        |row| {
            Ok(City {
                geoname_id: row.get(0)?,
                name: row.get(1)?,
                country_code: row.get(2)?,
                country: row.get(2)?,
                population: row.get(3)?,
                latitude: row.get(4)?,
                longitude: row.get(5)?,
                timezone: row.get(6)?,
                image: String::new(),
            })
        },
    )?;

    Ok(city)
}

#[derive(Debug, Deserialize)]
struct WikipediaSummary {
    extract: String,
}

pub async fn wikipedia_summary(city: &str) -> Result<String> {
    let client = reqwest::Client::new();

    let response: WikipediaSummary = client
        .get(format!(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{city}"
        ))
        .header(reqwest::header::USER_AGENT, "city-explorer/0.1")
        .send()
        .await?
        .json()
        .await?;

    Ok(response.extract)
}

pub async fn wikipedia_title(qid: &str) -> Result<String> {
    let client = reqwest::Client::new();

    let url = format!("https://www.wikidata.org/wiki/Special:EntityData/{qid}.json");

    let response: EntityResponse = client
        .get(url)
        .header(reqwest::header::USER_AGENT, "city-explorer/0.1")
        .send()
        .await?
        .json()
        .await?;

    let entity = response
        .entities
        .get(qid)
        .ok_or_else(|| anyhow::anyhow!("QID not found"))?;

    Ok(entity.sitelinks.enwiki.title.clone())
}

pub async fn wikidata_qid(geoname_id: i64) -> Result<String> {
    let query = format!(
        r#"
        SELECT ?item WHERE {{
            ?item wdt:P1566 "{}"
        }}
        "#,
        geoname_id
    );

    let client = reqwest::Client::new();

    let response: SparqlResponse = client
        .get("https://query.wikidata.org/sparql")
        .header(reqwest::header::USER_AGENT, "city-explorer/0.1")
        .query(&[("query", query.as_str()), ("format", "json")])
        .send()
        .await?
        .json()
        .await?;

    let uri = &response.results.bindings[0].item.value;

    Ok(uri.rsplit('/').next().unwrap().to_string())
}
