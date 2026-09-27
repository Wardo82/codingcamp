use anyhow::Result;
use clap::Parser;
use rusqlite;

use citydex::db::city::*;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Name of city
    city: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let conn = rusqlite::Connection::open("assets/db/cities.db")?;

    let city = get_city(&conn, &args.city)?;
    let qid = wikidata_qid(city.geoname_id)?;
    let title = wikipedia_title(&qid)?;
    let summary = wikipedia_summary(&title)?;
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
