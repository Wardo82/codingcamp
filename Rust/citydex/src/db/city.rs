use rusqlite;

use crate::city::City;

pub fn get_city(conn: &rusqlite::Connection, name: &str) -> anyhow::Result<City> {
    // Rusqlite has a convenient query_row method for fetching a single row.
    let city = conn.query_row(
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
                image: "".to_string(),
            })
        },
    )?;

    Ok(city)
}
