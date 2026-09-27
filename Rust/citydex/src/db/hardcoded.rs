use dioxus::prelude::*;

use crate::city::City;

pub fn load_cities() -> Vec<City> {
    vec![
        City {
            geoname_id: 123456,
            name: "Tokyo".to_string(),
            country: "Japan".to_string(),
            image: asset!("/assets/images/tokyo.jpg").to_string(),
            latitude: 123.0,
            longitude: 123.0,
            country_code: "".to_string(),
            population: 10,
            timezone: "".to_string(),
        },
        City {
            geoname_id: 123456,
            name: "New York".to_string(),
            country: "USA".to_string(),
            image: asset!("/assets/images/new_york.jpg").to_string(),
            latitude: 123.0,
            longitude: 123.0,
            country_code: "".to_string(),
            population: 10,
            timezone: "".to_string(),
        },
        City {
            geoname_id: 123456,
            name: "Tirana".to_string(),
            country: "Albania".to_string(),
            image: asset!("/assets/images/tirana.jpg").to_string(),
            latitude: 123.0,
            longitude: 123.0,
            country_code: "".to_string(),
            population: 10,
            timezone: "".to_string(),
        },
        City {
            geoname_id: 123456,
            name: "Caracas".to_string(),
            country: "Venezuela".to_string(),
            image: asset!("/assets/images/caracas.jpg").to_string(),
            latitude: 123.0,
            longitude: 123.0,
            country_code: "".to_string(),
            population: 10,
            timezone: "".to_string(),
        },
        City {
            geoname_id: 123456,
            name: "Barcelona".to_string(),
            country: "España".to_string(),
            image: asset!("/assets/images/barcelona.jpg").to_string(),
            latitude: 123.0,
            longitude: 123.0,
            country_code: "".to_string(),
            population: 10,
            timezone: "".to_string(),
        },
    ]
}
