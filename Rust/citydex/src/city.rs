#[derive(Debug, Clone, PartialEq)]
pub struct City {
    pub geoname_id: i64,
    pub name: String,
    pub country_code: String,
    pub country: String,
    pub latitude: f64,
    pub longitude: f64,
    pub population: i64,
    pub timezone: String,
    pub image: String,
}
