# Citydex

Citydex is a Rust-powered city exploration application that combines geographic data, Wikipedia knowledge, and a modern Dioxus user interface into a lightweight, offline-first city discovery experience.

The project started as a learning vehicle for Rust and Dioxus, but its long-term vision is to become a world-wide city encyclopedia and travel companion, eventually integrating OpenStreetMap contributions and community-driven exploration.

## Vision

Every city in the world has a story.

Citydex aims to provide a fast, searchable catalog of cities enriched with:

* Geographic information
* Population and demographic data
* Historical and cultural context
* Photography
* Travel-oriented discovery
* OpenStreetMap integration

The application should remain lightweight, responsive, and capable of functioning with minimal network access.

---

# Current Architecture

```text
GeoNames
    ↓
Importer
    ↓
SQLite (cities.db)
    ↓
City Repository
    ↓
Dioxus UI
    ↓
Wikipedia / Wikidata Enrichment
```

---

# Data Sources

## GeoNames

Primary source of city records.

Current dataset:

```text
cities15000.txt
```

Contains cities with populations greater than approximately 15,000 inhabitants.

Imported fields:

* GeoNames ID
* Name
* Country Code
* Latitude
* Longitude
* Population
* Timezone

Stored in:

```text
assets/db/cities.db
```

---

## Wikidata

Used as the canonical identifier layer.

Lookup flow:

```text
GeoNames ID
    ↓
Wikidata P1566
    ↓
QID
```

Example:

```text
1850147
    ↓
Q1490
```

---

## Wikipedia

Used for human-readable content.

Current enrichment:

```text
Q1490
    ↓
Tokyo
    ↓
Summary API
```

Example endpoint:

```text
https://en.wikipedia.org/api/rest_v1/page/summary/Tokyo
```

Retrieved information:

* First paragraph
* Featured image URL (available but not yet integrated)

---

# Technology Stack

## Language

* Rust

## UI

* Dioxus

## Database

* SQLite
* Rusqlite

## Networking

* Reqwest
* Serde

## CLI Utilities

* Clap
* Anyhow

---

# Current Features

## City Importer

Reads:

```text
cities15000.txt
```

Produces:

```text
cities.db
```

Features:

* Creates database schema
* Parses GeoNames records
* Imports city data
* Idempotent imports via `INSERT OR REPLACE`

---

## City Search

Current capability:

```rust
get_city("Tokyo")
```

Returns:

```rust
City {
    geoname_id,
    name,
    country,
    population,
    latitude,
    longitude,
    timezone,
}
```

---

## Wikidata Integration

Current capability:

```rust
wikidata_qid(1850147)
```

Returns:

```text
Q1490
```

---

## Wikipedia Integration

Current capability:

```rust
wikipedia_summary("Tokyo")
```

Returns:

```text
Tokyo is the capital and most populous prefecture...
```

---

## Dioxus Frontend

Implemented:

* Hero cards
* CSS-based styling
* Dynamic image loading
* Signals
* Event handlers
* Async actions
* Modal / popup experimentation

Current user flow:

```text
Click "Explore"
    ↓
Lookup city
    ↓
Resolve Wikidata QID
    ↓
Resolve Wikipedia article
    ↓
Fetch summary
    ↓
Display popup
```

---

# Design Principles

## Offline First

The local SQLite database is considered the primary data source.

The application should remain useful without a backend server.

---

## Separation of Concerns

Model layer:

```text
City
```

should remain independent from:

```text
Dioxus Assets
Dioxus Signals
UI-specific Types
```

UI consumes data but does not own it.

---

## Progressive Enrichment

Base city information should always be available.

Additional layers are optional:

```text
GeoNames
    ↓
Wikidata
    ↓
Wikipedia
    ↓
Images
    ↓
OpenStreetMap
```

---

# Planned Database Evolution

Current schema:

```sql
cities
```

Future schema:

```sql
cities
city_metadata
countries
tags
favorites
```

Example:

```sql
city_metadata
-------------
geoname_id
wikidata_id
wikipedia_title
summary
image_url
last_updated
```

This allows caching external requests and reducing API traffic.

---

# Planned Features

## Search

* Search by city name
* Search by country
* Search by population range
* Fuzzy matching

Examples:

```text
Cities over 500k inhabitants
German cities
Coastal cities
```

---

## Rich City Profiles

Each city should eventually expose:

* Summary
* Population
* Coordinates
* Timezone
* Hero image
* Wikipedia article
* OpenStreetMap links

---

## Navigation

Planned pages:

### Home

Google-like search interface.

### Discover

Browse featured cities.

### City

Detailed city profile.

### Favorites

Personal collection.

---

## Theming

Planned support:

* Light theme
* Dark theme
* Regional themes

---

## OpenStreetMap Integration

Long-term objective.

Potential features:

* View OSM data
* Show nearby landmarks
* Surface missing map data
* Contribute edits
* OSM account integration

---

# Long-Term Roadmap

## Phase 1 — Foundation

* GeoNames importer
* SQLite storage
* Dioxus UI
* Wikipedia summaries

Status: In progress.

---

## Phase 2 — Discovery

* Search
* Filtering
* Country pages
* Rich city profiles
* Image integration

---

## Phase 3 — Personal Travel Companion

* Favorites
* Notes
* Visited cities
* Custom collections

---

## Phase 4 — OpenStreetMap Citizen Explorer

* OSM integration
* Missing-data discovery
* Community contributions
* Local exploration tools

---

# Current Learning Goals

This project is intentionally used to learn:

* Rust ownership and lifetimes
* Error handling with `Result`
* Async Rust
* SQLite and data modeling
* HTTP APIs
* Dioxus signals, events, and contexts
* Application architecture
* Offline-first software design

The project prioritizes understanding and maintainability over premature optimization or large-scale infrastructure.
