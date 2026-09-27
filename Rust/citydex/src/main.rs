use dioxus::prelude::*;

pub mod city;
pub mod components;
pub mod db;

use city::City;
use components::eventcard::EventCard;
use components::herocard::HeroCard;
use db::hardcoded::load_cities;

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Load cities from backend
    let cities = load_cities();
    rsx! {
        document::Stylesheet { href: asset!("/assets/styles/base.css") }
        document::Stylesheet { href: asset!("/assets/styles/card.css") }
        document::Stylesheet { href: asset!("/assets/styles/layout.css") }
        document::Stylesheet { href: asset!("/assets/styles/variables.css")}

        div { id: "title",
            h1 { "Citydex! 📍" }
        }
        div {
            class: "page",

            for city in cities {
                HeroCard {
                    city: city.clone()
                }
                EventCard {
                    title: city.name,
                    location: city.country,
                    image: city.image,
                }
            }
        }
    }
}
