use dioxus::prelude::*;

pub mod city;
pub mod components;
pub mod db;

use components::citycard::CityCard;
use components::eventcard::EventCard;
use components::herocard::HeroCard;
use components::landing::LandingPage;

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
        document::Stylesheet { href: asset!("/assets/styles/popup.css") }
        document::Stylesheet { href: asset!("/assets/styles/variables.css")}
        document::Stylesheet { href: asset!("/assets/styles/landing.css") }
        document::Stylesheet { href: asset!("/assets/styles/citycard.css")}

        LandingPage {  }

        CityCard {
            city: cities[0].clone(),
            description: "Heelp".to_string()
        }

        div {
            class: "page",

            for city in cities {
                HeroCard {
                    title: city.name.clone(),
                    location: city.country.clone(),
                    image: city.image.clone(),
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
