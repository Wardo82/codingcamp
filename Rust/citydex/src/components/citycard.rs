use dioxus::prelude::*;

use crate::city::City;

#[component]
pub fn CityCard(city: City, description: String) -> Element {
    rsx! {
        div {
            class: "city-card",

            div {
                class: "city-info",

                span {
                    class: "city-country",
                    "{city.country}"
                }

                h1 {
                    class: "city-title",
                    "{city.name}"
                }

                p {
                    class: "city-description",
                    "{description}"
                }

                button {
                    class: "explore-btn",
                    "Explore city"
                }
            }

            div {
                class: "city-image-container",

                img {
                    class: "city-image",
                    src: "{city.image}",
                }
            }
        }
    }
}
