use dioxus::prelude::*;

use crate::city::City;

#[component]
pub fn HeroCard(city: City) -> Element {
    let open_wiki = move |_| {
        let a = 2;
    };

    rsx! {
        div {
            class: "card hero-card",

            style: "background-image: url('{city.image}')",

            div {
                class: "hero-card__overlay",

                h2 {
                    class: "hero-card__title",
                    "{city.name}"
                }

                p {
                    class: "hero-card__subtitle",
                    "{city.country}"
                }

                div {
                    class: "hero-card__meta",

                    span { "Direct flight" }
                }

                button {
                    class: "hero-card__button",
                    onclick: open_wiki,
                    "Explore"
                }
            }
        }
    }
}
