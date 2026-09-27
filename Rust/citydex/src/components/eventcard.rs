use dioxus::prelude::*;

#[component]
pub fn EventCard(title: String, location: String, image: String) -> Element {
    rsx! {
        div {
            class: "card event-card",

            h2 {
                class: "event-card__title",
                "{title}"
            }

            p {
                class: "event-card__location",
                "{location}"
            }

            div {
                class: "event-card__image",

                img {
                    src: "{image}"
                }
            }

            div {
                class: "event-card__footer",

                div {
                    class: "badge",
                    "Interested"
                }

                span {
                    "View"
                }
            }
        }
    }
}
