use dioxus::prelude::*;

#[component]
pub fn LandingPage() -> Element {
    rsx! {
        div {
            class: "landing-page",

            h1 {
                class: "logo",
                "Citydex"
            }


            div {
                class: "search-container",

                input {
                    class: "search-input",
                    r#type: "text",
                    placeholder: "Search for a city...",

                }
            }
        }
    }
}
