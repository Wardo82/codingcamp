use crate::db::city::*;
use dioxus::prelude::*;

#[component]
pub fn HeroCard(title: String, location: String, image: String) -> Element {
    let mut show_popup = use_signal(|| false);
    let mut summary = use_signal(|| "Empty".to_string());
    let city_name = title.clone();
    let open_wiki = move |_| {
        let city_name = city_name.clone();

        async move {
            let result = async {
                let city = get_city(&city_name)?;
                let qid = wikidata_qid(city.geoname_id).await?;
                let title = wikipedia_title(&qid).await?;
                let summary_text = wikipedia_summary(&title).await?;

                Ok::<_, anyhow::Error>(summary_text)
            }
            .await;

            match result {
                Ok(text) => {
                    summary.set(text);
                    show_popup.set(true);
                }
                Err(err) => {
                    summary.set(format!("Error: {err}"));
                    show_popup.set(true);
                }
            }
            show_popup.set(true);
        }
    };

    rsx! {
        div {
            class: "card hero-card",
            style: "background-image: url('{image}')",

            div {
                class: "hero-card__overlay",

                h2 {
                    class: "hero-card__title",
                    "{title}"
                }

                p {
                    class: "hero-card__subtitle",
                    "{location}"
                }

                button {
                    class: "hero-card__button",

                    onclick: open_wiki,

                    "Explore"
                }
            }

            if show_popup() {
                div {
                    class: "popup",

                    h3 { "{title}" }

                    p {
                        "{summary}"
                    }

                    button {
                        onclick: move |_| {
                            show_popup.set(false);
                        },

                        "Close"
                    }
                }
            }
        }
    }
}
