from __future__ import annotations

from dataclasses import KW_ONLY, field
import typing as t

import rio

from .. import components as comps


class NavigationBar(rio.Component):

    def navigate_home(self) -> None:
        self.session.navigate_to("/")


    def navigate_game(self) -> None:
        self.session.navigate_to("/tictac")

    def build(self) -> rio.Component:

        return rio.Row(
            rio.Button(
                "Home",
                icon="material/home",
                style="colored-text",
                on_press=self.navigate_home,
            ),
            rio.Button(
                "TicTac",
                icon="material/smart_display",
                style="colored-text",
                on_press=self.navigate_game,
            ),
        )
