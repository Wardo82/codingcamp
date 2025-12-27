from __future__ import annotations

from dataclasses import KW_ONLY, field
import functools
import typing as t

import rio

from .. import components as comps

@rio.page(
    name="Main",
    url_segment="",
)
class HomePage(rio.Component):
    counter: int = 0

    def countup(self) -> None:
        self.counter = self.counter + 1

    def on_reset(self) -> None:
        self.counter = 0

    def build(self) -> rio.Component:
        message = f"Main page count: {self.counter}"

        # Arrange all components in a grid
        return rio.Column(
            rio.Text(message, style="heading1"),
            rio.Row(
                rio.Button(
                    "Countup",
                    icon="material/smart_display",
                    style="colored-text",
                    on_press=self.countup,
                ),
                rio.Button(
                    "Reset",
                    icon="material/refresh",
                    style="colored-text",
                    on_press=self.on_reset,
                ),
            ),
            spacing=2,
            margin=2,
            align_x=0.5,
            align_y=0.0,
        )
