from __future__ import annotations

from dataclasses import KW_ONLY, field
import functools
import typing as t

import rio

from .. import components as comps

@rio.page(
    name="Tic Tac Toe",
    url_segment="tictac",
)
class TicTacToePage(rio.Component):
    fields: list[t.Literal["X", "O", ""]] = [""] * 9
    player: t.Literal["X", "O"] = "X"
    winner: t.Literal["X", "O", "draw"] | None = None
    # If there is a winner, these are the indices of the fields which made them win
    winning_indices: set[int] = set()

    def find_winner(self) -> None:
        """
        Look if there is a winner on the board, and stores it in the component's state.
        Also updates the winning indices accordingly.
        """
        winning_combinations = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]

        # Look for winners
        for combination in winning_combinations:
            values = [self.fields[i] for i in combination]
            if values.count("X") == 3:
                self.winner = "X"
                self.winning_indices = set(combination)
                return

            if values.count("O") == 3:
                self.winner = "O"
                self.winning_indices = set(combination)
                return

        # If no fields are empty, it's a draw
        if "" not in self.fields:
            self.winner = "draw"
            return

    def on_press(self, index: int) -> None:
        """ This function reacts to presses on the fields, and updates the game
        state accordingly.
        """

        if self.winner:
            return

        # Set the field on that index
        self.fields[index] = self.player
        # Next player
        self.player = "X" if self.player == "O" else "O"
        # See if there is a winner
        self.find_winner()

    def on_reset(self) -> None:
        """
        Reset the game to its initial state.
        """
        self.fields = [""] * 9
        self.player = "X"
        self.winner = None
        self.winning_indices = set()

    def build(self) -> rio.Component:
        # Spawn components for the fields
        field_components: list[rio.Component] = []

        for index, field in enumerate(self.fields):
            field_components.append(
                comps.Field(
                    value=field,
                    dim=self.winner is not None
                        and index not in self.winning_indices,
                    on_press=functools.partial(self.on_press, index),
                )
            )

        # come up with a status message
        if self.winner in ("X", "O"):
            message = f"{self.winner} won!"
        elif self.winner == "draw":
            message = "It's a draw!"
        else:
            message = f"{self.player}'s turn"

        # Arrange all components in a grid
        return rio.Column(
            rio.Text(message, style="heading1"),
            rio.Grid(
                field_components[0:3],
                field_components[3:6],
                field_components[6:9],
                row_spacing=1,
                column_spacing=1,
                align_x=0.5,
                ),
            rio.Button(
                "Reset",
                icon="material/refresh",
                style="colored-text",
                on_press=self.on_reset,
            ),
            spacing=2,
            margin=2,
            align_x=0.5,
            align_y=0.0,
        )
