""" Game extension that adds a Sorcerer character """

from dataclasses import dataclass

from game import factory

@dataclass
class Sorcerer:

    name: str

    def make_a_noise(self) -> None:
        print(f"Sorcerer {self.name}")

def initialize() -> None:
    factory.register("sorcerer", Sorcerer)
