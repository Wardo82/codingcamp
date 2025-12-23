""" Game extension that adds a bard character """

from dataclasses import dataclass

from game import factory

@dataclass
class Wizard:

    name: str

    def attack(self) -> None:
        print(f"Wizard[{self.name}]: Expelium!!! -100pts")

    def make_a_noise(self) -> None:
        print(f"Wizard[{self.name}]: You shall not pass!!!!!")

def initialize() -> None:
    factory.register("wizard", Wizard)
