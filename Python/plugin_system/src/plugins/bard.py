""" Game extension that adds a bard character """

from dataclasses import dataclass

from game import factory

@dataclass
class Bard:
    name: str
    
    def salute(self) -> None:
        print(f"Bard: Hello, my name is {self.name}")

    def make_a_noise(self) -> None:
        print(f"WOOF: Bard {self.name}")

def initialize() -> None:
    factory.register("bard", Bard)
