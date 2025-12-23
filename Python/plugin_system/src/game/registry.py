from typing import runtime_checkable

from game.capabilities.attacker import Attacker
from game.capabilities.character import GameCharacter
from game.capabilities.noisemaker import NoiseMaker

class CharacterRegistry:
    def __init__(self):
        self.noise_makers = []
        self.saluters = []
        self.attackers = []

    def register(self, character):
        if isinstance(character, NoiseMaker):
            self.noise_makers.append(character)
        if isinstance(character, GameCharacter):
            self.saluters.append(character)
        if isinstance(character, Attacker):
            self.attackers.append(character)
