from typing import Protocol, runtime_checkable

@runtime_checkable
class Attacker(Protocol):
    def attack(self) -> None: ...
