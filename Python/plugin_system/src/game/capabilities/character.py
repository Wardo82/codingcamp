from typing import Protocol, runtime_checkable

@runtime_checkable
class GameCharacter(Protocol):
    def salute(self) -> None: ...
