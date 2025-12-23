from typing import Protocol, runtime_checkable

@runtime_checkable
class NoiseMaker(Protocol):

    def make_a_noise(self) -> None: ...
