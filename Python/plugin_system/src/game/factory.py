from typing import Callable, Any

from game.capabilities import GameCharacter

character_creation_funcs: dict[str, Callable[..., GameCharacter]] = {}

def register(character_type: str, creation_func: Callable[..., GameCharacter]):
    """ Registers a new character type """
    character_creation_funcs[character_type] = creation_func

def unregister(character_type: str):
    """ Unregister a character """
    character_creation_funcs.pop(character_type, None)

def create(arguments: dict[str, Any]) -> GameCharacter:
    """ Create a game character of a specific type, given a dictionary of arguments. """
    args_copy = arguments.copy()
    character_type = args_copy.pop("type")
    try:
        creation_func = character_creation_funcs[character_type]
        return creation_func(**args_copy)
    except KeyError:
        raise ValueError(f"Unknown character type {character_type!r}") from None
