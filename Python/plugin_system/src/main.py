import json

from game import factory, loader
from game import registry

def main():
    """Creates game characters from a file containing a level definition"""

    # Read data form a JSON file:
    with open("./level.json") as file:
        data = json.load(file)

        character_registry = registry.CharacterRegistry()

        # Load the plugins
        loader.load_plugins(data["plugins"])

        # Create the characters
        characters = [factory.create(item) for item in data["characters"]]

        for character in characters:
            character_registry.register(character)

        for character in character_registry.noise_makers:
            character.make_a_noise()

        for character in character_registry.saluters:
            character.salute()

        for character in character_registry.attackers:
            character.attack()

if __name__ == '__main__':
    main()
