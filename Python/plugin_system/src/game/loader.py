""" A simple plugin loader. """

import importlib

class PluginInterface:
    """ A plugin has a single function called initialized. """

    @staticmethod
    def initialize() -> None:
        """ Initializes and registers the plugin """

def import_module(name: str) -> PluginInterface:
    return importlib.import_module(name) # type: ignore

def load_plugins(plugins: list[str]):
    """ Load the plugins defined in the plugins list. """
    for plugin_name in plugins:
        plugin = import_module(plugin_name)
        plugin.initialize()
