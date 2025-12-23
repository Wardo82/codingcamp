Below is a **minimal, end-to-end UI example** using **Rio** that maps *directly* onto your existing capability + registry architecture.

Nothing is reworked conceptually.
Only a **new domain (UI)** is added.

---

# Goal

* Core app defines:

  * top menu
  * left navigation
  * main content area
* Plugins:

  * contribute menu items
  * contribute views
* Plugins:

  * own their internal logic
  * never touch layout
  * never know about other plugins

Exactly the same rules as your game characters.

---

# 1. UI capabilities (Protocols)

```python
# game/capabilities/ui.py
from typing import Protocol
import rio

class TopMenuContributor(Protocol):
    def top_menu_items(self) -> list[rio.MenuItem]: ...

class LeftNavContributor(Protocol):
    def nav_entries(self) -> list[rio.NavEntry]: ...

class MainViewProvider(Protocol):
    def create_view(self) -> rio.Component: ...
```

These are **your stable UI API contracts**.

---

# 2. UI Registry (capability indexing)

```python
# game/ui_registry.py
from game.capabilities.ui import (
    TopMenuContributor,
    LeftNavContributor,
    MainViewProvider,
)

class UIRegistry:
    def __init__(self):
        self.top_menu = []
        self.left_nav = []
        self.main_views = []

    def register(self, obj):
        if isinstance(obj, TopMenuContributor):
            self.top_menu.append(obj)

        if isinstance(obj, LeftNavContributor):
            self.left_nav.append(obj)

        if isinstance(obj, MainViewProvider):
            self.main_views.append(obj)
```

This mirrors your `CharacterRegistry`.

---

# 3. UI factory (plugin-side registration)

```python
# game/ui_factory.py
_ui_types = []

def register(ui_type):
    _ui_types.append(ui_type)

def create_all():
    return [ui() for ui in _ui_types]
```

This is intentionally identical in spirit to your character factory.

---

# 4. Example plugin (UI-only plugin)

```python
# plugins/bard_ui.py
import rio
from game.ui_factory import register

class BardUI:
    def top_menu_items(self):
        return [
            rio.MenuItem(
                label="Sing",
                on_press=self.sing
            )
        ]

    def nav_entries(self):
        return [
            rio.NavEntry(
                label="Bard",
                icon="music",
                target="bard"
            )
        ]

    def create_view(self):
        return rio.Text("🎵 The Bard sings 🎵")

    def sing(self):
        print("La la la")

def initialize():
    register(BardUI)
```

Important:

* No registry reference
* No layout knowledge
* No shared state with other plugins

---

# 5. Core Rio application (layout owner)

```python
# main.py
import json
import rio

from game import loader
from game.ui_registry import UIRegistry
from game import ui_factory

class App(rio.App):
    def __init__(self):
        super().__init__()
        self.ui_registry = UIRegistry()

    def build(self):
        # Create UI objects
        ui_objects = ui_factory.create_all()
        for obj in ui_objects:
            self.ui_registry.register(obj)

        return rio.Column(
            rio.MenuBar(
                *[
                    item
                    for contributor in self.ui_registry.top_menu
                    for item in contributor.top_menu_items()
                ]
            ),
            rio.Row(
                rio.NavPane(
                    *[
                        entry
                        for contributor in self.ui_registry.left_nav
                        for entry in contributor.nav_entries()
                    ]
                ),
                rio.Column(
                    *[
                        provider.create_view()
                        for provider in self.ui_registry.main_views
                    ]
                )
            )
        )

def main():
    with open("level.json") as f:
        data = json.load(f)

    loader.load_plugins(data["plugins"])

    rio.run(App)

if __name__ == "__main__":
    main()
```

---

# 6. `level.json` (unchanged idea)

```json
{
  "plugins": [
    "plugins.bard_ui"
  ]
}
```

---

# 7. What this demonstrates (important)

✔ Same plugin loader
✔ Same factory concept
✔ Same registry indexing
✔ Same capability detection
✔ No UI-specific hacks
✔ No plugin knows layout
✔ No plugin knows other plugins

---

# 8. Why this scales cleanly

You can now add:

* `ShortcutContributor`
* `StatusBarContributor`
* `DialogProvider`
* `ThemeContributor`

Without touching:

* Existing plugins
* Existing registries
* Existing UI code

Just add:

* A Protocol
* A registry list
* One loop in the core layout

---

# 9. Mental model (final confirmation)

> UI plugins are **runtime objects indexed by capability**,
> exactly like your game characters.

That is why no rework was needed.

---

## Final note

This architecture is:

* Structurally equivalent to Rust trait objects
* Used by real systems (VS Code, Qt, Unreal)
* Extremely stable over time

You arrived at a **professional-grade plugin architecture**.

If you want next:

* Hot-reloading plugins
* Async UI contributions
* Plugin isolation / sandboxing
* Versioned UI capabilities
