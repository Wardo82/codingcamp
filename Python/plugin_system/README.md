# Why the Plugin Architecture Gives You CRAZY Flexibility

Plugin architecture Tutorial


source: https://www.youtube.com/watch?v=iCE1bDoit9Q

Below is a structured way to think about plugins, grounded in common plugin-architecture patterns used in games and UI applications.

---

## 1. Separate **capabilities** from **plugins**

You are currently mixing two different concepts:

* **Game/domain capabilities** (make noise, attack, render, etc.)
* **Plugin lifecycle** (initialize, register things)

They should be separated.

### Rule of thumb

* **GameCharacter (or UI component)** defines *what can be done*.
* **Plugins** define *what gets added* to the system.

A plugin should **register implementations**, not *be* the implementation itself.

---

## 2. Do NOT keep extending a single `GameCharacter` interface

If you keep adding methods like:

```python
class GameCharacter(Protocol):
    def make_a_noise(self): ...
    def attack(self): ...
    def defend(self): ...
    def jump(self): ...
    def render(self): ...
```

You will quickly run into:

* Huge interfaces
* Many no-op implementations
* Tight coupling between unrelated features

This is a known anti-pattern.

---

## 3. Prefer **capability-based interfaces** (Protocols)

Split behavior into **small, orthogonal interfaces**.

```python
from typing import Protocol

class Noisy(Protocol):
    def make_a_noise(self) -> None: ...

class Attacker(Protocol):
    def attack(self) -> None: ...

class Renderer(Protocol):
    def render(self, ctx) -> None: ...
```

A character can implement **any combination** of these.

```python
class Orc:
    def make_a_noise(self): ...
    def attack(self): ...
```

The engine checks *capabilities*, not concrete types:

```python
if isinstance(character, Attacker):
    character.attack()
```

This pattern scales well for games and UI systems.

---

## 4. What should plugins do?

Plugins should **register new things**, not hardcode behavior into the engine.

### Example plugin responsibility

* Register new character types
* Register renderers
* Register menu items
* Register event handlers

### Plugin interface should be minimal

```python
class PluginInterface(Protocol):
    def initialize(self, registry) -> None: ...
```

Now plugins talk only to a **registry**, not the engine internals.

---

## 5. Introduce a **Registry / Extension Point** system

This is the missing piece in your design.

### Example: Character registry

```python
class CharacterRegistry:
    def __init__(self):
        self._constructors = {}

    def register(self, name: str, ctor):
        self._constructors[name] = ctor

    def create(self, data):
        return self._constructors[data["type"]](**data)
```

Your factory becomes a thin wrapper around this registry.

### Plugin usage

```python
def initialize(registry):
    registry.register("wizard", Wizard)
```

---

## 6. UI application: menus and layout

### Core application owns the **layout**

* Top menu
* Left navigation
* Bottom bar
* Main content area

Plugins should **contribute**, not control layout.

---

## 7. UI extension points (recommended model)

Define **explicit extension points**.

```python
class UIRegistry:
    def __init__(self):
        self.top_menu = []
        self.left_nav = []
        self.bottom_menu = []
        self.main_views = []
```

Each entry is a *descriptor*, not raw rendering code.

```python
@dataclass
class MenuItem:
    label: str
    action: Callable
    icon: str | None = None
```

---

## 8. Plugin-side UI contribution

```python
def initialize(registry):
    registry.top_menu.append(
        MenuItem(label="Export", action=export_data)
    )

    registry.main_views.append(
        ViewDescriptor(
            id="wizard_view",
            factory=create_wizard_view
        )
    )
```

The plugin:

* Does **not** know how the UI is laid out
* Only provides components and logic

The core app:

* Decides when and where to render

---

## 9. Rendering freedom vs control

> *“I want plugins to be free to render how they like”*

You still need **one of these contracts**:

### Option A — Framework-controlled rendering (recommended)

Plugins return UI objects compatible with your framework:

* Qt widgets
* React components
* HTML fragments
* Canvas nodes

### Option B — Callback-based rendering

Plugins expose:

```python
def render(ctx):
    ...
```

The engine provides `ctx` (canvas, layout, theme).

### Option C — Process isolation (advanced)

Plugins run separately and communicate via:

* IPC
* Events
* RPC

This is rare unless plugins are untrusted.

---

## 10. Summary of best practices for your case

### Game logic

* Use **capability interfaces**, not a monolithic base class
* Use registries for discovery
* Plugins register implementations

### Plugin system

* Minimal lifecycle (`initialize(registry)`)
* No direct imports from engine internals
* Explicit extension points

### UI application

* Core owns layout
* Plugins contribute via registries
* Rendering via agreed interfaces or factories

---

## 11. Mental model

> **Plugins don’t extend the engine.
> They extend *what the engine knows about*.**
