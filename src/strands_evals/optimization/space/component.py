"""Fresh component construction from fixed and tunable arguments, and choices between components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar

from . import space
from .domains import SuggestionTrial

T = TypeVar("T")


@dataclass(frozen=True)
class Component(Generic[T]):
    """Describe how to construct one fresh agent component. Arguments may be domains, components, ``Switch`` or
    ``OneOf`` choices, or fixed values (copied fresh on every build)."""

    constructor: Callable[..., T]
    arguments: Mapping[str, Any]

    def defaults(self, prefix: str) -> dict[str, Any]:
        return space.defaults(self, prefix)

    def suggest(self, trial: SuggestionTrial, prefix: str) -> dict[str, Any]:
        return space.suggest(self, trial, prefix)

    def build(self, candidate: Mapping[str, Any], prefix: str) -> T:
        return space.build(self, candidate, prefix)


def component(constructor: Callable[..., T], **arguments: Any) -> Component[T]:
    """Declare a fresh component; plain argument values remain fixed."""
    return Component(constructor=constructor, arguments=arguments)


@dataclass(frozen=True)
class OneOf:
    """A structural choice between components, each with its own tunable arguments (a strategy and its settings).

    ``OneOf(component(SlidingWindowConversationManager, window_size=Int(10, 80, default=40)),
    component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)), default=1)``

    The candidate holds the chosen option's label (the constructor's name unless ``labels`` are given) at the
    argument's key, and only the chosen option's arguments, under ``<key>.<label>.<argument>``.
    """

    options: tuple[Component[Any], ...]
    labels: tuple[str, ...] = ()
    default: int | str = 0

    def __init__(
        self, *options: Component[Any], default: int | str = 0, labels: list[str] | tuple[str, ...] | None = None
    ):
        if len(options) < 2:
            raise ValueError("OneOf needs at least two options")
        if not all(isinstance(o, Component) for o in options):
            raise TypeError("OneOf options are components: component(Constructor, **arguments)")
        names = (
            tuple(labels)
            if labels
            else tuple(getattr(o.constructor, "__name__", f"option{i}") for i, o in enumerate(options))
        )
        if len(names) != len(options) or len(set(names)) != len(names):
            raise ValueError("OneOf options need distinct labels; pass labels=[...] when constructors share a name")
        if isinstance(default, int):
            if not 0 <= default < len(options):
                raise ValueError("OneOf default index is out of range")
        elif default not in names:
            raise ValueError(f"OneOf default {default!r} is not one of {list(names)}")
        object.__setattr__(self, "options", tuple(options))
        object.__setattr__(self, "labels", names)
        object.__setattr__(self, "default", default)

    @property
    def default_label(self) -> str:
        return self.labels[self.default] if isinstance(self.default, int) else self.default

    def option(self, label: str) -> Component[Any]:
        return self.options[self.labels.index(label)]

    def defaults(self, name: str) -> dict[str, Any]:
        return space.defaults(self, name)

    def suggest(self, trial: SuggestionTrial, name: str) -> dict[str, Any]:
        return space.suggest(self, trial, name)

    def build(self, candidate: Mapping[str, Any], name: str) -> Any:
        return space.build(self, candidate, name)
