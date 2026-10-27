"""A directly callable agent factory with a declarative search space."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import update_wrapper
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar

from . import skills, space
from .domains import SuggestionTrial

T = TypeVar("T")
_VALUES_ARGUMENT = "__strands_evals_values__"

# What a benchmark may provide to a factory. Each is passed by name, and only if the factory declares it, so the
# same factory runs on benchmarks that provide a sandbox, task tools, both, or neither.
RESOURCES = ("sandbox", "tools")


@dataclass(frozen=True)
class Values:
    """Serializable values for one point of a search space: one candidate's configuration."""

    values: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.values)


class TunableFactory(Generic[T]):
    """An agent factory with a search space: called with values (or none, for the defaults), it builds the agent."""

    def __init__(self, function: Callable[..., T], bindings: Mapping[str, Any]) -> None:
        self.function = function
        self.bindings = dict(bindings)
        self.parameters = inspect.signature(function).parameters
        update_wrapper(self, function)

    # ---- the space ------------------------------------------------------------------------------------------------
    @property
    def default_values(self) -> Values:
        values: dict[str, Any] = {}
        for name, spec in self.bindings.items():
            values.update(space.defaults(spec, name))
        return Values(values)

    def suggest(self, trial: SuggestionTrial) -> Values:
        values: dict[str, Any] = {}
        for name, spec in self.bindings.items():
            values.update(space.suggest(spec, trial, name))
        return Values(values)

    def params(self) -> list[space.ParamInfo]:
        """Every tunable value, including those under options not chosen by default."""
        return [info for name, spec in self.bindings.items() for info in space.params(spec, name)]

    def references(self) -> list[tuple[Any, tuple[str, ...], bool]]:
        """Every declared object a search can refer to (``Block(TPE(), window_size)``), with the variables it stands for
        and whether it is itself a variable (``space.references``)."""
        return [ref for name, spec in self.bindings.items() for ref in space.references(spec, name)]

    def active(self, values: Values | Mapping[str, Any]) -> list[space.ParamInfo]:
        values = _values(values)
        return [info for name, spec in self.bindings.items() for info in space.active(spec, values, name)]

    def complete(self, values: Values | Mapping[str, Any]) -> Values:
        """Fill missing values from the defaults, following the choices the values make."""
        values = _values(values)
        out: dict[str, Any] = {}
        for name, spec in self.bindings.items():
            out.update(space.complete(spec, values, name))
        return Values(out)

    def validate(self, values: Values | Mapping[str, Any], *, artifacts: str | Path | None = None) -> None:
        """Every value is active and inside its domain, and none is missing. ``artifacts``: the study's artifact store,
        where a stored value (a skill-library version) is found."""
        values = _values(values)
        active = {info.key: info for info in self.active(values)}
        unknown = sorted(set(values) - set(active))
        if unknown:
            raise space.SpaceError(f"not part of the search space (or of the options chosen): {unknown}")
        missing = sorted(set(active) - set(values))
        if missing:
            raise space.SpaceError(f"missing values: {missing}")
        for key, info in active.items():
            space.validate_value(info, values[key], artifacts)

    def resolve(self, values: Values | Mapping[str, Any], artifacts: str | Path | None) -> Values:
        """``values`` as the factory receives them: each skill library a stored version names (``skills/<hash>``,
        relative to the study's artifact store) made its absolute path under ``artifacts`` (``skills.resolve``)."""
        values = _values(values)
        for info in self.params():
            if info.type == "skills" and info.key in values:
                values[info.key] = skills.resolve(values[info.key], artifacts)
        return Values(values)

    def distributions(self, values: Values | Mapping[str, Any]) -> dict[str, Any]:
        values = _values(values)
        out: dict[str, Any] = {}
        for name, spec in self.bindings.items():
            out.update(space.distributions(spec, values, name))
        return out

    def space_dict(self) -> dict[str, Any]:
        return {name: space.to_dict(spec) for name, spec in self.bindings.items()}

    def with_space(self, bindings: Mapping[str, Any]) -> "TunableFactory[T]":
        """The same factory over another space (e.g. one read back from a serialized study)."""
        return TunableFactory(self.function, bindings)

    # ---- calling ---------------------------------------------------------------------------------------------------
    def bind(self, values: Values, *, artifacts: str | Path | None = None) -> dict[str, Any]:
        """Return serializable kwargs understood by this decorated factory (the values are validated). ``artifacts``:
        the study's artifact store, under which a stored value (a skill-library version) becomes the absolute path the
        factory receives."""
        full = self.complete(values)
        if artifacts is not None:
            full = self.resolve(full, artifacts)
        self.validate(full)
        return {_VALUES_ARGUMENT: full.to_dict()}

    @property
    def import_path(self) -> str:
        module = self.function.__module__
        if module == "__main__":
            source = inspect.getsourcefile(self.function)
            if source is None:
                raise ValueError("Cannot determine import path for tunable factory")
            module = Path(source).stem
        return f"{module}:{self.function.__name__}"

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        raw_values = kwargs.pop(_VALUES_ARGUMENT, None)
        values = self.complete(raw_values) if raw_values is not None else self.default_values
        # Benchmark resources the factory does not declare are dropped, so one factory fits many benchmarks.
        for resource in RESOURCES:
            if resource in kwargs and resource not in self.parameters:
                kwargs.pop(resource)
        for name, spec in self.bindings.items():
            if name in kwargs:
                continue
            kwargs[name] = space.build(spec, values.values, name)
        return self.function(*args, **kwargs)


def _values(values: Values | Mapping[str, Any]) -> dict[str, Any]:
    return dict(values.values if isinstance(values, Values) else values)


def tunable(**bindings: Any) -> Callable[[Callable[..., T]], TunableFactory[T]]:
    """Decorate a normal factory and bind search spaces to its arguments.

    Each binding is a domain (``Float``, ``Int``, ``Choice``), a ``component(...)``, a ``Switch`` or a ``OneOf``. With
    no bindings the factory is still usable by a study, which then evaluates the fixed agent.
    """

    def decorate(function: Callable[..., T]) -> TunableFactory[T]:
        signature = inspect.signature(function)
        unknown = sorted(set(bindings) - set(signature.parameters))
        if unknown:
            raise TypeError("@tunable bindings must name factory arguments; unknown: " + ", ".join(unknown))
        return TunableFactory(function, bindings)

    return decorate
