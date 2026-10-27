"""One recursive model of a search space, shared by every specification type.

A binding (the value given to one ``@tunable`` argument) is any of:

- a ``Domain`` (``Float``, ``Int``, ``Choice``): one tunable value;
- a ``Component``: a constructor whose arguments may themselves be bindings;
- a ``Switch``: a choice whose options carry their own bindings (built as ``(option, params)``);
- a ``OneOf``: a choice between components, each with its own bindings (built as the chosen component);
- anything else: a fixed value, copied fresh on every build.

Keys are dotted paths. A component argument adds ``.<argument>``; a ``Switch`` or ``OneOf`` holds its chosen
option at its own key and adds ``.<option>.<param>`` for that option's parameters, so parameters of options not chosen
never appear in a candidate.
"""

from __future__ import annotations

import importlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .domains import Choice, Domain, Float, Int, SkillLibrary, SuggestionTrial, TextValue


class SpaceError(ValueError):
    pass


@dataclass(frozen=True)
class ParamInfo:
    """One tunable value of a space, for documentation, validation, plotting and serialization."""

    key: str
    type: str  # float | int | choice | text | skills (a Domain) | switch | one_of (a structural choice)
    default: Any
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: tuple[Any, ...] = ()
    active_when: tuple[str, str] | None = None  # (selector key, option)
    about: str | None = None  # text and skills: what the value is, for the model that rewrites it
    max_chars: int | None = None  # text: the longest value that is measured

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"key": self.key, "type": self.type, "default": self.default}
        if self.type in ("float", "int"):
            out.update(low=self.low, high=self.high, log=self.log, step=self.step)
        if self.choices:
            out["choices"] = list(self.choices)
        if self.about is not None:
            out["about"] = self.about
        if self.max_chars is not None:
            out["max_chars"] = self.max_chars
        out["active_when"] = {"key": self.active_when[0], "option": self.active_when[1]} if self.active_when else None
        return out


def _component_types() -> tuple[type, ...]:
    from .component import Component, OneOf
    from .switch import Switch

    return Component, Switch, OneOf


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def is_spec(value: Any) -> bool:
    return isinstance(value, (Domain, *_component_types()))


# ---- structure ----------------------------------------------------------------------------------------------------
def params(spec: Any, key: str, active_when: tuple[str, str] | None = None) -> list[ParamInfo]:
    """Every tunable value under ``spec``, including those of options not chosen by default."""
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Float):
        return [ParamInfo(key, "float", spec.default, spec.low, spec.high, spec.log, spec.step, (), active_when)]
    if isinstance(spec, Int):
        return [ParamInfo(key, "int", spec.default, spec.low, spec.high, spec.log, spec.step, (), active_when)]
    if isinstance(spec, Choice):
        return [ParamInfo(key, "choice", spec.default, choices=tuple(spec.choices), active_when=active_when)]
    if isinstance(spec, TextValue):
        return [
            ParamInfo(key, "text", spec.default, active_when=active_when, about=spec.about, max_chars=spec.max_chars)
        ]
    if isinstance(spec, SkillLibrary):
        return [ParamInfo(key, "skills", spec.default, active_when=active_when, about=spec.about)]
    if isinstance(spec, Component):
        out: list[ParamInfo] = []
        for name, arg in spec.arguments.items():
            out += params(arg, _join(key, name), active_when)
        return out
    if isinstance(spec, Switch):
        out = [ParamInfo(key, "switch", spec.default, choices=tuple(spec.branches), active_when=active_when)]
        for option, branch in spec.branches.items():
            for name, arg in branch.items():
                out += params(arg, f"{key}.{option}.{name}", (key, option))
        return out
    if isinstance(spec, OneOf):
        out = [ParamInfo(key, "one_of", spec.default_label, choices=spec.labels, active_when=active_when)]
        for label, option in zip(spec.labels, spec.options, strict=True):
            out += params(option, f"{key}.{label}", (key, label))
        return out
    return []


def references(spec: Any, key: str) -> list[tuple[Any, tuple[str, ...], bool]]:
    """Every declared object under ``spec`` that a search can refer to, with the variables it stands for and whether it
    is itself a variable: a value (a domain) and a choice (``OneOf``, ``Switch``) are their own variable; a component
    stands for every variable declared in it."""
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return [(spec, (key,), True)]
    if isinstance(spec, Component):
        out: list[tuple[Any, tuple[str, ...], bool]] = [(spec, tuple(p.key for p in params(spec, key)), False)]
        for name, arg in spec.arguments.items():
            out += references(arg, _join(key, name))
        return out
    if isinstance(spec, Switch):
        out = [(spec, (key,), True)]
        for option, branch in spec.branches.items():
            for name, arg in branch.items():
                out += references(arg, f"{key}.{option}.{name}")
        return out
    if isinstance(spec, OneOf):
        out = [(spec, (key,), True)]
        for label, option in zip(spec.labels, spec.options, strict=True):
            out += references(option, f"{key}.{label}")
        return out
    return []


def defaults(spec: Any, key: str) -> dict[str, Any]:
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return {key: spec.default}
    if isinstance(spec, Component):
        out: dict[str, Any] = {}
        for name, arg in spec.arguments.items():
            out.update(defaults(arg, _join(key, name)))
        return out
    if isinstance(spec, Switch):
        out = {key: spec.default}
        for name, arg in spec.branches[spec.default].items():
            out.update(defaults(arg, f"{key}.{spec.default}.{name}"))
        return out
    if isinstance(spec, OneOf):
        label = spec.default_label
        return {key: label, **defaults(spec.option(label), f"{key}.{label}")}
    return {}


def suggest(spec: Any, trial: SuggestionTrial, key: str) -> dict[str, Any]:
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return {key: spec.suggest(trial, key)}
    if isinstance(spec, Component):
        out: dict[str, Any] = {}
        for name, arg in spec.arguments.items():
            out.update(suggest(arg, trial, _join(key, name)))
        return out
    if isinstance(spec, Switch):
        option = trial.suggest_categorical(key, list(spec.branches))
        out = {key: option}
        for name, arg in spec.branches[option].items():
            out.update(suggest(arg, trial, f"{key}.{option}.{name}"))
        return out
    if isinstance(spec, OneOf):
        label = trial.suggest_categorical(key, list(spec.labels))
        return {key: label, **suggest(spec.option(label), trial, f"{key}.{label}")}
    return {}


def active(spec: Any, values: Mapping[str, Any], key: str) -> list[ParamInfo]:
    """The tunable values that a candidate's own choices make active."""
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return params(spec, key)
    if isinstance(spec, Component):
        out: list[ParamInfo] = []
        for name, arg in spec.arguments.items():
            out += active(arg, values, _join(key, name))
        return out
    if isinstance(spec, Switch):
        out = params(spec, key)[:1]
        option = values.get(key, spec.default)
        for name, arg in spec.branches.get(option, {}).items():
            out += active(arg, values, f"{key}.{option}.{name}")
        return out
    if isinstance(spec, OneOf):
        out = params(spec, key)[:1]
        label = values.get(key, spec.default_label)
        if label in spec.labels:
            out += active(spec.option(label), values, f"{key}.{label}")
        return out
    return []


def complete(spec: Any, values: Mapping[str, Any], key: str) -> dict[str, Any]:
    """``values`` with every missing active value filled from the defaults (following the given choices)."""
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return {key: values.get(key, spec.default)}
    if isinstance(spec, Component):
        out: dict[str, Any] = {}
        for name, arg in spec.arguments.items():
            out.update(complete(arg, values, _join(key, name)))
        return out
    if isinstance(spec, Switch):
        option = values.get(key, spec.default)
        out = {key: option}
        for name, arg in spec.branches.get(option, {}).items():
            out.update(complete(arg, values, f"{key}.{option}.{name}"))
        return out
    if isinstance(spec, OneOf):
        label = values.get(key, spec.default_label)
        if label not in spec.labels:
            return {key: label}
        return {key: label, **complete(spec.option(label), values, f"{key}.{label}")}
    return {}


def validate_value(info: ParamInfo, value: Any, artifacts: Any = None) -> None:
    """Raise when ``value`` is not one of ``info``'s values. ``artifacts``: the study's artifact store, under which a
    stored skill-library version (``skills/<hash>``) is found."""
    if info.type in ("float", "int"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SpaceError(f"{info.key}={value!r}: expected a number")
        if info.type == "int" and int(value) != value:
            raise SpaceError(f"{info.key}={value!r}: expected an integer")
        if not info.low <= value <= info.high:  # type: ignore[operator]
            raise SpaceError(f"{info.key}={value!r} is outside [{info.low}, {info.high}]")
    elif info.type == "text":
        if not isinstance(value, str):
            raise SpaceError(f"{info.key}={value!r}: expected text")
        if info.max_chars is not None and len(value) > info.max_chars:
            raise SpaceError(f"{info.key} is {len(value)} characters, over max_chars={info.max_chars}")
    elif info.type == "skills":
        if value is not None and not isinstance(value, str):
            raise SpaceError(f"{info.key}={value!r}: expected the path of a skill library, or None")
        if value:
            from .skills import check_library, resolve

            problem = check_library(resolve(value, artifacts) or value)
            if problem and artifacts is None and not os.path.isabs(value):
                problem += " (a value a study stored is relative to its artifact store: see Study.artifact_path)"
            if problem:
                raise SpaceError(f"{info.key}: {problem}")
    elif value not in info.choices:
        raise SpaceError(f"{info.key}={value!r} is not one of {list(info.choices)}")


def json_problem(value: Any) -> str | None:
    """Why ``value`` cannot be a candidate's value, if it cannot: a record holds every value as plain JSON, and reads it
    back as the same value (so a value is never written as its ``repr``, and a candidate can be rebuilt from its
    record)."""
    try:
        text = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"it is not JSON ({exc})"
    if json.loads(text) != value:
        return "it does not read back from JSON as itself (a tuple reads back as a list, other keys as strings)"
    return None


JSON_HINT = (
    "give numbers, strings, booleans, None, or lists and dicts of them, and build objects with component() or OneOf()"
)


def check_json(infos: Iterable[ParamInfo]) -> None:
    """Raise, naming the variable, when a declared default or option is not plain JSON (``json_problem``)."""
    for info in infos:
        problem = json_problem(info.default)
        if problem:
            raise SpaceError(f"{info.key}: its default {info.default!r} cannot be recorded: {problem}; {JSON_HINT}")
        for option in info.choices:
            problem = json_problem(option)
            if problem:
                raise SpaceError(f"{info.key}: its option {option!r} cannot be recorded: {problem}; {JSON_HINT}")


def build(spec: Any, values: Mapping[str, Any], key: str) -> Any:
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Domain):
        return values[key]
    if isinstance(spec, Component):
        return spec.constructor(**{name: build(arg, values, _join(key, name)) for name, arg in spec.arguments.items()})
    if isinstance(spec, Switch):
        option = values[key]
        return option, {
            name: build(arg, values, f"{key}.{option}.{name}") for name, arg in spec.branches[option].items()
        }
    if isinstance(spec, OneOf):
        label = values[key]
        return build(spec.option(label), values, f"{key}.{label}")
    return deepcopy(spec)


def distributions(spec: Any, values: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Optuna distributions of the active values (to record an observation the optimizer did not propose)."""
    import optuna.distributions as d

    out: dict[str, Any] = {}
    for info in active(spec, values, key):
        if info.type == "float":
            out[info.key] = d.FloatDistribution(info.low, info.high, log=info.log, step=info.step)
        elif info.type == "int":
            out[info.key] = d.IntDistribution(int(info.low), int(info.high), log=info.log, step=int(info.step or 1))
        elif info.type in ("text", "skills"):
            continue  # written by a model, not sampled
        else:
            out[info.key] = d.CategoricalDistribution(list(info.choices))
    return out


# ---- serialization ------------------------------------------------------------------------------------------------
def import_path(obj: Any) -> str:
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None)
    if not module or not name or module == "__main__" or "<locals>" in name:
        raise SpaceError(f"{obj!r} has no stable import path (define it at module level in an importable module)")
    return f"{module}:{name}"


def resolve(path: str) -> Any:
    module, _, name = path.partition(":")
    target: Any = importlib.import_module(module)
    for part in name.split("."):
        target = getattr(target, part)
    return target


def to_dict(spec: Any) -> Any:
    Component, Switch, OneOf = _component_types()
    if isinstance(spec, Float):
        return {
            "type": "float",
            "low": spec.low,
            "high": spec.high,
            "default": spec.default,
            **({"log": True} if spec.log else {}),
            **({"step": spec.step} if spec.step is not None else {}),
        }
    if isinstance(spec, Int):
        return {
            "type": "int",
            "low": spec.low,
            "high": spec.high,
            "default": spec.default,
            **({"log": True} if spec.log else {}),
            **({"step": spec.step} if spec.step != 1 else {}),
        }
    if isinstance(spec, Choice):
        return {"type": "choice", "choices": list(spec.choices), "default": spec.default}
    if isinstance(spec, TextValue):
        return {
            "type": "text",
            "default": spec.default,
            **({"about": spec.about} if spec.about is not None else {}),
            **({"max_chars": spec.max_chars} if spec.max_chars is not None else {}),
        }
    if isinstance(spec, SkillLibrary):
        return {"type": "skills", "default": spec.default, **({"about": spec.about} if spec.about is not None else {})}
    if isinstance(spec, Component):
        return {
            "type": "component",
            "constructor": import_path(spec.constructor),
            "arguments": {name: to_dict(arg) for name, arg in spec.arguments.items()},
        }
    if isinstance(spec, Switch):
        return {
            "type": "switch",
            "default": spec.default,
            "branches": {o: {n: to_dict(a) for n, a in b.items()} for o, b in spec.branches.items()},
        }
    if isinstance(spec, OneOf):
        return {
            "type": "one_of",
            "default": spec.default_label,
            "options": {label: to_dict(option) for label, option in zip(spec.labels, spec.options, strict=True)},
        }
    return {"type": "fixed", "value": _json(spec)}


def from_dict(data: Any) -> Any:
    from .component import Component, OneOf
    from .switch import Switch

    kind = data.get("type") if isinstance(data, dict) else None
    if kind == "float":
        return Float(
            data["low"], data["high"], default=data["default"], step=data.get("step"), log=data.get("log", False)
        )
    if kind == "int":
        return Int(
            data["low"], data["high"], default=data["default"], step=data.get("step", 1), log=data.get("log", False)
        )
    if kind == "choice":
        return Choice(data["choices"], default=data["default"])
    if kind == "text":
        return TextValue(default=data["default"], about=data.get("about"), max_chars=data.get("max_chars"))
    if kind == "skills":
        return SkillLibrary(default=data.get("default"), about=data.get("about"))
    if kind == "component":
        return Component(resolve(data["constructor"]), {n: from_dict(a) for n, a in data["arguments"].items()})
    if kind == "switch":
        return Switch(
            {o: {n: from_dict(a) for n, a in b.items()} for o, b in data["branches"].items()}, default=data["default"]
        )
    if kind == "one_of":
        labels = list(data["options"])
        return OneOf(*(from_dict(data["options"][label]) for label in labels), labels=labels, default=data["default"])
    if kind == "fixed":
        return data["value"]
    raise SpaceError(f"cannot read a search-space entry of type {kind!r}")


def _json(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        raise SpaceError(f"fixed value {value!r} cannot be serialized; give it as a component or plain JSON") from None
    return value
