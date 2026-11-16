from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, TypeGuard

from .._entrypoint import EntryPointError

TemporaryWithout = dict[str, list[dict[str, Any]]]

_KEY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ITEM_FIELDS = ("index", "key", "name")
_SCALAR_TYPES = (str, int, float, bool, type(None))
_NAME_ATTRIBUTES = ("tool_name", "name", "__name__")
_TEXT_MIN_SIZE_CAP = 8000


class TemporaryVariableKind(str, Enum):
    TEXT = "text"
    CHOICE = "choice"
    SWITCH = "switch"


class _SpaceProblem(ValueError):
    pass


def temporary_text_value(default: str, about: str, optimization: ModuleType) -> Any:
    return optimization.TextValue(
        default=default, size_cap=max(_TEXT_MIN_SIZE_CAP, 2 * len(default)), edit_instructions=about or None
    )


@dataclass(frozen=True)
class TemporarySpaceVariable:
    key: str
    kind: TemporaryVariableKind
    option: str
    default: Any
    values: tuple[Any, ...] = ()
    about: str = ""
    item: dict[str, Any] | None = None

    def domain(self, current: dict[str, Any], optimization: ModuleType) -> Any:
        if self.kind is TemporaryVariableKind.TEXT:
            return temporary_text_value(self._text_default(current), self.about, optimization)
        if self.kind is TemporaryVariableKind.CHOICE:
            return optimization.OneOf(*self.values, default=self.default)
        return optimization.OneOf(True, False, default=self.default)

    def _text_default(self, current: dict[str, Any]) -> str:
        text = self.default if self.default is not None else (current.get(self.option) or "")
        if not isinstance(text, str):
            raise EntryPointError(
                f"--space variable {self.key} is text, but the agent's {self.option} is not text "
                f"({type(text).__name__}). Give the variable a default, or make it a choice, then try again."
            )
        return text


@dataclass(frozen=True)
class TemporarySpace:
    variables: tuple[TemporarySpaceVariable, ...]

    def bindings(self, current: dict[str, Any], optimization: ModuleType) -> dict[str, Any]:
        return {variable.key: variable.domain(current, optimization) for variable in self.variables}

    def overrides(self, values: dict[str, Any]) -> tuple[dict[str, Any], TemporaryWithout]:
        settings = {variable.option: values[variable.key] for variable in self.variables if variable.item is None}
        without: TemporaryWithout = {}
        for variable in self.variables:
            if variable.item is not None and not values[variable.key]:
                without.setdefault(variable.option, []).append(variable.item)
        return settings, without


def read_temporary_space(path_text: str) -> TemporarySpace:
    try:
        return _space(_document(path_text))
    except _SpaceProblem as e:
        raise EntryPointError(
            f"--space {path_text} is not a usable space file: {e}. Fix the file, then try again."
        ) from e


def temporary_without(collection: Any, items: list[dict[str, Any]]) -> Any:
    if isinstance(collection, dict):
        keys = {item["key"] for item in items}
        return {key: value for key, value in collection.items() if key not in keys}
    indexes = {item["index"] for item in items if "index" in item}
    names = {item["name"] for item in items if "name" in item}
    return [entry for index, entry in enumerate(collection) if index not in indexes and not _is_named(entry, names)]


def temporary_item_problem(options: dict[str, Any], option: str, item: dict[str, Any]) -> str | None:
    collection = options.get(option)
    container = dict if "key" in item else list
    if collection is None:
        return f"the agent passes no {option}"
    if not isinstance(collection, container):
        return f"the agent's {option} is a {type(collection).__name__}, not a {container.__name__}"
    if len(temporary_without(collection, [item])) == len(collection):
        return f"the agent's {option} has no such item"
    return None


def _is_named(entry: Any, names: set[str]) -> bool:
    identifiers = (entry, *(getattr(entry, attribute, None) for attribute in _NAME_ATTRIBUTES))
    return any(isinstance(identifier, str) and identifier in names for identifier in identifiers)


def _document(path_text: str) -> Any:
    try:
        return json.loads(Path(path_text).expanduser().read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise _SpaceProblem(f"it cannot be read as JSON ({e})") from e


def _space(document: Any) -> TemporarySpace:
    entries = document.get("variables") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise _SpaceProblem('it must be a JSON object with a "variables" list')
    variables = tuple(_variable(position, entry) for position, entry in enumerate(entries, start=1))
    _check_keys(variables)
    _check_options(variables)
    return TemporarySpace(variables)


def _variable(position: int, entry: Any) -> TemporarySpaceVariable:
    if not isinstance(entry, dict):
        raise _SpaceProblem(f"variable {position} is not a JSON object")
    key = entry.get("key")
    if not isinstance(key, str) or not _KEY_PATTERN.fullmatch(key):
        raise _SpaceProblem(
            f"variable {position} has the key {json.dumps(key)}, but a key must match [A-Za-z_][A-Za-z0-9_]*"
        )
    option = entry.get("option")
    if not isinstance(option, str) or not option:
        raise _SpaceProblem(f"variable {key} needs an option: the name of the createHarness option it changes")
    kind = entry.get("kind")
    if kind == TemporaryVariableKind.TEXT:
        return _text(key, option, entry)
    if kind == TemporaryVariableKind.CHOICE:
        return _choice(key, option, entry)
    if kind == TemporaryVariableKind.SWITCH:
        return _switch(key, option, entry)
    raise _SpaceProblem(f"variable {key} has the kind {json.dumps(kind)}, but a kind is text, choice or switch")


def _text(key: str, option: str, entry: dict[str, Any]) -> TemporarySpaceVariable:
    default, about = entry.get("default"), entry.get("about")
    if not isinstance(default, (str, type(None))) or not isinstance(about, (str, type(None))):
        raise _SpaceProblem(f"text {key} needs a default and an about that are strings, if it has them")
    return TemporarySpaceVariable(key, TemporaryVariableKind.TEXT, option, default, about=about or "")


def _choice(key: str, option: str, entry: dict[str, Any]) -> TemporarySpaceVariable:
    values, default = entry.get("values"), entry.get("default")
    if not isinstance(values, list) or not all(isinstance(value, _SCALAR_TYPES) for value in values):
        raise _SpaceProblem(f"choice {key} needs values: a list of strings, numbers, booleans or nulls")
    distinct = dict.fromkeys(map(_identity, values))
    if len(distinct) < 2:
        raise _SpaceProblem(f"choice {key} needs at least two distinct values")
    if "default" not in entry or not isinstance(default, _SCALAR_TYPES) or _identity(default) not in distinct:
        raise _SpaceProblem(f"choice {key} needs a default that is one of its values")
    choices = tuple(value for _, value in distinct)
    return TemporarySpaceVariable(key, TemporaryVariableKind.CHOICE, option, default, values=choices)


def _switch(key: str, option: str, entry: dict[str, Any]) -> TemporarySpaceVariable:
    default, item = entry.get("default"), entry.get("item")
    if not isinstance(default, bool):
        raise _SpaceProblem(f"switch {key} needs a default of true or false")
    if not _is_item(item):
        raise _SpaceProblem(f'switch {key} needs an item: {{"index": N}}, {{"key": "NAME"}} or {{"name": "NAME"}}')
    return TemporarySpaceVariable(key, TemporaryVariableKind.SWITCH, option, default, item=dict(item))


def _identity(value: Any) -> tuple[bool, Any]:
    return isinstance(value, bool), value


def _is_item(item: Any) -> TypeGuard[dict[str, Any]]:
    fields = [field for field in _ITEM_FIELDS if field in item] if isinstance(item, dict) else []
    if len(fields) != 1:
        return False
    value = item[fields[0]]
    if fields[0] == "index":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    return isinstance(value, str)


def _check_keys(variables: tuple[TemporarySpaceVariable, ...]) -> None:
    seen: set[str] = set()
    for variable in variables:
        if variable.key in seen:
            raise _SpaceProblem(f"two variables have the key {variable.key}; give each variable its own key")
        seen.add(variable.key)


def _check_options(variables: tuple[TemporarySpaceVariable, ...]) -> None:
    owners: dict[str, TemporarySpaceVariable] = {}
    for variable in variables:
        owner = owners.setdefault(variable.option, variable)
        if owner is not variable and (owner.item is None or variable.item is None):
            raise _SpaceProblem(
                f"variables {owner.key} and {variable.key} both change {variable.option}; "
                "only switches can share an option"
            )
