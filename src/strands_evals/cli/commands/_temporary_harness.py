from __future__ import annotations

import ast
import importlib
import importlib.machinery
import inspect
import sys
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from strands import Agent

from .._entrypoint import EntryPointError, _import_from_path
from ._temporary_model_effort import ModelEffort, read_model_effort
from ._temporary_optimize_space import TemporaryWithout, temporary_item_problem, temporary_without
from ._temporary_typescript_runner import TYPESCRIPT_SUFFIXES, load_typescript_harness

_SHARED_STATE_OPTIONS = frozenset({"session_manager", "memory_manager", "storage"})
_TRIAL_OVERRIDES: dict[str, Any] = {"session": False, "memory": False, "callback_handler": None}


class TemporaryHarness(Protocol):
    @property
    def options(self) -> dict[str, Any]: ...

    def build(self, values: dict[str, Any], without: TemporaryWithout | None = None) -> Any: ...

    def builds_with(self, **values: Any) -> bool: ...

    def temporary_without_problem(self, option: str, item: dict[str, Any]) -> str | None: ...

    def effort_unavailable(self) -> str | None: ...

    def effort_baseline(self) -> str: ...


@dataclass(frozen=True)
class _Construction:
    agent: Any
    args: tuple[Any, ...]
    options: dict[str, Any]


class _HarnessCapture:
    def __init__(self, harness: ModuleType) -> None:
        self.harness = harness
        self.original: Callable[..., Any] = harness.create_harness
        self.constructions: list[_Construction] = []
        self.recording = False

    def create_harness(self, *args: Any, **options: Any) -> Any:
        agent = self.original(*args, **options)
        if self.recording:
            self.constructions.append(_Construction(agent, args, dict(options)))
        return agent

    def __enter__(self) -> _HarnessCapture:
        self.recording = True
        self.harness.create_harness = self.create_harness
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.harness.create_harness = self.original
        self.recording = False


@dataclass(frozen=True)
class TemporaryHarnessFile:
    factory: Callable[..., Any]
    options: dict[str, Any]

    @cached_property
    def _model_effort(self) -> ModelEffort | None:
        model = self.options.get("model")
        return None if model is None or isinstance(model, str) else read_model_effort(model)

    def build(self, values: dict[str, Any], without: TemporaryWithout | None = None) -> Any:
        kept = {option: temporary_without(self.options[option], items) for option, items in (without or {}).items()}
        overrides = {**values, **kept, **_TRIAL_OVERRIDES}
        if "effort" in overrides and self._model_effort is not None:
            overrides.update(self._model_effort.options_for(overrides.pop("effort")))
        if overrides.get("instructions") == "":
            overrides["instructions"] = None
        agent = self.factory(**overrides)
        if not isinstance(agent, Agent):
            raise TypeError(
                f"create_agent(**overrides) returned {type(agent).__name__}, not a Strands Agent. "
                "Return the agent that create_harness(...) builds, then try again."
            )
        return agent

    def builds_with(self, **values: Any) -> bool:
        try:
            self.build(values)
        except ValueError:
            return False
        return True

    def temporary_without_problem(self, option: str, item: dict[str, Any]) -> str | None:
        return temporary_item_problem(self.options, option, item)

    def effort_unavailable(self) -> str | None:
        return None if self._model_effort is None else self._model_effort.unavailable

    def effort_baseline(self) -> str:
        return self.options.get("effort", "auto") if self._model_effort is None else self._model_effort.baseline


def _strands_harness() -> ModuleType:
    try:
        return importlib.import_module("strands_harness")
    except ImportError as e:
        raise EntryPointError(
            f"a Python harness file needs the strands-harness package, which is missing ({e}). "
            "Install strands-harness in the project's environment, then try again."
        ) from e


def _is_agent_built_by_create_harness(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    if not any(isinstance(target, ast.Name) and target.id == "agent" for target in targets):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    function = call.func
    return (isinstance(function, ast.Name) and function.id == "create_harness") or (
        isinstance(function, ast.Attribute) and function.attr == "create_harness"
    )


def _assigns_agent_from_create_harness(path: Path) -> bool:
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    return any(_is_agent_built_by_create_harness(node) for node in tree.body)


def _installed_module_name(parts: list[str], root: Path) -> str | None:
    for index in range(1, len(parts) - 1):
        name = parts[index]
        if not name.isidentifier():
            continue
        spec = importlib.machinery.PathFinder.find_spec(name)
        if spec is None:
            continue
        expected = root.joinpath(*parts[: index + 1]).resolve()
        if any(Path(location).resolve() == expected for location in spec.submodule_search_locations or []):
            return ".".join(parts[index:])
    return None


def _module_name(path: Path, root: Path) -> str | None:
    if not path.is_relative_to(root):
        return None
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    packaged = bool(parts) and all(
        root.joinpath(*parts[:index], "__init__.py").is_file() for index in range(1, len(parts))
    )
    return ".".join(parts) if packaged else _installed_module_name(parts, root)


def _forget_module_at(path: Path) -> None:
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if isinstance(location, str) and Path(location).resolve() == path:
            del sys.modules[name]


def _import_harness_module(path: Path, path_text: str) -> ModuleType:
    name = _module_name(path, Path.cwd().resolve())
    if name is None:
        _forget_module_at(path)
        return _import_from_path(str(path), path_text)
    sys.modules.pop(name, None)
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise EntryPointError(
            f"failed to import '{path_text}' as {name}: {e}. Fix that error, then try again; strands dev shows the "
            "same error when it opens this agent"
        ) from e


def load_temporary_harness_file(path_text: str) -> TemporaryHarnessFile:
    path = Path(path_text).expanduser().resolve()
    harness = _strands_harness()
    sys.path[:0] = [d for d in (str(Path.cwd()), str(path.parent)) if d not in sys.path]
    with _HarnessCapture(harness) as capture:
        module = _import_harness_module(path, path_text)
    agent = getattr(module, "agent", None)
    if not isinstance(agent, Agent):
        raise EntryPointError(
            f'{path_text} does not export a Strands Agent named "agent". '
            "Export one, for example agent = create_harness(...), then try again."
        )
    construction = next((c for c in capture.constructions if c.agent is agent), None)
    if construction is None:
        raise EntryPointError(
            f"{path_text} builds its agent without create_harness(...), so optimize cannot read its settings. "
            "Build the agent with create_harness(...), then try again."
        )
    shared = sorted(set(construction.options) & _SHARED_STATE_OPTIONS)
    if shared:
        raise EntryPointError(
            f"{path_text} passes {', '.join(shared)} to create_harness(...); "
            "every trial would share it, so remove it to optimize this agent"
        )
    factory = getattr(module, "create_agent", None)
    if inspect.iscoroutinefunction(factory):
        raise EntryPointError(
            f"{path_text} defines create_agent as async, and the optimizer calls it as a regular function. "
            "Change it to def create_agent(**overrides), then try again."
        )
    if callable(factory):
        return TemporaryHarnessFile(factory, construction.options)
    if not _assigns_agent_from_create_harness(path):
        raise EntryPointError(
            f"{path_text} changes its agent after create_harness(...); "
            "define create_agent(**overrides) so each trial can build a fresh agent"
        )
    original = capture.original

    def create_agent(**overrides: Any) -> Any:
        return original(*construction.args, **{**construction.options, **overrides})

    return TemporaryHarnessFile(create_agent, construction.options)


def load_temporary_harness(spec: str, typescript_runner: str | None) -> TemporaryHarness | None:
    suffix = Path(spec).suffix
    if suffix == ".py":
        return load_temporary_harness_file(spec)
    if suffix in TYPESCRIPT_SUFFIXES:
        return load_typescript_harness(spec, typescript_runner)
    return None


def temporary_answer_text(reply: Any) -> str:
    return str(reply).rstrip("\n")
