from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ...case import Case
from ...experiment import Experiment
from .._entrypoint import EntryPointError, resolve_custom_evaluators
from ._temporary_evaluator_catalog import temporary_optimize_problem
from ._temporary_harness import TemporaryHarness, load_temporary_harness, temporary_answer_text
from ._temporary_model_effort import EFFORT_LEVELS
from ._temporary_optimize_space import (
    TemporarySpace,
    TemporarySpaceVariable,
    read_temporary_space,
    temporary_text_value,
)
from ._temporary_typescript_runner import TYPESCRIPT_SUFFIXES

TEMPORARY_HARNESS_SETTINGS = ("instructions", "effort")
_DEFAULT_HARNESS_SETTINGS = ("instructions",)
_LEVELS_WITHOUT_REASONING = frozenset({"auto", "off"})


@dataclass(frozen=True)
class TemporaryTask:
    agent: Any
    rollout: Callable[..., Any]
    search: Any


def _optimization() -> ModuleType:
    try:
        from ... import optimization
    except ImportError as e:
        raise EntryPointError(
            f"strands-evals optimize needs the optimization extra: "
            f"pip install 'strands-agents-evals[optimization]' ({e})"
        ) from e
    return optimization


def _temporary_check_instructions(harness: TemporaryHarness) -> None:
    if "system_prompt" in harness.options:
        raise EntryPointError(
            "instructions cannot be tuned: the agent passes a full system prompt, which replaces its instructions. "
            "Move that text into instructions, or tune another setting such as effort."
        )


def _instructions_marker(harness: TemporaryHarness, optimization: ModuleType) -> Any:
    _temporary_check_instructions(harness)
    return temporary_text_value(
        harness.options.get("instructions") or "",
        "the agent's domain instructions, which the harness appends to its own system prompt",
        optimization,
    )


def _effort_marker(harness: TemporaryHarness, optimization: ModuleType) -> Any:
    reason = harness.effort_unavailable()
    if reason is not None:
        raise EntryPointError(f"effort cannot be tuned: {reason}")
    baseline = harness.effort_baseline()
    candidates = [level for level in EFFORT_LEVELS if level != "auto" or baseline == "auto"]
    levels = [level for level in candidates if level == baseline or harness.builds_with(effort=level)]
    if len(levels) < 2 or all(level in _LEVELS_WITHOUT_REASONING for level in levels):
        raise EntryPointError(
            f"effort cannot be tuned: the agent's model accepts no reasoning levels besides {levels}. "
            "Tune instructions instead."
        )
    return optimization.OneOf(*levels, default=baseline)


_SETTING_MARKERS: dict[str, Callable[[TemporaryHarness, ModuleType], Any]] = {
    "instructions": _instructions_marker,
    "effort": _effort_marker,
}


def _setting_factory(harness: TemporaryHarness, settings: tuple[str, ...], optimization: ModuleType) -> Any:
    unknown = [setting for setting in settings if setting not in _SETTING_MARKERS]
    if unknown or not settings:
        raise EntryPointError(
            f"settings must name one or more of {', '.join(TEMPORARY_HARNESS_SETTINGS)}, got {list(settings)}"
        )
    markers = {name: _SETTING_MARKERS[name](harness, optimization) for name in settings}

    def create_agent(**values: Any) -> Any:
        return harness.build(values)

    return optimization.tunable(create_agent, inject=markers)


def _temporary_check_variable(harness: TemporaryHarness, variable: TemporarySpaceVariable) -> None:
    if variable.option == "instructions":
        _temporary_check_instructions(harness)
    if variable.item is None:
        return
    problem = harness.temporary_without_problem(variable.option, variable.item)
    if problem:
        raise EntryPointError(
            f"the space variable {variable.key} cannot leave out {json.dumps(variable.item)} of {variable.option}: "
            f"{problem}. Make the space file again from the agent file as it is now, then try again."
        )


def _space_factory(harness: TemporaryHarness, space: TemporarySpace, optimization: ModuleType) -> Any:
    for variable in space.variables:
        _temporary_check_variable(harness, variable)
    markers = space.bindings(harness.options, optimization)

    def create_agent(**values: Any) -> Any:
        settings, without = space.overrides(values)
        return harness.build(settings, without)

    return optimization.tunable(create_agent, inject=markers)


def _baseline_only_search(optimization: ModuleType) -> Any:
    from ...optimization import Schedule

    class TemporaryBaselineOnly(Schedule):
        name = "temporary_baseline_only"

        def next(self, ctx: Any) -> None:
            return None

    return optimization.Search([], schedule=TemporaryBaselineOnly())


async def temporary_rollout(agent_factory: Callable[..., Any], case: Case) -> dict[str, Any]:
    agent = agent_factory()
    return {"output": temporary_answer_text(await agent.invoke_async(case.input))}


def _check_experiment(experiment_file: str, custom_evaluators: Sequence[str]) -> None:
    evaluators = resolve_custom_evaluators(list(custom_evaluators))
    try:
        experiment = Experiment.from_file(experiment_file, custom_evaluators=evaluators)
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, TypeError) as e:
        raise EntryPointError(
            f"{experiment_file} is not a usable Experiment file ({type(e).__name__}: {e}). Fix the file "
            f"(strands-evals validate {experiment_file} checks it), or create a new one with /experiment in strands dev"
        ) from e
    problem = temporary_optimize_problem(experiment.evaluators)
    if problem:
        raise EntryPointError(problem)


def temporary_task(
    agent_file: str,
    *,
    experiment_file: str,
    space_file: str | None = None,
    settings: Sequence[str] = _DEFAULT_HARNESS_SETTINGS,
    typescript_runner: str | None = None,
    custom_evaluators: Sequence[str] = (),
    baseline_only: bool = False,
) -> TemporaryTask:
    optimization = _optimization()
    _check_experiment(experiment_file, custom_evaluators)
    if Path(agent_file).suffix in TYPESCRIPT_SUFFIXES and not typescript_runner:
        raise EntryPointError(
            f"{agent_file} is a TypeScript agent; give typescript_runner, the command that starts its runner"
        )
    space = read_temporary_space(space_file) if space_file is not None else None
    with contextlib.redirect_stdout(sys.stderr):
        harness = load_temporary_harness(agent_file, typescript_runner)
    if harness is None:
        raise EntryPointError(f"{agent_file} is not a harness file; give a PATH.py or a PATH.ts")
    factory = (
        _space_factory(harness, space, optimization)
        if space is not None
        else _setting_factory(harness, tuple(dict.fromkeys(settings)), optimization)
    )
    return TemporaryTask(
        agent=factory,
        rollout=temporary_rollout,
        search=_baseline_only_search(optimization) if baseline_only else None,
    )
