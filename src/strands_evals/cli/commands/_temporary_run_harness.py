from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...case import Case
from .._agent_task import synthesize_task_function
from .._entrypoint import ResolvedEntryPoint
from ._temporary_typescript_runner import TYPESCRIPT_SUFFIXES, load_typescript_harness
from .temporary_optimize import _load_harness_file, temporary_answer_text


def temporary_harness_task(
    spec: str, typescript_runner: str | None, trace_attributes: dict[str, Any]
) -> Callable[[Case], Any] | None:
    suffix = Path(spec).suffix
    if suffix == ".py":
        harness = _load_harness_file(spec)
        entry = ResolvedEntryPoint(kind="callable_no_arg", obj=lambda: harness.build({}), spec=spec)
        return _with_answer_text(synthesize_task_function(entry, extra_trace_attributes=trace_attributes))
    if suffix not in TYPESCRIPT_SUFFIXES:
        return None
    typescript = load_typescript_harness(spec, typescript_runner)

    async def run_typescript_case(case: Case) -> dict[str, Any]:
        return {"output": temporary_answer_text(await typescript.build({}).invoke_async(case.input))}

    return run_typescript_case


def _with_answer_text(task: Callable[[Case], dict[str, Any]]) -> Callable[[Case], dict[str, Any]]:
    def run_python_case(case: Case) -> dict[str, Any]:
        result = task(case)
        return {**result, "output": temporary_answer_text(result["output"])}

    return run_python_case
