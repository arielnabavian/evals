from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...case import Case
from .._agent_task import synthesize_task_function
from .._entrypoint import ResolvedEntryPoint
from ._temporary_typescript_runner import TYPESCRIPT_SUFFIXES, load_typescript_harness
from .temporary_optimize import _load_harness_file


def temporary_harness_task(
    spec: str, typescript_runner: str | None, trace_attributes: dict[str, Any]
) -> Callable[[Case], Any] | None:
    suffix = Path(spec).suffix
    if suffix == ".py":
        harness = _load_harness_file(spec)
        entry = ResolvedEntryPoint(kind="callable_no_arg", obj=lambda: harness.build({}), spec=spec)
        return synthesize_task_function(entry, extra_trace_attributes=trace_attributes)
    if suffix not in TYPESCRIPT_SUFFIXES:
        return None
    typescript = load_typescript_harness(spec, typescript_runner)

    async def run_typescript_case(case: Case) -> dict[str, Any]:
        return {"output": await typescript.build({}).invoke_async(case.input)}

    return run_typescript_case
