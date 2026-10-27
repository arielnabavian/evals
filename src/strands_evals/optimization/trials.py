"""One candidate's trials: every case of the experiment, ``repetitions`` times, each with a status.

A trial that measured something is ``COMPLETED`` or ``AGENT_FAILURE`` (the agent raised); one that measured nothing is
``INFRASTRUCTURE_INVALID`` (throttling, connection, credentials) or ``VERIFIER_INVALID`` (every evaluator failed). The
status is decided here, never inferred from a score: ``Experiment`` scores a failed task 0 for every evaluator, and
that zero is discarded.

A trial runs a ``task`` with the agent built for it: by default one ``agent.invoke_async(case.input)``; a workload
whose trial is more than one call (a simulated user's conversation, a benchmark's own loop) gives its own. A task
raises ``TrialInvalid`` when what failed was not the agent (the simulated user, the environment); such a trial, and one
that failed to connect, runs again ``retries`` times before it is recorded as measuring nothing.

Usage comes from the agent's own metrics. Model calls a conversation manager makes itself (summaries) are not in
them until Strands records them (an upstream fix), so ``tokens`` leaves them out. A trial keeps every attempt: its
metrics are the final attempt's, and an attempt run again (it measured nothing: the infrastructure failed, or the model
service throttled it and Strands Evals ran it again) is kept in ``attempts`` with what it spent, which a study's cost
counts (``cost.py``) and nothing it decides uses.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping

from ..experiment import Experiment
from .space import TunableFactory, Values


class TrialStatus(str, Enum):
    """What one trial (a case, one repetition) measured."""

    COMPLETED = "completed"
    AGENT_FAILURE = "agent_failure"  # the agent raised: its usage counts, its scores take the study's failure value
    INFRASTRUCTURE_INVALID = "infrastructure_invalid"  # throttling, connection, credentials: nothing measured
    VERIFIER_INVALID = "verifier_invalid"  # every evaluator failed: nothing measured
    CANCELLED = "cancelled"


# Measured something (the trial has values); the other statuses measured nothing.
SCORED = frozenset({TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE})

# Agent usage, by metric name (``agent.event_loop_metrics.accumulated_usage``).
TOKENS = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cache_read_tokens": "cacheReadInputTokens",
    "cache_write_tokens": "cacheWriteInputTokens",
}
USAGE = (*TOKENS, "tokens", "seconds", "model_calls", "removed_messages")
SPENT = (*TOKENS, "tokens", "model_calls", "seconds")  # the usage that is spent, which a study's cost sums
NEVER_RAN = "the task never ran"

RETRY_WAIT = 5.0  # seconds before a trial that measured nothing runs again, doubling each time


@dataclass(frozen=True)
class Attempt:
    """An attempt of a trial that was run again: it measured nothing, but what it spent (``SPENT``) counts."""

    status: TrialStatus
    error: str | None
    usage: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "error": self.error, "usage": dict(self.usage)}


@dataclass(frozen=True)
class Trial:
    """One case, one repetition: its status, metrics (evaluator scores and usage), output and messages, from its final
    attempt; and the attempts before it that were run again."""

    case: str
    repetition: int  # 1-based
    status: TrialStatus
    metrics: Mapping[str, float]  # evaluator scores measured, and usage
    unavailable: frozenset[str] = frozenset()  # evaluators that failed on this trial
    error: str | None = None
    output: str | None = None
    messages: tuple[Any, ...] = field(default=(), repr=False)
    reasons: Mapping[str, str] = field(default_factory=dict, repr=False)
    attempts: tuple[Attempt, ...] = field(default=(), repr=False)  # earlier attempts, run again: spent, never used


class TrialInvalid(Exception):
    """Raised from a task when the trial measured nothing for a reason other than the agent: its simulated user or
    its environment failed. The trial is ``INFRASTRUCTURE_INVALID`` and runs again (``Study(retries=...)``)."""


Task = Callable[[Any, Any], Any | Awaitable[Any]]


async def invoke(agent: Any, case: Any) -> str:
    """The default task: one call of the agent with the case's input; its reply is the output."""
    return str(await agent.invoke_async(case.input))


class _AgentFailed(Exception):
    """Raised from the task so Strands Evals neither retries it nor runs the evaluators."""


def unwrap(exc: BaseException) -> BaseException:
    """The exception that caused ``exc``: Strands wraps failures inside the event loop in ``EventLoopException``."""
    from strands.types.exceptions import EventLoopException

    while isinstance(exc, EventLoopException) and getattr(exc, "original_exception", None) is not None:
        exc = exc.original_exception
    return exc


def classify(exc: BaseException) -> TrialStatus:
    """The status of a trial whose agent raised ``exc``: a failure of the infrastructure, or of the agent."""
    from strands.types.exceptions import EventLoopException

    from ..utils import is_throttling_error

    exc = unwrap(exc)
    if isinstance(exc, TrialInvalid):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if is_throttling_error(exc) and not isinstance(exc, EventLoopException):
        return TrialStatus.INFRASTRUCTURE_INVALID
    if isinstance(exc, _infrastructure_errors()):
        return TrialStatus.INFRASTRUCTURE_INVALID
    code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", "")).lower()
    if code in _SERVICE_ERRORS:  # the model service's own failure, in a streamed response (botocore's EventStreamError)
        return TrialStatus.INFRASTRUCTURE_INVALID
    return TrialStatus.AGENT_FAILURE


_SERVICE_ERRORS = {
    "internalserverexception",
    "serviceunavailableexception",
    "modelstreamerrorexception",
    "modelnotreadyexception",
    "throttlingexception",
}


def _infrastructure_errors() -> tuple[type[BaseException], ...]:
    """The transport's errors: no connection, no credentials, or a response that stopped or broke while it was read
    (urllib3's own timeouts reach the agent unwrapped when a streamed response stalls)."""
    errors: list[type[BaseException]] = []
    try:
        import botocore.exceptions as boto

        for name in (
            "ConnectTimeoutError",
            "EndpointConnectionError",
            "NoCredentialsError",
            "ReadTimeoutError",
            "ResponseStreamingError",
            "ConnectionClosedError",
        ):
            if hasattr(boto, name):
                errors.append(getattr(boto, name))
    except ImportError:  # pragma: no cover - botocore ships with strands' Bedrock model
        pass
    try:
        import urllib3.exceptions as http

        errors += [http.TimeoutError, http.ProtocolError, http.NewConnectionError]
    except ImportError:  # pragma: no cover - urllib3 ships with botocore
        pass
    return tuple(errors)


def _usage(agent: Any, seconds: float) -> dict[str, float]:
    metrics = getattr(agent, "event_loop_metrics", None)
    used = dict(getattr(metrics, "accumulated_usage", {}) or {})
    out = {name: float(used.get(key, 0) or 0) for name, key in TOKENS.items()}
    out["tokens"] = sum(out[name] for name in TOKENS)
    out["seconds"] = seconds
    out["model_calls"] = float(getattr(metrics, "cycle_count", 0) or 0)
    manager = getattr(agent, "conversation_manager", None)
    out["removed_messages"] = float(getattr(manager, "removed_message_count", 0) or 0)
    return out


async def run_trials(
    factory: TunableFactory,
    values: Values,
    experiment: Experiment,
    *,
    repetitions: int,
    max_concurrency: int,
    task: Task = invoke,
    retries: int = 2,
) -> list[Trial]:
    """Run every case ``repetitions`` times, each with a fresh agent built from ``values`` and run by ``task``."""
    from ..utils import is_throttling_error

    repetitions = max(1, int(repetitions))
    expanded, origin = [], {}
    for case in experiment.cases:
        for r in range(1, repetitions + 1):
            name = f"{case.name}#{r}"
            origin[name] = (case.name, r)
            expanded.append(case.model_copy(update={"name": name}))
    # Every attempt of each trial, in order: one run() retries a trial that measured nothing, and Strands Evals calls
    # run() again for one the model service throttled.
    history: dict[str, list[dict[str, Any]]] = {}
    kwargs = factory.bind(values)
    # A sync task runs in a pool as wide as the concurrency: asyncio.to_thread's shared pool holds at most
    # min(32, cpus + 4) threads, which would cap the trials running at once there.
    pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="trial")
    loop = asyncio.get_running_loop()

    async def run(case: Any) -> Any:
        for attempt in range(retries + 1):
            start, agent = time.monotonic(), None
            try:
                agent = factory(**kwargs)
                if inspect.iscoroutinefunction(task) or inspect.iscoroutinefunction(type(task).__call__):
                    result = await task(agent, case)
                else:
                    call = functools.partial(contextvars.copy_context().run, task, agent, case)
                    result = await loop.run_in_executor(pool, call)
            except Exception as exc:  # noqa: BLE001 - every failure becomes a status
                status, cause = classify(exc), unwrap(exc)
                history.setdefault(case.name, []).append(
                    {
                        "status": status,
                        "error": f"{type(cause).__name__}: {cause}"[:2000],
                        "usage": _usage(agent, time.monotonic() - start),
                        "messages": tuple(getattr(agent, "messages", ()) or ()),
                        "output": None,
                    }
                )
                if status == TrialStatus.INFRASTRUCTURE_INVALID:
                    if is_throttling_error(cause):
                        raise  # Strands Evals retries throttling itself
                    if attempt < retries:
                        await asyncio.sleep(RETRY_WAIT * 2**attempt)
                        continue
                raise _AgentFailed(str(exc)) from exc
            output = result.get("output") if isinstance(result, dict) else result
            history.setdefault(case.name, []).append(
                {
                    "status": TrialStatus.COMPLETED,
                    "error": None,
                    "usage": _usage(agent, time.monotonic() - start),
                    "messages": tuple(getattr(agent, "messages", ()) or ()),
                    "output": None if output is None else str(output),
                }
            )
            return result

    view = Experiment(cases=expanded, evaluators=experiment.evaluators)
    try:
        report = await view.run_evaluations_async(run, max_workers=max_concurrency)
    finally:
        pool.shutdown(wait=False)
    scores: dict[str, dict[str, float]] = {name: {} for name in origin}
    reasons: dict[str, dict[str, str]] = {name: {} for name in origin}
    measured: dict[str, set[str]] = {name: set() for name in origin}
    for row, score, reason, detail in zip(
        report.cases, report.scores, report.reasons, report.detailed_results, strict=True
    ):
        name, evaluator = row["name"], row["evaluator"]
        reasons[name][evaluator] = reason
        if detail:  # Strands Evals leaves an evaluator's outputs empty when it failed
            scores[name][evaluator] = float(score)
            measured[name].add(evaluator)
    evaluators = {e.get_name() for e in experiment.evaluators}
    trials = []
    for name, (case, r) in origin.items():
        *earlier, run = history.get(name) or [
            {
                "status": TrialStatus.INFRASTRUCTURE_INVALID,
                "error": NEVER_RAN,
                "usage": {},
                "messages": (),
                "output": None,
            }
        ]
        status, measured_values, unavailable = run["status"], dict(run["usage"]), frozenset()
        if status == TrialStatus.COMPLETED:
            unavailable = frozenset(evaluators - measured[name])
            measured_values.update(scores[name])
            if unavailable == evaluators:
                status = TrialStatus.VERIFIER_INVALID
        trials.append(
            Trial(
                case=case,
                repetition=r,
                status=status,
                metrics=measured_values,
                unavailable=unavailable,
                error=run["error"]
                or ("; ".join(f"{e}: {reasons[name].get(e, '')}" for e in sorted(unavailable)) or None),
                output=run["output"],
                messages=run["messages"],
                reasons=reasons[name] if status == TrialStatus.COMPLETED else {},
                attempts=tuple(Attempt(a["status"], a["error"], {k: a["usage"][k] for k in SPENT}) for a in earlier),
            )
        )
    return trials


def metric_names(experiment: Experiment) -> set[str]:
    """The scores an experiment produces: its evaluators' names."""
    return {e.get_name() for e in experiment.evaluators}
