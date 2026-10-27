"""Tune a Strands harness (``create_harness``) as an AppWorld code agent: the same search spaces and goals as
tau2_harness.py, on AppWorld's tasks (appworld_workload.py).

    python examples/optimization/appworld_harness.py --baseline --split dev --out dev-baseline.json
    BYPASS_TOOL_CONSENT=true python examples/optimization/appworld_harness.py --space prompt --goal best --out studies

The agent is ``create_harness``'s, with AppWorld's canonical code-agent instructions (and their worked example) as
its instructions, fixed, and one tool, ``execute``, which runs Python in the task's world. The harness's defaults stay:
its contract (the system prompt before the instructions), the todos plugin, thinking at create_harness's effort
"auto", prompt caching, and ``context_manager="auto"``. Off is what one AppWorld task cannot use: memory, sessions and
skills (they would carry over between tasks), the environment plugin, the built-in tools, and background tasks.

Haiku 4.5 thinks with a token budget; Haiku 5.5 thinks adaptively, at an effort (low to max). The ``numbers`` and
``full`` spaces tune whichever the agent's model takes. ``max_tokens`` stays bounded, because Bedrock reserves
``max_tokens`` of quota for every request.

Search spaces:

    prompt   the harness's contract (text: reflection)
    numbers  thinking (budget or effort) and the context manager's settings (TPE)
    full     the contract, ``execute``'s description, thinking, the todos plugin, and the context manager:
             create_harness's, one that drops stale tool results, or none, each with its settings

--max-effort caps the thinking effort the numbers and full spaces try (Haiku 5.5; e.g. xhigh leaves out max).

A study searches on AppWorld's train tasks (90), selects on its dev tasks (57), and tests the selection and the
baseline on --test (test_normal's 168 by default, or test_challenge's 417), which nothing before saw: each case
--repetitions times, until the best has not changed for --patience iterations (at most --iterations). The goals are
tau2_harness.py's (floor's --floor is absolute, judged on train and dev, whose rewards run higher than test_challenge's;
--floor-gain is a gain over the baseline on the same tasks instead), and so is what a run ships when nothing meets a
goal's limit on dev: the closest, unconfirmed, or with ``--unmet none`` nothing (the baseline, as round 3 and 3b).
The goals with a cost objective (joint, floor) search reward first, then cost (--schedule reward-first, round 3d's
default), as tau2_harness.py describes; --schedule joint is round 3 and 3b's.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from appworld_workload import AppWorldTask, Reward, TestsPassed, cases, execute, instructions
from strands import ModelRetryStrategy
from strands.experimental.context_manager import ContextManager, Offload
from strands_harness import create_harness
from strands_harness.models import resolve_model
from strands_harness.prompt import HARNESS_CONTRACT

from strands_evals import Experiment
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Block,
    CheapestOnFront,
    Choice,
    Derived,
    Float,
    Int,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    PrimaryFirst,
    Reflection,
    Search,
    Seesaw,
    Selection,
    Split,
    Study,
    TextValue,
    component,
    tool_descriptions,
    tunable,
)
from strands_evals.optimization.selection import ALSO, UNMET
from strands_evals.optimization.trials import run_trials

AGENT_MODEL = os.environ.get("APPWORLD_AGENT_MODEL", "us.anthropic.claude-haiku-5-5")
REFLECTION_MODEL = os.environ.get("APPWORLD_REFLECTION_MODEL", "us.anthropic.claude-sonnet-5-5")
REFLECTION_MAX_TOKENS = 32000  # a full space's texts are written in one turn: over the model's default output
ADAPTIVE = "haiku-5" in AGENT_MODEL  # Haiku 5.5 rejects budget thinking: thinking.type adaptive, at an effort
MAX_TOKENS = 32768  # Bedrock reserves max_tokens of quota per request; create_harness's 128k would exhaust it

# Each kind of token, weighted by its price relative to an uncached input token (the same ratios for Haiku 4.5 and 5.5).
WEIGHTS = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_write_tokens": 1.25, "cache_read_tokens": 0.1}
WEIGHTED_TOKENS = Derived(
    "weighted_tokens", lambda m: sum(m[k] * weight for k, weight in WEIGHTS.items()), range=(0, None)
)


def patient() -> ModelRetryStrategy:
    """A model call Bedrock throttles waits for capacity (up to 20 attempts, backing off to 4 minutes: about an hour)
    rather than failing its conversation: a conversation that fails on throttling measures nothing, and long ones fail
    most, which would bias what is measured. Strands' default is 6 attempts. One per agent: it counts attempts."""
    return ModelRetryStrategy(max_attempts=20, initial_delay=4, max_delay=240)


# A thinking block is bound to the conversation before it. When a context strategy later drops or cuts an earlier
# message (the full space's drop_stale), Bedrock may reject the next call ("Invalid `signature` in `thinking` block. The
# block is bound to a different conversation"; round 3 saw it in one account) and the trial ends as the agent's
# failure. A mismatched block is dropped instead (Anthropic's own setting for it, behind its beta header, as
# tau2_harness.py and loca_workload.py do): a call whose prefix is unchanged is sent and answered as before.
BLOCK_BINDING = {"prefix_mismatch_behavior": "drop_block"}
BINDING_BETA = "thinking-binding-controls-2026-08-01"


def bound(model: Any) -> Any:
    """``model`` with a thinking block whose earlier conversation changed dropped, not refused (when it thinks)."""
    fields = dict(model.config.get("additional_request_fields") or {})
    thinking = fields.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "adaptive":
        return model
    betas = [b for b in fields.get("anthropic_beta") or [] if b != BINDING_BETA]
    fields.update(thinking={**thinking, "block_binding": BLOCK_BINDING}, anthropic_beta=[*betas, BINDING_BETA])
    model.update_config(additional_request_fields=fields)
    return model


def haiku(thinking: Any) -> Any:
    """Haiku as create_harness builds it (thinking, prompt caching). ``thinking`` is an effort for Haiku 5.5 ("auto"
    is create_harness's: high), a thinking block whose earlier conversation changed dropped (``bound``), and a budget
    of tokens per call for Haiku 4.5."""
    if ADAPTIVE:
        model = resolve_model(f"bedrock/{AGENT_MODEL}", AGENT_MODEL, thinking, caching=True)
        model.update_config(max_tokens=MAX_TOKENS)
        return bound(model)
    model = resolve_model(f"bedrock/{AGENT_MODEL}", AGENT_MODEL, "auto", caching=True)
    budget = int(thinking)
    model.update_config(
        max_tokens=budget + 4096, additional_request_fields={"thinking": {"type": "enabled", "budget_tokens": budget}}
    )
    return model


class Counted:
    """A context strategy that counts, in the agent's state, each time it changes the context (the ``reductions``
    metric): whether a context setting acted at all. A strategy changes the context in its ``apply`` and, for one that
    changes each message as it is added (truncating tool results), in the hooks its ``init`` registers; both count."""

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        return self._strategy.name

    def __getattr__(self, attr: str) -> Any:
        if attr == "_strategy":
            raise AttributeError(attr)
        return getattr(self._strategy, attr)

    def init(self, agent: Any, stash: Any = None) -> None:
        self._strategy.init(_CountingHooks(agent), stash=stash)

    async def apply(self, context: Any) -> bool:
        acted = await self._strategy.apply(context)
        if acted:
            _count(context.agent)
        return acted


def _count(agent: Any) -> None:
    agent.state.set("context_reductions", int(agent.state.get("context_reductions") or 0) + 1)


class _CountingHooks:
    """The agent, as a strategy's ``init`` sees it: each hook it registers counts when it changes the message added."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def __getattr__(self, attr: str) -> Any:
        if attr == "_agent":
            raise AttributeError(attr)
        return getattr(self._agent, attr)

    @property
    def hooks(self) -> Any:
        return self

    def add_callback(self, event_type: Any, callback: Any) -> None:
        async def counted(event: Any) -> None:
            message = getattr(event, "message", None)
            before = repr(message)
            result = callback(event)
            if hasattr(result, "__await__"):
                await result
            if message is not None and repr(message) != before:
                _count(event.agent)

        self._agent.hooks.add_callback(event_type, counted)


class Metered:
    """The agent's model, for the context manager's summaries: their usage is added to the agent's, which the
    summarizer's calls otherwise bypass."""

    def __init__(self, model: Any) -> None:
        self._model, self.agent = model, None

    def __getattr__(self, attr: str) -> Any:
        if attr == "_model":
            raise AttributeError(attr)
        return getattr(self._model, attr)

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        async for event in self._model.stream(*args, **kwargs):
            usage = event.get("metadata", {}).get("usage") if isinstance(event, dict) else None
            if usage and self.agent is not None:
                total = self.agent.event_loop_metrics.accumulated_usage
                for key, n in usage.items():
                    if isinstance(n, (int, float)):
                        total[key] = total.get(key, 0) + n
            yield event


@dataclass(frozen=True)
class Context:
    """The context manager to build: by default create_harness's ("auto": tool results over 1,500 tokens cut to a
    750-token preview, the conversation summarized at 85% of the window), each step with its setting; None leaves a
    step out."""

    truncate_above: int | None = 1500
    preview_share: float = 0.5
    keep_tool_results: int | None = None
    summarize_at: float | None = 0.85

    def build(self, model: Any) -> tuple[Any, Metered | None]:
        steps, summarizer = [], None
        if self.truncate_above is not None:
            config = {"preview_tokens": max(1, int(self.truncate_above * self.preview_share))}
            steps.append(Offload.truncate("tool_results", config).when(threshold=self.truncate_above))
        if self.keep_tool_results is not None:
            steps.append(Offload.drop("tool_results").when(preserve_recent=self.keep_tool_results))
        if self.summarize_at is not None:
            summarizer = Metered(model)
            summarize = Offload.summarize("*", {"model": summarizer})
            steps.append(summarize.when(utilization=self.summarize_at, preserve_recent=4))
        if not steps:
            return False, None
        return ContextManager(strategies=[Counted(s) for s in steps]), summarizer


AUTO = Context()  # create_harness's own
DEFAULT_THINKING: Any = "auto" if ADAPTIVE else 16384


def harness(
    contract: str = HARNESS_CONTRACT,
    tools: Any = (execute,),
    thinking: Any = DEFAULT_THINKING,
    context: Context = AUTO,
    todos: bool = True,
) -> Any:
    """The agent every space tunes; what a space does not tune stays at create_harness's default."""
    model = haiku(thinking)
    manager, summarizer = context.build(model)
    agent = create_harness(
        model=model,
        system_prompt=f"{contract}\n\n<instructions>\n{instructions()}\n</instructions>",
        tools=list(tools),
        caching=False,  # the model carries its own cache settings, as resolve_model made them
        context_manager=manager,
        builtin_tools=[],
        builtin_plugins=["todos"] if todos else [],
        memory=False,
        session=False,
        skills=False,
        background_tasks=False,
        callback_handler=None,
        retry_strategy=patient(),
    )
    if summarizer is not None:
        summarizer.agent = agent
    return agent


CONTRACT = TextValue(
    default=HARNESS_CONTRACT,
    max_chars=6000,
    about="the harness's behavioral contract, given before AppWorld's code-agent instructions (fixed)",
)
TOOL_TEXTS = tool_descriptions(execute, max_chars=1500)
# Haiku 5.5: create_harness's efforts ("auto" is high); Haiku 4.5: tokens of thinking per model call.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING = Choice(list(EFFORTS), default="high") if ADAPTIVE else Int(1024, 32768, log=True, default=16384)
# An execution's output is capped at 7,200 characters (about 1,800 tokens); a task's context reaches about 30k tokens.
TRUNCATE_ABOVE = Int(200, 4000, log=True, default=1500)
PREVIEW_SHARE = Float(0.05, 0.9, default=0.5)
SUMMARIZE_AT = Float(0.05, 0.85, log=True, default=0.85)
KEEP_TOOL_RESULTS = Int(2, 64, log=True, default=64)
MANAGERS = OneOf(
    component(Context, truncate_above=TRUNCATE_ABOVE, preview_share=PREVIEW_SHARE, summarize_at=SUMMARIZE_AT),
    component(Context, keep_tool_results=Int(2, 32, log=True, default=8)),
    component(Context, truncate_above=None, summarize_at=None),
    labels=["auto", "drop_stale", "off"],
)


@tunable(contract=CONTRACT)
def prompt(contract):
    return harness(contract)


@tunable(
    thinking=THINKING,
    truncate_above=TRUNCATE_ABOVE,
    preview_share=PREVIEW_SHARE,
    summarize_at=SUMMARIZE_AT,
    keep_tool_results=KEEP_TOOL_RESULTS,
)
def numbers(thinking, truncate_above, preview_share, summarize_at, keep_tool_results):
    context = Context(truncate_above, preview_share, keep_tool_results, summarize_at)
    return harness(thinking=thinking, context=context)


@tunable(
    contract=CONTRACT,
    tools=TOOL_TEXTS,
    thinking=THINKING,
    todos=Choice([True, False], default=True),
    context=MANAGERS,
)
def full(contract, tools, thinking, todos, context):
    return harness(contract, tools, thinking, context, todos)


SPACES = {"prompt": (prompt, [CONTRACT]), "numbers": (numbers, []), "full": (full, [CONTRACT, TOOL_TEXTS])}


def capped(factory: Any, max_effort: str | None) -> Any:
    """``factory`` searching Haiku 5.5's thinking effort only up to ``max_effort`` (None: every effort; a space that
    does not tune thinking is unchanged), as tau2_harness.py's: a step waits for its slowest candidate, so a study
    with a deadline may leave the top levels out. The record's space lists the levels searched."""
    if max_effort is None or "thinking" not in factory.bindings:
        return factory
    if not ADAPTIVE:
        raise ValueError(f"--max-effort caps Haiku 5.5's effort; {AGENT_MODEL} thinks with a budget of tokens")
    if EFFORTS.index(max_effort) < EFFORTS.index(THINKING.default):
        raise ValueError(f"--max-effort {max_effort} is below the default effort, {THINKING.default}")
    levels = [e for e in THINKING.choices if EFFORTS.index(e) <= EFFORTS.index(max_effort)]
    return factory.with_space({**factory.bindings, "thinking": Choice(levels, default=THINKING.default)})


GOALS = ("best", "budget", "cheapest", "joint", "floor")
FEASIBILITY = 0.8  # the probability with which a goal's limit must hold
# How a goal with a cost objective (joint, floor) searches: reward first, then cost (PrimaryFirst, round 3d's default),
# or every step told both objectives (round 3 and 3b's: "joint").
SCHEDULES = ("reward-first", "joint")
COST_GOALS = ("joint", "floor")
ALSO_DEFAULT = ",".join(ALSO)  # recorded (and tested where they differ): unmet=none, unanchored (round 3b's), anchor


def phases(goal: str, schedule: str, margin: float | None, guard: int | None, cap: float) -> Any:
    """``Search(phases=...)`` for ``goal``: PrimaryFirst for a goal with a cost objective under "reward-first" (its
    non-inferiority ``margin`` in reward, None: the rule's default; the seesaw ``guard``'s k, None: none; phase A's
    ``cap``, a share of the steps), None (no phases) under "joint" or for any other goal."""
    if schedule not in SCHEDULES:
        raise ValueError(f"--schedule is one of {SCHEDULES}, not {schedule!r}")
    if schedule == "joint" or goal not in COST_GOALS:
        return None
    return PrimaryFirst(margin=margin, guard=Seesaw(guard) if guard is not None else None, cap=cap)


NEAR = 0.05  # floor: weighted tokens within 5% of the cheapest front member's tie, and reward breaks the tie
SPLITS = ("train", "dev", "test_normal", "test_challenge")
EVALUATORS = [Reward(name="reward"), TestsPassed(name="tests_passed")]


def goals(
    goal: str,
    budget_ratio: float = 0.8,
    floor_drop: float = 0.05,
    floor: float | None = None,
    floor_gain: float | None = None,
) -> dict[str, Any]:
    """The goal's objectives and constraints, as tau2_harness.py's (limits relative to the baseline; floor's, absolute,
    or with floor_gain a gain over the baseline on the same tasks)."""
    if goal == "best":
        return dict(objectives=Maximize("reward"))
    if goal == "floor":
        if (floor is None) == (floor_gain is None):
            raise ValueError("--goal floor needs --floor (absolute) or --floor-gain (over the baseline), one of them")
        return dict(
            objectives=CheapestOnFront(Minimize(WEIGHTED_TOKENS), Maximize("reward"), within=NEAR),
            constraints=[
                AtLeast("reward", floor) if floor is not None else AtLeast("reward", floor_gain, compare="difference")
            ],
        )
    if goal == "joint":
        return dict(objectives=Pareto(Maximize("reward"), Minimize(WEIGHTED_TOKENS)))
    if goal == "budget":
        return dict(objectives=Maximize("reward"), constraints=[AtMost(WEIGHTED_TOKENS, budget_ratio, compare="ratio")])
    return dict(
        objectives=Minimize(WEIGHTED_TOKENS), constraints=[AtLeast("reward", -floor_drop, compare="difference")]
    )


def make_study(
    space: str,
    goal: str,
    *,
    repetitions: int,
    record: str | None,
    test: str = "test_normal",
    n: int | None = None,
    concurrency: int = 40,
    width: int = Search().width,
    max_effort: str | None = None,
    unmet: str = Selection().unmet,
    schedule: str = SCHEDULES[0],
    margin: float | None = None,
    guard: int | None = 1,
    phase_cap: float = 2 / 3,
    also: tuple[str, ...] = tuple(ALSO),
    **limits: Any,
) -> Study:
    """A study of ``space`` for ``goal``: the search on AppWorld's train tasks, the selection on its dev tasks, the
    test on ``test``'s (``n`` of each split's tasks when given); a space that tunes thinking tries efforts up to
    ``max_effort`` (``capped``); ``unmet``: what a run ships when nothing meets the goal's limit (``Selection.unmet``).
    """
    factory, texts = SPACES[space]
    factory = capped(factory, max_effort)
    rewrite = (
        Block(Reflection(model_config={"model_id": REFLECTION_MODEL, "max_tokens": REFLECTION_MAX_TOKENS}), *texts)
        if texts
        else None
    )
    train, dev, held = cases("train", n), cases("dev", n), cases(test, n)
    return Study(
        agent=factory,
        experiment=Experiment(cases=train + dev + held, evaluators=EVALUATORS),
        task=AppWorldTask(),
        **goals(goal, **limits),
        search=Search(
            blocks=[rewrite] if rewrite else [],
            width=width,
            phases=phases(goal, schedule, margin, guard, phase_cap),
        ),
        repetitions=repetitions,
        split=Split(train=train, validation=dev, test=held),
        max_concurrency=concurrency,  # each running trial holds one AppWorld server
        selection=Selection(
            feasibility=FEASIBILITY, unmet=unmet, also=also if schedule == "reward-first" and goal in COST_GOALS else ()
        ),
        retries=4,
        record=record,
    )


def baseline(split: str, n: int | None, repetitions: int, concurrency: int) -> dict:
    """The default harness on ``split``: every trial, and task and scenario success over the cases."""
    chosen = cases(split, n)
    experiment = Experiment(cases=chosen, evaluators=EVALUATORS)
    run = run_trials(
        prompt, {}, experiment, repetitions=repetitions, max_concurrency=concurrency, task=AppWorldTask(), retries=4
    )
    trials = asyncio.run(run)
    rows = [
        {"case": t.case, "repetition": t.repetition, "status": t.status.value, "error": t.error, **t.metrics}
        for t in trials
    ]
    scored = [r for r in rows if isinstance(r.get("reward"), (int, float))]
    by_scenario: dict[str, list[float]] = {}
    for r in scored:
        by_scenario.setdefault(r["case"].removeprefix("appworld-").rsplit("_", 1)[0], []).append(r["reward"])
    weighted = [sum(r.get(k, 0) * w for k, w in WEIGHTS.items()) for r in scored]
    summary = {
        "model": AGENT_MODEL,
        "split": split,
        "cases": len(chosen),
        "repetitions": repetitions,
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
        "task_success": statistics.fmean(r["reward"] for r in scored) if scored else None,
        "scenario_success": statistics.fmean(all(v == 1.0 for v in vs) for vs in by_scenario.values())
        if by_scenario
        else None,
        "tests_passed": statistics.fmean(r["tests_passed"] for r in scored if "tests_passed" in r) if scored else None,
        "weighted_tokens": statistics.fmean(weighted) if weighted else None,
        "model_calls": statistics.fmean(r.get("model_calls", 0) for r in scored) if scored else None,
        **{f"{k}": statistics.fmean(r.get(k, 0) for r in scored) if scored else None for k in WEIGHTS},
    }
    return {"summary": summary, "trials": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", action="store_true", help="measure the default harness only, on --split")
    parser.add_argument("--split", default="dev", choices=SPLITS, help="--baseline: the tasks measured")
    parser.add_argument("--space", choices=SPACES, default="prompt")
    parser.add_argument("--goal", choices=GOALS, default="best")
    parser.add_argument("--test", default="test_normal", choices=("test_normal", "test_challenge"))
    parser.add_argument(
        "--budget-ratio", type=float, default=0.8, help="budget: the most weighted tokens, per baseline's"
    )
    parser.add_argument("--floor-drop", type=float, default=0.05, help="cheapest: the most reward below the baseline's")
    parser.add_argument("--floor", type=float, default=None, help="floor: the lowest reward allowed (absolute)")
    parser.add_argument("--floor-gain", type=float, default=None, help="floor: the least reward gain over the baseline")
    parser.add_argument("--n", type=int, default=None, help="a sample of this many of each split's tasks")
    parser.add_argument("--repetitions", type=int, default=1, help="of each case, every time a configuration runs")
    parser.add_argument("--iterations", type=int, default=100, help="steps of the search, each of --width candidates")
    parser.add_argument("--width", type=int, default=Search().width, help="candidates each step proposes and measures")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument(
        "--max-effort",
        choices=EFFORTS,
        default=None,
        help="numbers, full: the highest thinking effort tried (Haiku 5.5)",
    )
    parser.add_argument(
        "--unmet",
        choices=UNMET,
        default=Selection().unmet,
        help="when nothing meets the limit: ship the closest, unconfirmed, or none (the baseline, as round 3)",
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default=SCHEDULES[0],
        help="joint, floor: reward first, then cost (round 3d), or both from the first step (round 3 and 3b)",
    )
    parser.add_argument(
        "--margin", type=float, default=None, help="reward-first: phase B's non-inferiority margin in reward (0.02)"
    )
    parser.add_argument(
        "--guard", type=int, default=1, help="reward-first: the seesaw guard's k on validation (-1: no guard)"
    )
    parser.add_argument("--phase-cap", type=float, default=2 / 3, help="reward-first: phase A's most steps, a share")
    parser.add_argument(
        "--also",
        default=ALSO_DEFAULT,
        help="reward-first: rules recorded beside the selection and tested where they differ (comma-separated; '')",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="--baseline: the result's file; else: the records' directory"
    )
    args = parser.parse_args()
    if args.baseline:
        result = baseline(args.split, args.n, args.repetitions, args.concurrency)
        args.out.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(result["summary"], indent=1))  # noqa: T201
    else:
        record = args.out / f"appworld-harness-{args.space}-{args.goal}-{args.test}-{time.strftime('%Y%m%d-%H%M%S')}"
        study = make_study(
            args.space,
            args.goal,
            repetitions=args.repetitions,
            record=str(record),
            test=args.test,
            n=args.n,
            concurrency=args.concurrency,
            width=args.width,
            max_effort=args.max_effort,
            unmet=args.unmet,
            schedule=args.schedule,
            margin=args.margin,
            guard=None if args.guard < 0 else args.guard,
            phase_cap=args.phase_cap,
            also=tuple(a for a in args.also.split(",") if a),
            budget_ratio=args.budget_ratio,
            floor_drop=args.floor_drop,
            floor=args.floor,
            floor_gain=args.floor_gain,
        )
        study.optimize(iterations=args.iterations, patience=args.patience)
