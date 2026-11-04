"""Tune a Strands harness (``create_harness``) as a τ²-bench customer-service agent (airline; or retail, telecom or
banking_knowledge with ``TAU2_DOMAIN``): three search spaces by four goals.

    BYPASS_TOOL_CONSENT=true python examples/optimization/tau2_harness.py --space prompt --goal best --rotate
    python examples/optimization/tau2_harness.py --baseline    # the baseline alone, on every task

The agent is ``create_harness``'s, on Claude Haiku 5.5 (or Haiku 4.5 with ``TAU2_AGENT_MODEL``), with τ²'s
instructions and the domain's policy as its instructions. The harness's defaults stay: its contract (the system prompt
before the instructions), the todos plugin, thinking as ``effort="auto"`` gives the model (Haiku 5.5: adaptive, effort
high; Haiku 4.5: a budget), prompt caching, and ``context_manager="auto"``. Off is what a τ² conversation cannot use:
memory, sessions and skills (they would carry over between tasks), the environment plugin (it gives today's date; τ²'s
worlds have their own, e.g. May 2024 for airline), the built-in tools, and background tasks (τ² executes every tool
call itself, in turn). ``max_tokens`` is bounded (Haiku 5.5: 32,768; Haiku 4.5: the thinking budget plus 4,096) rather
than the model's own, because Bedrock reserves ``max_tokens`` of quota for every request.

Search spaces:

    prompt   the harness's contract (text: reflection)
    numbers  the thinking (Haiku 5.5: its effort; Haiku 4.5: its budget) and the context manager's settings, over ranges
             τ²'s conversations reach (TPE)
    full     the contract, the tools' descriptions, the thinking, the todos plugin, and the context manager:
             create_harness's, one that drops stale tool results, or none, each with its settings (rounds: Nested)

--max-effort caps the thinking effort the numbers and full spaces try (Haiku 5.5; e.g. xhigh leaves out max).

Goals, on reward (τ²'s, 1 or 0 per conversation: its mean is pass^1) and weighted tokens (the agent's tokens per
conversation, each kind weighted by its price relative to an uncached input token; the context manager's summaries
included):

    best      the highest reward
    budget    the highest reward within --budget-ratio of the baseline's weighted tokens (or --budget per conversation)
    cheapest  the fewest weighted tokens with a reward at most --floor-drop below the baseline's (or at least --floor)
    joint     the highest reward and the fewest weighted tokens together (the Pareto set)
    floor     the fewest weighted tokens with a reward of at least --floor (absolute), taken from the feasible front of
              (reward, weighted tokens): within 5% of the cheapest front member's tokens counts as a tie, and the tie
              goes to the higher reward; the finalists are front members and the incumbent is held on the front
              (``CheapestOnFront``)

The goals with a cost objective (joint, floor) search reward first, then cost (--schedule reward-first, the
default; ``PrimaryFirst``): phase A is a max-reward search until it stops gaining (2 steps in a row with no candidate
beating its incumbent with probability 0.8, paired over the train tasks, once the reward is within 0.25 of 1) or until
a floor is met with confidence, at most --phase-cap of the steps; phase B spends the rest lowering the weighted tokens
from phase A's result, A*, keeping its reward (paired over the same tasks: within --margin for joint, the floor for
floor).
The selection is A* or a cheaper candidate that keeps its reward on validation and passes the seesaw guard (--guard: it
loses at most k of the tasks A* solves on every repetition); for joint, the validated front is tested as before, with
the pick among it. --also: what other rules would ship (unmet=none, the unanchored rules, A* itself), recorded and
tested where they differ. --schedule joint: every step told both objectives.

Every decision is made with the study's model of each goal metric; a limit must hold with probability 0.8. When no
finalist (nor the baseline) meets it on validation, the run ships the one closest to meeting it (the highest
probability, then the least shortfall, then the goal's own order), marked unconfirmed in the record and the result;
``--unmet none`` ships nothing instead (the baseline). The domain's
tasks (all of τ²'s, the train and test tasks together) are dealt by --split into shares: the search runs on the train
share, each case --repetitions times, until its best has not changed for --patience iterations (at most --iterations);
the finalists and the baseline are measured on the validation share, and the selection is made there; last, the
selection and the baseline are measured on the test share, which nothing before saw (result.json and test.jsonl in the
study's record). With --rotate the study runs once per test share (4 runs for the default shares), each testing a
different part of the tasks, so every task is tested exactly once (run-k/ in the record, one per run).

(Reflection's agent runs a shell over the traces: run the text spaces where model-written shell commands are
acceptable, such as a VM or a container.)
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import boto3
import strands.models.bedrock as strands_bedrock
from strands import ModelRetryStrategy
from strands.experimental.context_manager import ContextManager, Offload
from strands_harness import create_harness
from strands_harness.models import resolve_model
from strands_harness.prompt import HARNESS_CONTRACT
from tau2_workload import AGENT_INSTRUCTION, Reductions, Reward, Tau2Task, cases, policy, tools

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

DOMAIN = os.environ.get("TAU2_DOMAIN", "airline")
# Each run picks its models' cross-region profiles (us. or global.) and its region (AWS_REGION), so that runs spread
# over Bedrock's quotas.
AGENT_MODEL = os.environ.get("TAU2_AGENT_MODEL", "global.anthropic.claude-haiku-5-5")
ADAPTIVE = "haiku-5" in AGENT_MODEL  # Haiku 5.5 thinks adaptively, at an effort; Haiku 4.5 with a budget
USER_MODEL = os.environ.get("TAU2_USER_MODEL", "bedrock/us.anthropic.claude-sonnet-4-6")
REFLECTION_MODEL = os.environ.get("TAU2_REFLECTION_MODEL", "us.anthropic.claude-sonnet-5-5")
REFLECTION_MAX_TOKENS = 32000  # a full space's texts are written in one turn: over the model's default output

# τ²'s own agent's instructions and the domain's policy: part of the benchmark, never tuned.
INSTRUCTIONS = f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n<policy>\n{policy(DOMAIN)}\n</policy>"
TOOLS = tools(DOMAIN)

# Each kind of token, weighted by its price relative to an uncached input token (Bedrock's prices for Claude Haiku 4.5
# and 5.5, the same ratios).
WEIGHTS = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_write_tokens": 1.25, "cache_read_tokens": 0.1}
WEIGHTED_TOKENS = Derived(
    "weighted_tokens", lambda m: sum(m[k] * weight for k, weight in WEIGHTS.items()), range=(0, None)
)


def patient() -> ModelRetryStrategy:
    """A model call Bedrock throttles waits for capacity (up to 20 attempts, backing off to 4 minutes: about an hour)
    rather than failing its conversation: a conversation that fails on throttling measures nothing, and long ones fail
    most, which would bias what is measured. Strands' default is 6 attempts. One per agent: it counts attempts."""
    return ModelRetryStrategy(max_attempts=20, initial_delay=4, max_delay=240)


class OneSession:
    """``boto3`` as Strands' ``BedrockModel`` sees it, with one session for the whole process.

    ``BedrockModel()`` calls ``boto3.Session()`` for each model it builds, and a study builds one per conversation (the
    agent's, and each rewrite's). Each new session reads the AWS config files, parses botocore's data files, and
    resolves its own credentials: for a role profile, an AssumeRole through the instance metadata service. With
    hundreds of conversations at once, that ran the process out of file descriptors and the metadata service out of
    capacity, so trials failed as if the agent had (``ProfileNotFound``, ``CredentialRetrievalError``, ``Too many open
    files``), and it held the GIL most of the time.

    Here every model's client comes from one session (the environment's profile and region), so credentials are
    resolved and data files read once, and botocore refreshes the credentials. A session is not thread-safe, so clients
    are made under a lock; the clients themselves are thread-safe. What is measured is unchanged: the same model, region
    and account.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._lock = threading.RLock()

    def Session(self, *args: Any, **kwargs: Any) -> Any:  # noqa: N802 - boto3's name
        return boto3.Session(*args, **kwargs) if args or kwargs else self

    @property
    def region_name(self) -> str | None:
        return self._shared().region_name

    def client(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._shared().client(*args, **kwargs)

    def _shared(self) -> Any:
        with self._lock:
            if self._session is None:
                self._session = boto3.Session()
            return self._session

    def __getattr__(self, name: str) -> Any:
        return getattr(boto3, name)


strands_bedrock.boto3 = OneSession()  # type: ignore[assignment]


ADAPTIVE_MAX_TOKENS = 32768  # Bedrock reserves max_tokens of quota per request; create_harness's 128k would exhaust it

# A thinking block is bound to the conversation before it. When an earlier message changes after the block was written
# (a context strategy drops or cuts an earlier tool result; a reasoning block is taken off the end of a turn), Bedrock
# may reject the next call ("Invalid `signature` in `thinking` block. The block is bound to a different conversation"),
# and the trial ends as the agent's failure (in our τ² studies, up to a few percent of a study's trials). A mismatched
# block is dropped instead (Anthropic's own setting for it, behind its beta header): a call whose prefix is unchanged
# is sent and answered as before.
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


# create_harness's effort "off" sends no thinking field, and Haiku 5.5 on Bedrock thinks when the field is absent (its
# default: adaptive, the block's text omitted, the block signed). So "off" thought, and its signed blocks went without
# the binding control (``bound`` binds an adaptive request): trials of thinking-"off" candidates could be refused, and
# "off" was the model's default thinking. Off is sent as off instead: a model that does not think writes no block to
# bind. (Bedrock takes no block_binding with it:
# "thinking.disabled.block_binding: Extra inputs are not permitted".)
DISABLED = {"type": "disabled"}


def disabled(model: Any) -> Any:
    """``model`` not thinking (Haiku 5.5's effort "off")."""
    fields = dict(model.config.get("additional_request_fields") or {})
    model.update_config(additional_request_fields={**fields, "thinking": dict(DISABLED)})
    return model


def haiku(thinking: str | int) -> Any:
    """Haiku as create_harness builds it (thinking, prompt caching). Haiku 5.5 thinks adaptively at the effort
    ``thinking`` (create_harness's levels: "off", "low", ..., "max"; its "auto" is "high"; "off" sent as disabled), a
    thinking block whose earlier conversation changed dropped (``bound``); Haiku 4.5 thinks with a budget, ``thinking``
    tokens a call."""
    if ADAPTIVE:
        model = resolve_model(f"bedrock/{AGENT_MODEL}", AGENT_MODEL, thinking, caching=True)
        model.update_config(max_tokens=ADAPTIVE_MAX_TOKENS)
        return disabled(model) if thinking == "off" else bound(model)
    model = resolve_model(f"bedrock/{AGENT_MODEL}", AGENT_MODEL, "auto", caching=True)
    budget = {"thinking": {"type": "enabled", "budget_tokens": thinking}}
    model.update_config(max_tokens=thinking + 4096, additional_request_fields=budget)
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

    truncate_above: int | None = 1500  # tool results over this many tokens are cut to a preview
    preview_share: float = 0.5  # the preview's size, as a share of the threshold (always below it)
    keep_tool_results: int | None = None  # tool results beyond the most recent this many are dropped
    summarize_at: float | None = 0.85  # the share of the context window at which older messages are summarized

    def build(self, model: Any) -> tuple[Any, Metered | None]:
        """The context manager (False for none) and its metered summarizer."""
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


def harness(
    contract: str = HARNESS_CONTRACT,
    tools: Any = TOOLS,
    thinking: str | int | None = None,
    context: Context = AUTO,
    todos: bool = True,
) -> Any:
    """The agent every space tunes; what a space does not tune stays at create_harness's default."""
    model = haiku(THINKING.default if thinking is None else thinking)
    manager, summarizer = context.build(model)
    agent = create_harness(
        model=model,
        system_prompt=f"{contract}\n\n{INSTRUCTIONS}",  # as create_harness assembles it: contract, instructions
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
    about="the harness's behavioral contract, given before τ²'s instructions and the domain's policy (both fixed)",
)
TOOL_TEXTS = tool_descriptions(*TOOLS, max_chars=1500)  # one text per tool, starting from τ²'s own description
EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")  # create_harness's levels for Claude; "high" is its "auto"
if ADAPTIVE:
    THINKING = Choice(list(EFFORTS), default="high")
else:  # tokens of thinking per model call
    THINKING = Int(1024, 32768, log=True, default=16384)
WINDOW = haiku(THINKING.default).context_window_limit  # Haiku 5.5: 1,000,000 tokens; Haiku 4.5: 200,000
# Over ranges a τ² conversation reaches. Its tool results: airline's run to about 2,000 tokens (median 180); banking's
# knowledge-base searches return 3,000 to 8,000, so create_harness's 1,500 previews nearly every one. Its context:
# airline's reaches about 20,000 tokens, banking's about 130,000 (median 43,000); the policy and tools alone are about
# 6,000 (airline). Summarizing can start from 8,000 tokens.
TRUNCATE_ABOVE = Int(200, 16000, log=True, default=1500)
PREVIEW_SHARE = Float(0.05, 0.9, default=0.5)  # 750 of create_harness's 1,500
SUMMARIZE_AT = Float(8000 / WINDOW, 0.85, log=True, default=0.85)
KEEP_TOOL_RESULTS = Int(2, 256, log=True, default=256)  # 256 keeps them all: banking's longest has 136 model calls
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


# Each space: its factory and the texts reflection rewrites.
SPACES = {"prompt": (prompt, [CONTRACT]), "numbers": (numbers, []), "full": (full, [CONTRACT, TOOL_TEXTS])}


def capped(factory: Any, max_effort: str | None) -> Any:
    """``factory`` searching Haiku 5.5's thinking effort only up to ``max_effort`` (None: every effort; a space that
    does not tune thinking is unchanged). A step waits for its slowest candidate, and a candidate at effort "max" can
    run for hours (τ³ banking), so a study with a deadline may leave the top levels out. The record's space lists the
    levels searched."""
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
# How a goal with a cost objective (joint, floor) searches: reward first, then cost (PrimaryFirst, the default), or
# every step told both objectives ("joint").
SCHEDULES = ("reward-first", "joint")
COST_GOALS = ("joint", "floor")
ALSO_DEFAULT = ",".join(ALSO)  # recorded (and tested where they differ): unmet=none, unanchored, anchor


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


def goals(
    goal: str, budget: float | None = None, floor: float | None = None, budget_ratio: float = 0.8, floor_drop=0.05
) -> dict:
    """The goal's objectives and constraints; a limit is relative to the baseline unless an absolute one is given."""
    if goal == "best":
        return dict(objectives=Maximize("reward"))
    if goal == "floor":
        if floor is None:
            raise ValueError("--goal floor needs --floor: the lowest reward allowed (absolute, e.g. 0.95)")
        return dict(
            objectives=CheapestOnFront(Minimize(WEIGHTED_TOKENS), Maximize("reward"), within=NEAR),
            constraints=[AtLeast("reward", floor)],
        )
    if goal == "joint":
        return dict(objectives=Pareto(Maximize("reward"), Minimize(WEIGHTED_TOKENS)))
    if goal == "budget":
        limit = (
            AtMost(WEIGHTED_TOKENS, budget_ratio, compare="ratio")
            if budget is None
            else AtMost(WEIGHTED_TOKENS, budget)
        )
        return dict(objectives=Maximize("reward"), constraints=[limit])
    limit = AtLeast("reward", -floor_drop, compare="difference") if floor is None else AtLeast("reward", floor)
    return dict(objectives=Minimize(WEIGHTED_TOKENS), constraints=[limit])


def make_study(
    space: str,
    goal: str,
    *,
    repetitions: int,
    record: str | None,
    split: Split | None = None,
    n: int | None = None,
    concurrency: int = 150,
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
    """A study of ``space`` for ``goal`` on ``n`` of the domain's tasks (all of them by default), dealt by ``split``
    (None: the search and the selection on every task, nothing tested); a space that tunes thinking tries efforts up
    to ``max_effort`` (``capped``); ``unmet``: what a run ships when nothing meets the goal's limit
    (``Selection.unmet``)."""
    factory, texts = SPACES[space]
    factory = capped(factory, max_effort)
    rewrite = (
        Block(Reflection(model_config={"model_id": REFLECTION_MODEL, "max_tokens": REFLECTION_MAX_TOKENS}), *texts)
        if texts
        else None
    )
    return Study(
        agent=factory,
        experiment=Experiment(
            cases=cases(DOMAIN, "base", n), evaluators=[Reward(name="reward"), Reductions(name="reductions")]
        ),
        task=Tau2Task(DOMAIN, user_llm=USER_MODEL),  # one τ² conversation per trial
        **goals(goal, **limits),
        search=Search(
            blocks=[rewrite] if rewrite else [],
            width=width,
            phases=phases(goal, schedule, margin, guard, phase_cap),
        ),
        repetitions=repetitions,
        split=split,
        max_concurrency=concurrency,  # every trial of a candidate at once, up to what one Bedrock quota holds
        selection=Selection(
            feasibility=FEASIBILITY, unmet=unmet, also=also if schedule == "reward-first" and goal in COST_GOALS else ()
        ),
        retries=4,
        record=record,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", choices=SPACES, default="prompt")
    parser.add_argument("--goal", choices=GOALS, default="best")
    parser.add_argument("--baseline", action="store_true", help="measure only the baseline, on every task")
    parser.add_argument(
        "--budget-ratio", type=float, default=0.8, help="budget: the most weighted tokens, per baseline's"
    )
    parser.add_argument("--floor-drop", type=float, default=0.05, help="cheapest: the most reward below the baseline's")
    parser.add_argument("--budget", type=float, help="budget: an absolute limit, weighted tokens per conversation")
    parser.add_argument("--floor", type=float, help="cheapest, floor: an absolute limit, the lowest reward allowed")
    parser.add_argument("--repetitions", type=int, default=3, help="of each case, every time a configuration runs")
    parser.add_argument(
        "--split", default="0.5,0.25,0.25", help="the tasks' shares: train, validation, test (summing to 1)"
    )
    parser.add_argument("--rotate", action="store_true", help="one run per test share: every task tested once")
    parser.add_argument("--seed", type=int, default=0, help="how the shares deal the tasks")
    parser.add_argument("--iterations", type=int, default=100, help="steps of the search, each of --width candidates")
    parser.add_argument("--width", type=int, default=Search().width, help="candidates each step proposes and measures")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--tasks", type=int, default=None, help="a sample of this many of the domain's tasks")
    parser.add_argument("--concurrency", type=int, default=150, help="trials at once")
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
        help="when nothing meets the limit: ship the closest, unconfirmed, or none (the baseline)",
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default=SCHEDULES[0],
        help="joint, floor: reward first, then cost (the default), or both from the first step",
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
    parser.add_argument("--out", default="studies")
    args = parser.parse_args()
    train, validation, test = (float(x) for x in args.split.split(","))
    split = None if args.baseline else Split(train, validation, test, rotate=args.rotate, seed=args.seed)
    name = "baseline" if args.baseline else f"{args.space}-{args.goal}"
    record = f"{args.out}/tau2-harness-{DOMAIN}-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    study = make_study(
        args.space,
        "best" if args.baseline else args.goal,
        repetitions=args.repetitions,
        record=record,
        split=split,
        n=args.tasks,
        concurrency=args.concurrency,
        width=args.width,
        max_effort=args.max_effort,
        unmet=args.unmet,
        schedule=args.schedule,
        margin=args.margin,
        guard=None if args.guard < 0 else args.guard,
        phase_cap=args.phase_cap,
        also=tuple(a for a in args.also.split(",") if a),
        budget=args.budget,
        floor=args.floor,
        budget_ratio=args.budget_ratio,
        floor_drop=args.floor_drop,
    )
    if args.baseline:
        study.optimize(iterations=0)
    else:
        study.optimize(iterations=args.iterations, patience=args.patience)
