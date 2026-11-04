"""The τ²-bench workload: a Strands agent as τ²'s customer-service agent, τ²'s tasks as cases, and τ²'s reward as the
score. Not a study: see tau2_airline.py (one study) and tau2_airline_steps.py (a ladder of them).

τ² runs each conversation. Its simulated user talks to the agent, and it executes every tool call the agent makes
itself, because its evaluator replays those calls against a fresh database. So the agent never runs a τ² tool: a hook
interrupts each call, τ² executes it, and τ²'s result resumes the agent. Everything else (system prompt, model,
conversation manager, hooks) is the agent's own, built by the study's factory.

    pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench"
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from functools import cache
from typing import Any

import regex
import tau2.evaluator.evaluator_nl_assertions as nl_assertions
from loguru import logger
from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry, MessageAddedEvent
from strands.tools.tools import PythonAgentTool
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import AGENT_INSTRUCTION
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.environment.toolkit import ToolType, get_tool_types
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.knowledge.retrievers.grep_retriever import GrepRetriever
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner import build_environment, build_user, get_tasks, load_task_splits
from tau2.runner.build import _derive_read_log_allowlist  # τ²'s own banking setup (tau2.runner.build._build_env_kwargs)

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import Study, TrialInvalid
from strands_evals.optimization.trials import TrialStatus, run_trials
from strands_evals.types.evaluation import EvaluationOutput

__all__ = [
    "AGENT_INSTRUCTION",
    "ConfirmWrites",
    "DirectWrites",
    "Reductions",
    "Reward",
    "Tau2Task",
    "cases",
    "env_kwargs",
    "held_out",
    "policy",
    "task_ids",
    "tools",
]

logger.remove()  # τ² logs every message it routes
logger.add(sys.stderr, level="ERROR")

# τ²'s simulated user: the same for every candidate, so only the agent varies (τ²'s leaderboard uses gpt-4.1).
USER_LLM = "bedrock/us.anthropic.claude-sonnet-4-6"
# The simulated user's and the judge's sampling temperature: 0 (for the default user, Sonnet 4.6) unless the
# environment says otherwise. Sonnet 5.5 refuses any temperature ("`temperature` is deprecated for this model"), so a
# study whose user is Sonnet 5.5 sets TAU2_USER_TEMPERATURE=none: no temperature is sent and the model samples as it
# does by default. The judge (banking's one natural-language assertion, retail's) follows the user's setting unless
# TAU2_JUDGE_TEMPERATURE is given.
USER_TEMPERATURE = os.environ.get("TAU2_USER_TEMPERATURE", "0")
JUDGE_TEMPERATURE = os.environ.get("TAU2_JUDGE_TEMPERATURE", USER_TEMPERATURE)


def _temperature(setting: str) -> dict[str, float]:
    """LiteLLM's ``temperature`` argument for ``setting``: a number, or none at all for "none"."""
    return {} if setting.strip().lower() == "none" else {"temperature": float(setting)}


NO_REPLY = "(no reply)"
# τ³ banking_knowledge searches its knowledge base with this retrieval variant: bm25_grep (BM25 search and grep) needs
# no embedding service; τ²'s default, alltools, includes OpenAI and OpenRouter embeddings.
RETRIEVAL_VARIANT = os.environ.get("TAU2_RETRIEVAL_VARIANT", "bm25_grep")


class _LenientJson:
    """json for τ²'s judge of natural-language assertions: its reply may be fenced, or keep the trailing comma of the
    prompt's example."""

    @staticmethod
    def loads(text: str) -> Any:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            body = text[text.find("{") : text.rfind("}") + 1]
            return json.loads(re.sub(r",\s*([}\]])", r"\1", body))


# τ²'s judge of natural-language assertions (part of retail's reward) is gpt-4.1; here it is Claude on Bedrock, like the
# simulated user, and retried when Bedrock throttles.
nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = os.environ.get(
    "TAU2_JUDGE_MODEL", os.environ.get("TAU2_USER_MODEL", USER_LLM)
)
nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {**_temperature(JUDGE_TEMPERATURE), "num_retries": 8, "max_tokens": 4096}
nl_assertions.json = _LenientJson
NUDGE = "(Your last turn contained no message. Reply to the customer or call a tool.)"

# τ³ banking's grep tool matches the agent's pattern against every document of the knowledge base with re, whose
# matcher backtracks without limit and never releases the GIL. A pattern such as `(\w+ ?)*Account.*...` never finishes
# on a 1,246-character document, and every thread of the study waits behind it: a study can freeze for hours. So the
# same search runs on the regex module in its VERSION0 mode (re's syntax and matching), which releases the GIL while
# it matches and gives up after GREP_SECONDS: the agent then gets the tool's error. The pattern is compiled as τ²'s
# does it (re, a pattern re rejects searched as literal text), and the documents, scores and order are τ²'s. On 21,769
# distinct patterns agents gave the tool in our studies (109k of 255k calls), every document's score was the same as
# re's; re took at most 1.7 s on any of them (median 14 ms), so 30 s ends only a runaway.
GREP_SECONDS = float(os.environ.get("TAU2_GREP_SECONDS", "30"))


class GrepTimeout(Exception):
    """A grep pattern that did not finish within GREP_SECONDS."""


def _grep_retrieve(self: Any, input_data: dict[str, Any], state: dict[str, Any]) -> list[tuple[str, float]]:
    """tau2.knowledge.retrievers.grep_retriever.GrepRetriever.retrieve, on the regex module with a time limit."""
    query = input_data[self.query_key]
    doc_content_map = state[self.content_state_key]
    if not query.strip():
        return []
    try:
        re.compile(query, 0 if self.case_sensitive else re.IGNORECASE)
        source = query
    except re.error:
        source = re.escape(query)
    try:
        pattern = regex.compile(source, regex.VERSION0 | (0 if self.case_sensitive else regex.IGNORECASE))
    except regex.error:  # a pattern re takes and regex does not: τ²'s own search
        return _RE_RETRIEVE(self, input_data, state)
    deadline = time.monotonic() + GREP_SECONDS
    results = []
    for doc_id, content in doc_content_map.items():
        try:
            matches = pattern.findall(content, concurrent=True, timeout=max(deadline - time.monotonic(), 0.001))
        except TimeoutError:
            raise GrepTimeout(
                f"the pattern did not finish within {GREP_SECONDS:g} seconds (too much backtracking); use a simpler one"
            ) from None
        if matches:
            results.append((doc_id, float(len(matches))))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[: self.top_k]


_RE_RETRIEVE = getattr(GrepRetriever.retrieve, "re_retrieve", GrepRetriever.retrieve)  # τ²'s own (also if reloaded)
_grep_retrieve.re_retrieve = _RE_RETRIEVE  # type: ignore[attr-defined]
GrepRetriever.retrieve = _grep_retrieve


def env_kwargs(domain: str, task: Any = None) -> dict[str, Any]:
    """What τ² passes the domain's environment constructor, as its own runner does: for banking_knowledge, the retrieval
    variant and, for a task, the task and the discoverable reads its golden trajectory requires (other reads are then
    not logged, so verifying a balance does not change the database the reward compares); nothing for the others."""
    if domain != "banking_knowledge":
        return {}
    kwargs: dict[str, Any] = {"retrieval_variant": RETRIEVAL_VARIANT}
    if task is not None:
        kwargs |= {"task": task, "read_log_allowlist": _derive_read_log_allowlist(task)}
    return kwargs


@cache
def _environment(domain: str) -> Any:
    return build_environment(domain, env_kwargs=env_kwargs(domain))


def policy(domain: str) -> str:
    """The domain's policy, which the agent must follow (fixed: it is part of the benchmark)."""
    return _environment(domain).get_policy()


def tools(domain: str) -> list[PythonAgentTool]:
    """The domain's tools as the agent sees them: their names, descriptions and inputs. τ² executes every call."""

    def executed_by_tau2(tool_use: dict[str, Any], **_: Any) -> dict[str, Any]:
        raise RuntimeError(f"{tool_use['name']} ran outside τ²: run the agent with Tau2Task")

    out = []
    for t in _environment(domain).get_tools():
        fn = t.openai_schema["function"]
        spec = {"name": fn["name"], "description": fn["description"], "inputSchema": {"json": fn["parameters"]}}
        out.append(PythonAgentTool(fn["name"], spec, executed_by_tau2))
    return out


def write_tools(domain: str) -> frozenset[str]:
    """The tools that change the customer's records."""
    return frozenset(n for n, kind in get_tool_types(_environment(domain).tools).items() if kind == ToolType.WRITE)


class DirectWrites(HookProvider):
    """Write calls go to τ² as the agent makes them."""

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        pass


class ConfirmWrites(HookProvider):
    """Each write call is returned unexecuted once, with ``message``; the agent re-checks it and calls it again."""

    def __init__(self, message: str, domain: str = "airline") -> None:
        self.message, self.writes = message, write_tools(domain)
        self._reviewed: set[str] = set()
        self._held: set[str] = set()  # by toolUseId: a call resumed after an interrupt gets the same decision

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        use = event.tool_use
        if use["name"] not in self.writes:
            return
        signature = json.dumps([use["name"], use["input"]], sort_keys=True)
        if use["toolUseId"] in self._held or signature not in self._reviewed:
            self._held.add(use["toolUseId"])
            self._reviewed.add(signature)
            event.cancel_tool = self.message


class _DeferToTau2(HookProvider):
    """Interrupts each τ² tool call; on resume, the call returns the result τ² gave."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        use = event.tool_use
        if use["name"] not in self.names or event.cancel_tool:
            return
        result = event.interrupt(f"tau2:{use['toolUseId']}", reason={**use})
        text, failed = result["content"] or "", bool(result["error"])

        def returns(tool_use: dict[str, Any], **_: Any) -> dict[str, Any]:
            status = "error" if failed else "success"
            return {"toolUseId": tool_use["toolUseId"], "status": status, "content": [{"text": text}]}

        event.selected_tool = PythonAgentTool(use["name"], event.selected_tool.tool_spec, returns)


class _NoThinkingTail(HookProvider):
    """Bedrock rejects an assistant message that ends in a reasoning block, which a turn occasionally does."""

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(MessageAddedEvent, self._on_message)

    def _on_message(self, event: MessageAddedEvent) -> None:
        content = event.message.get("content") or []
        if event.message.get("role") != "assistant" or not content or "reasoningContent" not in content[-1]:
            return
        while content and "reasoningContent" in content[-1] and any("reasoningContent" not in b for b in content):
            content.pop()
        if all("reasoningContent" in b for b in content):
            content.append({"text": NO_REPLY})


class _Bridge(HalfDuplexAgent[None]):
    """τ²'s agent interface over a Strands agent: τ² tool calls leave as interrupts and come back as their results."""

    def __init__(self, agent: Any, tau2_tools: list[Any], domain_policy: str) -> None:
        super().__init__(tools=tau2_tools, domain_policy=domain_policy)
        self.agent = agent
        agent.hooks.add_hook(_DeferToTau2({t.name for t in tau2_tools}))
        agent.hooks.add_hook(_NoThinkingTail())
        self._pending: list[Any] = []

    def get_init_state(self, message_history: list[Message] | None = None) -> None:
        if message_history:  # τ² opens with a scripted greeting; Bedrock needs a user turn first
            text = "\n".join(m.content or "" for m in message_history if isinstance(m, AssistantMessage))
            self.agent.messages = [
                {"role": "user", "content": [{"text": "(customer connected)"}]},
                {"role": "assistant", "content": [{"text": text}]},
            ]

    def generate_next_message(self, message: ValidAgentInputMessage, state: None) -> tuple[AssistantMessage, None]:
        if isinstance(message, UserMessage):
            prompt: Any = message.content or ""
        else:
            results = {m.id: m for m in (message.tool_messages if isinstance(message, MultiToolMessage) else [message])}
            prompt = [
                {
                    "interruptResponse": {
                        "interruptId": i.id,
                        "response": {
                            "content": results[i.reason["toolUseId"]].content,
                            "error": results[i.reason["toolUseId"]].error,
                        },
                    }
                }
                for i in self._pending
            ]
        result = self.agent(prompt)
        for _ in range(2):
            if result.stop_reason == "interrupt" or self._text(result):
                break
            result = self.agent(NUDGE)
        if result.stop_reason == "interrupt":
            self._pending = list(result.interrupts)
            calls = [
                ToolCall(id=i.reason["toolUseId"], name=i.reason["name"], arguments=i.reason["input"])
                for i in self._pending
            ]
            return AssistantMessage(role="assistant", content=None, tool_calls=calls), state
        self._pending = []
        text = self._text(result)
        if not text:
            raise RuntimeError("the agent gave no reply after two nudges")
        return AssistantMessage(role="assistant", content=text), state

    @staticmethod
    def _text(result: Any) -> str:
        text = "".join(b.get("text", "") for b in result.message.get("content", [])).strip()
        return "" if text == NO_REPLY else text


class Tau2Run:
    """A finished conversation; its text is how it ended and the agent's last message. (A plain object: Strands Evals
    keeps a task's output as it is.)"""

    def __init__(self, simulation: SimulationRun, domain: str, reductions: int = 0) -> None:
        self.simulation, self.domain, self.reductions = simulation, domain, reductions

    def __str__(self) -> str:
        said = [m.content for m in self.simulation.messages if isinstance(m, AssistantMessage) and m.content]
        reason = getattr(self.simulation.termination_reason, "value", self.simulation.termination_reason)
        return f"{reason} after {len(self.simulation.messages)} messages: {(said or [''])[-1][:300]}"


def _user_side(exc: BaseException) -> bool:
    """Whether the failure was τ²'s simulated user's (its LLM calls go through litellm), not the agent's."""
    for e in (exc, exc.__cause__, exc.__context__):
        if e is None:
            continue
        if type(e).__module__.startswith(("litellm", "httpx", "openai")):
            return True
        if isinstance(e, ValueError) and str(e).startswith("UserMessage must have"):
            return True  # the simulated user produced an empty turn
    return False


class Tau2Task:
    """The study's task: one τ² conversation between the agent and τ²'s simulated user, on the case's τ² task."""

    def __init__(self, domain: str = "airline", user_llm: str = USER_LLM, max_steps: int = 200, max_errors: int = 10):
        self.domain, self.user_llm, self.max_steps, self.max_errors = domain, user_llm, max_steps, max_errors

    def __call__(self, agent: Any, case: Case) -> dict[str, Any]:
        task = _task(self.domain, case.metadata["task_id"])
        env = build_environment(self.domain, env_kwargs=env_kwargs(self.domain, task))
        bridge = _Bridge(agent, env.get_tools(), env.get_policy())
        try:
            llm_args = {
                **_temperature(USER_TEMPERATURE),
                "num_retries": 8,  # litellm backs off when Bedrock throttles
                "max_tokens": 2048,  # Bedrock reserves max_tokens of quota for every request
                # The conversation so far is cached, as the agent's is: cache reads are cheap, not held against quota.
                "cache_control_injection_points": [
                    {"location": "message", "role": "system"},
                    {"location": "message", "index": -1},
                ],
            }
            user = build_user("user_simulator", env, task, llm=self.user_llm, llm_args=llm_args)
            orchestrator = Orchestrator(
                domain=self.domain,
                agent=bridge,
                user=user,
                environment=env,
                task=task,
                max_steps=self.max_steps,
                max_errors=self.max_errors,
            )
            simulation = orchestrator.run()
        except Exception as exc:
            if _user_side(exc):
                raise TrialInvalid(f"τ²'s simulated user failed: {type(exc).__name__}: {exc}") from exc
            raise
        if simulation.termination_reason in (TerminationReason.USER_ERROR, TerminationReason.INFRASTRUCTURE_ERROR):
            raise TrialInvalid(f"τ² ended the conversation: {simulation.termination_reason.value}")
        # How many times the agent's context strategies changed its context, if they count it (tau2_harness.py).
        reductions = int(agent.state.get("context_reductions") or 0)
        return {"output": Tau2Run(simulation, self.domain, reductions)}


@cache
def _task(domain: str, task_id: str) -> Any:
    return get_tasks(domain, task_ids=[task_id])[0]


class Reward(Evaluator):
    """τ²'s reward for the conversation: 1 when the database ends as the task requires and the agent told the user
    what it had to (airline), else 0."""

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        run: Tau2Run = case.actual_output
        task = _task(run.domain, case.metadata["task_id"])
        info = evaluate_simulation(
            simulation=run.simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=run.domain,
            env_kwargs=env_kwargs(run.domain, task),
        )
        parts = ", ".join(f"{getattr(k, 'value', k)}={v:g}" for k, v in (info.reward_breakdown or {}).items())
        return [EvaluationOutput(score=info.reward, test_pass=info.reward == 1.0, reason=parts or str(info.info))]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return await asyncio.to_thread(self.evaluate, case)  # τ²'s replay of the conversation takes a while


class Reductions(Evaluator):
    """How many times the agent's context strategies changed its context in the conversation: whether a context
    setting acted at all. Not a score."""

    score_range = (0, None)

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        n = case.actual_output.reductions
        return [EvaluationOutput(score=float(n), test_pass=True, reason=f"{n} reductions")]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)


def task_ids(domain: str, split: str) -> list[str]:
    """The ids of the domain's tasks in τ²'s ``split`` ("train", "test", ...); "base" is every task, also for a domain
    with no split file (banking_knowledge has none: its 97 tasks are one set)."""
    splits = load_task_splits(domain) or {}
    if split in splits:
        return sorted(splits[split], key=lambda i: (len(i), i))
    if split == "base":
        return sorted((t.id for t in get_tasks(domain)), key=lambda i: (len(i), i))
    raise ValueError(f"τ² {domain} has no split {split!r}; its splits are {sorted(splits) or ['base']}")


# Tasks no conversation can be run on, and why: left out of every case set.
UNRUNNABLE = {
    "banking_knowledge": {
        "task_034": "τ²'s simulated user never produces its first message (UserMessage must have either content or "
        "tool calls), in every trial of two Haiku 5.5 baselines",
    },
}


def cases(domain: str, split: str, n: int | None = None, seed: int = 0) -> list[Case]:
    """``n`` of the domain's tasks from τ²'s ``split`` (see ``task_ids``), drawn with ``seed``; all of them by
    default, less those no conversation can be run on (``UNRUNNABLE``)."""
    ids = [i for i in task_ids(domain, split) if i not in UNRUNNABLE.get(domain, {})]
    n = len(ids) if n is None else n
    chosen = sorted(random.Random(seed).sample(ids, n), key=lambda i: (len(i), i))
    return [
        Case(name=f"{domain}-{i}", input=f"τ² {domain} task {i}", metadata={"domain": domain, "task_id": i})
        for i in chosen
    ]


def held_out(study: Study, test_cases: list[Case], repetitions: int | None = None) -> list[dict]:
    """The baseline and the best candidates on cases the search never saw, each run ``repetitions`` times (default:
    the study's), scored by the study's evaluators."""
    repetitions = repetitions or study.repetitions
    experiment = Experiment(cases=test_cases, evaluators=study.experiment.evaluators)
    chosen = [study.baseline, *(c for c in study.best_candidates if c is not study.baseline)]
    out, baseline = [], None
    for c in chosen:
        run = run_trials(
            study.agent,
            c.params,
            experiment,
            repetitions=repetitions,
            max_concurrency=min(len(test_cases) * repetitions, study.max_concurrency or 150),
            task=study.task,
            retries=study.retries,
        )
        trials = [study.projection.prepare(t) for t in asyncio.run(run)]
        if not any(t.status == TrialStatus.COMPLETED for t in trials):
            failed = next((t.error for t in trials if t.error), "no trial completed")
            raise RuntimeError(f"held out: every trial of candidate {c.number} failed: {failed}")
        estimates = study.projection.estimates(trials)
        means = {m: e.mean for m, e in estimates.items()}
        baseline = (
            baseline or means
        )  # the first is the baseline: a constraint relative to it is judged on its test means
        violations = study.projection.violations(means, baseline=baseline)
        out.append(
            {
                "candidate": c.number,
                "params": c.params,
                "search": {m: e.mean for m, e in c.estimates.items()},
                "held_out": {m: {"mean": e.mean, "ci": e.ci and list(e.ci), "n": e.n} for m, e in estimates.items()},
                "held_out_feasible": all(v is not None and v <= 0 for v in violations),
                "repetitions": repetitions,
                "trials": [
                    {"case": t.case, "repetition": t.repetition, "status": t.status.value, **t.metrics} for t in trials
                ],
            }
        )
        goals = "  ".join(f"{m} {means[m]:.3g} (search {c.metrics[m]:.3g})" for m in study.projection.goal_metrics)
        limits = f"  limits {'met' if out[-1]['held_out_feasible'] else 'broken'}" if violations else ""
        print(f"held out: candidate {c.number}: {goals}{limits}")  # noqa: T201
    return out
