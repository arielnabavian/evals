"""
Base class for optimizers that use a strands Agent with shell tools.

Ported from strands-labs/harness-optimizer: ``strands_harness_optimizer/optimizers/base_agentic_optimizer.py``
(``BaseAgenticOptimizer``), with the accumulate-then-step protocol of its parent
``strands_harness_optimizer/optimizers/optimizer.py`` (``FormulaOptimizer``) folded in. The Formula abstraction is
not ported: the optimizer is given any object with ``get_tunable_params()`` and ``update_params()`` (see
:class:`TunableValues`) and reads and writes its parameters through those two methods.

Provides infrastructure for writing rollouts to temp folders, creating
strands Agents with shell tool access, output guardrails, and a
submit_optimized_params tool for reliable parameter extraction.

It handles the parts every agentic optimizer needs (which traces to look at, how to
build and invoke the agent, how to checkpoint) and leaves the optimization algorithm
to ``step()``. Subclasses live beside it: ``contrastive.py`` and ``skill_curation.py``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Protocol

# For strands_tools' shell (skill curation's agent uses it): without it shell initializes in interactive-consent mode
# and the model refuses to run any command.
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

from botocore.config import Config as BotocoreConfig
from strands import Agent, tool
from strands.models import BedrockModel
from strands.models.model import CacheConfig

from .datamodels import Reward, Rollout
from .guardrails import ToolOutputGuardrail

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_SUFFIX = """

## Submitting the Optimized Parameters

You have access to a `submit_optimized_params` tool. When you have finished
analyzing the traces and crafted the optimized parameters:

1. Write each optimized parameter to a separate file in your working folder using the shell tool
2. Call `submit_optimized_params(file_path_dict={"param_name": "<working folder>/<file>", ...})` to submit

Each file must contain ONLY the parameter value, nothing else.
Do NOT include the parameters in your response text — use the tool to submit them.

Example for system prompt optimization:
```
# Write the optimized prompt to a file in your working folder (the shell's working directory)
echo '...' > system_prompt.txt
# Submit it
submit_optimized_params(file_path_dict={"system_prompt": "<working folder>/system_prompt.txt"})
```"""

# Directories where other work on the machine keeps its files: a command that names a path in one of them, outside the
# optimizer's own folders, is refused.
PRIVATE_DIRS = ("/tmp", "/private", "/home", "/Users", "/root", "/var", "/mnt", "/opt", "/srv", "/data", "/run/user")
_PATH = re.compile(r"(?<![\w.:/~-])(/[^\s'\"|;&<>()`*?\[\]{},]*|\.\.(?=/|$|[\s'\"]))")


def outside(command: str, roots: Sequence[str], cwd: str) -> list[str]:
    """The paths ``command`` names that lie outside ``roots``, in a directory where other work keeps its files (see
    ``PRIVATE_DIRS``), or above ``cwd`` (``..``). A guard against a cooperative model's mistakes, such as reading or
    writing a fixed name in a shared ``/tmp``; it is not a sandbox (a program can still build a path)."""
    found = []
    for token in _PATH.findall(command):
        path = os.path.normpath(os.path.join(cwd, token))
        if any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots):
            continue
        if token.startswith("..") or any(path == d or path.startswith(d + "/") for d in PRIVATE_DIRS):
            found.append(token)
    return sorted(set(found))


# ---- the digest of every trace (round 3b, F5) ------------------------------------------------------------------------
DIGEST_CHARS = 20000  # about 5k tokens: the agent reads it once instead of opening traces to learn how each one went


def _short(text: Any, limit: int) -> str:
    """``text`` on one line, cut to ``limit`` characters."""
    flat = " ".join(str(text).split()) if text is not None else ""
    return flat if len(flat) <= limit else flat[: max(0, limit - 3)] + "..."


def _digest_row(file: str, rollout: Any, reward: float, reward_obj: Any, output: Any) -> dict:
    """One trace, as the digest lists it."""
    meta = (reward_obj.metadata if reward_obj else None) or {}
    task = rollout.data_sample or {}
    messages = rollout.messages or []
    calls = sum(
        1
        for m in messages
        if isinstance(m, dict)
        for b in (m.get("content") or [])
        if isinstance(b, dict) and "toolUse" in b
    )
    reasons = {k: v for k, v in (meta.get("reasons") or {}).items() if v}
    return {
        "file": file,
        "case": str(task.get("name") or task.get("task_id") or file),
        "repetition": task.get("repetition"),
        "reward": float(reward),
        "status": meta.get("status"),
        "error": meta.get("error"),
        "reasons": reasons,
        "objectives": {k: v for k, v in (meta.get("objectives") or {}).items() if v is not None},
        "ended": output,
        "messages": len(messages),
        "calls": calls,
    }


def _number(x: Any) -> str:
    if isinstance(x, (int, float)) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def _trial_line(r: dict, cap: int) -> str:
    parts = [f"{r['file']}: reward {r['reward']:.3g}"]
    if r["status"] and r["status"] != "completed":
        parts[0] += f" ({r['status']}{': ' + _short(r['error'], cap) if r['error'] else ''})"
    if r["ended"] is not None:
        parts.append(f"ended: {_short(r['ended'], cap)}")
    if r["reasons"]:
        parts.append("reasons: " + "; ".join(f"{k}: {_short(v, cap)}" for k, v in sorted(r["reasons"].items())))
    others = {k: v for k, v in r["objectives"].items() if not isinstance(v, float) or not 0 <= v <= 1}
    if others:
        parts.append(", ".join(f"{k} {_number(v)}" for k, v in others.items()))
    parts.append(f"{r['messages']} messages, {r['calls']} tool calls")
    return "- " + "; ".join(parts)


def _case_cost(rs: Sequence[dict], costs: Sequence[str]) -> str:
    """A case's mean of each cost metric over its conversations (those that have it)."""
    parts = []
    for m in costs:
        vs = [r["objectives"][m] for r in rs if isinstance(r["objectives"].get(m), (int, float))]
        if vs:
            parts.append(f"{m} {_number(sum(vs) / len(vs))}")
    return ", ".join(parts)


def digest_text(rows: Sequence[dict], threshold: float, limit: int, costs: Sequence[str] = ()) -> str:
    """The digest of every trace: a header, then every case with a conversation whose reward is below ``threshold``
    (each such conversation: its file, reward, how it ended, the evaluators' reasons, its size), then the cases where
    every conversation succeeded. Its fields are cut until it fits in ``limit`` characters; every case that did not
    succeed is always named (when even their names do not fit, the digest says how many are left out).

    ``costs``: metrics (each conversation's, among its objectives) the digest gives per case for every case: the
    total over the conversations in the header, and each case's mean beside it, the cases that succeeded listed most
    costly first, so a rewriter that is lowering cost sees where it goes without losing a success (round 3d)."""
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    failing = sorted(
        (c for c, rs in by_case.items() if any(r["reward"] < threshold for r in rs)),
        key=lambda c: (-sum(r["reward"] < threshold for r in by_case[c]), c),
    )
    passing = sorted(c for c in by_case if c not in failing)
    if costs:  # the costliest successes first: where cost can be cut without losing one
        first = costs[0]

        def spent(c: str) -> float:
            vs = [r["objectives"].get(first) for r in by_case[c]]
            vs = [v for v in vs if isinstance(v, (int, float))]
            return -(sum(vs) / len(vs)) if vs else 0.0

        passing.sort(key=lambda c: (spent(c), c))
    bad = sum(r["reward"] < threshold for r in rows)
    head = [
        f"# Digest: the current version's {len(rows)} conversations, {len(by_case)} cases",
        "",
        f"{bad} conversations did not succeed (reward below {threshold:g}), in {len(failing)} cases; "
        f"every case where every conversation succeeded: {len(passing)}. Each conversation is a file in this folder: "
        "its messages in `response.messages`, its `reward`, `output` (how it ended) and `metrics`.",
        "",
    ]
    if costs:
        total = _case_cost(rows, costs)
        head += [
            f"Cost per conversation, over all {len(rows)}: {total}. Every case below gives its mean cost per "
            "conversation; the cases where every conversation succeeded are listed most costly first.",
            "",
        ]

    def build(cap: int, detailed: bool) -> str:
        out = list(head)
        out.append(f"## Every case with a conversation that did not succeed ({len(failing)})")
        for c in failing:
            rs = by_case[c]
            ok = sum(r["reward"] >= threshold for r in rs)
            spent = f"; {_case_cost(rs, costs)}" if costs and _case_cost(rs, costs) else ""
            if detailed:
                out.append(f"### {c}: {ok} of {len(rs)} succeeded{spent}")
                out += [_trial_line(r, cap) for r in sorted(rs, key=lambda r: (r["reward"], r["file"]))]
            else:
                worst = min(rs, key=lambda r: r["reward"])
                ended = f"; ended: {_short(worst['ended'], cap)}" if worst["ended"] is not None else ""
                files = ", ".join(r["file"] for r in rs if r["reward"] < threshold)
                out.append(f"- {c}: {ok} of {len(rs)} succeeded ({files}){ended}{spent}")
        out += ["", f"## Cases where every conversation succeeded ({len(passing)})"]
        if costs:
            out += [f"- {c} ({len(by_case[c])}/{len(by_case[c])}): {_case_cost(by_case[c], costs)}" for c in passing]
            if not passing:
                out.append("(none)")
        else:
            out.append(", ".join(f"{c} ({len(by_case[c])}/{len(by_case[c])})" for c in passing) or "(none)")
        return "\n".join(out) + "\n"

    for detailed in (True, False):
        cap = 300
        while cap >= 40:
            text = build(cap, detailed)
            if len(text) <= limit:
                return text
            cap //= 2
    text = build(40, False)
    lines, kept = text.splitlines(), []
    for line in lines:  # still over: keep what fits, and say what was left out
        if sum(len(x) + 1 for x in kept) + len(line) + 1 > limit - 120:
            break
        kept.append(line)
    left = len(lines) - len(kept)
    note = f"({left} more lines left out at the digest's limit of {limit:,} characters: see the files)"
    return "\n".join(kept) + f"\n\n{note}\n"


class TunableValues(Protocol):
    """What an optimizer reads its parameters from and writes them back to."""

    def get_tunable_params(self) -> dict[str, Any]: ...

    def update_params(self, params: dict[str, Any]) -> None: ...


class AgenticOptimizer(ABC):
    """
    Base class for agent-based optimizers.

    Follows PyTorch's optimizer pattern:
    1. Initialize with the values to tune (like PyTorch optimizer takes model.parameters())
    2. Accumulate rollouts and rewards (like loss.backward() accumulates gradients)
    3. Call step() to update the values (like optimizer.step())
    4. Call zero() to reset accumulated data (like optimizer.zero_grad())

    Rollouts are expected to contain the data sample information (e.g., task input,
    expected answer) alongside the conversation trace, so there is no separate
    add_data_samples() method. The number of rollouts and rewards must be consistent
    (aligned by index) when step() is called.

    Provides:
    - Writing rollouts to a temp folder as JSON files
    - Creating a strands Agent with shell tool and output guardrail
    - A submit_optimized_params tool for reliable parameter extraction
    - Stratified sampling of traces by reward
    - Configurable boto, model, and guardrail settings

    Subclasses implement step() to define the optimization algorithm,
    using the inherited helper methods. None of the above is specific to a
    particular kind of value: ContrastiveReflection submits parameter VALUES,
    while SkillLibraryCuration has its agent write a decision tree, and both
    reuse everything here. Override ``_get_extra_tools`` to add tools, and
    ``_get_tools`` to replace the default set.
    """

    _DEFAULT_MODEL_CONFIG = {
        "model_id": "global.anthropic.claude-sonnet-5-5",
        "temperature": 1.0,
        "streaming": True,
    }

    _DEFAULT_BOTO_CONFIG = BotocoreConfig(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=256,
        read_timeout=256,
    )

    def __init__(
        self,
        values: TunableValues,
        model_config: dict | None = None,
        region_name: str | None = None,
        boto_config: dict | BotocoreConfig | None = None,
        n_sample_traces: int = -1,
        stratified_sampling: bool = True,
        success_threshold: float = 0.5,
        max_output_chars: int = 100000,
        system_prompt_suffix: str | None = None,
        readable_folders: Sequence[str] = (),
        digest: bool = True,
        digest_chars: int = DIGEST_CHARS,
        digest_costs: Sequence[str] = (),
    ):
        """
        Initialize the base agentic optimizer.

        Args:
            values: The values whose parameters will be optimized: any object with
                ``get_tunable_params()`` and ``update_params(params)``.
            model_config: Bedrock model configuration dict. Supported fields
                include model_id, temperature, max_tokens, top_p, streaming,
                stop_sequences, and any other BedrockModel config fields.
                Partial overrides are merged with defaults (model_id=claude-sonnet,
                temperature=1.0, streaming=True).
            region_name: AWS region for the Bedrock service (default: the environment's, as boto3 finds it).
            boto_config: Botocore config to merge with defaults. Accepts a
                BotocoreConfig instance or a dict (e.g., {"read_timeout": 300}).
                Partial overrides are supported. Uses default settings if None.
            n_sample_traces: Number of traces to sample per step (-1 for all).
            stratified_sampling: Balance successful/failed traces in sampling.
            success_threshold: Reward threshold for classifying success.
            max_output_chars: Maximum characters for tool output before truncation.
            system_prompt_suffix: Suffix appended to the agent's system prompt
                with instructions for using the submit_optimized_prompt tool.
                Uses default instructions if None.
            readable_folders: Folders the agent may read besides its traces (its
                shell is confined to those, the traces and its own working folder).
            digest: Write each trace's messages once (``response.messages``), with
                how the conversation ended (``output``), and a digest of every
                trace (``DIGEST.md`` in the traces folder): every case with a
                conversation that did not succeed, how each ended, and the
                evaluators' reasons. False: round 3's traces (the messages twice,
                under ``data`` and ``response``; how it ended left out).
            digest_chars: The digest's length limit, in characters (its fields
                are cut to fit; every case that did not succeed is listed).
            digest_costs: Cost metrics the digest gives for every case, those
                that succeeded too (``digest_text``): a phase that lowers cost.
        """
        self.values = values
        self._rollouts: list[Rollout] = []
        self._rewards: list[Reward] = []

        self.model_config = {**self._DEFAULT_MODEL_CONFIG, **(model_config or {})}
        self.region_name = region_name
        self.boto_config = boto_config
        self.n_sample_traces = n_sample_traces
        self.stratified_sampling = stratified_sampling
        self.success_threshold = success_threshold
        self.max_output_chars = max_output_chars
        self.system_prompt_suffix = system_prompt_suffix or DEFAULT_SYSTEM_PROMPT_SUFFIX
        self.readable_folders = [os.path.abspath(f) for f in readable_folders]
        self.digest, self.digest_chars, self.digest_costs = digest, digest_chars, list(digest_costs)
        self.digest_file: Optional[str] = None  # the digest written with the traces, when there is one
        self.digest_text: Optional[str] = None  # its text, kept after the traces folder is removed
        self._work_dir: Optional[str] = None
        self.refused: list[dict] = []  # the commands the confined shell refused, and the paths they named

        self._temp_dir: Optional[str] = None
        self._submitted_params: Optional[dict] = None
        self._step_count: int = 0
        self._prompt_history: list = []
        self._sampled_indices: list[int] = []

        self.last_metrics: dict = {}
        self.last_wall_clock_s: float = 0.0
        self.last_complete: bool = False

    # ── accumulate, step, zero ───────────────────────────────────────────────
    def add_rollouts(self, rollouts: list[Rollout]) -> None:
        """
        Accumulate rollouts.

        Args:
            rollouts: List of Rollout objects containing data_sample, messages,
                and optional metrics and metadata.
        """
        self._rollouts.extend(rollouts)

    def add_rewards(self, rewards: list[Reward]) -> None:
        """
        Accumulate Reward objects, aligned with rollouts.

        Args:
            rewards: List of Reward objects containing reward value and metadata.
        """
        self._rewards.extend(rewards)

    @abstractmethod
    def step(self) -> None:
        """
        Propose new parameters and update the values.

        Uses accumulated rollouts and rewards to compute new parameters,
        then calls self.values.update_params() internally.
        """

    def zero(self) -> None:
        """Reset accumulated rollouts and rewards."""
        self._rollouts = []
        self._rewards = []

    # ── the agent ────────────────────────────────────────────────────────────
    def _create_agent(self, system_prompt: str) -> Agent:
        """Create a strands Agent with shell tool, submit tool, and output guardrail."""
        if self.boto_config:
            override = (
                self.boto_config if isinstance(self.boto_config, BotocoreConfig) else BotocoreConfig(**self.boto_config)
            )
            boto_config = self._DEFAULT_BOTO_CONFIG.merge(override)
        else:
            boto_config = self._DEFAULT_BOTO_CONFIG

        # Prompt caching on, unless the model config says otherwise: the agent rereads its whole conversation (system
        # prompt, tools, every trace it has read) on each of its many turns, so a turn reads the earlier ones from the
        # cache. The SDK places the cache points for Anthropic models on Bedrock, and none for others.
        config = {"cache_config": CacheConfig(strategy="auto", tools_ttl=True), **self.model_config}
        model = BedrockModel(region_name=self.region_name, boto_client_config=boto_config, **config)

        # Create submit tool that captures the params
        optimizer_ref = self

        @tool
        def submit_optimized_params(file_path_dict: dict) -> str:
            """Submit optimized parameters by providing a mapping of param names to file paths.

            Each file should contain ONLY the parameter value as plain text.

            Args:
                file_path_dict: Dict mapping parameter names to file paths.
                    e.g., {"system_prompt": "<working folder>/system_prompt.txt"}
            """
            params = {}
            errors = []
            work = optimizer_ref._work_folder()
            for key, path in file_path_dict.items():
                full = os.path.realpath(os.path.join(work, path))
                if not full.startswith(os.path.realpath(work) + os.sep):
                    errors.append(f"{key}: {path} is not in your working folder; write it in {work} and submit that")
                    continue
                path = full
                try:
                    with open(path) as f:
                        params[key] = f.read().strip()
                except FileNotFoundError:
                    errors.append(f"{key}: file not found: {path}")
                except Exception as e:
                    errors.append(f"{key}: error reading {path}: {e}")

            if errors:
                return "Errors:\n" + "\n".join(errors)

            optimizer_ref._submitted_params = params
            summary = ", ".join(f"{k} ({len(v)} chars)" for k, v in params.items())
            return f"Parameters submitted successfully: {summary}"

        # Append suffix with tool usage instructions
        full_system_prompt = system_prompt + self.system_prompt_suffix

        agent_kwargs: dict[str, Any] = dict(
            system_prompt=full_system_prompt,
            model=model,
            tools=self._get_tools(submit_optimized_params) + self._get_extra_tools(),
        )
        conversation_manager = self._conversation_manager()
        if conversation_manager is not None:
            agent_kwargs["conversation_manager"] = conversation_manager
        agent = Agent(**agent_kwargs)

        # Register output truncation guardrail
        guardrail = ToolOutputGuardrail(max_chars=self.max_output_chars)
        guardrail.register(agent)

        return agent

    def _get_tools(self, submit_optimized_params) -> list:
        """Return the agent's BASE toolset.

        Default is ``[shell, submit_optimized_params]``, the shell confined to the agent's folders
        (``_confined_shell``). Override to replace the
        set — a subclass whose agent submits its result some other way (writing a
        folder of files, say) should drop the submit tool rather than be handed one
        that does nothing for it, since an unused tool in the schema is an
        invitation to call it.

        To ADD tools while keeping these, override ``_get_extra_tools`` instead.
        """
        return [self._confined_shell(), submit_optimized_params]

    def _work_folder(self) -> str:
        """The agent's own working folder (created on first use, removed with the traces): its shell's working
        directory, ``HOME`` and ``TMPDIR``, and the only place it may write."""
        if self._work_dir is None or not os.path.isdir(self._work_dir):
            self._work_dir = tempfile.mkdtemp(prefix="reflection_work_")
        return self._work_dir

    def _roots(self) -> list[str]:
        """The folders the agent may use: its traces, the readable folders, its working folder."""
        traces = [self._temp_dir] if self._temp_dir else []
        return [os.path.abspath(f) for f in [*traces, *self.readable_folders, self._work_folder()]]

    def _confined_shell(self) -> Any:
        """A shell for the agent that runs each command in its working folder (with ``HOME`` and ``TMPDIR`` there) and
        refuses a command that names a path outside its folders where other work keeps files (``outside``): several
        optimizers on one machine never share a scratch file. The refusals are kept in ``refused``."""
        optimizer = self

        @tool
        def shell(command: str, timeout: int = 300) -> str:
            """Run a bash command in your working folder and return its output (stdout, stderr and the exit code).
            Read only your trajectory folder and the other folders you were given, and write only in your working
            folder (also HOME and TMPDIR): other work on this machine is not yours to read.

            Args:
                command: The bash command to run.
                timeout: Seconds before the command is stopped.
            """
            work = optimizer._work_folder()
            roots = optimizer._roots()
            named = outside(command, roots, work)
            if named:
                optimizer.refused.append({"command": command[:500], "paths": named})
                return (
                    f"Refused: {', '.join(named)} is outside your folders ({', '.join(roots)}). Read only those "
                    f"folders and write only in your working folder, {work}."
                )
            env = {**os.environ, "HOME": work, "TMPDIR": work}
            try:
                done = subprocess.run(
                    ["bash", "-c", command], cwd=work, env=env, capture_output=True, text=True, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                return f"Stopped after {timeout} seconds."
            err = f"\n[stderr]\n{done.stderr}" if done.stderr else ""
            return f"{done.stdout}{err}\n[exit code {done.returncode}]"

        return shell

    def _get_extra_tools(self) -> list:
        """Return additional tools for the agent. Override in subclasses."""
        return []

    def _conversation_manager(self):
        """Conversation manager for the agent, or None for Strands' default.

        Strands' default is a sliding window of 40 messages with nothing pinned,
        trimmed oldest-first. The task message is ``messages[0]``, so an agent that
        runs long enough loses its own instructions. A subclass whose agent takes
        many shell/editor turns should return a manager sized for that and pin the
        task; see harness-optimizer's ``MultiSurfaceOptimizer``. ``None`` keeps
        existing behaviour.
        """
        return None

    def _invoke_agent(self, agent: Agent, message: str):
        """Invoke the agent and record wall-clock + Strands metrics on self: also when the call raises, from the
        agent's own metrics (every model call that finished before the failure was spent); ``last_complete`` says
        whether it returned."""
        t0 = time.time()
        self.last_metrics, self.last_wall_clock_s, self.last_complete = {}, 0.0, False
        try:
            response = agent(message)
        except BaseException:
            self.last_wall_clock_s = time.time() - t0
            self.last_metrics = self._metrics(getattr(agent, "event_loop_metrics", None))
            raise
        self.last_wall_clock_s = time.time() - t0
        self.last_metrics = self._metrics(response.metrics)
        self.last_complete = True
        return response

    @staticmethod
    def _metrics(m: Any) -> dict:
        """Strands' event-loop metrics as ``last_metrics`` holds them (empty when there are none)."""
        if m is None:
            return {}
        usage = getattr(m, "accumulated_usage", None) or {}
        return {
            "cycle_count": getattr(m, "cycle_count", 0),
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
            "latency_ms": (getattr(m, "accumulated_metrics", None) or {}).get("latencyMs", 0),
            "tool_calls": {k: v.call_count for k, v in (getattr(m, "tool_metrics", None) or {}).items()},
        }

    def _get_submitted_params(self) -> Optional[dict]:
        """Get the params submitted via the tool, then reset."""
        params = self._submitted_params
        self._submitted_params = None
        return params

    # ── which traces the agent sees ──────────────────────────────────────────
    def _sample_traces(self) -> list[int]:
        """Sample trace indices from accumulated rollouts and rewards.

        Returns:
            List of indices into self._rollouts / self._rewards.
        """
        n_traces = len(self._rollouts)
        all_indices = list(range(n_traces))

        if self.n_sample_traces < 0 or self.n_sample_traces >= n_traces:
            self._sampled_indices = all_indices
            return all_indices

        if self.stratified_sampling:
            if not self._rewards:
                raise ValueError(
                    "Stratified sampling requires rewards but none were provided. "
                    "Call add_rewards() before step(), or set stratified_sampling=False."
                )
            sampled = self._stratified_sample(all_indices)
        else:
            sampled = random.sample(all_indices, self.n_sample_traces)

        self._sampled_indices = sampled
        logger.info(f"Sampled {len(sampled)} of {n_traces} traces")
        return sampled

    def _sample_reward(self, index: int) -> float:
        """The scalar that classifies rollout ``index`` for stratified sampling.

        ``Reward.reward`` by default. A subclass whose objective is not that number
        (a weighted combination of scores, say) overrides this so the sample is
        stratified on the same quantity the optimizer shows its agent.
        """
        return self._rewards[index].reward

    def _sample_threshold(self, all_indices: list[int]) -> float:
        """The cut between "successful" and "failed" for stratified sampling.

        ``success_threshold`` by default. A subclass may adapt it to the pool, for
        example falling back to the median when every rollout lands on one side.
        """
        return self.success_threshold

    def _stratified_sample(self, all_indices: list[int]) -> list[int]:
        """Balance successful and failed traces in sampling."""
        successful, failed = [], []

        threshold = self._sample_threshold(all_indices)
        for i in all_indices:
            reward = self._sample_reward(i)
            if reward >= threshold:
                successful.append(i)
            else:
                failed.append(i)

        # Distribute budget across successful and failed
        n_each = self.n_sample_traces // 2
        n_success = min(n_each, len(successful))
        n_fail = min(self.n_sample_traces - n_success, len(failed))

        if n_success < n_each and len(failed) > n_fail:
            n_fail = min(self.n_sample_traces - n_success, len(failed))
        if n_fail < (self.n_sample_traces - n_success) and len(successful) > n_success:
            n_success = min(self.n_sample_traces - n_fail, len(successful))

        sampled = []
        if successful:
            sampled.extend(random.sample(successful, n_success))
        if failed:
            sampled.extend(random.sample(failed, n_fail))

        logger.info(f"Stratified sample: {n_success} successful + {n_fail} failed")
        return sampled

    def _write_traces_to_temp(self, indices: list[int]) -> str:
        """Write selected rollouts with rewards to a temp folder as JSON files.

        Each file carries the same episode under TWO shapes, because prompts in the
        wild read one or the other and a prompt that reads the wrong one silently
        sees nothing:

          ``data.{messages,reward,metrics,task}``
              The original nesting. The built-in contrastive_reflection templates
              read this.
          ``{reward, response.messages, eval_result, task_id}``
              The shape a deployed AgentCore runtime returns and that rollout
              traces are conventionally stored in. Prompts written against real
              trace folders read ``d["reward"]`` and
              ``d["response"]["messages"]`` -- with only the nested shape, such a
              prompt reports `reward=None` and finds zero tool calls for EVERY
              trace, so its census, its success/failure split and its per-tool
              statistics are all empty while it reports having analyzed the
              folder.

        Duplication costs a little disk in a temp folder that is deleted at the end
        of the step; a prompt silently analyzing nothing costs a whole optimizer
        run.

        With ``digest`` (the default since round 3b), a file holds its messages
        once, in the second shape (``response.messages``, which skill curation's
        template reads), with ``output`` (how the conversation ended) and the task,
        and ``DIGEST.md`` beside the files lists every trace (``digest_text``), so
        the agent learns from one bounded file which conversations failed and how,
        and reads the messages only where it needs them. Without it, both shapes as
        above, and how each conversation ended is not written.

        Args:
            indices: List of indices into self._rollouts / self._rewards to write.

        Returns:
            Path to the temp folder containing the trace files.
        """
        self._temp_dir = tempfile.mkdtemp(prefix="contrastive_traces_")
        self.digest_file = None
        rows = []

        for i in indices:
            rollout = self._rollouts[i]
            reward_obj = self._rewards[i] if i < len(self._rewards) else None
            reward_value = reward_obj.reward if reward_obj else 0
            metrics = {"reward": reward_value, "metadata": reward_obj.metadata} if reward_obj else {}
            # The runtime's own evaluator output, when the rollout engine kept it;
            # otherwise the one number we have. Copied to BOTH places a reader may
            # look: top level (invoke.py convention) and under `response` (what an
            # AgentCore runtime actually returns).
            eval_result = (rollout.metadata or {}).get("eval_result") or {"reward": reward_value}
            name = self._trace_filename(rollout, i)
            if self.digest:
                # the messages once, where the AgentCore shape keeps them, and how the conversation ended
                output = (rollout.metadata or {}).get("output")
                data = {
                    "reward": reward_value,
                    "output": output,
                    "response": {"messages": rollout.messages, "eval_result": eval_result},
                    "eval_result": eval_result,
                    "task_id": rollout.data_sample.get("task_id", f"rollout_{i:04d}"),
                    "task": rollout.data_sample,
                    "metrics": metrics,
                }
                data.update(self._trace_extra(rollout, reward_obj, i))
                with open(os.path.join(self._temp_dir, f"{name}.json"), "w") as f:
                    json.dump(data, f, indent=2, default=str)
                rows.append(_digest_row(f"{name}.json", rollout, reward_value, reward_obj, output))
                continue
            data = {
                # Legacy nesting -- the built-in contrastive_reflection templates.
                "data": {
                    "messages": rollout.messages,
                    "reward": reward_value,
                    "metrics": metrics,
                    "task": rollout.data_sample,
                },
                # invoke.py / AgentCore-runtime convention.
                "reward": reward_value,
                "response": {"messages": rollout.messages, "eval_result": eval_result},
                "eval_result": eval_result,
                "task_id": rollout.data_sample.get("task_id", f"rollout_{i:04d}"),
                "metrics": metrics,
            }
            data.update(self._trace_extra(rollout, reward_obj, i))

            path = os.path.join(self._temp_dir, f"{name}.json")
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        if self.digest and rows:
            self.digest_file = os.path.join(self._temp_dir, "DIGEST.md")
            self.digest_text = digest_text(rows, self.success_threshold, self.digest_chars, self.digest_costs)
            with open(self.digest_file, "w") as f:
                f.write(self.digest_text)
        logger.info(f"Wrote {len(indices)} traces to {self._temp_dir}")
        return self._temp_dir

    def _trace_extra(self, rollout, reward, index: int) -> dict:
        """Extra top-level keys for one written trace. Override in subclasses.

        Keys returned here are applied LAST, so a subclass may also override a
        default key (``reward``, say, to make it the configured objective total).
        """
        return {}

    @staticmethod
    def _trace_filename(rollout, index: int) -> str:
        """Filesystem-safe name for one trace, preserving a task_id if present."""
        raw = str(rollout.data_sample.get("task_id") or "").strip()
        if not raw:
            return f"rollout_{index:04d}"
        safe = raw.replace("/", "_").replace("\\", "_").replace("\0", "")
        # Keep it short enough for any filesystem, and unique regardless of the id.
        return f"{safe[:180]}_{index:04d}"

    def _cleanup_temp(self) -> None:
        """Clean up temporary folders: the traces and the agent's working folder."""
        for attr in ("_temp_dir", "_work_dir"):
            folder = getattr(self, attr)
            if folder and os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    logger.warning(f"Failed to clean up {folder}: {e}")
            setattr(self, attr, None)

    # ── checkpointing ────────────────────────────────────────────────────────
    def get_state(self) -> dict:
        """Get optimizer state for checkpointing.

        Returns:
            Dictionary containing optimizer state. Subclasses should call
            super().get_state() and merge their own state.

        Note:
            boto_config is not saved if it was provided as a BotocoreConfig
            instance (not JSON-serializable). Pass boto_config as a dict
            for full checkpoint support.
        """
        # Convert boto_config to dict if possible
        boto_config_dict = None
        if isinstance(self.boto_config, dict):
            boto_config_dict = self.boto_config
        elif self.boto_config is not None:
            logger.warning(
                "boto_config is a BotocoreConfig instance and cannot be serialized. "
                "Pass boto_config as a dict for full checkpoint support."
            )

        return {
            # Config
            "model_config": dict(self.model_config),
            "region_name": self.region_name,
            "boto_config": boto_config_dict,
            "n_sample_traces": self.n_sample_traces,
            "stratified_sampling": self.stratified_sampling,
            "success_threshold": self.success_threshold,
            "max_output_chars": self.max_output_chars,
            "system_prompt_suffix": self.system_prompt_suffix,
            # State
            "step_count": self._step_count,
            "prompt_history": list(self._prompt_history),
            "sampled_indices": list(self._sampled_indices),
        }

    def load_state(self, state: dict) -> None:
        """Load optimizer state from checkpoint.

        Args:
            state: Dictionary from get_state(). Subclasses should call
                super().load_state(state) and restore their own state.
        """
        # Config
        if "model_config" in state:
            self.model_config = dict(state["model_config"])
        if "region_name" in state:
            self.region_name = state["region_name"]
        if "boto_config" in state and state["boto_config"] is not None:
            self.boto_config = state["boto_config"]
        if "n_sample_traces" in state:
            self.n_sample_traces = state["n_sample_traces"]
        if "stratified_sampling" in state:
            self.stratified_sampling = state["stratified_sampling"]
        if "success_threshold" in state:
            self.success_threshold = state["success_threshold"]
        if "max_output_chars" in state:
            self.max_output_chars = state["max_output_chars"]
        if "system_prompt_suffix" in state:
            self.system_prompt_suffix = state["system_prompt_suffix"]
        # State
        if "step_count" in state:
            self._step_count = state["step_count"]
        if "prompt_history" in state:
            self._prompt_history = list(state["prompt_history"])
        if "sampled_indices" in state:
            self._sampled_indices = list(state["sampled_indices"])
