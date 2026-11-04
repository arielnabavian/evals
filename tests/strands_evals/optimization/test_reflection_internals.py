"""Tests for the reflection and skill-curation internals of ``strands_evals.optimization``.

Ported from strands-labs/harness-optimizer: ``tests/optimizers/test_optimizer.py``, the offline part of
``tests/optimizers/test_contrastive_reflection.py``, ``tests/optimizers/test_skill_library_optimizer.py``,
``tests/formulas/test_skill_library_formula.py`` (less the Formula-protocol tests: trigger timings and ``process()``)
and ``tests/utils/test_templates.py``; plus tests of the base optimizer's contract that upstream covered only through
subclasses not ported here. No model is called: as upstream, the agent run is stubbed (``_run_reflection`` /
``_run_curation`` replaced by a function that writes what a real agent would, or ``_create_agent`` / ``_invoke_agent``
faked).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from strands_evals.optimization._reflection import (
    MANIFEST_FILE,
    AgenticOptimizer,
    ContrastiveReflection,
    Reward,
    Rollout,
    SkillLibraryCuration,
    SkillSet,
    collect_decisions,
    create_template,
    load_template,
    skill_dirs_in,
    validate_skill_dir,
)
from strands_evals.optimization._reflection.agentic import outside
from strands_evals.optimization._reflection.guardrails import ToolOutputGuardrail
from strands_evals.optimization._reflection.skill_set import read_frontmatter, render_skill_index
from strands_evals.optimization._reflection.templates import list_templates

FRONTMATTER = "---\nname: {name}\ndescription: Use when {desc}.\n---\n\n{body}\n"


def write_skill(root, name, desc="testing", body="Do the thing.", extra=None):
    """Create <root>/<name>/SKILL.md, plus optional extra files."""
    d = os.path.join(str(root), name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w") as f:
        f.write(FRONTMATTER.format(name=name, desc=desc, body=body))
    for rel, content in (extra or {}).items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
    return d


def write_file(path, text):
    with open(path, "w") as f:
        f.write(text)


class TextValues:
    """Text values in the shape the reflection optimizer reads and writes."""

    def __init__(self, **values):
        self.values = dict(values)

    def get_tunable_params(self):
        return dict(self.values)

    def update_params(self, params):
        self.values.update({k: v for k, v in params.items() if k in self.values})


def graded_rollouts(n=2):
    """``n`` rollouts alternating reward 0.0 / 1.0."""
    messages = [{"role": "user", "content": "x"}]
    rollouts = [Rollout(data_sample={"task_id": f"t{i}"}, messages=list(messages)) for i in range(n)]
    rewards = [Reward(reward=1.0 if i % 2 else 0.0) for i in range(n)]
    return rollouts, rewards


def feed(opt, n=2):
    """Feed the optimizer enough graded rollouts to have something to sample."""
    rollouts, rewards = graded_rollouts(n)
    opt.add_rollouts(rollouts)
    opt.add_rewards(rewards)


def fake_response(**usage):
    """What ``agent(message)`` returns, as far as ``_invoke_agent`` reads it."""
    return SimpleNamespace(
        metrics=SimpleNamespace(
            cycle_count=3,
            accumulated_usage={"inputTokens": 100, "outputTokens": 20, "totalTokens": 120, **usage},
            accumulated_metrics={"latencyMs": 42},
            tool_metrics={"shell": SimpleNamespace(call_count=5)},
        )
    )


# ── templates ────────────────────────────────────────────────────────────────


class TestTemplates:
    def test_variables_are_not_html_escaped(self):
        """Prompts are not HTML: quotes, angle brackets and ampersands must pass through."""
        body = 'search(query: str) -> "x" & <tag>'
        assert create_template("{{ body }}").render(body=body) == body

    def test_sandbox_still_blocks_attribute_escapes(self):
        """Turning autoescape off must not turn the sandbox off."""
        from jinja2.exceptions import SecurityError

        t = create_template("{{ x.__class__.__mro__ }}")
        with pytest.raises(SecurityError):
            t.render(x="s")

    def test_builtin_templates_load(self):
        assert list_templates() == [
            "reflection/system_prompt.jinja",
            "reflection/task_message.jinja",
            "skill_library/system_prompt.jinja",
            "skill_library/task_message.jinja",
        ]
        for name in list_templates():
            load_template(name)

    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError):
            load_template("nope/never.jinja")

    def test_reflection_task_message_renders_each_variable_and_the_versions(self):
        """The reflection task message reads ``variables`` (what each one is and its limit) and the evidence: the
        versions ranked for the goal, and what a conversation's label is."""
        version = {
            "rank": 1,
            "number": 4,
            "from": 2,
            "current": True,
            "baseline": False,
            "changed": ["system_prompt"],
            "cells": ["0.62", "61,234"],
            "meets_limits": True,
            "conversations": 171,
        }
        text = load_template("reflection/task_message.jinja").render(
            traces_folder="/tmp/traces",
            params={"system_prompt": "Be brief.", "tool_x": "Searches."},
            variables=[
                {"name": "system_prompt", "about": "the system prompt", "max_chars": None},
                {"name": "tool_x", "about": None, "max_chars": 200},
            ],
            versions=[
                version,
                {
                    **version,
                    "rank": None,
                    "number": 0,
                    "from": None,
                    "current": False,
                    "baseline": True,
                    "changed": [],
                    "meets_limits": None,
                },
            ],
            versions_folder="/tmp/versions",
            goal_metrics=["reward", "tokens"],
            limits=["reward - baseline >= -0.05"],
            label_metrics=["reward"],
            label_ranked=False,
            judged_by_version=["tokens"],
        )
        assert "/tmp/traces" in text and "`/tmp/versions`" in text
        assert "results in `version.md` and each parameter alone in a file of its own (`<name>.txt`" in text
        assert "diff -r /tmp/versions/rank-001-version-* /tmp/versions/rank-002-version-*" in text
        assert '<parameter name="system_prompt">\nBe brief.' in text
        assert "the system prompt" in text and "at most 200 characters" in text
        assert "| 1 | 4 ← | 2 | system_prompt | 0.62 | 61,234 | yes | 171 |" in text
        assert "| - | 0 (baseline) | - | - | 0.62 | 61,234 | - | 171 |" in text
        assert "its limits: reward - baseline >= -0.05" in text
        assert "from 0 (worst) to 1 (best): its reward." in text and "tokens is not in its label" in text
        assert "Learned Behaviors" not in text and "append" not in text.lower()


# ── the base optimizer ───────────────────────────────────────────────────────


class PickBest(AgenticOptimizer):
    """Appends the hint of the highest-reward rollout."""

    def step(self):
        best = max(range(len(self._rewards)), key=lambda i: self._rewards[i].reward)
        hint = self._rollouts[best].data_sample.get("hint", "")
        params = self.values.get_tunable_params()
        self.values.update_params({"system_prompt": f"{params['system_prompt']} {hint}".strip()})
        self._step_count += 1


class TestAgenticOptimizer:
    def test_accumulate_step_zero_and_checkpoint(self):
        """ABC enforcement, add_rollouts + add_rewards + step + zero, checkpoint roundtrip."""
        with pytest.raises(TypeError):
            AgenticOptimizer(TextValues(system_prompt="test"))  # type: ignore[abstract]

        values = TextValues(system_prompt="original")
        optimizer = PickBest(values)
        optimizer.add_rollouts(
            [
                Rollout(data_sample={"hint": "be brief"}, messages=[]),
                Rollout(data_sample={"hint": "use examples"}, messages=[]),
            ]
        )
        optimizer.add_rewards([Reward(reward=0.2), Reward(reward=0.8)])
        optimizer.step()
        assert values.get_tunable_params()["system_prompt"] == "original use examples"

        optimizer.zero()
        assert optimizer._rollouts == []
        assert optimizer._rewards == []

        state = json.loads(json.dumps(optimizer.get_state()))  # must be JSON-able
        restored = PickBest(TextValues(system_prompt="x"))
        restored.load_state(state)
        assert restored._step_count == 1

    def test_model_and_boto_config_merge_with_defaults(self):
        opt = PickBest(TextValues(), model_config={"temperature": 0.2}, boto_config={"read_timeout": 300})
        assert opt.model_config["temperature"] == 0.2
        assert opt.model_config["model_id"] == AgenticOptimizer._DEFAULT_MODEL_CONFIG["model_id"]
        assert opt.get_state()["boto_config"] == {"read_timeout": 300}

    def test_samples_everything_by_default(self):
        opt = PickBest(TextValues())
        feed(opt, 5)
        assert opt._sample_traces() == [0, 1, 2, 3, 4]

    def test_stratified_sample_balances_success_and_failure(self):
        opt = PickBest(TextValues(), n_sample_traces=4)
        feed(opt, 10)
        sampled = opt._sample_traces()
        assert len(sampled) == 4
        assert sorted(opt._rewards[i].reward for i in sampled) == [0.0, 0.0, 1.0, 1.0]

    def test_stratified_sample_fills_from_the_larger_side(self):
        opt = PickBest(TextValues(), n_sample_traces=4)
        opt.add_rollouts([Rollout(data_sample={}, messages=[]) for _ in range(6)])
        opt.add_rewards([Reward(reward=r) for r in (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)])
        sampled = opt._sample_traces()
        assert len(sampled) == 4
        assert sum(opt._rewards[i].reward for i in sampled) == 1.0

    def test_stratified_sampling_requires_rewards(self):
        opt = PickBest(TextValues(), n_sample_traces=1)
        opt.add_rollouts(graded_rollouts(3)[0])
        with pytest.raises(ValueError, match="add_rewards"):
            opt._sample_traces()

    def test_trace_files_carry_both_shapes(self):
        opt = PickBest(TextValues(), digest=False)  # round 3's traces: the messages under both shapes, no digest
        opt.add_rollouts(
            [
                Rollout(
                    data_sample={"task_id": "a/b", "input": "q"},
                    messages=[{"role": "user", "content": [{"text": "q"}]}],
                    metadata={"eval_result": {"reward": 0.5, "reason": "partial"}},
                ),
                Rollout(data_sample={"input": "r"}, messages=[]),
            ]
        )
        opt.add_rewards([Reward(reward=0.5, metadata={"reason": "partial"}), Reward(reward=1.0)])
        folder = opt._write_traces_to_temp([0, 1])
        try:
            assert sorted(os.listdir(folder)) == ["a_b_0000.json", "rollout_0001.json"]
            with open(os.path.join(folder, "a_b_0000.json")) as f:
                d = json.load(f)
            assert d["data"]["reward"] == 0.5 and d["data"]["task"]["input"] == "q"
            assert d["reward"] == 0.5
            assert d["response"]["messages"] == d["data"]["messages"]
            assert d["eval_result"] == {"reward": 0.5, "reason": "partial"}
            assert d["metrics"] == {"reward": 0.5, "metadata": {"reason": "partial"}}
            assert d["task_id"] == "a/b"
            with open(os.path.join(folder, "rollout_0001.json")) as f:
                d = json.load(f)
            assert d["task_id"] == "rollout_0001"
            assert d["eval_result"] == {"reward": 1.0}
        finally:
            opt._cleanup_temp()
        assert not os.path.exists(folder)

    def test_with_the_digest_each_trace_holds_its_messages_once_and_how_it_ended(self):
        opt = PickBest(TextValues())  # the digest is the default (round 3b)
        messages = [{"role": "user", "content": [{"text": "q"}]}, {"role": "assistant", "content": [{"toolUse": {}}]}]
        opt.add_rollouts(
            [
                Rollout(data_sample={"name": "c1", "repetition": 1}, messages=messages, metadata={"output": "gave up"}),
                Rollout(data_sample={"name": "c1", "repetition": 2}, messages=[], metadata={"output": "solved it"}),
                Rollout(data_sample={"name": "c2", "repetition": 1}, messages=[], metadata={"output": "solved it"}),
            ]
        )
        meta = {"status": "completed", "reasons": {"reward": "DB=0", "other": ""}, "objectives": {"tokens": 1234.0}}
        opt.add_rewards([Reward(reward=0.0, metadata=meta), Reward(reward=1.0), Reward(reward=1.0)])
        folder = opt._write_traces_to_temp([0, 1, 2])
        try:
            files = ["DIGEST.md", "rollout_0000.json", "rollout_0001.json", "rollout_0002.json"]
            assert sorted(os.listdir(folder)) == files
            with open(os.path.join(folder, "rollout_0000.json")) as f:
                d = json.load(f)
            assert "data" not in d and d["response"]["messages"] == messages  # once
            assert d["output"] == "gave up" and d["reward"] == 0.0 and d["task"] == {"name": "c1", "repetition": 1}
            digest = open(opt.digest_file).read()
            assert opt.digest_file == os.path.join(folder, "DIGEST.md")
            assert "1 conversations did not succeed (reward below 0.5), in 1 cases" in digest
            assert "### c1: 1 of 2 succeeded" in digest
            line = next(x for x in digest.splitlines() if x.startswith("- rollout_0000.json"))
            assert "ended: gave up" in line and "reasons: reward: DB=0" in line and "tokens 1,234" in line
            assert "2 messages, 1 tool calls" in line and "other:" not in line
            assert "Cases where every conversation succeeded (1)" in digest and "c2 (1/1)" in digest
            assert "- rollout_0001.json: reward 1; ended: solved it" in digest  # the same case's success, to contrast
            assert "rollout_0002.json" not in digest  # a case where every conversation succeeded: named and counted
        finally:
            opt._cleanup_temp()

    def test_the_digest_fits_its_limit_and_still_names_every_case_that_did_not_succeed(self):
        from strands_evals.optimization._reflection.agentic import digest_text

        rows = [
            {
                "file": f"case{k:03d}_{r}.json",
                "case": f"case{k:03d}",
                "repetition": r,
                "reward": 0.0 if k < 60 else 1.0,
                "status": "completed",
                "error": None,
                "reasons": {"reward": "x" * 500},
                "objectives": {},
                "ended": "y" * 500,
                "messages": 40,
                "calls": 12,
            }
            for k in range(80)
            for r in (1, 2, 3)
        ]
        text = digest_text(rows, 0.5, 20000)
        assert len(text) <= 20000
        assert all(f"case{k:03d}" in text for k in range(60))  # every failing case, with its fields cut to fit
        tiny = digest_text(rows, 0.5, 2000)
        assert len(tiny) <= 2000 and "left out at the digest's limit" in tiny

    def test_invoke_records_metrics_and_wall_clock(self):
        opt = PickBest(TextValues())
        opt._invoke_agent(lambda message: fake_response(cacheReadInputTokens=7), "go")
        assert opt.last_metrics == {
            "cycle_count": 3,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_read_tokens": 7,
            "cache_write_tokens": 0,
            "latency_ms": 42,
            "tool_calls": {"shell": 5},
        }
        assert opt.last_wall_clock_s >= 0.0
        assert opt.last_complete

    def test_invoke_records_what_was_spent_when_the_agent_raises(self):
        """A call that fails part-way (a model service error mid-session) spent every model call that finished."""
        from strands import Agent, tool

        from .helpers import Says

        @tool
        def look() -> str:
            """Look around."""
            return "nothing here"

        class LooksThenFails(Says):
            async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("the model service failed")
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.calls}", "name": "look"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
                usage = {"inputTokens": 70, "outputTokens": 5, "totalTokens": 75, "cacheReadInputTokens": 30}
                yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}

        opt = PickBest(TextValues())
        agent = Agent(model=LooksThenFails(), tools=[look], callback_handler=None)
        with pytest.raises(Exception, match="the model service failed"):
            opt._invoke_agent(agent, "go")
        m = opt.last_metrics
        assert (m["input_tokens"], m["output_tokens"], m["cache_read_tokens"]) == (140, 10, 60)  # two calls finished
        assert m["tool_calls"] == {"look": 2} and opt.last_wall_clock_s > 0.0 and not opt.last_complete

    def test_agent_has_shell_and_submit_tools(self, tmp_path):
        opt = PickBest(TextValues(system_prompt="x"))
        agent = opt._create_agent("persona")
        assert sorted(agent.tool_names) == ["shell", "submit_optimized_params"]
        assert agent.system_prompt.startswith("persona")
        assert "submit_optimized_params" in agent.system_prompt

    def test_the_agent_caches_its_prompt_unless_told_otherwise(self):
        """The reflection agent rereads its conversation every turn: its model caches it (Bedrock cache points)."""
        agent = PickBest(TextValues(system_prompt="x"), region_name="us-east-1")._create_agent("persona")
        assert agent.model.get_config()["cache_config"].strategy == "auto"
        off = PickBest(TextValues(system_prompt="x"), region_name="us-east-1", model_config={"cache_config": None})
        assert off._create_agent("persona").model.get_config().get("cache_config") is None

    def test_submit_tool_reads_files_and_reports_missing_ones(self):
        opt = PickBest(TextValues(system_prompt="x"))
        submit = opt._create_agent("persona").tool_registry.registry["submit_optimized_params"]
        work = opt._work_folder()
        path = os.path.join(work, "p.txt")
        write_file(path, "  new prompt \n")

        assert "file not found" in submit(file_path_dict={"system_prompt": os.path.join(work, "missing.txt")})
        assert opt._get_submitted_params() is None

        assert "submitted successfully" in submit(file_path_dict={"system_prompt": str(path)})
        assert opt._get_submitted_params() == {"system_prompt": "new prompt"}
        assert opt._get_submitted_params() is None  # read once, then reset

    def test_submit_tool_refuses_what_its_check_refuses_and_says_why(self):
        opt = PickBest(TextValues(system_prompt="x"), check=lambda k, v: "it is shouting" if v.isupper() else None)
        submit = opt._create_agent("persona").tool_registry.registry["submit_optimized_params"]
        write_file(os.path.join(opt._work_folder(), "p.txt"), "LOUD")
        result = submit(file_path_dict={"system_prompt": "p.txt"})
        assert result == "Refused, nothing was submitted:\nsystem_prompt: it is shouting"
        assert opt._get_submitted_params() is None and opt._refusal == result
        assert opt.rejected == [{"name": "system_prompt", "reason": "it is shouting"}]
        write_file(os.path.join(opt._work_folder(), "p.txt"), "quiet")
        assert "submitted successfully" in submit(file_path_dict={"system_prompt": "p.txt"})
        assert opt._get_submitted_params() == {"system_prompt": "quiet"} and opt._refusal is None

    def test_guardrail_truncates_long_tool_output(self):
        event = SimpleNamespace(result={"content": [{"text": "x" * 50}, {"text": "short"}]})
        ToolOutputGuardrail(max_chars=10)(event)
        first, second = event.result["content"]
        assert first["text"].startswith("x" * 10 + "\n\n[TRUNCATED")
        assert "Original length: 50" in first["text"]
        assert second["text"] == "short"


# ── contrastive reflection ───────────────────────────────────────────────────


def make_reflection(values, **kwargs):
    return ContrastiveReflection(
        values,
        system_prompt_template=load_template("reflection/system_prompt.jinja"),
        task_message_template=kwargs.pop("task_message_template", "traces: {{ traces_folder }}"),
        **kwargs,
    )


def stub_reflection(opt, submit):
    """Replace the LLM call: ``submit(traces_folder, params)`` returns what the agent would submit (or None)."""
    seen = {}

    def run(traces_folder, params):
        seen.update(traces_folder=traces_folder, files=sorted(os.listdir(traces_folder)), params=params)
        opt._submitted_params = submit(traces_folder, params)

    opt._run_reflection = run
    return seen


class TestContrastiveReflection:
    def test_writes_rollouts_and_cleans_up(self):
        """Rollouts are written to temp folder and cleaned up."""
        opt = make_reflection(
            TextValues(system_prompt="original"),
            model_config={"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
        )
        opt.add_rollouts([Rollout(data_sample={"prompt": "test"}, messages=[{"role": "user", "content": "hello"}])])
        opt.add_rewards([Reward(reward=1)])

        indices = opt._sample_traces()
        traces_folder = opt._write_traces_to_temp(indices)
        assert os.path.exists(traces_folder)
        assert sorted(os.listdir(traces_folder)) == ["DIGEST.md", "rollout_0000.json"]

        opt._cleanup_temp()
        assert not os.path.exists(traces_folder)

    def test_step_applies_the_submitted_params(self):
        values = TextValues(system_prompt="original", tool_x="Searches.")
        opt = make_reflection(values)
        feed(opt, 3)
        seen = stub_reflection(opt, lambda folder, params: {"system_prompt": params["system_prompt"] + " Learned."})

        opt.step()

        assert values.values == {"system_prompt": "original Learned.", "tool_x": "Searches."}
        assert seen["params"] == {"system_prompt": "original", "tool_x": "Searches."}
        assert seen["files"] == ["DIGEST.md", "t0_0000.json", "t1_0001.json", "t2_0002.json"]
        assert not os.path.exists(seen["traces_folder"])
        assert opt._step_count == 1
        assert opt.get_state()["prompt_history"] == [{"system_prompt": "original Learned."}]

    def test_nothing_submitted_raises_and_still_cleans_up(self):
        values = TextValues(system_prompt="original")
        opt = make_reflection(values)
        feed(opt)
        seen = stub_reflection(opt, lambda folder, params: None)

        with pytest.raises(RuntimeError, match="submit_optimized_params"):
            opt.step()
        assert values.values == {"system_prompt": "original"}
        assert not os.path.exists(seen["traces_folder"])
        assert opt._step_count == 0

    def test_other_errors_are_wrapped(self):
        opt = make_reflection(TextValues(system_prompt="original"))
        feed(opt)

        def boom(traces_folder, params):
            raise KeyError("model")

        opt._run_reflection = boom
        with pytest.raises(RuntimeError, match="Error during contrastive reflection"):
            opt.step()

    def test_no_rollouts_is_a_noop(self):
        values = TextValues(system_prompt="original")
        opt = make_reflection(values)
        opt.step()  # must not raise
        assert values.values == {"system_prompt": "original"}

    def test_a_value_over_its_limit_is_sent_back_to_be_shortened_in_the_same_conversation(self):
        values = TextValues(system_prompt="original")
        opt = make_reflection(values, limits={"system_prompt": 10}, over_limit_retries=2)
        feed(opt)
        told = []

        def agent(message):  # the conversation the first submission was made in: each reply shortens it
            told.append(message)
            opt._submitted_params = {"system_prompt": ["y" * 15, "z" * 5][len(told) - 1]}
            return fake_response()

        def run(traces_folder, params):
            opt._submitted_params = {"system_prompt": "x" * 30}
            opt.last_metrics, opt.last_wall_clock_s = {"input_tokens": 1000, "tool_calls": {"shell": 2}}, 10.0
            return agent

        opt._run_reflection = run
        opt.step()
        assert values.values == {"system_prompt": "zzzzz"}
        assert opt.revisions == [{"system_prompt": 30}, {"system_prompt": 15}]
        assert "`system_prompt` is 30 characters, its limit 10 (20 over)" in told[0] and "wc -c" in told[0]
        assert opt.last_metrics["input_tokens"] == 1000 + 2 * 100  # every call of the conversation counted
        assert opt.last_metrics["tool_calls"] == {"shell": 2 + 2 * 5}

    def test_a_value_still_over_its_limit_after_the_retries_is_returned_for_the_caller_to_drop(self):
        for retries, calls in ((2, 2), (0, 0)):  # 0: round 3's, never sent back
            values = TextValues(system_prompt="original")
            opt = make_reflection(values, limits={"system_prompt": 10}, over_limit_retries=retries)
            feed(opt)
            told = []

            def agent(message, told=told, opt=opt):
                told.append(message)
                opt._submitted_params = {"system_prompt": "w" * 20}
                return fake_response()

            def run(traces_folder, params, opt=opt, agent=agent):
                opt._submitted_params = {"system_prompt": "x" * 30}
                return agent

            opt._run_reflection = run
            opt.step()
            assert len(told) == calls and len(opt.revisions) == calls
            assert len(values.values["system_prompt"]) == (20 if calls else 30)

    def test_templates_receive_traces_folder_and_params(self):
        """A task message may be any object with ``render(**kwargs)``, such as one that adds its own variables."""

        class WithVariables:
            def render(self, **kwargs):
                return create_template("{{ traces_folder }}|{{ params.system_prompt }}|{{ variables }}").render(
                    **kwargs, variables="v1"
                )

        seen = {}
        opt = ContrastiveReflection(
            TextValues(system_prompt="original"),
            system_prompt_template="persona for {{ params | length }} param(s)",
            task_message_template=WithVariables(),
        )

        def fake_create_agent(system_prompt):
            seen["system"] = system_prompt
            return lambda message: fake_response()

        def fake_invoke(agent, message):
            seen["task"] = message
            return AgenticOptimizer._invoke_agent(opt, agent, message)

        opt._create_agent = fake_create_agent
        opt._invoke_agent = fake_invoke
        feed(opt)
        with pytest.raises(RuntimeError):
            opt.step()  # the fake agent submits nothing; we only care about the render and the metrics

        assert seen["system"] == "persona for 1 param(s)"
        traces, prompt, variables = seen["task"].split("|")
        assert os.path.isabs(traces) and "contrastive_traces_" in traces
        assert (prompt, variables) == ("original", "v1")
        assert opt.last_metrics["input_tokens"] == 100


# ── the confined shell ───────────────────────────────────────────────────────


class TestConfinedShell:
    """Several optimizers run on one machine: each agent's shell keeps to its own folders, so none reads or writes
    another's scratch files (a fixed name in a shared /tmp was how one study's text reached another)."""

    def test_paths_outside_its_folders_where_other_work_lives_are_named(self):
        roots, cwd = ["/tmp/traces_x", "/tmp/work_y"], "/tmp/work_y"
        assert outside("ls -lh /tmp/traces_x/*.json && cat /tmp/traces_x/a.json > /tmp/work_y/b", roots, cwd) == []
        assert (
            outside("python3 -c 'print(6 / 3)' 2>/dev/null; grep -c x a.json; echo and/or; sed s/a/b/", roots, cwd)
            == []
        )
        assert outside("/usr/bin/env python3 x.py https://example.com ~/notes.txt ./x 'Wait... ok'", roots, cwd) == []
        assert outside("cat /tmp/optimized_contract.txt", roots, cwd) == ["/tmp/optimized_contract.txt"]
        assert outside("ls /home/ubuntu/work/studies; cd .. && ls", roots, cwd) == ["..", "/home/ubuntu/work/studies"]
        assert outside("ls /tmp/*", roots, cwd) == ["/tmp/"]

    def test_commands_run_in_its_working_folder_and_results_are_taken_only_from_there(self, tmp_path):
        opt = make_reflection(TextValues(system_prompt="original"), region_name="us-east-1")
        work = opt._work_folder()
        shell = opt._confined_shell()
        out = shell(command='pwd; echo "$HOME|$TMPDIR"; echo hi > note.txt; mktemp').splitlines()
        assert out[0] == os.path.realpath(work) and out[1] == f"{work}|{work}"
        assert os.path.dirname(os.path.realpath(out[2])) == os.path.realpath(work)
        assert open(os.path.join(work, "note.txt")).read() == "hi\n"
        refused = shell(command=f"cat {tmp_path}/elsewhere.txt")
        assert refused.startswith("Refused") and opt.refused[-1]["paths"] == [f"{tmp_path}/elsewhere.txt"]

        write_file(tmp_path / "elsewhere.txt", "Not mine.")
        agent = opt._create_agent("persona")
        result = agent.tool.submit_optimized_params(file_path_dict={"system_prompt": str(tmp_path / "elsewhere.txt")})
        assert "not in your working folder" in result["content"][0]["text"] and opt._submitted_params is None
        write_file(os.path.join(work, "prompt.txt"), "Mine.")
        agent.tool.submit_optimized_params(file_path_dict={"system_prompt": os.path.join(work, "prompt.txt")})
        assert opt._submitted_params == {"system_prompt": "Mine."}
        opt._cleanup_temp()
        assert not os.path.exists(work)


# ── skill-library curation ───────────────────────────────────────────────────


def make_curation(skill_set, output_folder, **kwargs):
    return SkillLibraryCuration(
        skill_set,
        system_prompt_template="persona: {{ skill_folder }}",
        task_message_template="traces: {{ traces_folder }} out: {{ output_folder }}",
        output_folder=str(output_folder),
        **kwargs,
    )


def stub_curation(opt, writer):
    """Replace the LLM call with `writer(output_folder)`."""
    opt._run_curation = lambda traces_folder: writer(opt.output_folder)


class TestCurationInit:
    def test_rejects_values_that_are_not_a_skill_set(self, tmp_path):
        """Text values have no set to curate."""
        with pytest.raises(TypeError, match="SkillSet"):
            make_curation(TextValues(instructions="i"), tmp_path / "out")

    def test_accepts_a_skill_set(self, tmp_path):
        skill_set = SkillSet()
        opt = make_curation(skill_set, tmp_path / "out")
        assert opt.output_folder == str(tmp_path / "out")
        assert opt.skill_set is skill_set and opt.values is skill_set

    def test_drops_the_submit_tool_instructions(self, tmp_path):
        """The curator writes files itself; submit_optimized_params does not apply."""
        opt = make_curation(SkillSet(), tmp_path / "out")
        assert opt.system_prompt_suffix == ""

    def test_an_explicit_suffix_is_kept(self, tmp_path):
        opt = make_curation(SkillSet(), tmp_path / "out", system_prompt_suffix="\nextra")
        assert opt.system_prompt_suffix == "\nextra"

    def test_drops_the_submit_tool_itself(self, tmp_path):
        """An unused tool in the schema is an invitation to call it."""
        opt = make_curation(SkillSet(), tmp_path / "out")
        agent = opt._create_agent("persona")
        assert sorted(agent.tool_names) == ["shell"]


class TestCurationStep:
    def test_no_rollouts_is_a_noop(self, tmp_path):
        opt = make_curation(SkillSet(), tmp_path / "out")
        opt.step()  # must not raise
        assert opt.skill_set.skill_names() == []

    def test_applies_a_create(self, tmp_path):
        skill_set = SkillSet()
        opt = make_curation(skill_set, tmp_path / "out")
        feed(opt)

        def writer(out):
            write_skill(os.path.join(out, "create"), "new-skill")
            write_file(
                os.path.join(out, MANIFEST_FILE),
                json.dumps({"actions": [{"action": "create", "skill_name": "new-skill"}]}),
            )

        stub_curation(opt, writer)
        opt.step()
        assert skill_set.skill_names() == ["new-skill"]
        assert opt.last_decisions["create"]
        assert opt._last_decisions == opt.last_decisions

    def test_applies_a_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        skill_set = SkillSet(str(existing))
        opt = make_curation(skill_set, tmp_path / "out")
        feed(opt)

        stub_curation(opt, lambda out: write_file(os.path.join(out, "retire.txt"), "beta\n"))
        opt.step()
        assert skill_set.skill_names() == ["alpha"]

    def test_other_errors_are_wrapped(self, tmp_path):
        opt = make_curation(SkillSet(), tmp_path / "out")
        feed(opt)

        def boom(out):
            raise KeyError("model")

        stub_curation(opt, boom)
        with pytest.raises(RuntimeError, match="Error during skill-library optimization"):
            opt.step()


class TestSkipIsValid:
    """A library whose recurring patterns are covered should stop, not churn."""

    def test_manifest_only_is_accepted(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        skill_set = SkillSet(str(existing))
        opt = make_curation(skill_set, tmp_path / "out")
        feed(opt)

        def writer(out):
            write_file(
                os.path.join(out, MANIFEST_FILE),
                json.dumps({"actions": [{"action": "skip", "evidence": "no recurring gaps"}]}),
            )

        stub_curation(opt, writer)
        opt.step()  # must NOT raise
        assert skill_set.skill_names() == ["alpha"]
        assert opt.last_decisions == {"create": [], "optimize": [], "retire": []}

    def test_skip_still_counts_as_a_step(self, tmp_path):
        opt = make_curation(SkillSet(), tmp_path / "out")
        feed(opt)
        stub_curation(opt, lambda out: write_file(os.path.join(out, MANIFEST_FILE), '{"actions": []}'))
        opt.step()
        assert opt._step_count == 1

    def test_nothing_at_all_raises(self, tmp_path):
        """No decisions AND no manifest means the agent did not do the work."""
        opt = make_curation(SkillSet(), tmp_path / "out")
        feed(opt)
        stub_curation(opt, lambda out: None)
        with pytest.raises(RuntimeError, match="manifest"):
            opt.step()


class TestCurationTemplateVars:
    def test_receives_the_three_inputs_plus_index(self, tmp_path):
        write_skill(tmp_path / "skills", "alpha", desc="searching a catalog")
        skill_set = SkillSet(str(tmp_path / "skills"))

        seen = {}
        opt = SkillLibraryCuration(
            skill_set,
            system_prompt_template="{{ skill_index }}",
            task_message_template="{{ traces_folder }}|{{ output_folder }}|{{ skill_folder }}",
            output_folder=str(tmp_path / "out"),
        )

        def fake_create_agent(system_prompt):
            seen["system"] = system_prompt
            return object()

        opt._create_agent = fake_create_agent
        opt._invoke_agent = lambda agent, message: seen.setdefault("task", message)
        feed(opt)
        with pytest.raises(RuntimeError):
            opt.step()  # the fake agent writes nothing; we only care about the render

        assert "searching a catalog" in seen["system"]
        traces, out, skills = seen["task"].split("|")
        assert os.path.isabs(out) and out.endswith("out")
        assert skills.endswith("skills")
        assert "contrastive_traces_" in traces

    def test_cold_start_passes_an_empty_skill_folder(self, tmp_path):
        seen = {}
        opt = SkillLibraryCuration(
            SkillSet(),
            system_prompt_template="x",
            task_message_template="[{{ skill_folder }}][{{ skill_index }}]",
            output_folder=str(tmp_path / "out"),
        )
        opt._create_agent = lambda sp: object()
        opt._invoke_agent = lambda agent, message: seen.setdefault("task", message)
        feed(opt)
        with pytest.raises(RuntimeError):
            opt.step()
        assert seen["task"] == "[][(none deployed)]"

    def test_builtin_templates_render(self, tmp_path):
        write_skill(tmp_path / "skills", "alpha", desc="searching a catalog")
        seen = {}
        opt = SkillLibraryCuration(
            SkillSet(str(tmp_path / "skills")),
            system_prompt_template=load_template("skill_library/system_prompt.jinja"),
            task_message_template=load_template("skill_library/task_message.jinja"),
            output_folder=str(tmp_path / "out"),
        )
        opt._create_agent = lambda sp: seen.setdefault("system", sp)
        opt._invoke_agent = lambda agent, message: seen.setdefault("task", message)
        feed(opt)
        with pytest.raises(RuntimeError):
            opt.step()
        assert "skill library" in seen["system"]
        assert str(tmp_path / "out") in seen["task"]
        assert "searching a catalog" in seen["task"]


class TestCurationCheckpointing:
    def test_round_trips_the_set(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        opt = make_curation(SkillSet(str(existing)), tmp_path / "out")
        state = json.loads(json.dumps(opt.get_state()))  # must be JSON-able

        restored = make_curation(SkillSet(), tmp_path / "elsewhere")
        restored.load_state(state)
        assert restored.skill_set.skill_names() == ["alpha"]
        assert restored.output_folder == str(tmp_path / "out")

    def test_state_records_skill_names(self, tmp_path):
        write_skill(tmp_path / "skills", "alpha")
        opt = make_curation(SkillSet(str(tmp_path / "skills")), tmp_path / "out")
        assert opt.get_state()["skill_names"] == ["alpha"]


# ── the skill set ────────────────────────────────────────────────────────────


class TestSkillSetInit:
    def test_cold_start_has_no_members(self):
        skill_set = SkillSet()
        assert skill_set.members == {}
        assert skill_set.get_tunable_params() == {"skill_dir": ""}

    def test_loads_existing_set(self, tmp_path):
        write_skill(tmp_path, "alpha")
        write_skill(tmp_path, "beta")
        skill_set = SkillSet(str(tmp_path))
        assert skill_set.skill_names() == ["alpha", "beta"]

    def test_tunable_param_is_the_directory(self, tmp_path):
        write_skill(tmp_path, "alpha")
        skill_set = SkillSet(str(tmp_path))
        assert skill_set.get_tunable_params() == {"skill_dir": str(tmp_path)}
        assert skill_set.skill_dir == str(tmp_path)

    def test_missing_dir_starts_cold(self, tmp_path):
        skill_set = SkillSet(str(tmp_path / "nope"))
        assert skill_set.members == {}

    def test_missing_dir_raises_in_strict_mode(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SkillSet(str(tmp_path / "nope"), strict=True)

    def test_skill_without_frontmatter_is_skipped(self, tmp_path):
        write_skill(tmp_path, "good")
        bad = os.path.join(str(tmp_path), "bad")
        os.makedirs(bad)
        write_file(os.path.join(bad, "SKILL.md"), "## Trigger\nUse when something.\n")
        skill_set = SkillSet(str(tmp_path))
        assert skill_set.skill_names() == ["good"]

    def test_bad_skill_raises_in_strict_mode(self, tmp_path):
        bad = os.path.join(str(tmp_path), "bad")
        os.makedirs(bad)
        write_file(os.path.join(bad, "SKILL.md"), "no frontmatter here")
        with pytest.raises(ValueError, match="frontmatter"):
            SkillSet(str(tmp_path), strict=True)


class TestSetAlgebra:
    """existing - retired - optimized_old + created."""

    def test_create_adds(self, tmp_path):
        skill_set = SkillSet()
        out = tmp_path / "out"
        write_skill(out / "create", "new-skill")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["new-skill"]

    def test_optimize_replaces_in_place(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha", body="OLD")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        write_skill(out / "optimize", "alpha", body="NEW")
        skill_set.update_params({"decisions_dir": str(out)})

        assert skill_set.skill_names() == ["alpha"]
        # Points at the optimized copy, not the original.
        assert "out" in skill_set.members["alpha"]

    def test_retire_removes(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        os.makedirs(out)
        (out / "retire.txt").write_text("beta\n")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["alpha"]

    def test_untouched_skills_survive(self, tmp_path):
        """The agent only reports what CHANGED; the rest must carry over."""
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        write_skill(existing, "beta")
        write_skill(existing, "gamma")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "delta")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["alpha", "beta", "delta", "gamma"]

    def test_merge_is_create_plus_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "search-basic")
        write_skill(existing, "search-refine")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "product-search")
        (out / "retire.txt").write_text("search-basic\nsearch-refine\n")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["product-search"]

    def test_split_is_creates_plus_retire(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "everything")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        write_skill(out / "create", "part-one")
        write_skill(out / "create", "part-two")
        (out / "retire.txt").write_text("everything\n")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["part-one", "part-two"]

    def test_retire_unknown_skill_is_ignored(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        skill_set = SkillSet(str(existing))

        out = tmp_path / "out"
        os.makedirs(out)
        (out / "retire.txt").write_text("never-deployed\n")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["alpha"]

    def test_no_decisions_leaves_set_unchanged(self, tmp_path):
        existing = tmp_path / "skills"
        write_skill(existing, "alpha")
        skill_set = SkillSet(str(existing))
        out = tmp_path / "out"
        os.makedirs(out)
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["alpha"]

    def test_malformed_create_is_rejected(self, tmp_path):
        """A skill the runtime could never load must not enter the set."""
        skill_set = SkillSet()
        out = tmp_path / "out"
        d = out / "create" / "broken"
        os.makedirs(d)
        (d / "SKILL.md").write_text("## Trigger\nUse when X.\n")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == []

    def test_malformed_create_raises_in_strict_mode(self, tmp_path):
        skill_set = SkillSet(strict=True)
        d = tmp_path / "out" / "create" / "broken"
        os.makedirs(d)
        (d / "SKILL.md").write_text("## Trigger\nUse when X.\n")
        with pytest.raises(ValueError, match="rejecting skill 'broken'"):
            skill_set.update_params({"decisions_dir": str(tmp_path / "out")})

    def test_deploy_dir_accepted_as_create_alias(self, tmp_path):
        skill_set = SkillSet()
        out = tmp_path / "out"
        write_skill(out / "deploy", "legacy-named")
        skill_set.update_params({"decisions_dir": str(out)})
        assert skill_set.skill_names() == ["legacy-named"]

    def test_update_params_requires_a_known_key(self):
        skill_set = SkillSet()
        with pytest.raises(ValueError, match="decisions_dir"):
            skill_set.update_params({"instructions": "nope"})

    def test_skill_dir_replaces_wholesale(self, tmp_path):
        first = tmp_path / "first"
        write_skill(first, "alpha")
        skill_set = SkillSet(str(first))

        second = tmp_path / "second"
        write_skill(second, "beta")
        skill_set.update_params({"skill_dir": str(second)})
        assert skill_set.skill_names() == ["beta"]

    def test_empty_skill_dir_clears_the_set(self, tmp_path):
        write_skill(tmp_path, "alpha")
        skill_set = SkillSet(str(tmp_path))
        skill_set.update_params({"skill_dir": ""})
        assert skill_set.members == {} and skill_set.skill_dir is None


class TestMaterialize:
    def test_writes_flat_folder(self, tmp_path):
        skill_set = SkillSet()
        out = tmp_path / "out"
        write_skill(out / "create", "alpha")
        write_skill(out / "create", "beta")
        skill_set.update_params({"decisions_dir": str(out)})

        dest = skill_set.materialize(str(tmp_path / "skill_set"))
        assert sorted(os.listdir(dest)) == ["alpha", "beta"]
        assert os.path.isfile(os.path.join(dest, "alpha", "SKILL.md"))

    def test_repoints_skill_dir(self, tmp_path):
        skill_set = SkillSet()
        write_skill(tmp_path / "out" / "create", "alpha")
        skill_set.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = skill_set.materialize(str(tmp_path / "skill_set"))
        assert skill_set.get_tunable_params() == {"skill_dir": dest}

    def test_carries_subdirectories(self, tmp_path):
        """A skill may ship scripts/ or resources/ alongside SKILL.md."""
        skill_set = SkillSet()
        write_skill(
            tmp_path / "out" / "create",
            "alpha",
            extra={"scripts/check.py": "print('hi')", "resources/notes.md": "# notes"},
        )
        skill_set.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = skill_set.materialize(str(tmp_path / "skill_set"))
        assert os.path.isfile(os.path.join(dest, "alpha", "scripts", "check.py"))
        assert os.path.isfile(os.path.join(dest, "alpha", "resources", "notes.md"))

    def test_is_idempotent(self, tmp_path):
        skill_set = SkillSet()
        write_skill(tmp_path / "out" / "create", "alpha")
        skill_set.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = str(tmp_path / "skill_set")
        skill_set.materialize(dest)
        skill_set.materialize(dest)
        assert sorted(os.listdir(dest)) == ["alpha"]

    def test_round_trips_through_a_new_set(self, tmp_path):
        skill_set = SkillSet()
        write_skill(tmp_path / "out" / "create", "alpha")
        skill_set.update_params({"decisions_dir": str(tmp_path / "out")})
        dest = skill_set.materialize(str(tmp_path / "skill_set"))
        assert SkillSet(dest).skill_names() == ["alpha"]


class TestFrontmatter:
    def test_reads_past_a_long_description(self, tmp_path):
        """A fixed-size window would cut mid-value and lose the closing ---."""
        long_desc = "x" * 900
        d = write_skill(tmp_path, "alpha", desc=long_desc)
        block = read_frontmatter(os.path.join(d, "SKILL.md"))
        assert "name: alpha" in block
        assert "description:" in block
        assert validate_skill_dir(d) == []

    def test_missing_frontmatter_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "bad")
        os.makedirs(d)
        write_file(os.path.join(d, "SKILL.md"), "## Trigger\nUse when X.\n")
        problems = validate_skill_dir(d)
        assert problems and "frontmatter" in problems[0]

    def test_missing_description_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "bad")
        os.makedirs(d)
        write_file(os.path.join(d, "SKILL.md"), "---\nname: bad\n---\n\nbody\n")
        problems = validate_skill_dir(d)
        assert any("description:" in p for p in problems)

    def test_missing_skill_md_is_reported(self, tmp_path):
        d = os.path.join(str(tmp_path), "empty")
        os.makedirs(d)
        assert validate_skill_dir(d)


class TestRenderSkillIndex:
    def test_cold_start(self):
        assert render_skill_index(None) == "(none deployed)"
        assert render_skill_index("") == "(none deployed)"

    def test_includes_name_path_and_trigger(self, tmp_path):
        write_skill(tmp_path, "alpha", desc="searching a catalog")
        index = render_skill_index(str(tmp_path))
        assert "### alpha" in index
        assert "searching a catalog" in index
        assert str(tmp_path) in index

    def test_omits_the_body(self, tmp_path):
        """Triggers only: reading bodies up front anchors toward editing them."""
        write_skill(tmp_path, "alpha", body="SECRET_BODY_MARKER")
        assert "SECRET_BODY_MARKER" not in render_skill_index(str(tmp_path))


class TestCollectDecisions:
    def test_empty_folder(self, tmp_path):
        assert collect_decisions(str(tmp_path)) == {"create": [], "optimize": [], "retire": []}

    def test_reads_all_three(self, tmp_path):
        write_skill(tmp_path / "create", "a")
        write_skill(tmp_path / "optimize", "b")
        (tmp_path / "retire.txt").write_text("c\nd\n")
        d = collect_decisions(str(tmp_path))
        assert len(d["create"]) == 1 and len(d["optimize"]) == 1
        assert d["retire"] == ["c", "d"]

    def test_ignores_blank_retire_lines(self, tmp_path):
        (tmp_path / "retire.txt").write_text("a\n\n  \nb\n")
        assert collect_decisions(str(tmp_path))["retire"] == ["a", "b"]

    def test_update_is_an_alias_of_optimize(self, tmp_path):
        """The multi-surface templates write update/; the set algebra reads optimize/."""
        write_skill(tmp_path / "optimize", "a", body="A")
        write_skill(tmp_path / "update", "b", body="B")
        d = collect_decisions(str(tmp_path))
        assert sorted(os.path.basename(x) for x in d["optimize"]) == ["a", "b"]
        assert d["create"] == []


class TestSkillDirsIn:
    def test_root_may_be_the_skill_or_its_parent(self, tmp_path):
        alpha = write_skill(tmp_path, "alpha")
        write_skill(tmp_path, "beta")
        os.makedirs(tmp_path / "not-a-skill")
        assert skill_dirs_in(alpha) == [alpha]
        assert [os.path.basename(p) for p in skill_dirs_in(str(tmp_path))] == ["alpha", "beta"]
        assert skill_dirs_in(str(tmp_path / "missing")) == []
