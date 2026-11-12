from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from strands_evals.cli.commands._temporary_generate_progress import (
    TemporaryGenerateProgress,
    TemporaryProgressGenerator,
    temporary_error_text,
)
from strands_evals.cli.main import main
from strands_evals.evaluators import OutputEvaluator
from strands_evals.generators.topic_planner import Topic, TopicPlan

AGENT = "strands_evals.cli.commands._temporary_generate_progress.Agent"
RUBRIC_AGENT = "strands_evals.generators.experiment_generator.Agent"


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class _CaseAgent:
    failures: list[BaseException] = []

    def __init__(self, **_kwargs):
        self.calls = 0

    async def structured_output_async(self, output_model, prompt):
        if _CaseAgent.failures:
            raise _CaseAgent.failures.pop(0)
        return output_model(name="refund", input=f"question for {prompt[-20:]}", expected_output="answer")


class _RubricAgent:
    def __init__(self, **_kwargs):
        pass

    async def invoke_async(self, _prompt):
        return "Score answers that issue the refund."


@pytest.fixture(autouse=True)
def _reset_failures():
    _CaseAgent.failures = []


def _generator(path: Path) -> tuple[TemporaryProgressGenerator, TemporaryGenerateProgress]:
    progress = TemporaryGenerateProgress(str(path))
    return TemporaryProgressGenerator(progress, input_type=str, output_type=str), progress


@pytest.mark.asyncio
async def test_should_report_each_written_case_when_generation_runs(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    generator, progress = _generator(path)

    with patch(AGENT, _CaseAgent):
        cases = await generator.generate_cases_async("Write refund cases.", num_cases=3)
    progress.close()

    assert len(cases) == 3
    assert _events(path) == [
        {"event": "cases", "total": 3},
        {"event": "case", "ok": True, "done": 1, "failed": 0, "total": 3},
        {"event": "case", "ok": True, "done": 2, "failed": 0, "total": 3},
        {"event": "case", "ok": True, "done": 3, "failed": 0, "total": 3},
    ]


@pytest.mark.asyncio
async def test_should_report_the_error_of_a_failed_case_when_the_model_call_raises(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    generator, progress = _generator(path)
    _CaseAgent.failures = [RuntimeError("Too many requests,\n  please wait")]

    with patch(AGENT, _CaseAgent):
        cases = await generator.generate_cases_async("Write refund cases.", num_cases=2)
    progress.close()

    failed = [event for event in _events(path) if event.get("ok") is False]
    assert len(cases) == 1
    assert failed == [
        {
            "event": "case",
            "ok": False,
            "done": 0,
            "failed": 1,
            "total": 2,
            "error": "RuntimeError: Too many requests, please wait",
        }
    ]
    assert _events(path)[-1]["done"] + _events(path)[-1]["failed"] == 2


@pytest.mark.asyncio
async def test_should_fail_every_queued_case_instead_of_hanging_when_the_agent_cannot_start(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    generator, progress = _generator(path)

    with patch(AGENT, side_effect=ValueError("unknown model id")):
        cases = await generator.generate_cases_async("Write refund cases.", num_cases=4)
    progress.close()

    last = _events(path)[-1]
    assert cases == []
    assert last == {
        "event": "case",
        "ok": False,
        "done": 0,
        "failed": 4,
        "total": 4,
        "error": "ValueError: unknown model id",
    }


@pytest.mark.asyncio
async def test_should_report_topic_and_rubric_phases_when_the_context_mode_uses_them(tmp_path: Path):
    path = tmp_path / "progress.jsonl"
    generator, progress = _generator(path)
    plan = TopicPlan(
        topics=[
            Topic(title="refunds", description="refund rules", key_aspects=["window"]),
            Topic(title="exchanges", description="exchange rules", key_aspects=["size"]),
        ]
    )

    with (
        patch(AGENT, _CaseAgent),
        patch(RUBRIC_AGENT, _RubricAgent),
        patch("strands_evals.generators.topic_planner.TopicPlanner.plan_topics_async", AsyncMock(return_value=plan)),
    ):
        experiment = await generator.from_context_async(
            context="A refund agent.",
            task_description="Answer refund questions",
            num_cases=3,
            evaluator=OutputEvaluator,
            num_topics=2,
        )
    progress.close()

    phases = [event["event"] for event in _events(path) if event["event"] != "case"]
    assert len(experiment.cases) == 3
    assert phases == ["topics", "cases", "rubric"]
    assert _events(path)[1] == {"event": "cases", "total": 3}


def test_should_write_a_failed_event_when_the_generate_command_raises(tmp_path: Path):
    path = tmp_path / "progress.jsonl"

    with patch(
        "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async",
        AsyncMock(side_effect=PermissionError("AccessDeniedException: no access to the model")),
    ):
        exit_code = main(["generate", "--context", "ctx", "--temporary-progress", str(path)])

    assert exit_code == 3
    assert _events(path) == [
        {"event": "failed", "error": "PermissionError: AccessDeniedException: no access to the model"}
    ]


def test_should_cut_long_error_text_when_it_passes_the_limit():
    text = temporary_error_text(RuntimeError("x" * 900))

    assert len(text) == 500
    assert text.endswith("…")
