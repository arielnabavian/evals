import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from strands_evals.case import Case
from strands_evals.cli.commands._temporary_typescript_runner import TypeScriptRunner
from strands_evals.cli.main import main
from strands_evals.evaluators.deterministic import Contains
from strands_evals.experiment import Experiment

RUNNER = Path(__file__).parent / "fixtures" / "temporary_typescript_runner.py"
RUNNER_COMMAND = f"{shlex.quote(sys.executable)} {shlex.quote(str(RUNNER))}"
DESCRIPTION = {"instructions": "Answer briefly.", "effort": "low", "model": "bedrock/test"}


@pytest.fixture
def runner_log(tmp_path, monkeypatch):
    log = tmp_path / "runner.jsonl"
    monkeypatch.setenv("FAKE_RUNNER_LOG", str(log))
    return log


def requests_of(log: Path, kind: str) -> list[dict]:
    if not log.exists():
        return []
    return [request for line in log.read_text().splitlines() if (request := json.loads(line))["type"] == kind]


def optimize_typescript(tmp_path, monkeypatch, *flags, description=None, runner=RUNNER_COMMAND):
    monkeypatch.setenv("FAKE_RUNNER_DESCRIPTION", json.dumps(DESCRIPTION if description is None else description))
    experiment = tmp_path / "experiment.json"
    Experiment(
        cases=[Case(name=f"case{index}", input=f"question {index}") for index in range(2)],
        evaluators=[Contains(value="effort high")],
    ).to_file(str(experiment))
    agent = tmp_path / "agent.ts"
    agent.write_text("export const agent = createHarness({})\n")
    runner_flags = ["--typescript-runner", runner] if runner else []
    return main(
        [
            "optimize",
            str(experiment),
            "--agent",
            str(agent),
            "--record",
            str(tmp_path / "record"),
            *runner_flags,
            *flags,
        ]
    )


def header_of(tmp_path) -> dict:
    lines = (tmp_path / "record" / "study.jsonl").read_text().splitlines()
    return next(event for event in map(json.loads, lines) if event["type"] == "study")


def space_of(tmp_path) -> dict[str, dict]:
    return {info["key"]: info for info in header_of(tmp_path)["space"]}


def baseline_trials(tmp_path) -> list[dict]:
    lines = (tmp_path / "record" / "study.jsonl").read_text().splitlines()
    return next(event for event in map(json.loads, lines) if event["type"] == "candidate")["trials"]


def transcript_input(tmp_path, trial: dict) -> str:
    messages = json.loads((tmp_path / "record" / "transcripts" / f"{trial['transcript']['hash']}.json").read_text())
    return messages[0]["content"][0]["text"]


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr("strands_evals.optimization.trials.RETRY_WAIT", 0.0)


def test_should_tune_effort_through_the_runner_when_given_a_typescript_agent(tmp_path, monkeypatch, runner_log, capsys):
    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "1", "--no-split", "--json")

    assert code == 0
    assert json.loads(capsys.readouterr().out)["format"] == "strands-evals.optimization.result"
    effort = space_of(tmp_path)["effort"]
    assert effort["default"] == "low"
    assert {"low", "high"} <= set(effort["choices"])
    assert not {"max", "auto"} & set(effort["choices"])
    assert requests_of(runner_log, "run")
    assert list((tmp_path / "record" / "transcripts").iterdir())


def test_should_send_the_file_instructions_to_each_trial_when_tuning_instructions(tmp_path, monkeypatch, runner_log):
    code = optimize_typescript(tmp_path, monkeypatch, "--iterations", "0", "--no-split")

    assert code == 0
    assert space_of(tmp_path)["instructions"]["default"] == "Answer briefly."
    runs = requests_of(runner_log, "run")
    assert runs
    assert all(run["values"] == {"instructions": "Answer briefly."} for run in runs)


def test_should_count_the_runner_usage_when_trials_finish(tmp_path, monkeypatch, runner_log, capsys):
    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "0", "--no-split", "--json")

    assert code == 0
    agent_cost = json.loads(capsys.readouterr().out)["cost"]["by_role"]["agent"]
    assert (agent_cost["tokens"], agent_cost["model_calls"]) == (24, 2)


@pytest.mark.parametrize(
    ("description", "tune", "message"),
    [
        ({**DESCRIPTION, "shared_state": ["sessionManager"]}, "effort", "sessionManager"),
        ({**DESCRIPTION, "system_prompt": True}, "instructions", "system_prompt"),
        ({**DESCRIPTION, "model": None, "model_is_instance": True}, "effort", "Model instance"),
        ({"error": "Cannot find module '@strands-agents/harness'"}, "effort", "could not be loaded"),
    ],
)
def test_should_reject_the_typescript_agent_when_the_runner_reports_it_cannot_be_tuned(
    tmp_path, monkeypatch, capsys, description, tune, message
):
    code = optimize_typescript(tmp_path, monkeypatch, "--tune", tune, description=description)

    assert code == 2
    assert message in capsys.readouterr().err


def test_should_ask_for_the_runner_when_a_typescript_agent_has_none(tmp_path, monkeypatch, capsys):
    code = optimize_typescript(tmp_path, monkeypatch, runner=None)

    assert code == 2
    assert "--typescript-runner" in capsys.readouterr().err


def test_should_report_the_runner_command_when_it_cannot_start(tmp_path, monkeypatch, capsys):
    code = optimize_typescript(tmp_path, monkeypatch, runner=str(tmp_path / "missing-runner"))

    assert code == 2
    assert "cannot start the TypeScript runner" in capsys.readouterr().err


def test_should_match_each_reply_to_its_trial_when_the_runner_answers_out_of_order(tmp_path, monkeypatch, runner_log):
    monkeypatch.setenv("FAKE_RUNNER_MODE", "delayed")

    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "0", "--no-split")

    assert code == 0
    trials = baseline_trials(tmp_path)
    assert len(trials) == 2
    assert all(transcript_input(tmp_path, trial) == f"question {trial['case'][-1]}" for trial in trials)


def test_should_count_the_usage_of_a_failed_trial_when_the_typescript_agent_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_RUNNER_MODE", "error")

    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "0", "--no-split", "--json")

    assert code == 0
    assert json.loads(capsys.readouterr().out)["cost"]["by_role"]["agent"]["tokens"] == 24
    assert {trial["case"]: trial["status"] for trial in baseline_trials(tmp_path)} == {
        "case0": "agent_failure",
        "case1": "completed",
    }


def test_should_run_the_trial_again_when_the_runner_reports_a_retryable_failure(
    tmp_path, monkeypatch, runner_log, no_retry_wait
):
    monkeypatch.setenv("FAKE_RUNNER_MODE", "retryable")

    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "0", "--no-split")

    assert code == 0
    assert [run["input"] for run in requests_of(runner_log, "run")].count("question 0") == 2
    assert {trial["status"] for trial in baseline_trials(tmp_path)} == {"completed"}


def test_should_stop_with_a_runtime_error_when_the_runner_exits_during_trials(
    tmp_path, monkeypatch, capsys, no_retry_wait
):
    monkeypatch.setenv("FAKE_RUNNER_MODE", "exit")

    code = optimize_typescript(tmp_path, monkeypatch, "--tune", "effort", "--iterations", "0", "--no-split")

    assert code == 3
    assert "TrialInvalid: the TypeScript runner has exited" in capsys.readouterr().err


async def test_should_keep_answering_requests_when_a_cancelled_run_replies_late(monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_MODE", "delayed")
    monkeypatch.setenv("FAKE_RUNNER_DESCRIPTION", json.dumps(DESCRIPTION))
    runner = TypeScriptRunner([*shlex.split(RUNNER_COMMAND), "agent.ts", "."])
    slow = asyncio.create_task(runner.call_async({"type": "run", "values": {}, "input": "question 0"}))
    await asyncio.sleep(0.05)

    slow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await slow
    quick = await asyncio.wait_for(runner.call_async({"type": "run", "values": {}, "input": "question 1"}), 5)
    await asyncio.sleep(0.4)
    described = await asyncio.wait_for(runner.call_async({"type": "describe"}), 5)
    runner.close()

    assert quick["output"] == "effort low"
    assert described["instructions"] == "Answer briefly."
