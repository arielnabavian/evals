import json
import shlex
import sys
from pathlib import Path

import pytest

from strands_evals.case import Case
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
