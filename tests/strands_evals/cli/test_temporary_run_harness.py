import json
import sys

import pytest

from strands_evals.case import Case
from strands_evals.cli.main import main
from strands_evals.evaluators.deterministic import Contains, Equals
from strands_evals.experiment import Experiment
from tests.strands_evals.cli.test_temporary_optimize import DIRECT_HARNESS, FakeHarness, write_harness
from tests.strands_evals.cli.test_temporary_typescript_optimize import DESCRIPTION, RUNNER_COMMAND


@pytest.fixture
def harness(monkeypatch):
    fake = FakeHarness()
    monkeypatch.setitem(sys.modules, "strands_harness", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return fake


def write_experiment(tmp_path):
    path = tmp_path / "experiment.json"
    Experiment(
        cases=[Case(name=f"case{index}", input=f"question {index}") for index in range(2)],
        evaluators=[Contains(value="effort low")],
    ).to_file(str(path))
    return path


def report_rows(path) -> list[dict]:
    report = json.loads(path.read_text())
    return [
        {"output": case["actual_output"].strip(), "passed": passed}
        for case, passed in zip(report["cases"], report["test_passes"], strict=True)
    ]


def test_should_run_each_case_on_a_fresh_agent_when_given_a_python_harness_file(harness, tmp_path):
    report = tmp_path / "report.json"
    agent = write_harness(tmp_path, DIRECT_HARNESS)

    code = main(["run", str(write_experiment(tmp_path)), "--agent", str(agent), "-o", str(report), "--fail-on", "none"])

    assert code == 0
    assert report_rows(report) == [{"output": "effort low", "passed": True}] * 2
    trial_calls = harness.calls[1:]
    assert len(trial_calls) == 2
    assert all(call["session"] is False and call["memory"] is False for call in trial_calls)


def test_should_score_the_answer_without_its_trailing_newline_when_given_a_python_harness_file(harness, tmp_path):
    report = tmp_path / "report.json"
    experiment = tmp_path / "experiment.json"
    Experiment(cases=[Case(name="case0", input="question 0")], evaluators=[Equals(value="effort low")]).to_file(
        str(experiment)
    )
    agent = write_harness(tmp_path, DIRECT_HARNESS)

    code = main(["run", str(experiment), "--agent", str(agent), "-o", str(report), "--fail-on", "none"])

    written = json.loads(report.read_text())
    assert code == 0
    assert [case["actual_output"] for case in written["cases"]] == ["effort low"]
    assert written["test_passes"] == [True]


def test_should_run_each_case_through_the_runner_when_given_a_typescript_harness_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_RUNNER_DESCRIPTION", json.dumps(DESCRIPTION))
    report = tmp_path / "report.json"
    agent = tmp_path / "agent.ts"
    agent.write_text("export const agent = createHarness({})\n")

    code = main(
        [
            "run",
            str(write_experiment(tmp_path)),
            "--agent",
            str(agent),
            "--typescript-runner",
            RUNNER_COMMAND,
            "-o",
            str(report),
            "--fail-on",
            "none",
        ]
    )

    assert code == 0
    assert report_rows(report) == [{"output": "effort low", "passed": True}] * 2


def test_should_ask_for_the_runner_when_run_gets_a_typescript_agent_without_one(tmp_path, capsys):
    agent = tmp_path / "agent.ts"
    agent.write_text("export const agent = createHarness({})\n")

    code = main(["run", str(write_experiment(tmp_path)), "--agent", str(agent), "-o", str(tmp_path / "report.json")])

    assert code == 2
    assert "--typescript-runner" in capsys.readouterr().err
