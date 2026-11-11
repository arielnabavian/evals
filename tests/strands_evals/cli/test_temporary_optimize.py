import json
import sys
import textwrap
import types
from pathlib import Path

import pytest
from strands import Agent

from strands_evals.case import Case
from strands_evals.cli.main import main
from strands_evals.evaluators.deterministic import Contains, Equals
from strands_evals.experiment import Experiment
from tests.strands_evals.optimization.helpers import Says

DIRECT_HARNESS = """
from strands_harness import create_harness

print("loading the agent")
agent = create_harness(model="bedrock/test", instructions="Answer briefly.", effort="low")
"""

FACTORY_HARNESS = """
from strands_harness import create_harness


def create_agent(**overrides):
    return create_harness(instructions="Answer briefly.", **overrides)


agent = create_agent()
"""


class FakeHarness(types.ModuleType):
    def __init__(self):
        super().__init__("strands_harness")
        self.calls = []

    def create_harness(self, **options):
        self.calls.append(options)
        effort = options.get("effort", "auto")
        if effort == "max":
            raise ValueError("Effort 'max' is not supported by Bedrock model test.")
        return Agent(model=Says(text=f"effort {effort}"), callback_handler=None)


@pytest.fixture
def harness(monkeypatch):
    fake = FakeHarness()
    monkeypatch.setitem(sys.modules, "strands_harness", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return fake


def write_experiment(directory: Path, cases: int = 2, evaluators=None) -> Path:
    path = directory / "experiment.json"
    experiment = Experiment(
        cases=[Case(name=f"case{index}", input=f"question {index}") for index in range(cases)],
        evaluators=evaluators or [Contains(value="effort high")],
    )
    experiment.to_file(str(path))
    return path


def write_harness(directory: Path, source: str) -> Path:
    path = directory / "agent.py"
    path.write_text(textwrap.dedent(source))
    return path


def optimize(*arguments: str) -> int:
    return main(["optimize", *arguments])


def read_record(record: Path) -> list[dict]:
    return [json.loads(line) for line in (record / "study.jsonl").read_text().splitlines()]


def space_of(events: list[dict]) -> dict[str, dict]:
    header = next(event for event in events if event["type"] == "study")
    return {info["key"]: info for info in header["space"]}


def test_should_tune_effort_over_the_levels_the_model_accepts_when_given_a_harness_file(harness, tmp_path, capsys):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, DIRECT_HARNESS)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(record),
        "--iterations",
        "1",
        "--no-split",
        "--json",
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["format"] == "strands-evals.optimization.result"
    effort = space_of(read_record(record))["effort"]
    assert effort["default"] == "low"
    assert "max" not in effort["choices"]
    assert {"low", "high"} <= set(effort["choices"])


def test_should_build_quiet_trial_agents_without_session_or_memory_when_given_a_harness_file(harness, tmp_path):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, DIRECT_HARNESS)

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(tmp_path / "record"),
        "--iterations",
        "1",
        "--no-split",
        "--json",
    )

    assert code == 0
    trial_calls = harness.calls[1:]
    assert trial_calls
    assert all(call["session"] is False and call["memory"] is False for call in trial_calls)
    assert all(call["callback_handler"] is None for call in trial_calls)
    assert all(call["model"] == "bedrock/test" for call in trial_calls)
    assert all(call["instructions"] == "Answer briefly." for call in trial_calls)


def test_should_start_from_the_file_instructions_when_tuning_instructions(harness, tmp_path):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, DIRECT_HARNESS)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--record",
        str(record),
        "--iterations",
        "0",
        "--no-split",
        "--json",
    )

    assert code == 0
    space = space_of(read_record(record))
    assert list(space) == ["instructions"]
    assert space["instructions"]["default"] == "Answer briefly."


def test_should_pass_tuned_settings_to_create_agent_when_the_file_defines_it(harness, tmp_path):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, FACTORY_HARNESS)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(record),
        "--iterations",
        "1",
        "--no-split",
        "--json",
    )

    assert code == 0
    measured = {e["params"]["effort"] for e in read_record(record) if e["type"] == "candidate"}
    built = {call.get("effort") for call in harness.calls[1:]}
    assert measured <= built
    assert all(call["instructions"] == "Answer briefly." for call in harness.calls)


def test_should_test_the_selection_on_held_out_cases_when_the_split_is_default(harness, tmp_path, capsys):
    experiment = write_experiment(tmp_path, cases=8)
    agent = write_harness(tmp_path, DIRECT_HARNESS)

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(tmp_path / "record"),
        "--iterations",
        "1",
        "--json",
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["tested"] is True


def test_should_maximize_every_evaluator_when_no_metric_is_given(harness, tmp_path):
    experiment = write_experiment(tmp_path, evaluators=[Contains(value="effort high"), Equals(value="effort high")])
    agent = write_harness(tmp_path, DIRECT_HARNESS)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(record),
        "--iterations",
        "0",
        "--no-split",
        "--json",
    )

    assert code == 0
    header = read_record(record)[0]
    assert [goal["metric"] for goal in header["objectives"]] == ["Contains", "Equals"]
    assert header["combine"]["kind"] == "WeightedSum"


def test_should_print_a_summary_when_the_output_is_rich(harness, tmp_path, capsys):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, DIRECT_HARNESS)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        str(agent),
        "--tune",
        "effort",
        "--record",
        str(record),
        "--iterations",
        "1",
        "--no-split",
        "--rich",
    )

    assert code == 0
    output = capsys.readouterr().out
    assert output.startswith("ships ")
    assert f"record: {record}" in output


def test_should_tune_a_tunable_factory_when_given_module_attr(tmp_path):
    experiment = write_experiment(tmp_path)
    record = tmp_path / "record"

    code = optimize(
        str(experiment),
        "--agent",
        "tests.strands_evals.cli.fixtures.temporary_optimize_agents:create_agent",
        "--record",
        str(record),
        "--iterations",
        "1",
        "--no-split",
        "--json",
    )

    assert code == 0
    assert list(space_of(read_record(record))) == ["style"]


@pytest.mark.parametrize(
    ("source", "tune", "message"),
    [
        ("from strands_harness import create_harness\nhelper = create_harness()\n", "effort", 'as "agent"'),
        (
            "from strands_harness import create_harness\nbuilt = create_harness()\nagent = built\n",
            "effort",
            "changes its agent after create_harness",
        ),
        (
            "from strands_harness import create_harness\n"
            "from tests.strands_evals.optimization.helpers import Says\n"
            "agent = create_harness(model=Says(text='x'))\n",
            "effort",
            "Model instance",
        ),
        (
            "from strands import Agent\nfrom tests.strands_evals.optimization.helpers import Says\n"
            "agent = Agent(model=Says(text='x'), callback_handler=None)\n",
            "instructions",
            "must build its agent with create_harness",
        ),
    ],
)
def test_should_reject_the_harness_file_when_it_cannot_build_fresh_agents(
    harness, tmp_path, capsys, source, tune, message
):
    experiment = write_experiment(tmp_path)
    agent = write_harness(tmp_path, source)

    code = optimize(str(experiment), "--agent", str(agent), "--tune", tune, "--record", str(tmp_path / "record"))

    assert code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ("tests.strands_evals.cli.fixtures.temporary_optimize_agents:plain_factory", "must name a @tunable factory"),
        ("tests.strands_evals.cli.fixtures.temporary_optimize_agents:create_agent --tune effort", "--tune applies"),
    ],
)
def test_should_reject_the_agent_when_module_attr_is_not_a_tunable_factory_or_takes_tune(
    tmp_path, capsys, agent, message
):
    experiment = write_experiment(tmp_path)

    code = optimize(str(experiment), "--agent", *agent.split(), "--record", str(tmp_path / "record"))

    assert code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [
        ["--tune", "temperature"],
        ["--split", "0.5,0.5"],
        ["--at-most", "tokens"],
        ["--iterations", "-1"],
        ["--width", "0"],
        ["--split", "0.5,0.25,0.25", "--no-split"],
    ],
)
def test_should_exit_with_bad_input_when_a_flag_is_invalid(tmp_path, flags):
    experiment = write_experiment(tmp_path)

    with pytest.raises(SystemExit) as raised:
        optimize(str(experiment), "--agent", "agent.py", "--record", str(tmp_path / "record"), *flags)

    assert raised.value.code == 2
