import json
import logging
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

FACTORY = "tests.strands_evals.cli.fixtures.temporary_optimize_agents"


class FakeHarness(types.ModuleType):
    def __init__(self):
        super().__init__("strands_harness")
        self.calls = []

    def create_harness(self, **options):
        self.calls.append(options)
        effort = options.get("effort", "auto")
        if options.get("model") == "bedrock/nolevels" and effort not in ("auto", "off"):
            raise ValueError(f"Effort {effort!r} is not supported by Bedrock model nolevels.")
        if effort == "max":
            raise ValueError("Effort 'max' is not supported by Bedrock model test.")
        return Agent(model=Says(text=f"effort {effort}"), callback_handler=None)


@pytest.fixture
def harness(monkeypatch):
    fake = FakeHarness()
    monkeypatch.setitem(sys.modules, "strands_harness", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return fake


@pytest.fixture
def fresh_modules():
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


def write_experiment(directory: Path, cases: int = 2, evaluators=None) -> Path:
    path = directory / "experiment.json"
    experiment = Experiment(
        cases=[Case(name=f"case{index}", input=f"question {index}") for index in range(cases)],
        evaluators=evaluators or [Contains(value="effort high")],
    )
    experiment.to_file(str(path))
    return path


def write_harness(directory: Path, source: str, name: str = "agent.py") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


def optimize(*arguments: str) -> int:
    return main(["optimize", *arguments])


def optimize_harness(tmp_path: Path, source: str, *flags: str, cases: int = 2, evaluators=None) -> int:
    experiment = write_experiment(tmp_path, cases=cases, evaluators=evaluators)
    agent = write_harness(tmp_path, source)
    return optimize(str(experiment), "--agent", str(agent), "--record", str(tmp_path / "record"), *flags)


def read_record(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "record" / "study.jsonl").read_text().splitlines()]


def header_of(tmp_path: Path) -> dict:
    return next(event for event in read_record(tmp_path) if event["type"] == "study")


def space_of(tmp_path: Path) -> dict[str, dict]:
    return {info["key"]: info for info in header_of(tmp_path)["space"]}


def test_should_tune_effort_over_the_levels_the_model_accepts_when_given_a_harness_file(harness, tmp_path, capsys):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "1", "--no-split", "--json")

    assert code == 0
    assert json.loads(capsys.readouterr().out)["format"] == "strands-evals.optimization.result"
    effort = space_of(tmp_path)["effort"]
    assert effort["default"] == "low"
    assert {"low", "high"} <= set(effort["choices"])
    assert not {"max", "auto"} & set(effort["choices"])


def test_should_build_quiet_trial_agents_without_session_or_memory_when_given_a_harness_file(harness, tmp_path):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "1", "--no-split", "--json")

    assert code == 0
    trial_calls = harness.calls[1:]
    assert trial_calls
    assert all(call["session"] is False and call["memory"] is False for call in trial_calls)
    assert all(call["callback_handler"] is None for call in trial_calls)
    assert all(call["model"] == "bedrock/test" for call in trial_calls)
    assert all(call["instructions"] == "Answer briefly." for call in trial_calls)


def test_should_start_from_the_file_instructions_when_tuning_instructions(harness, tmp_path):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--iterations", "0", "--no-split", "--json")

    assert code == 0
    space = space_of(tmp_path)
    assert list(space) == ["instructions"]
    assert space["instructions"]["default"] == "Answer briefly."


def test_should_build_agents_without_instructions_when_the_file_has_none(harness, tmp_path):
    source = "from strands_harness import create_harness\nagent = create_harness(model='bedrock/test')\n"

    code = optimize_harness(tmp_path, source, "--iterations", "0", "--no-split", "--json")

    assert code == 0
    assert space_of(tmp_path)["instructions"]["default"] == ""
    assert all(call["instructions"] is None for call in harness.calls[1:])


def test_should_pass_tuned_settings_to_create_agent_when_the_file_defines_it(harness, tmp_path):
    code = optimize_harness(tmp_path, FACTORY_HARNESS, "--tune", "effort", "--iterations", "1", "--no-split", "--json")

    assert code == 0
    measured = {event["params"]["effort"] for event in read_record(tmp_path) if event["type"] == "candidate"}
    built = {call.get("effort") for call in harness.calls[1:]}
    assert measured <= built
    assert all(call["instructions"] == "Answer briefly." for call in harness.calls)


def test_should_import_the_file_as_a_package_module_when_it_uses_relative_imports(
    harness, fresh_modules, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    write_harness(tmp_path, "", "desk/__init__.py")
    write_harness(tmp_path, "INSTRUCTIONS = 'Answer from the desk.'\n", "desk/prompts.py")
    agent = write_harness(
        tmp_path,
        "from strands_harness import create_harness\n"
        "from .prompts import INSTRUCTIONS\n"
        "agent = create_harness(model='bedrock/test', instructions=INSTRUCTIONS)\n",
        "desk/agent.py",
    )
    experiment = write_experiment(tmp_path)

    code = optimize(str(experiment), "--agent", str(agent), "--record", "record", "--iterations", "0", "--no-split")

    assert code == 0
    assert space_of(tmp_path)["instructions"]["default"] == "Answer from the desk."


def test_should_test_the_selection_on_held_out_cases_when_the_split_is_default(harness, tmp_path, capsys):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "1", "--json", cases=8)

    assert code == 0
    assert json.loads(capsys.readouterr().out)["tested"] is True


def test_should_write_json_when_stdout_is_not_a_terminal(harness, tmp_path, capsys):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "0", "--no-split")

    assert code == 0
    assert json.loads(capsys.readouterr().out)["format"] == "strands-evals.optimization.result"


def test_should_print_a_summary_when_the_output_is_rich(harness, tmp_path, capsys):
    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "1", "--no-split", "--rich")

    assert code == 0
    output = capsys.readouterr().out
    assert output.startswith("ships ")
    assert f"record: {tmp_path / 'record'}" in output


def test_should_maximize_every_evaluator_when_no_metric_is_given(harness, tmp_path):
    evaluators = [Contains(value="effort high"), Equals(value="effort high")]

    code = optimize_harness(
        tmp_path, DIRECT_HARNESS, "--tune", "effort", "--iterations", "0", "--no-split", evaluators=evaluators
    )

    assert code == 0
    header = header_of(tmp_path)
    assert [goal["metric"] for goal in header["objectives"]] == ["Contains", "Equals"]
    assert header["combine"]["kind"] == "WeightedSum"


def test_should_use_the_given_goals_when_maximize_and_at_most_are_given(harness, tmp_path):
    evaluators = [Contains(value="effort high"), Equals(value="effort high")]

    code = optimize_harness(
        tmp_path,
        DIRECT_HARNESS,
        "--tune",
        "effort",
        "--maximize",
        "Equals",
        "--at-most",
        "tokens=500",
        "--iterations",
        "0",
        "--no-split",
        evaluators=evaluators,
    )

    assert code == 0
    header = header_of(tmp_path)
    assert [goal["metric"] for goal in header["objectives"]] == ["Equals"]
    assert [(goal["metric"], goal["limit"]) for goal in header["constraints"]] == [("tokens", 500.0)]


def test_should_rewrite_instructions_with_the_given_model_when_rewriter_model_is_given(harness, tmp_path):
    code = optimize_harness(
        tmp_path, DIRECT_HARNESS, "--rewriter-model", "rewriter-x", "--iterations", "0", "--no-split"
    )

    assert code == 0
    blocks = header_of(tmp_path)["blocks"].values()
    rewriters = [block for block in blocks if block["method"]["kind"] == "Reflection"]
    assert [block["keys"] for block in rewriters] == [["instructions"]]
    assert "rewriter-x" in rewriters[0]["method"]["model"]


def test_should_ignore_the_rewriter_model_when_no_setting_is_text(harness, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="strands_evals"):
        code = optimize_harness(
            tmp_path, DIRECT_HARNESS, "--tune", "effort", "--rewriter-model", "rewriter-x", "--iterations", "0"
        )

    assert code == 0
    assert "no text to rewrite" in caplog.text
    assert all(block["method"]["kind"] != "Reflection" for block in header_of(tmp_path)["blocks"].values())


def test_should_exit_with_a_runtime_error_when_every_baseline_trial_fails(harness, tmp_path, capsys):
    source = """
    from strands_harness import create_harness


    def create_agent(**overrides):
        if overrides:
            raise RuntimeError("the agent cannot be built")
        return create_harness(instructions="Answer briefly.")


    agent = create_agent()
    """

    code = optimize_harness(tmp_path, source, "--iterations", "0", "--no-split")

    assert code == 3
    assert "strands-evals: error:" in capsys.readouterr().err


def test_should_refuse_the_record_before_loading_the_agent_when_it_holds_a_study(harness, tmp_path, capsys):
    (tmp_path / "record").mkdir()
    (tmp_path / "record" / "study.jsonl").write_text('{"type": "study"}\n')

    code = optimize_harness(tmp_path, DIRECT_HARNESS, "--iterations", "0")

    assert code == 2
    assert "must be a new or empty directory" in capsys.readouterr().err
    assert harness.calls == []


def test_should_tune_a_tunable_factory_when_given_module_attr(tmp_path):
    experiment = write_experiment(tmp_path)

    code = optimize(
        str(experiment),
        "--agent",
        f"{FACTORY}:create_agent",
        "--record",
        str(tmp_path / "record"),
        "--iterations",
        "1",
        "--no-split",
        "--json",
    )

    assert code == 0
    assert list(space_of(tmp_path)) == ["style"]


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
        (
            "from strands_harness import create_harness\nagent = create_harness(session_manager=object())\n",
            "effort",
            "session_manager",
        ),
        (
            "from strands_harness import create_harness\nagent = create_harness(system_prompt='You are terse.')\n",
            "instructions",
            "system_prompt",
        ),
        (
            "from strands_harness import create_harness\n"
            "async def create_agent(**overrides):\n"
            "    return create_harness(**overrides)\n"
            "agent = create_harness()\n",
            "effort",
            "async",
        ),
        (
            "from strands_harness import create_harness\nagent = create_harness(model='bedrock/nolevels')\n",
            "effort",
            "no reasoning levels",
        ),
    ],
)
def test_should_reject_the_harness_file_when_it_cannot_build_fresh_agents(
    harness, tmp_path, capsys, source, tune, message
):
    code = optimize_harness(tmp_path, source, "--tune", tune)

    assert code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        (f"{FACTORY}:plain_factory", "must name a @tunable factory"),
        (f"{FACTORY}:create_agent --tune effort", "--tune applies"),
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
