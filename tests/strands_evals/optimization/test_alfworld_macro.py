"""The macro in the ALFWorld workload and harness (alfworld_programs.py wired into alfworld_workload.py and
alfworld_harness.py): a trial whose agent calls the macro tool, in the scripted game process; the harness's --macro
space (no ALFWorld install, no network; the harness's checks need ALFWorld's data, for its command list)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("openai")

from strands import Agent  # noqa: E402
from strands.tools.executors import SequentialToolExecutor  # noqa: E402

from strands_evals import Case, Experiment  # noqa: E402
from strands_evals.optimization import tunable  # noqa: E402
from strands_evals.optimization.programs import EMPTY_VALUE  # noqa: E402
from strands_evals.optimization.trials import TrialStatus, run_trials  # noqa: E402

from .test_alfworld import FAKE_GAME, Plays  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "optimization"))

import alfworld_programs as programs  # noqa: E402
import alfworld_workload as workload  # noqa: E402

CARRY = """import re


class Carry(Macro):
    description = "Go to where the object is, take it, and move it to the destination."
    parameters = {"thing": "the object, for example mug", "destination": "where it goes, for example shelf 1"}

    def run(self, act, start, thing, destination):
        for place in re.findall(r"an? ([a-z]+ \\d+)", start.split("Your task")[0]):
            seen = act(f"go to {place}")
            found = re.search(rf"an? ({thing} \\d+)", seen)
            if found:
                act(f"take {found.group(1)} from {place}")
                return act(f"move {found.group(1)} to {destination}")
        return f"No {thing} found."
"""


class CallsMacro(Plays):
    """Calls the macro tool once with ``arguments``, then replies in text."""

    def __init__(self, arguments):
        super().__init__([], tool="carry")
        self.arguments = arguments

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        index, self.calls = self.calls, self.calls + 1
        yield {"messageStart": {"role": "assistant"}}
        if index == 0:
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "m0", "name": "carry"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.arguments)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "I am done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                "metrics": {"latencyMs": 1},
            }
        }


@pytest.fixture
def game(tmp_path, monkeypatch):
    script = tmp_path / "fake_game.py"
    script.write_text(FAKE_GAME)
    monkeypatch.setattr(workload, "GAME_SCRIPT", script)
    return str(tmp_path / "game.tw-pddl")


def test_a_trial_whose_agent_calls_the_macro_counts_and_records_its_commands(game):
    cls = programs.macro_variable().arguments["version"].load_source(CARRY)

    @tunable()
    def factory():
        tools = [workload.act, *programs.macro_tools(cls())]
        model = CallsMacro({"thing": "mug", "destination": "shelf 1"})
        return Agent(model=model, tools=tools, tool_executor=SequentialToolExecutor(), callback_handler=None)

    case = Case(name="g", input="Your task is to: put a mug in shelf.", metadata={"game_file": game, "task_type": "x"})
    experiment = Experiment(cases=[case], evaluators=[workload.Success(name="success"), workload.Steps(name="steps")])
    run = run_trials(
        factory,
        factory.default_values,
        experiment,
        repetitions=1,
        max_concurrency=1,
        task=workload.ALFWorldTask(),
        retries=0,
    )
    (trial,) = asyncio.run(run)
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["success"] == 1.0 and trial.metrics["steps"] == 4.0  # the macro's commands are the game's
    assert trial.metrics["model_calls"] == 1.0  # the game was won inside the macro: the agent was not asked again
    output = json.loads(trial.output)
    assert output["commands"] == ["go to shelf 1", "go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1"]
    (call,) = output["macro_calls"]
    assert call["macro"] == "Carry" and call["arguments"] == {"thing": "mug", "destination": "shelf 1"}
    assert [c for c, _ in call["steps"]] == output["commands"]
    text = programs.transcript(trial, 4000)
    assert '[call] carry {"thing": "mug", "destination": "shelf 1"}' in text
    assert "    > move mug 1 to shelf 1 => You move the mug 1 to the shelf 1." in text


def test_a_trial_without_a_macro_has_no_macro_calls_in_its_output(game):
    @tunable()
    def factory():
        return Agent(model=Plays(["look"]), tools=[workload.act, *programs.macro_tools(None)], callback_handler=None)

    case = Case(name="g", input="Your task is to: put a mug in shelf.", metadata={"game_file": game, "task_type": "x"})
    experiment = Experiment(cases=[case], evaluators=[workload.Success(name="success")])
    run = run_trials(
        factory,
        factory.default_values,
        experiment,
        repetitions=1,
        max_concurrency=1,
        task=workload.ALFWorldTask(),
        retries=0,
    )
    (trial,) = asyncio.run(run)
    assert "macro_calls" not in json.loads(trial.output)


@pytest.mark.skipif(not (workload.DATA.parent / "logic" / "alfred.twl2").exists(), reason="needs ALFWorld's data")
def test_the_harness_tunes_the_macro_from_none(monkeypatch):
    monkeypatch.setenv("ALFWORLD_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ALFWORLD_MODEL", "m")
    import alfworld_harness as harness

    factory, texts = harness.SPACES["macro"]
    assert texts == [] and [p.key for p in factory.params()] == ["macro.version"]
    assert factory.default_values.values == {"macro.version": EMPTY_VALUE}
    agent = factory()
    assert sorted(agent.tool_names) == ["act"]  # the baseline has no macro tool
    mini = harness.with_macro(harness.mini)
    assert "macro.version" in {p.key for p in mini.params()} and "system_prompt" in {p.key for p in mini.params()}
    context = harness.MACRO.arguments["version"].context
    assert context.startswith(programs.CONTEXT) and harness.SYSTEM_PROMPT in context
    cls = harness.MACRO.arguments["version"].load_source(CARRY)
    agent = harness.player(macro=cls())
    assert sorted(agent.tool_names) == ["act", "carry"]


@pytest.mark.skipif(not (workload.DATA.parent / "logic" / "alfred.twl2").exists(), reason="needs ALFWorld's data")
def test_the_harness_tunes_skills_from_none_and_an_empty_library_adds_no_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("ALFWORLD_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ALFWORLD_MODEL", "m")
    import alfworld_harness as harness

    both = harness.with_skills(harness.with_macro(harness.ladder))
    keys = {p.key: p.type for p in both.params()}
    assert keys["skills"] == "skills" and keys["macro.version"] == "program" and "conversation_manager" in keys
    assert both.default_values.values["skills"] is None
    plain = harness.player(skills=None)
    assert sorted(plain.tool_names) == ["act"] and "available_skills" not in str(plain.system_prompt)
    library = tmp_path / "lib" / "find-things"
    library.mkdir(parents=True)
    (library / "SKILL.md").write_text(
        "---\nname: find-things\ndescription: How to search.\n---\nGo to each receptacle.\n"
    )
    agent = harness.player(skills=str(tmp_path / "lib"))
    assert sorted(agent.tool_names) == ["act", "skills"]
    study = harness.make_study(
        "L2", "best", repetitions=1, record=None, search_per_type=1, validation_per_type=1, test_games=1, skills=True
    )
    curation = study.blocks["skills"].method
    assert curation.options["model_config"]["max_tokens"] == harness.REFLECTION_MAX_TOKENS
    helper = harness.with_skills(harness.with_macro(harness.ladder, helper=True))
    assert next(p for p in helper.params() if p.key == "macro.version").domain.interface is programs.HelperMacro
    assert next(p for p in harness.SPACES["macro-helper"][0].params()).to_dict()["settings"]["max_commands"] == 8
    floor = harness.goals("floor", floor=0.5)
    assert floor["constraints"][0].limit == 0.5 and floor["constraints"][0].compare == "absolute"
