"""ALFWorld macros (examples/optimization/alfworld_programs.py): a macro version loaded under the declaration's rules,
run as the agent's tool against a game, its commands counted and recorded; its checks; how the proposer reads a trial
(no network, no ALFWorld)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from strands import Agent, tool
from strands.models.model import Model

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "optimization"))

import alfworld_programs as programs  # noqa: E402
from alfworld_programs import (  # noqa: E402
    ALLOWED_IMPORTS,
    CONTEXT,
    Macro,
    check_macro,
    macro_tools,
    macro_variable,
    tool_name,
    transcript,
)

from strands_evals.optimization.programs import EMPTY, EMPTY_VALUE  # noqa: E402

FIND = """import re


class FindObject(Macro):
    description = "Go to each receptacle in the room until one holds the object; say where it is."
    parameters = {"name": "the object's name without its number, for example apple"}

    def run(self, act, start, name):
        places = re.findall(r"an? ([a-z]+ \\d+)", start.split("Your task")[0])
        for place in places:
            seen = act(f"go to {place}")
            if "closed" in seen:
                seen = act(f"open {place}")
            if name in seen:
                return f"Found at {place}: {seen}"
        return f"No {name} in: {', '.join(places)}"
"""


class Game:
    """An episode's interface (``step``, ``over``, ``won``): a room of three receptacles, the apple in the closed
    cabinet 1; taking it wins."""

    def __init__(self, max_steps=50):
        self.max_steps, self.steps, self.won, self.commands = max_steps, 0, False, []
        self.first = (
            "You are in the middle of a room. Looking quickly around you, you see a countertop 1, a cabinet 1, and a "
            "fridge 1.\n\nYour task is to: find the apple."
        )

    @property
    def over(self):
        return self.won or self.steps >= self.max_steps

    def step(self, command):
        if self.over:
            return "The game is over."
        self.steps += 1
        self.commands.append(command)
        answers = {
            "look": "You are in the middle of a room. Looking quickly around you, you see nothing.",
            "go to countertop 1": "You arrive at countertop 1. On the countertop 1, you see a knife 1.",
            "go to cabinet 1": "You arrive at cabinet 1. The cabinet 1 is closed.",
            "open cabinet 1": "You open the cabinet 1. The cabinet 1 is open. In it, you see a apple 1.",
            "take apple 1 from cabinet 1": "You pick up the apple 1 from the cabinet 1.",
        }
        self.won = command == "take apple 1 from cabinet 1"
        return answers.get(command, "Nothing happens.")


class Calls(Model):
    """Calls the tools in ``uses`` (name, input), one per model call, then answers "done"."""

    def __init__(self, uses):
        self.uses, self.calls, self.config = list(uses), 0, {"model_id": "calls"}

    def update_config(self, **kw):
        self.config.update(kw)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.specs = tool_specs
        yield {"messageStart": {"role": "assistant"}}
        if self.calls <= len(self.uses):
            name, given = self.uses[self.calls - 1]
            use = {"toolUseId": f"t{self.calls}", "name": name}
            yield {"contentBlockStart": {"start": {"toolUse": use}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(given)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}
        }


@tool(context=True)
def act(command: str, tool_context) -> str:
    """Run one command in the game.

    Args:
        command: The command.
    """
    episode = tool_context.invocation_state["episode"]
    observation = episode.step(command)
    if episode.over:
        tool_context.invocation_state.setdefault("request_state", {})["stop_event_loop"] = True
    return observation


def version():
    return macro_variable().arguments["version"]


def test_the_declaration_starts_from_no_macro_and_allows_only_its_imports():
    v = version()
    assert v.default == EMPTY_VALUE and v.seeds == (EMPTY,) and v.imports == ALLOWED_IMPORTS
    assert v.context == CONTEXT and v.load_source(EMPTY) is None and macro_tools(None) == []
    assert "import of 'os' is not allowed" in v.problem("import os\n" + FIND)
    assert "import of 'json' is not allowed" in v.problem("import json\n" + FIND)
    assert v.problem(FIND) is None and tool_name(v.load_source(FIND)) == "find_object"


def test_the_macro_is_a_tool_whose_commands_count_and_are_recorded():
    cls = version().load_source(FIND)
    game = Game()
    model = Calls([("find_object", {"name": "apple"}), ("act", {"command": "take apple 1 from cabinet 1"})])
    agent = Agent(model=model, tools=[act, *macro_tools(cls())], callback_handler=None)
    agent("Your task is to: find the apple.", episode=game)

    spec = next(s for s in model.specs if s["name"] == "find_object")
    assert spec["description"].startswith("Go to each receptacle")
    assert spec["inputSchema"]["json"]["required"] == ["name"]
    # the macro's commands are the game's: counted, in order, before the agent's own
    assert game.commands == [
        "go to countertop 1",
        "go to cabinet 1",
        "open cabinet 1",
        "take apple 1 from cabinet 1",
    ]
    assert game.won and game.steps == 4
    (call,) = game.macro_calls
    assert call["tool_use_id"] == "t1" and call["macro"] == "FindObject" and call["arguments"] == {"name": "apple"}
    assert [c for c, _ in call["steps"]] == game.commands[:3] and call["returned"].startswith("Found at cabinet 1")
    results = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    assert results[0]["status"] == "success" and results[0]["content"][0]["text"].startswith("Found at cabinet 1")


def test_a_macro_at_the_command_limit_stops_and_the_agent_is_told():
    cls = version().load_source(FIND.replace("if name in seen:", "if False:"))
    game = Game(max_steps=3)
    model = Calls([("find_object", {"name": "apple"})])
    agent = Agent(model=model, tools=[act, *macro_tools(cls())], callback_handler=None)
    agent("Your task is to: find the apple.", episode=game)
    assert game.steps == 3 and model.calls == 1  # the game ended in the macro: the agent was not asked again
    text = next(
        b["toolResult"]["content"][0]["text"] for m in agent.messages for b in m["content"] if "toolResult" in b
    )
    assert text.endswith("The limit of 3 commands is reached.")


def test_a_failing_macro_is_the_tools_error_and_the_game_goes_on():
    knife = (
        '            seen = act(f"go to {place}")\n'
        '            if "knife" in seen:\n'
        '                raise ValueError("no luck")\n'
    )
    cls = version().load_source(FIND.replace('            seen = act(f"go to {place}")\n', knife))
    game = Game()
    model = Calls([("find_object", {"name": "apple"}), ("act", {"command": "look"})])
    agent = Agent(model=model, tools=[act, *macro_tools(cls())], callback_handler=None)
    agent("Your task is to: find the apple.", episode=game)
    results = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    text = results[0]["content"][0]["text"]
    assert results[0]["status"] == "error" and text.startswith(
        "find_object failed after 1 commands: ValueError: no luck"
    )
    assert "Its last observation: You arrive at countertop 1." in text and results[1]["status"] == "success"
    assert game.macro_calls[0]["error"] == "ValueError: no luck" and game.steps == 2


def test_check_macro_refuses_what_the_agent_could_not_use():
    v = version()
    loops = FIND.replace(
        "        for place in places:", "        while True:\n            act('look')\n        for place in places:"
    )
    spins = FIND.replace("        places = re", "        while True:\n            pass\n        places = re")
    for source, why in (
        (FIND.replace('description = "Go', 'description = ""  # "Go'), "description must be a non-empty string"),
        (
            FIND.replace('{"name": "the', '{"class": "the').replace(
                "def run(self, act, start, name)", "def run(self, act, start, **kw)"
            ),
            "must be a Python identifier",
        ),
        (
            FIND.replace("def run(self, act, start, name)", "def run(self, act, start, thing)"),
            "does not match its signature",
        ),
        (FIND.replace("def run(self, act, start, name)", "def run(self, act, name)"), "does not match its signature"),
        (FIND.replace("class FindObject", "class Act"), "not 'act'"),
        (FIND.replace('return f"No {name}', 'return 3\n        return f"No {name}'), "must return a string"),
        (spins, "a dry run did not finish"),
        (FIND.replace("re.findall", "re.find_all"), "a dry run raised AttributeError"),
        (
            FIND.replace("        places = re", "        raise ValueError('nothing here')\n        places = re"),
            "raised ValueError",
        ),
    ):
        problem = v.problem(source)
        assert problem is not None and why in problem, (why, problem)
    assert v.problem(loops) is None  # a macro that plays to the command limit ends the game: allowed


def test_the_proposer_reads_each_macro_call_with_the_commands_it_ran():
    output = {
        "won": False,
        "macro_calls": [{"tool_use_id": "t1", "macro": "FindObject", "steps": [["look", "You see a cabinet 1."]]}],
    }
    messages = [
        {"role": "user", "content": [{"text": "Your task is to: find the apple."}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "find_object", "input": {"name": "apple"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "No apple"}]}}],
        },
    ]
    text = transcript(SimpleNamespace(output=json.dumps(output), messages=messages), 2000)
    assert text.splitlines() == [
        "[user] Your task is to: find the apple.",
        '[call] find_object {"name": "apple"}',
        "    > look => You see a cabinet 1.",
        "[result] No apple",
    ]
    cut = transcript(SimpleNamespace(output="not json", messages=messages * 30), 400)
    assert len(cut) <= 400 and "lines left out" in cut and cut.startswith("[user] Your task is to")
    assert check_macro(version().load_source(FIND)) is None and issubclass(version().load_source(FIND), Macro)


# ---- the helper variant: at most 8 commands a call, no task line, no task sentence as an argument -----------------
WALK = """import re


class Walk(HelperMacro):
    description = "Go to each receptacle in the room in turn and say what is there."
    parameters = {"name": "the object's name without its number, for example apple"}

    def run(self, act, start, name):
        seen = []
        for place in re.findall(r"an? ([a-z]+ \\d+)", start) * 5:
            seen.append(act(f"go to {place}"))
        return " | ".join(seen) + " | room: " + start
"""


TAKES_TASK = WALK.replace('{"name": "the', '{"task": "the').replace("start, name)", "start, task)")


def helper():
    return programs.macro_variable(helper=True).arguments["version"]


def test_the_helper_variant_is_declared_with_its_settings():
    v = helper()
    assert v.interface is programs.HelperMacro and v.context.endswith(programs.HELPER_CONTEXT)
    assert v.record_settings() == {
        "interface": "alfworld_programs:HelperMacro",
        "max_commands": 8,
        "start_without_task": True,
        "task_argument_refused": True,
    }
    assert version().record_settings()["max_commands"] is None  # the unconstrained macro, as P runs it


def test_a_helper_call_stops_at_8_commands_and_does_not_see_the_task_line():
    cls = helper().load_source(WALK)
    game = Game()
    model = Calls([("walk", {"name": "apple"})])
    agent = Agent(model=model, tools=[act, *macro_tools(cls())], callback_handler=None)
    agent("Your task is to: find the apple.", episode=game)
    (call,) = game.macro_calls
    assert len(call["steps"]) == 8 and game.steps == 8 and call["stopped"] == "command limit"
    text = next(
        b["toolResult"]["content"][0]["text"] for m in agent.messages for b in m["content"] if "toolResult" in b
    )
    assert text.endswith("(walk stopped: a call runs at most 8 commands.)")
    assert "Your task" not in programs._without_task(game.first) and "countertop 1" in programs._without_task(
        game.first
    )


def test_a_helper_call_whose_argument_holds_the_task_sentence_runs_no_command():
    cls = helper().load_source(WALK)
    game = Game()
    model = Calls([("walk", {"name": "find the apple"}), ("walk", {"name": "apple"})])
    agent = Agent(model=model, tools=[act, *macro_tools(cls())], callback_handler=None)
    game.first = game.first.replace("find the apple", "put a clean apple in fridge")
    model.uses[0] = ("walk", {"name": "a clean apple in fridge"})
    agent("Your task is to: put a clean apple in fridge.", episode=game)
    refused, ran = game.macro_calls
    assert refused["refused"] and refused["steps"] == [] and len(ran["steps"]) == 8
    results = [b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b]
    assert results[0]["status"] == "error" and "holds the task sentence" in results[0]["content"][0]["text"]
    assert programs._holds_task(["put a clean apple in fridge"], "Your task is to: put a clean apple in fridge.")
    assert not programs._holds_task(["clean apple", "fridge 1"], "Your task is to: put a clean apple in fridge.")


def test_a_helper_version_cannot_take_the_task_or_change_its_settings():
    v = helper()
    assert v.problem(WALK) is None
    for source, why in (
        (TAKES_TASK, "would carry the task"),
        (
            WALK.replace("the object's name without its number, for example apple", "your task sentence"),
            "would carry the task",
        ),
        (
            WALK.replace('    description = "Go', '    max_commands = 50\n    description = "Go'),
            "max_commands is HelperMacro's",
        ),
        (WALK.replace("class Walk(HelperMacro)", "class Walk(Macro)"), "name 'Macro' is not defined"),
    ):
        problem = v.problem(source)
        assert problem is not None and why in problem, (why, problem)
