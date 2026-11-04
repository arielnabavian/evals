"""The ALFWorld workload (examples/optimization): the tool, the episode's end, the score and the usage, with a scripted
game process and a scripted model (no ALFWorld install, no network)."""

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest
from strands import Agent
from strands.models.model import Model
from strands.tools.executors import SequentialToolExecutor

from strands_evals import Case, Experiment
from strands_evals.optimization import tunable
from strands_evals.optimization.trials import TrialStatus, run_trials

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "optimization"))

pytest.importorskip("openai")  # the workload's model client (strands-agents[openai]) is not a dependency

import alfworld_workload as workload  # noqa: E402

# A game with the protocol of alfworld_game.py: three commands win it.
FAKE_GAME = textwrap.dedent(
    """
    import json, sys

    ROOM = ("-= Welcome to TextWorld, ALFRED! =-\\n\\nYou are in the middle of a room. Looking quickly around you, "
            "you see a shelf 1, and a table 1.\\n\\nYour task is to: put a mug in shelf.")
    PLAN = ["go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1"]
    REPLIES = ["You arrive at table 1. On the table 1, you see a mug 1.", "You pick up the mug 1 from the table 1.",
               "You move the mug 1 to the shelf 1."]
    COMMANDS = ["go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1", "look"]

    def reply(feedback, won=False):
        print(json.dumps({"feedback": feedback, "won": won, "admissible": COMMANDS}), flush=True)

    reply(ROOM)
    done = 0
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            break
        if "walkthrough" in request:
            print(json.dumps({"walkthrough": PLAN}), flush=True)
            continue
        command = request["command"]
        if done < len(PLAN) and command == PLAN[done]:
            done += 1
            reply(REPLIES[done - 1], won=done == len(PLAN))
        else:
            reply("Nothing happens.")
    """
)


class Plays(Model):
    """Calls ``act`` with each of ``commands`` in turn, then replies in text. Every call spends 100 input tokens and 10
    output tokens. ``tool`` names the tool it calls; with ``cut_at``, that call stops at the output limit instead."""

    def __init__(self, commands, tool="act", cut_at=None):
        self.commands, self.tool, self.calls, self.cut_at = list(commands), tool, 0, cut_at
        self.config = {"model_id": "plays"}

    def update_config(self, **kw):
        self.config.update(kw)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        index, self.calls = self.calls, self.calls + 1
        yield {"messageStart": {"role": "assistant"}}
        if index == self.cut_at:
            yield {"contentBlockDelta": {"delta": {"text": "Let me think about this again and again"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "max_tokens"}}
        elif index < len(self.commands):
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{index}", "name": self.tool}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"command": "%s"}' % self.commands[index]}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "I am done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


@pytest.fixture
def game(tmp_path, monkeypatch):
    script = tmp_path / "fake_game.py"
    script.write_text(FAKE_GAME)
    monkeypatch.setattr(workload, "GAME_SCRIPT", script)
    return str(tmp_path / "game.tw-pddl")


def play(game_file, commands, *, max_steps=50, hints=False, tool="act", cut_at=None):
    """One trial of ``commands`` played in the scripted game; the trial and its output."""

    @tunable()
    def factory():
        agent = Agent(
            model=Plays(commands, tool, cut_at),
            tools=[workload.act],
            tool_executor=SequentialToolExecutor(),
            callback_handler=None,
        )
        agent.state.set("hints", hints)
        return agent

    case = Case(
        name="g", input="Your task is to: put a mug in shelf.", metadata={"game_file": game_file, "task_type": "x"}
    )
    experiment = Experiment(
        cases=[case],
        evaluators=[
            workload.Success(name="success"),
            workload.Steps(name="steps"),
            workload.InvalidCommands(name="invalid"),
        ],
    )
    trials = asyncio.run(
        run_trials(
            factory,
            factory.default_values,
            experiment,
            repetitions=1,
            max_concurrency=2,
            task=workload.ALFWorldTask(max_steps),
            retries=0,
        )
    )
    return trials[0]


def test_a_won_game_scores_one_and_the_loop_stops_at_the_win(game):
    trial = play(game, ["go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1", "look", "look"])
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["success"] == 1.0 and trial.metrics["steps"] == 3.0 and trial.metrics["invalid"] == 0.0
    assert trial.metrics["model_calls"] == 3.0  # the model is not asked again once the game is won
    assert (
        trial.metrics["input_tokens"] == 300 and trial.metrics["output_tokens"] == 30 and trial.metrics["tokens"] == 330
    )
    assert '"end": "won"' in trial.output and "go to table 1" in trial.output


def test_the_step_cap_ends_the_game_unwon(game):
    trial = play(game, ["look"] * 10, max_steps=4)
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["success"] == 0.0 and trial.metrics["steps"] == 4.0
    assert trial.metrics["model_calls"] == 4.0
    assert '"end": "step_limit"' in trial.output


def test_invalid_commands_are_counted_and_a_game_can_still_be_won(game):
    trial = play(game, ["dance", "go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1"])
    assert trial.metrics["success"] == 1.0 and trial.metrics["steps"] == 4.0 and trial.metrics["invalid"] == 1.0


def test_an_answer_without_a_command_ends_the_game_as_stopped(game):
    trial = play(game, [])
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["success"] == 0.0 and trial.metrics["steps"] == 0.0
    assert '"end": "stopped"' in trial.output


def test_calls_that_run_no_command_end_at_the_model_call_limit(game):
    trial = play(game, ["look"] * 100, max_steps=2, tool="no_such_tool")
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["steps"] == 0.0 and trial.metrics["model_calls"] >= 2 * workload.MODEL_CALLS_PER_STEP
    assert '"end": "model_call_limit"' in trial.output


def test_hints_append_the_admissible_commands_to_each_observation(game):
    trial = play(game, ["go to table 1"], hints=True)
    texts = [b["text"] for m in trial.messages for b in m["content"] if "text" in b]
    results = [b["toolResult"]["content"][0]["text"] for m in trial.messages for b in m["content"] if "toolResult" in b]
    assert "Admissible commands: go to table 1" in results[0]
    assert "Admissible commands:" in texts[0]  # the first message too
    plain = play(game, ["go to table 1"])
    assert all(
        "Admissible" not in b["toolResult"]["content"][0]["text"]
        for m in plain.messages
        for b in m["content"]
        if "toolResult" in b
    )


def test_the_episode_reads_the_task_and_the_walkthrough(game):
    episode = workload.Episode(game)
    try:
        assert episode.goal == "Your task is to: put a mug in shelf."
        assert "Welcome to TextWorld" not in episode.first
        assert episode.walkthrough() == ["go to table 1", "take mug 1 from table 1", "move mug 1 to shelf 1"]
        assert "game is over" not in episode.step("look")
        episode.max_steps = episode.steps  # no commands left
        assert "limit of" in episode.step("look")
    finally:
        episode.close()
    assert workload.demonstration(game).splitlines()[-2:] == [
        "> move mug 1 to shelf 1",
        "You move the mug 1 to the shelf 1.",
    ]


def test_a_missing_game_process_is_an_error_not_a_hang(tmp_path, monkeypatch):
    broken = tmp_path / "broken.py"
    broken.write_text("import sys; sys.exit(3)")
    monkeypatch.setattr(workload, "GAME_SCRIPT", broken)
    with pytest.raises(RuntimeError, match="exit 3"):
        workload.Episode("x")


def test_a_reply_cut_at_the_output_limit_ends_the_game_lost_not_as_the_agents_failure(game):
    trial = play(game, ["go to table 1", "look", "look"], cut_at=2)
    assert trial.status == TrialStatus.COMPLETED
    assert trial.metrics["success"] == 0.0 and trial.metrics["steps"] == 2.0 and trial.metrics["model_calls"] == 3.0
    assert '"end": "max_tokens"' in trial.output
