"""The ALFWorld workload: a Strands agent playing ALFWorld's text games, ALFWorld's games as cases, and winning the
game as the score. Not a study: see alfworld_harness.py.

ALFWorld (Shridhar et al., 2021) places an agent in a simulated household and gives it a goal such as "put a clean
lettuce in diningtable". Its text version (TextWorld games generated from the ALFRED scenes) answers each command with
a sentence; the game is won when the goal state holds. Each trial plays one game in an environment of its own, in a
process of its own (alfworld_game.py). The agent's one tool, ``act``, runs a command in that game; the task passes the
game to the agent with the call (``agent(message, invocation_state={"episode": episode})``) and the tool reads it from
the call's invocation state, so the tool holds no state and its description can be tuned like any other tool's. The
episode ends when the game is won, after ``MAX_STEPS`` commands, or when the agent answers without calling the tool.

The game's admissible commands (the list of commands the environment accepts in the current state) are not given to
the agent unless ``hints`` is on: ALFWorld's standard protocol does not give them.

Configuration, from the environment:

    ALFWORLD_DATA       ALFWorld's data directory, from ``alfworld-download`` (default ~/.cache/alfworld)
    ALFWORLD_BASE_URL   the OpenAI-compatible endpoint of the model that plays (default ``BASE_URL``: a server on
                        this machine, at vLLM's default port)
    ALFWORLD_MODEL      that model's id, as the endpoint serves it (its ``/v1/models`` lists it; required)

    pip install alfworld "strands-agents[openai]"
    ALFWORLD_DATA=... alfworld-download
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import select
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Any

import openai
from strands import tool
from strands.hooks import BeforeModelCallEvent, HookProvider, HookRegistry
from strands.types.exceptions import MaxTokensReachedException
from strands.types.tools import ToolContext

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import TrialInvalid
from strands_evals.types.evaluation import EvaluationOutput

DATA = Path(os.environ.get("ALFWORLD_DATA", Path.home() / ".cache" / "alfworld")) / "json_2.1.1"
SPLITS = ("train", "valid_seen", "valid_unseen", "valid_train")
# ALFWorld's task types, as its games name them, in its order.
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
MAX_STEPS = 50  # commands per game (ALFWorld's standard text protocol)
MODEL_CALLS_PER_STEP = 3  # a game's model calls are bounded: malformed tool calls spend a model call and no step
BANNER = "-= Welcome to TextWorld, ALFRED! =-"  # the game's first line; the observation starts after it

__all__ = [
    "MAX_STEPS",
    "TASK_TYPES",
    "ALFWorldRun",
    "ALFWorldTask",
    "Episode",
    "InvalidCommands",
    "Steps",
    "Success",
    "act",
    "cases",
    "demonstration",
    "game_commands",
    "game_index",
    "sample",
    "serve",
]

# Strands warns on every model call that reasoning blocks are not sent back through Chat Completions; they are not.
logging.getLogger("strands.models.openai").setLevel(logging.ERROR)


# ---- the endpoint ---------------------------------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8000/v1"  # a server on this machine (vLLM's default port), bound to 127.0.0.1


def serve() -> tuple[str, str]:
    """The model's endpoint and id: ``ALFWORLD_BASE_URL`` (default ``BASE_URL``) and ``ALFWORLD_MODEL``."""
    url, model = os.environ.get("ALFWORLD_BASE_URL") or BASE_URL, os.environ.get("ALFWORLD_MODEL")
    if not model:
        raise RuntimeError(
            f"no model: set ALFWORLD_MODEL to the id the endpoint at {url} serves the model under (its /v1/models "
            "lists it), and ALFWORLD_BASE_URL if the endpoint is not there"
        )
    return url, model


# ---- the game's commands --------------------------------------------------------------------------------------------
def game_commands() -> str:
    """The commands the game accepts, as it lists them for its ``help`` command (ALFWorld's grammar, alfred.twl2),
    verbatim."""
    path = DATA.parent / "logic" / "alfred.twl2"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run alfworld-download with ALFWORLD_DATA={DATA.parent}")
    found = re.search(r'"rhs":\s*("\\nAvailable commands:.*?")\s*[,}\n]', path.read_text())
    if found is None:
        raise ValueError(f"{path} has no help text")
    return json.loads(found.group(1)).strip()


# ---- the games ------------------------------------------------------------------------------------------------------
def _game_dirs(split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"ALFWorld's splits are {SPLITS}, not {split!r}")
    return sorted(p.parent for p in (DATA / split).rglob("traj_data.json"))


@cache
def game_index(split: str) -> tuple[dict[str, str], ...]:
    """The split's text games as ALFWorld's own loader (``AlfredTWEnv``) selects them: those with a TextWorld game file
    that is marked solvable, not the ones about movable or sliced objects. Each has its ``name`` (the trial's id), its
    ``game_file``, and its ``task_type``. Read once and kept next to the data."""
    path = DATA / f"index-{split}.json"
    if path.exists():
        return tuple(json.loads(path.read_text()))
    games = []
    for directory in _game_dirs(split):
        game_file = directory / "game.tw-pddl"
        if "movable" in str(directory) or "Sliced" in str(directory) or not game_file.exists():
            continue
        if not json.loads(game_file.read_text()).get("solvable"):
            continue
        task_type = json.loads((directory / "traj_data.json").read_text())["task_type"]
        games.append({"name": directory.name, "game_file": str(game_file), "task_type": task_type})
    path.write_text(json.dumps(games))
    return tuple(games)


def sample(split: str, per_type: int, *, seed: int = 0, exclude: frozenset[str] = frozenset()) -> list[dict[str, str]]:
    """``per_type`` games of each task type from ``split``, drawn at random with ``seed`` (without those named in
    ``exclude``), in the index's order."""
    chosen = []
    for task_type in TASK_TYPES:
        pool = [g for g in game_index(split) if g["task_type"] == task_type and g["name"] not in exclude]
        if len(pool) < per_type:
            raise ValueError(f"{split} has {len(pool)} {task_type} games not excluded, not {per_type}")
        chosen += random.Random(f"{seed}/{task_type}").sample(pool, per_type)
    return sorted(chosen, key=lambda g: g["name"])


# ---- an episode: one game --------------------------------------------------------------------------------------------
GAME_SCRIPT = Path(__file__).with_name("alfworld_game.py")
REPLY_SECONDS = 300.0  # a game answers a command in milliseconds; a process that is silent this long is stuck


class Episode:
    """One game, played command by command. The game runs in a process of its own (alfworld_game.py: TextWorld's
    environment with ALFWorld's wrappers, the object names as the game's text writes them)."""

    def __init__(self, game_file: str, max_steps: int = MAX_STEPS, hints: bool = False) -> None:
        self.game_file, self.max_steps, self.hints = game_file, max_steps, hints
        self.steps, self.won = 0, False
        self.commands: list[str] = []
        self.observations: list[str] = []
        self.admissible: list[str] = []
        self._lock = threading.Lock()  # one command at a time, if the model asks for several in one turn
        self._process = subprocess.Popen(
            [sys.executable, str(GAME_SCRIPT), game_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            state = self._read()
        except Exception:
            self.close()
            raise
        self.admissible = state["admissible"]
        self.first = self._text(state["feedback"])

    def _read(self) -> dict[str, Any]:
        ready, _, _ = select.select([self._process.stdout], [], [], REPLY_SECONDS)
        line = self._process.stdout.readline() if ready else ""
        if not line:
            code = self._process.poll()
            raise RuntimeError(f"the game process gave no reply ({'running' if code is None else f'exit {code}'})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        return self._read()

    @staticmethod
    def _text(feedback: str) -> str:
        return feedback.replace(BANNER, "").strip()

    @property
    def over(self) -> bool:
        return self.won or self.steps >= self.max_steps

    @property
    def goal(self) -> str:
        """The game's task, as it states it ("Your task is to: ...")."""
        return self.first[self.first.rfind("Your task is to:") :]

    def observe(self, text: str) -> str:
        """``text`` as the agent reads it: with the admissible commands, if the harness gives them."""
        if not self.hints or self.over:
            return text
        return f"{text}\nAdmissible commands: {', '.join(self.admissible)}"

    def step(self, command: str) -> str:
        """Run ``command`` in the game; the observation. A command after the game is over does nothing."""
        with self._lock:
            if self.over:
                return "The game is over." if self.won else f"The limit of {self.max_steps} commands is reached."
            command = " ".join(str(command).split())
            state = self._call({"command": command})
            self.steps += 1
            self.won = state["won"]
            self.admissible = state["admissible"]
            text = self._text(state["feedback"])
            self.commands.append(command)
            self.observations.append(text)
            return self.observe(text)

    def walkthrough(self) -> list[str]:
        """The game's expert plan, in the commands the game accepts."""
        with self._lock:
            return self._call({"walkthrough": True})["walkthrough"]

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            try:
                process.stdin.write(json.dumps({"close": True}) + "\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - closing is cleanup; the game was played
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        process.wait()


def _task_of(game_file: str) -> str:
    episode = Episode(game_file)
    try:
        return episode.goal
    finally:
        episode.close()


def goals(game_files: list[str]) -> dict[str, str]:
    """Each game's task as the game states it. The game has to be started to say it, so the answers are kept next to the
    data (goals.json) and the missing ones are started 32 at a time."""
    path = DATA / "goals.json"
    known: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}
    missing = [g for g in game_files if g not in known]
    if missing:
        with ThreadPoolExecutor(max_workers=32) as pool:
            known.update(zip(missing, pool.map(_task_of, missing), strict=True))
        path.write_text(json.dumps(known))
    return {g: known[g] for g in game_files}


def cases(games: list[dict[str, str]], split: str = "") -> list[Case]:
    """The games as cases: the input is the game's task, the metadata its game file and task type."""
    tasks = goals([g["game_file"] for g in games])
    return [
        Case(
            name=f"alfworld-{g['name']}",
            input=tasks[g["game_file"]],
            metadata={"game_file": g["game_file"], "task_type": g["task_type"], "split": split},
        )
        for g in games
    ]


def demonstration(game_file: str) -> str:
    """An expert's play of the game, as the transcript the agent would see: the game's own walkthrough, run in the
    game."""
    episode = Episode(game_file)
    try:
        episode.max_steps = len(episode.walkthrough()) + 1
        lines = [episode.first]
        for command in episode.walkthrough():
            lines += [f"> {command}", episode.step(command)]
        return "\n".join(lines)
    finally:
        episode.close()


# ---- the tool -------------------------------------------------------------------------------------------------------
@tool(context=True)
def act(command: str, tool_context: ToolContext) -> str:
    """Run one command in the household game and return what happens.

    Args:
        command: The command to run, in the game's syntax, for example "go to countertop 1" or "take apple 1 from
            countertop 1"
    """
    episode: Episode | None = (tool_context.invocation_state or {}).get("episode")
    if episode is None:
        return "No game is running."
    observation = episode.step(command)
    if episode.over:  # the game is won or the commands are used up: the agent need not be asked again
        stop(tool_context)
    return observation


def stop(tool_context: ToolContext) -> None:
    """End the agent's loop once the tools of this turn have run (``Agent.cancel``; earlier Strands: the loop's
    flag)."""
    try:
        tool_context.agent.cancel(after_current_tools=True)
    except TypeError:
        tool_context.invocation_state.setdefault("request_state", {})["stop_event_loop"] = True


class ModelCallLimit(Exception):
    """The agent made as many model calls as a game allows (``MODEL_CALLS_PER_STEP`` per command)."""


class _CallLimit(HookProvider):
    def __init__(self, limit: int) -> None:
        self.limit, self.calls = limit, 0

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model_call)

    def _before_model_call(self, event: BeforeModelCallEvent) -> None:
        self.calls += 1
        if self.calls > self.limit:
            raise ModelCallLimit(f"more than {self.limit} model calls")


# ---- the study's task -----------------------------------------------------------------------------------------------
class ALFWorldRun:
    """A finished game: how it ended and the commands played. (A plain object: Strands Evals keeps a task's output as
    it is.) Its text is the run as JSON."""

    def __init__(self, case: Case, episode: Episode, end: str, model_calls: int) -> None:
        self.game, self.task_type = case.metadata["game_file"], case.metadata["task_type"]
        self.goal, self.won, self.steps = episode.goal, episode.won, episode.steps
        self.commands, self.observations, self.end, self.model_calls = (
            list(episode.commands),
            list(episode.observations),
            end,
            model_calls,
        )
        # the calls of a macro tool (alfworld_programs.py), each with the commands it ran: none without one
        self.macro_calls = list(getattr(episode, "macro_calls", None) or [])

    @property
    def invalid(self) -> int:
        """Commands the game answered with "Nothing happens."."""
        return sum(o.strip().lower().startswith("nothing happens") for o in self.observations)

    @property
    def repeated(self) -> int:
        """Commands that repeat the one before."""
        return sum(a == b for a, b in zip(self.commands, self.commands[1:], strict=False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": Path(self.game).parent.name,
            "task_type": self.task_type,
            "goal": self.goal,
            "won": self.won,
            "end": self.end,
            "steps": self.steps,
            "model_calls": self.model_calls,
            "invalid": self.invalid,
            "repeated": self.repeated,
            "commands": self.commands,
            "final": self.observations[-1] if self.observations else "",
            **({"macro_calls": self.macro_calls} if self.macro_calls else {}),
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict())


# How an episode ended: won; "step_limit" (MAX_STEPS commands, not won); "stopped" (the agent answered without a tool
# call); "model_call_limit" (too many model calls for the commands run); "max_tokens" (a reply reached the model's
# output limit, as a thinking model's does when its reasoning loops). Each is a game played to its end, won or lost.
ENDS = ("won", "step_limit", "stopped", "model_call_limit", "max_tokens")


def _transport_error(exc: BaseException) -> bool:
    """Whether the model service failed to answer (connection, timeout, server error), not the agent."""
    for e in (exc, getattr(exc, "original_exception", None), exc.__cause__):
        if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
            return True
    return False


class ALFWorldTask:
    """The study's task: one ALFWorld game, played by the agent in an environment of its own."""

    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        self.max_steps = max_steps

    def __call__(self, agent: Any, case: Case) -> dict[str, Any]:
        hints = bool(agent.state.get("hints"))  # a harness choice the factory records (alfworld_harness.py)
        limit = _CallLimit(self.max_steps * MODEL_CALLS_PER_STEP)
        agent.hooks.add_hook(limit)
        try:
            episode = Episode(case.metadata["game_file"], self.max_steps, hints)
        except Exception as exc:
            raise TrialInvalid(f"ALFWorld's game did not load: {type(exc).__name__}: {exc}") from exc
        try:
            end = None
            try:
                agent(episode.observe(episode.first), invocation_state={"episode": episode})
            except Exception as exc:  # noqa: BLE001 - classified below
                cause = getattr(exc, "original_exception", None) or exc
                if isinstance(cause, ModelCallLimit):
                    end = "model_call_limit"
                elif isinstance(cause, MaxTokensReachedException):
                    end = "max_tokens"
                elif _transport_error(exc):
                    raise TrialInvalid(f"the model service failed: {type(cause).__name__}: {cause}") from exc
                else:
                    raise
            if episode.won:
                end = "won"
            elif episode.over:
                end = "step_limit"
            elif end is None:
                end = "stopped"
            return {"output": ALFWorldRun(case, episode, end, limit.calls)}
        finally:
            episode.close()


# ---- scores ---------------------------------------------------------------------------------------------------------
class Success(Evaluator):
    """1 when the game was won, else 0."""

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        run: ALFWorldRun = case.actual_output
        return [
            EvaluationOutput(score=1.0 if run.won else 0.0, test_pass=run.won, reason=f"{run.end} after {run.steps}")
        ]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)


class Steps(Evaluator):
    """The commands the agent ran. Not a score."""

    score_range = (0, None)

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        n = case.actual_output.steps
        return [EvaluationOutput(score=float(n), test_pass=True, reason=f"{n} commands")]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)


class InvalidCommands(Evaluator):
    """The commands the game did not accept ("Nothing happens."). Not a score."""

    score_range = (0, None)

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        n = case.actual_output.invalid
        return [EvaluationOutput(score=float(n), test_pass=True, reason=f"{n} invalid commands")]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)
