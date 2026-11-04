"""ALFWorld macros: a program-valued variable whose versions are tools that run several game commands in one call.

A version is a subclass of ``Macro``, written by ``ProgramProposer`` (Sonnet 5.5 on Bedrock, by default). The agent
gets it as one more Strands tool beside its ``act`` tool, named from the class (``FindObject`` -> ``find_object``),
described by its ``description`` and taking its ``parameters`` as string arguments. The tool runs the macro's ``run``
with ``act``, which sends one command to the episode's game and returns the observation, so every command a macro runs
counts toward the game's command limit and is recorded with the game's own commands; ``run`` also receives the game's
first observation (``start``), the only one that lists every receptacle in the room. The seed is the empty program:
the baseline agent has no macro, and the first version is written from the baseline's trials.

In an ALFWorld harness (alfworld_harness.py):

    from alfworld_programs import macro_tools, macro_variable

    @tunable(..., macro=macro_variable())
    def create_agent(..., macro):
        return Agent(model=..., tools=[act, *macro_tools(macro)], ...)

The tool reads the game from the call's invocation state (``agent(message, invocation_state={"episode": episode})``),
as ``act`` does: an ``episode`` with ``step(command) -> observation``, ``over``, ``won`` and ``first`` (its first
observation). Each macro call is recorded on the episode (``episode.macro_calls``: the call's tool-use id, the macro,
its arguments, the commands it ran with their observations, what it returned, and its error if it failed); a task that
writes them into its output (``{"macro_calls": [...]}`` in the output's JSON) lets ``transcript`` show the proposer
what each call did.

Not importing ALFWorld: the module runs against any object with that episode's interface.
"""

from __future__ import annotations

import inspect
import json
import keyword
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from strands.tools.tools import PythonAgentTool

from strands_evals.optimization.programs import EMPTY, Program, ProgramTimeout, messages_transcript, run_bounded

MAX_STEPS = 50  # the game's command limit (ALFWorld's standard protocol); a macro's commands count toward it
MAX_PARAMETERS = 4
MAX_DESCRIPTION = 1500  # characters
MAX_RESULT = 3000  # characters of what ``run`` returns that the agent reads
LINES, SECONDS = 2_000_000, 120.0  # each macro call's budget (``run_bounded``)
# What a version may import: no files, processes or network.
ALLOWED_IMPORTS = (
    "__future__",
    "re",
    "collections",
    "itertools",
    "dataclasses",
    "typing",
    "math",
    "string",
    "functools",
)

__all__ = [
    "ALLOWED_IMPORTS",
    "CONTEXT",
    "CommandLimit",
    "GameOver",
    "HELPER_CONTEXT",
    "HelperMacro",
    "Macro",
    "check_macro",
    "macro_tools",
    "macro_variable",
    "tool_name",
    "transcript",
]


class Macro(ABC):
    """A tool the agent can call to run several ALFWorld commands at once.

    Subclass it with:

    - ``description``: what the tool does and when to call it, as the agent reads it (the tool's description).
    - ``parameters``: each argument's name and what it is, as the agent reads them. Every argument is a string, and
      every one is required.
    - ``run(act, start, **arguments)``: plays commands with ``act(command)``, which sends one command to the game and
      returns the game's observation (the text the game answers with), and returns what the agent reads as the tool's
      result. ``start`` is the game's first observation: the receptacles in the room (the only observation that lists
      them all) and the task.

    ``act`` raises ``GameOver`` once the game is over (won, or the command limit reached): let it propagate (it is not
    an ``Exception``), and the tool tells the agent how the game ended. A fresh instance plays each game, so an instance
    may keep what it learned between calls within one game.
    """

    description: ClassVar[str] = ""
    parameters: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def run(self, act: Callable[[str], str], start: str, **arguments: str) -> str:
        """Run commands with ``act`` (``start``: the game's first observation) and return what the agent should
        read."""


class GameOver(BaseException):  # noqa: N818 - a BaseException, so a macro's `except Exception` cannot keep it
    """The game is over (won, or the command limit reached): a macro's next command is not run."""


class CommandLimit(BaseException):  # noqa: N818 - a BaseException, so a macro's `except Exception` cannot keep it
    """A helper macro's call has run as many commands as one call may: its next command is not run."""


class HelperMacro(Macro):
    """A Macro that helps the agent with one part of a game rather than playing the game for it:

    - each call runs at most ``max_commands`` commands (8): ``act`` raises ``CommandLimit`` on the next one, and the
      tool tells the agent the call stopped there;
    - ``start`` is the game's first observation without its task line ("Your task is to: ..."): the room's
      receptacles;
    - an argument that holds the task sentence is refused (the call runs no command), and so is a parameter meant for
      it (named task, goal, ..., or described as the task sentence): the agent decides each step and passes object
      and receptacle names.

    A version does not change these settings.
    """

    max_commands: ClassVar[int] = 8
    hides_task: ClassVar[bool] = True
    refuses_task: ClassVar[bool] = True


# What the record's space row keeps of each variant (Program(settings=...)).
SETTINGS = {"max_commands": None, "start_without_task": False, "task_argument_refused": False}
HELPER_SETTINGS = {"max_commands": 8, "start_without_task": True, "task_argument_refused": True}
TASK_NAMES = {"task", "goal", "instruction", "instructions", "objective", "sentence", "mission", "request", "query"}
TASK_WORDS = re.compile(
    r"\btask sentence|\byour task\b|\bfirst message\b|\bgoal sentence|\bwhole task\b|\bthe goal\b", re.I
)


CONTEXT = """\
The agent plays ALFWorld's text games (TextWorld versions of ALFRED's household scenes). Each game places it in a room \
with numbered receptacles ("cabinet 3", "countertop 1", "fridge 1") and objects ("apple 2", "mug 1"), and gives it a \
goal. The first observation lists the receptacles and states the goal, for example:

    You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a fridge 1, \
a microwave 1, a sinkbasin 1, and a toaster 1.

    Your task is to: put a clean lettuce in diningtable.

The game's commands (its own help text):

    look:                             look around your current location
    inventory:                        check your current inventory
    go to (receptacle):               move to a receptacle
    open (receptacle):                open a receptacle
    close (receptacle):               close a receptacle
    take (object) from (receptacle):  take an object from a receptacle
    move (object) to (receptacle):    place an object in or on a receptacle
    examine (something):              examine a receptacle or an object
    use (object):                     use an object
    heat (object) with (receptacle):  heat an object using a receptacle
    clean (object) with (receptacle): clean an object using a receptacle
    cool (object) with (receptacle):  cool an object using a receptacle
    slice (object) with (object):     slice an object using a sharp object

Names are written as the game writes them, with their number ("take apple 2 from countertop 1"). The agent holds at \
most one object; it must be at a receptacle (go to it) to take from it, put in it, open it, or use it. Only the first \
observation lists the room's receptacles: `look` describes only where the agent is ("You are in the middle of a room. \
Looking quickly around you, you see nothing." at the start, then "You are facing the desk 1. Next to it, you see \
nothing."), so a macro reads the receptacles from `start`, the first observation, which `run` receives. Objects are \
seen only by going to a receptacle (and opening it when it is closed). The game answers each command with one \
observation, for example:

    You arrive at countertop 1. On the countertop 1, you see a apple 2, and a knife 1.
    You arrive at cabinet 3. The cabinet 3 is closed.
    You open the cabinet 3. The cabinet 3 is open. In it, you see a mug 1.
    You pick up the apple 2 from the countertop 1.
    You move the apple 2 to the diningtable 1.
    You heat the apple 2 using the microwave 1.   (also: You cool ... using the fridge 1; You clean ... using the \
sinkbasin 1)
    You are carrying: a apple 2.
    Nothing happens.   (the command was not valid here: a wrong name, a closed receptacle, nothing in hand, ...)

The six task types, as their goals read:

- pick_and_place_simple: "put a/some X in/on Y": find an X, take it, go to Y, move it to Y.
- look_at_obj_in_light: "look at X under the desklamp" / "examine the X with the desklamp": take the X, go to the \
desklamp (or floorlamp), use it.
- pick_clean_then_place_in_recep: "put a clean X in Y" / "clean some X and put it in Y": take the X, clean it with a \
sinkbasin, then put it in Y.
- pick_heat_then_place_in_recep: "put a hot X in Y" / "heat some X and put it in Y": heat it with a microwave.
- pick_cool_then_place_in_recep: "put a cool X in Y" / "cool some X and put it in Y": cool it with a fridge.
- pick_two_obj_and_place: "put two X in Y" / "find two X and put them in Y": one at a time, both to Y.

The game is won as soon as the goal holds. It allows 50 commands, those a macro runs included; it ends at the 50th. \
When the harness gives hints, each observation ends with "Admissible commands: ..." (the commands valid in that state).

The agent is a small model (Qwen3.5-2B, served with vLLM) that plays with its `act` tool, one command per call, and, \
when the study gives it a macro, with the macro's tool too. It calls the macro only if the tool's name and \
description make clear when it helps, and passes the arguments as it reads the observations. Each call is one model \
turn, so a macro that does several commands' work saves the agent turns and context. What `run` returns is all the \
agent sees of the call (it does not see the macro's individual observations), so return the observations it needs \
to go on: where things are, what it holds, what was not found.

Each macro call may run at most 2,000,000 lines of its own code and 120 seconds, and its commands share the game's 50.
"""

HELPER_CONTEXT = """
This study's macro is a HelperMacro: each call runs at most 8 commands (`act` raises CommandLimit on the 9th, and the \
agent is told the call stopped), `start` holds the room without the task line, and an argument holding the task \
sentence is refused, as is a parameter meant for it. The macro cannot play the game for the agent: write a tool for \
one step the agent chooses, taking object and receptacle names, for example finding an object and taking it, carrying \
the held object to a receptacle and putting it there, or heating, cooling or cleaning the held object.
"""

ABOUT = (
    "a tool the agent calls to run several game commands at once (search receptacles for an object, carry an object "
    "somewhere, heat/cool/clean it on the way), so that a small model wins more games within the command limit"
)


# ---- the declaration -----------------------------------------------------------------------------------------------
def macro_variable(helper: bool = False, **overrides: Any) -> Any:
    """The ``macro`` variable: ``Program(Macro, ...)`` with the empty program as its seed, the imports in
    ``ALLOWED_IMPORTS`` only, the ALFWorld ``CONTEXT``, ``check_macro`` and ``transcript``; with ``helper``,
    ``Program(HelperMacro, ...)``, whose context also says its limits (``HELPER_CONTEXT``). The variant's settings are
    kept in the record. ``overrides`` go to ``Program`` (``seeds=[...]`` to start from written versions; a
    ``context`` replaces ``CONTEXT``, and a helper's still gets ``HELPER_CONTEXT``)."""
    context = overrides.pop("context", CONTEXT)
    options: dict[str, Any] = {
        "seeds": [EMPTY],
        "about": ABOUT,
        "imports": ALLOWED_IMPORTS,
        "standard": False,
        "context": context + HELPER_CONTEXT if helper else context,
        "check": check_macro,
        "transcript": transcript,
        "settings": HELPER_SETTINGS if helper else SETTINGS,
        "label": "macro",
        **overrides,
    }
    return Program(HelperMacro if helper else Macro, **options)


def tool_name(cls: type) -> str:
    """The tool's name: the class's name in snake case (``FindObject`` -> ``find_object``)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", cls.__name__).lower()


START = (
    "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, and a countertop 1.\n\n"
    "Your task is to: put a apple in cabinet."
)


class _DryRun:
    """A game for ``check_macro``'s dry run (its first observation: ``START``): ``look`` and ``inventory`` answer,
    every other command does nothing; after ``MAX_STEPS`` commands the game is over, and after ``limit`` the call
    stops (a helper macro's limit)."""

    def __init__(self, limit: int | None = None) -> None:
        self.steps, self.limit = 0, limit

    def __call__(self, command: str) -> str:
        if self.limit is not None and self.steps >= self.limit:
            raise CommandLimit(f"a call runs at most {self.limit} commands")
        if self.steps >= MAX_STEPS:
            raise GameOver(f"The limit of {MAX_STEPS} commands is reached.")
        self.steps += 1
        command = " ".join(str(command).split())
        if command == "look":
            return "You are in the middle of a room. Looking quickly around you, you see nothing."
        if command == "inventory":
            return "You are not carrying anything."
        return "Nothing happens."


def check_macro(cls: type) -> str | None:
    """Why a loaded macro class cannot be a tool, or None: its description, its parameters, its tool name, and a dry run
    (each argument "apple 1", in a two-receptacle room where every command but ``look`` and ``inventory`` does nothing)
    that must return a string or end the game, within a smaller budget."""
    name = tool_name(cls)
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) or name == "act":
        return f"the tool's name, {name!r} (from the class name), must be 1-64 letters, digits or _, and not 'act'"
    description = getattr(cls, "description", None)
    if not isinstance(description, str) or not description.strip():
        return "description must be a non-empty string: the agent reads it to decide when to call the tool"
    if len(description) > MAX_DESCRIPTION:
        return f"description is {len(description)} characters; at most {MAX_DESCRIPTION}"
    parameters = getattr(cls, "parameters", None)
    if not isinstance(parameters, dict) or len(parameters) > MAX_PARAMETERS:
        return f"parameters must be a dict of at most {MAX_PARAMETERS} names, each to what the argument is"
    for key, about in parameters.items():
        if not isinstance(key, str) or not key.isidentifier() or keyword.iskeyword(key) or key in ("act", "start"):
            return f"parameter {key!r} must be a Python identifier (not a keyword, not 'act' or 'start')"
        if not isinstance(about, str) or not about.strip():
            return f"parameter {key!r} needs a description (a non-empty string)"
    interface = next((c for c in cls.__mro__[1:] if c in (HelperMacro, Macro)), Macro)
    for setting in ("max_commands", "hides_task", "refuses_task"):
        if getattr(cls, setting, None) != getattr(interface, setting, None):
            return f"{setting} is {interface.__name__}'s setting; a version does not change it"
    if getattr(cls, "refuses_task", False):
        for key, about in parameters.items():
            if key.lower() in TASK_NAMES or TASK_WORDS.search(about):
                return (
                    f"parameter {key!r} would carry the task: a HelperMacro takes object and receptacle names, and the "
                    "agent decides each step"
                )
    macro = cls()
    arguments = dict.fromkeys(parameters, "apple 1")
    start = _without_task(START) if getattr(cls, "hides_task", False) else START
    limit = getattr(cls, "max_commands", None)
    try:
        inspect.signature(macro.run).bind(_DryRun(), start, **arguments)
    except TypeError as exc:
        return f"run(act, start, {', '.join(f'{k}=...' for k in parameters)}) does not match its signature: {exc}"
    try:
        out = run_bounded(lambda: macro.run(_DryRun(limit), start, **arguments), lines=200_000, seconds=10)
    except (GameOver, CommandLimit):
        return None
    except ProgramTimeout as exc:
        return f"a dry run did not finish: {exc} (every command but look and inventory answered 'Nothing happens.')"
    except Exception as exc:  # noqa: BLE001 - any error in the dry run is the reason
        return f"a dry run raised {type(exc).__name__}: {exc}"
    if not isinstance(out, str):
        return f"run must return a string (the tool's result); it returned {type(out).__name__}"
    return None


# ---- the tool ------------------------------------------------------------------------------------------------------
def macro_tools(macro: Macro | None) -> list[PythonAgentTool]:
    """The agent's tools for ``macro``: none for the empty program, else one tool that runs it."""
    if macro is None:
        return []
    cls = type(macro)
    name = tool_name(cls)
    limit = getattr(cls, "max_commands", None)
    hides, refuses = bool(getattr(cls, "hides_task", False)), bool(getattr(cls, "refuses_task", False))
    spec = {
        "name": name,
        "description": cls.description.strip(),
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in cls.parameters.items()},
                "required": list(cls.parameters),
            }
        },
    }

    def call(tool_use: dict[str, Any], **state: Any) -> dict[str, Any]:
        use_id = tool_use.get("toolUseId", "")
        episode = state.get("episode")

        def result(text: str, error: bool = False) -> dict[str, Any]:
            return {"toolUseId": use_id, "status": "error" if error else "success", "content": [{"text": text}]}

        if episode is None:
            return result("No game is running.", error=True)
        raw = tool_use.get("input") or {}
        arguments = {k: " ".join(str(raw.get(k, "")).split()) for k in cls.parameters}
        steps: list[list[str]] = []
        record: dict[str, Any] = {"tool_use_id": use_id, "macro": cls.__name__, "arguments": arguments, "steps": steps}
        calls = getattr(episode, "macro_calls", None)
        if calls is None:
            calls = []
            episode.macro_calls = calls
        calls.append(record)
        first = str(getattr(episode, "first", "") or "")
        if refuses and _holds_task(arguments.values(), str(getattr(episode, "goal", "") or first)):
            record["refused"] = "an argument holds the task sentence"
            return result(
                f"{name} was not run: an argument holds the task sentence. Pass object and receptacle names; you "
                "decide each step.",
                error=True,
            )

        def act(command: str) -> str:
            if episode.over:
                raise GameOver("won" if episode.won else "limit")
            if limit is not None and len(steps) >= limit:
                raise CommandLimit(f"a call runs at most {limit} commands")
            command = " ".join(str(command).split())
            observation = episode.step(command)
            steps.append([command, observation])
            if episode.over:  # as act does: the agent need not be asked again
                state.setdefault("request_state", {})["stop_event_loop"] = True
            return observation

        start = _without_task(first) if hides else first
        try:
            out = run_bounded(lambda: macro.run(act, start, **arguments), lines=LINES, seconds=SECONDS)
            text = out if isinstance(out, str) else json.dumps(out, default=str)
            text = text.strip() or (steps[-1][1] if steps else "(the macro returned nothing)")
            if episode.over:
                text += f"\n{_ended(episode)}"
        except GameOver:
            text = (steps[-1][1] + "\n" if steps else "") + _ended(episode)
        except CommandLimit:
            record["stopped"] = "command limit"
            text = (steps[-1][1] + "\n" if steps else "") + f"({name} stopped: a call runs at most {limit} commands.)"
        except (Exception, ProgramTimeout) as exc:  # noqa: BLE001 - a failed macro is the tool's error; the game goes on
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            last = f" Its last observation: {steps[-1][1]}" if steps else ""
            note = f"\n{_ended(episode)}" if episode.over else ""
            return result(f"{name} failed after {len(steps)} commands: {record['error']}.{last}{note}", error=True)
        record["returned"] = text[:MAX_RESULT]
        return result(text[:MAX_RESULT])

    return [PythonAgentTool(name, spec, call)]


def _without_task(observation: str) -> str:
    """The first observation without its task line: the room."""
    return observation.split("Your task is to:")[0].strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _holds_task(arguments: Any, goal: str) -> bool:
    """Whether an argument holds the task sentence: all of it, or four or more of its words in a row."""
    task = _words(goal.split("Your task is to:")[-1])
    if not task:
        return False
    for argument in arguments:
        words = _words(argument)
        for size in range(min(len(words), len(task)), 3, -1):
            spans = {tuple(task[i : i + size]) for i in range(len(task) - size + 1)}
            if any(tuple(words[i : i + size]) in spans for i in range(len(words) - size + 1)):
                return True
    return False


def _ended(episode: Any) -> str:
    return (
        "The game is won."
        if episode.won
        else f"The limit of {getattr(episode, 'max_steps', MAX_STEPS)} commands is reached."
    )


# ---- what the proposer reads of a trial ------------------------------------------------------------------------------
def transcript(trial: Any, limit: int) -> str:
    """A trial as the proposer reads it: its messages (``messages_transcript``), each macro call followed by the
    commands it ran and their observations (from the output's ``macro_calls``, when the task writes them)."""
    inner: dict[str, Sequence[str]] = {}
    try:
        output = json.loads(trial.output) if isinstance(trial.output, str) else {}
    except ValueError:
        output = {}
    for call in (output.get("macro_calls") or []) if isinstance(output, dict) else []:
        lines = [f"> {command} => {observation}" for command, observation in call.get("steps") or []]
        if call.get("error"):
            lines.append(f"(failed: {call['error']})")
        inner[call.get("tool_use_id", "")] = lines or ["(no commands)"]
    return messages_transcript(trial.messages, limit, inner)
