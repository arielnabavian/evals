"""Tune a small open-weight model's harness on ALFWorld's text games: its system prompt, the description of its one
tool, whether it thinks, its sampling temperature, whether the game's admissible commands are shown, and its
conversation manager. The goals are the share of games won and the weighted tokens spent to win them.

    python examples/optimization/alfworld_harness.py --baseline --out baseline.json
    BYPASS_TOOL_CONSENT=true python examples/optimization/alfworld_harness.py --space mini --goal best --out studies

The agent is a Strands ``Agent`` on an OpenAI-compatible endpoint (vLLM serving Qwen3.5-2B; ``ALFWORLD_BASE_URL``, by
default a server on this machine, and ``ALFWORLD_MODEL``, alfworld_workload.py), with one tool, ``act``, that runs a
command in the game. No examples are given (``--demos`` puts one expert play of each task type in the system prompt),
and the game's admissible commands are not shown unless the ``hints`` setting is on.

Search spaces:

    text, L1    the system prompt and ``act``'s description (text: reflection)
    mini        text, thinking and the sampling temperature (the numbers: TPE)
    mini-hints  mini, and whether the admissible commands are appended to each observation
    L2          mini, and the conversation manager (a sliding window or summaries) with its settings
    L2h, full   L2, and whether the admissible commands are appended to each observation
    macro       a macro tool beside ``act`` only (alfworld_programs.py: a program the proposer agent writes), from none
    macro-helper  the same with the helper variant: at most 8 commands a call, no task line, no task sentence argument

``--macro`` adds the macro to any space: one more tool, a class ``ProgramProposer`` (Sonnet 5.5 on Bedrock) writes from
the trials, starting from none (the baseline has no macro). ``--skills`` adds a skill library to any space: Strands'
``AgentSkills`` plugin over a library ``SkillCuration`` writes from the trials, starting empty (the baseline has no
plugin, so no ``skills`` tool and nothing added to the system prompt).

Every sliding window keeps the first message (the room and the task): Qwen3.5's chat template refuses a conversation
without a user message.

A study searches on games drawn from ALFWorld's train split (``--search-per-type`` of each of the six task types),
selects on other train games (``--validation-per-type`` of each type), and tests the selection and the baseline on all
134 games of valid_unseen, which nothing before saw: each case ``--repetitions`` times, for ``--iterations`` steps.

Goals, on success (1 for a game won, else 0) and weighted tokens (the agent's tokens per game, each kind weighted by its
price relative to an uncached input token, as tau2_harness.py does):

    best      the highest success
    budget    the highest success within the baseline's weighted tokens
    joint     the highest success and the fewest weighted tokens together (the Pareto set)
    cheapest  the fewest weighted tokens on the front, with success at least --floor-gain above the baseline's
    floor     the fewest weighted tokens on the front, with success at least --floor (an absolute share of games won)

Reflection's agent runs a shell over the traces: run the text spaces where model-written shell commands are acceptable,
such as a VM or a container. The macro is model-written code that the agent's tool runs in this process, loaded under
restrictions (strands_evals.optimization.programs) that are not a sandbox: run the macro spaces there too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from alfworld_programs import CONTEXT as MACRO_CONTEXT
from alfworld_programs import macro_tools, macro_variable
from alfworld_workload import (
    MAX_STEPS,
    TASK_TYPES,
    ALFWorldTask,
    InvalidCommands,
    Steps,
    Success,
    act,
    cases,
    demonstration,
    game_commands,
    game_index,
    sample,
    serve,
)
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.models.openai import OpenAIModel
from strands.tools.executors import SequentialToolExecutor
from strands.vended_plugins.skills import AgentSkills

from strands_evals import Experiment
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Block,
    CheapestOnFront,
    Choice,
    Derived,
    Float,
    Int,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    Reflection,
    Search,
    Selection,
    SkillCuration,
    SkillLibrary,
    Split,
    Study,
    TextValue,
    component,
    tool_descriptions,
    tunable,
)
from strands_evals.optimization.trials import run_trials

REFLECTION_MODEL = os.environ.get("ALFWORLD_REFLECTION_MODEL", "global.anthropic.claude-sonnet-5-5")
REFLECTION_MAX_TOKENS = 32000  # the texts are written in one turn: over the model's default output
MAX_TOKENS = int(os.environ.get("ALFWORLD_MAX_TOKENS", "8192"))  # per model call, thinking included
CONTEXT = int(os.environ.get("ALFWORLD_CONTEXT", "32768"))  # the served model's context window, in tokens

# Each kind of token, weighted by its price relative to an uncached input token. An OpenAI-compatible endpoint counts
# cached tokens in its prompt tokens, so they are taken out of the input before the cache's weight is added.
WEIGHTS = {"input_tokens": 1.0, "output_tokens": 5.0, "cache_write_tokens": 1.25, "cache_read_tokens": 0.1}
WEIGHTED_TOKENS = Derived(
    "weighted_tokens",
    lambda m: sum(m[k] * w for k, w in WEIGHTS.items()) - m["cache_read_tokens"] * WEIGHTS["input_tokens"],
    range=(0, None),
)

SYSTEM_PROMPT = """You are playing a text-based household game. Your task is in the first message. Complete it by \
running commands with the act tool, one command per call; each call returns what you observe.

%s

Objects and receptacles are named with a number, such as "apple 1" or "countertop 2", and must be written exactly as \
the game writes them. A command the game does not accept returns "Nothing happens." You have at most %d commands. The \
game ends when the task is complete.""" % (game_commands(), MAX_STEPS)  # the command list is the game's own


def model(thinking: str, temperature: float) -> OpenAIModel:
    """The served model, thinking or not (Qwen3.5's chat template takes ``enable_thinking``) at ``temperature``."""
    url, model_id = serve()
    return OpenAIModel(
        client_args={"base_url": url, "api_key": "EMPTY", "timeout": 900.0, "max_retries": 4},
        model_id=model_id,
        params={
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking == "on"}},
        },
        context_window_limit=CONTEXT,
    )


# The default context manager: a window of 120 messages with the first (the room and the task) pinned. A game of 50
# commands holds 101 messages, more when model calls run no command. Qwen3.5's chat template refuses a conversation
# with no user message ("No user query found in messages."), so every window keeps the first one.
WINDOW = {"window_size": 120, "pin_first": 1, "per_turn": True}


def player(
    system_prompt: str = SYSTEM_PROMPT,
    tools: Any = (act,),
    thinking: str = "off",
    temperature: float = 0.7,
    conversation_manager: Any = None,
    hints: bool = False,
    macro: Any = None,
    skills: str | None = None,
) -> Agent:
    """The agent every space tunes; what a space does not tune stays at its default here. ``macro``: a loaded macro
    (alfworld_programs.py), one more tool beside ``act``; None, the default, adds none. ``skills``: a skill library's
    directory, given to the agent with Strands' ``AgentSkills`` plugin; None (an empty library), the default, adds no
    plugin (an empty plugin would still add its tool and a block to the system prompt)."""
    agent = Agent(
        model=model(thinking, temperature),
        tools=[*tools, *macro_tools(macro)],
        plugins=[AgentSkills(skills=skills)] if skills else [],
        system_prompt=system_prompt,
        conversation_manager=conversation_manager or SlidingWindowConversationManager(**WINDOW),
        tool_executor=SequentialToolExecutor(),  # a turn with several commands plays them in order
        callback_handler=None,
    )
    agent.state.set("hints", hints)  # the task shows the admissible commands (alfworld_workload.py)
    return agent


# What can be tuned. A conversation of 50 commands holds 101 messages, so the default window (120) never cuts one.
PROMPT = TextValue(
    default=SYSTEM_PROMPT, max_chars=6000, about="the system prompt: the goal, the tool and the commands"
)
TOOLS = tool_descriptions(act, max_chars=1500)
THINKING = Choice(["off", "on"], default="off")
TEMPERATURE = Float(0.0, 1.2, default=0.7)
HINTS = Choice([False, True], default=False)  # append the game's admissible commands to each observation
WINDOW_SIZE = Int(8, 120, log=True, default=120)
SUMMARY_RATIO = Float(0.1, 0.8, default=0.3)
PRESERVE_RECENT = Int(4, 40, default=10)
MANAGER = OneOf(
    component(SlidingWindowConversationManager, window_size=WINDOW_SIZE, pin_first=1, per_turn=True),
    component(
        SummarizingConversationManager,
        summary_ratio=SUMMARY_RATIO,
        preserve_recent_messages=PRESERVE_RECENT,
        proactive_compression={"compression_threshold": 0.6},
    ),
    labels=["sliding_window", "summarizing"],
)


# The macro (--macro, and the macro space): its proposer is also told the agent's default system prompt.
MACRO = macro_variable(context=f"{MACRO_CONTEXT}\nThe agent's system prompt (the default):\n\n{SYSTEM_PROMPT}\n")
# The helper variant (macro="helper", and the macro-helper space): at most 8 commands a call, no task line, no task
# sentence as an argument (alfworld_programs.HelperMacro): code that helps the agent rather than plays for it.
MACRO_HELPER = macro_variable(
    helper=True, context=f"{MACRO_CONTEXT}\nThe agent's system prompt (the default):\n\n{SYSTEM_PROMPT}\n"
)
# The skill library (--skills): empty at the baseline, written by SkillCuration (its default block).
SKILLS = SkillLibrary(
    default=None,
    about="skills the agent can load while it plays ALFWorld's household games: how to find objects, the commands each "
    "task type needs (heat, cool, clean, use a lamp, two objects), and how to recover from 'Nothing happens.'",
)


# Each factory takes ``macro`` and ``skills`` (default none), which --macro and --skills bind (``with_macro``,
# ``with_skills``).
@tunable(system_prompt=PROMPT, tools=TOOLS)
def text(system_prompt, tools, macro=None, skills=None):
    return player(system_prompt, tools, macro=macro, skills=skills)


@tunable(system_prompt=PROMPT, tools=TOOLS, thinking=THINKING, temperature=TEMPERATURE)
def mini(system_prompt, tools, thinking, temperature, macro=None, skills=None):
    return player(system_prompt, tools, thinking, temperature, macro=macro, skills=skills)


@tunable(system_prompt=PROMPT, tools=TOOLS, thinking=THINKING, temperature=TEMPERATURE, hints=HINTS)
def mini_hints(system_prompt, tools, thinking, temperature, hints, macro=None, skills=None):
    return player(system_prompt, tools, thinking, temperature, hints=hints, macro=macro, skills=skills)


@tunable(system_prompt=PROMPT, tools=TOOLS, thinking=THINKING, temperature=TEMPERATURE, conversation_manager=MANAGER)
def ladder(system_prompt, tools, thinking, temperature, conversation_manager, macro=None, skills=None):
    return player(system_prompt, tools, thinking, temperature, conversation_manager, macro=macro, skills=skills)


@tunable(
    system_prompt=PROMPT,
    tools=TOOLS,
    thinking=THINKING,
    temperature=TEMPERATURE,
    conversation_manager=MANAGER,
    hints=HINTS,
)
def ladder_hints(system_prompt, tools, thinking, temperature, conversation_manager, hints, macro=None, skills=None):
    return player(system_prompt, tools, thinking, temperature, conversation_manager, hints, macro, skills)


@tunable(macro=MACRO)
def macro_only(macro, skills=None):
    return player(macro=macro, skills=skills)


@tunable(macro=MACRO_HELPER)
def macro_helper_only(macro, skills=None):
    return player(macro=macro, skills=skills)


def with_macro(factory: Any, helper: bool = False) -> Any:
    """``factory`` with the macro tuned too (``MACRO``, or ``MACRO_HELPER`` with ``helper``: none at the baseline,
    then written by ``ProgramProposer``)."""
    return factory.with_space({**factory.bindings, "macro": MACRO_HELPER if helper else MACRO})


def with_skills(factory: Any) -> Any:
    """``factory`` with the skill library tuned too (``SKILLS``: empty at the baseline, then written by
    ``SkillCuration``)."""
    return factory.with_space({**factory.bindings, "skills": SKILLS})


# Each space: its factory and the texts reflection rewrites.
SPACES = {
    "text": (text, [PROMPT, TOOLS]),
    "mini": (mini, [PROMPT, TOOLS]),
    "mini-hints": (mini_hints, [PROMPT, TOOLS]),
    "L1": (text, [PROMPT, TOOLS]),
    "L2": (ladder, [PROMPT, TOOLS]),
    "L2h": (ladder_hints, [PROMPT, TOOLS]),
    "macro": (macro_only, []),
    "macro-helper": (macro_helper_only, []),
}
SPACES["full"] = SPACES["L2h"]


def with_demonstrations(factory: Any, games: list[dict[str, str]]) -> tuple[Any, TextValue]:
    """``factory`` whose system prompt starts from the baseline's followed by one expert play of each task type (the
    first of ``games`` of that type): a seed for the rewrites that holds examples. The demonstrations are the games'
    own walkthroughs, replayed in the game; their games belong to no role of the study."""
    plays = []
    for task_type in TASK_TYPES:
        game = next(g for g in games if g["task_type"] == task_type)
        plays.append(f"Example ({task_type.replace('_', ' ')}):\n{demonstration(game['game_file'])}")
    seed = (
        SYSTEM_PROMPT
        + "\n\nHere is how the game goes, in six examples, each a task and the commands that complete it.\n\n"
    )
    seed += "\n\n".join(plays)
    prompt = TextValue(default=seed, max_chars=24000, about=PROMPT.about)
    return factory.with_space({**factory.bindings, "system_prompt": prompt}), prompt


GOALS = ("best", "budget", "joint", "cheapest", "floor")


def goals(goal: str, floor_gain: float = 0.10, floor: float = 0.5) -> dict[str, Any]:
    """The goal's objectives and constraints; a limit is relative to the baseline's, except ``floor``'s."""
    if goal == "best":
        return dict(objectives=Maximize("success"))
    if goal == "budget":
        return dict(objectives=Maximize("success"), constraints=[AtMost(WEIGHTED_TOKENS, 1.0, compare="ratio")])
    if goal == "joint":
        return dict(objectives=Pareto(Maximize("success"), Minimize(WEIGHTED_TOKENS)))
    if goal == "cheapest":
        return dict(
            objectives=CheapestOnFront(Minimize(WEIGHTED_TOKENS), Maximize("success")),
            constraints=[AtLeast("success", floor_gain, compare="difference")],
        )
    if goal == "floor":
        return dict(
            objectives=CheapestOnFront(Minimize(WEIGHTED_TOKENS), Maximize("success")),
            constraints=[AtLeast("success", floor)],
        )
    raise ValueError(f"--goal is one of {GOALS}, not {goal!r}")


EVALUATORS = [Success(name="success"), Steps(name="steps"), InvalidCommands(name="invalid_commands")]
SEED = 0  # how the search and validation games are drawn from the train split


def splits(search_per_type: int, validation_per_type: int, seed: int = SEED) -> tuple[list, list, list]:
    """(search, validation, test) cases: disjoint stratified samples of the train split, and all of valid_unseen."""
    search = sample("train", search_per_type, seed=seed)
    validation = sample("train", validation_per_type, seed=seed + 1, exclude=frozenset(g["name"] for g in search))
    return cases(search, "train"), cases(validation, "train"), cases(list(game_index("valid_unseen")), "valid_unseen")


def make_study(
    space: str,
    goal: str,
    *,
    repetitions: int,
    record: str | None,
    search_per_type: int = 12,
    validation_per_type: int = 6,
    test_games: int | None = None,
    concurrency: int = 128,
    width: int = Search().width,
    demos: bool = False,
    finalists: int = Selection().finalists,
    phases: Any = "auto",
    macro: bool | str = False,
    skills: bool = False,
    **limits: Any,
) -> Study:
    """A study of ``space`` for ``goal``: the search on ``search_per_type`` train games of each task type, the selection
    on ``validation_per_type`` others, the test on valid_unseen (the first ``test_games`` of it, if given); ``demos``
    starts the system prompt from one with an expert play of each task type. ``phases``: ``Search(phases=...)``; the
    default searches a goal with a cost objective reward first (PrimaryFirst), whose finalists are phase A's result
    and cheaper candidates no worse than it; None searches the goal as declared, so a Pareto goal's finalists are the
    members of its decided front (up to 5), and the test measures every member of the front validation leaves.
    ``macro`` tunes the macro too (its default block: ``ProgramProposer``; ``"helper"``: the helper variant),
    ``skills`` a skill library (its default block: ``SkillCuration``)."""
    factory, texts = SPACES[space]
    if macro:
        factory = with_macro(factory, helper=macro == "helper")
    if skills:
        factory = with_skills(factory)
    if demos and not texts:
        raise ValueError(f"--demos starts the system prompt from examples: the {space} space does not tune it")
    train, validation, test = splits(search_per_type, validation_per_type)
    if test_games is not None:
        test = test[:test_games]
    if demos:  # demonstrations come from train games the search and validation never use
        used = {c.metadata["game_file"] for c in train + validation}
        names = frozenset(Path(f).parent.name for f in used)
        factory, prompt = with_demonstrations(factory, sample("train", 1, seed=SEED + 2, exclude=names))
        texts = [prompt, *texts[1:]]
    config = {"model_id": REFLECTION_MODEL, "max_tokens": REFLECTION_MAX_TOKENS}
    blocks = [Block(Reflection(model_config=config), *texts)] if texts else []  # the macro space has no texts
    if skills:  # the curator writes its decisions in one turn too: Reflection's model and output limit
        blocks.append(Block(SkillCuration(model_config=config), SKILLS))
    return Study(
        agent=factory,
        experiment=Experiment(cases=train + validation + test, evaluators=EVALUATORS),
        task=ALFWorldTask(),
        **goals(goal, **limits),
        search=Search(blocks=blocks, width=width, phases=phases),
        repetitions=repetitions,
        split=Split(train=train, validation=validation, test=test),
        max_concurrency=concurrency,  # each running trial holds one game process
        selection=Selection(finalists=finalists),
        retries=4,
        record=record,
    )


def baseline(cases_: list, repetitions: int, concurrency: int, values: dict[str, Any] | None = None) -> dict:
    """The default harness (with ``values`` set, if given) on ``cases_``: every trial, and the share of games won and
    the cost over them."""
    experiment = Experiment(cases=cases_, evaluators=EVALUATORS)
    run = run_trials(
        mini,
        values or {},
        experiment,
        repetitions=repetitions,
        max_concurrency=concurrency,
        task=ALFWorldTask(),
        retries=4,
    )
    trials = asyncio.run(run)
    rows = [
        {"case": t.case, "repetition": t.repetition, "status": t.status.value, "error": t.error, **t.metrics}
        for t in trials
    ]
    scored = [r for r in rows if isinstance(r.get("success"), (int, float))]
    weighted = [sum(r.get(k, 0) * w for k, w in WEIGHTS.items()) - r.get("cache_read_tokens", 0) for r in scored]
    summary = {
        "model": serve()[1],
        "cases": len(cases_),
        "repetitions": repetitions,
        "statuses": {s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}},
        "success": statistics.fmean(r["success"] for r in scored) if scored else None,
        "steps": statistics.fmean(r["steps"] for r in scored) if scored else None,
        "weighted_tokens": statistics.fmean(weighted) if weighted else None,
        "model_calls": statistics.fmean(r.get("model_calls", 0) for r in scored) if scored else None,
        **{k: statistics.fmean(r.get(k, 0) for r in scored) if scored else None for k in WEIGHTS},
    }
    return {
        "summary": summary,
        "trials": rows,
        "outputs": {f"{t.case}#{t.repetition}": t.output for t in trials},
        "messages": {f"{t.case}#{t.repetition}": list(t.messages) for t in trials},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", action="store_true", help="measure the default harness only, on --split")
    parser.add_argument("--split", default="valid_unseen", help="--baseline: the games measured (train: a sample)")
    parser.add_argument("--space", choices=SPACES, default="mini")
    parser.add_argument("--goal", choices=GOALS, default="best")
    parser.add_argument(
        "--floor-gain", type=float, default=0.10, help="cheapest: the least success gain over the baseline"
    )
    parser.add_argument("--floor", type=float, default=0.5, help="floor: the least share of games won (absolute)")
    parser.add_argument("--search-per-type", type=int, default=12, help="train games per task type the search uses")
    parser.add_argument(
        "--validation-per-type", type=int, default=6, help="other train games per task type for selection"
    )
    parser.add_argument(
        "--test-games", type=int, default=None, help="a smaller test: the first this many valid_unseen games"
    )
    parser.add_argument("--repetitions", type=int, default=2, help="of each case, every time a configuration runs")
    parser.add_argument("--iterations", type=int, default=16, help="steps of the search, each of --width candidates")
    parser.add_argument("--width", type=int, default=Search().width, help="candidates each step proposes and measures")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--finalists", type=int, default=Selection().finalists)
    parser.add_argument("--concurrency", type=int, default=128, help="games at once")
    parser.add_argument(
        "--demos", action="store_true", help="the system prompt starts with an expert play of each task type"
    )
    parser.add_argument("--macro", action="store_true", help="also tune a macro tool beside act, from none")
    parser.add_argument("--skills", action="store_true", help="also tune a skill library (AgentSkills), from empty")
    parser.add_argument(
        "--out", type=Path, required=True, help="--baseline: the result's file; else: the records' directory"
    )
    args = parser.parse_args()
    if args.baseline:
        if args.split == "train":
            chosen = cases(sample("train", args.search_per_type), "train")
        else:
            chosen = cases(list(game_index(args.split))[: args.test_games], args.split)
        result = baseline(chosen, args.repetitions, args.concurrency)
        args.out.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps(result["summary"], indent=1))  # noqa: T201
    else:
        record = args.out / f"alfworld-{args.space}-{args.goal}-{time.strftime('%Y%m%d-%H%M%S')}"
        study = make_study(
            args.space,
            args.goal,
            repetitions=args.repetitions,
            record=str(record),
            search_per_type=args.search_per_type,
            validation_per_type=args.validation_per_type,
            test_games=args.test_games,
            concurrency=args.concurrency,
            width=args.width,
            demos=args.demos,
            macro=args.macro,
            skills=args.skills,
            finalists=args.finalists,
            floor_gain=args.floor_gain,
            floor=args.floor,
        )
        study.optimize(iterations=args.iterations, patience=args.patience)
