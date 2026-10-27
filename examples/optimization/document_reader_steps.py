"""The document-reader study in six steps, each adding one idea to the step before, and two with limits.

The agent must read a chained document page by page (document_reader_workload.py) and report a code spelled out across
the pages; what its conversation manager keeps decides whether it can.

    python examples/optimization/document_reader_steps.py --step 1
    BYPASS_TOOL_CONSENT=true python examples/optimization/document_reader_steps.py --step 3

1  window         One number, the sliding window's size, searched by TPE for the best answer.
2  tradeoff       A categorical setting beside it (keep the first message, which holds the ticket), and tokens as a
                  second objective: the result is the Pareto set of answer and tokens.
3  prompt         Text alone: the system prompt, rewritten by reflection from the trials' transcripts, under the
                  baseline's small window.
4  prompt_window  The prompt and the window's settings together: two blocks (text, numbers), each with its own method,
                  both changing in every iteration (Joint).
5  managers       Structure: a choice between two conversation managers, each with its own settings. Rounds each
                  choose a manager, then search its settings and the prompt under it (Nested).
6  full           The tools' descriptions as text too, and limits: a quality floor and a token budget. The
                  configuration of document_reader.py, where every argument is annotated.

Steps 7 and 8 keep an earlier step's space and change its goal: a limit (a constraint) in place of a second objective.
A candidate that breaks a limit is infeasible: it is never the best, and the methods are told by how much it missed.

7  floor          Step 2's space: the fewest tokens among candidates whose answer is at least 0.9.
8  budget         Step 4's space: the best answer within 25,000 tokens per trial, which rules out the large windows,
                  so the prompt has to make up for a small one.

(From step 3 on, text is rewritten by contrastive reflection, whose agent runs a shell over the traces: run those
steps where model-written shell commands are acceptable, such as a VM or a container.)
"""

import argparse
import time

from document_reader_workload import Answer, cases, open_document, read_page
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.models.bedrock import BedrockModel

from strands_evals import Experiment
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Choice,
    Float,
    Int,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    Study,
    TextValue,
    component,
    tool_descriptions,
    tunable,
)


def make_model():
    return BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", context_window_limit=4000)


def reader(system_prompt=None, conversation_manager=None, tools=(open_document, read_page)):
    """The agent every step tunes; what a step does not tune stays as in the baseline."""
    return Agent(
        model=make_model(),
        tools=list(tools),
        conversation_manager=conversation_manager or SlidingWindowConversationManager(window_size=6, per_turn=True),
        system_prompt=system_prompt or PROMPT,
        callback_handler=None,
    )


PROMPT = "Answer the request exactly in the format it asks for."
WINDOW_SIZE = Int(4, 40, default=6)  # messages kept
PIN_FIRST = Choice([0, 1], default=0)  # keep the first message (the request, with the ticket) however small the window


# 1. One number.
@tunable(window_size=WINDOW_SIZE)
def window(window_size):
    return reader(conversation_manager=SlidingWindowConversationManager(window_size=window_size, per_turn=True))


# 2. A categorical setting, and a second objective.
@tunable(window_size=WINDOW_SIZE, pin_first=PIN_FIRST)
def tradeoff(window_size, pin_first):
    manager = SlidingWindowConversationManager(window_size=window_size, pin_first=pin_first, per_turn=True)
    return reader(conversation_manager=manager)


# 3. Text.
@tunable(system_prompt=TextValue(default=PROMPT))
def prompt(system_prompt):
    return reader(system_prompt)


# 4. Text and numbers: two blocks.
@tunable(system_prompt=TextValue(default=PROMPT), window_size=WINDOW_SIZE, pin_first=PIN_FIRST)
def prompt_window(system_prompt, window_size, pin_first):
    manager = SlidingWindowConversationManager(window_size=window_size, pin_first=pin_first, per_turn=True)
    return reader(system_prompt, manager)


# 5. Structure: a choice between components, each with its own settings.
MANAGERS = OneOf(
    component(SlidingWindowConversationManager, window_size=WINDOW_SIZE, pin_first=PIN_FIRST, per_turn=True),
    component(
        SummarizingConversationManager,
        summary_ratio=Float(0.1, 0.8, default=0.3),
        preserve_recent_messages=Int(2, 12, default=10),
        proactive_compression={"compression_threshold": 0.6},
    ),
    labels=["sliding_window", "summarizing"],
)


@tunable(system_prompt=TextValue(default=PROMPT), conversation_manager=MANAGERS)
def managers(system_prompt, conversation_manager):
    return reader(system_prompt, conversation_manager)


# 6. The tools' descriptions, and limits.
@tunable(
    system_prompt=TextValue(default=PROMPT),
    tools=tool_descriptions(open_document, read_page),
    conversation_manager=MANAGERS,
)
def full(system_prompt, tools, conversation_manager):
    return reader(system_prompt, conversation_manager, tools)


ANSWER, BOTH = Maximize("answer"), Pareto(Maximize("answer"), Minimize("tokens"))
LIMITS = [AtLeast("answer", 0.8), AtMost("tokens", 30_000)]
# Each step: its name, its factory, its goals, and its iterations (each Nested round is 1 + 4 of them).
STEPS = {
    1: ("window", window, dict(objectives=ANSWER), 9),
    2: ("tradeoff", tradeoff, dict(objectives=BOTH), 9),
    3: ("prompt", prompt, dict(objectives=ANSWER), 7),
    4: ("prompt_window", prompt_window, dict(objectives=BOTH), 9),
    5: ("managers", managers, dict(objectives=BOTH), 10),
    6: ("full", full, dict(objectives=BOTH, constraints=LIMITS), 10),
    7: ("floor", tradeoff, dict(objectives=Minimize("tokens"), constraints=[AtLeast("answer", 0.9)]), 9),
    8: ("budget", prompt_window, dict(objectives=ANSWER, constraints=[AtMost("tokens", 25_000)]), 9),
}


def study(step: int, record: str | None, **options) -> Study:
    """Every step's study, with the default search for its variables (text by reflection; numbers and choices by TPE;
    rounds over a structural choice), so each step changes only what it declares."""
    _, factory, goals, _ = STEPS[step]
    return Study(
        agent=factory,
        experiment=Experiment(cases=cases(), evaluators=[Answer(name="answer")]),
        repetitions=2,
        max_concurrency=6,  # every trial of a candidate at once
        record=record,
        **goals,
        **options,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, choices=list(STEPS), required=True)
    parser.add_argument("--iterations", type=int, default=None, help="default: the step's own")
    args = parser.parse_args()
    name, _, _, n = STEPS[args.step]
    record = f"studies/document-reader-{args.step}-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    study(args.step, record).optimize(iterations=args.iterations or n)
