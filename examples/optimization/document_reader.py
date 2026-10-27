"""Tune a Strands agent that reads a long document: its system prompt, its tools' descriptions, which conversation
manager it uses, and that manager's settings. Quality and token use are optimized together, within a quality floor and
a token budget.

The agent must read a chained document page by page (document_reader_workload.py) and report a code spelled out across
the pages; what the conversation manager keeps decides whether it can.

    BYPASS_TOOL_CONSENT=true python examples/optimization/document_reader.py

(The system prompt is rewritten by contrastive reflection, whose agent runs a shell over the traces: run this where
model-written shell commands are acceptable, such as a VM or a container.)

The study runs in this process, against a Strands Evals `Experiment`. A version with each document as a packaged task,
the agent acting in the task's container, comes with Strands Evals' Harbor integration.
"""

import time

from document_reader_workload import Answer, cases, open_document, read_page
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.models.bedrock import BedrockModel

from strands_evals import Experiment
from strands_evals.optimization import (
    TPE,
    AtLeast,
    AtMost,
    Block,
    Choice,
    Float,
    Int,
    Joint,
    Maximize,
    Minimize,
    Nested,
    OneOf,
    Pareto,
    Reflection,
    Search,
    Study,
    TextValue,
    component,
    tool_descriptions,
    tunable,
)

# What can be tuned. Each variable is a Python object: the agent's factory uses it below, and so does the search.
# Plain values (per_turn=True) are fixed.
system_prompt = TextValue(default="Answer the request exactly in the format it asks for.")  # free text
# One TextValue per tool, each starting from the tool's own description (at most 800 characters); the factory gets
# copies of the tools with the candidate's descriptions. Also: a skill library, SkillLibrary("./skills") (or
# SkillLibrary() for an empty one), passed to Agent(plugins=[AgentSkills(skills=skills or [])]).
tools = tool_descriptions(open_document, read_page)
window_size = Int(4, 40, default=6)  # also: Int(..., step=2), Int(..., log=True)
pin_first = Choice([0, 1], default=0)  # a categorical setting inside one option
summary_ratio = Float(0.1, 0.8, default=0.3)  # also: Float(..., log=True), Float(..., step=0.05)
preserve_recent = Int(2, 12, default=10)
manager = OneOf(  # a choice between components, each with its own settings
    component(SlidingWindowConversationManager, window_size=window_size, pin_first=pin_first, per_turn=True),
    component(
        SummarizingConversationManager,
        summary_ratio=summary_ratio,
        preserve_recent_messages=preserve_recent,
        proactive_compression={"compression_threshold": 0.6},
    ),
    labels=["sliding_window", "summarizing"],
)


@tunable(system_prompt=system_prompt, tools=tools, conversation_manager=manager)
def create_reader(system_prompt, tools, conversation_manager):
    return Agent(
        model=BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", context_window_limit=4000),
        tools=tools,
        conversation_manager=conversation_manager,
        system_prompt=system_prompt,
        callback_handler=None,
    )


# How to search (these are the defaults, written out). A block is a method and the variables it proposes: a variable
# itself, or a component for every variable declared in it (as tools is). Methods: TPE(seed=7, n_startup_trials=4)
# (Optuna's TPE, which proposes at random until a landscape holds 4 candidates), Optuna(lambda:
# optuna.samplers.GPSampler()), Reflection(model_config=...), SkillCuration() for a skill library. A variable no
# block holds gets the default block for its type.
choose_manager = Block(TPE(), manager)  # which conversation manager
tune_manager = Block(TPE(), window_size, pin_first, summary_ratio, preserve_recent)  # the chosen manager's settings
# One reflection pass reads the traces and changes the texts it has a reason to change. Also: the tools in a block of
# their own, Block(Reflection(), tools), to rewrite their descriptions in a pass of their own.
rewrite_texts = Block(Reflection(), system_prompt, tools)
search = Search(
    blocks=[choose_manager, tune_manager, rewrite_texts],
    # Which blocks change in each iteration: rounds that each propose a manager, then search the other blocks jointly
    # under it for 4 more iterations; the manager's method is told each round's best. Also: Joint() (every block in
    # every iteration), Alternate(tune_manager, rewrite_texts, iterations=1) (one block at a time), or
    # Nested(..., inner=Nested(outer=..., inner=Joint())) (rounds within rounds).
    schedule=Nested(outer=choose_manager, inner=Joint(), inner_iterations=4),
    redraws=3,  # how many more times a step proposes when it repeats a candidate already measured
)

study = Study(
    agent=create_reader,
    # Each evaluator gives one metric, named by its name (its class name when it has none).
    experiment=Experiment(cases=cases(), evaluators=[Answer(name="answer")]),
    # Each candidate (one set of values) runs every case this many times; each run is a trial. The default is 1.
    repetitions=2,
    # Goals name metrics: an evaluator's ("answer", scored in 0 to 1 unless the evaluator sets score_range), or the
    # agent's usage, measured by the study: "tokens" (= input_tokens + output_tokens + cache_read_tokens +
    # cache_write_tokens), "seconds", "model_calls", "removed_messages". Or one computed from those, per trial:
    # Derived("cost", lambda m: m["input_tokens"] * 1e-6 + m["output_tokens"] * 5e-6), used in place of a name.
    # Several objectives are wrapped in how they combine: Pareto(...) optimizes them jointly and reports the Pareto set.
    # Also: WeightedSum(...), Chebyshev(...), Lexicographic(...), or Independent(...) (one copy of every method per
    # objective); one objective alone needs no wrapper: objectives=Maximize("answer"). Maximize/Minimize also take
    # weight=, scale= (a meaningful change, needed to combine an unbounded metric such as tokens by WeightedSum or
    # Chebyshev) and tolerance= (for Lexicographic).
    objectives=Pareto(Maximize("answer"), Minimize("tokens")),
    constraints=[AtLeast("answer", 0.8), AtMost("tokens", 30_000)],  # any metric; [] for none (the default)
    search=search,  # omitted: the defaults, as written out above
    max_concurrency=4,  # trials (a candidate's cases and repetitions) run at once
    record=f"studies/document-reader-{time.strftime('%Y%m%d-%H%M%S')}",  # written as candidates finish; None for none
    log=print,  # one line per finished candidate (the default); None for silence
)

if __name__ == "__main__":
    # Measure the baseline (the agent at its defaults), then 10 iterations, each proposing and measuring a candidate.
    study.optimize(iterations=10)  # also: patience=4 (stop once the best has not changed for 4 iterations)
