"""Quickstart: a whole study, offline, in under a minute.

The agent is a real Strands agent with a scripted model (``ScriptedModel``) in place of an LLM. It answers sums. It
answers correctly more often when it thinks harder (``effort``), works in more steps (the ``stepwise`` style) or samples
at a lower ``temperature``; effort and steps cost it tokens. Everything else runs as it would on a real model: the
search, the trials, the model of every goal metric, the selection on validation cases, the test on cases nothing saw
before, and the record.

    python examples/optimization/quickstart.py                  # the highest accuracy
    python examples/optimization/quickstart.py --goal budget    # the highest accuracy within --budget tokens a trial
    python examples/optimization/quickstart.py --goal floor     # the fewest tokens with accuracy of at least --floor
    python examples/optimization/quickstart.py --goal pareto    # accuracy and tokens together

The scripted model draws its mistakes at random, so your numbers will differ a little from run to run.
"""

import argparse
import random
import re
import time

from strands import Agent
from strands.models.model import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    CheapestOnFront,
    Choice,
    Float,
    Int,
    Maximize,
    Minimize,
    OneOf,
    Pareto,
    Split,
    Study,
    component,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

# ---- the agent: a scripted model whose accuracy and cost depend on its settings --------------------------------------
EFFORT = {"low": (0.55, 150), "medium": (0.75, 500), "high": (0.90, 1400)}  # skill, and thinking tokens an answer


class Concise:
    """Answer at once."""

    bonus, output_tokens = 0.0, 20


class Stepwise:
    """Work in ``steps`` steps before answering: each step adds accuracy and 120 output tokens."""

    def __init__(self, steps: int) -> None:
        self.bonus, self.output_tokens = 0.025 * steps, 20 + 120 * steps


class ScriptedModel(Model):
    """Stands in for an LLM: right with a probability that rises with effort and steps, and falls with temperature and
    with the digits in the sum; it reports 300 input tokens, and its thinking and steps as output tokens."""

    def __init__(self, effort: str, style: Concise | Stepwise, temperature: float) -> None:
        self.skill, self.thinking = EFFORT[effort]
        self.style, self.temperature = style, temperature
        self.config = {"model_id": "scripted"}

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        a, b = (int(n) for n in re.findall(r"\d+", messages[-1]["content"][0]["text"]))
        difficulty = len(f"{a}{b}") / 8
        right = random.random() < min(0.99, self.skill + self.style.bonus - 0.35 * difficulty - 0.15 * self.temperature)
        output = self.thinking + self.style.output_tokens
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": str(a + b if right else a + b + 1)}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": 300, "outputTokens": output, "totalTokens": 300 + output}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


# What can be tuned: a category, a number, and a choice between two styles, one of which has a setting of its own.
@tunable(
    effort=Choice(["low", "medium", "high"], default="low"),
    temperature=Float(0.0, 1.0, default=0.7),
    style=OneOf(component(Concise), component(Stepwise, steps=Int(1, 8, default=2)), labels=["concise", "stepwise"]),
)
def create_agent(effort, temperature, style):
    return Agent(model=ScriptedModel(effort, style, temperature), callback_handler=None)


# ---- the workload: 48 sums, and an evaluator that scores each reply 1 or 0 -------------------------------------------
class Correct(Evaluator):
    def evaluate(self, case):
        right = str(case.actual_output).strip() == case.expected_output
        return [EvaluationOutput(score=float(right), test_pass=right, reason=f"replied {case.actual_output!r}")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def sums(n: int = 48) -> list[Case]:
    rng = random.Random(0)
    cases = []
    for i in range(n):
        digits = 1 + i % 4
        a, b = (rng.randrange(10 ** (digits - 1), 10**digits) for _ in range(2))
        cases.append(Case(name=f"sum-{i:02d}", input=f"What is {a} + {b}?", expected_output=str(a + b)))
    return cases


# ---- the goals -------------------------------------------------------------------------------------------------------
def goal(name: str, budget: float, floor: float) -> dict:
    """The objectives and constraints of each goal."""
    return {
        "best": dict(objectives=Maximize("correct")),
        "budget": dict(objectives=Maximize("correct"), constraints=[AtMost("tokens", budget)]),
        "floor": dict(
            objectives=CheapestOnFront(Minimize("tokens"), Maximize("correct")), constraints=[AtLeast("correct", floor)]
        ),
        "pareto": dict(objectives=Pareto(Maximize("correct"), Minimize("tokens"))),
    }[name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", choices=("best", "budget", "floor", "pareto"), default="best")
    parser.add_argument("--budget", type=float, default=1200, help="budget: the most tokens a trial, on average")
    parser.add_argument("--floor", type=float, default=0.6, help="floor: the least accuracy")
    parser.add_argument("--iterations", type=int, default=8, help="steps of the search, each of 4 candidates")
    args = parser.parse_args()

    study = Study(
        agent=create_agent,
        experiment=Experiment(cases=sums(), evaluators=[Correct(name="correct")]),
        **goal(args.goal, args.budget, args.floor),
        repetitions=2,  # each candidate runs every case twice
        split=Split(train=0.5, validation=0.25, test=0.25),  # search on 24 sums, select on 12, test on 12
        max_concurrency=200,  # a step's 4 x 48 trials at once: a scripted model has no rate limit (Bedrock has)
        record=f"studies/quickstart-{args.goal}-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    result = study.optimize(iterations=args.iterations)

    run = result.runs[0]  # one search (Split(rotate=True) would run one per test share)
    shipped = run.pick if run.pick is not None else (run.selected[0].number if run.selected else 0)
    print(f"\nships candidate {shipped}: {study.candidates[shipped].params}")  # noqa: T201
    print(f"record: {study.record.directory}")  # noqa: T201
