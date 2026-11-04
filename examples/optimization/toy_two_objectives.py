"""Two objectives with closed-form answers, searched by the real framework, with no model calls.

    python examples/optimization/toy_two_objectives.py                      # every goal and schedule, then a table
    python examples/optimization/toy_two_objectives.py --goal floor --schedule joint --out studies

The "agent" is a point (x1, x2): x1 is a choice of three options, a, b and c, and x2 a number in [0, 10]. Its quality
y1 = f(x1, x2) is a bump in x2 with its own height and centre for each option, and its cost y2 = g(x1, x2) rises in a
straight line with x2 from a base that depends on x1:

    f(x1, x2) = H[x1] * exp(-(x2 - M[x1])^2 / (2 * 2^2))      H = 0.60, 0.80, 0.95     M = 3, 5, 7
    g(x1, x2) = B[x1] + x2                                      B = 1, 2, 4

The cost's noise grows with the cost, and the study's model assumes one noise variance for every observation, so in
some cost runs its held-out checks flag the model of y2 after about 20 candidates and the study decides with measured
means from then on (as can happen with an agent's token counts); with additive noise they do not.

Both are known in closed form (``answers``), so every answer the study ships can be checked against the true one:

    the highest y1                   x1 = c, x2 = 7: y1 = 0.95, y2 = 11
    the cheapest y2 with y1 >= q     on each option, x2 = M - 2 * sqrt(2 * ln(H / q)) (or 0 when that is negative),
                                     the cheapest option at that x2; q = 0.7: x1 = b, x2 = 3.97, y2 = 5.97
    the Pareto front (max y1, min y2) the cheapest point at every level q: option a up to y1 = 0.6 (y2 1 to 4), b from
                                     0.6 to 0.8 (y2 5.48 to 7), c from 0.8 to 0.95 (y2 9.83 to 11)

Measurement is noisy the way an agent's is. A case is a task with its own difficulty: its success probability is f
times the case's factor, between 0.95 and 1.05 (mean 1 over all cases), and each trial is one Bernoulli draw from it, so
y1 is a pass rate with case-to-case variation. Each trial's cost is g times the case's cost factor, between 0.9 and 1.1,
times lognormal noise (0.2 on the log scale, mean 1). Validation and test cases are other cases, so their draws are
fresh. A draw depends only on the point, the case and the repetition (and --seed), so a run can be repeated exactly.

Goals (--goal):

    best   Maximize("y1")
    joint  Pareto(Maximize("y1"), Minimize(y2)): the validated front is tested, and the pick shipped is the cheapest
           validated candidate within --margin of the best validated y1 (for --schedule joint, the same rule applied to
           the record's validated front)
    floor  CheapestOnFront(Minimize(y2), Maximize("y1")) with AtLeast("y1", --floor)

The cost goals (joint, floor) search y1 first, then y2 (``PrimaryFirst``, --schedule reward-first, the default), or
are told both from the first step (--schedule joint). Every study searches with TPE (x1 and x2 in one block), in steps
of --width candidates, on the train share of --cases cases, and selects on the validation share.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    TPE,
    AtLeast,
    Block,
    CheapestOnFront,
    Choice,
    Derived,
    Float,
    Maximize,
    Minimize,
    Pareto,
    PrimaryFirst,
    Search,
    Selection,
    Split,
    Study,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

OPTIONS = ("a", "b", "c")
HEIGHT = {"a": 0.60, "b": 0.80, "c": 0.95}  # the best y1 each option reaches
CENTRE = {"a": 3.0, "b": 5.0, "c": 7.0}  # where it reaches it
WIDTH = 2.0  # how fast y1 falls away from the centre
BASE = {"a": 1.0, "b": 2.0, "c": 4.0}  # y2 at x2 = 0; it rises by 1 per unit of x2 under every option
LOW, HIGH = 0.0, 10.0


def f(x1: str, x2: float) -> float:
    """The true quality: the probability that a trial on an average case succeeds."""
    return HEIGHT[x1] * math.exp(-((x2 - CENTRE[x1]) ** 2) / (2 * WIDTH**2))


def g(x1: str, x2: float) -> float:
    """The true cost: a trial's expected cost on an average case."""
    return BASE[x1] + x2


def cheapest_at(q: float, x1: str) -> float | None:
    """The smallest x2 at which option x1 reaches y1 >= q (f rises up to the centre), or None when it never does."""
    if q > HEIGHT[x1]:
        return None
    if q <= f(x1, LOW):
        return LOW
    return CENTRE[x1] - WIDTH * math.sqrt(2 * math.log(HEIGHT[x1] / q))


def cheapest(q: float) -> tuple[str, float, float] | None:
    """The cheapest point with y1 >= q: (x1, x2, y2), or None when no option reaches q."""
    found = [(g(k, x), k, x) for k in OPTIONS if (x := cheapest_at(q, k)) is not None]
    if not found:
        return None
    cost, k, x = min(found)
    return k, x, cost


def front(levels: int = 200) -> list[tuple[str, float, float, float]]:
    """The true Pareto front of (max y1, min y2), as (x1, x2, y1, y2) at evenly spaced levels of y1: at each level,
    the cheapest point that reaches it (points past an option's centre lose y1 and cost more, so none is on it)."""
    top = max(HEIGHT.values())
    lowest = min(f(k, LOW) for k in OPTIONS)
    out = []
    for i in range(levels + 1):
        q = lowest + (top - lowest) * i / levels
        k, x, cost = cheapest(q)  # type: ignore[misc]
        out.append((k, x, f(k, x), cost))
    return out


def answers(floor: float, margin: float) -> dict[str, dict[str, Any]]:
    """Each goal's true answer, in closed form."""
    best = max(OPTIONS, key=lambda k: HEIGHT[k])
    k, x, cost = cheapest(floor)  # type: ignore[misc]
    pk, px, pcost = cheapest(HEIGHT[best] - margin)  # type: ignore[misc]
    return {
        "best": {"x1": best, "x2": CENTRE[best], "y1": HEIGHT[best], "y2": g(best, CENTRE[best])},
        "floor": {"x1": k, "x2": x, "y1": f(k, x), "y2": cost},
        "joint": {"x1": pk, "x2": px, "y1": f(pk, px), "y2": pcost, "rule": f"cheapest with y1 >= max y1 - {margin}"},
    }


# ---- the study ---------------------------------------------------------------------------------------------------
X1 = Choice(list(OPTIONS), default="a")
X2 = Float(LOW, HIGH, default=1.0)


@dataclass(frozen=True)
class Observation:
    success: int  # 1 when the trial succeeded
    cost: float

    def __str__(self) -> str:
        return f"success={self.success} cost={self.cost:.3f}"


@dataclass(frozen=True)
class Point:
    """The "agent": a point in the space, measured by ``trial``."""

    x1: str
    x2: float


@tunable(x1=X1, x2=X2)
def point(x1: str, x2: float) -> Point:
    return Point(x1, float(x2))


SEED = 0


def _uniform(*key: Any) -> random.Random:
    """A generator for one draw, fixed by the seed and ``key``."""
    digest = hashlib.sha256(repr((SEED, *key)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def trial(agent: Point, case: Case) -> Observation:
    """One trial: a Bernoulli draw with the case's success probability, and a noisy cost. ``case.name`` carries the
    repetition (``case-07#2``), so each repetition draws afresh."""
    rng = _uniform(agent.x1, round(agent.x2, 12), case.name)
    p = f(agent.x1, agent.x2) * case.input["quality"]
    cost = g(agent.x1, agent.x2) * case.input["cost"] * math.exp(0.2 * rng.gauss(0.0, 1.0) - 0.02)
    return Observation(int(rng.random() < p), cost)


class Quality(Evaluator):
    """y1: 1 when the trial succeeded, else 0."""

    def evaluate(self, data: Any) -> list[EvaluationOutput]:
        s = float(data.actual_output.success)
        return [EvaluationOutput(score=s, test_pass=s == 1.0, reason=str(data.actual_output))]

    async def evaluate_async(self, data: Any) -> list[EvaluationOutput]:
        return self.evaluate(data)


class Cost(Evaluator):
    """The trial's cost, 0 or more."""

    score_range = (0, None)

    def evaluate(self, data: Any) -> list[EvaluationOutput]:
        c = float(data.actual_output.cost)
        return [EvaluationOutput(score=c, test_pass=True, reason=f"{c:.3f}")]

    async def evaluate_async(self, data: Any) -> list[EvaluationOutput]:
        return self.evaluate(data)


Y2 = Derived("y2", lambda m: m["cost"], range=(0, None))  # a cost: minimized, with no upper bound


def cases(n: int, seed: int) -> list[Case]:
    """``n`` cases, each with its own quality factor in [0.95, 1.05] and cost factor in [0.9, 1.1]."""
    rng = random.Random(seed)
    return [
        Case(name=f"case-{i:02d}", input={"quality": rng.uniform(0.95, 1.05), "cost": rng.uniform(0.9, 1.1)})
        for i in range(n)
    ]


GOALS = ("best", "joint", "floor")
SCHEDULES = ("reward-first", "joint")


def make_study(
    goal: str,
    schedule: str,
    *,
    floor: float,
    margin: float,
    width: int,
    repetitions: int,
    n_cases: int,
    seed: int,
    record: str | None,
) -> Study:
    if goal == "best":
        goals: dict[str, Any] = {"objectives": Maximize("y1")}
    elif goal == "joint":
        goals = {"objectives": Pareto(Maximize("y1"), Minimize(Y2))}
    else:
        goals = {
            "objectives": CheapestOnFront(Minimize(Y2), Maximize("y1"), within=0.05),
            "constraints": [AtLeast("y1", floor)],
        }
    phased = goal != "best" and schedule == "reward-first"
    phases = PrimaryFirst(margin=margin if goal == "joint" else None) if phased else None
    return Study(
        agent=point,
        experiment=Experiment(cases=cases(n_cases, seed), evaluators=[Quality(name="y1"), Cost(name="cost")]),
        task=trial,
        **goals,
        search=Search(blocks=[Block(TPE(seed=seed), X1, X2)], width=width, phases=phases),
        repetitions=repetitions,
        split=Split(train=0.5, validation=0.25, test=0.25, seed=seed),
        selection=Selection(feasibility=0.8),
        max_concurrency=width * n_cases * repetitions,
        record=record,
        log=None,
    )


def case_mean(trials: Any, metric: str) -> float | None:
    """A metric's mean over the cases, each case averaged over its repetitions (the study's estimate)."""
    by_case: dict[str, list[float]] = {}
    for t in trials:
        if metric in t.metrics:
            by_case.setdefault(t.case, []).append(t.metrics[metric])
    if not by_case:
        return None
    return sum(sum(v) / len(v) for v in by_case.values()) / len(by_case)


def _events(record: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(record, "study.jsonl").read_text().splitlines()]


def joint_pick(record: str, margin: float) -> int | None:
    """For a Pareto goal searched without phases: the cheapest validated candidate whose decided y1 is within
    ``margin`` of the best decided y1 on validation (the rule a phased run ships by), from the record."""
    selected = next(e for e in _events(record) if e["type"] == "selected")
    decided = {int(n): d["metrics"] for n, d in selected["decided"].items() if d and d.get("metrics")}
    members = [n for n in selected["best"] if n in decided]
    if not members:
        return None
    top = max(decided[n]["y1"]["mean"] for n in members)
    within = [n for n in members if decided[n]["y1"]["mean"] >= top - margin]
    return min(within, key=lambda n: (decided[n]["y2"]["mean"], -decided[n]["y1"]["mean"], n))


def run(goal: str, schedule: str, args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="toy-two-objectives-"))
    record = str(out / f"toy-{goal}-{schedule}")
    study = make_study(
        goal,
        schedule,
        floor=args.floor,
        margin=args.margin,
        width=args.width,
        repetitions=args.repetitions,
        n_cases=args.cases,
        seed=args.seed,
        record=record,
    )
    result = study.optimize(iterations=args.iterations)
    r = result.runs[0]
    if r.pick is not None:
        shipped = r.pick
    elif goal == "joint":
        shipped = joint_pick(record, args.margin)
    else:
        shipped = r.selected[0].number if r.selected else None
    c = study.candidates[shipped] if shipped is not None else study.baseline
    arms = {a.number: a for a in [r.test.baseline, *r.test.selected, *r.test.alternatives]} if r.test else {}
    tested = arms.get(c.number)
    return {
        "goal": goal,
        "schedule": schedule if goal != "best" else "-",
        "candidates": len(study.candidates) - 1,
        "shipped": c.number,
        "x1": c.params["x1"],
        "x2": c.params["x2"],
        "true_y1": f(c.params["x1"], c.params["x2"]),
        "true_y2": g(c.params["x1"], c.params["x2"]),
        "test_y1": case_mean(tested.trials, "y1") if tested else None,
        "test_y2": case_mean(tested.trials, "cost") if tested else None,
        "anchor": r.anchor,
        "record": record,
    }


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", choices=GOALS, help="one goal (default: every goal)")
    parser.add_argument("--schedule", choices=SCHEDULES, help="one schedule for the cost goals (default: both)")
    parser.add_argument("--floor", type=float, default=0.7, help="floor: the lowest y1 allowed")
    parser.add_argument("--margin", type=float, default=0.03, help="joint: the pick's margin in y1")
    parser.add_argument("--iterations", type=int, default=10, help="steps of the search")
    parser.add_argument("--width", type=int, default=4, help="candidates each step proposes")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", type=int, default=60, help="cases: half train, a quarter each validation and test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="a directory for the records (default: a temporary one)")
    args = parser.parse_args()
    SEED = args.seed
    truth = answers(args.floor, args.margin)
    print("closed-form answers:")  # noqa: T201
    for goal, a in truth.items():
        print(f"  {goal:6} x1={a['x1']} x2={a['x2']:.2f} y1={a['y1']:.3f} y2={a['y2']:.2f}")  # noqa: T201
    runs = [
        (goal, schedule)
        for goal in ([args.goal] if args.goal else GOALS)
        for schedule in (["-"] if goal == "best" else [args.schedule] if args.schedule else SCHEDULES)
    ]
    rows = [run(goal, schedule if schedule != "-" else "reward-first", args) for goal, schedule in runs]
    print("\nshipped (true values from f and g; test values measured on the test cases):")  # noqa: T201
    for row in rows:
        a = truth[row["goal"]]
        test = f"test y1={row['test_y1']:.3f} y2={row['test_y2']:.2f}" if row["test_y1"] is not None else "not tested"
        print(  # noqa: T201
            f"  {row['goal']:6} {row['schedule']:12} candidate {row['shipped']:2} of {row['candidates']}: "
            f"x1={row['x1']} x2={row['x2']:.2f} true y1={row['true_y1']:.3f} (answer {a['y1']:.3f}) "
            f"true y2={row['true_y2']:.2f} (answer {a['y2']:.2f}); {test}"
        )
    print("\nrecords:", *(row["record"] for row in rows), sep="\n  ")  # noqa: T201


if __name__ == "__main__":
    main()
