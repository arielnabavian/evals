"""The effects model recovers known effects: a study whose score is a known sum of per-choice effects plus noise."""

import json
import random

import optuna
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtLeast,
    Block,
    Choice,
    Float,
    Joint,
    Maximize,
    OneOf,
    Optuna,
    Search,
    Study,
    component,
    effects,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says, needs_bayesbench

# score = 0.3 + mode + manager (+ its own setting) + case, each additive; noise sd 0.04
MODE = {"a": 0.0, "b": 0.2}
WINDOW_KEEP = {"few": 0.0, "many": 0.06}
SUMMARY_STYLE = {"brief": 0.0, "full": 0.1}
SUMMARY = 0.15
CASE = {"sea": 0.0, "city": 0.05}
rng = random.Random(3)


class Window:
    def __init__(self, keep):
        self.effect = WINDOW_KEEP[keep]


class Summary:
    def __init__(self, style):
        self.effect = SUMMARY + SUMMARY_STYLE[style]


MODE_CHOICE = Choice(["a", "b"], default="a")
# no effect: it makes every candidate new (a study never measures the same values twice, and the choices alone have 8
# combinations)
JITTER = Float(0.0, 1.0, default=0.5)
KEEP = Choice(["few", "many"], default="few")
STYLE = Choice(["brief", "full"], default="brief")
MANAGER = OneOf(component(Window, keep=KEEP), component(Summary, style=STYLE), labels=["window", "summary"])


@tunable(mode=MODE_CHOICE, jitter=JITTER, manager=MANAGER)
def create(mode, jitter, manager):
    return Agent(model=Says(f"{0.3 + MODE[mode] + manager.effect:.4f}"), callback_handler=None)


class Score(Evaluator):
    def evaluate(self, case):
        value = float(str(case.actual_output).strip()) + CASE[case.name.split("#")[0]] + rng.gauss(0, 0.04)
        return [EvaluationOutput(score=min(1.0, max(0.0, value)), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


@needs_bayesbench
def test_known_effects_lie_in_their_intervals_and_the_model_fits(tmp_path):
    experiment = Experiment(cases=[Case(name="sea", input="x"), Case(name="city", input="y")], evaluators=[Score()])
    random_search = lambda: Optuna(lambda: optuna.samplers.RandomSampler(seed=1))  # noqa: E731
    study = Study(
        agent=create,
        experiment=experiment,
        objectives=Maximize("Score"),
        repetitions=2,
        search=Search(
            blocks=[Block(random_search(), MANAGER), Block(random_search(), MODE_CHOICE, JITTER, KEEP, STYLE)],
            schedule=Joint(),
        ),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=39)
    result = json.loads((tmp_path / "effects.json").read_text())  # written at the end of the run
    fitted = {(e["key"], e["level"]): e for e in result["metrics"]["Score"]["effects"]}
    truth = {
        ("mode", "b"): MODE["b"],
        ("manager", "summary"): SUMMARY + sum(SUMMARY_STYLE.values()) / 2 - sum(WINDOW_KEEP.values()) / 2,
        ("manager.window.keep", "many"): WINDOW_KEEP["many"],
        ("manager.summary.style", "full"): SUMMARY_STYLE["full"],
    }
    assert set(truth) <= set(fitted)
    covered = [lo <= truth[k] <= hi for k in truth for lo, hi in [fitted[k]["ci90"]]]
    assert sum(covered) >= len(truth) - 1, {k: (truth[k], fitted[k]["ci90"]) for k in truth}
    assert fitted[("mode", "b")]["p_positive"] > 0.99
    assert not result["metrics"]["Score"]["diagnostics"]["flagged"]
    assert result["goals"] == [{"role": "objective", "index": 0, "metric": "Score", "better": "higher"}]
    assert {o["key"] for o in result["omitted"]} == set()


def test_text_is_a_factor_by_how_many_rewrites_deep_it_is():
    candidates = [
        {"number": 0, "params": {"p": "x"}, "origin": {}},
        {"number": 1, "params": {"p": "x+a"}, "origin": {"text": {"from_candidate": 0}}},
        {"number": 2, "params": {"p": "x+a"}, "origin": {}},  # held, not rewritten
        {"number": 3, "params": {"p": "x+a+b"}, "origin": {"text": {"from_candidate": 2}}},
        {"number": 4, "params": {"p": "x+c"}, "origin": {"text": {"from_candidate": 0}}},
    ]
    assert effects.rewrites("p", "x", "text", candidates) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 1}


@needs_bayesbench
def test_a_text_that_never_varies_in_depth_is_left_out(tmp_path):
    (tmp_path / "study.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {
                    "type": "study",
                    "space": [{"key": "system_prompt", "type": "text", "default": "x", "active_when": None}],
                    "cases": ["a"],
                    "objectives": [{"kind": "Maximize", "metric": "m"}],
                    "constraints": [],
                    "metrics": {},
                },
                *[
                    {
                        "type": "candidate",
                        "number": n,
                        "role": "baseline" if n == 0 else "step",
                        "eligible": True,
                        "params": {"system_prompt": f"v{n}"},
                        "trials": [],
                        "estimates": {},
                    }
                    for n in range(4)
                ],
            ]
        )
    )
    result = effects.fit(tmp_path)
    assert result["omitted"] == [{"key": "system_prompt", "why": "text"}] and result["messages"] == ["no choice varies"]


@needs_bayesbench
def test_a_metric_that_is_an_objective_and_a_constraint_has_its_effects_once(tmp_path):
    experiment = Experiment(cases=[Case(name="sea", input="x"), Case(name="city", input="y")], evaluators=[Score()])
    study = Study(
        agent=create,
        experiment=experiment,
        objectives=Maximize("Score"),
        constraints=[AtLeast("Score", 0.4)],  # a floor on the objective's own metric
        repetitions=2,
        search=Search(blocks=[Block(Optuna(lambda: optuna.samplers.RandomSampler(seed=1)), MODE_CHOICE, JITTER)]),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=12)
    result = json.loads((tmp_path / "effects.json").read_text())
    assert [g["role"] for g in result["goals"]] == ["objective", "constraint"]  # both goals are stated
    levels = [(e["key"], e["level"]) for e in result["metrics"]["Score"]["effects"]]
    assert levels and len(levels) == len(set(levels))  # each effect once
