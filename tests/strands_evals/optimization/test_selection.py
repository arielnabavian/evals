"""What a run ships when nothing meets the constraints (``Selection(unmet=...)``): the closest, unconfirmed,
or nothing."""

import json
import random

import pytest
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Block,
    Candidate,
    CheapestOnFront,
    Int,
    Maximize,
    Minimize,
    Pareto,
    Search,
    Selection,
    Split,
    Study,
    Trial,
    tunable,
)
from strands_evals.optimization.projection import Projection
from strands_evals.optimization.result import Result, Run
from strands_evals.optimization.selection import P_RESOLUTION, View, closest
from strands_evals.optimization.trials import TrialStatus
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says, needs_bayesbench


# ---- the rule, on views ----------------------------------------------------------------------------------------------
def projection(objectives, constraints):
    return Projection(objectives, constraints, evaluators={"Score": (0, 1), "Polite": (0, 1), "Words": (0, None)})


def view(p, number, metrics, p_met=()):
    """A candidate of one case with ``metrics``, as a decision sees it, with the given probabilities of meeting each
    constraint (none: decided by measured means)."""
    trials = (Trial(case="a", repetition=1, status=TrialStatus.COMPLETED, metrics=metrics),)
    estimates = p.estimates(trials)
    c = Candidate(
        number=number,
        params={},
        status="complete",
        trials=trials,
        estimates=estimates,
        eligible=True,
        violations=p.violations({k: e.mean for k, e in estimates.items()}),
        proposed=(),
        base=None,
        objective=None,
        origin={},
    )
    return View(c, c.metrics, c.violations, True, {}, tuple(p_met))


def test_the_closest_is_the_highest_probability_of_meeting_every_constraint():
    # finalists that validated at 0.954-0.975 against a 0.95 floor, with P(met) 0.56-0.75
    p = projection([Minimize("Words"), Maximize("Score")], [AtLeast("Score", 0.95)])
    views = [
        view(p, 0, {"Words": 40, "Score": 0.24}, [0.0]),  # the baseline
        view(p, 3, {"Words": 50, "Score": 0.96}, [0.61]),
        view(p, 7, {"Words": 55, "Score": 0.975}, [0.75]),
        view(p, 9, {"Words": 45, "Score": 0.954}, [0.56]),
    ]
    found = closest(p, CheapestOnFront(*p.objectives, within=0.05), views, views[0])
    assert [v.number for v in found.chosen] == [7] and found.by == "probability"
    assert found.closeness[7] == {"p_met": 0.75, "excess": 0.0}  # it meets the floor as measured, not at P 0.8
    assert found.closeness[0]["p_met"] == 0.0 and found.closeness[0]["excess"] == pytest.approx(0.71)
    described = found.describe()
    assert described["rule"] == "closest" and described["resolution"] == P_RESOLUTION
    assert set(described["closeness"]) == {"0", "3", "7", "9"}


def test_every_constraint_counts_its_probabilities_multiply():
    p = projection([Maximize("Score")], [AtLeast("Score", 0.9), AtLeast("Polite", 0.9)])
    a = view(p, 1, {"Score": 0.95, "Polite": 0.5}, [0.9, 0.3])  # 0.27
    b = view(p, 2, {"Score": 0.85, "Polite": 0.85}, [0.6, 0.6])  # 0.36
    found = closest(p, Pareto(*p.objectives), [a, b], a)
    assert [v.number for v in found.chosen] == [2] and found.closeness[2]["p_met"] == pytest.approx(0.36)


def test_probabilities_equal_to_two_decimals_tie_and_the_least_excess_decides():
    p = projection([Maximize("Score")], [AtLeast("Score", 0.9)])
    a = view(p, 1, {"Score": 0.7}, [0.741])  # misses by 0.2
    b = view(p, 2, {"Score": 0.8}, [0.744])  # misses by 0.1: the same probability to two decimals
    assert [v.number for v in closest(p, Pareto(*p.objectives), [a, b], a).chosen] == [2]
    c = view(p, 3, {"Score": 0.6}, [0.75])  # higher to two decimals: the probability decides first
    assert [v.number for v in closest(p, Pareto(*p.objectives), [a, b, c], a).chosen] == [3]


def test_when_nothing_has_a_chance_the_least_excess_decides_not_the_objective():
    # a budget goal: the most reward within 100 words; everything is far over it, with no chance of meeting it
    p = projection([Maximize("Score")], [AtMost("Words", 100)])
    a = view(p, 1, {"Words": 300, "Score": 0.9}, [0.0])
    b = view(p, 2, {"Words": 150, "Score": 0.4}, [0.0])
    assert [v.number for v in closest(p, Pareto(*p.objectives), [a, b], a).chosen] == [2]  # nearest the budget


def test_still_tied_the_combinations_own_order_decides():
    # equally close to the politeness floor: CheapestOnFront's near-tie rule (cost within 5%, the higher value)
    p = projection([Minimize("Words"), Maximize("Score")], [AtLeast("Polite", 0.9)])
    a = view(p, 1, {"Words": 100, "Score": 0.5, "Polite": 0.6}, [0.3])
    b = view(p, 2, {"Words": 103, "Score": 0.8, "Polite": 0.6}, [0.3])
    c = view(p, 3, {"Words": 90, "Score": 0.3, "Polite": 0.5}, [0.1])
    assert [v.number for v in closest(p, CheapestOnFront(*p.objectives, within=0.05), [a, b, c], a).chosen] == [2]
    assert [v.number for v in closest(p, CheapestOnFront(*p.objectives, within=0.0), [a, b, c], a).chosen] == [1]
    # a Pareto set ships its front among the tied
    assert {v.number for v in closest(p, Pareto(*p.objectives), [a, b, c], a).chosen} == {1, 2}


def test_by_measured_means_the_least_excess_decides():
    p = projection([Maximize("Score")], [AtLeast("Score", 0.9)])
    a, b = view(p, 1, {"Score": 0.7}), view(p, 2, {"Score": 0.8})
    found = closest(p, Pareto(*p.objectives), [a, b], a)
    assert [v.number for v in found.chosen] == [2] and found.by == "excess"
    assert found.closeness[2] == {"p_met": None, "excess": pytest.approx(0.1)}
    assert found.describe()["resolution"] is None


def test_nothing_qualifies_without_constraints_or_eligible_candidates():
    p = projection([Maximize("Score")], [])
    assert closest(p, Pareto(*p.objectives), [view(p, 1, {"Score": 0.5})], view(p, 0, {"Score": 0.4})) is None
    p = projection([Maximize("Score")], [AtLeast("Score", 0.9)])
    v = view(p, 1, {"Score": 0.5})
    ineligible = View(v.candidate, v.metrics, v.violations, False, {}, ())
    assert closest(p, Pareto(*p.objectives), [ineligible], ineligible) is None


def test_the_setting():
    assert Selection().unmet == "closest" and Selection().describe()["unmet"] == "closest"
    assert Selection(unmet="none").describe()["unmet"] == "none"
    with pytest.raises(ValueError, match="unmet"):
        Selection(unmet="baseline")


# ---- in a study ------------------------------------------------------------------------------------------------------
class Shifted(Evaluator):
    """How long the output is, to 10 words (0.1 a word), plus ``bonus`` on the cases whose names start with ``on`` (by
    default the search cases, t...) and ``noise`` (seeded by the trial): the finalists meet a floor on the search cases
    and miss it on validation."""

    score_range = (0, 1)

    def __init__(self, bonus=0.15, noise=0.0, on="t"):
        super().__init__(name="long")
        self.bonus, self.noise, self.on = bonus, noise, on

    def evaluate(self, case):
        n = len(str(case.actual_output).split())
        shift = self.bonus if case.name.startswith(self.on) else 0.0
        jitter = random.Random(f"{case.name}/{case.actual_output}").gauss(0, self.noise) if self.noise else 0.0
        return [EvaluationOutput(score=min(1.0, max(0.0, n / 10 + shift + jitter)), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


class Words(Evaluator):
    score_range = (0, None)

    def __init__(self):
        super().__init__(name="words")

    def evaluate(self, case):
        return [EvaluationOutput(score=float(len(str(case.actual_output).split())), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


LENGTH = Int(1, 20, default=2)


@tunable(n=LENGTH)
def sayer(n):
    return Agent(model=Says(" ".join(["word"] * n)), callback_handler=None)


def scripted(values):
    from strands_evals.optimization import Method
    from strands_evals.optimization.methods import Draw

    class _Next(Draw):
        def __init__(self, value):
            self.value, self.origin = value, {"method": "scripted"}

        def suggest_int(self, name, low, high, **kw):
            return self.value

    class Scripted(Method):
        kinds = ("int",)

        def __init__(self):
            self.values = list(values)

        def begin(self, block, base, history, structure, study, objective):
            return _Next(self.values.pop(0))

    return Scripted()


def c3(unmet, record=None, log=None, on="t"):
    """C3 in small: the cheapest on the (words, long) front with long at least 0.94. On the search cases 8 and 9 words
    meet it (0.95, 1.0) and 10 is dominated by 9; on validation they score 0.8 and 0.9 and the baseline's 2 words 0.2,
    so nothing meets it there (with ``on=""`` validation scores as the search cases do, and 8 and 9 meet it)."""
    train = [Case(name=f"t{k}", input="say") for k in range(2)]
    validation, test = [Case(name="v0", input="say")], [Case(name="x0", input="say")]
    return Study(
        agent=sayer,
        experiment=Experiment(cases=train + validation + test, evaluators=[Words(), Shifted(on=on)]),
        objectives=CheapestOnFront(Minimize("words"), Maximize("long"), within=0.25),
        constraints=[AtLeast("long", 0.94)],
        search=Search(width=1, blocks=[Block(scripted([8, 9, 10]), LENGTH)]),
        split=Split(train=train, validation=validation, test=test),
        selection=Selection(by="measured", finalists=3, unmet=unmet),
        record=record,
        log=log,
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_when_no_finalist_meets_the_floor_the_closest_ships_unconfirmed(tmp_path):
    lines = []
    study = c3("closest", record=tmp_path, log=lines.append)
    result = study.optimize(iterations=3)
    by_n = {c.params["n"]: c.number for c in study.candidates}
    assert [c.number for c in study.validated] == [0, by_n[9], by_n[8]]  # the front, as ranked (10 is dominated)
    assert not any(v.feasible for v in study.validated)
    assert [c.number for c in result.selected] == [by_n[9]]  # misses the floor by 0.04 on validation; 8 by 0.14
    assert study.unconfirmed.by == "excess" and set(study.unconfirmed.closeness) == {0, by_n[8], by_n[9]}
    # the record
    selected = next(e for e in events(tmp_path / "study.jsonl") if e["type"] == "selected")
    assert selected["best"] == [by_n[9]] and selected["unconfirmed"] is True and selected["rule"]["unmet"] == "closest"
    assert selected["unmet"]["closeness"][str(by_n[9])]["excess"] == pytest.approx(0.04)
    finished = next(e for e in events(tmp_path / "study.jsonl") if e["type"] == "finished")
    assert finished["best"] == [by_n[9]] and finished["unconfirmed"] is True
    # the test: of the closest, against the baseline, its arm marked
    arms = [e for e in events(tmp_path / "test.jsonl") if e["type"] == "tested"]
    assert [(a["role"], a["number"], a["unconfirmed"]) for a in arms] == [
        ("baseline", 0, False),
        ("selected", by_n[9], True),
    ]
    assert result.test.selected[0].unconfirmed and result.test.difference["long"].mean == pytest.approx(0.7)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["unconfirmed"] == [1] and saved["runs"][0]["unconfirmed"] is True
    # the log says so
    assert any("UNCONFIRMED" in line and f"candidate {by_n[9]}" in line for line in lines)
    assert any(line.startswith("test") and "(selected, unconfirmed)" in line for line in lines)


def test_unmet_none_ships_nothing_as_before(tmp_path):
    study = c3("none", record=tmp_path)
    result = study.optimize(iterations=3)
    assert result.selected == [] and study.unconfirmed is None
    selected = next(e for e in events(tmp_path / "study.jsonl") if e["type"] == "selected")
    assert selected["best"] == [] and selected["unconfirmed"] is False and selected["unmet"] is None
    arms = [e for e in events(tmp_path / "test.jsonl") if e["type"] == "tested"]
    roles = [(a["role"], a["number"], a["unconfirmed"]) for a in arms]
    assert roles == [("baseline", 0, False), ("fallback", 0, False)]
    assert json.loads((tmp_path / "result.json").read_text())["unconfirmed"] == []


def test_a_selection_that_meets_the_constraints_is_unchanged():
    # without the validation shift, 8 and 9 words meet the floor on validation too: the near-tie rule picks 9
    study = c3("closest", on="")
    result = study.optimize(iterations=3)
    by_n = {c.params["n"]: c.number for c in study.candidates}
    assert [c.number for c in result.selected] == [by_n[9]] and study.unconfirmed is None
    assert not result.test.selected[0].unconfirmed and result.runs[0].unconfirmed is False


@needs_bayesbench
def test_by_the_model_the_closest_is_the_highest_probability(tmp_path):
    # noisy scores, decided by the model at P 0.8: the floor (0.85) is above every candidate's validation mean
    train = [Case(name=f"t{k}", input="say") for k in range(4)]
    validation = [Case(name=f"v{k}", input="say") for k in range(3)]
    test = [Case(name="x0", input="say")]
    study = Study(
        agent=sayer,
        experiment=Experiment(cases=train + validation + test, evaluators=[Shifted(bonus=0.0, noise=0.05)]),
        objectives=Maximize("long"),
        constraints=[AtLeast("long", 0.85)],
        search=Search(width=1, blocks=[Block(scripted([5, 6, 7, 8]), LENGTH)]),
        repetitions=2,
        split=Split(train=train, validation=validation, test=test),
        selection=Selection(feasibility=0.8, finalists=3),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=4)
    found = study.unconfirmed
    assert found is not None and found.by == "probability"
    assert set(found.closeness) == {c.number for c in study.validated}
    p = {n: c["p_met"] for n, c in found.closeness.items()}
    assert all(q < 0.8 for q in p.values())  # none met the floor at the required probability
    (chosen,) = result.selected
    assert round(p[chosen.number], 2) == max(round(q, 2) for q in p.values())
    selected = next(e for e in events(tmp_path / "study.jsonl") if e["type"] == "selected")
    assert selected["by"] == "model" and selected["unconfirmed"] is True
    assert {n: d["p_met"] for n, d in selected["decided"].items()} == {str(n): [q] for n, q in p.items()}


def test_a_rotated_result_names_its_unconfirmed_runs():
    runs = [Run(1, {}, [], None), Run(2, {}, [], None, unconfirmed=True), Run(3, {}, [], None)]
    result = Result(runs=runs, test=None)
    assert result.unconfirmed == [2] and result.to_dict()["unconfirmed"] == [2]
    assert [r["unconfirmed"] for r in result.to_dict()["runs"]] == [False, True, False]
