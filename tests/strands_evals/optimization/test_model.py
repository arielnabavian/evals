"""The model of each goal metric (model.py) on data with known effects, and deciding and selecting with it."""

import random
from dataclasses import dataclass, field

import numpy as np
import pytest
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtLeast,
    Block,
    Choice,
    Maximize,
    Minimize,
    OneOf,
    Search,
    Selection,
    Study,
    Thompson,
    component,
    tunable,
)
from strands_evals.optimization import model as M
from strands_evals.optimization.projection import Projection
from strands_evals.optimization.space.space import ParamInfo
from strands_evals.optimization.trials import Trial, TrialStatus
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says, needs_bayesbench

# ---- the model alone, on candidates with known effects ---------------------------------------------------------------
SPACE = {
    "option": ParamInfo(key="option", type="choice", default="a", choices=("a", "b", "c")),
    "level": ParamInfo(key="level", type="int", default=5, low=1, high=10),
    "text": ParamInfo(key="text", type="text", default="default"),
}
OPTION = {"a": 0.0, "b": 0.08, "c": -0.06}


@dataclass
class _Candidate:
    number: int
    params: dict
    trials: tuple
    eligible: bool = True
    estimates: dict = field(default_factory=dict)

    @property
    def metrics(self):
        return {}


def _truth(params, text_effects):
    return 0.5 + OPTION[params["option"]] + 0.005 * params["level"] + text_effects.get(params["text"], 0.0)


def _world(seed, *, n=40, cases=20, repetitions=3, noise=0.15, text_spread=0.0, options=True):
    """Candidates at random configurations, each measured on the same cases; the truth of each one's mean over them."""
    rng = random.Random(seed)
    case_effects = {f"case{k}": rng.gauss(0, 0.1) for k in range(cases)}
    text_effects = {}
    candidates, truths = [], {}
    for i in range(n):
        if i == 0:
            params = {"option": "a", "level": 5, "text": "default"}
        else:
            params = {
                "option": rng.choice("abc") if options else "a",
                "level": rng.randint(1, 10),
                "text": f"rewrite {i}" if text_spread is not None else "default",
            }
            text_effects[params["text"]] = rng.gauss(0, text_spread) if text_spread else 0.0
        trials = []
        for case, offset in case_effects.items():
            for r in range(1, repetitions + 1):
                value = _truth(params, text_effects) + offset + rng.gauss(0, noise)
                trials.append(Trial(case=case, repetition=r, status=TrialStatus.COMPLETED, metrics={"score": value}))
        candidates.append(_Candidate(i, params, tuple(trials)))
        truths[i] = _truth(params, text_effects) + sum(case_effects.values()) / cases
    return candidates, truths, list(case_effects)


PROJECTION = Projection([Maximize("score")], [], evaluators={"score": (None, None)})


def _fit(candidates, cases, projection=PROJECTION):
    return M.fit(projection, SPACE, {"text": ["text"]}, candidates, candidates[0], cases)


def _measured(candidate):
    return float(np.mean([t.metrics["score"] for t in candidate.trials]))


@needs_bayesbench
def test_decided_intervals_cover_the_truth():
    covered, total = 0, 0
    for seed in range(12):
        candidates, truths, cases = _world(seed, text_spread=0.03)
        fit = _fit(candidates, cases)
        for c in candidates:
            lo, hi = fit.metrics["score"].at(c.params, c.number).ci90
            covered += lo <= truths[c.number] <= hi
            total += 1
    assert 0.8 <= covered / total <= 0.98  # 90% intervals


@needs_bayesbench
def test_a_candidate_measured_lucky_is_shrunk_toward_what_the_others_say():
    # no rewrite does anything: the best measured rewrite is only the luckiest
    closer = 0
    for seed in range(12):
        candidates, truths, cases = _world(seed, text_spread=0.0, options=False)
        fit = _fit(candidates, cases)
        luckiest = max(candidates[1:], key=_measured)
        decided = fit.metrics["score"].at(luckiest.params, luckiest.number).mean
        closer += abs(decided - truths[luckiest.number]) < abs(_measured(luckiest) - truths[luckiest.number])
    assert closer >= 10


@needs_bayesbench
def test_the_model_selects_truly_better_candidates_than_the_highest_measured_mean():
    gains = []
    for seed in range(16):
        candidates, truths, cases = _world(seed, repetitions=1, noise=0.25, text_spread=0.02)
        fit = _fit(candidates, cases)
        by_model = max(candidates, key=lambda c: fit.metrics["score"].at(c.params, c.number).mean)
        by_measure = max(candidates, key=_measured)
        gains.append(truths[by_model.number] - truths[by_measure.number])
    assert np.mean(gains) > 0


@needs_bayesbench
def test_a_deterministic_metric_is_decided_as_measured():
    candidates, truths, cases = _world(3, repetitions=2, noise=0.0, text_spread=0.03)
    fit = _fit(candidates, cases)
    for c in candidates:
        assert fit.metrics["score"].at(c.params, c.number).mean == pytest.approx(_measured(c), abs=1e-3)


@needs_bayesbench
def test_effects_recover_an_option_effect():
    candidates, truths, cases = _world(5, n=60, text_spread=0.0)
    fit = _fit(candidates, cases)
    effects = {e["level"]: e for e in M.effects(fit, [("score", "higher")]) if e["key"] == "option"}
    for level in ("b", "c"):
        lo, hi = effects[level]["ci90"]
        assert lo <= OPTION[level] <= hi
    assert effects["b"]["p_positive"] > 0.95 and effects["c"]["p_positive"] < 0.05


@needs_bayesbench
def test_a_constraint_is_met_with_a_probability_that_is_about_right():
    met_when_sure, sure = 0, 0
    for seed in range(10):
        candidates, truths, cases = _world(seed, text_spread=0.03)
        fit = _fit(candidates, cases)
        limit = float(np.median(list(truths.values())))
        for c in candidates:
            p = M.p_met(AtLeast("score", limit), fit.metrics["score"].at(c.params, c.number))
            if p >= 0.9:
                sure += 1
                met_when_sure += truths[c.number] >= limit
    assert sure and met_when_sure / sure >= 0.85


# ---- deciding and selecting in a study -------------------------------------------------------------------------------
class Noisy(Evaluator):
    """Scores the number of words plus noise (each call its own draw, from a seeded generator)."""

    score_range = (0, None)

    def __init__(self, sd, seed=0):
        super().__init__(name="Noisy")
        self.sd, self.rng = sd, random.Random(seed)

    def evaluate(self, case):
        n = len(str(case.actual_output).split())
        return [EvaluationOutput(score=max(n + self.rng.gauss(0, self.sd), 0.0), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def experiment(sd=0.0, seed=0, cases=4):
    return Experiment(cases=[Case(name=f"c{k}", input="write") for k in range(cases)], evaluators=[Noisy(sd, seed)])


STANZAS = Choice([1, 2, 3, 4, 5, 6], default=2)


@tunable(stanzas=STANZAS)
def poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def grid_search(values):
    from strands_evals.optimization import Method
    from strands_evals.optimization.methods import _ChoiceDraw

    class _Grid(Method):
        kinds = ("choice",)

        def __init__(self):
            self.values = list(values)

        def begin(self, block, base, history, structure, study, objective):
            return _ChoiceDraw({"stanzas": self.values.pop(0)}, {"method": "grid"})

    return Search(width=1, blocks=[Block(_Grid(), STANZAS)])


@needs_bayesbench
def test_the_finalists_are_measured_again_before_the_selection(tmp_path):
    study = Study(
        agent=poet,
        experiment=experiment(sd=5.0, seed=1),
        objectives=Maximize("Noisy"),
        search=grid_search([1, 3, 4, 6]),
        repetitions=2,
        selection=Selection(finalists=2),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=4)
    remeasured = [c for c in study.candidates if len(c.trials) > 2 * 4]
    assert len(remeasured) == 2 and all(len(c.trials) == (2 + 2) * 4 for c in remeasured)
    assert study.best_candidate.params == {"stanzas": 6}
    import json

    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    assert {e["number"] for e in events if e["type"] == "remeasured"} == {c.number for c in remeasured}
    selected = next(e for e in events if e["type"] == "selected")
    assert selected["best"] == [study.best_candidate.number] and selected["decided"]


@needs_bayesbench
def test_a_clear_leader_is_selected_when_its_interval_overlaps_the_baselines():
    # 3 stanzas score 10 more than the baseline's 2; the noise is wide enough that their intervals overlap, but the
    # model decides 3 stanzas are better, so they are selected
    study = Study(
        agent=poet,
        experiment=experiment(sd=SEED_SD, seed=SEED),
        objectives=Maximize("Noisy"),
        search=grid_search([3]),
        repetitions=2,
        selection=Selection(finalists=0),
        log=None,
    )
    study.optimize(iterations=1)
    base, three = (study.view(c).decided["Noisy"] for c in study.candidates)
    assert three.mean > base.mean and three.ci90[0] <= base.ci90[1]  # a clear leader, overlapping intervals
    assert study.best_candidate.params == {"stanzas": 3}


SEED, SEED_SD = 0, 12.0


def test_a_constraint_relative_to_the_baseline():
    study = Study(
        agent=poet,
        experiment=experiment(),
        objectives=Minimize("output_tokens"),
        constraints=[AtLeast("Noisy", -5, compare="difference")],  # at most 5 words fewer than the baseline's 20
        search=grid_search([1, 3]),
        selection=Selection(by="measured"),
        log=None,
    )
    study.optimize(iterations=2)
    by_stanzas = {c.params["stanzas"]: c for c in study.candidates}
    assert by_stanzas[1].violations == (5.0,) and by_stanzas[2].violations == (-5.0,)
    assert study.best_candidate.params == {"stanzas": 2}


def test_lower_bound_decides_with_the_worse_side():
    from strands_evals.optimization.model import Decided
    from strands_evals.optimization.selection import LowerBound, Mean

    d = Decided(mean=1.0, sd=0.5)
    assert Mean().value(d, True) == 1.0
    assert LowerBound(0.1).value(d, True) < 1.0 < LowerBound(0.1).value(d, False)


# ---- Thompson --------------------------------------------------------------------------------------------------------
STYLE = OneOf(component(lambda: 1), component(lambda: 3), component(lambda: 6), labels=["short", "medium", "long"])


@tunable(style=STYLE)
def styled(style):
    return Agent(model=Says(" ".join(["word"] * 10 * style)), callback_handler=None)


@needs_bayesbench
def test_thompson_tries_an_option_not_yet_measured_then_prefers_the_best():
    study = Study(
        agent=styled,
        experiment=experiment(sd=2.0, seed=3),
        objectives=Maximize("Noisy"),
        search=Search(width=1, blocks=[Block(Thompson(seed=4), STYLE)]),
        selection=Selection(finalists=0),
        log=None,
    )
    study.optimize(iterations=6)
    first = study.candidates[1]
    assert first.params["style"] != "short" and first.origin["structure"]["by"] == "unmeasured option"
    assert study.best_candidate.params == {"style": "long"}
    # with a model, it proposes by posterior draws; an option far behind is not tried again
    later = [c.origin["structure"]["by"] for c in study.candidates[2:]]
    assert all(by == "posterior draw" for by in later)


# ---- validation and folds ------------------------------------------------------------------------------------------
def test_the_selection_is_made_on_the_validation_cases(tmp_path):
    import json

    from strands_evals.optimization import Split

    cases = [Case(name=f"c{k}", input="write") for k in range(4)]
    validation = [Case(name=f"v{k}", input="write") for k in range(3)]
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases + validation, evaluators=[Noisy(4.0, 5)]),
        objectives=Maximize("Noisy"),
        search=grid_search([1, 3, 5]),
        repetitions=2,
        selection=Selection(finalists=2),
        split=Split(train=cases, validation=validation),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=3)
    assert len(study.validated) == 3 and study.validated[0].number == 0  # the baseline and the two finalists
    assert all({t.case for t in c.trials} == {"v0", "v1", "v2"} and len(c.trials) == 6 for c in study.validated)
    assert all(len(c.trials) == 2 * 4 for c in study.candidates)  # the search measurements are not touched
    assert study.best_candidate.params == {"stanzas": 5}
    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    assert events[0]["validation"] == ["v0", "v1", "v2"] and events[0]["cases"] == ["c0", "c1", "c2", "c3"]
    assert [e["number"] for e in events if e["type"] == "validated"] == [c.number for c in study.validated]
    selected = next(e for e in events if e["type"] == "selected")
    assert selected["on"] == "validation" and selected["best"] == [study.best_candidate.number]
    assert set(selected["decided"]) == {str(c.number) for c in study.validated}


def test_a_constraint_on_the_ratio_to_the_baseline():
    from strands_evals.optimization import AtMost

    for selection in (Selection(by="measured"), Selection(finalists=0)):
        study = Study(
            agent=poet,
            experiment=experiment(sd=0.5, seed=6),
            objectives=Maximize("Noisy"),
            constraints=[AtMost("output_tokens", 1.6, compare="ratio")],  # at most 1.6 times the baseline's tokens
            search=grid_search([1, 3, 4]),
            selection=selection,
            log=None,
        )
        study.optimize(iterations=3)
        by_stanzas = {c.params["stanzas"]: study.view(c) for c in study.candidates}
        assert by_stanzas[3].feasible and not by_stanzas[4].feasible  # 1.5 and 2 times the baseline's 2 stanzas
        assert study.best_candidate.params == {"stanzas": 3}


def test_decided_values_closer_than_their_resolution_tie():
    from strands_evals.optimization.selection import resolved

    assert resolved(0.4999960020499944, 0.000707) == resolved(0.49999999999941763, 0.000707) == 0.5
    assert resolved(0.52, 0.000707) != resolved(0.5, 0.000707)
    assert resolved(3.25, 0.0) == 3.25


@needs_bayesbench
def test_a_configuration_not_fitted_is_predicted_with_its_own_effect_unknown():
    candidates, truths, cases = _world(2, text_spread=0.05)
    fit = _fit(candidates, cases)
    model = fit.metrics["score"]
    c = candidates[5]
    assert model.candidate_sd > 0
    assert model.at(c.params).sd > model.at(c.params, c.number).sd  # a configuration fitted knows its own effect
    assert model.at(candidates[0].params, 0).sd < model.at(candidates[0].params).sd


@needs_bayesbench
def test_a_number_is_predicted_along_its_trend_within_its_bin():
    rng = random.Random(7)
    space = {"level": ParamInfo(key="level", type="int", default=5, low=1, high=100)}
    cases = [f"case{k}" for k in range(6)]
    candidates = []
    for i, level in enumerate([5, *rng.sample(range(1, 101), 30)]):
        trials = tuple(
            Trial(case=case, repetition=1, status=TrialStatus.COMPLETED, metrics={"score": 0.2 + 0.005 * level})
            for case in cases
        )
        candidates.append(_Candidate(i, {"level": level}, trials))
    fit = M.fit(PROJECTION, space, {}, candidates, candidates[0], cases)
    for level in (3, 50, 97):  # values never measured
        assert fit.metrics["score"].at({"level": level}).mean == pytest.approx(0.2 + 0.005 * level, abs=0.02)


def test_a_baseline_whose_every_trial_fails_stops_the_study():
    @tunable(stanzas=STANZAS)
    def broken(stanzas):
        return Agent(model=Says("never", fail=ValueError("bad request: thinking.type.enabled")), callback_handler=None)

    study = Study(search=Search(width=1), agent=broken, experiment=experiment(), objectives=Maximize("Noisy"), log=None)
    with pytest.raises(RuntimeError, match="every trial of the baseline .* failed: .*thinking.type.enabled"):
        study.optimize(iterations=1)
