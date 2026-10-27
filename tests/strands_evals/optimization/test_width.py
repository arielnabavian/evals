"""Width: each step proposes Search.width candidates from the same incumbent and measures them together; iterations and
patience count steps (the search's depth)."""

import json
import threading
import time
import warnings

import pytest
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import TPE, Block, Choice, Int, Maximize, Method, Search, Selection, Study, tunable
from strands_evals.optimization.methods import _ChoiceDraw
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says


class Words(Evaluator):
    score_range = (0, None)

    def __init__(self):
        super().__init__(name="words")

    def evaluate(self, case):
        return [EvaluationOutput(score=len(str(case.actual_output).split()), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


STANZAS = Choice([1, 2, 3, 4, 5, 6, 7, 8], default=1)


@tunable(stanzas=STANZAS)
def poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def cases(n=2):
    return [Case(name=f"c{k}", input="write") for k in range(n)]


class Counting(Method):
    """Proposes the next value of ``values`` each time; records how many proposals overlap in time."""

    kinds = ("choice",)

    def __init__(self, values, delay=0.0, concurrent=False):
        self.values, self.delay, self.concurrent = list(values), delay, concurrent
        self.lock, self.active, self.most, self.observed = threading.Lock(), 0, 0, []

    def begin(self, block, base, history, structure, study, objective):
        with self.lock:
            self.active += 1
            self.most = max(self.most, self.active)
            value = self.values.pop(0)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return _ChoiceDraw({"stanzas": value}, {"method": "counting", "base": base.number if base else None})

    def observe(self, block, candidate, draw, study, objective):
        if draw is not None:
            self.observed.append((draw.values["stanzas"], candidate.number if candidate else None))


def test_a_step_proposes_width_candidates_from_the_same_incumbent_and_measures_them_together(tmp_path):
    method = Counting([2, 3, 4, 5, 6, 7])
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases(), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(method, STANZAS)], width=3),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=2)  # two steps of three
    steps = [(c.step, c.slot, c.base, c.params["stanzas"]) for c in study.candidates[1:]]
    assert steps == [(0, 0, 0, 2), (0, 1, 0, 3), (0, 2, 0, 4), (1, 0, 3, 5), (1, 1, 3, 6), (1, 2, 3, 7)]
    assert study.best_candidate.params == {"stanzas": 7}
    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    assert events[0]["width"] == 3
    assert [e["slot"] for e in events if e["type"] == "candidate"][1:] == [0, 1, 2, 0, 1, 2]
    assert sorted(method.observed) == [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]


WIDE = Choice(list(range(1, 13)), default=1)


@tunable(stanzas=WIDE)
def wide_poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def test_the_model_is_refitted_once_a_step_and_patience_counts_steps(monkeypatch):
    # a new best in step 0 (12), then two steps that do not change it: patience 2 ends the run after them (counted in
    # candidates, it would end within the second step)
    method = Counting([10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9, 1])
    study = Study(
        agent=wide_poet,
        experiment=Experiment(cases=cases(), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(method, WIDE)], width=3),
        selection=Selection(finalists=0),
        log=None,
    )
    fits = []
    real = study._refit
    monkeypatch.setattr(study, "_refit", lambda: (fits.append(len(study.candidates)), real())[1])
    study.optimize(iterations=10, patience=2)
    assert [c.params["stanzas"] for c in study.candidates] == [1, 10, 11, 12, 2, 3, 4, 5, 6, 7]
    assert fits == [1, 4, 7, 10]  # once for the baseline and once a step


def test_identical_proposals_in_a_step_are_measured_once_and_each_is_told_the_result():
    method = Counting([4, 4, 4])
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases(), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(method, STANZAS)], width=3, redraws=0),
        log=None,
    )
    study.optimize(iterations=1)
    assert len(study.candidates) == 2
    assert method.observed == [(4, 1), (4, 1), (4, 1)]


def test_a_concurrent_method_proposes_in_parallel_and_any_other_one_at_a_time():
    slow = Counting([2, 3, 4, 5], delay=0.3, concurrent=True)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases(), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(slow, STANZAS)], width=4),
        log=None,
    )
    study.optimize(iterations=1)
    assert slow.most == 4  # the four rewrites at once

    locked = Counting([2, 3, 4, 5], delay=0.05)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases(), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(locked, STANZAS)], width=4),
        log=None,
    )
    study.optimize(iterations=1)
    assert locked.most == 1 and len(study.candidates) == 5


N = Int(1, 200, default=1)


@tunable(n=N)
def counter(n):
    return Agent(model=Says(" ".join(["word"] * n)), callback_handler=None)


def test_tpe_proposes_a_step_of_distinct_values():
    study = Study(
        agent=counter,
        experiment=Experiment(cases=cases(1), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(blocks=[Block(TPE(seed=1, n_startup_trials=2), N)], width=4),
        log=None,
    )
    study.optimize(iterations=3)
    for step in range(3):
        values = [c.params["n"] for c in study.candidates if c.step == step]
        assert len(values) == len(set(values)) >= 3


def test_width_must_be_at_least_one():
    with pytest.raises(ValueError, match="width must be at least 1"):
        Search(width=0)


def test_a_step_that_cannot_run_at_once_warns_that_width_will_not_save_time():
    def study(max_concurrency):
        return Study(
            agent=poet,
            experiment=Experiment(cases=cases(), evaluators=[Words()]),
            objectives=Maximize("words"),
            search=Search(blocks=[Block(Counting([2, 3, 4, 5]), STANZAS)], width=4),
            repetitions=2,
            max_concurrency=max_concurrency,
            log=None,
        )

    with pytest.warns(UserWarning, match=r"4 x 4 trials .* max_concurrency is 15.*max_concurrency=16"):
        study(15).optimize(iterations=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study(16).optimize(iterations=1)
