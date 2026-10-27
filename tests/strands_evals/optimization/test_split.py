"""Split: which cases each part of a study uses; the test step; rotation (every case tested once) and its pooling."""

import asyncio
import json
import random
import threading
import time

import pytest
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtLeast,
    Block,
    CheapestOnFront,
    Choice,
    Int,
    Maximize,
    Minimize,
    Pareto,
    Search,
    Selection,
    Split,
    Study,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says


def cases(n, prefix="c"):
    return [Case(name=f"{prefix}{k}", input="write") for k in range(n)]


# ---- dealing cases ---------------------------------------------------------------------------------------------------
def test_shares_deal_every_case_once_independent_of_order():
    pool = cases(10)
    (a,) = Split(train=0.5, validation=0.2, test=0.3, seed=1).assign(pool)
    (b,) = Split(train=0.5, validation=0.2, test=0.3, seed=1).assign(list(reversed(pool)))
    assert (len(a.train), len(a.validation), len(a.test)) == (5, 2, 3)
    assert a.names() == b.names()
    assert sorted(a.names()["train"] + a.names()["validation"] + a.names()["test"]) == sorted(c.name for c in pool)


def test_rotation_tests_every_case_exactly_once_and_never_trains_or_validates_on_it():
    pool = cases(13)
    runs = Split(train=0.5, validation=0.25, test=0.25, rotate=True, seed=3).assign(pool)
    assert len(runs) == 4
    tested = [c.name for r in runs for c in r.test]
    assert sorted(tested) == sorted(c.name for c in pool)  # every case once
    for r in runs:
        names = r.names()
        assert not set(names["test"]) & (set(names["train"]) | set(names["validation"]))
        assert len(names["train"]) + len(names["validation"]) + len(names["test"]) == 13
    validated = [c.name for r in runs for c in r.validation]  # a whole part each: validation rotates too
    assert sorted(validated) == sorted(c.name for c in pool)


@pytest.mark.parametrize(
    "split, message",
    [
        (lambda p: Split(train=0.5, validation=p[:2]), "not a mix"),
        (lambda p: Split(train=0.5, validation=0.2, test=0.2), "sum to 1"),
        (lambda p: Split(train=0.6, validation=0.1, test=0.3, rotate=True), "whole number"),
        (lambda p: Split(train=p[:5], test=p[5:], rotate=True), "needs shares"),
        (lambda p: Split(train=p[:6], test=p[5:]).assign(p), "more than one role"),
        (lambda p: Split(train=p[:5], test=p[5:9]).assign(p), "have no role"),
        (lambda p: Split(train=p, test=cases(1, "x")).assign(p), "not in the experiment"),
        (lambda p: Split(train=0.0, test=1.0), "train must be more than 0"),
    ],
)
def test_a_split_that_cannot_be_dealt_says_why(split, message):
    with pytest.raises(ValueError, match=message):
        split(cases(10))


# ---- studies ---------------------------------------------------------------------------------------------------------
class Words(Evaluator):
    """The number of words, plus noise seeded by the trial (its case and output), whatever order the trials run in."""

    score_range = (0, None)

    def __init__(self, sd=0.0, seed=0):
        super().__init__(name="words")
        self.sd, self.seed = sd, seed

    def evaluate(self, case):
        n = len(str(case.actual_output).split())
        noise = random.Random(f"{self.seed}/{case.name}/{case.actual_output}").gauss(0, self.sd)
        return [EvaluationOutput(score=max(n + noise, 0.0), test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


STANZAS = Choice([1, 2, 3, 4], default=2)


@tunable(stanzas=STANZAS)
def poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def grid(values):
    from strands_evals.optimization import Method
    from strands_evals.optimization.methods import _ChoiceDraw

    class Grid(Method):
        kinds = ("choice",)

        def __init__(self):
            self.values, self.seen = list(values), []

        def begin(self, block, base, history, structure, study, objective):
            self.seen.append(len(study.candidates))
            return _ChoiceDraw({"stanzas": self.values.pop(0)}, {"method": "grid"})

    return Grid()


def test_a_split_searches_on_train_selects_on_validation_and_tests_the_selection(tmp_path):
    train, validation, test = cases(4, "t"), cases(3, "v"), cases(5, "x")
    study = Study(
        agent=poet,
        experiment=Experiment(cases=train + validation + test, evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(grid([4]), STANZAS)]),
        repetitions=2,
        split=Split(train=train, validation=validation, test=test),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=1)
    assert {t.case for c in study.candidates for t in c.trials} == {c.name for c in train}  # search: train only
    assert {t.case for c in study.validated for t in c.trials} == {c.name for c in validation}
    assert [c.params for c in result.selected] == [{"stanzas": 4}]
    diff = result.test.difference["words"]
    assert diff.mean == pytest.approx(20.0) and diff.cases == 5  # 40 words against the baseline's 20, per test case
    assert result.test.selected[0].ratio["words"].mean == pytest.approx(2.0)
    assert {t.case for t in result.test.baseline.trials} == {c.name for c in test}
    lines = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    assert [e["type"] for e in lines] == ["test", "tested", "tested", "test_end"]
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["tested"] and saved["test"]["selected"][0]["difference"]["words"]["mean"] == pytest.approx(20.0)
    with pytest.raises(RuntimeError, match="measured its test cases"):
        study.optimize(iterations=1)


def test_when_the_baseline_is_selected_its_test_difference_is_exactly_zero():
    pool = cases(6)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Words()]),
        objectives=Minimize("words"),  # the baseline's 20 words beat 40
        search=Search(width=1, blocks=[Block(grid([4]), STANZAS)]),
        split=Split(train=pool[:3], validation=pool[3:4], test=pool[4:]),
        log=None,
    )
    result = study.optimize(iterations=1)
    assert [c.number for c in result.selected] == [0]
    assert result.test.difference["words"].mean == 0.0


class Flat(Evaluator):
    """The same score for every output: every configuration ties."""

    score_range = (0, 1)

    def __init__(self):
        super().__init__(name="flat")

    def evaluate(self, case):
        return [EvaluationOutput(score=0.0, test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def test_a_tie_on_validation_selects_the_baseline_alone():
    pool = cases(6)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Flat()]),
        objectives=Maximize("flat"),
        search=Search(width=1, blocks=[Block(grid([4]), STANZAS)]),
        split=Split(train=pool[:3], validation=pool[3:5], test=pool[5:]),
        log=None,
    )
    result = study.optimize(iterations=1)
    assert [c.number for c in study.validated] == [0, 1]
    assert [c.number for c in result.selected] == [0]
    assert [a.role for a in result.test.selected] == ["selected"] and result.test.selected[0].number == 0


def test_rotation_runs_fresh_searches_and_pools_every_case(tmp_path):
    pool = cases(8)
    method = grid([4, 4, 4, 4])  # one per run: each run copies the method as given, untouched by the others
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Words(sd=2.0, seed=3)]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(method, STANZAS)]),
        split=Split(train=0.5, validation=0.25, test=0.25, rotate=True, seed=2),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=1)
    assert len(result.runs) == 4 and result.selected is None
    assert sorted(case for r in result.runs for case in r.test.cases) == sorted(c.name for c in pool)
    assert all(len(r.selected) == 1 and r.selected[0].params == {"stanzas": 4} for r in result.runs)
    assert method.values == [4, 4, 4, 4]  # the given method itself never ran: each run had its own copy
    pooled = result.test
    assert pooled.cases == 8
    lo, hi = pooled.difference["words"].ci90
    assert lo <= 20.0 <= hi  # 40 words against 20, with noise
    lo, hi = pooled.procedure["words"].ci90
    assert lo <= pooled.procedure["words"].mean <= hi
    plan = json.loads((tmp_path / "split.json").read_text())
    assert plan["split"]["runs"] == 4 and len(plan["runs"]) == 4
    assert all((tmp_path / f"run-{i}" / "study.jsonl").exists() for i in range(1, 5))
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["test"]["runs"] == 4 and saved["test"]["cases"] == 8


def test_a_pareto_goal_under_rotation_reports_each_run_and_pools_nothing():
    pool = cases(8)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Words()]),
        objectives=Pareto(Maximize("words"), Minimize("output_tokens")),
        search=Search(width=1, blocks=[Block(grid([4, 4, 4, 4]), STANZAS)]),
        split=Split(train=0.5, validation=0.25, test=0.25, rotate=True),
        log=None,
    )
    result = study.optimize(iterations=1)
    assert result.test is None and all(r.test is not None for r in result.runs)
    assert result.to_dict()["tested"]


def test_validation_cases_need_finalists():
    pool = cases(6)
    with pytest.raises(ValueError, match="finalists=0"):
        Study(
            search=Search(width=1),
            agent=poet,
            experiment=Experiment(cases=pool, evaluators=[Words()]),
            objectives=Maximize("words"),
            selection=Selection(finalists=0),
            split=Split(train=pool[:4], validation=pool[4:]),
            log=None,
        )


def test_a_tested_configuration_whose_every_trial_fails_stops_the_study():
    @tunable(stanzas=STANZAS)
    def fragile(stanzas):
        fail = ValueError("too long") if stanzas == 4 else None
        return Agent(model=Says(" ".join(["word"] * 10 * stanzas), fail=fail), callback_handler=None)

    pool = cases(6)
    study = Study(
        agent=fragile,
        experiment=Experiment(cases=pool, evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(grid([4]), STANZAS)]),
        split=Split(train=pool[:3], test=pool[3:]),
        selection=Selection(finalists=0),
        agent_error_score=100.0,  # so the failing candidate is selected, and its test fails
        log=None,
    )
    with pytest.raises(RuntimeError, match="test: every trial of candidate 1 failed"):
        study.optimize(iterations=1)


def test_without_a_split_nothing_is_tested():
    study = Study(
        agent=poet,
        experiment=Experiment(cases=cases(3), evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(grid([3]), STANZAS)]),
        log=None,
    )
    result = study.optimize(iterations=1)
    assert result.test is None and result.selected[0].params == {"stanzas": 3}
    assert not result.to_dict()["tested"]


# ---- rotation side by side -------------------------------------------------------------------------------------------
PROPOSING = {"now": 0, "most": 0}  # module state: every run's copy of the method counts here together
PROPOSING_LOCK = threading.Lock()


def slow_grid(values, delay=0.2, fail_on=None):
    """Proposes ``values`` in turn, each after ``delay`` seconds, counting the runs proposing at once; raises in a run
    whose train cases include the case named ``fail_on``."""
    from strands_evals.optimization import Method
    from strands_evals.optimization.methods import _ChoiceDraw

    class SlowGrid(Method):
        kinds = ("choice",)

        def __init__(self):
            self.values = list(values)

        def begin(self, block, base, history, structure, study, objective):
            if fail_on in [c.name for c in study.experiment.cases]:
                raise ValueError("boom")
            with PROPOSING_LOCK:
                PROPOSING["now"] += 1
                PROPOSING["most"] = max(PROPOSING["most"], PROPOSING["now"])
            time.sleep(delay)
            with PROPOSING_LOCK:
                PROPOSING["now"] -= 1
            return _ChoiceDraw({"stanzas": self.values.pop(0)}, {"method": "slow grid"})

    return SlowGrid()


def measured(record):
    events = [json.loads(line) for line in (record / "study.jsonl").read_text().splitlines()]
    return [
        (e["type"], e["number"], e.get("params"), e["estimates"]["words"]["mean"])
        for e in events
        if e["type"] in ("candidate", "validated")
    ]


def test_rotation_runs_side_by_side_and_each_run_finds_what_it_would_alone(tmp_path):
    pool = cases(8)
    split = Split(train=0.5, validation=0.25, test=0.25, rotate=True, seed=2)

    def study(split, record):
        return Study(
            agent=poet,
            experiment=Experiment(cases=pool, evaluators=[Words()]),
            objectives=Maximize("words"),
            search=Search(width=1, blocks=[Block(slow_grid([3, 4, 1]), STANZAS)]),
            split=split,
            record=record,
            log=None,
        )

    PROPOSING.update(now=0, most=0)
    result = study(split, tmp_path / "side-by-side").optimize(iterations=3)
    assert PROPOSING["most"] == 4  # the four runs proposed at the same time
    for k, (run, roles) in enumerate(zip(result.runs, split.assign(pool), strict=True), 1):
        alone = study(Split(train=roles.train, validation=roles.validation, test=roles.test), tmp_path / f"alone-{k}")
        found = alone.optimize(iterations=3).runs[0]
        assert run.index == k and run.cases == found.cases
        assert [c.params for c in run.selected] == [c.params for c in found.selected] == [{"stanzas": 4}]
        assert measured(tmp_path / "side-by-side" / f"run-{k}") == measured(tmp_path / f"alone-{k}")
        assert run.test.selected[0].means["words"] == found.test.selected[0].means["words"]


def test_when_a_run_fails_the_others_stop_at_their_next_step_and_its_error_is_raised(tmp_path):
    pool = cases(8)
    split = Split(train=0.5, validation=0.25, test=0.25, rotate=True, seed=2)
    failing = {k for k, roles in enumerate(split.assign(pool), 1) if "c0" in [c.name for c in roles.train]}
    assert 0 < len(failing) < 4
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(slow_grid([3, 4, 1], delay=0.5, fail_on="c0"), STANZAS)]),
        split=split,
        record=tmp_path,
        log=None,
    )
    with pytest.raises(ValueError, match="boom"):
        study.optimize(iterations=3)
    errors = {}
    for k in range(1, 5):
        events = [json.loads(line) for line in (tmp_path / f"run-{k}" / "study.jsonl").read_text().splitlines()]
        (stopped,) = [e for e in events if e["type"] == "stopped"]
        errors[k] = stopped["error"]
        assert not any(e["type"] == "finished" for e in events)
    # a run that would fail may be stopped by the first failure before it gets there; no other run fails
    assert any("boom" in errors[k] for k in failing)
    assert all(errors[k].startswith("_Stopped") for k in errors if k not in failing)
    assert not (tmp_path / "result.json").exists()


def test_the_runs_divide_max_concurrency_among_those_still_searching(monkeypatch):
    from strands_evals.optimization import study as study_module

    shares, real = [], study_module.run_trials

    def recording(*args, max_concurrency, **kwargs):
        shares.append(max_concurrency)
        return real(*args, max_concurrency=max_concurrency, **kwargs)

    monkeypatch.setattr(study_module, "run_trials", recording)
    running, lock = {"now": 0, "most": 0}, threading.Lock()

    async def task(agent, case):
        with lock:
            running["now"] += 1
            running["most"] = max(running["most"], running["now"])
        await asyncio.sleep(0.02)
        with lock:
            running["now"] -= 1
        return str(await agent.invoke_async(case.input))

    pool = cases(8)
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[Words()]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(grid([4, 4, 4, 4]), STANZAS)]),
        split=Split(train=0.5, validation=0.25, test=0.25, rotate=True),
        repetitions=2,
        max_concurrency=8,
        task=task,
        log=None,
    )
    with pytest.warns(UserWarning, match="in each of 4 runs side by side: 32 trials at once, but max_concurrency is 8"):
        study.optimize(iterations=1)
    assert shares[0] == 2 and max(shares) <= 8  # 2 each while all four search; a finished run's share is passed on
    assert running["most"] <= 8


# ---- the cost goal with a floor: finalists from the front (round 3b, F4) ---------------------------------------------
class Long(Evaluator):
    """How long the output is, to 10 words: 0.1 a word."""

    score_range = (0, 1)

    def __init__(self):
        super().__init__(name="long")

    def evaluate(self, case):
        return [EvaluationOutput(score=min(1.0, len(str(case.actual_output).split()) / 10), test_pass=True, reason="")]

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


def test_a_cost_goal_with_a_floor_validates_front_members_and_holds_its_incumbent_on_the_front(tmp_path):
    pool = cases(4)

    def study(objectives, record=None):
        return Study(
            agent=sayer,
            experiment=Experiment(cases=pool, evaluators=[Words(), Long()]),
            objectives=objectives,
            constraints=[AtLeast("long", 0.5)],  # an absolute floor: 5 words or more
            search=Search(width=1, blocks=[Block(scripted([3, 5, 6, 7, 8, 12]), LENGTH)]),
            split=Split(train=pool[:2], validation=pool[2:3], test=pool[3:]),
            selection=Selection(by="measured", finalists=3),
            record=record,
            log=None,
        )

    front = study(CheapestOnFront(Minimize("words"), Maximize("long"), within=0.25), record=tmp_path)
    result = front.optimize(iterations=6)
    by_n = {c.params["n"]: c.number for c in front.candidates}
    # the feasible front is 5, 6, 7, 8 and 12 words; 6 costs within 25% of 5 and is longer, so it is held (and ranked
    # first); then 5; then 8 (within 25% of 7, and longer)
    assert front.incumbent().number == by_n[6]
    assert [c.number for c in front.validated] == [0, by_n[6], by_n[5], by_n[8]]
    assert [c.number for c in result.selected] == [by_n[6]]
    head = json.loads((tmp_path / "study.jsonl").read_text().splitlines()[0])
    combine = head["combine"]
    assert (combine["kind"], combine["within"], combine["told"]) == ("CheapestOnFront", 0.25, [["words"]])
    assert [o["kind"] for o in head["objectives"]] == ["Minimize", "Maximize"]

    cheapest = study(Minimize("words"))  # round 3's cost goal: the 3 cheapest feasible, the cheapest selected
    result = cheapest.optimize(iterations=6)
    by_n = {c.params["n"]: c.number for c in cheapest.candidates}
    assert [c.number for c in cheapest.validated] == [0, by_n[5], by_n[6], by_n[7]]
    assert [c.number for c in result.selected] == [by_n[5]]


def test_a_floor_can_be_a_gain_over_the_baseline_on_the_same_cases():
    # the baseline says 2 words (long 0.2): a gain of at least 0.25 is 5 words or more, the absolute floor's front again
    pool = cases(4)
    study = Study(
        agent=sayer,
        experiment=Experiment(cases=pool, evaluators=[Words(), Long()]),
        objectives=CheapestOnFront(Minimize("words"), Maximize("long"), within=0.25),
        constraints=[AtLeast("long", 0.25, compare="difference")],
        search=Search(width=1, blocks=[Block(scripted([3, 5, 6, 7, 8, 12]), LENGTH)]),
        split=Split(train=pool[:2], validation=pool[2:3], test=pool[3:]),
        selection=Selection(by="measured", finalists=3),
        log=None,
    )
    result = study.optimize(iterations=6)
    by_n = {c.params["n"]: c.number for c in study.candidates}
    assert not study.candidates[by_n[3]].feasible and study.candidates[by_n[5]].feasible
    assert [c.number for c in study.validated] == [0, by_n[6], by_n[5], by_n[8]]
    assert [c.number for c in result.selected] == [by_n[6]]
