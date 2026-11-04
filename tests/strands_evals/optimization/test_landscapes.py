"""What an Optuna method's landscapes hold: every trial told again, as decided now, before each proposal; a step's
pending proposals running for the constant liar; and a round's results kept in the round's own landscape.

Scripted studies (no model calls, no model of the metrics): the "agent" is a point, a task measures it, and a scripted
evaluator scores it, so every candidate's measured means are known in advance."""

import dataclasses
import json
import math
import warnings

import optuna
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    TPE,
    AtLeast,
    Block,
    Choice,
    Derived,
    Float,
    Int,
    Joint,
    Maximize,
    Minimize,
    Nested,
    OneOf,
    Optuna,
    Pareto,
    PrimaryFirst,
    Search,
    Selection,
    Study,
    component,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---- a point, measured exactly ---------------------------------------------------------------------------------------
X1 = Choice(["a", "b", "c"], default="a")
X2 = Float(0.0, 10.0, default=1.0)
HEIGHT = {"a": 0.6, "b": 0.8, "c": 0.95}


def quality(x1, x2):
    return HEIGHT[x1] * math.exp(-((x2 - 5.0) ** 2) / 8.0)


@dataclasses.dataclass(frozen=True)
class Point:
    x1: str
    x2: float


@tunable(x1=X1, x2=X2)
def point(x1, x2):
    return Point(x1, float(x2))


def measure(agent, case):
    return agent  # the task's output: the point itself


class Score(Evaluator):
    """The point's quality, the same on every case."""

    def __init__(self):
        super().__init__(name="score")

    def evaluate(self, data):
        p = data.actual_output
        return [EvaluationOutput(score=quality(p.x1, p.x2), test_pass=True, reason="")]

    async def evaluate_async(self, data):
        return self.evaluate(data)


CASES = [Case(name=f"c{i}", input=i) for i in range(2)]


class Drifting(Study):
    """Decides each odd-numbered candidate a little better the longer ago it was measured, so the order of earlier
    candidates changes from one step to the next, as it does when the model is refitted (or starts deciding by
    measured means)."""

    def view(self, candidate):
        v = super().view(candidate)
        if candidate.number % 2 == 0 or "score" not in v.metrics:
            return v
        age = len(self.candidates) - candidate.number
        return dataclasses.replace(v, metrics={**v.metrics, "score": v.metrics["score"] + 0.05 * age})


class Recording(BaseSampler):
    """A sampler that records the trials (number, state, values, params) it sees at every proposal, then proposes as
    ``inner`` does."""

    def __init__(self, inner, seen):
        self.inner, self.seen = inner, seen

    def infer_relative_search_space(self, study, trial):
        self.seen.append(
            [
                (t.number, t.state, t.values, dict(t.params))
                for t in study.get_trials(deepcopy=False)
                if t.number != trial.number
            ]
        )
        return self.inner.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        return self.inner.sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        return self.inner.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study, trial):
        self.inner.before_trial(study, trial)

    def after_trial(self, study, trial, state, values):
        self.inner.after_trial(study, trial, state, values)


def tpe_sampler(seed):
    return TPESampler(seed=seed, n_startup_trials=4, multivariate=True, group=True, constant_liar=True)


def point_study(method, *, width=1, cls=Study, selection=None):
    return cls(
        agent=point,
        experiment=Experiment(cases=CASES, evaluators=[Score()]),
        objectives=Maximize("score"),
        task=measure,
        search=Search(blocks=[Block(method, X1, X2)], width=width),
        selection=selection or Selection(by="measured", finalists=0),
        log=None,
    )


# ---- what a landscape is told ----------------------------------------------------------------------------------------
def by_values(study, params):
    """The candidate that measured these values of the block."""
    return next(c for c in study.candidates if c.params == params)


def test_every_proposal_sees_each_earlier_candidate_as_decided_now_not_when_it_was_told():
    seen = []
    study = point_study(Optuna(lambda: Recording(tpe_sampler(3), seen)), cls=Drifting)
    study.optimize(iterations=8)
    assert len(seen) == 8
    stale = 0
    for m, trials in enumerate(seen, start=1):  # the proposal that made candidate m
        assert len(trials) == m  # the baseline (evidence) and every candidate before it, all told
        for _, state, values, params in trials:
            c = by_values(study, params)
            age = m - c.number  # what Drifting decides at the m-th proposal: m candidates measured
            now = c.metrics["score"] + (0.05 * age if c.number % 2 else 0.0)
            assert state == TrialState.COMPLETE and values == [pytest.approx(now)], (m, c.number)
            # told once, as before: an odd candidate's value one step after it was measured
            stale += c.number % 2 == 1 and age > 1
    assert stale > 0


class Constrained(Recording):
    """Records each trial's params and constraints at every proposal, then proposes as TPE does (which writes its own
    attributes for each trial, kept across rebuilds beside the constraints)."""

    def infer_relative_search_space(self, study, trial):
        self.seen.append(
            [
                (dict(t.params), dict(t.constraints))
                for t in study.get_trials(deepcopy=False)
                if t.number != trial.number
            ]
        )
        return self.inner.infer_relative_search_space(study, trial)


def test_a_constraint_is_judged_again_before_each_proposal_too():
    class Slipping(Study):
        """Judges each odd-numbered candidate as missing the floor once two more candidates have been measured."""

        def view(self, candidate):
            v = super().view(candidate)
            if candidate.number % 2 and len(self.candidates) - candidate.number > 2 and v.violations:
                return dataclasses.replace(v, violations=(1.0,))
            return v

    seen = []
    study = Slipping(
        agent=point,
        experiment=Experiment(cases=CASES, evaluators=[Score()]),
        objectives=Maximize("score"),
        constraints=[AtLeast("score", 0.0)],  # every candidate meets it, as measured
        task=measure,
        search=Search(blocks=[Block(Optuna(lambda: Constrained(tpe_sampler(4), seen)), X1, X2)], width=1),
        selection=Selection(by="measured", finalists=0),
        log=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(iterations=8)
    assert not [w for w in caught if "already set" in str(w.message)]  # no constraint kept from an earlier rebuild
    name = "score >= 0.0"
    slipped = 0
    for m, trials in enumerate(seen, start=1):
        for params, constraints in trials:
            c = by_values(study, params)
            missing = c.number % 2 == 1 and m - c.number > 2
            assert (constraints[name] > 0) == missing, (m, c.number, constraints)
            slipped += missing
    assert slipped > 0


def test_a_steps_pending_proposals_stay_running_so_the_constant_liar_keeps_them_apart():
    seen = []
    method = Optuna(lambda: Recording(tpe_sampler(5), seen))
    study = point_study(method, width=3)
    study.optimize(iterations=4)
    assert len(seen) == 12
    for step in range(4):
        for slot in range(3):
            trials = seen[3 * step + slot]
            running = [(n, p) for n, state, _, p in trials if state == TrialState.RUNNING]
            assert len(running) == slot  # the step's earlier proposals, each with the values it proposed
            mine = [c for c in study.candidates if c.step == step]
            assert [p for _, p in running] == [{"x1": c.params["x1"], "x2": c.params["x2"]} for c in mine[:slot]]
            assert sum(1 for _, state, _, _ in trials if state == TrialState.COMPLETE) == 1 + 3 * step
    for step in range(4):
        values = [(c.params["x1"], c.params["x2"]) for c in study.candidates if c.step == step]
        assert len(set(values)) == 3


def test_with_nothing_decided_differently_the_proposals_are_those_of_one_optuna_study_told_as_it_goes():
    # by measured means, with no re-measurement, a candidate's decided value never changes: rebuilding the landscape
    # before each proposal must then propose exactly what one Optuna study, asked and told in turn, proposes
    study = point_study(TPE(seed=11))
    study.optimize(iterations=10)
    reference = optuna.create_study(direction="maximize", sampler=tpe_sampler(11))
    d = {"x1": CategoricalDistribution(["a", "b", "c"]), "x2": FloatDistribution(0.0, 10.0)}
    reference.add_trial(
        optuna.trial.create_trial(params={"x1": "a", "x2": 1.0}, distributions=d, values=[quality("a", 1.0)])
    )
    proposed = []
    for _ in range(10):
        t = reference.ask()
        x1, x2 = t.suggest_categorical("x1", ["a", "b", "c"]), t.suggest_float("x2", 0.0, 10.0)
        proposed.append((x1, x2))
        reference.tell(t, quality(x1, x2))
    assert [(c.params["x1"], c.params["x2"]) for c in study.candidates[1:]] == proposed


def test_an_ineligible_candidate_stays_a_failed_trial_after_every_rebuild():
    seen = []

    def breaks_on_b(agent, case):
        if agent.x1 == "b":
            raise RuntimeError("b cannot run")
        return agent

    study = Study(
        agent=point,
        experiment=Experiment(cases=CASES, evaluators=[Score()]),
        objectives=Maximize("score"),
        task=breaks_on_b,
        search=Search(blocks=[Block(Optuna(lambda: Recording(tpe_sampler(2), seen)), X1, X2)], width=1),
        selection=Selection(by="measured", finalists=0),
        agent_error_score=None,
        log=None,
    )
    study.optimize(iterations=12)
    failed = [c.number for c in study.candidates if not c.eligible]
    assert failed
    for trials in seen:
        for _, state, _, params in trials:
            if state == TrialState.FAIL:
                assert params["x1"] == "b"  # a failed trial holds the values it proposed
            else:
                assert by_values(study, params).eligible
    last = len(seen)  # the last proposal made candidate 12: it saw candidates 0 to 11
    assert sum(t[1] == TrialState.FAIL for t in seen[-1]) == len([n for n in failed if n < last])


# ---- a round's results stay in its own landscape ---------------------------------------------------------------------
class Plain:
    size, solves = 10, 0.75


class Good:
    def __init__(self, size):
        self.size, self.solves = size, 1.0


class Bad:
    def __init__(self, size):
        self.size, self.solves = size, 0.5


SIZE_G, SIZE_B = Int(1, 20, default=10), Int(1, 20, default=10)
MANAGER = OneOf(
    component(Plain), component(Good, size=SIZE_G), component(Bad, size=SIZE_B), labels=["plain", "good", "bad"]
)


@tunable(manager=MANAGER)
def managed(manager):
    return manager


@dataclasses.dataclass(frozen=True)
class Outcome:
    solved: float
    cost: float


def run_managed(agent, case):
    # "good" solves every case, "plain" three in four, "bad" one in two; the cost is the manager's size
    return Outcome(float(case.input % 4 < round(4 * agent.solves)), float(agent.size))


class Solved(Evaluator):
    def __init__(self):
        super().__init__(name="reward")

    def evaluate(self, data):
        return [EvaluationOutput(score=data.actual_output.solved, test_pass=True, reason="")]

    async def evaluate_async(self, data):
        return self.evaluate(data)


class Spent(Evaluator):
    score_range = (0, None)

    def __init__(self):
        super().__init__(name="cost")

    def evaluate(self, data):
        return [EvaluationOutput(score=data.actual_output.cost, test_pass=True, reason="")]

    async def evaluate_async(self, data):
        return self.evaluate(data)


COST = Derived("spent", lambda m: m["cost"], range=(0, None))


class Scripted(BaseSampler):
    """Proposes the next value of ``script`` (one list shared by every landscape the factory opens) for the manager,
    and a smaller size each time it is asked for one."""

    def __init__(self, script, sizes):
        self.script, self.sizes = script, sizes

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == "manager":
            return self.script.pop(0)
        self.sizes[0] -= 1
        return max(param_distribution.low, self.sizes[0])


def managed_study(record, script):
    sizes = [20]
    structure = Block(Optuna(lambda: Scripted(script, sizes)), MANAGER, name="structure")
    numbers = Block(Optuna(lambda: Scripted(script, sizes)), SIZE_G, SIZE_B, name="numbers")
    return Study(
        agent=managed,
        experiment=Experiment(cases=[Case(name=f"c{i}", input=i) for i in range(8)], evaluators=[Solved(), Spent()]),
        objectives=Pareto(Maximize("reward"), Minimize(COST)),
        task=run_managed,
        search=Search(
            blocks=[structure, numbers],
            width=1,
            schedule=Nested(outer=structure, inner=Joint(), inner_iterations=2),
            phases=PrimaryFirst(cap=3, stall=None),
        ),
        selection=Selection(by="measured", finalists=0),
        record=record,
        log=None,
    )


def events(path):
    return [json.loads(line) for line in (path / "study.jsonl").read_text().splitlines()]


def test_a_phase_b_round_whose_opener_loses_a_stars_level_searches_and_is_credited_under_its_own_choice(tmp_path):
    # from "plain" (the default): phase A, a round of "good" (A* is its opener); phase B, a round of "bad" (which never
    # keeps A*'s reward), then "good" again
    study = managed_study(tmp_path, ["good", "bad", "good", "good"])
    study.optimize(iterations=9)
    log = events(tmp_path)
    scopes = {e["id"]: e for e in log if e["type"] == "scope"}
    rounds = [s for s in scopes.values() if s.get("outer") == "structure" and s.get("phase") == "cost"]
    first = rounds[0]["id"]
    members = [c for c in study.candidates if c.scope == first]
    opener = next(c for c in members if c.role == "outer")
    assert opener.params["manager"] == "bad" and opener.eligible
    # its inner steps hold the opener (no candidate of the round keeps A*'s level), never A*, which is "good"
    assert all(c.params["manager"] == "bad" for c in members)
    assert {c.base for c in members if c.role == "step"} <= {c.number for c in members}
    # the outer draw is told the round's nearest candidate, under "bad": not a "good" result from outside the round
    told = next(e for e in log if e["type"] == "told" and e["scope"] == first)
    assert told["draw_candidate"] == opener.number
    assert told["candidate"] is not None and study.candidates[told["candidate"]].params["manager"] == "bad"
    # and the phase-B copy's structure landscape holds it as infeasible (it misses the margin against A*)
    b_structure = study.methods[1]["structure"]
    built = b_structure._studies[()]
    entries = b_structure._landscapes[()].entries
    mine = [t for t, e in zip(built.trials, entries, strict=True) if e.params.get("manager") == "bad"]
    assert mine and all(
        t.state == TrialState.COMPLETE and sum(max(v, 0) for v in t.constraints.values()) > 0 for t in mine
    )


def test_a_rounds_value_is_decided_again_whenever_its_draw_is_told_again(tmp_path):
    class Later(Study):
        """Decides the round's first inner candidate best once the next round has begun."""

        def view(self, candidate):
            v = super().view(candidate)
            if candidate.number == 2 and len(self.candidates) > 4 and "spent" in v.metrics:
                return dataclasses.replace(v, metrics={**v.metrics, "reward": 1.0, "spent": 0.0})
            return v

    sizes = [20]
    script = ["good", "good", "good", "good"]
    structure = Block(Optuna(lambda: Scripted(script, sizes)), MANAGER, name="structure")
    numbers = Block(Optuna(lambda: Scripted(script, sizes)), SIZE_G, SIZE_B, name="numbers")
    study = Later(
        agent=managed,
        experiment=Experiment(cases=[Case(name=f"c{i}", input=i) for i in range(4)], evaluators=[Solved(), Spent()]),
        objectives=Pareto(Maximize("reward"), Minimize(COST)),
        task=run_managed,
        search=Search(
            blocks=[structure, numbers],
            width=1,
            schedule=Nested(outer=structure, inner=Joint(), inner_iterations=2),
            phases=None,
        ),
        selection=Selection(by="measured", finalists=0),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=6)
    told = [e for e in events(tmp_path) if e["type"] == "told"]
    assert told[0]["candidate"] != 2  # at the round's end, candidate 2 was not its best
    entry = study.methods[0]["structure"]._landscapes[()].entries[1]  # after the baseline's evidence
    assert entry.number == told[0]["candidate"] and entry.standing is not None
    assert entry.standing().number == 2  # decided now, candidate 2 is the round's best, and it stands for the draw
