"""Reward first, then cost (``PrimaryFirst``, round 3d): the default schedule of a goal with a cost objective.

Scripted studies (no network): the agent's reply says which cases it solves and how many words it writes, so each
candidate's reward on every case and its output tokens (two per word) are known in advance."""

import json

import pytest
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    TPE,
    AtLeast,
    Block,
    CheapestOnFront,
    Choice,
    Derived,
    Independent,
    Joint,
    Maximize,
    Method,
    Minimize,
    Pareto,
    PrimaryFirst,
    Search,
    Seesaw,
    Selection,
    Split,
    Study,
    Trial,
    tunable,
)
from strands_evals.optimization.methods import _ChoiceDraw
from strands_evals.optimization.phases import Anchor, Cheaper, Paired, Primary, paired
from strands_evals.optimization.projection import Projection
from strands_evals.optimization.trials import TrialStatus
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says

ALL = "-"  # fails no case
NONE = ",".join(str(i) for i in range(20))  # fails every case
FAILS = [ALL, NONE, "0", "1", "2", "3", "7", "8", "11", "0,1", "0,1,2,3", "8,9", "2,3,4,5,6,7"]
WORDS = [5, 10, 15, 20, 25, 30, 35, 38, 39, 40]
FAIL = Choice(FAILS, default=NONE)
WORDS_ = Choice(WORDS, default=40)


@tunable(fails=FAIL, words=WORDS_)
def worker(fails, words):
    return Agent(model=Says(f"fails:{fails} " + " ".join(["w"] * (words - 1))), callback_handler=None)


class Solves(Evaluator):
    """1 when the reply does not name the case among those it fails, else 0."""

    def __init__(self):
        super().__init__(name="reward")

    def evaluate(self, case):
        spec = str(case.actual_output).split()[0].split(":", 1)[1]
        failed = set() if spec == ALL else {int(x) for x in spec.split(",")}
        index = int(case.name.split("#")[0][1:])
        return [EvaluationOutput(score=0.0 if index in failed else 1.0, test_pass=True, reason="")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


class Script(Method):
    """Proposes the given (fails, words) in turn: ``phases[k]`` for method copy k (a phase); remembers what each copy
    was told of its own proposals, and the evidence it saw."""

    kinds = ("choice",)

    def __init__(self, *phases):
        self.phases = [list(p) for p in phases]
        self.told: list[tuple[int, int, list[float] | None]] = []
        self.evidence: list[tuple[int, int]] = []

    def copy(self, k):
        import copy

        return copy.deepcopy(self)

    def begin(self, block, base, history, structure, study, objective):
        script = self.phases[objective or 0]
        fails, words = script.pop(0) if script else (ALL, 40)
        return _ChoiceDraw({"fails": fails, "words": words}, {"method": "script", "phase": objective})

    def observe(self, block, candidate, draw, study, objective):
        if candidate is None:
            return
        if draw is not None:
            aim = study.aim(objective)
            self.told.append(
                (objective, candidate.number, aim.told(study.projection, study.view(candidate), objective, None))
            )
        else:
            self.evidence.append((objective, candidate.number))


TRAIN = [f"c{i}" for i in range(8)]
VALIDATION = [f"c{i}" for i in range(8, 12)]
TEST = [f"c{i}" for i in range(12, 16)]
PARETO = Pareto(Maximize("reward"), Minimize("output_tokens"))


def study(
    script,
    objectives=PARETO,
    constraints=(),
    *,
    phases="auto",
    split=True,
    record=None,
    selection=None,
    repetitions=2,
    **kw,
):
    pool = [Case(name=n, input="go") for n in TRAIN + VALIDATION + TEST]
    return Study(
        agent=worker,
        experiment=Experiment(cases=pool if split else pool[:8], evaluators=[Solves()]),
        objectives=objectives,
        constraints=list(constraints),
        search=Search(width=1, blocks=[Block(script, FAIL, WORDS_)], phases=phases),
        repetitions=repetitions,
        split=Split(
            train=[c for c in pool if c.name in TRAIN],
            validation=[c for c in pool if c.name in VALIDATION],
            test=[c for c in pool if c.name in TEST],
        )
        if split
        else None,
        selection=selection or Selection(by="measured", feasibility=0.8, also=("unmet=none", "unanchored", "anchor")),
        record=record,
        log=None,
        **kw,
    )


def events(path):
    return [json.loads(line) for line in (path / "study.jsonl").read_text().splitlines()]


# ---- which goals search in phases -----------------------------------------------------------------------------------
def test_the_primary_objective_is_the_first_that_is_not_a_cost():
    ev = {"reward": (0, 1)}
    wtc = Derived("wtc", lambda m: m["input_tokens"], range=(0, None))
    share = Derived("share", lambda m: m["input_tokens"], range=(0, 1))
    assert Projection([Maximize("reward"), Minimize("tokens")], [], evaluators=ev).primary_and_cost() == (0, 1)
    assert Projection([Minimize(wtc), Maximize("reward")], [], evaluators=ev).primary_and_cost() == (1, 0)
    assert Projection([Minimize("tokens")], [], evaluators=ev).primary_and_cost() is None  # a cost alone is primary
    assert Projection([Maximize("reward")], [], evaluators=ev).primary_and_cost() is None
    assert Projection([Maximize("reward"), Minimize(share)], [], evaluators=ev).primary_and_cost() is None  # bounded


def test_a_goal_with_a_cost_objective_searches_in_phases_by_default_and_phases_none_restores_round_3b():
    phased = study(Script([]))
    assert isinstance(phased.schedule, PrimaryFirst) and isinstance(phased.schedule.inner, Joint)
    assert [type(a).__name__ for a in phased.aims] == ["Primary", "Cheaper"] and len(phased.methods) == 2
    old = study(Script([]), phases=None)
    assert isinstance(old.schedule, Joint) and old.aims == [old.combine] and len(old.methods) == 1
    assert old.aim(0).told_names(old.projection, 0) == ["reward", "output_tokens"]  # both, from the first step
    best = study(Script([]), objectives=Maximize("reward"))
    assert isinstance(best.schedule, Joint)  # max reward: unchanged
    cheapest = study(Script([]), objectives=Minimize("output_tokens"), constraints=[AtLeast("reward", 0.5)])
    assert isinstance(cheapest.schedule, Joint)  # a cost declared alone is primary: no phases
    both = study(Script([]), objectives=Independent(Maximize("reward"), Minimize("output_tokens")))
    assert isinstance(both.schedule, Joint)  # Independent searches each objective itself
    with pytest.raises(ValueError, match="primary objective and a cost"):
        study(Script([]), objectives=Maximize("reward"), phases=PrimaryFirst())


# ---- the phase switch, the stopping rule, the anchor and the budget --------------------------------------------------
def test_phase_a_searches_reward_until_it_stalls_then_phase_b_lowers_the_cost_from_a_star(tmp_path):
    script = Script(
        [(ALL, 40), (ALL, 39), (ALL, 38), (ALL, 35)],  # phase A: 1.0 at the first step, then no gain twice
        [(ALL, 20), ("0", 5), (ALL, 10)],  # phase B: cheaper and as good; cheaper still but loses a case; cheaper
    )
    s = study(script, record=tmp_path)
    result = s.optimize(iterations=6)
    a_star = s.anchor
    assert a_star is not None and a_star.number == 1  # the first 1.0 (later ties do not move the incumbent)
    phases = [c.phase for c in s.candidates]
    assert phases == [None, "primary", "primary", "primary", "cost", "cost", "cost"]
    assert len(s.candidates) - 1 == 6  # the budget: 3 + 3, the same as one phase of 6
    # each phase's copy is told its own aim: phase A the reward, phase B the cost
    copies = [s.methods[k][next(iter(s.blocks))] for k in (0, 1)]
    told = {(k, n): v for c in copies for k, n, v in c.told}
    assert told[(0, 1)] == [1.0] and told[(1, 4)] == [s.candidates[4].metrics["output_tokens"]]
    assert {k for k, _, _ in copies[0].told} == {0} and {k for k, _, _ in copies[1].told} == {1}
    # phase B's copy heard nothing while phase A ran (not even the baseline), then every candidate so far at once
    assert [n for _, n in copies[1].evidence] == [0, 1, 2, 3]
    # phase B held the cheapest candidate no worse than A*: 4 (20 words), then 6 (10 words); 5 lost a case
    v = {c.number: s.view(c) for c in s.candidates}
    assert v[5].anchored.within is False and v[4].anchored.within and v[6].anchored.within
    assert s.candidates[6].base == 4 and s.candidates[5].base == 4
    assert result.runs[0].pick == 6 and result.runs[0].anchor == 1
    ev = events(tmp_path)
    anchor = next(e for e in ev if e["type"] == "anchor")
    assert anchor["candidate"] == 1 and anchor["phase"]["reason"] == "stall" and anchor["phase"]["steps"] == 3
    assert [st["gain"] for st in anchor["steps"]] == [True, False, False] and anchor["cost"]["total"]["tokens"] > 0
    scopes = {e["phase"]: e for e in ev if e["type"] == "scope" and "phase" in e}
    assert set(scopes) == {"primary", "cost"} and scopes["primary"]["limit"] == 4  # 2/3 of 6 steps
    header = ev[0]
    assert header["schedule"]["kind"] == "PrimaryFirst" and set(header["schedule"]["children"]) == {"primary", "cost"}
    assert [a["kind"] for a in header["aims"]] == ["Primary", "Cheaper"]
    assert header["combine"]["told"] == [["reward"], ["output_tokens"]]
    cand = [e for e in ev if e["type"] == "candidate"]
    assert cand[4]["phase"] == "cost" and cand[4]["aim"] == 1 and cand[4]["objective"] is None


def test_phase_a_ends_at_its_cap_while_the_incumbent_has_headroom(tmp_path):
    script = Script([("0,1,2,3", 40)] * 1 + [("0,1,2,3", w) for w in (39, 38, 35, 30)], [(ALL, 20)] * 3)
    s = study(script, split=False, selection=Selection(by="measured", feasibility=0.8), record=tmp_path)
    s.optimize(iterations=6)
    assert s.schedule.ended["reason"] == "cap" and s.schedule.ended["steps"] == 4  # reward 0.5: no stall below 0.75
    assert [c.phase for c in s.candidates].count("cost") == 2
    anchor = next(e for e in events(tmp_path) if e["type"] == "anchor")
    assert len(anchor["steps"]) == anchor["phase"]["steps"] == 4  # every step's evidence, the last at the cap too
    assert [st["gain"] for st in anchor["steps"]] == [True, False, False, False]


def test_a_floor_goal_ends_phase_a_once_the_floor_is_met():
    goal = CheapestOnFront(Minimize("output_tokens"), Maximize("reward"))
    script = Script([("0,1", 40), (ALL, 40), (ALL, 39)], [(ALL, 20), (ALL, 10), (ALL, 5), (ALL, 15)])
    s = study(
        script,
        objectives=goal,
        constraints=[AtLeast("reward", 0.9)],
        split=False,
        selection=Selection(by="measured", feasibility=0.8),
    )
    s.optimize(iterations=6)
    assert s.schedule.ended["reason"] == "met" and s.anchor.number == 2  # 0.75, then 1.0 meets 0.9
    assert [c.phase for c in s.candidates].count("cost") == 4
    assert s.best_candidates[0].params["words"] == 5  # the cheapest that keeps the floor


def test_the_budget_is_the_same_with_and_without_phases():
    a = study(Script([(ALL, w) for w in (40, 39, 38, 35)], [(ALL, w) for w in (30, 25, 20, 15, 10, 5)]))
    a.optimize(iterations=8)
    b = study(Script([(ALL, w) for w in (40, 39, 38, 35, 30, 25, 20, 15, 10)]), phases=None)
    b.optimize(iterations=8)
    assert len(a.candidates) == len(b.candidates) == 9


# ---- paired evidence and the anchored judgement ----------------------------------------------------------------------
def trials(values, case_prefix="c"):
    return tuple(
        Trial(case=f"{case_prefix}{i}", repetition=r, status=TrialStatus.COMPLETED, metrics={"reward": v})
        for i, vs in enumerate(values)
        for r, v in enumerate(vs, 1)
    )


def test_paired_evidence_is_over_the_cases_both_measured():
    p = Projection([Maximize("reward")], [], evaluators={"reward": (0, 1)})
    same = paired(p, trials([[1, 1]] * 6), trials([[1, 1]] * 6), "reward")
    assert same == Paired(0.0, 0.0, 6) and same.probability(-0.02) == 1.0 and same.beats() == 0.0
    worse = paired(p, trials([[1, 1]] * 5 + [[0, 0]]), trials([[1, 1]] * 6), "reward")
    assert worse.difference == pytest.approx(-1 / 6) and worse.cases == 6
    assert worse.probability(-0.02) < 0.5 and worse.probability(-0.5) > 0.9
    better = paired(p, trials([[1, 1]] * 6), trials([[1, 0]] * 3 + [[1, 1]] * 3), "reward")
    assert better.beats() > 0.9
    assert paired(p, trials([[1]]), trials([[1]], "x"), "reward") is None  # no case in common


def test_the_anchor_itself_always_keeps_its_own_level():
    goal = Pareto(Maximize("reward"), Minimize("output_tokens"))
    s = study(Script([(ALL, 40)], [(ALL, 20)]), objectives=goal)
    s.optimize(iterations=2)
    a = s.candidates[s.anchor.number]
    ev = s.view(a).anchored
    assert ev.reference == a.number and ev.within and ev.pair.difference == 0.0 and ev.probability == 1.0
    assert isinstance(s.aims[0], Primary) and isinstance(s.aims[1], Cheaper)


def test_a_floor_is_judged_as_a_stars_level_plus_the_paired_difference():
    p = Projection([Minimize("tokens"), Maximize("reward")], [AtLeast("reward", 0.9)], evaluators={"reward": (0, 1)})
    from strands_evals.optimization import Candidate
    from strands_evals.optimization.selection import View

    def cand(n, ts):
        est = p.estimates(ts)
        c = Candidate(
            number=n,
            params={},
            status="complete",
            trials=ts,
            estimates=est,
            eligible=True,
            violations=p.violations({k: e.mean for k, e in est.items()}),
            proposed=(),
            base=None,
            objective=None,
            origin={},
        )
        return View(c, c.metrics, (0.0,), True, {}, (0.5,), (est["reward"].mean,))

    # A* measured at 0.95 on 20 cases with its own uncertainty; c identical on every case, d identical but one case
    a = cand(1, trials([[1, 1]] * 19 + [[0, 0]]))
    same = cand(2, trials([[1, 1]] * 19 + [[0, 0]]))
    lost = cand(3, trials([[1, 1]] * 18 + [[0, 0]] * 2))
    anchor = Anchor(number=1, metric="reward", maximize=True, margin=None, confidence=0.8, feasibility=0.8, guard=None)
    assert anchor.rejudge(p, a, a).p_met == (1.0,) and anchor.rejudge(p, a, a).violations[0] <= 0  # 0.95 >= 0.9
    assert anchor.rejudge(p, same, a).p_met == (1.0,)  # the same on every case: as A*, whatever A*'s own spread
    judged = anchor.rejudge(p, lost, a)
    assert judged.violations[0] > 0 and judged.p_met[0] < 0.8  # 0.90 as measured, but not with confidence


# ---- the seesaw guard and A* as the fallback -------------------------------------------------------------------------
@pytest.mark.parametrize("k, pick", [(0, 3), (1, 4)])
def test_the_seesaw_guard_drops_a_cheaper_candidate_that_trades_cases_a_star_always_solves(k, pick):
    # A* (1) fails validation case 11; candidate 4 matches its mean there (0.75) but fails case 8, which A* solves on
    # every repetition: with k = 0 the guard drops it, and the pick is the cheapest of the rest (3, at 38 words)
    script = Script([("11", 40), ("11", 39), ("11", 38)], [("8", 20)])
    s = study(script, phases=PrimaryFirst(guard=Seesaw(k), margin=0.02))
    result = s.optimize(iterations=4)
    assert s.anchor.number == 1
    validated = {v.number: v for v in s.validated}
    assert validated[4].metrics["reward"] == validated[1].metrics["reward"] == 0.75
    assert result.runs[0].pick == pick


def test_with_nothing_cheaper_that_keeps_a_stars_reward_the_selection_is_a_star(tmp_path):
    script = Script([(ALL, 35), (ALL, 38), (ALL, 39)], [("0", 20), ("0,1", 10), ("2,3,4,5,6,7", 5)])
    s = study(script, record=tmp_path)
    result = s.optimize(iterations=6)
    assert s.anchor.number == 1
    assert [c.number for c in s._finalists()][0] == 1
    assert result.runs[0].pick == 1  # A*: nothing cheaper keeps its reward
    test = result.runs[0].test
    shipped = test.shipped("selection")
    assert shipped.number == 1 and shipped.means["reward"] == 1.0


def test_a_floor_goal_ships_the_closest_when_even_a_star_misses_it_on_validation_and_records_the_old_rule(tmp_path):
    goal = CheapestOnFront(Minimize("output_tokens"), Maximize("reward"))
    # A* solves every train case but two of the four validation cases (8, 9): 0.5 against a floor of 0.9
    script = Script([("8,9", 40), ("8,9", 39), ("8,9", 38)], [("8,9", 20)])
    s = study(script, objectives=goal, constraints=[AtLeast("reward", 0.9)], record=tmp_path)
    result = s.optimize(iterations=4)
    run = result.runs[0]
    assert run.unconfirmed and run.pick is not None
    assert run.alternatives["unmet=none"] == []  # the old fallback ships the baseline
    assert run.alternatives["anchor"] == [s.anchor.number]
    selected = next(e for e in events(tmp_path) if e["type"] == "selected")
    assert selected["alternatives"]["unmet=none"] == [] and selected["unconfirmed"]
    assert selected["anchor"] == s.anchor.number and selected["finalists"][0] == s.anchor.number
    base = run.test.baseline
    assert "unmet=none" in base.rules  # the baseline arm is what unmet="none" ships


def test_every_rule_the_selection_records_is_tested_when_it_differs(tmp_path):
    script = Script([(ALL, 40), (ALL, 39), (ALL, 38)], [(ALL, 20), (ALL, 10)])
    s = study(script, record=tmp_path)
    result = s.optimize(iterations=5)
    run = result.runs[0]
    assert run.pick == 5 and run.alternatives["anchor"] == [1]
    arms = {a.number: a for a in [*run.test.selected, *run.test.alternatives]}
    assert 1 in arms and "anchor" in arms[1].rules  # A* is tested, as its own arm when it is not a member
    assert arms[5].pick or len(run.test.selected) == 1
    lines = [json.loads(line) for line in (tmp_path / "test.jsonl").read_text().splitlines()]
    roles = [(e["number"], e["role"], e.get("rules")) for e in lines if e["type"] == "tested"]
    assert any(n == 1 and "anchor" in (rules or []) for n, _, rules in roles)


# ---- the methods each phase tells ------------------------------------------------------------------------------------
def test_tpes_phase_b_copy_is_told_the_cost_with_the_anchor_as_its_constraint():
    tpe = TPE(seed=3, n_startup_trials=2)
    pool = [Case(name=n, input="go") for n in TRAIN]
    s = Study(
        agent=worker,
        experiment=Experiment(cases=pool, evaluators=[Solves()]),
        objectives=PARETO,
        search=Search(width=1, blocks=[Block(tpe, FAIL, WORDS_)], phases=PrimaryFirst(cap=3)),
        selection=Selection(by="measured", feasibility=0.8),
        log=None,
    )
    s.optimize(iterations=6)
    a_copy, b_copy = s.methods[0][next(iter(s.blocks))], s.methods[1][next(iter(s.blocks))]
    for st in a_copy._studies.values():
        assert [d.name for d in st.directions] == ["MAXIMIZE"]
    for st in b_copy._studies.values():
        assert [d.name for d in st.directions] == ["MINIMIZE"]
        constrained = [t for t in st.trials if t.constraints]
        assert constrained and len(constrained) == len([t for t in st.trials if t.state.name == "COMPLETE"])


def test_the_rewriter_in_phase_b_is_shown_both_and_told_its_limits_against_a_star():
    s = study(Script([(ALL, 40), (ALL, 39), (ALL, 38)], [(ALL, 20)]))
    s.optimize(iterations=4)
    p = s.projection
    assert s.aim(0).shown(p) == ["reward"] and s.aim(0).per_trial(p) == ["reward"]
    assert s.aim(1).shown(p) == ["reward", "output_tokens"] and s.aim(1).digest_costs(p) == ["output_tokens"]
    limits = s.aim(1).limits(p)
    assert len(limits) == 1 and "no worse than candidate 1's" in limits[0]


def test_the_digest_gives_every_cases_cost_in_phase_b_only():
    from strands_evals.optimization._reflection.agentic import digest_text

    rows = [
        {
            "file": f"f{i}.json",
            "case": f"c{i % 3}",
            "repetition": 1,
            "reward": 1.0 if i % 3 else 0.0,
            "status": "completed",
            "error": None,
            "reasons": {},
            "objectives": {"reward": 1.0, "wtc": 1000.0 * (i + 1)},
            "ended": "done",
            "messages": 4,
            "calls": 1,
        }
        for i in range(6)
    ]
    plain = digest_text(rows, 0.5, 20000)
    costed = digest_text(rows, 0.5, 20000, costs=["wtc"])
    assert "Cost per conversation" not in plain and "c1 (2/2), c2 (2/2)" in plain
    assert "Cost per conversation, over all 6: wtc 3,500" in costed
    assert costed.index("- c2 (2/2): wtc 4,500") < costed.index("- c1 (2/2): wtc 3,500")  # costliest first


# ---- rotation --------------------------------------------------------------------------------------------------------
def test_under_rotation_each_runs_pick_is_pooled():
    pool = [Case(name=f"c{i}", input="go") for i in range(16)]
    s = Study(
        agent=worker,
        experiment=Experiment(cases=pool, evaluators=[Solves()]),
        objectives=PARETO,
        search=Search(width=1, blocks=[Block(Script([(ALL, 40), (ALL, 39), (ALL, 38)], [(ALL, 20)]), FAIL, WORDS_)]),
        split=Split(train=0.5, validation=0.25, test=0.25, rotate=True, seed=1),
        selection=Selection(by="measured", feasibility=0.8, also=("anchor",)),
        log=None,
    )
    result = s.optimize(iterations=4)
    assert all(r.pick is not None and r.anchor is not None for r in result.runs)
    assert set(result.picks) >= {"selection", "anchor"}
    assert result.picks["selection"].cases == 16
    assert result.to_dict()["picks"]["selection"]["runs"] == 4
