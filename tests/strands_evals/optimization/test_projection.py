"""Projection: per-trial values, rewards for scalar methods, how several objectives combine, and derived metrics."""

import pytest

from strands_evals.optimization import (
    AtLeast,
    AtMost,
    Candidate,
    CheapestOnFront,
    Chebyshev,
    Derived,
    Independent,
    Lexicographic,
    Maximize,
    Minimize,
    Pareto,
    Trial,
    WeightedSum,
)
from strands_evals.optimization.projection import Projection
from strands_evals.optimization.trials import TrialStatus

OK, FAILED, INVALID = TrialStatus.COMPLETED, TrialStatus.AGENT_FAILURE, TrialStatus.INFRASTRUCTURE_INVALID


def projection(objectives, constraints=(), **kw):
    return Projection(objectives, constraints, evaluators={"Score": (0, 1), "Polite": (0, 1), "Words": (0, None)}, **kw)


def candidate(p, number, rows):
    """rows: (case, status, metrics)."""
    trials = tuple(
        Trial(case=c, repetition=1, status=s, metrics=m, error="boom" if s == FAILED else None) for c, s, m in rows
    )
    estimates = p.estimates(trials)
    return Candidate(
        number=number,
        params={},
        status="complete",
        trials=trials,
        estimates=estimates,
        eligible=p.eligible(trials),
        violations=p.violations({k: e.mean for k, e in estimates.items()}),
        proposed=(),
        base=None,
        objective=None,
        origin={},
    )


def test_rewards_for_a_bounded_score_are_the_score_oriented():
    p = projection([Maximize("Score")])
    t = candidate(p, 0, [("a", OK, {"Score": 0.25}), ("b", OK, {"Score": 1.0})])
    assert [r for _, r, _ in p.rewards(t, Pareto(*p.objectives), None)] == [0.25, 1.0]
    p = projection([Minimize("Score")])
    assert [r for _, r, _ in p.rewards(candidate(p, 0, [("a", OK, {"Score": 0.25})]), Pareto(*p.objectives), None)] == [
        0.75
    ]


def test_rewards_for_an_unbounded_metric_are_mid_ranks_when_no_goal_metric_is_bounded():
    p = projection([Minimize("tokens")])
    t = candidate(p, 0, [("a", OK, {"tokens": 100.0}), ("b", OK, {"tokens": 300.0}), ("c", OK, {"tokens": 200.0})])
    assert [r for _, r, _ in p.rewards(t, Pareto(*p.objectives), None)] == [1.0, 0.0, 0.5]
    single = candidate(p, 0, [("a", OK, {"tokens": 100.0})])
    assert [r for _, r, _ in p.rewards(single, Pareto(*p.objectives), None)] == [0.5]


def test_rewards_combine_bounded_objectives_as_the_setting_says():
    p = projection([Maximize("Score"), Minimize("Polite")])
    t = candidate(p, 0, [("a", OK, {"Score": 1.0, "Polite": 1.0}), ("b", OK, {"Score": 0.5, "Polite": 0.0})])
    # Score -> 1.0, 0.5; Polite (lower is better) -> 0.0, 1.0
    assert [r for _, r, _ in p.rewards(t, Pareto(*p.objectives), None)] == [0.5, 0.75]
    assert [r for _, r, _ in p.rewards(t, Pareto(*p.objectives, weights=(3, 1)), None)] == [0.75, 0.625]
    assert [r for _, r, _ in p.rewards(t, Independent(*p.objectives), 1)] == [0.0, 1.0]
    assert [r for _, r, _ in p.rewards(t, Chebyshev(*p.objectives), None)] == [0.0, 0.5]
    assert [r for _, r, _ in p.rewards(t, Lexicographic(*p.objectives), None)] == [1.0, 0.5]


@pytest.mark.parametrize(
    "objectives, constraints",
    [
        ([Maximize("Score"), Minimize("tokens")], []),  # the Pareto set of success and tokens
        ([Minimize("tokens")], [AtLeast("Score", -0.05, compare="difference")]),  # the fewest tokens, success kept
        ([Maximize("Score")], [AtMost("tokens", 0.8, compare="ratio")]),  # the most success within a budget
    ],
)
def test_a_trial_is_rewarded_for_what_it_achieved_never_for_being_cheap(objectives, constraints):
    """Tokens rank a trial only against the candidate's others: a cheap failure would score as well as a costly
    success. A trial's reward is the goal's bounded metrics only; tokens are judged on whole candidates."""
    p = projection(objectives, constraints)
    rows = [("quit", OK, {"Score": 0.0, "tokens": 10.0}), ("solved", OK, {"Score": 1.0, "tokens": 900.0})]
    rewards = p.rewards(candidate(p, 0, rows), Pareto(*p.objectives), None)
    assert [(a.case, r) for a, r, _ in rewards] == [("quit", 0.0), ("solved", 1.0)]
    assert p.label_metrics() == (["Score"], False)


def test_rewards_for_constraints_failures_and_invalid_attempts():
    p = projection([Maximize("Score")], [AtMost("tokens", 200)])
    t = candidate(
        p, 0, [("a", OK, {"Score": 0.8, "tokens": 300.0}), ("b", FAILED, {"tokens": 50.0}), ("c", INVALID, {})]
    )
    rewards = p.rewards(t, Pareto(*p.objectives), None)
    assert [(a.case, r) for a, r, _ in rewards] == [("a", 0.8), ("b", 0.0)]  # the limit is the mean's; zero; left out
    assert rewards[0][2]["constraints_broken"] == {"tokens <= 200": 100.0}
    assert rewards[1][2]["error"] == "boom"


def test_an_agent_failure_counts_as_the_failure_policy_says():
    rows = [("a", OK, {"Score": 1.0}), ("b", FAILED, {"tokens": 5.0})]
    p = projection([Maximize("Score")])
    assert candidate(p, 0, rows).metrics["Score"] == 0.5
    p = projection([Maximize("Score")], agent_error_score=None)
    t = candidate(p, 0, rows)
    assert t.metrics["Score"] == 1.0 and not t.eligible  # case b has no Score: the candidate cannot be told


def two_objective_trials():
    p = projection([Maximize("Score"), Minimize("tokens")])
    base = candidate(p, 0, [("a", OK, {"Score": 0.5, "tokens": 200.0}), ("b", OK, {"Score": 0.5, "tokens": 200.0})])
    cheap = candidate(p, 1, [("a", OK, {"Score": 0.4, "tokens": 100.0}), ("b", OK, {"Score": 0.4, "tokens": 100.0})])
    good = candidate(p, 2, [("a", OK, {"Score": 0.9, "tokens": 300.0}), ("b", OK, {"Score": 0.9, "tokens": 300.0})])
    worse = candidate(p, 3, [("a", OK, {"Score": 0.3, "tokens": 400.0}), ("b", OK, {"Score": 0.3, "tokens": 400.0})])
    return p, [base, cheap, good, worse]


def test_pareto_tells_every_objective_and_holds_a_front_member():
    p, candidates = two_objective_trials()
    assert Pareto(*p.objectives).told(p, candidates[2], None, candidates[0]) == [0.9, 300.0]
    assert Pareto(*p.objectives).directions(p, None) == ["maximize", "minimize"]
    assert {t.number for t in Pareto(*p.objectives).best(p, candidates, candidates[0])} == {0, 1, 2}
    assert Pareto(*p.objectives).held(p, candidates, None, candidates[0]).number in {0, 1, 2}


def test_independent_tells_each_objective_its_own():
    p, candidates = two_objective_trials()
    assert Independent(*p.objectives).instances(2) == 2
    assert Independent(*p.objectives).told(p, candidates[2], 1, candidates[0]) == [300.0]
    assert Independent(*p.objectives).held(p, candidates, 0, candidates[0]).number == 2  # best Score
    assert Independent(*p.objectives).held(p, candidates, 1, candidates[0]).number == 1  # fewest tokens


def test_scalarizations_and_lexicographic():
    p = projection([Maximize("Score", scale=0.1), Minimize("tokens", scale=100)])
    _, candidates = two_objective_trials()
    candidates = [candidate(p, t.number, [(a.case, a.status, a.metrics) for a in t.trials]) for t in candidates]
    # gains over the baseline: cheap (-1, +1) -> 0; good (+4, -1) -> 3; worse (-2, -2) -> -4
    assert WeightedSum(*p.objectives).told(p, candidates[2], None, candidates[0]) == [pytest.approx(3.0)]
    assert [t.number for t in WeightedSum(*p.objectives).best(p, candidates, candidates[0])] == [2]
    assert Chebyshev(*p.objectives).told(p, candidates[2], None, candidates[0]) == [pytest.approx(-1.0)]
    assert [t.number for t in Chebyshev(*p.objectives).best(p, candidates, candidates[0])] == [
        0
    ]  # no objective worse than the baseline
    assert [t.number for t in Lexicographic(*p.objectives).best(p, candidates, candidates[0])] == [2]
    unscaled = projection([Maximize("Score"), Minimize("tokens")])
    with pytest.raises(ValueError, match="scale="):
        WeightedSum(*unscaled.objectives).told(unscaled, candidates[2], None, candidates[0])


def test_infeasible_and_ineligible_trials_are_held_last():
    p = projection([Maximize("Score")], [AtMost("tokens", 150)])
    over = candidate(p, 0, [("a", OK, {"Score": 0.9, "tokens": 300.0})])
    under = candidate(p, 1, [("a", OK, {"Score": 0.2, "tokens": 100.0})])
    assert Pareto(*p.objectives).held(p, [over, under], None, over).number == 1
    assert [t.number for t in Pareto(*p.objectives).best(p, [over, under], over)] == [1]
    assert Pareto(*p.objectives).held(p, [over], None, over).number == 0  # nothing feasible: the least infeasible


def test_an_agent_failure_scores_the_worst_end_of_the_range_for_the_goals_using_it():
    p = Projection(
        [Maximize("Score"), Minimize("Words")],
        [AtMost("Latency", 5)],
        evaluators={"Score": (0, 1), "Words": (0, None), "Latency": (0, 10), "Other": (0, 1)},
    )
    assert (p.failure_value("Score"), p.failure_value("Latency")) == (0.0, 10)
    assert p.failure_value("Words") is None  # unbounded on its bad side: left out, not scored as the best
    assert p.failure_value("Other") is None  # no goal says which end is worse
    fixed = Projection([Minimize("Words")], [], evaluators={"Words": (0, None)}, agent_error_score=0.0)
    assert fixed.failure_value("Words") == 0.0


# ---- derived metrics -------------------------------------------------------------------------------------------------
COST = Derived("cost", lambda m: m["input_tokens"] * 0.001 + m["output_tokens"] * 0.005, range=(0, None))


def test_a_derived_metric_is_computed_per_trial_then_estimated_like_any_other():
    p = projection([Maximize("Score")], [AtMost(COST, 1.0)])
    assert p.goal_metrics == ["Score", "cost"] and p.role("cost") == "derived"
    trials = [
        p.prepare(Trial(case=c, repetition=1, status=OK, metrics={"Score": 0.5, "input_tokens": i, "output_tokens": o}))
        for c, i, o in [("a", 100.0, 20.0), ("b", 300.0, 60.0)]
    ]
    assert [t.metrics["cost"] for t in trials] == [pytest.approx(0.2), pytest.approx(0.6)]
    assert p.estimates(trials)["cost"].mean == pytest.approx(0.4)
    assert p.violations({"Score": 0.5, "cost": 0.4}) == (pytest.approx(-0.6),)


def test_a_failed_agent_s_derived_metric_is_computed_when_it_can_be_and_otherwise_counts_as_its_worst():
    margin = Derived("margin", lambda m: m["Score"] - 0.5, range=(-0.5, 0.5))
    p = projection([Maximize(margin)], [AtMost(COST, 1.0)])
    failed = p.prepare(
        Trial(case="a", repetition=1, status=FAILED, metrics={"input_tokens": 100.0, "output_tokens": 0.0})
    )
    assert failed.metrics["cost"] == pytest.approx(0.1)  # from its usage, which a failed agent still has
    assert failed.metrics["margin"] == -0.5  # it needs a score the agent never got: the worst end of its range


def test_a_derived_metric_that_needs_a_metric_the_trial_lacks_is_unavailable_and_says_which():
    p = projection([Maximize("Score")], [AtLeast(Derived("typo", lambda m: m["Scroe"]), 0.5)])
    t = p.prepare(Trial(case="a", repetition=1, status=OK, metrics={"Score": 0.5}))
    assert "typo" in t.unavailable and "'Scroe'" in t.error and "typo" not in t.metrics
    with pytest.raises(ValueError, match="name of a measured one"):
        projection([Maximize(Derived("tokens", lambda m: 0.0))])


def test_a_combination_holds_its_objectives_and_a_single_objective_needs_none():
    both = Pareto(Maximize("Score"), Minimize("tokens"), weights=(2, 1))
    assert [o.metric for o in both.objectives] == ["Score", "tokens"]
    assert both.describe() == {"kind": "Pareto", "weights": [2, 1]}
    with pytest.raises(ValueError, match="at least one objective"):
        WeightedSum()
    with pytest.raises(TypeError, match="takes objectives"):
        Pareto(AtMost("tokens", 5))


# ---- the cost goal with a floor: the cheapest on the feasible front (round 3b, F4) -----------------------------------
def floored(*rows):
    """Candidates of (Words, Score): Words is the cost (unbounded), Score the value, at least 0.5."""
    p = projection([Minimize("Words"), Maximize("Score")], [AtLeast("Score", 0.5)])
    return p, [candidate(p, n, [("a", OK, {"Words": w, "Score": s})]) for n, (w, s) in enumerate(rows)]


def test_the_cheapest_on_the_front_breaks_a_near_tie_in_cost_by_the_value():
    p, c = floored((100, 0.3), (80, 0.6), (83, 0.9), (90, 0.95), (85, 0.7), (60, 0.4))
    goal = CheapestOnFront(*p.objectives, within=0.05)
    # 0 and 5 miss the floor; 4 is dominated by 2; 2 costs within 5% of 1, the cheapest feasible, and scores higher
    assert [t.number for t in goal.finalist_pool(p, c)] == [1, 2, 3]
    assert goal.held(p, c, None, c[0]).number == 2 and [t.number for t in goal.best(p, c, c[0])] == [2]
    assert goal.told(p, c[2], None, c[0]) == [83.0] and goal.directions(p, None) == ["minimize"]
    assert CheapestOnFront(*p.objectives, within=0.0).held(p, c, None, c[0]).number == 1  # no tolerance: the cheapest
    old = projection([Minimize("Words")], [AtLeast("Score", 0.5)])  # round 3's cost goal: the cheapest feasible
    oc = [candidate(old, t.number, [(a.case, a.status, a.metrics) for a in t.trials]) for t in c]
    assert Pareto(*old.objectives).held(old, oc, None, oc[0]).number == 1
    # while nothing meets the floor, the candidates that miss it least stand in: the incumbent climbs toward it
    assert goal.held(p, [c[0], c[5]], None, c[0]).number == 5 and goal.best(p, [c[0], c[5]], c[0]) == []
    assert goal.describe() == {"kind": "CheapestOnFront", "within": 0.05}
    with pytest.raises(TypeError, match="the cost first"):
        CheapestOnFront(Maximize("Score"), Minimize("Words"))
    with pytest.raises(ValueError, match="within"):
        CheapestOnFront(*p.objectives, within=1.5)
