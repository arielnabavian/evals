"""What a study found: its selection and, with test cases, how the selection did against the baseline on cases no part
of the search or the selection ever saw.

Every test number is paired by case: each case's mean over its repetitions, the selection's minus the baseline's (or
their ratio), so the unit is the case. Intervals are 90%, from the t distribution over cases. With
``Split(rotate=True)`` the runs' test cases together are every case once; their differences pool into ``Result.test``:

- ``difference``: what these runs' selections gained on cases they never saw (its interval weighs each run's cases by
  their share, ``sum_k (n_k / N)^2 s_k^2 / n_k``);
- ``procedure``: what running the study gains, the selections themselves varying from run to run (the corrected
  variance of the runs' estimates, ``(1 / K + n_test / n_train) S^2``, Nadeau and Bengio 2003, with K - 1 degrees of
  freedom); wider, and the honest interval for "running this optimizer gains".

A Pareto selection is a set; sets are not pooled: each run reports its members on its own test cases. A run that
ships one of its set's members (its pick: ``phases.PrimaryFirst``) pools it, as ``Result.picks["selection"]``; what
other rules ship (``Selection.also``) pools the same way, under each rule's name.

What it cost, in tokens, by role and phase (``cost.py``): ``Run.cost`` for each search, ``Result.cost`` for all of them.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .cost import Cost
from .trials import SCORED, Trial

LEVEL = 0.90


@dataclass(frozen=True)
class Difference:
    """A paired comparison over the test cases: ``mean`` with its 90% interval, from ``cases`` cases."""

    mean: float
    ci90: tuple[float, float] | None
    cases: int

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "ci90": list(self.ci90) if self.ci90 else None, "cases": self.cases}


@dataclass(frozen=True)
class Arm:
    """One configuration measured on the test cases: its trials, each metric's mean over the cases, and (for a
    selection) its difference from the baseline and ratio to it, per goal metric."""

    number: int
    role: str  # baseline | selected | fallback (the run selected nothing; it ships the baseline) | alternative
    params: dict[str, Any]
    trials: tuple[Trial, ...]
    means: dict[str, float]
    difference: dict[str, Difference] = field(default_factory=dict)
    ratio: dict[str, Difference] = field(default_factory=dict)
    unconfirmed: bool = False  # selected although nothing met the constraints: the closest (Selection.unmet)
    pick: bool = False  # the member of a selected set the run ships (phases.PrimaryFirst)
    rules: tuple[str, ...] = ()  # the rules that ship it: "selection", and Selection.also's

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "role": self.role,
            "unconfirmed": self.unconfirmed,
            **({"pick": self.pick} if self.pick else {}),
            **({"rules": list(self.rules)} if self.rules else {}),
            "means": self.means,
            "difference": {m: d.to_dict() for m, d in self.difference.items()},
            "ratio": {m: d.to_dict() for m, d in self.ratio.items()},
        }


@dataclass(frozen=True)
class Test:
    """One run's test: the baseline and its selection (a set's members for a Pareto goal) on its test cases."""

    cases: list[str]
    repetitions: int
    baseline: Arm
    selected: list[Arm]
    missing: dict[int, int]  # per configuration: test cases without a scored trial (left out of its comparison)
    alternatives: list[Arm] = field(default_factory=list)  # what other rules ship, not already among the arms

    def shipped(self, rule: str = "selection") -> Arm | None:
        """The arm ``rule`` ships (``Arm.rules``): the selection's pick (a single selection, or a set's pick), or what
        one of ``Selection.also``'s rules ships (the baseline when it ships nothing); None when no arm says so."""
        arms = [*self.selected, *self.alternatives]
        if rule == "selection" and len(self.selected) == 1 and not self.selected[0].rules:
            return self.selected[0]
        return next((a for a in [*arms, self.baseline] if rule in a.rules), None)

    @property
    def difference(self) -> dict[str, Difference] | None:
        """The selection's difference from the baseline, per goal metric (None for a set)."""
        return self.selected[0].difference if len(self.selected) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "repetitions": self.repetitions,
            "baseline": self.baseline.to_dict(),
            "selected": [a.to_dict() for a in self.selected],
            **({"alternatives": [a.to_dict() for a in self.alternatives]} if self.alternatives else {}),
            "missing": {str(k): v for k, v in self.missing.items()},
        }


@dataclass(frozen=True)
class Pooled:
    """The rotated runs' tests pooled: every case once (see the module's docstring)."""

    runs: int
    cases: int
    difference: dict[str, Difference]
    procedure: dict[str, Difference]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "cases": self.cases,
            "difference": {m: d.to_dict() for m, d in self.difference.items()},
            "procedure": {m: d.to_dict() for m, d in self.procedure.items()},
        }


@dataclass(frozen=True)
class Run:
    """One search: the cases it used by role, what it selected (``unconfirmed`` when none of the candidates it
    selected from met the constraints, so it shipped the closest: ``Selection.unmet``), its test (None without test
    cases), and what it spent, in tokens (``cost.py``)."""

    index: int
    cases: dict[str, list[str]]
    selected: list[Any]
    test: Test | None
    record: str | None = None
    cost: Cost | None = None
    unconfirmed: bool = False
    pick: int | None = None  # the candidate it ships, when its selection is a set (phases.PrimaryFirst)
    alternatives: dict[str, list[int]] = field(default_factory=dict)  # what Selection.also's rules ship
    anchor: int | None = None  # phase A's result, A*, when the run searched in phases

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cases": self.cases,
            "selected": [c.number for c in self.selected],
            "unconfirmed": self.unconfirmed,
            **({"pick": self.pick} if self.pick is not None else {}),
            **({"alternatives": self.alternatives} if self.alternatives else {}),
            **({"anchor": self.anchor} if self.anchor is not None else {}),
            "test": self.test.to_dict() if self.test else None,
            "record": self.record,
            "cost": self.cost.to_dict() if self.cost is not None else None,
        }


@dataclass(frozen=True)
class Result:
    """What a study found: one row of a results table.

    ``selected``: the configurations to ship (one search only; with rotation each run selected its own: see ``runs``).
    ``unconfirmed``: the runs (by index) whose selection met no constraint: each shipped the closest
    (``Selection.unmet``), and its test, pooled with the others, is of that.
    ``test``: the selection against the baseline on the test cases: one run's ``Test``, the runs' ``Pooled`` with
    rotation, or None when the study had no test cases (its numbers are then from the cases it searched).
    ``cost``: what every run spent, in tokens, by role and phase (each run's is ``Run.cost``)."""

    runs: list[Run]
    test: Test | Pooled | None
    picks: dict[str, Pooled] = field(default_factory=dict)  # each rule's shipped arm pooled over the runs (rotation)

    @property
    def selected(self) -> list[Any] | None:
        return self.runs[0].selected if len(self.runs) == 1 else None

    @property
    def cost(self) -> Cost | None:
        costs = [r.cost for r in self.runs if r.cost is not None]
        return sum(costs[1:], costs[0]) if costs else None

    @property
    def unconfirmed(self) -> list[int]:
        return [r.index for r in self.runs if r.unconfirmed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "strands-evals.optimization.result",
            "schema": 1,
            "level": LEVEL,
            "runs": [r.to_dict() for r in self.runs],
            "unconfirmed": self.unconfirmed,
            "test": self.test.to_dict() if self.test else None,
            "tested": any(r.test is not None for r in self.runs),
            "cost": self.cost.to_dict() if self.cost is not None else None,
            **({"picks": {rule: p.to_dict() for rule, p in self.picks.items()}} if self.picks else {}),
        }


# ---- computing -------------------------------------------------------------------------------------------------------
def case_means(trials: Sequence[Trial], metric: str, value: Any) -> dict[str, float]:
    """Each case's mean of ``metric`` over its scored trials (``value(trial, metric)``, the projection's)."""
    per: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        if t.status not in SCORED:
            continue
        v = value(t, metric)
        if v is not None:
            per[t.case].append(float(v))
    return {case: statistics.fmean(vs) for case, vs in per.items()}


def _t(df: int) -> float:
    from scipy.stats import t  # optional extra: optimization

    return float(t.ppf(0.5 + LEVEL / 2, df))


def paired(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[Difference | None, Difference | None, list[str]]:
    """``a`` minus ``b`` and ``a`` over ``b``, over the cases both measured; and those cases."""
    cases = sorted(set(a) & set(b))
    n = len(cases)
    if n == 0:
        return None, None, cases
    d = [a[c] - b[c] for c in cases]
    mean = statistics.fmean(d)
    ci = None
    if n >= 2:
        half = _t(n - 1) * statistics.stdev(d) / math.sqrt(n)
        ci = (mean - half, mean + half)
    ma, mb = statistics.fmean(a[c] for c in cases), statistics.fmean(b[c] for c in cases)
    ratio = None
    if mb:
        r = ma / mb
        rci = None
        if n >= 2:
            va = statistics.variance([a[c] for c in cases])
            vb = statistics.variance([b[c] for c in cases])
            cov = statistics.covariance([a[c] for c in cases], [b[c] for c in cases])
            var = max(va + r * r * vb - 2 * r * cov, 0.0) / (mb * mb * n)  # the delta method
            half = _t(n - 1) * math.sqrt(var)
            rci = (r - half, r + half)
        ratio = Difference(r, rci, n)
    return Difference(mean, ci, n), ratio, cases


def pool(runs: Sequence[Run], metrics: Sequence[str], value: Any, train_sizes: Sequence[int]) -> Pooled | None:
    """The rotated runs' single selections against the baseline, pooled over every test case (None when a run's
    selection is a set)."""
    if any(r.test is None or len(r.test.selected) != 1 for r in runs):
        return None
    return _pooled([r.test.selected[0] for r in runs if r.test], runs, metrics, value, train_sizes)


def pool_rules(
    runs: Sequence[Run], metrics: Sequence[str], value: Any, train_sizes: Sequence[int]
) -> dict[str, Pooled]:
    """Each rule's shipped arm (``Test.shipped``: the selection's pick, and ``Selection.also``'s rules) pooled over the
    rotated runs, when every run's test says which arm that is; empty when no run names its rules."""
    if any(r.test is None for r in runs):
        return {}
    named = sorted(
        {
            rule
            for r in runs
            for a in [*r.test.selected, *r.test.alternatives, r.test.baseline]  # type: ignore[union-attr]
            for rule in a.rules
        }
    )
    out = {}
    for rule in named:
        arms = [r.test.shipped(rule) for r in runs]  # type: ignore[union-attr]
        if all(a is not None for a in arms):
            pooled = _pooled(arms, runs, metrics, value, train_sizes)  # type: ignore[arg-type]
            if pooled is not None:
                out[rule] = pooled
    return out


def _pooled(
    arms: Sequence[Arm], runs: Sequence[Run], metrics: Sequence[str], value: Any, train_sizes: Sequence[int]
) -> Pooled | None:
    """One arm per run against its baseline, pooled over every run's test cases."""
    k = len(runs)
    difference: dict[str, Difference] = {}
    procedure: dict[str, Difference] = {}
    for m in metrics:
        per_run = []
        for arm, r in zip(arms, runs, strict=True):
            assert r.test is not None
            a = case_means(arm.trials, m, value)
            b = case_means(r.test.baseline.trials, m, value)
            per_run.append([a[c] - b[c] for c in sorted(set(a) & set(b))])
        n = sum(len(d) for d in per_run)
        if n == 0 or any(not d for d in per_run):
            continue
        mean = sum(sum(d) for d in per_run) / n
        var = sum((len(d) / n) ** 2 * (statistics.variance(d) / len(d) if len(d) >= 2 else 0.0) for d in per_run)
        half = _t(max(n - k, 1)) * math.sqrt(var)
        difference[m] = Difference(mean, (mean - half, mean + half), n)
        estimates = [statistics.fmean(d) for d in per_run]
        if k >= 2:
            ratio = statistics.fmean(len(d) / t for d, t in zip(per_run, train_sizes, strict=True))
            corrected = (1 / k + ratio) * statistics.variance(estimates)
            phalf = _t(k - 1) * math.sqrt(corrected)
            procedure[m] = Difference(mean, (mean - phalf, mean + phalf), n)
    return Pooled(
        runs=k, cases=sum(len(r.test.cases) for r in runs if r.test), difference=difference, procedure=procedure
    )


__all__ = ["Arm", "Difference", "Pooled", "Result", "Run", "Test"]
