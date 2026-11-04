"""Reward first, then cost: how a goal with a cost objective searches by default (``PrimaryFirst``).

A goal says what is wanted (``Pareto(Maximize("reward"), Minimize("tokens"))``, or
``CheapestOnFront(Minimize("tokens"), Maximize("reward"))`` with a floor on the reward); a schedule says how the search
gets there. Told both objectives from the first step, a search spends its budget on easy cost cuts (cheap harnesses
that fail), and its incumbents and finalists come from there, so it can ship a cheap harness that has lost much of the
reward a max-reward search reaches. ``PrimaryFirst`` searches in two phases instead:

- **Phase A** ("primary") is a search for the primary objective alone, exactly as a max-reward study searches: its
  methods are told the primary objective, and its incumbent is the best on it. It ends when the search stops gaining
  (``Stall``: some steps in a row in which no candidate beats the incumbent it was proposed from, paired over the same
  cases; only once the incumbent is within the stall's headroom of the metric's best), when a goal with constraints
  meets them (``until_met``: the floor is met with confidence), at its cap (a share of the run's steps), or when the run
  has nothing left. Its incumbent is the **anchor**, A*: the best candidate on the primary objective.
- **Phase B** ("cost") spends the rest of the same budget lowering the cost from A*: its methods are told the cost,
  with the constraint that a candidate be no worse than A* (``Cheaper``), and its incumbent is the cheapest candidate
  that is, starting at A*. "No worse" is judged on the paired difference over the cases both measured
  (``Evidence``): with a ``margin`` δ, P(primary_c − primary_A* ≥ −δ) ≥ ``confidence``; a goal's own constraint on the
  primary metric (a floor) is judged as A*'s value plus that paired difference, so the uncertainty of A*'s own level,
  which every candidate shares, does not hide which candidates keep it.

The selection is anchored too (``PrimaryFirst.decide``): the finalists are A* and the cheapest candidates no worse
than it; on validation, a goal with one best ships the goal's choice among those that keep A*'s level (judged as in
phase B) and pass the seesaw guard (``Seesaw``), else the closest (usually A*); a Pareto goal keeps its validated front,
as before, and ships one member of it, the pick: the cheapest validated candidate within the margin of the best
validated value of the primary objective that passes the guard (the cheapest-within-margin pick rule, with the
guard, and with A* always validated). A study therefore never ships something worse than its own phase-A result by
more than its margin (as validated). What other rules would have chosen from the same candidates is recorded
(``Selection.also``).

The primary objective is the first declared one that is not a cost; the cost, the first that is (a cost is a minimized
usage metric, such as tokens, or a derived metric with no upper bound: ``Projection.cost_type``). A goal with no cost
objective, or nothing but costs, searches as before. ``Search(phases=None)`` searches every goal without phases (its
methods told the goal as declared from the first step).

The record states it all: the schedule (its ``describe``), each phase's scope (``phase``), each candidate's phase, the
``anchor`` event (why phase A ended, after how many steps and candidates, the evidence of each of its steps, A*, and the
cost spent by then), each phase-B candidate's paired evidence (``anchored``), and the selection's anchored evidence,
pick and the other rules' choices.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .projection import AtLeast, Combine, Maximize, Projection, _feasible_first, constraint_name
from .schedules import Schedule, default_schedule

if TYPE_CHECKING:
    from .selection import Closest, View
    from .study import Candidate, Scope, Study

PHASES = ("primary", "cost")
EXACT = 1e-12


def _t_cdf(x: float, df: int) -> float:
    from scipy.stats import t  # optional extra: optimization

    return float(t.cdf(x, df))


def _t_ppf(q: float, df: int) -> float:
    from scipy.stats import t  # optional extra: optimization

    return float(t.ppf(q, df))


# ---- paired evidence -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Paired:
    """One candidate's metric minus another's over the cases both measured, each case's mean over its repetitions:
    the mean difference, its standard error (the differences' sd over the root of their number), and the number of
    cases. Probabilities are a t over the cases (n - 1 degrees of freedom); with no spread, 1 or 0."""

    difference: float
    se: float
    cases: int

    @property
    def spread(self) -> bool:
        return self.cases >= 2 and self.se > 0

    def probability(self, at_least: float) -> float:
        """P(the mean difference over the cases is at least ``at_least``)."""
        if not self.spread:
            return 1.0 if self.difference >= at_least - EXACT else 0.0
        return _t_cdf((self.difference - at_least) / self.se, self.cases - 1)

    def beats(self) -> float:
        """P(the mean difference is above 0): the candidate is better."""
        if not self.spread:
            return 1.0 if self.difference > EXACT else 0.0
        return _t_cdf(self.difference / self.se, self.cases - 1)

    def quantile(self, q: float) -> float:
        if not self.spread:
            return self.difference
        return self.difference + _t_ppf(q, self.cases - 1) * self.se

    def to_dict(self) -> dict[str, Any]:
        return {"difference": self.difference, "se": self.se, "cases": self.cases}


def paired(projection: Projection, a: Sequence[Any], b: Sequence[Any], metric: str) -> Paired | None:
    """``a``'s trials minus ``b``'s on ``metric``, over the cases both measured (None when they share none)."""
    x, y = projection.case_means(a, metric), projection.case_means(b, metric)
    cases = sorted(set(x) & set(y))
    if not cases:
        return None
    d = [x[c] - y[c] for c in cases]
    se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) >= 2 else 0.0
    return Paired(statistics.fmean(d), se, len(d))


# ---- policies -------------------------------------------------------------------------------------------------------
class Stall:
    """Phase A's stopping rule: phase A ends after ``steps`` steps in a row in which no candidate beats the incumbent it
    was proposed from on the primary metric with probability ``probability``, paired over the search cases.

    ``per``: ``"step"`` (default) asks it of the step: its best candidate beats its incumbent with probability at
    least ``probability ** (1 / m)``, m the step's candidates, so a step of candidates no better than their incumbent
    shows a gain with probability 1 - ``probability`` whatever its width (Šidák); ``"candidate"`` asks it of each
    candidate (a step of 4 equal candidates then shows a gain with probability 0.59). A round's opening proposal is a
    step of one.
    ``headroom``: the rule applies only once the incumbent is within this share of a bounded metric's range of its best
    end (0.25: a reward of at least 0.75); below that, phase A goes on to its cap, where gains can keep arriving late
    (a search far from its metric's ceiling may gain several points after the rule would have stopped it). None:
    always."""

    def __init__(
        self, steps: int = 2, probability: float = 0.8, per: str = "step", headroom: float | None = 0.25
    ) -> None:
        if steps < 1:
            raise ValueError("Stall(steps=...) must be at least 1")
        if not 0.5 <= probability < 1:
            raise ValueError("Stall(probability=...) needs 0.5 <= probability < 1")
        if per not in ("step", "candidate"):
            raise ValueError('Stall(per=...) is "step" or "candidate"')
        if headroom is not None and not 0 < headroom <= 1:
            raise ValueError("Stall(headroom=...) is a share of the metric's range: 0 < headroom <= 1, or None")
        self.steps, self.probability, self.per, self.headroom = steps, probability, per, headroom

    def threshold(self, m: int) -> float:
        return self.probability ** (1 / max(m, 1)) if self.per == "step" else self.probability

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "Stall",
            "steps": self.steps,
            "probability": self.probability,
            "per": self.per,
            "headroom": self.headroom,
        }


class Seesaw:
    """The seesaw guard: a candidate may lose at most ``k`` of the cases its reference solves on every repetition (a
    case is lost when the candidate solves it on at most half of its own repetitions). On a validation set near its
    ceiling, a candidate can match its reference on the mean while trading cases the reference solves every time for
    others, and then fall well behind it on test. Solving is reaching the best end of the metric's range, so the guard
    needs a bounded primary metric (it passes everything otherwise). k = 1 allows for noise: a configuration measured
    twice on the same validation cases typically loses 0 or 1 such cases against itself."""

    def __init__(self, k: int = 1) -> None:
        if k < 0:
            raise ValueError("Seesaw(k=...) must be 0 or more")
        self.k = k

    def lost(
        self, projection: Projection, candidate: Sequence[Any], reference: Sequence[Any], metric: str, maximize: bool
    ) -> tuple[int, int] | None:
        """How many of the reference's always-solved cases the candidate loses, and how many there are (None for an
        unbounded metric)."""
        lo, hi = projection.range(metric)
        if lo is None or hi is None:
            return None
        top = hi if maximize else lo

        def solved(v: float) -> bool:
            return v >= top - 1e-9 if maximize else v <= top + 1e-9

        def per_case(trials: Sequence[Any]) -> dict[str, list[float]]:
            out: dict[str, list[float]] = defaultdict(list)
            for t in trials:
                v = projection.value(t, metric)
                if v is not None:
                    out[t.case].append(v)
            return out

        ref, cand = per_case(reference), per_case(candidate)
        robust = [c for c, vs in ref.items() if vs and all(solved(v) for v in vs)]
        lost = sum(1 for c in robust if cand.get(c) and sum(solved(v) for v in cand[c]) <= len(cand[c]) / 2)
        return lost, len(robust)

    def describe(self) -> dict[str, Any]:
        return {"kind": "Seesaw", "k": self.k}


STALL = Stall()  # PrimaryFirst's defaults
GUARD = Seesaw(1)


# ---- the anchor ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Evidence:
    """A candidate against its reference (the anchor, A*), paired over the cases both measured: the primary metric's
    difference; whether that keeps within the margin (P(difference >= -margin) >= confidence; always with no margin);
    and, where the guard was asked, the seesaw count."""

    reference: int
    metric: str
    pair: Paired | None
    margin: float | None
    probability: float | None  # P(difference >= -margin)
    within: bool
    violation: float | None  # -margin less the difference at the (1 - confidence) quantile: <= 0 when within
    lost: int | None = None
    robust: int | None = None
    guard: bool | None = None  # the seesaw guard passed (None: not asked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "metric": self.metric,
            **(self.pair.to_dict() if self.pair else {"difference": None, "se": None, "cases": 0}),
            "margin": self.margin,
            "p": self.probability,
            "within": self.within,
            **({"lost": self.lost, "robust": self.robust, "guard": self.guard} if self.guard is not None else {}),
        }


@dataclass(frozen=True)
class Anchor:
    """Phase A's result, A* (candidate ``number``), and how later decisions are judged against it: the primary
    ``metric`` (``maximize``), the non-inferiority ``margin`` (None: none of its own; a goal's constraint on the metric
    still holds), the ``confidence`` it needs, the selection's ``feasibility``, and the seesaw ``guard``."""

    number: int
    metric: str
    maximize: bool
    margin: float | None
    confidence: float
    feasibility: float
    guard: Seesaw | None

    def evidence(self, projection: Projection, view: "View", reference: "View", guard: bool) -> Evidence:
        pair = (
            Paired(0.0, 0.0, len({t.case for t in reference.trials}))
            if view.number == reference.number
            else paired(projection, view.trials, reference.trials, self.metric)
        )
        if pair is not None and not self.maximize:  # oriented: positive is better
            pair = Paired(-pair.difference, pair.se, pair.cases)
        if self.margin is None:
            probability, within, violation = None, pair is not None, None
        elif pair is None:
            probability, within, violation = 0.0, False, None
        else:
            probability = pair.probability(-self.margin)
            within = probability >= self.confidence - EXACT
            violation = -self.margin - pair.quantile(1 - self.confidence)
        lost = robust = ok = None
        if guard and self.guard is not None:
            counted = self.guard.lost(projection, view.trials, reference.trials, self.metric, self.maximize)
            if counted is None:
                ok = True
            else:
                lost, robust = counted
                ok = lost <= self.guard.k
        return Evidence(
            reference.number, self.metric, pair, self.margin, probability, within, violation, lost, robust, ok
        )

    def rejudge(self, projection: Projection, view: "View", reference: "View", guard: bool = False) -> "View":
        """``view`` judged against the anchor's ``reference`` (its view on the same cases): its evidence, and every
        constraint on the anchor's metric (absolute, or a difference from the baseline) judged as the reference's
        decided value plus the paired difference, with the difference's uncertainty (the reference's own, which every
        candidate shares, is left out). The reference meets such a constraint exactly when its decided value does."""
        evidence = self.evidence(projection, view, reference, guard)
        violations, probabilities = list(view.violations), list(view.p_met)
        raw = (
            paired(projection, view.trials, reference.trials, self.metric)
            if view.number != reference.number
            else Paired(0.0, 0.0, evidence.pair.cases if evidence.pair else 0)
        )
        for i, c in enumerate(projection.constraints):
            if c.metric != self.metric or c.compare == "ratio" or raw is None:
                continue
            level = reference.values[i] if i < len(reference.values) else None
            if level is None:
                continue
            if isinstance(c, AtLeast):
                violations[i] = c.limit - (level + raw.quantile(1 - self.feasibility))
                p = raw.probability(c.limit - level)
            else:
                violations[i] = (level + raw.quantile(self.feasibility)) - c.limit
                p = (
                    1.0 - raw.probability(c.limit - level)
                    if raw.spread
                    else float(level + raw.difference <= c.limit + EXACT)
                )
            if i < len(probabilities):
                probabilities[i] = p
        return dataclasses.replace(view, violations=tuple(violations), p_met=tuple(probabilities), anchored=evidence)

    def describe(self) -> dict[str, Any]:
        return {
            "candidate": self.number,
            "metric": self.metric,
            "margin": self.margin,
            "confidence": self.confidence,
            "feasibility": self.feasibility,
            "guard": self.guard.describe() if self.guard else None,
        }


# ---- what each phase searches for ------------------------------------------------------------------------------------
class Primary(Combine):
    """Phase A's aim: the primary objective alone (objective ``index`` of the goal's), as a study of that objective
    alone searches: methods are told it, the incumbent is the best on it (among those that meet the goal's
    constraints, else those that miss them least), and a rewriter is shown it and the constraints' metrics."""

    def __init__(self, goal: Combine, index: int) -> None:
        super().__init__(*goal.objectives)
        self.index = index

    def settings(self) -> dict[str, Any]:
        return {"objective": self.objectives[self.index].metric}

    def focus(self, projection: Projection, k: int | None) -> list[int]:
        return [self.index]

    def told_names(self, projection, k):
        return [projection.objectives[self.index].metric]

    def directions(self, projection, k):
        return [projection.directions[self.index]]

    def told(self, projection, candidate, k, baseline):
        return [projection.values(candidate)[self.index]]

    def held(self, projection, pool, k, baseline):
        candidates = _feasible_first(pool)
        return max(candidates, key=lambda t: projection.oriented(t)[self.index]) if candidates else None

    def best(self, projection, candidates, baseline):
        feasible = [t for t in candidates if t.eligible and t.feasible]
        held = self.held(projection, feasible, None, baseline)
        return [held] if held is not None else []

    def scalar(self, scores, objectives, k):
        return scores[self.index]

    def shown(self, projection):
        own = projection.objectives[self.index].metric
        return list(dict.fromkeys([own, *(c.metric for c in projection.constraints)]))

    def per_trial(self, projection):
        return [projection.objectives[self.index].metric]


class Cheaper(Combine):
    """Phase B's aim: the cheapest candidate no worse than the anchor. Methods are told the cost (objective ``cost``),
    with the constraint that the candidate keep the anchor's primary objective (``primary``): within the margin, by
    the paired evidence (``Evidence``), and meeting the goal's constraints as rejudged against the anchor. The
    incumbent is the cheapest candidate that does (ties: the better primary value), and none when none does, so the
    search holds the anchor rather than a cheaper candidate that lost what phase A found. A rewriter is shown both,
    ranked so, with every case's cost in its digest."""

    waits = True  # its methods hear nothing until there is an anchor to judge against (Study._evidence)

    def __init__(self, goal: Combine, cost: int, primary: int) -> None:
        super().__init__(*goal.objectives)
        self.cost, self.primary = cost, primary
        self.anchor: Anchor | None = None  # set when phase B starts

    def settings(self) -> dict[str, Any]:
        return {"cost": self.objectives[self.cost].metric, "primary": self.objectives[self.primary].metric}

    def focus(self, projection: Projection, k: int | None) -> list[int]:
        return [self.cost]

    def qualifies(self, view: Any) -> bool:
        evidence = getattr(view, "anchored", None)
        return bool(view.eligible and view.feasible and evidence is not None and evidence.within)

    def told_names(self, projection, k):
        return [projection.objectives[self.cost].metric]

    def directions(self, projection, k):
        return [projection.directions[self.cost]]

    def told(self, projection, candidate, k, baseline):
        return [projection.values(candidate)[self.cost]]

    def held(self, projection, pool, k, baseline):
        keep = [v for v in pool if self.qualifies(v)]
        if not keep:
            return None
        return min(
            keep, key=lambda t: (-projection.oriented(t)[self.cost], -projection.oriented(t)[self.primary], t.number)
        )

    def best(self, projection, candidates, baseline):
        held = self.held(projection, candidates, None, baseline)
        return [held] if held is not None else []

    def scalar(self, scores, objectives, k):
        return scores[self.cost]

    def _margin_name(self, projection: Projection) -> str:
        a = self.anchor
        metric = projection.objectives[self.primary].metric
        at = f"candidate {a.number}'s" if a is not None else "the anchor's"
        margin = a.margin if a is not None else None
        p = a.confidence if a is not None else None
        return (
            f"{metric} no worse than {at} by more than {margin:g} (paired over the cases, P >= {p:g})"
            if margin is not None and p is not None
            else f"{metric} kept at {at} (paired over the cases)"
        )

    def told_constraints(self, projection, view):
        out = super().told_constraints(projection, view)
        evidence = getattr(view, "anchored", None)
        if self.anchor is not None and self.anchor.margin is not None:
            violation = evidence.violation if evidence is not None and evidence.violation is not None else 1.0
            out[self._margin_name(projection)] = violation
        return out

    def limits(self, projection):
        own = [
            f"{constraint_name(c)} (judged as candidate {self.anchor.number}'s value plus the paired difference)"
            if self.anchor is not None and c.metric == self.anchor.metric and c.compare != "ratio"
            else constraint_name(c)
            for c in projection.constraints
        ]
        return [*own, self._margin_name(projection)]

    def meets(self, projection, view):
        return self.qualifies(view)

    def digest_costs(self, projection):
        return [projection.objectives[self.cost].metric]


# ---- the selection's result ------------------------------------------------------------------------------------------
@dataclass
class Decision:
    """What an anchored selection ships: ``best`` (the goal's best: one candidate, or a Pareto goal's validated front
    with the pick), ``pick`` (the one to ship), ``unconfirmed`` (nothing met the constraints: the closest), the views
    it judged (``views``), and what each of ``Selection.also``'s rules would ship (``alternatives``: numbers; [] ships
    the baseline, as with ``unmet="none"``)."""

    best: list["Candidate"]
    pick: "Candidate | None"
    unconfirmed: "Closest | None"
    views: list["View"]
    alternatives: dict[str, list[int]]
    reference: int | None = None  # the pick's reference (a Pareto goal: the best validated on the primary)


# ---- the schedule ----------------------------------------------------------------------------------------------------
class PrimaryFirst(Schedule):
    """Search the primary objective first (phase A), then the cost from its result, A* (phase B), in one budget; see
    the module's docstring.

    ``inner``: what each phase runs (default: the schedule the search runs without phases, built from the space).
    ``stall``: phase A's stopping rule (``Stall()`` by default; None: none). ``cap``: phase A's most steps, a share of
    the run's steps (default 2/3) or a number of steps. ``until_met``: phase A also ends once its incumbent meets the
    goal's constraints (a floor), when the goal has some.
    ``margin``: phase B's non-inferiority margin on the primary objective, in its units (None: 0.02 of a bounded
    metric's range for a goal with no constraint on the primary metric, none for a goal with one, whose constraint is
    judged against A* instead). ``confidence``: the probability non-inferiority needs (None: the selection's
    feasibility). ``pick``: what a Pareto goal's pick asks of "within the margin": None (default), the decided values;
    a probability, the paired evidence at that probability. ``guard``: the seesaw guard on
    validation (``Seesaw(1)`` by default; None: none).
    ``primary``, ``cost``: the objectives' metrics, when not the rule's (``Projection.primary_and_cost``)."""

    def __init__(
        self,
        inner: Schedule | None = None,
        *,
        stall: Stall | None = STALL,
        cap: float | int = 2 / 3,
        until_met: bool = True,
        margin: float | None = None,
        confidence: float | None = None,
        pick: float | None = None,
        guard: Seesaw | None = GUARD,
        primary: str | None = None,
        cost: str | None = None,
    ) -> None:
        if inner is not None and not isinstance(inner, Schedule):
            raise TypeError(f"PrimaryFirst(inner=...) takes a Schedule, not {inner!r}")
        if isinstance(cap, bool) or not (isinstance(cap, int) and cap >= 1 or isinstance(cap, float) and 0 < cap < 1):
            raise ValueError("PrimaryFirst(cap=...) is a share of the run's steps (0 < cap < 1) or a number of steps")
        if margin is not None and margin < 0:
            raise ValueError("PrimaryFirst(margin=...) must be 0 or more")
        for name, p in (("confidence", confidence), ("pick", pick)):
            if p is not None and not 0.5 <= p < 1:
                raise ValueError(f"PrimaryFirst({name}=...) needs 0.5 <= {name} < 1")
        self.inner = inner
        self.stall = stall
        self.cap, self.until_met, self.margin, self.confidence = cap, until_met, margin, confidence
        self.pick, self.guard = pick, guard
        self.primary, self.cost = primary, cost
        self.indices: tuple[int, int] | None = None  # bound to a study: (primary, cost)
        self._aims: list[Combine] | None = None
        self.ended: dict[str, Any] | None = None  # how phase A ended, once it has

    # ---- binding to a study ----
    def bind(self, study: "Study", inner: Schedule) -> "PrimaryFirst":
        """This schedule for ``study``: around ``inner`` when none was given, with its objectives and margin resolved
        (raises when the goal has no primary and cost objective)."""
        p = study.projection
        found = p.primary_and_cost()
        metrics = [o.metric for o in p.objectives]
        if self.primary is not None or self.cost is not None:
            try:
                i = metrics.index(self.primary) if self.primary is not None else found[0]  # type: ignore[index]
                j = metrics.index(self.cost) if self.cost is not None else found[1]  # type: ignore[index]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"PrimaryFirst: no objective {self.primary or self.cost!r} in {metrics}") from exc
            found = (i, j)
        if found is None or found[0] == found[1]:
            raise ValueError(
                f"PrimaryFirst needs a goal with a primary objective and a cost objective (a minimized usage or "
                f"unbounded derived metric); the objectives are {metrics}"
            )
        if study.combine.instances(len(p.objectives)) > 1:
            raise ValueError("PrimaryFirst does not run under Independent, which searches each objective on its own")
        bound = copy.copy(self)  # the settings as given stay as given: a Search can be used again
        bound.inner = self.inner if self.inner is not None else inner
        bound.indices = found
        primary = p.objectives[found[0]]
        if bound.margin is None and not any(c.metric == primary.metric for c in p.constraints):
            lo, hi = p.range(primary.metric)
            bound.margin = 0.02 * (hi - lo) if lo is not None and hi is not None and hi > lo else 0.0
        if bound.confidence is None:
            bound.confidence = study.selection.feasibility
        bound._aims = [Primary(study.combine, found[0]), Cheaper(study.combine, found[1], found[0])]
        bound.ended = None
        return bound

    def aims(self, study: "Study") -> list[Combine] | None:
        return self._aims

    def children(self) -> dict[str, Schedule]:
        return {name: self.inner for name in PHASES} if self.inner is not None else {}

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        if self.inner is not None:
            self.inner.check(blocks, study)

    def describe(self, study: "Study") -> dict[str, Any]:
        p = study.projection
        i, j = self.indices if self.indices else (None, None)
        return {
            "kind": "PrimaryFirst",
            "primary": p.objectives[i].metric if i is not None else self.primary,
            "cost": p.objectives[j].metric if j is not None else self.cost,
            "stall": self.stall.describe() if self.stall else None,
            "cap": self.cap,
            "until_met": self.until_met,
            "margin": self.margin,
            "confidence": self.confidence,
            "pick": {"kind": "within the margin", "by": "decided" if self.pick is None else "paired", "p": self.pick},
            "guard": self.guard.describe() if self.guard else None,
        }

    # ---- running ----
    def _cap_steps(self, steps: int) -> int:
        if isinstance(self.cap, int):
            return max(1, min(self.cap, steps))
        return max(1, min(steps, round(self.cap * steps)))

    def run(self, scope: "Scope") -> None:
        study = scope.study
        assert self.indices is not None and self._aims is not None, "PrimaryFirst runs bound to its study (bind)"
        i, _ = self.indices
        steps = scope.budget
        cap = self._cap_steps(steps)
        cost_aim = self._aims[1]
        assert isinstance(cost_aim, Cheaper)
        study.anchor = None  # a new run (or a study's next optimize) finds its own
        study._listening[1] = False  # phase B's methods learn nothing until A* exists, then everything at once
        until = _Until(self, study)
        with scope.child("primary", self.inner, blocks=scope.blocks, limit=cap, objective=0, phase="primary") as a:
            a.stop = until
            self.inner.run(a)
            spent = a.used
            until.absorb(a)  # the steps no budget check asked about (the last, at the cap)
        anchor = study._held(study.candidates, 0) or study.baseline
        if until.reason is not None:
            reason = until.reason
        elif scope.budget <= 0:
            reason = "budget"
        elif spent >= cap * study.search.width:
            reason = "cap"
        else:
            reason = "exhausted"
        primary = study.projection.objectives[i]
        a_star = Anchor(
            number=anchor.number,
            metric=primary.metric,
            maximize=isinstance(primary, Maximize),
            margin=self.margin,
            confidence=float(self.confidence or study.selection.feasibility),
            feasibility=study.selection.feasibility,
            guard=self.guard,
        )
        self.ended = {"reason": reason, "steps": len(a.history), "candidates": len(a.candidates), "steps_cap": cap}
        cost_aim.anchor = a_star
        study._anchor(a_star, phase=self.ended, steps=until.log)
        study._evidence(1)
        if scope.budget <= 0:
            return
        with scope.child(
            "cost", self.inner, blocks=scope.blocks, limit=scope.budget, objective=1, phase="cost", seed=anchor
        ) as b:
            self.inner.run(b)

    # ---- selecting ----
    def finalists(self, study: "Study", n: int) -> list["Candidate"] | None:
        """A* and the cheapest candidates no worse than it (phase B's ranking, every candidate measured), ``n`` in all;
        when fewer than that are no worse than it with confidence, the cheapest whose measured difference is within
        the margin fill the rest."""
        if study.anchor is None or self._aims is None:
            return None
        a = study.candidates[study.anchor.number]
        out = [a]
        rest = [c for c in study.candidates if c.eligible and c.number != a.number]
        ranked = study._ranked(rest, n - 1, aim=1)
        out += ranked
        if len(out) < n:
            taken = {c.number for c in out}
            views = study._views([c for c in rest if c.number not in taken])
            margin = study.anchor.margin or 0.0
            close = [
                v
                for v in views
                if v.feasible
                and v.anchored is not None
                and v.anchored.pair is not None
                and v.anchored.pair.difference >= -margin - EXACT
            ]
            _, j = self.indices  # type: ignore[misc]
            close.sort(key=lambda v: (-study.projection.oriented(v)[j], v.number))
            out += [v.candidate for v in close[: n - len(out)]]
        return out

    def decide(self, study: "Study", views: list["View"], baseline: "View", on: str) -> Decision | None:
        """The anchored selection from ``views`` (the baseline and the finalists on validation, or every candidate on
        the search cases), or None when the run has no anchor among them (the goal's own selection applies)."""
        anchor = study.anchor
        if anchor is None or self.indices is None:
            return None
        reference = next((v for v in views if v.number == anchor.number), None)
        if reference is None:
            return None
        p = study.projection
        judged = [anchor.rejudge(p, v, reference, guard=True) if v.eligible else v for v in views]
        shippable = [v for v in judged if v.anchored is None or v.anchored.guard is not False]
        unconfirmed: Any = None
        if study.combine.best_is_set(len(p.objectives)):
            front = study.combine.best(p, judged, baseline)
            pick_view, ref = self._pick_of(study, judged, guard=True)
            if pick_view is None:  # nothing feasible: the closest, as for every goal (none with unmet="none")
                closest = study._closest(shippable, baseline, on=on)
                pick_view = next((v for v in judged if closest and v.number == closest[0].number), None)
                unconfirmed = study._unconfirmed
            best = [v.candidate for v in front]
            if pick_view is not None and pick_view.number not in {c.number for c in best}:
                best.append(pick_view.candidate)
            pick = pick_view.candidate if pick_view is not None else None
        else:
            chosen = [v.candidate for v in study.combine.best(p, shippable, baseline)]
            if not chosen:  # nothing keeps the goal's constraints: the closest (none with unmet="none")
                chosen = study._closest(shippable, baseline, on=on)
                unconfirmed = study._unconfirmed
            best, ref = chosen[:1], anchor.number
            pick = chosen[0] if chosen else None
        alternatives: dict[str, list[int]] = {}
        for rule in study.selection.also:
            if rule == "unmet=none":
                alternatives[rule] = [] if unconfirmed is not None or pick is None else [pick.number]
            elif rule == "anchor":
                alternatives[rule] = [anchor.number]
            elif rule == "unanchored":
                alternatives[rule] = self._unanchored(study, views, baseline)
        shipped = study.candidates[pick.number] if pick is not None else None
        return Decision(best, shipped, unconfirmed, judged, alternatives, ref)

    def _pick_of(
        self, study: "Study", views: Sequence["View"], *, guard: bool, members: set[int] | None = None
    ) -> tuple["View | None", int | None]:
        """A Pareto goal's pick: of the feasible ``views`` (``members`` among them), the reference is the best on the
        primary objective (ties: A*, then the cheaper); within the margin are those whose decided primary value is no
        more than the margin below the reference's (``pick=None``) or whose paired difference is at least -margin with
        probability ``pick``; with ``guard``, the seesaw guard against A* (the views' evidence) drops more; the pick is
        the cheapest left (ties: the better primary value, then the earlier number)."""
        p = study.projection
        i, j = self.indices  # type: ignore[misc]
        anchor = study.anchor
        assert anchor is not None
        pool = [v for v in views if v.eligible and v.feasible and (members is None or v.number in members)]
        if not pool:
            return None, None
        ref = max(pool, key=lambda v: (p.oriented(v)[i], v.number == anchor.number, p.oriented(v)[j], -v.number))
        margin = self.margin or 0.0
        kept = []
        for v in pool:
            if v.number == ref.number:
                if not (guard and v.anchored is not None and v.anchored.guard is False):
                    kept.append(v)
                continue
            if self.pick is None:
                within = p.oriented(v)[i] >= p.oriented(ref)[i] - margin - EXACT
            else:
                pair = paired(p, v.trials, ref.trials, anchor.metric)
                if pair is not None and not anchor.maximize:
                    pair = Paired(-pair.difference, pair.se, pair.cases)
                within = pair is not None and pair.probability(-margin) >= self.pick - EXACT
            if within and guard and v.anchored is not None:
                within = v.anchored.guard is not False
            if within:
                kept.append(v)
        if not kept:  # the reference itself trades A*'s cases: A*, which keeps its own
            kept = [v for v in pool if v.number == anchor.number] or [ref]
        pick = min(kept, key=lambda v: (-p.oriented(v)[j], -p.oriented(v)[i], v.number))
        return pick, ref.number

    def _unanchored(self, study: "Study", views: list["View"], baseline: "View") -> list[int]:
        """What the unanchored rules would ship from the same candidates: the goal's own selection without the anchor
        (constraints judged as the model decides them, no guard, nothing when none meets them); a Pareto goal's pick
        by the cheapest-within-margin rule (within the margin of the best by decided values, the baseline a member only
        when it was a finalist; no guard)."""
        p = study.projection
        if study.combine.best_is_set(len(p.objectives)):
            finalists = set(study._finalist_numbers)
            members = {v.number for v in views if v.number != study.baseline.number or v.number in finalists}
            saved, self.pick = self.pick, None  # the unanchored rule decides by the decided values
            try:
                pick, _ = self._pick_of(study, views, guard=False, members=members)
            finally:
                self.pick = saved
            return [pick.number] if pick is not None else []
        best = study.combine.best(p, views, baseline)
        return [best[0].number] if best else []


class _Until:
    """Phase A's end, asked before each of its steps (``Scope.stop``): the stall rule, or the goal's constraints met;
    ``log`` keeps each step's evidence for the record."""

    def __init__(self, schedule: PrimaryFirst, study: "Study") -> None:
        self.schedule, self.study = schedule, study
        self.reason: str | None = None
        self.log: list[dict[str, Any]] = []
        self._seen = 0

    def __call__(self, scope: "Scope") -> bool:
        if self.reason is not None:
            return True
        if len(scope.history) == self._seen:
            return False
        self.absorb(scope)
        return self._ends(scope)

    def absorb(self, scope: "Scope") -> None:
        """Log the evidence of every step of ``scope`` not yet logged (also once the phase has ended, by its cap or
        the run's budget, when no step after it asked)."""
        study, rule = self.study, self.schedule.stall
        p = study.projection
        i, _ = self.schedule.indices  # type: ignore[misc]
        metric = p.objectives[i].metric
        by_number = {c.number: c for c in study.candidates}
        for base, numbers in scope.history[self._seen :]:
            best, best_n = -1.0, None  # the step's most likely gain (its candidate is named even at 0)
            b = by_number.get(base) if base is not None else None
            for n in numbers:
                c = by_number[n]
                if b is None or not c.eligible or not b.eligible:
                    continue
                pair = paired(p, c.trials, b.trials, metric)
                if pair is None:
                    continue
                if not isinstance(p.objectives[i], Maximize):
                    pair = Paired(-pair.difference, pair.se, pair.cases)
                if pair.beats() > best:
                    best, best_n = pair.beats(), n
            threshold = rule.threshold(len(numbers)) if rule else None
            self.log.append(
                {
                    "base": base,
                    "candidates": list(numbers),
                    "best": best_n,
                    "p": max(best, 0.0),
                    "threshold": threshold,
                    "gain": bool(threshold is not None and best_n is not None and best >= threshold - EXACT),
                }
            )
        self._seen = len(scope.history)

    def _ends(self, scope: "Scope") -> bool:
        study, rule = self.study, self.schedule.stall
        p = study.projection
        i, _ = self.schedule.indices  # type: ignore[misc]
        metric = p.objectives[i].metric
        held = study._held(study.candidates, 0)
        if self.schedule.until_met and p.constraints and held is not None and study.view(held).feasible:
            self.reason = "met"
            return True
        if rule is None or len(self.log) < rule.steps:
            return False
        if rule.headroom is not None and held is not None:
            lo, hi = p.range(metric)
            if lo is not None and hi is not None and hi > lo:
                value = study.view(held).metrics.get(metric)
                maximize = isinstance(p.objectives[i], Maximize)
                if value is not None and (hi - value if maximize else value - lo) > rule.headroom * (hi - lo):
                    return False
        if not any(s["gain"] for s in self.log[-rule.steps :]):
            self.reason = "stall"
            return True
        return False


def phased(study: "Study", schedule: Schedule) -> Schedule:
    """The schedule a study runs: ``schedule`` (its search's, or the default one), inside ``PrimaryFirst`` when the
    search's ``phases`` say so (``"auto"``: when the goal has a primary and a cost objective, and the search gives no
    schedule of its own; a ``PrimaryFirst``: always; None: never)."""
    phases = study.search.phases
    if isinstance(schedule, PrimaryFirst):  # the search's own schedule
        return schedule.bind(study, schedule.inner or default_schedule(study))
    if phases is None:
        return schedule
    if isinstance(phases, PrimaryFirst):
        return phases.bind(study, schedule)
    if phases != "auto":
        raise ValueError(f'Search(phases=...) is "auto", None or a PrimaryFirst(...), not {phases!r}')
    if study.search.schedule is not None:
        return schedule
    if study.projection.primary_and_cost() is None or study.combine.instances(len(study.projection.objectives)) > 1:
        return schedule
    return PrimaryFirst().bind(study, schedule)


__all__ = [
    "Anchor",
    "Cheaper",
    "Decision",
    "Evidence",
    "Paired",
    "Primary",
    "PrimaryFirst",
    "Seesaw",
    "Stall",
    "paired",
]
