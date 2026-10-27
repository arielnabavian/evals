"""How a study decides: with the model of each goal metric (``model.py``), or with measured means.

Every combination of objectives (``Pareto``, ``WeightedSum``, ...) works on a candidate's values and on whether it
meets the constraints. Deciding by the model, it works on a *view* of each candidate: its decided values (the model's
posterior mean, or a cautious quantile of it, per ``Selection.rule``) and its constraints judged at the probability
``Selection.feasibility``.

When a run ends and no candidate it selects from meets the constraints, it ships the one that comes closest to meeting
them (``closest``), unconfirmed, unless ``Selection(unmet="none")``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .model import Decided, Fit, constraint_value, p_met
from .projection import AtMost, Combine, Maximize, Projection, _excess

if TYPE_CHECKING:
    from .study import Candidate

UNMET = ("closest", "none")  # what a run ships when no candidate meets the constraints (Selection.unmet)
ALSO = ("unmet=none", "unanchored", "anchor")  # rules recorded (and tested) beside the selection (Selection.also)
P_RESOLUTION = 0.01  # probabilities of meeting the constraints equal at this resolution tie (``closest``)


@dataclass(frozen=True)
class Mean:
    """Decide with the model's posterior mean of each objective."""

    def value(self, decided: Decided, maximize: bool) -> float:
        return decided.mean

    def describe(self) -> dict[str, Any]:
        return {"kind": "Mean"}


@dataclass(frozen=True)
class LowerBound:
    """Decide cautiously: each objective's ``q``-quantile on its worse side (the lower one for a maximized metric)."""

    q: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.q < 0.5:
            raise ValueError("LowerBound(q) needs 0 < q < 0.5")

    def value(self, decided: Decided, maximize: bool) -> float:
        return decided.quantile(self.q if maximize else 1 - self.q)

    def describe(self) -> dict[str, Any]:
        return {"kind": "LowerBound", "q": self.q}


@dataclass(frozen=True)
class Selection:
    """How the study decides and selects.

    ``by="model"`` (default) decides with the model of every goal metric; ``by="measured"`` with measured means, as
    before the model.
    ``rule``: ``Mean()`` or ``LowerBound(q)``, what an objective's decided value is.
    ``feasibility``: the probability with which every constraint must hold.
    ``finalists``: when a run ends, the best (a Pareto set's members, up to ``max(finalists, 5)``) are measured again,
    the study's ``repetitions`` times, on the validation cases (or, without them, on the search cases) before
    selecting; 0 selects at once.
    ``unmet``: what a run ships when no candidate it selects from (on validation: the baseline and the finalists)
    meets the constraints. ``"closest"`` (default): the one that comes closest to meeting them (``closest``), marked
    unconfirmed in the record and the result. ``"none"``: nothing, so the test measures the baseline alone (the
    behaviour before ``unmet``).
    ``also``: other rules whose choice, from the same candidates, is recorded beside the selection and measured on
    the test cases too when it differs from every arm already tested (``ALSO``; they apply when a schedule anchors
    the selection, as ``PrimaryFirst`` does): ``"unmet=none"``, the selection with ``unmet="none"``; ``"unanchored"``,
    the decision without the anchor (round 3b's rules); ``"anchor"``, the anchor itself (phase A's result)."""

    by: str = "model"
    rule: Mean | LowerBound = Mean()
    feasibility: float = 0.9
    finalists: int = 3
    unmet: str = "closest"
    also: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.by not in ("model", "measured"):
            raise ValueError('Selection(by=...) is "model" or "measured"')
        if not 0.5 <= self.feasibility < 1:
            raise ValueError("Selection(feasibility=...) needs 0.5 <= feasibility < 1")
        if self.finalists < 0:
            raise ValueError("Selection(finalists=...) must be 0 or more")
        if self.unmet not in UNMET:
            raise ValueError(f"Selection(unmet=...) is one of {UNMET}, not {self.unmet!r}")
        object.__setattr__(self, "also", tuple(self.also))
        wrong = [a for a in self.also if a not in ALSO]
        if wrong:
            raise ValueError(f"Selection(also=...) takes rules among {ALSO}, not {wrong}")

    def describe(self) -> dict[str, Any]:
        return {
            "by": self.by,
            "rule": self.rule.describe(),
            "feasibility": self.feasibility,
            "finalists": self.finalists,
            "unmet": self.unmet,
            **({"also": list(self.also)} if self.also else {}),
        }


@dataclass(frozen=True)
class View:
    """A candidate as a decision sees it: its decided values and whether it meets the constraints."""

    candidate: "Candidate"
    metrics: Mapping[str, float]
    violations: tuple[float | None, ...]
    eligible: bool
    decided: Mapping[str, Decided]
    p_met: tuple[float | None, ...]
    values: tuple[float | None, ...] = ()  # each constraint's value as decided (its mean), the limit aside
    anchored: Any = None  # the paired evidence against the study's anchor, when it has one (phases.Evidence)

    @property
    def number(self) -> int:
        return self.candidate.number

    @property
    def params(self) -> dict[str, Any]:
        return self.candidate.params

    @property
    def trials(self) -> Any:
        return self.candidate.trials

    @property
    def feasible(self) -> bool | None:
        if not self.eligible:
            return None
        return all(v is not None and v <= 0 for v in self.violations)


def resolved(value: float, sd: float) -> float:
    """``value`` to a tenth of ``sd`` (its leading digit's place): candidates the model cannot tell apart at that
    resolution tie, rather than one dominating another by a difference the model's arithmetic made."""
    if not sd > 0 or not math.isfinite(value):
        return value
    quantum = 10.0 ** math.floor(math.log10(sd / 10))
    return round(value / quantum) * quantum


def view(
    candidate: "Candidate", projection: Projection, selection: Selection, fit: Fit | None, baseline: "Candidate"
) -> View:
    """The candidate as decided: by the model when it covers the candidate, else by its measured means."""
    if fit is None or candidate.number not in fit.candidates:
        return View(
            candidate,
            candidate.metrics,
            candidate.violations,
            candidate.eligible,
            {},
            (),
            _measured_values(candidate, projection, baseline),
        )
    decided = fit.decided(candidate.params, candidate.number)
    metrics = dict(candidate.metrics)
    for o in projection.objectives:
        d = decided[o.metric]
        metrics[o.metric] = resolved(selection.rule.value(d, isinstance(o, Maximize)), d.sd)
    violations, probabilities, values = [], [], []
    for c in projection.constraints:
        d = constraint_value(fit, c, candidate.params, candidate.number, baseline.params, baseline.number)
        probabilities.append(p_met(c, d))
        values.append(d.mean)
        # judged at the required probability: the value the metric exceeds (or stays below) with that probability
        q = selection.feasibility if isinstance(c, AtMost) else 1 - selection.feasibility
        violations.append(d.quantile(q) - c.limit if isinstance(c, AtMost) else c.limit - d.quantile(q))
    return View(candidate, metrics, tuple(violations), candidate.eligible, decided, tuple(probabilities), tuple(values))


def _measured_values(candidate: "Candidate", projection: Projection, baseline: "Candidate") -> tuple[float | None, ...]:
    """Each constraint's value by measured means: the metric, its difference from the baseline's, or its ratio."""
    out: list[float | None] = []
    for c in projection.constraints:
        v, b = candidate.metrics.get(c.metric), baseline.metrics.get(c.metric)
        if v is None or (c.compare != "absolute" and b is None):
            out.append(None)
        elif c.compare == "difference":
            out.append(v - b)
        elif c.compare == "ratio":
            out.append(v / b if b else None)
        else:
            out.append(v)
    return tuple(out)


@dataclass(frozen=True)
class Closest:
    """What a run ships when none of the candidates it selects from meets the constraints (``closest``): ``chosen``,
    and each candidate's closeness (``closeness``, by number: ``p_met``, its probability of meeting every constraint,
    or None when decided by measured means; ``excess``, how far it misses them), judged ``by`` "probability" (then
    excess) or "excess"."""

    chosen: list[View]
    by: str
    closeness: dict[int, dict[str, float | None]]

    def describe(self) -> dict[str, Any]:
        return {
            "rule": "closest",
            "by": self.by,
            "resolution": P_RESOLUTION if self.by == "probability" else None,
            "closeness": {str(n): c for n, c in self.closeness.items()},
        }


def closest(projection: Projection, combine: Combine, views: Sequence[View], baseline: View) -> Closest | None:
    """When none of ``views`` meets the constraints: the one that comes closest to meeting them, unconfirmed.

    Closest is the highest probability of meeting every constraint (the product of each one's, the model's ``p_met``;
    probabilities equal to two decimals, ``P_RESOLUTION``, tie), then the least excess (the sum of what each constraint
    misses by, as judged: the "miss them least" rule every ``Combine.held`` uses), then the combination's own order
    among those still tied (``Combine.best``; for ``CheapestOnFront``, its near-tie rule). Decided by measured means (no
    probabilities), it is the least excess, then the combination's order. Only eligible candidates whose every
    constraint was measured qualify; None when there are none."""
    pool = [v for v in views if v.eligible and all(x is not None for x in v.violations)]
    if not pool or not projection.constraints:
        return None
    n = len(projection.constraints)
    by_probability = all(len(v.p_met) == n and all(p is not None for p in v.p_met) for v in pool)

    def probability(v: View) -> float | None:
        return math.prod(p for p in v.p_met if p is not None) if by_probability else None

    def key(v: View) -> tuple[float, float]:
        p = probability(v)
        return (round(p / P_RESOLUTION) if p is not None else 0.0, -_excess(v))

    top = max(key(v) for v in pool)
    tied = [v for v in pool if key(v) == top]
    # the combination's own order among the tied, each taken as meeting the constraints (none does)
    met = [dataclasses.replace(v, violations=tuple(0.0 for _ in v.violations)) for v in tied]
    chosen = {v.number for v in combine.best(projection, met, baseline)}
    return Closest(
        chosen=[v for v in tied if v.number in chosen],
        by="probability" if by_probability else "excess",
        closeness={v.number: {"p_met": probability(v), "excess": _excess(v)} for v in pool},
    )


__all__ = ["Closest", "LowerBound", "Mean", "Selection", "View", "closest", "resolved", "view"]
