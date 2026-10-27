"""Which of an experiment's cases each part of a study uses: ``train`` (the search runs on them), ``validation`` (the
finalists are measured on them and the selection is made there), ``test`` (the selection and the baseline are measured
on them once, after the selection, and never seen before).

    Split(train=train_cases, validation=dev_cases, test=test_cases)   # the cases themselves
    Split(train=0.5, validation=0.25, test=0.25)                      # shares of the experiment's cases
    Split(train=0.5, validation=0.25, test=0.25, rotate=True)         # every case tested once: 1 / test runs
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Assignment:
    """One run's cases by role."""

    train: list[Any]
    validation: list[Any]
    test: list[Any]

    def names(self) -> dict[str, list[str]]:
        return {role: [c.name for c in getattr(self, role)] for role in ("train", "validation", "test")}


@dataclass(frozen=True)
class Split:
    """The experiment's cases by role: each of ``train``, ``validation`` and ``test`` is either a list of cases or a
    share of the experiment's cases (shares sum to 1; lists are disjoint and give every case of the experiment a role).
    ``rotate`` (shares only): run the study once per test share, each run testing a different part of the cases, so
    every case is tested exactly once and the test results pool over all of them; it costs 1 / ``test`` searches, run
    side by side (they divide the study's ``max_concurrency``).
    ``seed`` fixes how shares deal the cases."""

    train: Sequence[Any] | float
    validation: Sequence[Any] | float | None = None
    test: Sequence[Any] | float | None = None
    rotate: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        values = [v for v in (self.train, self.validation, self.test) if v is not None]
        shares = [isinstance(v, (int, float)) and not isinstance(v, bool) for v in values]
        if any(shares) and not all(shares):
            raise ValueError("Split: give every role a share or every role a list of cases, not a mix")
        if all(shares):
            parts = [float(v) for v in (self.train, self.validation or 0.0, self.test or 0.0)]
            if any(p < 0 or p > 1 for p in parts) or parts[0] <= 0:
                raise ValueError(f"Split: shares must lie in 0 to 1 and train must be more than 0, got {parts}")
            if abs(sum(parts) - 1.0) > _TOLERANCE:
                raise ValueError(f"Split: shares must sum to 1, got {sum(parts):g}")
            if self.rotate:
                test = parts[2]
                if test <= 0 or abs(1 / test - round(1 / test)) > 1e-6 or round(1 / test) < 2:
                    raise ValueError(
                        f"Split(rotate=True): 1 / test must be a whole number of 2 or more, got test={test}"
                    )
        elif self.rotate:
            raise ValueError("Split(rotate=True) needs shares: fixed lists of cases have nothing to rotate")

    @property
    def shares(self) -> bool:
        return isinstance(self.train, (int, float)) and not isinstance(self.train, bool)

    @property
    def runs(self) -> int:
        """How many searches the study runs: 1, or 1 / ``test`` with ``rotate``."""
        return round(1 / float(self.test)) if self.rotate and self.test else 1

    def assign(self, cases: Sequence[Any]) -> list[Assignment]:
        """Each run's cases by role, from the experiment's ``cases``."""
        names = [c.name for c in cases]
        if len(set(names)) != len(names):
            raise ValueError("Split: the experiment's cases need unique names")
        return self._by_share(list(cases)) if self.shares else [self._by_list(list(cases))]

    def _by_list(self, cases: list[Any]) -> Assignment:
        roles = {r: list(getattr(self, r) or ()) for r in ("train", "validation", "test")}
        seen: dict[str, str] = {}
        twice = []
        for role, items in roles.items():
            for c in items:
                if c.name in seen:
                    twice.append(c.name)
                seen[c.name] = role
        if twice:
            raise ValueError(f"Split: cases in more than one role: {sorted(set(twice))}")
        if not roles["train"]:
            raise ValueError("Split: no train cases")
        pool = {c.name for c in cases}
        outside = sorted(set(seen) - pool)
        if outside:
            raise ValueError(f"Split: cases not in the experiment: {outside}")
        unassigned = sorted(pool - set(seen))
        if unassigned:
            raise ValueError(
                f"Split: the experiment's cases {unassigned} have no role; give them one or leave them out of the "
                "experiment"
            )
        by_name = {c.name: c for c in cases}  # the experiment's own objects
        return Assignment(*([by_name[c.name] for c in roles[r]] for r in ("train", "validation", "test")))

    def _by_share(self, cases: list[Any]) -> list[Assignment]:
        order = sorted(cases, key=lambda c: c.name)  # independent of the order given
        random.Random(self.seed).shuffle(order)
        n = len(order)
        train, validation, test = float(self.train), float(self.validation or 0.0), float(self.test or 0.0)
        if not self.rotate:
            counts = _counts(n, [train, validation, test])
            out = Assignment(
                order[: counts[0]], order[counts[0] : counts[0] + counts[1]], order[counts[0] + counts[1] :]
            )
            _check(out, n)
            return [out]
        k = self.runs
        if n < k:
            raise ValueError(f"Split(rotate=True): {n} cases cannot make {k} test parts")
        parts = [order[i::k] for i in range(k)]
        whole = validation / test  # validation as a whole number of parts, when it is one
        runs = []
        for i in range(k):
            rest = [c for j, part in enumerate(parts) if j != i for c in part]
            if abs(whole - round(whole)) <= 1e-6:
                held = [c for j in range(1, round(whole) + 1) for c in parts[(i + j) % k]]
            else:  # not a whole number of parts: each run draws its validation cases from its non-test cases
                drawn = list(rest)
                random.Random(self.seed + i + 1).shuffle(drawn)
                held = drawn[: round(validation * n)]
            names = {c.name for c in held}
            out = Assignment([c for c in rest if c.name not in names], held, parts[i])
            _check(out, n)
            runs.append(out)
        return runs

    def describe(self) -> dict[str, Any]:
        def role(v: Any) -> Any:
            return v if v is None or isinstance(v, (int, float)) else len(v)

        return {
            "train": role(self.train),
            "validation": role(self.validation),
            "test": role(self.test),
            "rotate": self.rotate,
            "seed": self.seed,
            "runs": self.runs,
        }


def _counts(n: int, shares: list[float]) -> list[int]:
    """``n`` dealt by ``shares``, largest remainders first, summing to ``n``."""
    exact = [s * n for s in shares]
    counts = [math.floor(x) for x in exact]
    for i in sorted(range(len(shares)), key=lambda i: exact[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    return counts


def _check(assignment: Assignment, n: int) -> None:
    if not assignment.train:
        raise ValueError(f"Split: these shares leave no train cases among {n}")


__all__ = ["Assignment", "Split"]
