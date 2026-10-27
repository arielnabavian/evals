"""What a study spends, in tokens: every model call of the agent in its trials (every attempt, those run again
included) and every model call a method makes itself, by role and phase. Nothing is priced: a cost objective or
constraint is in these same tokens.

Roles:

- ``agent``: the agent under optimization, in its trials (``Trial.metrics`` for the final attempt, ``Trial.attempts``
  for those run again);
- ``method:<block>``: a block's method, in its own model calls (a rewriter, a curator): what it reports in
  ``origin.usage``, whether its call succeeded or failed;
- ``probes``: reserved for trials a method runs to probe a proposal, which are recorded apart (none yet).

A model a workload calls inside its task or its evaluators (a simulated user, a judge that scores) is part of the
benchmark, not of the optimization, and is not counted.

Phases: ``search`` (the baseline and every candidate on the search cases, and every method call), ``remeasurement``
(finalists measured again on the search cases), ``validation`` and ``test``.

Each cell holds the usage metrics, in the names the trials and goals use (``input_tokens``, ``output_tokens``,
``cache_read_tokens``, ``cache_write_tokens``, ``tokens``, ``model_calls``, ``seconds``; seconds are summed, so they are
conversation or call seconds, not wall time), and what it counts: an agent cell its ``trials`` and ``attempts`` (every
time a trial's task ran), a method cell its ``calls`` and how many ``failed``.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .trials import NEVER_RAN, SPENT, TOKENS, Trial

PHASES = ("search", "remeasurement", "validation", "test")
AGENT, PROBES = "agent", "probes"
METHOD = "method:"


def method_role(block: str) -> str:
    """The role of a block's method's own model calls."""
    return f"{METHOD}{block}"


def spent(usage: Mapping[str, Any], seconds: float | None = None) -> dict[str, float]:
    """What a trial's metrics or a method's ``origin.usage`` spent (``SPENT``; a kind not given is 0, and ``tokens``
    the sum of the token kinds when not given)."""
    out = {k: float(usage.get(k) or 0) for k in SPENT}
    if usage.get("tokens") is None:
        out["tokens"] = sum(out[k] for k in TOKENS)
    if seconds is not None:
        out["seconds"] = float(seconds)
    return out


def _add(into: dict[str, Any], usage: Mapping[str, Any], keys: Sequence[str] | None = None) -> None:
    for k in keys if keys is not None else usage:
        if k in usage:
            into[k] = into.get(k, 0) + usage[k]


def _plain(v: Any) -> Any:
    return int(v) if isinstance(v, float) and v.is_integer() and abs(v) < 2**53 else v


@dataclass(frozen=True)
class Cost:
    """What a study spent, in tokens: ``roles[role][phase]``, each cell the usage it spent and what it counts (see the
    module's docstring). ``Result.cost`` is a study's; the sum of two (``a + b``) is what both spent."""

    roles: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)

    def cell(self, role: str, phase: str) -> dict[str, Any]:
        """What ``role`` spent in ``phase`` (zeros when it spent nothing there)."""
        return dict(self.roles.get(role, {}).get(phase) or {k: 0.0 for k in SPENT})

    @property
    def by_role(self) -> dict[str, dict[str, Any]]:
        """What each role spent over every phase, with its counts."""
        out: dict[str, dict[str, Any]] = {}
        for role, phases in self._ordered().items():
            out[role] = {k: 0.0 for k in SPENT}
            for cell in phases.values():
                _add(out[role], cell)
        return out

    @property
    def by_phase(self) -> dict[str, dict[str, float]]:
        """What every role together spent in each phase."""
        out: dict[str, dict[str, float]] = {}
        for phases in self.roles.values():
            for phase, cell in phases.items():
                _add(out.setdefault(phase, {k: 0.0 for k in SPENT}), cell, SPENT)
        return {p: out[p] for p in [*PHASES, *sorted(set(out) - set(PHASES))] if p in out}

    @property
    def total(self) -> dict[str, float]:
        """What the study spent."""
        out = {k: 0.0 for k in SPENT}
        for cell in self.by_phase.values():
            _add(out, cell, SPENT)
        return out

    def tokens(self, role: str | None = None, phase: str | None = None) -> float:
        """Tokens spent by ``role`` (default: every role) in ``phase`` (default: every phase)."""
        return float(
            sum(
                cell.get("tokens", 0)
                for r, phases in self.roles.items()
                if role is None or r == role
                for p, cell in phases.items()
                if phase is None or p == phase
            )
        )

    def __add__(self, other: Cost) -> Cost:
        roles: dict[str, dict[str, dict[str, Any]]] = {}
        for cost in (self, other):
            for role, phases in cost.roles.items():
                for phase, cell in phases.items():
                    _add(roles.setdefault(role, {}).setdefault(phase, {}), cell)
        return Cost(roles)

    def _ordered(self) -> dict[str, dict[str, Mapping[str, Any]]]:
        """The roles in their order (the agent, the methods, probes), each with its phases in theirs."""
        out: dict[str, dict[str, Mapping[str, Any]]] = {}
        for role in [AGENT, *sorted(r for r in self.roles if r not in (AGENT, PROBES)), PROBES]:
            phases = self.roles.get(role, {})
            out[role] = {p: phases[p] for p in [*PHASES, *sorted(set(phases) - set(PHASES))] if p in phases}
        return out

    def to_dict(self) -> dict[str, Any]:
        def plain(cell: Mapping[str, Any]) -> dict[str, Any]:
            return {k: _plain(v) for k, v in cell.items()}

        return {
            "unit": "tokens",
            "roles": {r: {p: plain(c) for p, c in phases.items()} for r, phases in self._ordered().items()},
            "by_role": {r: plain(c) for r, c in self.by_role.items()},
            "by_phase": {p: plain(c) for p, c in self.by_phase.items()},
            "total": plain(self.total),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Cost:
        """A cost as ``to_dict`` wrote it (its roles; the sums are recomputed)."""
        return cls({r: {p: dict(c) for p, c in phases.items()} for r, phases in (data.get("roles") or {}).items()})

    def describe(self) -> str:
        """One line: the total and each role's tokens (the agent's by phase)."""
        agent = self.roles.get(AGENT, {})
        phases = ", ".join(f"{p} {_n(agent[p]['tokens'])}" for p in self._ordered()[AGENT])
        parts = [f"agent {_n(self.tokens(AGENT))}" + (f" ({phases})" if phases else "")]
        parts += [
            f"{r} {_n(self.tokens(r))}" for r in self._ordered() if r != AGENT and (r != PROBES or self.tokens(r))
        ]
        return f"{_n(self.total['tokens'])} tokens: " + "; ".join(parts)


def _n(x: float) -> str:
    return f"{x:,.0f}"


class Ledger:
    """A study's cost as it runs (``Study.cost`` is its ``cost()``). Thread-safe: a step's proposals may be made in
    threads, and each method call is counted as it returns."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roles: dict[str, dict[str, dict[str, Any]]] = {AGENT: {}, PROBES: {}}

    def trials(self, phase: str, trials: Sequence[Trial]) -> None:
        """Count the agent's usage in ``trials``: each final attempt and every attempt run again."""
        with self._lock:
            cell = self._roles[AGENT].setdefault(phase, {**{k: 0.0 for k in SPENT}, "trials": 0, "attempts": 0})
            for t in trials:
                _add(cell, spent(t.metrics))
                for a in t.attempts:
                    _add(cell, spent(a.usage))
                cell["trials"] += 1
                cell["attempts"] += len(t.attempts) + (0 if t.error == NEVER_RAN else 1)

    def call(self, block: str, origin: Mapping[str, Any] | None, phase: str = "search") -> None:
        """Count one call of ``block``'s method: the model usage it reports in ``origin.usage`` (a method that reports
        none made no model call of its own)."""
        usage = (origin or {}).get("usage")
        if not isinstance(usage, Mapping):
            return
        with self._lock:
            role = self._roles.setdefault(method_role(block), {})
            cell = role.setdefault(phase, {**{k: 0.0 for k in SPENT}, "calls": 0, "failed": 0})
            _add(cell, spent(usage, (origin or {}).get("seconds")))
            cell["calls"] += 1
            cell["failed"] += int((origin or {}).get("status") == "failed")

    def cost(self) -> Cost:
        """What has been spent so far."""
        with self._lock:
            return Cost({r: {p: dict(c) for p, c in phases.items()} for r, phases in self._roles.items()})


__all__ = ["Cost", "Ledger", "PHASES", "method_role", "spent"]
