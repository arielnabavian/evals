"""A study's record on disk: what was searched, how the search was scheduled, and every candidate as it finishes.

    <directory>/
      study.jsonl              one JSON object per line (an event), in the order they happened
      transcripts/<hash>.json  each trial's messages, stored once by content hash
      test.jsonl               the test, when the study has test cases
      artifacts/               what methods write (skill-library versions, curators' decisions)

Events (``format: strands_evals.optimization``, schema 3):

- ``study``: the header. The agent factory and the ``task`` that runs each trial (with ``retries`` and ``redraws``);
  objectives; constraints (in order, each with its exact ``name``); ``combine``; the ``schedule`` tree (each node with
  its ``node`` id and ``children``); ``blocks`` (method, variables, structural) and ``block_order``; the ``space``;
  every metric with its role and range (a default skill library is the version the study stored, as declared in
  ``stored_defaults``); cases and repetitions; ``width`` and ``max_concurrency``; ``roles``, what
  identifies each role of the cost (the agent's ``model``, as the agent built at the defaults names it, or null; each
  method that calls a model itself, with its ``block``, ``method`` and ``model``; ``probes``, reserved); and ``code``,
  the package's ``version`` and git ``commit`` (with ``dirty``), where they can be found.
- ``run``: one ``optimize()`` call began (``iterations``, not counting the baseline the first call measures, and
  ``patience`` or null).
- ``scope``: a schedule began running: the root of a run, or a round (``round``, ``outer``) inside its ``parent``, with
  the blocks it may change, the objective it serves (Independent) and its candidate ``limit`` (in steps; with
  ``candidates``, its budget in candidates, when that is not ``limit`` x ``width``: a round whose opener costs one).
- ``candidate``: a candidate finished: its values, status and estimates; ``scope`` and ``path`` (the scopes it ran in,
  outermost first), ``role`` (baseline | outer | step) and ``step``; what was ``proposed`` (blocks), ``drawn``
  (variables per block), ``inherited`` and ``defaulted``; ``violations`` (one per constraint, in order) and
  ``feasible``; which blocks it was given to (``shared``) and which wait for a round's result (``deferred``); what each
  method copy was ``told``; and ``best_after``, the best candidates once it finished. Each of its ``trials`` (here and
  wherever trials are recorded) is its final attempt, with ``attempts``, the attempts before it that were run again
  (each with its ``status``, ``error`` and what it spent), when there were any.
- ``repeat``: a proposal (in ``scope``, with ``role`` and ``step``) repeated the values of ``candidate``, which was
  not measured again. The blocks in ``told`` learned that their draws scored what ``candidate`` measured (``values``;
  with ``Independent``, as ``objective``'s methods are told it). A step that repeats proposes again; a round whose
  opening proposal repeats opens on ``candidate``, and its outer draw is told the round's best later (``told``).
- ``told``: a deferred draw (a round's outer proposal, made in ``draw_candidate``) was told ``candidate``, the round's
  best, or when none of the round's candidates can be held (phase B: none keeps A*'s level) the closest to it (null:
  the round had no eligible candidate, and the draw failed). ``values`` (here, and a candidate's ``told``) are as
  decided when it was told: an Optuna landscape tells every trial again, as decided then, before each proposal.
- ``scope_end``: a scope finished, having run ``ran`` candidates over ``blocks`` (the blocks that existed in it); a
  round that opened on an earlier candidate does not count it.
- ``model``: one fit of the model of every goal metric (its factors, diagnostics, and every candidate's decided values
  at that fit); ``remeasured``: a finalist measured again on the search cases; ``validated``: the baseline or a
  finalist measured on the validation cases; ``selected``: the selection a run ended with, where and how it was made
  (``unconfirmed`` when none of the candidates it chose from met the constraints and it is the closest, with
  ``unmet``, each candidate's closeness: ``Selection.unmet``; records before it lack both, and their ``best`` is then
  empty).
- ``finished`` (the run ended, with ``best``, ``unconfirmed`` and its ``reason``: ``candidates``, ``patience``, or
  ``exhausted``: its schedule had nothing new to propose) or ``stopped`` (it raised, with ``error``). ``test.jsonl``'s
  ``tested`` arms carry ``unconfirmed`` too.
- Phases (``phases.PrimaryFirst``, round 3d): the header's ``aims`` (what each method copy searches for) and
  ``schedule`` (kind ``PrimaryFirst``, with its settings, and a child per phase); each phase's ``scope`` with its
  ``phase``; each candidate's ``phase`` and ``aim`` (the copy that proposed it) and, once there is an anchor,
  ``anchored`` (its paired evidence against A*); ``anchor``, when phase A ends: A*, why and after how many steps and
  candidates the phase ended, each of its steps' evidence, and the ``cost`` spent by then; the ``selected`` event's
  ``pick`` (the one candidate shipped), ``reference``, ``alternatives`` (what ``Selection.also``'s rules ship) and each
  decided candidate's ``anchored`` evidence; ``test.jsonl``'s arms with their ``pick`` and ``rules``, and an arm of role
  ``alternative`` for what another rule ships that no other arm measured. ``objective`` stays Independent's alone.

``selected``, ``finished``, ``stopped`` and ``test.jsonl``'s ``test_end`` carry ``cost``: what the study had spent by
then, in tokens, by role (the agent; each method's own model calls; probes, reserved) and phase (search,
re-measurement, validation, test), with every attempt of every trial counted (``cost.py``). A run that stopped still
states what it spent.

Values are plain JSON; transcripts are referenced by hash, never inlined, so the record stays small and readable. A
value a method writes as files (a skill-library version) is stored in ``artifacts/`` once, by content, and the value is
its path there (``skills/<hash>``): relative, so the record can be moved (``Study.artifact_path``; records written
before 2026-10-24 hold absolute paths, which still read).

Rules every record keeps, so that records written by later extensions still read, and mean the same, beside these:

- A reader skips an event type or a field it does not know. A new fact is a new event or a new field, never a new
  meaning for an existing one.
- A method's ``origin.usage`` counts only that method's own model calls (``origin.seconds``, their time): never the
  trials it runs, which are the agent's.
- Probe runs (trials a method runs to try a proposal before it is measured) are their own records, their own events
  linked to the proposal that ran them, never candidates: candidate numbers stay dense, and no fit, incumbent,
  selection or repeat check sees a probe.
- A trial over its deadline is an ``agent_failure``, with its cause in ``error``: never ``infrastructure_invalid``,
  which would run it again and then drop it, so a deadline never lets a slow or hanging agent shed its hard cases.
- A program's own parameters are keyed per version (``<key>.<version>.<parameter>``, as ``OneOf`` keys an option's
  arguments), so a variable's ``active_when`` stays one ``(key, option)`` pair.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import urllib.parse
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .projection import constraint_name, describe_goal
from .trials import SCORED, USAGE

if TYPE_CHECKING:
    from .cost import Cost
    from .study import Candidate, Scope, Study

FORMAT = "strands_evals.optimization"
SCHEMA = 3


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))


def _values(params: dict[str, Any]) -> dict[str, Any]:
    """A candidate's values as the record holds them: plain JSON, checked here rather than written as a ``repr``
    (``_write`` gives anything else in an event its ``str``)."""
    try:
        json.dumps(params, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"a candidate's values must be plain JSON to be recorded: {exc}") from exc
    return params


def _version(here: Path) -> str | None:
    """The installed package's version, when the code running is that installation (installed, or installed editable
    from this checkout); None when it is another copy on the path, whose version the installed one does not say."""
    try:
        dist = metadata.distribution("strands-agents-evals")
        if Path(str(dist.locate_file("strands_evals/optimization/record.py"))).resolve() == here:
            return dist.version
        url = json.loads(dist.read_text("direct_url.json") or "{}").get("url", "")
        if url.startswith("file://") and here.is_relative_to(Path(urllib.parse.unquote(url[7:])).resolve()):
            return dist.version
    except (metadata.PackageNotFoundError, OSError, ValueError):
        pass
    return None


def _code() -> dict[str, Any]:
    """Which code wrote the record: the package's version, and its git commit (and whether the checkout had changes),
    where they can be found."""
    here = Path(__file__).resolve()
    out: dict[str, Any] = {"version": _version(here), "commit": None}
    here = here.parent.parent
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            out["commit"] = head.stdout.strip()
            changes = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=here,
                capture_output=True,
                text=True,
                timeout=10,
            )
            out["dirty"] = bool(changes.stdout.strip()) if changes.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):  # no git, or not a checkout
        pass
    return out


def _cost(cost: "Cost | None") -> dict[str, Any] | None:
    return cost.to_dict() if cost is not None else None


def _import_path(obj: Any) -> str:
    """Where a function, or a callable object's class, is defined."""
    named = obj if hasattr(obj, "__qualname__") else type(obj)
    return f"{named.__module__}.{named.__qualname__}"


def describe_schedule(schedule: Any, study: "Study", node: str = "") -> dict[str, Any]:
    children = schedule.children() if hasattr(schedule, "children") else {}
    describe = schedule.describe(study) if hasattr(schedule, "describe") else {"kind": type(schedule).__name__}
    return {
        "node": node,
        **describe,
        "children": {
            name: describe_schedule(child, study, f"{node}.{name}" if node else name)
            for name, child in children.items()
        },
    }


class Record:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.path = self.directory / "study.jsonl"
        if self.path.exists() and self.path.stat().st_size > 0:
            raise FileExistsError(f"{self.path} already holds a study; give each study its own record directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "transcripts").mkdir(exist_ok=True)

    def _write(self, event: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _transcript(self, messages: Any) -> dict[str, str] | None:
        if not messages:
            return None
        text = json.dumps(_plain(list(messages)), indent=1)
        digest = hashlib.sha256(text.encode()).hexdigest()[:16]
        path = self.directory / "transcripts" / f"{digest}.json"
        if not path.exists():
            path.write_text(text)
        return {"hash": digest, "format": "strands-messages"}

    def study(self, study: "Study") -> None:
        p = study.projection
        metrics = sorted(p.evaluators | set(USAGE) | set(p.derived) | set(p.goal_metrics))
        self._write(
            {
                "type": "study",
                "format": FORMAT,
                "schema": SCHEMA,
                "started_at": _now(),
                "agent": study.agent.import_path,
                "task": _import_path(study.task),
                "retries": study.retries,
                "redraws": study.search.redraws,
                "width": study.search.width,
                "max_concurrency": study.max_concurrency,
                "roles": study._roles(),
                "code": _code(),
                "objectives": [describe_goal(o) for o in p.objectives],
                "constraints": [{**describe_goal(c), "name": constraint_name(c)} for c in p.constraints],
                "combine": {
                    **study.combine.describe(),
                    "best_is_set": study.combine.best_is_set(len(p.objectives)),
                    "copies": len(study.methods),
                    "told": [study.aim(k).told_names(p, k) for k in range(len(study.methods))],
                },
                **(
                    {"aims": [a.describe() for a in study.aims]}
                    if any(a is not study.combine for a in study.aims)
                    else {}
                ),
                "schedule": describe_schedule(study.schedule, study),
                "blocks": {
                    name: {"method": b.method.describe(), "keys": list(b.keys), "structural": b.structural}
                    for name, b in study.blocks.items()
                },
                "block_order": study.block_order(),
                "space": [info.to_dict() for info in study.params.values()],
                **({"stored_defaults": dict(study._stored_defaults)} if study._stored_defaults else {}),
                "metrics": {m: {"role": p.role(m), "range": list(p.range(m))} for m in metrics},
                "cases": [c.name for c in study.experiment.cases],
                "repetitions": study.repetitions,
                "estimate_unit": "case" if len(study.experiment.cases) >= 2 else "trial",
                "agent_error_score": {m: p.failure_value(m) for m in sorted(p.evaluators)},
                "selection": study.selection.describe(),
                "validation": [c.name for c in study.validation.cases] if study.validation is not None else None,
                "test": [c.name for c in study.test_cases] or None,
                "split": study.split.describe() if study.split is not None else None,
            }
        )

    def run(self, iterations: int, patience: int | None = None) -> None:
        self._write({"type": "run", "started_at": _now(), "iterations": iterations, "patience": patience})

    def scope(self, scope: "Scope", id: int) -> None:
        self._write(
            {
                "type": "scope",
                "id": id,
                "parent": scope.parent.id if scope.parent else None,
                "node": scope.node,
                "kind": scope.kind,
                "blocks": list(scope.blocks),
                "objective": scope.objective if scope.study._independent else None,
                **({"aim": scope.objective, "phase": scope.phase} if scope.phase is not None else {}),
                "round": scope.round,
                "outer": scope.outer,
                "limit": scope.limit,
                # its budget in candidates, when that is not limit x width (a round whose opener costs one candidate)
                **({"candidates": scope.capacity} if scope.capacity != scope.limit * scope.width else {}),
                "opened_at": _now(),
            }
        )

    def scope_end(self, scope: "Scope") -> None:
        self._write(
            {
                "type": "scope_end",
                "id": scope.id,
                "ran": sum(scope.id in c.path for c in scope.candidates),  # not a candidate a round opened on again
                "blocks": list(scope.blocks),
                "closed_at": _now(),
            }
        )

    def _told(self, study: "Study", candidate: "Candidate", k: int) -> list[float] | None:
        if not candidate.eligible:
            return None
        return list(study.aim(k).told(study.projection, study.view(candidate), k, study.view(study.baseline)))

    @staticmethod
    def _decided(study: "Study", candidate: "Candidate") -> dict[str, Any] | None:
        """The candidate as decided at this point: per goal metric, the model's posterior of its mean over the cases;
        each constraint's probability of being met; whether that makes it feasible (None when not by the model)."""
        if not study.by_model or candidate.number not in study.fit.candidates:
            return None
        v = study.view(candidate)
        return {
            "metrics": {m: d.to_dict() for m, d in v.decided.items()},
            "p_met": list(v.p_met),
            "feasible": v.feasible,
            **({"anchored": v.anchored.to_dict()} if v.anchored is not None else {}),
        }

    def _trials(self, trials: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            {
                "case": t.case,
                "repetition": t.repetition,
                "status": t.status.value,
                "scored": t.status in SCORED,
                "metrics": dict(t.metrics),
                "unavailable": sorted(t.unavailable),
                "error": t.error,
                "output": t.output,
                "reasons": dict(t.reasons),
                "transcript": self._transcript(t.messages),
                **({"attempts": [a.to_dict() for a in t.attempts]} if t.attempts else {}),
            }
            for t in trials
        ]

    def model(self, study: "Study", fit: Any, problem: str | None) -> None:
        """One fit of the model: its factors, prior scales and diagnostics, and every candidate's decided values at
        this point (they change as evidence accrues)."""
        self._write(
            {
                "type": "model",
                "fitted_at": _now(),
                "candidates": list(fit.candidates),
                "factors": fit.design.describe(),
                "omitted": fit.design.omitted,
                "trends": [{"key": k, "log": log, "center": c, "spread": sd} for k, log, c, sd in fit.design.trends],
                "metrics": {
                    m: {"noise": model.noise, "unit": model.unit, "offset": model.offset}
                    for m, model in fit.metrics.items()
                },
                "diagnostics": _plain(fit.diagnostics),
                "used": problem is None,
                "problem": problem,
                "decided": {str(n): self._decided(study, study.candidates[n]) for n in fit.candidates},
            }
        )

    def remeasured(self, study: "Study", candidate: "Candidate", trials: Sequence[Any]) -> None:
        """A finalist measured again: its new trials, and its estimates over all its trials."""
        self._write(
            {
                "type": "remeasured",
                "remeasured_at": _now(),
                "number": candidate.number,
                "trials": self._trials(trials),
                "estimates": {
                    m: {"mean": e.mean, "ci": list(e.ci) if e.ci else None, "n": e.n}
                    for m, e in candidate.estimates.items()
                },
                "eligible": candidate.eligible,
                "violations": list(candidate.violations),
            }
        )

    def validated(self, study: "Study", candidate: "Candidate") -> None:
        """A finalist (or the baseline) measured on the validation cases: its trials there, and its estimates."""
        self._write(
            {
                "type": "validated",
                "validated_at": _now(),
                "number": candidate.number,
                "trials": self._trials(candidate.trials),
                "estimates": {
                    m: {"mean": e.mean, "ci": list(e.ci) if e.ci else None, "n": e.n}
                    for m, e in candidate.estimates.items()
                },
                "eligible": candidate.eligible,
                "violations": list(candidate.violations),
            }
        )

    def selected(
        self,
        study: "Study",
        best: Sequence["Candidate"],
        *,
        on: str = "search",
        views: dict[int, Any] | None = None,
        unconfirmed: Any = None,
        decision: Any = None,
    ) -> None:
        """The selection a run ended with, the cases it was made on (``search`` or ``validation``), and the decided
        values it was made on (on validation, of every candidate measured there). ``unconfirmed`` (true when none of
        the candidates met the constraints, so the selection is the closest, ``Selection(unmet="closest")``) and
        ``unmet``, how the closest was judged: each candidate's probability of meeting every constraint (``p_met``,
        null by measured means) and what it misses them by (``excess``). With ``unmet="none"``, ``best`` is then
        empty and ``unconfirmed`` false, as before."""
        if views is not None:
            decided = {
                str(n): {
                    "metrics": {m: d.to_dict() for m, d in v.decided.items()},
                    "p_met": list(v.p_met),
                    "feasible": v.feasible,
                    **({"anchored": v.anchored.to_dict()} if v.anchored is not None else {}),
                }
                if v.decided or v.anchored is not None
                else None
                for n, v in views.items()
            }
            by = "model" if any(v.decided for v in views.values()) else "measured"
        else:
            decided = {str(c.number): self._decided(study, c) for c in best}
            by = "model" if study.by_model else "measured"
        self._write(
            {
                "type": "selected",
                "selected_at": _now(),
                "on": on,
                "by": by,
                "problem": study._model_problem if study.selection.by == "model" else None,
                "rule": study.selection.describe(),
                "best": [c.number for c in best],
                "unconfirmed": unconfirmed is not None,
                "unmet": unconfirmed.describe() if unconfirmed is not None else None,
                **(
                    {
                        "pick": decision.pick.number if decision.pick is not None else None,
                        "anchor": study.anchor.number if study.anchor is not None else None,
                        "reference": decision.reference,
                        "finalists": list(study._finalist_numbers),
                        "alternatives": decision.alternatives,
                    }
                    if decision is not None
                    else {}
                ),
                "decided": decided,
                "cost": _cost(study.cost),
            }
        )

    def anchor(self, study: "Study", anchor: Any, *, phase: dict[str, Any], steps: list[dict[str, Any]]) -> None:
        """Phase A ended and set the anchor, A*: why (``reason``: stall, met, cap, budget or exhausted), after how many
        steps and candidates, each of its steps' evidence (its base, its candidates, the best's probability of beating
        the base, the threshold, and whether that is a gain), A*'s measured means, and what the study had spent by then
        (``cost``: phase B's is the run's minus this)."""
        a = study.candidates[anchor.number]
        self._write(
            {
                "type": "anchor",
                "anchored_at": _now(),
                **anchor.describe(),
                "phase": phase,
                "steps": _plain(steps),
                "estimates": {m: a.metrics.get(m) for m in study.projection.goal_metrics},
                "candidates": len(study.candidates),
                "cost": _cost(study.cost),
            }
        )

    def candidate(
        self, study: "Study", candidate: "Candidate", *, shared: Sequence[str], deferred: Sequence[str]
    ) -> None:
        self._write(
            {
                "type": "candidate",
                "finished_at": _now(),
                "number": candidate.number,
                "params": _values(candidate.params),
                "status": candidate.status,
                "reason": candidate.reason,
                "eligible": candidate.eligible,
                "feasible": candidate.feasible,
                "violations": list(candidate.violations),
                "scope": candidate.scope,
                "path": list(candidate.path),
                "role": candidate.role,
                "step": candidate.step,
                "slot": candidate.slot,
                "proposed": list(candidate.proposed),
                "drawn": {b: list(ks) for b, ks in candidate.drawn.items()},
                "inherited": dict(candidate.inherited),
                "defaulted": list(candidate.defaulted),
                "base": candidate.base,
                "objective": candidate.objective,
                **({"aim": candidate.aim, "phase": candidate.phase} if candidate.phase is not None else {}),
                "origin": _plain(candidate.origin),
                "shared": list(shared),
                "deferred": list(deferred),
                "told": [self._told(study, candidate, k) for k in range(len(study.methods))],
                "best_after": [c.number for c in study.best_candidates],
                "decided": self._decided(study, candidate),
                "estimates": {
                    m: {"mean": e.mean, "ci": list(e.ci) if e.ci else None, "n": e.n}
                    for m, e in candidate.estimates.items()
                },
                "trials": self._trials(candidate.trials),
            }
        )

    def told(
        self, study: "Study", scope: "Scope", block: str, draw_candidate: int, candidate: "Candidate | None", k: int
    ) -> None:
        self._write(
            {
                "type": "told",
                "scope": scope.id,
                "block": block,
                "objective": k if study._independent else None,
                "draw_candidate": draw_candidate,
                "candidate": candidate.number if candidate is not None else None,
                "values": self._told(study, candidate, k) if candidate is not None else None,
            }
        )

    def repeat(
        self,
        study: "Study",
        candidate: "Candidate",
        draws: dict[str, Any],
        drawn: dict[str, Sequence[str]],
        *,
        scope: "Scope | None",
        role: str,
        step: int | None,
        objective: int | None,
        told: Sequence[str],
    ) -> None:
        k = objective or 0
        self._write(
            {
                "type": "repeat",
                "repeated_at": _now(),
                "scope": scope.id if scope else None,
                "path": list(scope.path) if scope else [],
                "role": role,
                "step": step,
                "candidate": candidate.number,
                "proposed": list(draws),
                "drawn": {b: list(ks) for b, ks in drawn.items()},
                "origin": _plain({name: draw.origin for name, draw in draws.items()}),
                "objective": objective if study._independent else None,
                "told": list(told),
                "values": self._told(study, candidate, k) if told else None,
            }
        )

    def finish(self, study: "Study", reason: str = "candidates") -> None:
        """``reason``: ``candidates`` (every candidate asked for ran), ``patience`` (the best stopped changing) or
        ``exhausted`` (the policy had nothing new to propose)."""
        self._write(
            {
                "type": "finished",
                "finished_at": _now(),
                "best": [c.number for c in study.best_candidates],
                "unconfirmed": study.unconfirmed is not None,
                "reason": reason,
                "cost": _cost(study.cost),
            }
        )

    def test(self, test: Any, cost: "Cost | None" = None) -> None:
        """The run's test (``test.jsonl``, beside ``study.jsonl``): the baseline and the selection on the test cases,
        their raw trials and their comparison, so every reported test number can be recomputed; ``test_end`` holds
        what the study had spent once it was measured (``cost``)."""
        path = self.directory / "test.jsonl"
        with open(path, "a") as f:
            head = {
                "type": "test",
                "format": "strands-evals.optimization.test",
                "schema": 1,
                "measured_at": _now(),
                "cases": test.cases,
                "repetitions": test.repetitions,
            }
            f.write(json.dumps(head) + "\n")
            for arm in [test.baseline, *test.selected, *test.alternatives]:
                event = {
                    "type": "tested",
                    **arm.to_dict(),
                    "params": _values(arm.params),
                    "trials": self._trials(arm.trials),
                }
                f.write(json.dumps(event, default=str) + "\n")
            end = {"type": "test_end", "missing": {str(k): v for k, v in test.missing.items()}, "cost": _cost(cost)}
            f.write(json.dumps(end) + "\n")

    def stopped(self, error: str, cost: "Cost | None" = None) -> None:
        self._write({"type": "stopped", "stopped_at": _now(), "error": error[:2000], "cost": _cost(cost)})
