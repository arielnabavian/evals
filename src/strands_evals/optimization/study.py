"""The study: candidates, the incumbent and the best candidates, and building, validating and measuring the candidates
of each step (``Search.width`` of them, proposed together and measured at once)."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import json
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Sequence, TypeVar

from .cost import AGENT, PROBES, Cost, Ledger, method_role
from .methods import SELECTORS, Block, Draw, resolve_blocks
from .phases import Anchor, phased
from .projection import (
    AtLeast,
    AtMost,
    Combine,
    Estimate,
    Maximize,
    Minimize,
    Pareto,
    Projection,
    Range,
    _feasible_first,
    constraint_name,
)
from .result import Arm, Result, Run, Test, case_means, paired, pool, pool_rules
from .schedules import Schedule, default_schedule
from .search import Search
from .selection import Closest, Selection, View, closest
from .selection import view as decided_view
from .space import TunableFactory, Values, tunable
from .space.space import JSON_HINT, check_json, json_problem
from .split import Split
from .trials import SCORED, USAGE, Task, Trial, TrialStatus, invoke, run_trials

T = TypeVar("T")


@dataclass(frozen=True)
class Candidate:
    number: int
    params: dict[str, Any]  # the complete candidate: exactly the variables that exist under its choices
    status: str  # complete | partial (some trial or value missing) | failed (nothing measured) | invalid (not run)
    trials: tuple[Trial, ...] = field(repr=False)
    estimates: dict[str, Estimate] = field(repr=False)
    eligible: bool  # every case has every objective and constraint metric: the candidate can be held and told
    violations: tuple[float | None, ...]  # one per constraint, in order: <= 0 when met (None: not measured)
    proposed: tuple[str, ...]  # the blocks whose methods proposed values in this candidate (none for the baseline)
    base: int | None  # the candidate whose values were held for everything not proposed
    objective: int | None  # with Independent, the objective whose methods proposed it
    origin: dict[str, dict[str, Any]]  # per proposed block: which method proposed its values, and how
    drawn: dict[str, tuple[str, ...]] = field(default_factory=dict)  # per proposed block, the variables it gave
    inherited: dict[str, int] = field(
        default_factory=dict
    )  # variables ``base`` lacks, from the best candidate with them
    defaulted: tuple[str, ...] = ()  # variables no candidate had yet: at their defaults
    scope: int | None = None  # the scope (schedule run or round) it ran in; None for the baseline
    path: tuple[int, ...] = ()  # the scopes it ran in, outermost first
    role: str = "step"  # baseline | outer (opens a round) | step
    step: int | None = None  # its index in its schedule's run, where the schedule counts them
    slot: int = 0  # its place among the candidates its step proposed together (see Search.width)
    reason: str | None = None  # why the candidate is invalid
    aim: int | None = None  # with several method copies, the copy that proposed it (Independent's objective; a phase)
    phase: str | None = None  # the phase it ran in (PrimaryFirst: "primary" or "cost"); None outside phases
    _factory: TunableFactory | None = field(default=None, repr=False, compare=False)
    _artifacts: Path | None = field(default=None, repr=False, compare=False)  # where its stored values are

    @property
    def metrics(self) -> dict[str, float]:
        """Each metric's mean over the cases."""
        return {m: e.mean for m, e in self.estimates.items() if e.mean is not None}

    @property
    def cases(self) -> dict[str, dict[str, float]]:
        """Each case's metrics, averaged over its scored repetitions."""
        out: dict[str, dict[str, list[float]]] = {}
        for a in self.trials:
            if a.status in SCORED:
                for m, v in a.metrics.items():
                    out.setdefault(a.case, {}).setdefault(m, []).append(v)
        return {c: {m: sum(vs) / len(vs) for m, vs in ms.items()} for c, ms in out.items()}

    @property
    def feasible(self) -> bool | None:
        """Every constraint is met; None when the candidate is not eligible (it cannot be judged)."""
        if not self.eligible:
            return None
        return all(v is not None and v <= 0 for v in self.violations)

    def agent(self, **resources: Any) -> Any:
        """Build the agent this candidate measured."""
        assert self._factory is not None
        return self._factory(**resources, **self._factory.bind(Values(self.params), artifacts=self._artifacts))


class Scope:
    """One run of one schedule: the blocks it may change, how many steps it may still run, and the candidates it holds
    values from. A round of ``Nested`` is a child scope of the run it belongs to; ``optimize()`` opens the root.

    Policies create candidates only through a scope (``step``, or ``candidate`` for a step of one) and rounds only
    through ``child``, so every candidate's place in the schedule is recorded, the budget is kept in one place, and
    every result reaches the right methods. A step proposes ``Search.width`` candidates from the same incumbent and
    measures them together; the budget counts steps, kept in candidates (``slots``): a step costs ``Search.width`` of
    them, a round's opening proposal what its round says (``opener_slots``: one with ``Nested(opener="candidate")``),
    and a last step with fewer slots left than the width proposes only that many."""

    def __init__(
        self,
        study: "Study",
        *,
        schedule: Schedule,
        node: str,
        blocks: Sequence[str],
        limit: int,
        parent: "Scope | None" = None,
        round: int | None = None,
        outer: str | None = None,
        objective: int | None = None,
        capacity: int | None = None,
        opener_slots: int | None = None,
        phase: str | None = None,
        seed: "Candidate | None" = None,
    ) -> None:
        """``limit``: the steps this scope may run; ``capacity``: the same budget in candidates, when it is not
        ``limit`` x ``Search.width`` (a round whose opener costs one candidate); ``opener_slots``: what a round's
        opening proposal costs (default: a step's worth); ``phase``: the phase it runs (inherited by its rounds);
        ``seed``: a candidate it holds until one of its own is better (it opens from it)."""
        self.study, self.schedule, self.node, self.parent = study, schedule, node, parent
        self.blocks, self.limit = tuple(blocks), limit
        self.round, self.outer, self.objective = round, outer, objective
        self.phase = phase if phase is not None else (parent.phase if parent is not None else None)
        self.seed = seed
        # A rule that ends this scope early (a phase's: PrimaryFirst), asked whenever its budget is: once it says
        # so, the scope and the rounds inside it have no steps left.
        self.stop: Callable[["Scope"], bool] | None = None
        self.history: list[tuple[int | None, tuple[int, ...]]] = []  # its steps (and its rounds'): base, measured
        self.width = study.search.width
        self.capacity = limit * self.width if capacity is None else capacity
        self.opener_slots = self.width if opener_slots is None else opener_slots
        self.candidates: list[Candidate] = []  # the candidates run in this scope and its children
        self.steps = 0  # the steps run in this scope and its children
        self.used = 0  # the budget they spent, in candidates (a step: Search.width; an opener: opener_slots)
        self._deferred: dict[str, tuple[Draw, int]] = {}
        self.id = study._open(self)

    @property
    def kind(self) -> str:
        return type(self.schedule).__name__

    @property
    def path(self) -> tuple[int, ...]:
        return (*self.parent.path, self.id) if self.parent else (self.id,)

    @property
    def slots(self) -> int:
        """The candidates' worth of budget this scope has left (never more than the scopes around it; none once the run
        has lost patience)."""
        own = self.capacity - self.used
        if self.stop is not None and own > 0 and self.stop(self):
            return 0
        if self.parent is None:
            return 0 if self.study._out_of_patience() else own
        return min(own, self.parent.slots)

    @property
    def budget(self) -> int:
        """The steps this scope may still run (never more than the scopes around it; none once the run has lost
        patience); a last step narrower than ``Search.width`` counts as one."""
        slots = self.slots
        return -(-slots // self.width) if slots > 0 else 0

    def turn(self) -> int:
        """The objective whose methods propose next: this round's, or the next in turn (with ``Independent``)."""
        return self.objective if self.objective is not None else self.study.turn()

    def incumbent(self, objective: int | None = None) -> Candidate:
        """The candidate other values are held at: the best of this scope's candidates (the root holds from every
        candidate in the study), else the enclosing scope's."""
        k = objective if objective is not None else self.objective
        pool = self.study.candidates if self.parent is None else self.candidates
        if self.seed is not None and all(c.number != self.seed.number for c in pool):
            pool = [self.study.candidates[self.seed.number], *pool]
        held = self.study._held(pool, k)
        if held is not None:
            return held
        return self.parent.incumbent(objective) if self.parent else self.study.baseline

    def best(self, objective: int | None = None) -> Candidate | None:
        """The best of this scope's own candidates, or None when none of them can be compared."""
        k = objective if objective is not None else self.objective
        return self.study._held(self.candidates, k)

    def step(
        self,
        update: Sequence[str],
        *,
        step: int | None = None,
        base: Candidate | None = None,
        objective: int | None = None,
    ) -> list[Candidate]:
        """Propose ``Search.width`` candidates that change ``update`` (the blocks among them that exist under the
        structure being built), each holding everything else at ``base`` (default: this scope's incumbent), and
        measure them together: one step. Returns them, or an empty list, without running anything, when none of the
        blocks has anything new to propose. With fewer slots left than ``Search.width`` (after rounds' openers, which
        cost one candidate each), the step proposes as many as are left."""
        width = min(self.width, self.slots)
        return [
            c
            for c, _ in self._run(
                update, role="step", step=step, defer=(), base=base, objective=objective, width=width, cost=width
            )
        ]

    def candidate(
        self,
        update: Sequence[str],
        *,
        role: str = "step",
        step: int | None = None,
        defer: Sequence[str] = (),
        base: Candidate | None = None,
        objective: int | None = None,
    ) -> Candidate | None:
        """A step of one candidate (see ``step``); a round opens with one (``role="outer"``). Returns None when none of
        the blocks has anything to propose. ``defer`` blocks are not told the result now: ``tell`` gives them the
        result that stands for their draw. It costs a step; a round's opener, ``opener_slots``."""
        cost = self.opener_slots if role == "outer" else min(self.width, self.slots)
        made = self._run(update, role=role, step=step, defer=defer, base=base, objective=objective, width=1, cost=cost)
        if not made:
            return None
        candidate, draws = made[0]
        for block in defer:
            if block in draws:
                self._deferred[block] = (draws[block], candidate.number)
        return candidate

    def _run(
        self,
        update: Sequence[str],
        *,
        role: str,
        step: int | None,
        defer: Sequence[str],
        base: Candidate | None,
        objective: int | None,
        width: int,
        cost: int,
    ) -> list[tuple[Candidate, dict[str, Draw]]]:
        if self.budget <= 0:
            raise RuntimeError(f"scope {self.id} ({self.kind}) has no steps left")
        allowed = set(self.blocks) | ({self.outer} if role == "outer" and self.outer else set())
        outside = [b for b in update if b not in allowed]
        if outside:
            raise ValueError(f"blocks {outside} are outside this scope's blocks {list(self.blocks)}")
        k = objective if objective is not None else self.turn()
        return self.study._step(
            update,
            base if base is not None else self.incumbent(k),
            objective=k,
            scope=self,
            role=role,
            step=step,
            defer=defer,
            width=width,
            cost=cost,
        )

    def child(
        self,
        name: str,
        schedule: Schedule,
        *,
        blocks: Sequence[str],
        limit: int,
        round: int | None = None,
        outer: str | None = None,
        objective: int | None = None,
        capacity: int | None = None,
        opener_slots: int | None = None,
        phase: str | None = None,
        seed: "Candidate | None" = None,
    ) -> "Scope":
        """Open a scope inside this one (a round, or a phase), where ``schedule`` (this schedule's child ``name``)
        runs."""
        return Scope(
            self.study,
            schedule=schedule,
            node=f"{self.node}.{name}" if self.node else name,
            blocks=blocks,
            limit=limit,
            parent=self,
            round=round,
            outer=outer,
            objective=objective,
            capacity=capacity,
            opener_slots=opener_slots,
            phase=phase,
            seed=seed,
        )

    def narrow(self, candidate: Candidate) -> None:
        """Keep only the blocks with variables under ``candidate``'s choices."""
        self.blocks = tuple(b for b in self.blocks if self.study.active(b, candidate.params))

    def tell(self, block: str, candidate: Candidate | None) -> None:
        """Give ``block``'s deferred draw the result that stands for it (None: it failed)."""
        draw, number = self._deferred.pop(block)
        self.study._tell(block, draw, number, candidate, objective=self.objective, scope=self)

    def __enter__(self) -> "Scope":
        return self

    def __exit__(self, kind: Any, *exc: Any) -> bool:
        if kind is not None:  # interrupted: the scope stays open in the record, and its draws untold
            return False
        for block in list(self._deferred):  # a draw never told (the round ended early): it failed
            self.tell(block, None)
        self.study._close(self)
        return False


def _score_range(evaluator: Any) -> Range:
    """Where an evaluator's scores lie: its ``score_range``, else 0 to 1."""
    lo, hi = getattr(evaluator, "score_range", (0.0, 1.0))
    return (None if lo is None else float(lo), None if hi is None else float(hi))


class Study:
    def __init__(
        self,
        *,
        agent: TunableFactory | Callable[..., Any],
        experiment: Any,
        objectives: Maximize | Minimize | Combine,
        constraints: Sequence[AtMost | AtLeast] = (),
        search: Search | None = None,
        repetitions: int = 1,
        agent_error_score: float | str | None = "worst",
        max_concurrency: int = 4,
        task: Task = invoke,
        retries: int = 2,
        selection: Selection | None = None,
        split: Split | None = None,
        record: Any = None,
        log: Callable[[str], None] | None = print,
    ) -> None:
        """``objectives``: one objective, or several wrapped in how they combine: ``Pareto(...)`` (jointly),
        ``WeightedSum(...)``, ``Chebyshev(...)``, ``Lexicographic(...)`` or ``Independent(...)``. An objective or
        constraint names a metric: an evaluator's (its name), the agent's usage (``tokens``, ``seconds``, ...), or a
        ``Derived`` one. An evaluator's scores lie in 0 to 1 unless it says otherwise (``score_range = (0, None)``).
        ``repetitions``: how many times each candidate runs every case (each run is a trial; a candidate's estimates
        are case-level means with 95% intervals).
        ``agent_error_score`` is the evaluator score a trial gets when its agent raised: ``"worst"`` (the end of the
        metric's range that is worst for the goals using it, e.g. 0 for a maximized score in [0, 1]), a number, or
        ``None`` (left out).
        ``task(agent, case)`` runs one trial with the agent built for it and returns its output, as a Strands Evals task
        does (or a dict with ``output``, ``trajectory``, ...); the default calls ``agent.invoke_async(case.input)``. It
        may be async. ``retries``: how many more times a trial runs when it measured nothing because something other
        than the agent failed (the task raised ``TrialInvalid``, or a connection failed; Strands Evals retries
        throttling).
        ``search``: how to search (``Search``: blocks, each a method and the variables it proposes, and the schedule
        that says when each moves); the default puts every variable in a block for its type and builds the schedule
        from the space. Values already measured are never measured again (``Search(redraws=...)``).
        ``selection``: how the study decides (which candidate it holds, what methods are told, which candidates are
        best) and selects when a run ends: by default with a model of every goal metric, constraints judged at a
        probability, and the best measured again before selecting (``Selection``).
        ``split``: which of the experiment's cases each part uses (``Split``): the search runs on ``train``; the
        finalists and the baseline are measured on ``validation`` and the selection is made there; the selection and
        the baseline are then measured on ``test``, which nothing before saw. With ``Split(rotate=True)`` the study runs
        once per test share, the runs side by side, each on fresh search state, and pools their tests (``Result``).
        Without a split,
        the search and the selection use every case, and nothing is tested.
        ``max_concurrency``: the most trials running at once; a step's candidates, and a rotation's runs while they
        search, divide it.
        ``log`` receives one line per finished candidate (``None`` for silence); ``record`` is a directory where the
        study and every candidate are written as they finish (``record.py``), from the first ``optimize()`` on.
        What the study spends, in tokens, by role (the agent; each method's own model calls) and phase (search,
        re-measurement, validation, test), is kept as it runs (``cost``, ``cost.py``) and given with what it found
        (``Result.cost``).

        ``experiment`` is a Strands Evals experiment: its cases, scored by its evaluators, run in this process."""
        self._args = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.agent = agent if isinstance(agent, TunableFactory) else tunable()(agent)
        self.split = split
        self._runs = split.assign(experiment.cases) if split is not None else None
        if split is not None and split.runs > 1:
            try:  # each run searches from the search as given, untouched by the others
                self._search_template = self._copy_search(search)
            except Exception as exc:  # noqa: BLE001 - reported with what cannot be copied
                raise ValueError(f"Split(rotate=True) needs a search it can copy for each run: {exc}") from exc
        roles = self._runs[0] if self._runs is not None and len(self._runs) == 1 else None
        if roles is not None:
            from ..experiment import Experiment

            experiment = Experiment(cases=roles.train, evaluators=experiment.evaluators)
        self.experiment = experiment
        if isinstance(objectives, (list, tuple)):
            raise TypeError(
                "give several objectives wrapped in how they combine: objectives=Pareto(...) (jointly), "
                "WeightedSum(...), Chebyshev(...), Lexicographic(...) or Independent(...)"
            )
        self.combine = objectives if isinstance(objectives, Combine) else Pareto(objectives)
        evaluators = {e.get_name(): _score_range(e) for e in experiment.evaluators}
        self.selection = selection if selection is not None else Selection()
        self.projection = Projection(
            list(self.combine.objectives), constraints, evaluators=evaluators, agent_error_score=agent_error_score
        )
        self.combine.check(self.projection)
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        self.repetitions, self.max_concurrency, self.log = repetitions, max_concurrency, log
        self.task, self.retries = task, retries
        self.fit: Any = None  # the model of every goal metric, refitted after each candidate (model.py)
        self._trackers: dict[str, Any] = {}  # held-out diagnostics, per metric, across fits
        self._model_problem: str | None = None  # why the last fit is not used, if it is not
        self._selected: list[Candidate] | None = None  # the selection the last run ended with
        self._unconfirmed: Closest | None = None  # when it met no constraint: how it was chosen (Selection.unmet)
        self.validation = self._validation_experiment(roles.validation if roles is not None else None)
        self.validated: list[Candidate] = []  # the last run's finalists and baseline, as measured on validation
        self.test_cases: list[Any] = list(roles.test) if roles is not None else []
        self.tested: Test | None = None  # the test, once measured: a study tests once
        self.result: Result | None = None
        if self.validation is not None and self.selection.finalists <= 0:
            raise ValueError("the split has validation cases, but Selection(finalists=0) selects without them")
        self.search = search if search is not None else Search()
        self._check(set(evaluators))
        self.params = {p.key: p for p in self.agent.params()}
        check_json(self.params.values())  # every value is recorded as JSON: a default or option that is not, fails here
        self._skills = [k for k, p in self.params.items() if p.type == "skills"]  # values stored in the artifact store
        self._stored_defaults: dict[str, dict[str, str]] = {}  # each default library, as declared and as stored
        blocks, self._names = resolve_blocks(self.params, self.search.blocks, self.agent.references())
        self.blocks = {b.name: b for b in blocks}
        self.block_of = {key: b.name for b in self.blocks.values() for key in b.keys}
        # The schedule: the search's or the one built from the space, in phases when the goal has a cost objective
        # (phases.PrimaryFirst; Search(phases=None) never).
        self.schedule = phased(self, self.search.schedule or default_schedule(self))
        # What each method copy searches for (its aim): the goal's own combination, once (or once per objective with
        # Independent); a schedule with parts that search for other combinations (phases) says which.
        self.aims = self.schedule.aims(self) or [self.combine] * self.combine.instances(len(self.combine.objectives))
        # Every block's method as given, plus a fresh copy of it for each further aim.
        self.methods = [
            {name: b.method if k == 0 else b.method.copy(k) for name, b in self.blocks.items()}
            for k in range(len(self.aims))
        ]
        # which copies are told each result as it comes: an aim that waits (phase B's) hears nothing until its phase
        # starts, then everything at once (_evidence)
        self._listening = [not getattr(a, "waits", False) for a in self.aims]
        self.anchor: Anchor | None = None  # a schedule's anchor for later decisions (phase A's result, A*)
        self._pick: Candidate | None = None  # the one candidate the run ships, when its selection is a set
        self._alternatives: dict[str, list[int]] = {}  # what other rules would ship (Selection.also)
        self._finalist_numbers: list[int] = []  # the last run's finalists
        self.schedule.check(list(self.blocks), self)
        self.candidates: list[Candidate] = []
        self._turn = 0
        self._scopes = 0
        self._patience: int | None = None
        self._unchanged = 0  # steps since the standing (see _standing) last changed
        # A step whose blocks include a ``concurrent`` method (reflection) makes its proposals in threads; methods
        # that hold state between proposals (Optuna's studies, Thompson's generator) and the record still propose one
        # at a time, under this lock. A step of such methods alone proposes in slot order, without threads.
        self._lock = threading.RLock()
        self._folds: _Folds | None = None  # a rotation's runs, when this study is one of them
        self._record_dir = record  # opened by the first optimize(): building a Study touches nothing on disk
        self.record: Any = None
        self._ledger = Ledger()  # what the study spends, in tokens, as it runs (cost.py)

    def name_of(self, block: Block) -> str:
        """The name of one of this study's blocks: a ``Block`` given in its ``Search``, or one it resolved."""
        if id(block) in self._names:
            return self._names[id(block)]
        if isinstance(block, Block) and self.blocks.get(block.name) is block:
            return block.name
        raise ValueError(f"{block!r} is not one of this study's blocks: give it in Search(blocks=...)")

    def _validation_experiment(self, cases: Sequence[Any] | None) -> Any:
        if not cases:
            return None
        from ..experiment import Experiment

        return Experiment(cases=list(cases), evaluators=self.experiment.evaluators)

    def _check(self, evaluators: set[str]) -> None:
        if not self.projection.objectives:
            raise ValueError("give at least one objective")
        names = [case.name for case in self.experiment.cases]
        if not names:
            raise ValueError("the experiment has no cases")
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("give every Case a unique name: each case's metrics are recorded by name")
        available = evaluators | set(USAGE) | set(self.projection.derived)
        for goal in [*self.projection.objectives, *self.projection.constraints]:
            if goal.metric not in available:
                raise ValueError(f"unknown metric {goal.metric!r}; the metrics are {sorted(available)}")

    @property
    def baseline(self) -> Candidate:
        return self.candidates[0]

    @property
    def cost(self) -> Cost:
        """What the study has spent so far, in tokens, by role and phase (``cost.py``): every attempt of every trial
        and every model call its methods made themselves."""
        return self._ledger.cost()

    def _roles(self) -> dict[str, dict[str, Any]]:
        """What identifies each role of the cost (the record's header): the agent's model, as the agent built at the
        defaults (the baseline's, so its first trial's) names it; each method's that calls a model itself."""
        return {
            AGENT: {"model": self._agent_model()},
            **{
                method_role(str(name)): {"block": name, "method": type(b.method).__name__, "model": b.method.model_id}
                for name, b in self.blocks.items()
                if b.method.model_id is not None
            },
            PROBES: {},
        }

    def _agent_model(self) -> str | None:
        """The model ID of the agent built at the defaults, or None when it cannot be read (or built)."""
        try:
            values = self.agent.suggest(_Source(self, {}, {}, None))
            agent = self.agent(**self.agent.bind(values, artifacts=self._artifact_root()))
            model = getattr(agent, "model", None)
            get_config = getattr(model, "get_config", None)
            config = get_config() if callable(get_config) else getattr(model, "config", None)
            model_id = config.get("model_id") if isinstance(config, dict) else None
        except Exception:  # noqa: BLE001 - the header records what it can; a factory that cannot build says so later
            return None
        return model_id if isinstance(model_id, str) else None

    # ---- the space ------------------------------------------------------------------------------------------------
    def ancestors(self, key: str) -> list[str]:
        """The choices ``key`` exists under, innermost first."""
        out, at = [], self.params[key].active_when
        while at is not None:
            out.append(at[0])
            at = self.params[at[0]].active_when
        return out

    def active(self, block: str, params: dict[str, Any]) -> bool:
        """Whether any of the block's variables exists in ``params`` (a candidate holds exactly the variables its
        choices make exist)."""
        return any(k in params for k in self.blocks[block].keys)

    def structural_order(self) -> list[str]:
        """The structural blocks, a block whose variables exist only under another's choices after that block."""
        structural = [name for name, b in self.blocks.items() if b.structural]
        after = {
            name: {self.block_of[s] for k in self.blocks[name].keys for s in self.ancestors(k)} - {name}
            for name in structural
        }
        order: list[str] = []
        while len(order) < len(structural):
            ready = [n for n in structural if n not in order and after[n] <= set(order)]
            if not ready:
                raise ValueError(f"the structural blocks depend on each other in a cycle: {after}")
            order.append(ready[0])
        return order

    def artifact_path(self, value: str | None) -> Path | None:
        """The path of a value the study stored, as a candidate (and the record) holds it: a skill-library version,
        ``skills/<hash>``, is relative to the study's artifact store (``artifacts``), so a record can be moved. An
        absolute path (records written before 2026-10-24) is returned as it is; None (an empty library) stays None."""
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.artifacts / path

    def _artifact_root(self) -> Path | None:
        """The artifact store, where the space has values stored in it (skill libraries); else None (no directory is
        made)."""
        return self.artifacts if self._skills else None

    def _store_defaults(self) -> None:
        """Store each default skill library in the artifact store, so the baseline holds a stored version like every
        candidate after it (a default that is not a directory is left as declared, and the baseline says why)."""
        from .space.skills import store_directory

        for key in self._skills:
            info = self.params[key]
            if key in self._stored_defaults or not isinstance(info.default, str) or not os.path.isdir(info.default):
                continue
            value = store_directory(info.default, self.artifacts)
            self._stored_defaults[key] = {"declared": info.default, "value": value}
            self.params[key] = dataclasses.replace(info, default=value)

    @property
    def artifacts(self) -> Path:
        """Where the study keeps what its methods write (skill-library versions, curators' decisions): ``artifacts/``
        in the record directory, or a temporary directory for a study without a record."""
        if getattr(self, "_artifacts", None) is None:
            import tempfile

            self._artifacts = (
                Path(self._record_dir) / "artifacts"
                if self._record_dir is not None
                else Path(tempfile.mkdtemp(prefix="study-artifacts-"))
            )
        return self._artifacts

    def block_order(self) -> list[str]:
        """Every block: the structural ones outermost first, then the rest in the order the agent declares them."""
        position = {key: i for i, key in enumerate(self.params)}
        rest = sorted(
            (b for b in self.blocks if not self.blocks[b].structural),
            key=lambda b: min(position[k] for k in self.blocks[b].keys),
        )
        return [*self.structural_order(), *rest]

    # ---- running ----------------------------------------------------------------------------------------------------
    def optimize(self, iterations: int, *, patience: int | None = None) -> Result:
        """Run ``iterations`` more iterations (steps), each proposing ``Search.width`` candidates from the incumbent and
        measuring them together: the search's depth. A study's first call measures the agent at its defaults (the
        baseline) first.

        ``patience``: stop early once the best candidates (as the objectives' combination defines best; while none
        meets the constraints, those that miss them least) have not changed for this many iterations in a row;
        omitted, the run uses every iteration. A round in progress ends there, and its outer proposal is told the
        round's best so far. A run also ends early when its schedule has nothing new to propose (every proposal repeats
        a measured candidate). With test cases, the selection and the baseline are measured on them last; with
        ``Split(rotate=True)``, the runs search, select and test side by side (``iterations`` and ``patience`` per
        run). Returns what the study found (``Result``)."""
        if self._runs is not None and len(self._runs) > 1:
            return self._optimize_runs(iterations, patience)
        if self.tested is not None:
            raise RuntimeError(
                "this study has measured its test cases; searching again would use them: make a new Study"
            )
        if iterations < 0:
            raise ValueError("iterations must be at least 0")
        if patience is not None and patience < 1:
            raise ValueError("patience must be at least 1 (or None, to run every iteration)")
        if iterations > 0 and self._folds is None:
            self._warn_concurrency(runs=1, cases=len(self.experiment.cases))
        if not self.candidates:
            self._store_defaults()
        if self._record_dir is not None and self.record is None:
            from .record import Record

            self.record = Record(self._record_dir)
            self.record.study(self)
        if self.record is not None:
            self.record.run(iterations, patience)
        self._patience, self._unchanged = patience, 0
        try:
            if not self.candidates:
                made = self._step([], None, objective=None, scope=None, role="baseline", step=None, defer=(), width=1)
                assert made
                baseline = made[0][0]
                if not baseline.eligible:
                    raise RuntimeError(
                        f"the baseline (the agent at its defaults) measured nothing usable: "
                        f"{baseline.status} ({baseline.reason or self._first_error(baseline)})"
                    )
                if not any(t.status == TrialStatus.COMPLETED for t in baseline.trials):
                    raise RuntimeError(  # its scores would be the failure value alone, not a measurement
                        f"every trial of the baseline (the agent at its defaults) failed: {self._first_error(baseline)}"
                    )
            left = 0
            self._selected, self._unconfirmed = None, None
            if iterations > 0:
                with Scope(self, schedule=self.schedule, node="", blocks=list(self.blocks), limit=iterations) as root:
                    self.schedule.run(root)
                    left = root.budget
            self._select()
            if self.test_cases:
                self.tested = self._test()
        except BaseException as exc:
            if self.record is not None:
                self.record.stopped(f"{type(exc).__name__}: {exc}", cost=self.cost)
            raise
        if self.record is not None:
            reason = "patience" if self._out_of_patience() else "exhausted" if left > 0 else "candidates"
            self.record.finish(self, reason)
            self._effects()
        if self.log:
            self.log(f"{'cost':<13}{self.cost.describe()}")
        self._patience = None
        cases = (
            self._runs[0].names()
            if self._runs is not None
            else {"train": [c.name for c in self.experiment.cases], "validation": [], "test": []}
        )
        run = Run(
            1,
            cases,
            list(self.best_candidates),
            self.tested,
            str(self._record_dir) if self._record_dir else None,
            cost=self.cost,
            unconfirmed=self._unconfirmed is not None,
            pick=self._pick.number if self._pick is not None else None,
            alternatives=dict(self._alternatives),
            anchor=self.anchor.number if self.anchor is not None else None,
        )
        self.result = Result(runs=[run], test=self.tested)
        self._write_result(self.result)
        return self.result

    def _optimize_runs(self, iterations: int, patience: int | None) -> Result:
        """Rotation: one study per run, each built afresh from this one's arguments with that run's cases, the runs
        side by side (a thread each), then their tests pooled. The runs divide ``max_concurrency`` among those still
        searching; when one fails, the others stop at their next step and its error is raised. The runs and their cases
        are stated (log, ``split.json``) before any runs."""
        assert self._runs is not None
        if self.result is not None:
            raise RuntimeError("this study has run its rotations and measured their test cases: make a new Study")
        root = Path(self._record_dir) if self._record_dir is not None else None
        sizes = ", ".join(f"{len(getattr(self._runs[0], r))} {r}" for r in ("train", "validation", "test"))
        if self.log:
            self.log(
                f"split: {len(self._runs)} runs side by side, each searching from scratch ({sizes}); every case "
                "tested once"
            )
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            plan = {"split": self.split.describe() if self.split else None, "runs": [a.names() for a in self._runs]}
            (root / "split.json").write_text(json.dumps(plan, indent=1))
        if iterations > 0:
            self._warn_concurrency(runs=len(self._runs), cases=max(len(r.train) for r in self._runs))
        folds = _Folds(len(self._runs))
        children = []
        for i, roles in enumerate(self._runs, 1):
            child = Study(
                **{
                    **self._args,
                    "split": Split(train=roles.train, validation=roles.validation or None, test=roles.test or None),
                    "search": self._copy_search(self._search_template),
                    "record": root / f"run-{i}" if root is not None else None,
                    "log": (lambda line, i=i: self.log(f"run {i}: {line}")) if self.log else None,
                }
            )
            child._folds = folds
            children.append(child)
        found: list[Any] = [None] * len(children)

        def search(k: int) -> None:
            try:
                found[k] = children[k].optimize(iterations, patience=patience).runs[0]
            except BaseException as exc:  # noqa: BLE001 - raised below, once every run has stopped
                found[k] = exc
                folds.failed.set()
            finally:
                folds.finish()

        threads = [
            threading.Thread(target=search, args=(k,), name=f"run-{k + 1}", daemon=True) for k in range(len(children))
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except BaseException:  # interrupted: the runs stop at their next step
            folds.failed.set()
            raise
        errors = [(k + 1, out) for k, out in enumerate(found) if isinstance(out, BaseException)]
        if errors:
            k, error = next(((k, e) for k, e in errors if not isinstance(e, _Stopped)), errors[0])
            if hasattr(error, "add_note"):
                error.add_note(f"in run {k} of {len(children)}")
            raise error
        runs = [dataclasses.replace(r, index=k) for k, r in enumerate(found, 1)]
        self.runs = runs
        train = [len(r.train) + len(r.validation) for r in self._runs]
        self.result = Result(
            runs=runs,
            test=pool(runs, self.projection.goal_metrics, self.projection.value, train),
            picks=pool_rules(runs, self.projection.goal_metrics, self.projection.value, train),
        )
        if self.log and self.result.cost is not None:
            self.log(f"{'cost':<13}(every run) {self.result.cost.describe()}")
        self._write_result(self.result)
        return self.result

    def _copy_search(self, search: Search | None) -> Search | None:
        """A copy of ``search`` whose methods start afresh, sharing the agent's declared variables (blocks refer to
        them by identity)."""
        declared = {id(obj): obj for obj, _keys, _variable in self.agent.references()}
        return copy.deepcopy(search, memo=dict(declared))

    def _warn_concurrency(self, runs: int, cases: int) -> None:
        """Warn when a step's trials, in every run searching side by side, cannot all run at once: the step then runs
        them in batches, so width (and the runs side by side) save less wall time than they could."""
        width, trials = self.search.width, cases * self.repetitions
        need = runs * width * trials
        if runs * width > 1 and need > self.max_concurrency:
            side = f", in each of {runs} runs side by side" if runs > 1 else ""
            warnings.warn(
                f"Search(width={width}) measures {width} x {trials} trials ({cases} cases x {self.repetitions} "
                f"repetitions) in a step{side}: {need} trials at once, but max_concurrency is {self.max_concurrency}, "
                f"so steps run in batches and take longer; give max_concurrency={need} to run them at once",
                stacklevel=3,
            )

    def _concurrency(self) -> int:
        """The trials this study may run at once: ``max_concurrency``, divided among a rotation's runs still
        searching."""
        return max(1, self.max_concurrency // (self._folds.running if self._folds is not None else 1))

    def _write_result(self, result: Result) -> None:
        if self._record_dir is not None:
            Path(self._record_dir).mkdir(parents=True, exist_ok=True)
            (Path(self._record_dir) / "result.json").write_text(json.dumps(result.to_dict(), indent=1, default=str))

    def _test(self) -> Test:
        """Measure the baseline and the selection (a set's members for a Pareto goal; the baseline again when nothing
        was selected) on the test cases, at the same time, and compare them case by case."""
        from ..experiment import Experiment

        experiment = Experiment(cases=self.test_cases, evaluators=self.experiment.evaluators)
        selected = [c for c in (self._selected or [])]
        measured = [self.baseline, *(c for c in selected if c.number != self.baseline.number)]
        # what the other rules ship (Selection.also), when it is not already measured: their own arms
        extra = sorted({n for ns in self._alternatives.values() for n in ns} - {c.number for c in measured})
        measured += [self.candidates[n] for n in extra]
        share = max(1, self._concurrency() // len(measured))

        async def measure() -> list[Any]:
            runs = [
                self._measure(
                    Values(c.params), experiment, repetitions=self.repetitions, max_concurrency=share, phase="test"
                )
                for c in measured
            ]
            return list(await asyncio.gather(*runs))

        trials = {
            c.number: tuple(self.projection.prepare(t) for t in out)
            for c, out in zip(measured, self._run(measure()), strict=True)
        }
        for c in measured:
            if not any(t.status == TrialStatus.COMPLETED for t in trials[c.number]):
                error = next((t.error for t in trials[c.number] if t.error), "no trial completed")
                raise RuntimeError(f"test: every trial of candidate {c.number} failed: {error}")
        names = [c.name for c in self.test_cases]
        metrics = list(self.projection.goal_metrics)

        pick = self._pick.number if self._pick is not None else None

        def rules(number: int, role: str) -> tuple[str, ...]:
            """The rules that ship this arm: the selection's pick, and Selection.also's ([] ships the baseline)."""
            ships = [r for r, ns in self._alternatives.items() if (ns and number in ns) or (not ns and number == 0)]
            if role in ("selected", "fallback") and (pick is None or number == pick):
                ships = ["selection", *ships]
            return tuple(ships)

        def arm(c: Candidate, role: str) -> Arm:
            own = trials[c.number]
            estimates = self.projection.estimates(own)
            means = {m: e.mean for m, e in estimates.items() if e.mean is not None}
            shipped = rules(c.number, role) if (self._alternatives or self._pick is not None) else ()
            if role == "baseline":
                return Arm(c.number, role, dict(c.params), own, means, rules=shipped)
            unconfirmed = role == "selected" and self._unconfirmed is not None
            difference, ratio = {}, {}
            for m in metrics:
                d, r, _ = paired(
                    case_means(own, m, self.projection.value), case_means(base_trials, m, self.projection.value)
                )
                if d is not None:
                    difference[m] = d
                if r is not None:
                    ratio[m] = r
            return Arm(
                c.number,
                role,
                dict(c.params),
                own,
                means,
                difference,
                ratio,
                unconfirmed=unconfirmed,
                pick=role == "selected" and pick is not None and c.number == pick and len(selected) > 1,
                rules=shipped,
            )

        base_trials = trials[self.baseline.number]
        baseline = arm(self.baseline, "baseline")
        arms = [arm(c, "selected") for c in selected] or [arm(self.baseline, "fallback")]
        others = [arm(self.candidates[n], "alternative") for n in extra]
        scored = {n: {t.case for t in ts if t.status in SCORED} for n, ts in trials.items()}
        missing = {n: sum(1 for case in names if case not in got) for n, got in scored.items()}
        test = Test(
            cases=names,
            repetitions=self.repetitions,
            baseline=baseline,
            selected=arms,
            missing=missing,
            alternatives=others,
        )
        if self.record is not None:
            self.record.test(test, cost=self.cost)
        if self.log:
            for a in [*arms, *others]:
                parts = "  ".join(
                    f"{m} {a.means.get(m, float('nan')):.3g} vs {baseline.means.get(m, float('nan')):.3g}"
                    + (f" ({d.mean:+.3g} [{d.ci90[0]:+.3g}, {d.ci90[1]:+.3g}])" if d.ci90 else "")
                    for m, d in a.difference.items()
                )
                role = f"{a.role}, unconfirmed" if a.unconfirmed else a.role
                role += ", the pick" if a.pick else ""
                role += f"; shipped by {', '.join(a.rules)}" if a.rules else ""
                self.log(f"{'test':<13}candidate {a.number} ({role}) on {len(names)} cases: {parts}")
        return test

    def _standing(self) -> frozenset[int]:
        """What patience watches: the best candidates or, while none meets the constraints, those that miss them least
        (so a run that is still closing in on its limits is not stopped)."""
        best = self._decided_best() or [v.candidate for v in _feasible_first(self._views(self.candidates))]
        return frozenset(t.number for t in best)

    # ---- deciding ---------------------------------------------------------------------------------------------------
    @property
    def by_model(self) -> bool:
        """Whether decisions are being made with the model now (it is fitted, and its diagnostics have not flagged)."""
        return self.selection.by == "model" and self.fit is not None and self._model_problem is None

    def view(self, candidate: Candidate) -> View:
        """The candidate as decisions see it: its decided values, and its constraints judged at the required
        probability (by measured means when the study does not decide with the model); with an anchor, judged against
        it too (``phases.Anchor.rejudge``)."""
        fit = self.fit if self.by_model else None
        v = decided_view(candidate, self.projection, self.selection, fit, self.baseline)
        if self.anchor is not None and candidate.eligible:
            a = self.candidates[self.anchor.number]
            reference = (
                v
                if a.number == candidate.number
                else decided_view(a, self.projection, self.selection, fit, self.baseline)
            )
            v = self.anchor.rejudge(self.projection, v, reference)
        return v

    def _views(self, pool: Sequence[Candidate]) -> list[View]:
        return [self.view(c) for c in pool]

    def aim(self, objective: int | None) -> Combine:
        """What a method copy searches for: the goal's combination, or the aim of the copy (a phase's)."""
        return self.combine if objective is None else self.aims[objective]

    def _held(self, pool: Sequence[Candidate], objective: int | None) -> Candidate | None:
        held = self.aim(objective).held(self.projection, self._views(pool), objective, self.view(self.baseline))
        return held.candidate if held is not None else None

    def _decided_best(self) -> list[Candidate]:
        best = self.combine.best(self.projection, self._views(self.candidates), self.view(self.baseline))
        return [v.candidate for v in best]

    def _text_blocks(self) -> dict[str, list[str]]:
        """The texts each method rewrites together, by block."""
        out = {}
        for name, block in self.blocks.items():
            keys = [k for k in block.keys if self.params[k].type in ("text", "skills")]
            if keys:
                out[name] = keys
        return out

    def _refit(self) -> None:
        """Fit the model of every goal metric to every eligible candidate (deciding by the model only)."""
        if self.selection.by != "model":
            return
        from . import model

        try:
            fitted = model.fit(
                self.projection,
                self.params,
                self._text_blocks(),
                self.candidates,
                self.baseline,
                [c.name for c in self.experiment.cases],
                trackers=self._trackers,
                previous=self.fit,
            )
        except Exception as exc:  # noqa: BLE001 - a model that cannot be fitted is recorded, and measured means decide
            self._model_problem = f"the model could not be fitted ({type(exc).__name__}: {exc})"
            if self.log:
                self.log(f"model: {self._model_problem}; deciding with measured means")
            return
        if fitted is None:
            return
        self.fit = fitted
        flagged = fitted.flagged
        self._model_problem = (
            f"its held-out diagnostics flagged {', '.join(flagged)}: deciding with measured means" if flagged else None
        )
        if self.record is not None:
            self.record.model(self, fitted, self._model_problem)

    def _ranked(self, pool: Sequence[Candidate], n: int, aim: int | None = None) -> list[Candidate]:
        """The ``n`` best of ``pool`` as decided (for the goal, or for ``aim``'s copy): the held candidate, then the
        held one among the rest, and so on."""
        out: list[Candidate] = []
        rest = list(pool)
        while rest and len(out) < n:
            held = self._held(rest, aim)
            if held is None:
                break
            out.append(held)
            rest = [c for c in rest if c.number != held.number]
        return out

    def _finalists(self) -> list[Candidate]:
        """The best as decided, to measure again before selecting: ``Selection.finalists`` of them, or a Pareto set's
        members (up to ``max(finalists, 5)``)."""
        if self.selection.finalists <= 0:
            return []
        several = self.combine.best_is_set(len(self.projection.objectives))
        own = self.schedule.finalists(self, max(self.selection.finalists, 5) if several else self.selection.finalists)
        if own is not None:  # the schedule's (phases: A* and the cheapest candidates no worse than it)
            return own
        front = self._decided_best()
        n = min(len(front), max(self.selection.finalists, 5)) if several else self.selection.finalists
        if several:
            pool = front
        else:  # as decided: every eligible candidate, or what the combination narrows them to (its front)
            views = self.combine.finalist_pool(self.projection, self._views([c for c in self.candidates if c.eligible]))
            pool = [v.candidate for v in views]
        return self._ranked(pool, n)

    def _select(self) -> None:
        """End a run: measure the finalists again (on the validation cases when there are some, else on the search
        cases, deciding by the model), and select."""
        finalists = self._finalists()
        self._finalist_numbers = [c.number for c in finalists]
        self._pick, self._alternatives = None, {}
        if self.validation is not None and finalists:
            self._select_on_validation(finalists)
            return
        if self.by_model and finalists:
            for c in finalists:
                self._remeasure(c, self.repetitions)
            self._refit()
        views = self._views(self.candidates)
        decision = self.schedule.decide(self, views, self.view(self.baseline), "search")
        if decision is not None:  # anchored (phases): the views judged against A*, the pick, the other rules
            best, views = decision.best, decision.views
            self._pick, self._alternatives = decision.pick, decision.alternatives
            self._log_decision(decision)
        else:
            best = [v.candidate for v in self.combine.best(self.projection, views, self.view(self.baseline))]
            if not best:
                best = self._closest(views, self.view(self.baseline), on="search")
            elif len(self.projection.objectives) == 1:  # one objective selects one candidate, also on a tie
                best = [self._among_equals(best[0], views)]
        self._selected = [self.candidates[c.number] for c in best]
        if self.record is not None:
            self.record.selected(
                self,
                self._selected,
                views={v.number: v for v in views} if decision is not None else None,
                unconfirmed=self._unconfirmed,
                decision=decision,
            )

    def _select_on_validation(self, finalists: list[Candidate]) -> None:
        """Measure the baseline and the finalists on the validation cases and select from those measurements (with a
        model fitted to them, deciding by the model)."""
        from . import model

        repetitions = self.repetitions
        measured = [self.baseline, *(f for f in finalists if f.number != self.baseline.number)]
        share = max(1, self._concurrency() // len(measured))  # measured together, sharing the concurrency

        async def measure() -> list[Any]:
            runs = [
                self._measure(
                    Values(c.params),
                    self.validation,
                    repetitions=repetitions,
                    max_concurrency=share,
                    phase="validation",
                )
                for c in measured
            ]
            return list(await asyncio.gather(*runs))

        validated: list[Candidate] = []
        for c, out in zip(measured, self._run(measure()), strict=True):
            trials = tuple(self.projection.prepare(t) for t in out)
            estimates = self.projection.estimates(trials)
            means = {m: e.mean for m, e in estimates.items()}
            validated_c = dataclasses.replace(
                c,
                trials=trials,
                estimates=estimates,
                eligible=self.projection.eligible(trials),
                violations=self.projection.violations(means, validated[0].metrics if validated else None),
            )
            validated.append(validated_c)
            if self.record is not None:
                self.record.validated(self, validated_c)
            if self.log:
                self.log(f"{'validated':<13}candidate {c.number}: " + "n=" + self._line(validated_c).split(" n=", 1)[1])
        self.validated = validated
        fit = None
        if self.selection.by == "model":
            self._model_problem = None  # the search model's problem, if any, is not the validation model's
            try:
                fit = model.fit(
                    self.projection,
                    self.params,
                    self._text_blocks(),
                    validated,
                    validated[0],
                    [c.name for c in self.validation.cases],
                )
            except Exception as exc:  # noqa: BLE001 - recorded with the selection; measured means decide
                self._model_problem = f"the validation model could not be fitted ({type(exc).__name__}: {exc})"
            if fit is None and self._model_problem is None:
                self._model_problem = "the validation model could not be fitted (fewer than 2 eligible candidates)"
        views = [decided_view(c, self.projection, self.selection, fit, validated[0]) for c in validated]
        decision = self.schedule.decide(self, views, views[0], "validation")
        if decision is not None:  # anchored (phases): the views judged against A*, the pick, the other rules
            best, views = decision.best, decision.views
            self._pick, self._alternatives = decision.pick, decision.alternatives
            self._log_decision(decision)
        else:
            best = [v.candidate for v in self.combine.best(self.projection, views, views[0])]
            if not best:
                best = self._closest(views, views[0], on="validation")
            elif len(self.projection.objectives) == 1:  # one objective selects one candidate, also on a tie
                best = [self._among_equals(best[0], views)]
        self._selected = [self.candidates[c.number] for c in best]
        if self.record is not None:
            self.record.selected(
                self,
                self._selected,
                on="validation",
                views={v.number: v for v in views},
                unconfirmed=self._unconfirmed,
                decision=decision,
            )

    def _log_decision(self, decision: Any) -> None:
        if not self.log:
            return
        pick = decision.pick.number if decision.pick is not None else "none (the baseline ships)"
        guarded = [
            f"{v.number} (loses {v.anchored.lost} of {v.anchored.robust})"
            for v in decision.views
            if v.anchored is not None and v.anchored.guard is False
        ]
        self.log(
            f"{'selected':<13}candidate {pick} against A* = candidate {self.anchor.number if self.anchor else '-'}"
            + (f"; the seesaw guard dropped {', '.join(guarded)}" if guarded else "")
            + (
                "; the other rules: "
                + ", ".join(f"{r} {n or 'the baseline'}" for r, n in decision.alternatives.items())
                if decision.alternatives
                else ""
            )
        )

    def _closest(self, views: Sequence[View], baseline: View, on: str) -> list[Candidate]:
        """No candidate in ``views`` meets the constraints: with ``Selection(unmet="closest")``, the one that comes
        closest to meeting them (``selection.closest``; one candidate for one objective: of those still tied, the one
        with the most trials, the first on a tie), kept in ``_unconfirmed``; with ``unmet="none"``, nothing."""
        found = closest(self.projection, self.combine, views, baseline) if self.selection.unmet == "closest" else None
        if found is None or not found.chosen:
            return []
        if len(self.projection.objectives) == 1 and len(found.chosen) > 1:
            found = dataclasses.replace(found, chosen=[max(found.chosen, key=lambda v: len(v.trials))])
        self._unconfirmed = found
        if self.log:
            chosen = {v.number for v in found.chosen}
            ranked = sorted(
                (n for n in found.closeness if n not in chosen),
                key=lambda n: (-(found.closeness[n]["p_met"] or 0.0), found.closeness[n]["excess"] or 0.0),
            )
            others = "; ".join(f"candidate {n}: {self._closeness(found.closeness[n])}" for n in ranked[:4])
            more = f" and {len(ranked) - 4} more" if len(ranked) > 4 else ""
            for v in found.chosen:
                self.log(
                    f"{'selected':<13}candidate {v.number}, UNCONFIRMED: none of the {len(found.closeness)} met the "
                    f"constraints on {on}; it is the closest ({self._closeness(found.closeness[v.number])})"
                    + (f"; next {others}{more}" if others else "")
                )
        return [v.candidate for v in found.chosen]

    @staticmethod
    def _closeness(c: dict[str, float | None]) -> str:
        p, excess = c.get("p_met"), c.get("excess") or 0.0
        return (f"P(met) {p:.2f}, " if p is not None else "") + f"misses by {excess:.3g}"

    def _remeasure(self, candidate: Candidate, repetitions: int) -> None:
        """Measure a candidate ``repetitions`` more times on every case; its trials and estimates grow in place."""
        run = self._measure(
            Values(candidate.params),
            self.experiment,
            repetitions=repetitions,
            max_concurrency=self._concurrency(),
            phase="remeasurement",
        )
        offset = max((t.repetition for t in candidate.trials), default=0)
        new = tuple(
            dataclasses.replace(self.projection.prepare(t), repetition=t.repetition + offset) for t in self._run(run)
        )
        trials = (*candidate.trials, *new)
        estimates = self.projection.estimates(trials)
        means = {m: e.mean for m, e in estimates.items()}
        updated = dataclasses.replace(
            candidate,
            trials=trials,
            estimates=estimates,
            eligible=self.projection.eligible(trials),
            violations=self.projection.violations(means, None if candidate.number == 0 else self.baseline.metrics),
        )
        self.candidates[candidate.number] = updated
        if self.record is not None:
            self.record.remeasured(self, updated, new)
        if self.log:
            self.log(f"{'remeasured':<13}candidate {candidate.number}: {repetitions} more repetitions")

    def _among_equals(self, best: Candidate, views: Sequence[View]) -> Candidate:
        """Among the feasible candidates whose decided objective equals ``best``'s (at the model's resolution,
        ``selection.resolved``), the one with the most trials (``best`` itself on a tie, the first of the pool: on
        validation, the baseline): an exact tie, never an overlap of intervals, and never another metric."""
        metric = self.projection.objectives[0].metric
        b = next((v for v in views if v.number == best.number), None)
        if b is None or metric not in b.decided:
            return best
        tied = [v for v in views if v.feasible and metric in v.decided and v.metrics[metric] == b.metrics[metric]]
        return max(tied, key=lambda v: (len(v.trials), v.number == best.number)).candidate

    def _out_of_patience(self) -> bool:
        return self._patience is not None and self._unchanged >= self._patience

    def _effects(self) -> None:
        """Write each choice's effects (``effects.json``): from the study's model when it decides by one, else from
        an effects model fitted to the record; a study without them still finishes."""
        import datetime as dt
        import json

        from . import effects, model

        try:
            if self.by_model:
                now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                data = model.effects_json(self.fit, self.projection, now)
                (Path(self.record.directory) / "effects.json").write_text(json.dumps(data, indent=1))
                return
            effects.write(self.record.directory)
        except Exception as exc:  # noqa: BLE001 - analysis is never a reason to fail a study
            if self.log:
                self.log(f"effects: not written ({type(exc).__name__}: {exc})")

    @staticmethod
    def _first_error(candidate: Candidate) -> str:
        return next((a.error for a in candidate.trials if a.error), "no trial had a value for every goal metric")

    @property
    def _independent(self) -> bool:
        """Whether the method copies serve the goal's objectives one each (Independent)."""
        return self.combine.instances(len(self.combine.objectives)) > 1

    def _anchor(self, anchor: Anchor, *, phase: dict[str, Any], steps: list[dict[str, Any]]) -> None:
        """A schedule set the anchor later decisions are judged against (phases: phase A's result, A*): record it,
        with how the phase before it ended and what it spent, and say so."""
        self.anchor = anchor
        a = self.candidates[anchor.number]
        if self.record is not None:
            self.record.anchor(self, anchor, phase=phase, steps=steps)
        if self.log:
            means = "  ".join(f"{m} {a.metrics.get(m, float('nan')):.3g}" for m in self.projection.goal_metrics)
            within = anchor.margin if anchor.margin is not None else "its limits"
            self.log(
                f"{'phase':<13}primary ended ({phase.get('reason')}, {phase.get('steps')} steps, "
                f"{phase.get('candidates')} candidates): A* = candidate {anchor.number} ({means}); "
                f"the cost phase keeps {anchor.metric} within {within}"
            )

    def _evidence(self, k: int) -> None:
        """Copy ``k``'s methods start listening: they are told every eligible candidate measured so far, as evidence
        (judged as it is now, against the anchor), then each result as it comes."""
        self._listening[k] = True
        for candidate in self.candidates:
            if candidate.eligible:
                for name in self.blocks:
                    self.methods[k][name].observe(self.blocks[name], candidate, None, self, k)

    def turn(self) -> int:
        """The objective whose methods propose next (they take turns with Independent)."""
        k = self._turn % len(self.methods)
        self._turn += 1
        return k

    def method(self, block: str, objective: int | None = None) -> Any:
        return self.methods[objective or 0][block]

    def _open(self, scope: Scope) -> int:
        self._scopes += 1
        if self.record is not None:
            self.record.scope(scope, self._scopes)
        return self._scopes

    def _close(self, scope: Scope) -> None:
        if self.record is not None:
            self.record.scope_end(scope)

    def _proposing(self, method: Any) -> Any:
        """The lock a method proposes under: none for a ``concurrent`` method, the study's for any other."""
        return contextlib.nullcontext() if getattr(method, "concurrent", False) else self._lock

    def _propose(
        self,
        update: Sequence[str],
        base: Candidate | None,
        objective: int | None,
    ) -> tuple[Values, dict[str, Draw], _Source] | None:
        """The candidate's values: the ``update`` blocks that exist under the structure being built proposed by their
        methods, structural blocks first (outermost first, each given the structure so far), everything else held."""
        held = dict(base.params) if base else {}
        draws: dict[str, Draw] = {}
        values = self.agent.suggest(_Source(self, held, draws, objective)).values
        choices = lambda: {k: v for k, v in values.items() if self.params[k].type in SELECTORS}  # noqa: E731
        for name in self.structural_order():
            if name in update and self.active(name, values):
                method = self.method(name, objective)
                with self._proposing(method):
                    draws[name] = method.begin(self.blocks[name], base, self.candidates, choices(), self, objective)
                self._ledger.call(name, getattr(draws[name], "origin", None))
                values = self.agent.suggest(_Source(self, held, draws, objective)).values
        structure = choices()
        for name in update:
            if not self.blocks[name].structural and self.active(name, values):
                method = self.method(name, objective)
                with self._proposing(method):
                    draws[name] = method.begin(self.blocks[name], base, self.candidates, structure, self, objective)
                self._ledger.call(name, getattr(draws[name], "origin", None))
        if update and not draws:
            return None
        source = _Source(self, held, draws, objective)
        proposed = self.agent.suggest(source)
        for key in source.drawn:  # every value is recorded as JSON: a method that proposes another kind fails here
            problem = json_problem(proposed.values[key])
            if problem:
                block = self.block_of[key]
                raise ValueError(
                    f"{key}={proposed.values[key]!r}, proposed by block {block!r} "
                    f"({type(self.blocks[block].method).__name__}), cannot be recorded: {problem}; {JSON_HINT}"
                )
        return proposed, draws, source

    def _proposal(
        self,
        update: Sequence[str],
        base: Candidate | None,
        *,
        objective: int | None,
        scope: Scope | None,
        role: str,
        step: int | None,
        defer: Sequence[str],
    ) -> tuple[str, Any, dict[str, Draw], Any] | None:
        """One new proposal: ("new", values, draws, source); a round's opening proposal that repeats a measured
        candidate opens on it: ("open", that candidate, draws, None); None when there is nothing new to propose."""
        for _ in range(self.search.redraws + 1):
            proposal = self._propose(update, base, objective)
            if proposal is None:
                return None
            values, draws, source = proposal
            earlier = self._measured(values.to_dict())
            if earlier is None:
                return "new", values, draws, source
            opens = role == "outer"
            with self._lock:
                self._repeat(
                    earlier,
                    draws,
                    source,
                    scope=scope,
                    role=role,
                    step=step,
                    objective=objective,
                    keep=defer if opens else (),
                )
            if opens:
                return "open", earlier, draws, None
        return None  # every proposal repeated a measured candidate

    def _step(
        self,
        update: Sequence[str],
        base: Candidate | None,
        *,
        objective: int | None,
        scope: Scope | None,
        role: str,
        step: int | None,
        defer: Sequence[str],
        width: int,
        cost: int | None = None,
    ) -> list[tuple[Candidate, dict[str, Draw]]]:
        """One step: ``width`` proposals from ``base`` (made at once, in threads, when a method of the step proposes
        concurrently; in slot order otherwise), measured together; then the model
        is refitted once and each candidate's result goes to the methods of the scope's blocks. Two proposals of the
        step with the same values are measured once, and the second's methods are told the first's result."""
        if self._folds is not None and self._folds.failed.is_set():
            raise _Stopped("another run of the rotation failed or was interrupted")
        unknown = [name for name in update if name not in self.blocks]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}; the blocks are {sorted(self.blocks)}")

        def one(_: int) -> tuple[str, Any, dict[str, Draw], Any] | None:
            return self._proposal(update, base, objective=objective, scope=scope, role=role, step=step, defer=defer)

        if width == 1 or not any(getattr(self.method(n, objective), "concurrent", False) for n in update):
            proposals = [one(k) for k in range(width)]  # in slot order: the methods propose one at a time anyway
        else:
            with ThreadPoolExecutor(max_workers=width, thread_name_prefix="propose") as pool:
                proposals = list(pool.map(one, range(width)))
        opened = next((p for p in proposals if p is not None and p[0] == "open"), None)
        spent = width if cost is None else cost  # the budget this step uses, in candidates (Scope.used)
        if opened is not None:  # a round opens on the earlier candidate: not measured again, nor counted outside it
            assert scope is not None
            scope.candidates.append(opened[1])
            scope.steps += 1
            scope.used += spent
            return [(opened[1], opened[2])]
        batch: list[tuple[Any, dict[str, Draw], Any]] = []
        twins: list[tuple[int, dict[str, Draw], Any]] = []
        for p in proposals:
            if p is None:
                continue
            _, values, draws, source = p
            same = next((i for i, (v, _, _) in enumerate(batch) if v.to_dict() == values.to_dict()), None)
            if same is None:
                batch.append((values, draws, source))
            else:
                twins.append((same, draws, source))
        if not batch:
            return []
        first = len(self.candidates)
        problems = [self._validate(values) for values, _, _ in batch]
        runnable = [i for i, problem in enumerate(problems) if not problem]
        share = max(1, self._concurrency() // max(1, len(runnable)))  # the step's candidates share the concurrency

        async def measure() -> list[Any]:
            runs = [
                self._measure(
                    batch[i][0], self.experiment, repetitions=self.repetitions, max_concurrency=share, phase="search"
                )
                for i in runnable
            ]
            return list(await asyncio.gather(*runs))

        measured = dict(zip(runnable, self._run(measure()) if runnable else [], strict=True))
        candidates: list[Candidate] = []
        for i, (values, draws, source) in enumerate(batch):
            common = dict(
                number=first + i,
                params=values.to_dict(),
                proposed=tuple(draws),
                base=base.number if base else None,
                objective=objective if self._independent else None,
                aim=objective if len(self.methods) > 1 else None,
                phase=scope.phase if scope else None,
                origin={name: draw.origin for name, draw in draws.items()},
                drawn={name: tuple(k for k in source.drawn if self.block_of[k] == name) for name in draws},
                inherited=dict(source.inherited),
                defaulted=tuple(source.defaulted),
                scope=scope.id if scope else None,
                path=scope.path if scope else (),
                role=role,
                step=step,
                slot=i,
                _factory=self.agent,
                _artifacts=self._artifact_root(),
            )
            if problems[i]:
                candidate = Candidate(
                    status="invalid",
                    trials=(),
                    estimates={},
                    eligible=False,
                    violations=(None,) * len(self.projection.constraints),
                    reason=problems[i],
                    **common,
                )
            else:
                trials = tuple(self.projection.prepare(t) for t in measured[i])
                estimates = self.projection.estimates(trials)
                eligible = self.projection.eligible(trials)
                scored = [a for a in trials if a.status in SCORED]
                status = (
                    "failed" if not scored else ("complete" if eligible and len(scored) == len(trials) else "partial")
                )
                violations = self.projection.violations(
                    {m: e.mean for m, e in estimates.items()}, self.baseline.metrics if self.candidates else None
                )
                candidate = Candidate(
                    status=status,
                    trials=trials,
                    estimates=estimates,
                    eligible=eligible,
                    violations=violations,
                    **common,
                )
            candidates.append(candidate)
        before = self._standing() if self.candidates else None
        for candidate in candidates:
            self.candidates.append(candidate)
            at = scope
            while at is not None:
                at.candidates.append(candidate)
                at = at.parent
        at = scope
        while at is not None:
            at.steps += 1
            at.used += spent
            at.history.append((base.number if base else None, tuple(c.number for c in candidates)))
            at = at.parent
        self._refit()
        self._unchanged = self._unchanged + 1 if before == self._standing() else 0
        shared = [b for b in (scope.blocks if scope else self.blocks) if b not in defer]
        for candidate, (_, draws, _) in zip(candidates, batch, strict=True):
            self._share(candidate, draws, shared)
            if self.record is not None:
                self.record.candidate(self, candidate, shared=shared, deferred=[b for b in defer if b in draws])
            if self.log:
                self.log(self._line(candidate))
        for i, draws, source in twins:
            self._repeat(candidates[i], draws, source, scope=scope, role=role, step=step, objective=objective, keep=())
        return [(c, d) for c, (_, d, _) in zip(candidates, batch, strict=True)]

    def _measured(self, values: dict[str, Any]) -> Candidate | None:
        """The candidate that already measured these values, if one did (a candidate that measured nothing usable can
        be measured again)."""
        return next((c for c in self.candidates if c.eligible and c.params == values), None)

    def _repeat(
        self,
        earlier: Candidate,
        draws: dict[str, Draw],
        source: "_Source",
        *,
        scope: Scope | None,
        role: str,
        step: int | None,
        objective: int | None,
        keep: Sequence[str],
    ) -> None:
        """A proposal repeated ``earlier``: each of its draws (but those in ``keep``, told later) learns that its values
        scored what ``earlier`` measured."""
        k = objective or 0
        told = [name for name in draws if name not in keep]
        for name in told:
            self.methods[k][name].observe(self.blocks[name], earlier, draws[name], self, k)
        if self.record is not None:
            drawn = {name: tuple(key for key in source.drawn if self.block_of[key] == name) for name in draws}
            self.record.repeat(
                self, earlier, draws, drawn, scope=scope, role=role, step=step, objective=objective, told=told
            )
        if self.log:
            self.log(f"{'repeat':<13}{'/'.join(draws):<18} = candidate {earlier.number}: not measured again")

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run ``coro`` to its end on the study's event loop (``_loop``): every measurement the study makes runs
        here."""
        return self._loop().run_until_complete(coro)

    async def _measure(
        self, values: Values, experiment: Any, *, repetitions: int, max_concurrency: int, phase: str
    ) -> list[Trial]:
        """Measure ``values`` on every case of ``experiment``, ``repetitions`` times, at most ``max_concurrency`` trials
        at once, with the study's task and retries: the one place the study runs trials (search, re-measurement,
        validation and test alike), awaited through ``_run``. What the trials spent counts in ``phase``. The factory
        receives the values with each stored one (a skill-library version) resolved to its path."""
        if self._skills:
            values = self.agent.resolve(values, self.artifacts)
        trials = await run_trials(
            self.agent,
            values,
            experiment,
            repetitions=repetitions,
            max_concurrency=max_concurrency,
            task=self.task,
            retries=self.retries,
        )
        self._ledger.trials(phase, trials)
        return trials

    def _loop(self) -> asyncio.AbstractEventLoop:
        """One event loop for the whole study: libraries keep asyncio objects between candidates (Harbor's per-image
        build locks), and those are bound to the loop that made them."""
        if getattr(self, "_event_loop", None) is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()
        return self._event_loop

    def _validate(self, candidate: Candidate) -> str | None:
        """Why the candidate cannot be measured, if it cannot: it is outside the space, or the agent cannot be built."""
        root = self._artifact_root()
        try:
            self.agent.validate(candidate, artifacts=root)
            self.agent(**self.agent.bind(candidate, artifacts=root))
        except Exception as exc:  # noqa: BLE001 - any failure to build makes the candidate invalid
            return f"{type(exc).__name__}: {exc}"
        return None

    def _share(self, candidate: Candidate, draws: dict[str, Draw], blocks: Sequence[str]) -> None:
        """Give a candidate's result to the methods of ``blocks``: to the copy that proposed it as the outcome of its
        own draw, to every other copy as evidence."""
        for k, methods in enumerate(self.methods):
            if not self._listening[k]:  # told everything at once when it starts listening (_evidence)
                continue
            mine = (candidate.aim or 0) == k
            for name in blocks:
                methods[name].observe(self.blocks[name], candidate, draws.get(name) if mine else None, self, k)

    def _tell(
        self, block: str, draw: Draw, number: int, candidate: Candidate | None, *, objective: int | None, scope: Scope
    ) -> None:
        """Tell the copy that drew ``draw`` (in candidate ``number``) the result that stands for it, and give that
        result to the block's other copies as evidence."""
        k = objective or 0
        for i, methods in enumerate(self.methods):
            if i == k:
                methods[block].observe(self.blocks[block], candidate, draw, self, i)
            elif candidate is not None and self._listening[i]:
                methods[block].observe(self.blocks[block], candidate, None, self, i)
        if self.record is not None:
            self.record.told(self, scope, block, number, candidate, k)

    # ---- results ----------------------------------------------------------------------------------------------------
    def incumbent(self, among: Sequence[Candidate] | None = None, objective: int | None = None) -> Candidate:
        """The candidate other blocks are held at, chosen by ``combine`` from ``among`` (default: every candidate)."""
        pool = list(among) if among is not None else self.candidates
        held = self._held(pool, objective)
        return held if held is not None else self.baseline

    @property
    def best_candidates(self) -> list[Candidate]:
        """The best feasible candidates as ``combine`` defines best (the Pareto set, or the single best), as decided:
        the selection the last run ended with, else the best so far. A selection made when none met the constraints is
        the closest, unconfirmed (``Selection.unmet``; ``unconfirmed``)."""
        if self._selected is not None:
            return [self.candidates[c.number] for c in self._selected]
        return self._decided_best()

    @property
    def unconfirmed(self) -> Closest | None:
        """Whether the last run's selection is unconfirmed: none of the candidates it selected from met the
        constraints, so it shipped the closest (``Selection(unmet="closest")``); how it was chosen, else None."""
        return self._unconfirmed

    @property
    def best_candidate(self) -> Candidate:
        best = self.best_candidates
        if self.combine.best_is_set(len(self.projection.objectives)):
            raise ValueError("with several objectives there is no single best candidate: see best_candidates")
        if not best:
            raise ValueError("no candidate meets the constraints")
        return best[0]

    def report(self) -> str:
        """One line per candidate: what changed, each goal metric's mean and interval, status."""
        return "\n".join(self._line(t) for t in self.candidates)

    def _line(self, t: Candidate) -> str:
        parts = []
        for m in self.projection.goal_metrics:
            e = t.estimates.get(m)
            if e and e.mean is not None:
                ci = f" [{e.ci[0]:.3g}, {e.ci[1]:.3g}]" if e.ci else ""
                parts.append(f"{m} {e.mean:.3g}{ci}")
        n = len({a.case for a in t.trials})
        return (
            f"candidate {t.number:2} {'/'.join(t.proposed) or t.role:18} {t.status:8} n={n} "
            + "  ".join(parts)
            + ("  (infeasible)" if t.feasible is False else "")
            + (f"  {t.reason}" if t.reason else "")
        )


class _Source:
    """What the factory's search space asks for while a candidate is built: values of the proposed blocks from their
    methods' draws; every other value as held; a value the held candidate lacks (it exists only under a choice the held
    candidate did not make) from the best candidate that has it, else its default."""

    def __init__(self, study: Study, held: dict[str, Any], draws: dict[str, Draw], objective: int | None) -> None:
        self.study, self.held, self.draws, self.objective = study, held, draws, objective
        self.drawn: list[str] = []
        self.inherited: dict[str, int] = {}
        self.defaulted: list[str] = []

    def _value(self, name: str, ask: Callable[[Draw], Any]) -> Any:
        block = self.study.block_of[name]
        if block in self.draws:
            self.drawn.append(name)
            return ask(self.draws[block])
        if name in self.held:
            return self.held[name]
        having = [t for t in self.study.candidates if name in t.params]
        if having:
            best = self.study.incumbent(having, self.objective)
            if name in best.params:
                self.inherited[name] = best.number
                return best.params[name]
        self.defaulted.append(name)
        return self.study.params[name].default

    def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        return self._value(name, lambda d: d.suggest_float(name, *args, **kwargs))

    def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
        return self._value(name, lambda d: d.suggest_int(name, *args, **kwargs))

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        return self._value(name, lambda d: d.suggest_categorical(name, choices))

    def suggest_text(self, name: str, default: str) -> str:
        return self._value(name, lambda d: d.suggest_text(name, default))

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        return self._value(name, lambda d: d.suggest_skills(name, default))


__all__ = ["AtLeast", "AtMost", "Maximize", "Minimize", "Scope", "Study", "Candidate", "constraint_name"]


class _Folds:
    """A rotation's runs searching side by side: how many still search (they divide ``max_concurrency``), and whether
    one failed or the rotation was interrupted (the others then stop at their next step)."""

    def __init__(self, runs: int) -> None:
        self.running, self.failed, self._lock = runs, threading.Event(), threading.Lock()

    def finish(self) -> None:
        with self._lock:
            self.running = max(1, self.running - 1)


class _Stopped(RuntimeError):
    """A rotation's run stopped because another failed, or the rotation was interrupted."""
