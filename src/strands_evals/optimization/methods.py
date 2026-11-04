"""Blocks of variables and the methods that update them.

A block is a method and the variables it proposes. The variables are the objects declared in ``@tunable``:

    window_size = Int(4, 40, default=6)
    manager = OneOf(component(SlidingWindowConversationManager, window_size=window_size), ...)

    @tunable(conversation_manager=manager, system_prompt=system_prompt, tools=tool_descriptions(...))
    def create_agent(...): ...

    Block(TPE(), manager)                       # the choice of manager
    Block(TPE(), window_size)                   # that one setting
    Block(Reflection(), system_prompt, tools)   # a component (here, tools) stands for every variable declared in it

A value (``Int``, ``TextValue``, ...) and a choice (``OneOf``, ``Switch``) are each one variable; a component stands for
the variables declared in it, unless another block names one of them itself. Every variable no block names goes to a
default block for its type: ``structure`` (the choices; TPE with a startup of ``STRUCTURE_STARTUP``, as it is asked
once a round) and ``numbers`` (``Float``, ``Int``, ``Choice``) by TPE, ``text`` (``TextValue``) by contrastive
reflection, ``skills`` (``SkillLibrary``) by skill-library curation. A block that holds a choice is *structural*: it
is proposed before the blocks whose variables exist only under it.

A method proposes values for its block given the values held for everything else (``begin`` returns a draw the
candidate is built from) and learns from results (``observe``): the outcome of its own proposal, or evidence from
candidates it did not propose. A method that calls a model itself reports that call's usage in its draw's
``origin["usage"]`` (and ``origin["seconds"]``), also when the call fails: the study counts it as the method's own cost
(``cost.py``), and names the model in its record (``Method.model_id``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .trials import TrialStatus

if TYPE_CHECKING:
    from .study import Candidate, Study

SELECTORS = ("one_of", "switch")  # variables that choose which other variables exist


class Draw:
    """One proposal in progress: the values a method gives while the candidate is built."""

    origin: dict[str, Any]

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        raise TypeError(f"{name}: this method does not propose numbers")

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        raise TypeError(f"{name}: this method does not propose numbers")

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        raise TypeError(f"{name}: this method does not propose choices")

    def suggest_text(self, name: str, default: str) -> str:
        raise TypeError(f"{name}: this method does not write text")

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        raise TypeError(f"{name}: this method does not curate skills")


class Method(ABC):
    """Proposes values for a block (``begin``) and learns from results (``observe``).

    What a method writes into its draws' ``origin`` is recorded as it is, so every method keeps the record's rules
    (``record.py``):

    - ``origin["usage"]`` counts only the method's own model calls (``origin["seconds"]``, their time), also when a
      call fails: never the trials it runs, which are the agent's;
    - trials it runs to probe a proposal before it is measured are their own records, never candidates;
    - a trial over its deadline is an agent failure, never an infrastructure one;
    - a program's parameters are keyed per version (``<key>.<version>.<parameter>``).
    """

    kinds: tuple[str, ...] = ()  # the variable types this method can propose
    # Whether a step's proposals (``Search.width`` of them) may be made in parallel: a method whose ``begin`` keeps no
    # state between proposals, such as reflection, which starts a fresh optimizer each time. Others propose one at a
    # time, under the study's lock.
    concurrent = False

    @abstractmethod
    def begin(
        self,
        block: "Block",
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        """Start a proposal for ``block``. ``base`` is the candidate whose values are held for every other block, and
        ``structure`` the choices this candidate has so far (structural blocks are proposed first, outermost first,
        so a structural block sees the choices of the blocks around it). ``objective`` is the objective this copy of
        the method serves (with ``Independent``)."""

    def observe(  # noqa: B027 - optional: a method that learns nothing from results need not implement it
        self, block: "Block", candidate: "Candidate | None", draw: Draw | None, study: "Study", objective: int
    ) -> None:
        """Learn from a candidate: the outcome of ``draw`` when this method proposed it, else evidence. A ``candidate``
        of ``None`` with a draw means the draw failed (nothing could be told for it)."""

    def copy(self, k: int) -> "Method":
        """A fresh copy of this method for the k-th further objective (with ``Independent``)."""
        import copy

        return copy.deepcopy(self)

    def describe(self) -> dict[str, Any]:
        """The method's kind and settings, for the record."""
        return {"kind": type(self).__name__}

    @property
    def model_id(self) -> str | None:
        """The model this method calls itself, if it calls one (the record's header names it for the method's role)."""
        return None


def _agentic_model(options: dict[str, Any]) -> str | None:
    """The model a reflection or curation optimizer calls: its ``model_config``'s, else the optimizer's default."""
    from ._reflection.agentic import AgenticOptimizer

    return {**AgenticOptimizer._DEFAULT_MODEL_CONFIG, **(options.get("model_config") or {})}.get("model_id")


class Block:
    """A method and the variables it proposes: ``Block(TPE(), window_size, pin_first)``. ``name`` labels the block in
    the record and the explorer (default: the kind of variable it holds)."""

    def __init__(self, method: Method, *variables: Any, name: str | None = None) -> None:
        if not isinstance(method, Method):
            raise TypeError("Block(method, *variables): the method comes first, as in Block(TPE(), window_size)")
        if not variables:
            raise ValueError("a block needs at least one variable: the objects declared in @tunable")
        self.method, self.variables, self.name = method, tuple(variables), name
        self.keys: tuple[str, ...] = ()  # the variables' keys, once the study has resolved them
        self.structural = False  # whether it holds a choice, once resolved

    @classmethod
    def resolved(cls, name: str, method: Method, keys: Sequence[str], structural: bool) -> "Block":
        """A block as the study runs it: named, with the keys of its variables."""
        block = cls.__new__(cls)
        block.method, block.variables, block.name = method, (), name
        block.keys, block.structural = tuple(keys), structural
        return block

    def __repr__(self) -> str:
        return f"Block({self.name!r}, {type(self.method).__name__}, keys={list(self.keys)})"


def context(block: Block, params: dict[str, Any], study: "Study") -> tuple:
    """Which landscape a block's values belong to: a structural block has one; any other block one per combination of
    the choices it is built under. Proposals and evidence are both filed by this."""
    if block.structural:
        return ()
    return tuple(sorted((k, v) for k, v in params.items() if study.params[k].type in SELECTORS and k not in block.keys))


# ---- Optuna ---------------------------------------------------------------------------------------------------------
CONSTRAINTS = "constraints:"  # the prefix of the system attributes Optuna keeps a trial's constraint values in


def _kept(attrs: dict[str, Any]) -> dict[str, Any]:
    """What a trial's sampler wrote for it, kept when its landscape is rebuilt; not its constraints, which Optuna keeps
    among the same attributes, and which would then never be judged again."""
    return {k: v for k, v in attrs.items() if not k.startswith(CONSTRAINTS)}


class _Entry:
    """One trial of a landscape, in the order it was told: the block's values (as proposed, or as a candidate it did not
    propose holds them) and the candidate whose result stands for them, so that its value can be told again as the
    study's decisions change. A draw not yet told is pending (``number`` None); one that could not be told failed."""

    __slots__ = ("params", "distributions", "number", "failed", "draw", "standing", "attrs")

    def __init__(
        self,
        params: dict[str, Any],
        distributions: dict[str, Any],
        number: int | None = None,
        draw: "_OptunaDraw | None" = None,
    ) -> None:
        self.params, self.distributions, self.number, self.draw = params, distributions, number, draw
        self.failed = False
        # how to find the candidate that stands for a draw told at a round's end, as decided now (Scope.tell)
        self.standing: Callable[[], "Candidate | None"] | None = None
        self.attrs: dict[str, Any] = {}  # the system attributes its sampler wrote for the trial, kept across rebuilds

    @property
    def pending(self) -> bool:
        return self.number is None and not self.failed


class _Landscape:
    """One landscape of an Optuna method (``context``): its sampler, kept for the whole study, and every trial it was
    told (``_Entry``). Its Optuna study is rebuilt from them before each proposal (``build``), each trial told the
    current decided value of the candidate that stands for it, so TPE never compares values decided by different fits
    (the model is refitted after every step, and decides with measured means once its diagnostics flag a metric)."""

    def __init__(self, sampler: Any, objective: int | None, study: "Study") -> None:
        self.sampler, self.objective, self.study = sampler, objective, study
        self.entries: list[_Entry] = []

    def candidate(self, entry: _Entry) -> "Candidate | None":
        if entry.standing is not None:
            return entry.standing()
        return self.study.candidates[entry.number] if entry.number is not None else None

    def build(self) -> Any:
        """A fresh Optuna study holding every trial, in order: each told trial with the current decided value of its
        candidate (and its constraints as judged now), each failed one failed, and each pending one running, so that
        Optuna's constant liar still keeps a step's proposals apart."""
        import optuna
        from optuna.trial import TrialState, create_trial

        study, k = self.study, self.objective
        aim = study.aim(k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            built = optuna.create_study(directions=aim.directions(study.projection, k), sampler=self.sampler)
        built.stop = lambda: None  # the study's budget ends the search, not the sampler (grid, brute force)
        baseline = study.view(study.baseline)
        for e in self.entries:
            if e.pending:  # as its draw's own study holds it now: what it has proposed so far, its sampler's attributes
                assert e.draw is not None
                asked = e.draw.study.get_trials(deepcopy=False)[e.draw.ask.number]
                built.add_trial(
                    create_trial(
                        state=TrialState.RUNNING,
                        params=dict(asked.params),
                        distributions=dict(asked.distributions),
                        system_attrs=_kept(asked.system_attrs),
                    )
                )
                continue
            c = None if e.failed else self.candidate(e)
            if c is None or not c.eligible:
                built.add_trial(
                    create_trial(
                        state=TrialState.FAIL,
                        params=e.params,
                        distributions=e.distributions,
                        system_attrs=dict(e.attrs),
                    )
                )
                continue
            decided = study.view(c)
            built.add_trial(
                create_trial(
                    params=e.params,
                    distributions=e.distributions,
                    values=aim.told(study.projection, decided, k, baseline),
                    constraints=aim.told_constraints(study.projection, decided) or None,
                    system_attrs=dict(e.attrs),  # a copy: create_trial writes the constraints into the dict it is given
                )
            )
        return built


class _OptunaDraw(Draw):
    def __init__(self, ask: Any, study: Any, origin: dict[str, Any], entry: _Entry | None = None) -> None:
        self.ask, self.study, self.origin = ask, study, origin  # the Optuna trial this draw asked for
        self.entry = entry  # its trial in the landscape
        self.standing: Callable[[], "Candidate | None"] | None = None  # set when told at a round's end

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        return self.ask.suggest_float(name, low, high, step=step, log=log)

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        return self.ask.suggest_int(name, low, high, step=step, log=log)

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        return self.ask.suggest_categorical(name, choices)


class Optuna(Method):
    """An Optuna sampler over the block's numbers and choices: ``Optuna(lambda: optuna.samplers.GPSampler())``.

    It keeps one landscape per ``context`` (for a non-structural block, each combination of the choices outside it),
    and every candidate in that landscape is evidence for it, whoever proposed it. A landscape's Optuna study is rebuilt
    before each proposal from every trial it was told, each with the current decided value of the candidate that stands
    for it (``_Landscape``): what TPE splits into good and bad is what the study decides now, as its incumbent and its
    selection are, never a value from an earlier fit. Its sampler is kept across rebuilds, and the attributes the
    sampler wrote for each trial with it."""

    kinds = ("float", "int", "choice", "one_of", "switch")

    def __init__(self, sampler: Callable[[], Any] | None = None) -> None:
        self.sampler = sampler
        self._landscapes: dict[tuple, _Landscape] = {}

    @property
    def _studies(self) -> dict[tuple, Any]:
        """Each landscape's Optuna study as it would be built now (for inspection)."""
        return {given: land.build() for given, land in self._landscapes.items()}

    def _landscape(self, context: tuple, study: "Study", objective: int | None) -> _Landscape:
        import optuna

        if context not in self._landscapes:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
                sampler = self._sampler(len(self._landscapes))
            self._landscapes[context] = _Landscape(sampler, objective, study)
        return self._landscapes[context]

    def _sampler(self, landscape: int) -> Any:
        """The sampler for the ``landscape``-th landscape this method opens."""
        return self.sampler() if self.sampler else None

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        given = context(block, structure, study)
        land = self._landscape(given, study, objective)
        optuna_study = land.build()  # every trial told its current decided value, the pending ones running
        ask = optuna_study.ask()
        entry = _Entry({}, {})
        draw = _OptunaDraw(
            ask, optuna_study, {"method": type(self).__name__, "optuna_trial": ask.number, "given": dict(given)}, entry
        )
        entry.draw = draw
        land.entries.append(entry)
        return draw

    def observe(
        self, block: Block, candidate: "Candidate | None", draw: Draw | None, study: "Study", objective: int
    ) -> None:
        """Told what ``combine`` tells this objective's methods, with the constraint violations; a candidate that cannot
        be told (none, invalid, or some case lacks a goal metric) fails its own draw and is no evidence. What is told
        is told again, as decided then, before each later proposal (``_Landscape.build``)."""
        import optuna

        aim = study.aim(objective)  # what this copy searches for: the goal's combination, or a phase's
        if isinstance(draw, _OptunaDraw):
            entry = draw.entry
            if candidate is None or not candidate.eligible:
                draw.study.tell(draw.ask, state=optuna.trial.TrialState.FAIL)
            else:
                decided, baseline = study.view(candidate), study.view(study.baseline)  # as the study decides now
                for name, violation in aim.told_constraints(study.projection, decided).items():
                    draw.ask.set_constraint(name, violation)
                draw.study.tell(draw.ask, aim.told(study.projection, decided, objective, baseline))
            if entry is not None:  # the trial as its landscape holds it from now on
                told = draw.study.get_trials(deepcopy=False)[draw.ask.number]
                entry.params, entry.distributions = dict(told.params), dict(told.distributions)
                entry.attrs = _kept(told.system_attrs)
                entry.draw = None
                if candidate is None or not candidate.eligible:
                    entry.failed = True
                else:
                    entry.number, entry.standing = candidate.number, draw.standing
            return
        if candidate is None or not candidate.eligible:
            return
        distributions = {k: d for k, d in study.agent.distributions(candidate.params).items() if k in block.keys}
        if not distributions:
            return
        self._landscape(context(block, candidate.params, study), study, objective).entries.append(
            _Entry({k: candidate.params[k] for k in distributions}, distributions, candidate.number)
        )


class TPE(Optuna):
    """Optuna's tree-structured Parzen estimator: ``TPE(seed=7, n_startup_trials=4)``.

    ``n_startup_trials`` is how many candidates a landscape holds (proposed or shared as evidence) before TPE's model is
    used; until then proposals are random. Optuna's default of 10 would make a short study random search, so the
    default here is 4. A step's proposals are made before any of them is told (``Search.width``): Optuna's constant liar
    keeps them apart (a landscape rebuilt for the next proposal holds them as running). Every trial is told the current
    decided value of its candidate before each proposal (``Optuna``)."""

    def __init__(self, seed: int | None = None, n_startup_trials: int = 4, **options: Any) -> None:
        self.seed, self.n_startup_trials, self.options = seed, n_startup_trials, options
        super().__init__()

    def _sampler(self, landscape: int) -> Any:
        import optuna

        seed = None if self.seed is None else self.seed + landscape  # each landscape draws its own startup points
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=self.n_startup_trials,
            # constant_liar: a step's proposals made before any is told keep apart (each pending one counts as bad)
            **{"multivariate": True, "group": True, "constant_liar": True, **self.options},
        )

    def copy(self, k: int) -> "TPE":
        return TPE(
            seed=None if self.seed is None else self.seed + 1000 * k,
            n_startup_trials=self.n_startup_trials,
            **self.options,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "seed": self.seed,
            "n_startup_trials": self.n_startup_trials,
            **self.options,
        }


# ---- Thompson sampling on the study's model --------------------------------------------------------------------------
class _ChoiceDraw(Draw):
    def __init__(self, values: dict[str, Any], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        v = self.values.get(name)
        return v if v in list(choices) else list(choices)[0]


def _drawn(fit: Any, draws: dict[str, Any], metric: str, params: dict[str, Any]) -> float:
    """A metric's value at a configuration under one drawn set of its effects."""
    return float(fit.metrics[metric].drawn(draws[metric][None, :], params)[0])


class Thompson(Method):
    """Proposes a block of choices (``Choice``, ``OneOf``, ``Switch``) by Thompson sampling from the study's model of
    every goal metric (``model.py``): it draws the metrics' effects from their posteriors, predicts every combination
    of the block's options with the rest held, and proposes the best under the draw (among those that meet the drawn
    constraints, else the one that misses them least; several objectives are weighted at random for each draw, each in
    its model's unit). An option no candidate has measured is predicted by its prior, so it is tried in proportion to
    how likely it is to be best. A combination already measured (with the rest as held) is not proposed again.
    Until the study has a model, it proposes an option not yet measured.

    ``Thompson(seed=7)``; the model is the study's, so evidence needs no telling."""

    kinds = ("choice", "one_of", "switch")
    MAX_COMBINATIONS = 512

    def __init__(self, seed: int | None = None) -> None:
        import numpy as np

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def copy(self, k: int) -> "Thompson":
        return Thompson(seed=None if self.seed is None else self.seed + 1000 * k)

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "seed": self.seed}

    def _combinations(self, block: Block, study: "Study") -> list[dict[str, Any]]:
        import itertools

        keys = [k for k in block.keys if study.params[k].type in self.kinds]
        options = [list(study.params[k].choices) for k in keys]
        combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*options)]
        if len(combos) > self.MAX_COMBINATIONS:
            picks = self._rng.choice(len(combos), size=self.MAX_COMBINATIONS, replace=False)
            combos = [combos[i] for i in sorted(picks)]
        return combos

    def _template(
        self, combo: dict[str, Any], base: "Candidate | None", history: Sequence["Candidate"], study: "Study"
    ) -> dict[str, Any]:
        """The values a combination would be predicted at: the best measured candidate that made these choices, else
        the held values with these choices (variables under other options left out)."""
        made = [c for c in history if c.eligible and all(c.params.get(k) == v for k, v in combo.items())]
        if made:
            best = study._held(made, None)
            if best is not None:
                return dict(best.params)
        params = {**(base.params if base else {}), **combo}
        return {
            k: v
            for k, v in params.items()
            if not (
                study.params[k].active_when
                and study.params[k].active_when[0] in combo
                and combo[study.params[k].active_when[0]] != study.params[k].active_when[1]
            )
        }

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        import numpy as np

        from .model import p_met  # noqa: F401 - the model is the study's (optional extra: bayesbench)

        combos = self._combinations(block, study)
        keys = [k for k in block.keys if study.params[k].type in self.kinds]
        held = {k: v for k, v in (base.params if base else {}).items() if k not in block.keys}
        # a combination already measured with the rest as held would only repeat that candidate: never proposed
        measured = {
            tuple(c.params.get(k) for k in keys)
            for c in history
            if c.eligible and all(c.params.get(k) == v for k, v in held.items() if k in c.params)
        }
        fit = study.fit if study.by_model else None
        if fit is None:
            fresh = [c for c in combos if tuple(c[k] for k in keys) not in measured] or combos
            pick = fresh[int(self._rng.integers(len(fresh)))]
            return _ChoiceDraw(pick, {"method": "Thompson", "by": "unmeasured option", "given": dict(structure)})
        templates = [self._template(c, base, history, study) for c in combos]
        draws = {m: model.draws(1, self._rng)[0] for m, model in fit.metrics.items()}
        p = study.projection
        objectives = [p.objectives[i] for i in study.aim(objective).focus(p, objective)]
        weights = self._rng.dirichlet(np.ones(len(objectives))) if len(objectives) > 1 else np.ones(1)
        scores, misses = [], []
        for params in templates:
            score = 0.0
            for w, o in zip(weights, objectives, strict=True):
                sign = 1.0 if o.__class__.__name__ == "Maximize" else -1.0
                score += w * sign * _drawn(fit, draws, o.metric, params) / fit.metrics[o.metric].unit
            miss = 0.0
            for c in p.constraints:
                v = _drawn(fit, draws, c.metric, params)
                if c.compare == "difference":
                    v -= _drawn(fit, draws, c.metric, study.baseline.params)
                elif c.compare == "ratio":
                    ref = _drawn(fit, draws, c.metric, study.baseline.params)
                    v = v / ref if ref else math.inf
                over = (v - c.limit) if c.__class__.__name__ == "AtMost" else (c.limit - v)
                miss += max(over, 0.0) / fit.metrics[c.metric].unit
            scores.append(score)
            misses.append(miss)
        fresh = [i for i, c in enumerate(combos) if tuple(c[k] for k in keys) not in measured]
        candidates = fresh or list(range(len(combos)))
        feasible = [i for i in candidates if misses[i] == 0.0]
        pool = feasible or [min(candidates, key=lambda j: misses[j])]
        i = max(pool, key=lambda j: scores[j])
        return _ChoiceDraw(
            combos[i],
            {"method": "Thompson", "by": "posterior draw", "given": dict(structure), "options": len(combos)},
        )


# ---- contrastive reflection -----------------------------------------------------------------------------------------
class _TextValues:
    """Text values in the shape the reflection optimizer reads and writes."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = dict(values)

    def get_tunable_params(self) -> dict:
        return dict(self.values)

    def update_params(self, params: dict) -> None:
        self.values.update({k: v for k, v in params.items() if k in self.values})


class _TextDraw(Draw):
    def __init__(self, values: dict[str, str], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_text(self, name: str, default: str) -> str:
        return self.values.get(name, default)


class _TaskMessage:
    """The reflection task message: the template, given what each variable is and how it may change, and the
    evidence beyond the traces (the versions measured so far: see ``_versions``)."""

    def __init__(self, template: Any, variables: list[dict[str, Any]], evidence: dict[str, Any] | None = None) -> None:
        self.template, self.variables, self.evidence = template, variables, evidence or {}

    def render(self, **values: Any) -> str:
        return self.template.render(**values, variables=self.variables, **self.evidence)


def _number(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def _versions(study: "Study", keys: list[str], base: "Candidate", objective: int | None) -> list[dict[str, Any]]:
    """Every candidate measured so far, best first as the study ranks them for its goal (with ``Independent``, for
    ``objective``), then those it cannot rank (a goal metric missing): each with its values of ``keys`` (the texts),
    its other values, what it changed from the candidate it was built from, its goal metrics' means over its trials,
    and whether it meets the goal's limits. The texts a reflection rewrites are judged here, as whole candidates."""
    measured = [c for c in study.candidates if c.status != "invalid"]
    aim = study.aim(objective)
    views, ranked = study._views([c for c in measured if c.eligible]), []
    by_view = {v.number: v for v in views}
    baseline = study.view(study.baseline)
    while views:
        held = aim.held(study.projection, views, objective, baseline)
        if held is None:
            break
        ranked.append(held.candidate)
        views = [v for v in views if v.number != held.number]
    shown = aim.shown(study.projection)
    numbers = {c.number for c in ranked}
    order = ranked + [c for c in measured if c.number not in numbers]
    by_number = {c.number: c for c in study.candidates}
    out = []
    for i, c in enumerate(order):
        parent = by_number.get(c.base) if c.base is not None else None
        changed = [k for k in keys if parent is not None and c.params.get(k) != parent.params.get(k)]
        if parent is not None and any(c.params.get(k) != parent.params.get(k) for k in set(c.params) - set(keys)):
            changed.append("other values")
        out.append(
            {
                "rank": i + 1 if c.number in numbers else None,
                "number": c.number,
                "from": c.base,
                "current": c.number == base.number,
                "baseline": c.number == study.baseline.number,
                "changed": changed,
                "metrics": {m: c.metrics.get(m) for m in shown},
                "cells": [_number(c.metrics.get(m)) for m in shown],
                "meets_limits": _meets(aim, study, c, by_view),
                "conversations": sum(1 for t in c.trials if t.status == TrialStatus.COMPLETED),
                "texts": {k: c.params.get(k) for k in keys},
                "others": {k: v for k, v in c.params.items() if k not in keys},
            }
        )
    return out


def _meets(aim: Any, study: "Study", c: "Candidate", views: dict[int, Any]) -> bool | None:
    """Whether a version meets the limits, as the aim judges them (by its measured means when it has no view: it is
    not eligible)."""
    if c.number in views:
        return aim.meets(study.projection, views[c.number])
    return None if not aim.limits(study.projection) else c.feasible


VERSION_FILE = "version.md"  # a version's results and other values, in its folder; its texts are files beside it
# A version's header, as ``_write_versions`` writes it: a line ``# Version N`` and the line below it.
_VERSION_HEADER = re.compile(r"^# Version \d+(?: \(the baseline\))?[ \t]*\r?\n(?:rank|built from): ", re.M)


def _text_files(names: Sequence[str]) -> dict[str, str]:
    """Each text's file name in a version's folder: its name, with anything but letters, digits, ``.``, ``_`` and
    ``-`` made ``_``, then ``.txt`` (numbered when two names would share one)."""
    out: dict[str, str] = {}
    for name in names:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or "text"
        file, i = f"{stem}.txt", 2
        while file in out.values() or file == VERSION_FILE:
            file, i = f"{stem}-{i}.txt", i + 1
        out[name] = file
    return out


def _write_versions(folder: str, versions: list[dict[str, Any]], goal_metrics: list[str]) -> None:
    """One folder per version, named by its rank (``rank-001-version-26``): ``version.md`` holds its results and its
    other values, and each of its texts is a file of its own (``_text_files``) that holds the text and nothing else.

    A rewriter that wants to start from a version's text copies that text's file, so what it edits and submits is
    the text alone. With one file per version, its header and results above its texts, a rewriter that copied the
    file into its working folder to edit the text below them submitted the header too, and the header became part of
    the text it measured. A submission that still carries the header is refused (``_no_version_header``)."""
    for v in versions:
        rank = f"rank-{v['rank']:03d}" if v["rank"] is not None else "unranked"
        path = os.path.join(folder, f"{rank}-version-{v['number']}")
        os.makedirs(path)
        files = _text_files(list(v["texts"]))
        lines = [f"# Version {v['number']}" + (" (the baseline)" if v["baseline"] else "")]
        lines.append(f"rank: {v['rank'] if v['rank'] is not None else 'not ranked (a result is missing)'}")
        lines.append(f"built from: version {v['from']}" if v["from"] is not None else "built from: nothing (the start)")
        lines += [f"{m}: {cell}" for m, cell in zip(goal_metrics, v["cells"], strict=True)]
        if v["meets_limits"] is not None:
            lines.append(f"meets the limits: {'yes' if v['meets_limits'] else 'no'}")
        lines.append(f"conversations: {v['conversations']}")
        if v["others"]:
            lines += ["", "## Other values", json.dumps(v["others"], indent=1, default=str)]
        lines += ["", "## Its texts, each in its own file in this folder (the text alone)"]
        for k, text in v["texts"].items():
            value = "" if text is None else str(text)
            lines.append(f"- {k}: `{files[k]}` ({len(value):,} characters)")
            with open(os.path.join(path, files[k]), "w") as f:
                f.write(value)
        with open(os.path.join(path, VERSION_FILE), "w") as f:
            f.write("\n".join(lines) + "\n")


def _no_version_header(name: str, text: str) -> str | None:
    """Why a submitted text is refused, or None: it carries a version's header from the versions folder
    (``_write_versions``), which is the record of a version and never part of a text."""
    found = _VERSION_HEADER.search(text)
    if found is None:
        return None
    first = found.group(0).splitlines()[0].strip()
    return (
        f"it holds a version's header from the versions folder (`{first}` and the results below it), which is not "
        f"part of the text. Write the text alone to the file (a version's text is alone in its `.txt` file, never "
        f"in {VERSION_FILE}) and submit it again"
    )


def _feed(optimizer: Any, rewards: list, study: "Study") -> None:
    """Give a reflection optimizer the held candidate's trials: messages, rewards and their reasons."""
    from ._reflection import Reward, Rollout

    cases = {case.name: case for case in study.experiment.cases}
    for trial, reward, metadata in rewards:
        case = cases[trial.case]
        optimizer.add_rollouts(
            [
                Rollout(
                    data_sample={
                        "name": trial.case,
                        "repetition": trial.repetition,
                        "input": case.input,
                        "expected_output": case.expected_output,
                    },
                    messages=list(trial.messages),
                    metadata={"output": trial.output},
                )
            ]
        )
        optimizer.add_rewards([Reward(reward=reward, metadata=metadata)])


class Reflection(Method):
    """Reflection over the block's text variables (on harness-optimizer's ``ContrastiveReflectionOptimizer``): a model
    writes a new version of the texts from two kinds of evidence:

    - the versions: every candidate measured so far, ranked best first for the study's goal, each with its texts and
      its goal metrics' means over all its trials (the table in its task message; one folder per version, each text
      in full in a file of its own: ``_write_versions``): what kinds of text have done well. A metric with no range,
      such as tokens, is judged here only;
    - the held candidate's trials: their messages, the evaluators' reasons, and each trial's reward from the goal's
      bounded metrics (task success: see ``Projection.rewards``), contrasting those that went well with those that
      did not: where its agent goes wrong.

    It writes each text it changes in full, and may add, rewrite, reorder or remove anything. Each variable says what
    it is (``TextValue(about=...)``) and its limit (``max_chars``); a value over its limit is sent back to be shortened,
    in the same conversation, up to ``over_limit_retries`` times (2; 0: never), and one still over is dropped and
    the held one kept. A submission whose text carries a version's header is refused (``_no_version_header``) and the
    agent told why; a turn that ends on such a refusal is sent back the same way. Options go to the optimizer
    (``model_config``, ``n_sample_traces``, templates, ...).

    The optimizer gives its model a shell over the traces: run studies with text blocks where model-written shell
    commands are acceptable (a VM or a container), with ``BYPASS_TOOL_CONSENT=true``."""

    kinds = ("text",)
    concurrent = True  # each proposal starts its own optimizer and folders

    def __init__(self, over_limit_retries: int = 2, **options: Any) -> None:
        if over_limit_retries < 0:
            raise ValueError("Reflection(over_limit_retries=...) must be 0 or more")
        self.over_limit_retries, self.options = over_limit_retries, options

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "over_limit_retries": self.over_limit_retries,
            **{k: v for k, v in self.options.items() if k != "model_config"},
            **({"model": str(self.options["model_config"])} if "model_config" in self.options else {}),
        }

    @property
    def model_id(self) -> str | None:
        return _agentic_model(self.options)

    def optimizer(
        self, values: "_TextValues", variables: list[dict[str, Any]], evidence: dict[str, Any] | None = None
    ) -> Any:
        """The reflection optimizer for one proposal (a seam for tests and for other reflection optimizers):
        ``evidence`` goes to the task message (the versions, their folder, the goal's metrics and limits, and what a
        trial's reward is)."""
        from ._reflection import ContrastiveReflection, create_template, load_template

        task = self.options.get("task_message_template", load_template("reflection/task_message.jinja"))
        template = create_template(task) if isinstance(task, str) else task
        options = {
            "system_prompt_template": load_template("reflection/system_prompt.jinja"),
            **(
                {"readable_folders": [evidence["versions_folder"]]}
                if evidence and "versions_folder" in evidence
                else {}
            ),
            # a phase that lowers cost: the digest gives every case's cost, those that succeeded too
            **({"digest_costs": evidence["digest_costs"]} if evidence and evidence.get("digest_costs") else {}),
            **self.options,
            "task_message_template": _TaskMessage(template, variables, evidence),
            "limits": {v["name"]: v["max_chars"] for v in variables if v.get("max_chars") is not None},
            "over_limit_retries": self.over_limit_retries,
            "check": _no_version_header,  # a text with a version's header is refused at submission
        }
        return ContrastiveReflection(values, **options)

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _TextDraw({}, {"method": name, "from_candidate": None})
        held = {k: base.params[k] for k in keys}
        aim = study.aim(objective)  # what this proposal searches for: the goal's, or its phase's
        rewards = study.projection.rewards(base, aim, objective, metrics=aim.per_trial(study.projection))
        if not rewards:
            return _TextDraw(held, {"method": name, "from_candidate": base.number, "status": "nothing to learn from"})
        variables = [{"name": k, "about": study.params[k].about, "max_chars": study.params[k].max_chars} for k in keys]
        versions = _versions(study, keys, base, objective)
        label, ranked = study.projection.label_metrics()
        shown = aim.shown(study.projection)
        with tempfile.TemporaryDirectory(prefix="versions_") as folder:
            _write_versions(folder, versions, shown)
            evidence = {
                "versions": versions,
                "versions_folder": folder,
                "goal_metrics": shown,
                "limits": aim.limits(study.projection),
                "label_metrics": label,
                "label_ranked": ranked,
                "judged_by_version": [m for m in shown if m not in label],
                **({"digest_costs": aim.digest_costs(study.projection)} if aim.digest_costs(study.projection) else {}),
            }
            text = _TextValues(held)
            optimizer = self.optimizer(text, variables, evidence)
            _feed(optimizer, rewards, study)
            origin: dict[str, Any] = {
                "method": name,
                "from_candidate": base.number,
                "trials": len(rewards),
                "versions": len(versions),
            }
            try:
                optimizer.step()
            except Exception as exc:  # noqa: BLE001 - a failed reflection keeps the held text; the study goes on
                rejected = getattr(optimizer, "rejected", None)
                return _TextDraw(
                    held,
                    {
                        **origin,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "usage": usage(getattr(optimizer, "last_metrics", None)),  # what it spent before it failed
                        "seconds": getattr(optimizer, "last_wall_clock_s", None),
                        **({"rejected": rejected[:10]} if rejected else {}),
                    },
                )
        values, dropped = dict(text.values), {}
        for k in keys:
            limit = study.params[k].max_chars
            if limit is not None and len(values[k]) > limit:
                dropped[k] = f"{len(values[k])} characters, over max_chars={limit}"
                values[k] = held[k]
        changed = [k for k in keys if values[k] != held[k]]
        origin.update(
            status="rewritten" if changed else "unchanged",
            changed=changed,
            usage=usage(getattr(optimizer, "last_metrics", None)),
            seconds=getattr(optimizer, "last_wall_clock_s", None),
        )
        digest = getattr(optimizer, "digest_text", None)
        if isinstance(digest, str):  # what the rewriter read of the trials: kept with the study, named in the record
            origin["digest"] = _store_digest(digest, study)
        revised = getattr(optimizer, "revisions", None)
        if revised:  # submissions sent back to be shortened: the lengths that were over their limits
            origin["revised"] = revised
        if dropped:
            origin["dropped"] = dropped
        rejected = getattr(optimizer, "rejected", None)
        if rejected:  # submissions refused for a version's header in a text: which text, and why
            origin["rejected"] = rejected[:10]
        refused = getattr(optimizer, "refused", None)
        if refused:  # commands that named paths outside its folders (see the optimizer's confined shell)
            origin["refused"] = refused[:10]
        return _TextDraw(values, origin)


def _store_digest(text: str, study: "Study") -> dict[str, Any]:
    """Store a digest the rewriter read in the study's artifact store, once by content (``digests/<hash>.md``), and say
    where and how big it is (with how many cases it lists as not succeeding)."""
    name = f"digests/{hashlib.sha256(text.encode()).hexdigest()[:16]}.md"
    path = study.artifacts / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    header = next((line for line in text.splitlines() if line.startswith("## Every case")), "")
    failing = header.rsplit("(", 1)[-1].rstrip(")") if "(" in header else None
    return {"file": name, "chars": len(text), "failing_cases": int(failing) if failing and failing.isdigit() else None}


class _SkillsDraw(Draw):
    def __init__(self, values: dict[str, str | None], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        return self.values.get(name, default)


class SkillCuration(Method):
    """Skill-library curation (harness-optimizer's ``SkillLibraryOptimizer``, ported unchanged): a model reads the held
    candidate's trials and the deployed skills, and decides which skills to create, revise or retire; or that the
    library should stay as it is, which is a legitimate outcome. Skills whose frontmatter cannot be loaded are rejected.
    Each new library is stored once, by content, in the study's artifact store, and its value is its path there
    (``skills/<hash>``); each curation writes its decisions in a folder of its own. Options go to the optimizer.

    The optimizer gives its model a shell over the traces: run it where model-written shell commands are acceptable,
    with ``BYPASS_TOOL_CONSENT=true``."""

    kinds = ("skills",)

    def __init__(self, **options: Any) -> None:
        self.options = options

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            **{k: v for k, v in self.options.items() if k != "model_config"},
            **({"model": str(self.options["model_config"])} if "model_config" in self.options else {}),
        }

    @property
    def model_id(self) -> str | None:
        return _agentic_model(self.options)

    def optimizer(self, skills: Any, output_folder: str) -> Any:
        """The curation optimizer for one proposal (a seam for tests)."""
        from ._reflection import SkillLibraryCuration, load_template

        options = {
            "system_prompt_template": load_template("skill_library/system_prompt.jinja"),
            "task_message_template": load_template("skill_library/task_message.jinja"),
            **self.options,
        }
        return SkillLibraryCuration(skills, output_folder=output_folder, **options)

    def begin(
        self,
        block: Block,
        base: "Candidate | None",
        history: Sequence["Candidate"],
        structure: dict[str, Any],
        study: "Study",
        objective: int | None,
    ) -> Draw:
        from ._reflection import SkillSet
        from .space.skills import store

        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _SkillsDraw({}, {"method": name, "from_candidate": None})
        held = {k: base.params[k] for k in keys}
        aim = study.aim(objective)
        rewards = study.projection.rewards(base, aim, objective, metrics=aim.per_trial(study.projection))
        if not rewards:
            return _SkillsDraw(held, {"method": name, "from_candidate": base.number, "status": "nothing to learn from"})
        values, origin = dict(held), {"method": name, "from_candidate": base.number, "trials": len(rewards)}
        spent: dict[str, float] = {}  # every key's curation, each its own optimizer
        seconds = 0.0
        for key in keys:
            stored = study.artifact_path(held[key])
            library = SkillSet(skill_dir=str(stored) if stored else None, strict=False)
            folder = _fresh(study.artifacts / "skill-decisions", f"candidate-{len(study.candidates):03d}-{key}")
            optimizer = self.optimizer(library, str(folder))
            _feed(optimizer, rewards, study)
            try:
                optimizer.step()
            except Exception as exc:  # noqa: BLE001 - a failed curation keeps the held library; the study goes on
                _sum(spent, usage(getattr(optimizer, "last_metrics", None)))  # what it spent before it failed
                return _SkillsDraw(
                    held,
                    {
                        **origin,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "usage": spent,
                        "seconds": seconds + (getattr(optimizer, "last_wall_clock_s", None) or 0.0),
                    },
                )
            _sum(spent, usage(getattr(optimizer, "last_metrics", None)))
            seconds += getattr(optimizer, "last_wall_clock_s", None) or 0.0
            decisions = getattr(optimizer, "_last_decisions", {}) or {}
            if any(decisions.values()):
                values[key] = store(library.members, study.artifacts)
            names = {
                k: [os.path.basename(os.path.normpath(x)) if isinstance(x, str) else x for x in v]
                for k, v in decisions.items()
            }
            origin[key] = {"decisions": names, "folder": folder.relative_to(study.artifacts).as_posix()}
        changed = [k for k in keys if values[k] != held[k]]
        origin.update(status="rewritten" if changed else "unchanged", changed=changed, usage=spent, seconds=seconds)
        return _SkillsDraw(values, origin)


def usage(metrics: dict[str, Any] | None) -> dict[str, float]:
    """A method's own model usage in the trials' metric names (``tokens`` = input + output + cache), with its model
    calls."""
    m = metrics or {}
    out = {
        name: float(m.get(name, 0) or 0)
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    }
    out["tokens"] = sum(out.values())
    out["model_calls"] = float(m.get("cycle_count", 0) or 0)
    return out


def _fresh(root: Any, name: str) -> Any:
    """A new folder under ``root``: ``name``, or ``name-2``, ``name-3``, ... when an earlier curation of the same step
    wrote that one (a step's proposals, and its redraws, each curate from an empty folder)."""
    folder, n = root / name, 2
    while folder.exists():
        folder, n = root / f"{name}-{n}", n + 1
    folder.mkdir(parents=True)
    return folder


def _sum(into: dict[str, float], usage: dict[str, float]) -> None:
    for k, v in usage.items():
        into[k] = into.get(k, 0.0) + v


# ---- resolving the blocks -------------------------------------------------------------------------------------------
KINDS = {"structure": SELECTORS, "numbers": ("float", "int", "choice"), "text": ("text",), "skills": ("skills",)}
# The default structure block's TPE startup: a structural block is asked once a round (Nested), and a run has few rounds
# (3 at width 4, 12 iterations, Fixed(4)). TPE's startup of 4 counts the baseline (shared as evidence) and every round
# told, so every round's structure was a random draw (and most candidates of a full space ran under structures never
# selected). 2: the baseline and one random round, then TPE's model chooses each round from the rounds' results.
# A structure block given in Search(blocks=...) keeps its own TPE(n_startup_trials=...).
STRUCTURE_STARTUP = 2
DEFAULT_METHODS: dict[str, Callable[[], Method]] = {"text": Reflection, "skills": SkillCuration}  # any other: TPE


def register_kind(kind: str, types: Sequence[str], method: Callable[[], Method]) -> None:
    """Add a kind of variable defined outside the package: its default block is named ``kind``, holds the variables of
    ``types`` (a plug-in domain's ``ParamInfo.type``) no block names, and is proposed by a fresh ``method()``."""
    KINDS[kind] = tuple(types)
    DEFAULT_METHODS[kind] = method


def _kind(types: set[str]) -> str | None:
    return next((name for name, kinds in KINDS.items() if types <= set(kinds)), None)


def resolve_blocks(
    params: dict[str, Any], given: Sequence[Block], references: Sequence[tuple[Any, tuple[str, ...], bool]]
) -> tuple[list[Block], dict[int, str]]:
    """Every variable in exactly one block: the given block that names it (naming a variable itself outranks naming a
    component it is declared in), else a default block for its type. Returns the blocks and each given block's name."""
    found: dict[int, list[tuple[tuple[str, ...], bool]]] = {}
    for obj, keys, direct in references:
        found.setdefault(id(obj), []).append((keys, direct))
    claims: dict[str, tuple[int, bool]] = {}
    for i, block in enumerate(given):
        for obj in block.variables:
            if id(obj) not in found:
                raise ValueError(
                    f"block {i + 1} holds {obj!r}, which the agent's @tunable does not declare: declare the variable "
                    "once, as a Python object, and give that same object to both"
                )
            for keys, direct in found[id(obj)]:
                for key in keys:
                    held = claims.get(key)
                    if held is None or (direct and not held[1]):
                        claims[key] = (i, direct)
                    elif held[0] != i and held[1] == direct:
                        raise ValueError(f"{key} is in blocks {held[0] + 1} and {i + 1}; give each variable one block")
    groups: list[tuple[str | None, Method, list[str]]] = [(b.name, b.method, []) for b in given]
    defaults: dict[str, list[str]] = {}
    for key, info in params.items():
        if key in claims:
            groups[claims[key][0]][2].append(key)
        else:
            defaults.setdefault(next(n for n, kinds in KINDS.items() if info.type in kinds), []).append(key)
    for i, (_, _, keys) in enumerate(groups):
        if not keys:
            raise ValueError(f"block {i + 1} holds no variable of its own (another block names all of them)")
    taken = {name for name, _, _ in groups if name}
    out, names = [], {}

    def unique(base: str) -> str:
        name, n = base, 2
        while name in taken:
            name, n = f"{base} {n}", n + 1
        taken.add(name)
        return name

    for i, (name, method, keys) in enumerate(groups):
        types = {params[k].type for k in keys}
        name = name or unique(_kind(types) or "block")
        names[id(given[i])] = name
        out.append((name, method, keys))
    for kind, keys in defaults.items():
        method = TPE(n_startup_trials=STRUCTURE_STARTUP) if kind == "structure" else DEFAULT_METHODS.get(kind, TPE)()
        out.append((unique(kind), method, keys))
    if len({name for name, _, _ in out}) != len(out):
        raise ValueError(f"two blocks have the same name: {[name for name, _, _ in out]}")
    blocks = []
    for name, method, keys in out:
        types = {params[k].type for k in keys}
        unsupported = types - set(method.kinds)
        if unsupported:
            raise ValueError(
                f"block {name!r} has {sorted(unsupported)} variables, which {type(method).__name__} "
                f"does not propose; put them in another block"
            )
        blocks.append(Block.resolved(name, method, keys, bool(types & set(SELECTORS))))
    return blocks, names
