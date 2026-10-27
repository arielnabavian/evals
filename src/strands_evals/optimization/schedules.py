"""Schedules: which blocks change in each iteration, and which candidate the other blocks are held at.

- ``Joint()``: every block in scope proposes in every iteration.
- ``Alternate(numbers, texts, iterations=1)``: one block at a time, in turn (block-coordinate updates), each for
  ``iterations`` iterations.
- ``Nested(outer=choose_manager, inner=Joint(), inner_iterations=4)``: rounds, the one construct for multilevel
  search. Each round proposes the outer block, then runs ``inner`` for ``inner_iterations`` more with the outer values
  fixed; the outer method is told the round's best result, so outer choices are compared at their inner best. ``inner``
  may itself be ``Nested``, to any depth. A round whose outer proposal repeats a measured candidate opens on it; rounds
  stop once ``redraws + 1`` of them in a row measure nothing new. A round's opening proposal costs the budget around it
  one candidate (``opener="candidate"``), not a step, so rounds do not shrink the candidates a run measures.

  Its four rules are policies, each given as an object (the defaults are the rules above):

  - ``allocation``: how many inner steps a round gets (``Fixed(n)``; ``inner_iterations=n`` is its shorthand);
  - ``value``: what the outer method is told (``BestOfRound()``: the round's best candidate, as decided);
  - ``start``: where a round opens from (``FromIncumbent()``: the enclosing incumbent);
  - ``order``: in what order a round moves its levels (``OuterThenInner()``).

  Another policy (racing or halving the outer choices, values at a matched budget, warm starts, choosing the level
  that moves next) is a new object behind the same small interface (``Allocation``, ``RoundValue``, ``RoundStart``,
  ``RoundOrder``), and is recorded with the schedule (its ``describe``).

Blocks are given as the ``Block`` objects of the study's ``Search``. A schedule runs in a ``Scope`` (``study.py``): the
blocks it may change, the iterations (steps) it may still run, and the candidates it holds values from.
``scope.step(blocks)`` proposes ``Search.width`` candidates from the incumbent and measures them together (one
iteration); ``scope.candidate(blocks)`` a step of one, as a round's opening proposal is; ``scope.child(...)`` opens a
round. The study records
every scope and every candidate's place in them, so the schedule at any depth is in the record. A custom schedule
implements ``run(scope)`` (or just ``blocks(k, in_scope)``) and gets the same record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from .methods import Block
    from .study import Candidate, Scope, Study


class Schedule:
    def blocks(self, k: int, in_scope: Sequence[str]) -> list[str]:
        """The blocks (by name) to update in the k-th iteration of a run, given the blocks in scope."""
        raise NotImplementedError

    def run(self, scope: "Scope") -> None:
        """Run iterations in ``scope`` until its budget is spent (or nothing in it can change)."""
        k = 0
        while scope.budget > 0:
            if not scope.step(self.blocks(k, scope.blocks), step=k):
                return
            k += 1

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        """Raise if this schedule cannot run over ``blocks`` (called before anything runs)."""

    def describe(self, study: "Study") -> dict[str, Any]:
        """The schedule's kind and settings, for the record (its sub-schedules come from ``children``)."""
        return {"kind": type(self).__name__}

    def children(self) -> dict[str, "Schedule"]:
        """The schedules this one runs inside its rounds, by name."""
        return {}

    # What a schedule may decide besides which blocks move (``phases.PrimaryFirst`` does): None leaves it to the goal.
    def aims(self, study: "Study") -> list[Any] | None:
        """The combinations its parts search for, one method copy each (a scope's ``objective`` picks one), or None:
        the goal's own (one copy, or one per objective with ``Independent``)."""
        return None

    def finalists(self, study: "Study", n: int) -> "list[Candidate] | None":
        """The ``n`` candidates to measure again before selecting, or None: the goal's best."""
        return None

    def decide(self, study: "Study", views: list[Any], baseline: Any, on: str) -> Any:
        """The selection from ``views`` (``phases.Decision``), or None: the goal's own."""
        return None


class Joint(Schedule):
    def blocks(self, k: int, in_scope: Sequence[str]) -> list[str]:
        return list(in_scope)


class Alternate(Schedule):
    def __init__(self, *blocks: "Block", iterations: int = 1) -> None:
        """``blocks``: the blocks in turn (default: every block in scope); ``iterations``: how many each turn runs."""
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        self.order, self.iterations = list(blocks), iterations

    def describe(self, study: "Study") -> dict[str, Any]:
        return {"kind": "Alternate", "blocks": [study.name_of(b) for b in self.order], "iterations": self.iterations}

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        outside = [study.name_of(b) for b in self.order if study.name_of(b) not in blocks]
        if outside:
            raise ValueError(f"Alternate names blocks outside its scope: {outside} (the scope is {list(blocks)})")

    def run(self, scope: "Scope") -> None:
        """One block per iteration, in turn; a block with nothing to propose under the current structure is skipped.
        With ``Independent``, an objective's turn is one whole cycle of the blocks."""
        order = [scope.study.name_of(b) for b in self.order] or list(scope.blocks)
        cycle = len(order) * self.iterations
        k, misses, objective = 0, 0, None
        while scope.budget > 0 and misses < cycle:
            if k % cycle == 0:
                objective = scope.turn()
            block = order[(k // self.iterations) % len(order)]
            made = scope.step([block], step=k, objective=objective) if block in scope.blocks else []
            misses = 0 if made else misses + 1
            k += 1


# ---- Nested's policies -----------------------------------------------------------------------------------------------
class Policy:
    """One of ``Nested``'s rules, as an object: plain data (a rotation copies the search for each run), recorded with
    the schedule by ``describe``."""

    def describe(self) -> dict[str, Any]:
        """The policy's kind and settings, for the record."""
        return {"kind": type(self).__name__}


class Allocation(Policy):
    """How many inner steps a round gets. Asked as each round opens; the round's limit is its opening proposal and
    these steps, recorded in its ``scope`` event, and never changes while it runs (a policy that gives an outer choice
    more steps opens another round on it)."""

    def inner_steps(self, scope: "Scope", round: int) -> int:
        """The inner steps round ``round`` gets, in ``scope`` (the run or round the rounds run in)."""
        raise NotImplementedError


class Fixed(Allocation):
    """Every round gets ``steps`` inner steps (``Nested(inner_iterations=steps)`` is this)."""

    def __init__(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("inner_iterations (a Fixed allocation's steps) must be at least 0")
        self.steps = steps

    def inner_steps(self, scope: "Scope", round: int) -> int:
        return self.steps

    def describe(self) -> dict[str, Any]:
        return {"kind": "Fixed", "steps": self.steps}


class RoundValue(Policy):
    """What a round's outer draw is told when the round ends: the candidate whose result stands for it (the method is
    told it as every method is told a candidate: its decided values), or None when nothing in the round can (its draw
    failed)."""

    def told(self, round: "Scope") -> "Candidate | None":
        raise NotImplementedError


class BestOfRound(RoundValue):
    """The round's best candidate, as decided: outer choices are compared at their inner best."""

    def told(self, round: "Scope") -> "Candidate | None":
        return round.best()


class RoundStart(Policy):
    """Where a round opens from: the candidate its opening proposal holds every block but the outer one at."""

    def base(self, scope: "Scope", objective: int | None) -> "Candidate":
        """The candidate the round opens from, in ``scope`` (the run or round the rounds run in), for ``objective`` (the
        round's, with ``Independent``)."""
        raise NotImplementedError


class FromIncumbent(RoundStart):
    """The enclosing scope's incumbent."""

    def base(self, scope: "Scope", objective: int | None) -> "Candidate":
        return scope.incumbent(objective)


class RoundOrder(Policy):
    """In what order a round moves its levels. ``open()`` makes the round's opening proposal of the outer block (a step
    of one) and returns it, or None when the outer block has nothing to propose; ``inner()`` runs the inner schedule in
    the round, for the round's budget."""

    def run(self, round: "Scope", open: Callable[[], "Candidate | None"], inner: Callable[[], None]) -> bool:
        """Run one round; False when it could not open (the rounds end there)."""
        raise NotImplementedError


class OuterThenInner(RoundOrder):
    """The outer proposal, then, when it measured something, the inner schedule under it."""

    def run(self, round: "Scope", open: Callable[[], "Candidate | None"], inner: Callable[[], None]) -> bool:
        opener = open()
        if opener is None:
            return False
        if opener.eligible:
            inner()
        return True


OPENER_COSTS = ("candidate", "step")


class Nested(Schedule):
    def __init__(
        self,
        outer: "Block",
        inner: Schedule | None = None,
        inner_iterations: int | None = None,
        *,
        allocation: Allocation | None = None,
        value: RoundValue | None = None,
        start: RoundStart | None = None,
        order: RoundOrder | None = None,
        opener: str = "candidate",
    ) -> None:
        """``outer``: the block each round proposes first; ``inner``: what runs under it (default ``Joint()``), for
        ``inner_iterations`` iterations per round (default 4). The rules, as policies (module docstring):
        ``allocation`` (default ``Fixed(inner_iterations)``; give one or the other), ``value`` (``BestOfRound()``),
        ``start`` (``FromIncumbent()``) and ``order`` (``OuterThenInner()``).

        ``opener``: what a round's opening proposal (one candidate) costs the budget around the round. ``"candidate"``
        (default): one candidate's worth, 1 / ``Search.width`` of a step, so a run measures ``width`` x ``iterations``
        candidates whatever its rounds (its last step may be narrower), and an opener that repeats a measured candidate
        costs nothing. ``"step"``: a whole step, as before round 3b (at width 4 and 12 iterations of rounds of 1 + 4
        steps, a run measured 39 candidates, not 48)."""
        if inner_iterations is not None and allocation is not None:
            raise ValueError(
                "give Nested inner_iterations=n or allocation=..., not both (inner_iterations=n is Fixed(n))"
            )
        given = {"allocation": allocation, "value": value, "start": start, "order": order}
        kinds = {"allocation": Allocation, "value": RoundValue, "start": RoundStart, "order": RoundOrder}
        wrong = [f"{k}={v!r}" for k, v in given.items() if v is not None and not isinstance(v, kinds[k])]
        if wrong:
            raise TypeError(f"Nested: {', '.join(wrong)} is not a policy of its kind ({', '.join(kinds)})")
        if opener not in OPENER_COSTS:
            raise ValueError(f"Nested(opener=...) is one of {OPENER_COSTS}, not {opener!r}")
        self.outer, self.inner = outer, inner or Joint()
        self.allocation = allocation or Fixed(4 if inner_iterations is None else inner_iterations)
        self.value, self.start = value or BestOfRound(), start or FromIncumbent()
        self.order = order or OuterThenInner()
        self.opener = opener

    @property
    def inner_iterations(self) -> int | None:
        """The inner steps every round gets, when the allocation is ``Fixed`` (None for another allocation)."""
        return self.allocation.steps if isinstance(self.allocation, Fixed) else None

    def describe(self, study: "Study") -> dict[str, Any]:
        return {
            "kind": "Nested",
            "outer": study.name_of(self.outer),
            **({"inner_iterations": self.inner_iterations} if self.inner_iterations is not None else {}),
            "allocation": self.allocation.describe(),
            "value": self.value.describe(),
            "start": self.start.describe(),
            "order": self.order.describe(),
            "opener": self.opener,
        }

    def children(self) -> dict[str, Schedule]:
        return {"inner": self.inner}

    def check(self, blocks: Sequence[str], study: "Study") -> None:
        outer = study.name_of(self.outer)
        if outer not in blocks:
            raise ValueError(f"no block {outer!r} to search over; the blocks in scope are {list(blocks)}")
        inner = [b for b in blocks if b != outer]
        for key in study.blocks[outer].keys:
            for selector in study.ancestors(key):
                if study.block_of[selector] in inner:
                    raise ValueError(
                        f"Nested(outer={outer!r}): {key} exists only for some values of {selector}, which is in "
                        f"the inner block {study.block_of[selector]!r}; search {study.block_of[selector]!r} in an "
                        f"enclosing round instead"
                    )
        self.inner.check(inner, study)

    def run(self, scope: "Scope") -> None:
        outer = scope.study.name_of(self.outer)
        if outer not in scope.blocks:  # nothing of the outer block exists under this structure: no rounds here
            self.inner.run(scope)
            return
        inner = [b for b in scope.blocks if b != outer]
        width = scope.study.search.width
        r, empty = 0, 0
        while scope.budget > 0 and empty <= scope.study.search.redraws:
            measured = len(scope.study.candidates)
            objective = scope.turn()
            steps = self.allocation.inner_steps(scope, r)
            shared = self.opener == "candidate"  # the opener costs one candidate, not a step
            with scope.child(
                "inner",
                self.inner,
                blocks=inner,
                limit=1 + steps,
                round=r,
                outer=outer,
                objective=objective,
                capacity=1 + steps * width if shared else None,
                opener_slots=1 if shared else None,
            ) as round_:

                def open_(round_: "Scope" = round_, objective: int | None = objective) -> "Candidate | None":
                    base = self.start.base(scope, objective)
                    opener = round_.candidate([outer], role="outer", defer=[outer], base=base)
                    if opener is not None:
                        round_.narrow(opener)  # the inner blocks that exist under the structure just chosen
                    return opener

                def inner_(round_: "Scope" = round_) -> None:
                    self.inner.run(round_)

                if not self.order.run(round_, open_, inner_):
                    return
                round_.tell(outer, self.value.told(round_))  # None: nothing in the round stands for it, the draw failed
            # A round that opened on a measured candidate and measured nothing new under it: rounds that keep doing
            # so have nothing left to try.
            empty = empty + 1 if len(scope.study.candidates) == measured else 0
            r += 1


def default_schedule(study: "Study") -> Schedule:
    """Rounds over the structural blocks, outermost first (a block whose choices exist only under another's comes
    after it), around a search of everything else: ``Search.within``, by default ``Alternate()`` (one block a step, in
    turn); ``Joint()`` when there is nothing to nest."""
    structural = study.structural_order()
    others = [b for b in study.blocks if b not in structural]
    levels = structural if others else structural[:-1]
    within = study.search.within if study.search.within is not None else Alternate()
    schedule: Schedule = within if levels else Joint()
    for name in reversed(levels):
        schedule = Nested(outer=study.blocks[name], inner=schedule)
    return schedule
