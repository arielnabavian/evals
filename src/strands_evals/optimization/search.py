"""How a study searches: its blocks (each a method and the variables it proposes), the schedule that says when each
block moves, and how often a proposal that repeats a measured candidate is made again.

    choose_manager = Block(TPE(), manager)
    tune_manager = Block(TPE(), window_size, pin_first, summary_ratio, preserve_recent)
    rewrite_texts = Block(Reflection(), system_prompt, tools)

    Search(
        blocks=[choose_manager, tune_manager, rewrite_texts],
        schedule=Nested(outer=choose_manager, inner=Joint(), inner_iterations=4),
        width=4,
    )

Without blocks, every variable is in a default block for its type; without a schedule, the study builds rounds over
the structural blocks around a search of the rest (``default_schedule``), which moves one block at a time
(``within=Alternate()``, the default) or every block in every step (``within=Joint()``, as before round 3b).

**Width and depth.** A search runs in steps, one after another: each step's proposals are made from the candidates the
steps before it measured, so the number of steps (``optimize(iterations=...)``) is the search's depth, and its wall time
is about the number of steps times the time to measure one. ``width`` is how many candidates each step proposes from
the same incumbent and measures at once (sharing ``max_concurrency``) before the study's model is refitted and the
methods are told; it costs little more wall time than one candidate, because a step waits for its slowest trial either
way. In the τ² telecom records, a step of width 4 found about as much as 2 to 3 steps of width 1, and width 8 about 3
to 3.5: wider steps buy wall time with trials (about 1.5 to 2.5 times as many for the same gain), and gain less past
about 8. ``width=1`` proposes one candidate at a time.

**Phases (round 3d).** A goal with a cost objective (a primary objective, such as reward, and a cost, such as tokens:
``Pareto(Maximize(...), Minimize(...))``, ``CheapestOnFront(...)``) searches reward first, then cost
(``phases.PrimaryFirst``): phase A searches the primary objective alone, as a max-reward study does, until it stops
gaining; phase B lowers the cost from phase A's result under paired non-inferiority to it, in the same budget.
``phases="auto"`` (default) does so whenever the goal has both and the search gives no schedule of its own;
``phases=PrimaryFirst(...)`` with its settings, around the schedule given or built; ``phases=None`` never: every step's
methods are told the goal as declared, round 3 and 3b's schedule.
"""

from __future__ import annotations

from typing import Any, Sequence

from .methods import Block
from .schedules import Schedule


class Search:
    def __init__(
        self,
        blocks: Sequence[Block] = (),
        schedule: Schedule | None = None,
        redraws: int = 3,
        width: int = 4,
        within: Schedule | None = None,
        phases: Any = "auto",
    ) -> None:
        """``redraws``: how many more times a proposal is made again when it repeats a measured candidate (each time
        its methods are told that candidate's result); a proposal that still repeats has nothing new to measure, and a
        step with no new proposal ends its schedule's run. ``width``: how many candidates each step proposes from the
        same incumbent and measures together (see the module's docstring); a round's opening proposal (``Nested``),
        which sets the structure its inner steps search under, is always one. ``within``: what the default schedule
        runs inside its rounds over the structural blocks (``Alternate()`` when omitted: one block a step, in turn, so a
        text rewrite and a draw of the numbers are measured apart; ``Joint()``: every block every step, round 3's);
        unused with a ``schedule`` of one's own, or with no structural block (then every step is ``Joint()``).
        ``phases``: how a goal with a cost objective searches (see the module's docstring): ``"auto"``, a
        ``PrimaryFirst(...)``, or None (round 3 and 3b's schedule)."""
        if redraws < 0:
            raise ValueError("redraws must be at least 0")
        if width < 1:
            raise ValueError("width must be at least 1")
        wrong = [b for b in blocks if not isinstance(b, Block)]
        if wrong:
            raise TypeError(f"Search(blocks=...) takes Block objects, not {wrong}")
        if len({id(b) for b in blocks}) != len(blocks):
            raise ValueError("the same Block is given twice")
        if within is not None and not isinstance(within, Schedule):
            raise TypeError(f"Search(within=...) takes a Schedule (Alternate(), Joint(), ...), not {within!r}")
        if phases is not None and phases != "auto" and type(phases).__name__ != "PrimaryFirst":
            raise TypeError(f'Search(phases=...) is "auto", None or a PrimaryFirst(...), not {phases!r}')
        self.blocks, self.schedule, self.redraws, self.width = list(blocks), schedule, redraws, width
        self.within, self.phases = within, phases
