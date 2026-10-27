"""Optimize a Strands agent against a Strands Evals experiment.

    @tunable(
        system_prompt=TextValue(default="You answer support tickets concisely."),
        conversation_manager=OneOf(
            component(SlidingWindowConversationManager, window_size=Int(10, 80, default=40)),
            component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8, default=0.3)),
        ),
    )
    def create_agent(system_prompt, conversation_manager): ...

    study = Study(agent=create_agent, experiment=experiment, repetitions=2,
                  objectives=Maximize("Helpfulness"), constraints=[AtMost("tokens", 4000)])
    study.optimize(iterations=30)
    study.best_candidate.params

A study evaluates **candidates**: values for the agent's tunable arguments. Each candidate runs every case of the
experiment ``repetitions`` times; each run is a **trial**, with a status (an agent that raised, or an evaluator that
failed, is a status, never a zero score). Estimates are case-level means with 95% intervals.

A **search** says how candidates are proposed: **blocks**, each a method and the variables it proposes (the objects
declared in ``@tunable``: ``Block(TPE(), window_size)``, ``Block(Reflection(), system_prompt, tools)``), and a
**schedule** that says which blocks change in each iteration, holding the rest at the incumbent: ``Joint()`` (all of
them), ``Alternate(...)`` (one at a time), or ``Nested(outer=..., inner=...)`` (rounds: an outer proposal, then a
bounded inner search, compared at its best; nests to any depth; its rules are policies: ``allocation``, ``value``,
``start`` and ``order``). By default every variable is in a block for its type
(choices and numbers by Optuna's TPE, ``TextValue`` by contrastive reflection, ``SkillLibrary`` by skill-library
curation) and the schedule is built from the space.

Several objectives are given wrapped in how they combine: ``Pareto(...)`` (jointly), ``Independent(...)`` (one copy of
every method per objective), ``WeightedSum(...)``, ``Chebyshev(...)``, ``Lexicographic(...)`` or
``CheapestOnFront(...)``. A goal names a metric:
an evaluator's, the agent's usage (``tokens``, ``seconds``, ...), or a ``Derived`` one computed from those. With
``record=``, the study writes every candidate as it finishes, which the explorer reads. What a study spends is kept in
tokens, by role (the agent; each method's own model calls) and phase (search, re-measurement, validation, test), every
attempt of every trial counted: ``Result.cost``, and in the record as each phase ends.

The study decides with a **model** of every goal metric (``model.py``: BayesBench's Bayesian posterior over the choices,
the binned numbers, each rewritten text and the case): which candidate it holds, what methods are told, and which are
best use each candidate's decided values, and a constraint holds when the model gives it the required probability. When
a run ends, the best are measured again and the selection is made from the refitted model (``Selection``).

A goal with a cost objective (``Pareto(Maximize("reward"), Minimize("tokens"))``, ``CheapestOnFront(...)``) searches
reward first, then cost (``PrimaryFirst``, ``phases.py``): phase A is a max-reward search until it stops gaining;
phase B lowers the cost from its result, A*, under paired non-inferiority to it, in the same budget; the selection
never ships worse than A* by more than its margin. ``Search(phases=None)`` searches as before (every step told the goal
as declared).
"""

from .cost import Cost
from .methods import TPE, Block, Draw, Method, Optuna, Reflection, SkillCuration, Thompson
from .phases import PrimaryFirst, Seesaw, Stall
from .projection import (
    AtLeast,
    AtMost,
    CheapestOnFront,
    Chebyshev,
    Combine,
    Derived,
    Estimate,
    Independent,
    Lexicographic,
    Maximize,
    Minimize,
    Pareto,
    WeightedSum,
)
from .result import Result
from .schedules import (
    Allocation,
    Alternate,
    BestOfRound,
    Fixed,
    FromIncumbent,
    Joint,
    Nested,
    OuterThenInner,
    RoundOrder,
    RoundStart,
    RoundValue,
    Schedule,
)
from .search import Search
from .selection import LowerBound, Mean, Selection
from .space import (
    Choice,
    Component,
    Float,
    Int,
    OneOf,
    SkillLibrary,
    Switch,
    TextValue,
    TunableFactory,
    Values,
    component,
    tool_descriptions,
    tunable,
)
from .split import Split
from .study import Candidate, Scope, Study
from .trials import Trial, TrialInvalid, TrialStatus

__all__ = [
    "Allocation",
    "Alternate",
    "AtLeast",
    "AtMost",
    "BestOfRound",
    "Block",
    "Candidate",
    "CheapestOnFront",
    "Chebyshev",
    "Choice",
    "Combine",
    "Component",
    "Cost",
    "Derived",
    "Draw",
    "Estimate",
    "Fixed",
    "Float",
    "FromIncumbent",
    "Independent",
    "Int",
    "Joint",
    "Lexicographic",
    "LowerBound",
    "Maximize",
    "Mean",
    "Method",
    "Minimize",
    "Nested",
    "OneOf",
    "Optuna",
    "OuterThenInner",
    "Pareto",
    "PrimaryFirst",
    "Reflection",
    "RoundOrder",
    "RoundStart",
    "RoundValue",
    "Schedule",
    "Search",
    "Selection",
    "Scope",
    "Seesaw",
    "SkillCuration",
    "SkillLibrary",
    "Stall",
    "Study",
    "Switch",
    "TPE",
    "TextValue",
    "Thompson",
    "Trial",
    "TrialInvalid",
    "TrialStatus",
    "TunableFactory",
    "Values",
    "WeightedSum",
    "component",
    "Split",
    "Result",
    "tool_descriptions",
    "tunable",
]
