"""Reflection and skill-curation optimizers: the private machinery behind ``strands_evals.optimization``.

Ported from strands-labs/harness-optimizer (``strands_harness_optimizer/optimizers/``,
``formulas/skill_library_formula.py``, ``datamodels.py``, ``utils/templates.py``, ``utils/guardrails/`` and
``templates/``) without its Formula abstraction: an optimizer is given any object with ``get_tunable_params()`` and
``update_params()`` and reads and writes its parameters through those two methods.

- :class:`AgenticOptimizer`: the base. Samples traces, writes them to a temp folder, runs a Strands agent with the
  ``shell`` tool over them, and records ``last_metrics`` / ``last_wall_clock_s``.
- :class:`ContrastiveReflection`: compares successful and failed traces and submits new parameter values.
- :class:`SkillLibraryCuration` over a :class:`SkillSet`: writes a decision tree (create, optimize, retire) that the
  set resolves as ``existing - retired - optimized_old + created``.

The agents get a shell over the traces, and importing this package sets ``BYPASS_TOOL_CONSENT=true`` (unless already
set) so the shell tool runs without interactive consent: run them where model-written shell commands are acceptable.
"""

from .agentic import AgenticOptimizer
from .contrastive import ContrastiveReflection
from .datamodels import Reward, Rollout
from .skill_curation import SkillLibraryCuration
from .skill_set import MANIFEST_FILE, SkillSet, collect_decisions, skill_dirs_in, validate_skill_dir
from .templates import create_template, load_template

__all__ = [
    "AgenticOptimizer",
    "ContrastiveReflection",
    "SkillLibraryCuration",
    "SkillSet",
    "Rollout",
    "Reward",
    "create_template",
    "load_template",
    "collect_decisions",
    "skill_dirs_in",
    "validate_skill_dir",
    "MANIFEST_FILE",
]
