"""
Skill Library Curation.

Ported from strands-labs/harness-optimizer, ``strands_harness_optimizer/optimizers/skills/skill_library.py``
(``SkillLibraryOptimizer``), without the Formula abstraction: it curates a :class:`SkillSet` (the port of
``SkillLibraryFormula``) instead.

Curates a whole skill LIBRARY from rollout traces: an agent reads the traces and
the deployed skills, then decides which skills to create, revise or retire. Pairs
with :class:`SkillSet`.

Two things differ from :class:`ContrastiveReflection`, and both follow
from optimizing a set of files rather than a string:

**The agent writes a decision tree, not a value.** ``submit_optimized_params``
reads each submitted path as a plain-text parameter value, which cannot express
"create this skill, retire that one". So this optimizer gives the agent an
``output_folder`` and reads the tree it wrote:

    output_folder/
      create/<name>/SKILL.md      a skill that did not exist
      optimize/<name>/SKILL.md    a revision of a deployed skill
      retire.txt                  one name per line
      manifest.json               the agent's own record of its reasoning

MERGE and SPLIT need no folders of their own — both are written as a create plus a
retire of every source.

**"No change" is a correct outcome.** ``ContrastiveReflection`` raises
when the agent submits nothing, which is right for a prompt: you asked for an
improvement and got none. A skill library is different — once the recurring
patterns are covered, the right move is to stop, and churn actively degrades a
healthy library (each extra rule risks over-fitting one noisy sample). So a
deliberate SKIP is accepted; only an agent that produced *nothing at all* — no
decisions and no manifest — is treated as a failure. See :meth:`SkillLibraryCuration.step`.

The ``output_folder`` also PERSISTS, unlike the temp trace folder. Everything the
agent writes there (manifest, changelogs, merge notes) is the run's record of why
the library looks the way it does, and the caller chose the location, so there is
nothing to clean up or hand back.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from jinja2 import Template

from .agentic import AgenticOptimizer
from .skill_set import MANIFEST_FILE, SkillSet, collect_decisions, render_skill_index
from .templates import Renderable, create_template

logger = logging.getLogger(__name__)

SKILL_SYSTEM_PROMPT_SUFFIX = ""


class SkillLibraryCuration(AgenticOptimizer):
    """Optimizer that curates a set of skills from rollout traces.

    Inherits trace sampling, agent construction, the tool-output guardrail and
    metrics capture from :class:`AgenticOptimizer`; replaces the
    submit-a-value contract with a decision tree on disk.

    The templates receive exactly three variables:

    ``traces_folder``
        Temp folder holding the sampled traces. Deleted after the step.
    ``output_folder``
        Where to write decisions. Persists.
    ``skill_folder``
        The deployed skill set, or ``""`` on a cold start.

    plus ``skill_index`` — the deployed skills' names and frontmatter, rendered
    from ``skill_folder`` so the two cannot disagree. Templates that build their
    own index can ignore it.

    Example:
        skill_set = SkillSet(skill_dir="./skills")
        optimizer = SkillLibraryCuration(
            skill_set,
            system_prompt_template=load_template("skill_library/system_prompt.jinja"),
            task_message_template=load_template("skill_library/task_message.jinja"),
            output_folder="./runs/iter0",
        )
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
        optimizer.step()
        skill_set.materialize("./runs/iter0/skill_set")

    Args:
        skill_set: The :class:`SkillSet` whose set will be optimized.
        system_prompt_template: Jinja2 template for the curator agent's persona.
        task_message_template: Jinja2 template for the task message.
        output_folder: Directory the agent writes its decisions to. Created if
            absent. Unlike the trace folder this is NOT cleaned up — it is the
            run's record.
        **kwargs: Passed to :class:`AgenticOptimizer` (model_config,
            region_name, boto_config, n_sample_traces, stratified_sampling,
            success_threshold, max_output_chars).
    """

    def __init__(
        self,
        skill_set: SkillSet,
        system_prompt_template: str | Template | Renderable,
        task_message_template: str | Template | Renderable,
        output_folder: str,
        **kwargs: Any,
    ):
        if not isinstance(skill_set, SkillSet):
            raise TypeError(
                "SkillLibraryCuration requires a SkillSet (it optimizes a "
                f"set of skill directories); got {type(skill_set).__name__}. For a "
                "single text value use ContrastiveReflection."
            )
        explicit_suffix = kwargs.pop("system_prompt_suffix", None)
        super().__init__(skill_set, **kwargs)

        self.system_prompt_suffix = explicit_suffix if explicit_suffix is not None else SKILL_SYSTEM_PROMPT_SUFFIX

        self.output_folder = output_folder
        self._system_prompt_template: Renderable = (
            create_template(system_prompt_template)
            if isinstance(system_prompt_template, str)
            else system_prompt_template
        )
        self._task_message_template: Renderable = (
            create_template(task_message_template) if isinstance(task_message_template, str) else task_message_template
        )
        self._last_decisions: dict = {}

        logger.info(
            "Initialized SkillLibraryCuration (model=%s, n_sample_traces=%s, output_folder=%s)",
            self.model_config.get("model_id", "default"),
            self.n_sample_traces,
            output_folder,
        )

    @property
    def skill_set(self) -> SkillSet:
        """The :class:`SkillSet` being curated (the values this optimizer updates)."""
        return cast(SkillSet, self.values)

    def _get_tools(self, submit_optimized_params) -> list:
        """Drop ``submit_optimized_params``: this agent writes a decision tree.

        That tool reads each submitted path as one plain-text parameter VALUE,
        which cannot express "create this skill, retire that one". Leaving it in
        the schema while the task message asks for a folder of decisions gives the
        agent two contradictory ways to finish — and the one that "succeeds"
        silently produces nothing this optimizer can read.
        """
        from strands_tools import shell

        return [shell]

    def step(self) -> None:
        """Curate the library from the accumulated rollouts.

        Samples traces, runs the curator agent against a persistent
        ``output_folder``, then applies whatever decisions it wrote.

        A step that changes nothing is NOT an error, provided the agent left a
        manifest explaining why. An agent that wrote neither decisions nor a
        manifest did not do the work, and that DOES raise — the two look identical
        from the folder alone and mean opposite things.
        """
        if not self._rollouts:
            logger.warning("No rollouts accumulated, skipping step")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        try:
            indices = self._sample_traces()
            traces_folder = self._write_traces_to_temp(indices)
            self._run_curation(traces_folder)

            decisions = collect_decisions(self.output_folder)
            self._last_decisions = decisions
            n = sum(len(v) for v in decisions.values())

            if n == 0:
                if os.path.isfile(os.path.join(self.output_folder, MANIFEST_FILE)):
                    self._step_count += 1
                    logger.info(
                        "Step %d: SKIP -- no qualifying patterns; the library is unchanged (see %s/%s)",
                        self._step_count,
                        self.output_folder,
                        MANIFEST_FILE,
                    )
                    return
                raise RuntimeError(
                    "The curator agent produced neither decisions nor a "
                    f"{MANIFEST_FILE} in {self.output_folder}. A deliberate SKIP "
                    f"must still write {MANIFEST_FILE} saying why, so an agent that "
                    "failed part-way is not mistaken for one that correctly decided "
                    "to change nothing."
                )

            self.skill_set.update_params({"decisions_dir": self.output_folder})
            self._step_count += 1
            self._prompt_history.append({"skills": self.skill_set.skill_names()})
            logger.info(
                "Step %d: %d decision(s) applied -> %d skill(s): %s",
                self._step_count,
                n,
                len(self.skill_set.members),
                ", ".join(self.skill_set.skill_names()) or "(none)",
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error during skill-library optimization: {e}") from e
        finally:
            self._cleanup_temp()

    def _run_curation(self, traces_folder: str) -> None:
        """Render the templates and run the curator agent."""
        skill_folder = self.skill_set.get_tunable_params().get("skill_dir", "")
        template_vars = {
            "traces_folder": os.path.abspath(traces_folder),
            "output_folder": os.path.abspath(self.output_folder),
            "skill_folder": os.path.abspath(skill_folder) if skill_folder else "",
            "skill_index": render_skill_index(skill_folder or None),
        }
        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)

        agent = self._create_agent(system_prompt)
        self._invoke_agent(agent, task_message)

    @property
    def last_decisions(self) -> dict:
        """The decisions read from the most recent step.

        ``{"create": [...], "optimize": [...], "retire": [...]}``; all empty for a
        SKIP.
        """
        return dict(self._last_decisions)

    def get_state(self) -> dict:
        """Add the skill set and output folder to the inherited checkpoint state."""
        state = super().get_state()
        state["output_folder"] = self.output_folder
        state["skill_names"] = self.skill_set.skill_names()
        state["skill_dir"] = self.skill_set.get_tunable_params().get("skill_dir", "")
        state["last_decisions"] = dict(self._last_decisions)
        return state

    def load_state(self, state: dict) -> None:
        """Restore from a checkpoint written by :meth:`get_state`."""
        super().load_state(state)
        if "output_folder" in state:
            self.output_folder = state["output_folder"]
        if "last_decisions" in state:
            self._last_decisions = dict(state["last_decisions"])
        if state.get("skill_dir"):
            self.skill_set.update_params({"skill_dir": state["skill_dir"]})
