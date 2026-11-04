"""
Contrastive Reflection Optimizer.

Ported from strands-labs/harness-optimizer,
``strands_harness_optimizer/optimizers/system_prompt/contrastive_reflection.py`` (``ContrastiveReflectionOptimizer``),
without the Formula abstraction: it reads and writes any object with ``get_tunable_params()`` and ``update_params()``.

Uses a strands LLM agent with shell tools to analyze rollout traces,
compare successful vs failed trajectories, and generate optimized parameters.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from jinja2 import Template

from .agentic import AgenticOptimizer, TunableValues
from .templates import Renderable, create_template

logger = logging.getLogger(__name__)

# What the agent is told when a value it submitted is over its limit: in the same conversation, so it shortens what
# it wrote rather than starting again.
OVER_LIMIT = (
    "{what} over {whose} limit: {details}. A value over its limit is discarded and the current value kept. Shorten "
    "{it} to fit, keeping what matters most (check each file with `wc -c`), write {it} to {its} file in your working "
    "folder again, and submit with submit_optimized_params."
)


# What the agent is told when its turn ended with nothing accepted and its last submission refused (``check``): in
# the same conversation, as a value over its limit is.
REFUSED = (
    "{refusal}\n\nYour turn ended with nothing submitted. Fix the file(s) in your working folder and submit again with "
    "submit_optimized_params."
)


def over_limit_message(over: dict[str, tuple[int, int]]) -> str:
    """The message for values over their limits: ``over`` maps a name to (its length, its limit)."""
    details = "; ".join(
        f"`{k}` is {n:,} characters, its limit {lim:,} ({n - lim:,} over)" for k, (n, lim) in over.items()
    )
    one = len(over) == 1
    names = ", ".join(f"`{k}`" for k in over)
    return OVER_LIMIT.format(
        what=f"{names} {'is' if one else 'are'}",
        whose="its" if one else "their",
        details=details,
        it="it" if one else "them",
        its="its" if one else "their",
    )


class ContrastiveReflection(AgenticOptimizer):
    """
    Optimizer that uses an LLM agent with shell tools for contrastive learning.

    The optimizer:
    1. Writes accumulated rollouts to a temp folder as JSON files
    2. Optionally samples a subset (with stratified sampling by reward)
    3. Creates a strands Agent with shell tool and submit_optimized_params tool
    4. The agent analyzes traces contrastively (success vs failure)
    5. The agent writes optimized params to files and submits via tool

    Example:
        optimizer = ContrastiveReflection(values, system_prompt_template, task_message_template)
        optimizer.add_rollouts(rollouts)
        optimizer.add_rewards(rewards)
        optimizer.step()
        optimizer.zero()
    """

    def __init__(
        self,
        values: TunableValues,
        system_prompt_template: str | Template | Renderable,
        task_message_template: str | Template | Renderable,
        limits: dict[str, int] | None = None,
        over_limit_retries: int = 0,
        **kwargs: Any,
    ):
        """
        Initialize the contrastive reflection optimizer.

        Args:
            values: The values whose parameters will be optimized: any object with
                ``get_tunable_params()`` and ``update_params(params)``.
            system_prompt_template: Jinja2 template (str or Template) for the
                reflection agent's system prompt. Use load_template()
                to load the default contrastive analysis template.
            task_message_template: Jinja2 template (str or Template, or any object
                with ``render(**kwargs)``) for the task message. Use load_template()
                to load a built-in template, or provide a custom one. Templates receive:
                traces_folder, work_folder (the agent's own), params (dict of current tunable params).
            limits: Each parameter's length limit, in characters (by name).
            over_limit_retries: How many times a submission with a value over its
                limit is sent back, in the same conversation, to be shortened
                (``OVER_LIMIT``); what is still over is the caller's to drop.
                ``revisions`` lists each time: the lengths that were over. A
                turn that ended on a submission ``check`` refused, with nothing
                accepted, is sent back the same way (``REFUSED``), within the
                same number of times.
            **kwargs: Additional arguments passed to AgenticOptimizer
                (model_config, region_name, boto_config, n_sample_traces,
                stratified_sampling, success_threshold, max_output_chars,
                system_prompt_suffix).
        """
        super().__init__(values, **kwargs)
        self.limits = dict(limits or {})
        self.over_limit_retries = over_limit_retries
        self.revisions: list[dict[str, int]] = []  # each submission sent back: the lengths over their limits

        self._system_prompt_template: Renderable
        if isinstance(system_prompt_template, str):
            self._system_prompt_template = create_template(system_prompt_template)
        else:
            self._system_prompt_template = system_prompt_template

        self._task_message_template: Renderable
        if isinstance(task_message_template, str):
            self._task_message_template = create_template(task_message_template)
        else:
            self._task_message_template = task_message_template

        logger.info(
            f"Initialized ContrastiveReflection "
            f"(model={self.model_config.get('model_id', 'default')}, "
            f"n_sample_traces={self.n_sample_traces})"
        )

    def step(self) -> None:
        """
        Analyze accumulated rollouts and update the values' parameters.

        Writes rollouts to a temp folder, runs the reflection agent with
        shell and submit_optimized_params tools, and updates the values.
        """
        if not self._rollouts:
            logger.warning("No rollouts accumulated, skipping step")
            return

        try:
            # Sample traces and write to temp folder
            indices = self._sample_traces()
            traces_folder = self._write_traces_to_temp(indices)
            self._refusal, refused_before = None, len(self.rejected)

            # Run reflection
            params = self.values.get_tunable_params()
            agent = self._run_reflection(traces_folder, params)

            # Read submitted params; a value over its limit (or a refused submission) is sent back
            optimized_params = self._get_submitted_params()
            optimized_params = self._shorten(agent, optimized_params)

            if optimized_params:
                self.values.update_params(optimized_params)
                self._prompt_history.append(optimized_params)
                self._step_count += 1
                summary = ", ".join(f"{k} ({len(str(v))} chars)" for k, v in optimized_params.items())
                logger.info(f"Step {self._step_count}: updated params: {summary}")
            elif len(self.rejected) > refused_before:
                last = self.rejected[-1]
                raise RuntimeError(
                    f"Every submission was refused ({len(self.rejected) - refused_before}); the last, "
                    f"{last['name']}: {last['reason']}"
                )
            else:
                raise RuntimeError(
                    "Optimization agent did not call submit_optimized_params tool. "
                    "The agent must write optimized params to files and submit via the tool."
                )

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error during contrastive reflection: {e}") from e
        finally:
            self._cleanup_temp()

    def _over(self, params: dict | None) -> dict[str, tuple[int, int]]:
        """The submitted values over their limits: name -> (length, limit)."""
        return {
            k: (len(str(v)), self.limits[k])
            for k, v in (params or {}).items()
            if self.limits.get(k) is not None and len(str(v)) > self.limits[k]
        }

    def _shorten(self, agent: Any, params: dict | None) -> dict | None:
        """Send a submission with values over their limits back to the agent, in its conversation, up to
        ``over_limit_retries`` times, and likewise a turn that ended with nothing accepted on a refused submission
        (``check``); each new submission replaces the values it gives. What it spent is added to ``last_metrics``
        and ``last_wall_clock_s``."""
        self.revisions = []
        seconds, metrics = self.last_wall_clock_s, dict(self.last_metrics)
        sent = 0
        while agent is not None and sent < self.over_limit_retries:
            over = self._over(params)
            if params and over:
                self.revisions.append({k: n for k, (n, _) in over.items()})
                message = over_limit_message(over)
            elif not params and self._refusal:
                message = REFUSED.format(refusal=self._refusal)
            else:
                break
            sent, self._refusal = sent + 1, None
            response = self._invoke_agent(agent, message)
            seconds += self.last_wall_clock_s
            if getattr(response, "metrics", None) is not getattr(agent, "event_loop_metrics", None):
                metrics = _added(metrics, self.last_metrics)  # this call's alone: add it to the first's
            else:
                metrics = dict(self.last_metrics)  # the agent's own metrics: every call so far
            again = self._get_submitted_params()
            if again:
                params = {**(params or {}), **again}
        self.last_wall_clock_s, self.last_metrics = seconds, metrics
        return params

    def _run_reflection(self, traces_folder: str, params: dict) -> Any:
        """Run the reflection agent to analyze traces; returns the agent (its conversation goes on when a submission is
        over a limit)."""
        template_vars = {
            "traces_folder": os.path.abspath(traces_folder),
            "work_folder": self._work_folder(),
            "params": params,
            "digest_file": self.digest_file,  # every trace's digest, or None
        }

        system_prompt = self._system_prompt_template.render(**template_vars)
        task_message = self._task_message_template.render(**template_vars)

        agent = self._create_agent(system_prompt)
        self._invoke_agent(agent, task_message)
        return agent

    # get_state() and load_state() inherited from AgenticOptimizer


def _added(a: dict, b: dict) -> dict:
    """Two calls' metrics (``AgenticOptimizer._metrics``'s) added up."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, (int, float)) and isinstance(out.get(k, 0), (int, float)):
            out[k] = out.get(k, 0) + v
        elif isinstance(v, dict):
            out[k] = {**(out.get(k) or {}), **{n: (out.get(k) or {}).get(n, 0) + c for n, c in v.items()}}
    return out
