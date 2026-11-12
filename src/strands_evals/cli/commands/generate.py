"""`strands-evals generate` — synthesize an Experiment via ExperimentGenerator."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...generators import ExperimentGenerator
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error
from .._entrypoint import resolve_custom_evaluators
from .._io import write_text_output
from ._temporary_evaluator_catalog import (
    TEMPORARY_DEV_LANGUAGES,
    TemporaryEvaluatorChoice,
    temporary_evaluator_choice,
)
from ._temporary_generate_progress import TemporaryGenerateProgress, TemporaryProgressGenerator
from .temporary_optimize import _count


def _validate_args(args: argparse.Namespace) -> str | None:
    """Return an error string if argument combinations are invalid, else None.

    The argparse mutually-exclusive group already enforces "exactly one of
    --context/--experiment". This layer guards mode-specific flag misuse:

    - `--evaluator` and `--num-topics` only apply to context mode; the
      `from_experiment_async` library API derives evaluators from the source
      and doesn't take a topic plan, so silently dropping them would mislead
      users into thinking the flag took effect.
    - `--extra-information` and `--custom-evaluator` only apply to experiment
      mode; in context mode the equivalent guidance goes inline into
      `--context` and there's no source experiment to load custom evaluators
      for.
    """
    if args.experiment is not None:
        if args.evaluator is not None:
            return "--evaluator is not supported with --experiment (evaluators are inherited from the source)."
        if args.num_topics is not None:
            return "--num-topics is not supported with --experiment (topic planning runs only in --context mode)."
    else:
        if args.extra_information is not None:
            return "--extra-information requires --experiment."
        if args.custom_evaluator:
            return "--custom-evaluator requires --experiment."
        choices = args.evaluator or []
        return _repeated_evaluator_error(choices) or _agent_language_error(choices, args.temporary_agent_language)
    return None


def _agent_language_error(choices: list[TemporaryEvaluatorChoice], language: str | None) -> str | None:
    if language is None:
        return None
    agent = TEMPORARY_DEV_LANGUAGES[language]
    reasons = [
        f"{choice.entry.name} cannot score a {agent} agent in strands dev. {limit}"
        for choice in choices
        if (limit := choice.entry.dev_limit(language))
    ]
    return " ".join(reasons) or None


def _repeated_evaluator_error(choices: list[TemporaryEvaluatorChoice]) -> str | None:
    labels = [choice.label for choice in choices]
    repeated = sorted({label for label in labels if labels.count(label) > 1})
    if not repeated:
        return None
    return (
        f"--evaluator {', '.join(repeated)} is set more than one time. "
        'Give each copy a different "name" setting, or remove the copies.'
    )


def _run(args: argparse.Namespace) -> int:
    error = _validate_args(args)
    if error is not None:
        emit_error(error)
        return EXIT_BAD_INPUT

    options = _generator_options(args)
    if args.temporary_progress is None:
        return _generate(args, ExperimentGenerator(**options))
    progress = TemporaryGenerateProgress(args.temporary_progress)
    try:
        return _generate(args, TemporaryProgressGenerator(progress, **options))
    except Exception as failure:
        progress.failed(failure)
        raise
    finally:
        progress.close()


def _generator_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "input_type": str,
        "output_type": str,
        "model": args.model,
        "include_expected_output": args.include_expected_output,
        "include_expected_trajectory": args.include_expected_trajectory,
        "include_expected_interactions": args.include_expected_interactions,
        "include_metadata": args.include_metadata,
    }
    if args.max_parallel_num_cases is not None:
        options["max_parallel_num_cases"] = args.max_parallel_num_cases
    return options


def _generate(args: argparse.Namespace, generator: ExperimentGenerator[str, str]) -> int:
    if args.experiment is not None:
        custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
        source = Experiment.from_file(args.experiment, custom_evaluators=custom_evaluators)
        experiment = asyncio.run(
            generator.from_experiment_async(
                source_experiment=source,
                task_description=args.task_description or "",
                num_cases=args.num_cases,
                extra_information=args.extra_information,
            )
        )
    else:
        experiment = asyncio.run(_from_context(args, generator, args.evaluator or []))

    if args.output is not None:
        # `experiment.to_file` enforces a `.json` extension and writes via
        # `to_dict()`; route through it so the on-disk shape matches what
        # `Experiment.from_file` round-trips.
        experiment.to_file(args.output)
    else:
        write_text_output(json.dumps(experiment.to_dict(), indent=2, ensure_ascii=False), None)

    print(  # noqa: T201
        f"generated {len(experiment.cases)} case(s), {len(experiment.evaluators)} evaluator(s)",
        file=sys.stderr,
    )
    return EXIT_OK


async def _from_context(
    args: argparse.Namespace,
    generator: ExperimentGenerator[str, str],
    choices: list[TemporaryEvaluatorChoice],
) -> Experiment:
    task_description = args.task_description or ""
    experiment = await generator.from_context_async(
        context=args.context,
        task_description=task_description,
        num_cases=args.num_cases,
        num_topics=args.num_topics,
    )
    if not choices:
        return experiment
    rubric_prompt = f"Create a rubric with the following context: {args.context} for this task: {task_description} "
    evaluators = await asyncio.gather(*(_evaluator(generator, choice, rubric_prompt) for choice in choices))
    return Experiment(cases=experiment.cases, evaluators=list(evaluators))


async def _evaluator(
    generator: ExperimentGenerator[str, str], choice: TemporaryEvaluatorChoice, rubric_prompt: str
) -> Evaluator:
    if not choice.needs_rubric:
        return choice.build()
    written = await generator.construct_evaluator_async(prompt=rubric_prompt, evaluator=choice.entry.evaluator)
    return choice.build(rubric=written.rubric)


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate",
        parents=[parent],
        help="generate a starter Experiment via ExperimentGenerator",
        description=(
            "Synthesize an Experiment by wrapping ExperimentGenerator. Two "
            "modes: --context creates cases (and an optional rubric) from a "
            "free-form description via from_context_async; --experiment "
            "derives new cases from an existing Experiment file via "
            "from_experiment_async, inheriting the source's evaluators. "
            "--context/--experiment are mutually exclusive. With -o the "
            "experiment is written via Experiment.to_file (.json enforced); "
            "without -o the JSON document is emitted on stdout."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--context",
        metavar="CONTEXT",
        default=None,
        help=(
            "context describing the task, tools, or domain (from_context_async). "
            'Use shell substitution for file contents, e.g. --context "$(cat tools.txt)".'
        ),
    )
    source.add_argument(
        "--experiment",
        metavar="EXPERIMENT_FILE",
        default=None,
        help=(
            "path to an existing Experiment JSON used as a reference "
            "(from_experiment_async). Cases are inspired by the source; "
            "evaluators are derived from the source's defaults."
        ),
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="number of test cases to generate (default: 5)",
    )
    parser.add_argument(
        "--task-description",
        metavar="STR",
        default=None,
        help="short description of the task the agent will perform (default: empty)",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=None,
        help="--context only: expand into N topic-specific prompts for diverse coverage",
    )
    parser.add_argument(
        "--evaluator",
        metavar="NAME|JSON",
        type=temporary_evaluator_choice,
        action="append",
        default=None,
        help=(
            "--context only, repeatable (TEMPORARY): an evaluator to attach. Give its name, for example "
            'HelpfulnessEvaluator, or a JSON object with its settings, for example {"evaluator_type": '
            '"ToolCalled", "tool_name": "search"}. Rubric judges get a rubric written from the context. '
            "Run strands-evals evaluators to see every evaluator. "
            "Omit to produce an experiment with a placeholder Evaluator."
        ),
    )
    parser.add_argument(
        "--extra-information",
        metavar="TEXT",
        default=None,
        help="--experiment only: extra context for the new cases/rubric",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help=(
            "--experiment only: custom Evaluator subclass to register before "
            "loading the source experiment file (repeatable)"
        ),
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="override the judge model used by the generator",
    )
    parser.add_argument(
        "--max-parallel-num-cases",
        metavar="N",
        type=_count(1),
        default=None,
        help="TEMPORARY: generate at most N cases at the same time (max_parallel_num_cases, default: 10)",
    )
    parser.add_argument(
        "--no-expected-output",
        dest="include_expected_output",
        action="store_false",
        help="TEMPORARY: generate cases without an expected output (include_expected_output=False)",
    )
    parser.add_argument(
        "--expected-trajectory",
        dest="include_expected_trajectory",
        action="store_true",
        help="TEMPORARY: also generate the expected tool trajectory of each case (include_expected_trajectory)",
    )
    parser.add_argument(
        "--expected-interactions",
        dest="include_expected_interactions",
        action="store_true",
        help="TEMPORARY: also generate the expected interactions of each case (include_expected_interactions)",
    )
    parser.add_argument(
        "--metadata",
        dest="include_metadata",
        action="store_true",
        help="TEMPORARY: also generate metadata for each case (include_metadata)",
    )
    parser.add_argument(
        "--temporary-agent-language",
        choices=sorted(TEMPORARY_DEV_LANGUAGES),
        default=None,
        help=(
            "TEMPORARY: the language of the strands dev agent that the experiment is for. generate rejects "
            "an evaluator that a strands dev run of that agent cannot give data to"
        ),
    )
    parser.add_argument(
        "--temporary-progress",
        metavar="PATH",
        default=None,
        help=(
            "TEMPORARY: write one JSON line to PATH when topic planning, case generation, or rubric "
            "generation starts, when each case is written or fails, and when the command fails"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the generated experiment JSON to PATH (via Experiment.to_file)",
    )
    parser.set_defaults(func=_run)
    return parser
