from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from ...case import Case
from ...experiment import Experiment
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error, resolve_format
from .._entrypoint import EntryPointError, import_attr, resolve_custom_evaluators
from ._temporary_evaluator_catalog import temporary_optimize_problem
from ._temporary_harness import TemporaryHarness, load_temporary_harness, temporary_answer_text
from ._temporary_optimize_space import (
    TemporarySpace,
    TemporarySpaceVariable,
    read_temporary_space,
    temporary_text_value,
)

logger = logging.getLogger(__name__)

_HARNESS_SETTINGS = ("instructions", "effort")
_DEFAULT_HARNESS_SETTINGS = ("instructions",)
_EFFORT_LEVELS = ("auto", "off", "minimal", "low", "medium", "high", "xhigh", "max")
_LEVELS_WITHOUT_REASONING = frozenset({"auto", "off"})
_DEFAULT_ITERATIONS = 4
_DEFAULT_SPLIT = (0.5, 0.25, 0.25)
_DEFAULT_MAX_ROLLOUTS = 500
_BASELINE_NEXT_STEP = (
    "If the agent failed, check that it can answer one of the cases in a strands dev chat; if a judge or the model "
    "service failed, check its credentials, then try again"
)
_NEXT_STEPS = (
    (
        "at least 1 is needed",
        "Add more cases to the experiment, or pass a --split that gives train and validation one case each at least",
    ),
    ("share above 0", "Give train and validation a share above 0 in --split"),
    ("sums to", "Fix --split so that the three shares sum to 1"),
    ("case names must be unique", "Give every case in the experiment file a unique name, then try again"),
    ("every case needs a name", "Give every case in the experiment file a unique name, then try again"),
    ("nothing to optimize", "Add an evaluator to the experiment file, or pass --maximize METRIC"),
    ("can't cover the baseline", "Pass a larger --max-rollouts, or use an experiment with fewer cases"),
    ("unmeasured for the baseline", _BASELINE_NEXT_STEP),
    ("trial of the baseline failed", _BASELINE_NEXT_STEP),
    ("nothing to resume", "Start the optimization again with --record DIR"),
    (
        "changed:",
        "Pass the experiment, the agent, and the flags of the first run, or start a new run with --record DIR",
    ),
)


def _optimization() -> ModuleType:
    try:
        from ... import optimization
    except ImportError as e:
        raise EntryPointError(
            f"strands-evals optimize needs the optimization extra: "
            f"pip install 'strands-agents-evals[optimization]' ({e})"
        ) from e
    return optimization


def _temporary_check_instructions(harness: TemporaryHarness) -> None:
    if "system_prompt" in harness.options:
        raise EntryPointError(
            "instructions cannot be tuned: the agent passes a full system prompt, which replaces its instructions. "
            "Move that text into instructions, or tune another setting with --tune effort."
        )


def _instructions_marker(harness: TemporaryHarness, optimization: ModuleType) -> Any:
    _temporary_check_instructions(harness)
    return temporary_text_value(
        harness.options.get("instructions") or "",
        "the agent's domain instructions, which the harness appends to its own system prompt",
        optimization,
    )


def _effort_marker(harness: TemporaryHarness, optimization: ModuleType) -> Any:
    model = harness.options.get("model")
    if model is not None and not isinstance(model, str):
        raise EntryPointError(
            "effort cannot be tuned: the agent's model is a model object, and the harness ignores effort for model "
            "objects. Pass the model as an id string such as bedrock/<model id>, or tune instructions (the default)."
        )
    baseline = harness.options.get("effort", "auto")
    candidates = [level for level in _EFFORT_LEVELS if level != "auto" or baseline == "auto"]
    levels = [level for level in candidates if level == baseline or harness.builds_with(effort=level)]
    if len(levels) < 2 or all(level in _LEVELS_WITHOUT_REASONING for level in levels):
        raise EntryPointError(
            f"effort cannot be tuned: the agent's model accepts no reasoning levels besides {levels}. "
            "Tune instructions instead (the default)."
        )
    return optimization.OneOf(*levels, default=baseline)


_SETTING_MARKERS: dict[str, Callable[[TemporaryHarness, ModuleType], Any]] = {
    "instructions": _instructions_marker,
    "effort": _effort_marker,
}


def _setting_factory(harness: TemporaryHarness, settings: tuple[str, ...], optimization: ModuleType) -> Any:
    markers = {name: _SETTING_MARKERS[name](harness, optimization) for name in settings}

    def create_agent(**values: Any) -> Any:
        return harness.build(values)

    return optimization.tunable(create_agent, inject=markers)


def _temporary_check_variable(harness: TemporaryHarness, variable: TemporarySpaceVariable) -> None:
    if variable.option == "instructions":
        _temporary_check_instructions(harness)
    if variable.item is None:
        return
    problem = harness.temporary_without_problem(variable.option, variable.item)
    if problem:
        raise EntryPointError(
            f"--space variable {variable.key} cannot leave out {json.dumps(variable.item)} of {variable.option}: "
            f"{problem}. Make the space file again from the agent file as it is now, then try again."
        )


def _space_factory(harness: TemporaryHarness, space: TemporarySpace, optimization: ModuleType) -> Any:
    for variable in space.variables:
        _temporary_check_variable(harness, variable)
    markers = space.bindings(harness.options, optimization)

    def create_agent(**values: Any) -> Any:
        settings, without = space.overrides(values)
        return harness.build(settings, without)

    return optimization.tunable(create_agent, inject=markers)


def _tunable_agent(args: argparse.Namespace, optimization: ModuleType) -> Any:
    space = read_temporary_space(args.space) if args.space is not None else None
    harness = load_temporary_harness(args.agent, args.typescript_runner)
    if harness is not None and space is not None:
        return _space_factory(harness, space, optimization)
    if harness is not None:
        return _setting_factory(harness, args.tune or _DEFAULT_HARNESS_SETTINGS, optimization)
    if args.tune is not None or space is not None:
        flag = "--tune" if args.tune is not None else "--space"
        raise EntryPointError(
            f"{flag} applies to a harness file (PATH.py or PATH.ts); a @tunable factory declares its own space"
        )
    factory = import_attr(args.agent)
    if not isinstance(factory, optimization.TunableFactory):
        raise EntryPointError(
            f"--agent {args.agent} must name a @tunable factory, or be a harness file (PATH.py or PATH.ts)"
        )
    return factory


def _rewriter_blocks(factory: Any, rewriter_model: str | None, optimization: ModuleType) -> list[Any]:
    if rewriter_model is None:
        return []
    texts = list(dict.fromkeys(info.name for info in factory.space.params() if info.kind == "text"))
    if not texts:
        logger.warning("rewriter_model=<%s> | ignored because the agent has no text to rewrite", rewriter_model)
        return []
    return [optimization.Block(optimization.Reflection(model=rewriter_model), *texts)]


def _baseline_only() -> Any:
    from ...optimization import Schedule

    class TemporaryBaselineOnly(Schedule):
        name = "temporary_baseline_only"

        def next(self, ctx: Any) -> None:
            return None

    return TemporaryBaselineOnly()


def _search(factory: Any, args: argparse.Namespace, optimization: ModuleType) -> Any:
    blocks = _rewriter_blocks(factory, args.rewriter_model, optimization)
    if args.iterations == 0:
        return optimization.Search(blocks, schedule=_baseline_only())
    return optimization.Search(blocks) if blocks else None


def _objective(metrics: list[str] | None, optimization: ModuleType) -> Any:
    if not metrics:
        return None
    goals = [optimization.Maximize(name) for name in dict.fromkeys(metrics)]
    return goals[0] if len(goals) == 1 else optimization.WeightedSum(*goals)


async def _temporary_rollout(agent_factory: Callable[..., Any], case: Case) -> dict[str, Any]:
    agent = agent_factory()
    return {"output": temporary_answer_text(await agent.invoke_async(case.input))}


def _load_experiment(path: str, custom_evaluators: list[Any]) -> Experiment:
    try:
        return Experiment.from_file(path, custom_evaluators=custom_evaluators)
    except FileNotFoundError:
        raise
    except (ValueError, KeyError, TypeError) as e:
        raise EntryPointError(
            f"{path} is not a usable Experiment file ({type(e).__name__}: {e}). Fix the file (strands-evals validate "
            f"{path} checks it), or create a new one with /experiment in strands dev"
        ) from e


def _optimizer(args: argparse.Namespace, optimization: ModuleType, resumed: Path | None) -> Any:
    experiment = _load_experiment(args.experiment_file, resolve_custom_evaluators(args.custom_evaluator or []))
    problem = temporary_optimize_problem(experiment.evaluators)
    if problem:
        raise ValueError(problem)
    factory = _tunable_agent(args, optimization)
    shared: dict[str, Any] = {
        "rollout_fn": _temporary_rollout,
        "objective": _objective(args.maximize, optimization),
        "constraints": [optimization.AtMost(metric, limit) for metric, limit in args.at_most or []],
        "search": _search(factory, args, optimization),
        "budget": optimization.Budget(max_rollouts=args.max_rollouts, max_iterations=args.iterations or None),
        "max_workers": args.max_concurrency,
    }
    if resumed is not None:
        return optimization.Optimizer.resume(resumed, experiment=experiment, factory=factory, **shared)
    train, validation, test = args.split
    return optimization.Optimizer(
        experiment,
        factory,
        split=optimization.Split(train=train, validation=validation, test=test),
        runs_per_candidate=args.repetitions,
        proposals_per_iteration=args.width,
        seed=args.seed,
        base_dir=args.record,
        **shared,
    )


def _run_to_resume(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if (path / "run.json").is_file():
        return path
    runs = sorted(child for child in path.iterdir() if (child / "run.json").is_file()) if path.is_dir() else []
    if len(runs) == 1:
        return runs[0]
    found = f"holds {len(runs)} runs; pass one of their folders" if runs else "holds no run to continue"
    raise EntryPointError(f"--resume {path_text} {found}. Or start the optimization again with --record DIR.")


def _summary_lines(result: Any) -> list[str]:
    return [*result.summary().splitlines(), f"record: {result.run_dir}"]


def _is_new_directory(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _with_next_step(message: str) -> str:
    for marker, next_step in _NEXT_STEPS:
        if marker in message:
            return f"{message.rstrip('.')}. {next_step}."
    return message


def _optimize(args: argparse.Namespace) -> int:
    if args.record is not None and not _is_new_directory(Path(args.record).expanduser()):
        emit_error(f"--record {args.record} must be a new or empty directory; each optimization needs its own")
        return EXIT_BAD_INPUT
    output_format = resolve_format(args)
    with contextlib.redirect_stdout(sys.stderr):
        optimization = _optimization()
        try:
            resumed = _run_to_resume(args.resume) if args.resume is not None else None
            optimizer = _optimizer(args, optimization, resumed)
        except (TypeError, ValueError) as e:
            emit_error(_with_next_step(str(e)), debug=getattr(args, "debug", False), exc=e)
            return EXIT_BAD_INPUT
        try:
            result = optimizer.run()
        except RuntimeError as e:
            raise RuntimeError(_with_next_step(str(e))) from e
    if output_format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        for line in _summary_lines(result):
            print(line)  # noqa: T201
    return EXIT_OK


def _parse_settings(raw: str) -> tuple[str, ...]:
    settings = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = [setting for setting in settings if setting not in _HARNESS_SETTINGS]
    if unknown or not settings:
        raise argparse.ArgumentTypeError(f"--tune takes a comma-separated list of {', '.join(_HARNESS_SETTINGS)}")
    return settings


def _parse_split(raw: str) -> tuple[float, float, float]:
    try:
        train, validation, test = (float(part) for part in raw.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--split must be TRAIN,VALIDATION,TEST shares, got '{raw}'") from e
    return train, validation, test


def _parse_limit(raw: str) -> tuple[str, float]:
    metric, separator, value = raw.partition("=")
    try:
        limit = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--at-most must be METRIC=NUMBER, got '{raw}'") from e
    if not separator or not metric:
        raise argparse.ArgumentTypeError(f"--at-most must be METRIC=NUMBER, got '{raw}'")
    return metric, limit


def _count(minimum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"expected a whole number, got '{raw}'") from e
        if value < minimum:
            raise argparse.ArgumentTypeError(f"expected a number of at least {minimum}, got {value}")
        return value

    return parse


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "optimize",
        parents=[parent],
        help="TEMPORARY: tune an agent's settings against an experiment",
        description=(
            "TEMPORARY command until the strands_evals.optimization CLI ships. Run an Optimizer: score the agent "
            "at its current settings, search other settings on the train cases, pick on the validation cases, and "
            "compare the pick with the baseline on the test cases. The run folder inside --record holds run.json, "
            "events.jsonl as the run goes, then best.values.json, best.diff, and result.json. Text settings are "
            "rewritten by a Reflection agent that reads the trial transcripts."
        ),
    )
    parser.add_argument("experiment_file", metavar="EXPERIMENT_FILE", help="path to a serialized Experiment JSON file")
    parser.add_argument(
        "--agent",
        metavar="PATH.py|PATH.ts|MODULE:ATTR",
        required=True,
        help=(
            "a Python harness file that exports a Strands Agent as 'agent', built with create_harness(...) or by "
            "create_agent(**overrides); a TypeScript harness file that --typescript-runner can build (TEMPORARY); "
            "or a @tunable factory. Run from the project root so the file's imports resolve."
        ),
    )
    tuning = parser.add_mutually_exclusive_group()
    tuning.add_argument(
        "--tune",
        metavar="SETTINGS",
        type=_parse_settings,
        default=None,
        help=f"harness settings to tune, comma-separated: {', '.join(_HARNESS_SETTINGS)} (default: instructions)",
    )
    tuning.add_argument(
        "--space",
        metavar="PATH",
        default=None,
        help=(
            'TEMPORARY: JSON file {"variables": [...]} with the whole search space of a harness file. Each variable '
            "has a key (its name in the run's values), a kind (text, choice or switch), the createHarness option it "
            "changes, and a default (a text without one starts from the agent's value). A choice lists its values; "
            'a switch keeps or leaves out one item of its option: {"index": N}, {"key": "NAME"} or {"name": "NAME"}'
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--record",
        metavar="DIR",
        default=None,
        help="new or empty directory; the run folder is made inside it",
    )
    target.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help=(
            "continue a stopped run from its events.jsonl: its run folder, or the --record folder that holds it. "
            "Pass the experiment, the agent, and the flags of the first run again"
        ),
    )
    parser.add_argument(
        "--maximize",
        metavar="METRIC",
        action="append",
        default=None,
        help="evaluator metric to maximize (repeatable; default: every evaluator, equally weighted)",
    )
    parser.add_argument(
        "--at-most",
        metavar="METRIC=NUMBER",
        action="append",
        type=_parse_limit,
        default=None,
        help="constraint on a metric, such as tokens=4000 (repeatable)",
    )
    parser.add_argument(
        "--iterations",
        type=_count(0),
        default=_DEFAULT_ITERATIONS,
        help=f"search steps after the baseline; 0 scores only the baseline (default: {_DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--width", type=_count(1), default=4, help="candidates per step (default: 4)")
    parser.add_argument("--repetitions", type=_count(1), default=1, help="runs of each case per candidate (default: 1)")
    parser.add_argument("--max-concurrency", type=_count(1), default=4, help="trials running at once (default: 4)")
    parser.add_argument(
        "--max-rollouts",
        type=_count(1),
        default=_DEFAULT_MAX_ROLLOUTS,
        help=f"most agent runs in the whole optimization (default: {_DEFAULT_MAX_ROLLOUTS})",
    )
    parser.add_argument(
        "--split",
        metavar="TRAIN,VALIDATION,TEST",
        type=_parse_split,
        default=_DEFAULT_SPLIT,
        help="shares of the cases for search, picking and test (default: 0.5,0.25,0.25)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the split and the search (default: 0)")
    parser.add_argument(
        "--typescript-runner",
        metavar="COMMAND",
        default=None,
        help=(
            "TEMPORARY: command that starts the Node runner for a TypeScript agent (PATH.ts). The command gets the "
            "agent file and the project root as its last two arguments, and speaks JSON lines on stdin and stdout"
        ),
    )
    parser.add_argument(
        "--rewriter-model",
        metavar="MODEL_ID",
        default=None,
        help="model id of the Reflection agent that rewrites text settings (default: the Reflection default)",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help="custom Evaluator subclass that the experiment file names (repeatable)",
    )
    parser.set_defaults(func=_optimize)
    return parser
