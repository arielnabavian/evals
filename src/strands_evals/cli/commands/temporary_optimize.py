from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.machinery
import inspect
import json
import logging
import sys
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from strands import Agent

from ...experiment import Experiment
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error, resolve_format
from .._entrypoint import EntryPointError, _import_from_path, import_attr, resolve_custom_evaluators
from ._temporary_evaluator_catalog import temporary_optimize_problem
from ._temporary_typescript_runner import TYPESCRIPT_SUFFIXES, load_typescript_harness

logger = logging.getLogger(__name__)

_HARNESS_SETTINGS = ("instructions", "effort")
_DEFAULT_HARNESS_SETTINGS = ("instructions",)
_EFFORT_LEVELS = ("auto", "off", "minimal", "low", "medium", "high", "xhigh", "max")
_LEVELS_WITHOUT_REASONING = frozenset({"auto", "off"})
_SHARED_STATE_OPTIONS = frozenset({"session_manager", "memory_manager", "storage"})
_TRIAL_OVERRIDES: dict[str, Any] = {"session": False, "memory": False, "callback_handler": None}
_INSTRUCTIONS_MIN_CHAR_LIMIT = 8000
_DEFAULT_ITERATIONS = 4
_DEFAULT_SPLIT = (0.5, 0.25, 0.25)
_NEXT_STEPS = (
    (
        "leave no train cases",
        "Add more cases to the experiment, or pass --no-split to search and select on every case",
    ),
    ("shares must", "Fix --split so that the three shares sum to 1, or pass --no-split"),
    ("unique name", "Give every case in the experiment file a unique name, then try again"),
    ("has no cases", "Add cases to the experiment file, or create a new one with /experiment in strands dev"),
    (
        "the baseline (the agent at its defaults)",
        "The cause is in parentheses: if the agent failed, check that it can answer one of the cases in a strands "
        "dev chat; if a judge or the model service failed, check its credentials, then try again",
    ),
)


class _Harness(Protocol):
    @property
    def options(self) -> dict[str, Any]: ...

    def build(self, values: dict[str, Any]) -> Any: ...

    def builds_with(self, **values: Any) -> bool: ...


@dataclass(frozen=True)
class _Construction:
    agent: Any
    args: tuple[Any, ...]
    options: dict[str, Any]


class _HarnessCapture:
    def __init__(self, harness: ModuleType) -> None:
        self.harness = harness
        self.original: Callable[..., Any] = harness.create_harness
        self.constructions: list[_Construction] = []
        self.recording = False

    def create_harness(self, *args: Any, **options: Any) -> Any:
        agent = self.original(*args, **options)
        if self.recording:
            self.constructions.append(_Construction(agent, args, dict(options)))
        return agent

    def __enter__(self) -> _HarnessCapture:
        self.recording = True
        self.harness.create_harness = self.create_harness
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.harness.create_harness = self.original
        self.recording = False


@dataclass(frozen=True)
class _HarnessFile:
    factory: Callable[..., Any]
    options: dict[str, Any]

    def build(self, values: dict[str, Any]) -> Any:
        overrides = {**values, **_TRIAL_OVERRIDES}
        if overrides.get("instructions") == "":
            overrides["instructions"] = None
        agent = self.factory(**overrides)
        if not isinstance(agent, Agent):
            raise TypeError(
                f"create_agent(**overrides) returned {type(agent).__name__}, not a Strands Agent. "
                "Return the agent that create_harness(...) builds, then try again."
            )
        return agent

    def builds_with(self, **values: Any) -> bool:
        try:
            self.build(values)
        except ValueError:
            return False
        return True


def _strands_harness() -> ModuleType:
    try:
        return importlib.import_module("strands_harness")
    except ImportError as e:
        raise EntryPointError(
            f"a Python harness file needs the strands-harness package, which is missing ({e}). "
            "Install strands-harness in the project's environment, then try again."
        ) from e


def _optimization() -> ModuleType:
    try:
        from ... import optimization
    except ImportError as e:
        raise EntryPointError(
            f"strands-evals optimize needs the optimization extra: "
            f"pip install 'strands-agents-evals[optimization]' ({e})"
        ) from e
    return optimization


def _is_agent_built_by_create_harness(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    if not any(isinstance(target, ast.Name) and target.id == "agent" for target in targets):
        return False
    call = node.value
    if not isinstance(call, ast.Call):
        return False
    function = call.func
    return (isinstance(function, ast.Name) and function.id == "create_harness") or (
        isinstance(function, ast.Attribute) and function.attr == "create_harness"
    )


def _assigns_agent_from_create_harness(path: Path) -> bool:
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    return any(_is_agent_built_by_create_harness(node) for node in tree.body)


def _installed_module_name(parts: list[str], root: Path) -> str | None:
    for index in range(1, len(parts) - 1):
        name = parts[index]
        if not name.isidentifier():
            continue
        spec = importlib.machinery.PathFinder.find_spec(name)
        if spec is None:
            continue
        expected = root.joinpath(*parts[: index + 1]).resolve()
        if any(Path(location).resolve() == expected for location in spec.submodule_search_locations or []):
            return ".".join(parts[index:])
    return None


def _module_name(path: Path, root: Path) -> str | None:
    if not path.is_relative_to(root):
        return None
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    packaged = bool(parts) and all(
        root.joinpath(*parts[:index], "__init__.py").is_file() for index in range(1, len(parts))
    )
    return ".".join(parts) if packaged else _installed_module_name(parts, root)


def _import_harness_module(path: Path, path_text: str) -> ModuleType:
    name = _module_name(path, Path.cwd().resolve())
    if name is None:
        return _import_from_path(str(path), path_text)
    sys.modules.pop(name, None)
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise EntryPointError(
            f"failed to import '{path_text}' as {name}: {e}. Fix that error, then try again; strands dev shows the "
            "same error when it opens this agent"
        ) from e


def _load_harness_file(path_text: str) -> _HarnessFile:
    path = Path(path_text).expanduser().resolve()
    harness = _strands_harness()
    sys.path[:0] = [d for d in (str(Path.cwd()), str(path.parent)) if d not in sys.path]
    with _HarnessCapture(harness) as capture:
        module = _import_harness_module(path, path_text)
    agent = getattr(module, "agent", None)
    if not isinstance(agent, Agent):
        raise EntryPointError(
            f'{path_text} does not export a Strands Agent named "agent". '
            "Export one, for example agent = create_harness(...), then try again."
        )
    construction = next((c for c in capture.constructions if c.agent is agent), None)
    if construction is None:
        raise EntryPointError(
            f"{path_text} builds its agent without create_harness(...), so optimize cannot read its settings. "
            "Build the agent with create_harness(...), then try again."
        )
    shared = sorted(set(construction.options) & _SHARED_STATE_OPTIONS)
    if shared:
        raise EntryPointError(
            f"{path_text} passes {', '.join(shared)} to create_harness(...); "
            "every trial would share it, so remove it to optimize this agent"
        )
    factory = getattr(module, "create_agent", None)
    if inspect.iscoroutinefunction(factory):
        raise EntryPointError(
            f"{path_text} defines create_agent as async, and the optimizer calls it as a regular function. "
            "Change it to def create_agent(**overrides), then try again."
        )
    if callable(factory):
        return _HarnessFile(factory, construction.options)
    if not _assigns_agent_from_create_harness(path):
        raise EntryPointError(
            f"{path_text} changes its agent after create_harness(...); "
            "define create_agent(**overrides) so each trial can build a fresh agent"
        )
    original = capture.original

    def create_agent(**overrides: Any) -> Any:
        return original(*construction.args, **{**construction.options, **overrides})

    return _HarnessFile(create_agent, construction.options)


def _instructions_domain(harness_file: _Harness, optimization: ModuleType) -> Any:
    if "system_prompt" in harness_file.options:
        raise EntryPointError(
            "instructions cannot be tuned: the agent passes a full system prompt, which replaces its instructions. "
            "Move that text into instructions, or tune another setting with --tune effort."
        )
    baseline = harness_file.options.get("instructions") or ""
    return optimization.TextValue(
        default=baseline,
        about="the agent's domain instructions, which the harness appends to its own system prompt",
        max_chars=max(_INSTRUCTIONS_MIN_CHAR_LIMIT, 2 * len(baseline)),
    )


def _effort_domain(harness_file: _Harness, optimization: ModuleType) -> Any:
    model = harness_file.options.get("model")
    if model is not None and not isinstance(model, str):
        raise EntryPointError(
            "effort cannot be tuned: the agent's model is a model object, and the harness ignores effort for model "
            "objects. Pass the model as an id string such as bedrock/<model id>, or tune instructions (the default)."
        )
    baseline = harness_file.options.get("effort", "auto")
    candidates = [level for level in _EFFORT_LEVELS if level != "auto" or baseline == "auto"]
    levels = [level for level in candidates if level == baseline or harness_file.builds_with(effort=level)]
    if len(levels) < 2 or all(level in _LEVELS_WITHOUT_REASONING for level in levels):
        raise EntryPointError(
            f"effort cannot be tuned: the agent's model accepts no reasoning levels besides {levels}. "
            "Tune instructions instead (the default)."
        )
    return optimization.Choice(levels, default=baseline)


_HARNESS_DOMAINS: dict[str, Callable[[_Harness, ModuleType], Any]] = {
    "instructions": _instructions_domain,
    "effort": _effort_domain,
}


def _harness_agent(harness_file: _Harness, settings: tuple[str, ...], optimization: ModuleType) -> Any:
    bindings = {name: _HARNESS_DOMAINS[name](harness_file, optimization) for name in settings}

    def create_agent(**values: Any) -> Any:
        return harness_file.build(values)

    return optimization.TunableFactory(create_agent, bindings)


def _load_harness(spec: str, typescript_runner: str | None) -> _Harness | None:
    suffix = Path(spec).suffix
    if suffix == ".py":
        return _load_harness_file(spec)
    if suffix in TYPESCRIPT_SUFFIXES:
        return load_typescript_harness(spec, typescript_runner)
    return None


def _tunable_agent(
    spec: str, settings: tuple[str, ...] | None, optimization: ModuleType, typescript_runner: str | None = None
) -> Any:
    harness = _load_harness(spec, typescript_runner)
    if harness is not None:
        return _harness_agent(harness, settings or _DEFAULT_HARNESS_SETTINGS, optimization)
    if settings is not None:
        raise EntryPointError(
            "--tune applies to a harness file (PATH.py or PATH.ts); a @tunable factory declares its own space"
        )
    factory = import_attr(spec)
    if not isinstance(factory, optimization.TunableFactory):
        raise EntryPointError(f"--agent {spec} must name a @tunable factory, or be a harness file (PATH.py or PATH.ts)")
    return factory


def _text_variables(factory: Any) -> list[Any]:
    text_keys = {info.key for info in factory.params() if info.type == "text"}
    return [obj for obj, keys, is_variable in factory.references() if is_variable and set(keys) <= text_keys]


def _search(factory: Any, args: argparse.Namespace, optimization: ModuleType) -> Any:
    blocks = []
    if args.rewriter_model is not None:
        texts = _text_variables(factory)
        if texts:
            rewriter = optimization.Reflection(model_config={"model_id": args.rewriter_model})
            blocks.append(optimization.Block(rewriter, *texts))
        else:
            logger.warning(
                "rewriter_model=<%s> | ignored because the agent has no text to rewrite", args.rewriter_model
            )
    return optimization.Search(blocks=blocks, width=args.width)


def _objectives(metrics: list[str] | None, experiment: Experiment, optimization: ModuleType) -> Any:
    names = list(dict.fromkeys(metrics or [evaluator.get_name() for evaluator in experiment.evaluators]))
    if not names:
        raise EntryPointError(
            "the experiment has no evaluators, so there is nothing to score. "
            "Add an evaluator to the experiment file, or pass --maximize METRIC."
        )
    goals = [optimization.Maximize(name) for name in names]
    return goals[0] if len(goals) == 1 else optimization.WeightedSum(*goals)


def _split(shares: tuple[float, float, float], optimization: ModuleType) -> Any:
    train, validation, test = shares
    return optimization.Split(train=train, validation=validation, test=test)


def _summary_lines(result: Any, record: str) -> list[str]:
    selected = result.runs[0].selected
    lines = [f"ships candidate {c.number}: {dict(c.params)}" for c in selected] or ["ships the baseline"]
    if result.test is not None:
        for arm in result.test.selected:
            for metric, difference in arm.difference.items():
                lines.append(f"test {metric}: {difference.mean:+.3f} vs baseline over {difference.cases} cases")
    lines.append(f"record: {record}")
    return lines


def _log_to_stderr(line: str) -> None:
    print(line, file=sys.stderr)  # noqa: T201


def temporary_answer_text(reply: Any) -> str:
    return str(reply).rstrip("\n")


async def _temporary_answer(agent: Any, case: Any) -> str:
    return temporary_answer_text(await agent.invoke_async(case.input))


def _study(args: argparse.Namespace, optimization: ModuleType) -> Any:
    custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
    experiment = _load_experiment(args.experiment_file, custom_evaluators)
    problem = temporary_optimize_problem(experiment.evaluators)
    if problem:
        raise ValueError(problem)
    factory = _tunable_agent(args.agent, args.tune, optimization, args.typescript_runner)
    return optimization.Study(
        agent=factory,
        task=_temporary_answer,
        experiment=experiment,
        objectives=_objectives(args.maximize, experiment, optimization),
        constraints=[optimization.AtMost(metric, limit) for metric, limit in args.at_most or []],
        search=_search(factory, args, optimization),
        repetitions=args.repetitions,
        max_concurrency=args.max_concurrency,
        split=None if args.no_split else _split(args.split, optimization),
        record=args.record,
        log=_log_to_stderr,
    )


def _is_new_directory(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _optimize(args: argparse.Namespace) -> int:
    if not _is_new_directory(Path(args.record).expanduser()):
        emit_error(f"--record {args.record} must be a new or empty directory; each study needs its own")
        return EXIT_BAD_INPUT
    optimization = _optimization()
    output_format = resolve_format(args)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            study = _study(args, optimization)
        except (TypeError, ValueError) as e:
            emit_error(_with_next_step(str(e)), debug=getattr(args, "debug", False), exc=e)
            return EXIT_BAD_INPUT
        try:
            result = study.optimize(args.iterations, patience=args.patience)
        except RuntimeError as e:
            raise RuntimeError(_with_next_step(str(e))) from e
    if output_format == "json":
        sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    else:
        for line in _summary_lines(result, args.record):
            print(line)  # noqa: T201
    return EXIT_OK


def _with_next_step(message: str) -> str:
    for marker, next_step in _NEXT_STEPS:
        if marker in message:
            return f"{message.rstrip('.')}. {next_step}."
    return message


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
            "TEMPORARY command on the feat/harness-optimization fork. Run a strands_evals.optimization Study: "
            "measure the agent at its current settings, search other settings on the train cases, select on the "
            "validation cases, and compare the selection with the baseline on the test cases. Every candidate "
            "and trial goes into the record directory as the study runs. Text settings are rewritten by a model "
            "that runs shell commands over the trial transcripts without approval. Run text studies only where "
            "that is acceptable."
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
    parser.add_argument(
        "--tune",
        metavar="SETTINGS",
        type=_parse_settings,
        default=None,
        help=f"harness settings to tune, comma-separated: {', '.join(_HARNESS_SETTINGS)} (default: instructions)",
    )
    parser.add_argument("--record", metavar="DIR", required=True, help="new directory for the study record")
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
        help=f"search steps after the baseline (default: {_DEFAULT_ITERATIONS})",
    )
    parser.add_argument("--width", type=_count(1), default=4, help="candidates per step (default: 4)")
    parser.add_argument(
        "--patience",
        type=_count(1),
        default=None,
        help="stop when the best candidates do not change for this many steps",
    )
    parser.add_argument("--repetitions", type=_count(1), default=1, help="runs of each case per candidate (default: 1)")
    parser.add_argument("--max-concurrency", type=_count(1), default=4, help="trials running at once (default: 4)")
    split = parser.add_mutually_exclusive_group()
    split.add_argument(
        "--split",
        metavar="TRAIN,VALIDATION,TEST",
        type=_parse_split,
        default=_DEFAULT_SPLIT,
        help="shares of the cases for search, selection and test (default: 0.5,0.25,0.25)",
    )
    split.add_argument(
        "--no-split",
        action="store_true",
        help="search and select on every case, and test nothing",
    )
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
        help="Bedrock model id that rewrites text settings (default: the fork's Reflection model)",
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
