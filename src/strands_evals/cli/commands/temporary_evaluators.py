from __future__ import annotations

import argparse
import json

from .._common import EXIT_OK, resolve_format
from .._io import write_text_output
from ._temporary_evaluator_catalog import (
    TEMPORARY_DEV_LANGUAGES,
    TEMPORARY_EVALUATOR_CATALOG,
    TemporaryEvaluatorEntry,
    TemporaryEvaluatorKind,
)

_KIND_HEADINGS: dict[TemporaryEvaluatorKind, str] = {
    TemporaryEvaluatorKind.RUBRIC: (
        "Rubric judges. A model scores each answer against a rubric. "
        "generate writes the rubric from the context, unless you give one."
    ),
    TemporaryEvaluatorKind.JUDGE: "Built-in judges. A model scores each answer with a fixed rubric.",
    TemporaryEvaluatorKind.MULTIMODAL: "Multimodal judges. A model scores answers about images or documents.",
    TemporaryEvaluatorKind.CHECK: "Checks. Code compares the answer with a value. No model is used.",
    TemporaryEvaluatorKind.CHAOS: "Chaos judges. A model scores how the agent handles injected tool failures.",
}

_SETTINGS_HINT = (
    'Give settings as JSON, for example --evaluator \'{"evaluator_type": "ToolCalled", "tool_name": "search"}\'. '
    "Run strands-evals evaluators --json to see every setting."
)


def _run(args: argparse.Namespace) -> int:
    if resolve_format(args) == "json":
        write_text_output(json.dumps([entry.to_dict() for entry in TEMPORARY_EVALUATOR_CATALOG], indent=2), None)
    else:
        write_text_output(_catalog_text(), None)
    return EXIT_OK


def _catalog_text() -> str:
    width = max(len(_entry_title(entry)) for entry in TEMPORARY_EVALUATOR_CATALOG)
    sections = [
        "\n".join(
            [
                _KIND_HEADINGS[kind],
                *(_entry_line(entry, width) for entry in TEMPORARY_EVALUATOR_CATALOG if entry.kind is kind),
            ]
        )
        for kind in TemporaryEvaluatorKind
    ]
    return "\n\n".join([*sections, _SETTINGS_HINT])


def _entry_line(entry: TemporaryEvaluatorEntry, width: int) -> str:
    languages = [label for language, label in TEMPORARY_DEV_LANGUAGES.items() if entry.dev_limit(language) is None]
    scope = (
        ""
        if len(languages) == len(TEMPORARY_DEV_LANGUAGES)
        else f" [strands dev: {' and '.join(languages)} agents only]"
        if languages
        else " [not in strands dev]"
    )
    return f"  {_entry_title(entry).ljust(width)}  {entry.description}{scope}"


def _entry_title(entry: TemporaryEvaluatorEntry) -> str:
    return f"{entry.name} ({', '.join(entry.required_settings)})" if entry.required_settings else entry.name


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "evaluators",
        parents=[parent],
        help="TEMPORARY: list every strands-evals evaluator, what it does, and its settings",
        description=(
            "TEMPORARY command on the feat/harness-optimization fork. List every strands-evals evaluator with a "
            "description, the settings that it takes, and the strands dev agents that it can score. Pass a name, "
            "or a JSON object with its settings, to generate --evaluator."
        ),
    )
    parser.set_defaults(func=_run)
    return parser
