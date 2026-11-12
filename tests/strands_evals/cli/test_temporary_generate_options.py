from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from strands_evals.case import Case
from strands_evals.cli.main import main
from strands_evals.experiment import Experiment

FROM_CONTEXT = "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async"
AGENT = "strands_evals.cli.commands._temporary_generate_progress.Agent"
CONSTRUCTOR_FLAGS = [
    "--max-parallel-num-cases",
    "3",
    "--no-expected-output",
    "--expected-trajectory",
    "--expected-interactions",
    "--metadata",
]


def _capture_generator():
    generators = []

    async def from_context(self, **_kwargs):
        generators.append(self)
        return Experiment(cases=[Case[str, str](name="c0", input="q0")])

    return generators, from_context


def _constructor_settings(generator) -> dict:
    return {
        "include_expected_output": generator.include_expected_output,
        "include_expected_trajectory": generator.include_expected_trajectory,
        "include_expected_interactions": generator.include_expected_interactions,
        "include_metadata": generator.include_metadata,
        "max_parallel_num_cases": generator.max_parallel_num_cases,
    }


@pytest.mark.parametrize("progress", [False, True])
def test_should_pass_every_constructor_flag_to_the_generator_when_the_flags_are_set(tmp_path: Path, progress: bool):
    generators, from_context = _capture_generator()
    progress_flags = ["--temporary-progress", str(tmp_path / "progress.jsonl")] if progress else []

    with patch(FROM_CONTEXT, new=from_context):
        exit_code = main(["generate", "--context", "ctx", *CONSTRUCTOR_FLAGS, *progress_flags])

    assert exit_code == 0
    assert _constructor_settings(generators[0]) == {
        "include_expected_output": False,
        "include_expected_trajectory": True,
        "include_expected_interactions": True,
        "include_metadata": True,
        "max_parallel_num_cases": 3,
    }


def test_should_keep_the_generator_defaults_when_no_constructor_flag_is_set():
    generators, from_context = _capture_generator()

    with patch(FROM_CONTEXT, new=from_context):
        exit_code = main(["generate", "--context", "ctx"])

    assert exit_code == 0
    assert _constructor_settings(generators[0]) == {
        "include_expected_output": True,
        "include_expected_trajectory": False,
        "include_expected_interactions": False,
        "include_metadata": False,
        "max_parallel_num_cases": 10,
    }


@pytest.mark.parametrize("value", ["0", "-2", "many"])
def test_should_reject_the_parallel_limit_when_it_is_not_a_positive_whole_number(value: str, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--context", "ctx", "--max-parallel-num-cases", value])

    assert exit_info.value.code == 2
    assert "--max-parallel-num-cases" in capsys.readouterr().err


def test_should_list_every_new_flag_when_generate_help_runs(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["generate", "--help"])

    help_text = capsys.readouterr().out
    assert exit_info.value.code == 0
    for flag in [
        "--max-parallel-num-cases",
        "--no-expected-output",
        "--expected-trajectory",
        "--expected-interactions",
        "--metadata",
        "--temporary-progress",
    ]:
        assert flag in help_text


class _SchemaAgent:
    def __init__(self, **_kwargs):
        pass

    async def structured_output_async(self, output_model, _prompt):
        values = {
            "name": "refund",
            "input": "Refund order 2001",
            "expected_output": "Refund it",
            "expected_trajectory": ["lookup_order", "refund"],
            "metadata": {"topic": "refunds"},
        }
        return output_model(**{field: values[field] for field in output_model.model_fields})


def test_should_write_cases_with_the_requested_fields_when_progress_and_schema_flags_are_set(tmp_path: Path):
    output = tmp_path / "experiment.json"
    progress = tmp_path / "progress.jsonl"

    with patch(AGENT, _SchemaAgent):
        exit_code = main(
            [
                "generate",
                "--context",
                "ctx",
                "--num-cases",
                "2",
                "--no-expected-output",
                "--expected-trajectory",
                "--metadata",
                "--temporary-progress",
                str(progress),
                "-o",
                str(output),
            ]
        )

    cases = json.loads(output.read_text())["cases"]
    assert exit_code == 0
    assert [case["expected_trajectory"] for case in cases] == [["lookup_order", "refund"]] * 2
    assert [case["metadata"] for case in cases] == [{"topic": "refunds"}] * 2
    assert [case.get("expected_output") for case in cases] == [None, None]
    assert json.loads(progress.read_text().splitlines()[-1])["done"] == 2
