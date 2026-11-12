from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import strands_evals.evaluators as evaluators_package
import strands_evals.evaluators.chaos as chaos_package
from strands_evals.case import Case
from strands_evals.cli.commands._temporary_evaluator_catalog import (
    TEMPORARY_EVALUATOR_CATALOG,
    TemporaryEvaluatorKind,
    temporary_evaluator_choice,
)
from strands_evals.cli.main import main
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator
from strands_evals.experiment import Experiment

FROM_CONTEXT = "strands_evals.cli.commands.generate.ExperimentGenerator.from_context_async"
CONSTRUCT = "strands_evals.cli.commands.generate.ExperimentGenerator.construct_evaluator_async"
SAMPLE_SETTINGS = {"value": "refund", "tool_name": "lookup_order", "name": "cart", "rubric": "Score the answer."}


def _exported_evaluator_names() -> set[str]:
    exported = {name for name in evaluators_package.__all__ if name != "Evaluator"}
    return exported | set(chaos_package.__all__)


def _required_constructor_settings(evaluator: type) -> set[str]:
    parameters = list(inspect.signature(evaluator.__init__).parameters.values())[1:]
    return {parameter.name for parameter in parameters if parameter.default is inspect.Parameter.empty}


def _spec(name: str) -> str:
    entry = next(entry for entry in TEMPORARY_EVALUATOR_CATALOG if entry.name == name)
    return json.dumps(
        {"evaluator_type": name, **{setting: SAMPLE_SETTINGS[setting] for setting in entry.required_settings}}
    )


def _cases_only() -> AsyncMock:
    return AsyncMock(return_value=Experiment(cases=[Case[str, str](name="c0", input="q0")]))


def _written_rubric(rubric: str = "written rubric") -> AsyncMock:
    async def construct(prompt, evaluator, message_history=None):
        return evaluator(rubric=rubric)

    return AsyncMock(side_effect=construct)


def _generate(tmp_path: Path, *evaluators: str) -> dict:
    output = tmp_path / "experiment.json"
    exit_code = main(
        ["generate", "--context", "A refund agent.", *[f"--evaluator={spec}" for spec in evaluators], "-o", str(output)]
    )
    assert exit_code == 0
    return json.loads(output.read_text())


def test_should_list_every_exported_evaluator_when_the_catalog_is_built():
    assert {entry.name for entry in TEMPORARY_EVALUATOR_CATALOG} == _exported_evaluator_names()


def test_should_list_each_evaluator_once_when_the_catalog_is_built():
    names = [entry.name for entry in TEMPORARY_EVALUATOR_CATALOG]

    assert len(names) == len(set(names))


@pytest.mark.parametrize("entry", TEMPORARY_EVALUATOR_CATALOG, ids=lambda entry: entry.name)
def test_should_name_only_constructor_parameters_when_an_entry_lists_settings(entry):
    parameters = set(inspect.signature(entry.evaluator.__init__).parameters)

    assert {setting.name for setting in entry.settings} <= parameters


@pytest.mark.parametrize("entry", TEMPORARY_EVALUATOR_CATALOG, ids=lambda entry: entry.name)
def test_should_mark_every_constructor_argument_without_a_default_as_required_when_generate_cannot_write_it(entry):
    written = {"rubric"} if entry.kind is TemporaryEvaluatorKind.RUBRIC else set()

    assert set(entry.required_settings) == _required_constructor_settings(entry.evaluator) - written


@pytest.mark.parametrize(
    "entry",
    [entry for entry in TEMPORARY_EVALUATOR_CATALOG if entry.kind is not TemporaryEvaluatorKind.CHAOS],
    ids=lambda entry: entry.name,
)
def test_should_load_the_evaluator_from_an_experiment_file_when_generate_accepts_it(entry):
    choice = temporary_evaluator_choice(_spec(entry.name))
    evaluator = choice.build(rubric="r" if choice.needs_rubric else None)
    written = Experiment(cases=[Case[str, str](name="c0", input="q0")], evaluators=[evaluator])

    loaded = Experiment.from_dict(json.loads(json.dumps(written.to_dict())))

    assert [type(evaluator) for evaluator in loaded.evaluators] == [entry.evaluator]


@pytest.mark.parametrize(
    "entry",
    [entry for entry in TEMPORARY_EVALUATOR_CATALOG if entry.kind is TemporaryEvaluatorKind.CHAOS],
    ids=lambda entry: entry.name,
)
def test_should_refuse_a_chaos_evaluator_when_an_experiment_file_cannot_load_it(entry):
    stored = {"cases": [], "evaluators": [entry.evaluator().to_dict()]}

    with pytest.raises(Exception, match="Cannot find"):
        Experiment.from_dict(stored)
    with pytest.raises(Exception, match="chaos evaluator"):
        temporary_evaluator_choice(entry.name)


def test_should_print_every_evaluator_as_json_when_json_is_requested(capsys):
    exit_code = main(["evaluators", "--json"])

    listed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [item["name"] for item in listed] == [entry.name for entry in TEMPORARY_EVALUATOR_CATALOG]
    assert all(item["description"] and item["kind"] and item["reads"] for item in listed)


@pytest.mark.parametrize(
    ("name", "agents", "limit"),
    [
        ("OutputEvaluator", ["python", "typescript"], None),
        ("HelpfulnessEvaluator", ["python"], "only for Python agents"),
        ("ToolCalled", ["python"], "only for Python agents"),
        ("InteractionsEvaluator", [], "do not record them"),
        ("StateEquals", [], "do not record it"),
        ("RecoveryStrategyEvaluator", [], "chaos experiment"),
    ],
)
def test_should_say_which_strands_dev_agents_an_evaluator_can_score_when_json_is_requested(capsys, name, agents, limit):
    main(["evaluators", "--json"])

    item = next(item for item in json.loads(capsys.readouterr().out) if item["name"] == name)
    assert item["agents"] == agents
    assert (limit is None and "limit" not in item) or limit in item["limit"]


def test_should_give_the_type_of_each_setting_when_json_is_requested(capsys):
    main(["evaluators", "--json"])

    contains = next(item for item in json.loads(capsys.readouterr().out) if item["name"] == "Contains")
    assert {setting["name"]: (setting["type"], setting["required"]) for setting in contains["settings"]} == {
        "value": ("string", True),
        "case_sensitive": ("boolean", False),
        "name": ("string", False),
    }


def test_should_print_each_name_with_its_required_settings_when_text_is_requested(capsys):
    exit_code = main(["evaluators", "--rich"])

    text = capsys.readouterr().out
    assert exit_code == 0
    assert "ToolCalled (tool_name)" in text
    assert all(entry.name in text and entry.description in text for entry in TEMPORARY_EVALUATOR_CATALOG)


def test_should_attach_every_evaluator_when_several_are_given(tmp_path: Path):
    with patch(FROM_CONTEXT, new=_cases_only()), patch(CONSTRUCT, new=_written_rubric()):
        written = _generate(
            tmp_path,
            "OutputEvaluator",
            "HelpfulnessEvaluator",
            '{"evaluator_type": "ToolCalled", "tool_name": "lookup_order"}',
            '{"evaluator_type": "Contains", "value": "refund", "case_sensitive": false}',
        )

    output, helpfulness, tool_called, contains = written["evaluators"]
    assert [item["evaluator_type"] for item in written["evaluators"]] == [
        "OutputEvaluator",
        "HelpfulnessEvaluator",
        "ToolCalled",
        "Contains",
    ]
    assert output["rubric"] == "written rubric"
    assert "rubric" not in helpfulness
    assert tool_called["tool_name"] == "lookup_order"
    assert (contains["value"], contains["case_sensitive"]) == ("refund", False)
    assert len(written["cases"]) == 1


def test_should_write_one_rubric_for_each_rubric_judge_when_several_are_given(tmp_path: Path):
    construct = _written_rubric()

    with patch(FROM_CONTEXT, new=_cases_only()), patch(CONSTRUCT, new=construct):
        _generate(tmp_path, "OutputEvaluator", "TrajectoryEvaluator")

    assert sorted(call.kwargs["evaluator"].__name__ for call in construct.await_args_list) == [
        OutputEvaluator.__name__,
        TrajectoryEvaluator.__name__,
    ]
    assert all("A refund agent." in call.kwargs["prompt"] for call in construct.await_args_list)


def test_should_keep_the_given_rubric_when_the_evaluator_has_one(tmp_path: Path):
    construct = _written_rubric()

    with patch(FROM_CONTEXT, new=_cases_only()), patch(CONSTRUCT, new=construct):
        written = _generate(tmp_path, '{"evaluator_type": "OutputEvaluator", "rubric": "my rubric"}')

    assert [(item["evaluator_type"], item["rubric"]) for item in written["evaluators"]] == [
        ("OutputEvaluator", "my rubric")
    ]
    construct.assert_not_awaited()


def test_should_keep_two_copies_of_an_evaluator_when_their_names_differ(tmp_path: Path):
    with patch(FROM_CONTEXT, new=_cases_only()):
        written = _generate(
            tmp_path,
            '{"evaluator_type": "Contains", "value": "refund", "name": "says refund"}',
            '{"evaluator_type": "Contains", "value": "days", "name": "says days"}',
        )

    assert [evaluator["name"] for evaluator in written["evaluators"]] == ["says refund", "says days"]


def test_should_reject_two_copies_of_an_evaluator_when_they_have_the_same_name(capsys):
    from_context = _cases_only()

    with patch(FROM_CONTEXT, new=from_context):
        exit_code = main(
            [
                "generate",
                "--context",
                "ctx",
                "--evaluator",
                "HelpfulnessEvaluator",
                "--evaluator",
                "HelpfulnessEvaluator",
            ]
        )

    assert exit_code == 2
    assert 'different "name" setting' in capsys.readouterr().err
    from_context.assert_not_awaited()


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("Helpfulness", "is not a strands-evals evaluator"),
        ("FailureCommunicationEvaluator", "is a chaos evaluator"),
        ("ToolCalled", 'ToolCalled needs tool_name. Pass it as JSON: {"evaluator_type": "ToolCalled"'),
        ('{"evaluator_type": "Contains", "value": "x", "tool": "y"}', "Contains does not take tool"),
        ('{"evaluator_type": "Contains", "value": "x", "case_sensitive": "no"}', "case_sensitive must be a boolean"),
        ('{"evaluator_type": "ToolCalled", "tool_name": 3}', "tool_name must be a string"),
        ('{"tool_name": "search"}', 'an "evaluator_type" name'),
        ("{not json", "the evaluator JSON is not valid"),
    ],
)
def test_should_reject_the_evaluator_before_any_generation_when_its_spec_is_invalid(capsys, spec, message):
    from_context = _cases_only()

    with patch(FROM_CONTEXT, new=from_context), pytest.raises(SystemExit) as exited:
        main(["generate", "--context", "ctx", f"--evaluator={spec}"])

    assert exited.value.code == 2
    assert message in capsys.readouterr().err
    from_context.assert_not_awaited()


@pytest.mark.parametrize(
    ("language", "spec", "message"),
    [
        ("typescript", "HelpfulnessEvaluator", "HelpfulnessEvaluator cannot score a TypeScript agent in strands dev"),
        ("typescript", '{"evaluator_type": "ToolCalled", "tool_name": "search"}', "only for Python agents"),
        ("python", "InteractionsEvaluator", "InteractionsEvaluator cannot score a Python agent in strands dev"),
        ("python", '{"evaluator_type": "StateEquals", "name": "cart"}', "do not record it"),
    ],
)
def test_should_reject_the_evaluator_when_a_strands_dev_run_of_the_agent_cannot_give_it_data(
    capsys, language, spec, message
):
    from_context = _cases_only()

    with patch(FROM_CONTEXT, new=from_context):
        exit_code = main(
            ["generate", "--context", "ctx", "--temporary-agent-language", language, f"--evaluator={spec}"]
        )

    assert exit_code == 2
    assert message in capsys.readouterr().err
    from_context.assert_not_awaited()


@pytest.mark.parametrize(
    ("language", "spec"),
    [
        ("typescript", "OutputEvaluator"),
        ("typescript", '{"evaluator_type": "Contains", "value": "refund"}'),
        ("python", "HelpfulnessEvaluator"),
        ("python", '{"evaluator_type": "ToolCalled", "tool_name": "search"}'),
    ],
)
def test_should_accept_the_evaluator_when_a_strands_dev_run_of_the_agent_gives_it_data(tmp_path: Path, language, spec):
    output = tmp_path / "experiment.json"

    with patch(FROM_CONTEXT, new=_cases_only()), patch(CONSTRUCT, new=_written_rubric()):
        exit_code = main(
            [
                "generate",
                "--context",
                "ctx",
                "--temporary-agent-language",
                language,
                f"--evaluator={spec}",
                "-o",
                str(output),
            ]
        )

    assert exit_code == 0
    assert len(json.loads(output.read_text())["evaluators"]) == 1
