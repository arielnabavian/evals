import json
from types import SimpleNamespace

import optuna
import pytest
from strands import Agent
from strands.models.model import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    TPE,
    Alternate,
    AtMost,
    Block,
    Choice,
    Float,
    Independent,
    Int,
    Joint,
    Maximize,
    Method,
    Minimize,
    Nested,
    OneOf,
    Optuna,
    Pareto,
    Search,
    Selection,
    Study,
    TextValue,
    component,
    tunable,
)
from strands_evals.optimization.methods import Draw
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import needs_bayesbench


class Says(Model):
    """Replies with fixed text; two output tokens per word."""

    def __init__(self, text):
        self.text, self.config = text, {"model_id": "says"}

    def update_config(self, **kw):
        self.config.update(kw)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        text = self.text if self.text is not None else system_prompt
        words = len(text.split())
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": 50, "outputTokens": 2 * words, "totalTokens": 50 + 2 * words}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


class WordCount(Evaluator):
    score_range = (0, None)

    def evaluate(self, case):
        n = len(str(case.actual_output).split())
        return [EvaluationOutput(score=n, test_pass=True, reason=f"{n} words")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


EXPERIMENT = Experiment(
    cases=[Case(name="sea", input="the sea"), Case(name="city", input="a city at night")], evaluators=[WordCount()]
)


def grid(values, *variables):
    """The study's search: Optuna's grid over ``values`` (by name), for ``variables``."""
    return Search(width=1, blocks=[Block(Optuna(lambda: optuna.samplers.GridSampler(values, seed=0)), *variables)])


# ---- numbers ---------------------------------------------------------------------------------------------------------
STANZAS = Int(1, 6, default=2)


@tunable(stanzas=STANZAS)
def create_poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


def test_longest_poem_within_a_token_budget():
    # deciding with measured means, a candidate exactly at the limit meets it (by the model, see below, it does not:
    # it is not 90% sure to)
    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        constraints=[AtMost("output_tokens", 80)],
        search=grid({"stanzas": [1, 2, 3, 4, 5, 6]}, STANZAS),
        selection=Selection(by="measured"),
    )
    study.optimize(iterations=6)
    assert study.candidates[0].params == {"stanzas": 2} and study.candidates[0].proposed == ()  # the baseline
    assert {t.params["stanzas"] for t in study.candidates} == {1, 2, 3, 4, 5, 6}
    best = study.best_candidate
    assert best.params == {"stanzas": 4} and best.feasible
    assert {k: best.metrics[k] for k in ("WordCount", "input_tokens", "output_tokens", "tokens")} == {
        "WordCount": 40.0,
        "input_tokens": 50.0,
        "output_tokens": 80.0,
        "tokens": 130.0,
    }
    assert best.status == "complete" and best.estimates["WordCount"].n == 2
    assert best.agent().model.text.count("word") == 40


@needs_bayesbench
def test_by_the_model_a_constraint_must_hold_with_the_required_probability():
    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        constraints=[AtMost("output_tokens", 80)],
        search=grid({"stanzas": [1, 2, 3, 4, 5, 6]}, STANZAS),
        log=None,
    )
    study.optimize(iterations=6)
    assert study.by_model
    decided = {c.params["stanzas"]: study.view(c) for c in study.candidates}
    assert decided[3].p_met[0] > 0.99 and decided[5].p_met[0] < 0.01  # 60 and 100 output tokens
    assert 0.2 < decided[4].p_met[0] < 0.8  # exactly at the limit: as likely over it as under
    assert study.best_candidate.params == {"stanzas": 3}
    # a deterministic metric is decided as measured
    assert decided[3].decided["WordCount"].mean == pytest.approx(30.0, rel=1e-3)


def test_several_objectives_give_a_pareto_set():
    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Pareto(Maximize("WordCount"), Minimize("tokens")),
        search=grid({"stanzas": [1, 2, 3, 4, 5, 6]}, STANZAS),
    )
    study.optimize(iterations=6)
    assert {t.params["stanzas"] for t in study.best_candidates} == {1, 2, 3, 4, 5, 6}
    with pytest.raises(ValueError, match="best_candidates"):
        _ = study.best_candidate


def test_unknown_metrics_and_blocks_are_caught():
    with pytest.raises(ValueError, match=r"unknown metric 'words'.*'WordCount'"):
        Study(search=Search(width=1), agent=create_poet, experiment=EXPERIMENT, objectives=Maximize("words"))
    with pytest.raises(ValueError, match="does not declare"):
        Study(
            agent=create_poet,
            experiment=EXPERIMENT,
            objectives=Maximize("WordCount"),
            search=Search(width=1, blocks=[Block(TPE(), Int(1, 6, default=2))]),  # equal to STANZAS, but another object
        )


class Manager:
    """A live object, which a record cannot hold."""


class _Proposes(Draw):
    def __init__(self, value):
        self.value, self.origin = value, {"method": "Proposes"}

    def suggest_categorical(self, name, choices):
        return self.value


class Proposes(Method):
    """Proposes one fixed value, whatever the options are."""

    kinds = ("choice",)

    def __init__(self, value):
        self.value = value

    def begin(self, block, base, history, structure, study, objective):
        return _Proposes(self.value)


def test_every_value_is_plain_json_or_the_study_says_which_is_not(tmp_path):
    def study(binding, **kw):
        @tunable(style=binding)
        def create(style):
            return Agent(model=Says("a poem"), callback_handler=None)

        return Study(agent=create, experiment=EXPERIMENT, objectives=Maximize("WordCount"), log=None, **kw)

    with pytest.raises(ValueError, match=r"style: its option <.*Manager object .*> cannot be recorded: it is not JSON"):
        study(Choice([Manager(), "plain"], default="plain"))
    with pytest.raises(ValueError, match=r"style: its default \(1, 2\) cannot be recorded: it does not read back"):
        study(Choice([(1, 2), (3, 4)], default=(1, 2)))
    style = Choice(["brief", "long"], default="brief")
    odd = study(style, search=Search(width=1, blocks=[Block(Proposes(Manager()), style)]), record=tmp_path)
    with pytest.raises(ValueError, match=r"style=<.*Manager object .*>, proposed by block 'numbers' \(Proposes\)"):
        odd.optimize(iterations=1)
    log = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    assert [e["params"] for e in log if e["type"] == "candidate"] == [{"style": "brief"}]  # never written as a repr
    assert log[-1]["type"] == "stopped" and "proposed by block 'numbers'" in log[-1]["error"]


# ---- text ------------------------------------------------------------------------------------------------------------
class _Rewrite(Draw):
    def __init__(self, text):
        self.text, self.origin = text, {"method": "append_rule"}

    def suggest_text(self, name, default):
        return self.text


class AppendRule(Method):
    """Stands in for contrastive reflection (no model): appends a rule to the held candidate's text."""

    kinds = ("text",)

    def __init__(self):
        self.seen = []

    def begin(self, block, base, history, structure, study, objective):
        prompt = base.params["system_prompt"]
        rewards = study.projection.rewards(base, study.combine, objective)
        self.seen.append(
            (
                base.number,
                prompt,
                sorted(a.case for a, _, _ in rewards),
                [base.cases[c]["WordCount"] for c in sorted(base.cases)],
            )
        )
        return _Rewrite(prompt + " Use more words.")


POEM_PROMPT = TextValue(default="Write a poem.")


@tunable(system_prompt=POEM_PROMPT)
def create_echo(system_prompt):
    return Agent(model=Says(None), system_prompt=system_prompt, callback_handler=None)


def test_text_is_rewritten_from_the_held_trial():
    rule = AppendRule()
    study = Study(
        agent=create_echo,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(rule, POEM_PROMPT)]),
    )
    study.optimize(iterations=2)
    assert [t.params["system_prompt"] for t in study.candidates] == [
        "Write a poem.",
        "Write a poem. Use more words.",
        "Write a poem. Use more words. Use more words.",
    ]
    assert [t.base for t in study.candidates] == [None, 0, 1]
    # each rewrite starts from the held candidate's own text, cases and per-case scores
    assert rule.seen == [
        (0, "Write a poem.", ["city", "sea"], [3.0, 3.0]),
        (1, "Write a poem. Use more words.", ["city", "sea"], [6.0, 6.0]),
    ]


# ---- structure: a choice between components, each with its own numbers and choices ----------------------------------
class Window:
    def __init__(self, size, truncate):
        self.words = size // 10 + (5 if truncate else 0)


class Summary:
    def __init__(self, ratio, keep):
        self.words = round(ratio * 10) + keep


SIZE, TRUNCATE = Int(10, 80, default=40), Choice([True, False], default=True)
RATIO, KEEP = Float(0.1, 0.8, default=0.3), Int(2, 16, default=4)
MANAGER = OneOf(
    component(Window, size=SIZE, truncate=TRUNCATE),
    component(Summary, ratio=RATIO, keep=KEEP),
    labels=["window", "summary"],
)
PROMPT = TextValue(default="Write.")


@tunable(manager=MANAGER, system_prompt=PROMPT)
def create_managed(manager, system_prompt):
    return Agent(
        model=Says(" ".join(["w"] * manager.words + system_prompt.split())),
        system_prompt=system_prompt,
        callback_handler=None,
    )


def managed(schedule, rule, **kw):
    """The choice, the numbers and the prompt in a block each; ``schedule(b)`` builds the schedule from them
    (``b.structure``, ``b.numbers``, ``b.text``)."""
    b = SimpleNamespace(
        structure=Block(TPE(seed=1), MANAGER),
        numbers=Block(TPE(seed=2), SIZE, TRUNCATE, RATIO, KEEP),
        text=Block(rule, PROMPT),
    )
    return Study(
        agent=create_managed,
        experiment=EXPERIMENT,
        objectives=kw.pop("objectives", Maximize("WordCount")),
        search=Search(width=1, blocks=[b.structure, b.numbers, b.text], schedule=schedule(b)),
        **kw,
    )


def test_the_blocks_follow_the_variable_types():
    study = managed(lambda b: Joint(), AppendRule())
    assert {name: b.keys for name, b in study.blocks.items()} == {
        "structure": ("manager",),
        "numbers": ("manager.window.size", "manager.window.truncate", "manager.summary.ratio", "manager.summary.keep"),
        "text": ("system_prompt",),
    }
    assert (
        type(
            Study(
                search=Search(width=1), agent=create_managed, experiment=EXPERIMENT, objectives=Maximize("WordCount")
            ).schedule
        )
        is Nested
    )


def test_joint_updates_every_block_in_every_trial():
    study = managed(lambda b: Joint(), AppendRule())
    study.optimize(iterations=4)
    assert [t.proposed for t in study.candidates[1:]] == [("structure", "numbers", "text")] * 4
    assert all(t.base == study.incumbent(study.candidates[: t.number]).number for t in study.candidates[1:])


def test_alternate_updates_one_block_at_a_time_holding_the_rest():
    study = managed(lambda b: Alternate(b.structure, b.numbers, b.text), AppendRule())
    study.optimize(iterations=6)
    assert [t.proposed for t in study.candidates[1:]] == [("structure",), ("numbers",), ("text",)] * 2
    for t in study.candidates[1:]:
        held = study.candidates[t.base]
        for key, value in t.params.items():
            block = study.block_of[key]
            if block not in t.proposed and key in held.params and held.params["manager"] == t.params["manager"]:
                assert value == held.params[key], (t.number, key)


def round_groups(study, path):
    """Each round of the outermost Nested: its opening candidate and the candidates measured in it, in order. A round
    whose opening proposal repeated an earlier candidate opened on that candidate (``repeat`` in the record)."""
    log = [json.loads(line) for line in (path / "study.jsonl").read_text().splitlines()]
    opened_on = {e["scope"]: e for e in log if e["type"] == "repeat" and e["role"] == "outer"}
    groups = []
    for r in [e["id"] for e in log if e["type"] == "scope" and e["parent"] == 1]:
        here = [c for c in study.candidates if r in c.path]
        if r in opened_on:
            assert opened_on[r]["proposed"] == ["structure"]
            groups.append((study.candidates[opened_on[r]["candidate"]], here))
        else:
            assert here[0].proposed == ("structure",) and here[0].role == "outer"
            groups.append((here[0], here[1:]))
    return groups


def test_nested_searches_inside_each_structure_and_compares_at_the_inner_best(tmp_path):
    study = managed(
        lambda b: Nested(outer=b.structure, inner=Alternate(b.numbers, b.text), inner_iterations=3),
        AppendRule(),
        record=tmp_path,
    )
    study.optimize(iterations=8)
    groups = round_groups(study, tmp_path)
    assert [len(inner) for _, inner in groups] == [3, 3, 0]  # the budget ends at the third round's opening
    for opener, inner in groups:
        assert [t.proposed for t in inner] == [("numbers",), ("text",), ("numbers",)][: len(inner)]
        assert len({t.params["manager"] for t in [opener, *inner]}) == 1  # the structure is fixed inside its round
    # the second round chose the first round's structure again: it opened on that structure's best, not measured again
    first, second = groups[0], groups[1]
    assert second[0] is max([first[0], *first[1]], key=lambda t: t.metrics["WordCount"])
    assert len({tuple(sorted(t.params.items())) for t in study.candidates}) == len(study.candidates)
    # the structure method was told each round's best result (the candidate it opened on included), not its first
    outer = study.method("structure")._studies[()]
    told = [t.values[0] for t in outer.trials if t.state.name == "COMPLETE"]
    # (as decided: the model's values, the measured ones for a deterministic metric)
    expected = [max(t.metrics["WordCount"] for t in [opener, *inner]) for opener, inner in groups]
    assert told[1:] == pytest.approx(expected, rel=1e-3)
    # the numbers method keeps one Optuna study per structure, with every candidate in it as evidence
    numbers = study.method("numbers")._studies
    assert set(numbers) <= {(("manager", "window"),), (("manager", "summary"),)}


def test_nested_nests_and_each_scope_is_told_only_its_own_best(tmp_path):
    study = managed(
        lambda b: Nested(
            outer=b.structure, inner=Nested(outer=b.numbers, inner=Joint(), inner_iterations=2), inner_iterations=6
        ),
        AppendRule(),
        record=tmp_path,
    )
    study.optimize(iterations=14)  # the baseline, then rounds of 1 + 6 (each: 2 subrounds of 1 + 2)
    groups = round_groups(study, tmp_path)
    for opener, inner in groups[:2]:
        assert [t.proposed for t in inner] == [("numbers",), ("text",), ("text",), ("numbers",), ("text",), ("text",)]
        assert len({t.params["manager"] for t in [opener, *inner]}) == 1
    outer = study.method("structure")._studies[()]
    told = [t.values[0] for t in outer.trials if t.state.name == "COMPLETE"]
    bests = [max(t.metrics["WordCount"] for t in [opener, *inner]) for opener, inner in groups]
    assert told == pytest.approx([study.candidates[0].metrics["WordCount"], *bests], rel=1e-3)  # as decided


def test_independent_objectives_take_turns_with_their_own_methods_and_incumbents():
    study = managed(lambda b: Joint(), AppendRule(), objectives=Independent(Maximize("WordCount"), Minimize("tokens")))
    study.optimize(iterations=4)
    assert [t.objective for t in study.candidates[1:]] == [0, 1, 0, 1]
    assert len(study.methods) == 2 and study.method("numbers", 0) is not study.method("numbers", 1)
    for t in study.candidates[1:]:
        assert t.base == study.incumbent(study.candidates[: t.number], t.objective).number
    words = study.method("numbers", 0)._studies
    assert all(s.directions[0].name == "MAXIMIZE" for s in words.values())


def test_an_ineligible_trial_is_told_as_failed_and_never_held():
    stanzas = Int(1, 6, default=2)

    @tunable(stanzas=stanzas)
    def sometimes_broken(stanzas):
        from .helpers import Says as Scripted

        model = Scripted(" ".join(["word"] * 10 * stanzas), fail=ValueError("bug") if stanzas == 5 else None)
        return Agent(model=model, callback_handler=None)

    study = Study(
        agent=sometimes_broken,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        agent_error_score=None,
        search=grid({"stanzas": [5, 1]}, stanzas),
    )
    study.optimize(iterations=2)
    broken = next(t for t in study.candidates if t.params["stanzas"] == 5)
    assert broken.status == "partial" and not broken.eligible  # its agent failures are measured, but no score counts
    assert all(a.status.value == "agent_failure" and "bug" in a.error for a in broken.trials)
    optuna_trials = study.method("numbers")._studies[()].trials
    assert any(t.state.name == "FAIL" for t in optuna_trials)
    assert study.incumbent().params["stanzas"] != 5


@needs_bayesbench
def test_the_record_holds_the_study_and_every_trial(tmp_path):
    import json

    study = Study(
        agent=create_poet,
        experiment=EXPERIMENT,
        objectives=Maximize("WordCount"),
        constraints=[AtMost("output_tokens", 80)],
        search=grid({"stanzas": [1, 4]}, STANZAS),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=2)
    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    assert [e["type"] for e in events] == [
        "study",
        "run",
        "candidate",  # the baseline: one candidate is too few to model
        "scope",
        "model",  # each fit, before the candidate it includes is written
        "candidate",
        "model",
        "candidate",
        "scope_end",
        "remeasured",  # the finalists, measured again
        "remeasured",
        "remeasured",
        "model",
        "selected",
        "finished",
    ]
    selected = next(e for e in events if e["type"] == "selected")
    assert selected["by"] == "model" and selected["best"] == [c.number for c in study.best_candidates]
    remeasured = [e for e in events if e["type"] == "remeasured"]
    assert all(len(e["trials"]) == study.repetitions * len(EXPERIMENT.cases) for e in remeasured)
    head = events[0]
    assert head["objectives"] == [
        {"kind": "Maximize", "metric": "WordCount", "weight": 1.0, "scale": None, "tolerance": 0.0}
    ]
    assert head["constraints"] == [
        {"kind": "AtMost", "metric": "output_tokens", "limit": 80, "compare": "absolute", "name": "output_tokens <= 80"}
    ]
    assert head["blocks"] == {"numbers": {"method": {"kind": "Optuna"}, "keys": ["stanzas"], "structural": False}}
    assert head["schedule"] == {"node": "", "kind": "Joint", "children": {}}
    assert head["cases"] == ["sea", "city"] and head["space"][0]["key"] == "stanzas"
    candidate = events[5]  # (events[4] is the model fitted to it)
    assert candidate["estimates"]["WordCount"]["n"] == 2 and len(candidate["trials"]) == 2
    assert (candidate["scope"], candidate["path"], candidate["role"], candidate["proposed"]) == (
        1,
        [1],
        "step",
        ["numbers"],
    )
    digest = candidate["trials"][0]["transcript"]["hash"]
    assert json.loads((tmp_path / "transcripts" / f"{digest}.json").read_text())[0]["role"] == "user"
    assert events[-1]["best"] == [study.best_candidate.number] == events[-2]["best"]  # finished, selected


def test_several_objectives_are_given_wrapped_in_how_they_combine():
    with pytest.raises(TypeError, match=r"objectives=Pareto\(\.\.\.\)"):
        Study(
            search=Search(width=1),
            agent=create_managed,
            experiment=EXPERIMENT,
            objectives=[Maximize("WordCount"), Minimize("tokens")],
        )
    study = Study(
        search=Search(width=1),
        agent=create_managed,
        experiment=EXPERIMENT,
        objectives=Independent(Maximize("WordCount"), Minimize("tokens")),
        log=None,
    )
    assert len(study.methods) == 2 and study.projection.range("WordCount") == (0.0, None)  # the evaluator's own


def test_iterations_do_not_count_the_baseline():
    study = Study(
        search=Search(width=1), agent=create_managed, experiment=EXPERIMENT, objectives=Maximize("WordCount"), log=None
    )
    study.optimize(iterations=3)
    assert len(study.candidates) == 4 and study.candidates[0].role == "baseline"
    study.optimize(iterations=2)  # a later call measures no baseline
    assert len(study.candidates) == 6
