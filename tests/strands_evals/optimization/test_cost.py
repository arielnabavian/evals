"""What a study spends, in tokens: every attempt of every trial, every method call (failed ones too), by role and phase,
in the record as it runs and in the result."""

import asyncio
import json
from pathlib import Path

import pytest
from strands import Agent
from strands.types.exceptions import ModelThrottledException

from strands_evals import Case, Experiment
from strands_evals import experiment as experiment_module
from strands_evals.optimization import (
    Block,
    Choice,
    Maximize,
    Method,
    Reflection,
    Schedule,
    Search,
    SkillLibrary,
    Split,
    Study,
    TextValue,
    TrialInvalid,
    tunable,
)
from strands_evals.optimization import trials as trials_module
from strands_evals.optimization.cost import Cost, Ledger
from strands_evals.optimization.methods import SkillCuration, _ChoiceDraw
from strands_evals.optimization.trials import SPENT, TrialStatus, run_trials

from .helpers import Says, WordCount, experiment, needs_bayesbench

CALL = 50 + 6 + 10  # one model call of Says("a short poem"): 50 in, 2 per word out, 10 read from the cache


def events(path: Path, name: str = "study.jsonl") -> list[dict]:
    return [json.loads(line) for line in (path / name).read_text().splitlines()]


# ---- attempts --------------------------------------------------------------------------------------------------------
def test_every_attempt_of_a_trial_is_kept_and_counted_and_decisions_see_only_the_last(monkeypatch, tmp_path):
    monkeypatch.setattr(trials_module, "RETRY_WAIT", 0.0)
    monkeypatch.setattr(experiment_module, "_INITIAL_RETRY_DELAY", 0)
    ran = {"n": 0}

    def flaky(agent, case):  # the agent answers each time; then its simulated user fails twice, then it is throttled
        ran["n"] += 1
        reply = str(agent(case.input))
        if ran["n"] in (1, 2):
            raise TrialInvalid("the simulated user did not answer")
        if ran["n"] == 3:
            raise ModelThrottledException("slow down")  # Strands Evals runs the trial again itself
        return reply

    @tunable()
    def poet():
        return Agent(model=Says("a short poem"), callback_handler=None)

    exp = experiment(WordCount(), cases=("sea",))
    (t,) = asyncio.run(
        run_trials(poet, poet.default_values, exp, repetitions=1, max_concurrency=1, task=flaky, retries=2)
    )
    assert ran["n"] == 4 and t.status == TrialStatus.COMPLETED and t.metrics["WordCount"] == 3
    assert [a.status for a in t.attempts] == [TrialStatus.INFRASTRUCTURE_INVALID] * 3
    assert "did not answer" in t.attempts[0].error and "slow down" in t.attempts[2].error
    assert all(a.usage["tokens"] == CALL and a.usage["model_calls"] == 1 for a in t.attempts)
    assert set(t.attempts[0].usage) == set(SPENT) and t.metrics["tokens"] == CALL  # the trial is its last attempt

    ledger = Ledger()
    ledger.trials("search", [t])
    cell = ledger.cost().cell("agent", "search")
    assert (cell["tokens"], cell["model_calls"], cell["trials"], cell["attempts"]) == (4 * CALL, 4, 1, 4)
    assert cell["cache_read_tokens"] == 4 * 10 and cell["input_tokens"] == 4 * 50

    ran["n"] = 0
    study = Study(agent=poet, experiment=exp, objectives=Maximize("WordCount"), task=flaky, record=tmp_path, log=None)
    result = study.optimize(iterations=0)
    baseline = study.baseline
    assert baseline.status == "complete" and baseline.metrics["WordCount"] == 3  # estimates: the last attempt alone
    (trial,) = [e for e in events(tmp_path) if e["type"] == "candidate"][0]["trials"]
    assert [a["status"] for a in trial["attempts"]] == ["infrastructure_invalid"] * 3
    assert trial["attempts"][0]["usage"]["tokens"] == CALL
    assert result.cost.cell("agent", "search")["tokens"] == 4 * CALL


# ---- the methods' own calls ------------------------------------------------------------------------------------------
class Rewriter:
    """Stands in for the reflection optimizer: each call spends what a rewriter's model calls would, and the calls in
    ``fail`` fail part-way (as when the model service fails mid-session), the held text kept."""

    def __init__(self, values, calls, fail):
        self.values, self.calls, self.fail = values, calls, fail
        self.last_metrics, self.last_wall_clock_s = {}, 0.0

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        self.calls.append(len(self.calls) + 1)
        k = len(self.calls)
        self.last_wall_clock_s = 2.0
        if k in self.fail:
            self.last_metrics = {"cycle_count": 2, "input_tokens": 900, "output_tokens": 40, "cache_read_tokens": 3000}
            raise RuntimeError("Error during contrastive reflection: internalServerException")
        self.last_metrics = {"cycle_count": 1, "input_tokens": 100, "output_tokens": 10, "cache_write_tokens": 5}
        self.values.update_params({"system_prompt": self.values.values["system_prompt"] + " More."})


class Rewrites(Reflection):
    def __init__(self, fail=()):
        super().__init__()
        self.calls, self.fail = [], set(fail)

    def optimizer(self, values, variables, evidence=None):
        return Rewriter(values, self.calls, self.fail)


PROMPT = TextValue(default="Write a poem.")


@tunable(system_prompt=PROMPT)
def echo(system_prompt):
    return Agent(model=Says(), system_prompt=system_prompt, callback_handler=None)  # it says its system prompt


def test_a_failed_rewrite_counts_what_it_spent(tmp_path):
    rewrites = Rewrites(fail={1})
    study = Study(
        agent=echo,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(rewrites, PROMPT)]),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=1)
    # the first rewrite failed and kept the held text (the baseline's: a repeat, so proposed again); the second wrote
    failed = next(e for e in events(tmp_path) if e["type"] == "repeat")["origin"]["text"]
    assert failed["status"] == "failed" and failed["seconds"] == 2.0
    assert failed["usage"] == {
        "input_tokens": 900.0,
        "output_tokens": 40.0,
        "cache_read_tokens": 3000.0,
        "cache_write_tokens": 0.0,
        "tokens": 3940.0,
        "model_calls": 2.0,
    }
    cell = result.cost.cell("method:text", "search")
    assert (cell["calls"], cell["failed"]) == (2, 1)
    assert (cell["tokens"], cell["model_calls"], cell["seconds"]) == (3940 + 115, 3, 4.0)


class Curator:
    """Stands in for the skill-library optimizer: one per skill library; spends 110 tokens in 2 model calls, writes a
    new skill, or fails part-way when told to."""

    def __init__(self, formula, output_folder, fail):
        self.formula, self.output_folder, self.fail = formula, output_folder, fail
        self.last_metrics, self.last_wall_clock_s, self._last_decisions = {}, 0.0, {}

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        from strands_evals.optimization._reflection import collect_decisions

        self.last_metrics, self.last_wall_clock_s = {"cycle_count": 2, "input_tokens": 100, "output_tokens": 10}, 3.0
        if self.fail:
            raise RuntimeError("the curator's model service failed")
        skill = Path(self.output_folder) / "create" / "checkout"
        skill.mkdir(parents=True, exist_ok=True)
        (skill / "SKILL.md").write_text("---\nname: checkout\ndescription: Use when buying.\n---\n# checkout\n")
        self._last_decisions = collect_decisions(self.output_folder)
        self.formula.update_params({"decisions_dir": self.output_folder})


class Curate(SkillCuration):
    def __init__(self, fail_second=False):
        super().__init__()
        self.made, self.fail_second = 0, fail_second

    def optimizer(self, formula, output_folder):
        self.made += 1
        return Curator(formula, output_folder, fail=self.fail_second and self.made % 2 == 0)


@pytest.mark.parametrize("fail_second", [False, True])
def test_skill_curation_counts_every_librarys_curation(tmp_path, fail_second):
    lib = tmp_path / "lib" / "web-search"
    lib.mkdir(parents=True)
    (lib / "SKILL.md").write_text("---\nname: web-search\ndescription: Use when searching.\n---\n# web-search\n")
    first, second = SkillLibrary(default=str(lib.parent)), SkillLibrary()

    @tunable(first=first, second=second)
    def create(first, second):
        return Agent(model=Says("a poem"), callback_handler=None)

    study = Study(
        agent=create,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Curate(fail_second), first, second)]),
        record=tmp_path / "record",
        log=None,
    )
    result = study.optimize(iterations=1)
    origin = next(e for e in events(tmp_path / "record") if e["type"] in ("candidate", "repeat") and e.get("origin"))
    origin = origin["origin"]["skills"]
    assert origin["status"] == ("failed" if fail_second else "rewritten")
    assert origin["usage"]["tokens"] == 220.0 and origin["usage"]["model_calls"] == 4.0 and origin["seconds"] == 6.0
    calls = result.cost.cell("method:skills", "search")["calls"]  # a failed curation keeps the library: tried again
    assert calls == (1 + study.search.redraws if fail_second else 1)
    assert result.cost.tokens("method:skills") == 220.0 * calls


# ---- the breakdown ---------------------------------------------------------------------------------------------------
def sums(cost: Cost) -> None:
    """Every total is the sum of its parts."""
    for k in SPENT:
        by_phase = sum(cell[k] for cell in cost.by_phase.values())
        by_role = sum(cell[k] for cell in cost.by_role.values())
        cells = sum(cell.get(k, 0) for phases in cost.roles.values() for cell in phases.values())
        assert cost.total[k] == pytest.approx(by_phase) == pytest.approx(by_role) == pytest.approx(cells)


def tokens(trials: list[dict]) -> float:
    return sum(t["metrics"]["tokens"] + sum(a["usage"]["tokens"] for a in t.get("attempts", [])) for t in trials)


@needs_bayesbench
def test_the_breakdown_is_every_role_by_phase_and_its_totals_are_the_sum_of_its_parts(tmp_path):
    study = Study(
        agent=echo,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Rewrites(), PROMPT)]),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=2)
    cost = result.cost
    sums(cost)
    log = events(tmp_path)
    candidates = [e for e in log if e["type"] == "candidate"]
    remeasured = [e for e in log if e["type"] == "remeasured"]
    assert remeasured  # the finalists were measured again on the search cases
    assert cost.tokens("agent", "search") == tokens([t for e in candidates for t in e["trials"]])
    assert cost.tokens("agent", "remeasurement") == tokens([t for e in remeasured for t in e["trials"]])
    origins = [e["origin"]["text"] for e in log if e["type"] in ("candidate", "repeat") and "text" in e["origin"]]
    assert cost.tokens("method:text") == sum(o["usage"]["tokens"] for o in origins) == 2 * 115
    assert cost.cell("agent", "search")["trials"] == 3 * 2  # the baseline and two candidates, on two cases
    assert set(cost.to_dict()["roles"]) == {"agent", "method:text", "probes"} and cost.roles["probes"] == {}
    # the record states the cost as it stood at each phase's end, and the result the same as the record's last
    assert [e["type"] for e in log if e.get("cost")] == ["selected", "finished"]
    assert log[-1]["cost"] == result.to_dict()["cost"] == json.loads((tmp_path / "result.json").read_text())["cost"]
    assert Cost.from_dict(log[-1]["cost"]).to_dict() == log[-1]["cost"]
    head = log[0]
    assert head["roles"]["agent"] == {"model": "says"}  # read from the agent built at the defaults
    assert head["roles"]["method:text"] == {
        "block": "text",
        "method": "Rewrites",
        "model": "global.anthropic.claude-sonnet-5-5",  # the rewriter's default
    }
    assert head["max_concurrency"] == 4 and head["width"] == 1 and set(head["code"]) >= {"version", "commit"}


STANZAS = Choice([1, 2, 3, 4], default=2)


@tunable(stanzas=STANZAS)
def poet(stanzas):
    return Agent(model=Says(" ".join(["word"] * 10 * stanzas)), callback_handler=None)


class Grid(Method):
    """Proposes the given numbers of stanzas in turn (no model call of its own)."""

    kinds = ("choice",)

    def __init__(self, values):
        self.values = list(values)

    def begin(self, block, base, history, structure, study, objective):
        return _ChoiceDraw({"stanzas": self.values.pop(0)}, {"method": "grid"})


def test_a_rotation_gives_each_runs_cost_and_their_total(tmp_path):
    pool = [Case(name=f"c{k}", input="write") for k in range(8)]
    study = Study(
        agent=poet,
        experiment=Experiment(cases=pool, evaluators=[WordCount(name="words")]),
        objectives=Maximize("words"),
        search=Search(width=1, blocks=[Block(Grid([4, 4]), STANZAS)]),
        split=Split(train=0.5, validation=0.25, test=0.25, rotate=True),
        record=tmp_path,
        log=None,
    )
    result = study.optimize(iterations=1)
    runs = [r.cost for r in result.runs]
    assert len(runs) == 4 and result.cost.to_dict() == sum(runs[1:], runs[0]).to_dict()
    sums(result.cost)
    for k in SPENT:
        assert result.cost.total[k] == pytest.approx(sum(c.total[k] for c in runs))
    assert list(result.cost.by_phase) == ["search", "validation", "test"]
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["cost"] == result.cost.to_dict()
    assert [r["cost"] for r in saved["runs"]] == [c.to_dict() for c in runs]
    for i, c in enumerate(runs, 1):
        assert events(tmp_path / f"run-{i}")[-1]["cost"] == c.to_dict()  # each run's record ends with its own
        end = events(tmp_path / f"run-{i}", "test.jsonl")[-1]
        assert end["type"] == "test_end" and end["cost"]["by_phase"]["test"] == c.to_dict()["by_phase"]["test"]


class Stops(Schedule):
    """Two steps, then the machine running the study goes away."""

    def run(self, scope):
        scope.step(list(scope.blocks))
        scope.step(list(scope.blocks))
        raise RuntimeError("reclaimed")


def test_a_stopped_run_states_what_it_spent(tmp_path):
    study = Study(
        agent=echo,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Rewrites(), PROMPT)], schedule=Stops()),
        record=tmp_path,
        log=None,
    )
    with pytest.raises(RuntimeError, match="reclaimed"):
        study.optimize(iterations=5)
    log = events(tmp_path)
    stopped = log[-1]
    assert stopped["type"] == "stopped" and stopped["cost"] == study.cost.to_dict()
    trials = [t for e in log if e["type"] == "candidate" for t in e["trials"]]
    assert stopped["cost"]["roles"]["agent"]["search"]["tokens"] == tokens(trials) > 0
    assert stopped["cost"]["by_role"]["method:text"]["tokens"] == 2 * 115


def test_costs_add_and_read_back():
    a = Ledger()
    a.trials("search", [])
    a.call("text", {"usage": {"input_tokens": 10, "output_tokens": 2}, "seconds": 1.5})
    a.call("text", {"method": "TPE"})  # no usage: no model call of its own
    b = Ledger()
    b.call("text", {"usage": {"input_tokens": 1}, "status": "failed"})
    total = a.cost() + b.cost()
    cell = total.cell("method:text", "search")
    assert (cell["input_tokens"], cell["tokens"], cell["calls"], cell["failed"], cell["seconds"]) == (11, 13, 2, 1, 1.5)
    assert Cost.from_dict(total.to_dict()) == Cost.from_dict(json.loads(json.dumps(total.to_dict())))
    assert total.describe() == "13 tokens: agent 0 (search 0); method:text 13"


def test_the_header_gives_the_package_version_only_when_it_is_the_code_that_ran(monkeypatch, tmp_path):
    from strands_evals.optimization import record

    here = Path(record.__file__).resolve()

    class Dist:
        version = "9.9.9"

        def __init__(self, root, url=None):
            self.root, self.url = root, url

        def locate_file(self, name):
            return self.root / name

        def read_text(self, name):
            return json.dumps({"url": self.url}) if self.url else None

    def installed(dist):
        monkeypatch.setattr(record.metadata, "distribution", lambda name: dist)

    installed(Dist(here.parents[2]))  # installed: its files are the code running
    assert record._version(here) == "9.9.9"
    installed(Dist(tmp_path, f"file://{here.parents[3]}"))  # installed editable from this checkout
    assert record._version(here) == "9.9.9"
    installed(Dist(tmp_path, f"file://{tmp_path}"))  # another checkout is installed; this one runs from the path
    assert record._version(here) is None

    def missing(name):
        raise record.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(record.metadata, "distribution", missing)
    assert record._version(here) is None
