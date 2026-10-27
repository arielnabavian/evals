"""Language-valued variables: tool descriptions as text, skill libraries, and reflection over several of them."""

import json
import os
import shutil
from pathlib import Path

import pytest
from strands import Agent, tool

from strands_evals.optimization import (
    Block,
    Maximize,
    Reflection,
    Search,
    SkillLibrary,
    Study,
    TextValue,
    Values,
    tool_descriptions,
    tunable,
)
from strands_evals.optimization.methods import SkillCuration
from strands_evals.optimization.space.skills import check_library, resolve, store

from .helpers import Says, experiment


@tool
def search(query: str) -> str:
    """Search the catalog."""
    return query


@tool
def click(target: str) -> str:
    """Click an element."""
    return target


# ---- tool descriptions -----------------------------------------------------------------------------------------------
def test_each_tool_description_is_a_text_variable_and_the_tools_are_never_changed():
    @tunable(tools=tool_descriptions(search, click))
    def build(tools):
        return tools

    params = {p.key: p for p in build.params()}
    assert set(params) == {"tools.search", "tools.click"}
    assert (params["tools.search"].type, params["tools.search"].default, params["tools.search"].max_chars) == (
        "text",
        "Search the catalog.",
        800,
    )
    values = {k: p.default for k, p in params.items()} | {"tools.search": "Search the catalog by keyword."}
    built = build(**build.bind(Values(values)))
    assert [t.tool_spec["description"] for t in built] == ["Search the catalog by keyword.", "Click an element."]
    assert search.tool_spec["description"] == "Search the catalog."
    with pytest.raises(ValueError, match="over max_chars=800"):
        build.validate(Values(values | {"tools.click": "x" * 801}))


# ---- reflection over several texts ----------------------------------------------------------------------------------
class Scripted:
    """Stands in for the reflection optimizer: records what it was given (and the version files while it runs), then
    submits the edits it was told to (the next of a list, when given one)."""

    seen: list = []

    def __init__(self, values, variables, evidence, edits):
        self.values, self.variables, self.evidence, self.edits = values, variables, evidence, edits
        self.last_metrics, self.last_wall_clock_s = {"input_tokens": 100, "output_tokens": 10}, 1.0

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        folder = self.evidence["versions_folder"]
        files = {name: (Path(folder) / name).read_text() for name in sorted(os.listdir(folder))}
        Scripted.seen.append({"variables": self.variables, "evidence": self.evidence, "files": files})
        self.values.update_params(self.edits.pop(0) if isinstance(self.edits, list) else self.edits)


class Edits(Reflection):
    def __init__(self, edits):
        super().__init__()
        self.edits = edits

    def optimizer(self, values, variables, evidence=None):
        return Scripted(values, variables, evidence, self.edits)


HELP = TextValue(default="Help.", about="the system prompt")
TOOLS = tool_descriptions(search, click)


@tunable(system_prompt=HELP, tools=TOOLS)
def create(system_prompt, tools):
    return Agent(model=Says("ok"), system_prompt=system_prompt, tools=tools, callback_handler=None)


def test_reflection_is_told_what_each_text_is_and_changes_only_what_it_submits():
    edits = {"tools.search": "Search the catalog by keyword; returns ids.", "tools.click": "y" * 900}
    study = Study(
        agent=create,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Edits(edits), HELP, TOOLS)]),
        log=None,
    )
    study.optimize(iterations=1)
    candidate = study.candidates[1]
    told = {v["name"]: v for v in Scripted.seen[-1]["variables"]}
    assert told["system_prompt"]["about"] == "the system prompt" and told["system_prompt"]["max_chars"] is None
    assert told["tools.search"]["max_chars"] == 800 and "search tool" in told["tools.search"]["about"]
    assert candidate.params["tools.search"] == edits["tools.search"]
    assert candidate.params["tools.click"] == "Click an element."  # over its limit: dropped, the held text kept
    assert candidate.params["system_prompt"] == "Help."  # not submitted: unchanged
    origin = candidate.origin["text"]
    assert origin["changed"] == ["tools.search"] and "over max_chars=800" in origin["dropped"]["tools.click"]


class Shortened(Scripted):
    """Stands in for an optimizer that sent an over-limit submission back once (``revisions``) and got it shortened."""

    def step(self):
        super().step()
        self.revisions = [{"tools.click": 900}]


class ShortenedEdits(Edits):
    def optimizer(self, values, variables, evidence=None):
        return Shortened(values, variables, evidence, self.edits)


def test_reflection_sends_a_text_over_its_limit_back_and_records_that_it_did():
    study = Study(
        agent=create,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(ShortenedEdits({"tools.click": "Click it."}), HELP, TOOLS)]),
        log=None,
    )
    study.optimize(iterations=1)
    origin = study.candidates[1].origin["text"]
    assert origin["revised"] == [{"tools.click": 900}] and "dropped" not in origin
    assert study.candidates[1].params["tools.click"] == "Click it."
    method = Reflection()
    assert method.over_limit_retries == 2 and method.describe()["over_limit_retries"] == 2
    variables = [{"name": "a", "about": "", "max_chars": 50}, {"name": "b", "about": "", "max_chars": None}]
    optimizer = method.optimizer(Scripted.__new__(Scripted), variables)
    assert optimizer.limits == {"a": 50} and optimizer.over_limit_retries == 2
    assert Reflection(over_limit_retries=0).optimizer(Scripted.__new__(Scripted), []).over_limit_retries == 0
    with pytest.raises(ValueError, match="over_limit_retries"):
        Reflection(over_limit_retries=-1)


class Digested(Scripted):
    """Stands in for an optimizer that wrote a digest of the trials it was given."""

    def step(self):
        super().step()
        self.digest_text = "# Digest\n\n## Every case with a conversation that did not succeed (2)\n- a\n- b\n"


class DigestedEdits(Edits):
    def optimizer(self, values, variables, evidence=None):
        return Digested(values, variables, evidence, self.edits)


def test_the_digest_the_rewriter_read_is_kept_with_the_study_and_named_in_the_record(tmp_path):
    study = Study(
        agent=create,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(DigestedEdits({"tools.click": "Click it."}), HELP, TOOLS)]),
        record=tmp_path,
        log=None,
    )
    study.optimize(iterations=1)
    digest = study.candidates[1].origin["text"]["digest"]
    assert digest["failing_cases"] == 2 and digest["file"].startswith("digests/") and digest["chars"] > 0
    assert (tmp_path / "artifacts" / digest["file"]).read_text().startswith("# Digest")
    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    recorded = next(e for e in events if e["type"] == "candidate" and e["number"] == 1)
    assert recorded["origin"]["text"]["digest"] == digest


@tunable(system_prompt=HELP)
def echo(system_prompt):
    return Agent(model=Says(), system_prompt=system_prompt, callback_handler=None)  # it says its system prompt


def test_reflection_is_given_every_version_ranked_for_the_goal():
    edits = [{"system_prompt": "one two three"}, {"system_prompt": "one"}, {"system_prompt": "one two"}]
    study = Study(
        agent=echo,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Edits(edits), HELP)]),
        log=None,
    )
    study.optimize(iterations=3)
    last = Scripted.seen[-1]
    versions = last["evidence"]["versions"]
    assert [(v["rank"], v["number"]) for v in versions][:1] == [(1, 1)]  # three words: the best so far
    assert versions[0]["current"] and versions[0]["from"] == 0 and versions[0]["changed"] == ["system_prompt"]
    assert sorted(v["number"] for v in versions) == [0, 1, 2] and [v["rank"] for v in versions] == [1, 2, 3]
    assert last["evidence"]["goal_metrics"] == ["WordCount"] and last["evidence"]["label_ranked"]
    assert "rank-001-version-1.md" in last["files"]
    assert "## system_prompt\n\none two three" in last["files"]["rank-001-version-1.md"]
    assert study.candidates[3].origin["text"]["versions"] == 3


# ---- skill libraries ------------------------------------------------------------------------------------------------
def skill(root: Path, name: str, description: str | None = "Use when testing.") -> None:
    (root / name).mkdir(parents=True)
    front = f"---\nname: {name}\ndescription: {description}\n---\n" if description else ""
    (root / name / "SKILL.md").write_text(front + f"# {name}\n")


def test_a_skill_library_is_checked_and_stored_once_by_content(tmp_path):
    lib = tmp_path / "lib"
    skill(lib, "web-search")
    assert check_library(str(lib)) is None
    skill(lib, "broken", description=None)
    assert "broken" in check_library(str(lib))
    a = store({"web-search": str(lib / "web-search")}, tmp_path / "store")
    b = store({"web-search": str(lib / "web-search")}, tmp_path / "store")
    assert a == b and a.startswith("skills/")  # relative to the store: the record can be moved
    assert os.path.isfile(tmp_path / "store" / a / "web-search" / "SKILL.md")
    assert resolve(a, tmp_path / "store") == str((tmp_path / "store" / a).resolve())


class Curator:
    """Stands in for the skill-library optimizer: writes a decision tree, then applies it as the real one does."""

    found: list = []  # what each curator found in its folder before it wrote anything

    def __init__(self, formula, output_folder):
        self.formula, self.output_folder = formula, output_folder
        self.last_metrics, self.last_wall_clock_s, self._last_decisions = {}, 1.0, {}
        Curator.found.append(sorted(os.listdir(output_folder)) if os.path.isdir(output_folder) else [])

    def add_rollouts(self, rollouts):
        pass

    def add_rewards(self, rewards):
        pass

    def step(self):
        from strands_evals.optimization._reflection import collect_decisions

        skill(Path(self.output_folder) / "create", "checkout", "Use when buying.")
        self._last_decisions = collect_decisions(self.output_folder)
        self.formula.update_params({"decisions_dir": self.output_folder})


class Curate(SkillCuration):
    def optimizer(self, formula, output_folder):
        return Curator(formula, output_folder)


def skills_study(tmp_path, width=1, record="record"):
    lib = tmp_path / "lib"
    skill(lib, "web-search")
    library = SkillLibrary(default=str(lib))
    received = []

    @tunable(skills=library)
    def create_with_skills(skills):
        received.append(skills)
        assert os.path.isabs(skills)  # the factory receives a path it can use, wherever the study runs
        return Agent(model=Says(" ".join(sorted(os.listdir(skills)))), callback_handler=None)

    study = Study(
        agent=create_with_skills,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=width, blocks=[Block(Curate(), library)]),
        record=tmp_path / record,
        log=None,
    )
    return study, received


def test_skill_curation_proposes_a_new_library_version(tmp_path):
    study, received = skills_study(tmp_path)
    assert study.blocks["skills"].keys == ("skills",)
    study.optimize(iterations=1)
    new = study.candidates[1].params["skills"]
    assert new.startswith("skills/") and study.artifact_path(new) == tmp_path / "record" / "artifacts" / new
    assert sorted(os.listdir(study.artifact_path(new))) == ["checkout", "web-search"]
    assert study.candidates[1].origin["skills"]["skills"]["decisions"]["create"] == ["checkout"]
    assert study.candidates[1].origin["skills"]["skills"]["folder"] == "skill-decisions/candidate-001-skills"
    assert study.candidates[1].metrics["WordCount"] == 2  # the agent was built with the new library
    assert received and all(r.startswith(str((tmp_path / "record" / "artifacts").resolve())) for r in received)


def test_the_default_library_is_stored_too_and_a_moved_record_still_finds_every_version(tmp_path):
    study, _ = skills_study(tmp_path)
    study.optimize(iterations=1)
    baseline = study.baseline.params["skills"]
    assert baseline.startswith("skills/") and sorted(os.listdir(study.artifact_path(baseline))) == ["web-search"]
    head = json.loads((tmp_path / "record" / "study.jsonl").read_text().splitlines()[0])
    assert head["space"][0]["default"] == baseline  # the baseline holds the stored version, as every candidate does
    assert head["stored_defaults"] == {"skills": {"declared": str(tmp_path / "lib"), "value": baseline}}

    shutil.move(tmp_path / "record", tmp_path / "moved")
    shutil.rmtree(tmp_path / "lib")  # nothing outside the record is needed
    values = [
        json.loads(line)["params"]["skills"]
        for line in (tmp_path / "moved" / "study.jsonl").read_text().splitlines()
        if json.loads(line)["type"] == "candidate"
    ]
    moved, _ = skills_study(tmp_path / "other", record="../moved")
    for value in values:
        assert check_library(str(moved.artifact_path(value))) is None
        kwargs = moved.agent.bind(Values({"skills": value}), artifacts=moved.artifacts)  # as a factory receives it
        assert os.path.isdir(kwargs["__strands_evals_values__"]["skills"])


def test_a_record_with_absolute_library_paths_still_reads(tmp_path):
    lib = tmp_path / "lib"
    skill(lib, "web-search")
    study, _ = skills_study(tmp_path / "s")
    assert study.artifact_path(str(lib)) == lib and study.artifact_path(None) is None
    assert resolve(str(lib), study.artifacts) == str(lib)
    study.agent.validate(Values({"skills": str(lib)}), artifacts=study.artifacts)  # as records before 2026-10-24 hold
    with pytest.raises(ValueError, match="relative to its artifact store"):
        study.agent.validate(Values({"skills": "skills/0123456789abcdef"}))  # a stored value, without its store


def test_each_curation_writes_its_decisions_in_a_folder_of_its_own(tmp_path):
    Curator.found.clear()
    study, _ = skills_study(tmp_path, width=2)  # two curations from the same incumbent, in the same step
    study.optimize(iterations=1)
    assert Curator.found == [[], []]  # neither saw another's decisions
    folders = sorted((tmp_path / "record" / "artifacts" / "skill-decisions").iterdir())
    assert [f.name for f in folders] == ["candidate-001-skills", "candidate-001-skills-2"]
