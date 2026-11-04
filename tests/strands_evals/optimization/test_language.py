"""Language-valued variables: tool descriptions as text, skill libraries, and reflection over several of them."""

import json
import os
import re
import shlex
import shutil
from pathlib import Path

import pytest
from strands import Agent, tool
from strands.models.model import Model

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
from strands_evals.optimization._reflection import agentic
from strands_evals.optimization.methods import SkillCuration, _no_version_header, _write_versions
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
        folder = Path(self.evidence["versions_folder"])
        files = {str(p.relative_to(folder)): p.read_text() for p in sorted(folder.rglob("*")) if p.is_file()}
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
    assert last["files"]["rank-001-version-1/system_prompt.txt"] == "one two three"  # the text alone
    record = last["files"]["rank-001-version-1/version.md"]
    assert record.startswith("# Version 1\nrank: 1\nbuilt from: version 0\n") and "one two three" not in record
    assert "- system_prompt: `system_prompt.txt` (13 characters)" in record
    assert [t for f, t in last["files"].items() if f.endswith("-version-0/system_prompt.txt")] == ["Help."]
    assert study.candidates[3].origin["text"]["versions"] == 3


# ---- the rewriter's own files never reach a text ---------------------------------------------------------------------
HEADER = re.compile(r"^# Version \d+", re.M)  # the first line of a version's header in the versions folder


class Rewriter(Model):
    """A scripted reflection model: each call takes the next step of ``script`` and makes it a tool call (``shell``
    with a command, ``submit`` with files) or a reply that ends its turn (``say``). A step may be a function of what
    the model has read so far (``read``: the versions folder, the current text, what it was last told: a tool's
    result or a message sent back to it). ``told`` keeps everything it was told after the task message."""

    def __init__(self, script):
        self.script, self.told = list(script), []
        self.config = {"model_id": "rewriter"}

    def update_config(self, **kw):
        self.config.update(kw)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    @staticmethod
    def read(messages):
        task = messages[0]["content"][0]["text"]
        last = [
            c["toolResult"]["content"][0].get("text", "") if "toolResult" in c else c.get("text", "")
            for c in (messages[-1].get("content", []) if len(messages) > 1 else [])
            if isinstance(c, dict)
        ]
        return {
            "versions": re.search(r"`(/[^`]*versions_[^`/]*)`", task).group(1),
            "text": re.search(r'<parameter name="system_prompt">\n(.*?)\n</parameter>', task, re.S).group(1),
            "last": "\n".join(last),
        }

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        seen = self.read(messages)
        if seen["last"]:
            self.told.append(seen["last"])
        kind, arg = self.script.pop(0) if self.script else ("say", "Nothing more.")
        arg = arg(seen) if callable(arg) else arg
        yield {"messageStart": {"role": "assistant"}}
        if kind == "say":
            yield {"contentBlockDelta": {"delta": {"text": arg}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            name, args = ("shell", {"command": arg}) if kind == "shell" else ("submit_optimized_params", arg)
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{len(self.script)}", "name": name}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({**args})}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        usage = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


SUBMIT = ("submit", {"file_path_dict": {"system_prompt": "optimized_system_prompt.txt"}})


def rewrite_once(monkeypatch, script, record=None, **options):
    """One reflection step on ``echo`` ("Help.") with the real optimizer, confined shell and submit tool, and
    ``Rewriter(script)`` as its model; returns the study and the model."""
    model = Rewriter(script)
    monkeypatch.setattr(agentic, "BedrockModel", lambda **kwargs: model)
    study = Study(
        agent=echo,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(Reflection(**options), HELP)]),
        record=record,
        log=None,
    )
    study.optimize(iterations=1)
    return study, model


def test_a_rewriter_that_copies_the_current_texts_file_from_the_versions_folder_submits_the_text_alone(monkeypatch):
    """With one file per version, a rewriter that copied the versions folder's file holding the current text into
    its working folder, edited the text there and submitted it submitted the version's whole record, so its header
    (`# Version 26`, rank, results) became part of the text."""
    copy_and_edit = (
        "shell",
        lambda seen: (
            f"f=$(grep -rlxF -- {shlex.quote(seen['text'])} {seen['versions']} | sort | head -1); "
            "sed 's/^Help\\.$/Help. Be brief./' \"$f\" > optimized_system_prompt.txt"
        ),
    )
    study, _ = rewrite_once(monkeypatch, [copy_and_edit, SUBMIT, ("say", "Submitted.")])
    candidate = study.candidates[1]
    assert candidate.origin["text"]["status"] == "rewritten"
    assert not HEADER.search(candidate.params["system_prompt"]), candidate.params["system_prompt"]
    assert candidate.params["system_prompt"] == "Help. Be brief."


# what a rewriter that pasted a version's whole record writes: its header and results above the text
PASTED = (
    "printf '# Version 0 (the baseline)\\nrank: 1\\nbuilt from: nothing (the start)\\nWordCount: 1\\n\\n"
    "Help. Be brief.\\n' > optimized_system_prompt.txt"
)
ALONE = ("shell", "printf 'Help. Be brief.\\n' > optimized_system_prompt.txt")


def test_a_submission_whose_text_carries_a_versions_header_is_refused_and_the_rewriter_told_why(monkeypatch):
    script = [("shell", PASTED), SUBMIT, ALONE, SUBMIT, ("say", "Submitted.")]
    study, model = rewrite_once(monkeypatch, script)
    candidate = study.candidates[1]
    assert candidate.params["system_prompt"] == "Help. Be brief."
    refusal = next(t for t in model.told if t.startswith("Refused"))
    assert "system_prompt: it holds a version's header from the versions folder (`# Version 0 (the baseline)`" in (
        refusal
    )
    assert "Parameters submitted successfully" in model.told[-1]
    [rejected] = candidate.origin["text"]["rejected"]
    assert rejected["name"] == "system_prompt" and "version's header" in rejected["reason"]


def test_a_turn_that_ends_on_a_refusal_is_sent_back_and_one_never_fixed_keeps_the_held_text(monkeypatch, tmp_path):
    script = [("shell", PASTED), SUBMIT, ("say", "Done."), ALONE, SUBMIT, ("say", "Submitted.")]
    study, model = rewrite_once(monkeypatch, script)  # over_limit_retries=2: sent back once
    assert study.candidates[1].params["system_prompt"] == "Help. Be brief."
    assert any("Your turn ended with nothing submitted" in t and "Refused" in t for t in model.told)

    script = [("shell", PASTED), SUBMIT, ("say", "Done.")]
    study, model = rewrite_once(monkeypatch, script, record=tmp_path, over_limit_retries=0)
    assert not any(HEADER.search(c.params["system_prompt"]) for c in study.candidates)
    events = [json.loads(line) for line in (tmp_path / "study.jsonl").read_text().splitlines()]
    # nothing accepted: the draw kept the held text (the baseline's, so a repeat), never the header
    origin = next(e for e in events if e["type"] in ("candidate", "repeat") and e.get("origin"))["origin"]["text"]
    assert origin["status"] == "failed" and "Every submission was refused (1)" in origin["error"]
    assert origin["rejected"][0]["name"] == "system_prompt"


def test_each_version_is_a_folder_of_its_texts_alone_and_its_record_is_never_a_text(tmp_path):
    version = {
        "rank": 1,
        "number": 26,
        "from": 15,
        "baseline": False,
        "cells": ["0.424", "465,938"],
        "meets_limits": True,
        "conversations": 144,
        "others": {"effort": "high"},
        "texts": {"contract": "You are an agent.\n## Acting\n- act", "tools/kb search": "Search the KB."},
    }
    _write_versions(str(tmp_path), [version], ["reward", "weighted_tokens"])
    folder = tmp_path / "rank-001-version-26"
    assert sorted(p.name for p in folder.iterdir()) == ["contract.txt", "tools_kb_search.txt", "version.md"]
    assert (folder / "contract.txt").read_text() == "You are an agent.\n## Acting\n- act"
    record = (folder / "version.md").read_text()
    assert record.startswith("# Version 26\nrank: 1\nbuilt from: version 15\nreward: 0.424\n")
    assert "You are an agent." not in record and '"effort": "high"' in record
    assert _no_version_header("contract", (folder / "contract.txt").read_text()) is None
    assert "`# Version 26`" in _no_version_header("contract", record + "\nYou are an agent.")
    pasted = "# Version 26\nrank: 1\nbuilt from: version 15\nreward: 0.424\n\n## contract\n\nYou are an agent."
    assert _no_version_header("contract", pasted) and _no_version_header("contract", "Intro.\n\n" + pasted)
    assert _no_version_header("contract", "# Version 2 of the API\nrank: by date\n# Versions\n") is None


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
