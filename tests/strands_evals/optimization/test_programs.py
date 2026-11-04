"""Program-valued variables (strands_evals.optimization.programs): a ConversationManager written by a scripted proposer
agent, standalone and inside a OneOf; the empty program; the loader's restrictions; what the proposer is told (no
network)."""

import json
import sys
from abc import ABC, abstractmethod

import pytest
from strands import Agent
from strands.agent.conversation_manager import ConversationManager, SlidingWindowConversationManager

from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    Block,
    Int,
    Joint,
    Maximize,
    Nested,
    OneOf,
    Search,
    Study,
    Thompson,
    component,
    tunable,
)
from strands_evals.optimization.programs import (
    EMPTY,
    EMPTY_VALUE,
    Program,
    ProgramProposer,
    ProgramTimeout,
    load,
    run_bounded,
    value_of,
)
from strands_evals.types.evaluation import EvaluationOutput

from .helpers import Says, experiment, needs_bayesbench

SEED = """from strands.agent.conversation_manager import ConversationManager


class KeepLast(ConversationManager):
    def __init__(self):
        super().__init__()
        self.keep = 8

    def apply_management(self, agent, **kwargs):
        excess = len(agent.messages) - self.keep
        if excess > 0:
            del agent.messages[:excess]
            self.removed_message_count += excess

    def reduce_context(self, agent, e=None, **kwargs):
        self.keep = max(2, self.keep // 2)
        self.apply_management(agent)
"""
GOOD = SEED.replace("KeepLast", "KeepFour").replace("self.keep = 8", "self.keep = 4")
BAD_IMPORT = "import os\n" + GOOD.replace("KeepFour", "KeepOs")
ABSTRACT = GOOD.replace("KeepFour", "NoReduce").split("    def reduce_context")[0]
KEY = "conversation_manager.version"


def fenced(source):
    return f"Here is the new version.\n```python\n{source}```\nIt keeps fewer messages."


def scripted(*sources):
    """A proposer whose agent runs on a scripted model: one reply (a program's source) per proposal."""
    replies = iter([fenced(s) for s in sources])
    return ProgramProposer(model=lambda: Says(next(replies)), model_id="says")


def factory(spec, received):
    @tunable(conversation_manager=spec)
    def create(conversation_manager):
        received.append(type(conversation_manager).__name__)
        return Agent(model=Says("a short answer"), conversation_manager=conversation_manager, callback_handler=None)

    return create


def test_program_binds_to_the_proposer_by_type_and_its_baseline_is_the_seed():
    manager = Program(ConversationManager, seeds=[SEED], about="how the agent trims its history")
    study = Study(agent=factory(manager, []), experiment=experiment(), objectives=Maximize("WordCount"), log=None)
    assert list(study.blocks) == ["programs"] and isinstance(study.blocks["programs"].method, ProgramProposer)
    assert study.params[KEY].type == "program" and study.params[KEY].default == value_of(SEED)
    assert value_of(SEED).startswith("programs/") and len(value_of(SEED)) == len("programs/") + 16


def test_standalone_program_written_stored_by_hash_invalid_when_it_fails_and_counted(tmp_path):
    manager = Program(ConversationManager, seeds=[SEED], about="how the agent trims its history")
    proposer = scripted(GOOD, BAD_IMPORT, ABSTRACT)
    received = []
    study = Study(
        agent=factory(manager, received),
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(width=1, blocks=[Block(proposer, manager)]),  # overrides the default proposer
        record=tmp_path / "record",
        log=None,
    )
    assert study.blocks["programs"].method is proposer
    result = study.optimize(iterations=3)
    baseline, written, bad_import, abstract = study.candidates[:4]

    # every value is a content hash; each version is stored once, immutable, in the study's artifact store
    assert baseline.params[KEY] == value_of(SEED) and written.params[KEY] == value_of(GOOD)
    assert (study.artifact_path(baseline.params[KEY]) / "program.py").read_text() == SEED
    assert (study.artifact_path(written.params[KEY]) / "program.py").read_text() == GOOD
    manifest = json.loads((study.artifact_path(written.params[KEY]) / "manifest.json").read_text())
    assert manifest["parent"] == value_of(SEED)
    assert manifest["interface"].endswith(":ConversationManager")
    assert baseline.status == written.status == "complete" and {"KeepLast", "KeepFour"} <= set(received)

    # code that fails to load or check is an invalid candidate: never run, so never a zero
    for bad, why in ((bad_import, "import of 'os' is not allowed"), (abstract, "does not implement")):
        assert bad.status == "invalid" and why in bad.reason, bad.reason
        assert bad.trials == () and not bad.eligible and bad.params[KEY].startswith("programs/")
    assert "NoReduce" not in received and "KeepOs" not in received

    # the proposer's own model tokens are in the native ledger, as method:<block>
    cell = result.cost.roles["method:programs"]["search"]
    spent = sum(c.origin["programs"]["usage"]["tokens"] for c in (written, bad_import, abstract))
    assert cell["calls"] == 3 and cell["tokens"] == spent > 0
    head = json.loads((tmp_path / "record" / "study.jsonl").read_text().splitlines()[0])
    assert head["roles"]["method:programs"]["model"] == "says"
    assert {"key": KEY, "type": "program"}.items() <= head["space"][0].items()


@needs_bayesbench  # Thompson samples from the study's model
def test_program_inside_a_one_of_is_compared_with_a_hand_written_manager_by_nested(tmp_path):
    manager = Program(ConversationManager, seeds=[SEED])
    choice = OneOf(component(SlidingWindowConversationManager, window_size=Int(2, 20, default=10)), manager)
    outer, programs = Block(Thompson(seed=0), choice), Block(scripted(GOOD, GOOD.replace("= 4", "= 3")), manager)
    received = []
    study = Study(
        agent=factory(choice, received),
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        search=Search(
            width=1, blocks=[outer, programs], schedule=Nested(outer=outer, inner=Joint(), inner_iterations=1)
        ),
        record=tmp_path / "record",
        log=None,
    )
    assert study.params["conversation_manager"].choices == ("SlidingWindowConversationManager", "program")
    result = study.optimize(iterations=4)
    labels = [c.params["conversation_manager"] for c in study.candidates]
    assert labels[0] == "SlidingWindowConversationManager" and "program" in labels
    versions = [c.params.get("conversation_manager.program.version") for c in study.candidates]
    assert value_of(SEED) in versions and value_of(GOOD) in versions  # the seed opens its round; the proposer writes
    assert {"SlidingWindowConversationManager", "KeepLast", "KeepFour"} <= set(received)
    assert result.cost.roles["method:programs"]["search"]["calls"] >= 1
    records = [json.loads(line) for line in (tmp_path / "record" / "study.jsonl").read_text().splitlines()]
    told = [r for r in records if r["type"] == "told" and r.get("block") == "structure"]
    assert told  # the outer method is told each round's best: the program compared at its inner best


def test_the_loader_refuses_what_it_should():
    for source, why in (
        ("import subprocess\n" + GOOD, "subprocess"),
        (GOOD + "\nopen('/etc/passwd')\n", "'open'"),
        (GOOD + "\nx = ().__class__.__base__.__subclasses__()\n", "attribute '__"),
        (GOOD + "\nx = '{0.__globals__}'.format(len)\n", "format"),
        (GOOD + "\nclass Second(KeepFour):\n    pass\n", "expected one subclass"),
    ):
        try:
            load(source, ConversationManager, ("strands.agent", "strands.types"))
        except Exception as exc:  # noqa: BLE001
            assert why in str(exc), (why, exc)
        else:
            raise AssertionError(f"loaded: {why}")


# ---- the empty program, the proposer's task message, sending back a version that does not load -----------------------
class Scorer(ABC):
    """Scores an answer."""

    label: str = ""

    @abstractmethod
    def score(self, text: str) -> int:
        """The answer's score."""


SCORER = """import re


class Words(Scorer):
    label = "words"

    def score(self, text):
        return len(re.findall(r"\\w+", text))
"""
SCORER_ABSTRACT = "class Nothing(Scorer):\n    label = 'nothing'\n"


class Passes(Evaluator):
    """1 on the case named "sea", else 0 (a bounded score: the other case's trials did not succeed)."""

    def evaluate(self, case):
        ok = case.name.split("#")[0] == "sea"
        return [EvaluationOutput(score=1.0 if ok else 0.0, test_pass=ok, reason="the sea" if ok else "not the sea")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


class Heard(Says):
    """``Says``, one reply per call in turn, keeping each message it was sent (the last user message's text)."""

    def __init__(self, replies, heard):
        super().__init__(replies[0])
        self.replies, self.heard = list(replies), heard

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.heard.append(" ".join(b.get("text", "") for b in messages[-1]["content"] if isinstance(b, dict)))
        self.text = self.replies[min(len(self.heard), len(self.replies)) - 1]
        async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
            yield event


def scorer_factory(spec, received):
    @tunable(scorer=spec)
    def create(scorer):
        received.append(scorer)
        return Agent(model=Says("a short answer"), callback_handler=None)

    return create


def test_the_empty_program_is_the_baseline_and_the_proposer_writes_the_first_version_from_the_trials(tmp_path):
    scorer = Program(
        Scorer,
        seeds=[EMPTY],
        about="how answers are scored",
        context="Answers are short.",
        imports=("re",),
        standard=False,
    )
    heard, received = [], []
    proposer = ProgramProposer(model=lambda: Heard([fenced(SCORER)], heard), model_id="heard")
    study = Study(
        agent=scorer_factory(scorer, received),
        experiment=experiment(Passes()),
        objectives=Maximize("Passes"),
        search=Search(width=1, blocks=[Block(proposer, scorer)]),
        record=tmp_path / "record",
        log=None,
    )
    key = "scorer.version"
    assert study.params[key].default == EMPTY_VALUE == "programs/none"
    study.optimize(iterations=1)
    baseline, written = study.candidates[:2]
    assert baseline.params[key] == EMPTY_VALUE and baseline.status == "complete"
    assert written.params[key] == value_of(SCORER) and written.status == "complete"
    assert received[0] is None and type(received[-1]).__name__ == "Words" and received[-1].score("two words") == 2

    # the task: the rules (only the declared imports), the interface, the environment, no held version, the versions
    # measured with their goal metrics, reflection's digest of the held candidate's trials and the transcripts of
    # those that did not succeed
    task = heard[0]
    assert "Import only from: re." in task and "already defined in the module's namespace" in task
    assert "class Scorer(ABC)" in task and "## The environment\nAnswers are short." in task
    assert "There is none: the agent runs without the program" in task
    assert (
        "Goal metrics: Passes (maximize)" in task
        and "- candidate 0: none (the empty program) [held, baseline]: Passes=0.5" in task
    )
    assert "# Digest: the current version's 2 conversations, 2 cases" in task
    assert "## Every case with a conversation that did not succeed (1)" in task and "### city:" in task
    assert "follow the digest, under the same names" in task and "Each conversation is a file" not in task
    assert "## Transcripts of the trials that did not succeed: 1 of 1" in task
    assert (
        "trial-002: city, repetition 1, reward 0" in task
        and "[user] city" in task
        and "[assistant] a short answer" in task
    )

    # what it read of the trials is kept with the study and named in the record
    origin = written.origin["programs"]["scorer.version"]
    assert origin["status"] == "written" and origin["digest"]["trials"] == 2 and origin["digest"]["failing_cases"] == 1
    assert (study.artifacts / origin["digest"]["file"]).read_text() in task
    head = json.loads((tmp_path / "record" / "study.jsonl").read_text().splitlines()[0])
    assert {"key": key, "type": "program", "written": True}.items() <= head["space"][0].items()
    assert head["space"][0]["settings"] == {"interface": f"{Scorer.__module__}:Scorer"}
    capped = Program(Scorer, seeds=[EMPTY], settings={"max_commands": 8}).arguments["version"]
    assert capped.record_settings() == {"interface": f"{Scorer.__module__}:Scorer", "max_commands": 8}


def test_a_version_that_does_not_load_is_sent_back_with_the_error_in_the_same_conversation(tmp_path):
    scorer = Program(Scorer, seeds=[EMPTY], imports=("re",), standard=False)
    heard = []
    replies = [fenced("import os\n" + SCORER), fenced(SCORER_ABSTRACT), fenced(SCORER)]
    study = Study(
        agent=scorer_factory(scorer, []),
        experiment=experiment(Passes()),
        objectives=Maximize("Passes"),
        search=Search(width=1, blocks=[Block(ProgramProposer(model=lambda: Heard(replies, heard)), scorer)]),
        log=None,
    )
    study.optimize(iterations=1)
    written = study.candidates[1]
    assert written.params["scorer.version"] == value_of(SCORER) and written.status == "complete"
    assert (
        len(heard) == 3 and "import of 'os' is not allowed" in heard[1] and "does not implement ['score']" in heard[2]
    )
    origin = written.origin["programs"]["scorer.version"]
    assert len(origin["revised"]) == 2 and origin["usage"]["model_calls"] == 3
    assert origin["usage"]["input_tokens"] == 3 * 50  # the conversation's three calls, each counted once

    # with no retries, the first version is kept as written, and its candidate is invalid
    heard.clear()
    study = Study(
        agent=scorer_factory(scorer, []),
        experiment=experiment(Passes()),
        objectives=Maximize("Passes"),
        search=Search(
            width=1, blocks=[Block(ProgramProposer(model=lambda: Heard(replies, heard), load_retries=0), scorer)]
        ),
        log=None,
    )
    study.optimize(iterations=1)
    assert len(heard) == 1 and study.candidates[1].status == "invalid" and "'os'" in study.candidates[1].reason


def test_a_declarations_check_refuses_a_class_that_loads():
    version = Program(
        Scorer,
        seeds=[EMPTY],
        imports=("re",),
        standard=False,
        check=lambda cls: None if cls.label else "label is empty",
    ).arguments["version"]
    assert version.problem(SCORER) is None and version.problem(EMPTY) is None
    assert "label is empty" in version.problem(SCORER.replace('label = "words"', 'label = ""'))
    # the empty program is a version only where it is a seed
    other = Program(Scorer, seeds=[SCORER], imports=("re",), standard=False).arguments["version"]
    assert "expected one subclass of Scorer" in other.problem(EMPTY)


def test_allowed_modules_are_seen_through_their_public_names_and_frames_are_out_of_reach():
    imports = ("re", "typing", "dataclasses", "collections", "functools", "__future__")
    good = (
        "from __future__ import annotations\nimport collections.abc\nfrom collections import Counter, abc\n"
        "from dataclasses import dataclass, field\nfrom typing import Any, Callable\nimport re\n\n"
        "@dataclass\nclass Counts:\n    seen: Any = field(default_factory=Counter)\n\n"
        "class Ok(Scorer):\n    label = 'ok'\n\n    def score(self, text):\n"
        "        assert isinstance(Counter(), collections.abc.Mapping) and abc.Mapping is collections.abc.Mapping\n"
        "        return len(re.split(' ', text))\n"
    )
    assert load(good, Scorer, imports)().score("a b c") == 3
    for source, why in (
        ("import typing\nx = typing.sys\n", "has no attribute 'sys'"),
        ("import dataclasses\nx = dataclasses.inspect\n", "has no attribute 'inspect'"),
        ("import json\n", "import of 'json' is not allowed"),
        ("def f():\n    yield 1\ng = f()\nx = g.gi_frame\n", "'gi_frame'"),
        ("try:\n    pass\nexcept:\n    pass\n", "bare `except:`"),
        ("try:\n    pass\nexcept BaseException:\n    pass\n", "'BaseException'"),
        ("import string\nx = string.Formatter()\n", "'Formatter'"),
    ):
        try:
            load(
                source + "\nclass S(Scorer):\n    def score(self, text):\n        return 0\n",
                Scorer,
                imports + ("string",),
            )
        except Exception as exc:  # noqa: BLE001
            assert why in str(exc), (why, exc)
        else:
            raise AssertionError(f"loaded: {why}")


def test_run_bounded_stops_a_loop_that_catches_exception_and_puts_the_trace_back():
    loops = load(
        "class Spin(Scorer):\n    def score(self, text):\n        n = 0\n        while True:\n            try:\n"
        "                n += 1\n            except Exception:\n                pass\n",
        Scorer,
        (),
    )()
    before = sys.gettrace()
    with pytest.raises(ProgramTimeout, match="more than 10,000 lines"):
        run_bounded(lambda: loops.score("x"), lines=10_000)
    with pytest.raises(ProgramTimeout, match="seconds"):
        run_bounded(lambda: loops.score("x"), lines=10**12, seconds=0.2)
    assert sys.gettrace() is before
    # a version whose module, or whose constructor, does not finish does not load
    score = "    def score(self, text):\n        return 0\n"
    module_spins = "while True:\n    pass\n\n\nclass T(Scorer):\n" + score
    init_spins = "class S(Scorer):\n    def __init__(self):\n        while True:\n            pass\n\n" + score
    for source in (module_spins, init_spins):
        with pytest.raises(TimeoutError, match="loading it: the program ran more than"):
            load(source, Scorer, ())
        assert value_of(source).rsplit("/", 1)[1] not in " ".join(sys.modules)  # not left in sys.modules
    assert run_bounded(lambda: load(SCORER, Scorer, ("re",))().score("one two"), lines=1000) == 2


def test_the_effects_model_treats_a_program_as_written():
    from strands_evals.optimization.effects import _factors

    space = [{"key": "p.version", "type": "program", "default": EMPTY_VALUE, "written": True, "active_when": None}]
    candidates = [
        {"number": i, "params": {"p.version": v}, "origin": {} if i == 0 else {"programs": {"from_candidate": 0}}}
        for i, v in enumerate([EMPTY_VALUE, "programs/aaaa", "programs/bbbb"])
    ]
    factors, omitted = _factors(space, candidates, candidates, {"p.version": "programs"})
    assert [(f.key, f.kind, f.levels) for f in factors] == [("p.version", "rewrites", ["default", "1"])] and not omitted


class CutOff(Heard):
    """``Heard``, but its first reply stops at the output limit (``max_tokens``)."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.heard:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        self.heard.append(" ".join(b.get("text", "") for b in messages[-1]["content"] if isinstance(b, dict)))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "Here is a long analysis ```python\nimport re\n"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "max_tokens"}}
        usage = {"inputTokens": 50, "outputTokens": 7, "totalTokens": 57}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


def test_a_reply_cut_off_at_the_output_limit_is_asked_again_shorter():
    version = Program(Scorer, seeds=[EMPTY], imports=("re",), standard=False).arguments["version"]
    heard = []
    proposer = ProgramProposer(model=lambda: CutOff([fenced(SCORER), fenced(SCORER)], heard))
    source, origin = proposer.write(version, EMPTY, [])
    assert source == SCORER and origin["status"] == "written"
    assert (
        origin["revised"] == ["the reply was cut off at the output limit"] and "cut off at the output limit" in heard[1]
    )
    assert origin["usage"]["model_calls"] == 2
    # with no retries left, the cut-off reply fails the proposal (the held version is kept)
    heard.clear()
    source, origin = ProgramProposer(model=lambda: CutOff([fenced(SCORER)], heard), load_retries=0).write(
        version, EMPTY, []
    )
    assert source is None and origin["status"] == "failed" and "MaxTokensReached" in origin["error"]
    assert ProgramProposer().describe()["max_tokens"] == 32000
