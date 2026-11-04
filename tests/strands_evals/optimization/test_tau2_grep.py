"""The τ² workload's guard on τ³ banking's grep tool (tau2_workload._grep_retrieve): τ²'s results for a pattern that
finishes, the tool's error, not a frozen process, for one that does not. Needs τ² (skipped without it)."""

import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("tau2")
pytest.importorskip("regex")

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "optimization"))

import tau2_workload as workload  # noqa: E402
from tau2.data_model.message import ToolCall  # noqa: E402
from tau2.knowledge.retrievers.grep_retriever import GrepRetriever  # noqa: E402

DOCS = {
    # short runs of words between punctuation, so that re finishes FROZEN on every document but "flow"
    "fees": "ATM fees: $2.50 each. Intl. ATM fee: 3%. Foreign fee, rebate: 1%.",
    "cards": "Credit card rewards. Card replacement: order_replacement_credit_card_7291. CREDIT limits.",
    "empty": "",
    "accounts": "Silver Account, Platinum Account, Diamond Elite Account, Silver Plus Account.",
    # a document a study froze on starts with prose like this
    "flow": "## Overview of Data Flow\n- Transaction categorization: Card purchases are automatically classified by "
    "merchant category and the rewards engine applies the matching earn rate to every eligible purchase.\n" * 3,
}
STATE = {"doc_content_map": DOCS}
# the agent's pattern in the frozen trial: (\w+ ?)* before a literal that does not follow backtracks exponentially in re
FROZEN = r"(?i)(\w+ ?)*Account.*(ATM|international|foreign)|(ATM|international|foreign).*(fee|rebate).*(\$|%)"


def test_the_tool_uses_the_guard():
    assert GrepRetriever.retrieve is workload._grep_retrieve
    assert workload._RE_RETRIEVE.__module__ == "tau2.knowledge.retrievers.grep_retriever"


@pytest.mark.parametrize(
    "pattern",
    [
        "credit.*card",
        "fee|charge",
        "ATM.*fee",
        r"(international|foreign).*(fee|rebate)",
        r"\$\d+\.\d\d",
        "account",
        "x*",  # empty matches count as re counts them
        r"\bcard\b",
        "order_replacement_credit_card_7291",
        "(unbalanced",  # re rejects it: τ² searches it as literal text
        "[card",
        "   ",
        "Diamond Elite|Silver Plus",
    ],
)
@pytest.mark.parametrize("case_sensitive", [False, True])
@pytest.mark.parametrize("top_k", [10, 2])
def test_same_results_as_tau2(pattern, case_sensitive, top_k):
    retriever = GrepRetriever(top_k=top_k, case_sensitive=case_sensitive)
    assert workload._grep_retrieve(retriever, {"query": pattern}, STATE) == workload._RE_RETRIEVE(
        retriever, {"query": pattern}, STATE
    )


# backtracks exponentially in the regex module too (re's runaway patterns above do not)
RUNAWAY = ("(a|aa)+$", {"d": "a" * 60 + "b"})


def test_a_runaway_pattern_ends_with_the_tools_error(monkeypatch):
    monkeypatch.setattr(workload, "GREP_SECONDS", 1.0)
    started = time.monotonic()
    with pytest.raises(workload.GrepTimeout, match="did not finish within 1 seconds"):
        workload._grep_retrieve(GrepRetriever(), {"query": RUNAWAY[0]}, {"doc_content_map": RUNAWAY[1]})
    assert time.monotonic() - started < 5


def test_the_frozen_pattern_finishes():
    """A pattern that froze a study: the guard answers at once, with τ²'s result on the documents re can
    finish (re never finishes "flow", which has no match; "cards" has a 34-character word: slow for re)."""
    retriever = GrepRetriever()
    started = time.monotonic()
    found = workload._grep_retrieve(retriever, {"query": FROZEN}, STATE)
    assert time.monotonic() - started < 5
    rest = {"doc_content_map": {k: v for k, v in DOCS.items() if k in ("fees", "accounts", "empty")}}
    assert found == workload._RE_RETRIEVE(retriever, {"query": FROZEN}, rest)
    assert [doc for doc, _ in found] == ["fees"]


def test_other_threads_run_while_a_pattern_matches(monkeypatch):
    """re holds the GIL for the whole match; the guard releases it, so the study's other threads keep going."""
    monkeypatch.setattr(workload, "GREP_SECONDS", 2.0)
    ticks = []
    done = threading.Event()

    def tick():
        while not done.is_set():
            ticks.append(time.monotonic())
            time.sleep(0.01)

    t = threading.Thread(target=tick)
    t.start()
    try:
        with pytest.raises(workload.GrepTimeout):
            workload._grep_retrieve(GrepRetriever(), {"query": RUNAWAY[0]}, {"doc_content_map": RUNAWAY[1]})
    finally:
        done.set()
        t.join()
    assert len(ticks) > 50  # ~200 in 2 s; a held GIL would allow about 1


def test_a_timeout_reaches_the_agent_as_tau2s_tool_error(monkeypatch):
    """On the real knowledge base, through τ²'s Environment.get_response: the tool message 'Error: ...'."""
    env = workload._environment("banking_knowledge")
    ok = env.get_response(ToolCall(id="1", name="grep", arguments={"pattern": "ATM.*fee"}))
    assert not ok.error and "Score" in ok.content
    monkeypatch.setattr(workload, "GREP_SECONDS", 0.5)
    started = time.monotonic()
    msg = env.get_response(ToolCall(id="2", name="grep", arguments={"pattern": "(.|..)+$"}))
    assert time.monotonic() - started < 5
    assert msg.error and "did not finish within 0.5 seconds" in msg.content
