"""The document-reader workload: a document whose pages must be read one at a time, the tools that read it, the cases,
and how an answer is scored. Not a study: see document_reader.py."""

from __future__ import annotations

import random

from strands import tool

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationOutput

PAGES = 10
rng = random.Random(7)
WORDS = ["ledger", "harbor", "copper", "lantern", "meadow", "signal", "quarry", "willow", "ember", "basin"]
DOCUMENTS = {name: "".join(rng.choice("0123456789") for _ in range(PAGES)) for name in ["atlas", "beacon", "cinder"]}
TICKETS = {name: f"T-{rng.randrange(1000, 9999)}" for name in DOCUMENTS}
KEYS = {(name, page): f"{rng.getrandbits(32):08x}" for name in DOCUMENTS for page in range(1, PAGES + 1)}
PAGE_OF = {key: name_page for name_page, key in KEYS.items()}


def page_text(name: str, page: int) -> str:
    words = random.Random(f"{name}/{page}")
    filler = lambda n: " ".join(words.choice(WORDS) for _ in range(n))  # noqa: E731
    next_key = (
        f" The key to page {page + 1} is {KEYS[(name, page + 1)]}." if page < PAGES else " This is the last page."
    )
    return (
        f"Document {name}, page {page} of {PAGES}. {filler(80)} Digit {page} of the code is "
        f"{DOCUMENTS[name][page - 1]}. {filler(80)}{next_key}"
    )


@tool
def open_document(name: str) -> str:
    """Open a document and read its first page."""
    return page_text(name, 1)


@tool
def read_page(key: str) -> str:
    """Read the page that a key from the previous page unlocks."""
    if key not in PAGE_OF:
        return "No page has that key."
    return page_text(*PAGE_OF[key])


def score_answer(reply: str, expected: str) -> float:
    """The ticket id, and each digit of the code in its place: out of 1 + the code's length."""
    ticket, code = expected.split(": ")
    words = [w.strip("*.`'\"") for w in reply.replace(":", " ").split()]
    found = next((w for w in words if w.isdigit() and len(w) == len(code)), "")
    return (float(ticket in reply) + sum(a == b for a, b in zip(found, code, strict=False))) / (1 + len(code))


class Answer(Evaluator):
    """`score_answer` of the agent's reply (a plain function, so a packaged task's verifier can be the same code)."""

    def evaluate(self, case):
        score = score_answer(str(case.actual_output), case.expected_output)
        return [
            EvaluationOutput(score=score, test_pass=score == 1.0, reason=f"replied {str(case.actual_output)[-60:]!r}")
        ]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def cases() -> list[Case]:
    return [
        Case(
            name=name,
            input=f"Ticket {TICKETS[name]}: what is the code in document {name!r}? Read every page. "
            f"Reply exactly as '<ticket>: <code>'.",
            expected_output=f"{TICKETS[name]}: {code}",
        )
        for name, code in DOCUMENTS.items()
    ]
