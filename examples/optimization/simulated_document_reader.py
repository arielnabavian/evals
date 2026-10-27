"""The document-reader steps (document_reader_steps.py) with a scripted reader in place of the LLM; everything else is
real.

Real: the tools, Strands' agent loop and conversation managers (which trim and summarize the real conversation), the
evaluator, the Study with its blocks, policies and record. Scripted: the model, a careful reader that reads the pages
in order through the tools and answers only from what is still in its context, so a small window really loses
digits and the ticket; it keeps a running note when its system prompt says so, and reports token usage from the
actual size of its context, with a little noise. Contrastive reflection is replaced by a scripted rewrite that adds
real-looking rules (its origin has the real method's shape).

For developing the explorer and for fast end-to-end checks; no network, about a minute for every study:

    python examples/optimization/simulated_document_reader.py --out studies/ [--step 5]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from pathlib import Path
from typing import Any

import document_reader_steps as steps
from document_reader_workload import PAGES
from strands.models.model import Model

from strands_evals.optimization import Block, Draw, Method, Search, Study, TextValue

NOTE_RULE = (
    "WHEN reading a multi-page document, DO keep a running note in each reply with the ticket id and every "
    "digit found so far, so nothing is lost if earlier messages are trimmed."
)
CHECK_RULE = "WHEN answering, DO check that every digit position is filled before writing the final line."


class SimulatedReader(Model):
    """Reads the chained document through the real tools and answers from its context."""

    rng = random.Random(7)

    def __init__(self) -> None:
        self.config = {"model_id": "simulated-reader", "context_window_limit": 4000}

    def update_config(self, **kw: Any) -> None:
        self.config.update(kw)

    def get_config(self) -> dict:
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    @staticmethod
    def _texts(messages: list) -> list[tuple[str, str]]:
        out = []
        for m in messages:
            for block in m.get("content", []):
                if "text" in block:
                    out.append((m["role"], block["text"]))
                if "toolResult" in block:
                    for c in block["toolResult"].get("content", []):
                        if "text" in c:
                            out.append(("tool", c["text"]))
        return out

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        texts = self._texts(messages)
        context = sum(len(t) for _, t in texts) + len(system_prompt or "")
        if system_prompt and "summarizer" in system_prompt:
            everything = " ".join(t for _, t in texts)
            seen = sorted(set(re.findall(r"Digit (\d+) of the code is (\d)", everything)), key=lambda d: int(d[0]))
            ticket = re.search(r"T-\d{4}", everything)
            kept = [f"digit {k} = {d}" for k, d in seen if self.rng.random() > 0.08]
            reply = (
                "- Summary: "
                + (f"ticket {ticket.group(0)}; " if ticket and self.rng.random() > 0.3 else "")
                + ("; ".join(kept) or "no digits yet")
            )
            async for e in self._text(reply, context):
                yield e
            return
        notes = NOTE_RULE in (system_prompt or "")
        careful = CHECK_RULE in (system_prompt or "")
        everything = " ".join(t for _, t in texts)
        request = next((t for r, t in texts if r == "user" and "Ticket" in t), "")
        ticket = re.search(r"T-\d{4}", request) or re.search(r"ticket (T-\d{4})", everything)
        digits = dict(re.findall(r"Digit (\d+) of the code is (\d)", everything))
        for note in re.findall(r"digits so far: ([0-9?]+)", everything):  # its own running notes
            for i, d in enumerate(note, start=1):
                if d != "?":
                    digits.setdefault(str(i), d)
        last = texts[-1][1] if texts else ""
        page = re.search(r"page (\d+) of (\d+)", last)
        key = re.search(r"key to page \d+ is ([0-9a-f]{8})", last)
        name = re.search(r"document '(\w+)'", request) or re.search(r"Document (\w+), page", everything)
        code = "".join(digits.get(str(i), "?") for i in range(1, PAGES + 1))
        if careful and "?" in code and self.rng.random() < 0.5:
            code = code.replace("?", str(self.rng.randrange(10)), 1)  # a careful guess fills one gap
        note = f"Note: ticket {self._ticket(ticket) or 'unknown'}; digits so far: {code}. " if notes else ""
        if not page and name:
            opening = note + "I'll open the document."
            async for e in self._tool("open_document", {"name": name.group(1)}, opening, context):
                yield e
        elif page and key and int(page.group(1)) < int(page.group(2)):
            if self.rng.random() < 0.03:  # occasionally loses its place and answers early
                async for e in self._answer(ticket, code, context):
                    yield e
                return
            reading = note + f"Reading page {int(page.group(1)) + 1}."
            async for e in self._tool("read_page", {"key": key.group(1)}, reading, context):
                yield e
        else:
            async for e in self._answer(ticket, code, context):
                yield e

    @staticmethod
    def _ticket(match: re.Match | None) -> str | None:
        """The ticket id a match found (its group, when the pattern has one)."""
        if match is None:
            return None
        return match.group(1) if match.re.groups else match.group(0)

    async def _answer(self, ticket, code, context):
        if self.rng.random() < 0.05 and "?" not in code:  # a slip on one digit
            i = self.rng.randrange(len(code))
            code = code[:i] + str((int(code[i]) + 1) % 10) + code[i + 1 :]
        t = self._ticket(ticket)
        async for e in self._text(f"{t}: {code}" if t else code, context):
            yield e

    def _usage(self, context: int, out: str) -> dict:
        noise = self.rng.uniform(0.9, 1.1)
        inp = int(context / 4 * noise) + 120
        return {"inputTokens": inp, "outputTokens": len(out) // 4 + 5, "totalTokens": inp + len(out) // 4 + 5}

    async def _text(self, text, context):
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": self._usage(context, text), "metrics": {"latencyMs": 400}}}

    async def _tool(self, name, arguments, text, context):
        self._n = getattr(self, "_n", 0) + 1
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self._n}", "name": name}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {"usage": self._usage(context, text), "metrics": {"latencyMs": 600}}}


class _Rewrite(Draw):
    def __init__(self, text: str, origin: dict) -> None:
        self.text, self.origin = text, origin

    def suggest_text(self, name: str, default: str) -> str:
        return self.text if name == "system_prompt" else default


class ScriptedReflection(Method):
    """Adds the next rule a reflection would learn from the held candidate's failures (none once both are in)."""

    kinds = ("text",)
    rng = random.Random(11)

    def begin(self, block, base, history, structure, study, objective):
        prompt = base.params["system_prompt"]
        rewards = study.projection.rewards(base, study.combine, objective)
        lost = sum(1 for _, r, _ in rewards if r < 1.0)
        rule = next((r for r in (NOTE_RULE, CHECK_RULE) if r not in prompt), None)
        usage = {
            "input_tokens": float(self.rng.randint(90_000, 140_000)),
            "output_tokens": float(self.rng.randint(2_000, 5_000)),
            "cache_read_tokens": 0.0,
            "cache_write_tokens": 0.0,
        }
        usage["tokens"] = usage["input_tokens"] + usage["output_tokens"]
        origin = {
            "method": type(self).__name__,
            "from_candidate": base.number,
            "trials": len(rewards),
            "usage": usage,
            "seconds": round(self.rng.uniform(80, 170), 1),
        }
        if rule is None or lost == 0:
            return _Rewrite(prompt, {**origin, "status": "unchanged", "changed": []})
        section = "" if "## Learned Behaviors" in prompt else "\n\n---\n\n## Learned Behaviors\n"
        return _Rewrite(f"{prompt}{section}\n{rule}", {**origin, "status": "rewritten", "changed": ["system_prompt"]})


def study(step: int, record: str | None, **options: Any) -> Study:
    steps.make_model = SimulatedReader
    factory = steps.STEPS[step][1]
    texts = [obj for obj, _, _ in factory.references() if isinstance(obj, TextValue)]
    search = Search(blocks=[Block(ScriptedReflection(), *texts)] if texts else [])
    return steps.study(step, record, search=search, **options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="studies")
    parser.add_argument("--step", type=int, choices=list(steps.STEPS), default=None, help="default: every step")
    parser.add_argument("--iterations", type=int, default=None, help="iterations per study (default: the step's own)")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for step in [args.step] if args.step else list(steps.STEPS):
        name, _, _, n = steps.STEPS[step]
        out = Path(args.out) / f"sim-{step}-{name}-{stamp}"
        print(f"\n== step {step} ({name}) -> {out}")  # noqa: T201
        study(step, str(out)).optimize(iterations=args.iterations or n)
