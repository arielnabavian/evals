"""Scripted models and evaluators for study tests (no network)."""

import importlib.util

import pytest
from strands.models.model import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationOutput

# The study's model (model.py) is BayesBench's posterior. BayesBench is not public yet, and without it a study
# decides with measured means; the tests of the model, and of what only a fitted model does, need it.
needs_bayesbench = pytest.mark.skipif(
    importlib.util.find_spec("bayesbench") is None, reason="needs BayesBench, the study's model (not public yet)"
)


class Says(Model):
    """Replies with fixed text (or its system prompt when ``text`` is None); two output tokens per word. Raises
    ``fail`` on the given call numbers."""

    def __init__(self, text=None, fail=None, on_calls=()):
        self.text, self.fail, self.on_calls, self.calls = text, fail, set(on_calls), 0
        self.config = {"model_id": "says"}

    def update_config(self, **kw):
        self.config.update(kw)

    def get_config(self):
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.fail is not None and (not self.on_calls or self.calls in self.on_calls):
            raise self.fail
        text = self.text if self.text is not None else system_prompt
        words = len(text.split())
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {
            "inputTokens": 50,
            "outputTokens": 2 * words,
            "totalTokens": 50 + 2 * words,
            "cacheReadInputTokens": 10,
            "cacheWriteInputTokens": 0,
        }
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


class WordCount(Evaluator):
    """Scores the number of words (unbounded, so it says where its scores lie)."""

    score_range = (0, None)

    def __init__(self, name=None):
        super().__init__(name=name)
        self.calls = 0

    def evaluate(self, case):
        self.calls += 1
        n = len(str(case.actual_output).split())
        return [EvaluationOutput(score=n, test_pass=True, reason=f"{n} words")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


class Fails(Evaluator):
    """Raises on the cases named in ``on``."""

    def __init__(self, on=()):
        super().__init__()
        self.on = set(on)

    def evaluate(self, case):
        if case.name.split("#")[0] in self.on:
            raise RuntimeError("grader crashed")
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def experiment(*evaluators, cases=("sea", "city")):
    return Experiment(cases=[Case(name=c, input=c) for c in cases], evaluators=list(evaluators or [WordCount()]))
