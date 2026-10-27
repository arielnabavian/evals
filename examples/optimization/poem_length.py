"""Steel thread on Amazon Bedrock: learn a system prompt that makes a Strands agent write poems of 40 to 60 words.

The poet is Claude Haiku 4.5 (``POEM_AGENT_MODEL``); its system prompt is rewritten by contrastive reflection on Claude
Sonnet 5.5 (``POEM_REWRITER_MODEL``). The rewriter is an agent with a shell over the trial transcripts, confined to its
own folders: run this where model-written shell commands are acceptable (a VM or a container).

    BYPASS_TOOL_CONSENT=true python examples/optimization/poem_length.py

It needs AWS credentials with access to both models (``AWS_PROFILE``, ``AWS_REGION``).
"""

import os
import time

from strands import Agent
from strands.models.bedrock import BedrockModel

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import Block, Maximize, Reflection, Search, Study, TextValue, tunable
from strands_evals.types.evaluation import EvaluationOutput

AGENT_MODEL = os.environ.get("POEM_AGENT_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
REWRITER_MODEL = os.environ.get("POEM_REWRITER_MODEL", "global.anthropic.claude-sonnet-5-5")

SYSTEM_PROMPT = TextValue(
    default="You write poems about the topic you are given.",
    about="the poet's system prompt",
    max_chars=2000,
)


@tunable(system_prompt=SYSTEM_PROMPT)
def create_poet(system_prompt):
    return Agent(
        model=BedrockModel(model_id=AGENT_MODEL, max_tokens=1024),
        system_prompt=system_prompt,
        callback_handler=None,
    )


class WordCount(Evaluator):
    """Passes when the poem has 40 to 60 words."""

    def evaluate(self, case):
        words = len(str(case.actual_output).split())
        passed = 40 <= words <= 60
        return [EvaluationOutput(score=float(passed), test_pass=passed, reason=f"{words} words (40 to 60 passes)")]

    async def evaluate_async(self, case):
        return self.evaluate(case)


topics = ["the sea", "a city at night", "first snow", "an old dog", "the last train home", "a garden in August"]
experiment = Experiment(cases=[Case(name=t, input=t) for t in topics], evaluators=[WordCount()])

study = Study(
    agent=create_poet,
    experiment=experiment,
    objectives=Maximize("WordCount"),
    repetitions=2,
    # One block: the rewriter, on the system prompt. Width 1: one rewrite a step, so each sees the last one's result.
    search=Search(blocks=[Block(Reflection(model_config={"model_id": REWRITER_MODEL}), SYSTEM_PROMPT)], width=1),
    max_concurrency=12,  # a candidate's 6 topics x 2 repetitions at once
    record=f"studies/poem-length-{time.strftime('%Y%m%d-%H%M%S')}",
)

if __name__ == "__main__":
    study.optimize(iterations=4)
    for candidate in study.candidates[1:]:
        text = candidate.origin.get("text", {})
        print(  # noqa: T201
            f"candidate {candidate.number}: reflection {text.get('status')}, {text.get('trials')} trials, "
            f"rewriter tokens {text.get('usage', {}).get('tokens', 0):,.0f}"
        )
    print("\nbest system prompt:\n" + study.best_candidate.params["system_prompt"])  # noqa: T201
