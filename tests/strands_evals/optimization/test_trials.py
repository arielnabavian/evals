"""Trials: statuses decided by the task wrapper, never inferred from a score; usage up to a failure; repetitions."""

import asyncio

import pytest
from strands import Agent
from strands.types.exceptions import EventLoopException, ModelThrottledException

from strands_evals.optimization import Maximize, Study, tunable
from strands_evals.optimization.trials import TrialStatus, classify, run_trials

from .helpers import Fails, Says, WordCount, experiment


def trials_of(factory, exp, repetitions=1):
    return asyncio.run(run_trials(factory, factory.default_values, exp, repetitions=repetitions, max_concurrency=4))


def test_an_agent_that_raises_is_an_agent_failure_not_a_zero_and_is_not_retried():
    models = []

    @tunable()
    def broken():
        model = Says("hello there", fail=ValueError("the harness has a bug"))
        models.append(model)
        return Agent(model=model, callback_handler=None)

    counter = WordCount()
    trials = trials_of(broken, experiment(counter))
    assert [a.status for a in trials] == [TrialStatus.AGENT_FAILURE] * 2
    assert all("the harness has a bug" in a.error for a in trials)
    assert counter.calls == 0  # the evaluators never ran
    assert all(m.calls == 1 for m in models)  # no retry (Strands Evals would retry 6 times, minutes of backoff)
    assert all("WordCount" not in a.metrics for a in trials)  # no score was invented


def test_usage_up_to_a_failure_is_kept():
    @tunable()
    def fails_on_second_call():
        from strands import tool

        @tool
        def look() -> str:
            """Look around."""
            return "nothing here"

        class ToolThenFail(Says):
            async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise ValueError("crashed after one model call")
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "t1", "name": "look"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
                yield {"contentBlockStop": {}}
                yield {"messageStop": {"stopReason": "tool_use"}}
                yield {
                    "metadata": {
                        "usage": {"inputTokens": 70, "outputTokens": 5, "totalTokens": 75},
                        "metrics": {"latencyMs": 1},
                    }
                }

        return Agent(model=ToolThenFail(), tools=[look], callback_handler=None)

    a = trials_of(fails_on_second_call, experiment(cases=("sea",)))[0]
    assert a.status == TrialStatus.AGENT_FAILURE
    assert a.metrics["input_tokens"] == 70 and a.metrics["tokens"] == 75 and a.metrics["model_calls"] >= 1
    assert len(a.messages) >= 2  # the conversation up to the failure


def test_classify():
    assert classify(ModelThrottledException("slow down")) == TrialStatus.INFRASTRUCTURE_INVALID
    assert classify(EventLoopException(ModelThrottledException("slow down"))) == TrialStatus.INFRASTRUCTURE_INVALID
    assert classify(EventLoopException(ValueError("bug"))) == TrialStatus.AGENT_FAILURE
    assert classify(ValueError("bug")) == TrialStatus.AGENT_FAILURE
    from botocore.exceptions import EndpointConnectionError
    from urllib3.exceptions import ProtocolError, ReadTimeoutError

    assert classify(EndpointConnectionError(endpoint_url="https://x")) == TrialStatus.INFRASTRUCTURE_INVALID
    stalled = ReadTimeoutError(None, None, "Read timed out.")  # a streamed response that stopped, as urllib3 raises it
    assert classify(stalled) == TrialStatus.INFRASTRUCTURE_INVALID
    assert classify(EventLoopException(stalled)) == TrialStatus.INFRASTRUCTURE_INVALID
    assert classify(ProtocolError("Connection broken")) == TrialStatus.INFRASTRUCTURE_INVALID
    assert classify(TimeoutError("a tool of the agent's timed out")) == TrialStatus.AGENT_FAILURE
    from botocore.exceptions import ClientError, EventStreamError

    broke = EventStreamError({"Error": {"Code": "internalServerException", "Message": "x"}}, "ConverseStream")
    assert classify(EventLoopException(broke)) == TrialStatus.INFRASTRUCTURE_INVALID
    refused = ClientError({"Error": {"Code": "ValidationException", "Message": "bad input"}}, "Converse")
    assert classify(refused) == TrialStatus.AGENT_FAILURE  # a request the harness built wrong is the agent's


@tunable()
def poet():
    return Agent(model=Says("one two three"), callback_handler=None)


def test_a_failed_evaluator_measures_nothing():
    trials = {a.case: a for a in trials_of(poet, experiment(WordCount(), Fails(on=["sea"])))}
    assert trials["sea"].status == TrialStatus.COMPLETED and trials["sea"].unavailable == {"Fails"}
    assert "Fails" not in trials["sea"].metrics and trials["sea"].metrics["WordCount"] == 3.0
    only = {a.case: a for a in trials_of(poet, experiment(Fails(on=["sea"])))}
    assert only["sea"].status == TrialStatus.VERIFIER_INVALID and "grader crashed" in only["sea"].error
    assert only["city"].status == TrialStatus.COMPLETED


def test_repetitions_run_every_case_that_many_times_and_record_usage():
    trials = trials_of(poet, experiment(), repetitions=3)
    assert sorted((a.case, a.repetition) for a in trials) == [(c, r) for c in ("city", "sea") for r in (1, 2, 3)]
    a = trials[0]
    assert a.metrics["output_tokens"] == 6 and a.metrics["cache_read_tokens"] == 10
    assert a.metrics["tokens"] == 50 + 6 + 10 and a.metrics["model_calls"] == 1 and a.metrics["seconds"] >= 0


def test_estimates_are_case_means_with_intervals():
    lengths = iter([2, 4, 6, 1, 1, 1])  # sea: 2, 4, 6 -> 4; city: 1, 1, 1 -> 1 (one model call per trial, in order)

    class Varying(Says):
        async def stream(self, *args, **kwargs):
            self.text = " ".join(["w"] * next(lengths))
            async for event in super().stream(*args, **kwargs):
                yield event

    @tunable()
    def varying():
        return Agent(model=Varying(), callback_handler=None)

    study = Study(
        agent=varying,
        experiment=experiment(),
        objectives=Maximize("WordCount"),
        repetitions=3,
        max_concurrency=1,
    )
    study.optimize(iterations=0)
    e = study.candidates[0].estimates["WordCount"]
    assert e.n == 2 and e.mean == pytest.approx(2.5)
    # sd of (4, 1) is 2.1213; se 1.5; t(1) = 12.706 -> half-width 19.06, clipped at 0 below
    assert e.ci[0] == 0.0 and e.ci[1] == pytest.approx(2.5 + 12.706 * 1.5, rel=1e-3)


class BoundedWordCount(WordCount):
    score_range = (0, 1)  # as an evaluator that says nothing else has


def test_a_score_outside_its_range_is_not_measured_and_says_why():
    study = Study(
        agent=poet, experiment=experiment(BoundedWordCount(name="WordCount")), objectives=Maximize("WordCount")
    )
    with pytest.raises(RuntimeError, match=r"WordCount=3.0 is outside its range"):
        study.optimize(iterations=0)  # the baseline measured nothing usable
    trial = study.candidates[0].trials[0]
    assert trial.status.value == "verifier_invalid" and "WordCount" in trial.unavailable
    assert "WordCount" not in trial.metrics and trial.metrics["tokens"] > 0  # usage is kept


def test_an_invalid_candidate_is_not_run():
    @tunable()
    def cannot_build():
        raise RuntimeError("missing credentials for the model")

    study = Study(agent=cannot_build, experiment=experiment(), objectives=Maximize("WordCount"))
    with pytest.raises(RuntimeError, match="baseline.*invalid.*missing credentials"):
        study.optimize(iterations=0)
    assert study.candidates[0].status == "invalid" and study.candidates[0].trials == ()


def test_a_task_runs_the_agent_its_own_way_and_its_output_reaches_the_evaluators():
    def twice(agent, case):  # a trial of two calls, as a simulated user's conversation would be
        agent(case.input)
        return {"output": str(agent("and again"))}

    @tunable()
    def talker():
        return Agent(model=Says("one two three"), callback_handler=None)

    exp = experiment(WordCount(), cases=("sea",))
    t = asyncio.run(run_trials(talker, talker.default_values, exp, repetitions=1, max_concurrency=1, task=twice))[0]
    assert t.status == TrialStatus.COMPLETED and t.output == "one two three\n"
    assert t.metrics["WordCount"] == 3 and t.metrics["model_calls"] == 2


def test_a_trial_whose_task_raises_trial_invalid_runs_again_then_measures_nothing(monkeypatch):
    from strands_evals.optimization import TrialInvalid
    from strands_evals.optimization import trials as trials_module

    monkeypatch.setattr(trials_module, "RETRY_WAIT", 0.0)
    attempts = {"flaky": 0, "down": 0}

    def user_fails(agent, case):
        attempts[case.input] += 1
        if case.input == "down" or attempts[case.input] == 1:
            raise TrialInvalid("the simulated user did not answer")
        return str(agent(case.input))

    @tunable()
    def talker():
        return Agent(model=Says("fine"), callback_handler=None)

    exp = experiment(WordCount(), cases=("flaky", "down"))
    run = run_trials(talker, talker.default_values, exp, repetitions=1, max_concurrency=2, task=user_fails, retries=2)
    by_case = {t.case: t for t in asyncio.run(run)}
    assert attempts == {"flaky": 2, "down": 3}
    assert by_case["flaky"].status == TrialStatus.COMPLETED
    assert by_case["down"].status == TrialStatus.INFRASTRUCTURE_INVALID
    assert "did not answer" in by_case["down"].error and "WordCount" not in by_case["down"].metrics
