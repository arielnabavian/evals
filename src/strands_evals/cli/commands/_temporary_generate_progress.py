from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from strands import Agent

from ...case import Case
from ...evaluators.evaluator import Evaluator
from ...generators import ExperimentGenerator

logger = logging.getLogger(__name__)

_ERROR_TEXT_LIMIT = 500


def temporary_error_text(error: BaseException) -> str:
    text = " ".join(str(error).split())
    summary = f"{type(error).__name__}: {text}" if text else type(error).__name__
    return summary if len(summary) <= _ERROR_TEXT_LIMIT else f"{summary[: _ERROR_TEXT_LIMIT - 1]}…"


class TemporaryGenerateProgress:
    def __init__(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._stream = target.open("w", encoding="utf-8")
        self._total = 0
        self._done = 0
        self._failed = 0

    def topics(self) -> None:
        self._write({"event": "topics"})

    def cases(self, total: int) -> None:
        self._total = total
        self._write({"event": "cases", "total": total})

    def case_done(self) -> None:
        self._done += 1
        self._write(self._case_event(ok=True))

    def case_failed(self, error: BaseException) -> None:
        self._failed += 1
        self._write({**self._case_event(ok=False), "error": temporary_error_text(error)})

    def rubric(self) -> None:
        self._write({"event": "rubric"})

    def failed(self, error: BaseException) -> None:
        self._write({"event": "failed", "error": temporary_error_text(error)})

    def close(self) -> None:
        self._stream.close()

    def _case_event(self, ok: bool) -> dict[str, Any]:
        return {"event": "case", "ok": ok, "done": self._done, "failed": self._failed, "total": self._total}

    def _write(self, event: dict[str, Any]) -> None:
        self._stream.write(f"{json.dumps(event, ensure_ascii=False)}\n")
        self._stream.flush()


class TemporaryProgressGenerator(ExperimentGenerator[str, str]):
    def __init__(self, progress: TemporaryGenerateProgress, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._progress = progress

    async def _prepare_generation_prompts(
        self, base_prompt: str, num_cases: int, num_topics: int | None = None
    ) -> list[tuple[str, int]]:
        if num_topics is not None:
            self._progress.topics()
        prompt_specs = await super()._prepare_generation_prompts(base_prompt, num_cases, num_topics)
        self._progress.cases(sum(count for _prompt, count in prompt_specs))
        return prompt_specs

    async def _case_worker(self, queue: asyncio.Queue, prompt: str, message_history: list | None, results: list):
        try:
            case_generator = Agent(
                model=self.model,
                system_prompt=self.case_system_prompt,
                callback_handler=None,
                messages=message_history if message_history else [],
            )
        except Exception as error:
            logger.exception("Could not start the case generator agent")
            self._fail_queued_cases(queue, error)
            return

        while True:
            try:
                difficulty = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                full_prompt = prompt + f"Ensure that the test case has a difficulty level of {difficulty}."
                gen_case = await case_generator.structured_output_async(self._Case, full_prompt)
                results.append(Case(**gen_case.model_dump()))
                self._progress.case_done()
            except Exception as error:
                logger.exception(f"Error generating case: {error}")
                self._progress.case_failed(error)
            finally:
                queue.task_done()

    def _fail_queued_cases(self, queue: asyncio.Queue, error: BaseException) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._progress.case_failed(error)
            queue.task_done()

    async def construct_evaluator_async(
        self, prompt: str, evaluator: Evaluator, message_history: list | None = None
    ) -> Evaluator:
        self._progress.rubric()
        return await super().construct_evaluator_async(prompt, evaluator, message_history)
