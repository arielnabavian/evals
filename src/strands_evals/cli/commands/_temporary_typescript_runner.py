from __future__ import annotations

import asyncio
import atexit
import json
import logging
import shlex
import subprocess
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .._entrypoint import EntryPointError

logger = logging.getLogger(__name__)

TYPESCRIPT_SUFFIXES = frozenset({".ts", ".mts", ".cts", ".js", ".mjs", ".cjs"})
MODEL_INSTANCE = object()
_SETUP_TIMEOUT_SECONDS = 300.0
_CLOSE_TIMEOUT_SECONDS = 5.0


class RunnerGone(RuntimeError):
    pass


class TypeScriptRunner:
    def __init__(self, command: list[str]) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as e:
            raise EntryPointError(
                f"cannot start the TypeScript runner {command[0]}: {e}. Restart strands dev, then try again."
            ) from e
        self._pending: dict[int, Future[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._next_id = 0
        threading.Thread(target=self._read_replies, name="typescript-runner", daemon=True).start()
        atexit.register(self.close)

    def _write(self, payload: dict[str, Any]) -> None:
        stdin = self._process.stdin
        if stdin is None or stdin.closed or self._process.poll() is not None:
            raise RunnerGone("the TypeScript runner has exited")
        try:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()
        except OSError as e:
            raise RunnerGone(f"the TypeScript runner stopped reading requests: {e}") from e

    def _request(self, message: dict[str, Any]) -> tuple[int, Future[dict[str, Any]]]:
        future: Future[dict[str, Any]] = Future()
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            self._pending[request_id] = future
            try:
                self._write({**message, "id": request_id})
            except RunnerGone:
                self._pending.pop(request_id)
                raise
        return request_id, future

    def _cancel(self, request_id: int) -> None:
        with self._lock:
            self._pending.pop(request_id, None)
            try:
                self._write({"type": "cancel", "target": request_id})
            except RunnerGone:
                logger.debug("request_id=<%s> | the typescript runner exited before the cancel", request_id)

    def _read_replies(self) -> None:
        try:
            for line in self._process.stdout or ():
                self._deliver(line)
        finally:
            self._fail_pending()

    def _deliver(self, line: str) -> None:
        try:
            reply = json.loads(line)
        except ValueError:
            logger.warning("line=<%s> | ignored a typescript runner line that is not json", line.strip()[:200])
            return
        with self._lock:
            future = self._pending.pop(reply.get("id"), None)
        if future is not None and not future.done():
            future.set_result(reply)

    def _fail_pending(self) -> None:
        try:
            code: int | None = self._process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            code = None
        failure = RunnerGone(f"the TypeScript runner exited with code {code}")
        with self._lock:
            pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(failure)

    def call(self, message: dict[str, Any]) -> dict[str, Any]:
        return self._request(message)[1].result(timeout=_SETUP_TIMEOUT_SECONDS)

    async def call_async(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id, future = self._request(message)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            self._cancel(request_id)
            raise

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            with self._lock:
                self._process.stdin.close()
        try:
            self._process.wait(timeout=_CLOSE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()


class _TypeScriptAgent:
    def __init__(self, runner: TypeScriptRunner, values: dict[str, Any]) -> None:
        self._runner = runner
        self._values = values
        self.messages: list[Any] = []
        self.event_loop_metrics = SimpleNamespace(accumulated_usage={}, cycle_count=0)

    async def invoke_async(self, prompt: Any) -> str:
        from ...optimization.trials import TrialInvalid

        try:
            reply = await self._runner.call_async({"type": "run", "values": self._values, "input": prompt})
        except RunnerGone as e:
            raise TrialInvalid(str(e)) from e
        self.messages = list(reply.get("messages") or [])
        self.event_loop_metrics = SimpleNamespace(
            accumulated_usage=dict(reply.get("usage") or {}), cycle_count=int(reply.get("cycles") or 0)
        )
        if reply.get("error") and reply.get("retryable"):
            raise TrialInvalid(f"the TypeScript agent's model call failed: {reply['error']}")
        if reply.get("error"):
            raise RuntimeError(f"the TypeScript agent failed: {reply['error']}")
        return str(reply.get("output") or "")


@dataclass(frozen=True)
class TypeScriptHarness:
    runner: TypeScriptRunner
    options: dict[str, Any]

    def build(self, values: dict[str, Any]) -> _TypeScriptAgent:
        if values.get("instructions") == "":
            values = {**values, "instructions": None}
        return _TypeScriptAgent(self.runner, values)

    def builds_with(self, **values: Any) -> bool:
        return bool(self.runner.call({"type": "probe", "values": values}).get("ok"))


def _python_options(description: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "instructions": description.get("instructions"),
        "effort": description.get("effort") or "auto",
    }
    if description.get("model_is_instance"):
        options["model"] = MODEL_INSTANCE
    elif description.get("model"):
        options["model"] = description["model"]
    if description.get("system_prompt"):
        options["system_prompt"] = True
    return options


def load_typescript_harness(path_text: str, runner_command: str | None) -> TypeScriptHarness:
    if not runner_command:
        raise EntryPointError(
            f"{path_text} is a TypeScript agent; pass --typescript-runner COMMAND to start the runner that builds it"
        )
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise EntryPointError(f"cannot find the TypeScript agent {path_text}. Check the path, then try again.")
    runner = TypeScriptRunner([*shlex.split(runner_command), str(path), str(Path.cwd())])
    try:
        description = runner.call({"type": "describe"})
    except Exception as e:
        runner.close()
        raise EntryPointError(
            f"{path_text} could not be loaded: {e}. Fix that error, check that strands dev can open this agent, "
            "then try again."
        ) from e
    if description.get("error"):
        runner.close()
        raise EntryPointError(
            f"{path_text} could not be loaded: {description['error']}. Fix that error, check that strands dev can "
            "open this agent, then try again."
        )
    shared = list(description.get("shared_state") or [])
    if shared:
        runner.close()
        raise EntryPointError(
            f"{path_text} passes {', '.join(shared)} to createHarness(...); "
            "every trial would share it, so remove it to optimize this agent"
        )
    return TypeScriptHarness(runner, _python_options(description))
