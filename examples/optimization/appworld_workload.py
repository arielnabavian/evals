"""The AppWorld workload: a Strands agent as AppWorld's code agent, AppWorld's tasks as cases, and AppWorld's own
evaluation as the score. Not a study: see appworld_harness.py.

Each trial runs one AppWorld task against a fresh environment server of its own (``appworld serve environment``,
from AppWorld 0.1.3 in its own venv, as harness-optimizer's AppWorld runtime does), bound to 127.0.0.1 (``_Server``:
AppWorld's server would bind every interface, and a server that reports another address is stopped before it is used).
The agent's one tool, ``execute``,
runs Python in that task's world; the task passes the world to the agent with the call (``agent(message,
appworld=world)``) and the tool reads it from the call's invocation state, so the tool itself holds no state and its
description can be tuned like any other tool's. When the agent is done, the world is saved and AppWorld evaluates it:
the task's unit tests against the final database.

Configuration, from the environment (``appworld_setup.sh`` builds this layout under ``APPWORLD_HOME``):

    APPWORLD_HOME           where the rest are, unless each is given (default ~/appworld)
    APPWORLD_ROOT           AppWorld's root, holding its data (default $APPWORLD_HOME/root)
    APPWORLD_CACHE          what AppWorld's install unpacked besides the package (default $APPWORLD_HOME/cache)
    APPWORLD_SERVER_ENV     the venv whose ``appworld`` runs each world's server (default $APPWORLD_HOME/venv_server)
    APPWORLD_INSTRUCTIONS   AppWorld's canonical code-agent instructions (default $APPWORLD_HOME/instructions.yaml, the
                            harness-optimizer runtime's ``appworld_code_instructions.yaml``)

The client (``appworld`` 0.2.0, on pydantic 2.12 or 2.13) must be importable where the study runs.
"""

from __future__ import annotations

import json
import os
import random
import select
import signal
import socket
import subprocess
import threading
import time
import uuid
from functools import cache
from pathlib import Path
from typing import Any

from strands import tool
from strands.types.tools import ToolContext

from strands_evals import Case
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import TrialInvalid
from strands_evals.types.evaluation import EvaluationOutput

BASE = Path(os.environ.get("APPWORLD_HOME", str(Path.home() / "appworld"))).expanduser()
ROOT = Path(os.environ.setdefault("APPWORLD_ROOT", str(BASE / "root")))
os.environ.setdefault("APPWORLD_CACHE", str(BASE / "cache"))  # the server inherits the environment
SERVER_ENV = Path(os.environ.get("APPWORLD_SERVER_ENV", str(BASE / "venv_server")))
INSTRUCTIONS_PATH = Path(os.environ.get("APPWORLD_INSTRUCTIONS", str(BASE / "instructions.yaml")))
SPLITS = ("train", "dev", "test_normal", "test_challenge")
OUTPUT_CHARS = 7200  # an execution's output is capped at this, as harness-optimizer's runtime does
START_TIMEOUT = 60.0  # seconds for a world's server to come up (the runtime's 10 s failed under 14-20 starts at once)
STARTS_AT_ONCE = 6  # servers starting at the same time; each start imports AppWorld's apps, which is CPU-heavy
START_ATTEMPTS = 3

__all__ = ["AppWorldRun", "AppWorldTask", "Reward", "TestsPassed", "cases", "execute", "instructions", "task_ids"]

# The per-task user message: the trailing block of AppWorld's canonical prompt (its instructions and worked example
# are the agent's fixed instructions; this carries the task's own details).
USER_MESSAGE = (
    "Using these APIs, now generate code to solve the actual task:\n\n"
    "My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.\n"
    "Task: {instruction}"
)


@cache
def instructions() -> str:
    """AppWorld's canonical code-agent instructions with their worked example: part of the benchmark, never tuned."""
    import yaml

    text = yaml.safe_load(INSTRUCTIONS_PATH.read_text())["system_prompt"]
    head = text.split("Using these APIs, now generate code to solve the actual task:")[0]
    return head.rstrip().removesuffix("USER:").rstrip()


@cache
def task_ids(split: str) -> tuple[str, ...]:
    """The task ids of one of AppWorld's splits, in AppWorld's order."""
    if split not in SPLITS:
        raise ValueError(f"AppWorld's splits are {SPLITS}, not {split!r}")
    path = ROOT / "data" / "datasets" / f"{split}.txt"
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def cases(split: str, n: int | None = None, seed: int = 0) -> list[Case]:
    """The tasks of AppWorld's ``split`` as cases (``n`` of them, drawn with ``seed``, when given). A task's scenario
    (the id before its last ``_``) is in its metadata: AppWorld's scenario success counts scenarios whose every task
    succeeded."""
    ids = list(task_ids(split))
    if n is not None and n < len(ids):
        chosen = set(random.Random(seed).sample(ids, n))
        ids = [i for i in ids if i in chosen]
    return [
        Case(
            name=f"appworld-{i}",
            input=f"AppWorld task {i}",
            metadata={"task_id": i, "split": split, "scenario": i.rsplit("_", 1)[0]},
        )
        for i in ids
    ]


def _capped(text: str) -> str:
    suffix = "... (truncated)"
    return text if len(text) <= OUTPUT_CHARS else text[: OUTPUT_CHARS - len(suffix)] + suffix


@tool(context=True)
def execute(code: str, tool_context: ToolContext) -> str:
    """Execute Python code in the AppWorld environment.

    Args:
        code: Python code to execute in AppWorld

    Returns:
        JSON string with execution output and task completion status
    """
    world = (tool_context.invocation_state or {}).get("appworld")
    if world is None:
        return json.dumps({"error": "No AppWorld instance set", "success": False})
    try:
        output = world.execute(code)
        return _capped(json.dumps({"output": output, "task_completed": world.task_completed(), "success": True}))
    except Exception as exc:  # noqa: BLE001 - the agent's code failed: its error is the tool's result
        return _capped(json.dumps({"error_type": type(exc).__name__, "success": False, "error": str(exc)}))


# ---- a world's environment server -----------------------------------------------------------------------------------
LOOPBACK = "127.0.0.1"
# AppWorld 0.1.3's server binds every interface (``uvicorn.run(app, host="0.0.0.0", ...)``) and takes no option to
# change it. Each world's server runs AppWorld's own command line through this wrapper, which binds it to loopback,
# where the workload reaches it. (Never AppWorld's ``--docker``: it publishes the port on every interface.)
_SERVE_ON_LOOPBACK = f"""
import sys
import uvicorn

_run = uvicorn.run


def _on_loopback(app, *args, **kwargs):
    kwargs["host"] = {LOOPBACK!r}
    return _run(app, *args, **kwargs)


uvicorn.run = _on_loopback
from appworld.cli import main

sys.argv = ["appworld", *sys.argv[1:]]
sys.exit(main())
"""
_starting = threading.Semaphore(STARTS_AT_ONCE)
_ports_lock = threading.Lock()
_ports_taken: set[int] = set()


def _free_port() -> int:
    with _ports_lock:
        for _ in range(1000):
            port = random.randint(20000, 60000)
            if port in _ports_taken:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _ports_taken.add(port)
            return port
    raise RuntimeError("no free port for an AppWorld environment server")


class NotLoopback(RuntimeError):
    """A world's server said it serves somewhere other than 127.0.0.1. It was stopped and its trial fails (never
    retried), so a study whose baseline cannot start a loopback server stops there."""


class _Server:
    """One ``appworld serve environment`` process, on loopback: started, waited on until it serves (and stopped at once
    if it says it serves anywhere but 127.0.0.1), its output drained (it logs every request; a full pipe would block
    it), and stopped with its process group."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def start(self) -> None:
        python = str(SERVER_ENV / "bin" / "python")
        command = [
            python,
            "-c",
            _SERVE_ON_LOOPBACK,
            "serve",
            "environment",
            "--port",
            str(self.port),
            "--no-show-usage",
        ]
        env = {**os.environ, "APPWORLD_ROOT": str(ROOT)}
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
        deadline, seen = time.monotonic() + START_TIMEOUT, []
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited ({self.proc.returncode}): {''.join(seen)[-500:]}")
            ready, _, _ = select.select([self.proc.stdout], [], [], 0.2)
            if ready:
                line = self.proc.stdout.readline()
                seen.append(line)
                if "Uvicorn running on" in line:
                    if f"http://{LOOPBACK}:{self.port}" not in line:
                        self.stop()
                        raise NotLoopback(f"AppWorld's server is not on loopback, so it was stopped: {line.strip()}")
                    threading.Thread(target=self._drain, daemon=True).start()
                    return
        raise RuntimeError(f"the server did not come up in {START_TIMEOUT:.0f} s: {''.join(seen)[-500:]}")

    def _drain(self) -> None:
        try:
            for _ in self.proc.stdout:
                pass
        except (ValueError, OSError):
            pass

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        with _ports_lock:
            _ports_taken.discard(self.port)


def _serve() -> _Server:
    """A started server, or TrialInvalid: a server that will not start is the infrastructure's failure."""
    errors = []
    for _ in range(START_ATTEMPTS):
        server = _Server()
        try:
            with _starting:
                server.start()
            return server
        except NotLoopback:
            raise
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            server.stop()
            errors.append(f"{type(exc).__name__}: {exc}")
    raise TrialInvalid(f"AppWorld's environment server did not start ({START_ATTEMPTS} attempts): {errors[-1]}")


# ---- the study's task -----------------------------------------------------------------------------------------------
class AppWorldRun:
    """A finished task: AppWorld's evaluation of the world the agent left. (A plain object: Strands Evals keeps a
    task's output as it is.)"""

    def __init__(self, task_id: str, evaluation: dict[str, Any], completed: bool) -> None:
        self.task_id, self.evaluation, self.completed = task_id, evaluation, completed

    @property
    def success(self) -> bool:
        return bool(self.evaluation.get("success"))

    @property
    def tests(self) -> tuple[int, int]:
        """(tests passed, tests)."""
        passes = self.evaluation.get("passes") or []
        n = int(self.evaluation.get("num_tests") or 0)
        return (len(passes) if isinstance(passes, list) else int(passes or 0)), n

    def __str__(self) -> str:
        passed, n = self.tests
        state = "completed" if self.completed else "not marked complete"
        return f"{'success' if self.success else 'failure'}: {passed}/{n} tests passed ({state})"


class AppWorldTask:
    """The study's task: one AppWorld task, run by the agent in a fresh world of its own, then evaluated."""

    def __init__(self, experiment: str = "strands-evals") -> None:
        self.experiment = experiment  # AppWorld writes each world's outputs under experiments/outputs/<this>/...

    def __call__(self, agent: Any, case: Case) -> dict[str, Any]:
        from appworld import AppWorld  # optional: only AppWorld studies need its client

        task_id = case.metadata["task_id"]
        server = _serve()
        try:
            try:
                world = AppWorld(
                    task_id=task_id,
                    remote_environment_url=server.url,
                    experiment_name=f"{self.experiment}/{task_id}-{uuid.uuid4().hex[:8]}",
                )
            except Exception as exc:
                raise TrialInvalid(f"AppWorld's world did not load: {type(exc).__name__}: {exc}") from exc
            with world:
                supervisor = world.task.supervisor if isinstance(world.task.supervisor, dict) else {}
                message = USER_MESSAGE.format(
                    first_name=supervisor.get("first_name", ""),
                    last_name=supervisor.get("last_name", ""),
                    email=supervisor.get("email", ""),
                    phone_number=supervisor.get("phone_number", ""),
                    instruction=world.task.instruction,
                )
                agent(message, appworld=world)
                completed = bool(world.task_completed())
                try:
                    world.save()
                except Exception:  # noqa: BLE001 - saving is for inspection; the evaluation does not need it
                    pass
                try:
                    evaluation = world.evaluate().to_dict()
                except Exception as exc:
                    raise TrialInvalid(f"AppWorld's evaluation failed: {type(exc).__name__}: {exc}") from exc
        finally:
            server.stop()
        return {"output": AppWorldRun(task_id, evaluation, completed)}


# ---- scores ---------------------------------------------------------------------------------------------------------
class Reward(Evaluator):
    """AppWorld's task success (TGC): 1 when every one of the task's unit tests passes on the final world, else 0."""

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        run: AppWorldRun = case.actual_output
        return [EvaluationOutput(score=1.0 if run.success else 0.0, test_pass=run.success, reason=str(run))]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)


class TestsPassed(Evaluator):
    """The share of the task's unit tests that pass on the final world: partial credit, finer than success."""

    def evaluate(self, case: Any) -> list[EvaluationOutput]:
        run: AppWorldRun = case.actual_output
        passed, n = run.tests
        share = passed / n if n else 0.0
        return [EvaluationOutput(score=share, test_pass=share == 1.0, reason=f"{passed}/{n} tests passed")]

    async def evaluate_async(self, case: Any) -> list[EvaluationOutput]:
        return self.evaluate(case)
