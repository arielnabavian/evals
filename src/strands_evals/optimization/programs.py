"""Program-valued variables: a class implementing an interface, whose versions a proposer agent writes.

    manager = Program(ConversationManager, seeds=[SOURCE], about="how the agent trims its history")

    @tunable(conversation_manager=manager)                       # standalone: the baseline is the seed
    @tunable(conversation_manager=OneOf(component(SlidingWindowConversationManager, ...), manager))  # beside a manager

    macro = Program(Macro, seeds=[EMPTY], imports=("re", "math"), standard=False)  # the baseline has no program

- ``Program`` is a component: its constructor loads the version its one variable (``<key>.version``) names, so it fits
  wherever a component does (``OneOf``'s label for it: ``program``). The factory receives an instance of the version's
  class, or None for the empty program (``EMPTY``, a seed that leaves the agent as it is).
- A version's value is ``programs/<hash>``: the content hash of its source, stored once in the study's artifact store
  (``programs/<hash>/program.py`` and a manifest), as a skill library is (``skills/<hash>``); the empty program's value
  is ``programs/none``. Seeds are stored the first time a study resolves them.
- Loading is restricted (an import allowlist, each allowed module seen only through its public names, no file / exec /
  dunder / frame access, no bare ``except``, a small set of builtins) and checked against the interface (one concrete
  subclass, constructible with no arguments, and the declaration's own ``check``). A version that fails either makes its
  candidate invalid (never measured, so never a zero). This is not a sandbox: isolation (a worker process with a
  deadline) is the additive next step at this same boundary. ``run_bounded`` gives a call into a loaded program a budget
  of executed lines and seconds.
- ``ProgramProposer`` is the default method for programs (``register_kind``); ``Block(ProgramProposer(...), manager)``
  overrides it. Its agent is given the interface, the environment (``context``), the held version, every version
  measured so far with its goal metrics, and how the held version's trials went: the reflection digest of every trial
  and the transcripts of those that did not succeed. A version that does not load is sent back with the error, in the
  same conversation (``load_retries``). Its agent's tokens are its ``origin["usage"]``, so the study counts them as
  ``method:<block>``.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.config import Config as BotocoreConfig
from strands import Agent
from strands.models import BedrockModel
from strands.types.exceptions import MaxTokensReachedException

from .methods import Draw, Method, _store_digest, register_kind, usage
from .space import Component, Domain
from .space.space import ParamInfo

if TYPE_CHECKING:
    from .study import Candidate, Study

KIND = "programs"  # the artifact store's folder of program versions, and the default block's name
EMPTY = ""  # the empty program: a seed that builds the agent without one (the factory receives None)
EMPTY_VALUE = f"{KIND}/none"
DEFAULT_MODEL = "global.anthropic.claude-sonnet-5-5"
SAFE_MODULES = ("__future__", "typing", "math", "copy", "json", "dataclasses", "collections", "itertools", "functools")
SAFE_BUILTINS = (
    "abs all any bool dict enumerate filter float frozenset hasattr int isinstance issubclass iter len list map "
    "max min next object property range repr reversed round set slice sorted staticmethod classmethod str sum super "
    "tuple zip "
    "Exception ValueError TypeError KeyError IndexError RuntimeError NotImplementedError AttributeError"
).split()
FORBIDDEN = {"open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "input"}
FORBIDDEN |= {"breakpoint", "help", "exit", "quit", "memoryview", "format", "format_map", "vformat", "Formatter"}
FORBIDDEN |= {"BaseException", "mro", "with_traceback"}
# a frame reaches every caller's globals: no attribute that leads to one
FORBIDDEN |= {"gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "tb_frame", "tb_next"}
FORBIDDEN |= {"f_back", "f_globals", "f_locals", "f_builtins", "f_code"}
DUNDERS_ALLOWED = {"__init__", "__name__", "__class__"}
LINES = 2_000_000  # run_bounded: the lines a call may execute in the program's own code
SECONDS = 120.0  # run_bounded: the seconds a call may take
LOAD_LINES, LOAD_SECONDS = 1_000_000, 30.0  # load: running a version's module, and constructing its class


# ---- storing and loading versions ------------------------------------------------------------------------------------
def value_of(source: str) -> str:
    """A version's value: the content hash of its source, under the store's folder (``programs/none``: the empty
    program)."""
    if source == EMPTY:
        return EMPTY_VALUE
    return f"{KIND}/{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def store(source: str, artifacts: str | Path, manifest: dict[str, Any]) -> str:
    """Write a version once into the study's artifact store (``programs/<hash>/``) and return its value."""
    value = value_of(source)
    target = Path(artifacts) / value
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=target.parent))
        (tmp / "program.py").write_text(source)
        (tmp / "manifest.json").write_text(json.dumps({**manifest, "value": value}, indent=1))
        try:
            os.replace(tmp, target)
        except OSError:  # another proposal stored the same source first
            shutil.rmtree(tmp, ignore_errors=True)
    return value


def _check_source(tree: ast.AST, imports: Sequence[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ImportError("relative imports are not allowed")
            for name in names:
                if not _allowed(name, imports):
                    raise ImportError(f"import of {name!r} is not allowed (allowed: {', '.join(imports)})")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN or _dunder(node.id) and node.id != "__name__"):
            raise NameError(f"{node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute) and (
            node.attr in FORBIDDEN or _dunder(node.attr) and node.attr not in DUNDERS_ALLOWED
        ):
            raise AttributeError(f"attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            raise SyntaxError(f"a bare `except:` is not allowed (line {node.lineno}): catch Exception or narrower")


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _allowed(name: str, imports: Sequence[str]) -> bool:
    return any(name == m or name.startswith(m + ".") for m in imports)


def _public(module: types.ModuleType, imports: tuple[str, ...]) -> types.ModuleType:
    """``module`` as a program sees it: its public names only (``__all__``, else those without a leading underscore),
    without the modules it imported (``typing.sys``, ``dataclasses.inspect``) unless they are allowed too."""
    view = types.ModuleType(module.__name__)
    names = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, types.ModuleType):
            if not _allowed(value.__name__, imports):
                continue
            value = _public(value, imports)
        setattr(view, name, value)
    return view


def _guarded_import(imports: tuple[str, ...]) -> Callable[..., Any]:
    def guarded_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if level or not _allowed(name, imports):
            raise ImportError(f"import of {name!r} is not allowed")
        builtins.__import__(name, globals, locals, fromlist, level)
        if fromlist:  # from a.b import c: the module a.b, with any allowed submodule c
            view = _public(sys.modules[name], imports)
            for item in fromlist:
                child = f"{name}.{item}"
                if not hasattr(view, item) and _allowed(child, imports) and child in sys.modules:
                    setattr(view, item, _public(sys.modules[child], imports))
            return view
        parts = name.split(".")  # import a.b: the package a, with a.b reachable from it
        top = _public(sys.modules[parts[0]], imports)
        at = top
        for i in range(1, len(parts)):
            sub = _public(sys.modules[".".join(parts[: i + 1])], imports)
            setattr(at, parts[i], sub)
            at = sub
        return top

    return guarded_import


@lru_cache(maxsize=256)
def load(source: str, interface: type, imports: tuple[str, ...]) -> type:
    """The one concrete subclass of ``interface`` that ``source`` defines, loaded under the restrictions above; raises
    (any exception) when the source breaks a restriction or does not implement the interface. The interface is in the
    module's namespace under its own name, so a version may subclass it without importing it. Running the module and
    constructing the class are bounded (``run_bounded``: ``LOAD_LINES``, ``LOAD_SECONDS``): over it, a TimeoutError."""
    tree = ast.parse(source)
    _check_source(tree, imports)
    safe = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
    safe.update(__build_class__=builtins.__build_class__, __import__=_guarded_import(imports))
    module = f"strands_evals_program_{hashlib.sha256(source.encode()).hexdigest()[:16]}"
    loaded = types.ModuleType(module)  # in sys.modules, as dataclasses and typing look a class's module up there
    loaded.__dict__.update({"__builtins__": safe, interface.__name__: interface})
    sys.modules[module] = loaded
    namespace = loaded.__dict__
    code = compile(tree, f"<{module}>", "exec")
    try:
        # restricted above; isolation is the next step
        run_bounded(lambda: exec(code, namespace), lines=LOAD_LINES, seconds=LOAD_SECONDS)  # noqa: S102
        found = [
            v for v in namespace.values() if isinstance(v, type) and issubclass(v, interface) and v.__module__ == module
        ]
        if len(found) != 1:
            raise TypeError(f"expected one subclass of {interface.__name__}, found {[c.__name__ for c in found]}")
        cls = found[0]
        if inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} does not implement {sorted(cls.__abstractmethods__)}")
        run_bounded(cls, lines=LOAD_LINES, seconds=LOAD_SECONDS)  # constructible with no arguments
    except ProgramTimeout as exc:
        sys.modules.pop(module, None)
        raise TimeoutError(f"loading it: {exc}") from None
    except BaseException:
        sys.modules.pop(module, None)
        raise
    return cls


class ProgramTimeout(BaseException):  # noqa: N818 - a BaseException, so a program's `except Exception` cannot keep it
    """A call into a program ran more lines or seconds than ``run_bounded`` allowed."""


def run_bounded(call: Callable[[], Any], *, lines: int = LINES, seconds: float = SECONDS) -> Any:
    """``call()``, stopped with ``ProgramTimeout`` once the program's own code (the modules ``load`` makes) has run
    ``lines`` lines or the call has taken ``seconds``: a loop in a version cannot hold a trial forever. Time spent
    outside the program's code (a tool it calls) counts toward ``seconds``, not ``lines``. Traces the calling thread
    only, and puts back the trace it found."""
    start, count = time.monotonic(), 0

    def local(frame: Any, event: str, arg: Any) -> Any:
        nonlocal count
        if event == "line":
            count += 1
            if count > lines:
                raise ProgramTimeout(f"the program ran more than {lines:,} lines")
            if count % 1000 == 0 and time.monotonic() - start > seconds:
                raise ProgramTimeout(f"the program ran for more than {seconds:g} seconds")
        return local

    def calls(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename.startswith("<strands_evals_program_"):
            if time.monotonic() - start > seconds:
                raise ProgramTimeout(f"the program ran for more than {seconds:g} seconds")
            return local
        return None

    previous = sys.gettrace()
    sys.settrace(calls)
    try:
        return call()
    finally:
        sys.settrace(previous)


# ---- the domain and the declaration ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProgramVersion(Domain[str]):
    """The variable a ``Program`` declares: which version of the program the agent is built with."""

    interface: type
    seeds: tuple[str, ...]
    about: str | None = None
    imports: tuple[str, ...] = ()
    context: str | None = None  # the environment the program runs in, as its proposer is told it
    check: Callable[[type], str | None] | None = None  # a loaded class's own check: why it cannot be used, or None
    transcript: Callable[[Any, int], str] | None = None  # a trial as its proposer reads it, cut to a length
    settings: tuple[tuple[str, Any], ...] = ()  # the declaration's own settings, plain JSON, kept in the record
    written = True  # a model writes its values: never sampled, modelled as rewritten or not

    @property
    def default(self) -> str:  # type: ignore[override]
        return value_of(self.seeds[0])

    def param_info(self, key: str, active_when: tuple[str, str] | None) -> ParamInfo:
        return ParamInfo(key, "program", self.default, active_when=active_when, about=self.about, domain=self)

    def suggest(self, trial: Any, name: str) -> str:
        ask = getattr(trial, "suggest_program", None)
        if ask is None:
            raise TypeError(f"{name} is a program: it needs a method that writes programs (Study), not a sampler")
        return ask(name, self.default)

    def record_settings(self) -> dict[str, Any]:
        """What the record's space row keeps of the declaration: its interface and its settings."""
        return {"interface": f"{self.interface.__module__}:{self.interface.__qualname__}", **dict(self.settings)}

    def manifest(self, **extra: Any) -> dict[str, Any]:
        return {"interface": f"{self.interface.__module__}:{self.interface.__qualname__}", **extra}

    def resolve(self, value: str, artifacts: str | Path | None) -> str:
        """The stored version's directory (a seed is stored the first time it is resolved)."""
        if artifacts is None or os.path.isabs(value):
            return value
        path = Path(artifacts) / value
        seed = {value_of(s): s for s in self.seeds}.get(value)
        if not path.exists() and seed is not None:
            store(seed, artifacts, self.manifest(seed=True))
        return str(path.resolve())

    def source(self, value: str, artifacts: str | Path | None = None) -> str:
        seed = {value_of(s): s for s in self.seeds}.get(value)
        path = Path(self.resolve(value, artifacts)) / "program.py"
        if path.is_file():
            return path.read_text()
        if seed is not None:
            return seed
        raise FileNotFoundError(f"no stored program version {value!r}")

    def load_source(self, source: str) -> type | None:
        """The class ``source`` defines, loaded and checked (None for the empty program, when it is a seed); raises
        when it cannot be used. Checked once per source (a check may run the class)."""
        if source == EMPTY and EMPTY in self.seeds:
            return None
        return _checked(self, source)

    def problem(self, source: str) -> str | None:
        """Why ``source`` cannot be used, or None."""
        try:
            self.load_source(source)
        except Exception as exc:  # noqa: BLE001 - any failure to load or check is the reason
            return f"{type(exc).__name__}: {exc}"
        return None

    def check_value(self, value: Any, artifacts: str | Path | None) -> str | None:
        if not isinstance(value, str):
            return f"{value!r}: expected a program version ({KIND}/<hash>)"
        try:
            source = self.source(value, artifacts)
        except Exception as exc:  # noqa: BLE001 - a version that cannot be read makes the candidate invalid
            return f"program {os.path.basename(value)} does not load: {type(exc).__name__}: {exc}"
        problem = self.problem(source)
        return f"program {os.path.basename(value)} does not load: {problem}" if problem else None


@lru_cache(maxsize=256)
def _checked(version: ProgramVersion, source: str) -> type:
    cls = load(source, version.interface, version.imports)
    problem = version.check(cls) if version.check is not None else None
    if problem:
        raise TypeError(problem)
    return cls


class _Build:
    """A program's constructor: the agent receives a fresh instance of the version's class (None: the empty
    program)."""

    def __init__(self, version: ProgramVersion, label: str) -> None:
        self.version, self.__name__ = version, label  # the label a OneOf gives this option

    def __call__(self, version: str) -> Any:
        cls = self.version.load_source(self.version.source(version))
        return None if cls is None else cls()


def Program(  # noqa: N802 - a declaration, read like the domains
    interface: type,
    *,
    seeds: Sequence[str],
    about: str | None = None,
    imports: Sequence[str] = (),
    standard: bool = True,
    context: str | None = None,
    check: Callable[[type], str | None] | None = None,
    transcript: Callable[[Any, int], str] | None = None,
    settings: Mapping[str, Any] | None = None,
    label: str = "program",
) -> Component[Any]:
    """A program the agent is built with: a class implementing ``interface``, whose versions a proposer writes.

    Args:
        interface: The class every version subclasses (in each version's namespace under its own name).
        seeds: The starting source(s), the first the default; ``EMPTY`` is the empty program (the factory receives
            None), so a baseline can run without one.
        about: What the program is for, for the model that writes it.
        imports: Modules a version may import, besides the standard ones (``standard``).
        standard: Allow, besides ``imports``, a few modules of the standard library (``SAFE_MODULES``), the interface's
            own package and ``strands.types``. False: ``imports`` only.
        context: The environment the program runs in, as its proposer is told it (commands, observations, tasks).
        check: A loaded class's own check: why it cannot be used, or None. A version it refuses is invalid.
        transcript: How the proposer reads one trial: given the trial and a length, its text (default: its messages,
            ``messages_transcript``).
        settings: The declaration's own settings (plain JSON), kept in the record's space row with the interface, so
            records of differently constrained programs can be told apart.
        label: Its option's label in a ``OneOf``.
    """
    if not seeds:
        raise ValueError("Program needs at least one seed")
    base = (*SAFE_MODULES, interface.__module__.rsplit(".", 1)[0], "strands.types") if standard else ()
    version = ProgramVersion(
        interface,
        tuple(seeds),
        about,
        tuple(dict.fromkeys((*base, *imports))),
        context,
        check,
        transcript,
        tuple((settings or {}).items()),
    )
    return Component(constructor=_Build(version, label), arguments={"version": version})


# ---- what the proposer reads of a trial ------------------------------------------------------------------------------
def _cut(lines: list[str], limit: int) -> str:
    """``lines`` joined, within ``limit`` characters: the first two lines (the task) and as many of the last as fit, the
    lines between left out and counted."""
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    head = lines[:2]
    budget = limit - sum(len(x) + 1 for x in head) - 60
    tail: list[str] = []
    for line in reversed(lines[2:]):
        if budget - len(line) - 1 < 0:
            break
        tail.insert(0, line)
        budget -= len(line) + 1
    left = len(lines) - len(head) - len(tail)
    return "\n".join([*head, f"... ({left} lines left out) ...", *tail])


def messages_transcript(messages: Sequence[Any], limit: int, inner: Mapping[str, Sequence[str]] | None = None) -> str:
    """A conversation as lines, within ``limit`` characters (``_cut``): each text, tool call and tool result on its own
    line (each cut to 600 characters). ``inner``: lines to show under a tool call, by its ``toolUseId`` (what a tool did
    that its result does not say)."""
    from ._reflection.agentic import _short  # as methods.py does: importing _reflection sets BYPASS_TOOL_CONSENT

    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        for b in m.get("content") or []:
            if not isinstance(b, dict):
                continue
            if isinstance(b.get("text"), str) and b["text"].strip():
                lines.append(f"[{role}] {_short(b['text'], 600)}")
            elif "toolUse" in b:
                use = b["toolUse"]
                lines.append(f"[call] {use.get('name')} {_short(json.dumps(use.get('input'), default=str), 600)}")
                lines += [f"    {_short(x, 600)}" for x in (inner or {}).get(use.get("toolUseId", ""), ())]
            elif "toolResult" in b:
                result = b["toolResult"]
                text = " ".join(
                    x["text"]
                    for x in result.get("content") or []
                    if isinstance(x, dict) and isinstance(x.get("text"), str)
                )
                status = " (error)" if result.get("status") == "error" else ""
                lines.append(f"[result{status}] {_short(text, 600)}")
    return _cut(lines, limit)


# ---- the proposer ----------------------------------------------------------------------------------------------------
SYSTEM = (
    "You write Python modules, each defining one class that implements a given interface. Reply with the complete "
    "module in a single ```python block."
)
DIGEST_CHARS = 12000  # the digest of every trial of the held version (``digest_text``)
TRANSCRIPT_CHARS = 40000  # the transcripts of its trials that did not succeed, together
TRANSCRIPT_MIN = 1500  # the shortest a transcript is cut to: fewer are shown instead
# the default model's client: a module of several thousand tokens is one reply
BOTO_CONFIG = BotocoreConfig(connect_timeout=60, read_timeout=600, retries={"max_attempts": 3, "mode": "adaptive"})
SUCCESS = 0.5  # a trial's reward below this did not succeed (as reflection's digest counts them)
WHERE = "The transcripts of the conversations that did not succeed follow the digest, under the same names."


def _metrics(m: Any) -> dict[str, Any]:
    u = getattr(m, "accumulated_usage", None) or {}
    return {
        "cycle_count": getattr(m, "cycle_count", 0),
        "input_tokens": u.get("inputTokens", 0),
        "output_tokens": u.get("outputTokens", 0),
        "cache_read_tokens": u.get("cacheReadInputTokens", 0),
        "cache_write_tokens": u.get("cacheWriteInputTokens", 0),
    }


def _class_name(source: str) -> str | None:
    try:
        return next((n.name for n in ast.parse(source).body if isinstance(n, ast.ClassDef)), None)
    except SyntaxError:
        return None


class _ProgramDraw(Draw):
    def __init__(self, values: dict[str, str], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_program(self, name: str, default: str) -> str:
        return self.values.get(name, default)


class ProgramProposer(Method):
    """Writes a new version of each program in its block with an agent, given the interface, the environment it runs
    in, the held version's source (or that there is none), every version measured so far with its goal metrics, and
    the held candidate's trials: reflection's digest of every trial and the transcripts of those that did not succeed.
    A seed not yet measured is proposed first, with no model call. A version that does not load is sent back with the
    error, in the same conversation, up to ``load_retries`` times, as is a reply cut off at the output limit; one still
    failing is stored and its candidate is invalid.

    Args:
        model: A Strands model, or a function returning a fresh one per proposal (default: Bedrock, ``DEFAULT_MODEL``).
        model_id: The model's id, for the record.
        max_tokens: The default model's output limit per reply (a module is written in one reply).
        load_retries: How many times a version that does not load, or a reply cut off at the output limit, is sent back
            (0: never).
        digest_chars: The digest's length limit.
        transcript_chars: The transcripts' length limit, together; each is cut to an equal share, never below
            ``TRANSCRIPT_MIN`` (fewer are shown instead).
    """

    kinds = ("program",)
    concurrent = True  # each proposal runs its own agent

    def __init__(
        self,
        model: Any = None,
        *,
        model_id: str | None = None,
        max_tokens: int = 32000,
        load_retries: int = 2,
        digest_chars: int = DIGEST_CHARS,
        transcript_chars: int = TRANSCRIPT_CHARS,
    ) -> None:
        if load_retries < 0:
            raise ValueError("ProgramProposer(load_retries=...) must be 0 or more")
        self.model, self._model_id = model, model_id or (DEFAULT_MODEL if model is None else None)
        self.max_tokens, self.load_retries = max_tokens, load_retries
        self.digest_chars, self.transcript_chars = digest_chars, transcript_chars

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def describe(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "model": self._model_id,
            **({"max_tokens": self.max_tokens} if self.model is None else {}),
            "load_retries": self.load_retries,
            "digest_chars": self.digest_chars,
            "transcript_chars": self.transcript_chars,
        }

    def _model(self) -> Any:
        if self.model is None:
            return BedrockModel(
                model_id=DEFAULT_MODEL,
                region_name=os.environ.get("AWS_REGION", "us-west-2"),
                max_tokens=self.max_tokens,
                boto_client_config=BOTO_CONFIG,
            )
        return self.model() if callable(self.model) and not hasattr(self.model, "stream") else self.model

    # ---- the task message
    def task(self, version: ProgramVersion, held: str, measured: Sequence[str], trials: str = "") -> str:
        """The task message: the rules, the interface, the environment, the held version, the versions measured so far
        (``measured``: one line each) and how the held version's trials went (``trials``)."""
        name = version.interface.__name__
        interface = inspect.getsource(version.interface)[:8000]
        lines = [
            f"Write a new version of a {name} ({version.interface.__module__}).",
            f"What it is for: {version.about or '-'}",
            "",
            "## Rules",
            f"- Import only from: {', '.join(version.imports)}. Nothing else is importable.",
            "- No files, processes, network, exec/eval, getattr/setattr, or double-underscore attributes other than "
            "__init__/__name__/__class__; no bare `except:` (catch Exception or narrower).",
            f"- Builtins available: {', '.join(SAFE_BUILTINS)} (no print).",
            f"- `{name}` is already defined in the module's namespace: subclass it without importing it.",
            f"- Define exactly one subclass of {name} that implements every abstract method and can be constructed "
            "with no arguments.",
            "- Reply with the complete module in a single ```python block, after at most a few sentences on what you "
            "change and why.",
            "",
            f"## The interface\n```python\n{interface}\n```",
        ]
        if version.context:
            lines += ["", "## The environment", version.context.strip()]
        lines.append("")
        if held == EMPTY:
            lines += [
                "## The held version",
                "There is none: the agent runs without the program (that is the baseline). Write the first version.",
            ]
        else:
            lines += ["## The held version (improve it, or replace it)", f"```python\n{held}\n```"]
        lines += ["", "## Versions measured so far" if measured else "No version has been measured yet.", *measured]
        if trials:
            lines += ["", "## How the held version did", trials]
        return "\n".join(lines)

    def measured(
        self, key: str, version: ProgramVersion, base: Candidate, history: Sequence[Candidate], study: Study
    ) -> list[str]:
        """One line per candidate that has ``key``: its version (and class), its goal metrics, or why it is invalid."""
        from ._reflection.agentic import _short  # as methods.py does: importing _reflection sets BYPASS_TOOL_CONSENT

        metrics = list(study.projection.goal_metrics)
        directions = dict(
            zip([o.metric for o in study.projection.objectives], study.projection.directions, strict=True)
        )
        out = [
            "Goal metrics: "
            + ", ".join(f"{m} ({directions.get(m, 'a constraint')})" for m in metrics)
            + ". Candidates in the order they were measured:"
        ]
        for c in history:
            if key not in c.params:
                continue
            value = c.params[key]
            try:
                cls = (
                    "the empty program" if value == EMPTY_VALUE else _class_name(version.source(value, study.artifacts))
                )
            except Exception:  # noqa: BLE001 - a version that cannot be read is named by its value only
                cls = None
            marks = [m for m, on in (("held", c.number == base.number), ("baseline", c.base is None)) if on]
            line = f"- candidate {c.number}: {os.path.basename(value)}" + (f" ({cls})" if cls else "")
            line += f" [{', '.join(marks)}]" if marks else ""
            if c.status == "invalid":
                line += f": invalid ({_short(c.reason, 400)})"
            else:
                line += ": " + ", ".join(f"{m}={_value(c.metrics.get(m))}" for m in metrics)
            out.append(line)
        return out

    def trials(self, version: ProgramVersion, base: Candidate, study: Study, objective: int | None) -> tuple[str, int]:
        """How the held candidate's trials went: reflection's digest of every trial, then the transcripts of those
        that did not succeed (one per case first, then the rest, while they fit). With how many trials it covers."""
        from ._reflection import Reward, Rollout  # as methods.py does: importing _reflection sets BYPASS_TOOL_CONSENT
        from ._reflection.agentic import _digest_row, digest_text

        aim = study.aim(objective)
        rewards = study.projection.rewards(base, aim, objective, metrics=aim.per_trial(study.projection))
        if not rewards:
            return "", 0
        cases = {case.name: case for case in study.experiment.cases}
        rows, named = [], []
        for i, (trial, reward, metadata) in enumerate(rewards):
            name = f"trial-{i + 1:03d}"
            case = cases.get(trial.case)
            rollout = Rollout(
                data_sample={
                    "name": trial.case,
                    "repetition": trial.repetition,
                    "input": None if case is None else case.input,
                },
                messages=list(trial.messages),
                metadata={"output": trial.output},
            )
            rows.append(_digest_row(name, rollout, reward, Reward(reward=reward, metadata=metadata), trial.output))
            named.append((name, trial, reward))
        digest = digest_text(rows, SUCCESS, self.digest_chars, where=WHERE)
        failing = [(n, t, r) for n, t, r in named if r < SUCCESS]
        by_case: dict[str, list[tuple[str, Any, float]]] = {}
        for item in failing:
            by_case.setdefault(item[1].case, []).append(item)
        depth = max((len(v) for v in by_case.values()), default=0)
        order = [v[i] for i in range(depth) for v in by_case.values() if i < len(v)]
        shown = order[: max(1, self.transcript_chars // TRANSCRIPT_MIN)] if order else []
        each = self.transcript_chars // max(1, len(shown))
        render = version.transcript or (lambda trial, limit: messages_transcript(trial.messages, limit))
        parts = [digest]
        if shown:
            parts.append(
                f"## Transcripts of the trials that did not succeed: {len(shown)} of {len(failing)}, each cut to "
                f"{each:,} characters"
            )
            for name, trial, reward in shown:
                parts += [
                    f"### {name}: {trial.case}, repetition {trial.repetition}, reward {reward:.3g}",
                    render(trial, each),
                ]
        return "\n".join(parts), len(rewards)

    # ---- one proposal
    def write(
        self, version: ProgramVersion, held: str, measured: Sequence[str], trials: str = ""
    ) -> tuple[str | None, dict[str, Any]]:
        """One agent conversation: the source it wrote (None when it failed) and the origin it is recorded with. A
        version that does not load is sent back with the error, up to ``load_retries`` times."""
        agent = Agent(model=self._model(), system_prompt=SYSTEM, callback_handler=None)
        t0, revised = time.time(), []
        message = self.task(version, held, measured, trials)
        source = None
        try:
            for attempt in range(self.load_retries + 1):
                try:
                    text = str(agent(message))
                except MaxTokensReachedException:  # cut off at the output limit: asked again, shorter
                    if attempt == self.load_retries:
                        raise
                    revised.append("the reply was cut off at the output limit")
                    message = (
                        "Your reply was cut off at the output limit. Reply again with only the complete module, "
                        "shorter, in a single ```python block."
                    )
                    continue
                blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
                source = (blocks[-1] if blocks else text).strip() + "\n"
                problem = version.problem(source)
                if problem is None or attempt == self.load_retries:
                    break
                revised.append(problem[:1000])
                message = (
                    f"That module cannot be used: {problem}\nFix it and reply with the complete corrected module in a "
                    "single ```python block."
                )
        except Exception as exc:  # noqa: BLE001 - a failed proposal keeps the held version; the study goes on
            error = f"{type(exc).__name__}: {exc}"[:2000]
            spent = usage(_metrics(agent.event_loop_metrics))  # every call of the conversation, the failed one too
            return None, {"status": "failed", "error": error, "usage": spent, "seconds": time.time() - t0}
        # the agent's metrics accumulate over its calls: the conversation's usage, every send-back included
        origin: dict[str, Any] = {
            "status": "written",
            "usage": usage(_metrics(agent.event_loop_metrics)),
            "seconds": time.time() - t0,
        }
        if revised:  # versions sent back because they did not load: why
            origin["revised"] = revised
        return source, origin

    def begin(
        self,
        block: Any,
        base: Candidate | None,
        history: Sequence[Candidate],
        structure: dict[str, Any],
        study: Study,
        objective: int | None,
    ) -> Draw:
        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _ProgramDraw({}, {"method": name, "from_candidate": None})
        values: dict[str, str] = {}
        origin: dict[str, Any] = {"method": name, "from_candidate": base.number}
        for key in keys:  # one program per key; a block usually holds one
            version: ProgramVersion = study.params[key].domain
            seen = {c.params.get(key) for c in history}
            seed = next((value_of(s) for s in version.seeds if value_of(s) not in seen), None)
            if seed is not None:
                values[key], origin[key] = seed, {"status": "seed"}
                continue
            measured = self.measured(key, version, base, history, study)
            trials, n = self.trials(version, base, study, objective)
            source, origin[key] = self.write(
                version, version.source(base.params[key], study.artifacts), measured, trials
            )
            if trials:  # what the proposer read of the trials: kept with the study, named in the record
                origin[key]["digest"] = {**_store_digest(trials, study), "trials": n}
            values[key] = (
                base.params[key]
                if source is None
                else store(source, study.artifacts, version.manifest(parent=base.params[key]))
            )
            origin[key]["parent"] = base.params[key]
        spent: dict[str, float] = {}
        for key in keys:
            _add(spent, origin[key].get("usage") or {})
        if spent:  # the study counts a method's own model calls from origin["usage"] (role method:<block>)
            origin["usage"] = spent
            origin["seconds"] = sum(origin[k].get("seconds") or 0.0 for k in keys)
        if any(origin[k].get("status") == "failed" for k in keys):
            origin["status"] = "failed"
        return _ProgramDraw(values, origin)


def _add(into: dict[str, float], spent: Mapping[str, float]) -> None:
    for k, v in spent.items():
        into[k] = into.get(k, 0.0) + v


def _value(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
    return "-" if x is None else str(x)


register_kind(KIND, ("program",), ProgramProposer)  # Program -> ProgramProposer, unless a Block names another method

__all__ = [
    "EMPTY",
    "KIND",
    "Program",
    "ProgramProposer",
    "ProgramTimeout",
    "ProgramVersion",
    "load",
    "messages_transcript",
    "run_bounded",
    "store",
    "value_of",
]
