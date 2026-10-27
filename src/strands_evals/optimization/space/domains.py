"""The domains of a search space: numbers, choices, text and skill libraries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

T = TypeVar("T")


class SuggestionTrial(Protocol):
    """Small sampler protocol implemented by Optuna Trial."""

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int: ...

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...


class Domain(ABC, Generic[T]):
    """A serializable value that a search backend may vary."""

    default: T

    @abstractmethod
    def suggest(self, trial: SuggestionTrial, name: str) -> T:
        """Ask a backend trial for one value."""


@dataclass(frozen=True)
class Float(Domain[float]):
    low: float
    high: float
    default: float
    step: float | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Float low must not exceed high")
        if not self.low <= self.default <= self.high:
            raise ValueError("Float default must be within [low, high]")
        if self.step is not None and self.step <= 0:
            raise ValueError("Float step must be positive")
        if self.log and self.step is not None:
            raise ValueError("Float cannot use both log=True and step")

    def suggest(self, trial: SuggestionTrial, name: str) -> float:
        return trial.suggest_float(
            name,
            self.low,
            self.high,
            step=self.step,
            log=self.log,
        )


@dataclass(frozen=True)
class Int(Domain[int]):
    low: int
    high: int
    default: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Int low must not exceed high")
        if not self.low <= self.default <= self.high:
            raise ValueError("Int default must be within [low, high]")
        if self.step <= 0:
            raise ValueError("Int step must be positive")
        if self.log and self.step != 1:
            raise ValueError("Int cannot use both log=True and a custom step")

    def suggest(self, trial: SuggestionTrial, name: str) -> int:
        return trial.suggest_int(
            name,
            self.low,
            self.high,
            step=self.step,
            log=self.log,
        )


@dataclass(frozen=True)
class Choice(Domain[T]):
    choices: tuple[T, ...]
    default: T

    def __init__(self, choices: list[T] | tuple[T, ...], *, default: T):
        choices = tuple(choices)
        if not choices:
            raise ValueError("Choice requires at least one value")
        if default not in choices:
            raise ValueError("Choice default must be one of choices")
        object.__setattr__(self, "choices", choices)
        object.__setattr__(self, "default", default)

    def suggest(self, trial: SuggestionTrial, name: str) -> T:
        return trial.suggest_categorical(name, list(self.choices))


@dataclass(frozen=True)
class TextValue(Domain[str]):
    """Free text, such as a system prompt or a tool's description. A sampler cannot draw text: a ``Study`` has a model
    write each new value from the results so far.

    ``about`` says what the text is, for the model that rewrites it ("the description of the search tool, which the
    model reads to decide when to call it"); ``max_chars`` bounds its length (a longer value is never measured)."""

    default: str
    about: str | None = None
    max_chars: int | None = None

    def __post_init__(self) -> None:
        if self.max_chars is not None and len(self.default) > self.max_chars:
            raise ValueError(f"TextValue default is {len(self.default)} characters, over max_chars={self.max_chars}")

    def suggest(self, trial: SuggestionTrial, name: str) -> str:
        suggest_text = getattr(trial, "suggest_text", None)
        if suggest_text is None:
            raise TypeError(f"{name} is free text: it needs an optimizer that writes text (Study), not a sampler")
        return suggest_text(name, self.default)


@dataclass(frozen=True)
class SkillLibrary(Domain[str]):
    """A library of Strands Agent Skills: a directory of ``<name>/SKILL.md`` (plus any files a skill needs). A method
    that curates skills creates, revises and retires skills, and each new version is a new, immutable directory in the
    study's artifact store, as is the default when the study starts; a candidate's value is the version's path there,
    ``skills/<hash>`` (relative to the store, so a record can be moved: ``Study.artifact_path`` finds it). ``default``:
    the starting library, or None for an empty one.

    The factory receives the version's absolute path (``AgentSkills(skills=path)``), or None for an empty library."""

    default: str | None = None
    about: str | None = None

    def suggest(self, trial: SuggestionTrial, name: str) -> str | None:
        suggest_skills = getattr(trial, "suggest_skills", None)
        if suggest_skills is None:
            raise TypeError(f"{name} is a skill library: it needs a method that curates skills (Study), not a sampler")
        return suggest_skills(name, self.default)
