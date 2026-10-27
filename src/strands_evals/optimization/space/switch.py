"""Conditional (branch-dependent) search spaces and stable candidate keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from . import space
from .domains import SuggestionTrial


@dataclass(frozen=True)
class Switch:
    """A categorical choice whose options carry their own parameters.

    Only the selected option's parameters are sampled (define-by-run), so a sampler never proposes parameters of an
    inactive branch. Keys: ``name`` holds the option, ``name.<option>.<param>`` its parameters. Branch
    values may be Domains, components, nested Switches or OneOfs, or fixed values (fixed values are not part of the
    candidate). Built as ``(option, params)``; use ``OneOf`` to choose between components directly.
    """

    branches: Mapping[str, Mapping[str, Any]]
    default: str

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Switch requires at least one branch")
        if self.default not in self.branches:
            raise ValueError("Switch default must name a branch")

    def suggest(self, trial: SuggestionTrial, name: str) -> dict[str, Any]:
        return space.suggest(self, trial, name)

    def defaults(self, name: str) -> dict[str, Any]:
        return space.defaults(self, name)

    def build(self, candidate: Mapping[str, Any], name: str) -> tuple[str, dict[str, Any]]:
        """Return ``(option, params)`` for the selected branch, including fixed values."""
        return space.build(self, candidate, name)


def candidate_key(values: Mapping[str, Any], *, extras: Mapping[str, str] | None = None) -> str:
    """A stable 12-hex-digit key over canonical JSON of the values (and optional extra artifact hashes)."""
    payload = {"values": dict(values), "extras": dict(extras or {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
