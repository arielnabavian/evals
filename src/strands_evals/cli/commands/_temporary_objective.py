from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

_DIRECTIONS = ("Maximize", "Minimize")
_COMBINATIONS = ("WeightedSum", "Pareto")
_TERM_SETTINGS = ("weight", "scale")
_USAGE = (
    "--objective takes the objective of the optimization API, for example Maximize(Helpfulness), "
    "Minimize(tokens), or WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))"
)


@dataclass(frozen=True)
class TemporaryObjectiveTerm:
    direction: str
    metric: str
    settings: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class TemporaryObjectiveSpec:
    combination: str | None
    terms: tuple[TemporaryObjectiveTerm, ...]


def _refuse(part: str) -> argparse.ArgumentTypeError:
    return argparse.ArgumentTypeError(f"{_USAGE}; '{part}' does not fit it")


def _call_name(node: ast.expr) -> str | None:
    return node.func.id if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) else None


def _metric(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return node.value
    raise _refuse(ast.unparse(node))


def _setting(keyword: ast.keyword) -> tuple[str, float]:
    if keyword.arg not in _TERM_SETTINGS:
        raise _refuse(keyword.arg or ast.unparse(keyword.value))
    value = keyword.value
    if (
        not isinstance(value, ast.Constant)
        or isinstance(value.value, bool)
        or not isinstance(value.value, (int, float))
        or not math.isfinite(value.value)
        or value.value <= 0
    ):
        raise _refuse(ast.unparse(value))
    return keyword.arg, float(value.value)


def _term(node: ast.expr) -> TemporaryObjectiveTerm:
    direction = _call_name(node)
    if direction not in _DIRECTIONS or not isinstance(node, ast.Call):
        raise _refuse(direction or ast.unparse(node))
    if len(node.args) != 1:
        raise _refuse(ast.unparse(node))
    return TemporaryObjectiveTerm(direction, _metric(node.args[0]), tuple(_setting(k) for k in node.keywords))


def parse_temporary_objective(raw: str) -> TemporaryObjectiveSpec:
    try:
        tree = ast.parse(raw.strip(), mode="eval").body
    except SyntaxError as e:
        raise _refuse(raw.strip()) from e
    combination = _call_name(tree)
    if combination not in _COMBINATIONS or not isinstance(tree, ast.Call):
        return TemporaryObjectiveSpec(None, (_term(tree),))
    if tree.keywords:
        raise _refuse(tree.keywords[0].arg or ast.unparse(tree.keywords[0].value))
    return TemporaryObjectiveSpec(combination, tuple(_term(argument) for argument in tree.args))


def temporary_objective(spec: TemporaryObjectiveSpec, optimization: ModuleType) -> Any:
    terms = [getattr(optimization, term.direction)(term.metric, **dict(term.settings)) for term in spec.terms]
    return getattr(optimization, spec.combination)(*terms) if spec.combination else terms[0]
