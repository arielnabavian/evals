"""Which choices helped, by how much, and how sure: an effects model fitted to a study's record.

For every goal metric, BayesBench's conjugate Gaussian posterior over main effects and two-way interactions is fitted to
the candidates' per-case means. The factors are the agent's choices (``OneOf``, ``Switch``, ``Choice``) at any depth,
its numbers binned into three ranges of the values tried, each text variable by how many rewrites deep it is (default,
1, 2, 3+: the text itself is no factor, each rewrite being its own value), and the case (so case difficulty is not
mistaken for an effect). A variable that does not exist under a candidate's choices takes the level ``–``; combinations
the space cannot produce are excluded.

Metrics are put on BayesBench's [0, 1] scale before fitting (by their declared range, or for an unbounded metric by
twice the baseline's value) and every effect is reported back in the metric's own units: the change from the reference
level (the default) to each other level, with a 90% credible interval (the goal says which way is better; ``p_positive``
is the probability the change is above zero). Each metric is fitted once, however many goals use it. The model's
held-out diagnostics say whether it fits; read the effects only when it does.

    python -m strands_evals.optimization.effects studies/<study> [...]

writes ``effects.json`` next to each record's ``study.jsonl`` (a study writes it itself at the end of each run).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

SCHEMA = 1
MISSING = "–"
MAX_CELLS = 60_000
SELECTORS = ("one_of", "switch")


def _events(directory: Path) -> list[dict[str, Any]]:
    out = []
    for line in (directory / "study.jsonl").read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line still being written
    return out


def _label(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:.3g}"


class _Factor:
    """One factor of the model: a variable's levels (the reference, its default's, first), and a candidate's level."""

    def __init__(
        self,
        key: str,
        kind: str,
        levels: list[str],
        bins: list[tuple[float, float, str]] | None = None,
        reference: str | None = None,
        by_trial: dict[int, str] | None = None,
    ) -> None:
        self.key, self.kind, self.bins, self.by_trial = key, kind, bins, by_trial
        self.reference = reference if reference in levels else levels[0]
        self.levels = [self.reference, *[lv for lv in levels if lv != self.reference]]

    def level(self, candidate: dict[str, Any]) -> str:
        params = candidate["params"]
        if self.key not in params:
            return MISSING
        if self.by_trial is not None:
            return self.by_trial[candidate["number"]]
        v = params[self.key]
        if self.bins is None:
            return str(v)
        for _lo, hi, name in self.bins:
            if float(v) <= hi:
                return name
        return self.bins[-1][2]


def rewrites(key: str, default: str, block: str, every: list[dict[str, Any]]) -> dict[int, int]:
    """How many rewrites deep each candidate's text is: 0 for the default, else one more than the text it was rewritten
    from (the text its method held, ``origin[block].from_candidate``). A text a candidate only held keeps its depth."""
    depth: dict[str, int] = {default: 0}
    out: dict[int, int] = {}
    for t in every:
        if key not in t["params"]:
            continue
        text = t["params"][key]
        if text not in depth:
            parent = (t.get("origin") or {}).get(block, {}).get("from_candidate")
            source = every[parent]["params"].get(key) if isinstance(parent, int) and 0 <= parent < len(every) else None
            depth[text] = depth.get(source, 0) + 1 if source is not None else 1
        out[t["number"]] = depth[text]
    return out


def _factors(
    space: list[dict[str, Any]], candidates: list[dict[str, Any]], every: list[dict[str, Any]], blocks: dict[str, str]
) -> tuple[list[_Factor], list[dict[str, str]]]:
    factors: list[_Factor] = []
    omitted: list[dict[str, str]] = []
    for row in space:
        key, kind = row["key"], row["type"]
        present = [t["params"][key] for t in candidates if key in t["params"]]
        absent = [MISSING] if any(key not in t["params"] for t in candidates) else []
        if kind in ("text", "skills"):
            # the text itself is not a factor (each rewrite is its own value); how many rewrites deep it is, is
            depth = rewrites(key, str(row["default"]), blocks.get(key, ""), every)
            names = {d: ("default" if d == 0 else str(d) if d < 3 else "3+") for d in depth.values()}
            by_trial = {n: names[d] for n, d in depth.items()}
            seen = list(dict.fromkeys(by_trial[t["number"]] for t in candidates if t["number"] in by_trial))
            order = ["default", "1", "2", "3+"]
            f = _Factor(key, "rewrites", sorted(seen, key=order.index) + absent, reference="default", by_trial=by_trial)
            if len([lv for lv in f.levels if lv != MISSING]) < 2:
                omitted.append({"key": key, "why": "text" if len(seen) < 2 else "one value"})
                continue
            factors.append(f)
            continue
        if not present:
            omitted.append({"key": key, "why": "never active"})
            continue
        if kind in SELECTORS or kind == "choice":
            seen = list(dict.fromkeys(str(v) for v in present))
            f = _Factor(key, "choice", seen + absent, reference=str(row["default"]))
        else:
            values = sorted(set(float(v) for v in present))
            if len(values) <= 3:
                bins = [(v, v, _label(v)) for v in values]
            else:
                q = statistics.quantiles([float(v) for v in present], n=3, method="inclusive")
                if kind == "int":
                    q = [round(x) for x in q]
                edges = sorted(set([values[0], *q, values[-1]]))
                bins = [(a, b, f"{_label(a)}–{_label(b)}") for a, b in zip(edges, edges[1:], strict=False)]
            default = row["default"]
            probe = _Factor(key, "range", [b[2] for b in bins] or [MISSING], bins=bins)
            f = _Factor(
                key,
                "range",
                [b[2] for b in bins] + absent,
                bins=bins,
                reference=probe.level({"params": {key: default}, "number": -1}) if default is not None else None,
            )
        if len([lv for lv in f.levels if lv != MISSING]) < 2:
            omitted.append({"key": key, "why": "one value"})
            continue
        factors.append(f)
    return factors, omitted


def _consistent(cell: dict[str, str], factors: list[_Factor], space: dict[str, dict[str, Any]]) -> bool:
    """Whether the space can produce this combination: a variable has a level exactly when its choices make it exist."""
    level = dict(cell)

    def exists(key: str) -> bool:
        at = space[key].get("active_when")
        while at:
            sel, option = at["key"], at["option"]
            if sel in level and level[sel] != option:
                return False
            at = space[sel].get("active_when")
        return True

    return all((level[f.key] != MISSING) == exists(f.key) for f in factors)


def _attempt_value(trial: dict[str, Any], metric: str, failure: dict[str, float | None]) -> float | None:
    if not trial.get("scored"):
        return None
    if trial["status"] == "agent_failure" and metric in failure:
        return failure[metric]
    v = trial["metrics"].get(metric)
    return None if v is None else float(v)


def fit(directory: str | Path) -> dict[str, Any]:
    """The effects of every choice on every goal metric, from the record in ``directory``."""
    from bayesbench import ExperimentSpace
    from bayesbench.sampler import InformationGainSampler

    directory = Path(directory)
    events = _events(directory)
    head = events[0]
    candidates = [e for e in events if e["type"] == "candidate" and e["eligible"]]
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "record_events": len(events),
        "method": "bayesbench",
        "factors": [],
        "omitted": [],
        "goals": [],
        "metrics": {},
        "messages": [],
    }
    if len(candidates) < 3:
        out["messages"].append("fewer than 3 comparable candidates")
        return out
    space = {row["key"]: row for row in head["space"]}
    every = [e for e in events if e["type"] == "candidate"]
    blocks = {k: name for name, b in head.get("blocks", {}).items() for k in b.get("keys", [])}
    factors, omitted = _factors(head["space"], candidates, every, blocks)
    out["omitted"] = omitted
    cases = list(head["cases"])
    if len(cases) >= 2:
        factors.append(_Factor("case", "case", cases))

    def cells(fs: list[_Factor]) -> int:
        return math.prod(len(f.levels) for f in fs)

    while factors and cells(factors) > MAX_CELLS:
        drop = next((f for f in reversed(factors) if f.kind == "range"), None) or factors[-1]
        factors.remove(drop)
        out["omitted"].append({"key": drop.key, "why": "too many combinations"})
    model = [f for f in factors if f.kind != "case"]
    if not model:
        out["messages"].append("no choice varies")
        return out
    out["factors"] = [{"key": f.key, "kind": f.kind, "reference": f.reference, "levels": f.levels} for f in model]
    names = [f.key for f in factors]
    bb = ExperimentSpace({f.key: f.levels for f in factors})
    eligible = [dict(zip(names, cell, strict=True)) for cell in itertools.product(*(f.levels for f in factors))]
    eligible = [c for c in eligible if _consistent({k: v for k, v in c.items() if k != "case"}, model, space)]
    failure = head.get("agent_error_score", head.get("agent_failure")) or {}  # schema 2, schema 1
    baseline = next((e for e in events if e["type"] == "candidate" and e["role"] == "baseline"), None)

    out["goals"] = [
        {
            "role": "objective",
            "index": i,
            "metric": o["metric"],
            "better": "higher" if o["kind"] == "Maximize" else "lower",
        }
        for i, o in enumerate(head["objectives"])
    ]
    out["goals"] += [
        {
            "role": "constraint",
            "index": i,
            "metric": c["metric"],
            "better": "higher" if c["kind"] == "AtLeast" else "lower",
        }
        for i, c in enumerate(head["constraints"])
    ]
    out["metrics"] = {}
    for metric in dict.fromkeys(g["metric"] for g in out["goals"]):
        lo, hi = head["metrics"].get(metric, {}).get("range", [None, None])
        if lo is not None and hi is not None and hi > lo:
            offset, unit, scale = lo, hi - lo, {"kind": "range", "low": lo, "high": hi}
        else:
            b = (baseline or {}).get("estimates", {}).get(metric, {}).get("mean")
            offset, unit = 0.0, 2 * abs(b) if b else 1.0
            scale = {"kind": "baseline", "unit": unit}
        observations, spreads = [], []
        for t in candidates:
            for case in cases:
                vs = [
                    v for a in t["trials"] if a["case"] == case if (v := _attempt_value(a, metric, failure)) is not None
                ]
                if not vs:
                    continue
                ys = [(v - offset) / unit for v in vs]
                if len(ys) >= 2:
                    spreads.append(statistics.variance(ys))
                cell = {f.key: f.level(t) for f in model}
                if len(cases) >= 2:
                    cell["case"] = case
                observations.append((cell, statistics.fmean(ys), len(ys)))
        pooled = statistics.fmean(spreads) if spreads else None
        sampler = InformationGainSampler(bb, eligible_configs=eligible, seed=0)
        for cell, y, n in observations:
            variance = pooled / n if pooled is not None else None
            sampler.observe(cell, y, variance=variance, m=None if variance is not None else max(n, 1))
        keys = [bb.key(c) for c in eligible]
        position = {f.key: i for i, f in enumerate(factors)}
        effects = []
        for f in model:
            i = position[f.key]
            for level in f.levels:
                if level in (f.reference, MISSING):
                    continue
                # each side averaged over every combination the space can produce with it (for a choice between
                # components, each option over its own settings; for anything else, the same as a matched contrast)
                with_level = [k for k in keys if k[i] == level]
                with_reference = [k for k in keys if k[i] == f.reference]
                if not with_level or not with_reference:
                    continue
                L = sampler.encoder.design_matrix(with_level).mean(axis=0) - sampler.encoder.design_matrix(
                    with_reference
                ).mean(axis=0)
                c = sampler.linear_contrast(L, name=f"{f.key}={level}")
                effects.append(
                    {
                        "key": f.key,
                        "level": level,
                        "mean": c.mean * unit,
                        "ci90": [c.ci90[0] * unit, c.ci90[1] * unit],
                        "p_positive": c.p_positive,
                    }
                )
        d = sampler.diagnostics()
        out["metrics"][metric] = {
            "scale": scale,
            "observations": len(observations),
            "effects": effects,
            "diagnostics": {
                "n_holdout": d.n_holdout,
                "z_mean": d.z_mean,
                "z_sq_sum": d.z_sq_sum,
                "flagged": d.flagged,
                "messages": list(d.messages),
            },
        }
    return out


def write(directory: str | Path) -> Path:
    directory = Path(directory)
    path = directory / "effects.json"
    path.write_text(json.dumps(fit(directory), indent=1))
    return path


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        for record in sorted(Path(arg).glob("**/study.jsonl")) or [Path(arg) / "study.jsonl"]:
            sys.stdout.write(f"{write(record.parent)}\n")
