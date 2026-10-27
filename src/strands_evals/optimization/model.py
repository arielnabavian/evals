"""What the study decides with: a model of each goal metric, fitted to every eligible candidate's trials.

One model per goal metric (every objective's and constraint's metric): BayesBench's conjugate Gaussian posterior over
a design the study builds itself, so the space's cells are never enumerated.

- **Observations:** one per candidate and case, the mean of its scored repetitions, on BayesBench's scale (by the
  metric's range, or for an unbounded metric by twice the baseline's value). Agent failures count as the study's
  ``agent_error_score``, as in estimates.
- **Design:** the intercept; each choice (``Choice``, ``OneOf``, ``Switch``) by its option and each number binned at the
  quantiles of the values measured (``–`` where the space leaves the variable out), with main effects and pairwise
  interactions (BayesBench's encoder); each text block (the texts one method rewrites together) by whether it is
  rewritten; each candidate's own effect, for what the factors leave unexplained (a rewrite's text, a number within
  its bin); and the case, so case difficulty is shared by every candidate. All but the first group have main effects
  only.
- **Prior scales:** how far an option or bin moves the metric, and how far a candidate moves it beyond its factors,
  are fitted to the data (empirical Bayes: the scales that make the observations most likely), so a candidate measured
  lucky is shrunk toward what its factors predict by as much as the other candidates' spread says it should be.
- **Noise:** one variance for every observation, in two passes: the within-case variance over repetitions divided by
  the repetitions, then the first fit's residual mean square, whichever is larger.
- **Diagnostics:** before each new candidate is absorbed, its mean over the cases is scored against the previous fit's
  prediction (BayesBench's held-out z windows).

A candidate's *decided* value of a metric is the posterior mean and interval of its mean over the cases; a constraint's
probability of being met follows from it.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .projection import AtLeast, AtMost, Constraint, Projection

if TYPE_CHECKING:
    from .space import ParamInfo
    from .study import Candidate

MISSING = "–"
SELECTORS = ("one_of", "switch", "choice")
TEXTS = ("text", "skills")
Z90 = 1.6448536269514722

TAU_INTERCEPT = 0.25
TAU_MAIN = 0.15
TAU_PAIR = 0.03  # BayesBench's default for an interaction
TAU_CANDIDATE = 0.05  # one candidate, around what its factors predict
TAU_REWRITTEN = 0.15  # the average rewrite, around the default
TAU_CASE = 0.35
NOISE_FLOOR = 1e-6  # on BayesBench's scale: a deterministic metric is decided as measured


@dataclass(frozen=True)
class Decided:
    """A metric's posterior at one configuration, in the metric's own units: its mean over the cases, the standard
    deviation of that mean, and a 90% interval."""

    mean: float
    sd: float

    @property
    def ci90(self) -> tuple[float, float]:
        return (self.mean - Z90 * self.sd, self.mean + Z90 * self.sd)

    def quantile(self, q: float) -> float:
        return self.mean + _norm_ppf(q) * self.sd

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "sd": self.sd, "ci90": list(self.ci90)}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    return statistics.NormalDist().inv_cdf(q)


def p_met(constraint: Constraint, value: Decided) -> float:
    """The probability that the constraint holds, given the metric's decided value."""
    if value.sd <= 0:
        return (
            1.0
            if (value.mean <= constraint.limit if isinstance(constraint, AtMost) else value.mean >= constraint.limit)
            else 0.0
        )
    z = (constraint.limit - value.mean) / value.sd
    return _norm_cdf(z) if isinstance(constraint, AtMost) else 1.0 - _norm_cdf(z)


def _text_level(values: Mapping[str, Any], keys: Sequence[str], defaults: Mapping[str, Any]) -> str:
    if all(values.get(k) == defaults.get(k) for k in keys):
        return "default"
    digest = hashlib.sha256("\0".join(str(values.get(k)) for k in keys).encode()).hexdigest()[:12]
    return f"text:{digest}"


@dataclass
class _Factor:
    """One design factor: a choice, a binned number, or whether a text block is rewritten; the reference level (the
    default's) first."""

    key: str
    kind: str  # choice | range | rewritten
    levels: list[str]
    keys: tuple[str, ...] = ()  # a text block's variables
    bins: list[float] = field(default_factory=list)  # a range's upper edges
    defaults: dict[str, Any] = field(default_factory=dict)

    def level(self, params: Mapping[str, Any]) -> str:
        if self.kind == "rewritten":
            return "default" if _text_level(params, self.keys, self.defaults) == "default" else "rewritten"
        if self.key not in params:
            return MISSING
        v = params[self.key]
        if self.kind == "choice":
            return str(v)
        for i, edge in enumerate(self.bins):
            if float(v) <= edge:
                return self._bin_name(i)
        return self._bin_name(len(self.bins) - 1)

    def _bin_name(self, i: int) -> str:
        lo = "-inf" if i == 0 else f"{self.bins[i - 1]:.6g}"
        return f"({lo}, {self.bins[i]:.6g}]"


def _trend_value(v: Any, log: bool) -> float:
    v = float(v)
    return math.log(v) if log and v > 0 else v


def _bins(values: list[float]) -> list[float]:
    """Upper bin edges at the quantiles of the measured values: 3 bins, 4 from 12 distinct values, each with at least
    2 values where the values allow."""
    distinct = sorted(set(values))
    if len(distinct) <= 3:
        return distinct
    n = 4 if len(distinct) >= 12 else 3
    edges = statistics.quantiles(values, n=n, method="inclusive")
    edges = sorted(set([*edges, distinct[-1]]))
    keep, count, last = [], 0, -math.inf
    for e in edges:
        count = sum(1 for v in values if last < v <= e)
        if count >= 2 or e == edges[-1]:
            keep.append(e)
            last = e
    return keep if len(keep) >= 2 else [distinct[-1]]


class Design:
    """The study's factors at one fit, and each configuration's row of the design matrix."""

    def __init__(
        self,
        space: Mapping[str, "ParamInfo"],
        text_blocks: Mapping[str, Sequence[str]],
        candidates: Sequence["Candidate"],
        cases: Sequence[str],
    ) -> None:
        from bayesbench.encoding import Encoder  # optional extra: bayesbench
        from bayesbench.space import ExperimentSpace

        params = [c.params for c in candidates]
        self.factors: list[_Factor] = []
        self.omitted: list[dict[str, str]] = []
        in_texts = {k for keys in text_blocks.values() for k in keys}
        for key, info in space.items():
            if key in in_texts or info.type in TEXTS or info.type in ("component", "fixed"):
                continue
            present = [p[key] for p in params if key in p]
            absent = [MISSING] if any(key not in p for p in params) else []
            if not present:
                self.omitted.append({"key": key, "why": "never active"})
                continue
            if info.type in SELECTORS:
                # every option the space offers, measured or not: an option never measured is at its prior
                options = list(dict.fromkeys([*(str(v) for v in info.choices), *(str(v) for v in present)]))
                reference = str(info.default) if str(info.default) in options else options[0]
                f = _Factor(key, "choice", [reference, *[o for o in options if o != reference], *absent])
            else:
                edges = _bins([float(v) for v in present])
                f = _Factor(key, "range", [], bins=edges)
                names = list(dict.fromkeys(f._bin_name(i) for i in range(len(edges))))
                reference = f.level({key: info.default}) if info.default is not None else names[0]
                f.levels = [reference, *[n for n in names if n != reference], *absent]
            if len([lv for lv in f.levels if lv != MISSING]) < 2 and not absent:
                self.omitted.append({"key": key, "why": "one value"})
                continue
            self.factors.append(f)
        space_ = ExperimentSpace({f.key: f.levels for f in self.factors}) if self.factors else None
        self.encoder = Encoder(space_, interactions=2 if len(self.factors) >= 2 else 1) if space_ else None
        self.extra: list[_Factor] = []
        for name, keys in text_blocks.items():
            defaults = {k: space[k].default for k in keys}
            rewritten = _Factor(
                f"rewritten:{name}", "rewritten", ["default", "rewritten"], tuple(keys), defaults=defaults
            )
            if any(rewritten.level(p) == "rewritten" for p in params):
                self.extra.append(rewritten)
        self.cases = list(cases)
        self.extra_cols: list[tuple[int, str]] = []  # (index into self.extra, level), one per column
        for i, f in enumerate(self.extra):
            for level in f.levels[1:]:
                self.extra_cols.append((i, level))
        # each number's trend: a slope over its values (their logarithms for a log range), standardized over the
        # candidates fitted, so that a value anywhere in its bin is predicted along the trend rather than at the bin's
        self.trends: list[tuple[str, bool, float, float]] = []  # key, log, center, spread
        self.bin_centers: dict[tuple[str, str], float] = {}
        for f in self.factors:
            if f.kind != "range":
                continue
            log = bool(space[f.key].log)
            ts = [_trend_value(p[f.key], log) for p in params if f.key in p]
            spread = statistics.pstdev(ts) if len(ts) >= 2 else 0.0
            if spread > 0:
                self.trends.append((f.key, log, statistics.fmean(ts), spread))
                for level in f.levels:  # each bin at the mean of the values fitted in it, for its effect
                    inside = [_trend_value(p[f.key], log) for p in params if f.key in p and f.level(p) == level]
                    if inside:
                        self.bin_centers[(f.key, level)] = statistics.fmean(inside)
        # each candidate's own effect (the first, the baseline, is the reference)
        self.numbers = [c.number for c in candidates][1:]
        self.fitted = {c.number for c in candidates}
        self.n_design = self.encoder.p if self.encoder else 1
        self.n_extra = len(self.extra_cols) + len(self.trends)
        self.n_candidates = len(self.numbers)
        self.p = self.n_design + self.n_extra + self.n_candidates + max(len(self.cases) - 1, 0)
        self._extra_index = {(self.extra[i].key, level): j for j, (i, level) in enumerate(self.extra_cols)}
        self._candidate_index = {n: j for j, n in enumerate(self.numbers)}

    def prior(self, intercept: float, scales: Mapping[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The prior's mean and variances; ``scales`` overrides the standard deviation of a group of effects
        (``main``, ``pair``, ``rewritten``, ``candidate``, ``case``)."""
        tau = {
            "main": TAU_MAIN,
            "pair": TAU_PAIR,
            "rewritten": TAU_REWRITTEN,
            "candidate": TAU_CANDIDATE,
            "case": TAU_CASE,
        }
        tau.update(scales or {})
        mean = np.zeros(self.p)
        var = np.empty(self.p)
        mean[0] = intercept
        var[0] = TAU_INTERCEPT**2
        if self.encoder:
            for cols in self.encoder.main_idx.values():
                var[cols] = tau["main"] ** 2
            for cols in self.encoder.interaction_idx.values():
                var[cols] = tau["pair"] ** 2
        for j, (i, _level) in enumerate(self.extra_cols):
            var[self.n_design + j] = tau[self.extra[i].kind] ** 2
        trend_start = self.n_design + len(self.extra_cols)
        var[trend_start : trend_start + len(self.trends)] = tau["main"] ** 2  # a slope per standard deviation
        start = self.n_design + self.n_extra
        var[start : start + self.n_candidates] = tau["candidate"] ** 2
        var[start + self.n_candidates :] = tau["case"] ** 2
        return mean, var

    @property
    def groups(self) -> list[str]:
        """The groups of effects this design has, whose prior scales are fitted to the data."""
        out = []
        if self.encoder and self.encoder.main_idx:
            out.append("main")
        if self.encoder and self.encoder.interaction_idx:
            out.append("pair")
        if self.n_candidates:
            out.append("candidate")
        return out

    def row(self, params: Mapping[str, Any], case: str | None = None, number: int | None = None) -> np.ndarray:
        """A configuration's row: at one case, or (``case=None``) averaged over the cases; with the candidate's own
        effect when it is a candidate fitted (``number``), else at what its factors predict."""
        x = np.zeros(self.p)
        if self.encoder:
            key = tuple(self._level(f, params) for f in self.factors)
            x[: self.n_design] = self.encoder.phi(key)
        else:
            x[0] = 1.0
        for f in self.extra:
            j = self._extra_index.get((f.key, f.level(params)))
            if j is not None:
                x[self.n_design + j] = 1.0
        trend_start = self.n_design + len(self.extra_cols)
        for j, (key, log, center, spread) in enumerate(self.trends):
            if key in params:  # where the space leaves the number out, its bin factor's absent level says so
                x[trend_start + j] = (_trend_value(params[key], log) - center) / spread
        start = self.n_design + self.n_extra
        if number is not None and number in self._candidate_index:
            x[start + self._candidate_index[number]] = 1.0
        offset = start + self.n_candidates
        if case is None:
            x[offset:] = 1.0 / len(self.cases) if len(self.cases) > 1 else 0.0
        elif case in self.cases[1:]:
            x[offset + self.cases.index(case) - 1] = 1.0
        return x

    def _level(self, f: _Factor, params: Mapping[str, Any]) -> str:
        level = f.level(params)
        return level if level in f.levels else f.levels[0]  # a variable every fitted candidate had: at its reference

    def describe(self) -> list[dict[str, Any]]:
        out = [{"key": f.key, "kind": f.kind, "levels": f.levels, "reference": f.levels[0]} for f in self.factors]
        out += [{"key": f.key, "kind": f.kind, "levels": f.levels, "reference": f.levels[0]} for f in self.extra]
        return out


class MetricModel:
    """One goal metric's posterior at one fit."""

    def __init__(
        self,
        metric: str,
        design: Design,
        posterior: Any,
        offset: float,
        unit: float,
        noise: float,
        candidate_sd: float = 0.0,
    ) -> None:
        self.metric, self.design, self.posterior = metric, design, posterior
        self.offset, self.unit, self.noise = offset, unit, noise
        # the prior scale of a candidate's own effect (model units; over a mixture, the root of its mean square)
        self.candidate_sd = candidate_sd

    def own(self, number: int | None) -> float:
        """The variance a configuration's own effect adds: none for a candidate fitted, whose effect the posterior
        holds; for any other, the prior's (what its factors leave unexplained, as among the candidates fitted)."""
        return 0.0 if number is not None and number in self.design.fitted else self.candidate_sd**2

    def at(self, params: Mapping[str, Any], number: int | None = None) -> Decided:
        """The metric's mean over the cases at a configuration: a candidate fitted (``number``), or any other."""
        mean, var = self.posterior.predictive(self.design.row(params, number=number))
        sd = math.sqrt(max(float(var[0]) + self.own(number), 0.0)) * self.unit
        return Decided(mean=float(mean[0]) * self.unit + self.offset, sd=sd)

    def difference(
        self, params: Mapping[str, Any], reference: Mapping[str, Any], number: int | None, reference_number: int | None
    ) -> Decided:
        """The posterior of this configuration's value minus the reference's, over the same cases."""
        a = self.design.row(params, number=number) - self.design.row(reference, number=reference_number)
        mean, var = self.posterior.predictive(a)
        var = float(var[0]) + self.own(number) + self.own(reference_number)
        return Decided(mean=float(mean[0]) * self.unit, sd=math.sqrt(max(var, 0.0)) * self.unit)

    def ratio(
        self, params: Mapping[str, Any], reference: Mapping[str, Any], number: int | None, reference_number: int | None
    ) -> Decided:
        """The posterior of this configuration's value divided by the reference's, over the same cases (to first
        order: the ratio of the means, with the variance of a ratio of correlated estimates)."""
        a = self.design.row(params, number=number)
        b = self.design.row(reference, number=reference_number)
        mean, var = self.posterior.predictive(np.vstack([a, b]))
        va, vb, vab = float(var[0]), float(var[1]), self.posterior.covariance(a, b)
        va, vb = va + self.own(number), vb + self.own(reference_number)
        ma, mb = float(mean[0]) * self.unit + self.offset, float(mean[1]) * self.unit + self.offset
        if mb == 0:
            return Decided(mean=math.inf, sd=0.0)
        u2 = self.unit**2
        r = ma / mb
        var = (va * u2) / mb**2 + (ma**2 * vb * u2) / mb**4 - 2 * ma * vab * u2 / mb**3
        return Decided(mean=r, sd=math.sqrt(max(var, 0.0)))

    def draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.posterior.sample(n, rng)

    def drawn(self, theta: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """The metric's value at a configuration under each drawn set of effects (``theta``: draws × p)."""
        return (theta @ self.design.row(params)) * self.unit + self.offset


class Mixture:
    """The posterior averaged over the prior scales the observations allow: one conjugate posterior per point of the
    scale grid, weighted by the marginal likelihood it gives the observations (Bayesian model averaging), so that
    scales the data cannot tell apart (an effect of 0.001 or of 0.01) all count, rather than the likeliest alone."""

    def __init__(self, components: Sequence[tuple[float, Any]]) -> None:
        self.weights = np.array([w for w, _ in components], dtype=float)
        self.weights /= self.weights.sum()
        self.components = [c for _, c in components]

    @property
    def n(self) -> int:
        return self.components[0].n

    def predictive(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The mixture's mean and variance of ``a θ`` for each row of ``a``."""
        a = np.atleast_2d(np.asarray(a, dtype=float))
        parts = [c.predictive(a) for c in self.components]
        means, variances = np.array([m for m, _ in parts]), np.array([v for _, v in parts])
        mean = self.weights @ means
        return mean, np.maximum(self.weights @ (variances + means**2) - mean**2, 0.0)

    def covariance(self, a: np.ndarray, b: np.ndarray) -> float:
        """The mixture's covariance of ``a θ`` and ``b θ``."""
        ma = np.array([float(a @ c.mean) for c in self.components])
        mb = np.array([float(b @ c.mean) for c in self.components])
        within = np.array([float(a @ cho_solve(c._cho, b)) for c in self.components])  # a Σ b, without forming Σ
        return float(self.weights @ (within + ma * mb) - (self.weights @ ma) * (self.weights @ mb))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        picks = rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty((n, self.components[0].p))
        for j in np.unique(picks):
            out[picks == j] = self.components[j].sample(int(np.sum(picks == j)), rng)
        return out


@dataclass
class Fit:
    """Every goal metric's model at one point of the study."""

    metrics: dict[str, MetricModel]
    design: Design
    candidates: tuple[int, ...]  # the candidates it was fitted to
    diagnostics: dict[str, dict[str, Any]]

    @property
    def flagged(self) -> list[str]:
        return [m for m, d in self.diagnostics.items() if d.get("flagged")]

    def decided(self, params: Mapping[str, Any], number: int | None = None) -> dict[str, Decided]:
        return {m: model.at(params, number) for m, model in self.metrics.items()}


def _scale(projection: Projection, metric: str, baseline: "Candidate") -> tuple[float, float]:
    lo, hi = projection.range(metric)
    if lo is not None and hi is not None and hi > lo:
        return float(lo), float(hi - lo)
    b = baseline.metrics.get(metric)
    return 0.0, (2 * abs(b) if b else 1.0)


# The prior scales averaged over (every point equally likely before the observations): from effects too small to
# matter to large ones, in units of the metric's range (or of twice the baseline's value, for an unbounded metric).
GRID = {
    "main": (0.003, 0.01, 0.03, 0.08, 0.15),
    "pair": (0.001, 0.003, 0.01, 0.03),
    "candidate": (0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3),
}
MIXTURE_SIZE = 24  # the likeliest grid points kept, at most; those with less than 1e-4 of the weight are dropped


def _posterior_mean(gram: np.ndarray, cross: np.ndarray, noise: float, mu0: np.ndarray, var0: np.ndarray) -> np.ndarray:
    precision = np.diag(1.0 / var0) + gram / noise
    return np.linalg.solve(precision, mu0 / var0 + cross / noise)


def _log_evidence(
    gram: np.ndarray, cross: np.ndarray, yy: float, n: int, noise: float, mu0: np.ndarray, var0: np.ndarray
) -> float:
    """The marginal likelihood of the observations under the prior (the effects integrated out), up to a constant."""
    precision = np.diag(1.0 / var0) + gram / noise
    try:
        factor = cho_factor(precision)
    except np.linalg.LinAlgError:
        return -math.inf
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    r = mu0 / var0 + cross / noise
    quad = yy / noise + float(mu0 @ (mu0 / var0)) - float(r @ cho_solve(factor, r))
    return -0.5 * (n * math.log(noise) + float(np.sum(np.log(var0))) + logdet + quad)


def _empirical_bayes(
    design: Design, gram: np.ndarray, cross: np.ndarray, yy: float, n: int, noise: float, intercept: float
) -> list[tuple[dict[str, float], float]]:
    """Each point of the grid of prior scales (of the design's effect groups) with the weight the observations give it
    (its marginal likelihood, normalized), likeliest first: how far a rewrite, or an option, moves the metric is
    learned from the candidates measured, so a candidate measured lucky is shrunk by as much as the others' spread
    says it should be."""
    groups = design.groups
    if not groups:
        return [({}, 1.0)]
    scored = []
    for combo in _product([GRID[g] for g in groups]):
        scales = dict(zip(groups, combo, strict=True))
        mu0, var0 = design.prior(intercept, scales)
        scored.append((scales, _log_evidence(gram, cross, yy, n, noise, mu0, var0)))
    top = max(v for _, v in scored)
    if not math.isfinite(top):
        return [(scored[0][0], 1.0)]
    weighted = sorted(((s, math.exp(v - top)) for s, v in scored), key=lambda sv: -sv[1])
    total = sum(w for _, w in weighted)
    return [(s, w / total) for s, w in weighted]


def _product(grids: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = [()]
    for g in grids:
        out = [(*o, v) for o in out for v in g]
    return out


def fit(
    projection: Projection,
    space: Mapping[str, "ParamInfo"],
    text_blocks: Mapping[str, Sequence[str]],
    candidates: Sequence["Candidate"],
    baseline: "Candidate",
    cases: Sequence[str],
    trackers: dict[str, Any] | None = None,
    previous: Fit | None = None,
) -> Fit | None:
    """Every goal metric's model, fitted to the eligible candidates; None when fewer than two are eligible.

    ``trackers`` (one BayesBench diagnostics tracker per metric, kept by the study) score each candidate the previous
    fit had not seen against that fit's prediction, before it is absorbed."""
    from bayesbench.diagnostics import DiagnosticsTracker  # optional extra: bayesbench
    from bayesbench.posterior import Posterior

    pool = [c for c in candidates if c.eligible]
    if len(pool) < 2:
        return None
    design = Design(space, text_blocks, pool, cases)
    trackers = trackers if trackers is not None else {}
    models: dict[str, MetricModel] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    seen = set(previous.candidates) if previous else set()
    for metric in projection.goal_metrics:
        offset, unit = _scale(projection, metric, baseline)
        rows, ys, spreads = [], [], []
        means: dict[int, float] = {}
        for c in pool:
            per_case: dict[str, list[float]] = {}
            for t in c.trials:
                v = projection.value(t, metric)
                if v is not None:
                    per_case.setdefault(t.case, []).append((v - offset) / unit)
            for case, vs in per_case.items():
                rows.append(design.row(c.params, case, c.number))
                ys.append(statistics.fmean(vs))
                if len(vs) >= 2:
                    spreads.append(statistics.variance(vs) / len(vs))
            if per_case:
                means[c.number] = statistics.fmean(statistics.fmean(vs) for vs in per_case.values())
        phi, y = np.vstack(rows), np.asarray(ys)
        gram, cross, intercept = phi.T @ phi, phi.T @ y, float(np.mean(y))
        # without repetitions (or with identical ones), the residual pass sets it
        noise = max(statistics.fmean(spreads) if spreads else 0.0, NOISE_FLOOR)
        weighted = _empirical_bayes(design, gram, cross, float(y @ y), len(y), noise, intercept)
        mu0, var0 = design.prior(intercept, weighted[0][0])
        mean = _posterior_mean(gram, cross, noise, mu0, var0)
        residual = float(np.mean((y - phi @ mean) ** 2))
        if residual > noise:  # how differently candidates do on each case, beyond repetition noise
            noise = residual
            weighted = _empirical_bayes(design, gram, cross, float(y @ y), len(y), noise, intercept)
        kept = [(s, w) for s, w in weighted[:MIXTURE_SIZE] if w >= 1e-4]
        components = []
        for s, w in kept:
            component = Posterior(*design.prior(intercept, s))
            component.update(phi, y, np.full(len(y), noise))
            components.append((w, component))
        posterior = Mixture(components)
        total = sum(w for _, w in kept)
        scales = {g: sum(w * s[g] for s, w in kept) / total for g in kept[0][0]}  # each scale's posterior mean
        if previous is not None and metric in previous.metrics:
            tracker = trackers.setdefault(metric, DiagnosticsTracker())
            before = previous.metrics[metric]
            for c in pool:
                if c.number in seen or c.number not in means:
                    continue
                mean, var = before.posterior.predictive(before.design.row(c.params))
                var = var + before.own(None)  # a candidate it had not fitted: its own effect at the prior's scale
                n_cases = sum(1 for case in cases if any(t.case == case for t in c.trials))
                tracker.record(
                    means[c.number], float(mean[0]), float(var[0]), before.noise / max(n_cases, 1), (c.number,)
                )
        if metric in trackers:
            d = trackers[metric].snapshot()
            diagnostics[metric] = {
                "n_holdout": d.n_holdout,
                "z_mean": d.z_mean,
                "z_sq_sum": d.z_sq_sum,
                "flagged": d.flagged,
                "messages": list(d.messages),
            }
        else:
            diagnostics[metric] = {"n_holdout": 0, "flagged": False, "messages": ["no held-out candidates yet"]}
        diagnostics[metric]["prior_scales"] = scales
        diagnostics[metric]["prior_scale_weights"] = [{"scales": s, "weight": w} for s, w in kept[:8]]
        candidate_sd = (
            math.sqrt(sum(w * s.get("candidate", TAU_CANDIDATE) ** 2 for s, w in kept) / total)
            if design.n_candidates
            else 0.0
        )
        models[metric] = MetricModel(metric, design, posterior, offset, unit, noise, candidate_sd)
    return Fit(metrics=models, design=design, candidates=tuple(c.number for c in pool), diagnostics=diagnostics)


def effects(fit_: Fit, goals: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """Each factor level's effect on each goal metric, against the reference level with every other factor at its
    reference: the posterior mean and 90% interval, in the metric's units. ``goals``: (metric, "higher" | "lower")."""
    out = []
    design = fit_.design
    for metric, better in goals:
        model = fit_.metrics.get(metric)
        if model is None:
            continue
        for f in [*design.factors, *design.extra]:
            for level in f.levels[1:]:
                if level == MISSING:
                    continue
                a = np.zeros(design.p)
                if f in design.factors and design.encoder is not None:
                    ref = tuple(g.levels[0] for g in design.factors)
                    key = tuple(level if g is f else g.levels[0] for g in design.factors)
                    a[: design.n_design] = design.encoder.phi(key) - design.encoder.phi(ref)
                    for j, (tkey, _log, _center, spread) in enumerate(design.trends):
                        here, there = (
                            design.bin_centers.get((f.key, level)),
                            design.bin_centers.get((f.key, f.levels[0])),
                        )
                        if tkey == f.key and here is not None and there is not None:  # the trend between the bins
                            a[design.n_design + len(design.extra_cols) + j] = (here - there) / spread
                else:
                    j = design._extra_index.get((f.key, level))
                    if j is None:
                        continue
                    a[design.n_design + j] = 1.0
                mean, var = model.posterior.predictive(a)
                d = Decided(mean=float(mean[0]) * model.unit, sd=math.sqrt(max(float(var[0]), 0.0)) * model.unit)
                p_positive = 1.0 - _norm_cdf(-d.mean / d.sd) if d.sd > 0 else float(d.mean > 0)
                out.append(
                    {
                        "metric": metric,
                        "better": better,
                        "key": f.key,
                        "level": level,
                        "reference": f.levels[0],
                        "mean": d.mean,
                        "ci90": list(d.ci90),
                        "p_positive": p_positive,
                    }
                )
    return out


def effects_json(fit_: Fit, projection: Projection, computed_at: str) -> dict[str, Any]:
    """``effects.json`` (the explorer's effects view) from the study's model: each factor level's effect on each goal
    metric, with the model's diagnostics."""
    from .projection import Maximize as _Max

    goals = [
        {"role": "objective", "index": i, "metric": o.metric, "better": "higher" if isinstance(o, _Max) else "lower"}
        for i, o in enumerate(projection.objectives)
    ] + [
        {"role": "constraint", "index": i, "metric": c.metric, "better": "lower" if isinstance(c, AtMost) else "higher"}
        for i, c in enumerate(projection.constraints)
    ]
    # once per metric and direction: a metric both optimized and constrained (a floor on the reward) is one model
    found = effects(fit_, list(dict.fromkeys((g["metric"], g["better"]) for g in goals)))
    metrics = {}
    for metric, model in fit_.metrics.items():
        metrics[metric] = {
            "scale": {"kind": "model", "unit": model.unit, "offset": model.offset},
            "observations": int(model.posterior.n),
            "effects": [
                {k: e[k] for k in ("key", "level", "mean", "ci90", "p_positive")}
                for e in found
                if e["metric"] == metric
            ],
            "diagnostics": fit_.diagnostics.get(metric, {}),
        }
    return {
        "schema": 1,
        "computed_at": computed_at,
        "record_events": None,
        "method": "model",
        "factors": [f for f in fit_.design.describe()],
        "omitted": fit_.design.omitted,
        "goals": goals,
        "metrics": metrics,
        "messages": [],
    }


def constraint_value(
    fit_: Fit,
    constraint: Constraint,
    params: Mapping[str, Any],
    number: int | None,
    baseline: Mapping[str, Any],
    baseline_number: int | None,
) -> Decided:
    """The decided value a constraint is judged on: the metric itself, or its difference from the baseline over the
    same cases (``compare="difference"``), or its ratio to it (``compare="ratio"``)."""
    model = fit_.metrics[constraint.metric]
    if constraint.compare == "difference":
        return model.difference(params, baseline, number, baseline_number)
    if constraint.compare == "ratio":
        return model.ratio(params, baseline, number, baseline_number)
    return model.at(params, number)


__all__ = [
    "Decided",
    "Design",
    "Fit",
    "MetricModel",
    "AtLeast",
    "AtMost",
    "constraint_value",
    "effects",
    "fit",
    "p_met",
]
