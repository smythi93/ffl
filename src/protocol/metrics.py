"""Phase 4 of the registered protocol: rankings and outcome metrics.

Everything is computed from the artifacts produced by :mod:`protocol.extract`.

Ties are broken by repeated randomised simulation (10,000 replicates by
default, seeded from the master seed), exactly as pre-registered: a location
inside a group of equally suspicious locations takes a uniformly random
position inside that group, and every rank-based outcome is averaged over the
replicates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from protocol import MASTER_SEED
from sflkit.analysis.suggestion import Location
from sflkit.evaluation import Rank, Scenario

from protocol.formulas import FORMULAS, min_max, suspiciousness

REPEATS = 10_000
EPSILON = 1e-3
LAMBDA = 0.35
AGGREGATION = "max"
IMPACT_THRESHOLD = 0.6
TOP_K = (1, 5, 10)

#: Weighting conditions and how they are reported.
PRIMARY_METHOD = "rf"
VARIANT_METHODS = ("hybrid", "failfirst", "adaptive")
CONTROL_METHODS = ("neg", "pos")


def stable_seed(*parts) -> int:
    """Deterministic seed independent of Python's randomised string hashing."""
    digest = hashlib.blake2b(
        "|".join(str(part) for part in parts).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**63)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def blended_scores(
    base: np.ndarray,
    weights: np.ndarray,
    lam: float,
    epsilon: float,
    unmapped: str = "floor",
    product: str = "raw",
) -> np.ndarray:
    """susp(l) = (1-lambda) norm(susp'(l)) + lambda norm(susp'(l) * w(l)).

    ``unmapped`` controls what happens to lines whose mapped features all carry
    the minimum importance. The registered method floors them at ``epsilon``
    (``"floor"``). ``"neutral"`` instead leaves their baseline score untouched
    and is used only for the clearly labelled post-hoc probe that asks whether
    an absent effect originates in the floor or in the importances themselves.
    """
    weights = np.asarray(weights, dtype=float)
    if unmapped == "neutral":
        w = np.where(weights > 0.0, epsilon + (1.0 - epsilon) * weights, 1.0)
    else:
        w = epsilon + (1.0 - epsilon) * weights
    #: ``product`` selects what the weight multiplies. "raw" is the registered
    #: rule, susp'(l) x w(l). Naish2 is the one formula in the set whose scores
    #: go negative, and multiplying a negative score by a small weight moves it
    #: UP, so the weighting is inverted on exactly those lines. "prenorm"
    #: multiplies the already-normalised score instead, which keeps the operand
    #: non-negative for every formula. It is a clearly labelled post-hoc probe
    #: and never feeds a confirmatory result.
    operand = min_max(base) if product == "prenorm" else base
    return (1.0 - lam) * min_max(base) + lam * min_max(operand * w)


def file_top1(scores: np.ndarray, files: Sequence[str], faulty_files: Set[str]) -> float:
    """Probability that the first file inspected contains a faulty line.

    A file is scored by its most suspicious line; among the files tied for the
    top score, one is taken uniformly at random.
    """
    if not len(scores) or not faulty_files:
        return 0.0
    best: Dict[str, float] = {}
    for score, name in zip(np.asarray(scores, dtype=float), files):
        if score > best.get(name, -np.inf):
            best[name] = score
    top = max(best.values())
    tied = [name for name, value in best.items() if value == top]
    return sum(name in faulty_files for name in tied) / len(tied)


def tiebreak_scores(base: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Post-hoc combination rule: weights only break ties of the baseline.

    Distinct baseline scores keep their exact order; within a group of equal
    baseline scores, lines are ordered by weight. The weight term is scaled to
    half the smallest gap between distinct baseline scores, so it can never
    carry a line past one with a different baseline score.
    """
    normalized = min_max(np.asarray(base, dtype=float))
    distinct = np.unique(normalized)
    gap = float(np.diff(distinct).min()) if len(distinct) > 1 else 1.0
    return normalized + 0.5 * gap * min_max(np.asarray(weights, dtype=float))


@dataclass(frozen=True)
class Groups:
    """Tie groups of a ranking, ordered by decreasing suspiciousness."""

    group_of: np.ndarray  # per location: index of its tie group
    before: np.ndarray  # per group: number of locations ranked strictly higher
    size: np.ndarray  # per group: number of locations in the group

    @property
    def n_locations(self) -> int:
        return int(self.size.sum())


def tie_groups(scores: np.ndarray) -> Groups:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    # A new group starts wherever the score changes.
    starts = np.empty(ordered.shape, dtype=bool)
    starts[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts[1:])
    group_ids = np.cumsum(starts) - 1
    size = np.bincount(group_ids)
    before = np.concatenate([[0], np.cumsum(size)[:-1]])
    group_of = np.empty(scores.shape, dtype=np.int64)
    group_of[order] = group_ids
    return Groups(group_of=group_of, before=before, size=size)


def _sample_positions(k: int, f: int, repeats: int, rng: np.random.Generator) -> np.ndarray:
    """``repeats`` x ``f`` distinct positions in 1..k, uniform over f-subsets.

    Tie groups can hold tens of thousands of locations, so materialising a
    ``repeats`` x ``k`` matrix of sort keys is only affordable when the group is
    small relative to the number of targets. Otherwise we draw with replacement
    and redraw the rows that collided, which yields the same uniform
    distribution over f-subsets at a cost independent of ``k``.
    """
    if f >= k:
        return np.tile(np.arange(1, k + 1, dtype=np.int64), (repeats, 1))
    if f == 1:
        return rng.integers(1, k + 1, size=(repeats, 1), dtype=np.int64)
    if k <= 4 * f:
        keys = rng.random((repeats, k))
        return np.argpartition(keys, f - 1, axis=1)[:, :f].astype(np.int64) + 1
    draw = rng.integers(1, k + 1, size=(repeats, f), dtype=np.int64)
    for _ in range(200):
        ordered = np.sort(draw, axis=1)
        collided = (ordered[:, 1:] == ordered[:, :-1]).any(axis=1)
        if not collided.any():
            break
        draw[collided] = rng.integers(
            1, k + 1, size=(int(collided.sum()), f), dtype=np.int64
        )
    return draw


def simulate_ranks(
    groups: Groups,
    targets: Sequence[int],
    repeats: int,
    rng: np.random.Generator,
    default_rank: float,
    n_missing: int,
) -> np.ndarray:
    """Ranks of every target location for each tie-breaking replicate."""
    columns: List[np.ndarray] = []
    if len(targets):
        target_groups = groups.group_of[np.asarray(targets, dtype=np.int64)]
        for group in np.unique(target_groups):
            members = int((target_groups == group).sum())
            positions = _sample_positions(
                int(groups.size[group]), members, repeats, rng
            )
            columns.append(positions + int(groups.before[group]))
    if n_missing:
        columns.append(np.full((repeats, n_missing), default_rank, dtype=float))
    if not columns:
        return np.empty((repeats, 0), dtype=float)
    return np.concatenate([c.astype(float) for c in columns], axis=1)


def _scenario_values(ranks: np.ndarray) -> Dict[str, np.ndarray]:
    """Best, average and worst debugging case over the target set.

    The three cases differ in how much of a multi-line fault a developer has to
    see: the best case stops at the first faulty line, the worst case needs all
    of them, and the average case stops around half.

    The average-case index deliberately matches SFLKit's ``Ranking.get_rank``,
    which uses ``len(faulty) // 2 - 1``. For an odd target count that selects
    slightly fewer than half the targets, so it is marginally more optimistic
    than a literal "50% of faulty lines" reading. It is kept because the
    protocol defines these cases as SFLKit computes them, and because the
    numbers must stay comparable with the EFDD results.
    """
    ordered = np.sort(ranks, axis=1)
    median_index = max(ordered.shape[1] // 2 - 1, 0)
    return {
        "best": ordered[:, 0],
        "avg": ordered[:, median_index],
        "worst": ordered[:, -1],
    }


def build_rank(
    scores: np.ndarray,
    locations: Sequence,
    n_locations: int,
):
    """A SFLKit :class:`Rank` over ``scores``.

    SFLKit owns the definition of these metrics, so the ranking is built with its
    machinery rather than mirrored here. ``Rank`` groups equally scored locations
    into one suggestion and gives every member the group's midpoint rank, which
    is what makes EXAM and wasted effort deterministic: once the ranking
    exists, each location has exactly one rank and there is no tie left to break.
    """
    from sflkit.analysis.suggestion import Suggestion

    grouped: Dict[float, List] = {}
    for score, location in zip(np.asarray(scores, dtype=float), locations):
        grouped.setdefault(float(score), []).append(location)
    suggestions = [Suggestion(lines, score) for score, lines in grouped.items()]
    return Rank(suggestions, total_number_of_locations=max(n_locations, len(scores)))


def rank_outcomes(
    scores: np.ndarray,
    targets: Sequence[int],
    n_missing: int,
    n_locations: int,
    locations: Sequence,
    faulty: Sequence,
    repeats: int = REPEATS,
    top_k: Sequence[int] = TOP_K,
    seed: int = MASTER_SEED,
) -> Dict[str, float]:
    """Top-k, EXAM and wasted effort for one ranking and one target set.

    Only top-$k$ samples. It asks whether a faulty location is among the ``k``
    highest scored, and when the cut-off falls inside a group of equally scored
    locations that answer genuinely depends on which of them is taken, so SFLKit
    draws repeatedly, from a generator seeded per ``k`` from ``seed``. Groups
    entirely above the cut-off are always taken; this needs SFLKit's fixed
    ``Rank.top_n`` (branch fix/top-n-tie-breaking), which breaks ties only
    inside the group the cut-off falls into. EXAM and wasted effort read a
    location's rank straight off the ranking and need no sampling at all.
    """
    if not len(targets) and not n_missing:
        return {}
    ranking = build_rank(scores, locations, n_locations)
    faulty_set = set(faulty)
    outcomes: Dict[str, float] = {}
    for k in top_k:
        outcomes[f"top{k}"] = float(
            ranking.top_n(
                faulty_set,
                k,
                scenario=Scenario.BEST_CASE,
                repeat=repeats,
                seed=stable_seed(seed, "top", k),
            )
        )
    for name, scenario in (
        ("best", Scenario.BEST_CASE),
        ("avg", Scenario.AVG_CASE),
        ("worst", Scenario.WORST_CASE),
    ):
        outcomes[f"exam_{name}"] = float(ranking.exam(faulty_set, scenario))
        outcomes[f"wasted_{name}"] = float(ranking.wasted_effort(faulty_set, scenario))
    outcomes["min_rank"] = float(ranking.get_rank(faulty_set, Scenario.BEST_CASE))
    return outcomes


def impact_outcomes(
    scores: np.ndarray,
    targets: Sequence[int],
    n_locations: int,
    rng: np.random.Generator,
    repeats: int = REPEATS,
    top_k: Sequence[int] = TOP_K,
    impact_set_size: int = 0,
) -> Dict[str, float]:
    """impact-found@k, minimum impact rank and impact locations ranked."""
    if not len(targets):
        return {
            **{f"found@{k}": 0.0 for k in top_k},
            **{f"recall@{k}": 0.0 for k in top_k},
            "min_rank": float(max(n_locations, len(scores))),
            "total_ranked": 0.0,
        }
    groups = tie_groups(scores)
    ranks = simulate_ranks(groups, targets, repeats, rng, float(n_locations), 0)
    minimum = ranks.min(axis=1)
    outcomes: Dict[str, float] = {}
    denominator = float(impact_set_size or len(targets))
    for k in top_k:
        outcomes[f"found@{k}"] = float((minimum <= k).mean())
        outcomes[f"recall@{k}"] = float(
            ((ranks <= k).sum(axis=1) / denominator).mean()
        )
    outcomes["min_rank"] = float(minimum.mean())
    outcomes["total_ranked"] = float(len(targets))
    return outcomes


def function_outcomes(
    scores: np.ndarray,
    line_functions: np.ndarray,
    function_lengths: np.ndarray,
    faulty_functions: Set[int],
    rng: np.random.Generator,
    repeats: int = REPEATS,
) -> Dict[str, float]:
    """Function-level outcomes.

    Functions are scored by the maximum suspiciousness of their ranked lines,
    ranked with the same randomised tie-breaking, and turned into a probability
    mass by normalising the scores.
    """
    valid = line_functions >= 0
    if not valid.any():
        return {}
    ids = line_functions[valid]
    values = scores[valid]
    unique, inverse = np.unique(ids, return_inverse=True)
    function_scores = np.full(unique.shape, -np.inf)
    np.maximum.at(function_scores, inverse, values)
    n_functions = len(unique)
    lengths = function_lengths[unique].astype(float)

    positive = np.clip(function_scores, 0.0, None)
    total_mass = positive.sum()
    mass = (
        positive / total_mass
        if total_mass > 0
        else np.full(n_functions, 1.0 / n_functions)
    )
    faulty = np.array([int(f) in faulty_functions for f in unique], dtype=bool)
    outcomes: Dict[str, float] = {
        "function_faulty_mass": float(mass[faulty].sum()),
        "functions_ranked": float(n_functions),
        "faulty_functions_ranked": float(faulty.sum()),
    }
    if not faulty.any():
        outcomes.update(
            {
                "function_mass@1": 0.0,
                "function_mass@3": 0.0,
                "function_mass@5": 0.0,
                "function_exam": 1.0,
                "function_effort_lines": 1.0,
            }
        )
        return outcomes

    groups = tie_groups(function_scores)
    targets = np.where(faulty)[0]
    ranks = simulate_ranks(groups, targets, repeats, rng, float(n_functions), 0)
    minimum = ranks.min(axis=1)
    for k in (1, 3, 5):
        # Expected probability mass of faulty functions inside the top k.
        hit = (ranks <= k).astype(float)
        outcomes[f"function_mass@{k}"] = float((hit * mass[targets]).sum(axis=1).mean())
    outcomes["function_exam"] = float((minimum / n_functions).mean())

    # Practical effort: source lines inside the functions inspected until the
    # first faulty function is reached, normalised by all ranked function lines.
    order_lengths = np.empty(n_functions, dtype=float)
    ordering = np.argsort(-function_scores, kind="stable")
    order_lengths[:] = lengths[ordering]
    cumulative = np.cumsum(order_lengths)
    total_lines = float(cumulative[-1]) if n_functions else 1.0
    index = np.clip(minimum.astype(int) - 1, 0, n_functions - 1)
    outcomes["function_effort_lines"] = float(
        (cumulative[index] / max(total_lines, 1.0)).mean()
    )
    return outcomes


# ---------------------------------------------------------------------------
# Test-impact ground truth
# ---------------------------------------------------------------------------
def test_impact_indices(
    ef: np.ndarray, ep: np.ndarray, threshold: float = IMPACT_THRESHOLD
) -> np.ndarray:
    observed = ef > 0
    ratio = np.divide(
        ef, np.maximum(ef + ep, 1), out=np.zeros(ef.shape, dtype=float), where=True
    )
    selected = observed & ((ep == 0) | (ratio >= threshold))
    if not selected.any():
        selected = observed
    return np.where(selected)[0]


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    lam: float = LAMBDA
    epsilon: float = EPSILON
    aggregation: str = AGGREGATION
    threshold: float = IMPACT_THRESHOLD
    repeats: int = REPEATS
    unmapped: str = "floor"
    product: str = "raw"

    def tag(self) -> str:
        #: The tag seeds the tie-breaking RNG, so the registered configuration
        #: must keep producing the exact string it always has. The probe suffix
        #: is appended only when the probe is active.
        base = (
            f"lam{self.lam:g}_eps{self.epsilon:g}_"
            f"{self.aggregation}_thr{self.threshold:g}_{self.unmapped}"
        )
        return base if self.product == "raw" else f"{base}_{self.product}"


def subject_rows(
    artifact: Dict,
    methods: Sequence[str],
    config: Config = Config(),
    formulas: Sequence[str] = FORMULAS,
) -> List[Dict]:
    """One row per (subject, formula, method) with all outcome families."""
    lines = artifact["lines"]
    files = lines["file"]
    numbers = lines["line"]
    ef = np.asarray(lines["ef"], dtype=float)
    ep = np.asarray(lines["ep"], dtype=float)
    tf = int(artifact["totals"]["failed"])
    tp = int(artifact["totals"]["passed"])
    n_ranked = len(files)
    #: EXAM is the share of the *ranked* locations (the lines the tests
    #: executed) inspected before the fault. Dividing by the project's total
    #: LOC instead would shrink EXAM about tenfold, since most of a project is
    #: never executed by the tests and so never ranked. A fault line that is not
    #: ranked at all is charged the end of the ranking (EXAM 1).
    n_locations = n_ranked

    position = {(f, int(l)): i for i, (f, l) in enumerate(zip(files, numbers))}
    #: SFLKit owns the ranking, and it keys everything on Location objects.
    sfl_locations = [Location(f, int(l)) for f, l in zip(files, numbers)]
    reference = [tuple(x) for x in artifact["reference_lines"]]
    hunks = [tuple(x) for x in artifact["hunk_lines"]]
    reference_idx = [position[(f, int(l))] for f, l in reference if (f, int(l)) in position]
    reference_missing = len(reference) - len(reference_idx)
    #: The faulty set keeps every reference line, including those the ranking
    #: never saw: SFLKit charges those its default rank rather than dropping them.
    reference_locations = [Location(f, int(l)) for f, l in reference]
    hunk_idx = [position[(f, int(l))] for f, l in hunks if (f, int(l)) in position]
    impact_idx = test_impact_indices(ef, ep, config.threshold)

    line_functions = np.asarray(lines["function"], dtype=np.int64)
    faulty_files = {f for f, _ in reference} | {f for f, _ in hunks}
    table = artifact["functions"]
    function_lengths = np.array(
        [max(1, int(end) - int(start) + 1) for _, _, start, end in table], dtype=float
    )
    faulty_functions = {
        int(i)
        for i in list(artifact["reference_functions"]) + list(artifact["hunk_functions"])
        if int(i) >= 0
    }

    seed = int(artifact.get("seed", MASTER_SEED))
    rows: List[Dict] = []
    for formula in formulas:
        base = suspiciousness(formula, ef, ep, tf, tp)
        for method in methods:
            if method.startswith("s_") and method not in artifact["weights"]:
                if "_subset_weights" not in artifact:
                    from pathlib import Path as _Path

                    from protocol.subsets import subset_weights

                    artifact["_subset_weights"] = subset_weights(
                        artifact, _Path(__file__).resolve().parents[2] / "vectors")
                if method not in artifact["_subset_weights"]:
                    continue
                artifact["weights"][method] = artifact["_subset_weights"][method]
            if method.startswith("h_"):
                from protocol.hierarchical import hierarchical_scores, parse_method

                core, _, extra_name = method.partition("__")
                mode, source = parse_method(core)
                if extra_name and extra_name not in artifact["weights"]:
                    continue
                extra = (np.asarray(artifact["weights"][extra_name][config.aggregation], dtype=float)
                         if extra_name else None)
                if mode == "gran":
                    from protocol.hierarchical import granularity_weights

                    source_weights = granularity_weights(
                        artifact["weights"], source, config.aggregation)
                    if source_weights is None:
                        continue
                elif source is not None and source not in artifact["weights"]:
                    continue
                else:
                    source_weights = (
                        None if source is None
                        else np.asarray(artifact["weights"][source][config.aggregation],
                                        dtype=float)
                    )
                scores = hierarchical_scores(base, source_weights, files, line_functions, mode,
                                             extra)
            elif method.startswith("lex_"):
                #: Lexicographic: the weight first, SFL among equal weights
                #: (e.g. the slice as a filter in front of the SFL ranking).
                from protocol.hierarchical import _dense_scores

                source = method[len("lex_"):]
                if source not in artifact["weights"]:
                    continue
                weights = np.asarray(artifact["weights"][source][config.aggregation], dtype=float)
                scores = _dense_scores(np.column_stack([weights, min_max(base)]))
            elif method != "baseline" and method not in artifact["weights"]:
                continue  # condition not present for this subject
            elif method == "baseline":
                scores = base
            else:
                weights = np.asarray(
                    artifact["weights"][method][config.aggregation], dtype=float
                )
                if method.startswith("tie_"):
                    scores = tiebreak_scores(base, weights)
                else:
                    scores = blended_scores(
                        base, weights, config.lam, config.epsilon, config.unmapped,
                        config.product,
                    )
            rng = np.random.default_rng(
                stable_seed(seed, formula, method, config.tag())
            )
            row: Dict[str, object] = {
                "subject": artifact["subject"],
                "project": artifact["project"],
                "bug_id": artifact["bug_id"],
                "formula": formula,
                "method": method,
                "file_top1": file_top1(scores, files, faulty_files),
                **{
                    f"patch_{k}": v
                    for k, v in rank_outcomes(
                        scores,
                        reference_idx,
                        reference_missing,
                        n_locations,
                        sfl_locations,
                        reference_locations,
                        repeats=config.repeats,
                        seed=stable_seed(seed, formula, method, config.tag()),
                    ).items()
                },
                **{
                    f"hunk_{k}": v
                    for k, v in impact_outcomes(
                        scores,
                        hunk_idx,
                        n_locations,
                        rng,
                        repeats=config.repeats,
                        impact_set_size=len(hunks),
                    ).items()
                },
                **{
                    f"impact_{k}": v
                    for k, v in impact_outcomes(
                        scores,
                        impact_idx,
                        n_locations,
                        rng,
                        repeats=config.repeats,
                        impact_set_size=len(impact_idx),
                    ).items()
                },
                **function_outcomes(
                    scores,
                    line_functions,
                    function_lengths,
                    faulty_functions,
                    rng,
                    repeats=config.repeats,
                ),
            }
            rows.append(row)
    return rows


def confounders(artifact: Dict) -> Dict[str, float]:
    """Pre-registered confounders and feasibility descriptors of one subject."""
    lines = artifact["lines"]
    ef = np.asarray(lines["ef"], dtype=float)
    ep = np.asarray(lines["ep"], dtype=float)
    tf = int(artifact["totals"]["failed"])
    tp = int(artifact["totals"]["passed"])
    n_ranked = len(lines["line"])
    diagnostics = artifact.get("diagnostics", {})
    coverage_density = float((ef + ep).sum() / max(1, n_ranked * (tf + tp)))
    reference = artifact["reference_lines"]
    return {
        "subject": artifact["subject"],
        "project": artifact["project"],
        "loc": float(artifact.get("loc") or 0),
        "executable_lines": float(n_ranked),
        "test_suite_size": float(tf + tp),
        "failing_tests": float(tf),
        "passing_tests": float(tp),
        "coverage_density": coverage_density,
        "patch_size": float(len(reference)),
        "hunk_size": float(len(artifact["hunk_lines"])),
        "n_functions": float(len(artifact["functions"])),
        "constant_fraction": float(diagnostics.get("constant_fraction", 0.0)),
        "n_features_total": float(diagnostics.get("n_features_total", 0)),
        "n_features_used": float(diagnostics.get("n_features_used", 0)),
        "mapping_conflict_rate": float(diagnostics.get("mapping_conflict_rate", 0.0)),
        "line_evidence_rate": float(diagnostics.get("line_evidence_rate", 0.0)),
        "importance_gini": float(diagnostics.get("importance", {}).get("gini", 0.0)),
        "importance_entropy": float(
            diagnostics.get("importance", {}).get("entropy", 0.0)
        ),
        "importance_skewness": float(
            diagnostics.get("importance", {}).get("skewness", 0.0)
        ),
        "weak_signal": float(bool(diagnostics.get("weak_signal", False))),
        "runtime_total": float(artifact.get("timing", {}).get("total", 0.0)),
    }
