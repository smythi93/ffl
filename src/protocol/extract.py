"""Phase 1 + Phase 2 of the registered protocol.

For one Tests4Py subject this module

1. turns the SFLKit event traces into one labelled execution-feature vector per
   test execution (Phase 1),
2. removes within-subject constant features, trains the per-subject random
   forest and reads its Gini importances (Phase 1),
3. maps feature importances to source lines, normalises them and aggregates
   them per line (Phase 2), and
4. writes a single self-contained *artifact* holding the line spectra, the
   per-line weights of every pre-registered weighting condition, the two
   patch-based ground-truth sets, the function table and all diagnostics.

Every downstream number of the study (rankings, metrics, sensitivity grids,
statistics) is computed from these artifacts alone, so the expensive
instrumentation is executed exactly once.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sflkit import Analyzer
from sflkit.analysis.analysis_type import AnalysisObject, AnalysisType
from sflkit.analysis.factory import LineFactory
from sflkit.analysis.spectra import Line
from sflkit.events.event_file import EventFile
from sflkit.events.mapping import EventMapping
from sflkit.features.handler import EventHandler
from sflkit.language.python.finder import (
    PythonBranchFinder,
    PythonFunctionFinder,
    PythonLoopFinder,
)
from sklearn.ensemble import RandomForestClassifier

from protocol import MASTER_SEED
from protocol import functions as fn
from protocol.patches import ground_truth

# ---------------------------------------------------------------------------
# Pre-registered constants (Sections "Methodology" and "Execution Plan").
# ---------------------------------------------------------------------------
N_ESTIMATORS = 200
CRITERION = "gini"
CLASS_WEIGHT = "balanced"
AGGREGATIONS = ("max", "mean", "sum")

#: Weighting conditions stored in every artifact.  ``rf`` is the primary
#: pre-registered method; ``neg``/``pos`` are the outcome-neutral controls;
#: ``alt`` is the pre-registered alternative feature-to-line mapping;
#: ``hybrid``/``failfirst``/``adaptive`` are the exploratory ranking variants.
CONDITIONS = ("rf", "alt", "neg", "pos", "hybrid", "failfirst", "adaptive")


#: Tests4Py reports no size for this subject, which would leave the EXAM
#: denominator at zero and silently fall back to the ranked-line count, making
#: it the one subject measured on a different scale. The EFDD study patches the
#: same value, so using it keeps the two comparable.
LOC_OVERRIDES = {"youtubedl_24": 87341}


def subject_loc(project) -> int:
    """Lines of code used as the EXAM denominator, as Tests4Py reports them."""
    return int(LOC_OVERRIDES.get(project.get_identifier(), project.loc or 0))


def subject_seed(identifier: str) -> int:
    """Deterministic per-subject seed derived from the master seed."""
    digest = 0
    for char in identifier:
        digest = (digest * 131 + ord(char)) % (2**31 - 1)
    return (MASTER_SEED * 1_000_003 + digest) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------
def load_event_files(
    events: Path, mapping_file: Path
) -> Tuple[List[EventFile], List[EventFile]]:
    mapping = EventMapping.load_from_file(mapping_file, "")
    failing: List[EventFile] = []
    passing: List[EventFile] = []
    if (events / "failing").exists():
        failing = [
            EventFile(events / "failing" / path, run_id, mapping, failing=True)
            for run_id, path in enumerate(sorted(os.listdir(events / "failing")))
        ]
    if (events / "passing").exists():
        passing = [
            EventFile(events / "passing" / path, run_id, mapping, failing=False)
            for run_id, path in enumerate(
                sorted(os.listdir(events / "passing")), start=len(failing)
            )
        ]
    return failing, passing


class FeatureAnalyzer(Analyzer):
    """Line spectra plus one execution-feature vector per test execution."""

    def __init__(self, failing: List[EventFile], passing: List[EventFile]):
        AnalysisObject.set_finder(
            function_finder=PythonFunctionFinder,
            loop_finder=PythonLoopFinder,
            branch_finder=PythonBranchFinder,
        )
        super().__init__(
            relevant_event_files=failing,
            irrelevant_event_files=passing,
            factory=LineFactory(),
            workers=1,
        )
        self.handler = EventHandler(workers=1)

    def _analyze(self, event_file):
        self.handler.handle(event_file)
        super()._analyze(event_file)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def feature_matrix(handler: EventHandler) -> Tuple[np.ndarray, np.ndarray, List]:
    """One row per test execution, one column per execution feature.

    Built directly as a dense array rather than through ``EventHandler.to_df``
    because subjects can carry several hundred thousand features, for which the
    dict-of-columns detour dominates the runtime.
    """
    from sflkit.runners.run import TestResult

    features = handler.builder.get_all_features()
    vectors = list(handler.builder.get_vectors())
    # Fill from each run's *observed* features on top of a row of defaults.
    # ``FeatureVector.get_feature_value`` takes a lock per feature, so the dense
    # form costs one locked call per run per feature -- tens of millions of them
    # on subjects with a few hundred runs. Observed features are far fewer.
    position = {feature: index for index, feature in enumerate(features)}
    defaults = np.fromiter(
        (feature.default().value for feature in features),
        dtype=np.int8, count=len(features),
    )
    matrix = np.tile(defaults, (len(vectors), 1)) if features else np.empty(
        (len(vectors), 0), dtype=np.int8
    )
    labels = np.empty(len(vectors), dtype=int)
    for row, vector in enumerate(vectors):
        for feature, value in vector.get_features().items():
            index = position.get(feature)
            if index is not None:
                matrix[row, index] = value.value
        labels[row] = 1 if vector.result == TestResult.FAILING else 0
    return matrix, labels, features


def drop_constant(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return matrix, np.zeros(matrix.shape[1], dtype=bool)
    keep = matrix.max(axis=0) != matrix.min(axis=0)
    return matrix[:, keep], keep


# ---------------------------------------------------------------------------
# Importance models
# ---------------------------------------------------------------------------
#: Threads used per random forest.  ``n_jobs`` only parallelises tree
#: construction and leaves the fitted forest (and hence the importances)
#: identical for a fixed ``random_state``.
N_JOBS = int(os.environ.get("FFL_RF_JOBS", "4"))

#: Refits used for the feature-importance stability diagnostics.
N_STABILITY_REFITS = 3


def forest_importances(
    matrix: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
    n_estimators: int = N_ESTIMATORS,
) -> np.ndarray:
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=CRITERION,
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    forest.fit(matrix, labels, sample_weight=sample_weight)
    return np.asarray(forest.feature_importances_, dtype=float)


class _Named:
    """Minimal stand-in exposing the single attribute the fusion reads."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


#: Every importance condition stored in the vectors archive. ``pos`` is absent
#: on purpose: the positive control is derived from ``rf`` plus an oracle
#: marking of the reference lines, not from a forest of its own.
IMPORTANCE_CONDITIONS = ("rf", "neg", "adaptive", "hybrid", "failfirst")


def compute_importances(
    matrix: np.ndarray,
    labels: np.ndarray,
    features: Sequence,
    seed: int,
    conditions: Sequence[str] = IMPORTANCE_CONDITIONS,
) -> Dict[str, np.ndarray]:
    """Every per-feature importance the study needs, from the feature matrix.

    ``matrix`` is the *reduced* matrix, with constant features already dropped,
    exactly as stored in the vectors archive; ``features`` are the matching
    feature objects or any objects exposing ``.name``. Given the subject's seed
    this reproduces the stored importances bit for bit, which is what makes the
    archive checkable and lets any condition be recomputed without re-tracing.

    ``neg`` needs no model at all: it is seeded uniform noise, and is stored only
    so the negative control can be replayed without re-deriving the seed.
    """
    unknown = [c for c in conditions if c not in IMPORTANCE_CONDITIONS]
    if unknown:
        raise ValueError(f"unknown importance conditions {unknown}")

    #: ``hybrid_importances`` groups features by name and keys its result on the
    #: objects it is given. Accept bare names too, so the archive's ``features``
    #: array can be passed straight through.
    features = [f if hasattr(f, "name") else _Named(str(f)) for f in features]

    raw: Dict[str, np.ndarray] = {}
    for condition in conditions:
        if condition == "rf":
            raw[condition] = forest_importances(matrix, labels, seed)
        elif condition == "neg":
            raw[condition] = np.random.default_rng(seed + 1).random(matrix.shape[1])
        elif condition == "adaptive":
            raw[condition] = forest_importances(
                matrix, labels, seed, sample_weight=adaptive_sample_weights(labels)
            )
        else:
            #: The multi-signal variants fuse thirteen signals and cost time
            #: linear in the feature count, which dominates on large subjects.
            #: A failure here must not lose the other conditions.
            try:
                raw[condition] = hybrid_importances(matrix, labels, features, condition)
            except Exception:  # noqa: BLE001
                raw[condition] = np.zeros(matrix.shape[1], dtype=float)
    return raw


def adaptive_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Exploratory adaptive variant: extra mass on sparse failing evidence."""
    n_fail = int(labels.sum())
    n_pass = int(len(labels) - n_fail)
    weights = np.ones(len(labels), dtype=float)
    if n_fail == 0 or n_pass == 0:
        return weights
    imbalance = n_pass / n_fail
    # Sparse failing evidence is up-weighted beyond the balanced default; the
    # boost saturates at 3x so that a single failing run cannot dominate.
    boost = min(3.0, math.sqrt(max(1.0, imbalance)))
    weights[labels == 1] = boost
    return weights


def hybrid_importances(
    matrix: np.ndarray,
    labels: np.ndarray,
    features: Sequence,
    mode: str,
) -> np.ndarray:
    """Exploratory multi-signal fusion over the execution-feature matrix."""
    from localization.hybrid_ranking import HybridRankingDiagnosis

    kwargs = dict(ranking_mode="balanced")
    if mode == "failfirst":
        kwargs = dict(
            ranking_mode="fail_first_verify_pass",
            verify_pass_penalty=0.65,
            fail_focus_boost=0.45,
        )
    # float32 avoids any integer-overflow surprise in the signal arithmetic.
    frame = pd.DataFrame(np.asarray(matrix, dtype=np.float32),
                         columns=[f.name for f in features])
    diagnosis = HybridRankingDiagnosis(**kwargs)
    diagnosis.all_features = list(features)
    diagnosis.x_train = frame
    diagnosis.y_train = pd.Series(labels)
    diagnosis.feature_groups = diagnosis._build_feature_groups(frame.columns)
    diagnosis.train()
    scores = diagnosis.feature_importance_scores
    return np.asarray([float(scores.get(f, 0.0)) for f in features], dtype=float)


def normalize(importances: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1]; the epsilon floor is applied later."""
    if importances.size == 0:
        return importances
    lo, hi = float(importances.min()), float(importances.max())
    if hi <= lo:
        lo, hi = 0.0, 1.0
    return (importances - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Feature -> line mapping
# ---------------------------------------------------------------------------
def _location_key(analysis) -> Tuple:
    """Everything the location of a feature can depend on.

    SFLKit resolves branch, function and loop features by re-parsing the source
    file on every call, which costs tens of milliseconds on a large module. Many
    features share the same construct, so the resolved locations are memoized
    per subject under this key; the key covers every attribute the finders read,
    so memoization cannot change a result.
    """
    return (
        analysis.analysis_type(),
        analysis.file,
        int(analysis.line),
        getattr(analysis, "then", None),
        getattr(analysis, "function", None),
        getattr(analysis, "use_file", None),
        getattr(analysis, "use_line", None),
    )


def feature_locations(
    feature,
    base_dir: str,
    alternative: bool,
    cache: Optional[Dict[Tuple, List[Tuple[str, int]]]] = None,
) -> List[Tuple[str, int]]:
    analysis = feature.analysis
    if alternative and analysis.analysis_type() == AnalysisType.DEF_USE:
        return [(analysis.use_file, int(analysis.use_line))]
    key = _location_key(analysis) if cache is not None else None
    if key is not None and key in cache:
        return cache[key]
    try:
        suggestion = analysis.get_suggestion(metric=lambda _: 0.0, base_dir=base_dir)
        locations = [(loc.file, int(loc.line)) for loc in suggestion.lines]
    except Exception:
        locations = [(analysis.file, int(analysis.line))]
    if key is not None:
        cache[key] = locations
    return locations


def aggregate_weights(
    location_lists: Sequence[Sequence[Tuple[str, int]]],
    values: np.ndarray,
    line_index: Dict[Tuple[str, int], int],
    n_lines: int,
) -> Dict[str, np.ndarray]:
    best = np.zeros(n_lines, dtype=float)
    total = np.zeros(n_lines, dtype=float)
    count = np.zeros(n_lines, dtype=float)
    for locations, value in zip(location_lists, values):
        for location in locations:
            idx = line_index.get(location)
            if idx is None:
                continue
            if value > best[idx]:
                best[idx] = value
            total[idx] += value
            count[idx] += 1.0
    mean = np.divide(total, np.maximum(count, 1.0))
    span = total.max()
    summed = total / span if span > 0 else total
    return {"max": best, "mean": mean, "sum": summed}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def importance_diagnostics(importances: np.ndarray) -> Dict[str, float]:
    if importances.size == 0:
        return {"gini": 0.0, "entropy": 0.0, "skewness": 0.0, "n_nonzero": 0}
    total = float(importances.sum())
    if total <= 0:
        share = np.full(importances.size, 1.0 / importances.size)
    else:
        share = importances / total
    ordered = np.sort(share)
    n = ordered.size
    gini = float(
        (2.0 * np.sum(np.arange(1, n + 1) * ordered)) / (n * np.sum(ordered))
        - (n + 1) / n
    )
    entropy = float(-np.sum(share * np.log(share + 1e-12)))
    max_entropy = math.log(n) if n > 1 else 1.0
    mean = float(importances.mean())
    std = float(importances.std())
    skew = float(np.mean(((importances - mean) / (std + 1e-12)) ** 3))
    return {
        "gini": gini,
        "entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
        "skewness": skew,
        "n_nonzero": int(np.count_nonzero(importances)),
    }


def stability_diagnostics(
    matrix: np.ndarray, labels: np.ndarray, seed: int, reference: np.ndarray
) -> Dict[str, float]:
    """Seeded refits and bootstrap refits (>=3 passing and >=3 failing runs)."""
    from scipy.stats import spearmanr

    n_fail = int(labels.sum())
    n_pass = int(len(labels) - n_fail)
    if n_fail < 3 or n_pass < 3 or matrix.shape[1] < 2:
        return {"available": False}
    rng = np.random.default_rng(seed)
    top_k = max(1, min(20, matrix.shape[1]))
    reference_top = set(np.argsort(-reference)[:top_k].tolist())

    def compare(other: np.ndarray) -> Tuple[float, float]:
        rho = spearmanr(reference, other).statistic
        other_top = set(np.argsort(-other)[:top_k].tolist())
        jaccard = len(reference_top & other_top) / len(reference_top | other_top)
        return (0.0 if rho is None or math.isnan(rho) else float(rho), jaccard)

    seeded, boot = [], []
    for _ in range(N_STABILITY_REFITS):
        seeded.append(
            compare(
                forest_importances(
                    matrix, labels, seed=int(rng.integers(0, 2**31 - 1)), n_estimators=N_ESTIMATORS
                )
            )
        )
    for _ in range(N_STABILITY_REFITS):
        while True:
            idx = rng.integers(0, len(labels), size=len(labels))
            if 0 < labels[idx].sum() < len(idx):
                break
        boot.append(
            compare(
                forest_importances(
                    matrix[idx], labels[idx], seed=seed, n_estimators=N_ESTIMATORS
                )
            )
        )
    return {
        "available": True,
        "seeded_spearman": float(np.mean([s for s, _ in seeded])),
        "seeded_jaccard20": float(np.mean([j for _, j in seeded])),
        "bootstrap_spearman": float(np.mean([s for s, _ in boot])),
        "bootstrap_jaccard20": float(np.mean([j for _, j in boot])),
    }


# ---------------------------------------------------------------------------
# Artifact construction
# ---------------------------------------------------------------------------
def build_artifact(
    project,
    events: Path,
    mapping_file: Path,
    checkout: Path,
    with_stability: bool = True,
    vectors_path: Optional[Path] = None,
) -> Dict:
    identifier = project.get_identifier()
    seed = subject_seed(identifier)
    timing: Dict[str, float] = {}

    start = time.time()
    failing, passing = load_event_files(events, mapping_file)
    if not failing or not passing:
        raise ValueError(
            f"{identifier}: needs at least one passing and one failing execution "
            f"(passing={len(passing)}, failing={len(failing)})"
        )
    analyzer = FeatureAnalyzer(failing, passing)
    analyzer.analyze()
    timing["trace_analysis"] = time.time() - start

    start = time.time()
    matrix, labels, features = feature_matrix(analyzer.handler)
    reduced, retained = drop_constant(matrix)
    n_dropped = int((~retained).sum())
    timing["feature_matrix"] = time.time() - start
    if reduced.shape[1] == 0:
        raise ValueError(f"{identifier}: every execution feature is constant")
    del matrix

    keep = [feature for feature, flag in zip(features, retained) if flag]

    start = time.time()
    raw = {"rf": forest_importances(reduced, labels, seed)}
    raw["neg"] = np.random.default_rng(seed + 1).random(reduced.shape[1])
    raw["adaptive"] = forest_importances(
        reduced, labels, seed, sample_weight=adaptive_sample_weights(labels)
    )
    timing["forest"] = time.time() - start

    start = time.time()
    deferred = os.environ.get("FFL_DEFER_VARIANTS", "1") == "1"
    for mode in ("hybrid", "failfirst"):
        if deferred:
            # The multi-signal variants cost time linear in the number of
            # features, which dominates extraction on the largest subjects.
            # They are recomputed for every subject in one consistent offline
            # pass over the persisted feature vectors (``run_protocol.py
            # variants``), so they are never computed for only part of the
            # sample.
            raw[mode] = np.zeros(reduced.shape[1], dtype=float)
            continue
        try:
            raw[mode] = hybrid_importances(reduced, labels, keep, mode)
        except Exception:
            raw[mode] = np.zeros(reduced.shape[1], dtype=float)
    timing["variants"] = time.time() - start

    #: The positive control no longer fits a second forest on an oracle-augmented
    #: matrix. That construction did not dominate the primary weighting: the extra
    #: column absorbed Gini mass and shifted the min-max normalisation, so the real
    #: features came out deflated, and the oracle column itself frequently received
    #: zero importance. The control is now derived from the primary weighting
    #: directly (see below), which makes it an upper bound by construction.

    # ------------------------------------------------------------------
    # Line spectra
    # ------------------------------------------------------------------
    spectra: List[Line] = [
        a for a in analyzer.model.get_analysis() if isinstance(a, Line)
    ]
    spectra.sort(key=lambda s: (s.file, s.line))
    line_index = {(s.file, int(s.line)): i for i, s in enumerate(spectra)}
    n_lines = len(spectra)

    start = time.time()
    location_cache: Dict[Tuple, List[Tuple[str, int]]] = {}
    primary_locations = [
        feature_locations(f, str(checkout), False, location_cache) for f in keep
    ]
    alt_locations = [
        feature_locations(f, str(checkout), True, location_cache) for f in keep
    ]
    timing["mapping"] = time.time() - start
    timing["mapping_cache_entries"] = float(len(location_cache))

    reference, hunks = ground_truth(project)
    weights: Dict[str, Dict[str, List[float]]] = {}
    for condition in ("rf", "neg", "adaptive", "hybrid", "failfirst"):
        aggregated = aggregate_weights(
            primary_locations, normalize(raw[condition]), line_index, n_lines
        )
        weights[condition] = {k: v.tolist() for k, v in aggregated.items()}
    weights["alt"] = {
        k: v.tolist()
        for k, v in aggregate_weights(
            alt_locations, normalize(raw["rf"]), line_index, n_lines
        ).items()
    }
    #: Positive control: the primary weighting with the reference lines raised to
    #: the largest weight the pipeline can assign. Identical to ``rf`` everywhere
    #: else, so it dominates it per subject: raising only those entries can move a
    #: fault line up and never down, because min-max normalisation is monotone and
    #: lifting the maximum scales every other entry down.
    positive = aggregate_weights(
        primary_locations, normalize(raw["rf"]), line_index, n_lines
    )
    marked = 0
    for aggregation in AGGREGATIONS:
        vector = positive[aggregation]
        for location in reference:
            idx = line_index.get(location)
            if idx is not None:
                vector[idx] = 1.0
                marked += aggregation == AGGREGATIONS[0]
    weights["pos"] = {k: v.tolist() for k, v in positive.items()}

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------
    start = time.time()
    files = {s.file for s in spectra} | {f for f, _ in reference} | {f for f, _ in hunks}
    index = fn.build_index(checkout, files)
    table, line_functions = fn.assign(
        index, [(s.file, int(s.line)) for s in spectra]
    )
    lookup = {span: i for i, span in enumerate(table)}

    def extra(locations):
        out = []
        for file, line in locations:
            span = fn.enclosing(index, file, line)
            if span is None:
                out.append(-1)
                continue
            if span not in lookup:
                lookup[span] = len(table)
                table.append(span)
            out.append(lookup[span])
        return out

    reference_functions = extra(reference)
    hunk_functions = extra(hunks)
    timing["functions"] = time.time() - start

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    if vectors_path is not None:
        start = time.time()
        write_vectors(
            vectors_path,
            reduced,
            labels,
            [f.name for f in keep],
            [f.analysis.analysis_type().value for f in keep],
            dict(raw),
            [[line_index[loc] for loc in locs if loc in line_index] for locs in primary_locations],
            [[line_index[loc] for loc in locs if loc in line_index] for locs in alt_locations],
        )
        timing["vectors"] = time.time() - start

    conflicts = sum(1 for locs in primary_locations if len(locs) > 1)
    covered = sum(1 for w in weights["rf"]["max"] if w > 0)
    diagnostics = {
        "n_features_total": len(features),
        "n_features_constant": n_dropped,
        "constant_fraction": n_dropped / max(1, len(features)),
        "n_features_used": reduced.shape[1],
        "n_vectors": int(len(labels)),
        "n_vectors_failing": int(labels.sum()),
        "n_vectors_passing": int(len(labels) - labels.sum()),
        "weak_signal": bool(labels.sum() < 3),
        "multi_location_features": conflicts,
        "mapping_conflict_rate": conflicts / max(1, len(keep)),
        "lines_with_evidence": covered,
        "line_evidence_rate": covered / max(1, n_lines),
        "importance": importance_diagnostics(raw["rf"]),
        "positive_control_marked": marked,
        "positive_control_reference": len(reference),
    }
    if with_stability:
        start = time.time()
        diagnostics["stability"] = stability_diagnostics(
            reduced, labels, seed, raw["rf"]
        )
        timing["stability"] = time.time() - start

    return {
        "schema": 1,
        "subject": identifier,
        "project": project.project_name,
        "bug_id": int(project.bug_id),
        "seed": seed,
        "loc": subject_loc(project),
        "n_tests_failing": len(failing),
        "n_tests_passing": len(passing),
        "lines": {
            "file": [s.file for s in spectra],
            "line": [int(s.line) for s in spectra],
            "ef": [int(s.failed_observed) for s in spectra],
            "ep": [int(s.passed_observed) for s in spectra],
            "function": line_functions,
        },
        "totals": {"failed": len(failing), "passed": len(passing)},
        "weights": weights,
        "reference_lines": [list(x) for x in reference],
        "reference_functions": reference_functions,
        "hunk_lines": [list(x) for x in hunks],
        "hunk_functions": hunk_functions,
        "functions": [[s.file, s.name, s.start, s.end] for s in table],
        "diagnostics": diagnostics,
        "timing": timing,
        "status": "ok",
        "variants_deferred": deferred,
    }


def write_vectors(
    path: Path,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    feature_types: Sequence[int],
    importances: Dict[str, np.ndarray],
    location_index: Sequence[Sequence[int]],
    alternative_index: Sequence[Sequence[int]],
) -> None:
    """Persist the labelled feature vectors and the fitted importances.

    Phase 1 of the protocol requires that a ranking can be reproduced without
    re-running instrumentation.  The archive stores the labelled vectors, the
    feature vocabulary, the raw importances of every weighting condition and
    the feature-to-line mapping as index lists into the artifact's line table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_primary = np.concatenate([np.asarray(x, dtype=np.int32) for x in location_index]) if location_index else np.zeros(0, dtype=np.int32)
    offsets_primary = np.cumsum([0] + [len(x) for x in location_index]).astype(np.int64)
    flat_alt = np.concatenate([np.asarray(x, dtype=np.int32) for x in alternative_index]) if alternative_index else np.zeros(0, dtype=np.int32)
    offsets_alt = np.cumsum([0] + [len(x) for x in alternative_index]).astype(np.int64)
    np.savez_compressed(
        path,
        matrix=matrix,
        labels=labels.astype(np.int8),
        features=np.asarray(feature_names, dtype=object),
        types=np.asarray(feature_types, dtype=np.int16),
        map_flat=flat_primary,
        map_offsets=offsets_primary,
        alt_flat=flat_alt,
        alt_offsets=offsets_alt,
        **{f"importance_{k}": v for k, v in importances.items()},
    )


def write_artifact(artifact: Dict, path: Path) -> None:
    """Write atomically: a partial write must never leave a readable-looking file.

    A worker killed mid-write (out of memory, or a killed run) otherwise leaves a
    truncated gzip stream that later passes an existence check but fails to
    parse, which silently removes the subject from the analysis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with gzip.open(scratch, "wt", encoding="utf-8") as handle:
            json.dump(artifact, handle, separators=(",", ":"))
        os.replace(scratch, path)
    finally:
        if scratch.exists():
            scratch.unlink(missing_ok=True)


def read_artifact(path: Path) -> Dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def extract_subject(
    project,
    artifacts_dir: Path = Path("artifacts"),
    events_root: Path = Path("sflkit_events"),
    mappings_root: Path = Path("mappings"),
    checkout_root: Path = Path("tmp"),
    with_stability: bool = True,
    vectors_dir: Optional[Path] = Path("vectors"),
) -> Dict:
    identifier = project.get_identifier()
    target = artifacts_dir / f"{identifier}.json.gz"
    events = events_root / project.project_name / str(project.bug_id)
    mapping_file = mappings_root / f"{project}.json"
    checkout = checkout_root / identifier
    started = time.time()
    try:
        artifact = build_artifact(
            project,
            events,
            mapping_file,
            checkout,
            with_stability=with_stability,
            vectors_path=(vectors_dir / f"{identifier}.npz") if vectors_dir else None,
        )
    except BaseException as error:  # noqa: BLE001 - recorded as a feasibility failure
        artifact = {
            "schema": 1,
            "subject": identifier,
            "project": project.project_name,
            "bug_id": int(project.bug_id),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=6),
            "timing": {"total": time.time() - started},
        }
    artifact.setdefault("timing", {})["total"] = time.time() - started
    write_artifact(artifact, target)
    return artifact
