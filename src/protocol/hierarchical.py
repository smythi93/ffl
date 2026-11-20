"""Exploratory: hierarchical feature-based localization (file -> function -> line).

Not registered and not part of the study's verdict. Instead of modulating a
line ranking, the ranking is built top-down: files are ordered first, then the
functions inside each file, then the lines inside each function. A line can
therefore only be inspected once every line of every higher-ranked function
and file has been, which is how a developer who trusts the hierarchy would
read it.

Each level is ordered by a unit score derived from per-line evidence:

``feat``   the feature weight (maximum over the unit's lines); the SFL score
           of the unit only breaks ties.
``blend``  the mean of the normalized SFL score and the feature weight, so
           both signals count equally at every level.
``sfl``    the SFL score alone: the same hierarchy without features, which
           separates the effect of ranking hierarchically from the effect of
           the features.
``gran``   granularity-matched descent: files and functions by function-level
           features (function entered, exception exit), then blocks by branch
           and loop features, then lines by line-level features (line,
           def-use, condition, values); SFL breaks the remaining ties.

Method names are ``h_sfl`` and ``h_<mode>_<source>``, where ``<source>`` is any
line weighting stored in the artifact (``x_spectrum``, ``x_nn``, ``rf``, and the
controls ``neg`` and ``perm``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from protocol.formulas import min_max

MODES = ("feat", "blend", "sfl", "gran")

#: Feature types (SFLKit ``AnalysisType`` ids) that describe each level.
GRANULARITY = {
    "func": {2, 16},                                 # function entered, exception exit
    "block": {1, 3},                                 # branch, loop
    "line": {0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},  # line, def-use, condition, values
}
GRANULARITY_SOURCES = ("x_spectrum", "x_nn", "neg")


def granularity_weights(artifact_weights, source: str, aggregation: str):
    """The (function, block, line) weight vectors of ``source``, or None."""
    keys = [f"g_{level}_{source}" for level in ("func", "block", "line")]
    if not all(k in artifact_weights for k in keys):
        return None
    return tuple(np.asarray(artifact_weights[k][aggregation], dtype=float) for k in keys)


def parse_method(method: str) -> Tuple[str, Optional[str]]:
    """``h_feat_x_spectrum`` -> ("feat", "x_spectrum"); ``h_sfl`` -> ("sfl", None)."""
    if not method.startswith("h_"):
        raise ValueError(method)
    rest = method[2:]
    mode, _, source = rest.partition("_")
    if mode not in MODES:
        raise ValueError(f"unknown hierarchy mode in {method}")
    return mode, (source or None)


def _unit_ids(files, functions) -> Tuple[np.ndarray, np.ndarray]:
    """Dense file ids and function ids; module-level lines (function -1) form
    one pseudo-function per file."""
    _, file_id = np.unique(np.asarray(files, dtype=object), return_inverse=True)
    functions = np.asarray(functions, dtype=np.int64)
    keys = np.where(functions >= 0, functions, -(file_id + 1))
    _, function_id = np.unique(keys, return_inverse=True)
    return file_id, function_id


def _unit_max(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    best = np.full(ids.max() + 1, -np.inf)
    np.maximum.at(best, ids, values)
    return best[ids]


def hierarchical_scores(
    base: np.ndarray,
    weights: Optional[np.ndarray],
    files,
    functions,
    mode: str,
    extra: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scores whose descending order is the hierarchical ranking.

    Lines with identical keys at every level share a score, so they form a tie
    group and the usual randomized tie breaking applies to them. ``extra`` is an
    optional last key that breaks the remaining ties (method suffix
    ``__<weighting>``, e.g. ``h_feat_x_nn__recency``).
    """
    if len(base) == 0:
        return np.zeros(0, dtype=float)
    sfl = min_max(np.asarray(base, dtype=float))
    file_id, function_id = _unit_ids(files, functions)
    if mode == "gran":
        #: file -> function by function-level features, then block-level
        #: features, then line-level features, then SFL.
        func_w, block_w, line_w = weights
        keys = [
            _unit_max(func_w, file_id), _unit_max(func_w, function_id),
            block_w, line_w, sfl,
        ]
        if extra is not None:
            keys.append(np.asarray(extra, dtype=float))
        return _dense_scores(np.column_stack(keys))
    if mode == "sfl":
        primary, secondary = sfl, np.zeros_like(sfl)
    else:
        w = np.asarray(weights, dtype=float)
        w = min_max(w) if w.size and w.max() > 1.0 else w
        if mode == "feat":
            primary, secondary = w, sfl
        else:  # blend
            primary, secondary = 0.5 * sfl + 0.5 * w, sfl

    #: Keys from the most to the least significant level. Unit ids are
    #: deliberately not part of the key: ordering tied units by id would encode
    #: source order. Units that tie on their scores at a level are instead
    #: treated as one merged unit, whose lines are ordered by the next level.
    keys = [
        _unit_max(primary, file_id), _unit_max(secondary, file_id),
        _unit_max(primary, function_id), _unit_max(secondary, function_id),
        primary, secondary,
    ]
    if extra is not None:
        keys.append(np.asarray(extra, dtype=float))
    return _dense_scores(np.column_stack(keys))


def _dense_scores(matrix: np.ndarray) -> np.ndarray:
    """Scores whose descending order is the lexicographic order of the rows
    (first column most significant); equal rows get equal scores."""
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=float)
    order = np.lexsort([-matrix[:, i] for i in reversed(range(matrix.shape[1]))])
    ranked = matrix[order]
    #: Dense rank over full keys: equal keys -> equal score.
    changed = np.any(np.diff(ranked, axis=0) != 0, axis=1)
    dense = np.concatenate([[0], np.cumsum(changed)])
    scores = np.empty(len(order), dtype=float)
    scores[order] = -dense.astype(float)
    return scores
