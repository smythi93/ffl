"""Post-hoc (unregistered) alternatives to the random-forest importances.

The registered method scores execution features by random-forest impurity
importance. In this study 72% of subjects have exactly one failing test, so a
supervised learner sees a single positive example and many features separate
it perfectly; which of them the forest credits is close to arbitrary. The
scorers below replace only that step. Each produces one score per feature from
the stored (reduced) feature matrix and labels, and is then mapped to lines
through the registered feature-to-line map, aggregated by the registered
maximum, and blended with the registered lambda and epsilon, so any difference
to ``rf`` is attributable to the feature scorer alone.

None of this feeds a confirmatory result.

Scorers
-------
``x_spectrum``  SFL applied to features: Ochiai of the feature taking a value,
                for TRUE and FALSE separately, keeping the more
                failure-specific of the two. UNDEFINED (not evaluated) counts
                as neither. No learning, so no single-sample instability.
``x_mi``        Mutual information between the feature and the test outcome.
``x_l1``        |coefficient| of an L1-regularized, class-balanced logistic
                regression (a sparse linear learner instead of trees).
``x_et``        Extra-Trees impurity importance, same settings as the forest.
``x_nn``        Nearest-neighbour contrast: features in which a failing run
                differs from its most similar passing run(s).

Combination rule
----------------
``tie_<scorer>`` uses the weights only to break ties: the baseline order of
distinct scores is kept exactly, and the weight decides only within a tie
group (see :func:`protocol.metrics.tiebreak_scores`).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

TRUE, FALSE = 1, 0


def spectrum_scores(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    failing = labels == 1
    n_failing = max(int(failing.sum()), 1)
    best = np.zeros(matrix.shape[1], dtype=float)
    for value in (TRUE, FALSE):
        hit = matrix == value
        ef = hit[failing].sum(axis=0).astype(float)
        ep = hit[~failing].sum(axis=0).astype(float)
        denominator = np.sqrt(n_failing * (ef + ep))
        ochiai = np.divide(ef, denominator, out=np.zeros_like(ef), where=denominator > 0)
        best = np.maximum(best, ochiai)
    return best


def nearest_neighbour_scores(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nearest-neighbour contrast (after Renieris and Reiss).

    Instead of crediting every feature that separates a failing run from *all*
    passing runs, credit only the features in which a failing run differs from
    its most similar passing run(s). Similarity is the number of features with
    equal values (UNDEFINED included, so "executed here but not there" counts as
    a difference). Equidistant nearest passing runs are averaged, and the score
    of a feature is the mean over failing runs of the share of their nearest
    passing runs it differs from.
    """
    failing = np.flatnonzero(labels == 1)
    passing = np.flatnonzero(labels == 0)
    scores = np.zeros(matrix.shape[1], dtype=float)
    if len(failing) == 0 or len(passing) == 0 or matrix.shape[1] == 0:
        return scores
    reference = matrix[passing]
    for row in failing:
        differs = reference != matrix[row]
        distance = differs.sum(axis=1)
        nearest = differs[distance == distance.min()]
        scores += nearest.mean(axis=0)
    return scores / len(failing)


def mutual_information_scores(matrix: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_classif

    if matrix.shape[1] == 0:
        return np.zeros(0)
    return np.asarray(
        mutual_info_classif(matrix, labels, discrete_features=True, random_state=seed),
        dtype=float,
    )


def l1_logistic_scores(matrix: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    if matrix.shape[1] == 0 or len(np.unique(labels)) < 2:
        return np.zeros(matrix.shape[1])
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0, class_weight="balanced",
        random_state=seed, max_iter=1000,
    )
    model.fit(matrix.astype(float), labels)
    return np.abs(model.coef_[0]).astype(float)


def extra_trees_scores(matrix: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesClassifier

    from protocol.extract import CLASS_WEIGHT, CRITERION, N_ESTIMATORS, N_JOBS

    if matrix.shape[1] == 0:
        return np.zeros(0)
    model = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS, criterion=CRITERION, class_weight=CLASS_WEIGHT,
        random_state=seed, n_jobs=N_JOBS,
    )
    model.fit(matrix, labels)
    return np.asarray(model.feature_importances_, dtype=float)


SCORERS: Dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "x_spectrum": lambda m, y, s: spectrum_scores(m, y),
    "x_mi": mutual_information_scores,
    "x_l1": l1_logistic_scores,
    "x_et": extra_trees_scores,
    "x_nn": lambda m, y, s: nearest_neighbour_scores(m, y),
}

#: Tie-break-only variants and the weights they reuse.
TIEBREAK = {"tie_rf": "rf", "tie_spectrum": "x_spectrum", "tie_nn": "x_nn"}

POSTHOC_METHODS = tuple(SCORERS) + tuple(TIEBREAK)
