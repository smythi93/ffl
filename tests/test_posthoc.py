"""Post-hoc scorers and the tie-break-only combination rule."""

import numpy as np

from protocol.metrics import tiebreak_scores
from protocol.posthoc import spectrum_scores


def test_tiebreak_keeps_the_order_of_distinct_baseline_scores():
    rng = np.random.default_rng(0)
    base = rng.choice([-2.0, 0.1, 0.1000001, 0.5, 0.5, 0.5, 3.0], size=200)
    weights = rng.random(200)
    scores = tiebreak_scores(base, weights)
    for i in range(200):
        for j in range(200):
            if base[i] > base[j]:
                assert scores[i] > scores[j]


def test_tiebreak_orders_a_tie_group_by_weight():
    base = np.array([1.0, 0.5, 0.5, 0.5, 0.0])
    weights = np.array([0.0, 0.2, 0.9, 0.5, 1.0])
    order = np.argsort(-tiebreak_scores(base, weights), kind="stable")
    assert order.tolist() == [0, 2, 3, 1, 4]


def test_spectrum_prefers_features_specific_to_the_failing_run():
    #           f0: only failing TRUE   f1: everywhere TRUE   f2: undefined in failing
    matrix = np.array([
        [1, 1, -1],   # failing
        [0, 1, 1],
        [0, 1, 1],
    ], dtype=np.int8)
    labels = np.array([1, 0, 0])
    scores = spectrum_scores(matrix, labels)
    assert scores[0] == 1.0
    assert scores[0] > scores[1] > scores[2] == 0.0


def test_nearest_neighbour_credits_only_differences_to_the_closest_passing_run():
    from protocol.posthoc import nearest_neighbour_scores

    matrix = np.array([
        [1, 1, 0, 1],   # failing
        [1, 0, 0, 1],   # passing, differs from the failing run in feature 1 only
        [0, 0, 1, 0],   # passing, far away
    ], dtype=np.int8)
    labels = np.array([1, 0, 0])
    scores = nearest_neighbour_scores(matrix, labels)
    assert scores.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_nearest_neighbour_averages_equidistant_neighbours():
    from protocol.posthoc import nearest_neighbour_scores

    matrix = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.int8)
    labels = np.array([1, 0, 0])
    assert nearest_neighbour_scores(matrix, labels).tolist() == [0.5, 0.5]
