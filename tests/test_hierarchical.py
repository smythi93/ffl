"""Hierarchical (file -> function -> line) ranking."""

import numpy as np

from protocol.hierarchical import hierarchical_scores, parse_method
from protocol.metrics import file_top1


FILES = ["a.py", "a.py", "a.py", "b.py", "b.py"]
FUNCS = [0, 0, 1, 2, 2]


def _order(scores):
    return np.argsort(-scores, kind="stable").tolist()


def test_parse_method():
    assert parse_method("h_feat_x_spectrum") == ("feat", "x_spectrum")
    assert parse_method("h_sfl") == ("sfl", None)


def test_feature_hierarchy_ranks_whole_units_before_lines():
    base = np.array([0.9, 0.8, 0.1, 0.5, 0.4])
    # b.py carries the strongest feature, so all of b.py comes first even though
    # a.py holds the line with the highest SFL score.
    weights = np.array([0.1, 0.0, 0.2, 0.0, 1.0])
    order = _order(hierarchical_scores(base, weights, FILES, FUNCS, "feat"))
    assert order[:2] == [4, 3]
    # within a.py, function 1 (weight 0.2) precedes function 0 (max weight 0.1)
    assert order[2] == 2


def test_sfl_hierarchy_without_features():
    base = np.array([0.9, 0.8, 0.1, 0.5, 0.4])
    order = _order(hierarchical_scores(base, None, FILES, FUNCS, "sfl"))
    assert order == [0, 1, 2, 3, 4]


def test_tied_units_form_one_tie_group_not_source_order():
    base = np.array([0.5, 0.5])
    scores = hierarchical_scores(base, None, ["a.py", "b.py"], [0, 1], "sfl")
    assert scores[0] == scores[1]


def test_file_top1_counts_ties():
    scores = np.array([1.0, 0.2, 1.0])
    assert file_top1(scores, ["a", "a", "b"], {"b"}) == 0.5
    assert file_top1(scores, ["a", "a", "b"], {"c"}) == 0.0


def test_granularity_descent_uses_level_specific_evidence():
    files = ["a.py"] * 4 + ["b.py"] * 2
    funcs = [0, 0, 1, 1, 2, 2]
    base = np.array([0.9, 0.9, 0.1, 0.1, 0.5, 0.5])
    func_w = np.array([0.0, 0.0, 1.0, 1.0, 0.3, 0.3])   # function 1 entered only when failing
    block_w = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    line_w = np.array([1.0, 0.0, 0.0, 0.7, 0.0, 0.0])    # strong line evidence in function 0
    scores = hierarchical_scores(base, (func_w, block_w, line_w), files, funcs, "gran")
    order = _order(scores)
    # a.py (holds function 1) first, function 1 before function 0, line 3 before line 2
    assert order[:2] == [3, 2]
    assert order[2:4] == [0, 1]
    assert set(order[4:]) == {4, 5}


def test_empty_subject():
    assert hierarchical_scores(np.zeros(0), None, [], [], "sfl").size == 0


def test_extra_key_breaks_only_the_remaining_ties():
    import numpy as np
    from protocol.hierarchical import hierarchical_scores

    base = np.array([1.0, 1.0, 0.0])
    files = ["a.py", "a.py", "a.py"]
    functions = [0, 0, 0]
    plain = hierarchical_scores(base, None, files, functions, "sfl")
    assert plain[0] == plain[1]
    broken = hierarchical_scores(base, None, files, functions, "sfl", extra=np.array([0.0, 1.0, 5.0]))
    #: Line 1 now precedes line 0, and line 2 stays last despite its large extra key.
    assert broken[1] > broken[0] > broken[2]
