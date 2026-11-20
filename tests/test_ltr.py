import numpy as np
import pandas as pd

from protocol.ltr import _percentile, feature_columns, leave_one_project_out


def test_percentile_is_descending_and_shares_ties():
    p = _percentile(np.array([3.0, 1.0, 1.0, 5.0]))
    assert p[3] == 1.0 and p[0] > p[1] and p[1] == p[2]


def test_feature_groups_nest():
    groups = {"Ochiai": "sfl", "x_nn": "vec", "slice": "trace"}
    assert feature_columns(groups, "sfl") == ["Ochiai"]
    assert set(feature_columns(groups, "full")) == {"Ochiai", "x_nn", "slice"}


def test_leave_one_project_out_never_sees_the_held_out_project():
    rng = np.random.default_rng(0)
    rows = []
    for project in ("a", "b", "c"):
        x = rng.random(400)
        #: In project "c" the relation is inverted; a model that had seen "c"
        #: would not rank its positives this low.
        label = (x > 0.9) if project != "c" else (x < 0.1)
        rows.append(pd.DataFrame({"x": x, "label": label.astype(int), "project": project,
                                  "subject": project + "_1"}))
    table = pd.concat(rows, ignore_index=True)
    pred = leave_one_project_out(table, ["x"])
    c = table.project == "c"
    assert pred[c & (table.label == 1)].mean() < pred[c & (table.label == 0)].mean()


def test_quick_metrics_uses_the_best_fault_and_midpoint_ties():
    from protocol.ltr import quick_metrics

    table = pd.DataFrame({"subject": ["s"] * 4, "label": [0, 1, 0, 1]})
    m = quick_metrics(table, np.array([0.9, 0.5, 0.5, 0.1]))
    #: Best fault at score 0.5, tied with one other line: midpoint rank 2.5.
    assert m["top1"] == 0 and m["top5"] == 1 and m["exam"] == 2.5 / 4
