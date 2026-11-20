"""Exploratory: cross-subject learning to rank over SFL scores and features.

Not registered and not part of the study's verdict. Every other weighting in
this project is learned *within* one subject, from its own passing and failing
runs; with a single failing test in most subjects, that leaves almost nothing
to learn from. Here a model instead learns, from the bugs of *other* projects,
which combination of per-line evidence marks a faulty line.

Per line, all evidence is expressed as within-subject rank percentiles (so that
subjects of very different size are comparable):

``sfl``      the five SFL formulas, and the line's ``ef``/``ep`` shares;
``vec``      the feature-vector weightings (spectrum, nearest neighbour, forest);
``trace``    the ordered-trace signals (slice, slice depth, failure proximity,
             automaton spectrum and nearest neighbour);
``neg``/``perm``  the outcome-neutral controls (random / permuted importances),
             each standing in for the feature weightings in a control model;
``context``  for every signal above, its maximum in the line's function and file,
             and the line's position within its function.

Training is strictly leave-one-project-out: the lines of a project are scored by
a model fitted on all other projects only. Hyperparameters are fixed a priori.
The label is membership in the reference-patch lines.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from protocol.formulas import FORMULAS, suspiciousness

VECTOR = ("x_spectrum", "x_nn", "rf")
TRACE = ("slice", "slice_depth", "recency", "slice_recency", "fsm", "fsm_nn")
#: Outcome-neutral controls in place of the feature weightings: uniform random
#: importances (``neg``) and permuted importances (``perm``) through the same map.
CONTROLS = ("neg", "perm")
GROUPS = {
    "sfl": ("sfl",),
    "vec": ("sfl", "vec"),
    "trace": ("sfl", "trace"),
    "full": ("sfl", "vec", "trace"),
    "neg": ("sfl", "neg"),
    "perm": ("sfl", "perm"),
}
SEED = 20260519


def _percentile(values: np.ndarray) -> np.ndarray:
    """Descending rank percentile in (0, 1]; ties share their average rank."""
    from scipy.stats import rankdata

    if len(values) == 0:
        return values
    return 1.0 - (rankdata(-np.asarray(values, dtype=float), method="average") - 1) / len(values)


def line_table(artifact: Dict) -> Optional[pd.DataFrame]:
    """One row per ranked line of a subject, with every evidence column."""
    from protocol.hierarchical import _unit_ids, _unit_max

    lines = artifact["lines"]
    n = len(lines["file"])
    if n == 0:
        return None
    ef = np.asarray(lines["ef"], dtype=float)
    ep = np.asarray(lines["ep"], dtype=float)
    tf = int(artifact["totals"]["failed"])
    tp = int(artifact["totals"]["passed"])
    file_id, function_id = _unit_ids(lines["file"], np.asarray(lines["function"], dtype=np.int64))
    columns: Dict[str, np.ndarray] = {}
    groups: Dict[str, str] = {}

    def add(name: str, values: np.ndarray, group: str, context: bool = True):
        p = _percentile(values)
        columns[name] = p
        groups[name] = group
        if context:
            columns[f"{name}@fn"] = _unit_max(p, function_id)
            columns[f"{name}@file"] = _unit_max(p, file_id)
            columns[f"{name}-fn"] = p - columns[f"{name}@fn"]
            groups[f"{name}@fn"] = groups[f"{name}@file"] = groups[f"{name}-fn"] = group

    for formula in FORMULAS:
        add(formula, suspiciousness(formula, ef, ep, tf, tp), "sfl")
    columns["ef_share"] = ef / max(tf, 1)
    columns["ep_share"] = ep / max(tp, 1)
    groups["ef_share"] = groups["ep_share"] = "sfl"
    weights = artifact["weights"]
    for name in VECTOR + TRACE + CONTROLS:
        if name in weights:
            values = np.asarray(weights[name]["max"], dtype=float)
        else:
            values = np.zeros(n)
        group = "vec" if name in VECTOR else "trace" if name in TRACE else name
        add(name, values, group)
    frame = pd.DataFrame(columns)
    reference = {(f, int(l)) for f, l in artifact["reference_lines"]}
    frame["label"] = [int((f, int(l)) in reference) for f, l in zip(lines["file"], lines["line"])]
    frame["subject"] = artifact["subject"]
    frame["project"] = artifact["project"]
    frame.attrs["groups"] = groups
    return frame


def feature_columns(groups: Dict[str, str], variant: str) -> List[str]:
    keep = set(GROUPS[variant])
    return [c for c, g in groups.items() if g in keep]


DEFAULT_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                      min_samples_leaf=50, l2_regularization=1.0, class_weight="balanced")


def _model(kind: str, params: Optional[Dict], seed: int):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=seed))
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=seed, **{**DEFAULT_PARAMS, **(params or {})})


def leave_one_project_out(table: pd.DataFrame, columns: Sequence[str], params: Optional[Dict] = None,
                          kind: str = "hgb", seed: int = SEED,
                          permute: Optional[Dict[str, Sequence[str]]] = None):
    """Out-of-project predictions for every row of ``table``.

    With ``permute`` (group name -> columns), also returns, per group, the
    predictions after shuffling those columns within each held-out subject
    (permutation importance on unseen projects).
    """
    predictions = np.zeros(len(table))
    permuted = {g: np.zeros(len(table)) for g in (permute or {})}
    projects = table["project"].to_numpy()
    subjects = table["subject"].to_numpy()
    columns = list(columns)
    X = table[columns].to_numpy(dtype=np.float32)
    y = table["label"].to_numpy()
    rng = np.random.default_rng(seed)
    for project in np.unique(projects):
        test = projects == project
        model = _model(kind, params, seed)
        model.fit(X[~test], y[~test])
        predictions[test] = model.predict_proba(X[test])[:, 1]
        for group, group_columns in (permute or {}).items():
            Xt = X[test].copy()
            idx = [columns.index(c) for c in group_columns if c in columns]
            test_subjects = subjects[test]
            for subject in np.unique(test_subjects):
                rows = np.flatnonzero(test_subjects == subject)
                order = rng.permutation(len(rows))
                Xt[np.ix_(rows, idx)] = Xt[np.ix_(rows[order], idx)]
            permuted[group][test] = model.predict_proba(Xt)[:, 1]
    return (predictions, permuted) if permute else predictions


def quick_metrics(table: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    """Fast proxy outcomes from the best-ranked fault line of each subject
    (midpoint rank among equal scores); a subject whose fault lines are not
    ranked counts as a miss with EXAM 1."""
    frame = pd.DataFrame({"subject": table["subject"].to_numpy(), "label": table["label"].to_numpy(),
                          "score": scores})
    top1 = top5 = top10 = exam = 0.0
    groups = frame.groupby("subject", sort=False)
    for _, rows in groups:
        s, l = rows["score"].to_numpy(), rows["label"].to_numpy()
        if l.sum() == 0:
            exam += 1.0
            continue
        best = s[l == 1].max()
        rank = (s > best).sum() + ((s == best).sum() + 1) / 2.0
        top1 += rank <= 1
        top5 += rank <= 5
        top10 += rank <= 10
        exam += rank / len(s)
    n = groups.ngroups
    return {"top1": top1 / n, "top5": top5 / n, "top10": top10 / n, "exam": exam / n}


def signal_groups(columns: Sequence[str]) -> Dict[str, List[str]]:
    """Input columns grouped by the signal they derive from (with context)."""
    groups: Dict[str, List[str]] = {}
    for column in columns:
        base = column.split("@")[0].split("-fn")[0]
        if base in ("ef_share", "ep_share"):
            base = "ef/ep"
        groups.setdefault(base, []).append(column)
    return groups


#: Robustness grid: model seeds, alternative hyperparameters, and a linear model.
ROBUSTNESS = {"base": ("hgb", {}, None)}
ROBUSTNESS.update({f"seed{seed}": ("hgb", {}, seed) for seed in (1, 2, 3, 4, 5)})
ROBUSTNESS.update({
    "fast": ("hgb", {"max_iter": 150, "learning_rate": 0.1}, None),
    "slow": ("hgb", {"max_iter": 600, "learning_rate": 0.025}, None),
    "leaves15": ("hgb", {"max_leaf_nodes": 15}, None),
    "leaves63": ("hgb", {"max_leaf_nodes": 63}, None),
    "minleaf20": ("hgb", {"min_samples_leaf": 20}, None),
    "minleaf200": ("hgb", {"min_samples_leaf": 200}, None),
    "noweight": ("hgb", {"class_weight": None}, None),
    "logreg": ("logreg", {}, None),
})
ROBUSTNESS_VARIANTS = ("sfl", "neg", "vec", "trace", "full")

VARIANTS = tuple(GROUPS)
METHODS = tuple(f"lex_ltr_{variant}" for variant in VARIANTS)
