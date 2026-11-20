"""Render the manuscript's tables and figures from the study outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from protocol.analysis import FAMILIES, PRIMARY_METRICS
from protocol.formulas import FORMULAS

#: A left-aligned fixed-width column. Justification in a column this narrow
#: produces stretched word spacing and mid-word hyphenation.
RAGGED = ">{\\raggedright\\arraybackslash}p{%s}"

METRIC_LABELS = {
    "patch_top1": "top-1",
    "patch_top5": "top-5",
    "patch_top10": "top-10",
    "patch_exam_avg": "EXAM",
    "patch_wasted_avg": "WE",
    "patch_wasted_avg_norm": "WE (norm.)",
    "patch_exam_best": "EXAM (best)",
    "patch_exam_worst": "EXAM (worst)",
    "patch_wasted_best": "WE (best)",
    "patch_wasted_worst": "WE (worst)",
    "function_exam": "function EXAM",
    "function_effort_lines": "function effort",
    "function_faulty_mass": "faulty mass",
    "function_mass@1": "mass@1",
    "function_mass@3": "mass@3",
    "function_mass@5": "mass@5",
    "impact_found@1": "impact@1",
    "impact_found@5": "impact@5",
    "impact_found@10": "impact@10",
    "impact_min_rank": "min.\\ impact rank",
    "impact_total_ranked": "impact locations",
    "hunk_found@1": "hunk@1",
    "hunk_found@5": "hunk@5",
    "hunk_found@10": "hunk@10",
    "hunk_min_rank": "min.\\ hunk rank",
    "hunk_total_ranked": "hunk locations",
}

LTR_LABELS = {
    "lex_ltr_sfl": "SFL scores only", "lex_ltr_neg": "+ random importances",
    "lex_ltr_perm": "+ permuted importances", "lex_ltr_vec": "+ feature vectors",
    "lex_ltr_trace": "+ trace signals", "lex_ltr_full": "+ vectors and traces",
    "lex_ltr_full_ens": "+ vectors and traces, seed ensemble",
}
LTR_TABLE = ("lex_ltr_sfl", "lex_ltr_neg", "lex_ltr_perm", "lex_ltr_vec", "lex_ltr_trace",
             "lex_ltr_full", "lex_ltr_full_ens")

METHOD_LABELS = {
    "rf": "RF weighting",
    "alt": "alternative mapping",
    "neg": "negative control (random)",
    "perm": "permutation control",
    "pos": "positive control",
    "hybrid": "hybrid",
    "failfirst": "fail-first",
    "adaptive": "adaptive",
    "fam_coverage": "coverage features only",
    "fam_data_flow": "data-flow features only",
    "fam_value": "value features only",
    "fam_condition": "condition features only",
    "fam_exception": "exception features only",
    "x_spectrum": "feature spectrum (Ochiai)",
    "x_mi": "mutual information",
    "x_l1": "L1 logistic regression",
    "x_et": "Extra-Trees",
    "x_nn": "nearest-neighbour contrast",
    "tie_rf": "RF, tie-break only",
    "tie_spectrum": "feature spectrum, tie-break only",
    "tie_nn": "nearest-neighbour contrast, tie-break only",
}


#: Tables with more columns than the text width comfortably holds. They are
#: emitted inside a size group rather than a \\resizebox: sn-jnl's float
#: handling is incompatible with \\resizebox, and scaled type would also carry
#: stroke weights that no longer match the body text.
WIDE_TABLES = {
    "tab_primary.tex", "tab_superiority.tex", "tab_superiority_paired.tex",
    "tab_subjects.tex",
    "tab_correlations.tex", "tab_sensitivity.tex", "tab_effort.tex", "tab_function.tex", "tab_impact.tex",
    "tab_hunk.tex", "tab_sensitivity_threshold.tex", "tab_ablations.tex",
}

#: Tables that only need a modest reduction.
NARROW_TABLES = {
    "tab_diagnostics.tex", "tab_feasibility.tex", "tab_moderators.tex",
    "tab_example.tex", "tab_per_subject.tex", "tab_superiority_methods.tex",
    "tab_posthoc.tex", "tab_hierarchy.tex", "tab_ltr.tex",
}


def _column_tokens(spec: str) -> List[str]:
    """Split a tabular column spec into one token per column (with its modifiers)."""
    tokens, i, pending = [], 0, ""
    while i < len(spec):
        ch = spec[i]
        if ch in "><@!":
            depth, j = 0, i + 1
            while j < len(spec):
                depth += spec[j] == "{"
                depth -= spec[j] == "}"
                j += 1
                if depth == 0:
                    break
            pending += spec[i:j]
            i = j
            continue
        if ch in "pmbX":
            depth, j = 0, i + 1
            while j < len(spec):
                depth += spec[j] == "{"
                depth -= spec[j] == "}"
                j += 1
                if depth == 0:
                    break
            tokens.append(pending + spec[i:j])
            pending, i = "", j
            continue
        if ch.strip():
            tokens.append(pending + ch)
            pending = ""
        i += 1
    return tokens


def _split_cells(row: str) -> List[str]:
    """Cells of a tabular row (``&``-separated, ignoring escaped ``\\&``)."""
    cells, current, i = [], "", 0
    while i < len(row):
        if row[i] == "\\" and i + 1 < len(row):
            current += row[i:i + 2]
            i += 2
            continue
        if row[i] == "&":
            cells.append(current)
            current = ""
        else:
            current += row[i]
        i += 1
    cells.append(current)
    return cells


def house_style(body: str, wide: bool = False) -> str:
    """Restyle a generated ``tabular`` to the manuscript's table style.

    The table spans the line width (``tabularx``): leading label columns keep
    their natural width and the value columns share the rest. There are no
    rules inside the data. Rows alternate in background; where rows form groups
    (a label in the first column followed by rows with that cell empty), whole
    groups alternate and the label is set once, vertically centred over its
    group with a negative ``\\multirow`` in the group's last row, so the row
    colours cannot paint over it.
    """
    lines = body.strip().split("\n")
    head = lines[0]
    spec = head[len("\\begin{tabular}{"):-1]
    tokens = _column_tokens(spec)
    labels = 0
    while labels < len(tokens) and tokens[labels][-1] not in "rc":
        labels += 1
    labels = max(1, min(labels, len(tokens) - 1))
    if wide:
        #: Many columns: values keep their natural width and the last label
        #: column takes up the slack (and may wrap), so nothing can overflow.
        new_tokens = list(tokens)
        if tokens[labels - 1] == "l":
            new_tokens[labels - 1] = "L"
    else:
        #: Numeric (right-aligned) and centred columns stretch; text columns
        #: keep their natural width, since their cells cannot always break.
        stretch = {"r": "R", "c": "C"}
        new_tokens = tokens[:labels] + [stretch.get(t, t) if len(t) == 1 else t
                                        for t in tokens[labels:]]
    #: Header: everything up to the first \midrule; data: up to \bottomrule.
    first_mid = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")
    bottom = next(i for i, l in enumerate(lines) if l.strip() == "\\bottomrule")
    header, data, footer = lines[1:first_mid + 1], lines[first_mid + 1:bottom], lines[bottom:]
    rows = [l for l in data if l.strip() not in ("\\midrule",) and not l.strip().startswith("\\cmidrule")]
    cells = [_split_cells(r.rstrip().removesuffix("\\\\").rstrip()) for r in rows]
    grouped = any(c[0].strip() == "" for c in cells[1:]) and cells and cells[0][0].strip() != ""
    out = []
    if grouped:
        groups, current = [], []
        for c in cells:
            if c[0].strip() and current:
                groups.append(current)
                current = []
            current.append(c)
        groups.append(current)
        for g, group in enumerate(groups):
            label = group[0][0].strip()
            group[0][0] = ""
            if len(group) == 1:
                group[0][0] = label
            else:
                group[-1][0] = f"\\multirow{{-{len(group)}}}{{*}}{{{label}}}"
            for c in group:
                colour = "" if g % 2 else "\\rowcolor{rowstripe}"
                out.append(colour + " & ".join(x.strip() for x in c) + " \\\\")
    else:
        for r, c in enumerate(cells):
            colour = "" if r % 2 else "\\rowcolor{rowstripe}"
            out.append(colour + " & ".join(x.strip() for x in c) + " \\\\")
    footer = [l.replace("\\end{tabular}", "\\end{tabularx}") for l in footer]
    return "\n".join(["\\begin{tabularx}{\\linewidth}{" + "".join(new_tokens) + "}"]
                     + header + out + footer)


def write_table(out_dir: Path, name: str, body: str) -> None:
    if not body.lstrip().startswith("\\begin{tabular}"):
        (out_dir / name).write_text(body)
        return
    body = house_style(body, wide=name in WIDE_TABLES)
    if name in WIDE_TABLES:
        prologue = "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3.2pt}\n"
    elif name in NARROW_TABLES:
        prologue = "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
    else:
        prologue = "\\begingroup\n"
    (out_dir / name).write_text(prologue + body + "\n\\endgroup\n")


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _stars(q: float) -> str:
    if q < 0.001:
        return "$^{***}$"
    if q < 0.01:
        return "$^{**}$"
    if q < 0.05:
        return "$^{*}$"
    return ""


def table_primary(effects: pd.DataFrame, method: str = "rf") -> str:
    subset = effects[(effects.method == method)]
    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Formula & Outcome & Baseline & Augmented & $\\Delta$ & $d$ & $d_z$ & 95\\% CI & $q$ \\\\",
        "\\midrule",
    ]
    for formula in FORMULAS:
        first = True
        for outcome, _ in PRIMARY_METRICS:
            row = subset[(subset.formula == formula) & (subset.outcome == outcome)]
            if row.empty:
                continue
            row = row.iloc[0]
            label = formula if first else ""
            first = False
            digits = 0 if "wasted" in outcome else 3
            lines.append(
                f"{label} & {METRIC_LABELS[outcome]} & "
                f"{_fmt(row.baseline_mean, digits)} & {_fmt(row.augmented_mean, digits)} & "
                f"{_fmt(row.coefficient, digits)} & {_fmt(row.d, 2)} & "
                f"{_fmt(row.d_paired, 2)} & "
                f"[{_fmt(row.ci_low, digits)}, {_fmt(row.ci_high, digits)}] & "
                f"{_fmt(row.qvalue, 3)}{_stars(row.qvalue)} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def table_superiority(analysis: Dict, method: str = "rf", key: str = "superiority") -> str:
    """Per-formula superiority; ``key="superiority_paired"`` reads it off d_z."""
    verdict = analysis["methods"][method][key]
    lines = [
        "\\begin{tabular}{lccccccl}",
        "\\toprule",
        "Formula & top-1 & top-5 & top-10 & EXAM & wasted & improved & superior \\\\",
        "\\midrule",
    ]
    for formula in FORMULAS:
        entry = verdict["per_formula"][formula]
        cells = []
        for outcome, _ in PRIMARY_METRICS:
            detail = entry["detail"].get(outcome, {})
            d = detail.get("d", float("nan"))
            mark = "$\\uparrow$" if detail.get("improves") else ("$\\downarrow$" if detail.get("degrades") else "$\\approx$")
            cells.append(f"{mark}\\,{_fmt(abs(d), 2)}")
        lines.append(
            f"{formula} & " + " & ".join(cells) +
            f" & {entry['improves']}/5 & " + ("yes" if entry["superior"] else "no") + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_superiority_methods(analysis: Dict, methods: Sequence[str]) -> str:
    """Formulas for which each method is superior, under the registered d and
    under the paired d_z (exploratory)."""
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & superior ($d$) & superior ($d_z$) \\\\",
        "\\midrule",
    ]
    for method in methods:
        result = analysis["methods"].get(method)
        if not result or "superiority_paired" not in result:
            continue
        registered = result["superiority"]["n_superior"]
        paired = result["superiority_paired"]["n_superior"]
        label = METHOD_LABELS.get(method, method)
        lines.append(f"{label} & {registered}/{len(FORMULAS)} & {paired}/{len(FORMULAS)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_posthoc(analysis: Dict, effects: pd.DataFrame, methods: Sequence[str],
                  labels: Optional[Dict[str, str]] = None) -> str:
    """Post-hoc feature scorers: mean primary outcomes over the five formulas,
    and the number of formulas for which the four-of-five rule holds under the
    registered d and under the paired d_z."""
    outcomes = [o for o, _ in PRIMARY_METRICS]
    header = " & ".join(METRIC_LABELS[o] for o in outcomes)
    lines = [
        "\\begin{tabular}{l" + "r" * len(outcomes) + "cc}",
        "\\toprule",
        f"Method & {header} & sup.\\ ($d$) & sup.\\ ($d_z$) \\\\",
        "\\midrule",
    ]

    def cells(column: str, method: str) -> str:
        values = []
        for outcome in outcomes:
            rows = effects[(effects.method == method) & (effects.outcome == outcome)]
            mean = rows[column].mean()
            values.append(f"{mean:,.0f}" if "wasted" in outcome else _fmt(mean, 3))
        return " & ".join(values)

    reference = methods[0]
    lines.append(f"baseline & {cells('baseline_mean', reference)} & & \\\\")
    for method in methods:
        result = analysis["methods"].get(method)
        if result is None:
            continue
        lines.append(
            f"{(labels or METHOD_LABELS).get(method, method)} & {cells('augmented_mean', method)} & "
            f"{result['superiority']['n_superior']}/{len(FORMULAS)} & "
            f"{result['superiority_paired']['n_superior']}/{len(FORMULAS)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


HIERARCHY_LABELS = {
    "h_sfl": "SFL only (no features)",
    "x_spectrum": "feature spectrum", "x_nn": "nearest-neighbour contrast",
    "rf": "random forest", "neg": "random importances", "perm": "permuted importances",
}


def table_hierarchy(analysis: Dict, outcomes: pd.DataFrame, methods: Sequence[str]) -> str:
    """Exploratory hierarchical ranking: means over subjects and formulas, file@1,
    and the formulas for which the four-of-five rule holds under d and d_z."""
    columns = [("patch_top1", "top-1", 3), ("patch_top10", "top-10", 3),
               ("patch_exam_avg", "EXAM", 4), ("function_exam", "function EXAM", 3),
               ("file_top1", "file@1", 3)]
    lines = [
        "\\begin{tabular}{ll" + "r" * len(columns) + "cc}",
        "\\toprule",
        "Order & Evidence & " + " & ".join(c[1] for c in columns)
        + " & sup.\\ ($d$) & sup.\\ ($d_z$) \\\\",
        "\\midrule",
    ]

    def cells(method: str) -> str:
        rows = outcomes[outcomes.method == method]
        return " & ".join(_fmt(rows[c].mean(), digits) for c, _, digits in columns)

    lines.append(f"flat & baseline SFL & {cells('baseline')} & & \\\\")
    for method in methods:
        result = analysis["methods"].get(method)
        if result is None:
            continue
        if method == "h_sfl":
            order, evidence = "hierarchical", HIERARCHY_LABELS["h_sfl"]
        else:
            mode, _, source = method[2:].partition("_")
            order = f"hier., {mode}"
            evidence = HIERARCHY_LABELS.get(source, source)
        lines.append(
            f"{order} & {evidence} & {cells(method)} & "
            f"{result['superiority']['n_superior']}/{len(FORMULAS)} & "
            f"{result['superiority_paired']['n_superior']}/{len(FORMULAS)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_family(effects: pd.DataFrame, family: str, method: str = "rf") -> str:
    outcomes = [o for o, _ in FAMILIES[family]]
    subset = effects[(effects.method == method) & (effects.outcome.isin(outcomes))]
    aggregated = (
        subset.groupby("outcome")
        .agg(
            baseline=("baseline_mean", "mean"),
            augmented=("augmented_mean", "mean"),
            delta=("coefficient", "mean"),
            d=("d", "mean"),
            q=("qvalue", "min"),
            improve=("count_improve", "sum"),
            tie=("count_tie", "sum"),
            degrade=("count_degrade", "sum"),
        )
        .reindex(outcomes)
        .dropna(how="all")
    )
    lines = [
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Outcome & Baseline & Augmented & $\\Delta$ & $d$ & $q$ & improve/tie/degrade \\\\",
        "\\midrule",
    ]
    for outcome, row in aggregated.iterrows():
        digits = 0 if ("rank" in outcome or "wasted" in outcome or "locations" in outcome) else 3
        lines.append(
            f"{METRIC_LABELS.get(outcome, outcome)} & {_fmt(row.baseline, digits)} & "
            f"{_fmt(row.augmented, digits)} & {_fmt(row.delta, digits)} & {_fmt(row.d, 2)} & "
            f"{_fmt(row.q, 3)}{_stars(row.q)} & "
            f"{int(row.improve)}/{int(row.tie)}/{int(row.degrade)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


#: Groups of the method table: the registered method, its exploratory
#: variants, and the outcome-neutral controls, each with its short row label.
METHOD_GROUPS = (
    ("Method", (("baseline", "baseline SFL"), ("rf", "RF weighting"))),
    ("Variants", (("alt", "alternative mapping"), ("hybrid", "hybrid"),
                  ("failfirst", "fail-first"), ("adaptive", "adaptive"))),
    ("Controls", (("neg", "random importances"), ("perm", "permuted importances"),
                  ("pos", "oracle (positive)"))),
)


def table_methods(effects: pd.DataFrame) -> str:
    """The registered method beside its variants and the controls, grouped."""
    lines = [
        "\\begin{tabular}{ll" + "r" * len(PRIMARY_METRICS) + "}",
        "\\toprule",
        " & Condition & " + " & ".join(METRIC_LABELS[o] for o, _ in PRIMARY_METRICS) + " \\\\",
        "\\midrule",
    ]
    for group, members in METHOD_GROUPS:
        first = True
        for method, label in members:
            lookup = "rf" if method == "baseline" else method
            if effects[effects.method == lookup].empty:
                continue
            column = "baseline_mean" if method == "baseline" else "augmented_mean"
            cells = []
            for outcome, _ in PRIMARY_METRICS:
                rows = effects[(effects.method == lookup) & (effects.outcome == outcome)]
                cells.append(_fmt(rows[column].mean(), 0 if "wasted" in outcome else 3))
            lines.append(f"{group if first else ''} & {label} & " + " & ".join(cells) + " \\\\")
            first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_subjects(subjects: pd.DataFrame) -> str:
    ok = subjects[subjects.status == "ok"]
    grouped = (
        ok.groupby("project")
        .agg(
            bugs=("subject", "count"),
            loc=("loc", "median"),
            lines=("executable_lines", "median"),
            tests=("test_suite_size", "median"),
            failing=("failing_tests", "median"),
            features=("n_features_total", "median"),
            runtime=("runtime_total", "median"),
        )
        .sort_index()
    )
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Project & Bugs & LOC & Exec.\\ lines & Tests & Failing & Features & Runtime (s) \\\\",
        "\\midrule",
    ]
    for project, row in grouped.iterrows():
        lines.append(
            f"{project} & {int(row['bugs'])} & {int(row['loc']):,} & {int(row['lines']):,} & "
            f"{int(row['tests'])} & {int(row['failing'])} & {int(row['features']):,} & "
            f"{row['runtime']:,.0f} \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{All}} & {int(grouped['bugs'].sum())} & "
        f"{int(ok['loc'].median()):,} & {int(ok['executable_lines'].median()):,} & "
        f"{int(ok['test_suite_size'].median())} & {int(ok['failing_tests'].median())} & "
        f"{int(ok['n_features_total'].median()):,} & {ok['runtime_total'].median():,.0f} \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


IMPACT_METRICS = [
    ("impact_found@1", "impact@1"),
    ("impact_found@5", "impact@5"),
    ("impact_found@10", "impact@10"),
    ("impact_min_rank", "min.\\ impact rank"),
]


def _sensitivity_block(rows: List[Dict], columns: Sequence, title: str) -> str:
    lines = [
        "\\begin{tabular}{ll" + "r" * len(columns) + "}",
        "\\toprule",
        "Knob & Value & " + " & ".join(label for _, label in columns) + " \\\\",
        "\\midrule",
    ]
    current = None
    for row in rows:
        label = row["knob"] if row["knob"] != current else ""
        current = row["knob"]
        cells = []
        for key, _ in columns:
            value = row.get(key)
            digits = 0 if ("wasted" in key or "rank" in key) else 3
            cells.append(_fmt(value, digits) if value is not None else "--")
        lines.append(f"{label} & {row['value']} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_sensitivity(rows: List[Dict]) -> str:
    """Knobs that move the ranking, against the primary outcomes."""
    ranking = [r for r in rows if "threshold" not in r["knob"]]
    columns = [(o, METRIC_LABELS[o]) for o, _ in PRIMARY_METRICS]
    return _sensitivity_block(ranking, columns, "ranking")


def table_sensitivity_threshold(rows: List[Dict]) -> str:
    """The test-impact threshold, against the outcomes it can actually move."""
    threshold = [r for r in rows if "threshold" in r["knob"]]
    if not threshold:
        return "% no threshold grid\n"
    return _sensitivity_block(threshold, IMPACT_METRICS, "impact")


def table_diagnostics(diagnostics: pd.DataFrame) -> str:
    columns = [
        ("n_features_total", "execution features", 0),
        ("n_features_used", "non-constant features", 0),
        ("constant_fraction", "constant fraction", 3),
        ("n_vectors_failing", "failing vectors", 0),
        ("n_vectors_passing", "passing vectors", 0),
        ("mapping_conflict_rate", "multi-location features", 3),
        ("lines_mapped_rate", "lines reached by the mapping", 3),
        ("lines_with_importance_rate", "lines reached by a non-zero importance", 3),
        ("line_evidence_rate", "lines above the weight floor", 3),
        ("importance_n_nonzero", "features with non-zero importance", 0),
        ("importance_gini", "importance Gini", 3),
        ("importance_entropy", "importance entropy", 3),
        ("importance_skewness", "importance skewness", 2),
        ("stability_seeded_spearman", "seeded refit $\\rho$", 3),
        ("stability_bootstrap_spearman", "bootstrap refit $\\rho$", 3),
        ("stability_seeded_jaccard20", "seeded top-20 Jaccard", 3),
        ("stability_bootstrap_jaccard20", "bootstrap top-20 Jaccard", 3),
    ]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Diagnostic & Median & IQR low & IQR high & $n$ \\\\",
        "\\midrule",
    ]
    for key, label, digits in columns:
        if key not in diagnostics.columns:
            continue
        values = pd.to_numeric(diagnostics[key], errors="coerce").dropna()
        if values.empty:
            continue
        lines.append(
            f"{label} & {_fmt(values.median(), digits)} & "
            f"{_fmt(values.quantile(0.25), digits)} & {_fmt(values.quantile(0.75), digits)} & "
            f"{len(values)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_feasibility(diagnostics: pd.DataFrame, subjects: pd.DataFrame) -> str:
    phases = [
        ("time_trace_analysis", "trace analysis"),
        ("time_feature_matrix", "feature matrix"),
        ("time_forest", "random forest"),
        ("time_positive_control", "positive control"),
        ("time_variants", "exploratory variants"),
        ("time_mapping", "feature-to-line mapping"),
        ("time_stability", "stability refits"),
        ("time_total", "total per subject"),
    ]
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Phase & Median (s) & IQR low & IQR high \\\\",
        "\\midrule",
    ]
    for key, label in phases:
        if key not in diagnostics.columns:
            continue
        values = pd.to_numeric(diagnostics[key], errors="coerce").dropna()
        if values.empty:
            continue
        lines.append(
            f"{label} & {_fmt(values.median(), 1)} & {_fmt(values.quantile(0.25), 1)} & "
            f"{_fmt(values.quantile(0.75), 1)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def figure_project_effects(outcomes: pd.DataFrame, path: Path, outcome: str = "patch_exam_avg") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = outcomes[outcomes.method == "baseline"][["subject", "project", "formula", outcome]]
    aug = outcomes[outcomes.method == "rf"][["subject", "formula", outcome]]
    merged = base.merge(aug, on=["subject", "formula"], suffixes=("_base", "_aug"))
    merged["delta"] = merged[f"{outcome}_aug"] - merged[f"{outcome}_base"]
    order = merged.groupby("project")["delta"].median().sort_values().index.tolist()
    data = [merged[merged.project == project]["delta"].to_numpy() for project in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
    parts = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                       widths=0.62, medianprops=dict(color="#B45309", linewidth=1.4))
    for patch in parts["boxes"]:
        patch.set_facecolor("#DBEAFE")
        patch.set_edgecolor("#1D4ED8")
    ax.set_ylabel("paired $\\Delta$ EXAM (augmented $-$ baseline)")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def table_correlations(outcomes: pd.DataFrame) -> str:
    """Spearman correlations among the primary dependent variables."""
    columns = [o for o, _ in PRIMARY_METRICS] + [
        "function_exam", "impact_found@1", "hunk_found@1",
    ]
    columns = [c for c in columns if c in outcomes.columns]
    subset = outcomes[outcomes.method == "rf"][columns]
    matrix = subset.corr(method="spearman")
    #: Spelling every outcome out twice does not fit the text block, so the
    #: columns are numbered and the rows carry the names, as is conventional for
    #: a correlation matrix.
    header = " & ".join(f"({i})" for i in range(1, len(columns) + 1))
    lines = [
        "\\begin{tabular}{l" + "r" * len(columns) + "}",
        "\\toprule",
        f" & {header} \\\\",
        "\\midrule",
    ]
    for index, row in enumerate(columns, start=1):
        #: The matrix is symmetric; the upper triangle would only repeat it.
        cells = [
            _fmt(matrix.loc[row, column], 2) if position <= index else ""
            for position, column in enumerate(columns, start=1)
        ]
        label = f"({index}) {METRIC_LABELS.get(row, row)}"
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_example(
    artifact_dir: Path, subject: str = "middle_1", formula: str = "Tarantula"
) -> str:
    """Running example: per-line spectra, weight, baseline and augmented score."""
    import numpy as np

    from protocol.extract import read_artifact
    from protocol.formulas import suspiciousness
    from protocol.metrics import blended_scores

    artifact = read_artifact(artifact_dir / f"{subject}.json.gz")
    lines = artifact["lines"]
    ef = np.asarray(lines["ef"], dtype=float)
    ep = np.asarray(lines["ep"], dtype=float)
    weights = np.asarray(artifact["weights"]["rf"]["max"], dtype=float)
    base = suspiciousness(formula, ef, ep, artifact["totals"]["failed"],
                          artifact["totals"]["passed"])
    augmented = blended_scores(base, weights, 0.35, 1e-3)
    faulty = {(f, int(l)) for f, l in artifact["reference_lines"]}
    order = np.argsort(-augmented)
    rows = [
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Line & $e_f$ & $e_p$ & $w(l)$ & " + formula + " & Augmented \\\\",
        "\\midrule",
    ]
    for index in order:
        line = int(lines["line"][index])
        cells = [str(line), str(int(ef[index])), str(int(ep[index])),
                 f"{weights[index]:.3f}", f"{base[index]:.3f}", f"{augmented[index]:.3f}"]
        if (lines["file"][index], line) in faulty:
            #: The whole faulty row in bold, so it stands out against the striping.
            cells = ["\\textbf{" + c + "}" for c in cells]
        rows.append(" & ".join(cells) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows)


MODERATOR_LABELS = {
    "line_evidence_rate": "share of lines above the weight floor",
    "constant_fraction": "share of constant features",
    "importance_gini": "importance concentration (Gini)",
    "importance_entropy": "importance entropy",
    "mapping_conflict_rate": "share of multi-location features",
    "z_log_failing": "failing tests (log, standardized)",
    "z_log_lines": "executable lines (log, standardized)",
    "weak_signal": "fewer than three failing runs",
}


def table_moderators(analysis: Dict, outcome: str = "patch_exam_avg") -> str:
    """Subject properties that predict the size of the paired effect."""
    entry = analysis["methods"]["rf"].get("moderators", {}).get(outcome, {})
    if not entry.get("available"):
        return "% moderator analysis unavailable\n"
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Subject property & Coefficient & $p$ & Spearman $\\rho$ \\\\",
        "\\midrule",
    ]
    for name, values in entry["coefficients"].items():
        rho = entry["spearman"].get(name, float("nan"))
        lines.append(
            f"{MODERATOR_LABELS.get(name, name)} & {_fmt(values['coefficient'], 3)} & "
            f"{_fmt(values['pvalue'], 3)}{_stars(values['pvalue'])} & {_fmt(rho, 2)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


FAMILY_LABELS = {
    "fam_coverage": "coverage only",
    "fam_data_flow": "data flow only",
    "fam_value": "value only",
    "fam_condition": "condition only",
    "fam_exception": "exception only",
}

ABLATION_ORDER = [
    ("baseline", "unmodified SFL"),
    ("rf", "full method"),
    ("alt", "mapping: def-use to the use line only"),
    ("fam_coverage", "vocabulary: coverage features only"),
    ("fam_data_flow", "vocabulary: data-flow features only"),
    ("fam_value", "vocabulary: value features only"),
    ("fam_condition", "vocabulary: condition features only"),
    ("fam_exception", "vocabulary: exception features only"),
    ("perm", "association: importances permuted"),
    ("neg", "learning: random importances"),
    ("pos", "upper reference: the fault lines given the maximum weight"),
]


def table_ablations(outcomes: pd.DataFrame) -> str:
    """Every ablation of the method against the unmodified baseline."""
    metrics = [o for o, _ in PRIMARY_METRICS]
    metrics = [m for m in metrics if m in outcomes.columns]
    present = set(outcomes["method"])
    lines = [
        "\\begin{tabular}{" + RAGGED % "2.2cm" + RAGGED % "3.3cm"
        + "r" * len(metrics) + "}",
        "\\toprule",
        "Condition & What is removed & "
        + " & ".join(METRIC_LABELS.get(m, m) for m in metrics) + " \\\\",
        "\\midrule",
    ]
    for method, description in ABLATION_ORDER:
        if method not in present:
            continue
        rows = outcomes[outcomes.method == method]
        cells = [
            _fmt(float(rows[m].mean()), 0 if "wasted" in m else 3) for m in metrics
        ]
        label = METHOD_LABELS.get(method, FAMILY_LABELS.get(method, method))
        lines.append(f"{label} & {description} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def table_per_subject(analysis: Dict, method: str = "rf") -> str:
    """How often the method wins on an individual subject."""
    entry = analysis["methods"].get(method, {}).get("per_subject", {})
    if not entry.get("available"):
        return "% per-subject summary unavailable\n"
    rows = [
        ("subject/formula pairs", f"{entry['n_pairs']}"),
        ("mean primary outcomes improved", f"{entry['mean_metrics_improved']:.2f} of 5"),
        ("mean primary outcomes degraded", f"{entry['mean_metrics_degraded']:.2f} of 5"),
        ("superior ($\\ge$4 better, none worse)", f"{100*entry['share_superior']:.1f}\\%"),
        ("inferior ($\\ge$4 worse, none better)", f"{100*entry['share_inferior']:.1f}\\%"),
        ("more outcomes better than worse", f"{100*entry['share_more_better']:.1f}\\%"),
        ("more outcomes worse than better", f"{100*entry['share_more_worse']:.1f}\\%"),
        ("identical on all five", f"{100*entry['share_all_tied']:.1f}\\%"),
    ]
    lines = ["\\begin{tabular}{lr}", "\\toprule",
             "Per-subject outcome & Value \\\\", "\\midrule"]
    lines += [f"{name} & {value} \\\\" for name, value in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write_numbers(
    out_dir: Path,
    analysis: Dict,
    effects: pd.DataFrame,
    subjects: pd.DataFrame,
    diagnostics: pd.DataFrame,
    outcomes: Optional[pd.DataFrame] = None,
    categories: Optional[pd.DataFrame] = None,
) -> None:
    """Inline numbers used in the running text, one macro file each."""
    numbers = out_dir / "numbers"
    numbers.mkdir(parents=True, exist_ok=True)
    ok = subjects[subjects.status == "ok"]
    verdict = analysis["methods"]["rf"]["superiority"]
    checks_path = out_dir.parent / "study" / "checks.csv"

    values: Dict[str, str] = {
        "nsubjects": f"{len(ok)}",
        "nprojects": f"{ok['project'].nunique()}",
        "nexcluded": f"{len(subjects) - len(ok)}",
        "detectable": f"{analysis['detectable_effect_d']:.2f}",
        "nsuperior": f"{verdict['n_superior']}",
        "npairedsuperior": f"{analysis['methods']['rf']['superiority_paired']['n_superior']}",
        "signtestp": f"{verdict['sign_test_p']:.4f}",
        "medianfeatures": f"{ok['n_features_total'].median():,.0f}",
        "medianconstant": f"{100 * ok['constant_fraction'].median():.1f}",
        "medianruntime": f"{ok['runtime_total'].median():.1f}",
        "totalruntime": f"{ok['runtime_total'].sum() / 3600:.1f}",
        "weaksignal": f"{int(ok['weak_signal'].sum())}",
        "medianfailing": f"{ok['failing_tests'].median():.0f}",
        "medianevidence": f"{100 * ok['line_evidence_rate'].median():.1f}",
        "medianconflict": f"{100 * ok['mapping_conflict_rate'].median():.1f}",
        "medianlines": f"{ok['executable_lines'].median():,.0f}",
    }
    if checks_path.exists():
        checks = pd.read_csv(checks_path)
        if "reference_lines_ranked" in checks.columns:
            values["nounreached"] = f"{int((checks['reference_lines_ranked'] == 0).sum())}"
    for outcome, _ in PRIMARY_METRICS:
        rows = effects[(effects.method == "rf") & (effects.outcome == outcome)]
        key = outcome.replace("_", "").replace("@", "at")
        #: Effort is a line count: format it as the tables do, so the running text
        #: and the tables never disagree on the same quantity.
        wasted = "wasted" in outcome
        for prefix, column in (("base", "baseline_mean"), ("aug", "augmented_mean")):
            mean = rows[column].mean()
            values[f"{prefix}{key}"] = f"{mean:,.0f}" if wasted else f"{mean:.3f}"
        #: "-0.00" is an artifact of rounding a negative number that is zero to
        #: two places; it reads as a claim about direction that the value cannot
        #: support.
        d = rows["d"].mean()
        values[f"d{key}"] = f"{0.0 if abs(d) < 5e-3 else d:.2f}"
        if "d_paired" in rows.columns:
            dz = rows["d_paired"].mean()
            values[f"dz{key}"] = f"{0.0 if abs(dz) < 5e-3 else dz:.2f}"
            values[f"dz{key}min"] = f"{rows['d_paired'].min():.2f}"
            values[f"dz{key}max"] = f"{rows['d_paired'].max():.2f}"
    for method in ("neg", "pos", "perm"):
        rows = effects[(effects.method == method) & (effects.outcome == "patch_top1")]
        if not rows.empty:
            values[f"{method}top1"] = f"{rows['augmented_mean'].mean():.3f}"

    #: The weakest evidence against the null over the five primary outcomes: if
    #: even the smallest q is large, no outcome is close to significant.
    primary_names = [o for o, _ in PRIMARY_METRICS]
    rf_primary = effects[(effects.method == "rf") & (effects.outcome.isin(primary_names))]
    if "qvalue" in rf_primary.columns and not rf_primary.empty:
        values["minq"] = f"{rf_primary['qvalue'].min():.2f}"
    if not rf_primary.empty:
        top1 = rf_primary[rf_primary.outcome == "patch_top1"]
        if not top1.empty:
            best = top1.loc[top1["d"].idxmax()]
            values["bestformula"] = str(best["formula"])
            values["bestformulad"] = f"{best['d']:.2f}"

    #: Per-subject reading of the same superiority rule.
    per_subject = analysis["methods"]["rf"].get("per_subject", {})
    if per_subject.get("available"):
        values["pspairs"] = f"{per_subject['n_pairs']:,}"
        values["psimproved"] = f"{per_subject['mean_metrics_improved']:.2f}"
        values["psdegraded"] = f"{per_subject['mean_metrics_degraded']:.2f}"
        for share, key in (("share_superior", "pssuperior"),
                           ("share_inferior", "psinferior"),
                           ("share_more_better", "psmorebetter"),
                           ("share_more_worse", "psmoreworse"),
                           ("share_all_tied", "psalltied")):
            values[key] = f"{100 * per_subject[share]:.1f}"
        by_formula = per_subject.get("per_formula_superior") or {}
        if by_formula:
            name = max(by_formula, key=by_formula.get)
            values["psbestformula"] = name
            values["psbestformulashare"] = f"{100 * by_formula[name]:.1f}"
    if outcomes is not None:
        binary = ["patch_top1", "patch_top5", "patch_top10",
                  "impact_found@1", "hunk_found@1"]
        present = [c for c in binary if c in outcomes.columns]
        primary = outcomes[outcomes.method.isin(["baseline", "rf"])]
        stacked = primary[present].to_numpy(dtype=float).ravel()
        fractional = np.mean((stacked > 1e-9) & (stacked < 1 - 1e-9))
        values["fractionalshare"] = f"{100 * fractional:.1f}"
    if "importance_n_nonzero" in diagnostics.columns:
        values["mediannonzero"] = (
            f"{pd.to_numeric(diagnostics['importance_n_nonzero'], errors='coerce').median():,.0f}"
        )
    models = analysis["methods"]["rf"]["models"]
    converged = [m.get("converged", True) for m in models.values() if isinstance(m, dict)]
    values["nmodels"] = f"{len(converged)}"
    values["nconverged"] = f"{sum(1 for c in converged if c)}"
    values["nmodelsfailed"] = f"{sum(1 for c in converged if not c)}"
    if categories is not None and not categories.empty:
        for condition, key in (("neg", "negdensityrho"), ("rf", "rfdensityrho"),
                               ("perm", "permdensityrho")):
            column = f"density_rho_{condition}"
            if column in categories.columns:
                values[key] = f"{categories[column].median():.2f}"
        for column, key in (("lines_mapped_rate", "medianmapped"),
                            ("lines_with_importance_rate", "medianimportancereach")):
            if column in categories.columns:
                values[key] = f"{100 * categories[column].median():.1f}"
        for family in ("coverage", "data_flow", "value", "condition", "exception"):
            column = f"share_{family}"
            if column in categories.columns:
                values[f"share{family.replace('_', '')}"] = (
                    f"{100 * categories[column].mean():.1f}"
                )
    #: The positive control bounds what ANY weighting can achieve under the
    #: registered blend: because the blend multiplies the formula's own
    #: suspiciousness, a line the formula scores near zero cannot be promoted by
    #: any weight. Reporting the ceiling on the subjects where a reference line is
    #: actually ranked separates that structural limit from the reach problem.
    if outcomes is not None and checks_path.exists():
        checks = pd.read_csv(checks_path)
        if "reference_lines_ranked" in checks.columns:
            reachable = set(checks[checks.reference_lines_ranked > 0]["subject"])
            values["nreachable"] = f"{len(reachable)}"
            subset = outcomes[outcomes.subject.isin(reachable)]
            for method, key in (("baseline", "reachbase"), ("pos", "reachpos"),
                                ("rf", "reachrf")):
                rows = subset[subset.method == method]
                if not rows.empty and "patch_top1" in rows.columns:
                    values[key] = f"{rows['patch_top1'].mean():.3f}"

    #: Negative-range behaviour of the registered blend. Naish2 is the only
    #: formula in the set whose suspiciousness goes below zero, and the blend
    #: multiplies the weight into that score, which inverts the weighting there.
    #: The prenorm probe removes the inversion; reporting both lets the text say
    #: the inversion is real AND that it does not explain the Naish2 reversal.
    sign_path = out_dir.parent / "study" / "sign_probe.json"
    if sign_path.exists():
        sign = json.loads(sign_path.read_text())
        negatives = sign["negative_scores"]
        worst = max(negatives, key=lambda k: negatives[k]["share"])
        values["negformula"] = worst
        values["negshare"] = f"{100 * negatives[worst]['share']:.1f}"
        values["negsubjects"] = f"{negatives[worst]['subjects']}"
        values["negothers"] = f"{sum(1 for k, v in negatives.items() if v['share'] == 0)}"
        inversion = sign["inversion"]
        if inversion.get("rho_positive_median") is not None:
            values["rhopositive"] = f"{inversion['rho_positive_median']:.3f}"
            values["rhonegative"] = f"{inversion['rho_negative_median']:.3f}"

    #: The post-hoc prenorm probe, run through the same analysis as the
    #: confirmatory result so the effect sizes are directly comparable.
    probe_path = out_dir.parent / "study" / "analysis_prenorm.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())["methods"]["rf"]

        def _records(block):
            return {(r["outcome"], r["formula"]): r
                    for lst in block["families"].values()
                    for r in lst if isinstance(r, dict)}

        registered, probed = _records(analysis["methods"]["rf"]), _records(probe)
        key = ("function_faulty_mass", "Naish2")
        if key in registered and key in probed:
            values["naishd"] = f"{registered[key]['d']:.2f}"
            values["naishdprobe"] = f"{probed[key]['d']:.2f}"
        others = [r["d"] for (outcome, formula), r in registered.items()
                  if outcome == "function_faulty_mass" and formula != "Naish2"]
        if others:
            values["othersdmin"] = f"{min(others):.2f}"
            values["othersdmax"] = f"{max(others):.2f}"
        values["ndegrading"] = f"{sum(1 for r in registered.values() if r.get('degrades'))}"
        values["ndegradingprobe"] = f"{sum(1 for r in probed.values() if r.get('degrades'))}"
        values["probesuperior"] = f"{probe['superiority']['n_superior']}"
        model = analysis["methods"]["rf"]["models"].get("function_faulty_mass") or {}
        joint = model.get("interaction_joint") or {}
        if joint.get("pvalue") is not None:
            values["faminteraction"] = (
                "<0.001" if joint["pvalue"] < 0.001 else f"{joint['pvalue']:.3f}"
            )

    #: Moderator screen: how many of the fitted coefficients reach nominal
    #: significance, against how many were tested. Reporting both is the honest
    #: way to say "no reliable moderator" when multiplicity guarantees some hits.
    moderators = analysis["methods"]["rf"].get("moderators") or {}
    tested = hits = 0
    for entry in moderators.values():
        if not isinstance(entry, dict) or not entry.get("available"):
            continue
        for coefficient in entry.get("coefficients", {}).values():
            tested += 1
            hits += coefficient["pvalue"] < 0.05
    if tested:
        values["nmoderatortests"] = f"{tested}"
        values["nmoderatorsig"] = f"{hits}"
        values["nmoderatorexpected"] = f"{0.05 * tested:.1f}"
        #: How many of those survive a Holm correction over the whole screen.
        from statsmodels.stats.multitest import multipletests
        pvalues = [c["pvalue"] for e in moderators.values()
                   if isinstance(e, dict) and e.get("available")
                   for c in e.get("coefficients", {}).values()]
        values["nmoderatorholm"] = f"{int(multipletests(pvalues, method='holm')[0].sum())}"

    #: Distribution-free agreement for the effort outcomes, whose standardized
    #: effect is diluted by a heavy-tailed between-subject spread: the paired
    #: rank-biserial correlation (magnitude) over the five formulas.
    records = [r for lst in analysis["methods"]["rf"]["families"].values()
               for r in lst if isinstance(r, dict)]
    for outcome, key in (("patch_exam_avg", "exam"), ("patch_wasted_avg", "wasted")):
        rb = [abs(r["wilcoxon_rank_biserial"]) for r in records
              if r["outcome"] == outcome and r.get("wilcoxon_rank_biserial") is not None]
        if rb:
            values[f"rb{key}min"] = f"{min(rb):.2f}"
            values[f"rb{key}max"] = f"{max(rb):.2f}"
        pvalues = [r["wilcoxon_pvalue"] for r in records
                   if r["outcome"] == outcome and r.get("wilcoxon_pvalue") is not None]
        if pvalues:
            #: Formulas for which the Wilcoxon test rejects equality at 0.05.
            values[f"wil{key}nreject"] = f"{sum(p < 0.05 for p in pvalues)}"
    #: Pooled standardized effects for the broader ground truths quoted in the text.
    for outcome, key in (("impact_found@1", "dimpacttop1"), ("hunk_found@1", "dhunktop1"),
                         ("hunk_found@10", "dhunktop10"), ("function_exam", "dfunctionexam")):
        rows = effects[(effects.method == "rf") & (effects.outcome == outcome)]
        if not rows.empty:
            values[key] = f"{rows['d'].mean():.2f}"
    #: The weakest formula on top-1, beside the strongest one already reported.
    rf_top1 = effects[(effects.method == "rf") & (effects.outcome == "patch_top1")]
    if not rf_top1.empty:
        worst = rf_top1.loc[rf_top1["d"].idxmin()]
        values["worstformula"] = str(worst["formula"])
        values["worstformulad"] = f"{worst['d']:.2f}"
    top1_model = analysis["methods"]["rf"]["models"].get("patch_top1") or {}
    joint = top1_model.get("interaction_joint") or {}
    if joint.get("pvalue") is not None:
        values["topinteraction"] = (
            "<0.001" if joint["pvalue"] < 0.001 else f"={joint['pvalue']:.3f}"
        )
    #: Projects on which the mean paired top-1 change is positive.
    if outcomes is not None:
        pair = outcomes[outcomes.method.isin(["baseline", "rf"])].pivot_table(
            index=["subject", "formula"], columns="method", values="patch_top1")
        gain = (pair["rf"] - pair["baseline"]).groupby(
            pair.index.get_level_values("subject").str.rsplit("_", n=1).str[0]).mean()
        values["npositiveprojects"] = f"{int((gain > 1e-9).sum())}"
        for method in ("neg", "perm"):
            rows = effects[(effects.method == method) & (effects.outcome == "patch_top10")]
            if not rows.empty:
                values[f"{method}top10"] = f"{rows['augmented_mean'].mean():.3f}"

    #: How much of the oracle's headroom the registered method actually captures.
    #: Stated in the text, so it must come from the data rather than be typed.
    try:
        base = float(values["basepatchtop1"])
        aug = float(values["augpatchtop1"])
        oracle = float(values["postop1"])
    except (KeyError, ValueError):
        pass
    else:
        headroom = oracle - base
        if headroom > 1e-9:
            values["headroompoints"] = f"{100 * headroom:.1f}"
            values["headroomshare"] = f"{100 * (aug - base) / headroom:.0f}"

    #: Share of reference lines the baseline formula scores at zero: these cannot
    #: be promoted by any weight under a multiplicative blend.
    if checks_path.exists():
        checks = pd.read_csv(checks_path)
        shares = [c for c in checks.columns if c.startswith("zero_susp_")]
        if shares and "n_reference_ranked" in checks.columns:
            #: Pool over reference lines rather than averaging per-subject shares,
            #: so the number means what the text says it means: the share of
            #: reference lines, not the share for the average subject.
            counts = pd.to_numeric(checks["n_reference_ranked"], errors="coerce")
            total = counts.sum()
            weighted = checks[shares].mul(counts, axis=0).sum() / total
            values["zerosusp"] = f"{100 * weighted.min():.1f}"
            values["zerosuspmax"] = f"{100 * weighted.max():.1f}"
            worst = weighted.idxmax().removeprefix("zero_susp_")
            values["zerosuspworst"] = {
                "tarantula": "Tarantula", "ochiai": "Ochiai", "dstar": "DStar",
                "naish2": "Naish2", "gp13": "GP13",
            }.get(worst, worst)

    #: Best single feature family, to say plainly that no family carries the effect.
    if outcomes is not None:
        families = [f"fam_{f}" for f in
                    ("coverage", "data_flow", "value", "condition", "exception")]
        rows = outcomes[outcomes.method.isin(families)]
        if not rows.empty and "patch_top1" in rows.columns:
            means = rows.groupby("method")["patch_top1"].mean()
            best = means.idxmax()
            values["bestfamily"] = FAMILY_LABELS.get(
                best.removeprefix("fam_"), best.removeprefix("fam_").replace("_", " ")
            )
            values["bestfamilytopone"] = f"{means.max():.3f}"

    #: How far the blend parameter can move the headline outcome at all.
    sensitivity_path = out_dir.parent / "study" / "sensitivity.json"
    if sensitivity_path.exists():
        grid = json.loads(sensitivity_path.read_text())
        lam = [r["patch_top1"] for r in grid
               if "lambda" in str(r.get("knob", "")) and "patch_top1" in r]
        if len(lam) > 1:
            values["lambdaspan"] = f"{max(lam) - min(lam):.3f}"
        #: The direction matters as much as the span: if top-1 falls as lambda
        #: rises, the grid's optimum sits at the boundary, towards no weighting.
        pairs = sorted((r["value"], r["patch_top1"]) for r in grid
                       if "lambda" in str(r.get("knob", "")) and "patch_top1" in r)
        if len(pairs) > 1:
            values["lambdamin"] = f"{pairs[0][0]:.2f}"
            values["lambdamax"] = f"{pairs[-1][0]:.2f}"
            values["lambdaattmin"] = f"{pairs[0][1]:.3f}"
            values["lambdaattmax"] = f"{pairs[-1][1]:.3f}"

        def _grid(knob: str) -> List[Dict]:
            return sorted((r for r in grid if knob in str(r.get("knob", ""))),
                          key=lambda r: str(r["value"]) if knob == "aggregation" else r["value"])

        lam_rows = _grid("lambda")
        if lam_rows:
            for key, outcome in (("impact", "impact_found@1"), ("topten", "patch_top10"),
                                 ("topfive", "patch_top5")):
                values[f"lam{key}atmin"] = f"{lam_rows[0][outcome]:.3f}"
                values[f"lam{key}atmax"] = f"{lam_rows[-1][outcome]:.3f}"
            values["baseimpacttop1"] = f"{lam_rows[0]['baseline_impact_found@1']:.3f}"
        #: Largest deviation of any primary outcome (except wasted effort, which is
        #: a line count) from the registered epsilon over the epsilon grid.
        eps_rows = _grid("varepsilon")
        registered = next((r for r in eps_rows if abs(r["value"] - 1e-3) < 1e-12), None)
        if registered is not None:
            deviation = max(
                abs(r[o] - registered[o]) for r in eps_rows
                for o in ("patch_top1", "patch_top5", "patch_top10", "patch_exam_avg")
            )
            values["epsdeviation"] = f"{deviation:.3f}"
        for row in _grid("aggregation"):
            values[f"agg{row['value']}top1"] = f"{row['patch_top1']:.3f}"
            values[f"agg{row['value']}topten"] = f"{row['patch_top10']:.3f}"
        thresholds = _grid("threshold")
        if thresholds:
            gaps = [100 * (r["baseline_impact_found@1"] - r["impact_found@1"]) for r in thresholds]
            values["thresholdgapmin"] = f"{min(gaps):.0f}"
            values["thresholdgapmax"] = f"{max(gaps):.0f}"

    #: The largest standardized effect of the registered method on any primary
    #: outcome and formula: the whole verdict in one number.
    primary_rf = effects[(effects.method == "rf")
                         & (effects.outcome.isin([o for o, _ in PRIMARY_METRICS]))]
    if not primary_rf.empty:
        values["rfmaxabsd"] = f"{primary_rf['d'].abs().max():.2f}"
        if "d_paired" in primary_rf.columns:
            values["rfmaxabsdz"] = f"{primary_rf['d_paired'].abs().max():.2f}"
    #: How reproducible the forest's choice of features is when refitted.
    for column, key in (("stability_seeded_jaccard20", "seededjaccard"),
                        ("stability_bootstrap_jaccard20", "bootstrapjaccard"),
                        ("stability_seeded_spearman", "seededrho")):
        if column in diagnostics.columns and diagnostics[column].notna().any():
            values[key] = f"{diagnostics[column].median():.2f}"
            values["nstability"] = f"{int(diagnostics[column].notna().sum())}"
    if "failing_tests" in subjects.columns:
        ok_failing = ok["failing_tests"]
        values["shareonefailing"] = f"{100 * (ok_failing == 1).mean():.0f}"
        values["sharetwofailing"] = f"{100 * (ok_failing <= 2).mean():.0f}"

    #: Effect sizes of the controls and of the family ablations, quoted in the text.
    for method in ("rf", "neg", "perm", "pos", "alt", "hybrid", "failfirst", "adaptive",
                   "fam_coverage", "fam_data_flow", "fam_value", "fam_condition",
                   "fam_exception"):
        key = method.replace("_", "")
        for outcome, short in (("patch_top1", "topone"), ("patch_top10", "topten"),
                               ("patch_exam_avg", "exam"), ("patch_wasted_avg", "wasted")):
            rows = effects[(effects.method == method) & (effects.outcome == outcome)]
            if rows.empty:
                continue
            values[f"{key}{short}"] = (f"{rows['augmented_mean'].mean():,.0f}" if short == "wasted"
                                       else f"{rows['augmented_mean'].mean():.3f}")
            values[f"{key}d{short}"] = f"{rows['d'].mean():.2f}"
            if "d_paired" in rows.columns:
                values[f"{key}dz{short}"] = f"{rows['d_paired'].mean():.2f}"
            #: Subjects (mean over formulas) on which the outcome got better / worse.
            values[f"{key}improve{short}"] = f"{rows['count_improve'].mean():.0f}"
            values[f"{key}degrade{short}"] = f"{rows['count_degrade'].mean():.0f}"
        result = analysis["methods"].get(method)
        if result is not None:
            values[f"{key}nsuperior"] = f"{result['superiority']['n_superior']}"

    #: Post-hoc feature scorers and the tie-break-only rule (exploratory).
    study = out_dir.parent / "study"
    posthoc_analysis = study / "analysis_posthoc.json"
    posthoc_effects = study / "effects_posthoc.csv"
    if posthoc_analysis.exists() and posthoc_effects.exists():
        ph = json.loads(posthoc_analysis.read_text())
        phe = pd.read_csv(posthoc_effects)
        from protocol.posthoc import POSTHOC_METHODS

        scorers = phe[phe.method.isin(POSTHOC_METHODS)
                      & phe.outcome.isin([o for o, _ in PRIMARY_METRICS])]
        if not scorers.empty:
            values["phmaxabsd"] = f"{scorers['d'].abs().max():.2f}"
            values["phmaxabsdz"] = f"{scorers['d_paired'].abs().max():.2f}"
            values["phnsuperiorany"] = f"{max(ph['methods'][m]['superiority']['n_superior'] for m in POSTHOC_METHODS if m in ph['methods'])}"
            values["phnpairedsuperiorany"] = f"{max(ph['methods'][m]['superiority_paired']['n_superior'] for m in POSTHOC_METHODS if m in ph['methods'])}"
            beyond = scorers[scorers['d_paired'].abs() >= 0.2]
            values["phnbeyondz"] = f"{len(beyond)}"
            values["phnbeyondzexam"] = f"{int((beyond.outcome == 'patch_exam_avg').sum())}"
            values["phnbeyondzwasted"] = f"{int((beyond.outcome == 'patch_wasted_avg').sum())}"
            values["phnbeyondztopk"] = f"{int(beyond.outcome.isin(['patch_top1', 'patch_top5', 'patch_top10']).sum())}"
            values["phncells"] = f"{len(scorers)}"
        for method, result in ph["methods"].items():
            key = "ph" + method.replace("_", "")
            values[f"{key}nsuperior"] = f"{result['superiority']['n_superior']}"
            values[f"{key}npairedsuperior"] = f"{result['superiority_paired']['n_superior']}"
            for outcome, short in (("patch_top1", "topone"), ("patch_top5", "topfive"),
                                   ("patch_top10", "topten"), ("patch_exam_avg", "exam"),
                                   ("patch_wasted_avg", "wasted")):
                rows = phe[(phe.method == method) & (phe.outcome == outcome)]
                if rows.empty:
                    continue
                mean = rows["augmented_mean"].mean()
                values[f"{key}{short}"] = f"{mean:,.0f}" if short == "wasted" else f"{mean:.3f}"
                values[f"{key}d{short}"] = f"{rows['d'].mean():.2f}"
                values[f"{key}dz{short}"] = f"{rows['d_paired'].mean():.2f}"
                values[f"{key}d{short}max"] = f"{rows['d'].max():.2f}"
                values[f"{key}dz{short}max"] = f"{rows['d_paired'].max():.2f}"
                values[f"{key}improve{short}"] = f"{rows['count_improve'].mean():.0f}"
                values[f"{key}degrade{short}"] = f"{rows['count_degrade'].mean():.0f}"
                #: Formulas whose paired effect reaches the registered threshold.
                values[f"{key}nbeyondz{short}"] = f"{int((rows['d_paired'].abs() >= 0.2).sum())}"

    values.update(ltr_macros(study))

    #: A macro that renders as "nan" would ship silently into the manuscript. Drop
    #: it instead: the missing \input then fails the LaTeX build, which is visible.
    for key, value in sorted(values.items()):
        if value.strip().lower() in {"nan", "-nan", "inf", "-inf", "none"}:
            continue
        (numbers / f"{key}.tex").write_text(value)



def ltr_macros(study: Path) -> Dict[str, str]:
    """Macros of the cross-subject learning-to-rank analysis (``ltr*``)."""
    values: Dict[str, str] = {}
    analysis_path, effects_path = study / "analysis_ltr.json", study / "effects_ltr.csv"
    if not (analysis_path.exists() and effects_path.exists()):
        return values
    analysis = json.loads(analysis_path.read_text())["methods"]
    effects = pd.read_csv(effects_path)
    outcomes = (("patch_top1", "topone"), ("patch_top5", "topfive"), ("patch_top10", "topten"),
                ("patch_exam_avg", "exam"), ("patch_wasted_avg", "wasted"))
    base = effects[effects.method == "lex_ltr_full"]
    for outcome, short in outcomes:
        rows = base[base.outcome == outcome]
        if not rows.empty:
            mean = rows["baseline_mean"].mean()
            values[f"ltrbase{short}"] = f"{mean:,.0f}" if short == "wasted" else f"{mean:.3f}"
    for method, result in analysis.items():
        if not method.startswith("lex_ltr_"):
            continue
        key = "ltr" + method[len("lex_ltr_"):].replace("_", "")
        values[f"{key}nsuperior"] = f"{result['superiority']['n_superior']}"
        values[f"{key}npairedsuperior"] = f"{result['superiority_paired']['n_superior']}"
        for outcome, short in outcomes:
            rows = effects[(effects.method == method) & (effects.outcome == outcome)]
            if rows.empty:
                continue
            mean = rows["augmented_mean"].mean()
            values[f"{key}{short}"] = f"{mean:,.0f}" if short == "wasted" else f"{mean:.3f}"
            values[f"{key}d{short}"] = f"{rows['d'].mean():.2f}"
            values[f"{key}dz{short}"] = f"{rows['d_paired'].mean():.2f}"
            values[f"{key}d{short}min"] = f"{rows['d'].abs().min():.2f}"
            values[f"{key}dz{short}min"] = f"{rows['d_paired'].abs().min():.2f}"
            values[f"{key}improve{short}"] = f"{rows['count_improve'].mean():.0f}"
            values[f"{key}degrade{short}"] = f"{rows['count_degrade'].mean():.0f}"
    #: Seed spread: the pre-fixed seed and seeds 1-5.
    for variant in ("vec", "trace", "full"):
        names = [f"lex_ltr_{variant}"] + [f"lex_ltr_{variant}_s{s}" for s in range(1, 6)]
        present = [analysis[n] for n in names if n in analysis]
        if not present:
            continue
        sup = [r["superiority"]["n_superior"] for r in present]
        paired = [r["superiority_paired"]["n_superior"] for r in present]
        values[f"ltr{variant}seeds"] = f"{len(present)}"
        values[f"ltr{variant}seedsupmin"] = f"{min(sup)}"
        values[f"ltr{variant}seedsupmax"] = f"{max(sup)}"
        values[f"ltr{variant}seedmajority"] = f"{sum(v >= 3 for v in sup)}"
        values[f"ltr{variant}seedpairedmin"] = f"{min(paired)}"
        rows = effects[effects.method.isin(names)]
        for outcome, short in outcomes[:4]:
            per_seed = rows[rows.outcome == outcome].groupby("method")["augmented_mean"].mean()
            if not per_seed.empty:
                values[f"ltr{variant}seed{short}min"] = f"{per_seed.min():.3f}"
                values[f"ltr{variant}seed{short}max"] = f"{per_seed.max():.3f}"
    #: Robustness grid (fast proxy metrics) and permutation importance.
    grid_path = study / "ltr_robustness.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        wide = grid.pivot(index="config", columns="variant")
        values["ltrgridconfigs"] = f"{len(wide)}"
        #: Configurations in which the full ranker beats the random-importance
        #: control on top-1, top-10 and EXAM alike.
        better = ((wide[("top1", "full")] > wide[("top1", "neg")])
                  & (wide[("top10", "full")] > wide[("top10", "neg")])
                  & (wide[("exam", "full")] < wide[("exam", "neg")])).sum()
        values["ltrgridordered"] = f"{int(better)}"
        for variant in ("sfl", "neg", "full"):
            for metric, short in (("top10", "topten"), ("exam", "exam")):
                column = wide[(metric, variant)]
                values[f"ltrgrid{variant}{short}min"] = f"{column.min():.3f}"
                values[f"ltrgrid{variant}{short}max"] = f"{column.max():.3f}"
        if "logreg" in wide.index:
            values["ltrgridlogregfulltopten"] = f"{wide.loc['logreg', ('top10', 'full')]:.3f}"
            values["ltrgridlogregnegtopten"] = f"{wide.loc['logreg', ('top10', 'neg')]:.3f}"
    transplant_path = study / "transplant.csv"
    if transplant_path.exists():
        transplant = pd.read_csv(transplant_path)
        ok = transplant[transplant.status == "ok"]
        values["ltrtracesubjects"] = f"{len(ok)}"
        values["ltrtracecoveragemin"] = f"{100 * ok['line_coverage'].min():.0f}"
        values["ltrtracecoveragemedian"] = f"{100 * ok['line_coverage'].median():.0f}"
    importance_path = study / "ltr_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).set_index("signal")
        for signal in ("x_spectrum", "fsm"):
            if signal in imp.index:
                key = signal.replace("_", "")
                for metric, short in (("top1", "topone"), ("top5", "topfive"), ("top10", "topten")):
                    values[f"ltrimp{key}{short}"] = f"{imp.loc[signal, metric]:.3f}"
        others = imp.drop([s for s in ("x_spectrum", "fsm") if s in imp.index])
        values["ltrimpothermax"] = f"{others[['top1', 'top5', 'top10']].max().max():.3f}"
        values["ltrimpnsignals"] = f"{len(imp)}"
    return values


def render(study_dir: Path, out_dir: Path, tag: Optional[str] = None) -> None:
    suffix = "" if tag is None else f"_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(study_dir / f"effects{suffix}.csv")
    subjects = pd.read_csv(study_dir / f"subjects{suffix}.csv")
    analysis = json.loads((study_dir / f"analysis{suffix}.json").read_text())

    write_table(out_dir, "tab_primary.tex", table_primary(effects))
    write_table(out_dir, "tab_superiority.tex", table_superiority(analysis))
    hier_path = study_dir / "analysis_hier.json"
    if tag is None and hier_path.exists():
        hier = json.loads(hier_path.read_text())
        hier_outcomes = pd.read_csv(study_dir / "outcomes_hier.csv")
        write_table(out_dir, "tab_hierarchy.tex", table_hierarchy(
            hier, hier_outcomes, [m for m in hier["methods"] if m.startswith("h_")]))
    posthoc_path = study_dir / "analysis_posthoc.json"
    if tag is None and posthoc_path.exists():
        posthoc = json.loads(posthoc_path.read_text())
        posthoc_effects = pd.read_csv(study_dir / "effects_posthoc.csv")
        write_table(out_dir, "tab_posthoc.tex", table_posthoc(
            posthoc, posthoc_effects,
            [m for m in ("rf", "neg", "x_spectrum", "x_mi", "x_l1", "x_et", "x_nn", "tie_rf",
                         "tie_spectrum", "tie_nn", "pos") if m in posthoc["methods"]],
        ))
    ltr_path = study_dir / "analysis_ltr.json"
    if tag is None and ltr_path.exists():
        ltr = json.loads(ltr_path.read_text())
        ltr_effects = pd.read_csv(study_dir / "effects_ltr.csv")
        labels = dict(METHOD_LABELS, **LTR_LABELS)
        write_table(out_dir, "tab_ltr.tex", table_posthoc(
            ltr, ltr_effects, [m for m in LTR_TABLE if m in ltr["methods"]], labels))
    write_table(out_dir, "tab_superiority_paired.tex",
                table_superiority(analysis, key="superiority_paired"))
    write_table(out_dir, "tab_superiority_methods.tex",
        table_superiority_methods(
            analysis,
            [m for m in ("rf", "alt", "hybrid", "failfirst", "adaptive", "neg", "perm", "pos")
             if m in analysis["methods"]],
        )
    )
    write_table(out_dir, "tab_function.tex", table_family(effects, "function"))
    write_table(out_dir, "tab_impact.tex", table_family(effects, "impact"))
    write_table(out_dir, "tab_hunk.tex", table_family(effects, "hunk"))
    write_table(out_dir, "tab_effort.tex", table_family(effects, "patch_effort"))
    write_table(out_dir, "tab_subjects.tex", table_subjects(subjects))
    write_table(out_dir, "tab_methods.tex", table_methods(effects))
    diagnostics = pd.read_csv(study_dir / f"diagnostics{suffix}.csv")
    outcomes = pd.read_csv(study_dir / f"outcomes{suffix}.csv")
    write_table(out_dir, "tab_ablations.tex", table_ablations(outcomes))
    write_table(out_dir, "tab_per_subject.tex", table_per_subject(analysis))
    write_table(out_dir, "tab_correlations.tex", table_correlations(outcomes))
    write_table(out_dir, "tab_moderators.tex", table_moderators(analysis))
    example = study_dir.parent / "artifacts"
    if (example / "middle_1.json.gz").exists():
        write_table(out_dir, "tab_example.tex", table_example(example))
    # tab_diagnostics is written after the categories merge below.
    write_table(out_dir, "tab_feasibility.tex", table_feasibility(diagnostics, subjects))
    categories_path = study_dir / f"categories{suffix}.csv"
    categories = pd.read_csv(categories_path) if categories_path.exists() else None
    if categories is not None:
        reach = [c for c in categories.columns
                 if c.startswith("lines_") or c.startswith("share_")]
        diagnostics = diagnostics.merge(
            categories[["subject"] + reach], on="subject", how="left"
        )
    write_table(out_dir, "tab_diagnostics.tex", table_diagnostics(diagnostics))
    write_numbers(out_dir, analysis, effects, subjects, diagnostics, outcomes, categories)
    from protocol.figures import render_figures

    render_figures(study_dir, out_dir, tag)

    sensitivity_path = study_dir / f"sensitivity{suffix}.json"
    if sensitivity_path.exists():
        grid = json.loads(sensitivity_path.read_text())
        write_table(out_dir, "tab_sensitivity.tex", table_sensitivity(grid))
        write_table(out_dir, "tab_sensitivity_threshold.tex",
                    table_sensitivity_threshold(grid))
    print(f"wrote tables to {out_dir}")
