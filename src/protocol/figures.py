"""Manuscript figures.

One shared visual system: a validated categorical palette used in fixed slot
order, thin marks with rounded data-ends, recessive chrome, a legend whenever
two or more series share a panel, and selective direct labels. Every figure has
a corresponding table in the manuscript, which is also the relief required by
the palette's sub-3:1 contrast slots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from protocol.analysis import HIGHER, PRIMARY_METRICS  # noqa: E402
from protocol.formulas import FORMULAS  # noqa: E402

# --- design tokens ---------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8981"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948"]
BLUE_LIGHT = "#9ec5f4"
DIVERGE_NEG = "#d03b3b"

COLUMN_IN = 372.0 / 72.27  # sn-jnl single column, 372 pt
WRAP_IN = 0.46 * COLUMN_IN  # width of a wrapped figure

LABELS = {
    "patch_top1": "top-1",
    "patch_top5": "top-5",
    "patch_top10": "top-10",
    "patch_exam_avg": "EXAM",
    "patch_wasted_avg": "wasted effort",
}
METHOD_LABELS = {
    "baseline": "baseline SFL",
    "rf": "RF weighting",
    "alt": "alternative mapping",
    "neg": "negative control",
    "perm": "permutation control",
    "pos": "positive control",
    "hybrid": "hybrid",
    "failfirst": "fail-first",
    "adaptive": "adaptive",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "figure.dpi": 200,
    })


def _clean(ax, grid_axis: Optional[str] = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


def _rounded_barh(ax, y, width, height, color, radius_pt=2.0, **kwargs):
    """Horizontal bar anchored at the baseline with a rounded data-end."""
    radius = min(radius_pt / 72.0, abs(width) / 3.0) if width else 0.0
    patch = FancyBboxPatch(
        (0, y - height / 2), max(width - radius, 1e-9), height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, mutation_aspect=1e-9, **kwargs,
    )
    ax.add_patch(patch)
    return patch


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# --- figures ---------------------------------------------------------------
def figure_bottleneck(diagnostics: pd.DataFrame, path: Path) -> None:
    """How far the execution-feature evidence actually reaches down to lines."""
    needed = ["lines_mapped_rate", "lines_with_importance_rate"]
    if not all(c in diagnostics.columns for c in needed):
        return
    frame = diagnostics.dropna(subset=needed).copy()
    if frame.empty:
        return
    frame = frame.sort_values("lines_mapped_rate", ascending=False).reset_index(drop=True)
    x = np.arange(1, len(frame) + 1)

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 1.95))
    ax.fill_between(x, frame["lines_with_importance_rate"], frame["lines_mapped_rate"],
                    color=BLUE_LIGHT, alpha=0.45, linewidth=0, zorder=1)
    ax.plot(x, frame["lines_mapped_rate"], color=SERIES[0], zorder=3,
            label="reached by the feature-to-line mapping")
    ax.plot(x, frame["lines_with_importance_rate"], color=SERIES[1], zorder=3,
            label="reached by a non-zero importance")
    ax.set_xlim(1, len(frame))
    ax.set_ylim(0, 1)
    ax.set_xlabel("subjects, ordered by mapping reach")
    ax.set_ylabel("share of executable lines")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _clean(ax)
    median_gap = float((frame["lines_mapped_rate"] - frame["lines_with_importance_rate"]).median())
    ax.annotate(f"median gap {median_gap:.0%}",
                xy=(len(frame) * 0.55, 0.5), color=INK_2, fontsize=7.5, ha="center")
    ax.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 1.0),
              handlelength=1.5, columnspacing=1.4, borderaxespad=0.1)
    _save(fig, path)


def figure_controls(outcomes: pd.DataFrame, path: Path) -> None:
    """Each weighting condition against the unmodified formula."""
    metrics = ["patch_top1", "patch_top5", "patch_top10"]
    order = [m for m in ("rf", "neg", "perm", "pos") if m in set(outcomes["method"])]
    if not order:
        return
    baseline = outcomes[outcomes.method == "baseline"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(COLUMN_IN, 2.05), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        reference = float(baseline[metric].mean())
        values = [float(outcomes[outcomes.method == m][metric].mean()) for m in order]
        upper = max(max(values), reference) * 1.42
        for index, (method, value) in enumerate(zip(order, values)):
            color = SERIES[0] if method == "rf" else INK_MUTED
            _rounded_barh(ax, len(order) - 1 - index, value, 0.56, color, zorder=2)
            # Values sit at a fixed right edge so they never collide with the
            # baseline reference line.
            ax.text(upper, len(order) - 1 - index, f"{value:.2f}", ha="right",
                    va="center", fontsize=7, color=INK_2)
        ax.axvline(reference, color=DIVERGE_NEG, linewidth=1.0, linestyle=(0, (3, 2)),
                   zorder=3)
        ax.set_xlim(0, upper)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in reversed(order)])
        ax.set_title(LABELS[metric], color=INK)
        ax.set_xticks([])
        _clean(ax, grid_axis=None)
        ax.spines["bottom"].set_visible(False)
    handle = plt.Line2D([], [], color=DIVERGE_NEG, linewidth=1.0,
                        linestyle=(0, (3, 2)), label="baseline SFL")
    fig.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.5, -0.06),
               handlelength=1.6, borderaxespad=0.0)
    _save(fig, path)


def figure_forest(effects: pd.DataFrame, path: Path, method: str = "rf") -> None:
    """Paired standardized method effect d_z per baseline formula, oriented so
    that positive always means the augmented method is better. The interval is
    the bootstrap interval of the mean paired difference, divided by the
    standard deviation of the paired differences."""
    subset = effects[effects.method == method]
    if subset.empty:
        return
    metrics = [m for m, _ in PRIMARY_METRICS]
    directions = dict(PRIMARY_METRICS)
    fig, axes = plt.subplots(1, len(metrics), figsize=(COLUMN_IN, 2.1), sharey=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        sign = 1.0 if directions[metric] == HIGHER else -1.0
        ax.axvspan(-0.2, 0.2, color="#eeede9", linewidth=0, zorder=0)
        ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
        ax.tick_params(axis="y", length=0)
        for index, formula in enumerate(FORMULAS):
            row = subset[(subset.formula == formula) & (subset.outcome == metric)]
            if row.empty:
                continue
            row = row.iloc[0]
            spread = float(row.get("paired_sd", 0.0) or 0.0)
            d = sign * float(row["d_paired"])
            low = sign * (row["perm_ci_low"] / spread) if spread > 0 else d
            high = sign * (row["perm_ci_high"] / spread) if spread > 0 else d
            y = len(FORMULAS) - 1 - index
            ax.plot([min(low, high), max(low, high)], [y, y], color=SERIES[0],
                    linewidth=1.4, solid_capstyle="round", zorder=2)
            ax.plot([d], [y], marker="o", markersize=4.2, color=SERIES[0],
                    markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
        ax.set_title(LABELS[metric], color=INK)
        ax.set_yticks(range(len(FORMULAS)))
        ax.set_yticklabels(list(reversed(FORMULAS)))
        ax.set_ylim(-0.6, len(FORMULAS) - 0.4)
        _clean(ax, grid_axis=None)
    spans = []
    for metric in metrics:
        rows = subset[subset.outcome == metric]
        for _, row in rows.iterrows():
            spread = float(row.get("paired_sd", 0.0) or 0.0)
            if spread > 0:
                spans += [abs(row["perm_ci_low"] / spread), abs(row["perm_ci_high"] / spread)]
        spans += list(np.abs(rows["d_paired"].to_numpy()))
    finite = [v for v in spans if np.isfinite(v)]
    limit = float(np.percentile(finite, 98)) * 1.2 if finite else 0.5
    limit = min(max(limit, 0.35), 2.0)
    tick = round(limit / 2, 1) or 0.2
    for ax in np.atleast_1d(axes):
        ax.set_xlim(-limit, limit)
        ax.set_xticks([-tick, 0.0, tick])
        ax.set_xticklabels([f"{-tick:g}", "0", f"{tick:g}"], fontsize=6.5)
    fig.supxlabel("paired effect $d_z$  (positive $=$ augmented better; "
                  "shaded band $=$ negligible)", fontsize=7.5, color=INK_2, y=-0.06)
    _save(fig, path)


def figure_project_effects(outcomes: pd.DataFrame, path: Path,
                           outcome: str = "patch_exam_avg", method: str = "rf") -> None:
    """Per-project distribution of the paired change, on a robust scale.

    A handful of very small projects produce extreme paired deltas that would
    otherwise flatten every other project onto the zero line, so the axis is
    clipped to the bulk of the distribution and clipped whiskers are marked.
    """
    base = outcomes[outcomes.method == "baseline"][["subject", "project", "formula", outcome]]
    aug = outcomes[outcomes.method == method][["subject", "formula", outcome]]
    merged = base.merge(aug, on=["subject", "formula"], suffixes=("_base", "_aug"))
    if merged.empty:
        return
    # Lower EXAM is better, so plot the improvement.
    merged["delta"] = merged[f"{outcome}_base"] - merged[f"{outcome}_aug"]
    counts = merged.groupby("project")["subject"].nunique()
    order = merged.groupby("project")["delta"].median().sort_values().index.tolist()
    data = [merged[merged.project == p]["delta"].to_numpy() for p in order]

    pooled = np.abs(merged["delta"].to_numpy())
    limit = float(np.percentile(pooled, 90)) if pooled.size else 1.0
    limit = max(limit, 1e-6) * 1.5

    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.15))
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    parts = ax.boxplot(data, tick_labels=[f"{p} ({counts[p]})" for p in order],
                       showfliers=False, patch_artist=True, widths=0.6, zorder=2,
                       medianprops=dict(color=SERIES[1], linewidth=1.3),
                       whiskerprops=dict(color=SERIES[0], linewidth=0.9),
                       capprops=dict(color=SERIES[0], linewidth=0.9))
    for patch in parts["boxes"]:
        patch.set_facecolor(BLUE_LIGHT)
        patch.set_edgecolor(SERIES[0])
        patch.set_linewidth(0.9)
    # Mark the projects whose spread runs past the clipped axis.
    for index, values in enumerate(data, start=1):
        if values.size and (values.max() > limit or values.min() < -limit):
            ax.plot([index], [limit * 0.93], marker="^", markersize=3.6,
                    color=SERIES[1], markeredgewidth=0, zorder=4)
    ax.set_ylim(-limit, limit)
    ax.set_ylabel("EXAM improvement")
    ax.tick_params(axis="x", labelsize=6.5, rotation=55)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    _clean(ax)
    _save(fig, path)


def figure_density_confound(categories: pd.DataFrame, path: Path,
                            wrap: bool = False) -> None:
    """Why the registered negative control is not a pure noise control."""
    columns = [c for c in ("density_rho_neg", "density_rho_perm", "density_rho_rf")
               if c in categories.columns]
    if not columns:
        return
    labels = {"density_rho_neg": "negative control",
              "density_rho_perm": "permutation control",
              "density_rho_rf": "RF weighting"}
    width = WRAP_IN if wrap else COLUMN_IN
    fig, ax = plt.subplots(figsize=(width, 1.85 if wrap else 1.95))
    jitter = 0.05 if wrap else 0.055
    for index, column in enumerate(columns):
        values = categories[column].dropna().to_numpy()
        y = len(columns) - 1 - index
        ax.scatter(values,
                   np.full(values.shape, y)
                   + np.random.default_rng(4).normal(0, jitter, values.size),
                   s=7 if wrap else 9, color=SERIES[index], alpha=0.55,
                   linewidths=0, zorder=2)
        median = float(np.median(values))
        ax.plot([median, median], [y - 0.28, y + 0.28], color=INK, linewidth=1.4, zorder=3)
        ax.text(median, y + 0.33, f"{median:.2f}" if wrap
                else f"median $\\rho$ = {median:.2f}",
                ha="center", fontsize=6.6 if wrap else 7, color=INK_2)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels([labels[c] for c in reversed(columns)])
    ax.set_ylim(-0.6, len(columns) - 0.05)
    ax.set_xlim(-0.2, 1.05)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Spearman $\\rho$ (weight vs. mapped features)" if wrap
                  else "Spearman $\\rho$ between a line's weight and the number "
                       "of features mapped to it",
                  fontsize=7 if wrap else 8)
    if wrap:
        ax.tick_params(axis="both", labelsize=6.6)
    _clean(ax, grid_axis="x")
    _save(fig, path)


def figure_sensitivity(rows: List[Dict], path: Path) -> None:
    lam = [r for r in rows if r["knob"].endswith("lambda$")]
    if not lam:
        lam = [r for r in rows if "lambda" in r["knob"]]
    if not lam:
        return
    metrics = [m for m, _ in PRIMARY_METRICS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(COLUMN_IN, 1.85))
    fig.subplots_adjust(wspace=0.75)
    values = sorted({r["value"] for r in lam})
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        y = [next(r[metric] for r in lam if r["value"] == v) for v in values]
        reference = float(np.mean([r[f"baseline_{metric}"] for r in lam]))
        ax.axhline(reference, color=DIVERGE_NEG, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.plot(values, y, color=SERIES[0], marker="o", markersize=3.4,
                markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
        registered = 0.35
        if registered in values:
            index = values.index(registered)
            ax.plot([registered], [y[index]], marker="o", markersize=5.6,
                    color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
        ax.set_title(LABELS[metric], color=INK)
        ends = [values[0], values[len(values) // 2], values[-1]]
        ax.set_xticks(ends)
        ax.set_xticklabels([f"{v:g}" for v in ends], fontsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.margins(x=0.12)
        _clean(ax)
    fig.supxlabel("blend parameter $\\lambda$  (orange $=$ registered value; "
                  "dashed $=$ baseline SFL)", fontsize=7.5, color=INK_2, y=-0.08)
    _save(fig, path)


def figure_families(categories: pd.DataFrame, subjects: pd.DataFrame, path: Path) -> None:
    families = [("share_coverage", "coverage"), ("share_data_flow", "data flow"),
                ("share_value", "value"), ("share_condition", "condition"),
                ("share_exception", "exception")]
    families = [(c, l) for c, l in families if c in categories.columns]
    if not families:
        return
    frame = categories.merge(subjects[["subject", "project"]], on="subject", how="left")
    grouped = frame.groupby("project")[[c for c, _ in families]].mean()
    grouped = grouped.div(grouped.sum(axis=1), axis=0)
    grouped = grouped.sort_values(families[2][0] if len(families) > 2 else families[0][0])

    fig, ax = plt.subplots(figsize=(COLUMN_IN, min(2.9, max(1.6, 0.155 * len(grouped) + 0.85))))
    left = np.zeros(len(grouped))
    for index, (column, label) in enumerate(families):
        widths = grouped[column].to_numpy()
        ax.barh(np.arange(len(grouped)), widths, left=left, height=0.66,
                color=SERIES[index], label=label, linewidth=0.8, edgecolor=SURFACE,
                zorder=2)
        left = left + widths
    ax.set_yticks(np.arange(len(grouped)))
    ax.set_yticklabels(grouped.index, fontsize=7)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("share of the total random-forest importance mass")
    _clean(ax, grid_axis="x")
    ax.legend(ncols=len(families), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.1, columnspacing=1.1, borderaxespad=0.1)
    _save(fig, path)


def render_figures(study_dir: Path, out_dir: Path, tag: Optional[str] = None) -> None:
    _style()
    suffix = "" if tag is None else f"_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(name):
        path = study_dir / f"{name}{suffix}.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    outcomes, subjects = read("outcomes"), read("subjects")
    diagnostics, categories, effects = read("diagnostics"), read("categories"), read("effects")
    if not diagnostics.empty and not categories.empty:
        reach = [c for c in categories.columns if c.startswith("lines_")]
        diagnostics = diagnostics.merge(
            categories[["subject"] + reach], on="subject", how="left"
        )

    built = []
    if not diagnostics.empty:
        figure_bottleneck(diagnostics, out_dir / "fig_bottleneck.pdf")
        built.append("bottleneck")
    if not outcomes.empty:
        figure_controls(outcomes, out_dir / "fig_controls.pdf")
        figure_project_effects(outcomes, out_dir / "fig_project_effects.pdf")
        built += ["controls", "project_effects"]
    if not effects.empty:
        figure_forest(effects, out_dir / "fig_forest.pdf")
        built.append("forest")
    if not categories.empty:
        figure_density_confound(categories, out_dir / "fig_density_confound.pdf")
        figure_density_confound(categories, out_dir / "fig_density_wrap.pdf", wrap=True)
        built.append("density_confound")
        if not subjects.empty:
            figure_families(categories, subjects, out_dir / "fig_families.pdf")
            built.append("families")
    sensitivity = study_dir / f"sensitivity{suffix}.json"
    if sensitivity.exists():
        figure_sensitivity(json.loads(sensitivity.read_text()),
                           out_dir / "fig_sensitivity.pdf")
        built.append("sensitivity")
    print(f"figures: {', '.join(built)}")
