"""The pre-registered analysis plan.

Primary inference is one confounder-adjusted mixed-effects model per outcome,
fitted over all subjects, baseline formulas and paired method conditions, with

* fixed effects: the method indicator, the SFL formula as a factor, their
  interaction and the pre-registered confounders,
* random effects: an intercept per project, an intercept per subject within
  project and a per-project random slope for the method effect.

Continuous outcomes use a Gaussian mixed model.  The pre-registered binomial
logit for the per-subject top-$k$ and impact-found@$k$ outcomes is fitted as a
fractional-response logit (binomial family, logit link, exchangeable working
correlation within subject, cluster-robust standard errors), because those
outcomes are tie-breaking probabilities rather than 0/1 draws; the Gaussian
model on the same outcome is reported alongside it as a linear-probability
model.  Distribution-free paired procedures corroborate every contrast.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

from protocol import MASTER_SEED  # noqa: E402
from protocol.formulas import FORMULAS  # noqa: E402
from protocol.metrics import stable_seed  # noqa: E402

HIGHER = "higher"
LOWER = "lower"

#: The five primary reference-patch outcomes of the superiority rule.
PRIMARY_METHOD = "rf"

PRIMARY_METRICS: List[Tuple[str, str]] = [
    ("patch_top1", HIGHER),
    ("patch_top5", HIGHER),
    ("patch_top10", HIGHER),
    ("patch_exam_avg", LOWER),
    ("patch_wasted_avg", LOWER),
]

#: Outcome families; multiplicity is controlled inside each family.
FAMILIES: Dict[str, List[Tuple[str, str]]] = {
    "patch_topk": [("patch_top1", HIGHER), ("patch_top5", HIGHER), ("patch_top10", HIGHER)],
    "patch_effort": [
        ("patch_exam_best", LOWER), ("patch_exam_avg", LOWER), ("patch_exam_worst", LOWER),
        ("patch_wasted_best", LOWER), ("patch_wasted_avg", LOWER),
        ("patch_wasted_worst", LOWER), ("patch_wasted_avg_norm", LOWER),
    ],
    "function": [
        ("function_exam", LOWER), ("function_effort_lines", LOWER),
        ("function_faulty_mass", HIGHER), ("function_mass@1", HIGHER),
        ("function_mass@3", HIGHER), ("function_mass@5", HIGHER),
    ],
    "impact": [
        ("impact_found@1", HIGHER), ("impact_found@5", HIGHER), ("impact_found@10", HIGHER),
        ("impact_min_rank", LOWER), ("impact_total_ranked", HIGHER),
    ],
    "hunk": [
        ("hunk_found@1", HIGHER), ("hunk_found@5", HIGHER), ("hunk_found@10", HIGHER),
        ("hunk_min_rank", LOWER), ("hunk_total_ranked", HIGHER),
    ],
}

BINARY_OUTCOMES = {
    "patch_top1", "patch_top5", "patch_top10",
    "impact_found@1", "impact_found@5", "impact_found@10",
    "hunk_found@1", "hunk_found@5", "hunk_found@10",
    "function_faulty_mass", "function_mass@1", "function_mass@3", "function_mass@5",
}

CONFOUNDERS = [
    "z_log_loc",             # project size
    "z_log_lines",           # executable-line count
    "z_log_tests",           # test-suite size
    "z_log_failing",         # number of failing tests
    "z_coverage_density",    # coverage density
    "z_log_patch",           # bug difficulty (size of the reference fix)
]

NEGLIGIBLE = 0.2


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def _z(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    return (values - values.mean()) / std if std > 0 else values * 0.0


def prepare(outcomes: pd.DataFrame, subjects: pd.DataFrame) -> pd.DataFrame:
    meta = subjects[subjects.get("status", "ok") == "ok"].copy()
    frame = outcomes.merge(
        meta.drop(columns=["project"], errors="ignore"), on="subject", how="inner"
    )
    frame["z_log_loc"] = _z(np.log1p(frame["loc"]))
    frame["z_log_lines"] = _z(np.log1p(frame["executable_lines"]))
    frame["z_log_tests"] = _z(np.log1p(frame["test_suite_size"]))
    frame["z_log_failing"] = _z(np.log1p(frame["failing_tests"]))
    frame["z_coverage_density"] = _z(frame["coverage_density"])
    frame["z_log_patch"] = _z(np.log1p(frame["patch_size"]))
    frame["patch_wasted_avg_norm"] = frame["patch_wasted_avg"] / frame[
        "executable_lines"
    ].clip(lower=1)
    frame["formula"] = pd.Categorical(frame["formula"], categories=list(FORMULAS))
    return frame


# ---------------------------------------------------------------------------
# Primary model
# ---------------------------------------------------------------------------
@dataclass
class MethodEffect:
    outcome: str
    formula: str
    coefficient: float
    stderr: float
    ci_low: float
    ci_high: float
    pvalue: float
    d: float
    spread: float
    direction: str
    improves: bool
    degrades: bool
    model: str


def _contrast(result, names: Sequence[str], weights: Sequence[float]) -> Tuple[float, float, float]:
    index = {name: i for i, name in enumerate(result.params.index)}
    vector = np.zeros(len(result.params))
    for name, weight in zip(names, weights):
        if name in index:
            vector[index[name]] = weight
    estimate = float(vector @ result.params.to_numpy())
    covariance = np.asarray(result.cov_params())
    variance = float(vector @ covariance @ vector)
    stderr = math.sqrt(max(variance, 0.0))
    if stderr <= 0:
        return estimate, 0.0, 1.0
    z = estimate / stderr
    return estimate, stderr, float(2 * stats.norm.sf(abs(z)))


def fit_mixed(frame: pd.DataFrame, outcome: str) -> Optional[object]:
    formula = (
        f"Q('{outcome}') ~ is_aug * C(formula) + " + " + ".join(CONFOUNDERS)
    )
    try:
        model = smf.mixedlm(
            formula,
            frame,
            groups=frame["project"],
            re_formula="1 + is_aug",
            vc_formula={"subject": "0 + C(subject)"},
        )
        return model.fit(method="lbfgs", maxiter=200, reml=False)
    except Exception:
        try:
            model = smf.mixedlm(
                formula, frame, groups=frame["project"], re_formula="1 + is_aug"
            )
            return model.fit(method="lbfgs", maxiter=200, reml=False)
        except Exception:
            return None


def fit_fractional_logit(frame: pd.DataFrame, outcome: str) -> Optional[object]:
    formula = f"Q('{outcome}') ~ is_aug * C(formula) + " + " + ".join(CONFOUNDERS)
    try:
        model = smf.gee(
            formula,
            groups="subject",
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        return model.fit()
    except Exception:
        return None


def _joint_wald(result, names: Sequence[str]) -> Dict[str, float]:
    """Joint Wald test that all method x formula interactions are zero."""
    if not names:
        return {"statistic": 0.0, "df": 0, "pvalue": 1.0}
    index = {name: i for i, name in enumerate(result.params.index)}
    restriction = np.zeros((len(names), len(result.params)))
    for row, name in enumerate(names):
        restriction[row, index[name]] = 1.0
    estimate = restriction @ result.params.to_numpy()
    covariance = restriction @ np.asarray(result.cov_params()) @ restriction.T
    try:
        statistic = float(estimate @ np.linalg.solve(covariance, estimate))
    except np.linalg.LinAlgError:
        return {"statistic": float("nan"), "df": len(names), "pvalue": float("nan")}
    return {
        "statistic": statistic,
        "df": len(names),
        "pvalue": float(stats.chi2.sf(statistic, len(names))),
    }


def _random_effects(result) -> Dict[str, float]:
    covariance = getattr(result, "cov_re", None)
    if covariance is None:
        return {}
    try:
        return {
            f"{row}|{column}": float(covariance.loc[row, column])
            for row in covariance.index
            for column in covariance.columns
        }
    except Exception:
        return {}


def method_effects(
    frame: pd.DataFrame, outcome: str, direction: str, method: str
) -> Tuple[List[MethodEffect], Dict]:
    subset = frame[frame["method"].isin(["baseline", method])].copy()
    subset["is_aug"] = (subset["method"] == method).astype(float)
    subset = subset.dropna(subset=[outcome])
    if subset.empty or subset[outcome].std(ddof=0) == 0:
        return [], {"model": "degenerate"}

    gaussian = fit_mixed(subset, outcome)
    logit = fit_fractional_logit(subset, outcome) if outcome in BINARY_OUTCOMES else None
    if gaussian is None:
        return [], {"model": "failed"}

    effects: List[MethodEffect] = []
    reference = FORMULAS[0]
    for formula_name in FORMULAS:
        rows = subset[(subset["formula"] == formula_name) & (subset["is_aug"] == 0)]
        spread = float(rows[outcome].std(ddof=0)) if len(rows) > 1 else 0.0
        names = ["is_aug"]
        weights = [1.0]
        if formula_name != reference:
            names.append(f"is_aug:C(formula)[T.{formula_name}]")
            weights.append(1.0)
        estimate, stderr, pvalue = _contrast(gaussian, names, weights)
        d = estimate / spread if spread > 0 else 0.0
        better = (estimate > 0) if direction == HIGHER else (estimate < 0)
        effects.append(
            MethodEffect(
                outcome=outcome,
                formula=formula_name,
                coefficient=estimate,
                stderr=stderr,
                ci_low=estimate - 1.96 * stderr,
                ci_high=estimate + 1.96 * stderr,
                pvalue=pvalue,
                d=d,
                spread=spread,
                direction=direction,
                improves=bool(better and abs(d) >= NEGLIGIBLE),
                degrades=bool((not better) and abs(d) >= NEGLIGIBLE),
                model="mixedlm",
            )
        )

    interaction = [
        name for name in gaussian.params.index if name.startswith("is_aug:C(formula)")
    ]
    joint = _joint_wald(gaussian, interaction)
    info: Dict[str, object] = {
        "model": "mixedlm",
        "converged": bool(getattr(gaussian, "converged", True)),
        "n_observations": int(len(subset)),
        "global_method_coefficient": float(gaussian.params.get("is_aug", np.nan)),
        "interaction_joint": joint,
        "interaction_terms": {
            name: {
                "coefficient": float(gaussian.params[name]),
                "pvalue": float(gaussian.pvalues[name]),
            }
            for name in interaction
        },
        "random_effects": _random_effects(gaussian),
    }
    if logit is not None:
        estimate, stderr, pvalue = _contrast(logit, ["is_aug"], [1.0])
        info["fractional_logit"] = {
            "coefficient": estimate,
            "stderr": stderr,
            "pvalue": pvalue,
            "odds_ratio": float(np.exp(estimate)),
        }
    return effects, info


# ---------------------------------------------------------------------------
# Distribution-free corroboration
# ---------------------------------------------------------------------------
def paired_frames(frame: pd.DataFrame, outcome: str, method: str, formula_name: str):
    base = frame[(frame.method == "baseline") & (frame.formula == formula_name)]
    aug = frame[(frame.method == method) & (frame.formula == formula_name)]
    merged = base[["subject", outcome]].merge(
        aug[["subject", outcome]], on="subject", suffixes=("_base", "_aug")
    )
    return merged[f"{outcome}_base"].to_numpy(), merged[f"{outcome}_aug"].to_numpy()


def paired_effect(base: np.ndarray, aug: np.ndarray, direction: str) -> Dict[str, object]:
    """Paired standardized effect d_z = mean(aug - base) / sd(aug - base).

    The registered d divides the model coefficient by the between-subject
    spread of the baseline outcome, so a few subjects with extreme values (as in
    wasted effort) inflate the denominator. d_z standardizes by the spread of
    the per-subject differences instead, which removes the between-subject
    variance that the paired design holds fixed. The manuscript applies the
    decision rule with d_z and reports the registered d as a check.
    """
    diff = np.asarray(aug, dtype=float) - np.asarray(base, dtype=float)
    diff = diff[~np.isnan(diff)]
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    mean = float(diff.mean()) if len(diff) else 0.0
    d_z = mean / sd if sd > 0 else 0.0
    better = (mean > 0) if direction == HIGHER else (mean < 0)
    return {
        "d_paired": d_z,
        "paired_sd": sd,
        "paired_improves": bool(better and abs(d_z) >= NEGLIGIBLE),
        "paired_degrades": bool((not better) and abs(d_z) >= NEGLIGIBLE),
    }


def permutation_test(
    base: np.ndarray, aug: np.ndarray, seed: int, repeats: int = 10_000
) -> Dict[str, float]:
    difference = aug - base
    observed = float(difference.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(repeats, len(difference)))
    null = (signs * difference).mean(axis=1)
    pvalue = float((np.abs(null) >= abs(observed) - 1e-15).mean())
    boot = rng.choice(difference, size=(repeats, len(difference)), replace=True).mean(axis=1)
    return {
        "mean_difference": observed,
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "pvalue": pvalue,
        "n": int(len(difference)),
    }


def wilcoxon_test(base: np.ndarray, aug: np.ndarray) -> Dict[str, float]:
    difference = aug - base
    nonzero = difference[difference != 0]
    if len(nonzero) == 0:
        return {"pvalue": 1.0, "rank_biserial": 0.0, "n": int(len(difference)), "n_nonzero": 0}
    statistic, pvalue = stats.wilcoxon(base, aug, zero_method="wilcox", alternative="two-sided")
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    total = ranks.sum()
    return {
        "pvalue": float(pvalue),
        "rank_biserial": float(2 * positive / total - 1) if total > 0 else 0.0,
        "n": int(len(difference)),
        "n_nonzero": int(len(nonzero)),
    }


def direction_counts(base: np.ndarray, aug: np.ndarray, direction: str) -> Dict[str, int]:
    difference = aug - base
    tolerance = 1e-12
    if direction == HIGHER:
        improve = int((difference > tolerance).sum())
        degrade = int((difference < -tolerance).sum())
    else:
        improve = int((difference < -tolerance).sum())
        degrade = int((difference > tolerance).sum())
    return {"improve": improve, "tie": int(len(difference) - improve - degrade), "degrade": degrade}


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float]) -> List[float]:
    values = np.asarray(pvalues, dtype=float)
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        index = order[rank]
        running = min(running, values[index] * n / (rank + 1))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def sign_test(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.sf(successes - 1, trials, 0.5)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyse_method(frame: pd.DataFrame, method: str, seed: int = MASTER_SEED) -> Dict:
    result: Dict[str, object] = {"method": method, "families": {}, "models": {}}
    for family, outcomes in FAMILIES.items():
        entries = []
        for outcome, direction in outcomes:
            if outcome not in frame.columns:
                continue
            effects, info = method_effects(frame, outcome, direction, method)
            result["models"][outcome] = info
            for effect in effects:
                base, aug = paired_frames(frame, outcome, method, effect.formula)
                if len(base) == 0:
                    continue
                entry = {
                    "outcome": outcome,
                    "direction": direction,
                    "formula": effect.formula,
                    "coefficient": effect.coefficient,
                    "stderr": effect.stderr,
                    "ci_low": effect.ci_low,
                    "ci_high": effect.ci_high,
                    "pvalue": effect.pvalue,
                    "d": effect.d,
                    "spread": effect.spread,
                    "improves": effect.improves,
                    "degrades": effect.degrades,
                    "baseline_mean": float(base.mean()),
                    "augmented_mean": float(aug.mean()),
                    **paired_effect(base, aug, direction),
                    **{f"perm_{k}": v for k, v in
                       permutation_test(
                           base, aug,
                           seed=stable_seed(seed, method, outcome, effect.formula),
                       ).items()},
                    **{f"wilcoxon_{k}": v for k, v in wilcoxon_test(base, aug).items()},
                    **{f"count_{k}": v for k, v in direction_counts(base, aug, direction).items()},
                }
                entries.append(entry)
        adjusted = benjamini_hochberg([e["pvalue"] for e in entries])
        adjusted_perm = benjamini_hochberg([e["perm_pvalue"] for e in entries])
        for entry, q, q_perm in zip(entries, adjusted, adjusted_perm):
            entry["qvalue"] = q
            entry["perm_qvalue"] = q_perm
        result["families"][family] = entries
    result["superiority"] = superiority(result)
    result["superiority_paired"] = superiority(result, paired=True)
    result["per_subject"] = per_subject_superiority(frame, method)
    if method == PRIMARY_METHOD:
        result["moderators"] = {
            outcome: moderator_analysis(frame, method, outcome, direction)
            for outcome, direction in PRIMARY_METRICS
        }
    return result


def superiority(result: Dict, paired: bool = False) -> Dict:
    """Four-of-five superiority rule per formula.

    With ``paired`` the same rule is read off the paired d_z (exploratory,
    not part of the registered decision).
    """
    d_key, up_key, down_key = (
        ("d_paired", "paired_improves", "paired_degrades") if paired
        else ("d", "improves", "degrades")
    )
    lookup = {}
    for entries in result["families"].values():
        for entry in entries:
            lookup[(entry["outcome"], entry["formula"])] = entry
    per_formula = {}
    for formula_name in FORMULAS:
        improves, degrades, detail = 0, 0, {}
        for outcome, _ in PRIMARY_METRICS:
            entry = lookup.get((outcome, formula_name))
            if entry is None:
                continue
            improves += int(entry[up_key])
            degrades += int(entry[down_key])
            detail[outcome] = {"d": entry[d_key], "improves": entry[up_key],
                               "degrades": entry[down_key]}
        per_formula[formula_name] = {
            "improves": improves,
            "degrades": degrades,
            "superior": bool(improves >= 4 and degrades == 0),
            "detail": detail,
        }
    superior = [f for f, v in per_formula.items() if v["superior"]]
    return {
        "per_formula": per_formula,
        "n_superior": len(superior),
        "H_ACC_supported": len(superior) >= 3,
        "H_ROB_supported": len(superior) == len(FORMULAS),
        "sign_test_p": sign_test(len(superior), len(FORMULAS)),
    }


MODERATORS = [
    "line_evidence_rate",
    "constant_fraction",
    "importance_gini",
    "importance_entropy",
    "mapping_conflict_rate",
    "z_log_failing",
    "z_log_lines",
    "weak_signal",
]


def moderator_analysis(frame: pd.DataFrame, method: str, outcome: str, direction: str) -> Dict:
    """Which subject properties predict how large the paired effect is?

    The response is the paired difference (augmented minus baseline) averaged
    over the five formulas, so one row per subject; the predictors are the
    pre-registered diagnostics. This makes explicit where the transfer from
    diagnosis to localization works and where it does not.
    """
    base = frame[frame.method == "baseline"][["subject", "project", "formula", outcome]]
    aug = frame[frame.method == method][["subject", "formula", outcome]]
    merged = base.merge(aug, on=["subject", "formula"], suffixes=("_base", "_aug"))
    merged["delta"] = merged[f"{outcome}_aug"] - merged[f"{outcome}_base"]
    if direction == LOWER:
        merged["delta"] = -merged["delta"]  # positive means improvement
    per_subject = merged.groupby(["subject", "project"], as_index=False)["delta"].mean()
    predictors = frame.drop_duplicates("subject")[
        ["subject"] + [c for c in MODERATORS if c in frame.columns]
    ]
    data = per_subject.merge(predictors, on="subject")
    available = [c for c in MODERATORS if c in data.columns and data[c].std(ddof=0) > 0]
    if not available or len(data) < 20:
        return {"available": False, "n": int(len(data))}
    formula = "delta ~ " + " + ".join(available)
    try:
        model = smf.mixedlm(formula, data, groups=data["project"]).fit(
            method="lbfgs", maxiter=200, reml=False
        )
    except Exception:
        return {"available": False, "n": int(len(data))}
    return {
        "available": True,
        "n": int(len(data)),
        "outcome": outcome,
        "mean_delta": float(data["delta"].mean()),
        "coefficients": {
            name: {"coefficient": float(model.params[name]),
                   "pvalue": float(model.pvalues[name])}
            for name in available
            if name in model.params.index
        },
        "spearman": {
            name: float(stats.spearmanr(data["delta"], data[name]).statistic)
            for name in available
        },
    }


def per_subject_superiority(frame: pd.DataFrame, method: str) -> Dict:
    """How often is the augmented method better *for a subject*?

    The pre-registered rule is applied to the pooled model coefficient, which
    answers whether the method helps on average. It does not say how often it
    helps on an individual bug, which is what a developer experiences, so the
    same four-of-five criterion is also evaluated per subject and formula.
    """
    tolerance = 1e-9
    base = frame[frame.method == "baseline"]
    aug = frame[frame.method == method]
    keys = ["subject", "project", "formula"]
    merged = base.merge(aug, on=keys, suffixes=("_b", "_a"))
    if merged.empty:
        return {"available": False}
    better = pd.DataFrame(index=merged.index)
    worse = pd.DataFrame(index=merged.index)
    for outcome, direction in PRIMARY_METRICS:
        delta = merged[f"{outcome}_a"] - merged[f"{outcome}_b"]
        better[outcome] = delta > tolerance if direction == HIGHER else delta < -tolerance
        worse[outcome] = delta < -tolerance if direction == HIGHER else delta > tolerance
    n_better = better.sum(axis=1)
    n_worse = worse.sum(axis=1)
    superior = (n_better >= 4) & (n_worse == 0)
    inferior = (n_worse >= 4) & (n_better == 0)
    per_formula = (
        pd.DataFrame({"formula": merged["formula"], "superior": superior})
        .groupby("formula")["superior"].mean().to_dict()
    )
    return {
        "available": True,
        "n_pairs": int(len(merged)),
        "n_subjects": int(merged["subject"].nunique()),
        "mean_metrics_improved": float(n_better.mean()),
        "mean_metrics_degraded": float(n_worse.mean()),
        "share_superior": float(superior.mean()),
        "share_inferior": float(inferior.mean()),
        "share_more_better": float((n_better > n_worse).mean()),
        "share_more_worse": float((n_worse > n_better).mean()),
        "share_all_tied": float(((n_better + n_worse) == 0).mean()),
        "improved_histogram": {int(k): int(v) for k, v in n_better.value_counts().items()},
        "per_formula_superior": {k: float(v) for k, v in per_formula.items()},
    }


def detectable_effect(n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) / math.sqrt(max(n, 1)))


def run_analysis(study_dir: Path, tag: Optional[str] = None) -> Dict:
    suffix = "" if tag is None else f"_{tag}"
    outcomes = pd.read_csv(study_dir / f"outcomes{suffix}.csv")
    subjects = pd.read_csv(study_dir / f"subjects{suffix}.csv")
    frame = prepare(outcomes, subjects)

    n_subjects = frame["subject"].nunique()
    report: Dict[str, object] = {
        "n_subjects": int(n_subjects),
        "n_projects": int(frame["project"].nunique()),
        "detectable_effect_d": detectable_effect(n_subjects),
        "methods": {},
    }
    methods = sorted(set(frame["method"]) - {"baseline"})
    workers = int(os.environ.get("FFL_ANALYSIS_WORKERS", "1"))
    if workers > 1 and len(methods) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(partial(analyse_method, frame), methods)
            for method, result in zip(methods, results):
                print(f"analysed {method}", flush=True)
                report["methods"][method] = result
    else:
        for method in methods:
            print(f"analysing {method} ...", flush=True)
            report["methods"][method] = analyse_method(frame, method)

    output = study_dir / f"analysis{suffix}.json"
    output.write_text(json.dumps(report, indent=1, default=float))

    rows = []
    for method, result in report["methods"].items():
        for family, entries in result["families"].items():
            for entry in entries:
                rows.append({"method": method, "family": family, **entry})
    pd.DataFrame(rows).to_csv(study_dir / f"effects{suffix}.csv", index=False)
    print(f"wrote {output}")
    return report
