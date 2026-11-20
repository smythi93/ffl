"""Software-level controls for the analysis code.

These are not study results: they check that the pre-registered decision rule
behaves as specified on data with a *known* effect, so that a null finding in
the study cannot be an artifact of the analysis implementation.

Run with ``PYTHONPATH=src pytest tests/test_analysis_controls.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from protocol.analysis import run_analysis  # noqa: E402
from protocol.formulas import FORMULAS  # noqa: E402

N_SUBJECTS = 310
N_PROJECTS = 20


def _synthesize(tmp: Path, effect: float, seed: int = 7) -> Path:
    """A full-scale synthetic study in which the augmented method is better
    by ``effect`` (in top-$k$ probability, and by ``0.12 * effect`` in EXAM)."""
    rng = np.random.default_rng(seed)
    projects = [f"proj{i:02d}" for i in range(N_PROJECTS)]
    outcomes, subjects = [], []
    for index in range(N_SUBJECTS):
        project = projects[index % N_PROJECTS]
        subject = f"{project}_{index}"
        lines = int(rng.integers(50, 5000))
        subjects.append(
            dict(
                subject=subject, project=project, loc=lines * 3, executable_lines=lines,
                test_suite_size=int(rng.integers(5, 300)),
                failing_tests=int(rng.integers(1, 8)),
                passing_tests=int(rng.integers(4, 300)),
                coverage_density=float(rng.random()),
                patch_size=int(rng.integers(1, 10)), hunk_size=5, n_functions=50,
                constant_fraction=float(rng.random()), n_features_total=1000,
                n_features_used=500, mapping_conflict_rate=0.1,
                line_evidence_rate=float(rng.random()), importance_gini=0.5,
                importance_entropy=0.5, importance_skewness=0.0, weak_signal=0.0,
                runtime_total=100.0, status="ok",
            )
        )
        quality = rng.random()
        for formula in FORMULAS:
            for method in ("baseline", "rf"):
                gain = effect if method == "rf" else 0.0
                row = dict(subject=subject, project=project, bug_id=index,
                           formula=formula, method=method)
                for key, scale in (("patch_top1", 0.5), ("patch_top5", 0.8),
                                   ("patch_top10", 0.9)):
                    row[key] = float(rng.random() < np.clip(quality * scale + gain, 0, 1))
                exam = float(np.clip(0.30 - 0.48 * gain + rng.normal(0, 0.05), 0, 1))
                for key in ("patch_exam_best", "patch_exam_avg", "patch_exam_worst"):
                    row[key] = exam
                for key in ("patch_wasted_best", "patch_wasted_avg", "patch_wasted_worst"):
                    row[key] = exam * lines
                row["patch_min_rank"] = exam * lines + 1
                for prefix in ("hunk", "impact"):
                    for k in (1, 5, 10):
                        row[f"{prefix}_found@{k}"] = float(rng.random() < quality)
                        row[f"{prefix}_recall@{k}"] = quality
                    row[f"{prefix}_min_rank"] = float(rng.gamma(2, 20))
                    row[f"{prefix}_total_ranked"] = 10.0
                row["function_faulty_mass"] = quality * 0.3
                row["functions_ranked"] = 50.0
                row["faulty_functions_ranked"] = 1.0
                for k in (1, 3, 5):
                    row[f"function_mass@{k}"] = quality * 0.1 * k
                row["function_exam"] = exam
                row["function_effort_lines"] = 0.4
                outcomes.append(row)
    tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outcomes).to_csv(tmp / "outcomes.csv", index=False)
    pd.DataFrame(subjects).to_csv(tmp / "subjects.csv", index=False)
    return tmp


def test_decision_rule_fires_on_a_strong_effect(tmp_path):
    report = run_analysis(_synthesize(tmp_path / "strong", effect=0.25))
    verdict = report["methods"]["rf"]["superiority"]
    assert verdict["n_superior"] == len(FORMULAS)
    assert verdict["H_ACC_supported"]
    assert verdict["H_ROB_supported"]


def test_decision_rule_stays_silent_without_an_effect(tmp_path):
    report = run_analysis(_synthesize(tmp_path / "null", effect=0.0))
    verdict = report["methods"]["rf"]["superiority"]
    assert verdict["n_superior"] == 0
    assert not verdict["H_ACC_supported"]
    assert not verdict["H_ROB_supported"]


def test_power_matches_the_registered_design(tmp_path):
    report = run_analysis(_synthesize(tmp_path / "power", effect=0.0))
    assert 0.15 <= report["detectable_effect_d"] <= 0.17


def test_paired_effect_ignores_between_subject_spread():
    """d_z standardizes by the spread of the paired differences, so a subject
    with an extreme level but the same improvement does not dilute it."""
    from protocol.analysis import paired_effect

    base = np.array([10.0, 20.0, 30.0, 40.0, 5000.0])
    aug = base - np.array([2.0, 3.0, 2.0, 3.0, 2.5])
    effect = paired_effect(base, aug, "lower")
    assert effect["d_paired"] < -0.2 * 10
    assert effect["paired_improves"] and not effect["paired_degrades"]

    unchanged = paired_effect(base, base.copy(), "lower")
    assert unchanged["d_paired"] == 0.0
    assert not unchanged["paired_improves"] and not unchanged["paired_degrades"]


def test_paired_superiority_is_reported_next_to_the_registered_one(tmp_path):
    report = run_analysis(_synthesize(tmp_path, effect=0.25))
    rf = report["methods"]["rf"]
    assert "superiority_paired" in rf
    assert rf["superiority_paired"]["n_superior"] == len(FORMULAS)
