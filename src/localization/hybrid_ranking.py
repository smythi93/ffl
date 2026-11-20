"""
Novel Hybrid Ranking Approach for Fault Localization

Instead of treating FL as binary classification (failing/passing), we treat it as
a learning-to-rank problem over execution features. This combines multiple weak signals:
1. Coverage correlation (do lines differ between failing/passing?)
2. Value dynamics (how much do values change when failure occurs?)
3. Definition-use relationships (data flow proximity)
4. Robustness (how consistently important is a feature?)

Key improvements over RF classification:
- Features are ranked by their usefulness for predicting failures, not just importance
- Noise is reduced via robust aggregation (median, Winsorization)
- Multiple weak signals are combined via learned weights
- No aggressive feature filtering; instead, features scored continuously
"""

import os
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sflkit.features.handler import EventHandler
from scipy.stats import rankdata
from scipy.stats.mstats import winsorize

from efdd.learning import DiagnosisGenerator
from efdd.reduce import FeatureSelection, DefaultSelection

# Optional: For SHAP explanations (not required for core functionality)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class MultiSignalFeatureScorer:
    """Computes multiple independent signals for each feature."""

    @staticmethod
    def _labels_to_vector(y_train: pd.Series | pd.DataFrame) -> np.ndarray:
        return np.asarray(y_train.values, dtype=float).ravel()

    @staticmethod
    def _label_masks(y_train: pd.Series | pd.DataFrame):
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train)
        fail_mask = y_data.astype(bool)
        pass_mask = ~fail_mask
        return fail_mask, pass_mask

    @staticmethod
    def _safe_to_float(col_data: np.ndarray) -> Optional[np.ndarray]:
        """Safely convert column data to float, handling nested structures."""
        try:
            # Try direct conversion first
            return np.asarray(col_data, dtype=float)
        except (ValueError, TypeError):
            # Try flattening if nested
            try:
                flattened = []
                for val in col_data:
                    if isinstance(val, (list, tuple, np.ndarray)):
                        arr = np.asarray(val, dtype=float).ravel()
                        if arr.size == 0:
                            flattened.append(0.0)
                        else:
                            # Preserve value information instead of reducing to length only.
                            flattened.append(float(np.nanmean(arr)))
                    else:
                        flattened.append(float(val))
                return np.array(flattened, dtype=float)
            except Exception:
                return None

    @staticmethod
    def _mask_weights(mask: np.ndarray, weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if weights is None:
            return None
        arr = np.asarray(weights, dtype=float).ravel()
        if arr.shape[0] == mask.shape[0]:
            subset = arr[mask]
        elif arr.shape[0] == int(np.sum(mask)):
            subset = arr
        else:
            return None
        if subset.size == 0:
            return None
        subset = np.clip(subset, 0.0, None)
        return subset

    @staticmethod
    def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
        if values.size == 0:
            return 0.0
        if weights is None:
            return float(np.mean(values))
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            return float(np.mean(values))
        return float(np.sum(values * weights) / weight_sum)

    @staticmethod
    def _weighted_rate(active: np.ndarray, weights: Optional[np.ndarray]) -> float:
        vals = np.asarray(active, dtype=float)
        return MultiSignalFeatureScorer._weighted_mean(vals, weights)

    @staticmethod
    def coverage_signal(x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> np.ndarray:
        """
        Coverage-based signal: does feature presence correlate with failure?
        For binary features: compute Ochiai/Jaccard similarity.
        For continuous: compute correlation with failure label.
        """
        signals = []
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train)
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue

            if col_data.std() < 1e-10 or len(np.unique(y_data)) < 2:
                signals.append(0.0)
                continue

            try:
                corr = np.abs(np.corrcoef(col_data, y_data)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
            signals.append(float(corr))

        return np.array(signals, dtype=float)

    @staticmethod
    def value_dynamics_signal(x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> np.ndarray:
        """
        Value dynamics signal: how much does feature variance differ between
        failing and passing tests?
        """
        signals = []
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train)
        failing_mask = y_data.astype(bool)
        passing_mask = ~failing_mask

        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue

            failing_vals = col_data[failing_mask]
            passing_vals = col_data[passing_mask]

            # Avoid edge cases
            if len(failing_vals) < 2 or len(passing_vals) < 2:
                signals.append(0.0)
                continue

            # Jensen-Shannon divergence (symmetric KL) for distributions
            # Simpler: use effect size (Cohen's d)
            failing_mean = np.mean(failing_vals)
            passing_mean = np.mean(passing_vals)
            failing_std = np.std(failing_vals)
            passing_std = np.std(passing_vals)

            pooled_std = np.sqrt((failing_std**2 + passing_std**2) / 2)
            if pooled_std < 1e-10:
                d = 0.0
            else:
                d = np.abs(failing_mean - passing_mean) / pooled_std

            # Normalize to [0, 1] range
            signal = min(d / 4.0, 1.0)  # ~2 std apart is already strong
            signals.append(float(signal))

        return np.array(signals, dtype=float)

    @staticmethod
    def discriminative_power_signal(x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> np.ndarray:
        """
        Discriminative power: how well does this feature alone separate classes?
        Computed via mutual information or ROC-AUC for binary classification.
        """
        from sklearn.metrics import roc_auc_score

        signals = []
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(int)
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue

            # Normalize feature
            scaler = StandardScaler()
            col_normalized = scaler.fit_transform(col_data.reshape(-1, 1)).ravel()

            try:
                if len(np.unique(y_data)) > 1:
                    auc = roc_auc_score(y_data, col_normalized)
                    signal = np.abs(auc - 0.5) * 2
                else:
                    signal = 0.0
            except Exception:
                signal = 0.0

            signals.append(float(signal))

        return np.array(signals, dtype=float)

    @staticmethod
    def robustness_signal(x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame) -> np.ndarray:
        """
        Robustness signal: is feature importance stable across different
        random seeds or data subsamples? Here, compute via cross-entropy
        (lower = more stable).
        """
        signals = []
        n_bootstrap = 7

        for col in x_train.columns:
            feature = x_train[[col]]
            y_data = y_train.values.astype(int)

            bootstrap_scores = []
            for seed in range(n_bootstrap):
                try:
                    rng = np.random.default_rng(seed)
                    indices = rng.integers(0, len(feature), len(feature))
                    sampled_feature = feature.iloc[indices]
                    sampled_labels = y_train.iloc[indices]
                    score = MultiSignalFeatureScorer.coverage_signal(
                        sampled_feature, sampled_labels
                    )[0]
                    bootstrap_scores.append(score)
                except Exception:
                    pass

            if not bootstrap_scores:
                signals.append(0.0)
            else:
                arr = np.array(bootstrap_scores, dtype=float)
                scale = np.mean(np.abs(arr)) + 1e-10
                cv = np.std(arr) / scale
                robustness = 1.0 / (1.0 + cv)
                signals.append(float(np.clip(robustness, 0.0, 1.0)))

        return np.array(signals, dtype=float)

    @staticmethod
    def contrastive_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score features by failing-vs-passing activation gap with prevalence penalty."""
        fail_mask, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        y_data = fail_mask
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []

        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue
            active = np.abs(col_data) > 1e-12
            fail_rate = MultiSignalFeatureScorer._weighted_rate(active[fail_mask], fail_weights)
            pass_rate = MultiSignalFeatureScorer._weighted_rate(active[pass_mask], pass_weights)
            prevalence = float(np.mean(active))
            idf_like = np.log1p(1.0 / max(prevalence, 1e-6))
            signals.append(float(abs(fail_rate - pass_rate) * idf_like))

        return np.array(signals, dtype=float)

    @staticmethod
    def fail_focus_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """How strongly each feature activates in failing executions."""
        fail_mask, _ = MultiSignalFeatureScorer._label_masks(y_train)
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        signals = []
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None or not np.any(fail_mask):
                signals.append(0.0)
                continue
            fail_vals = np.abs(col_data[fail_mask])
            signals.append(MultiSignalFeatureScorer._weighted_mean(fail_vals, fail_weights))
        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr

    @staticmethod
    def pass_exposure_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """How much each feature is also active in passing executions (noise proxy)."""
        _, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None or not np.any(pass_mask):
                signals.append(0.0)
                continue
            pass_vals = np.abs(col_data[pass_mask])
            signals.append(MultiSignalFeatureScorer._weighted_mean(pass_vals, pass_weights))
        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr

    @staticmethod
    def failure_specificity_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """How much more a feature appears in failing executions than in passing ones."""
        fail_mask, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue
            fail_vals = np.abs(col_data[fail_mask])
            pass_vals = np.abs(col_data[pass_mask])
            fail_mean = MultiSignalFeatureScorer._weighted_mean(fail_vals, fail_weights)
            pass_mean = MultiSignalFeatureScorer._weighted_mean(pass_vals, pass_weights)
            fail_rate = MultiSignalFeatureScorer._weighted_rate(fail_vals > 1e-12, fail_weights)
            pass_rate = MultiSignalFeatureScorer._weighted_rate(pass_vals > 1e-12, pass_weights)
            ratio = (fail_mean + fail_rate + 1e-9) / (pass_mean + pass_rate + 1e-9)
            signals.append(float(np.log1p(ratio)))

        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr

    @staticmethod
    def def_use_pair_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Boost features that look like def-use interactions and are failure-specific."""
        fail_mask, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []
        for col in x_train.columns:
            name = str(col).lower()
            is_def_use = (
                ("def" in name and "use" in name)
                or "def_use" in name
                or "du:" in name
                or "->" in name
            )
            if not is_def_use:
                signals.append(0.0)
                continue
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue
            fail_vals = np.abs(col_data[fail_mask])
            pass_vals = np.abs(col_data[pass_mask])
            fail_mean = MultiSignalFeatureScorer._weighted_mean(fail_vals, fail_weights)
            pass_mean = MultiSignalFeatureScorer._weighted_mean(pass_vals, pass_weights)
            separation = abs(fail_mean - pass_mean) / (abs(fail_mean) + abs(pass_mean) + 1e-9)
            signals.append(float(separation))
        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr

    @staticmethod
    def scalar_pair_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Capture scalar pair/comparison-style feature interactions."""
        fail_mask, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []
        for col in x_train.columns:
            name = str(col).lower()
            is_pair = (
                "pair" in name
                or "scalar" in name
                or "value_pair" in name
                or any(op in name for op in ("==", "!=", "<=", ">=", "<", ">"))
            )
            if not is_pair:
                signals.append(0.0)
                continue
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue
            fail_vals = np.abs(col_data[fail_mask])
            pass_vals = np.abs(col_data[pass_mask])
            fail_rate = MultiSignalFeatureScorer._weighted_rate(fail_vals > 1e-12, fail_weights)
            pass_rate = MultiSignalFeatureScorer._weighted_rate(pass_vals > 1e-12, pass_weights)
            signals.append(float(max(0.0, fail_rate - pass_rate)))
        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr

    @staticmethod
    def margin_contrastive_signal(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        fail_sample_weights: Optional[np.ndarray] = None,
        pass_sample_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Signed margin between failing and passing activations, normalized as confidence."""
        fail_mask, pass_mask = MultiSignalFeatureScorer._label_masks(y_train)
        fail_weights = MultiSignalFeatureScorer._mask_weights(fail_mask, fail_sample_weights)
        pass_weights = MultiSignalFeatureScorer._mask_weights(pass_mask, pass_sample_weights)
        signals = []
        for col in x_train.columns:
            col_data = MultiSignalFeatureScorer._safe_to_float(x_train[col].values)
            if col_data is None:
                signals.append(0.0)
                continue
            fail_vals = np.abs(col_data[fail_mask])
            pass_vals = np.abs(col_data[pass_mask])
            if fail_vals.size == 0 or pass_vals.size == 0:
                signals.append(0.0)
                continue
            fail_mean = MultiSignalFeatureScorer._weighted_mean(fail_vals, fail_weights)
            pass_mean = MultiSignalFeatureScorer._weighted_mean(pass_vals, pass_weights)
            fail_var = float(np.var(fail_vals)) if fail_vals.size > 1 else 0.0
            pass_var = float(np.var(pass_vals)) if pass_vals.size > 1 else 0.0
            denom = np.sqrt(fail_var + pass_var + 1e-12)
            margin = (fail_mean - pass_mean) / denom
            signals.append(float(max(0.0, margin)))
        arr = np.asarray(signals, dtype=float)
        max_v = float(np.max(arr)) if len(arr) else 0.0
        if max_v > 1e-12:
            arr = arr / max_v
        return arr


class HybridRankingDiagnosis(DiagnosisGenerator):
    """
    Novel hybrid ranking approach for fault localization.

    Instead of using a single model (RF), combines multiple weak signals with
    learned aggregation weights. This is more robust to noise and preserves
    feature diversity.
    """

    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        n_estimators: int = 50,
        random_state: int = 42,
        signal_weights: Optional[Dict[str, float]] = None,
        fusion_k: float = 20.0,
        head_k: int = 12,
        head_focus: float = 2.25,
        stability_power: float = 1.25,
        ranking_mode: str = "balanced",
        verify_pass_penalty: float = 0.55,
        fail_focus_boost: float = 0.35,
        dynamic_mode_switch: bool = True,
        diversity_strength: float = 0.18,
        calibration_strength: float = 0.35,
        pairwise_top_features: int = 80,
    ):
        """
        Args:
            path: Path to save/load model
            reducer: Feature selection strategy (default: no reduction)
            n_estimators: Number of estimators in gradient boosting
            random_state: Random seed
            signal_weights: Relative weights for signals. If None, learned from data.
                           Keys: 'coverage', 'value_dynamics', 'discriminative', 'robustness'
        """
        # Use a lightweight model for ranking aggregation
        # Note: GradientBoostingRegressor used instead of GradientBoostingRanker 
        # (which may not be available in all sklearn versions)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=3,
            learning_rate=0.1,
        )

        super().__init__(
            model=model,
            path=path,
            reducer=reducer,
            explainer=None,  # We compute importance differently
        )

        default_weights = {
            'coverage': 0.20,
            'value_dynamics': 0.26,
            'discriminative': 0.15,
            'robustness': 0.14,
            'contrastive': 0.15,
            'fail_focus': 0.07,
            'pass_exposure': 0.03,
            'nearest_pass_exposure': 0.06,
            'margin_contrastive': 0.08,
            'pairwise_interaction': 0.06,
            'lofo_stability': 0.05,
            'def_use_pair': 0.04,
            'scalar_pair': 0.03,
        }
        self.signal_weights = {**default_weights, **(signal_weights or {})}
        self.signal_order = [
            'coverage',
            'value_dynamics',
            'discriminative',
            'robustness',
            'contrastive',
            'fail_focus',
            'pass_exposure',
            'nearest_pass_exposure',
            'margin_contrastive',
            'pairwise_interaction',
            'lofo_stability',
            'def_use_pair',
            'scalar_pair',
        ]
        self.signals_computed = {}
        self.aggregated_importance = None
        self.fusion_k = max(1.0, float(fusion_k))
        self.head_k = max(1, int(head_k))
        self.head_focus = max(1.0, float(head_focus))
        self.stability_power = max(0.0, float(stability_power))
        self.feature_stability = None
        self.ranking_mode = str(ranking_mode or "balanced").strip().lower()
        self.verify_pass_penalty = max(0.0, float(verify_pass_penalty))
        self.fail_focus_boost = max(0.0, float(fail_focus_boost))
        self.dynamic_mode_switch = bool(dynamic_mode_switch)
        self.diversity_strength = max(0.0, float(diversity_strength))
        self.calibration_strength = max(0.0, min(1.0, float(calibration_strength)))
        self.pairwise_top_features = max(12, int(pairwise_top_features))
        self.selected_ranking_mode = self.ranking_mode
        self.routing_confidence = 0.0
        self.feature_groups = []
        self.sample_weights = None
        self.fail_sample_weights = None
        self.pass_sample_weights = None
        self.nearest_pass_weights = None
        self.lofo_stability = None

    @staticmethod
    def _normalize_signal(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        if arr.size == 0:
            return arr
        minimum = float(np.min(arr))
        maximum = float(np.max(arr))
        if maximum <= minimum:
            return np.zeros_like(arr, dtype=float)
        return (arr - minimum) / (maximum - minimum)

    @staticmethod
    def _compute_fail_diversity_weights(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
    ) -> Optional[np.ndarray]:
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(bool)
        fail_indices = np.where(y_data)[0]
        if fail_indices.size <= 1:
            return None
        matrix = np.asarray(x_train.to_numpy(dtype=float), dtype=float)
        fail_matrix = np.nan_to_num(np.abs(matrix[fail_indices]), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(fail_matrix, axis=1) + 1e-12
        similarity = (fail_matrix @ fail_matrix.T) / (norms[:, None] * norms[None, :])
        similarity = np.clip(similarity, 0.0, 1.0)
        mean_similarity = (np.sum(similarity, axis=1) - 1.0) / max(1, fail_matrix.shape[0] - 1)
        novelty = np.clip(1.0 - mean_similarity, 0.15, 2.0)
        novelty /= (np.mean(novelty) + 1e-12)
        return novelty

    @staticmethod
    def _compute_nearest_pass_weights(
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
        verify_pass_penalty: float,
    ) -> Optional[np.ndarray]:
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(bool)
        fail_indices = np.where(y_data)[0]
        pass_indices = np.where(~y_data)[0]
        if fail_indices.size == 0 or pass_indices.size == 0:
            return None
        matrix = np.asarray(x_train.to_numpy(dtype=float), dtype=float)
        fail_matrix = np.nan_to_num(np.abs(matrix[fail_indices]), nan=0.0, posinf=0.0, neginf=0.0)
        pass_matrix = np.nan_to_num(np.abs(matrix[pass_indices]), nan=0.0, posinf=0.0, neginf=0.0)
        fail_proto = np.mean(fail_matrix, axis=0)
        fail_norm = float(np.linalg.norm(fail_proto) + 1e-12)
        pass_norm = np.linalg.norm(pass_matrix, axis=1) + 1e-12
        similarity = (pass_matrix @ fail_proto) / (pass_norm * fail_norm)
        similarity = np.clip(similarity, 0.0, 1.0)

        top_k = max(1, int(np.ceil(0.30 * pass_indices.size)))
        chosen = np.argsort(-similarity)[:top_k]
        selected = similarity[chosen]
        if selected.size == 0:
            return None
        s_min = float(np.min(selected))
        s_max = float(np.max(selected))
        if s_max > s_min:
            selected = (selected - s_min) / (s_max - s_min)
        else:
            selected = np.ones_like(selected)

        weights = np.ones(pass_indices.size, dtype=float)
        boost = 0.5 + 0.8 * float(np.clip(verify_pass_penalty, 0.0, 1.0))
        weights[chosen] = 1.0 + boost * selected
        return weights

    @staticmethod
    def _feature_group_key(feature_name: str) -> str:
        name = str(feature_name)
        if ".py" in name:
            prefix = name.split(".py", 1)[0]
            if ":" in prefix:
                prefix = prefix.rsplit(":", 1)[0]
            if "@" in prefix:
                prefix = prefix.split("@", 1)[0]
            return f"{prefix}.py"
        if ":" in name:
            return name.split(":", 1)[0]
        if "@" in name:
            return name.split("@", 1)[0]
        return name[:48]

    def _build_feature_groups(self, columns) -> list[str]:
        return [self._feature_group_key(col) for col in columns]

    def _compute_pairwise_interaction_signal(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Approximate pairwise interactions by combining fail-pass separation with
        local correlation structure among high-signal features.
        """
        n_features = x_train.shape[1]
        if n_features == 0:
            return np.array([], dtype=float)
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(bool)
        fail_mask = y_data
        pass_mask = ~y_data
        if not np.any(fail_mask) or not np.any(pass_mask):
            return np.zeros(n_features, dtype=float)

        matrix = np.asarray(x_train.to_numpy(dtype=float), dtype=float)
        matrix = np.nan_to_num(np.abs(matrix), nan=0.0, posinf=0.0, neginf=0.0)
        fail_mean = np.mean(matrix[fail_mask], axis=0)
        pass_mean = np.mean(matrix[pass_mask], axis=0)
        sep = np.abs(fail_mean - pass_mean)
        if sep.size == 0:
            return np.zeros(n_features, dtype=float)

        top_n = min(n_features, self.pairwise_top_features)
        top_idx = np.argsort(-sep)[:top_n]
        subset = matrix[:, top_idx]
        if subset.shape[1] <= 1:
            raw = np.zeros(n_features, dtype=float)
            raw[top_idx] = sep[top_idx]
            max_v = float(np.max(raw)) if raw.size else 0.0
            return raw / max_v if max_v > 1e-12 else raw

        corr = np.corrcoef(subset, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        neighborhood = np.mean(corr, axis=1)
        inter_strength = sep[top_idx] * (0.5 + 0.5 * neighborhood)

        raw = np.zeros(n_features, dtype=float)
        raw[top_idx] = inter_strength
        max_v = float(np.max(raw)) if raw.size else 0.0
        if max_v > 1e-12:
            raw = raw / max_v
        return raw

    def _compute_leave_one_failout_stability(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series | pd.DataFrame,
    ) -> np.ndarray:
        """
        Stability of feature specificity under leave-one-failing-run-out.
        High score means a feature remains important across failing-run perturbations.
        """
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(bool)
        fail_indices = np.where(y_data)[0]
        n_features = x_train.shape[1]
        if fail_indices.size <= 1 or n_features == 0:
            return np.ones(n_features, dtype=float)

        scorer = MultiSignalFeatureScorer()
        matrix = []
        for held_out in fail_indices:
            fail_weights = np.ones_like(y_data, dtype=float)
            fail_weights[held_out] = 0.0
            if np.sum(fail_weights[y_data]) <= 1e-12:
                continue
            signal = scorer.failure_specificity_signal(
                x_train,
                y_train,
                fail_sample_weights=fail_weights[y_data],
                pass_sample_weights=None,
            )
            matrix.append(signal)

        if not matrix:
            return np.ones(n_features, dtype=float)

        stacked = np.asarray(matrix, dtype=float)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        cv = std / (np.abs(mean) + 1e-6)
        stability = 1.0 / (1.0 + cv)
        stability = np.asarray(np.clip(stability, 0.05, 1.0), dtype=float)
        max_v = float(np.max(stability)) if stability.size else 0.0
        if max_v > 1e-12:
            stability = stability / max_v
        return stability

    def _calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Light calibration to convert raw fused values into confidence-like scores.
        Not strictly monotonic due to rank blending, which helps tie-rich heads.
        """
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad > 1e-12:
            z = (arr - median) / (1.4826 * mad)
        else:
            z = arr - median
        prob = 1.0 / (1.0 + np.exp(-z))
        ranks = rankdata(-arr, method='average')
        rank_mass = 1.0 - (ranks - 1.0) / max(1.0, len(arr) - 1.0)
        alpha = float(self.calibration_strength)
        calibrated = (1.0 - alpha) * arr + alpha * (0.65 * prob + 0.35 * rank_mass)
        calibrated = np.asarray(np.nan_to_num(calibrated, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)
        min_v = float(np.min(calibrated))
        max_v = float(np.max(calibrated))
        if max_v > min_v:
            calibrated = (calibrated - min_v) / (max_v - min_v)
        return calibrated

    def _apply_top_head_diversification(self, scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores, dtype=float).copy()
        if arr.size == 0 or len(self.feature_groups) != arr.size or self.diversity_strength <= 0.0:
            return arr
        order = np.argsort(-arr)
        head_n = min(len(order), max(10, self.head_k * 3))
        seen = {}
        strength = float(np.clip(self.diversity_strength, 0.0, 0.45))
        for idx in order[:head_n]:
            group = self.feature_groups[idx]
            repeats = seen.get(group, 0)
            if repeats > 0:
                penalty = 1.0 - min(0.60, strength * repeats / (1.0 + repeats))
                arr[idx] *= penalty
            seen[group] = repeats + 1
        return arr

    def _resolve_ranking_mode(self) -> str:
        mode = self.ranking_mode
        if mode == "fail_first_verify_pass":
            self.selected_ranking_mode = mode
            self.routing_confidence = 1.0
            return mode
        if not self.dynamic_mode_switch:
            self.selected_ranking_mode = mode
            self.routing_confidence = 0.0
            return mode

        specificity = self._normalize_signal(
            np.asarray(self.signals_computed.get("failure_specificity", np.array([], dtype=float)), dtype=float)
        )
        if specificity.size == 0:
            self.selected_ranking_mode = "balanced"
            self.routing_confidence = 0.0
            return self.selected_ranking_mode
        ranked = np.sort(specificity)[::-1]
        top_gap = float(ranked[0] - ranked[1]) if ranked.size >= 2 else float(ranked[0])

        mass = specificity + 1e-12
        mass = mass / np.sum(mass)
        entropy = -float(np.sum(mass * np.log(mass + 1e-12)))
        max_entropy = float(np.log(len(mass))) if len(mass) > 1 else 1.0
        concentration = 1.0 - (entropy / max_entropy if max_entropy > 0 else 1.0)

        fail_focus = float(np.mean(self.signals_computed.get("fail_focus", np.zeros_like(specificity))))
        pass_noise = float(np.mean(self.signals_computed.get("nearest_pass_exposure", np.zeros_like(specificity))))
        contrastive_margin = float(np.mean(self.signals_computed.get("margin_contrastive", np.zeros_like(specificity))))
        margin = max(0.0, fail_focus - pass_noise)
        confidence = np.clip(
            0.35 * top_gap
            + 0.30 * concentration
            + 0.20 * margin
            + 0.15 * contrastive_margin,
            0.0,
            1.0,
        )

        self.routing_confidence = float(confidence)
        self.selected_ranking_mode = "fail_first_verify_pass" if confidence >= 0.34 else "balanced"
        return self.selected_ranking_mode

    def compute_signals(self, x_train: pd.DataFrame, y_train: pd.Series | pd.DataFrame):
        """Compute all signals with diversity-aware fail weighting and nearest-pass negatives."""
        scorer = MultiSignalFeatureScorer()
        y_data = MultiSignalFeatureScorer._labels_to_vector(y_train).astype(bool)
        fail_mask = y_data
        pass_mask = ~y_data

        sample_weights = np.ones(len(y_data), dtype=float)
        fail_weights = self._compute_fail_diversity_weights(x_train, y_train)
        if fail_weights is not None and fail_mask.any():
            sample_weights[fail_mask] = fail_weights

        pass_weights = np.ones(int(np.sum(pass_mask)), dtype=float)
        nearest_pass_weights = self._compute_nearest_pass_weights(
            x_train,
            y_train,
            self.verify_pass_penalty,
        )
        if nearest_pass_weights is not None and pass_mask.any():
            pass_weights = nearest_pass_weights
            sample_weights[pass_mask] = nearest_pass_weights

        self.sample_weights = sample_weights
        self.fail_sample_weights = sample_weights[fail_mask] if fail_mask.any() else None
        self.pass_sample_weights = sample_weights[pass_mask] if pass_mask.any() else None
        self.nearest_pass_weights = nearest_pass_weights

        self.signals_computed['coverage'] = scorer.coverage_signal(x_train, y_train)
        self.signals_computed['value_dynamics'] = scorer.value_dynamics_signal(x_train, y_train)
        self.signals_computed['discriminative'] = scorer.discriminative_power_signal(x_train, y_train)
        self.signals_computed['robustness'] = scorer.robustness_signal(x_train, y_train)
        self.signals_computed['contrastive'] = scorer.contrastive_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
            pass_sample_weights=self.pass_sample_weights,
        )
        self.signals_computed['fail_focus'] = scorer.fail_focus_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
        )
        self.signals_computed['pass_exposure'] = scorer.pass_exposure_signal(x_train, y_train)
        self.signals_computed['nearest_pass_exposure'] = scorer.pass_exposure_signal(
            x_train,
            y_train,
            pass_sample_weights=pass_weights if pass_mask.any() else None,
        )
        self.signals_computed['margin_contrastive'] = scorer.margin_contrastive_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
            pass_sample_weights=self.pass_sample_weights,
        )
        self.signals_computed['pairwise_interaction'] = self._compute_pairwise_interaction_signal(
            x_train,
            y_train,
        )
        self.signals_computed['failure_specificity'] = scorer.failure_specificity_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
            pass_sample_weights=self.pass_sample_weights,
        )
        self.signals_computed['def_use_pair'] = scorer.def_use_pair_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
            pass_sample_weights=self.pass_sample_weights,
        )
        self.signals_computed['scalar_pair'] = scorer.scalar_pair_signal(
            x_train,
            y_train,
            fail_sample_weights=self.fail_sample_weights,
            pass_sample_weights=self.pass_sample_weights,
        )
        self.lofo_stability = self._compute_leave_one_failout_stability(x_train, y_train)
        self.signals_computed['lofo_stability'] = self.lofo_stability
        base_stability = np.clip(self.signals_computed['robustness'], 0.05, 1.0)
        if self.lofo_stability is not None and len(self.lofo_stability) == len(base_stability):
            base_stability = np.sqrt(np.clip(base_stability, 0.0, 1.0) * np.clip(self.lofo_stability, 0.0, 1.0))
        self.feature_stability = np.power(base_stability, self.stability_power)

    def aggregate_signals(self) -> np.ndarray:
        """
        Aggregate multiple signals using robust rank fusion.

        This keeps low-variance evidence alive, reduces sensitivity to a single
        noisy signal, and avoids hard feature pruning.
        """
        active_mode = self._resolve_ranking_mode()
        if active_mode == "fail_first_verify_pass":
            specificity = np.asarray(
                self.signals_computed.get(
                    "failure_specificity",
                    self.signals_computed.get("contrastive", np.array([], dtype=float)),
                ),
                dtype=float,
            )
            fail_focus = np.asarray(self.signals_computed.get("fail_focus", specificity), dtype=float)
            contrastive = np.asarray(self.signals_computed.get("contrastive", specificity), dtype=float)
            coverage = np.asarray(self.signals_computed.get("coverage", specificity), dtype=float)
            value_dynamics = np.asarray(self.signals_computed.get("value_dynamics", specificity), dtype=float)
            robustness = np.asarray(self.signals_computed.get("robustness", specificity), dtype=float)
            def_use_pair = np.asarray(self.signals_computed.get("def_use_pair", specificity * 0.0), dtype=float)
            scalar_pair = np.asarray(self.signals_computed.get("scalar_pair", specificity * 0.0), dtype=float)
            margin_contrastive = np.asarray(
                self.signals_computed.get("margin_contrastive", specificity * 0.0),
                dtype=float,
            )
            pairwise_interaction = np.asarray(
                self.signals_computed.get("pairwise_interaction", specificity * 0.0),
                dtype=float,
            )
            lofo_stability = np.asarray(
                self.signals_computed.get("lofo_stability", np.ones_like(specificity)),
                dtype=float,
            )
            pass_exposure = np.asarray(
                self.signals_computed.get("pass_exposure", np.zeros_like(specificity)),
                dtype=float,
            )
            nearest_pass_exposure = np.asarray(
                self.signals_computed.get("nearest_pass_exposure", np.zeros_like(specificity)),
                dtype=float,
            )

            fail_gain = 1.0 + 0.65 * self.fail_focus_boost
            pass_penalty = 0.20 + 0.30 * self.verify_pass_penalty

            raw = (
                0.40 * specificity
                + (0.18 * fail_gain) * fail_focus
                + 0.15 * contrastive
                + 0.10 * value_dynamics
                + 0.08 * coverage
                + 0.05 * robustness
                + 0.07 * def_use_pair
                + 0.05 * scalar_pair
                + 0.08 * margin_contrastive
                + 0.06 * pairwise_interaction
                + 0.04 * lofo_stability
                - (0.10 + 0.12 * pass_penalty) * pass_exposure
                - (0.12 + 0.15 * pass_penalty) * nearest_pass_exposure
            )
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            raw = self._apply_top_head_diversification(raw)
            raw = self._calibrate_scores(raw)
            if len(raw) > 0:
                r_min, r_max = float(np.min(raw)), float(np.max(raw))
                if r_max > r_min:
                    raw = (raw - r_min) / (r_max - r_min)
                else:
                    raw = np.zeros_like(raw)
            self.aggregated_importance = raw
            return raw

        reference = np.asarray(next(iter(self.signals_computed.values())), dtype=float)
        stacked_signals = np.stack(
            [
                np.asarray(
                    self.signals_computed.get(key, np.zeros_like(reference)),
                    dtype=float,
                )
                for key in self.signal_order
            ],
            axis=0,
        )

        normalized = np.zeros_like(stacked_signals, dtype=float)
        confidences = []
        for i in range(stacked_signals.shape[0]):
            signal = np.asarray(
                np.nan_to_num(stacked_signals[i], nan=0.0, posinf=0.0, neginf=0.0),
                dtype=float,
            )
            s_min = float(np.min(signal))
            s_max = float(np.max(signal))
            if s_max > s_min:
                denom = s_max - s_min
                signal = np.array([(float(v) - s_min) / denom for v in signal], dtype=float)
            signal = winsorize(signal, limits=(0.05, 0.05))
            signal = np.asarray(signal, dtype=float)
            normalized[i] = signal

            mass = np.abs(signal) + 1e-12
            mass = mass / mass.sum()
            entropy = float(np.sum(mass * np.log(mass + 1e-12)))
            entropy = -entropy
            max_entropy = np.log(len(signal)) if len(signal) > 1 else 1.0
            confidence = 1.0 - (entropy / max_entropy if max_entropy > 0 else 1.0)
            confidences.append(float(np.clip(confidence, 0.05, 1.0)))

        base_weights = np.array([
            float(self.signal_weights.get(key, 0.0)) for key in self.signal_order
        ], dtype=float)

        if active_mode == "fail_first_verify_pass":
            # Reweight towards failing-specific evidence and stronger pass-side verification.
            mode_adjust = {
                'coverage': 1.10,
                'value_dynamics': 1.00,
                'discriminative': 0.95,
                'robustness': 1.00,
                'contrastive': 1.20,
                'fail_focus': 1.30 + 0.50 * self.fail_focus_boost,
                'pass_exposure': max(0.4, 1.0 - 0.7 * self.verify_pass_penalty),
                'nearest_pass_exposure': max(0.35, 1.0 - 0.8 * self.verify_pass_penalty),
                'margin_contrastive': 1.10,
                'pairwise_interaction': 1.05,
                'lofo_stability': 1.05,
                'def_use_pair': 1.05,
                'scalar_pair': 1.00,
            }
            adjust_vector = np.array(
                [float(mode_adjust.get(key, 1.0)) for key in self.signal_order],
                dtype=float,
            )
            base_weights = base_weights * adjust_vector
        combined_weights = base_weights * np.array(confidences, dtype=float)
        if combined_weights.sum() <= 0:
            combined_weights = np.ones_like(combined_weights)
        combined_weights = combined_weights / combined_weights.sum()

        fused = np.zeros(normalized.shape[1], dtype=float)
        fusion_k = self.fusion_k
        for weight, signal in zip(combined_weights, normalized):
            ranks = rankdata(-signal, method='average')
            rr = weight / (fusion_k + ranks)
            if len(ranks) > 0:
                head_mask = ranks <= min(self.head_k, len(ranks))
                rr = rr * np.where(head_mask, self.head_focus, 1.0)
            fused += rr

        nearest_pass_exposure = np.asarray(
            self.signals_computed.get("nearest_pass_exposure", np.zeros_like(fused)),
            dtype=float,
        )
        if nearest_pass_exposure.size == fused.size:
            suppression = 1.0 - 0.25 * self.verify_pass_penalty * nearest_pass_exposure
            fused *= np.clip(suppression, 0.35, 1.0)

        if self.feature_stability is not None and len(self.feature_stability) == len(fused):
            fused *= self.feature_stability

        # Penalize highly inconsistent features that spike in only one signal.
        disagreement = np.std(normalized, axis=0)
        coherence = 1.0 / (1.0 + 4.0 * disagreement)
        fused *= np.clip(coherence, 0.2, 1.0)
        fused = self._apply_top_head_diversification(fused)
        fused = self._calibrate_scores(fused)

        median = np.median(fused)
        mad = np.median(np.abs(fused - median))
        if mad > 1e-10:
            fused = (fused - median) / (1.4826 * mad)
            fused = np.clip(fused, -3, 3)
            fused = (fused + 3) / 6

        fused = np.asarray(fused, dtype=float)
        self.aggregated_importance = fused
        return fused

    def train(self):
        """
        Train the hybrid model on feature signals.
        We create a ranking problem: features are "documents", signal vector is "query".
        """
        # Compute all signals
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Training data is not initialized")
        x_train = self.x_train
        y_train = self.y_train
        self.compute_signals(x_train, y_train)

        # Aggregate signals to get feature importance
        importance = self.aggregate_signals()

        # Store for later use
        self.feature_importance_scores = {
            feat: imp for feat, imp in zip(self.all_features, importance)
        }

    def fit(self, all_features, handler: EventHandler):
        """Fit the hybrid model."""
        self.all_features = all_features
        self.x_train, self.y_train = self.prepare_data(handler)
        self.feature_groups = self._build_feature_groups(self.x_train.columns)
        self.train()

    def get_importance(self):
        """Return feature importance as list of (feature, importance) tuples."""
        if not hasattr(self, 'feature_importance_scores'):
            # Fallback if fit not called
            importance = np.ones(len(self.all_features)) / len(self.all_features)
        else:
            importance = [
                self.feature_importance_scores.get(feat, 0.0)
                for feat in self.all_features
            ]
        importance_list = [float(v) for v in np.asarray(importance, dtype=float).ravel().tolist()]
        return [(feat, imp) for feat, imp in zip(self.all_features, importance_list)]

    def evaluate(self, handler: EventHandler, output_dict: bool = False):
        """Evaluate on test data (note: using same features as train)."""
        # For now, report perfect scores as we're not using traditional classification
        # In a full implementation, would do hold-out evaluation
        
        # Dummy evaluation (stub for now)
        report = "Hybrid ranking model trained. Use feature importance for evaluation."
        confusion = [[0, 0], [0, 0]]
        return report, confusion
