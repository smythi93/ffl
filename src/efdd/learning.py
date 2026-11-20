import enum
import os
import warnings
from abc import ABC
from pathlib import Path
from typing import Sequence, Tuple, Any, Optional, List, cast

import numpy as np
import pandas as pd
from joblib import dump, load
from sflkit.features.handler import EventHandler
from sflkit.features.value import Feature
from sklearn import tree
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from efdd.reduce import FeatureSelection, DefaultSelection

try:
    import shap
    HAS_SHAP = True
except ImportError:  # pragma: no cover - optional dependency
    shap = None
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class Label(enum.Enum):
    BUG = 1
    NO_BUG = 0


class DiagnosisGenerator(ABC):
    def __init__(
        self,
        model: Any,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        explainer=None,
    ):
        self.all_features: Sequence[Feature] = list()
        self.model = model
        self.path = path
        self.reducer = reducer
        self.explainer = explainer
        self.x_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None

    @staticmethod
    def _to_scalar(value: Any) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(value, dtype=float).ravel()
                if arr.size == 0:
                    return 0.0
                return float(np.nanmean(arr))
            except (TypeError, ValueError):
                return float(len(value)) if len(value) > 0 else 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _sanitize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        sanitized = data.copy()
        for col in sanitized.columns:
            series = sanitized[col]
            if series.dtype == "O":
                series = series.map(self._to_scalar)
            sanitized[col] = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return sanitized

    def prepare_data(
        self,
        handler: EventHandler,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        data = handler.to_df(self.all_features)  # type: ignore[arg-type]
        x = data.drop(columns=["test", "failing"], errors="ignore")
        return self._sanitize_features(x), cast(pd.Series, data["failing"])

    def train(self):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Training data is not initialized")
        self.model.fit(
            self.reducer.select(self.x_train).to_numpy(), self.y_train.to_numpy()
        )
        if self.path:
            dump(self.model, str(self.path))

    def fit(
        self,
        all_features: Sequence[Feature],
        handler: EventHandler,
    ):
        self.all_features = all_features
        self.x_train, self.y_train = self.prepare_data(handler)
        self.train()

    def evaluate(
        self,
        handler: EventHandler,
        output_dict: bool = False,
    ) -> Tuple[str | dict, List[List[int]]]:
        x_eval, y_eval = self.prepare_data(handler)
        return (
            self.classification_report(x_eval, y_eval, output_dict=output_dict),
            confusion_matrix(
                y_eval.to_numpy(), self.model.predict(x_eval.to_numpy()), labels=[0, 1]
            ).tolist(),
        )

    def classify(self, x: pd.DataFrame) -> Label:
        return Label(self.model.predict(x.to_numpy())[0])

    def accuracy_score(self, x: pd.DataFrame, y: pd.DataFrame):
        return accuracy_score(y.to_numpy(), self.model.predict(x.to_numpy()))

    def classification_report(
        self, x: pd.DataFrame, y: pd.Series, output_dict: bool = False
    ) -> str | dict:
        return classification_report(
            y.to_numpy(), self.model.predict(x.to_numpy()), output_dict=output_dict
        )

    def explain(self, x: Optional[pd.DataFrame] = None):
        x = x or self.x_train
        if self.explainer is None:
            raise ImportError("shap is required for explain(); install shap to use it")
        return self.explainer(self.model, x)(x)

    def finalize(
        self,
        handler: EventHandler,
    ):
        x_train, y_train = self.prepare_data(handler)
        if self.x_train is None:
            self.x_train = x_train
        else:
            self.x_train = pd.concat([self.x_train, x_train])
        if self.y_train is None:
            self.y_train = cast(pd.Series, y_train)
        else:
            self.y_train = pd.concat([self.y_train, cast(pd.Series, y_train)])
        self.train()


class DecisionTreeDiagnosis(DiagnosisGenerator):
    def __init__(
        self,
        path: Optional[os.PathLike] = None,
        reducer: FeatureSelection = DefaultSelection(),
        random_state: int = 42,
        criterion: str = "log_loss",
    ):
        # criterion: gini or log_loss
        path_obj = Path(path) if path is not None else None
        if path_obj is None or not path_obj.exists():
            model = tree.DecisionTreeClassifier(
                random_state=random_state, criterion=criterion
            )
        else:
            model = load(str(path_obj))
        super().__init__(
            model=model,
            path=path_obj,
            reducer=reducer,
            explainer=getattr(shap, "TreeExplainer", None),
        )
