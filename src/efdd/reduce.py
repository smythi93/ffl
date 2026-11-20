from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class FeatureSelection(ABC):
    def __init__(self):
        self.length = 0

    def prerequisite(self, features: pd.DataFrame):
        assert isinstance(features, pd.DataFrame), "features must be a pandas DataFrame"
        assert len(features.index) >= 1, "At least one feature vector required"
        assert len(features.columns) >= 1, "At least one feature column required"
        # In a DataFrame every row shares the same number of columns by construction.
        self.length = int(features.shape[1])

    def select(self, features: pd.DataFrame) -> pd.DataFrame:
        self.prerequisite(features)
        return self.choices(features)

    @abstractmethod
    def choices(self, features: pd.DataFrame) -> pd.DataFrame:
        pass


class DefaultSelection(FeatureSelection):
    def prerequisite(self, features: pd.DataFrame):
        pass

    def choices(self, features: pd.DataFrame) -> pd.DataFrame:
        return features


class RemoveIrrelevantFeatures(FeatureSelection):
    def choices(self, features: pd.DataFrame) -> pd.DataFrame:
        drop = []
        for col in features.columns:
            if len(set(features[col].to_list())) <= 1:
                drop.append(col)
        return features.drop(columns=drop)
