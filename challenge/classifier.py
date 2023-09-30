from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import xgboost
from sklearn import linear_model


class Classifier(ABC):
    """
    Classifier interface.
    """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit classifier with data.

        Args:
            features (pd.DataFrame): data.
            target (pd.DataFrame): target.
        """

    pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

    pass


class LogisticRegressionClassifier(Classifier):
    def __init__(
        self,
    ):
        self._clf: linear_model.LogisticRegression = linear_model.LogisticRegression()

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Compute class weights and set model parameter accordingly
        label_0_len = len(target[target["delay"] == 0])
        label_1_len = len(target[target["delay"] == 1])

        # Fit model
        self._clf.set_params(
            class_weight={1: label_0_len / len(target), 0: label_1_len / len(target)}
        )
        self._clf.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        # Predict
        predictions = self._clf.predict(features)
        return predictions.tolist()


class XGBoostClassifier(Classifier):
    def __init__(
        self,
    ):
        self._clf: xgboost.XGBClassifier = xgboost.XGBClassifier(
            random_state=1, learning_rate=0.01
        )

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Compute class weights and set model parameter accordingly
        label_0_len = len(target[target["delay"] == 0])
        label_1_len = len(target[target["delay"] == 1])
        scale = label_0_len / label_1_len
        self._clf.set_params(scale_pos_weight=scale)

        # Fit model
        self._clf.set_params(
            class_weight={1: label_0_len / len(target), 0: label_1_len / len(target)}
        )
        self._clf.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        # Predict
        predictions = self._clf.predict(features)
        return predictions.tolist()
