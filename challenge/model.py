import pickle
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from .classifier import LogisticRegressionClassifier

REQUIRED_FEATURE_DATA = ["OPERA", "TIPOVUELO", "MES"]
REQUIRED_TARGET_DATA = ["Fecha-O", "Fecha-I"]

ALLOWED_OPERATORS = [
    "Grupo LATAM",
    "Sky Airline",
    "Aerolineas Argentinas",
    "Copa Air",
    "Latin American Wings",
    "Avianca",
    "JetSmart SPA",
    "Gol Trans",
    "American Airlines",
    "Air Canada",
    "Iberia",
    "Delta Air",
    "Air France",
    "Aeromexico",
    "United Airlines",
    "Oceanair Linhas Aereas",
    "Alitalia",
    "K.L.M.",
    "British Airways",
    "Qantas Airways",
    "Lacsa",
    "Austral",
    "Plus Ultra Lineas Aereas",
]

ALLOWED_FLIGHT_TYPES = ["N", "I"]

ALLOWED_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

FEATURES = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]


class DelayModel:
    def __init__(self):
        self._model = (
            LogisticRegressionClassifier()
        )  # Model should be saved in this attribute.

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.

        Raises:
            ValueError: if requiered data columns not present.
            ValueError: if unkonwn values in categorical features.
        """
        # Check that all requiered columns are present
        required_columns = (
            REQUIRED_FEATURE_DATA + REQUIRED_TARGET_DATA
            if target_column
            else REQUIRED_FEATURE_DATA
        )
        self._check_requiered_columns_are_present(data, required_columns)
        self._check_categorical_values_are_valid(data)

        # Construct target if required, is None otherwise
        # Keep only the requiered features
        target = self._get_target(data, target_column)
        data = data[REQUIRED_FEATURE_DATA]

        # Construct features
        features = self._get_features(data)

        # Return features (and target if requiered)
        if target_column:
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        self._model.fit(features, target)
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

        # Predict
        return self._model.predict(features)

    def save(self, file_path) -> None:
        """
        Save DelayModel to pickle file.

        Args:
            file_path (str): file path.
        """
        pickle.dump(self, open(file_path, "wb"))
        return

    @classmethod
    def load(cls, model_filepath: str):
        """
        Load DelayModel from pickle file.

        Args:
            model_filepath (str): file path.

        Returns:
            DelayModel: model.
        """

        model = None
        try:
            model = pickle.load(open(model_filepath, "rb"))
            print(f"Model loaded from {model_filepath}")
        except:
            print(f"Error loading model from {model_filepath}")
            model = cls()
        return model

    @staticmethod
    def _check_requiered_columns_are_present(
        data: pd.DataFrame, required_columns: List[str]
    ):
        """
        Check that all requiered columns are present.
        """
        if not set(required_columns).issubset(set(data.columns)):
            raise ValueError("Missing requiered columns in data")

    @staticmethod
    def _check_categorical_values_are_valid(data: pd.DataFrame):
        """
        Check that all categorical values are valid.
        """
        # Check if only allowed values are present
        if not set(data["OPERA"]).issubset(set(ALLOWED_OPERATORS)):
            raise ValueError("Invalid values in OPERA column")
        if not set(data["TIPOVUELO"]).issubset(set(ALLOWED_FLIGHT_TYPES)):
            raise ValueError("Invalid values in TIPOVUELO column")
        if not set(data["MES"]).issubset(set(ALLOWED_MONTHS)):
            raise ValueError("Invalid values in MES column")

    @staticmethod
    def _get_target(
        data: pd.DataFrame, target_column: Union[str, None]
    ) -> Union[pd.DataFrame, None]:
        """
        Construct target from raw data.

        Args:
            data (pd.DataFrame): raw data.
            target_column (Union[str, None]): target column name.

        Returns:
            pd.DataFrame: target or None if target column not requiered.
        """
        if not target_column:
            return None

        data["min_diff"] = data.apply(_get_min_diff, axis=1)
        threshold_in_minutes = 15
        data[target_column] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)
        target = data[[target_column]]
        return target

    @staticmethod
    def _get_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Construct features from raw data.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            pd.DataFrame: features.
        """

        # Convert columns to categorical and set explicitly the allowed values
        data["OPERA"] = pd.Categorical(data["OPERA"], categories=ALLOWED_OPERATORS)
        data["TIPOVUELO"] = pd.Categorical(
            data["TIPOVUELO"], categories=ALLOWED_FLIGHT_TYPES
        )
        data["MES"] = pd.Categorical(data["MES"], categories=ALLOWED_MONTHS)

        # Dummy variables for categories
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        # Keep only the requiered features
        features = features[FEATURES]
        return features


# Utility functions


def _get_min_diff(data):
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff
