import pandas as pd

from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import numpy as np


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
    "OPERA_Copa Air"
]

class DelayModel:

    def __init__(
        self
    ):
        self._model = LogisticRegression() # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
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
        """

        # Construct target if required
        target = None
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis = 1)
            threshold_in_minutes = 15
            data[target_column] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            target = data[[target_column]]

        # Concstruct dummy features
        data = data[["OPERA", "TIPOVUELO", "MES"]]
        features = pd.concat(   
            [
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix = 'MES')
            ], 
            axis = 1
        )

        # Filter useful features
        features = features[FEATURES]

        if target_column:   
            return features, target        
        return features

    @staticmethod
    def _get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Compute class weights and set model parameter accordingly
        label_0_len = len(target[target['delay'] == 0])
        label_1_len = len(target[target['delay'] == 1])

        # Fit model
        self._model.set_params(class_weight={1: label_0_len/len(target), 0: label_1_len/len(target)})
        self._model.fit(features, target)
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        # Predict
        predictions = self._model.predict(features)
        return predictions.tolist()
    