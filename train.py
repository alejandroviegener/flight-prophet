"""
Train and serialize a DelayModel
"""

import pickle 
import pandas as pd

from challenge.model import DelayModel


def train_and_save_model():
    """
    Train and serialize a DelayModel
    """

    # Load data and create model instance
    data = pd.read_csv("data/data.csv")
    model = DelayModel()

    # Preprocess data and fit model
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)

    # Serialize model
    model.save("tmp/delay_model.pkl")
    return


if __name__ == "__main__":
    print("Training model...")
    train_and_save_model()
    print("Model trained and serialized!")