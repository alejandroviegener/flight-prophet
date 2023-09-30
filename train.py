"""
Train and serialize a DelayModel
"""

import argparse

import pandas as pd

from challenge.model import DelayModel


def train_and_save_model(source_file: str, output_file: str):
    """
    Train and serialize a DelayModel
    """

    # Load data and create model instance
    data = pd.read_csv(source_file)
    model = DelayModel()

    # Preprocess data and fit model
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)

    # Serialize model
    model.save(output_file)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using data from source file and save the trained model to an output file."
    )
    parser.add_argument(
        "source_file", help="Path to the source data file for training the model"
    )
    parser.add_argument(
        "output_file",
        help="Path where the trained model will be saved in pickled format",
    )

    args = parser.parse_args()

    print("Training model...")
    train_and_save_model(args.source_file, args.output_file)
    print("Model trained and serialized!")
