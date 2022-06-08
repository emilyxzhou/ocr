import matplotlib.pyplot as plt
import os

from os.path import isfile


def load_training_data():
    cwd = os.getcwd()
    train_data_path = os.path.join(
        cwd, "..", "data", "training"
    )
    char_image_paths = [
        f for f in os.listdir(
            os.path.join(train_data_path, "characters")
        )
    ]
    symbol_image_paths = [
        f for f in os.listdir(
            os.path.join(train_data_path, "symbols")
        )
    ]

    return char_image_paths + symbol_image_paths


if __name__ == "__main__":
    cwd = os.getcwd()
    train_data_path = os.path.join(
        cwd, "..", "data", "training"
    )
