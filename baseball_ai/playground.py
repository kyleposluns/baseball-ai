
import numpy as np
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
import argparse
from pathlib import Path
import json
from enum import Enum


parser = argparse.ArgumentParser(description="Lets train a model")
parser.add_argument("training_data_file", action="store", type=lambda p: Path(p).absolute())

class PlayResult(Enum):
    STEAL = [1, 0, 0]
    BUNT = [0, 1, 0]
    NONE = [0, 0, 1]

def read_training_data(file):
    inputs = []
    outputs = []
    with file.open() as f:
        training_data = json.load(f)

    for test, result in training_data.items():

        inputs.append([ord(character) for character in test])
        outputs.append(PlayResult[result].value)

    return np.array(inputs), np.array(outputs)





if __name__ == "__main__":
    args = parser.parse_args()

    inputs, outputs = read_training_data(args.training_data_file)

    model = Sequential([
        Dense(5, input_dim=10, activation="sigmoid"),
        Dense(3, activation="sigmoid")
    ])

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    model.fit(inputs, outputs, epochs=150, batch_size=10)
    print(model.evaluate())

