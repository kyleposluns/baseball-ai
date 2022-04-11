
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
parser.add_argument("testing_data_file", action="store", type=lambda p: Path(p).absolute())

class PlayResult(Enum):
    STEAL = [1, 0, 0]
    BUNT = [0, 1, 0]
    NONE = [0, 0, 1]

def read_data(file):
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

    training_inputs, training_outputs = read_data(args.training_data_file)
    testing_inputs, testing_outputs = read_data(args.testing_data_file)

    model = Sequential([
        Dense(10, input_dim=10, activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        Dense(5, activation="sigmoid"),
        Dense(3, activation="sigmoid")
    ])

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    model.fit(training_inputs, training_outputs, epochs=150, batch_size=1000)
    print(model.evaluate(testing_inputs, testing_outputs))

