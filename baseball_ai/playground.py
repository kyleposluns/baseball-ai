
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
parser.add_argument("--learning-rate", action="store", type=float)
parser.add_argument("--epochs", action="store", type=int)
parser.add_argument("--batch-size", action="store", type=int)

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

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    input_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    return input_norm, outputs





if __name__ == "__main__":
    args = parser.parse_args()

    training_inputs, training_outputs = read_data(args.training_data_file)
    testing_inputs, testing_outputs = read_data(args.testing_data_file)

    model = Sequential([
        Dense(len(training_inputs[0]), activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        Dense(3, activation="sigmoid")
    ])

    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), metrics=["accuracy"])
    model.fit(training_inputs, training_outputs, epochs=args.epochs, batch_size=args.batch_size)
    print(model.evaluate(testing_inputs, testing_outputs))

