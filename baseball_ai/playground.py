
import numpy as np
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
import argparse
from pathlib import Path

from baseball_ai.util import read_data

parser = argparse.ArgumentParser(description="Lets train a model")
parser.add_argument("training_data_file", action="store", type=lambda p: Path(p).absolute())
parser.add_argument("testing_data_file", action="store", type=lambda p: Path(p).absolute())



if __name__ == "__main__":
    args = parser.parse_args()

    training_inputs, training_outputs = read_data(args.training_data_file)
    testing_inputs, testing_outputs = read_data(args.testing_data_file)

    model = Sequential([
        Dense(10, input_dim=10, activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        Dense(10, activation="sigmoid"),
        Dense(3, activation="sigmoid")
    ])

    model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=0.1), metrics=["accuracy"])
    model.fit(training_inputs, training_outputs, epochs=150, batch_size=1000)
    print(model.evaluate(testing_inputs, testing_outputs))

