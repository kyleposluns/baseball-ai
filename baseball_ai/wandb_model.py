
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Input, SimpleRNN
from keras import optimizers
from tensorflow.keras import layers
import argparse
from pathlib import Path
import json
from enum import Enum
from keras.models import Model

import wandb
from wandb.keras import WandbCallback

wandb.init(project="ai-project", entity="cs4100")

parser = argparse.ArgumentParser(description="Lets train a model")
parser.add_argument("training_data_file", action="store", type=lambda p: Path(p).absolute())
parser.add_argument("testing_data_file", action="store", type=lambda p: Path(p).absolute())

class PlayResult(Enum):
    STEAL = [1, 0, 0]
    BUNT = [0, 1, 0]
    NONE = [0, 0, 1]

class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}   


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

    x_train, y_train = read_data(args.training_data_file)
    x_test, y_test = read_data(args.testing_data_file)

    wandb.config = {
        "learning_rate": 0.01,
        "epochs": 400,
        "batch_size": 200
    }

    #define model
    inp = Input(shape=(10,))
    hidden = Dense(units=10, activation='sigmoid')(inp)
    hidden = Dense(units=64, activation='sigmoid')(hidden)
    hidden = Dense(units=30, activation='sigmoid')(hidden)
    out = Dense(units=3, activation='sigmoid')(hidden)
    model = CustomModel(inp, out)
    
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[WandbCallback()])

    acc = model.evaluate(x_train, y_train)[1]
    for i in range (10):
        wandb.log({"accuracy": acc})
    # wandb.log({"epoch": 400, "accuracy": acc,
    #        "inputs": wandb.Image(inp),
    #        "logits": wandb.Histogram(out)})

    print('accuracy: ', acc)

