import argparse
from pathlib import Path

import tensorflow as tf
from keras.layers import Dense, Input
from tensorflow import keras

from baseball_ai.util import read_data

parser = argparse.ArgumentParser(description="Lets train a model")
parser.add_argument("training_data_file", action="store", type=lambda p: Path(p).absolute())
parser.add_argument("testing_data_file", action="store", type=lambda p: Path(p).absolute())


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


if __name__ == "__main__":
    args = parser.parse_args()

    training_inputs, training_outputs = read_data(args.training_data_file)
    testing_inputs, testing_outputs = read_data(args.testing_data_file)

    inp = Input(shape=(10,))
    hidden = Dense(units=10, activation='sigmoid')(inp)
    hidden = Dense(units=64, activation='sigmoid')(hidden)
    hidden = Dense(units=30, activation='sigmoid')(hidden)
    out = Dense(units=3, activation='sigmoid')(hidden)
    model = CustomModel(inp, out)
    
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
    model.fit(training_inputs, training_outputs, epochs=400, batch_size=100, shuffle=True)
    print(model.evaluate(testing_inputs, testing_outputs))

