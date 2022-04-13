import argparse
import json
from enum import Enum
from pathlib import Path

import numpy as np
import math


def sigmoid(x):
    return float(1 / (1 + math.exp(-x)))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


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

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    input_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    return input_norm.tolist(), outputs.tolist()

class MLP:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, activation, activation_derivative):
        self.weights = [
            np.random.rand(hidden_nodes, input_nodes + 1),
            np.random.rand(output_nodes, hidden_nodes + 1),
        ]
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.activation = np.vectorize(activation)
        self.activation_derivative = np.vectorize(activation_derivative)

    def compute(self, input_values):
        input_values = np.append([1], input_values)
        last_input = input_values
        weights_times_inputs = np.dot(self.weights[0], np.transpose(input_values))
        activated = self.activation(weights_times_inputs)
        last_hidden_output = np.append([1], activated)

        last_layer = np.dot(self.weights[1], last_hidden_output)
        last_output = self.activation(last_layer)
        return last_input, weights_times_inputs, last_hidden_output, last_output


class BackPropagation:
    def __init__(self, mlp, learning_rate, penalty):
        self.deltas = [
            np.zeros((mlp.hidden_nodes, mlp.input_nodes + 1)),
            np.zeros((mlp.output_nodes, mlp.hidden_nodes + 1)),
        ]

        self.learning_rate = learning_rate
        self.penalty = penalty
        self.mlp = mlp

    def iterate(self, input_values, output_values):
        for i in range(len(input_values)):
            last_input, weights_times_inputs, last_hidden_output, last_output = self.mlp.compute(input_values[i])
            diff = np.subtract(last_output, output_values[i])

            od3 = np.dot((self.mlp.weights[1].transpose()), diff)
            # dw = np.multiply(last_hidden_output, (1 - last_hidden_output))
            dw = np.append([0], self.mlp.activation_derivative(weights_times_inputs))

            delta2 = np.multiply(od3, dw)

            self.deltas[1] = np.add(
                self.deltas[1], np.outer(diff, last_hidden_output)
            )
            self.deltas[0] = np.add(
                self.deltas[0], np.delete(np.outer(delta2, last_input), 0, 0)
            )

        self.deltas[0] = (
                self.deltas[0] / len(input_values)
                + self.penalty * self.mlp.weights[0]
        )
        self.deltas[1] = (
                self.deltas[1] / len(input_values)
                + self.penalty * self.mlp.weights[1]
        )

        self.mlp.weights[0] -= self.learning_rate * self.deltas[0]
        self.mlp.weights[1] -= self.learning_rate * self.deltas[1]


if __name__ == "__main__":
    args = parser.parse_args()
    mlp = MLP(5, 10, 3, sigmoid, sigmoid_derivative)
    bprop = BackPropagation(mlp, learning_rate=0.1, penalty=0.1)
    train_inputs, train_outputs = read_data(args.training_data_file)
    test_inputs, test_outputs = read_data(args.testing_data_file)
    bprop.iterate(train_inputs, train_outputs)

    amount_correct = 0
    amount_seen = 0
    for i in range(len(test_inputs)):
        test_input = test_inputs[i]
        test_output = test_outputs[i]
        output = [round(i) for i in mlp.compute(test_input)[3].tolist()]
        if output == test_output:
            amount_correct += 1

        amount_seen += 1
    print(f"Accuracy: {amount_correct / amount_seen}")








