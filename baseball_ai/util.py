
import json
from enum import Enum

import numpy as np


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
