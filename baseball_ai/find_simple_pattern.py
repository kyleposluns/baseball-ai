import argparse
import json
from pathlib import Path

from baseball_ai.util import read_data

parser = argparse.ArgumentParser(description="Lets train a model")
parser.add_argument(
    "training_data_file", action="store", type=lambda p: Path(p).absolute()
)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def find_patterns(training_inputs, training_outputs):
    output_to_pattern = {}
    for i in range(len(training_outputs)):
        input = training_inputs[i]
        output = training_outputs[i]

        if output not in output_to_pattern:
            output_to_pattern[output] = list(input)
        else:
            output_to_pattern[output] = intersection(
                output_to_pattern.get(output), list(input)
            )

    return {key: "".join(value) for key, value in output_to_pattern.items()}


def read_data_file(training_file):
    with training_file.open() as f:
        training_data = json.load(f)

    return list(training_data.keys()), list(training_data.values())


if __name__ == "__main__":
    args = parser.parse_args()

    training_inputs, training_outputs = read_data_file(args.training_data_file)

    print(find_patterns(training_inputs, training_outputs))
