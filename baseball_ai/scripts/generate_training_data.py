import argparse
import json
import random
import sys
from dataclasses import dataclass
from random import SystemRandom
from typing import Dict

import jsonstream
from marshmallow import Schema, fields, post_load, pre_dump
from rstr import Xeger

parser = argparse.ArgumentParser(description="Create Training Data for baseball-ai")
parser.add_argument("size", type=int)
parser.add_argument(
    "--fixed-input-size", type=int, action="store", required=False, default=None
)


EXCLUDE_ALL = "EXCLUDE_ALL"


@dataclass(frozen=True)
class PatternSpecifier:
    simple: Dict[str, str]
    complex: Dict[str, str]


@dataclass(frozen=True)
class TrainingRecipe:
    alphabet: Dict[str, str]
    patterns: PatternSpecifier


class PatternSpecifierSchema(Schema):
    simple = fields.Dict(keys=fields.String, values=fields.String)
    complex = fields.Dict(keys=fields.String, values=fields.String)

    @post_load
    def __construct(self, data, **kwargs):
        return PatternSpecifier(**data)

    @pre_dump
    def __deconstruct(self, data, **kwargs):
        return data.__dict__


class TrainingRecipeSchema(Schema):
    alphabet = fields.Dict(keys=fields.String(), values=fields.String())
    patterns = fields.Nested(PatternSpecifierSchema)

    @post_load
    def __construct(self, data, **kwargs):
        return TrainingRecipe(**data)

    @pre_dump
    def __deconstruct(self, data, **kwargs):
        return data.__dict__


def main():
    args = parser.parse_args()
    json_stream = jsonstream.load(sys.stdin)

    recipe = TrainingRecipeSchema().load(next(json_stream))
    simple_values = list(recipe.patterns.simple.items())
    valid_patterns = set(
        filter(lambda x: x != EXCLUDE_ALL, list(recipe.patterns.simple.values()))
    )

    complex_values = list(recipe.patterns.complex.items())

    xeger = Xeger(SystemRandom())

    training_data = {}
    alphabet = list(recipe.alphabet.keys())
    while len(training_data) < args.size:
        if len(recipe.patterns.complex) > 0 and random.random() < 0.5:
            choice = random.choice(complex_values)
            training_data[xeger.xeger(choice[1])] = choice[0].upper()
        else:
            choice = random.choice(simple_values)
            include = choice[1]
            exclude_set = (
                valid_patterns - {include} if include != EXCLUDE_ALL else valid_patterns
            )
            if args.fixed_input_size:
                data = xeger.rstr(alphabet, args.fixed_input_size)
            else:
                data = xeger.rstr(alphabet)

            # insert sign
            if include != EXCLUDE_ALL:
                random_idx = random.randint(0, args.fixed_input_size - 1)
                data = data[:random_idx] + include + data[random_idx + len(include):]

            if any([exclude in data for exclude in exclude_set]) or (
                args.fixed_input_size is not None and len(data) != args.fixed_input_size
            ):
                continue
            else:
                training_data[data] = choice[0].upper()

    print(json.dumps(training_data))
