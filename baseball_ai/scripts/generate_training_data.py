import sys
import jsonstream

def main():
    json_stream = jsonstream.load(sys.stdin)

    pattern_definition = next(json_stream)
    print(pattern_definition)

