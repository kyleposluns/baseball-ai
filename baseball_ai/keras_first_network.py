from numpy import loadtxt
from argparse import ArgumentParser
import numpy as np
import json
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense


# To run: python keras_first_network.py
parser = ArgumentParser(description='train model')
parser.add_argument('training_data', action='store', type=lambda p: Path(p).absolute())

def load_tester(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

# load data
# TODO determine how to load json into numpy



dataset = loadtxt('training.json', delimiter=',')
#split into input (X) and output (y) variables
x = None
y = None

# define the keras model
model = Sequential()
model.add(Dense(5, input_dim=10, activation='sigmoid'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

# compile keras model
# loss fn= 
# adam= stochastic gradient descent algo
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(x, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.training_data

    data = load_tester(data_path)
    print(data)

