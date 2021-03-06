"""
This script attempts to reproduce a given digit using a trained neural network
and maximizing the likelihood of that digit by performing gradiend descent
on a randomly generated noise image
"""
from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.online import online_learn
from common import load_digits, map_output, test_cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sys import argv


def train_network():
    train_set, test_set = load_digits('digits.txt')
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]

    nnet = NeuralNetwork([400, 20, 10])
    online_learn(nnet, train_set, 500, lambda t: 500.0 / (500.0 + t), 2500)

    accuracy, validation_error = test_cv(nnet, test_set)

    return nnet, accuracy


def main(target_digit):
    target = np.array([map_output(target_digit)]).T

    print 'training'
    nnet, accuracy = train_network()
    print 'accuracy', accuracy

    img = np.random.random((1, 20 * 20))

    for i, _ in enumerate(iter(lambda: 0, 1)):
        val = nnet.value(img)
        _ = nnet.calculate_gradients(target, val)
        img += 1 * nnet._input_deltas.T
        print i, ';'.join('%.2f' % x for x in val.T.tolist()[0]), \
              np.average(nnet._input_deltas)

        if abs(np.average(nnet._input_deltas)) < 1e-05 or i > 1000:
           break

    plt.imshow(img.reshape(20, 20), cmap=cm.Greys_r, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    if len(argv) < 2:
        print 'specify a digit to reconstruct'
    else:
        main(int(argv[1]))
