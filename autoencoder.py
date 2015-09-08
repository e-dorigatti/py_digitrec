"""
This script attempts to create an autoencoder, i.e. a neural network with is able
to reconstruct its input by using an internal representation which has less parameters
than the input neurons. This effectively creates a recap, a compression, of the input
"""
from py_neuralnet.online import online_learn
from py_neuralnet.minibatch import minibatch
from py_neuralnet.neuralnet import NeuralNetwork
from common import load_digits, test_cv, map_output
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sys import argv
from math import sqrt
import random


def main(size):
    train_set, cv_set = load_digits('digits.txt')
    train_set.extend(cv_set)
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]
   
    nnet = NeuralNetwork([400, size, 400])

    epoch_length, i, error = 100, 0, 0.
    while True:
        inputs, _ = random.choice(train_set)
        outputs = nnet.value(inputs)

        lrate = 2000. / (2000. + i)
        error += nnet.backpropagate(inputs, outputs, lrate)

        i += 1
        if i % epoch_length == 0:
            error  /= epoch_length
            print i, error, lrate
            if error < 1.5 or i > 4500:
                break
            else:
                error = 0.

    hidden_weights = nnet.weights[0]
    subplots = int(sqrt(len(hidden_weights)))
    for i, neuron in enumerate(hidden_weights):
        neuron = neuron[1:]
        size = int(sqrt(len(neuron)))
        neuron = neuron.reshape((size, size)).T

        plt.subplot(subplots, subplots + 1, i + 1)
        plt.imshow(neuron, cmap=cm.Greys_r)
        plt.xticks([]); plt.yticks([])
    plt.tight_layout(); plt.show()

    img = random.choice(train_set)[0]
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img).reshape(20, 20).T, interpolation='nearest', cmap=cm.Greys_r)
    plt.subplot(1, 2, 2)
    out = nnet.value(img)
    plt.imshow(out.reshape(20, 20).T, interpolation='nearest', cmap=cm.Greys_r)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(int(argv[1]))
