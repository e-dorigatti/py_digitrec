from math import sqrt
import numpy as np
from common import test_cv, load_digits, map_output
from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.online import online_learn
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def train_network():
    nnet = NeuralNetwork([400, 20, 10])
    train_set, test_set = load_digits('digits.txt')
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]

    online_learn(nnet, train_set, 500, lambda t: 500.0 / (500.0 + t), 2500)
    accuracy, validation_error = test_cv(nnet, test_set)

    return nnet, accuracy

def main():
    print 'learning digits'
    nnet, accuracy = train_network()

    print 'plotting weights'
    hidden_weights = nnet.weights[0]
    
    for i, neuron in enumerate(hidden_weights):
        neuron = neuron[1:] # forget the bias
        size = int(sqrt(len(neuron)))
        neuron = neuron.reshape((size, size)).T

        plt.subplot(5, 4, i + 1)
        # use interpolation='nearest' to clearly see the individual pixels
        plt.imshow(neuron, cmap=cm.Greys_r)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Connection Weigths to Each Neuron of the Hidden Layer'+\
        '\nNetwork Accuracy: {:.1%}'.format(accuracy))
    plt.show()

if __name__ == '__main__':
    main()
