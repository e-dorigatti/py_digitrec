from math import sqrt
import numpy as np
import common
from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def train_network(train_set, test_set):
    nnet = NeuralNetwork([400, 20, 10], sigmoid, d_dx_sigmoid)
    nnet, data = common.learn_digits(nnet, train_set, test_set,
        lambda t: 10000.0 / (20000.0 + t), 10000, False)

    acc = common.test_cv(nnet, test_set)
    return nnet, acc

if __name__ == '__main__':
    print 'learning digits'
    train_set, test_set = common.load_digits('digits.txt')
    nnet, accuracy = train_network(train_set, test_set)

    print 'plotting weights'
    hidden_weights = nnet.layers[1].weights
    
    for i, neuron in enumerate(hidden_weights):
        neuron = neuron[1:] # forget the bias
        size = int(sqrt(len(neuron)))
        neuron = neuron.reshape((size, size)).T

        plt.subplot(5, 4, i)
        # use interpolation='nearest' to clearly see the individual pixels
        plt.imshow(neuron, cmap=cm.Greys_r)
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Connection Weigths to Each Neuron of the Hidden Layer'+\
        '\nNetwork Accuracy: {:.1%}'.format(accuracy))
    plt.savefig('hidden_weights.png')
    plt.show()
