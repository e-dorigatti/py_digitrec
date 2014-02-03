from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
from common import load_digits, learn_digits

from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print 'loading digits'
    d = load_digits('digits.txt')
    shuffle(d)
    train_set = d[0:4000]
    cv_set = d[4000:]

    # train a neural network to recognize digits
    print 'training network'
    nnet = NeuralNetwork([400, 100, 10], sigmoid, d_dx_sigmoid)

    # best values found empirically (python learning_curves.py)
    nnet, gd = learn_digits(nnet, train_set, cv_set, \
        lambda t: 1000.0 / (2000.0 + t), 10000, debug=False)

    # validate the network and build the data
    print 'validating results'
    data = []
    for digit, image in cv_set:
        correct = [0] * 10
        correct[digit] = 1

        out = nnet.value(image)
        prediction, confidency = max(enumerate(out), key=lambda x: x[1])

        if prediction == digit:
            data.append((1, confidency))
        else:
            data.append((0, confidency))

    acc = len(filter(lambda x: x[0] == 1, data)) / float(len(data))
    print 'accuracy:', acc

    # usual nice plot
    print 'plotting'
    
    corr = [y for x, y in data if x == 1]
    wrong = [y for x, y in data if x == 0]
    plt.hist([corr, wrong], 20)
    plt.grid(True)
    plt.show()
