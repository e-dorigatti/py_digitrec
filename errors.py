from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
from common import load_digits, learn_digits

from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    print 'loading digits'
    train_set, cv_set = load_digits('digits.txt')

    # train a neural network to recognize digits
    print 'training network'
    nnet = NeuralNetwork([400, 20, 10], sigmoid, d_dx_sigmoid)

    # best values found empirically (python learning_curves.py)
    nnet, gd = learn_digits(nnet, train_set, cv_set, \
        lambda t: 10000.0 / (20000.0 + t), 10000)

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
    print 'plotting'

    corr = [y for x, y in data if x == 1]
    wrong = [y for x, y in data if x == 0]

    n_corr, bins = np.histogram(corr, 10)
    n_wrong, bins = np.histogram(wrong, 10)

    count = np.array([c + w for c, w in zip(n_corr, n_wrong)], dtype=np.float)

    ticks = np.arange(0.0, 1.0, 0.1)
    width = 0.05

    plt.subplot(1, 2, 1)
    plt.bar(ticks, n_corr / count, width=width, color='b', label='Prediction Correctness')
    plt.legend(loc='upper left')
    plt.xlabel('Network Confidency')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(ticks, n_corr, width=width, color='g', label='Correct')
    plt.bar(ticks, n_wrong, width=width, color='r', label='Wrong', bottom=n_corr)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('Network Confidency')

    plt.suptitle('Error Analysis over {} Examples\nNetwork Accuracy: {:.1%}'\
        .format(len(data), acc))
    plt.savefig('errors.png')
    plt.show()
