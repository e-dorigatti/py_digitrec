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
    bars_corr = 100 * n_corr / count
    bars_wrong = 100 * n_wrong / count

    ticks = np.arange(0.1, 1.01, 0.1)
    width = 0.05

    plt.subplot(1, 2, 1)
    plt.bar(ticks-width/2, bars_corr, width=width, color='g', label='Correct')
    plt.bar(ticks-width/2, bars_wrong, width=width, bottom=bars_corr, \
        color='r',label='Wrong')
    plt.legend(loc='lower right')
    plt.xlim(xmax=1+width)
    plt.xlabel('Network Confidency')
    plt.title('Prediction Correctness (%)')

    plt.subplot(1, 2, 2)
    rel_cnt = count / np.sum(count)
    plt.bar(ticks-width/2, 100*rel_cnt, width=width, color='b')
    plt.title('Sample Distribution (%)')
    plt.xlabel('Network Confidency')
    plt.xlim(xmax=1+width)

    plt.suptitle('Error Analysis - Network Accuracy: {:.1%}'.format(acc))
    plt.grid(True)
    plt.savefig('errors.png')
    plt.show()
