from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
from common import load_digits, learn_digits

from random import randint, shuffle
import matplotlib.pyplot as plt
import numpy as np
from sys import argv

def compare_learning_rates(n, train_set, cv_set):
    iterations = 15000

    errors = []
    learning_rates = [ 
        (1, 'magenta', lambda t: 1000.0 / (1000.0 + t)),
        (2, 'green', lambda t: 1000.0 / (2000.0 + t)),
        (3, 'blue', lambda t: 10000.0 / (20000.0 + t)),
        (4, 'black', lambda t: 10000.0 / (50000.0 + t)),
        (5, 'red', lambda t: 0.1),
    ]

    dpi=96
    plt.figure(figsize=(1280.0/dpi, 720.0/dpi), dpi=dpi)

    for subplot, color, lr in learning_rates:
        nnet = NeuralNetwork([400, n, 10], sigmoid, d_dx_sigmoid)
        nnet, d = learn_digits(nnet, train_set, cv_set, lr, iterations)

        errors.append((color, d['iterations'], d['training_error']))
        plt.subplot(2, 3, subplot)
        plt.plot(d['iterations'], d['validation_accuracy'], \
            label='Validation Accuracy', color=color)
        plt.plot(d['iterations'], d['learning_rate'], \
            label='Learning Rate', color=color)
        plt.xticks(range(0,iterations+1,5000))
        plt.xlim(xmin=0, xmax=iterations)
        plt.ylim(ymin=0, ymax=1)
        plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.title('Training Errors')
    for color, its, err in errors:
        plt.plot(its, err, color=color)
        plt.xticks(range(0,iterations+1,5000))
        plt.xlim(xmin=0, xmax=iterations)
        plt.ylim(ymin=0, ymax=10)
        plt.grid(True)

    plt.suptitle('Comparing Learning Rates For a 400 x %d x 10 Neurons Network' % n)
    plt.savefig('%dneur_%sk.png' % (n, str(iterations)[0:-3]))
    plt.show()

if __name__ == '__main__':
    if len(argv) == 1:
        print 'Please specify network sizes as parameters'
    else:
        print 'loading digits...'
        d = load_digits('digits.txt')
        shuffle(d)
        train_set = d[0:4000]
        cv_set = d[4000:]

        for ns in [int(x) for x in argv[1:]]:
            print ns, 'neurons'
            compare_learning_rates(ns, train_set, cv_set)
        
