from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
from common import load_digits, learn_digits

from random import shuffle
import matplotlib.pyplot as plt
from sys import argv

iterations = 15000

def compare_size(sizes, f_learning_rate, train_set, cv_set):
    plot_data = {}
    for color, n in sizes:
        nnet = NeuralNetwork([400] + n + [10], sigmoid, d_dx_sigmoid)
        nnet, data = learn_digits(nnet, train_set, cv_set, f_learning_rate, \
            iterations, 100)
        plot_data[str(n)] = data
        plot_data[str(n)]['color'] = color
    return plot_data

def plot_subplot(index, plot_data):
    plt.subplot(2, 3, index)

    for key, value in sorted(plot_data.iteritems()):
        plt.plot(value['iterations'], value['validation_accuracy'], \
            color = value['color'], label = str(key) + ' Neurons')
        plt.plot(value['iterations'], value['learning_rate'], \
            color = 'black', linestyle = 'dashed')

    plt.xticks([i for i in range(0, iterations+1, 5000)])
    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)

if __name__ == '__main__':
    plt.figure(figsize=(1280/96.0, 720/96.0), dpi=96)
    
    train_set, cv_set = load_digits('digits.txt')

    sizes = [
        ('red', [20]),
        ('green', [100, 10]),
        ('blue', [200]) ]
    learning_rates = [ 
        (1, lambda t: 0.05),
        (2, lambda t: 10000.0 / (50000.0 + t)),
        (3, lambda t: 10000.0 / (20000.0 + t)),
        (4, lambda t: 1000.0 / (5000.0 + t)),
        (5, lambda t: 1000.0 / (2000.0 + t)),
        (6, lambda t: 1000.0 / (1000.0 + t)) ]

    for subplot, learning_rate in learning_rates:
        print '--------------------------------------------------------------'
        print '--- ', subplot
        print '--------------------------------------------------------------'
        plot_data = compare_size(sizes, learning_rate, train_set, cv_set)
        plot_subplot(subplot, plot_data)

    plt.legend(loc='lower right')
    plt.suptitle('Comparing Network Sizes and Learning Rates')
    plt.savefig('comparison_' + str(iterations)[0:-3] + 'k.png')
    plt.show()

