from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.online import online_learn
from common import load_digits, map_output, test_cv
from random import shuffle
import matplotlib.pyplot as plt
import multiprocessing as mp
from sys import argv
import time, Queue


def train_network(args):
    size, (lrate_a, lrate_b), train_set, cv_set, iterations, subplot, name = args
    lrate = (lambda i: lrate_a / (i + lrate_b)) if lrate_b else (lambda i:lrate_a)

    plot_data = {
        'iterations': [],
        'training_error': [],
        'validation_error': [],
        'validation_accuracy': [],
        'learning_rate': [],
    }

    def stop(i, trainerror, lrate):
        accuracy, valerror = test_cv(nnet, cv_set)
        plot_data['iterations'].append(i)
        plot_data['training_error'].append(trainerror)
        plot_data['validation_error'].append(valerror)
        plot_data['validation_accuracy'].append(accuracy)
        plot_data['learning_rate'].append(lrate)

        print '%s - iteration %d learning rate %f accuracy %f' % (
            name, i, lrate, accuracy)

        return i >= iterations


    nnet = NeuralNetwork([400] + size + [10])
    online_learn(nnet, train_set, 100, lrate, stop)

    return plot_data, subplot, name

def plot_subplot(index, plot_data, iterations):
    plt.subplot(2, 3, index)

    for net_size, data in sorted(plot_data.iteritems()):
        plt.plot(data['iterations'], data['validation_accuracy'],
            label=net_size + ' Neurons')
        plt.plot(data['iterations'], data['learning_rate'], color='black',
            linestyle='dashed')

    plt.xticks([i for i in range(0, iterations+1, 5000)])
    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)

def main(sizes):
    plt.figure(figsize=(1280/96.0, 720/96.0), dpi=96)
    train_set, cv_set = load_digits('digits.txt')
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]
    iterations = 25000

    # associates each subplot with the learning rate parameters displayed in it
    # subplot: ( learning rate A, learning rate B )
    # learning rate = learning rate A / (t + learning rate B) if learning rate B != 0
    # else learning rate = learning rate A
    learning_rates = {
        1: (0.05, 0),
        2: (10000.0, 20000.0),
        3: (10000.0, 10000.0),
        4: (1000.0, 2000.0),
        5: (500.0, 500.0),
        6: (0.5, 0),
    }

    # create the parameter list to submit to the processes
    tasks = list()
    for subplot, lrate in learning_rates.iteritems():
        for size in sizes:
            name = 'x'.join(str(x) for x in size)
            tasks.append((size, lrate, train_set, cv_set, iterations, subplot, name))
    
    # run the processes, train the network and collect plot data
    pool = mp.Pool()
    results = pool.imap(train_network, tasks)

    # aggregate the data from the processes
    plot_data = dict()
    for data, subplot, name in results:
        if not subplot in plot_data:
            plot_data[subplot] = dict()
        plot_data[subplot][name] = data

    # and plot
    for subplot, data in plot_data.iteritems():
        plot_subplot(subplot, data, iterations)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90)
    plt.legend(loc='lower right')
    plt.suptitle('Comparing Network Sizes and Learning Rates')
    plt.show()

    print 'Finished'

if __name__ == '__main__':
    if len(argv) == 1:
        print 'Please specify network sizes as parameters (hidden layers only)'
        print 'For example, 100x50 will create a network with a 400x100x50x10 architecture'
    else:
        sizes = [[int(n) for n in e.split('x')] for e in argv[1:]]
        main(sizes)
