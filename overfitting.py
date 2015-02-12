from py_neuralnet.online import online_learn
from py_neuralnet.neuralnet import NeuralNetwork
from common import load_digits, test_cv, map_output
import matplotlib.pyplot as plt
from sys import argv

def main(size):
    dpi = 96
    plt.figure(figsize=(1280.0/dpi, 720.0/dpi), dpi=dpi)
    plt.suptitle('Overfitting in neural networks')
    plt.subplot(1, 2, 1); plt.title('Errors'); plt.grid(True); plt.ylim(ymin=0)
    plt.subplot(1, 2, 2); plt.title('Acuracy'); plt.grid(True); plt.ylim(ymin=0, ymax=1)
    plt.show(block=False)

    train_set, cv_set = load_digits('digits.txt')
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]

    def stop(i, error, lrate):
        accuracy, valerror = test_cv(nnet, cv_set)

        plt.subplot(1, 2, 1)
        plt.plot(i, error, 'ro', label='Training Error')
        plt.plot(i, valerror, 'b^', label='Validation Error')
        
        plt.subplot(1, 2, 2)
        plt.plot(i, accuracy, 'go', label='Accuracy')
        plt.plot(i, lrate, 'r+', label='Learning Rate')
        plt.draw()

        print i, lrate, error, valerror, accuracy
        return False

    def lrate(i):
        return 10000.0 / (10000.0 + i)

    nnet = NeuralNetwork([400] + size + [10])
    online_learn(nnet, train_set, 500, lrate, stop)

if __name__ == '__main__':
    if len(argv) != 2:
        print 'Please specify network sizes as parameters (hidden layers only)'
        print 'For example, 100x50 will create a network with a 400x100x50x10 architecture'
    else:
        size = [int(n) for n in argv[1].split('x')]
        main(size)
