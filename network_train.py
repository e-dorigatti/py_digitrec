from py_neuralnet.online import online_learn
from py_neuralnet.minibatch import minibatch
from py_neuralnet.neuralnet import NeuralNetwork
from common import load_digits, test_cv, map_output
import matplotlib.pyplot as plt
import click
import pickle


@click.command()
@click.argument('filename', type=click.File('w'))
@click.option('-e', '--training-epochs', default=-1., help='Train for # epochs')
@click.option('-t', '--training-error', default=-1.,
              help='Stop when training error is less than this')
@click.option('-V', '--validation-error', default=-1.,
              help='Stop when validation error is less than this')
@click.option('-A', '--accuracy', default=-1.,
              help='Stop when the accuracy is larger than this')
@click.option('-h', '--hidden-units', default=[20], multiple=True, type=click.INT,
              help='Number of units in the hidden layer. Specify multiple times'
                   ' for multiple layers')
@click.option('-s', '--batch-size', default=200)
@click.option('-c', '--regularization', default=1.)
@click.option('-p', '--print-interval', default=500,
              help='Print status every number epochs')
@click.option('-a', '--lrate-a', default=2000.)
@click.option('-b', '--lrate-b', default=2000.)
@click.option('-v', '--verbose/--no-verbose', default=True,
              help='Print training process in csv form')
def main(filename, training_epochs, training_error, validation_error, accuracy,
         hidden_units, batch_size, regularization, print_interval, lrate_a, lrate_b,
         verbose):
    """ Trains a neural network to recognize digits.
    """

    train_set, cv_set = load_digits('digits.txt', normalize=True)
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]

    def stop(i, error, lrate):
        actual_acc, actual_val = -1, -1
        if verbose and i % print_interval == 0:
            actual_acc, actual_val = test_cv(nnet, cv_set)
            print ';'. join(map(str, (i, i*batch_size, error, 
                                     actual_val, actual_acc, lrate)))

        return (training_epochs > 0. and i >= training_epochs) or \
            error < training_error or \
            (actual_val > 0 and validation_error > 0 and actual_val < validation_error) \
            or (actual_acc > 0 and accuracy > 0 and actual_acc > accuracy)

    def lrate(i):
        return lrate_a / (lrate_b + i * batch_size)

    nnet = NeuralNetwork([400] + list(hidden_units) + [10])

    try:
        minibatch(nnet, train_set, batch_size, lrate, regularization, stop)
    except KeyboardInterrupt:
        pass

    pickle.dump(nnet, filename)


if __name__ == '__main__':
    main()
