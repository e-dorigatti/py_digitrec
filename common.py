from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.online import online_learn
from random import randint, shuffle, choice

def load_digits(path):
    """
    Loads the digits from the specified file and returns
    a training set and a test set; the training set has 80%
    of the data.

    Each element is a tuple (digit, input_vector)
    """
    stream = open(path)
    s_data = stream.read()
    stream.close()

    digits = []
    i = 0
    for line in s_data.replace(' ', '').split('\n'):
        if len(line) == 0:
            continue

        digits.append((i, [float(x) for x in line.split(',')]))
        if len(digits) % 500 == 0:
            i += 1

    shuffle(digits)
    split = int(0.8 * len(digits))
    return digits[0:split], digits[split:]

def map_output(i):
    """
    Maps a digit to a vector used to evaluate the network's output;
    digit '0' corresponds to the first neuron, digit '1' to the
    second etc.
    """
    out = [0] * 10
    out[i] = 1
    return out

def prediction(nnet, digit):
    """
    Returns the neural network's prediction for the given digit and
    its 'confidency' as second element of the tuple.
    """
    return max(enumerate(nnet.value(digit)), key = lambda x: x[1])

def test_cv(nnet, cv):
    """
    Tests the neural network against the cross validation set returning
    the accuracy obtained as number_correct / number_samples and the
    average error
    """
    outcome = (prediction(nnet, input)[0] == correct for correct, input in cv)
    errors = (nnet.prediction_error(map_output(correct), nnet.value(input))
        for correct, input in cv)
    return float(sum(outcome)) / len(cv), sum(errors) / len(cv)

def learn_digits(nnet, train_set, cv_set, learning_rate, iterations, step=-1):
    train_set = [(inputs, map_output(digit)) for digit, inputs in train_set]

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

        print 'epoch %d lrate %f training err %f validation err %f accuracy %f' % (
            i, lrate, trainerror, valerror, accuracy)

        return i >= iterations

    online_learn(nnet, train_set, step, learning_rate, stop)

    return nnet, plot_data
