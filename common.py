from py_neuralnet.neuralnet import NeuralNetwork
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
    the accuracy obtained as number_correct / number_samples
    """
    outcome = [prediction(nnet, input)[0] == correct for correct, input in cv]
    return float(sum(outcome)) / len(cv)

def learn_digits(nnet, train_set, cv_set, learning_rate, iterations, step=-1):
    """
    Attempts to teach the given digits (train_set) to the neural network
    running for the specified iterations. Returns the neural network and,
    if the 'debug mode' is activated (see below), some statistics about
    the training as a dictionary whose keys are strings and values
    are lists (to ease plotting the data).
    
    The digits must have the same format as those returned from load_digits:
    (digit, input_vector).

    cv_set is the cross validation set and is used in 'debug mode' to
    periodically test the network's performances. It can be None if
    you do not wish to do this.

    step is the step size in 'debug mode'. Every -steps- iterations the
    network is tested and various data is collected.

    The 'debug mode' is activated if cv_set is not none and steps is
    greater than 0. In this case the network is periodically tested
    and the following statistics are collected in a dictionary:
    iteration number, training error, validation accuracy and learning
    rate.

    learning_rate can be either a number or a function. In the latter case
    it will be called at every iteration to compute the learning rate
    to use for that particular iteration; therefore, it must accept
    one int parameter, the iteration, and return a number.
    """
    graph_data = {
        'iterations': [],
        'training_error': [],
        'validation_accuracy': [],
        'learning_rate': [],
    }

    debug = step > 0 and cv_set is not None
    error_accumulator, i= 0.0, 0
    train_set = [(map_output(digit), input) for digit, input in train_set]
    f_lr = learning_rate
    if isinstance(learning_rate, (int, long, float)):
        f_lr = lambda x: learning_rate

    while i < iterations:
        i += 1
        learning_rate = f_lr(i)
        correct, input = choice(train_set)
        error_accumulator += nnet.backprop(input, correct, learning_rate)
        
        if debug and i % step == 0:
            avg_error = error_accumulator / step
            error_accumulator = 0

            accuracy = test_cv(nnet, cv_set)

            graph_data['iterations'].append(i)
            graph_data['training_error'].append(avg_error)
            graph_data['validation_accuracy'].append(accuracy)
            graph_data['learning_rate'].append(learning_rate)
    
            print 'iteration', i, \
                  'learning rate', learning_rate, \
                  'error', avg_error, \
                  'accuracy', accuracy
    return nnet, graph_data
