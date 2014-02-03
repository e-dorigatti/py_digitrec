from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
from random import randint, shuffle

def load_digits(path):
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
    return digits

def random_sample(digits):
    return digits[randint(0, len(digits) - 1)]

def map_output(i):
    # maps the output from the correct digit to the vector of
    # outputs for the neural network
    out = [0] * 10
    out[i] = 1
    return out

def prediction(nnet, digit):
    out = nnet.value(digit)
    best = (0, 0)
    for i, y in enumerate(out):
        if y > best[1]:
            best = i, y
    return best

def test_cv(nnet, cv):
    # test the neural network agains the cross validation set
    # returns the accuracy
    correct = 0
    for x in cv:
        d = prediction(nnet, x[1])
        if d[0] == x[0]:
            correct += 1
    return float(correct) / len(cv)

def learn_digits(nnet, train, cv, f_learning_rate, iterations, debug=True):
    graph_data = {
        'iterations': [],
        'training_error': [],
        'validation_accuracy': [],
        'learning_rate': [],
    }

    last = []
    i = 0
    acc = 0
    while i < iterations: #acc < 0.92:
        i += 1
        learning_rate = f_learning_rate(i)
        x = random_sample(train)
        out = map_output(x[0])
        err = nnet.backprop(x[1], out, learning_rate)
        
        if not debug:
            continue

        last.append(err)
        if len(last) >= 100:
            avg_err = float(sum(last)) / len(last)
            acc = test_cv(nnet, cv)

            graph_data['iterations'].append(i)
            graph_data['training_error'].append(avg_err)
            graph_data['validation_accuracy'].append(acc)
            graph_data['learning_rate'].append(learning_rate)
            
            print 'iteration', i, \
                  'learning rate', learning_rate, \
                  'error', float(sum(last)) / len(last), \
                  'accuracy', acc
            last = []

    return nnet, graph_data

