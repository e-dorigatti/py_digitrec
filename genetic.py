from random import uniform, randint, random, randrange, choice
from py_neuralnet.neuralnet import NeuralNetwork
from py_neuralnet.utils import sigmoid, d_dx_sigmoid
import numpy as np

class GeneticAlgorithm:
    def __init__(self, pop_size, nnet_arch, fitness):
        self.fitness = fitness
        self.population = [NeuralNetwork(nnet_arch, sigmoid, d_dx_sigmoid)
            for i in range(pop_size)]

    def probabilistic_choice(self, score_list):
        """
        Chooses an element from scores (a list of tuples)
        with a probability proportional to the first element
        of the tuple.
        """
        total, cumulatives = 0, []
        for score, individual in score_list:
            total += score
            cumulatives.append((score, total, individual))

        choice = uniform(0, cumulatives[-1][1])
        for score, cumul, individual in cumulatives:
            if cumul > choice:
                return score, individual

    def mutation(self, nnet):
        layer = choice(nnet.layers[1:]).weights
        mut_x = randint(0, layer.shape[0] - 1)
        mut_y = randint(0, layer.shape[1] - 1)
        layer[mut_x, mut_y] = uniform(-2, 2)

    def slice_layer(self, layer, col, row):
        a = layer[0:row, 0:col]
        b = layer[0:row, col:]
        c = layer[row:, 0:col]
        d = layer[row:, col:]
        
        return a, b, c, d

    def create_layer(self, a, b, c, d):
        top = np.hstack((a, b))
        bottom = np.hstack((c, d))
        whole = np.vstack((top, bottom))
        return whole

    def combine(self, parent1, parent2):
        # ignore input layer as it has no weights
        layer_n = randint(1, len(parent1.layers) - 1)
        layer1 = parent1.layers[layer_n]
        layer2 = parent2.layers[layer_n]
       
        layer1_w = layer1.weights
        layer2_w = layer2.weights
    
        slice_row = randint(0, layer1_w.shape[0] - 1)
        slice_col = randint(0, layer1_w.shape[1] - 1)

        slice1_a, slice1_b, slice1_c, slice1_d = \
            self.slice_layer(layer1_w, slice_row, slice_col)
        slice2_a, slice2_b, slice2_c, slice2_d = \
            self.slice_layer(layer2_w, slice_row, slice_col)

        layer1.weights = self.create_layer( \
            slice1_a, slice2_b, slice2_c, slice1_d)
        layer2.weights = self.create_layer( \
            slice2_a, slice1_b, slice1_c, slice2_d)

    def evolve(self):
        scores = [(self.fitness(individual), individual)
            for individual in self.population]

        for i in self.population:
            fitness1, parent1 = self.probabilistic_choice(scores)
            fitness2, parent2 = self.probabilistic_choice(scores)
            
            self.combine(parent1, parent2)        

        #return max([score for score, individual in scores])

        s = sorted([s for s, i in scores])
        print s[0], s[-1]
        return s[-1]


def test_probabilistic_choice():
    a = [(1, 'a'),(2, 'b'),(3, 'c'),(4, 'd'),(5, 'e'),]

    choices = {}
    g = GeneticAlgorithm(10, [10,10,10], None)

    for i in range(500000):
        x = g.probabilistic_choice(a)
        if not x in choices.keys():
            choices[x] = 0
        choices[x] += 1

    print 'These numbers should be equal'
    v0 = float(choices[a[0]])
    for k, v in choices.iteritems():
        print '%f - %f' % (k[0], v / v0)

if __name__ == '__main__':
    test_probabilistic_choice()
