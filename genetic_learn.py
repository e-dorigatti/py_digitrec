from genetic import GeneticAlgorithm
import common

def build_fitness(cv):
    def fitness(nnet):
        return common.test_cv(nnet, cv)

    return fitness

if __name__ == '__main__':
    training_set, cv_set = common.load_digits('digits.txt')
    gen = GeneticAlgorithm(100, [400, 20, 10], build_fitness(cv_set))

    while True:
        gen.evolve()
