import random as rnd
import copy
from heuristic_lib.benchmarks.utility import Utility

__all__ = ['GeneticAlgorithm']


class Chromosome:
    def __init__(self, D, LB, UB):
        self.D = D
        self.LB = LB
        self.UB = UB

        self.Solution = []
        self.Fitness = float('inf')
        self.generateSolution()

    def generateSolution(self):
        self.Solution = [self.LB + (self.UB - self.LB) * rnd.random()
                         for _i in range(self.D)]

    def evaluate(self):
        self.Fitness = Chromosome.FuncEval(self.D, self.Solution)

    def repair(self):
        for i in range(self.D):
            if self.Solution[i] > self.UB:
                self.Solution[i] = self.UB
            if self.Solution[i] < self.LB:
                self.Solution[i] = self.LB

    def __eq__(self, other):
        return self.Solution == other.Solution and self.Fitness == other.Fitness

    def toString(self):
        print([i for i in self.Solution])


class GeneticAlgorithm:
    r"""Implementation of Genetic algorithm.

    **Algorithm:** Genetic algorithm

    **Date:** 2018

    **Author:** Uros Mlakar

    **License:** MIT
    """

    def __init__(self, D, NP, nFES, Ts, Mr, gamma, benchmark):
        r"""**__init__(self, D, NP, nFES, Ts, Mr, gamma, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            NP {integer} -- population size

            nFES {integer} -- number of function evaluations

            Ts {integer} -- tournament selection

            Mr {decimal} -- mutation rate

            gamma {decimal} -- minimum frequency

            benchmark {object} -- benchmark implementation object

        Raises:
            TypeError -- Raised when given benchmark function which does not exists.

        """
        self.benchmark = Utility().get_benchmark(benchmark)
        self.NP = NP  # population size; number of search agents
        self.D = D  # dimension of the problem
        self.Ts = Ts  # tournament selection
        self.Mr = Mr  # mutation rate
        self.gamma = gamma  # minimum frequency
        self.Lower = self.benchmark.Lower  # lower bound
        self.Upper = self.benchmark.Upper  # upper bound
        self.Population = []
        self.nFES = nFES  # number of function evaluations
        self.FEs = 0  # function evaluations
        self.Done = False
        Chromosome.FuncEval = staticmethod(self.benchmark.function())

        self.Best = Chromosome(self.D, self.Lower, self.Upper)

    def checkForBest(self, pChromosome):
        """Check best solution."""
        if pChromosome.Fitness <= self.Best.Fitness:
            self.Best = copy.deepcopy(pChromosome)

    def TournamentSelection(self):
        """Tournament selection."""
        indices = list(range(self.NP))
        rnd.shuffle(indices)
        tPop = []
        for i in range(self.Ts):
            tPop.append(self.Population[indices[i]])
        tPop.sort(key=lambda x: x.Fitness)

        self.Population.remove(tPop[0])
        self.Population.remove(tPop[1])
        return tPop[0], tPop[1]

    def CrossOver(self, parent1, parent2):
        """Crossover."""
        alpha = [-self.gamma + (1 + 2 * self.gamma) * rnd.random()
                 for i in range(self.D)]
        child1 = Chromosome(self.D, self.Lower, self.Upper)
        child2 = Chromosome(self.D, self.Lower, self.Upper)
        child1.Solution = [alpha[i] * parent1.Solution[i] +
                           (1 - alpha[i]) * parent2.Solution[i] for i in range(self.D)]
        child2.Solution = [alpha[i] * parent2.Solution[i] +
                           (1 - alpha[i]) * parent1.Solution[i] for i in range(self.D)]
        return child1, child2

    def Mutate(self, child):
        """Mutation."""
        for i in range(self.D):
            if rnd.random() < self.Mr:
                sigma = 0.20 * float(child.UB - child.LB)
                child.Solution[i] = min(
                    max(rnd.gauss(child.Solution[i], sigma), child.LB), child.UB)

    def init(self):
        """Initialize population."""
        for i in range(self.NP):
            self.Population.append(Chromosome(self.D, self.Lower, self.Upper))
            self.Population[i].evaluate()
            self.checkForBest(self.Population[i])

    def tryEval(self, c):
        """Check evaluations."""
        if self.FEs < self.nFES:
            self.FEs += 1
            c.evaluate()
        else:
            self.Done = True

    def run(self):
        """Run."""
        self.init()
        self.FEs = self.NP
        while not self.Done:
            for _k in range(int(self.NP / 2)):
                parent1, parent2 = self.TournamentSelection()
                child1, child2 = self.CrossOver(parent1, parent2)

                self.Mutate(child1)
                self.Mutate(child2)

                child1.repair()
                child2.repair()

                self.tryEval(child1)
                self.tryEval(child2)

                tPop = [parent1, parent2, child1, child2]
                tPop.sort(key=lambda x: x.Fitness)
                self.Population.append(tPop[0])
                self.Population.append(tPop[1])

            for i in range(self.NP):
                self.checkForBest(self.Population[i])
        return self.Best.Fitness
