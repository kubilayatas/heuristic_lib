from unittest import TestCase

from heuristic_lib.algorithms.custom import RedFoxOptimizationAlgorithm


class MyBenchmark:

    def __init__(self):
        self.Lower = -5.12
        self.Upper = 5.12

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class RFOTestCase(TestCase):

    def setUp(self):
        self.rfo = RedFoxOptimizationAlgorithm(
            10, 20, 10000, MyBenchmark())

        self.rfo_griewank = RedFoxOptimizationAlgorithm(
            10, 20, 10000, 'griewank')

    def test_works_fine(self):
        self.assertTrue(self.fa.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.fa_griewank.run())
