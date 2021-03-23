from heuristic_lib.algorithms.custom import RedFoxOptimizationAlgorithm
from heuristic_lib.benchmarks import Pinter

# initialize Pinter benchamrk with custom bound
pinterCustom = Pinter(-5, 5)

# we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function

for i in range(100):
    # first parameter takes dimension of problem
    # second parameter is population size
    # third parameter takes the number of function evaluations
    # fourth parameter is benchmark function
    algorithm = RedFoxOptimizationAlgorithm(10, 20 , 10000, pinterCustom)

    # running algorithm returns best found minimum
    best = algorithm.run()

    # printing best minimum
    print(best)


#algorithm = RedFoxOptimizationAlgorithm(10, 20 , 10000, pinterCustom)
#best = algorithm.run()
#print(best)