from tools import *

# algorithm test:
print("this genetic algorithm tries to search for the best path to fill a matrix\n")
rows = int(input("Quantity of rows in matrix: "))
columns = int(input("Quantity of columns in matrix: "))
space = generateWorkspace(rows, columns)
print(space)
populationQuantity = int(input("population size: "))
desiredFitness = int(input("wished fitness: "))
iterations = int(input("number of iterations: "))

bestChromosome = geneticAlgorithm(populationQuantity,desiredFitness,space, iterations)

print('\nbest chromosome found:\n' + str(bestChromosome[0]) + 'with fitness: ' + str(bestChromosome[1]))
