from tools import *

#workspace generation test:

print('\nworkspace generation test:\n')
rows = int(input("Number of rows in matrix: "))
columns = int(input("Number of columns in matrix: "))
matrix = generateWorkspace(rows, columns)
print(matrix)

#state change test:

print('\nstate mutation test:\n')
matrix[0][0].goThrough()
matrix[0][1].goThrough()
matrix[0][1].goThrough()
print("Rows: " + str(len(matrix)) + " Columns: " + str(len(matrix[0])))
print(matrix)

#chromosome representation test:

print('\nchromosome representation test:\n')
positionsArray = np.ndarray.tolist(np.asarray(matrix).reshape(-1))
individual = Chromosome(workspacePositions(positionsArray))


print(individual)

print('Individual' + str(positionsArray[2]))

#initial population test:

print('\ninitial population test:\n')
quantityOfIndividuals = int(input("Ingrese la cantidad de individuos: "))
initialPopulation = generateInitialPopulation(quantityOfIndividuals, matrix)

print(initialPopulation)

#fitness test:

print('\nfitness test:\n')
fitnessObject = Fitness(generateWorkspace(rows, columns))

fitnessValue = fitnessObject.fitnessOf(individual)
print("individual: " + str(individual) + '\n')
print("the fitness of the individual is: "+ str(fitnessValue))

fitnessObject.addChromosome(individual)
fitnessObject.addChromosome(individual)
print("the total fitness is: "+ str(fitnessObject.totalFitness()))

#crossover test:

print('\ncrossover test:\n')
father = Chromosome(workspacePositions(np.ndarray.tolist(np.asarray(matrix).reshape(-1))))
mother = copy.deepcopy(father)
mother.genes[0].row = 1
mother.genes[4].row = 1000

print('father:' + str(father))
print('mother:' + str(mother))
offspring = crossOver(father, mother)
print('son:' + str(offspring[0]))
print('daughter:' + str(offspring[1]))

#mutation test:

print('\nmutation test:\n')
originalChromosome = father
print('original chromosome:' + str(originalChromosome))
mutatedChromosome = mutation(originalChromosome)
print('mutated:' + str(mutatedChromosome))

#selection and best chromosome test:

print('\nselection and best chromosome test:\n')
population = generateInitialPopulation(3, matrix)
print('population: ' + str(population))
anotherFitnessObject = Fitness(matrix)
anotherFitnessObject.addChromosome(population[0])
anotherFitnessObject.addChromosome(population[1])
anotherFitnessObject.addChromosome(population[2])

print('\nselection:\n')
selected = stochasticSelection(anotherFitnessObject)
print('firstSelected:' + str(selected[0]) + '\n' + 'secondSelected:' + str(selected[1]))


print('chromosomes, fitness:\n' + str(anotherFitnessObject.chromosomes))
print('\nbest chromosome:\n' + str(anotherFitnessObject.bestChromosome()))

#newPopulation test (uses selection and best chromosome test):

print('old population: ' + str(population))

print('new population: ' + str(newPopulationFrom(population,anotherFitnessObject)))