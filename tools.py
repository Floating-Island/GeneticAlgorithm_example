import numpy as np
import copy as copy

def generateWorkspace(rows, columns):
    matrix = np.zeros((rows,columns))
    matrix = [[Node(Position(j,i)) for i in range(columns)] for j in range(rows)]
    return matrix

def nodePosition(node):
    return node.position

def workspacePositions(workspace):
    positions = list()
    for node in workspace:
        positions.append(nodePosition(node))
    return positions

class Node:
    def __init__(self, position):
        self.state = State()
        self.position = position
    def __repr__(self):
        return str(self.position) + str(self.state)
    def goThrough(self):
        fitnessValue = self.state.fitnessValue()
        self.state.changeState()
        return fitnessValue

class Position:
    def __init__(self, row, column):
        self.row = row
        self.column = column
    def __repr__(self):
        return 'row: ' + str(self.row) + ' . column: ' + str(self.column)

class State:
    def __init__(self):
        self.stateType = StateDirty.classInstance()
    def __repr__(self):
        return ' state: ' + str(self.stateType)
    def fitnessValue(self):
        return self.stateType.value()
    def changeState(self):
        self.stateType = self.stateType.changeState()

class StateDirty:
    __uniqueInstance = None
    @staticmethod
    def classInstance():
        if StateDirty.__uniqueInstance == None:
            StateDirty()
        return StateDirty.__uniqueInstance
    def __init__(self):
        if StateDirty.__uniqueInstance != None:
            raise Exception("use classInstance() to get this class instance.")
        else:
            StateDirty.__uniqueInstance = self
        self.fitnessValue = 1
    def __repr__(self):
        return 'dirty. value: ' + str(self.fitnessValue)
    def changeState(self):
        return StateClean.classInstance()
    def value(self):
        return self.fitnessValue

class StateClean:
    __uniqueInstance = None
    @staticmethod
    def classInstance():
        if StateClean.__uniqueInstance == None:
            StateClean()
        return StateClean.__uniqueInstance
    def __init__(self):
        if StateClean.__uniqueInstance != None:
            raise Exception("use classInstance() to get this class instance.")
        else:
            StateClean.__uniqueInstance = self
        self.fitnessValue = -2
    def __repr__(self):
        return 'clean. value: ' + str(self.fitnessValue)
    def changeState(self):
        return self.__uniqueInstance
    def value(self):
        return self.fitnessValue

def geneticAlgorithm(populationQuantity, desiredFitness, workspace, iterations):
    population = generateInitialPopulation(populationQuantity,workspace)
    fitness = Fitness(workspace)
    fitness.addPopulation(population)
    iteration = 0
    while (fitness.bestChromosome())[1] != desiredFitness and iteration < iterations:
        population = newPopulationFrom(population,fitness)
        fitness = Fitness(workspace)
        fitness.addPopulation(population)
        print('\niteration: ' + str(iteration) + '\n')
        print('chromosomes, fitness:\n' + str(fitness.chromosomes))
        print('\nbest chromosome:\n' + str(fitness.bestChromosome()))
        print('workspace:\n' + str(workspace))
        iteration += 1
    return fitness.bestChromosome()

def generateInitialPopulation(quantityOfIndividuals, workSpace):
    population = list()
    for times in range(0,quantityOfIndividuals):
        genes = workspacePositions(np.ndarray.tolist(np.asarray(workSpace).reshape(-1)))
        np.random.shuffle(genes)
        individual = Chromosome(genes)
        population.append(individual)
    return population

class Fitness:
    def __init__(self, workspace):
        self.workspace = workspace
        self.temporalWorkspace = copy.deepcopy(self.workspace)
        self.chromosomes = list()
    def fitnessOf(self, chromosome):
        chromosomeFitness = 0
        self.temporalWorkspace = copy.deepcopy(self.workspace)
        for gene in chromosome.genes:
            if chromosome.genes[-1] != gene:
                nextGene = chromosome.genes[chromosome.genes.index(gene) + 1]
                chromosomeFitness += self.toNextGene(gene, nextGene)
            else:
                chromosomeFitness += self.traverseNodeIn(gene, self.temporalWorkspace)
        return chromosomeFitness
    def toNextGene(self, current, target):
        intermediateFitness = 0
        while equalsPositions(current,target) == False:
            intermediateFitness += self.traverseNodeIn(current, self.temporalWorkspace)

            currentRow = current.row
            targetRow = target.row
            currentColumn = current.column
            targetColumn = target.column

            if currentRow > targetRow:
                current = nodePosition(self.temporalWorkspace[currentRow - 1][currentColumn])
            elif currentRow < targetRow:
                current = nodePosition(self.temporalWorkspace[currentRow + 1][currentColumn])
            elif currentColumn < targetColumn:
                current = nodePosition(self.temporalWorkspace[currentRow][currentColumn + 1])
            elif currentColumn > targetColumn:
                current = nodePosition(self.temporalWorkspace[currentRow][currentColumn - 1])

        return intermediateFitness
    def traverseNodeIn(self, position, workspace):
        return workspace[position.row][position.column].goThrough()
    def addChromosome(self, chromosome):
        chromosomeFitness = self.fitnessOf(chromosome)
        self.chromosomes.append((chromosome, chromosomeFitness))
    def totalFitness(self):
        total = 0
        for chromosome in self.chromosomes:
            total += chromosome[1]
        return total
    def bestChromosome(self):
        if self.chromosomes == None:
            raise Exception("no chromosomes added to Fitness.")
        best = self.chromosomes[0]
        for x in self.chromosomes:
            if x[1] > best[1]:
                best = x
        return best
    def addPopulation(self, population):
        for individual in population:
            self.addChromosome(individual)

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
    def __repr__(self):
        return 'Chromosome genes: ' + str(self.genes) + '\n'

def newPopulationFrom(oldPopulation, oldFitness):
    newPopulation = list()
    while len(newPopulation) != len(oldPopulation):
        selected = list()
        selected = bestChromosomeSelection(oldFitness)
        if doBasedOn(0.85):
            selected = mixedCrossover(selected[0], selected[1], oldFitness)
        for offspring in selected:
            if len(newPopulation) != len(oldPopulation):
                if doBasedOn(0.06):
                        newPopulation.append(mutation(offspring))
                else:
                    newPopulation.append(offspring)
            else:
                break
    return newPopulation

#selection methods:

def stochasticSelection(fitnessObject):
    selected = list()
    bestFitness = (fitnessObject.bestChromosome())[1]
    while len(selected) != 2:
        candidate = fitnessObject.chromosomes[np.random.randint(len(fitnessObject.chromosomes))]
        if bestFitness != 0:
            probability = abs(candidate[1] / bestFitness)
            if doBasedOn(probability):
                selected.append(candidate[0])
        else:
            selected.append(candidate[0])
    return selected

def bestChromosomeSelection(fitnessObject):
    selected = list()
    selected.append((fitnessObject.bestChromosome())[0])
    bestFitness = (fitnessObject.bestChromosome())[1]
    while len(selected) != 2:
        candidate = fitnessObject.chromosomes[np.random.randint(len(fitnessObject.chromosomes))]
        if bestFitness != 0:
            probability = abs(candidate[1] / abs(bestFitness))
            if doBasedOn(probability):
                selected.append(candidate[0])
        else:
            selected.append(candidate[0])
    return selected

#crossover methods:

def crossOver(chromosomeA, chromosomeB):
    fatherChromosome = copy.deepcopy(chromosomeA)
    motherChromosome = copy.deepcopy(chromosomeB)
    result = list()
    for slicePosition in range(0,len(fatherChromosome.genes)):
        if equalsPositions(fatherChromosome.genes[slicePosition], motherChromosome.genes[slicePosition]):
            result = generateOffspring(fatherChromosome, slicePosition, motherChromosome)
            break
    if result == None:
        result = randomCrossover(chromosomeA,chromosomeB)
    return result

def generateOffspring(fatherChromosome, sliceAt, motherChromosome):
    fatherTail = fatherChromosome.genes[sliceAt:]
    motherTail = motherChromosome.genes[sliceAt:]
    fatherHead = fatherChromosome.genes[0:sliceAt]
    motherHead = motherChromosome.genes[0:sliceAt]

    sonGenes = list()
    sonGenes.extend(fatherHead)
    sonGenes.extend(motherTail)
    daughterGenes = list()
    daughterGenes.extend(motherHead)
    daughterGenes.extend(fatherTail)
    offspring = list()
    offspring.append(Chromosome(sonGenes))
    offspring.append(Chromosome(daughterGenes))
    return offspring

def randomCrossover(chromosomeA, chromosomeB):
    fatherChromosome = copy.deepcopy(chromosomeA)
    motherChromosome = copy.deepcopy(chromosomeB)
    slicingPosition = np.random.randint(len(fatherChromosome.genes))
    return generateOffspring(fatherChromosome,slicingPosition,motherChromosome)

def mixedCrossover(chromosomeA, chromosomeB, fitness):
    father = copy.deepcopy(chromosomeA)
    mother = copy.deepcopy(chromosomeB)
    result = list()

    if (fitness.bestChromosome())[1] == 0:
        interchangeProbability = 0.5
    else:
        interchangeProbability = fitness.fitnessOf(mother) / abs((fitness.bestChromosome())[1])

    if doBasedOn(interchangeProbability):
        for eachIndex in range(0,len(father.genes)):
            if doBasedOn(interchangeProbability):
                changeValueOf(father,mother,eachIndex)
            else:
                acceptValueOf(father,mother,eachIndex)
        result.append(father)
        result.append(mother)
    else:
        result = randomCrossover(chromosomeA,chromosomeB)
    return result

def interchangeCrossover(chromosomeA, chromosomeB, fitness):
    father = copy.deepcopy(chromosomeA)
    mother = copy.deepcopy(chromosomeB)
    if (fitness.bestChromosome())[1] == 0:
        interchangeProbability = 0.5
    else:
        interchangeProbability = fitness.fitnessOf(mother) / abs((fitness.bestChromosome())[1])
    result = list()
    for eachIndex in range(0,len(father.genes)):
        if doBasedOn(interchangeProbability):
            changeValueOf(father,mother,eachIndex)
        else:
            acceptValueOf(father,mother,eachIndex)
    result.append(father)
    result.append(mother)
    return result

def tailToHeadCrossover(chromosomeA, chromosomeB):
    fatherChromosome = copy.deepcopy(chromosomeA)
    motherChromosome = copy.deepcopy(chromosomeB)
    result = list()
    for slicePosition in range(0,len(fatherChromosome.genes)):
        if equalsPositions(fatherChromosome.genes[slicePosition], motherChromosome.genes[slicePosition]):
            result = generateInverseOffspring(fatherChromosome, slicePosition, motherChromosome)
            break
    if result == None:
        result = randomCrossover(chromosomeA,chromosomeB)
    return result

def generateInverseOffspring(fatherChromosome, sliceAt, motherChromosome):
    fatherTail = fatherChromosome.genes[sliceAt:]
    motherTail = motherChromosome.genes[sliceAt:]
    fatherHead = fatherChromosome.genes[0:sliceAt]
    motherHead = motherChromosome.genes[0:sliceAt]

    sonGenes = list()
    sonGenes.extend(motherTail)
    sonGenes.extend(fatherHead)
    daughterGenes = list()
    daughterGenes.extend(fatherTail)
    daughterGenes.extend(motherHead)
    offspring = list()
    offspring.append(Chromosome(sonGenes))
    offspring.append(Chromosome(daughterGenes))
    return offspring

#mutation methods:

def mutation(chromosome):
    mutatedChromosome = copy.deepcopy(chromosome)
    geneToMutate = np.random.randint(len(mutatedChromosome.genes))
    mutatesTogene = np.random.randint(len(mutatedChromosome.genes))

    temporalGene = mutatedChromosome.genes[geneToMutate]

    mutatedChromosome.genes[geneToMutate] = mutatedChromosome.genes[mutatesTogene]
    mutatedChromosome.genes[mutatesTogene] = temporalGene
    return mutatedChromosome

#miscellaneous:

def doBasedOn(probability):
    if probability == 1:
        return True
    return (np.random.random() < probability)

def changeValueOf(father,mother, atIndex):
    temporalGene = father.genes[atIndex]
    father.genes[atIndex] = mother.genes[atIndex]
    mother.genes[atIndex] = temporalGene

def acceptValueOf(father,mother, atIndex):
    mother.genes[atIndex] = father.genes[atIndex]

def equalsPositions(aPosition, anotherPosition):
    if(aPosition.row == anotherPosition.row and aPosition.column == anotherPosition.column):
        return True
    else:
        return False
