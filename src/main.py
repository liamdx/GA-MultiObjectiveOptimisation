# Multi-Objective Optimisation || Liam Devlin || 22/10/18


from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random
import numpy
import os
from scoop import futures
from copy import copy
import imageio
import warnings
warnings.filterwarnings("ignore")

d = os.getcwd().split("src")
directory = "".join(d)
nrp1_path = "realistic-nrp/nrp-g4.txt"
nrp2_path = "classic-nrp/nrp4.txt"
data_file = open(directory + nrp1_path, "r")
nrp1 = data_file.read()
data_file = open(directory + nrp2_path, "r")
nrp2 = data_file.read()
data_file.close()

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("FitnessSingle", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMulti)
creator.create("SingleIndividual", list, fitness=creator.FitnessSingle)


def parseFile(raw_data):
    raw_data = raw_data.split('\n')
    numberOfLevels = int(raw_data[0])
    requirements = []
    line = 1

    while line in range(1 + (numberOfLevels * 2)):
        currentNumberRequirements = int(raw_data[line])
        rawRequirements = raw_data[line + 1]
        rawRequirements = rawRequirements.split(" ")
        for i in range(currentNumberRequirements):
            requirements.append(int(rawRequirements[i]))
        line = line + 2

    # skip dependencies
    line = line + int(raw_data[line]) + 1
    customers = []
    numberOfCustomers = int(raw_data[line])
    line = line + 1
    endOfFile = line + numberOfCustomers

    while(line < endOfFile):
        rawCustomer = raw_data[line]
        rawCustomer = rawCustomer.split(" ")
        customerValue = int(rawCustomer[0])
        customerRequirements = []
        numberOfRequirements = int(rawCustomer[1])
        for i in range(numberOfRequirements):
            requirementIndex = 2 + i
            customerRequirements.append(int(rawCustomer[requirementIndex]))

        customers.append((customerValue, customerRequirements))
        line = line + 1

    return requirements, customers


def getWeights(customers):
    weights = []
    totalProfit = 0

    for customer in customers:
        totalProfit += customer[0]
        weights.append(customer[0])

    for i in range(len(weights)):
        weights[i] = weights[i] / totalProfit

    return totalProfit, weights


def getValue(requirement, customer):
    customerRequirements = customer[1]
    numberOfCustomerRequirements = len(customerRequirements)
    if requirement in customerRequirements:
        # returns value in range 0-1 based on how important requirement is to customer
        return (numberOfCustomerRequirements - customerRequirements.index(requirement)) / numberOfCustomerRequirements
    else:
       return 0


def getScore(requirement, customers, customerWeights):
    score = 0
    for i in range(len(customers)):
        score += customerWeights[i] * getValue(requirement, customers[i])
    return score


def evaluate(individual, customers, customerWeights, requirements, numberOfRequirements):
    scores, costs = [], []
    for j in range(numberOfRequirements):
        if(individual[j] == 1):
            costs.insert(j, requirements[j])
            scores.insert(j, getScore(requirements[j], customers, customerWeights))
        else:
            costs.insert(j, 0)
            scores.insert(j, 0.0)


    # dot product with requirements for indiviudal final cost and score.
    finalScore = numpy.dot(scores, requirements)
    finalCost = numpy.dot(costs, requirements)
    return (finalScore, finalCost)


def singleEvaluate(individual, customers, customerWeights, requirements, numberOfRequirements, weight):
    # Single objective forumla
    # f(x) = w . f1(x) + (1 - w).f2(x)
    finalScore, finalCost = evaluate(individual, customers, customerWeights, requirements, numberOfRequirements)
    final = numpy.dot(weight, finalScore) + numpy.dot((1 - weight), finalCost)
    # must return two values for deap (expects all fitness functions to be multi-objective formatted)
    return final,


def generateGif(multi_all_gens,random_all_gens, single_last_gen,  nGens, popSize, name, title):
    print("Generating Results Gif")
    images = []
    for j in range(nGens):
        fig = plt.figure()
        canvas = FigureCanvas(fig)
        plt.title("Results for Gen %d" % j)
        plt.suptitle(title)
        plt.xlabel("Score")
        plt.ylabel("Cost")
        plt.gca().invert_yaxis()

        mo = vars()
        ran = vars()
        so = vars()
        for i in range(popSize):
            mo = plt.scatter(multi_all_gens[j][i][0], multi_all_gens[j][i][1], marker='x', color='b')
            ran = plt.scatter(random_all_gens[j][i][0], random_all_gens[j][i][1], marker='+', color='r')

        if(j == nGens - 1):
            for k in range(len(single_last_gen)):
                so = plt.scatter(single_last_gen[k][0], single_last_gen[k][1], marker='d')

        plt.legend((mo, ran, so),
                   ('Multi Objective', 'Random', 'Single Objective'),
                   scatterpoints=1,
                   loc='upper right',
                   ncol=3,
                   fontsize=8)

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        width = int(width)
        height = int(height)
        images.append(numpy.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3))
        if (j == nGens - 1):
            # show final result for 15 frames
            for k in range(15):
                images.append(images[nGens - 1])

    imageio.mimsave(name + ".gif", images)
    plt.close()


def singleObjective(raw_data, numberOfGenerations, populationSize, weight):
    popSize = populationSize
    crossover = 0.75
    mutation = 0.1
    nGens = numberOfGenerations

    requirements, customers = parseFile(raw_data)
    numberOfRequirements = len(requirements)
    totalProfit, customerWeights = getWeights(customers)


    singleObjectiveToolbox = base.Toolbox()
    singleObjectiveToolbox.register("map", futures.map)
    singleObjectiveToolbox.register("attr_bool", random.randint, 0, 1)

    singleObjectiveToolbox.register("individual", tools.initRepeat, creator.SingleIndividual,
                        singleObjectiveToolbox.attr_bool, n=numberOfRequirements)
    singleObjectiveToolbox.register("population", tools.initRepeat, list, singleObjectiveToolbox.individual)


    pop = singleObjectiveToolbox.population(n=popSize)
    singleObjectiveToolbox.register("mate", tools.cxTwoPoint)
    singleObjectiveToolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    singleObjectiveToolbox.register("select", tools.selNSGA2)
    singleObjectiveToolbox.register("evaluate",singleEvaluate,
                                    customers=customers, customerWeights=customerWeights,requirements=requirements,
                                    numberOfRequirements=numberOfRequirements, weight = weight)


    stats = tools.Statistics(lambda individual: individual.fitness.values) 
    stats.register("allGenerations", copy)
    stats.register("max", numpy.max)
    hof = tools.HallOfFame(nGens)

    pop, logbook = algorithms.eaMuPlusLambda(pop, singleObjectiveToolbox, popSize, popSize, crossover, mutation, nGens, stats, hof, True)

    bestIndividual = hof[nGens - 1]
    currentScore, currentCost = evaluate(bestIndividual, customers, customerWeights, requirements, numberOfRequirements)

    return (currentScore, currentCost)




def main(raw_data, numberOfGeneration, populationSize):
    popSize = populationSize
    crossover = 0.75
    mutation = 0.1
    nGens = numberOfGeneration

    requirements, customers = parseFile(raw_data)
    numberOfRequirements = len(requirements)
    totalProfit, customerWeights = getWeights(customers)

    # set up DEAP
    # https://deap.readthedocs.io/en/master/tutorials/basic/part1.html
    # https://deap.readthedocs.io/en/master/overview.html
    # define aliases for fitness function and individual representation
    # create an NSGA2 toolbox
    nsgaToolbox = base.Toolbox()
    # enable multi-threading
    nsgaToolbox.register("map", futures.map)
    nsgaToolbox.register("attr_bool", random.randint, 0 ,1 )
    nsgaToolbox.register("individual", tools.initRepeat, creator.Individual,
                         nsgaToolbox.attr_bool, n = numberOfRequirements)
    nsgaToolbox.register("population", tools.initRepeat, list, nsgaToolbox.individual)
    pop = nsgaToolbox.population(n=popSize)
    nsgaToolbox.register("mate", tools.cxTwoPoint)
    nsgaToolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    nsgaToolbox.register("select", tools.selNSGA2)
    nsgaToolbox.register("evaluate", evaluate, customers=customers, customerWeights=customerWeights,
                         requirements=requirements, numberOfRequirements=numberOfRequirements)


    # Multi objective statistics
    # https://deap.readthedocs.io/en/master/tutorials/basic/part3.html

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    # https://docs.python.org/3/library/copy.html
    # use shallow copy because we do not want to mess with underlying DEAP data, especially considering threading...
    stats.register("allGenerations", copy)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("std", numpy.std, axis = 0)
    stats.register("max", numpy.max, axis = 0)
    logbook = tools.Logbook()


    pop, logbook = algorithms.eaMuPlusLambda(pop, nsgaToolbox, popSize, popSize, crossover, mutation, nGens, stats, hof, True)


    fit_maxs = logbook.select("max")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")
    all_gens = logbook.select("allGenerations")

    return all_gens


def randomGenerate(popSize, numberOfRequirements):
    pop = []
    for i in range(popSize):
        individual = []
        for j in range(numberOfRequirements):
            individual.append(random.randint(0 , 1))
        pop.append(individual)

    return pop


def randomAlgorithm(raw_data, numberOfGenerations, populationSize):
    popSize = populationSize
    nGens = numberOfGenerations
    requirements, customers = parseFile(raw_data)
    numberOfRequirements = len(requirements)
    totalProfit, customerWeights = getWeights(customers)

    finalRandomResults = []
    for i in range(nGens):
        print("Random Algorithm: Gen %d" % i)
        pop = randomGenerate(popSize, numberOfRequirements)
        popResults = []
        for j in range(popSize):
            currentScore, currentCost = evaluate(pop[j], customers, customerWeights, requirements, numberOfRequirements)
            popResults.append((currentScore, currentCost))

        finalRandomResults.append(popResults)

    return finalRandomResults


if __name__ == '__main__':
    numberOfGenerations = 100
    populationSize = 50
    #NRP 1
    SO_NRP1_RESULTS = []
    SO_NRP1_RESULTS.append(singleObjective(nrp1,numberOfGenerations,populationSize,0.1))
    SO_NRP1_RESULTS.append(singleObjective(nrp1, numberOfGenerations, populationSize, 0.25))
    SO_NRP1_RESULTS.append(singleObjective(nrp1, numberOfGenerations, populationSize, 0.4))
    SO_NRP1_RESULTS.append(singleObjective(nrp1, numberOfGenerations, populationSize, 0.55))
    SO_NRP1_RESULTS.append(singleObjective(nrp1, numberOfGenerations, populationSize, 0.7))
    SO_NRP1_RESULTS.append(singleObjective(nrp1, numberOfGenerations, populationSize, 0.9))

    MO_NRP1_RESULTS = main(nrp1, numberOfGenerations, populationSize)
    RAN_NRP1_RESULTS = randomAlgorithm(nrp1, numberOfGenerations, populationSize)
    
    # NRP2
    SO_NRP2_RESULTS = []
    SO_NRP2_RESULTS.append(singleObjective(nrp2,numberOfGenerations,populationSize,0.1))
    SO_NRP2_RESULTS.append(singleObjective(nrp2, numberOfGenerations, populationSize, 0.25))
    SO_NRP2_RESULTS.append(singleObjective(nrp2, numberOfGenerations, populationSize, 0.4))
    SO_NRP2_RESULTS.append(singleObjective(nrp2, numberOfGenerations, populationSize, 0.55))
    SO_NRP2_RESULTS.append(singleObjective(nrp2, numberOfGenerations, populationSize, 0.7))
    SO_NRP2_RESULTS.append(singleObjective(nrp2, numberOfGenerations, populationSize, 0.9))

    MO_NRP2_RESULTS = main(nrp2, numberOfGenerations, populationSize)
    RAN_NRP2_RESULTS = randomAlgorithm(nrp2, numberOfGenerations, populationSize)

    generateGif(MO_NRP1_RESULTS,RAN_NRP1_RESULTS,SO_NRP1_RESULTS,numberOfGenerations, populationSize, "NRP G4 Results","NRP G4 Results")
    generateGif(MO_NRP2_RESULTS,RAN_NRP2_RESULTS,SO_NRP2_RESULTS,numberOfGenerations, populationSize, "NRP4 Results","NRP4 Results")