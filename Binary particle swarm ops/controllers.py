# ===============================================================================
# @author: Daniel V. Stankevich
# @organization: RMIT, School of Computer Science, 2012
#
#
# This package contains all PSO controllers
# ===============================================================================


# from Models import SwarmModel
from Models import ParticleModel
from Models import NeighbourhoodModel

import scipy.spatial as spp
import numpy as np
# import matplotlib as plt
from random import *
# from math import *
import math as mt


# ===============================================================================
# Particle controller
# ===============================================================================
class ParticleController:
    _solution = None

    def __init__(self, solution):
        self._solution = solution

    def initParticle(self, model):
        # Create position array'
        machine1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        machine2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        machine3 = np.mod(np.multiply(np.random.randint(0, 2, size=12), 1), 2)

        machine4 = np.mod(np.multiply(np.random.randint(0, 2, size=12), 1), 2)

        model._position = np.array([machine1, machine2, machine3, machine4])
        # Create Velocity array
        model._velocity = np.zeros([4, 12])

        # Save best Position so far as current Position
        model._bestPosition = model._position

        self.updateFitness(model)

    def updateFitness(self, model):
        # Get Differences of vector
        diff = np.subtract(model._position, self._solution)
        # Get Norm of diff vector
        newFitness = np.linalg.norm(diff)
        # Save it as best position if its better than previous best
        if newFitness < model._fitness or model._fitness is None:
            model._bestPosition = np.copy(model._position)
            model._fitness = newFitness

    def updatePosition(self, model, swarm):
        # VELOCITY NEEDS TO BE CONSTRICTED WITH VMAX
        # Get random coefficients e1 & e2
        w = model._inertia
        Dist = 0
        Gdist = 0

        Cvelocity = 0
        PPLC = 0
        CRMC = 0
        CCLC = 0
        velocity = 0

        # Apply equation to each component of the velocity, add it to corresponding position component


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


# ===============================================================================
# Binary Particle controller
# ================== =============================================================
Energy_cost_PTI = np.array(
    [214.7, 214.7, 214.7, 301.1, 301.1, 301.1, 301.1, 301.1, 301.1, 395.7, 395.7,
     395.7])
e1 = np.random.rand()
e2 = np.random.rand()
e3 = np.random.rand()
c1 = 3
c2 = 3
c3 = 2
vmax = 6
velocity = 0
c = 1
CCLC = 0
PPLC = 0
CRMC = 0
Nzeros = np.zeros([4, 12])
Nones = np.zeros([4, 12])
Vzeros = np.zeros([4, 12])
Vones = np.zeros([4, 12])
Gzeros = np.zeros([4, 12])
Gones = np.zeros([4, 12])
ARP = 0
BAF = 0
CGL = randint(2721, 3610)
CCL = randint(255, 1755)
PPL = randint(82, 399)
CRM = randint(3612, 6345)
MaxDID = 0
Fitcalcarr = np.zeros([4,12])


class BinaryParticleController:
    _solution = None

    def __init__(self, solution, model):
        self._solution = solution

    def initParticle(self, model, dimensions):
        # Create position array
        # random arrays need to be initialized
        CCLC = np.sum(model._position[1])
        CRMC = np.sum(model._position[2])
        PPLC = np.sum(model._position[3])

        while PPLC < 5 or CRMC < 7 or CCLC < 11:
            for i, position in enumerate(model._position):
                if i > 0:
                    for j, val in enumerate(position):
                        model._position[i, j] = randint(0, 1)
            CCLC = sum(model._position[1])
            CRMC = sum(model._position[2])
            PPLC = sum(model._position[3])

        # Create Velocity array
        # model._velocity = np.array.random.random(0,2 , size=(6,24))
        # model._position = np.array.random.random(0, 2, size=(6, 24))
        # Save best Position so far as current Position
        # model._bestPosition = model._position

        self.updateFitness(model)

    def updateFitness(self, model):

        CGL = randint(2721, 3610)
        CCL = randint(255, 1755)
        PPL = randint(82, 399)
        CRM = randint(3612, 6345)

        for i, position in enumerate (model._position):
            for d, j in enumerate (position):
                if i == 0:
                   Fitcalcarr[i,d] =  j * CGL * uniform(0.8, 1)
                if i == 1:
                    Fitcalcarr[i,d] = j * CCL * uniform(0.8, 1)
                if i == 2:
                    Fitcalcarr[i, d] = j * PPL * uniform(0.8, 1)
                if i == 3:
                    Fitcalcarr[i, d] = j * CRM * uniform(0.8, 1)
        # calculate fitness
        fx = (model._position[0] * CGL * uniform(0.5, 1)) + (model._position[1] * CCL * uniform(0.5, 1)) + (
                    model._position[2] * PPL * uniform(0.5, 1)) + (
                     model._position[3] * CRM * uniform(0.5, 1))
        fx = Fitcalcarr
        fx_cost = Energy_cost_PTI * Fitcalcarr * 2
        TotalEnergyCost = np.sum(fx_cost)

        model._fitness = TotalEnergyCost
        model.curve = fx

        if TotalEnergyCost < model._bestPositionFitness:
            model._bestPosition = model._position
            model._bestPositionFitness = TotalEnergyCost
            model._bestcurve = model.curve

    def updatePosition(self, model):
        # VELOCITY NEEDS TO BE CONSTRICTED WITH VMAX
        # Get random coefficients e1 & e2
        w = model._inertia
        Dist = 0
        Gdist = 0
        CCLC = sum(model._position[1])
        CRMC = sum(model._position[2])
        PPLC = sum(model._position[3])

        # Apply equation to each component of the velocity, add it to corresponding position component

        for i, position in enumerate(model._position):
            # vx =  model._velocity[i] ,
            SBP = model._sbest
            hdist = spp.distance.hamming(position, model._nbBestPosition[i])
            gdist = spp.distance.hamming(position, SBP[i])

            Dist += hdist
            Gdist += gdist

        while PPLC < 5 or  CRMC < 7 or CCLC < 11:  # vx = i
            for i, position in enumerate(model._position):
                if i > 0:
                    for j, val in enumerate(position):
                        Vs = model._velocity[i, j]

                        velocity = w * Vs + (c1 * e1 * Dist) + (c2 * e2 * Gdist)

                        if abs(velocity) > vmax and abs(velocity) == velocity:
                            model._velocity[i, j] = vmax
                        elif abs(Vs) < vmax:
                            model._velocity[i, j] = abs(velocity)

                        Cvelocity = 2 * abs(self.sigmoid(velocity) - 0.5)

                        if random() < Cvelocity:
                            if val == 1:
                                val = 0
                        else:
                            val = 1
            CCLC = sum(model._position[1])
            CRMC = sum(model._position[2])
            PPLC = sum(model._position[3])

            # Ds = self.sigmoid(Vs)
            # model.velocity[i,j] = Ds
            #            print "vel:", velocity
            # if Vs > randint(0, 1):
            #     model._position[i, j] = 1
            # else:
            #      model._position[i, j] = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + mt.exp(-x))

    # self.constrain(model)
    # print(velocity)
    # print(model._position)
    # print(prob_zeros, prob_ones)

    def updatePosition2(self, model):
        w = model._inertia

        CCLC = sum(model._position[1])
        CRMC = sum(model._position[2])
        PPLC = sum(model._position[3])
        fx = (model._position[0] * CGL) + (model._position[1] * CCL) + (
                    model._position[2] * PPL ) + (
                     model._position[3] * CRM )

        MaxDID = (sum(fx) * 2)

        while (PPLC < 6 or CRMC < 7 or CCLC < 10) and MaxDID <= 115824:
            for i, position in enumerate(model._position):

                if i > 1:
                    for j, val in enumerate(position):
                        nbest = model._nbBestPosition[i, j]
                        gbest = model._sbest[i, j]

                        if nbest == 1:
                            Nones[i, j] = c1 * e1
                            Nzeros[i, j] = -c1 * e1
                        else:
                            Nzeros[i, j] = c1 * e1
                            Nones[i, j] = -c1 * e1

                        if gbest == 1:
                            Gones[i, j] = c2 * e2
                            Gzeros[i, j] = -c2 * e2
                        else:
                            Gzeros[i, j] = c2 * e2
                            Gones[i, j] = -c2 * e2

                        Velocity1 = model._vones[i, j]
                        Velocity0 = model._vzeros[i, j]
                        Velocones = w * Velocity1 + Nones[i, j] + Gones[i, j]
                        Veloczero = w * Velocity0 + Nzeros[i, j] + Gzeros[i, j]
                        Vs1 = 2 * (self.sigmoid(Velocones) - 0.5)
                        Vs0 = 2 * (self.sigmoid(Veloczero) - 0.5)

                        print(Vs1, Vs0)

                        if Vs1 > Vs0:
                            val = 1
                        else:
                            val = 0
            fx = (model._position[0] * CGL * uniform(0.5, 1)) + (model._position[1] * CCL * uniform(0.5, 1)) + (
                        model._position[2] * PPL * uniform(0.5, 1)) + (
                         model._position[3] * CRM * uniform(0.5, 1))

            fx_cost = Energy_cost_PTI * fx

            MaxDID = (sum(fx_cost)* 2)

            CCLC = sum(model._position[1])
            CRMC = sum(model._position[2])
            PPLC = sum(model._position[3])

    def updatePosition1(self, model):

        CCLC = sum(model._position[1])
        CRMC = sum(model._position[2])
        PPLC = sum(model._position[3])

        fx = (model._position[0] * CGL * uniform(0.5, 1)) + (model._position[1] * CCL * uniform(0.5, 1)) + (
                    model._position[2] * PPL * uniform(0.5, 1)) + (
                     model._position[3] * CRM * uniform(0.5, 1))
        fx_cost = Energy_cost_PTI * fx

        MaxDID = (sum(fx) * 2)

        PeakDID = np.max

        while (PPLC < 6 or  CRMC < 7 or CCLC < 10):
            for i, position in enumerate(model._position):
                # vx =  model._velocity[i]
                # vx = i
                if i > 1:
                    for j, val in enumerate(position):
                        model._position[i, j] = randint(0, 1)

            # fx = (model._position[0] * CGL * uniform(0.5, 1)) + (model._position[1] * CCL * uniform(0.5, 1)) + (
            #            model._position[2] * PPL * uniform(0.5, 1)) + (model._position[3] * CRM * 2)
            # MaxDID = sum(fx)

            CCLC = sum(model._position[1])
            CRMC = sum(model._position[2])
            PPLC = sum(model._position[3])


# ===============================================================================
# Knapsack Particle Controller
# ===============================================================================
class KnapsackParticleController(BinaryParticleController):

    def __init__(self, solution):
        self._solution = solution

    def initParticle(self, model, dimensions):
        # Create position array
        model._position = np.random.randint(2, size=dimensions)
        # Create Velocity array
        model._velocity = np.random.randint(2, size=dimensions)
        # Save best Position so far as current Position
        # model._bestPosition = np.zeros((dimensions,), dtype= np.int())
        # print model._bestPosition
        self.updateFitness(model)

    def updateFitness(self, model):

        curWeight = curValue = 0
        for idx, (price, weight) in enumerate(self._solution._items):
            if model._position[idx] == 1:
                curWeight += weight
                curValue += price

        if curWeight != 0 and curWeight <= self._solution._knapsackSize and (
                1 / float(curValue) < model._fitness or model._fitness is None):
            model._fitness = 1 / float(curValue)
            model._bestPosition = np.copy(model._position)
            # self._solution._resValue = curValue
            # self._solution._resWeight = curWeight


# ===============================================================================
# TSP Particle Controller
# ===============================================================================
class TSPParticleController(BinaryParticleController):

    def __init__(self, solution):
        self._solution = solution

    def updateFitness(self, model):
        curWeight = 0
        curPath = []
        for idx, node in enumerate(self._solution._edges):
            if model._position[idx] == 1:
                curWeight += self._solution._edges[node]
                curPath.append(node)
        if self.validateNumOfNodes(curPath, self._solution._numOfCities):
            try:
                curPath = self.orderSolution(curPath, self._solution._startNode)
                if curWeight < model._fitness or model._fitness is None:
                    model._fitness = curWeight
                    model._bestPosition = np.copy(model._position)
                    # self._solution._bestPath = curPath[:]
            except:
                pass

    def countEdges(self, graph):
        result = {}
        for (start, dest) in graph:
            if start in result:
                result[start] = result[start] + 1
            else:
                result[start] = 1
        return result

    def validateNumOfNodes(self, graph, numOfCities):
        return (len(self.countEdges(graph)) == numOfCities)

    def orderSolution(self, graph, startNode, isFirst=True):
        result = []
        countMap = self.countEdges(graph)
        curPostion = startNode
        flag = True
        subGraph = []
        while flag is True:
            flag = False
            for (start, dest) in graph:
                if curPostion is start:
                    i = countMap[start]
                    graph.remove((start, dest))
                    while i > 1:
                        subGraph.append(self.orderSolution(graph, start, False))
                        i = i - 1
                    for item in subGraph:
                        if len(item) != 0 and item[len(item) - 1][1] is not startNode:
                            result += item
                            subGraph.remove(item)
                    result.append((start, dest))
                    curPostion = dest
                    flag = True
        for item in subGraph:
            result += item
        if isFirst:
            if len(graph) != 0:
                raise Exception("Invalid Graph")
            for i, node in enumerate(result):
                if node[1] != result[(i + 1) % len(result)][0]:
                    raise Exception("Invalid Graph")
        return result


# ===============================================================================
# Swarm Controller
# ===============================================================================
class SwarmController:
    _particleController = None
    _neighbourhoodController = None

    def __init__(self, type, solution):
        # Initialize ParticleController
        if type == "continuous":
            self._particleController = ParticleController(solution)
        elif type == "binary":
            self._particleController = BinaryParticleController(solution, model=ParticleModel)
        elif type == "knapsack":
            self._particleController = KnapsackParticleController(solution)
        elif type == "tsp":
            self._particleController = TSPParticleController(solution)

            # Initialize NeighbourhoodController
        self._neighbourhoodController = NeighbourhoodController()

    def initSwarm(self, swarm, nParticles=100, dimensions=5):
        swarm._neighbourhoods = []
        # Create Swarm
        for i in range(nParticles):
            newParticle = ParticleModel()
            self._particleController.initParticle(newParticle, dimensions)
            swarm._particles.append(newParticle)  # Adding new particle to swarm
        swarm._neighbourhoods = self._neighbourhoodController.initNeighbourhoods(swarm,
                                                                                 "gbest")  # initiallizing the swarm topology or the global best topology

        for curparticle in swarm._particles:
            curparticle._neighbourhoods = self._neighbourhoodController.initNeighbourhoods(swarm,
                                                                                           "lbest")  # initializing the neighbourhood topology.
            swarm._neighbourhoods += curparticle._neighbourhoods
        self.updateSwarmBestPosition(swarm)

    def updateSwarmBestPosition(self, swarm):
        # Find swarm best position and save it in swarm
        for nb in swarm._neighbourhoods:

            self._neighbourhoodController.updateNeighbourhoodBestPosition(nb)
            TotalDemand = ((np.sum(nb._curve))* 2)
            if swarm.bestPositionFitness > nb._bestPositionFitness:
                swarm._bestPositionFitness = nb._bestPositionFitness
                swarm._bestPosition = np.copy(nb._bestPosition)
                swarm._curve = np.copy(nb._curve)
                nb._sbest = swarm._bestPosition
                print("SwarmBEst:",swarm._bestPosition)

    # Update all particles in the swarm
    def updateSwarm(self, swarm):
        for i, curParticle in enumerate(swarm._particles):
            curParticle._inertia = swarm._inertia
            self._particleController.updatePosition1(curParticle)
            self._particleController.updateFitness(curParticle)

            print("best position ", curParticle.bestPosition)
            print("current Fitness", i, curParticle._fitness)

        self.updateSwarmBestPosition(swarm)

    def SwarmBestPos(self, swarm):
        x = swarm.BestPosition
        return x


# ===============================================================================
# Neighborhood Controller
# ===============================================================================
class NeighbourhoodController:

    def initNeighbourhoods(self, swarm, topology):
        # Neighbour1 = []
        # Neighbour2 = []
        if topology == "gbest":
            return [NeighbourhoodModel(swarm._particles)]
        elif topology == "lbest":
            neighbourhoods = []
            for idx, curParticle in enumerate(swarm._particles):
                if idx == 0:
                    Neighbour4 = swarm._particles[idx + 2]
                    Neighbour1 = swarm._particles[idx + 1]
                    Neighbour2 = swarm._particles[len(swarm._particles) - 1]
                    Neighbour3 = swarm._particles[len(swarm._particles) - 2]

                elif idx == len(swarm._particles) - 1:
                    # Previous is previous, next is first
                    Neighbour1 = swarm._particles[0]
                    Neighbour2 = swarm._particles[idx - 1]
                    Neighbour3 = swarm._particles[2]
                    Neighbour4 = swarm._particles[idx - 2]
                else:
                    # Previous is previous, next is next

                    Neighbour1 = swarm._particles[idx + 1]
                    Neighbour2 = swarm._particles[idx - 1]
                    Neighbour3 = swarm._particles[idx + 2]
                    Neighbour4 = swarm.particles[idx - 2]
                neighbourhoods.append(NeighbourhoodModel([Neighbour1, curParticle, Neighbour2, Neighbour3, Neighbour4]))

                return neighbourhoods

    def updateNeighbourhoodBestPosition(self, model):
        # Find the best one in the NB

        for curParticle in model._particles:
            MaxDemand = np.max(curParticle.curve)

            if model._bestPositionFitness > curParticle._bestPositionFitness and 5000 <= MaxDemand <= 7585:
                model._bestPositionFitness = curParticle._bestPositionFitness
                model._bestPosition = np.copy(curParticle._bestPosition)
                model._curve = np.copy(curParticle._bestcurve)
                curParticle._sbest = model._sbest
                print("Neighbourhood BestPosition",model._bestPosition)
            # Save nb best position in particles nbBestPosition

        for curParticle in model._particles:
            curParticle._nbBestPosition = np.copy(model._bestPosition)
