# ===============================================================================
# @author: Daniel V. Stankevich
# @organization: RMIT, School of Computer Science, 2012
#
#
# This package contains representations of the following models:
#  'Particle'            - an atomic element
#  'Swarm'               - a set of particles
#  'Neighbourhood'       - particles topology
#  'KnapsackSolution'    - representation for solution of the problem
#  'TSPSolution'         - representation for solution of the problem
# ===============================================================================


# ===============================================================================
#                             GENERIC MODELS
# ===============================================================================

# ---- Particle representation
import numpy as np

machine1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

machine3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

machine4 = np.mod(np.multiply(np.random.randint(0, 2, size=12), 1), 2)

machine5 = np.mod(np.multiply(np.random.randint(0, 2, size=12), 1), 2)


class ParticleModel:
    _position = None
    _velocity = None
    _vones = None
    _vzeros = None
    _bestPosition = None
    _bestPositionFitness = None
    _nbBestPosition = None
    _fitness = None
    _neighbourhoods = None
    _curve = None
    _bestcurve = None
    _sbest = None
    _inertia = None

    def __init__(self):
        self._position = np.array([machine1, machine5, machine3, machine4])
        self._velocity = np.ones([4, 12])
        self._vones = np.ones([4, 12])
        self._vzeros = np.ones([4, 12])
        self._bestPosition = np.zeros([4, 12])
        self._bestPositionFitness = 1000000000000
        self._nbBestPosition = np.zeros([4, 12])
        self._fitness = 10000000000
        self.neighbourhoods = []
        self._curve = np.zeros(12)
        self._bestcurve = np.zeros(12)
        self._Demand = 10000
        self._sbest = np.zeros([4, 12])
        self._inertia = 0

    @property
    def bestPosition(self):
        return self._bestPosition


# ---- Swarm representation
class SwarmModel:
    _particles = []
    _neighbourhoods = None
    _bestPosition = None
    _bestPositionFitness = None
    _curve = None
    _inertia = None

    def __init__(self):
        self._particles = []
        self._neighbourhoods = []
        self._bestPosition = np.zeros([4, 12])
        self._bestPositionFitness = 100000000
        self._curve = np.zeros(12)
        self._inertia = 0

    @property
    def bestPositionFitness(self):
        return self._bestPositionFitness

    @property
    def BestPosition(self):
        return self._bestPosition

    @property
    def curve(self):
        return self._curve


# ---- Neighbourhood representation
class NeighbourhoodModel:
    _particles = None
    _bestPosition = None
    _bestPositionFitness = None
    _curve = None
    _sbest = None

    def __init__(self, particles):
        self._particles = particles
        self._bestPosition = np.zeros([4, 12])
        self._bestPositionFitness = 100000000
        self._curve = np.zeros(12)
        self._sbest = np.zeros([4, 12])

# ===============================================================================
#                            PROBLEM SPECIFIC MODELS
# ===============================================================================
