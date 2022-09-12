import random
from pyexpat import model
from random import *
import math
import numpy as np
import Models
import controllers
from controllers import ParticleController
from controllers import BinaryParticleController
import matplotlib.pyplot as plt

x = [6538, 6538, 13984, 5758, 6342, 5562, 5758, 6538, 5758, 6538, 5758, 6538, 5758, 6342, 5562 ]
y = np.arange(24)

plt.plot(y, x)

plt.title('Optimized load curve')

plt.xlabel("Time Interval (2 Hrs) ")
plt.ylabel("Consumption")

plt.show()