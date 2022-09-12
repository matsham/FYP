#===============================================================================
# @author: Daniel V. Stankevich
# @organization: RMIT, School of Computer Science, 2012
#
# PSO Simple Test Functions
#===============================================================================

import numpy as np

Energy_cost_PTI = []
ARP = 0
BAF = 0
CGL = 0
CCL = 0
PPL = 0
CRM = 0
# Square function
def square(vector):
    return

def Energy_COst_function (vector1,vector2):     #vector1 is a solution #vector2 ia cost array vectpr 3 is
     fx = (model._position[0] * CCL) + (model._position[1] * CGL) + (model._position[2] * BAF) + (
                model._position[3] * ARP) + (model._position[4] * PPL) + (model._position[5] * CRM)
     fxcost = Energy_cost_PTI * fx
     TotalEnergyCost = sum(fxcost)
     #vector =
     vector =  np.multiply(vector1,vector2)



# Power function
def pow(vector, power):
    original = vector[:]
    for i in range(power-1):
        vector = np.multiply(vector, original)
    return vector