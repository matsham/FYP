#===============================================================================
# @author: Daniel V. Stankevich
# @organization: RMIT, School of Computer Science, 2012
#
# Main Project Script
#===============================================================================

#import PSOProblems
from PSOProblems import *


#cpsoProblem 	= CPSOProblem("lbest")
#psoProblem = CBPSOProblem("gbest")

#psoProblem.plotResults()

cbpsoProblem 	= CBPSOProblem("gbest")
cbpsoProblem.plotResults()

#cbpsoProblem.plotResults()

#knapsackProblem = BPSOKnapsackProblem()

#tspProblem      = BPSOTSPProblem()