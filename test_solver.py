from numpy import number
import HeatEq
from fileReader import ReadInputFile
import sys
listOfInputs=ReadInputFile()
domainSize,numberOfSd,Tl,Tr,D,tf,L=listOfInputs
dx=dy=L*numberOfSd/domainSize
HeatSolv=HeatEq.HeatEqSolver(domainSize,L,dx,dy,numberOfSd,leftTemperature=Tl,rightTemperature=Tr,DiffCoeff=D,dt=tf)
# HeatSolv.testICCFactorMethod()
if sys.argv[1]=="1":
    MPCG=HeatEq.MultiPreconditionedCG(HeatSolv)
else:
    HeatSolv.execute()
    HeatSolv.saveSolution()
    HeatSolv.showSolution()
