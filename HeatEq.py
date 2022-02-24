
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import sys
import petsc4py
import matplotlib.pyplot as plt
petsc4py.init(sys.argv)
class HeatEqSolver:
    def __init__(self, domainSize, L, dx, dy, NumberOfSd, leftTemperature=0,rightTemperature=10,DiffCoeff=1, dt=0.1):
        self.DomainSize = int(domainSize)
        self.NumberOfSd = int(NumberOfSd)
        self.sdSize=self.DomainSize//self.NumberOfSd
        self.leftTemperature=leftTemperature
        self.rightTemperature=rightTemperature
        self.D = DiffCoeff
        self.dt = dt
        self.L= L
        self.dx, self.dy = dx, dy
        self.alpha = 1+2*self.dt*(1/(self.dx*self.dx) + 1/(self.dy*self.dy))
        self.beta = -(self.dt*self.D)/(self.dx*self.dx)
        self.gamma = -(self.dt*self.D)/(self.dy*self.dy)
        self.glbIndices = np.arange(0, int(self.DomainSize), 1,dtype=np.int32)
        self.Sol=PETSc.Vec().createMPI(size=self.DomainSize,comm=MPI.COMM_WORLD)
        self.localMatrix=self.buildLocalMatrix()
        self.LGMapping=self.buildLocToGlbMapping()
        self.Sol.setLGMap(self.buildLocToGlbMapping())
        self.Sol.getArray()[:]=0
        self.Sol.assemble()
        self.glbMatrix=self.buildGlbMatrix()
        self.glbKSP=self.buildGlbKsp()
        # self.setGlbPc()
        self.glbRHS=self.buildGlbRHS()
        
    def buildGlbMatrix(self):
        """
        Build Heat equation Matrix
         """
        GlbMatrix = PETSc.Mat().createPython(self.DomainSize, comm=MPI.COMM_WORLD)
        GlbMatrix.setType(PETSc.Mat.Type.IS)
        LocToGlbMapping = self.LGMapping
        GlbMatrix.setLGMap(LocToGlbMapping)
        LocMatrix = self.localMatrix
        GlbMatrix.setISLocalMat(LocMatrix)
        GlbMatrix.assemble()
        return GlbMatrix
    def buildLocToGlbMapping(self):
        """
        build Local to Global Mapping
        """
        rank = self.getProcId()
        localSize = self.DomainSize//self.NumberOfSd
        sdIndices = np.split(self.glbIndices, self.NumberOfSd)[rank]
        if rank > 0 and rank < self.NumberOfSd-1:
            sdIndices = np.concatenate((
                sdIndices-localSize, sdIndices, sdIndices+localSize))
        else:
            if rank == 0:
                sdIndices = np.concatenate((sdIndices, sdIndices+localSize))
            else:
                sdIndices = np.concatenate((sdIndices-localSize, sdIndices))
        sdIndicesIs = PETSc.IS().createGeneral(sdIndices, comm=MPI.COMM_WORLD)
        LocGlbMaping = PETSc.LGMap().createIS(sdIndicesIs)
        return LocGlbMaping

    def buildLocalMatrix(self):
        """
        build local shell matrix
        """
        LocalMatrix=PETSc.Mat().createPython(size=self.sizeOfLocalMatrix(),comm=MPI.COMM_SELF)
        LocalMatrix.setPythonContext(localMatrixCtx(self,self.alpha,self.beta,self.gamma))
        LocalMatrix.setUp()
        return LocalMatrix
    def setGlbPc(self):
        """
        set Preconditionner for the global KSP
        """
        glbPc=self.glbKSP.getPC()
        glbPc.setType(PETSc.PC.Type.JACOBI)
    def buildGlbKsp(self):
        """
        build global KSP
        """
        PetscKsp=PETSc.KSP().create()
        PetscKsp.setOperators(self.glbMatrix)
        PetscKsp.setConvergenceHistory(reset=True)
        PetscKsp.setType(PETSc.KSP.Type.CG)
        return PetscKsp
    def sizeOfLocalMatrix(self):
        rank=self.getProcId()
        if rank>0 and rank < self.NumberOfSd-1:
            return (3*self.sdSize,3*self.sdSize)
        else:
            return (2*self.sdSize,2*self.sdSize)
    def getProcId(self):
        """
        return Processor ID
        """
        return MPI.COMM_WORLD.Get_rank()
    def getLeftTemperature(self):
        return self.leftTemperature
    def getRightTemperature(self):
        return self.rightTemperature
    def getSolutionArray(self):
        return self.Sol.getArray()
    def buildGlbRHS(self):
        """
        build Global RHS vector
        """
        assert(self.DomainSize==2*(self.DomainSize//2))
        glbRHS=np.zeros(self.DomainSize)
        # glbSol=self.Sol.getArray()
        # print(f"length of Sol vec : {len(glbSol)}")
        for i in range(self.DomainSize):
            if i%self.sdSize==0:
                glbRHS[i]-=self.beta*self.leftTemperature
            elif (i+1)%self.sdSize==0:
                glbRHS[i]-=self.beta*self.rightTemperature
        # print("------RHS Array -----------",glbRHS)
        PetscGlbRHS=PETSc.Vec().createWithArray(glbRHS,size=self.DomainSize,comm=MPI.COMM_WORLD)
        PetscGlbRHS.setLGMap(self.buildLocToGlbMapping())
        PetscGlbRHS.assemble()
        # print("Petsc RHS len :",len(PetscGlbRHS.getArray()))
        return PetscGlbRHS
    def updateRHS(self):
        self.glbRHS=self.buildGlbRHS()
    def execute(self):
        """
        solve the system
        """
        # print("----RHS--------",self.glbRHS.getArray(),"--------len----",len(self.glbRHS.getArray()))
        self.glbKSP.solve(self.glbRHS,self.Sol)
        # print("----Sol--------",self.Sol.getArray())
    def writeSolution(self):
        ibegin=self.rank*self.sdSize
    def saveSolution(self):
        solArray=self.Sol.getArray()
        rank=self.getProcId()
        Tfile=open(f"T{rank}.txt","w")
        print("number of Sd ;",self.sdSize)
        for i in range(self.sdSize):
            Tfile.write(str(solArray[i])+"\n")
        Tfile.close()
    def showSolution(self):
        heatMapList=[]
        if self.getProcId()==0:
            for i in range(self.NumberOfSd):
                Tfile=open(f"T{i}.txt",'r')
                heatMapListOfValues=[]
                for j in range(self.sdSize):
                    heatMapListOfValues.append(float(Tfile.readline()))
                Tfile.close()
                heatMapList.append(heatMapListOfValues)
            print("----------------- heat map array ----------", heatMapList)
            plt.imshow(heatMapList, cmap='viridis')
            plt.colorbar()
            plt.savefig('heatmap')

class localMatrixCtx:
    """
     Context of the local shell matrix
    """
    def __init__(self,Solver,alpha,beta,gamma):
        assert(isinstance(Solver,HeatEqSolver))
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.rank=Solver.getProcId()
        self.NumberOfSd=Solver.NumberOfSd
        self.sdSize=Solver.sdSize
    def mult(self,mat,PetscVecToMult,PetscMultRes):
        Res=PetscMultRes.getArray()
        
        vecToMult=PetscVecToMult.getArray(readonly=True)
        # print("--------------vecToMult len :",len(vecToMult))
        # vecToMult=np.zeros(3*self.sdSize)
        Res[:]=0
        if self.rank==1:
            print("vecToMult---:",vecToMult)
        for i in range(self.sdSize):
            if self.rank>0 and self.rank < self.NumberOfSd-1:
                Res[self.sdSize+i]+=vecToMult[self.sdSize+i]*self.alpha
                if i>0:
                    Res[self.sdSize+i]+=vecToMult[self.sdSize+i-1]*self.beta
                if i <self.sdSize-1:
                    Res[self.sdSize+i]+=vecToMult[self.sdSize+i+1]*self.beta
                Res[self.sdSize+i]+= vecToMult[i]*self.gamma +self.gamma*vecToMult[2*self.sdSize+i]
            if self.rank==0:
                Res[i]+=vecToMult[i]*self.alpha
                if i>0:
                    Res[i]+=vecToMult[i-1]*self.beta
                if i <self.sdSize-1:
                    Res[i]+=vecToMult[i+1]*self.beta
                Res[i]+=self.gamma*vecToMult[self.sdSize+i]
            if self.rank==self.NumberOfSd-1:
                Res[self.sdSize+i]+=vecToMult[self.sdSize+i]*self.alpha
                if i>0:
                    Res[self.sdSize+i]+=vecToMult[self.sdSize+i-1]*self.beta
                if i <self.sdSize-1:
                    Res[self.sdSize+i]+=vecToMult[self.sdSize+i+1]*self.beta
                Res[self.sdSize+i]+= vecToMult[i]*self.gamma
        