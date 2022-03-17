from lib2to3.pytree import convert
from math import perm
from pipes import Template
from re import I
from statistics import mode
from traceback import print_tb
from tracemalloc import DomainFilter
from defer import return_value
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
        GlbMatrix = PETSc.Mat().create(comm=MPI.COMM_WORLD)
        GlbMatrix.setSizes(self.DomainSize)
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
    
    def getGlbMatrix(self):
        return self.glbMatrix
    
    def getDomainSize(self):
        return self.DomainSize
    
    def getsdSize(self):
        return self.sdSize
    
    def getAlpha(self):
        return self.alpha
    
    def getGlbRHS(self):
        return self.glbRHS
    
    def buildGlbRHS(self):
        """
        build Global RHS vector
        """
        assert(self.DomainSize==2*(self.DomainSize//2))
        glbRHS=np.zeros(self.DomainSize)
        for i in range(self.DomainSize):
            if i%self.sdSize==0:
                glbRHS[i]-=self.beta*self.leftTemperature
            elif (i+1)%self.sdSize==0:
                glbRHS[i]-=self.beta*self.rightTemperature
        PetscGlbRHS=PETSc.Vec().createWithArray(glbRHS,size=self.DomainSize,comm=MPI.COMM_WORLD)
        PetscGlbRHS.setLGMap(self.buildLocToGlbMapping())
        PetscGlbRHS.assemble()
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
            plt.imshow(heatMapList, cmap='viridis')
            plt.colorbar()
            plt.savefig('heatmap')

    def testICCFactorMethod(self):
        permIS=PETSc.IS().createGeneral(np.arange(self.DomainSize,dtype=np.int32))
        convertedMatrix=PETSc.Mat().create(PETSc.COMM_SELF)
        convertedMatrix.setSizes((2,2))
        convertedMatrix.setType(PETSc.Mat.Type.SEQDENSE)
        convertedMatrix.setUp()
        # convertedMatrix.zeroEntries()
        diagVec=PETSc.Vec().createSeq(2)
        # diagVec.getArray()[:]=np.full(self.sdSize,9)
        # convertedMatrix.setDiagonal(diagVec)
        convertedMatrix.setValue(0,0,1)
        convertedMatrix.setValue(0,1,1)
        convertedMatrix.setValue(1,1,1)
        convertedMatrix.setValue(1,0,1)
        convertedMatrix.assemble()
        newMat=PETSc.Mat().createAIJ(2,comm=MPI.COMM_WORLD)
        newMat.setUp()
        convertedMatrix.convert(PETSc.Mat.Type.MPIAIJ,newMat)
        # tempKSP=PETSc.KSP().create(comm=PETSc.COMM_SELF)
        # tempKSP.setOperators(convertedMatrix)
        # PC=tempKSP.getPC()
        # PC.setType('cholesky')
        # PC.setFactorSolverType('mumps')
        # PC.setFactorSetUpSolverType()
        # tempKSP.setUp()
        # facMat=tempKSP.getPC().getFactorMatrix()
        # facMat.getDiagonal(diagVec)
        # convertedMatrix.factorLU(permIS,permIS)
        # convertedMatrix.setUnfactored()
        # convertedMatrix.view()

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
        Res[:]=0
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

class PCLocalMatrixCtx:
    """
    Local Matrix context of a jacobi perconditioner
    """
    def __init__(self,Solver,alpha):
        assert(isinstance(Solver,HeatEqSolver))
        self.alpha=alpha
        self.rank=Solver.getProcId()
        self.NumberOfSd=Solver.NumberOfSd
        self.sdSize=Solver.sdSize
        
    def mult(self,mat,PetscVecToMult,PetscMultRes):
        MultVecSize=PetscVecToMult.getSize()
        PetscVecToMultArray=PetscVecToMult.getArray(readonly=True)
        for i in range(MultVecSize):
            PetscMultRes.setValue(i,(1./self.alpha)*PetscVecToMultArray[i])
    
class MultiPreconditionedCG:
    
    """
    MultiPreconditoned Matrix

    """
    def __init__(self,baseSolver :HeatEqSolver):
        assert(isinstance(baseSolver,HeatEqSolver))
        self.BaseSolver=baseSolver
        self.alpha=self.BaseSolver.getAlpha()
        self.DomainSize=self.BaseSolver.getDomainSize()
        self.sdSize=self.BaseSolver.getsdSize()
        self.SdNumber=self.DomainSize//self.sdSize
        self.sizeOfLocalMatrix=self.BaseSolver.sizeOfLocalMatrix()
        self.GlbMatrix=self.BaseSolver.getGlbMatrix()
        self.PcIsMatrix=self.buildPC()
        self.Solve(self.BaseSolver.getGlbRHS(),self.BaseSolver.getGlbRHS(),0.0001)

    def buildPC(self):
        Pc=PETSc.Mat().createPython(self.DomainSize,comm=MPI.COMM_WORLD)
        Pc.setType(PETSc.Mat.Type.IS)
        LocToGlbMapping=self.BaseSolver.buildLocToGlbMapping()
        Pc.setLGMap(LocToGlbMapping)
        PcLocalMatrix=PETSc.Mat().createPython(self.sizeOfLocalMatrix, comm=MPI.COMM_SELF)
        PcLocalMatrix.setPythonContext(PCLocalMatrixCtx(self.BaseSolver,self.alpha))
        PcLocalMatrix.setUp()
        Pc.setISLocalMat(PcLocalMatrix)
        Pc.assemble()
        return Pc

    def buildsrDirMatrixCtx(self,ResidualVec):
        SrDirMatCtx=SearchDirectionsLocMatrixCtx(ResidualVec,self.PcIsMatrix)
        # LocToGlbMap=SrDirMatCtx.buildLGMap()
        # srDirMatrix.setLGMap(LocToGlbMap[0],LocToGlbMap[1])
        # srDirLocMatrix=PETSc.Mat().createPython((self.DomainSize,1),comm=MPI.COMM_SELF)
        # srDirLocMatrix.setPythonContext(SrDirMatCtx)
        # srDirLocMatrix.setUp()
        # srDirMatrix.setISLocalMat(srDirLocMatrix)
        # srDirMatrix.assemble()
        return SrDirMatCtx

    def Orthogonolize(self,lastSrDirMat,
                listOfQMats,lisfOfSrDirMats,listOfDeltaMatKSPs):
        """
        Do the full Orthonolization process 
        """
        iterNbr=len(listOfDeltaMatKSPs)
        # Results matrix Shape has to be  changed for optimized non singular versions of SrDirMats
        tempSrDirMat=lastSrDirMat.duplicate()
        tempDeltaMat=listOfDeltaMatKSPs[0].duplicate()
        for i in range(iterNbr):
            listOfQMats[i].matTransposeMult(lisfOfSrDirMats[i],tempDeltaMat)
            listOfDeltaMatKSPs[i].matsolve(tempDeltaMat,tempDeltaMat)
            lisfOfSrDirMats[i].matMult(tempDeltaMat,tempSrDirMat)
            lastSrDirMat.axpy(-1,tempSrDirMat)

    def lastError(self,srDirMatrix,ResidualVec):
        tempVec=srDirMatrix.getVecRight()
        tempVec.getArray()[:]=[1]
        srDirMatrix.mult(tempVec,tempVec)
        return np.sqrt(tempVec.dot(ResidualVec))
    
    def Solve(self,PetscSolVec,PetscRhsVec,tol):
        
        ResidualVec=PetscRhsVec.duplicate()
        PetscSolVec.getArray()[:]=- PetscSolVec.getArray()[:]
        listOfDeltaMatKSPs,listOfsrDirMats,listOfQMats=[],[],[]
        self.GlbMatrix.multAdd(PetscSolVec,PetscRhsVec,ResidualVec)
        PetscSolVec.zeroEntries()
        srDirMatCtx=self.buildsrDirMatrixCtx(ResidualVec)
        srDirMat=srDirMatCtx.getSrDirMat()
        listOfsrDirMats.append(srDirMat)
        deltaMatCtx=DeltaMatrixCtx(srDirMat,self.GlbMatrix,srDirMatCtx)
        gammaVecSeq=PETSc.Vec().createSeq(self.SdNumber)
        gammaVecPar=PETSc.Vec().createMPI(self.SdNumber,comm=MPI.COMM_WORLD)
        
        while self.lastError(srDirMat,ResidualVec)>tol:
            QSeqMat=deltaMatCtx.getQSeqMat()
            listOfQMats.append(QSeqMat)
            deltaMatKSP=deltaMatCtx.getLocalKSP()
            listOfDeltaMatKSPs.append(deltaMatKSP)
            srDirMat.multTranspose(ResidualVec,gammaVecPar)
            gammaVecSeq=srDirMatCtx.shareLocalVec(gammaVecPar,mode="reverse")
            deltaMatKSP.solve(gammaVecSeq,gammaVecSeq)
            gammaVecPar=srDirMatCtx.shareLocalVec(gammaVecSeq,mode="forward")
            srDirMat.multAdd(gammaVecPar,PetscSolVec,PetscSolVec)
            QSeqMat.multAdd(-gammaVecPar,ResidualVec,ResidualVec)
            srDirMatCtx.update(ResidualVec)
            srDirMat=srDirMatCtx.getSrDirMat()
            self.Orthogonolize()
            deltaMatCtx.update(srDirMat,srDirMatCtx)

class DeltaMatrixCtx:
    
    """
    build the context of  matrix W^T F W 
    
    """ 
    def __init__(self,srDirMatIs,GlbMatIS,srDirMatCtx):
        self.srDirMat=srDirMatIs
        self.srDirMatCtx=srDirMatCtx
        self.GlbIsMatrix=GlbMatIS
        self.glbSize=GlbMatIS.getSize()[0]
        self.sdSize=srDirMatIs.getSize()[1]
        self.sdNbr=MPI.COMM_WORLD.Get_size()
        self.sdRank=MPI.COMM_WORLD.Get_rank()
        self.deltaSeqMat,self.QSeqMat=self.buildSeqMat()
        # self.deltaSeqMat.view()
        self.localKSP=self.buildLocalKSP()
        self.computeAndSetNullSpace()
    
    def buildParMat(self):
        deltaMat=PETSc.Mat().createDense(((self.glbSize,self.glbSize),(1,self.sdSize)),comm=MPI.COMM_WORLD)
        basisElementiArr=np.zeros(self.sdNbr)
        basisElementiArr[self.sdRank]=1
        basisElementi=PETSc.Vec().createWithArray(basisElementiArr,size=self.sdNbr,comm=MPI.COMM_WORLD)
        tempVec=self.srDirMat.getVecLeft()
        self.srDirMat.mult(basisElementi,tempVec)
        self.GlbIsMatrix.mult(tempVec,tempVec)
        self.srDirMat.multTranspose(tempVec,basisElementi)
        deltaMat.getDenseArray()[:]=basisElementi.getArray()[:]
        return deltaMat    
    
    def buildSeqMat(self):
        """
        build for each subdomain its own explicit version of matrix W^t F W and matrix Q= F W
        """
        deltaMat=PETSc.Mat().createDense(self.sdNbr,comm=MPI.COMM_SELF)
        QMat=PETSc.Mat().createDense((self.glbSize, self.sdNbr),comm=MPI.COMM_SELF)
        deltaMat.setUp()
        QMat.setUp()
        basisElementi=PETSc.Vec().createMPI(size=self.sdNbr,comm=MPI.COMM_WORLD)
        for i in range(self.sdNbr):
            basisElementi.zeroEntries()
            basisElementi.setValue(i,1)
            basisElementi.assemble()
            tempVec1,tempVec2=self.srDirMat.getVecLeft(),self.GlbIsMatrix.getVecLeft()
            self.srDirMat.mult(basisElementi,tempVec1)
            self.GlbIsMatrix.mult(tempVec1,tempVec2)
            QMat.setValues(np.arange(0,self.glbSize,dtype=np.int32),[i],
                            self.srDirMatCtx.shareLocalVec(tempVec2,mode="reverse").getArray()[:])
            self.srDirMat.multTranspose(tempVec2,basisElementi)
            deltaMat.setValues(np.arange(0,self.sdNbr,dtype=np.int32),[i],
                            self.srDirMatCtx.shareLocalVec(basisElementi,mode="reverse").getArray()[:])
        deltaMat.assemble()
        QMat.assemble()
        deltaMat.view()
        return deltaMat,QMat
        
    def buildLocalKSP(self):
        """
        Build local solver for matrix W^T F W
        """
        PetscKsp = PETSc.KSP()
        PetscKsp.create(PETSc.COMM_SELF)
        deltaAsSparseMat=PETSc.Mat().createAIJ(self.deltaSeqMat.getSize(), comm=MPI.COMM_SELF)
        self.deltaSeqMat.convert(PETSc.Mat.Type.AIJ,deltaAsSparseMat)
        PetscKsp.setOperators(deltaAsSparseMat)
        PetscKsp.setType('preonly')
        PetscPc = PetscKsp.getPC()
        PetscPc.setType('lu')
        PetscPc.setFactorSolverType('mumps')
        PetscPc.setFactorSetUpSolverType()

        MumpsFactorMat = PetscPc.getFactorMatrix()
        MumpsFactorMat.setMumpsIcntl(14, 30)
        MumpsFactorMat.setMumpsIcntl(24, 1)
        MumpsFactorMat.setMumpsCntl(3, 1e-6)

        PetscKsp.setUp()
        return PetscKsp
    
    def computeAndSetNullSpace(self):
        """
        Return The Null Space Basis of Matrix W^T F W
        """
        mumpsFactorMatrix=self.localKSP.getPC().getFactorMatrix()
        NullSpaceBasis = []
        NullSpaceSize = mumpsFactorMatrix.getMumpsInfog(28)
        for idx in range(NullSpaceSize):
            TempVec = self.deltaSeqMat.getVecRight()
            mumpsFactorMatrix.setMumpsIcntl(25, idx + 1)
            self.localKSP.solve(TempVec,TempVec)
            for OtherVec in NullSpaceBasis[:idx]:
                TempVec -= TempVec.dot(OtherVec) * OtherVec
            TempVec.normalize()
            NullSpaceBasis.append(TempVec)
        NullSpace = PETSc.NullSpace().create(
            vectors=NullSpaceBasis, comm=MPI.COMM_SELF)
        assert(NullSpace.test(self.deltaSeqMat))
        mumpsFactorMatrix.setMumpsIcntl(25, 0)
        self.localKSP.getOperators()[0].setNullSpace(NullSpace)
    
    def getDeltaSeqMat(self):
        return self.deltaSeqMat

    def getQSeqMat(self):
        return self.QSeqMat

    def getLocalKSP(self):
        return self.localKSP
    
    def update(self,newSrDirMat,newSrDirMatCtx):
        self.srDirMat=newSrDirMat
        self.srDirMatCtx=newSrDirMatCtx
        self.deltaSeqMat,self.QSeqMat=self.buildSeqMat()
        self.localKSP=self.buildLocalKSP()
        self.computeAndSetNullSpace()

    def mult(self,mat,PetscVecToMult,PetscMultRes):
        """
        TO DO
        """    
class SearchDirectionsLocMatrixCtx:
    
    def __init__(self,ResidualVec,PcMatIS):
        self.ResidualVec=ResidualVec
        self.PcMatIS=PcMatIS
        self.glbSize=self.PcMatIS.getSize()[0]
        self.PcRowLGMapIS,self.PcColLGMapIS=self.PcMatIS.getLGMap()
        self.PcIsLocMat=self.PcMatIS.getISLocalMat()
        self.SdSearchDir=self.computeSdSearchDirection()
        self.sdRank=MPI.COMM_WORLD.Get_rank()
        self.sdNbr=MPI.COMM_WORLD.Get_size()
        self.srDirMat=self.buildSrDirMatrix()

    def buildLGMap(self):
        rowIS=PETSc.IS().createGeneral(np.arange(0,self.glbSize,step=1,dtype=np.int32),comm=MPI.COMM_WORLD)
        colIS=PETSc.IS().createGeneral([self.sdRank],comm=MPI.COMM_WORLD)
        rowLGMapping=PETSc.LGMap().createIS(rowIS)
        colLGMapping=PETSc.LGMap().createIS(colIS)
        return rowLGMapping,colLGMapping

    def buildVecScatter(self):
        """
        Build a vector scatter to copy data of shared vec on a self owned vec
        """
        VecToScatterFrom=self.GlbIsMatrix.getVecLeft()
        #------------------TO DO ---------------

    def shareLocalVec(self,vecToShareOrGather,mode="forward"):
        """
        Convert a vector defined on COMM_SELF communicator to a COMM_WORLD communicator where the
        output vector is now shared between all subdomains.

        If mode is set to 'reverse' it will do the reverse process

        """
        glbSize=vecToShareOrGather.getSize()
        if mode=="forward":
            # assert(vecToShareOrGather.getComm()==MPI.COMM_SELF)
            SharedVec=PETSc.Vec().createMPI(glbSize,comm=MPI.COMM_WORLD)
            vecToShareOrGatherArray=np.array(vecToShareOrGather.getArray())
            splitedLocVec=np.split(vecToShareOrGatherArray,self.sdNbr)
            SharedVec.getArray()[:]=splitedLocVec[self.sdRank][:]
            return SharedVec
        
        elif mode=="reverse":
            # assert(vecToShareOrGather.getComm()==PETSc.Comm.COMM_WORLD)
            OwnedVec=PETSc.Vec().createSeq(glbSize,comm=MPI.COMM_SELF)
            # ownershipRangeStartIndex=vecToShareOrGather.getOwnershipRange()[0]
            # ownershipRangeSize=vecToShareOrGather.getOwnershipRange()[1]-ownershipRangeStartIndex
            IndexSetToScatter=PETSc.IS().createStride(glbSize,
                            0,step=1,comm=MPI.COMM_WORLD)
            VecScatter=PETSc.Scatter().create(vecToShareOrGather,IndexSetToScatter,OwnedVec,IndexSetToScatter)
            VecScatter.scatter(vecToShareOrGather,OwnedVec,mode="forward")
            return OwnedVec
        
        else :
            raise RuntimeError(f"There is no mode called {mode}: try with 'forward' or 'reverse'")

    def computeSdSearchDirection(self):
        #------ To optimize if colIndices and rowIndices are the same ------------
        colIndices=np.array(self.PcColLGMapIS.getIndices())
        rowIndices=np.array(self.PcRowLGMapIS.getIndices())
        searchDirVec=PETSc.Vec().createSeq(self.glbSize,comm=MPI.COMM_SELF)
        searchDirVec.zeroEntries()
        reducedResidualVec=PETSc.Vec().createSeq(len(colIndices),comm=MPI.COMM_SELF)
        #-------To optimize if colIndices is the same as data parallel layout on the vector ResidualVec-------
        reducedResidualVec.getArray()[:]=self.shareLocalVec(self.ResidualVec,mode="reverse").getArray()[colIndices]
        localProdRes=self.PcIsLocMat.getVecLeft()
        self.PcIsLocMat.mult(reducedResidualVec,localProdRes)
        searchDirVec.getArray()[rowIndices]=localProdRes.getArray()[:]
        return searchDirVec
    
    
    def buildSrDirMatrix(self):
        SrDirMat=PETSc.Mat().createDense((self.glbSize,self.sdNbr),comm=MPI.COMM_WORLD)
        SrDirMat.setUp()
        SrDirMat.setValues(np.arange(0,self.glbSize,dtype=np.int32),[self.sdRank],self.SdSearchDir.getArray()[:])
        SrDirMat.assemble()
        return SrDirMat
    
    def getSdSearchDirection(self):
        return self.SdSearchDir
    
    def getSrDirMat(self):
        return self.srDirMat
    
    def update(self,newResidualVec):
        self.ResidualVec=newResidualVec
        self.SdSearchDir=self.computeSdSearchDirection()
        self.srDirMat=self.buildSrDirMatrix()

    def mult(self,mat,PetscVecToMult,PetscMultRes):
        SrDirVecArray,PetscMultResArray=self.SdSearchDir.getArray(),PetscMultRes.getArray()
        PetscVecToMultArray=PetscVecToMult.getArray(readonly=True)
        PetscMultResArray[:]=PetscVecToMultArray[0]*SrDirVecArray[:]
    
    def multTranspose(self,mat,PetscVecToMult,PetscMultRes):
        SrDirVecArray,PetscMultResArray=self.SdSearchDir.getArray(),PetscMultRes.getArray()
        PetscVecToMultArray=PetscVecToMult.getArray(readonly=True)
        PetscMultResArray[:]=np.dot(PetscVecToMultArray[:],SrDirVecArray[:])