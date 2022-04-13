
import sys
from copy import deepcopy
import numpy as np
from scipy.linalg.lapack import dpstrf
from scipy.linalg import solve_triangular
import scipy.sparse.linalg
import scipy.sparse
from mpi4py import MPI
from petsc4py import PETSc
import sys
import petsc4py
from petsc4py import PETSc
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
        self.Sol.setLGMap(self.LGMapping[1])
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
        GlbMatrix.setLGMap(LocToGlbMapping[0],LocToGlbMapping[1])
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
        rowSdIndices=sdIndices
        if rank > 0 and rank < self.NumberOfSd-1:
            sdIndices = np.concatenate((
                sdIndices-localSize, sdIndices, sdIndices+localSize))
        else:
            if rank == 0:
                sdIndices = np.concatenate((sdIndices, sdIndices+localSize))
            else:
                sdIndices = np.concatenate((sdIndices-localSize, sdIndices))
        sdIndicesIs = PETSc.IS().createGeneral(sdIndices, comm=MPI.COMM_WORLD)
        rowSdIndicesIs = PETSc.IS().createGeneral(rowSdIndices, comm=MPI.COMM_WORLD)
        LocGlbMaping = PETSc.LGMap().createIS(sdIndicesIs)
        rowLocGlbMaping = PETSc.LGMap().createIS(rowSdIndicesIs)
        return rowLocGlbMaping,LocGlbMaping

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
            return (self.sdSize,3*self.sdSize)
        else:
            return (self.sdSize,2*self.sdSize)
    
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
        ConvergedReason =self.glbKSP.getConvergedReason()
        if ConvergedReason > 0:
                print("The interface problem converged"
                    " in \t\t:", self.glbKSP.its, "iterations.")
        else : print("The interface problem "
                "diverged : {} ({} iterations performed)"
                "".format(ConvergedReason, self.GlobalKsp.its))
        # print("----Sol--------",self.Sol.getArray())
    def writeSolution(self):
        ibegin=self.rank*self.sdSize
    
    # def saveSolution(self):
    #     solArray=self.Sol.getArray()
    #     rank=self.getProcId()
    #     Tfile=open(f"T{rank}.txt","w")
    #     print("number of Sd ;",self.sdSize)
    #     for i in range(self.sdSize):
    #         Tfile.write(str(solArray[i])+"\n")
    #     Tfile.close()
    
    # def showSolution(self):
    #     heatMapList=[]
    #     if self.getProcId()==0:
    #         for i in range(self.NumberOfSd):
    #             Tfile=open(f"T{i}.txt",'r')
    #             heatMapListOfValues=[]
    #             for j in range(self.sdSize):
    #                 heatMapListOfValues.append(float(Tfile.readline()))
    #             Tfile.close()
    #             heatMapList.append(heatMapListOfValues)
    #         plt.imshow(heatMapList, cmap='viridis')
    #         plt.colorbar()
    #         plt.savefig('heatmap')

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
                Res[i]+=vecToMult[self.sdSize+i]*self.alpha
                if i>0:
                    Res[i]+=vecToMult[self.sdSize+i-1]*self.beta
                if i <self.sdSize-1:
                    Res[i]+=vecToMult[self.sdSize+i+1]*self.beta
                Res[i]+= vecToMult[i]*self.gamma +self.gamma*vecToMult[2*self.sdSize+i]
            if self.rank==0:
                Res[i]+=vecToMult[i]*self.alpha
                if i>0:
                    Res[i]+=vecToMult[i-1]*self.beta
                if i <self.sdSize-1:
                    Res[i]+=vecToMult[i+1]*self.beta
                Res[i]+=self.gamma*vecToMult[self.sdSize+i]
            if self.rank==self.NumberOfSd-1:
                Res[i]+=vecToMult[self.sdSize+i]*self.alpha
                if i>0:
                    Res[i]+=vecToMult[self.sdSize+i-1]*self.beta
                if i <self.sdSize-1:
                    Res[i]+=vecToMult[self.sdSize+i+1]*self.beta
                Res[i]+= vecToMult[i]*self.gamma

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
    

class MultiPreconditionedGMRES:
        
        def __init__(self,baseSolver:HeatEqSolver,KerProjMat=None):
            assert(isinstance(baseSolver,HeatEqSolver))
            self.BaseSolver=baseSolver
            self.alpha=self.BaseSolver.getAlpha()
            self.DomainSize=self.BaseSolver.getDomainSize()
            self.sdSize=self.BaseSolver.getsdSize()
            self.SdNumber=self.DomainSize//self.sdSize
            self.sdRank=MPI.COMM_WORLD.Get_rank()
            self.sizeOfLocalMatrix=self.BaseSolver.sizeOfLocalMatrix()
            self.GlbMatrix=self.BaseSolver.getGlbMatrix()
            self.glbSize=self.GlbMatrix.getSize()[0]
            self.PcIsMatrix=self.buildPC()
            glbRHS=self.BaseSolver.getGlbRHS()
            self.PcLocalMat=self.PcIsMatrix.getISLocalMat()
            self.LocRowsNbr,self.LocColsNbr=self.PcLocalMat.getSize()
            self.PcLocalMatCtx=self.PcLocalMat.getPythonContext()
            self.PcLocToGLbMap=self.PcIsMatrix.getLGMap()
            self.PcMatsArray=self.buildAllSdPcGlbMat()
            self.solVec,self.SrDirTempVec=self.PcIsMatrix.getVecs()
            self.solVec.zeroEntries()
            self.Solve(self.solVec,glbRHS,None,None)
        
        def buildPC(self):
            Pc=PETSc.Mat().create(comm=MPI.COMM_WORLD)
            Pc.setSizes(self.DomainSize)
            Pc.setType(PETSc.Mat.Type.IS)                                                                                                             
            LocToGlbMapping=self.BaseSolver.buildLocToGlbMapping()
            Pc.setLGMap(LocToGlbMapping[0],LocToGlbMapping[0])
            PcLocalMatrix=PETSc.Mat().createPython(self.sizeOfLocalMatrix[0], comm=MPI.COMM_SELF)
            PcLocalMatrix.setPythonContext(PCLocalMatrixCtx(self.BaseSolver,self.alpha))
            PcLocalMatrix.setUp()
            Pc.setISLocalMat(PcLocalMatrix)
            Pc.assemble()
            return Pc
        
        def buildAllSdPcGlbMat(self):
            ArrOfSdPcMats=[]
            for id in range(self.SdNumber):
                ArrOfSdPcMats.append(PETSc.Mat().createPython(self.glbSize,comm=MPI.COMM_WORLD))
                ArrOfSdPcMats[id].setType(PETSc.Mat.Type.IS)
                ArrOfSdPcMats[id].setLGMap(self.PcLocToGLbMap[0],self.PcLocToGLbMap[1])

            for id2 in range(self.SdNumber):
                
                if id2==self.sdRank:
                    LocPcMat=PETSc.Mat().createPython((self.LocRowsNbr,self.LocColsNbr), comm=MPI.COMM_SELF)    
                    LocPcMat.setPythonContext(self.PcLocalMatCtx)
                    LocPcMat.setUp()
                    ArrOfSdPcMats[id2].setISLocalMat(LocPcMat)
                
                else:
                    LocPcMat=PETSc.Mat().createPython((self.LocRowsNbr,self.LocColsNbr), comm=MPI.COMM_SELF)    
                    LocPcMat.setPythonContext(NullMatrixCtx())
                    LocPcMat.setUp()
                    ArrOfSdPcMats[id2].setISLocalMat(LocPcMat)                
            
            for pcMat in ArrOfSdPcMats:    
                pcMat.assemble()
            return ArrOfSdPcMats
        
        def ComputeError(self,SolVec,RhsVec):
            """
            update Residual vector and return its norm
            """
            tempvec=RhsVec.duplicate()
            self.GlbMatrix.multAdd(-SolVec,RhsVec,tempvec)
            return tempvec.norm()

        def updateBornes(self,iterNbr,firstVecIndex,lastVecIndex):
            firstVecIndex=lastVecIndex
            lastVecIndex+=self.SdNumber**iterNbr

        def updateSrDirVecsArray(self,SrDirVecsArr:list,BasisVecsArr:list):
            for basisVec in BasisVecsArr :
                for idx,sdPcGlbMat in enumerate(self.PcMatsArray):
                    TempVec=basisVec.duplicate()
                    sdPcGlbMat.mult(basisVec,TempVec)
                    SrDirVecsArr.append(TempVec)
        
        def createVecScatter(self,ParPetscVec):
            VecSize=ParPetscVec.getSize()
            ownedVec=PETSc.Vec().createSeq(ParPetscVec.getSize(),comm=MPI.COMM_SELF)
            isIndexSet=PETSc.IS().createStride(VecSize,
                                            0,step=1,comm=MPI.COMM_WORLD)
            return ownedVec,PETSc.Scatter().create(ParPetscVec,isIndexSet,ownedVec,isIndexSet)
            
        def SolveLSQRProblem(self,LSQRMat,firstResidualNorm):
            PetscKsp = PETSc.KSP()
            PetscKsp.create(MPI.COMM_WORLD)
            PetscKsp.setOperators(LSQRMat)
            PetscKsp.setType('lsqr')
            PetscKsp.setUp()
            PetscKsp.setTolerances(
            rtol=1e-10, atol=1e-10, max_it=100000)
            LSQRSolVec,LSQRRhsVec=LSQRMat.getVecs()
            LSQRRhsVec.zeroEntries()
            LSQRRhsVec.setValue(0,firstResidualNorm)
            LSQRRhsVec.assemble()
            PetscKsp.solve(LSQRRhsVec,LSQRSolVec)
            LSQRMat.multAdd(-LSQRSolVec,LSQRRhsVec,LSQRRhsVec)
            PetscKsp.destroy();LSQRRhsVec.destroy()
            return LSQRSolVec
        
        def ProcOwnershipRange(self,nRows):
            offSet=self.sdRank+1 if nRows%self.SdNumber>self.sdRank else nRows%self.SdNumber
            startindex=self.sdRank*int(nRows/self.SdNumber)+offSet*(self.sdRank>0)
            RangeSize=int(nRows/self.SdNumber)+(nRows%self.SdNumber>self.sdRank)
            endindex=startindex+RangeSize-1
            return startindex,endindex,RangeSize

        def ProcMissingRows(self,nRows,nr):
            return int((nRows-nr)/self.SdNumber)+(self.sdRank<(nRows-nr)%self.SdNumber)
        
        def getProcCsr(self,csr):
            procCsr=[[],[],[]]
            nRows=len(csr["indptr"])-1
            startind,endind,rangeSize=self.ProcOwnershipRange(nRows)
            leftOffset,rightOffset=csr["indptr"][startind],csr["indptr"][endind+1]
            procCsr[0]=np.array(csr["indptr"][startind+1:endind+2]-leftOffset,dtype=np.int32)
            procCsr[0]=np.insert(procCsr[0],0,0)
            procCsr[1]=np.array(csr["colsInd"][leftOffset:rightOffset],dtype=np.int32)
            procCsr[2]=np.array(csr["values"][leftOffset:rightOffset],dtype=np.double)
            return procCsr
        
        def initializeOwnCsr(self,nRows,nCols,LocRowsNbr):
            
            csr=[[0]]
            csr[0]=np.concatenate((csr[0],np.cumsum(np.full(LocRowsNbr,nCols,dtype=np.int32),dtype=np.int32)),dtype=np.int32)
            csr.append(np.indices((LocRowsNbr,nCols),dtype=np.int32)[1].flatten())
            csr.append(np.zeros(LocRowsNbr*nCols,dtype=np.double))
            return csr
        
        def Solve(self,PetscSolVec,PetscRhsVec,tol,maxiter):
            
            """
            this the vectorized version of MPGMRES
            """
            if tol==None :
                tol=1e-3
                print("***Warning***: Tolerance has not been set, and default tol={} is used instead"\
                                                                            .format(tol))
            if maxiter==None:
                maxiter=100000
                print("***Warning***: Maximum Number of iterations has not been set, and default maxiter={} is used instead"\
                                                                                            .format(maxiter))    
            if self.sdSize>1 :
                print("***Info*** : Solving in parallel with {} Procs".format(self.SdNumber))
                isSequential=False
            else :
                print("***Info*** : Solving in sequential ")
                isSequential=True

            ResidualVec,intialGuess=PetscRhsVec.duplicate(),PetscSolVec.copy()
            SrDirVecsArr,BasisVecsArr,LSQRMatsArr=[],[],[]
            self.GlbMatrix.multAdd(-PetscSolVec,PetscRhsVec,ResidualVec)
            firstResidualNorm=ResidualVec.norm()
            ResidualVec.normalize()
            print(ResidualVec.getSize())
            BasisVecsArr.append(ResidualVec)
            iterNbr=0
            firstVecIndex=0
            lastVecIndex=self.SdNumber
            basisVecsNbr=1
            self.updateSrDirVecsArray(SrDirVecsArr,BasisVecsArr)        
            csr={"indptr":[],"colsInd":[],"values":[]}
            csr["indptr"].append(0)
            while self.ComputeError(PetscSolVec,PetscRhsVec)>tol:
                if iterNbr>=maxiter:
                    print("***Warning***: Maximum number of iterations ({}) reached without achieving convergence"\
                                                                .format(maxiter))
                    return
                else :
                    iterNbr+=1
        
                TempVec=self.SrDirTempVec.duplicate()
                for SrDirVec in SrDirVecsArr[firstVecIndex:lastVecIndex]:
                    self.GlbMatrix.mult(SrDirVec,TempVec)
                    csr["indptr"]=np.pad(csr["indptr"],(0,1),mode="edge")
                    for idx,basisVec in enumerate(BasisVecsArr):
                        scalarprod=basisVec.dot(TempVec)
                        csr['indptr'][basisVecsNbr]+=1
                        csr["values"].append(scalarprod)
                        csr["colsInd"].append(idx)
                        TempVec.axpy(-scalarprod,basisVec)
                        
                    SrDirVecNorm=TempVec.norm()
                    csr['indptr'][basisVecsNbr]+=1
                    csr["values"].append(SrDirVecNorm)
                    csr["colsInd"].append(basisVecsNbr)       
                    TempVec.normalize()
                    BasisVecsArr.append(deepcopy(TempVec))
                    basisVecsNbr+=1
                if isSequential:
                    sparseLSQMat=scipy.sparse.csr_matrix((csr["values"],csr["colsInd"],csr["indptr"]),shape=(lastVecIndex,lastVecIndex+1),dtype=float)
                    sparseLSQMat=sparseLSQMat.transpose()
                    rhs=np.zeros(lastVecIndex+1)
                    rhs[0]=firstResidualNorm
                    LSQRSolVecArr=scipy.sparse.linalg.lsqr(sparseLSQMat,rhs,atol=1e-10,btol=1e-10)[0]
                else:
                    ProcCsr=self.getProcCsr(csr)
                    glbLSQRMat=PETSc.Mat().createAIJWithArrays((lastVecIndex,lastVecIndex+1),\
                                                                            ProcCsr,comm=MPI.COMM_WORLD)
                    glbLSQRMat.setUp();glbLSQRMat.assemble()       
                    glbLSQRMat.createTranspose(glbLSQRMat)
                    LSQRSolVec=self.SolveLSQRProblem(glbLSQRMat,firstResidualNorm)
                    LSQRseqSolVec,VecScatter=self.createVecScatter(LSQRSolVec)
                    VecScatter.scatter(LSQRSolVec,LSQRseqSolVec,mode="forward")
                    PetscSolVec.getArray()[:]=intialGuess.getArray()[:]
                    LSQRSolVecArr=LSQRseqSolVec.getArray()[:]
                    glbLSQRMat.destroy();LSQRSolVec.destroy();VecScatter.destroy();LSQRseqSolVec.destroy()
                
                for idx,SrDirVec in enumerate(SrDirVecsArr):
                    PetscSolVec.axpy(LSQRSolVecArr[idx],SrDirVec)
                # PetscSolVec.maxpy(LSQRSolVec.getArray()[:],SrDirVecsArr)
                self.updateSrDirVecsArray(SrDirVecsArr,BasisVecsArr[firstVecIndex+1:lastVecIndex+1])
                firstVecIndex=lastVecIndex
                lastVecIndex+=self.SdNumber**(iterNbr+1)
                
            print("l'algo a convergé dans {} itérations with a final residual ||r||={}"\
                                                .format(iterNbr,self.ComputeError(PetscSolVec,PetscRhsVec)))



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
        self.sdRank=MPI.COMM_WORLD.Get_rank()
        self.sizeOfLocalMatrix=self.BaseSolver.sizeOfLocalMatrix()
        self.GlbMatrix=self.BaseSolver.getGlbMatrix()
        self.glbSize=self.GlbMatrix.getSize()[0]
        self.PcIsMatrix=self.buildPC()
        glbRHS=self.BaseSolver.getGlbRHS()
        self.Solve1(glbRHS,glbRHS,0.0001)

    def buildPC(self):
            Pc=PETSc.Mat().create(comm=MPI.COMM_WORLD)
            Pc.setSizes(self.DomainSize)
            Pc.setType(PETSc.Mat.Type.IS)                                                                                                             
            LocToGlbMapping=self.BaseSolver.buildLocToGlbMapping()
            Pc.setLGMap(LocToGlbMapping[0],LocToGlbMapping[0])
            PcLocalMatrix=PETSc.Mat().createPython(self.sizeOfLocalMatrix[0], comm=MPI.COMM_SELF)
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

    def solveMat(self,ParMatToSolve,KspSolver,srDirMatCtx,ResMat):
        """
        solves a linear system with sequential square matrix as operator and a parallel Matrix as RHS
        
        """
        rowNbr,colNbr=ParMatToSolve.getSize()
        for idx in range(colNbr):
            Seqvec=srDirMatCtx.shareLocalVec(ParMatToSolve.getColumnVector(idx),mode="reverse")
            KspSolver.solve(Seqvec,Seqvec)
            ResMat.setValues(np.arange(0,rowNbr,dtype=np.int32),[idx],Seqvec.getArray()[:])
        ResMat.assemble()

    def updateSrDirMat(self,SrDirMat,SrDirMati,coeffSeqMat,srDirMatCtx):
        """
        remove from W its components on span(Wi) with respect to scalar product of F
        """
        ncol=coeffSeqMat.getSize()[1]
    
        for idx in range(ncol):
            multResParVec=SrDirMati.getVecLeft()
            SrDirMati.mult(srDirMatCtx.shareLocalVec(coeffSeqMat.getColumnVector(idx)),multResParVec)
            start,end=multResParVec.getOwnershipRange()
            SrDirMat.setValues(np.arange(start,end,dtype=np.int32),[idx],
                            -multResParVec.getArray()[:],addv=True)
        SrDirMat.assemble()
        # SrDirMat.transposeMatMult(self.GlbMatrix.matMult(SrDirMati)).view()

    def Orthogonolize(self,lastSrDirMat,
                listOfQMats,lisfOfSrDirMats,listOfDeltaMatKSPs=None, srDirMatCtx=None):
        """
        Do the full Orthonolization process 
        """
        iterNbr=len(lisfOfSrDirMats)
        if listOfDeltaMatKSPs!=None :
            
            # Results matrix Shape has to be  changed for optimized non singular versions of SrDirMats
            tempSrDirMat=lastSrDirMat.duplicate()
            tempDeltaMat=listOfDeltaMatKSPs[0].getOperators()[0].duplicate()
        
            for i in range(iterNbr):
                self.solveMat(listOfQMats[i].transposeMatMult(lastSrDirMat),
                            listOfDeltaMatKSPs[i],srDirMatCtx,tempDeltaMat)
                self.updateSrDirMat(lastSrDirMat,lisfOfSrDirMats[i],tempDeltaMat,srDirMatCtx)
        else :
            for i in range(iterNbr):
                lastSrDirMat.axpy(-1,lisfOfSrDirMats[i].matMult(listOfQMats[i].transposeMatMult(lastSrDirMat))) 
    
    def srDirMatOrthogonalization(self,petscDeltaMat,petscSrDirMat,petscQMat):
        """
        This method turn W to  F-orthogonal matrix, and recompute the new Q matrix
        """

        petscDeltaMat.convert(PETSc.Mat.Type.SEQDENSE)
        deltaMatAsArr=petscDeltaMat.getDenseArray()
        CholFactoredMat,piv,deltaRank,info=dpstrf(deltaMatAsArr,lower=1)
        rankRange=np.arange(0,deltaRank,dtype=np.int32)
        fullRankedLowerTriang=CholFactoredMat[np.ix_(rankRange,rankRange)]
        invertedTransposedL=solve_triangular(fullRankedLowerTriang,np.identity(deltaRank),trans=1,lower=True)
        np.pad(invertedTransposedL,((0,self.SdNumber-deltaRank),(0,0)),mode="constant",constant_values=0)
        invertedTransposedL[piv-1]=invertedTransposedL[rankRange]
        OrthOperator=PETSc.Mat().createAIJ(((1,self.SdNumber),(None,deltaRank)),comm=MPI.COMM_WORLD)
        OrthOperator.setUp()
        OrthOperator.setValues(rankRange,rankRange,invertedTransposedL.flatten())
        # OrthOperator.getDenseArray()[:]=invertedTransposedL[self.sdRank]
        OrthOperator.assemble()
        return petscSrDirMat.matMult(OrthOperator),petscQMat.matMult(OrthOperator)

    def lastError(self,srDirMatrix,ResidualVec):
        tempVec=self.PcIsMatrix.getVecRight()
        self.PcIsMatrix.mult(ResidualVec,tempVec)
        # destroy the two vectors
        return np.sqrt(tempVec.dot(ResidualVec))

    def Solve(self,PetscSolVec,PetscRhsVec,tol):
        
        ResidualVec=PetscRhsVec.duplicate()
        listOfDeltaMatKSPs,listOfsrDirMats,listOfQMats=[],[],[]
        self.GlbMatrix.multAdd(-PetscSolVec,PetscRhsVec,ResidualVec)
        PetscSolVec.zeroEntries()
        srDirMatCtx=self.buildsrDirMatrixCtx(ResidualVec)
        srDirMat=srDirMatCtx.getSrDirMat()
        listOfsrDirMats.append(srDirMat)
        deltaMatCtx=DeltaMatrixCtx(srDirMat,self.GlbMatrix,srDirMatCtx)
        gammaVecSeq=PETSc.Vec().createSeq(self.SdNumber)
        gammaVecPar=PETSc.Vec().createMPI(self.SdNumber,comm=MPI.COMM_WORLD)
        itNbr=0

        while self.lastError(srDirMat,ResidualVec)>tol:
            QSeqMat=deltaMatCtx.getQSeqMat()
            listOfQMats.append(QSeqMat)
            deltaMatKSP,deltaMat=deltaMatCtx.getLocalKSP(),deltaMatCtx.getDeltaSeqMat()
            self.srDirMatOrthogonalization(deltaMat,srDirMat,QSeqMat)
            listOfDeltaMatKSPs.append(deltaMatKSP)
            srDirMat.multTranspose(ResidualVec,gammaVecPar)
            gammaVecSeq=srDirMatCtx.shareLocalVec(gammaVecPar,mode="reverse")
            deltaMatKSP.solve(gammaVecSeq,gammaVecSeq)
            gammaVecPar=srDirMatCtx.shareLocalVec(gammaVecSeq,mode="forward")
            srDirMat.multAdd(gammaVecPar,PetscSolVec,PetscSolVec)
            # F prod Result must have same parallel layout as ResVec
            QSeqMat.multAdd(-gammaVecPar,ResidualVec,ResidualVec)
            srDirMatCtx.update(ResidualVec)
            srDirMat=srDirMatCtx.getSrDirMat()
            self.Orthogonolize(srDirMat,listOfQMats,listOfsrDirMats,listOfDeltaMatKSPs,srDirMatCtx)
            deltaMatCtx.update(srDirMat,srDirMatCtx)
            listOfsrDirMats.append(srDirMat)
            itNbr+=1
        print("l'algo a convergé dans ",itNbr,"itérations")

    def Solve1(self,PetscSolVec,PetscRhsVec,tol):
        
        ResidualVec=PetscRhsVec.duplicate()
        listOfsrDirMats,listOfQMats=[],[]
        self.GlbMatrix.multAdd(-PetscSolVec,PetscRhsVec,ResidualVec)
        PetscSolVec.zeroEntries()
        srDirMatCtx=self.buildsrDirMatrixCtx(ResidualVec)
        srDirMat=srDirMatCtx.getSrDirMat()
        deltaMatCtx=DeltaMatrixCtx(srDirMat,self.GlbMatrix,srDirMatCtx)
        gammaVecSeq=PETSc.Vec().createSeq(self.SdNumber)
        gammaVecPar=PETSc.Vec().createMPI(self.SdNumber,comm=MPI.COMM_WORLD)
        itNbr=0

        while self.lastError(srDirMat,ResidualVec)>tol:
            QSeqMat=deltaMatCtx.getQSeqMat()
            deltaMat=deltaMatCtx.getDeltaSeqMat()
            orthoSrDirMat,orthoQMat=self.srDirMatOrthogonalization(deltaMat,srDirMat,QSeqMat)
            deltaMat.destroy()
            srDirMat.destroy()
            QSeqMat.destroy()
            listOfsrDirMats.append(orthoSrDirMat)
            listOfQMats.append(orthoQMat)
            orthoSrDirMat.multTranspose(ResidualVec,gammaVecPar)
            orthoSrDirMat.multAdd(gammaVecPar,PetscSolVec,PetscSolVec)
            # F prod Result must have same parallel layout as ResVec
            orthoQMat.multAdd(-gammaVecPar,ResidualVec,ResidualVec)
            srDirMatCtx.update(ResidualVec)
            srDirMat=srDirMatCtx.getSrDirMat()
            self.Orthogonolize(srDirMat,listOfQMats,listOfsrDirMats)
            deltaMatCtx.update(srDirMat,srDirMatCtx)
            itNbr+=1
            print("itération \t:",itNbr)    
        print("l'algo a convergé dans ",itNbr,"itérations")

class DeltaMatrixCtx:
    
    """
    build the context of  matrix W^T F W 
    
    """ 
    def __init__(self,srDirMatIs,GlbMatIS,srDirMatCtx,version=None):
        self.srDirMat=srDirMatIs
        self.srDirMatCtx=srDirMatCtx
        self.GlbIsMatrix=GlbMatIS
        self.glbSize=GlbMatIS.getSize()[0]
        self.sdSize=srDirMatIs.getSize()[1]
        self.sdNbr=MPI.COMM_WORLD.Get_size()
        self.sdRank=MPI.COMM_WORLD.Get_rank()
        self.deltaSeqMat,self.QSeqMat=self.buildSeqMat()
        self.version=version
        # self.deltaSeqMat.view()
        if version=="initial":
            self.localKSP=self.buildLocalKSP()
            self.computeAndSetNullSpace()
        else :
            pass
    
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
        deltaMat=PETSc.Mat().createAIJ(self.sdNbr,comm=MPI.COMM_SELF)
        QMat=PETSc.Mat().createAIJ((self.glbSize,self.sdNbr),comm=MPI.COMM_WORLD)
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
            start,end=tempVec2.getOwnershipRange()
            QMat.setValues(np.arange(start,end,dtype=np.int32),[i],
                                tempVec2.getArray()[:])
            QMat.assemble()
            QMat.getValue(start,i)                    
            self.srDirMat.multTranspose(tempVec2,basisElementi)
            deltaMat.setValues(np.arange(0,self.sdNbr,dtype=np.int32),[i],
                            self.srDirMatCtx.shareLocalVec(basisElementi,mode="reverse").getArray()[:])
        deltaMat.assemble()
        QMat.assemble()
        # deltaMat.view()
        return deltaMat,QMat
        
    def buildLocalKSP(self):
        """
        Build local solver for matrix W^T F W
        """
        PetscKsp = PETSc.KSP()
        PetscKsp.create(MPI.COMM_SELF)
        PetscKsp.setOperators(self.deltaSeqMat)
        PetscKsp.setType('preonly')
        PetscPc = PetscKsp.getPC()
        PetscPc.setType('cholesky')
        PetscPc.setFactorSolverType('mumps')
        PetscPc.setFactorSetUpSolverType()

        MumpsFactorMat = PetscPc.getFactorMatrix()
        MumpsFactorMat.setMumpsIcntl(14, 30)
        MumpsFactorMat.setMumpsIcntl(24, 1)
        MumpsFactorMat.setMumpsCntl(3, 1e-6)
        # MumpsFactorMat.setUnfactored()
        # MumpsFactorMat.assemble()

        PetscKsp.setUp()
        # MumpsFactorMat.setUnfactored()
        # MumpsFactorMat.view()
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
        if self.version=="initial":
            self.localKSP=self.buildLocalKSP()
            self.computeAndSetNullSpace()

    
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
        self.colIndices=np.array(self.PcColLGMapIS.getIndices())
        self.rowIndices=np.array(self.PcRowLGMapIS.getIndices())

    def buildLGMap(self):
        rowIS=PETSc.IS().createGeneral(np.arange(0,self.glbSize,step=1,dtype=np.int32),comm=MPI.COMM_WORLD)
        colIS=PETSc.IS().createGeneral([self.sdRank],comm=MPI.COMM_WORLD)
        rowLGMapping=PETSc.LGMap().createIS(rowIS)
        colLGMapping=PETSc.LGMap().createIS(colIS)
        return rowLGMapping,colLGMapping

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
            IndexSetToScatter=PETSc.IS().createStride(glbSize,
                            0,step=1,comm=MPI.COMM_WORLD)
            VecScatter=PETSc.Scatter().create(vecToShareOrGather,IndexSetToScatter,OwnedVec,IndexSetToScatter)
            VecScatter.scatter(vecToShareOrGather,OwnedVec,mode="forward")
            return OwnedVec
        
        else :
            raise RuntimeError("There is no mode called {}: try with 'forward' or 'reverse'".format(mode))

    def computeSdSearchDirection(self):
        #------ To optimize if colIndices and rowIndices are the same ------------
        self.colIndices=np.array(self.PcColLGMapIS.getIndices())
        self.rowIndices=np.array(self.PcRowLGMapIS.getIndices())
        reducedResidualVec=PETSc.Vec().createSeq(len(self.colIndices),comm=MPI.COMM_SELF)
        #-------To optimize if colIndices is the same as data parallel layout on the vector ResidualVec-------
        reducedResidualVec.getArray()[:]=self.shareLocalVec(self.ResidualVec,mode="reverse").getArray()[self.colIndices]
        localProdRes=self.PcIsLocMat.getVecLeft()
        self.PcIsLocMat.mult(reducedResidualVec,localProdRes)
        searchDirVec=localProdRes.getArray()[:]
        return searchDirVec
    
    
    def buildSrDirMatrix(self):
        SrDirMat=PETSc.Mat().createAIJ((self.glbSize,self.sdNbr),comm=MPI.COMM_WORLD)
        SrDirMat.setUp()
        SrDirMat.setValues(self.rowIndices,[self.sdRank],self.SdSearchDir)
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
        SrDirVecArray,PetscMultResArray=self.SdSearchDir,PetscMultRes.getArray()
        PetscVecToMultArray=PetscVecToMult.getArray(readonly=True)
        PetscMultResArray[:]=PetscVecToMultArray[0]*SrDirVecArray[:]
    
    def multTranspose(self,mat,PetscVecToMult,PetscMultRes):
        SrDirVecArray,PetscMultResArray=self.SdSearchDir,PetscMultRes.getArray()
        PetscVecToMultArray=PetscVecToMult.getArray(readonly=True)
        PetscMultResArray[:]=np.dot(PetscVecToMultArray[:],SrDirVecArray[:])

class NullMatrixCtx:
    
    def __init__(self) -> None:
        pass
    
    def mult(self,mat,PetscVecToMult,PetscMultRes):
        # PetscMultRes.getArray()[:]=PetscVecToMult.getArray(readonly=True)[:]
        PetscMultRes.zeroEntries()
    def multTranspose(self,mat,PetscVecToMult,PetscMultRes):
        # PetscMultRes.getArray()[:]=PetscVecToMult.getArray(readonly=True)[:]
        PetscMultRes.zeroEntries()