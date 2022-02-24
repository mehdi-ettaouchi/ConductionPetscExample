import readline
import numpy as np


def ReadInputFile():
    inpFile=open("input_file.txt","r")
    listOfInputs=[]
    line=inpFile.readline()
    while len(line)>0:
        if line=="nombre de noeuds total" or line=="nombre de sous domaines":
            listOfInputs.append(np.int32(inpFile.readline()))
        else:
            listOfInputs.append(float(inpFile.readline()))
        line=inpFile.readline()
    return listOfInputs