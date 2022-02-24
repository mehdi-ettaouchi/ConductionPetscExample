# ConductionPetscExample
Conduction dans une plaque carré :  Système implémenté par le type de matrice IS de PETSc

                        Tup=0
   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
  |                                                     |
  |                                                     | 
  |                                                     |
  |                                                     |
  |                                                     |  
  |                                                     |
  |                                                     |  
Tl|                                                     | Tr
  |                                                     |
  |                                                     |
  |                                                     |  
  |                                                     |  
  |                                                     |
  |                                                     |  
  |                                                     |
  |__ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __|
                          Tdown=0
 le code fournis une heatmap de la température à l'instant choisi
 
 Pour executer le code :
        mpiexec -n "nombre de sous domaines choisi" python3 test_solver.py  
