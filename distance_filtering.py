import sys
import subprocess
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from micrograph_operations import assign_oligomers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


FileIn = sys.argv[1]
MinDist = float(sys.argv[2])  # The clearance distance for oligomers

npzfile  = np.load(FileIn, allow_pickle=True)
DataName = FileIn.split("../")[1].split(".npz")[0]
All_Coors = npzfile['arr_0'].tolist()
Oligomer_Coors_ByLength  = npzfile['arr_1'].tolist()
Oligomer_Scores_ByLength = npzfile['arr_2'].tolist()

print("\nThere are a total of", len(All_Coors), "beads\n")


All_Oligomers = []
All_Oligomer_Coors = []
for oligolen in Oligomer_Coors_ByLength.keys() :
   for oligomer in Oligomer_Coors_ByLength[oligolen] :
      All_Oligomers.append(oligomer)
      for bead in oligomer :
         All_Oligomer_Coors.append(bead)

print("There are", len(All_Oligomer_Coors), "beads assigned to", len(All_Oligomers), "oligomers\n")

[Singleton_Coors,
Cleared_Singleton_Coors,
UnCleared_Singleton_Coors,
Cleared_Oligomer_Coors_ByLength,
UnCleared_Oligomer_Coors_ByLength,
Blocked_Oligomer_Coors_ByLength,
Possible_Oligomer_Coors_ByLength,
Locked_Oligomer_Coors_ByLength,
Excluded_Coors]  =  assign_oligomers(All_Coors, Oligomer_Coors_ByLength, MinDist)


np.savez("Assigned_"+DataName+"_ClearanceDistance"+str(MinDist)+"nm.npz",
np.array(All_Coors),
np.array(Excluded_Coors),
np.array(Cleared_Oligomer_Coors_ByLength),
np.array(Blocked_Oligomer_Coors_ByLength),
np.array(Possible_Oligomer_Coors_ByLength),
np.array(Locked_Oligomer_Coors_ByLength),
np.array(Singleton_Coors),
np.array(Cleared_Singleton_Coors),
np.array(UnCleared_Singleton_Coors))

