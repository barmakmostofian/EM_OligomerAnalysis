import sys
from micrograph_operations import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



######### LOADING/DEFINING PARAMETERS ##########

if (len(sys.argv) == 6 and (sys.argv[5] == "debug")) :
 debug = 1
else :
 debug = 0

FileIn_Picks    = sys.argv[1]
FileIn_Peaks    = sys.argv[2]
CDFs_Dir        = sys.argv[3]
Score_Threshold = float(sys.argv[4])

DataName = FileIn_Picks.split("DogPicks/")[1].split(".star")[0]

Clustering_Threshold  = 8.0
Min_Percentile        = 0.995
Max_Percentile        = 0.005
Min_Oligolen = 2
Max_Oligolen = 8





######### CLUSTER DOG-PICK DATA ##########

print("### CLUSTERING ###\n")

All_Coors = []
FileIn = open(FileIn_Picks)
for Line in FileIn.readlines() :
   Words = Line.split()
   if len(Words) == 5 :   # Reading in pixel coordinates of star files
      All_Coors.append([0.437*float(Words[0]), 0.437*float(Words[1])])  ##Each pixel is worth 0.437nm
FileIn.close()

[Clusters_IDs, Clusters_Coors] = cluster_beads(All_Coors, Clustering_Threshold, debug)
print("There are", len(All_Coors), "data points in", len(Clusters_IDs.keys()), "clusters!\n")





######### ASSIGN DOG-PICKS TO AVG. PEAK PIXEL VALUES ##########

print("### ASSIGNING PEAK PIXEL VALUES ###\n")

GraphPeaks = np.flipud(matplotlib.pyplot.imread(FileIn_Peaks))
GraphPeaks_Normal = GraphPeaks/np.average(GraphPeaks)

AvgPixels_IDs = {}

for beadID in range(len(All_Coors)) :
   PickPosX = All_Coors[beadID][0]/0.437                                        #Converting nm back to pixels for this operation
   PickPosY = len(GraphPeaks) - float(All_Coors[beadID][1])/0.437 - 1           #Y-coors go upside down on the image
   PickPixels = []
   PickPixels.append(GraphPeaks_Normal[int(PickPosY)][int(PickPosX)])
   for j in range(1, 4) :
      PickPixels.append(GraphPeaks_Normal[int(PickPosY)-j][int(PickPosX)])
      PickPixels.append(GraphPeaks_Normal[int(PickPosY)+j][int(PickPosX)])
   for k in range(1, 4) :
      PickPixels.append(GraphPeaks_Normal[int(PickPosY)][int(PickPosX)]-k)
      PickPixels.append(GraphPeaks_Normal[int(PickPosY)][int(PickPosX)]+k)
   for j in range(1, 4) :
      for k in range(1, (4-j)) :
         PickPixels.append(GraphPeaks_Normal[int(PickPosY)-j][int(PickPosX)+k])
         PickPixels.append(GraphPeaks_Normal[int(PickPosY)+j][int(PickPosX)+k])
         PickPixels.append(GraphPeaks_Normal[int(PickPosY)-j][int(PickPosX)-k])
         PickPixels.append(GraphPeaks_Normal[int(PickPosY)+j][int(PickPosX)-k])
   AvgPixels_IDs[beadID] = np.average(PickPixels)





######### DEFINE SCORING FXNS ##########

print("### DEFINING SCORING FXNS ###\n")

#Distances

Values_Distances = []
Scores_Distances = []

FileIn = open(CDFs_Dir+"/CDF_Distances.dat")
for Line in FileIn.readlines() :
   Words = Line.split()
   Values_Distances.append(float(Words[0]))
   Scores_Distances.append(1-float(Words[1])) ##(1-CDF) is score function
FileIn.close()

Values_Distances = np.array(Values_Distances)
Scores_Distances = np.array(Scores_Distances)

MinDist =  Values_Distances[list(map(lambda i: i<Min_Percentile, Scores_Distances)).index(True)]
MaxDist =  Values_Distances[list(map(lambda i: i<Max_Percentile, Scores_Distances)).index(True)]

def Scoring_Bonds (x) :
   if (x < MinDist) or (x > MaxDist) :
      return 0
   else :
      return Scores_Distances[(np.abs(Values_Distances - x)).argmin()]


#Angles

Values_Angles = []
Scores_Angles = []

FileIn = open(CDFs_Dir+"/CDF_Angles.dat")
for Line in FileIn.readlines() :
   Words = Line.split()
   Values_Angles.append(float(Words[0]))
   Scores_Angles.append(1-float(Words[1])) ##(1-CDF) is score function
FileIn.close()

Values_Angles = np.array(Values_Angles)
Scores_Angles = np.array(Scores_Angles)

MaxAngle =  Values_Angles[list(map(lambda i: i<Max_Percentile, Scores_Angles)).index(True)]
# No minimum angle

def Scoring_Angles (x) :
   if (x > MaxAngle) :
      return 0
   else :
      return Scores_Angles[(np.abs(Values_Angles - x)).argmin()]


#Avg.Pixels

Values_AvgPixels = []
Scores_AvgPixels = []

FileIn = open(CDFs_Dir+"/CDF_AvgPixels.dat")
for Line in FileIn.readlines() :
   Words = Line.split()
   Values_AvgPixels.append(float(Words[0]))
   Scores_AvgPixels.append(float(Words[1])) ##CDF is score function
FileIn.close()

Values_AvgPixels = np.array(Values_AvgPixels)
Scores_AvgPixels = np.array(Scores_AvgPixels)

MinAvgPixels =  Values_AvgPixels[list(map(lambda i: i>Max_Percentile, Scores_AvgPixels)).index(True)]  ##This is the other way around in terms of min/max percentiles because the actual CDF is the score
# No maximum avg. pixel value

def Scoring_Beads (x) :
   if (x < MinAvgPixels) :
      return 0
   else :
      return Scores_AvgPixels[(np.abs(Values_AvgPixels - x)).argmin()]





########## SCORE OLIGOMERS IN ALL CLUSTERS ##########

print("### SCORING ###\n")

All_NonOverlapping_Oligomer_Coors_ByLength        = {}
All_NonOverlapping_Oligomer_NormalScores_ByLength = {}
for oligolen in range(Min_Oligolen, Max_Oligolen+1) :
   All_NonOverlapping_Oligomer_Coors_ByLength[oligolen]        = []
   All_NonOverlapping_Oligomer_NormalScores_ByLength[oligolen] = []
##These are the oligomer coordinates and (normalized) scores retrieved from all clusters and saved as a function of their lengths


for lab in sorted(Clusters_Coors.keys()) :

   Oligomer_IDs_ByLength    = {}   
   Oligomer_Scores_ByLength = {}   
   for oligolen in range(Min_Oligolen, Max_Oligolen+1) :
      Oligomer_IDs_ByLength[oligolen]    = []
      Oligomer_Scores_ByLength[oligolen] = []


   Distance_Score_Matrix      =  get_distance_score_matrix(Clusters_Coors[lab], Scoring_Bonds,   debug)
   Angle_Score_Matrix         =  get_angle_score_matrix(Clusters_Coors[lab], Scoring_Angles,   debug)

   
   oligolen = 2
   [Dimer_IDs, Dimer_Scores] =  get_initial_dimers(Distance_Score_Matrix, Scoring_Beads, AvgPixels_IDs, Clusters_IDs, lab,  debug)
   Oligomer_IDs_ByLength[oligolen]    = Dimer_IDs
   Oligomer_Scores_ByLength[oligolen] = Dimer_Scores

   if debug: print("Dimers:", Oligomer_IDs_ByLength[2], Oligomer_Scores_ByLength[2])


   for oligolen in range(3, Max_Oligolen+1) :
      [Curr_Oligomer_IDs, Curr_Oligomer_Scores]  =  extend_oligomers(Oligomer_IDs_ByLength, Oligomer_Scores_ByLength, oligolen, Distance_Score_Matrix, Angle_Score_Matrix, Scoring_Beads, AvgPixels_IDs, Clusters_IDs, lab,  debug)
      [Curr_Oligomer_IDs_Sorted, Curr_Oligomer_Scores_Sorted]  =  sort_oligomers_by_score(Curr_Oligomer_IDs, Curr_Oligomer_Scores, 'decr',   debug)
      ## Sorting is performed at every iteration in case one has a ton of particles and wishes to extend only the highest scoring oligomers
      for i in range(len(Curr_Oligomer_IDs_Sorted)) :
         Oligomer_IDs_ByLength[oligolen].append(Curr_Oligomer_IDs_Sorted[i])
         Oligomer_Scores_ByLength[oligolen].append(Curr_Oligomer_Scores_Sorted[i])

      if debug: print("These are found for the oligolen:\nIDs:",  Oligomer_IDs_ByLength[oligolen], "\nScores:", Oligomer_Scores_ByLength[oligolen], "\n")


   All_Possible_Oligomer_IDs    = []
   All_Possible_Oligomer_Scores = []
   for oligolen in range(Min_Oligolen, Max_Oligolen+1) :
      for oligomer_ids in Oligomer_IDs_ByLength[oligolen] :
         All_Possible_Oligomer_IDs.append(oligomer_ids)
      for oligomer_scores in Oligomer_Scores_ByLength[oligolen] :
         All_Possible_Oligomer_Scores.append(oligomer_scores)

   [Remaining_Oligomer_IDs, Remaining_Oligomer_NormalScores]  =  discard_oligomers_by_normalscore_threshold(All_Possible_Oligomer_IDs, All_Possible_Oligomer_Scores, Score_Threshold,   debug)

   if len(Remaining_Oligomer_IDs) > 0 :  ##if any left after thresholding, then sort, remove overlaps, and put the coordinates in the global dictionary

      [Remaining_Oligomer_IDs_SortedByScore, Remaining_Oligomer_NormalScores_SortedByScore] = sort_oligomers_by_score(Remaining_Oligomer_IDs, Remaining_Oligomer_NormalScores, 'incr',   debug)
      [Remaining_Oligomer_IDs_SortedByLengthAndScore, Remaining_Oligomer_NormalScores_SortedByLengthAndScore] = sort_oligomers_by_length(Remaining_Oligomer_IDs_SortedByScore, Remaining_Oligomer_NormalScores_SortedByScore, 'decr',   debug)
      [NonOverlap_Oligomer_IDs, NonOverlap_Oligomer_NormalScores] = remove_oligomer_overlaps(Remaining_Oligomer_IDs_SortedByLengthAndScore, Remaining_Oligomer_NormalScores_SortedByLengthAndScore)

      if debug: print("These are kept, sorted, and free of overlaps\nIDs: ", NonOverlap_Oligomer_IDs, "\nScores: ", NonOverlap_Oligomer_NormalScores, "\n\n")

      for oligonum in range(len(NonOverlap_Oligomer_IDs)) :
         oligomer     = NonOverlap_Oligomer_IDs[oligonum]
         oligolen     = len(oligomer)
         normalscore  = NonOverlap_Oligomer_NormalScores[oligonum]
         realoligomer = []
         for beadid in oligomer :
            realoligomer.append(Clusters_Coors[lab][beadid])
         All_NonOverlapping_Oligomer_Coors_ByLength[oligolen].append(realoligomer)
         All_NonOverlapping_Oligomer_NormalScores_ByLength[oligolen].append(normalscore)




FileOut = open("ScoredPop_"+DataName+"_ScoreCutOff"+"{0:.2f}".format(Score_Threshold)+"_NewCode.dat", 'w')
for oligolen in range(Min_Oligolen, Max_Oligolen+1) :
   print(oligolen, "\t", len(All_NonOverlapping_Oligomer_Coors_ByLength[oligolen]))
   print(oligolen, "\t", len(All_NonOverlapping_Oligomer_Coors_ByLength[oligolen]), file=FileOut)


np.savez("Scored_"+DataName+"_ScoreCutOff"+"{0:.2f}".format(Score_Threshold)+"_NewCode.npz", 
np.array(All_Coors), 
np.array(All_NonOverlapping_Oligomer_Coors_ByLength), 
np.array(All_NonOverlapping_Oligomer_NormalScores_ByLength)
)


