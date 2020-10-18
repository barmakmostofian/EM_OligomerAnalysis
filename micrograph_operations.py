import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering




#######################################################################################################################################################
def cluster_beads (All_Coors, CutOff, debug) :
### 'All_Coors' is a list of listed x-y coordinates, e.g. [[0.0, 0.1], [1.0, 1.2]]
### 'CutOff' is the threshold length used for single-linkage clustering

   clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=CutOff, compute_full_tree=True).fit(All_Coors)
   #print("There are", len(All_Coors), "data points in", len(np.unique(clustering.labels_)), "clusters!\n")

   Clusters_Coors = {}
   Clusters_IDs   = {}
   for i in range(len(clustering.labels_)) :
      lab = clustering.labels_[i]
      if lab in Clusters_Coors.keys() :
         Clusters_Coors[lab].append(All_Coors[i])
         Clusters_IDs[lab].append(i)
      else :
         Clusters_Coors[lab] = [All_Coors[i]]
         Clusters_IDs[lab]   = [i]

   return [Clusters_IDs, Clusters_Coors] 
#######################################################################################################################################################





#######################################################################################################################################################
def get_distance_score_matrix (All_Coors, Scoring_Func, debug) :
### 'All_Coors' is a list of listed x-y coordinates, e.g. [[0.0, 0.1], [1.0, 1.2]]; most likely a subset of the overall system, i.e., for instance, a cluster of data points
###
###

   Distance_Matrix       = distance.cdist(All_Coors, All_Coors, 'euclidean')

   Distance_Score_Matrix = np.zeros(Distance_Matrix.shape)
   for i in range(len(Distance_Matrix)) :
      for j in range(len(Distance_Matrix[i])) :
         if i != j :
            Distance_Score_Matrix[i][j] = Scoring_Func(Distance_Matrix[i][j])
         else :
            Distance_Score_Matrix[i][j] = 0
   
   return Distance_Score_Matrix
#######################################################################################################################################################





#######################################################################################################################################################
def get_angle_score_matrix (All_Coors, Scoring_Func, debug) :
### 'All_Coors' is a list of listed x-y coordinates, e.g. [[0.0, 0.1], [1.0, 1.2]]; most likely a subset of the overall system, i.e., for instance, a cluster of data points
###

   Angle_Matrix = np.zeros((len(All_Coors), len(All_Coors), len(All_Coors)))
   for i in range(0, len(Angle_Matrix)) :
      for j in range(0, len(Angle_Matrix)) :
         for k in range(0, len(Angle_Matrix)) :
            if (i == j) or (j == k) or (i == k):
               Angle_Matrix[i][j][k] = -1  ##We need to get rid of the zero set initially because that is actually a high-scoring angle
            else :
               Bond1 = np.array(All_Coors[j]) - np.array(All_Coors[i])
               Bond2 = np.array(All_Coors[k]) - np.array(All_Coors[j])
               cosine = dot(Bond1,Bond2)/norm(Bond1)/norm(Bond2)
               angle  = arccos(clip(cosine, -1, 1))
               Angle_Matrix[i][j][k] = angle/np.pi * 180

   Angle_Score_Matrix = np.zeros(Angle_Matrix.shape)
   for i in range(len(Angle_Matrix)) :
      for j in range(len(Angle_Matrix[i])) :
         for k in range(len(Angle_Matrix[j])) :
            if Angle_Matrix[i][j][k] >= 0  :  ##Zero score for undefined angles
               Angle_Score_Matrix[i][j][k] = Scoring_Func(Angle_Matrix[i][j][k])

   return Angle_Score_Matrix
#######################################################################################################################################################





#######################################################################################################################################################
def get_initial_dimers(Distance_Score_Matrix, Scoring_Func, AvgPixels_IDs, Clusters_IDs, lab, debug) :
###
###

   Dimer_IDs    = []
   Dimer_Scores = []
 
   [r, c] = np.triu_indices(len(Distance_Score_Matrix), 1)                      ##Indices for the upper triangle (i.e., offset by 1 from the diagonal) of a square matrix of given length
   Sorted_Distance_Scores = sorted(Distance_Score_Matrix[r,c], reverse=True)    ##Sorted values of that given matrix section (off-diagonal, thus non-zero distances with itself, but note that the distance scores may well be zero)
 
   scorenum = 0 
   while scorenum < len(Sorted_Distance_Scores) :  ##Using a while instead of for loop because there may be multiple pairs with the same score (so we increment the running parameter for every stored pair)
      if debug: print("Scorenum is", scorenum)
      distscore = Sorted_Distance_Scores[int(scorenum)] ##Making the parameter an integer because it becomes a (whole-numbered!) float below
      if debug: print("Currdistscore is", distscore)
      if distscore > 0 :  ##Consider only scores that are not beyond the min/max region
         Distance_Score_rowIDs = np.where(Distance_Score_Matrix == distscore)[0]  ##We sorted the distance scores only by the upper triangle of the distance matrix, but we are checking against the whole matrix ...
         Distance_Score_colIDs = np.where(Distance_Score_Matrix == distscore)[1]  ##... so, for each score, there will be both (i,j) and (j,i) pairs (and, of course, i!=j because we had disregarded the diagonal)
         for idnum in range(len(Distance_Score_rowIDs)) : ##This is needed if the same score occurs more than once, but also, in particular, each dimer is saved in both "directions", i.e., both (i,j) and (j,i) are continued here, as either could have a better trimer score
            beadscore1 = Scoring_Func(AvgPixels_IDs[Clusters_IDs[lab][Distance_Score_rowIDs[idnum]]])  
            beadscore2 = Scoring_Func(AvgPixels_IDs[Clusters_IDs[lab][Distance_Score_colIDs[idnum]]])
            if (beadscore1 > 0) and (beadscore2 > 0) :  ##Consider only  pixel-intensity scores that are not beyond the min/max region
               Dimer_IDs.append([Distance_Score_rowIDs[idnum], Distance_Score_colIDs[idnum]])
               Dimer_Scores.append(np.log(distscore) + np.log(beadscore1) + np.log(beadscore2)) ##The dimer score consists of the distance score plus the two individual bead scores.
               if debug: print("CurrDimerID", Dimer_IDs[-1])
               if debug: print("CurrDimerScore", Dimer_Scores[-1])
            scorenum += 0.5  ##Only adding one half because there will be two "idnum" elements, i.e., (i,j) and (j,i), for each "distscore"
      else :
         scorenum += 1
   
   return [Dimer_IDs, Dimer_Scores]	# Dimers are automatically sorted by score because the array is filled based on 'Sorted_Distance_Scores'
#######################################################################################################################################################





#######################################################################################################################################################
def extend_oligomers (Oligomer_IDs_ByLength, Oligomer_Scores_ByLength, oligolen, Distance_Score_Matrix, Angle_Score_Matrix, Scoring_Func, AvgPixels_IDs, Clusters_IDs, lab, debug) :
###
###

   Curr_Oligomer_IDs    = []
   Curr_Oligomer_Scores = []

   Prev_Oligomer_Num = len(Oligomer_IDs_ByLength[oligolen-1])
   for oligonum in range(Prev_Oligomer_Num) :
      oligomer     = Oligomer_IDs_ByLength[oligolen-1][oligonum]
      oligoscore   = Oligomer_Scores_ByLength[oligolen-1][oligonum]
      bead_pre     = oligomer[-1]
      bead_prepre  = oligomer[-2]
      if debug: print("Checking Oligomer", oligomer, "with oligoscore", oligoscore)

      Sorted_Distance_Scores = sorted(Distance_Score_Matrix[bead_pre], reverse=True)
      if debug: print("Its sorted distance scores to new beads are", Sorted_Distance_Scores)

      scorenum = 0
      while scorenum < len(Sorted_Distance_Scores) :  ##Similar approach to 'get_initial_dimers'
         distscore = Sorted_Distance_Scores[scorenum]
         if debug: print("ScoreNum is", scorenum, "and the corresponding distance score is", distscore)
         if distscore > 0 :
            Distance_Score_colIDs = np.where(Distance_Score_Matrix[bead_pre] == distscore)[0]
            if debug: print("It occurs with these potential new beads", Distance_Score_colIDs)
            for idnum in range(len(Distance_Score_colIDs)) :
               if Distance_Score_colIDs[idnum] in oligomer:  ##It might be one of the previous beads, in particular 'bead_prepre'
                  if debug: print(Distance_Score_colIDs[idnum], "already in", oligomer)
                  scorenum += 1
                  continue
               else :
                  beadscore = Scoring_Func(AvgPixels_IDs[Clusters_IDs[lab][Distance_Score_colIDs[idnum]]])   ##Cluster_IDs and lab defined in 'main'
                  anglescore = Angle_Score_Matrix[bead_prepre][bead_pre][Distance_Score_colIDs[idnum]]
                  if debug: print("Addding bead", Distance_Score_colIDs[idnum], "would give the new distance score", distscore, "and angle score", anglescore, "and bead score", beadscore)
                  if (beadscore > 0) and (anglescore > 0) :  ##Consider only scores that are not beyond the min/max angle and pixel-intensity region
                     new_oligomer = oligomer + [Distance_Score_colIDs[idnum]]
                     Curr_Oligomer_IDs.append(new_oligomer)
                     Curr_Oligomer_Scores.append(oligoscore + np.log(distscore) + np.log(anglescore) + np.log(beadscore))
                     if debug: print("new oligomer is", new_oligomer, "with log-score", Curr_Oligomer_Scores[-1], "\n")
                  scorenum += 1
         else :
            scorenum += 1  ##A distance score of 0.0 will occur this time because of distances to itself (diagonal values were excluded before)

   return [Curr_Oligomer_IDs, Curr_Oligomer_Scores]
#######################################################################################################################################################





#######################################################################################################################################################
def sort_oligomers_by_score (Oligomer_IDs, Oligomer_Scores, order,   debug) :
###
###

   ##Sorting by increasing score
   Oligomer_IDs_SortedByScore  =  [ Oligomer_IDs for Oligomer_Scores,Oligomer_IDs  in sorted(zip(Oligomer_Scores,Oligomer_IDs)) ]
   Oligomer_Scores.sort()
   Oligomer_Scores_SortedByScore = Oligomer_Scores  ##Just to have the names of 'IDs' and 'Scores' consistent
   if order == 'decr' :
      ##Sorting by decreasing score
      Oligomer_IDs_SortedByScore.reverse()
      Oligomer_Scores_SortedByScore.reverse()

   return [Oligomer_IDs_SortedByScore, Oligomer_Scores_SortedByScore]
#######################################################################################################################################################





#######################################################################################################################################################
def discard_oligomers_by_normalscore_threshold (Oligomer_IDs, Oligomer_Scores, score_threshold,   debug) :
###
###

   NormalScore_CutOff = np.log(score_threshold)  ##probability average per bond and angle used as threshold here
   Remaining_Oligomer_IDs    = []
   Remaining_Oligomer_Scores = []

   for i in range(len(Oligomer_IDs)) :
      oligomer  = Oligomer_IDs[i]
      bondnums  = len(oligomer) - 1
      anglenums = len(oligomer) - 2
      normalscore = Oligomer_Scores[i]/float(bondnums+anglenums)  ##We normalize the oligomer scores for cross-oligolen comparison within a cluster (e.g. when removing some because of overlapping)

      if normalscore > NormalScore_CutOff :
         Remaining_Oligomer_IDs.append(oligomer)
         Remaining_Oligomer_Scores.append(normalscore)

   if debug: print("These are found in the cluster:\nIDs:", Remaining_Oligomer_IDs, "\nScores:", Remaining_Oligomer_Scores, "\n")

   return [Remaining_Oligomer_IDs, Remaining_Oligomer_Scores]
#######################################################################################################################################################





#######################################################################################################################################################
def sort_oligomers_by_length (Oligomer_IDs, Oligomer_Scores, order,   debug) :
###
###

   ##Sorting by increasing length (if more than one has the same length and the items were sorted by increasing score, then they are still sorted by increasing score)
   Oligomer_Lengths = []
   for i in Oligomer_IDs :      
      Oligomer_Lengths.append(len(i))
   Oligomer_Scores_SortedByLen = [ Oligomer_Scores for Oligomer_Lengths,Oligomer_Scores  in sorted(zip(Oligomer_Lengths,Oligomer_Scores)) ]  ##This sorts by the length (the 1st element in the zipped list); when the lengths are identical, then it sorts by the score value (the 2nd element in the zipped list)
   Oligomer_IDs.sort(key=len)  ##Sorting by length
   Oligomer_IDs_SortedByLen = Oligomer_IDs  ##Just to have the names of 'IDs' and 'Scores' consistent
   if order == 'decr' :
      ##Sorting by decreasing length
      Oligomer_IDs_SortedByLen.reverse()
      Oligomer_Scores_SortedByLen.reverse()

   return [Oligomer_IDs_SortedByLen, Oligomer_Scores_SortedByLen]
#######################################################################################################################################################





#######################################################################################################################################################
def remove_oligomer_overlaps (Oligomer_IDs, Oligomer_Scores) :
###
###

   NonOverlap_Oligomer_IDs   = []
   NonOverlap_Oligomer_Score = [] 
   NonOverlap_Oligomer_IDs.append(Oligomer_IDs[0])
   NonOverlap_Oligomer_Score.append(Oligomer_Scores[0])

   for i in range(len(Oligomer_IDs)) :
      oligomer = Oligomer_IDs[i]
      score    = Oligomer_Scores[i] 
      overlap = 0
      for saved_oligomer in NonOverlap_Oligomer_IDs :
         if any(bead in saved_oligomer for bead in oligomer) :
            overlap += 1
      if not overlap :
         NonOverlap_Oligomer_IDs.append(oligomer)
         NonOverlap_Oligomer_Score.append(score)

   return [NonOverlap_Oligomer_IDs, NonOverlap_Oligomer_Score]
#######################################################################################################################################################









#######################################################################################################################################################
def assign_oligomers (All_Coors,  Oligomer_Coors_ByLength, MinDist) :
### 'AllCoors' is a list of listed x-y coordinates, e.g. [[0.0, 0.1], [1.0, 1.2]]
### 'Oligomer_Coors_ByLength' is a dictionary where the key is the oligomer length and the corresponding value is a list of oligomers (of that length), which itself are lists of listed x-y coordinantes, e.g. [[[0,1], [1,2], [2,3]],  [[1,0], [1,1]]] 
### 'MinDist' is the minimum distance that an oligomer is allowed to have to another bead or particle in order to be counted as 'cleared'
 

   Singleton_Coors 		     = []   # The list of singleton coordinates, that are NOT subject to any distance criterion
   Cleared_Singleton_Coors	     = []
   UnCleared_Singleton_Coors	     = []
   Cleared_Oligomer_Coors_ByLength   = {}   # The dictionary of cleared oligomers (with their length being the key value)
   UnCleared_Oligomer_Coors_ByLength = {}   # The dictionary of oligomers that are not cleared, i.e., they are in proximity to another bead by less than 'MinDist'
   UnCleared_Oligomer_Coors          = []   # The list of coordinates that are from uncleared oligomers (required to get the locked oligomers easily) 
   Locked_Oligomer_Coors_ByLength    = {}   # The dictionary of oligomers that are not cleared because they are in proximity to another oligomer by less than 'MinDist'
   Blocked_Oligomer_Coors_ByLength   = {}   # The dictionary of oligomers that are not cleared because they are in proximity to a singleton by less than 'MinDist'
   Possible_Oligomer_Coors_ByLength  = {}   # The dictionary of putative oligomers, i.e., they are either 'Cleared' or (currently) 'Blocked' by a singleton
   Excluded_Coors		     = []   # The list of all uncleared beads, i.e., uncleared oligomers AND singletons



####### IDENTIFY SINGLETONS #########

   print("Assigning all singletons")
   
   for bead in All_Coors :
      single = 1
      for oligolen in Oligomer_Coors_ByLength.keys() :
         for oligomer in Oligomer_Coors_ByLength[oligolen] :
            if bead in oligomer :
               single = 0
               continue
         if single == 0 :
            continue
      if single : 
         Singleton_Coors.append(bead)   
#      if len(Singleton_Coors)%200 == 0 : print("found another 200 singletons", len(Singleton_Coors))

######################################



###### IDENTIFY UNCLEARED OLIGOMERS  #######

   print("Assigning uncleared oligomers")

   for oligolen in Oligomer_Coors_ByLength.keys() :
      UnCleared_Oligomer_Coors_ByLength[oligolen] = []
      for oligomer in Oligomer_Coors_ByLength[oligolen] :
         for bead1 in oligomer :
            for bead2 in All_Coors :
               if bead2 in oligomer :
                  continue
               else :
                  bondvec = np.array(bead1) - np.array(bead2)
                  if (norm(bondvec) < MinDist)  and  (oligomer not in UnCleared_Oligomer_Coors_ByLength[oligolen]) : 
                     UnCleared_Oligomer_Coors_ByLength[oligolen].append(oligomer) 

   for oligolen in UnCleared_Oligomer_Coors_ByLength.keys() :
      for oligomer in UnCleared_Oligomer_Coors_ByLength[oligolen] :
         for bead in oligomer :
            UnCleared_Oligomer_Coors.append(bead)

#################################################


               
###### IDENTIFY CLEARED OLIGOMERS  #######

   print("Assigning cleared oligomers")

   for oligolen in Oligomer_Coors_ByLength.keys() :
      Cleared_Oligomer_Coors_ByLength[oligolen] = []
      for oligomer in Oligomer_Coors_ByLength[oligolen] :
         if oligomer not in UnCleared_Oligomer_Coors_ByLength[oligolen] :
            Cleared_Oligomer_Coors_ByLength[oligolen].append(oligomer)

###########################################



###### IDENTIFY LOCKED OLIGOMERS  #######

   print("Assigning locked oligomers")

   for oligolen in UnCleared_Oligomer_Coors_ByLength.keys() :
      Locked_Oligomer_Coors_ByLength[oligolen] = []
      for oligomer in UnCleared_Oligomer_Coors_ByLength[oligolen] :
         for bead1 in oligomer :
            for bead2 in UnCleared_Oligomer_Coors :
               if bead2 in oligomer:
                  continue
               else :
                  bondvec = np.array(bead1) - np.array(bead2)
                  if (norm(bondvec) < MinDist)  and  (oligomer not in Locked_Oligomer_Coors_ByLength[oligolen]) :
                     Locked_Oligomer_Coors_ByLength[oligolen].append(oligomer)

########################################



###### IDENTIFY BLOCKED OLIGOMERS  #######

   print("Assigning blocked (and thus the remaining 'possible') oligomers")

   for oligolen in UnCleared_Oligomer_Coors_ByLength.keys() :
      Blocked_Oligomer_Coors_ByLength[oligolen] = []
      Possible_Oligomer_Coors_ByLength[oligolen] = []
      for oligomer in UnCleared_Oligomer_Coors_ByLength[oligolen] :
         if oligomer not in Locked_Oligomer_Coors_ByLength[oligolen] :
            Blocked_Oligomer_Coors_ByLength[oligolen].append(oligomer)

   for oligolen in Blocked_Oligomer_Coors_ByLength.keys() :
      for oligomer in Blocked_Oligomer_Coors_ByLength[oligolen] :
         Possible_Oligomer_Coors_ByLength[oligolen].append(oligomer)
      for oligomer in Cleared_Oligomer_Coors_ByLength[oligolen] :
         Possible_Oligomer_Coors_ByLength[oligolen].append(oligomer)

##########################################



###### IDENTIFY UNCLEARED SINGLETONS  #######

   print("Assigning uncleared singletons (and thus all excluded beads)")

   for beadID1 in range(len(Singleton_Coors)-1) :
   #   if beadID1%200 == 0:  print("Checking bead", beadID1, "/", len(Singleton_Coors))
      bead1 = Singleton_Coors[beadID1]
      for beadID2 in range(beadID1+1, len(Singleton_Coors)) :
         bead2 = Singleton_Coors[beadID2]
         bondvec = np.array(bead1) - np.array(bead2)
         if (norm(bondvec) < MinDist) :
            if bead1 not in UnCleared_Singleton_Coors :  # We don't wannt to check this earlier and 'continue' b/c we might leave out a bead that should be counnted
               UnCleared_Singleton_Coors.append(bead1) 
            if bead2 not in UnCleared_Singleton_Coors : 
               UnCleared_Singleton_Coors.append(bead2) 

   for bead1 in Singleton_Coors :
      for bead2 in UnCleared_Oligomer_Coors :
         bondvec = np.array(bead1) - np.array(bead2)
         if (norm(bondvec) < MinDist)  and  (bead1 not in UnCleared_Singleton_Coors) :
            UnCleared_Singleton_Coors.append(bead1) 
            continue

      if bead1 not in UnCleared_Singleton_Coors :
         Cleared_Singleton_Coors.append(bead1)

   for bead in UnCleared_Singleton_Coors :
      Excluded_Coors.append(bead)
   for bead in UnCleared_Oligomer_Coors :
      Excluded_Coors.append(bead)

###############################################


   return [Singleton_Coors, Cleared_Singleton_Coors, UnCleared_Singleton_Coors, Cleared_Oligomer_Coors_ByLength, UnCleared_Oligomer_Coors_ByLength, Blocked_Oligomer_Coors_ByLength, Possible_Oligomer_Coors_ByLength, Locked_Oligomer_Coors_ByLength, Excluded_Coors]



