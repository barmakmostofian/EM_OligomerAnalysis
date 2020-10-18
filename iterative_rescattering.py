import sys
import subprocess
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
from scipy.spatial import distance
import random
from sklearn.cluster import AgglomerativeClustering
from micrograph_operations import assign_oligomers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



###### DEFINE PARAMETERS #######

GraphID  = sys.argv[1]
IterBeg  = 0
IterEnd  = 99
PlotFreq = 10



for IterID in range(IterBeg, IterEnd+1) :

   print("\n\n>>>>>>>   RUNNING ITERATION", IterID, "   <<<<<<<<\n\n")


   ###### LOAD DATA #####

   if IterID == 0 :
      FileIn = "../../AssignOligomersAndSingletons/Assigned_Scored_LC8_4mer_Q8_49k_def_1.97_"+GraphID+"_dog_r=8-t=4+1000_MaxDist6.0nm_MaxAngle75deg_WholeScoreCutOff0.10.npz" 
      DataName = FileIn.split("AssignOligomersAndSingletons/")[-1].split(".npz")[0]
   else :
      FileIn = "./ShuffledAndReScored_Assigned_Scored_LC8_4mer_Q8_49k_def_1.97_"+GraphID+"_dog_r=8-t=4+1000_MaxDist6.0nm_MaxAngle75deg_WholeScoreCutOff0.10_Iter"+'{0:02}'.format(IterID-1)+".npz"
      DataName = FileIn.split("ShuffledAndReScored_")[-1].split("_Iter")[0]

   npzfile  = np.load(FileIn, allow_pickle=True)
   All_Coors_ORIG                        = npzfile['arr_0'].tolist()
   Excluded_Coors_ORIG                   = npzfile['arr_1'].tolist()
   Cleared_Oligomer_Coors_ByLength_ORIG  = npzfile['arr_2'].tolist()
   Blocked_Oligomer_Coors_ByLength_ORIG  = npzfile['arr_3'].tolist()
   Possible_Oligomer_Coors_ByLength_ORIG = npzfile['arr_4'].tolist()
   Locked_Oligomer_Coors_ByLength_ORIG   = npzfile['arr_5'].tolist()
   Singleton_Coors_ORIG        	         = npzfile['arr_6'].tolist()
   Cleared_Singleton_Coors_ORIG          = npzfile['arr_7'].tolist()
   UnCleared_Singleton_Coors_ORIG        = npzfile['arr_8'].tolist()

   print("\nAll beads (original):", len(All_Coors_ORIG)) 
   print("All singletons (original):", len(Cleared_Singleton_Coors_ORIG))

   if IterID > 0 :
      Singleton_Coors_SHUFFLED              = npzfile['arr_9'].tolist()
      All_Coors_PREV                        = npzfile['arr_10'].tolist()
      Excluded_Coors_PREV                   = npzfile['arr_11'].tolist()
      Cleared_Oligomer_Coors_ByLength_PREV  = npzfile['arr_12'].tolist()
      Blocked_Oligomer_Coors_ByLength_PREV  = npzfile['arr_13'].tolist()
      Possible_Oligomer_Coors_ByLength_PREV = npzfile['arr_14'].tolist()
      Locked_Oligomer_Coors_ByLength_PREV   = npzfile['arr_15'].tolist()
      Singleton_Coors_PREV                  = npzfile['arr_16'].tolist()
      Cleared_Singleton_Coors_PREV          = npzfile['arr_17'].tolist()
      UnCleared_Singleton_Coors_PREV        = npzfile['arr_18'].tolist()
      Possible_Oligomer_Coors_ByLength_ALT   = npzfile['arr_19'].tolist()
      Stripped_Oligomer_Coors_ByLength_PREV = npzfile['arr_20'].tolist()

      print("All beads (previously):", len(All_Coors_PREV))
      print("All singletons (previously):", len(Cleared_Singleton_Coors_PREV))
   print("\n")

   #####################################################



   if IterID > 0 :   
      ######### PRUNE OLIGOMERS ##########

      print("\nGETTING DIFFERENCES, PRUNING CLEARED OLIGOMERS, AND MAKING A NEW LIST OF ALTERED OLIGOMERS\n")

      OligoLen = list(Cleared_Oligomer_Coors_ByLength_ORIG.keys())
      OligoLen.sort()
      OligoLen.reverse()   # IMPORTANT! Because we want to do things (i.e., the cascading stripping) from long to short oligomers
   
      OligoCountDiffs = {}
      for oligolen in OligoLen :
         OligoCountDiffs[oligolen] = []

      print("The oligomer difference to the previous iteration is:")   
      for oligolen in OligoLen :
         OligoCount_Orig = len(Cleared_Oligomer_Coors_ByLength_ORIG[oligolen])
         OligoCount_Prev = len(Cleared_Oligomer_Coors_ByLength_PREV[oligolen])
         OligoCountDiffs[oligolen] = (OligoCount_Prev - OligoCount_Orig)
         print(str(oligolen)+"-mers:", OligoCountDiffs[oligolen])
      print("\n")
   
      print("The original 'Possible' list lengths:")
      for oligolen in Cleared_Oligomer_Coors_ByLength_ORIG.keys() :
         print(str(oligolen)+"-mers: ",  len(Possible_Oligomer_Coors_ByLength_ORIG[oligolen]))
      print("\n")
   

      StripNums = {}  		# Number of n-mers to get stripped (or split in the case of 2-mers)
      for oligolen in OligoLen :
         StripNums[oligolen] = OligoCountDiffs[oligolen] 
         if IterID > 1 :	# Current difference PLUS added to number of previously stripped, thus we get a cumulative sum of differences 
            if oligolen == min(OligoLen) :
               StripNums[oligolen] += len(Stripped_Oligomer_Coors_ByLength_PREV[oligolen])/2.0 # IMPORTANT! For 2-mers, this list contains twice the number of pruned 2-mers because both beads are stripped and become singletons (i.e., the 2-mer is split)
            else :
               StripNums[oligolen] += len(Stripped_Oligomer_Coors_ByLength_PREV[oligolen])
         if StripNums[oligolen] < 0 :
            StripNums[oligolen] = 0   # May happen when a smaller cumulative count of n-mers has been produced upon shuffling than there are in the original/experimental micrograph!
         print(str(oligolen)+"-mers:  ", StripNums[oligolen], "to prune!")
      print("\n")


      ListLens = {} # Number of n-mers in the original list PLUS as many as are going to be stripped from the (n+1)-mer (unless there is no (n+1)-mer)
      for oligolen in OligoLen :
         if oligolen < max(OligoLen) :
            ListLens[oligolen] = len(Possible_Oligomer_Coors_ByLength_ORIG[oligolen]) + StripNums[oligolen+1]
            if ListLens[oligolen] < StripNums[oligolen] :
               print("There are fewer "+str(oligolen)+"-mers to prune in the original micrograph than what we want to!\nSo we just prune all "+str(oligolen)+"-mers (which are", ListLens[oligolen], "because they include the number of "+str(oligolen+1)+"-mers that got pruned)!")
               StripNums[oligolen] = ListLens[oligolen]
         else :
            ListLens[oligolen] = len(Possible_Oligomer_Coors_ByLength_ORIG[oligolen])
            if ListLens[oligolen] < StripNums[oligolen] :
               print("There are fewer "+str(oligolen)+"-mers to prune in the original micrograph than what we want to!\nSo we just prune all "+str(oligolen)+"-mers (which are", ListLens[oligolen], ")!")
               StripNums[oligolen] = ListLens[oligolen]

      for oligolen in OligoLen :
         print(str(oligolen)+"-mers:  ", ListLens[oligolen], "are in the list (incl. the stripped "+str(oligolen+1)+"-mers)")
      print("\n")      


      StripIDs = {}	# Actual list IDs that get stripped off the n-mer, randomly selected from 0 to the length of the corresponding list
      print("These IDs get stripped:")
      for oligolen in OligoLen :
         StripIDs[oligolen] = []
         i=1
         while i <= StripNums[oligolen] :
            r = random.choice(range(ListLens[oligolen])) 
            if r in StripIDs[oligolen] :
               continue
            else :
               StripIDs[oligolen].append(r)
               i+=1
         currlist = StripIDs[oligolen]
         currlist.sort()
         print(str(oligolen)+"-mers:\n", currlist)
      print("\n")


      Stripped_Oligomer_Coors_ByLength_NEW = {}  # Going to be single beads from each oligolen that become new singletons (and that will be added to 'All_Coors_NEW' during reshuffling)
      for oligolen in OligoLen :
         Stripped_Oligomer_Coors_ByLength_NEW[oligolen] = []
      
      Possible_Oligomer_Coors_ByLength_ALT = {}   # Going to be the altered list of all possible oligos, in which specific larger oligos have been stripped to shorter ones; WE ARE OVERWRITING WHAT WAS LOADED, BUT THAT'S OKAY B/C WE DON'T NEED THE OLDER ONE
      for oligolen in OligoLen :
         Possible_Oligomer_Coors_ByLength_ALT[oligolen] = []
      
      Possible_Oligomer_Coors_ByLength_TMP = {}   # We need this for the upcoming loop: it starts as the 'ORIG' list but shorter oligos get enhanced in a cascading fashion as we decide if something is kept or stripped and added or not to the above 'ALT' list
      ### AND WE FILL THIS LIST UP IN SUCH A CUMBERSOME WAY B/C OTHERWISE OLIGOS IN THE ORIGINAL LIST ALSO GET ALTERED WHEN OPERATING ON THE 'TMP' LIST
      for oligolen in OligoLen :
         Possible_Oligomer_Coors_ByLength_TMP[oligolen] = []
         for oligoID in range(len(Possible_Oligomer_Coors_ByLength_ORIG[oligolen])) :
            Possible_Oligomer_Coors_ByLength_TMP[oligolen].append([])
            for bead in Possible_Oligomer_Coors_ByLength_ORIG[oligolen][oligoID] :
               Possible_Oligomer_Coors_ByLength_TMP[oligolen][oligoID].append(bead)
      
      for oligolen in OligoLen :  # Here it is important to start from the larger oligo and go to the shorter one (the 'cascade') b/c of the stripping that changes the number of shorter oligos
         if oligolen > min(OligoLen) :
            for OligoID in range(len(Possible_Oligomer_Coors_ByLength_TMP[oligolen])) :
               oligomer = Possible_Oligomer_Coors_ByLength_TMP[oligolen][OligoID]
               if OligoID in StripIDs[oligolen] :
                  stripped_bead = oligomer.pop()
                  new_oligomer = oligomer
                  Stripped_Oligomer_Coors_ByLength_NEW[oligolen].append(stripped_bead)
                  Possible_Oligomer_Coors_ByLength_TMP[oligolen-1].append(new_oligomer)
               else :
                  Possible_Oligomer_Coors_ByLength_ALT[oligolen].append(oligomer)   # The unstripped oligomers are saved in the final ('ALT') list; Note that this is an altered list of the 'ORIG' list of all possible oligomers because the 'TMP' list at this oligolen has been updated, when oligolen+1 was being evaluated, with pruned (or altered) oligomers upon stripping one bead off!
         else :
            for OligoID in range(len(Possible_Oligomer_Coors_ByLength_TMP[oligolen])) :
               oligomer = Possible_Oligomer_Coors_ByLength_TMP[oligolen][OligoID]
               if OligoID in StripIDs[oligolen] :
                  stripped_bead1 = oligomer.pop()
                  stripped_bead2 = oligomer.pop()
                  Stripped_Oligomer_Coors_ByLength_NEW[oligolen].append(stripped_bead1)
                  Stripped_Oligomer_Coors_ByLength_NEW[oligolen].append(stripped_bead2)
               else :
                  Possible_Oligomer_Coors_ByLength_ALT[oligolen].append(oligomer)
      
      print("\nOriginal and alternative possible list of oligos:")
      for oligolen in Possible_Oligomer_Coors_ByLength_ORIG.keys() :
         print(str(oligolen)+"-mers:", len(Possible_Oligomer_Coors_ByLength_ORIG[oligolen]), len(Possible_Oligomer_Coors_ByLength_ALT[oligolen]))
      print("\n")


   elif IterID == 0 :
      Stripped_Oligomer_Coors_ByLength_NEW = {}
      Possible_Oligomer_Coors_ByLength_ALT = Possible_Oligomer_Coors_ByLength_ORIG   
            
   ###########################################################################################




   ######## RE-SHUFFLE SINGLETONS ##########

   print("\nSHUFFLING SINGLETONS AND ADDING THEM TO A NEW LIST OF ALL OLIGOMERS\n")

   All_Coors_NEW = []  # We need this b/c of the singleton shuffling; thus, some coordinates must be updated, the total num must be the same
   for oligolen in Possible_Oligomer_Coors_ByLength_ALT.keys() :
      for oligomer in Possible_Oligomer_Coors_ByLength_ALT[oligolen] :
         for bead in oligomer :
            All_Coors_NEW.append(bead)
   for oligolen in Locked_Oligomer_Coors_ByLength_ORIG.keys() :
      for oligomer in Locked_Oligomer_Coors_ByLength_ORIG[oligolen] :
         for bead in oligomer :
            All_Coors_NEW.append(bead) 


   DimX = 900	# Graph dimensions in nm
   DimY = 900
   
   MinDist_All  = 2  # The minimum distance (in nm) of any two dog picks. Used here to not plot the random singletons directly "on top" of the oligomers
   
   ##### 
   def CheckClash (x, List, ClashDist) :
   
      x=np.array(x)
      for bead in List :
         bead=np.array(bead)
      #     print x, " vs. ", bead
         if norm(x-bead) < ClashDist :
      #        print("CLASH ", x, " WITH ", bead, "(ClashDist=", ClashDist, ")")
            return 1
   
      return 0
   #####
   
   Singleton_Coors_SHUFFLED = []  # WE ARE OVERWRITING WHAT WAS LOADED, BUT THAT'S OKAY B/C WE DON'T NEED THE OLDER ONE
   SingleNum = len(Singleton_Coors_ORIG) # IMPORTANT: We always add to and reshuffle the ORIGINAL singletons! 
   if IterID > 0 :
      for oligolen in Stripped_Oligomer_Coors_ByLength_NEW.keys() :
         SingleNum += len(Stripped_Oligomer_Coors_ByLength_NEW[oligolen])  # CAUTION: These are the exact numbers of stripped beads, the 'StripNum' list-of-lists should have half that many for 2-mers b/c it counted the number of 2-mers to be split!
   print("shuffling ", SingleNum, "singletons")

   i=1   
   while i <= SingleNum :
      if i%200 == 0 :  print("adding singleton", i, "/", SingleNum)
      CoorX = np.random.rand()*DimX
      CoorY = np.random.rand()*DimY
      if (CheckClash([CoorX, CoorY], All_Coors_NEW, MinDist_All) == 1) :
         continue
      else :
         Singleton_Coors_SHUFFLED.append([CoorX, CoorY])
         All_Coors_NEW.append([CoorX, CoorY])
         i+=1
   
   print(len(Singleton_Coors_SHUFFLED), "singletons got shuffled\n")
   
   ####################################################
   



   ######## RE-CLUSTERING OLIGOMERS ##########
   
   print("\n\n", "\tPERFORMING THE RE-CLUSTERING\n")
   
   CutOffLen = 6.5  
   
   clustering = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=CutOffLen, compute_full_tree=True).fit(All_Coors_NEW)
   print("There are", len(All_Coors_NEW), "data points (incl. noise) in", len(np.unique(clustering.labels_)), "clusters!")
   
   
   Clusters = {}
   for i in range(len(clustering.labels_)) :
      lab = clustering.labels_[i]
      if lab in Clusters.keys() :
         Clusters[lab].append(All_Coors_NEW[i])
      else :
         Clusters[lab] = [All_Coors_NEW[i]]
   
   ###########################################
   
   
   
   
   ######### RE-SCORING OLIGOMERS ############
   
   print("\n\n", "\tPERFORMING THE RE-SCORING\n")
   
   debug = 0
   PolyLen = range(2,10)
   
   MaxDist  = float(6.0)
   MaxAngle = float(75)
   score_threshold = float(0.10)
   
   #####
   def Scoring_Bonds (x) :
      mu = 5.0
      if x <= mu :
    #     n = 1/50.0
    #     n = 1/3.0
    #     p = (x/mu)**n
         p = 1
      else :
         sigma = 0.4
         #p = np.exp((-(x-mu)**2)/2*sigma**2)/np.sqrt(2*np.pi*sigma**2)
         p = np.exp((-(x-mu)**2)/(2*sigma**2))
      return p
   #####
   
   #####
   def Scoring_Angles (x) :
      mu = 15
      sigma = 15
      #p = np.exp((-(x-mu)**2)/2*sigma**2)/np.sqrt(2*np.pi*sigma**2)
      p = np.exp((-(x-mu)**2)/(2*sigma**2))
      return p
   #####
   
   DistMinScore  = Scoring_Bonds(MaxDist)
   AngleMinScore = Scoring_Angles(MaxAngle)
   
   AllNonOverlappingPolymerCoors    = {}
   AllNonOverlappingPolymerNormalScores = {}
   for polylen in PolyLen :
      AllNonOverlappingPolymerCoors[polylen]    = []
      AllNonOverlappingPolymerNormalScores[polylen] = []
   ##These are the retrieved polymer coordinates and (normalized) scores, saved as a function of their lengths
   
   
   for lab in sorted(Clusters.keys()) :
      if lab%100 == 0 :  print("Checking cluster label", lab+1, "/", len(Clusters.keys()), "[", len(Clusters[lab]), "points]")
      if len(Clusters[lab]) < 2 :
         continue
   
      AllClusterPolymerIDs          = []   
      AllClusterPolymerNormalScores = []
   
      PolymerIDs    = {} ##Polymers saved in terms of IDs with respect to their position in the current cluster list
      PolymerScores = {}
      for polylen in PolyLen :
         PolymerIDs[polylen] = []
         PolymerScores[polylen] = []
      
   
      DistMat = distance.cdist(Clusters[lab], Clusters[lab], 'euclidean')   
      DistScoreMat = np.zeros(DistMat.shape)
      for i in range(len(DistMat)) :
         for j in range(len(DistMat[i])) :
            if i != j :
               DistScoreMat[i][j] = Scoring_Bonds(DistMat[i][j])
   
      AngleMat = np.zeros((len(Clusters[lab]), len(Clusters[lab]), len(Clusters[lab])))
      for i in range(0, len(AngleMat)) :
         for j in range(0, len(AngleMat)) :
            for k in range(0, len(AngleMat)) :
               if (i == j) or (j == k) or (i == k):
                  AngleMat[i][j][k] = -1  ##We need to get rid of the zero set initially because that is actually a high-scoring angle
               else :
                  Bond1 = np.array(Clusters[lab][j]) - np.array(Clusters[lab][i])
                  Bond2 = np.array(Clusters[lab][k]) - np.array(Clusters[lab][j]) 
                  cosine = dot(Bond1,Bond2)/norm(Bond1)/norm(Bond2)
                  angle  = arccos(clip(cosine, -1, 1))
                  AngleMat[i][j][k] = angle/np.pi * 180
      AngleScoreMat = np.zeros(AngleMat.shape)
      for i in range(len(AngleMat)) :
         for j in range(len(AngleMat[i])) :
            for k in range(len(AngleMat[j])) :
               if AngleMat[i][j][k] >= 0  :  ##Zero score for undefined angles 
                  AngleScoreMat[i][j][k] = Scoring_Angles(AngleMat[i][j][k])
   
      [r, c] = np.triu_indices(len(DistScoreMat), 1)  ##Indices for the upper triangle (i.e., offset by 1 from the diagonal) of a square matrix of given length 
      SortedDistScores = sorted(DistScoreMat[r,c], reverse=True)  ##Sorted values of that given matrix section (off-diagonal, thus non-zero distances with itself, but note that the distance scores may well be zero) 
   
      if debug: print("Sorted distance scores:", SortedDistScores)
   
   
      polylen = 2
   
      scorenum = 0
      while scorenum < len(SortedDistScores) :  ##Using a while instead of for loop because there may be multiple pairs with the same score (so we increment the running parameter for every stored pair)
         if debug: print("Scorenum is", scorenum)
         distscore = SortedDistScores[int(scorenum)] ##Making the parameter an integer because it becomes a (whole-numbered!) float below
         if debug: print("Currdistscore is", distscore)
         if distscore > DistMinScore : ##Consider only scores that are larger than a pre-defined minimum score  
            DistScore_rowIDs = np.where(DistScoreMat == distscore)[0]  ##We sorted the distance scores only by the upper triangle of the distance matrix, but we are checking against the whole matrix ...
            DistScore_colIDs = np.where(DistScoreMat == distscore)[1]  ##... so, for each score, there will be both (i,j) and (j,i) pairs (and, of course, i!=j because we had disregarded the diagonal)
            for idnum in range(len(DistScore_rowIDs)) : ##This is needed if the same score occurs more than once, but also, in particular, each dimer is saved in both "directions", i.e., both (i,j) and (j,i) are continued here, as either could have a better trimer score
               PolymerIDs[polylen].append([DistScore_rowIDs[idnum], DistScore_colIDs[idnum]])
               PolymerScores[polylen].append(np.log(distscore)) ##Saving the dimer score twice for the above reason.
               if debug: print("CurrPolyID", [DistScore_rowIDs[idnum], DistScore_colIDs[idnum]])
               if debug: print("CurrPolyScore", np.log(distscore))
               scorenum += 0.5  ##Only adding one half because there will be two "idnum" elements, i.e., (i,j) and (j,i), for each "distscore"
         else :
            scorenum += 1
   
      if debug: print("Dimers:", PolymerIDs[2], PolymerScores[2])
   
   #   print("finished polylen 2")
   
      for polylen in range(3,10) :  ##Similar to above, just for all 3-mers and larger
   
   #      print("start polylen", polylen, "\n")
   
         CurrPolyIDs    = []
         CurrPolyScores = []
   #      print("These many oligomers to grow now:  ", len(PolymerIDs[polylen-1]), "\n")
         #for polynum in range(len(PolymerIDs[polylen-1])) :
         PrevPolyNum = 2000  ##TAKING ONLY THIS MANY POLYMERS (HALF THAT MANY IN BOTH DIRECTIONS) TO GROW BY ONE MORE BEAD; THEY ARE SORTED BY THE PREVIOUSLY BEST DISTANCES TO GROW AND NOT BY THE SCORE UP TO THAT POINT!!!
         if len(PolymerIDs[polylen-1]) < PrevPolyNum :
            PrevPolyNum = len(PolymerIDs[polylen-1]) 
         for polynum in range(PrevPolyNum) : 
            polymer     = PolymerIDs[polylen-1][polynum]
            polyscore   = PolymerScores[polylen-1][polynum]
            bead_pre    = polymer[-1]
            bead_prepre = polymer[-2]
            if debug: print("Checking Polymer", polymer, "with polyscore", polyscore)
   
            SortedDistScores = sorted(DistScoreMat[bead_pre], reverse=True)
            if debug: print("Its sorted distance scores to new beads are", SortedDistScores)
            scorenum = 0
            while scorenum < len(SortedDistScores) :  ##Similar approach to above
               distscore = SortedDistScores[scorenum]
               if debug: print("ScoreNum is", scorenum, "and the corresponding distance score is", distscore)
               if distscore > DistMinScore :
                  DistScore_colIDs = np.where(DistScoreMat[bead_pre] == distscore)[0]  
                  if debug: print("It occurs with these potential new beads", DistScore_colIDs) 
                  for idnum in range(len(DistScore_colIDs)) :
                     if DistScore_colIDs[idnum] in polymer:  ##It might be one of the previous beads, in particular 'bead_prepre'
                        if debug: print(DistScore_colIDs[idnum], "already in", polymer)
                        scorenum += 1
                        continue
                     else :
                        anglescore = AngleScoreMat[bead_prepre][bead_pre][DistScore_colIDs[idnum]]
                        if debug: print("Addding bead", DistScore_colIDs[idnum], "would give the new distance score", distscore, "and angle score", anglescore)
                        if anglescore > AngleMinScore :
                           newpolymer = polymer + [DistScore_colIDs[idnum]]
                           CurrPolyIDs.append(newpolymer)
                           CurrPolyScores.append(polyscore + np.log(distscore) + np.log(anglescore))
                           if debug: print("new polymer is", newpolymer, "with log-score", CurrPolyScores[-1], "\n")
                        scorenum += 1
               else :
                  scorenum += 1  ##A distance score of 0.0 will occur this time because of distances to itself (diagonal values were excluded before)  
   
         ##Sorting by increasing score
         CurrPolyIDs_SortedByScore  =  [ CurrPolyIDs for CurrPolyScores,CurrPolyIDs  in sorted(zip(CurrPolyScores,CurrPolyIDs)) ]
         CurrPolyScores.sort()
         CurrPolyScores_SortedByScore = CurrPolyScores  ##Just to have the names consistent
         ##Sorting by decreasing score
         CurrPolyIDs_SortedByScore.reverse()
         CurrPolyScores_SortedByScore.reverse()
   
         for i in range(len(CurrPolyIDs_SortedByScore)) :
            PolymerIDs[polylen].append(CurrPolyIDs_SortedByScore[i])
            PolymerScores[polylen].append(CurrPolyScores_SortedByScore[i])   
   
         if debug: print("These are found for the polylen:\nIDs:",  PolymerIDs[polylen], "\nScores:", PolymerScores[polylen], "\n")
   
      for polylen in PolyLen :
         for polynum in range(len(PolymerIDs[polylen])) :
            polymer     = PolymerIDs[polylen][polynum]
            bondnums  = len(polymer) - 1
            anglenums = len(polymer) - 2
            normalscore = PolymerScores[polylen][polynum]/float(bondnums+anglenums)  ##At this point, we normalize the polymer scores for cross-polylen comparison within a cluster (e.g. when removing some because of overlapping) 
            AllClusterPolymerIDs.append(polymer)         
            AllClusterPolymerNormalScores.append(normalscore)
   
      if debug: print("These are found in the cluster:\nIDs:", AllClusterPolymerIDs, "\nScores:", AllClusterPolymerNormalScores, "\n")
   
      NormalScoreCutOff = np.log(score_threshold)  ##probability average per bond and angle used as threshold here
   
      ##Applying threshold
      AllClusterPolymerIDs_Cut = []
      AllClusterPolymerNormalScores_Cut = []
      for i in range(len(AllClusterPolymerIDs)) :
         if AllClusterPolymerNormalScores[i] > NormalScoreCutOff :
            AllClusterPolymerIDs_Cut.append(AllClusterPolymerIDs[i])
            AllClusterPolymerNormalScores_Cut.append(AllClusterPolymerNormalScores[i])
   
   
      if len(AllClusterPolymerIDs_Cut) > 0 :  ##if any left, then sort, remove overlaps, and put the coordinates in the global array
   
         ##Sorting by increasing score
         AllClusterPolymerIDs_CutAndSortedByScore  =  [ AllClusterPolymerIDs_Cut for AllClusterPolymerNormalScores_Cut,AllClusterPolymerIDs_Cut  in sorted(zip(AllClusterPolymerNormalScores_Cut,AllClusterPolymerIDs_Cut)) ]  ##This sorts by the score (the 1st element in the zipped list); if two scores were identical, it would sort by the ID-values (values of the 2nd element in the zipped list)
         AllClusterPolymerNormalScores_Cut.sort()
         AllClusterPolymerNormalScores_CutAndSortedByScore = AllClusterPolymerNormalScores_Cut  ##Just to have the names consistent
   
         ##Sorting by increasing length (if more than one has the same length, they are still sorted by increasing score)
         AllClusterPolymerLengths_CutAndSortedByScore = []
         for i in AllClusterPolymerIDs_CutAndSortedByScore :
            AllClusterPolymerLengths_CutAndSortedByScore.append(len(i))
         AllClusterPolymerNormalScores_CutAndSortedByLenAndScore  =  [ AllClusterPolymerNormalScores_CutAndSortedByScore for AllClusterPolymerLengths_CutAndSortedByScore,AllClusterPolymerNormalScores_CutAndSortedByScore  in  sorted(zip(AllClusterPolymerLengths_CutAndSortedByScore,AllClusterPolymerNormalScores_CutAndSortedByScore)) ]  ##This sorts by the length (the 1st element in the zipped list); when the lengths are identical, then it sorts by the score value (the 2nd element in the zipped list)
         AllClusterPolymerIDs_CutAndSortedByScore.sort(key=len)  ##Sorting by length
         AllClusterPolymerIDs_CutAndSortedByLenAndScore = AllClusterPolymerIDs_CutAndSortedByScore  ##Just to have the names consistent
   
   
         AllClusterPolymerIDs_CutAndSortedByLenAndScore.reverse()  ##Sort decrementally, NOT incrementally
         AllClusterPolymerNormalScores_CutAndSortedByLenAndScore.reverse()  ##Dito
   
   
         NonOverlapPolymerIDs           = []  
         NonOverlapPolymerNormalScores  = []
         NonOverlapPolymerIDs.append(AllClusterPolymerIDs_CutAndSortedByLenAndScore[0])
         NonOverlapPolymerNormalScores.append(AllClusterPolymerNormalScores_CutAndSortedByLenAndScore[0])
   
         for i in range(len(AllClusterPolymerIDs_CutAndSortedByLenAndScore)) :   
            polymer       = AllClusterPolymerIDs_CutAndSortedByLenAndScore[i]
            normalscore   = AllClusterPolymerNormalScores_CutAndSortedByLenAndScore[i]         
            overlap       = 0
            for saved_polymer in NonOverlapPolymerIDs :
               if any(bead in saved_polymer for bead in polymer) :
                  overlap += 1
            if not overlap :
               NonOverlapPolymerIDs.append(polymer)
               NonOverlapPolymerNormalScores.append(normalscore)   
   
         if debug: print("These are kept, sorted, and free of overlaps\nIDs: ", NonOverlapPolymerIDs, "\nScores: ", NonOverlapPolymerNormalScores, "\n\n") 
   
         for polynum in range(len(NonOverlapPolymerIDs)) :
            polymer     = NonOverlapPolymerIDs[polynum]
            polylen     = len(polymer)
            normalscore = NonOverlapPolymerNormalScores[polynum] 
            realpolymer = []
            for beadid in polymer :
               realpolymer.append(Clusters[lab][beadid]) 
            AllNonOverlappingPolymerCoors[polylen].append(realpolymer)
            AllNonOverlappingPolymerNormalScores[polylen].append(normalscore)
   
   print("\n\n")
   for polylen in PolyLen :
     print(polylen, "\t", len(AllNonOverlappingPolymerCoors[polylen]), "\t", len(AllNonOverlappingPolymerNormalScores[polylen]))
   
   
   ###############################################
   
   
   
   
   ######## RE-ASSIGN PARTICLES ##########
   
   print("\n\n", "\tPERFORMING THE RE-ASSIGNING\n")
   # WE NEED TO EVALUATE THE NEW SINGLETON AND OLIGOMER CATEGORIES BECAUSE THE PRUNING OF OLIGOMERS DEPENDS ON THE DIFFERENCE IN 'CLEARED', I.E., COUNTED, OLIGOMERS.  HOWEVER, IT IS IMPORTANT TO KEEP IN MIND THAT RE-SHUFFLING IS ALWAYS PERFORMED ON THE SET OF ORIGINAL SINGLETONS, WHICH ARE POSSIBLY ENHANCED AFTER PRUNING.  
   
   MinDist = 9	
   
   [Singleton_Coors_NEW, 
   Cleared_Singleton_Coors_NEW, 
   UnCleared_Singleton_Coors_NEW, 
   Cleared_Oligomer_Coors_ByLength_NEW, 
   UnCleared_Oligomer_Coors_ByLength_NEW, 
   Blocked_Oligomer_Coors_ByLength_NEW, 
   Possible_Oligomer_Coors_ByLength_NEW, 
   Locked_Oligomer_Coors_ByLength_NEW, 
   Excluded_Coors_NEW]  =  assign_oligomers(All_Coors_NEW, AllNonOverlappingPolymerCoors, MinDist)
   
   print("After the singleton shuffling, there are", len(Singleton_Coors_NEW), "new singletons assigned, of which", len(Cleared_Singleton_Coors_NEW), "are cleared and", len(UnCleared_Singleton_Coors_NEW), "are not")
   print("And the following number of new cleared, uncleared, blocked, possible, and locked oligomers:")
   for oligolen in Cleared_Oligomer_Coors_ByLength_NEW.keys() :
      print(oligolen, len(Cleared_Oligomer_Coors_ByLength_NEW[oligolen]), len(UnCleared_Oligomer_Coors_ByLength_NEW[oligolen]), len(Blocked_Oligomer_Coors_ByLength_NEW[oligolen]), len(Possible_Oligomer_Coors_ByLength_NEW[oligolen]), len(Locked_Oligomer_Coors_ByLength_NEW[oligolen]))
   print("Also, there are", len(Excluded_Coors_NEW), "new excluded beads (i.e., uncleared oligomers AND singletons)")
   
   ##########################################
   
   
   
   
   ####### RE-PLOTTING AND SAVING #########
  
   if (IterID % PlotFreq == 0) or (IterID == 1) : 
      print("\nPLOTTING DATA AT ITERATION", IterID)
      # This is being plotted such that the original oligomers (black dots) are first given and then topped by possible stripped beads (white dots) before showing that original oligomer by line (yellow line) and adding the reshuffled singletons (lightgreen dots) and new cleared(!) oligomers (purple line).  The excluded beads, i.e., uncleared oligos and singletons (brown dots), are as located before and "irrelevant" for the study.  
      
      fig = plt.figure(figsize=(7, 7), dpi=300)
      ax  = fig.add_subplot(1, 1, 1)
      
      gray       = (166/255., 166/255., 166/255.)
      yellow     = (240/255., 230/255., 80/255.)
      lightgreen = (115/255., 251/255., 121/255.)
      brown      = (200/255., 100/255., 0/255.)
      purple     = (140/255., 40/255., 200/255.)
      
      ax.set_facecolor(gray)
      
      #for bead in All_Coors_NEW :  ## Just plotting all dog picks to see if they all get overshadowed by all the other ones
      #   plt.plot(bead[0], bead[1],  'o', color='r', markeredgecolor='r', markersize=0.5)
      
      for oligolen in Possible_Oligomer_Coors_ByLength_ORIG.keys() :
         for oligomer in Possible_Oligomer_Coors_ByLength_ORIG[oligolen] :
            for bead in oligomer :
               plt.plot(bead[0], bead[1],  'o', color='k', markeredgecolor='k', markersize=0.5)
      
      if IterID > 0 :
         for oligolen in Stripped_Oligomer_Coors_ByLength_NEW.keys() :
            for bead in Stripped_Oligomer_Coors_ByLength_NEW[oligolen] :
               plt.plot(bead[0], bead[1],  'o', color='w', markeredgecolor='w', markersize=0.5)

      for oligolen in Cleared_Oligomer_Coors_ByLength_ORIG.keys() :
         for oligomer in Cleared_Oligomer_Coors_ByLength_ORIG[oligolen] :
            oligomer_xcoor = []
            oligomer_ycoor = []
            for bead in oligomer :
               oligomer_xcoor.append(bead[0])
               oligomer_ycoor.append(bead[1])
            plt.plot(oligomer_xcoor, oligomer_ycoor, '-', color=yellow, linewidth=0.4)

      for bead in Singleton_Coors_SHUFFLED :
         plt.plot(bead[0], bead[1],  'o', color=lightgreen, markeredgecolor=lightgreen, markersize=0.5)
      
      for oligolen in Cleared_Oligomer_Coors_ByLength_NEW.keys() :
         for oligomer in Cleared_Oligomer_Coors_ByLength_NEW[oligolen] :
            oligomer_xcoor = []
            oligomer_ycoor = []
            for bead in oligomer :
               oligomer_xcoor.append(bead[0])
               oligomer_ycoor.append(bead[1])
            plt.plot(oligomer_xcoor, oligomer_ycoor, '-', color=purple, linewidth=0.4)
      
      for oligolen in Cleared_Oligomer_Coors_ByLength_ORIG.keys() :
         for oligomer in Cleared_Oligomer_Coors_ByLength_ORIG[oligolen] :
            oligomer_xcoor = []
            oligomer_ycoor = []
            for bead in oligomer :
               oligomer_xcoor.append(bead[0])
               oligomer_ycoor.append(bead[1])
            plt.plot(oligomer_xcoor, oligomer_ycoor, ':', color=yellow, linewidth=0.4)  ### PLOTTING THESE LINES AGAIN AS DOTTED SO THAT THEY ARE SOMEWHAT VISIBLE WHEN PURPLE LINES ARE DRAWN
      
      for oligolen in Locked_Oligomer_Coors_ByLength_ORIG.keys() :
         for oligomer in Locked_Oligomer_Coors_ByLength_ORIG[oligolen] :
            for bead in oligomer :
               plt.plot(bead[0], bead[1],  'o', color=brown, markeredgecolor=brown, markersize=0.5) 
     
 
      plt.xticks(range(0, 901, 50), [val if (val%100==0) else ' ' for val in range(0, 901, 50)])
      plt.yticks(range(0, 901, 50), [val if (val%100==0) else ' ' for val in range(0, 901, 50)])
      plt.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=True, bottom=True, top=True, left=True, right=True)
      plt.xlabel('nm')
      plt.ylabel('nm')
      plt.savefig("ShuffledAndReScored_"+DataName+"_Iter"+'{0:02}'.format(IterID)+".pdf")
      
  

   print("\nSAVING DATA...")
 
   FileOut = open("ShuffledAndReScoredPop_"+DataName+"_Iter"+'{0:02}'.format(IterID)+".dat", 'w')
   for oligolen in Cleared_Oligomer_Coors_ByLength_NEW.keys() :
      print(oligolen, "\t", len(Cleared_Oligomer_Coors_ByLength_NEW[oligolen]), file=FileOut)
   FileOut.close()
   
   
   np.savez("ShuffledAndReScored_"+DataName+"_Iter"+'{0:02}'.format(IterID)+".npz",  
   np.array(All_Coors_ORIG),
   np.array(Excluded_Coors_ORIG),
   np.array(Cleared_Oligomer_Coors_ByLength_ORIG),
   np.array(Blocked_Oligomer_Coors_ByLength_ORIG),
   np.array(Possible_Oligomer_Coors_ByLength_ORIG),
   np.array(Locked_Oligomer_Coors_ByLength_ORIG),
   np.array(Singleton_Coors_ORIG),
   np.array(Cleared_Singleton_Coors_ORIG),
   np.array(UnCleared_Singleton_Coors_ORIG),
   np.array(Singleton_Coors_SHUFFLED),
   np.array(All_Coors_NEW),
   np.array(Excluded_Coors_NEW),
   np.array(Cleared_Oligomer_Coors_ByLength_NEW),
   np.array(Blocked_Oligomer_Coors_ByLength_NEW),
   np.array(Possible_Oligomer_Coors_ByLength_NEW),
   np.array(Locked_Oligomer_Coors_ByLength_NEW),
   np.array(Singleton_Coors_NEW),
   np.array(Cleared_Singleton_Coors_NEW),
   np.array(UnCleared_Singleton_Coors_NEW),
   np.array(Possible_Oligomer_Coors_ByLength_ALT),
   np.array(Stripped_Oligomer_Coors_ByLength_NEW),
   )
   
   print(len(All_Coors_ORIG), len(All_Coors_NEW), len(Singleton_Coors_ORIG), len(Singleton_Coors_SHUFFLED), len(Singleton_Coors_NEW), "\n")
   
   print("\n\nFINISHED ITERATION", IterID, "/", IterEnd, "\n\n")

   #########################################################################
