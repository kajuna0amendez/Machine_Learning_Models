# -*- coding: utf-8 -*-
"""
Here, we implement the basic of Dirichlet Clustering.

Thus, we can have an understanding of the basic algorithms
Author: Andres 

"""

import numpy as np
import sys
from scipy import spatial

class DirichletClassificationKDTree:
    """
    """
    def __init__(self):
        """
        Default Constructor.
        """
        self.p = []
        self.r = []
        self.Clustering_Index = []
        self.CHull_List = []
        self.X = []        
        
    def Clustering_Decay_Distance(self, X, a, alpha, selec, Error, K):
        """        self.Grouping_Clusters(Error)
        Here there is an implementation of the Chinese.
        
        INPUT:
            1.- X - The samples to cluster
            2.- alpha  - It is the prior controlling how much 
                         you weigh previously selected groups 
                         when selecting a new group assignment.
                    
                         Smaller alpha - the previous groups
                                         weight more
                         Larger alpha  - The previous groups 
                                         weight less
             3.- selec - The Possible Decay Function
                                        
        OUTPUT
           the clustering indeces for the elements in the X 
           data set
                 
        """
        d, N = X.shape 

        # In place make set
        self.MakeSet(N)
                
        # Here the K-D-Tree
        tuple_p = list()
        for i in range(d):
            tuple_p.append(list(X[i,:].tolist())[0])
        zipped = zip(*tuple_p)
        # Building the KDTree for Reducing Complexity
        tree = spatial.KDTree(zipped) 
        
        for i in range(N):
            # Get the i point
            point = X[:,i].T.tolist()
            # Query the nearest K points
            p_query = tree.query(point,K)[1]
            LData   = X[:,p_query[0]]
            pindex  = [i]+p_query[0].tolist()
            # Find the alpha distance 
            fD = self.Distance_Data( X[:,i],  LData, a,  alpha, selec)             
            Ni = np.sum(fD)
            Prob = fD/Ni           
            self.Union(i, self.Roulette_Wheel_Selection(pindex,Prob))

        # Group posteriro clustering for better clustering
        self.Grouping_Clusters(X, Error)

        for i in range(N):
            self.Find(i)
        
        
        tcluster = list(set(self.p))

        N = len(tcluster)

        for i in range(N):
            mask = [x == tcluster[i] for x in self.p ]            
            for j in range(len(self.p)):
                if mask[j]:
                    self.p[j] = i
        
        self.Clustering_Index = list(set(self.p))
        self.X = X        
        return self.p
    
    #################################################################
    ###### MAKE UNION FIND FOR CLUSTERING INDEXING
    #################################################################    
    
    def MakeSet(self,N):
        self.p = range(N)
        self.r = [0]*N
        
    def Union(self, x, y):
        xRoot = self.Find(x)
        yRoot = self.Find(y)
        
        if self.r[xRoot] > self.r[yRoot]:
            self.p[yRoot] = xRoot
        else: 
            self.p[xRoot] = yRoot
            if (self.r[xRoot] == self.r[yRoot]):
                self.r[yRoot] = self.r[yRoot]+1

    def Find(self,x):
        if self.p[x] == x:
            return x
        else:
            self.p[x] = self.Find(self.p[x])
            return self.p[x]

    #################################################################
    ###### DISTANCE AND DECAY FUNCTIONS
    #################################################################     
    
    def Distance_Data(self, point, X, a,  alpha, selec):
        """
        It genererates the Distance Function.
        """        
        Distance = np.sqrt(np.sum(np.power(X - point,2), axis=0))
        
        D = self.Decay_Function(Distance,a,selec)
        
        # Remove the 1 at diagonal and add the alpah
        D = np.concatenate([np.matrix(alpha),D], axis = 1)

        return D
    
 
 
    def Decay_Function(self, val, a, selec):
        if (selec == 0):
            return np.exp(-a*val)
        elif (selec==1):
            return np.exp(-val+a)/(1+np.exp(-val+a))
        else:
            sys.exit("not correct selection")

    #################################################################
    ###### MAKE UNION FIND FOR CLUSTERING INDEXING
    #################################################################    
       
    def Roulette_Wheel_Selection(self, pindex, prob):
        """
        Take the Probabilities p for finding a new table or assign to a old table.
        
        Note: Remember Sum p = 1
        """        
        # Generate a Random Sample from U(0,1)
        a = np.random.random_sample(1)[0]   
        
        # Use an accumulator for the probabilites
        accumulator = 0.0

        # Accumulate the values 
        # The original algorithm uses a binary search 
        # but by the accumulation you can discover the correct 
        # index
        N, M = prob.shape
        for i in range(M):
            accumulator+=prob[0,i] 
            if (a < accumulator):
                return pindex[i]

    #################################################################
    ###### Posterior Group Clustering Algorithm
    #################################################################    
  
    def Grouping_Clusters(self, X, Error):
         
        
        setsheads = list(set(self.p))
        
        for i in setsheads:
            Point = X[:,i]
            tlist = [j for j in setsheads if j!=i ]
            D = np.sqrt(np.sum(np.power(X[:,tlist] - Point,2), axis=0)) 
            mask = ( D < Error)

            for j in range(len(tlist)):
                if mask[0,j]:
                    self.Union(tlist[j],i)

    
    def ConvexClassification(self):
        """
        This function generate the list of the Convex Hulls for the 
        data set X
        """
        Points = np.array(self.X.T)        
        
        
        for i in self.Clustering_Index:
            # Generate Mask
            mask  = [el==i for el in self.p]
            lclst = [el for el in range(len(self.p)) if mask[el]]
            Chull_i = Points[lclst]
            if len(lclst)>2:
                Hull  =  spatial.ConvexHull(Chull_i)
                self.CHull_List.append([lclst,Hull])
            else:
                self.CHull_List.append([lclst,[]])
        return self.CHull_List
        
    def Delanuey_Identification(self):
        """
        Using Delanuey to find if a new x is in the 
        """
        return 1
            
        
        