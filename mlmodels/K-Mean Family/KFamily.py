# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:02:18 2016

@author: Andres Mendez Vazquez

@ Licencense BSD 3

This class implements several methods from the k-mean family:

* K-Means
* K-Medians
* K-Centers
* Fast K-Medoids
* Fuzzy C-Means

"""
import numpy as np

class KFamily:
    
    def __init__(self,k,X):
        """
        The constructor recieves two parameters
            k - the number of clusters
            X - The self.X set where the clusters are going to be found where the
                the self.X is a matrix d x N where
                    d = dimesnion of the self.X set
                    N = Number of samples
        """
        self.k = k
        self.X = X
    
    def CanopyCentroids(self):
        """
        This method uses the idea of building an iniital canopy by 
        """
        # This still under implementation 
        # For now, we are using a random init
        x1,x2 = self.X.shape
        return np.matrix(np.random.random_sample((x1,self.k)))
        
    def kMetric(self, x, y, dselect):
        """
        dselect allows to select between different metrics:
              1.- Euclidean Metric
              2.- Manhattan Metric
              3.- Chebyshev Metric
        """      
        # Given that we do not have cases in python we use a dictonary and 
        # Lambda functions form functional programming to obtain the same 
        # result
        result = {
            # Euclidean Metric 
            1: lambda x, y: np.sqrt(np.sum(np.power(x-y,2),axis=0)),
            # Manhattan Metric
            2: lambda x, y: np.sum(np.absolute(x-y)),
            # Chebyshev Metric
            3: lambda x, y: np.amax(np.absolute(x-y))
        }[dselect](x,y)
        
        return result
    
    def set_k(self,value):
        """
        This allows to reset the k values
        """
        self.k = value
        
    
    def k_means(self, error, dselect):
        """
        This method implements the k-mean algorithm using diferent metrics
        Input:
            1.- self-    self.X stored in the object 
            2.- error-   Threshold for the stopping criteria
            3.- dselect- Used to select the Metric
        Output:
            1.- centroids- The Final Centroids after running the algorithm
            2.- Labels-    The Final Cluster Labels for each point
        """
        # Find Shapes of the Data        
        x1,x2 = self.X.shape
        
        # Array to select the clusters for the Data
        Labels = np.zeros(x2)

        # Initial Centroids 
        centroids = self.CanopyCentroids()      
        
        # Shape of the Centroids
        y1,y2 = centroids.shape
       
        
        # Generate Old Random Centroids
        oldcentroids = np.matrix(np.random.random_sample((y1,y2)))

        # Array for the Metric to centroids for all points        
        dist = np.zeros(y2)
        # Loops to find the centroids
        while ( np.sqrt(np.sum(np.power(oldcentroids-centroids,2)))>error):
            # Loop to select the cluster for each Data points
            for i in range(0,x2):
                point = self.X[:,i]
                #loop for the centroids to assign the point to one of them
                for j in range(0, y2):
                    clustercenter = centroids[:,j]
                    dist[j] = self.kMetric(point,clustercenter,dselect)
                Labels[i] = np.argmin(dist)
                print  Labels[i]
            # Save the olf centroids
            oldcentroids = centroids
        
            #Update centroids     
            for j in range(0, y2):
                indexc = [i for i,val in enumerate(Labels) if val==(j)]
                Xc = self.X[:,indexc]
                # If the indexc is empty
                if (len(indexc)>0):
                    centroids[:,j]= Xc.sum(axis=1)/len(indexc)
        # Return the results
        return centroids, Labels
        
    def k_centers(self, dselect):
        """
         This method implements the k-centers algorithm using diferent metrics
        Input:
            1.- self-    self.X stored in the object 
            2.- dselect- Used to select the distance
        Output:
            1.- H-       The Final Centroids after running the algorithm
            2.- Labels-  The Final Cluster Labels for each point 
        """
        
        # Get the dimension and number of samples
        x1,x2 = self.X.shape
        
        # Initial distance to each x
        dist =  np.zeros(x2)
        
        # Initial Cluster Centroids
        H      =  np.matrix(np.zeros((x1,self.k)))
        
        # Labels 
        Labels =  np.zeros(x2)
        # For the max distance from the previous centroid being taken in 
        # consideration
        D      =  np.zeros(x2-1)
        
        # Random selection of h1 the initial centroid
        h1 = np.random.choice(x2)
        H[:,0] = self.X[:,h1] 
        
        
        # Obtain the initial distance to the inital cluster
        # and label everybody belonging to it
        for i in range(0, x2):
            dist[i] = self.kMetric(self.X[:,i],H[:,0],dselect)
            Labels[i]=0
        
        # Finding the rest of the centroids
        for i in range(1,self.k):
            # Getting the maximum distance from the previous centroid
            D[i-1] = np.amax(dist)
            # Finding the point that is dist[h_i]=D
            H[:,i]=self.X[:, np.argmax(dist)]
            
            # We update distance and labels
            for j in range(0, x2):
                L = self.kMetric(self.X[:,j],H[:,i],dselect)
                if L<= dist[j]:
                    dist[j]=L
                    Labels[j]=i
        
        return H, Labels

    def fuzzyCmeans(self,error,m):
        """
          This method implements the fuzzy C means algorithm using only the euclidean metric
          because it is the only derivable
          Input:
            1.- self-    self.X stored in the object 
            2.- error-   Threshold for the stopping criteria
            3.- m -      the fuzzification power for the A membership
        Output:
            1.- A-          The Final Fuzzy Memberships
            2.- vclusters-  The Final Cluster Labels for each point 
        """
    
        # Get the shape of the data
        x1,x2 = self.X.shape
        
        # Generate A memberships
        A = np.matrix(np.random.random_sample((self.k,x2)))

        # Use 0 centroids at the beginning
        vcluster = np.matrix(np.zeros([x1, self.k]))     
        
        while True:
            # We need a copy to avoid 
            oldA = A.copy()
            # Update the clusters
            for i in range(0,self.k):
                
                # The new membership numerator and denominator
                D_A = np.sum(np.power(A[i,:],m)) # Denominator
                D_Ax = np.sum(np.multiply(np.power(A[i,:],m),self.X), axis = 1) # Numerator
                #The new memeberships 
                vcluster[:,i]=D_Ax/D_A
    
            # update A memberships
            for i in range(0,self.k):
                for j in range(0, x2):
                    # The only problem you need a deriavable metric as the Euclidean  
                    n_up = np.power(self.kMetric(self.X[:,j],vcluster[:,i], 1),2)
                    n_down = np.power(self.kMetric(vcluster,self.X[:,j],1),2)
                    
                    # The Updating fuzzy membership
                    n_sum = np.power(n_up/n_down, 1.0/(m-1))                
                    n_f = np.sum(n_sum)
                    A[i,j] =  1.0/n_f
        
            # Break from the loop by stopping Criteria
            if (np.amax(np.amax(np.abs(oldA-A),axis=1),axis=0)[0,0]<error):
               break
        
        return A, vcluster