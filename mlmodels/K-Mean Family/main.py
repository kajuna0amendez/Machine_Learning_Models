# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 00:17:28 2016

@author: Andres Mendez Vazquez

@ Licencense BSD 3
"""

#import matplotlib.pyplot as plt
import numpy as np
from KFamily import KFamily 


def DataGeneration(cov, mean, number):
    # Give me a series of pts with N
    x,y = np.random.multivariate_normal(mean, cov , number).T
    return x,y


#Initial Values for the clusters and centroids
symbols=['bx','ro','kH','bo']
kclusters = 2
dim = 2
clusters_size = 300 

#error
error = 0.0000001

#Mean and cov for the clusters
mean1= [0,4]
mean2= [0,-4]
mean3 = [-4, 0]
cov1=0.05*np.identity(2)
cov2=1*np.identity(2)

# Class 1 
x1,y1 = DataGeneration(cov1, mean1, clusters_size)

# Class2
x2,y2 = DataGeneration(cov1, mean2, clusters_size)

# Putting the Data Together
Data=np.matrix([np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2), axis=0)])


# Build and object for clustering 
C1 = KFamily(kclusters,Data) 

# Testing the Basic Functions
centroids1, Labels1 = C1.k_means(error,1)
centroids2, Labels2 = C1.k_centers(1)
m_memb = 2.0
centroids3, Labels3 = C1.fuzzyCmeans(error,m_memb)