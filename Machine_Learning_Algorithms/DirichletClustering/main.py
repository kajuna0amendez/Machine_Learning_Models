# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 08:47:57 2016

@author: Andres Mendez-Vazquez
"""

from DirichletClassificationKDTree import DirichletClassificationKDTree
import numpy as np
import matplotlib.pyplot as plt


def DataGeneration(cov, mean, number):
    # Give me a series of pts with N
    x,y = np.random.multivariate_normal(mean, cov , number).T
    return np.matrix(np.vstack((x, y)))


# Data Generation 
markers = ['.','o','v','<','>','^','s','p','h','x','d','+']
colors  = ['r','b','g','c','m','y','k','w','g','m','m','y']

# Data Size
data_size = 1000

# Decay Function
selec = 0

#Mean and cov for the classes
mean1= [0,0]
mean2= [0,0.5]
mean3= [0.5,0.5]
mean4= [0.5,0]
cov1=0.001*np.identity(2)
cov2=0.01*np.identity(2)
cov3=0.001*np.identity(2)
cov4=0.01*np.identity(2)

# alpha  y a_exp
alpha = 3.05
a_exp = -30.0
#a_exp = 1e-2
K = 10 # Total number of points
Error = 0.08


Class1 = DataGeneration(cov1, mean1, data_size) 
Class2 = DataGeneration(cov2, mean2, data_size )
Class3 = DataGeneration(cov3, mean3, data_size )
Class4 = DataGeneration(cov4, mean4, data_size )

Data=np.matrix(np.concatenate((Class1, Class2, Class3, Class4), axis = 1))

##################### The Running Part ################################

Cluster1 = DirichletClassificationKDTree()

Cluster_Indexes = Cluster1.Clustering_Decay_Distance(Data,a_exp, 
                                                     alpha, 
                                                     selec,
                                                     Error,K)


tcluster = list(set(Cluster_Indexes))

print(tcluster)

# Plot Everything
plt.figure(1)
plt.plot(Class1[0,:], Class1[1,:], 'bx')
plt.plot(Class2[0,:], Class2[1,:], 'ro')
plt.plot(Class3[0,:], Class3[1,:], 'g>')
plt.plot(Class4[0,:], Class4[1,:], 'r+')
plt.axis('equal')

plt.figure(2)
n=0
for i in tcluster:
    mask  = ( np.array(Cluster_Indexes) == i)
    cluster = np.matrix(Data[:,mask])
    m1 = colors[n%len(markers)]+markers[n%len(markers)]
    n+=1    
    plt.plot(cluster[0,:],cluster[1,:], m1)
    plt.axis('equal')

plt.show()    
