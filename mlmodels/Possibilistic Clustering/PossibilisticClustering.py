

"""
Possibilistic Clustering code
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import matplotlib.animation as animation


def DataGeneration(cov, mean, number):
    # Give me a series of pts with N
    x,y = np.random.multivariate_normal(mean, cov , number).T
    return x,y

def fuzzyCmeans(Data,A,nclusters,error,m):

    # Get the shape of the data
    x1,x2 = Data.shape

    vcluster = np.matrix(np.zeros([x1, nclusters]))     
    
    while True:
        oldA = A.copy()
        # Update the clusters
        for i in range(0,nclusters):
            D_A = np.sum(np.power(A[i,:],m))
            
            D_Ax = np.sum(np.multiply(np.power(A[i,:],m),Data), axis = 1)
           
          
            vcluster[:,i]=D_Ax/D_A

        # update A
        for i in range(0,nclusters):
            for j in range(0, x2):
                n_up = np.power(np.linalg.norm(Data[:,j]-vcluster[:,i], axis = 0),2)
                n_down = np.power(np.linalg.norm(vcluster-Data[:,j], axis=0),2)
                
                #print  vcluster-Data[:,j]               
                
                n_sum = np.power(n_up/n_down, 1.0/(m-1))                
                n_f = np.sum(n_sum)
                A[i,j] =  1.0/n_f
    
        # Break from the loop
        if (np.amax(np.amax(np.abs(oldA-A),axis=1),axis=0)[0,0]<error):
           break
    
    return A, vcluster

def PossibilisticClustering(Data,Tipicallity,nclusters,error,m,M):

    # Get the shape of the data
    x1,x2 = Data.shape

    vcluster = np.matrix(np.zeros([x1, nclusters]))     
    weights  = np.matrix(np.zeros([1, nclusters]))      
    
    while True:
        oldTipicallity = Tipicallity.copy()
        # Update the clusters
        for i in range(0,nclusters):
            D_T   = np.sum(np.power(Tipicallity[i,:],m)) 
            D_Tw = np.sum(np.multiply(np.power(Tipicallity[i,:],m),Data), axis = 1)
            vcluster[:,i]=D_Tw/D_T

        # Update Weights
        for i in range(0,nclusters):
            t2power = np.power(Tipicallity[i,:],m)
            ndiff2 = np.power(np.linalg.norm(Data-vcluster[:,i], axis = 0),2)
            Numw = np.sum(np.multiply(t2power,ndiff2))  
            Denw = np.sum(t2power)
            weights[0,i] = M*(Numw/Denw)

        # update Tipicallity
        for i in range(0,nclusters):
            for j in range(0, x2):
                
                n_up =  np.power(np.linalg.norm(Data[:,j]-vcluster[:,i], axis = 0),2)  
                
                n_p = np.power(n_up/weights[0,i], 1.0/(m-1))                

                Tipicallity[i,j] =  1.0/(1.0+n_p)
    
        # Break from the loop
        if (np.amax(np.amax(np.abs(oldTipicallity-Tipicallity),axis=1),axis=0)[0,0]<error):
           break
    
    return Tipicallity, vcluster

#Initial Values
# For the clusters and centroids
symbols=['bx','ro','kH','rx']
nclusters = 2
dim = 2
clusters_size = 300
clusters_soutliers = 25
m_memb = 1.8
M = 1.8

#error
error = 1.0e-10

#Mean and cov for the clusters
mean1= [4,0]
mean2= [10,0]
mean3 = [15,0]
cov=1.0*np.identity(dim)
# Class 1 
x1,y1 = DataGeneration(cov, mean1, clusters_size)

# Class2
x2,y2 = DataGeneration(cov, mean2, clusters_size)

#Outliers
x3,y3 = DataGeneration(cov, mean3, clusters_soutliers)

#Generate A
Tipicallity = np.matrix(np.random.random_sample((nclusters,2*clusters_size+clusters_soutliers)))

Data1=np.matrix([np.concatenate((x1, x2, x3), axis=0), np.concatenate((y1, y2, y3), axis=0)])
Data2=np.matrix([np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2), axis=0)])

Tipicallity, centroids = fuzzyCmeans(Data1,Tipicallity,nclusters,error,m_memb)

newTipicallity, centroids = PossibilisticClustering(Data1,Tipicallity,nclusters,error,m_memb, M)

fig = plt.figure()

ax = Axes3D(fig)


z1 = np.array(newTipicallity[1,0:clusters_size])
z2 = np.array(newTipicallity[0,clusters_size:2*clusters_size])
z3 = np.array(newTipicallity[0,2*clusters_size:clusters_size+2*clusters_size])

fig.hold()

ax.scatter(x1, y1, z1,  color='purple',                            # marker colour
                                          marker='o',                                # marker shape
                                          s=30                                       # marker size
                                          )

ax.scatter(x2, y2, z2,  color='red',                            # marker colour
                                          marker='o',                                # marker shape
                                          s=30                                       # marker size
                                          )
                                          
ax.scatter(x3, y3, z3,  color='red',                            # marker colour
                                          marker='x',                                # marker shape
                                          s=30                                       # marker size
                                          )


ax.set_zlabel('Membership Label')
plt.axis('equal')

#Generate Tipicallity
Tipicallity = np.matrix(np.random.random_sample((nclusters,2*clusters_size)))


Tipicallity, centroids = fuzzyCmeans(Data2,Tipicallity,nclusters,error,m_memb)

newTipicallity, centroids = PossibilisticClustering(Data2,Tipicallity,nclusters,error,m_memb,M)

print(centroids)

fig = plt.figure()

ax = Axes3D(fig)


z1 = np.array(newTipicallity[0,0:clusters_size])
z2 = np.array(newTipicallity[1,clusters_size:2*clusters_size])

fig.hold()

ax.scatter(x1, y1, z1,  color='purple',                            # marker colour
                                          marker='o',                                # marker shape
                                          s=30                                       # marker size
                                          )

ax.scatter(x2, y2, z2,  color='red',                            # marker colour
                                          marker='o',                                # marker shape
                                          s=30                                       # marker size
                                          )


ax.set_zlabel('Membership Label')
plt.axis('equal')


plt.show()





