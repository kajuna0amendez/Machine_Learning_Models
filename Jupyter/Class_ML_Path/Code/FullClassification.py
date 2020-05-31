# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = 'Andres Mendez-Vazquez'
__copyright__ = 'Copyright 2019'
__credits__ = ['Andres Mendez-Vazquez']
__license__ = 'Open'
__version__ = '1.0.0'
__maintainer__ = 'Andres Mendez-Vazquez'
__email__ = 'amendez@gdl.cinvestav.mx'
__status__ = 'Development'


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
import itertools
from tabulate import tabulate

def Normalization(Data):
    Mean1 = np.mean(Data, axis = 0)
    Std1  = np.std(Data, axis = 0)
    return (Data-Mean1)/Std1

def MahalonobisDetection(Data, alpha):
    Data = Data - np.mean(Data, axis = 0)
    n1,n2 = Data.shape
    Cov = (1/float(n1-1))*Data.T*Data
    M = np.zeros(n1)
    for i in range(0,n1):
        M[i] = Data[i,:]*np.linalg.inv(Cov)*Data.T[:,i]
    c = chi2.isf(alpha,n2) 
    return  M, c , Cov

def ReturnDataFrame(path):
        return pd.read_csv(path, sep=',',skipinitialspace=True)      
    
def PCA(NData):
    NDataMean = NData - np.mean(NData,axis = 0)

    n1 , n2 = NDataMean.shape

    NCov = (NDataMean.T)*NDataMean
    NCov = (1/float(n1-1))*NCov
    NEigenvaluesc, NEigenvectorsc = np.linalg.eigh(NCov) 
    idx = NEigenvaluesc.argsort()[::-1]  
    NEigenvaluesc = NEigenvaluesc[idx]
    NEigenvectorsc  =  NEigenvectorsc [:,idx]
    return NEigenvaluesc, NEigenvectorsc

def SelectingBestSubset2class(Data, nfeat, fmask,mmask):
    
    t1 , t2 = Data.shape
    
    C1 = np.asmatrix(Data[fmask,:])
    C2 = np.asmatrix(Data[mmask,:])
    n1, dummy = C1.shape
    n2, dummy = C2.shape    
    
    P1 = float(n1)/float(t1)
    P2 = float(n2)/float(t1)
    
    Flag = True 
    
    L1   = range(t2)
    
    t2 = t2 -1
    
    J = -100000.0
    
    while(Flag):
        p1 = list(itertools.combinations(L1,t2))
        print(len(p1))
        for j in p1:
            TData = Data[:,j]
            C1 = np.asmatrix(TData[fmask,:])
            C2 = np.asmatrix(TData[mmask,:])
            Cov1 = (1/float(n1-1))*C1.T*C1
            Cov2 = (1/float(n2-1))*C2.T*C2         
            Sw = P1*Cov1+P2*Cov2
            m1 = (1/float(n1))*np.sum(C1,axis = 0)
            m2 = (1/float(n2))*np.sum(C2,axis = 0)
            m0 = P1*m1+P2*m2
            Sm = (1/float(t1-1))*(TData - m0).T*(TData-m0)
            
            Jt = np.trace(Sm)/np.trace(Sw)
            
            if (Jt > J):
                print(L1)
                J = Jt
                L1 = j
                
        if (t2 == nfeat):
            Flag = False
            print('The selected features ')
            print(L1)
            print('J value for selection '+str(J))

        t2 = t2-1
         
    return L1, J

def kmeans(Data,centroids,error):
    lbelong = []
    x1,x2 = Data.shape
    y1,y2 = centroids.shape
    oldcentroids = np.matrix(np.random.random_sample((y1,y2)))
    # Loop for the epochs
    # This allows to control the error
    trace = [];
    while ( np.sqrt(np.sum(np.power(oldcentroids-centroids,2)))>error):
        # Loop for the Data
        for i in range(0,x2):
            dist = []
            point = Data[:,i]
            #loop for the centroids
            for j in range(0, y2):
                centroid = centroids[:,j]
                dist.append(np.sqrt(np.sum(np.power(point-centroid,2))))
            lbelong.append(dist.index(min(dist)))        
        oldcentroids = centroids
        trace.append(centroids)
        
        #Update centroids     
        for j in range(0, y2):
            indexc = [i for i,val in enumerate(lbelong) if val==(j)]
            Datac = Data[:,indexc]
            print(len(indexc))
            if (len(indexc)>0):
                centroids[:,j]= Datac.sum(axis=1)/len(indexc)
    return centroids, lbelong, trace

def LinearRegression(Class1, Class2):
    # Generate the X
    n1, dummy = Class1.shape
    n2, dummy = Class2.shape

    C1 = np.hstack((np.ones((n1,1)),Class1))
    C2 = np.hstack((np.ones((n2,1)),Class2))
    X = np.matrix(np.vstack((C1,C2)))
    # Get the label array
    y = np.matrix(np.vstack((np.ones((n1,1)),-np.ones((n2,1)))))

    # Finally get the w for the decision surface
    w = np.linalg.inv((np.transpose(X)*X))*np.transpose(X)*y    
    
    return X[0:n1,:]*w, X[n1:n1+n2,:]*w

def gen_line(w,minr,maxr,nsamp):
    # Generate samples for x
    x = np.array(np.linspace(minr,maxr,nsamp))

    # Generate the samples for y
    y = -w[0,0]/w[2,0]-(w[1,0]/w[2,0])*x

    return x,y

def Confusion_Matrix(Y1,Y2):
  """
  Print the Confusion Matrix
  """
  P,M1 = Y1.shape 
  N,M2 = Y2.shape
  TP = np.sum(1*(Y1>0))
  TN = np.sum(1*(Y2<0))
  FP = np.sum(1*(Y1<=0))
  FN = np.sum(1*(Y2>=0))
  print('{}'.format(15*'='))
  print('Confusion Matrix')
  print('{}'.format(20*'='))
  print(tabulate([['C1', TP , FP], ['C2', FN, TN]], headers=['', 'C1', 'C2']))
  print('{}'.format(20*'='))
  print(' ')
  print('{}'.format(20*'='))
  print('Confusion Matrix As Probabilities')
  print('{}'.format(20*'='))
  print(tabulate([['C1', '{0:0.2f}'.format(float(TP)/float(P)) , '{0:0.2f}'.format(float(FP)/float(P))  ],
                   ['C2', '{0:0.2f}'.format(float(FN)/float(N)) , '{0:0.2f}'.format(float(TN)/float(N)) ]], 
                    headers=['', 'C1', 'C2']))
  print('{}'.format(20*'='))



if __name__ == '__main__':
  # Load CVS
  Path1 = 'voice.csv'
  DataMatrix = ReturnDataFrame(Path1)
  
  DataMatrix.replace({'male': -1.0, 'female': 1.0},
                      inplace=True)
  
  DataLabels = DataMatrix['label']
  
  DataMatrix.drop('label', axis=1, inplace=True)
  
  
  # Transform to an NP Array
  Data = DataMatrix.as_matrix()
  Label = DataLabels.as_matrix()
  
  fmask = (Label == 1.0)
  mmask = (Label == -1.0)
  
  # Normalize your Data # 
  NData = np.asmatrix(Normalization(Data))
  #NData = np.asmatrix(Data)
  
  
  # Select Best Features
  nfeat = 4
  L1  , J = SelectingBestSubset2class(NData, nfeat, fmask,mmask)
  
  # Select The Best 
  BNData = NData[:,L1]
  
  ### Apply the Eigenvalues
  Eigv, Eig = PCA(BNData)
  #
  idx = Eigv.argsort()[::-1]   
  #
  Eigv = Eigv[idx]
  Eig = Eig[:,idx]
  
  NP1 =  np.transpose(Eig)
  #
  PBNData = (NP1*BNData.T).T
  #
  Class1 = PBNData[fmask,:]
  Class2 = PBNData[mmask,:]
  
  
  ## Detect The Outliers and Remove Them # 
  
  alpha = 0.05
  
  M1, c1 , cov1 = MahalonobisDetection(Class1, alpha)
  M2, c2 , cov2 = MahalonobisDetection(Class2, alpha)
  #
  Class1 = Class1[(M1<c1),:]
  Class2 = Class2[(M2<c2),:] 
  #
  #
  for i in [[0,1,2]]: #list(itertools.combinations(range(len(L1)),3)):
      fig = plt.figure()
      ax = Axes3D(fig)
      ax.scatter(np.array(Class1[:,i[0]]),np.array(Class1[:,i[1]]),\
                 np.array(Class1[:,i[2]]),color='red',marker='o', \
                label ='Class 1')
      ax.scatter(np.array(Class2[:,i[0]]),np.array(Class2[:,i[1]]),\
                 np.array(Class2[:,i[2]]),color='blue',marker='x',
                label = 'Class 2')
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),\
                ncol=3, fancybox=True, shadow=True)
      ax.legend(title = 'Classes with different combination of features')
  dummy , dim = PBNData.shape
  
  ncentroids = 4
  
  error = 0.00001
  
  centroids = np.matrix(np.random.random_sample((dim,ncentroids)))
  
  fcentroids, fbelong, ftrace = kmeans(PBNData.T,centroids,error)
  
  clr = ['b','g','r','c','m','y','b','w']
  sym = ['.',',','o','v','^']
  
  clusters = []
  for j in range(0, ncentroids):
     indexc = [i for i,val in enumerate(fbelong) if val==(j)]
     clusters.append(PBNData[indexc,:])
     
  fig = plt.figure()  
  ax = Axes3D(fig)  
  for j in range(0, ncentroids):
      ax.scatter(np.array(clusters[j][:,1]),\
                 np.array(clusters[j][:,2]),\
                 np.array(clusters[j][:,3]),\
                 color=clr[j],marker=sym[j],)
      l1=ax.scatter(fcentroids[1,j],fcentroids[2,j],\
                 fcentroids[3,j],color=clr[6],marker=sym[4])
  plt.title('K-Means')
  plt.axis('equal')
  
  Classification1, Classification2 =  LinearRegression(Class1, Class2)
  
  Confusion_Matrix(Classification1,Classification2)

  plt.show()