#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2017 Sampler Project"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email__ = "amendez@gdl.cinvestav.mx"
__status__ = "Development"

from LRSampling import LRSampling
from Tools.BasicTools import BasicTools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression 

if __name__ == '__main__':
  
  print ' '  
  print 'Logistic Regression Using M-H'
  print ' '
  Display = True
  N       = 10000
  Burn    = 2000 
  Itera   = 4000
  nsamp   = 10
  mdisp   = 1000
  Labels  = [0,1]
  Disk    = 2
  # Generate Classes
  C, Y = BasicTools.Test_1(N)
  C1 = C[Y[:,0]==0]
  C2 = C[Y[:,0]==1] 
  # Parameter for seed distribution
  mu = np.mean(C,axis = 0)
  sigma = np.array([[Disk**2,0,0],[0,Disk**2,0],[0,0,Disk**2]])
  # Find the min and max of x coordinate
  minr = np.amin(np.concatenate((C1[:,1].T,C2[:,1].T)))
  maxr = np.amax(np.concatenate((C1[:,2].T,C2[:,2].T)))
  
  # The object for simulation
  Simulation = LRSampling(C,Y)
  # Simulation
  w = Simulation.RandomWalkMHLR(mu,sigma, Itera, mdisp)
  # Using the mode to find the estimation
  w_mode = Simulation.W_Estimation_Mode(Burn,Display)
  # Plot the Classes and Separaiting Line
  if Display == True:
      plt.figure()
      plt.plot(C1[:,1],C1[:,2],'rx')
      plt.plot(C2[:,1],C2[:,2],'bo')
      x,y = BasicTools.gen_line(w_mode,minr,maxr,nsamp)
      plt.plot(x, y, 'g') 
  # The Chains
  #BasicTools.DisplayMarkovChain(w)
  Y_est = Simulation.Estimation()
  BasicTools.Confusion_Matrix(Y,Y_est,Labels)
  LR = LogisticRegression()
  y = np.array(Y).reshape(2*N)
LR.fit(C,y)
nY_est = LR.predict(C).reshape(2*N,1)
BasicTools.Confusion_Matrix(Y,nY_est,Labels)
plt.show(block=False)