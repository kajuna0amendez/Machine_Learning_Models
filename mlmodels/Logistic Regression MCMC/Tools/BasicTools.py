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

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

class BasicTools():
  """
  Static Bassic Tools For Different Sampler Methods
  """
  def __init__():
    """
    Empty init method
    """
  @staticmethod
  def DisplayMarkovChain(w_chain):
    """
    Display the Markov Chain
    """
    # Colors for the chains
    clr   = ['g','c','b','r']
    # Length of the Chain
    LenMC = len(w_chain)
    # Number of chains
    N     = len(w_chain[0][0])
    # Plot subchains
    fig , subplt = plt.subplots(nrows=N, figsize=(8, 9))
    for i in range(N):
      chain = np.asarray([w[0][i] for w in w_chain])
      subplt[i].plot(range(LenMC),chain,clr[i%4], linewidth=2.5,\
                    label='Chain w[{}]'.format(i))
      subplt[i].legend()
    # Add title
    plt.suptitle('Markov Chains')
    plt.show(block=False)
  @staticmethod
  def gen_line(w,minr,maxr,nsamp=100):
    """
    Getting x and y for the weak learner
    INPUT
    1.- the weight w describing the line
    2.- minr and maxr for y
    3.- nsamp the number of samples
    OUTPUT
    """    
    #Generate samples for x
    y = np.array(np.linspace(minr,maxr,nsamp))
    # Generate the samples for y
    x = -w[0]/w[2]-(w[1]/w[2])*y
    return x,y

  @staticmethod    
  def Test_1(N):
    """
    Metod to building the First Test Samples
    """
    def class1(nsamp, mu, theta,scalex,scaley):
      """
      Class 1 type
      INPUT
      1.- nsamp number of samples to be generated
      2.- mu, theta for the multinomial distibutions
      4.- scalex, scalexy the parameter to deform the multinomial emisions
      OUTPUT
      """    
      #Define the covariance
      R = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), 
                      np.cos(theta)]])
      S = np.matrix([[scalex, 0.0],[0.0,scaley]])
      cov = R*S*S*np.transpose(R)
      x, y = np.random.multivariate_normal(mu, cov, nsamp).T
      return x,y
    #Class1
    x1,y1 = class1(N, [-2.0,1.0], np.pi/8.0 ,2.0,0.5)    
    #Class2
    x2,y2 = class1(N, [2.0,-1.0], np.pi/4.0 ,2.0,1.0)
    C1 = (np.stack((x1,y1),axis = 0)).T
    C2 = (np.stack((x2,y2),axis = 0)).T
    C1 = np.concatenate((np.ones([N,1]),C1), axis = 1)  
    C2 = np.concatenate((np.ones([N,1]),C2), axis = 1)
    Y1 =  np.ones([N,1])
    Y2 =  np.zeros([N,1])    
    return np.vstack((C1,C2)), np.vstack((Y1,Y2))

  @staticmethod
  def Confusion_Matrix(Y_real,Y_est,Labels):
    """
    Print the Confusion Matrix
    INPUT
    1.- Y_real the real labels
    2.- Y_est the estimated Y
    3.- Labels the labels given 
    OUTPUT
    2.- Confusion Matrix CF as count and prob
    """
    # Def function for the keys    
    def f_key(x):
      return x[1]
    # Getting the Confusion Matrix
    M = len(Labels)
    CF = np.zeros([M,M])
    N,_ = Y_real.shape
    for i in range(N):
      p   = [(k, abs(j-Y_est[i])) for k,j in enumerate(Labels)]
      y_l = min(p,key=f_key)[0]
      x_l = Labels.index(Y_real[i])
      CF[x_l,y_l]+=1.0
    # Printing the Confusion Matrix
    print '{}'.format(25*'=')
    print 'Confusion Matrix'
    print '{}'.format(25*'=')
    rows = list()
    for i,j in enumerate(Labels):
      row  = CF[i,:].tolist()
      rows.append([j]+row)
    print tabulate(rows, headers=['']+Labels)
    print '{}'.format(25*'=')
    print ' '
    # Printing Confusion 
    pCF = CF/np.sum(CF,axis=1)  
    prows = list()
    for i,j in enumerate(Labels):
      row  = pCF[i,:].tolist()
      prows.append([j]+row)    
    print '{}'.format(35*'=')
    print 'Confusion Matrix As Probabilities'
    print '{}'.format(35*'=')
    print tabulate(prows, headers=['']+Labels)
    return CF, pCF