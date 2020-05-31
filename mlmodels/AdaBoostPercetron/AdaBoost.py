# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:23:35 2017

@author: Andres Mendez-Vazquez
"""

import numpy as np

def AdaBoost(C1, C2, error, eta, M = 10):
  """
  Here is the implementation of the AdaBoost
  INPUT
  1.-Classes C1, C2
  2.-error for the theta
  3.-eta the step size for perceptron
  4.-M Number of Weak Learners
  OUTPUT
  1.- Theta List
  2.- Alpha List
  """
  # Get the dimension of the data  
  n1,m1 = C1.shape
  n2,m2 = C2.shape
  C1 = np.concatenate((np.ones([n1,1]),C1), axis = 1)  
  C2 = np.concatenate((np.ones([n2,1]),C2), axis = 1)
  # ThetaList
  ThetaList = list()
  # Unifrom Weights
  w1 = (1.0/float(n1+n2))*np.ones([n1,1])
  w2 = (1.0/float(n1+n2))*np.ones([n2,1])
  # Weights of correct incorrect errors
  W_e = 0
  W_c = 0
  W = 0
  alpha = list()
  # The building of the Machines
  for m in range(M):
    Theta = Perceptron(C1,C2, w1, w2, error, eta, m)
    W_c_1 =  np.sum(w1[C1.dot(Theta)>0])
    W_e_1 =  np.sum(w1[C1.dot(Theta)<=0])
    W_c_2 =  np.sum(w2[C2.dot(Theta)<0])
    W_e_2 =  np.sum(w2[C2.dot(Theta)>=0])
    
    W_c = W_c_1 + W_c_2
    W_e = W_e_1 + W_e_2
    W = W_c + W_e
    e_m = W_e/W
    c_e = np.sqrt((1.0-e_m)/e_m)
    c_c = np.sqrt(e_m/(1.0-e_m))
    # Here the weight update     
    w1[C1.dot(Theta)>0] = c_c*w1[C1.dot(Theta)>0]
    w1[C1.dot(Theta)<=0] = c_e*w1[C1.dot(Theta)<=0]
    w2[C2.dot(Theta)<0] = c_c*w2[C2.dot(Theta)<0]
    w2[C2.dot(Theta)>=0] = c_e*w2[C2.dot(Theta)>=0]
    # Reabalincding weigths into a Distribution
    Z = np.sum(w1)+np.sum(w2) 
    w1 = (1.0/Z)*w1
    w2 = (1.0/Z)*w2
    # the Alpha for the Weak Learner
    alpha_m =(1.0/2.0)*np.log((1.0-e_m)/e_m)
    # Save the important infro
    ThetaList.append(Theta)
    alpha.append(alpha_m)
    print ' '
    print 'Probability of error {} by machine {}'.format(e_m,m)
    print 'Alpha {} for machine {}'.format(alpha_m,m)
    print 'Weight Correct  {} by machine {}'.format(W_c,m)
    print 'Weight Incorrect {} by machine {}'.format(W_e,m)
    print ' '
  return alpha, ThetaList


def Machine_Ada_Perceptron_Eval(X, alpha, ThetaList):
  """
  Here, I am assuming no extended classes
  with samples at the rows
  """
  # Get the size of them 
  nsamples, m = X.shape
  # The Calculated Output
  Y = np.zeros([nsamples,1])
  M = len(ThetaList)
  X_e = np.concatenate((np.ones([nsamples,1]),X), axis = 1)  
  for i in range(M):
    Y = Y + alpha[i]*np.tanh(X_e.dot(ThetaList[i]))
    
  return np.sign(Y) 

def Perceptron(C1,C2, w1, w2, error, eta, M, IterPrint=60000):
  """
  Here is the weak learner
  INPUT
    1.- C1 class 1
    2.- C2 class 2
    3.- w1 weights from AdaBoost for Class 1
    4.- w2 weights from AdaBoost for Class 2
    5.- error for stoping training
    6.- eta learning step
  OUTPUT
    1.- Theta the weihgts of the perceptron
  """
  # Get the dimension of the data  
  n1,m1 = C1.shape
  n2,m2 = C2.shape
  # Get the labels for each class
  d1 = 1.0*np.ones([n1,1])
  d2 = -1.0*np.ones([n2,1])
  # Get the weight Theta
  OldTheta = 2*np.random.random_sample(m1)-1
  Theta = 2*np.random.random_sample(m1)-1
  # Convert it into a bidimensional array
  OldTheta.resize([m1,1])
  Theta.resize([m1,1])  
  # The errorwith respect to Theta 
  GE = DeltaTheta(OldTheta,Theta)
  #
  count = 0
  # Printing Weak Learner type 
  print 'Weak Learner Perceptron {}'.format(M)
  # The main loop    
  while GE>error:
    # Get the activations
    y1 = np.tanh(np.matmul(C1,Theta)) 
    y2 = np.tanh(np.matmul(C2,Theta))
    # the derivatives of activation     
    cy1 = 1-y1**2
    cy2 = 1-y2**2
    # the weights against y for the AdaBoost
    y1 = w1*y1
    y2 = w2*y2
    # the weights against y for the AdaBoost 
    tC1 = w1*C1
    tC2 = w2*C2
    # The error
    e1 = d1 - y1 
    e2 = d2 - y2
    # Use Brodcasting
    ne1 = (e1*cy1)*tC1
    ne2 = (e2*cy2)*tC2
    # The first part of the gradient
    TotalG = np.sum(ne1.T, axis=1)
    TotalG = TotalG + np.sum(ne2.T, axis=1)
    TotalG = (1.0/float(n1+n2))*TotalG.T
    TotalG.resize([m1,1])  
    OldTheta = Theta
    Theta = Theta-eta*TotalG
    GE = DeltaTheta(OldTheta,Theta)
    count+=1
    if count%IterPrint==0:
      print 'Iteration i={} - error={}'.format(count, GE)
  return Theta

def DeltaTheta(OldTheta, Theta):
  """
  Delta Theta
  """
  return ((Theta-OldTheta).T).dot(Theta-OldTheta)[0][0] 
  
def LinearSearchPerceptron():
  return 1
