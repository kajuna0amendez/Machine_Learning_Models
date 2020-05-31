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
import matplotlib.pyplot as plt

class LRSampling():
  """
  This class implements the sampling methods for the Logistic Regression 
  """
  def __init__(self,C,Y):
    """
    Init method
    """
    assert type(C) is np.ndarray, 'C it is not an array'
    assert type(Y) is np.ndarray, 'Y it is not an array'
    N1,N2 = C.shape
    M1, _ = Y.shape
    assert N1==M1, 'The multiplication C*Y is\
                    not possible for dim problem'
    self.__C = C
    self.__Y = Y
    # Storage for the chain
    self.__w_chain = None
    # Storage of the w
    self.__w       = None
  ##############################################################  
  ##############    Getters and Setters   ######################
  ##############################################################
  def get_C(self):
    return self.__C
  def set_C(self,C):
    self.__C = C
  def get_Y(self):
    return self.__Y
  def set_Y(self, Y):
    self.__Y = Y
  def get_w_chain(self):
    return self.w_chain
  def set_w_chain(self, w_chain):
    self.w_chain = w_chain
  def get_w(self):
    return self.w
  def set_w(self, w):
    self.w = w
  ##############################################################
  ##############################################################  
#  def LR(self,w):
#    """
#    The Loggistic Regression Cost Function
#    """
#    np.seterr(divide='ignore',over='ignore', invalid='ignore')
#    f = self.get_C().dot(w.T)
#    f1 = np.exp(0.1*f)
#    p1 = (f1/(1.0+f1))**self.get_Y()
#    p2 = (1.0/(1.0+f1))**(1-self.get_Y())
#    p3 = p1*p2
#    return np.prod(p3)
  
  def LR(self,w):
    """
    The Loggistic Regression Cost Function
    """
    np.seterr(divide='ignore',over='ignore', invalid='ignore')
    f = self.get_C().dot(w.T)
    f1 = np.exp(0.1*f)
    p1 = self.get_Y()*np.log(2.2*(f1/(1.0+f1)))
    p2 = (1-self.get_Y())*np.log(2.2*(1.0/(1.0+f1)))
    p3 = p1+p2
    return np.sum(p3)

  def RandomWalkMHLR(self,mu,sigma,Itera=100,  mdisp = 100 ):
    """
    Random Walk M-H using a Multinomial 
    Gaussian for emitting the samples
    INPUT
    1.- mu,sigma = The parameters for the q function 
    2.- Itera    = Number of iterations
    3.- mdisp    = Display the counter%mdisp == 0
    OUTPOUT
    1.- The Chain after simulation
    """
    print '='*25  
    print 'Random Walk MHLR Run'
    print '='*25
    # Sample the Multinomail Gaussian
    w_old = np.random.multivariate_normal(mu,sigma,1)
    w_chain = list()
    w_chain.append(w_old)
    # Sample until all the iterations are obtained
    while True:
      # Sample
      w = np.random.multivariate_normal(mu,sigma,1)
      # Evaluate Cost Function
      p1 =  self.LR(w)
      p2 =  self.LR(w_old)
      # Evaluate the ratio
      if p2 > 0.0:
        if p1/p2>1:
          A = 1.0
        else:
          A = p1/p2
      else:
        A = 1.0
      # Sample from a unifrom distribution
      u = np.random.uniform(0,1,1)[0]
      # Test tthe decision 
      if u<A: 
        # Display each mdisp 
        if len(w_chain)%mdisp==0:
          print 'Add w # {} - {}'.format(len(w_chain),w[0])
        w_chain.append(w)
        w_old = w   
      # Break once you have the correct number of samples
      if len(w_chain)>Itera:
        break
    self.set_w_chain(w_chain)
    return w_chain
  
  def Estimation(self):
    """
    Estimation of the values
    """
    print '='*25  
    print 'Estimation done'
    print '='*25
    f = self.get_C().dot(self.get_w().T)
    f1 = np.exp(f)
    p1 = (f1/(1.0+f1))
    p2 = (1.0/(1.0+f1))
    Y_est = [np.argmax([p2[i],p1[i]]) for i in range(len(p1)) ]
    return np.matrix(Y_est).T
  
  def W_Estimation_Mode(self, Burn, Display = False, nbins = 100):
    """
    Here,we calcualte the mode of each sub-chain
    INPUT:
        1.- nbins The number of bins for the histogram
        2.- Burn  The wasted samples 
    """
    print '='*25  
    print 'Etimation of W'
    print '='*25
    # Chain
    w_chain = self.get_w_chain()
    # Number of chains
    N     = len(w_chain[0][0])
    # Estimated w
    w_mode = list()
    # Get the mode using histogram
    for i in range(N):
      chain = np.asarray([w[0][i] for w in w_chain])
      count, rbin = np.histogram(chain[Burn:], bins=nbins)
      if Display == True:
          plt.figure()
          plt.hist(chain[Burn:],bins=nbins)
          plt.show(block=False)
      index = np.argmax(count)
      #w_mode.append(np.mean(chain))
      w_mode.append(rbin[index])
    self.set_w(np.array(w_mode))
    return np.array(w_mode)