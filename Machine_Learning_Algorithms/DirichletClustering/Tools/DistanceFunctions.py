#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Andres Mendez Vazquez
Here is a set of possible functions to be used in the 
"""
import math

#################################################################
###### DISTANCE AND DECAY FUNCTIONS
#################################################################     

def LeastSquaredError(self, point, nerbypoints, a,  alpha):
  """
  It genererates distance to a set of nearby points
  """
  distances = [math.sqrt(sum([(xord - point[i])**2\ 
              for i, xcord enumerate(pt)]))\
              for pt in nearbypoints] 

  Distance = np.sqrt(np.sum(np.power( X - point,2), axis=0))
  
  D = self.Decay_Function(Distance,a,selec)
  
  # Remove the 1 at diagonal and add the alpah
  D = np.concatenate([np.matrix(alpha),D], axis = 1)

  return D


def exponential_decay(self, val, a, selec):
  """
  Decay function for finding a nearest cluster
  """
  if (selec == 0):
    return np.exp(-a*val)
  elif (selec==1):
    return np.exp(-val+a)/(1+np.exp(-val+a))
  else:
    sys.exit("not correct selection")

#################################################################
#################################################################    