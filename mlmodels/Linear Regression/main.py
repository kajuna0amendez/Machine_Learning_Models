# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:31:25 2016

This is the main to test the viability of the Logistic Regression Algorithm

@author: andres
"""

import numpy as np
import matplotlib.pyplot as plt

from LinearClassification import LinearClassification

def gen_line(w,minr,maxr,nsamp):
    # Generate samples for x
    y = np.array(np.linspace(minr,maxr,nsamp))

    # Generate the samples for y
    x = -w[0,0]/w[2,0]-(w[1,0]/w[2,0])*y

    return x,y

def class1(numsamp, mu, theta,scalex,scaley):
    # Define the covariance
    R = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    S = np.matrix([[scalex, 0.0],[0.0,scaley]])
    cov = R*S*S*np.transpose(R)
    x, y = np.random.multivariate_normal(mu, cov, numsamp).T

    return x,y
    
    
# Sample number for Gaussians
nsamples = 400

# For the lines
nsamp = 10

terror = 1/10.0**100

# Building the samples
x1,y1 = class1(nsamples, [-2.2,1.0], np.pi/8.0 ,2.0,0.5)
x2,y2 = class1(nsamples, [2.2,-1.0], np.pi/4.0 ,2.0,1.0)

C1 = np.stack((x1,y1),axis = 1)
C2 = np.stack((x2,y2),axis = 1)
X = np.matrix(np.vstack((C1,C2)))

Y = np.matrix(np.vstack((np.zeros((nsamples,1)),np.ones((nsamples,1)))))

Classificator1 = LinearClassification(X,Y)


X_extend, B, L = Classificator1.Logistic_Discriminant(terror)


Clases = X_extend*B



# Plot Everything
plt.figure(1)
plt.plot(X[0:399,0], X[0:399,1], 'bx')
plt.plot(X[400:799,0], X[400:799,1], 'ro')

# Find the Straihg Line
# Find the min and max of x coordinate
minr = np.amin(np.concatenate((x1,x2)))
maxr = np.amax(np.concatenate((x1,x2)))

# Generate the line
x3,y3 = gen_line(B,minr,maxr,nsamp)


plt.plot(x3, y3, 'g')

plt.axis('equal')
plt.savefig('LMSRegression.svg', transparent=True)
plt.show()