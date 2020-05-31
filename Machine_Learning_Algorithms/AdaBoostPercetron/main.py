import numpy as np
import matplotlib.pyplot as plt
from AdaBoost import AdaBoost, Machine_Ada_Perceptron_Eval
from tabulate import tabulate

def gen_line(w,minr,maxr,nsamp):
  """
  Getting x and y for the weak learner
  """    
  #Generate samples for x
  y = np.array(np.linspace(minr,maxr,nsamp))
  # Generate the samples for y
  x = -w[0,0]/w[2,0]-(w[1,0]/w[2,0])*y
  return x,y
    
def Test_1(nC1,nC2):
  """
  Building the First Test
  """
  #Class1
  x1,y1 = class1(nC1, [-2.2,1.0], np.pi/8.0 ,2.0,0.5)    
  #Class2
  x2,y2 = class1(nC2, [2.2,-1.0], np.pi/4.0 ,2.0,1.0)
  C1 = (np.stack((x1,y1),axis = 0)).T
  C2 = (np.stack((x2,y2),axis = 0)).T     
  return C1,C2

def class1(numsamp, mu, theta,scalex,scaley):
  """
  Class 1 type
  """    
  #Define the covariance
  R = np.matrix([[np.cos(theta), -np.sin(theta)],[np.sin(theta), 
                  np.cos(theta)]])
  S = np.matrix([[scalex, 0.0],[0.0,scaley]])
  cov = R*S*S*np.transpose(R)
  x, y = np.random.multivariate_normal(mu, cov, numsamp).T
  return x,y

def Test_2(N):
  """
  Bulding Test Two
  """
  C1,C2 = class2(N, np.zeros(2), 4*np.eye(2), 2.0, 2.3)  
  nC1,m1 = C1.shape
  nC2,m1 = C2.shape  
  return C1, C2, nC1,nC2

def DrawingClasses(C1,C2,ThetaList, M, nslinea):
  plt.figure()
  plt.plot(C1[:,0], C1[:,1], 'bx')
  plt.plot(C2[:,0], C2[:,1], 'ro')
  
    # Find the Straihg Line
  # Find the min and max of x coordinate
  minr = np.amin(np.concatenate((C1[:,0].T,C2[:,0].T)))
  maxr = np.amax(np.concatenate((C1[:,1].T,C2[:,1].T)))

  for i in range(M): 
    x,y = gen_line(ThetaList[i],minr,maxr,nslinea )
    plt.plot(x, y, 'g')  
  
  plt.axis('equal')

def class2(N,mu, sigma, r1,r2):
  """
  Building a Ring Example
  """
  x,y = np.random.multivariate_normal(mu,sigma, N).T
  C = (np.stack((x,y),axis = 0)).T
  Value =  np.sqrt(np.sum(C*C,axis=1))
  mask = (Value>r2) & (Value<6.0)
  C1 = C[mask]
  C2 = C[Value<r1]
  
  return C1, C2
  
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
  print '{}'.format(15*'=')
  print 'Confusion Matrix'
  print '{}'.format(20*'=')
  print tabulate([['C1', TP , FP], ['C2', FN, TN]], headers=['', 'C1', 'C2'])
  print '{}'.format(20*'=')
  print ' '
  print '{}'.format(20*'=')
  print 'Confusion Matrix As Probabilities'
  print '{}'.format(20*'=')
  print tabulate([['C1', '{0:0.2f}'.format(float(TP)/float(P)) , '{0:0.2f}'.format(float(FP)/float(P))  ],
                   ['C2', '{0:0.2f}'.format(float(FN)/float(N)) , '{0:0.2f}'.format(float(TN)/float(N)) ]], 
                    headers=['', 'C1', 'C2'])
  print '{}'.format(20*'=')

  
if __name__ == '__main__':

#  ################### Test1 ####################
#  print ' '  
#  print 'TEST 1'
#  print ' '
#  nC1 = 200
#  nC2 = 400
#  C1, C2 = Test_1(nC1,nC2)
#   # For the lines
#  nslinea = 10
#  error = 1/10.0**9
#  eta   = 5.0
#  M = 10
#  alpha, ThetaList =  AdaBoost(C1, C2, error, 
#                               eta, M) 
#  DrawingClasses(C1,C2, ThetaList, M, nslinea)
#  L1 = Machine_Ada_Perceptron_Eval(C1, alpha, ThetaList) 
#  L2 = Machine_Ada_Perceptron_Eval(C2, alpha, ThetaList)                               
  #############################################                             
  
  ################### Test2 ####################
  print ' '  
  print 'TEST 2'
  print ' '
  N = 2000
  C3, C4, nC3,nC4 = Test_2(N)
  # Training the Weak Learners
  # For the lines
  nslinea = 10
  error = 1/10.0**10
  eta   = 100.0
  M = 25
  alpha, ThetaList =  AdaBoost(C3, C4, error, 
                               eta, M) 
                               
  DrawingClasses(C3,C4, ThetaList, M, nslinea) 
  Y1 = Machine_Ada_Perceptron_Eval(C3, alpha, ThetaList) 
  Y2 = Machine_Ada_Perceptron_Eval(C4, alpha, ThetaList) 
  Confusion_Matrix(Y1,Y2)
  #############################################

  plt.show()                             