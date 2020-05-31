import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return 2+np.cos(x)+0.5*np.cos(2*x-0.5)
    #return  np.power(x,3)-3*np.power(x,2)+x-2

def Gf1(x):
    return -np.sin(x)-np.sin(2*x-0.5) 
    #return  3*np.power(x,2)-6*x+1

def GradientDescent(x,Nm,eg,et,alpha,epsilon1,l):
    """ Gradient Descent Algorithm """
    xp    = np.zeros(Nm)
    xp[0] = x
    alphak = alpha
    for t in xrange(0,Nm-1):
        alphak = linear_search(xp[t],alphak,epsilon1,l)
        xp[t+1]=xp[t]-alphak*Gf1(xp[t])
        if np.abs(Gf1(xp[t+1]))<eg:
            print "Converged on critical point"             
            return xp,t+1
        if np.abs(xp[t+1]-xp[t])<et:
            print "Converged on an x value"
            return xp,t+1
        if f1(xp[t])<f1(xp[t+1]):
            print "Diverging"
            return xp,t+1
    print "Maximum number of iterations reached"
    return xp, Nm

def linear_search(x,alpha,epsilon1,l):
    """Linear Search of the best alpha"""
    ak = alpha - epsilon1
    bk = alpha + epsilon1
    while bk-ak > l:
        lk = ((ak+bk)/2+ak)/2
        mk = ((ak+bk)/2+bk)/2

        if f1(x-lk*Gf1(x))< f1(x-mk*Gf1(x)):
            bk = (ak+bk)/2 
        else:
            ak = (ak+bk)/2
    return (ak+bk)/2
            
    
if __name__ == '__main__':
  #Initial Values
  x     = 0.0
  Nm    = 500
  eg    = 0.001
  et    = 0.001
  alpha = 0.001 
  delta = 0.001
  epsilon1 = 0.01 
  l        = 0.00001
  
  xp, t = GradientDescent(x,Nm,eg,et,alpha,epsilon1,l)
  
  xr = np.arange(-6, 8, delta)
  
  y = f1(xr)
  yg = f1(xp)
  
  fig = plt.figure()
  
  plt.plot(xr,y,color='blue', linewidth=2)
  plt.plot(xp[:t],yg[:t],'ro')
  
  plt.grid()
  
  plt.show()