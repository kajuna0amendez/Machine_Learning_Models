# -*- coding: utf-8 -*-
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2018"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Closed"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

if __name__ == '__main__':
	
    # Use the Reader from MNIST
    mndata = MNIST('Data')
    mndata.gz = True
    # Get List of List of images and labels
    images, labels = mndata.load_training()
    	
    # Hyperparameters
    Number = 4
    cntnumber1 = 80   #Number of Samples
    cntnumber2 = 100
    Dimension = 784 - cntnumber2
    width1 = 8		  # Width Display of Numbers
    width2 = 10		  # Width Display of Eigenvectors
    #alpha = 0.00001   # Error
    	
    print('Number of Eigenvectors %i'%cntnumber2)
    print('Dimension taken in perc %i'%Dimension) 
    
    # Reasing Files
    ThreeList = []
    cnt = 0
    for i, number in enumerate(images):
        if labels[i] == Number and cnt < cntnumber1:
            ThreeList.append(number)
            cnt += 1
        if cnt > cntnumber1:
            break
    # Conver a List of images int an nparray
    npthree = np.array(ThreeList)

    # Plot the numbers
    h1 = plt.figure(1)
    for i in range(cntnumber1):
        row  = cntnumber1//width1
        if row*width1 >= cntnumber1:
            plt.subplot(row, width1, i+1)
        else:
            plt.subplot(row+1, width1, i+1)
        Num = npthree[i, :]
        Num = np.reshape(Num, [28,28])
        img = plt.imshow(Num)
        img.set_cmap('gray')
        plt.axis('off')
	
    #-------------PCA process--------------------
    meanval = np.mean(npthree, axis = 0)
	
    # Center the data set
    Xterm = npthree - meanval
	
    # Get the Covarance
    N, _ = Xterm.shape
    covnum = (1.0/float(N))*np.dot(Xterm.T, Xterm)
	
    # Get eigenvalues w and eigenvectors v column format
    w, v  =  np.linalg.eigh(covnum)
    #-------------PCA End--------------------
	
    # Extract the most important by percentage
    #Totalw = np.sum(w)
    #Percentage = Totalw - alpha*Totalw
    #N = len(w)
    #acc = 0.0
    #for i in range(N-1,-1,-1):
    #   acc += w[i]
    #   if acc > Percentage:
    #       Dimension = i
    #       break
    
    # Get the cntnumber2 of eigenvectors
    U = v[:, Dimension:]

    # Rebuild the samples
    R = np.dot(U.T, npthree.T)
    Result = np.dot(U, R).T
    # Plot them
    h2 = plt.figure(2)
    for i in range(cntnumber1):
        row  = cntnumber1//width1
        if row*width1 >= cntnumber1:
            plt.subplot(row, width1, i+1)
        else:
            plt.subplot(row+1, width1, i+1)
        Num = Result[i, :]
        Num = np.reshape(Num, [28,28])
        img = plt.imshow(Num)
        img.set_cmap('gray')
        plt.axis('off')
  
    # Plot the Eigenvectors
    h3 = plt.figure(3)
    for i in range(cntnumber2,1,-1):
        row  = cntnumber2//width2
        if row*width2 >= cntnumber2:
            plt.subplot(row, width2, cntnumber2-i+1)
        else:
            plt.subplot(row+1, width2, cntnumber2-i+1)
        Num = U.T[i-1, :]
        Num = np.reshape(Num, [28,28])
        img = plt.imshow(Num)
        img.set_cmap('gray')
        plt.axis('off')

    h1.savefig('Original.svg', transparent = True)
    h2.savefig('Reconstructed.svg', transparent = True)
    h3.savefig('Eigenvectors.svg', transparent = True)
    #plt.show(block = False)
	

	
	
		
	
	