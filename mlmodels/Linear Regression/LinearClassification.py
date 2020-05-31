# -*- coding: utf-8 -*-
"""
Autor Andres Mendez

A class with several of the possible methods for classification
for the Linear Regression

"""

import numpy as np

class LinearClassification:
    """
    This is a collection of linear classifiers
    """
    
    def __init__(self, X, Y):
        """
        Here the object recieves only the following things
        Input:    
            1.- A N x d matrix with samples X
                a.- N is the number of samples
                b.- d is the original dimension of the 
                    samples before extending them 
                    2.- A matrix of zeros and ones where the sample 
                    belongs to the class k Y_k = 1 else 
                    Y_K = 0
        """
        N,d = X.shape

        # Store the Values
        self.X = np.concatenate((np.ones([N,1]),X), axis = 1)
        self.Y = Y      

###############################################################################    

    def Linear_Discriminant(self, sregularization):
        """
        Here, there is an implelemtnation of the classic 
        Linear Discriminant Analysis for K classes
        Input: 
            1.- The instantiaded data in self
            2.- What kind of Regularization you need
                a.- None       = 0
                b.- Thichonov  = 1
                c.- LASSO      = 2
        Output :
            1.- The Matrix of Weights B for the multiple machines
            2.- The Classification L of each sample to one of the classes 

        """
        if (sregularization == 0): # No regularization
            B = np.linalg.inv(self.X.T*self.X)*(self.X.T)*self.Y     
        elif (sregularization == 1): # Tichonov Regularization
            B = 1        
        elif (sregularization == 2): # LASSO Regularization
            B = 2
        else:
            print("You have not selected a valid regularization")
            return -1
        
        # Generation of the class labels
        
        N,K = B.shape        
        # Two classes or K classes
        if (K == 1):
            L = self.X*B
        else:
            L = np.argmax(self.X*B, axis = 1)
        
        return B,L
        
###############################################################################                
     

    def Logistic_Discriminant(self, terror, alpha = 0.001 , epsilon1 = 0.01,
                              l = 0.00001):
        """
        Implmentation of the Logistic Discriminant for two classes. 
        
        This implements the classic Raphson Method because the log-likelihood
        derivative gives an equuation where is necessary to look for the zeros 
        of it.
        
        Thus, instead of using the derivativew we use the Hessian for the 
        method to work.
        
        Question Could Quasi Netwon method work better? The Problem is the W
        It is as big as the number of samples....!!!
        
        Input: 
            1.- The instantiaded data in self
            
        Output :
            1.- A Vector B of Weights for the Logistic machines
            2.- The Classification L of each sample to one of the classes 
            
        It is possible to extend to K classes usign a linear machine 
        """
        
        # Be sure you are working only with two classes 
        Ry,Cy = self.Y.shape        
        
        if (Cy>1):
            print("The Logistic can only work with two classes")
            return -1,-1
            
        
        # Here we generate the diagonal W initial matrix
        # First a rand vector with normalization\
        
        N, d = self.X.shape       
        
        # Here the initial random weights
        B_new = np.matrix(np.zeros([d,1])) 
        
        # Main Training Loop
        alpha = 0.1         
        while True:        
        
            # This elements are arrays be careful
            # There is a difference
            # 1.- As matrix elements * is matrix mult
            # 2.- As array * is element wise mult
            p_x_B   =  self.Class1_Probability(B_new)
            p_1_x_B = 1.0 - p_x_B
        
            # Array element wise multiplication for the
            # Element W in the Hessian
            MainDiag = np.array(np.multiply(p_x_B, p_1_x_B))
            
            # Convert it into a list there is no other way
            MainDiag = [ i[0] for i in MainDiag]
            
            # The W Diagonal matrix in the Hessian 
            W = np.diag(MainDiag)            
            
            # The necessary Elements for the Newton-Raphson Method
            Gradient_B = self.Gradient_Logistic_Regression(p_x_B)
            Hessian_B_inv = self.Hessian_Inverse_Losgistic(W) 
            
            # Remember the Old Values
            B_old = np.copy(B_new)
            
            #alpha = self.Linear_Search_Logistic(B_old,alpha,epsilon1,l,
            #                                    Hessian_B_inv*Gradient_B)
            B_new = B_old + alpha*Hessian_B_inv*Gradient_B 
            alpha /=2
            
            if (np.sqrt((B_new-B_old).T*(B_new-B_old))<terror):
                print("getting out ")
                break
        
        
        L1 = np.exp(self.X*B_new)/(1.0+np.exp(self.X*B_new))
        L2 = 1.0/(1.0+np.exp(self.X*B_new))
        
        L = np.concatenate((L1,L2), axis = 1)
        
        return self.X , B_new, L   
###############################################################################        

# Here are the Gradient, Inverse Hessian of the two class case 
    def Gradient_Logistic_Regression(self, p_x_B):
        """
        The Logistic Regression Gradient of the likelihhod sum(log p_i). 
        """
        return self.X.T*(self.Y-p_x_B) 
        
    def Hessian_Inverse_Losgistic(self, W):
        """
        The invers of the Logistic Regression Gradient of the likelihhod 
        sum(log p_i). 
        """
        return np.linalg.inv(-self.X.T*W*self.X)
        
###############################################################################
        
# First Class Probability 

    def Class1_Probability(self, B):
        """
        Here you calculate the sigmoidal probabilitiesa of B.
        """
        return np.multiply(np.exp(self.X*B),
                    1.0/(1.0+np.exp(self.X*B)))
        
        
###############################################################################    
    def Regularization(self, selec):
        """
        The Different Cases of Regularization
        """
        if (selec == 1):
            print("Dummy Function")
        return selec

