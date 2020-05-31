#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:52:11 2019

@author: andres
"""
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import svds
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import coo_matrix 
import collections
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import chi2
import itertools
#matplotlib inline


def Normalization(Data):
    Mean1 = np.mean(Data, axis = 0)
    Std1  = np.std(Data, axis = 0)
    return (Data-Mean1)/Std1

#@jit(nopython=True)
def MahalonobisDetection(Data, alpha):
    Data = Data - np.mean(Data, axis = 0)
    n1,n2 = Data.shape
    Cov = (1/float(n1-1))*np.dot(Data.T,Data)
    M = np.zeros(n1)
    for i in range(0,n1):
        M[i] = np.dot(Data[i,:],np.dot(np.linalg.inv(Cov),Data.T[:,i]))
    #M = np.diag(np.dot(Data,np.dot(np.linalg.inv(Cov),Data.T)))
    c = chi2.isf(alpha,n2) 
    return  M, c , Cov

def ReturnDataFrame(path):
        return pd.read_csv(path, sep=',',skipinitialspace=True)      
    
def PCA(NData):
    NDataMean = NData - np.mean(NData,axis = 0)

    n1 , n2 = NDataMean.shape

    NCov = np.dot((NDataMean.T),NDataMean)
    NCov = (1/float(n1-1))*NCov
    NEigenvaluesc, NEigenvectorsc = np.linalg.eigh(NCov) 
    idx = NEigenvaluesc.argsort()[::-1]  
    NEigenvaluesc = NEigenvaluesc[idx]
    NEigenvectorsc  =  NEigenvectorsc [:,idx]
    return NEigenvaluesc, NEigenvectorsc

def SelectingBestSubset2class(Data, nfeat, fmask,mmask):
    
    t1 , t2 = Data.shape
    
    C1 = np.asmatrix(Data[fmask,:])
    C2 = np.asmatrix(Data[mmask,:])
    n1, dummy = C1.shape
    n2, dummy = C2.shape    
    
    P1 = float(n1)/float(t1)
    P2 = float(n2)/float(t1)
    
    Flag = True 
    
    L1   = range(t2)
    
    t2 = t2 -1
    
    J = -1e6
    
    while(Flag):
        p1 = list(itertools.combinations(L1,t2))
        print(len(p1))
        for j in p1:
            TData = Data[:,j]
            C1 = np.asmatrix(TData[fmask,:])
            C2 = np.asmatrix(TData[mmask,:])
            C1 = C1 - np.mean(C1,axis=0)
            C2 = C2 - np.mean(C2,axis=0)
            Cov1 = (1/float(n1-1))*np.dot(C1.T,C1)
            Cov2 = (1/float(n2-1))*np.dot(C2.T,C2)         
            Sw = P1*Cov1+P2*Cov2
            m1 = (1/float(n1))*np.sum(C1,axis = 0)
            m2 = (1/float(n2))*np.sum(C2,axis = 0)
            m0 = P1*m1+P2*m2
            Sm = (1/float(t1-1))*np.dot((TData - m0).T,(TData-m0))
            
            Jt = np.trace(Sm)/np.trace(Sw)
            
            if (Jt > J):
                J = Jt
                L1 = j
        print('Best %i'%t2)
        print(L1)
        print('J Value %f'%J)
        if (t2 == nfeat):
            Flag = False
            print('The selected features ')
            print(L1)
            print('J value for selection '+str(J))
        Jold = J
        J = -1e6
        t2 = t2-1
         
    return L1, Jold

def Grid_Search_Selecting_Features( Data, fmask, mmask):
    """
    Here a Grid Search for the Feature Selection
    """
    # Set the minimal value
    Winer = {
        'L1' : [],#dummy value now
        'J'  : -1e6 # dummy value now
    }
    # Loop for the Grid Search
    for nfeat in range(1,20):
        L1, Jval = SelectingBestSubset2class(Data, nfeat, fmask, mmask)
        print(Jval)
        if Winer['J'] < Jval:
            Winer['J'] = Jval
            Winer['L1'] = L1
    return Winer['L1'], Winer['J']

def kcenter(Data,K):
    """
    k-center algorithms and data is in column format 
    """
    x1, x2 = Data.shape
    # Random selection of h1
    h1 = np.random.choice(x2)

    # Distance of each x
    distx =  np.zeros(x2)
    # Cluster Centroids
    H      =  np.matrix(np.zeros((x1,K)))
    # Labels 
    Labels =  np.zeros(x2) 
    D      =  np.zeros(x2-1)
    
    # Choose the correct element
    H[:,0] = Data[:,h1]    
    
        
    for i in range(0, x2):
        distx[i] = np.linalg.norm(Data[:,i]- H[:,0])
        Labels[i]=0
    
    for i in range(1,K):
        D[i-1] = np.amax(distx)
        H[:,i] = Data[:, np.argmax(distx)]
        for j in range(0, x2):
            L = np.linalg.norm(Data[:,j]- H[:,i])
            if L<= distx[j]:
                distx[j]=L
                Labels[j]=i
    
    return H, Labels

def kmeans(Data,centroids,error):
    """
    k-mean algorithms and data is in column format 
    """
    lbelong = []
    x1,x2 = Data.shape
    y1,y2 = centroids.shape
    oldcentroids = np.matrix(np.random.random_sample((y1,y2)))
    # Loop for the epochs
    # This allows to control the error
    trace = [];
    while ( np.sqrt(np.sum(np.power(oldcentroids-centroids,2)))>error):
        # Loop for the Data
        for i in range(0,x2):
            dist = []
            point = Data[:,i]
            #loop for the centroids
            for j in range(0, y2):
                centroid = centroids[:,j]
                dist.append(np.sqrt(np.sum(np.power(point-centroid,2))))
            lbelong.append(dist.index(min(dist)))        
        oldcentroids = centroids
        trace.append(centroids)
        
        #Update centroids     
        for j in range(0, y2):
            indexc = [i for i,val in enumerate(lbelong) if val==(j)]
            Datac = Data[:,indexc]
            print(len(indexc))
            if (len(indexc)>0):
                centroids[:,j]= Datac.sum(axis=1)/len(indexc)
    return centroids, lbelong, trace

def Grid_Search_Clusters(Data, minc, maxc):
    """
    Grid search algorithm for clusters, and data is in column format 
    """
    
    results = list()
    
    print("Shapes %i %i"%Data.shape)
    
    for K in range(minc,maxc+1):
        centroids, _ = kcenter(Data,K)
        centroids, lbelong, _ = kmeans(Data,centroids,error)
        array_belong = np.array(lbelong)
        acc = 0.0
        print('Total Numbers of Samples %i'%len(lbelong))
        for i in range(K):
            acc = np.sum(np.sum(np.power(Data[:, array_belong == i]-centroids[:,i],2), axis = 0),axis = 1)[0,0]
        
        results.append([K, acc])
    return results

def LinearRegression(Class1, Class2):
    # Generate the X
    n1, dummy = Class1.shape
    n2, dummy = Class2.shape

    C1 = np.hstack((np.ones((n1,1)),Class1))
    C2 = np.hstack((np.ones((n2,1)),Class2))
    X = np.matrix(np.vstack((C1,C2)))
    # Get the label array
    y = np.matrix(np.vstack((np.ones((n1,1)),-np.ones((n2,1)))))

    # Finally get the w for the decision surface
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))    
    
    return X[0:n1,:]*w, X[n1:n1+n2,:]*w, w

def gen_line(w,minr,maxr,nsamp):
    # Generate samples for x
    x = np.array(np.linspace(minr,maxr,nsamp))

    # Generate the samples for y
    y = -w[0,0]/w[2,0]-(w[1,0]/w[2,0])*x

    return x,y


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
    print('{}'.format(15*'='))
    print('Confusion Matrix')
    print('{}'.format(20*'='))
    print(' %i  %i '%(TP, FP))
    print(' %i  %i '%(FN, TN))
    print('{}'.format(20*'='))
    print(' ')
    print('{}'.format(20*'='))
    print('Confusion Matrix As Probabilities')
    print('{}'.format(20*'='))
    print(' %2f  %2f '%(float(TP)/float(P), float(FP)/float(P)))
    print(' %2f  %2f '%(float(FN)/float(N), float(TN)/float(N)))
    print('{}'.format(20*'='))

    
def plot_eiganvalues(eigv):
    """
    Pritning the eigenvalues 
    """
    x = np.array(range(1,len(eigv)+1))
    plt.figure()
    plt.plot(x, eigv, color='blue', linewidth=3)
    plt.show()

def fval(x):
    return x[1]
    
def roc_curve(estimate1,estimate2, P, N):
    """
    ROC Curve Plotting 
    """
    M = P+N
    
    tL = np.concatenate((estimate1, estimate2))

    L = [(i, val[0,0]) for i, val in enumerate(tL)]
    L.sort(key = fval, reverse=True)
    L = [v[0] for v in L]

    FP = 0.0
    TP = 0.0
    R = list()
    fprev = -1e10
    i = 0

    while i<M:
        if tL[i,0]!= fprev:
            R.append((float(FP)/float(N),float(TP)/float(P)))
            fprev = tL[i,0]
        if L[i] < P :
            TP +=1.0
        else:
            FP +=1.0
        i +=1
    R.append((float(FP)/float(N),float(TP)/float(P)))
    
    X = np.array([v[0] for v in R])
    Y = np.array([v[1] for v in R])
    plt.figure()
    plt.plot(X, Y, color='blue', linewidth=1)
    plt.show()
    
punctuations=['?',':','!','.',',',';','-','_']
stopwords = ['a', 'of', 'in', 'at', 'on', 'the']

path = 'ag_news_csv/train.csv'
df = pd.read_csv(path, sep=',', names = ['class', 'title', 'body'])
list_texts = [ word_tokenize(text.replace('\\',' ')) for text in df['body'] ]


for i, sentence in enumerate(list_texts):
    temp = list()
    for word in sentence:
        if word not in punctuations:
            temp.append(word)
    list_texts[i] = temp
for  i, sentence in enumerate(list_texts):
    temp = list()
    for word in sentence:
        if word not in stopwords:
            temp.append(word)
    list_texts[i] = temp
    
wordnet_lemmatizer = WordNetLemmatizer() 
for i, sentence in enumerate(list_texts):
    temp = list()
    for word in sentence:
        temp.append(wordnet_lemmatizer.lemmatize(word, pos = "v"))
    list_texts[i] = temp

allterms = []
for ls in list_texts:
    allterms += ls
counter = collections.Counter(allterms)

diffterms = list(set(allterms))

hashterms = {}
for i, word in enumerate(diffterms):
    hashterms.update({word:i})
    
M = len(hashterms)
N = len(list_texts)

irow = []
jcol = []
data = []
for j, text in enumerate(list_texts):
    for word in text:
        irow.append(hashterms[word])
        jcol.append(j)
        data.append(float(counter[word]))


compress_matrix = coo_matrix((data, (irow, jcol)), shape=(M, N))

CA = compress_matrix.tocsc()

U, S, V = svds(CA, 300)

U_k = csc_matrix(U[:,0:300].T)
DenseRep = np.array(((U_k.dot(CA)).T).todense())

DataLabels = df['class']

del CA
del list_texts
del df

c1mask = (DataLabels == 1)
c2mask = (DataLabels == 2)

# Normalize your Data # 
NData = np.asmatrix(Normalization(DenseRep))

Class1 = NData[c1mask,:]
Class2 = NData[c2mask,:]

alpha = 0.05
M1, c1 , cov1 = MahalonobisDetection(Class1, alpha)
M2, c2 , cov2 = MahalonobisDetection(Class2, alpha)
Class1 = Class1[(M1<c1),:]
Class2 = Class2[(M2<c2),:] 

Classification1, Classification2, weights =  LinearRegression(Class1, Class2)
Confusion_Matrix(Classification1,Classification2)



P,_ = Class1.shape
N,_ = Class2.shape
roc_curve(Classification1,Classification2, P,N )
