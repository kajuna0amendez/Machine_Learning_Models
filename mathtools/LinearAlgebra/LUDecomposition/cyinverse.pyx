import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf


@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef  LUP_Decomposition(double [:,:] B, int n):
    """
    Lower Upper Decomposition
    """
    cdef int[:] pi
    cdef double[:,:] A = np.zeros((n, n))
    cdef double[:,:] L = np.zeros((n, n))
    cdef double[:,:] U = np.zeros((n, n))
    cdef int temp
    # Py_ssize is the index of numpy objects
    cdef Py_ssize_t i, k, j, kp
    cdef double p, ftemp
    # copy array
    for i in range(n):
        for k in range(n):
            A[i,k] = B[i,k] 
    # Generate the Permutation Matrix
    pi = np.arange(0,n, dtype=np.intc)
    
    # Start the pivoting
    for k in range(n):
        p = 0.0
        # Find the Largest Pivote
        for i in range(k,n):
            if abs(A[i,k])>p:
                p = abs(A[i,k])
                kp = i
        # Throw error 
        if p == 0.0:
            printf('Singular Matrix')
        # Exchange permutations
        temp = pi[k]
        pi[k]= pi[kp]
        pi[kp] = temp
        # Exchange Rows
        for i in range(n):
            ftemp = A[k,i]
            A[k,i] = A[kp,i]
            A[kp,i] = ftemp
        # Do the necessary calculations    
        for i in range(k+1,n):
            A[i,k] = A[i,k]/A[k,k]
            for j in range(k+1, n):
                A[i,j] = A[i,j] - A[i,k]*A[k,j]
    # Build the Lower and Upper triangular matrixes
    for i in range(n):
        for k in range(n):
            if i>k:
                L[i,k] = A[i,k]
            else:
                U[i,k] = A[i,k]
                if i == k:
                   L[i,k] = 1.0 
                
    return L, U, pi

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef LUP_Solve(double[:,:] L, double[:,:] U, int[:] pi, double[:] b, int n):
    """
    Lower Upper Solver
    """
    cdef double[:] x = np.zeros(n)
    cdef double[:] y = np.zeros(n)
    cdef double temp
    cdef Py_ssize_t i, j
    
    for i in range(n):
        temp = 0.0
        for j in range(0,i):
            temp +=  L[i,j]*y[j] 
        y[i] = b[pi[i]] - temp
        
    for i in range(n-1,-1,-1):
        temp = 0.0
        for j in range(i,n):
            temp += U[i,j]*x[j]
        x[i] = (y[i] - temp)/U[i,i]
    
    return x

@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef M_Inverse(double[:,:] A, int n):
    """
    Inverse of the Matrix
    """
    cdef int[:] pi
    cdef double[:,:] L = np.zeros((n, n))
    cdef double[:,:] U = np.zeros((n, n))
    cdef np.ndarray[np.float64_t, ndim=2] X = np.zeros((n, n))
    cdef double[:,:] Ident = np.identity(n)
    cdef Py_ssize_t i
    # Get the LUP decomposition for A
    L, U, pi = LUP_Decomposition(A, n)
    for i in range(n):
        X[:,i] = LUP_Solve(L, U, pi, Ident[:,i], n)
    return X

cpdef find_smallest_eigenvalue(np.ndarray[dtype=double, ndim = 2] X, int n):
    """
    Finding smallest eigenvalue
    """
    cdef double[:,:] w = np.zeros((n,n))
    cdef double[:] v = np.zeros(n)
    v, w = np.linalg.eigh(X)
    
    return v[0]
