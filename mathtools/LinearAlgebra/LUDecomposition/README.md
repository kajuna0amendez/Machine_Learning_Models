# Here the routines for Lower Upper decomposition
We implemented these routines in Fortran given their capabilities for numerical applications

## LU Decomposition 

Here is an implementation using Crout’s algorithm, then for storage it returns 
the LU stored in the original matrix and indx the permutations done for getting
the decomposition. And although the result is a permutation matrix of the original one,  it can be recovered by using the indx vector in a backward fashion.

## Backward and Fordward procedures 

These procedures can be used to solve linear systems like the ones that happen
in solving the inverse matrix problem
