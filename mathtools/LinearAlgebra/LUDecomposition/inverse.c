/*
 * __author__ = "Andres Mendez-Vazquez"
 * __copyright__ = "Copyright 2020"
 * __credits__ = ["Andres Mendez-Vazquez"]
 * __license__ = "MIT License"
 * __version__ = "1.0.0"
 * __maintainer__ = "Andres Mendez-Vazquez"
 * __email__ = "kajuna0kajuna@gmail.com"
 * __status__ = "Development"
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "inverse.h"


struct LUPResult LUP(double *arr, int n){

    int i, j, k, kp;
    double p, ftemp;
    double* working = (double *)malloc(sizeof(double)*n*n);
    int* perm = (int *)malloc(sizeof(int)*n);
    struct LUPResult Res;
    


    // Copy array
    for(i = 0 ; i < n ; i++){
        for(j = 0; j < n; i++){
            working[n*i+j] = arr[n*i+j];
        }
    }
    // Generate the permutation matrix
    for(i = 0 ; i < n ; i++){
        perm[i] = i;
    }
    //Start Pivoting
    for(k = 0; k < n; k++){
        p = 0.0;
        for(i = 0 ; i < n ; i++){
            if(fabs(working[n*i + k]) > p){
                p = fabs(working[n*i + k]);
                kp = i;
            }
        }
        if(p == 0.0){
            printf("Singular Matrix");
        }
        // Xor Swap
        perm[k] = perm[k]^perm[kp];
        perm[kp] = perm[k]^perm[kp];
        perm[k] = perm[k]^perm[kp];
        // Exchange Rows
        for(i = 0 ; i < n ; i++){
            ftemp = working[n*k + i];
            working[n*k + i] = working[n*kp + i];
            working[n*kp + i] = ftemp;
        }
        // Do the necessary calculations
        for(i = k+1 ; i < n ; i++){
            working[n*i + k] = working[n*i + k]/working[n*k + k];
            for(j = k+1 ; j < n ; j++){
                working[n*i + j] = working[n*i + j] - working[n*i + k]*working[n*k + j];
            }
        }
    }
    Res.LU = working;
    Res.perm = perm;

    return Res;

}

double* LUP_Solve(struct LUPResult Res, double *b, int n){

    double *temp; 

    temp = (double *) malloc(sizeof(double)*2);

    temp[0] = 1.0;

    return temp;

} 

double* inverse(double *arr, int n){

    double *temp; 

    temp = (double *) malloc(sizeof(double)*2);

    temp[0] = 1.0;

    return temp;

}



