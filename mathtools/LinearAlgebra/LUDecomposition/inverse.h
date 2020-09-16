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

#ifndef INVERSE_FILE
#define INVERSE_FILE


// Return structure of LUP
struct LUPResult{
    double *LU; //Storage matrix
    int *perm;
};


/*
 * LUP
 * 
 * @param arr square array in flat representation
 * 
 * @return LUResult
 * 
 */
struct LUPResult LUP(double *arr, int n);

/*
 * LUP_Solve
 * 
 * @param LUResult
 * 
 * @param b double array
 * 
 * @oaram n int
 * 
 * @return LUResult
 * 
 */
double* LUP_Solve(struct LUPResult res, double *b, int n);

/*
 * inverse
 * 
 * @param arr to be inverted
 * 
 * @oaram n int
 * 
 * @return X inverse
 * 
 */
double* inverse(double *arr, int n);

#endif