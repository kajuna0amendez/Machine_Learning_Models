# -*- coding: utf-8 -*-
#!/usr/bin/env python 
# cython: language_level=2

__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2018"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Open"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.nonecheck(False)    # Deactivate none check
ccpdef Mahalanobis():
    """
    """
    return 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.nonecheck(False)    # Deactivate none check
ccdef float chi_statistics():
    """
    """
    return 1

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
@cython.nonecheck(False)    # Deactivate none check
ccdef z():
    """
    """
    return 1

