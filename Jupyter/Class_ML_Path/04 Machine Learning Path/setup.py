# -*- coding: utf-8 -*-
#!/usr/bin/env python 
# cython: language_level=2

__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2018"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Closed"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

"""
Basic setup for the cython file
"""

ext_modules = [
       Extension ('Mahalanobis',
                  sources = ['Mahalanobis.pyx'],
                  libraries=["m"],
                  include_dirs=[numpy.get_include()],
                  extra_compile_args=['-Wno-cpp']
                  )]

setup(
     name = "Mahalanobis",
     cmdclass = {"build_ext": build_ext},
     ext_modules = ext_modules)
