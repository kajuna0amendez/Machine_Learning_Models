#!/usr/bin/env ropython2
# -*- coding: utf-8 -*-
"""
Author: Andres Mendez-Vazquez
Class Implementation of different data structures for
the Dirichlet Clustering Processing
"""

class Matrices:
  """
  Matrix class implementing 
  Container:
    marray - flat matrix

  """
  def __init__(self, Array, rows, cols):
    """
    Constructore asserting the correctitud of the data
    """
    self.__matrix_assert(Array, rows, cols)
    self.__marray = Array
    self.__rows = rows
    self.__cols = cols

  ##############################################################################
  # We have only getters for the these values                                  #
  ##############################################################################

  def getrows(self):
    """
    Get of rows
    """
    return self.__rows

  def getcols(self):
    """
    Get of cols
    """
    return self.__getcols

  ##############################################################################
  ##############################################################################

  def mult(bmatrix, rows, cols):
    """
    Matrix Multiplication
    """
    rcmatrix = self.getcols()
    ccmatrix = cols 
    cmatrix = 
    for i in xrange(rows):
    return cmatrix

  def sum(bmatrix, rows, cols):
    """
    Matrix Addition
    """
    return 1
 
  def __matrix_assert(Array, rows, cols):
    assert isinstance(Array,list), 'It is not a list'
    assert rows*cols == len(Array), 'Dimensions are incorrect'
    return True

  def __vector_assert(Vector, row)


class Vectors():