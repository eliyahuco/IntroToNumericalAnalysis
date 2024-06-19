"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_2 Question 1 in the course intro to numerical analysis
the objective of this script is to finf the inverse of a matrix using the LU decomposition method
and show the result to the user
the matrix is:
⎡ 4  8  4  0 ⎤
⎢ 1  4  7  2 ⎢
⎢ 1  5  4 -3 ⎢
⎣ 1  3  0 -2 ⎦
"""
# Libraries in use
from tabulate import tabulate
import numpy as np
import Itna_HW_2_Q_1 as hw1
from Itna_HW_2_Q_1 import lu_decomposition_steps as lud
from Itna_HW_2_Q_1 import rankin_matrix as rm



