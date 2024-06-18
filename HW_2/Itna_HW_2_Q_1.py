"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_2 Question 1 in the course intro to numerical analysis
the objective of this script is to solve a system of linear equations using the fllowing methods: gauss elimination, LU decomposition, gauss-seidel
the script will give the user the option to choose the method he wants to use and will print the solution of the system of linear equations
the order of numbers of operations will be printed as well
the linear equations in matrix form are:
⎡ 3  -3  2 -4 ⎤ ⎡x₁⎤   ⎡ 7.9 ⎤
⎢-2  -1  3 -1 ⎢ ⎢x₂⎢   ⎢-12.5⎥
⎢ 5  -2 -3  2 ⎢ ⎢x₃⎢ = ⎢  18 ⎥
⎣-2   4  1  2 ⎦ ⎣x₄⎦   ⎣ -8.1⎦
"""

from tabulate import tabulate
import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve
import time

# Given parameters
A = np.array([[3, -3, 2, -4], [-2, -1, 3, -1], [5, -2, -3, 2], [-2, 4, 1, 2]])
b = np.array([7.9, -12.5, 18, -8.1])
n = len(b)

# Gauss elimination