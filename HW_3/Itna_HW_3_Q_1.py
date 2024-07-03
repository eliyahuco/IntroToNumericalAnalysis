"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_3 for the course intro to numerical analysis
the objective of this script is to create interpolation functions using the Lagrange method
the script will create the Lagrange interpolation function and will plot the function
first we will create the Lagrange interpolation in order four, then we will create two interpolation functions in second order
the points for the interpolation are:
xᵢ = [-1, 0, 2, 3, 4]
f(xᵢ) =[0,1,9,25,67]

---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Given parameters
x_i = np.array([-1, 0, 2, 3, 4])
f = np.array([0, 1, 9, 25, 67])
n = len(x_i)

def lagrange_interpolation(x, x_i, f):
    """
    This function creates the Lagrange interpolation function
    :param x: the x values
    :param x_i: the x values of the points
    :param f: the f values of the points
    :return: the Lagrange interpolation function
    """
    L = np.zeros(n)
    for i in range(n):
        L_i = 1
        for j in range(n):
            if i != j:
                L_i *= (x - x_i[j]) / (x_i[i] - x_i[j])
        L[i] = L_i
    return np.dot(L, f)







def main():
    pass


if __name__ == '__main__':
    main()