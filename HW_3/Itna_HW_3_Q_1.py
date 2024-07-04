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

import numpy as np

def craete_lagrange_polynomial(x_i, x, i, n = len(x_i)):
    """
    This function creates the Lagrange polynomial
    :param x_i: the x values of the interpolation points
    :param x: the x value to interpolate
    :param i: the index of the interpolation point
    :param n: the number of interpolation points
    :return: the value of the Lagrange polynomial at x
    """
    len(x_i)
    l_x = 1 # initialize the Lagrange polynomial
    for j in range(n):
        # calculate the Lagrange polynomial
        if i != j:
            l_x = l_x * (x - x_i[j]) / (x_i[i] - x_i[j])
    return l_x
for i in range(n):
    for x in x_i:
        print(f"The value of the Lagrange polynomial at x = {x} is and i = {i} is: {craete_lagrange_polynomial(x_i, x, i)}")
 def lagrange_interpolation_with_lagrange_polynomial(x_i, f, x, n = len(x_i)):
    """
    This function creates the Lagrange interpolation function
    :param x_i: the x values of the interpolation points
    :param f: the f values of the interpolation points
    :param x: the x value to interpolate
    :param n: the number of interpolation points
    :return: the value of the interpolation function at x
    """
    n = len(x_i)
    L_x = 0 # initialize the interpolation function
    for i in range(n):
        l_x = craete_lagrange_polynomial(x_i, x, i) # initialize the Lagrange polynomial
        # calculate the interpolation function
        L_x = L_x + f[i] * l_x
    return L_x

def lagrange_interpolation(x_i, f, x, n = len(x_i)):
    """
    This function creates the Lagrange interpolation function
    :param x_i: the x values of the interpolation points
    :param f: the f values of the interpolation points
    :param x: the x value to interpolate
    :param n: the number of interpolation points
    :return: the value of the interpolation function at x
    """
    n = len(x_i)
    L_x = 0 # initialize the interpolation function
    for i in range(n):
        l_x = 1 # initialize the Lagrange polynomial
        for j in range(n):
            # calculate the Lagrange polynomial
            if i != j:
                l_x = l_x * (x - x_i[j]) / (x_i[i] - x_i[j])
        # calculate the interpolation function
        L_x = L_x + f[i] * l
    return L_x




def main():
    pass


if __name__ == '__main__':
    main()