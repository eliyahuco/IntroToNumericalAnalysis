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


def lagrange_interpolation(x_i, f, x, n = len(x_i)):
    """
    This function creates the Lagrange interpolation function
    :param x_i: the x values of the interpolation points
    :param f: the f values of the interpolation points
    :param x: the x value to interpolate
    :param n: the number of interpolation points
    :return: the value of the interpolation function at x
    """
    # initialize the interpolation function
    L = 0
    for i in range(n):
        # initialize the Lagrange polynomial
        l = 1
        for j in range(n):
            # calculate the Lagrange polynomial
            if i != j:
                l = l * (x - x_i[j]) / (x_i[i] - x_i[j])
        # calculate the interpolation function
        L = L + f[i] * l
    return L
print(lagrange_interpolation(x_i, f, 2.5))
x = np.linspace(-1, 4, 1000)
y = lagrange_interpolation(x_i, f, x, n)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Lagrange interpolation", color='r')
plt.scatter(x_i, f, label="Interpolation points", color='b')
plt.xlabel('x', fontsize=14, fontweight='bold')
plt.ylabel('f(x)', fontsize=14, fontweight='bold')
plt.title('Lagrange interpolation', fontsize=16, fontweight='bold')
plt.legend()
plt.grid()
plt.show()




def main():
    pass


if __name__ == '__main__':
    main()