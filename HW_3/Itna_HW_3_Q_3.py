"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 3 in HW_3 for the course intro to numerical analysis
the objective of this script is to create interpolation functions using the cubic spline method
the cubic spline method is based on solution of a system of linear tridiagonal matrix equations
the question has two sections:
a) to show a plot of the cubic spline interpolation function for the points:
    (xᵢ,yᵢ) = [(0,4), (2,2), (3,8),(4,10),(7,4),(8,-2)]
    the program will calculate the coefficients of the cubic spline and will plot the interpolation function
    we will show also the two first derivatives of the interpolation function
    we will print the coefficients of the cubic spline
b) to show a plot of the cubic spline interpolation function for the points:
    (xᵢ,yᵢ) = [(3,4), (2,3), (2.5,1), (4,2), (5,3.5), (4,4.5)]
    for this section we will use parametric cubic spline interpolation
    the program will calculate the coefficients of the cubic spline and will plot the interpolation function
    we will show also the two first derivatives of the interpolation function
    we will print the coefficients of the cubic spline
    we will compare the results of the accuracy of the interpolation function
    accuracy required: 10^-4

will use the techniques of natural cubic spline and parametric cubic spline:
natural cubic spline means that the second derivative at the edges is zero
parametric cubic spline means that x and y are themselves functions of a parameter t

we will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import math as math
import numerical_analysis_methods_tools as na_tools

# Given parameters
x_i = np.array([0, 2, 3, 4, 7, 8])
y_i = np.array([4, 2, 8, 10, 4, -2])
n = len(x_i)

def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    This function solves a system of linear tridiagonal matrix equations
    :param a: the lower diagonal of the matrix
    :param b: the main diagonal of the matrix
    :param c: the upper diagonal of the matrix
    :param d: the right side of the equation
    :return: the solution of the system
    """
    n = len(d)
    # forward elimination
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]
    # backward substitution
    x = np.zeros(n)
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x



def natural_cubic_spline(x_i, y_i):
    """
    This function creates the natural cubic spline interpolation function and gives the coefficients of the cubic spline
    :param x_i: the x values of the interpolation points
    :param y_i: the y values of the interpolation points
    :return: the coefficients of the cubic spline, the x values of the interpolation function, the y values of the interpolation function
    """
    n = len(x_i)
    h = np.zeros(n - 1)
    for i in range(n - 1):
        h[i] = x_i[i + 1] - x_i[i]
    a = np.zeros(n)
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)
    for i in range(1, n - 1):
        a[i] = 2 * (h[i - 1] + h[i])
        b[i] = h[i]
        c[i] = h[i - 1]
        d[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])
    # solve the system of linear tridiagonal matrix equations
    c = tridiagonal_matrix_algorithm(c, a, b, d)
    for i in range(1, n):
        a[i] = y_i[i - 1]
        b[i] = (y_i[i] - y_i[i - 1]) / h[i - 1] - h[i - 1] * (c[i] + 2 * c[i - 1]) / 3
        c[i] = c[i - 1]
        d[i] = (c[i] - c[i - 1]) / (3 * h[i - 1])
    x = np.array([])
    y = np.array([])
    for i in range(n - 1):
        x_i = np.linspace(x_i[i], x_i[i + 1], 100)
        y_i = a[i] + b[i] * (x_i - x_i[i]) + c[i] * (x_i - x_i[i]) ** 2 + d[i] * (x_i - x_i[i]) ** 3
        x = np.concatenate((x, x_i))
        y = np.concatenate((y, y))
    return a, b, c, d, x, y



def parametric_cubic_spline(x_i, y_i):
    pass

def main():
    """
    The main function of the script
    :return: plots the interpolation functions
    """
    # create the interpolation functions
    a, b, c, d, x, y = natural_cubic_spline(x_i, y_i)
    print(f'the coefficients of the cubic spline are: a = {a}, b = {b}, c = {c}, d = {d}')
    plt.plot(x, y, label="cubic spline interpolation function", color='b')
    plt.scatter(x_i, y_i, label="interpolation points", color='r')
    plt.legend()
    plt.show()

    # create the interpolation functions
    a, b, c, d, x, y = parametric_cubic_spline(x_i, y_i)
    print(f'the coefficients of the cubic spline are: a = {a}, b = {b}, c = {c}, d = {d}')
    plt.plot(x, y, label="cubic spline interpolation function", color='b')
    plt.scatter(x_i, y_i, label="interpolation points", color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()