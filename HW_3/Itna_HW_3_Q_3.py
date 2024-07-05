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
from scipy import linalg

# Given parameters
x_i = np.array([0, 2, 3, 4, 7, 8])
y_i = np.array([4, 2, 8, 10, 4, -2])
n = len(x_i)

import numpy as np
import matplotlib.pyplot as plt


def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    Solves a tridiagonal system of linear equations using the Thomas algorithm.

    Parameters:
    a (ndarray): Sub-diagonal coefficients (n-1 elements).
    b (ndarray): Diagonal coefficients (n elements).
    c (ndarray): Super-diagonal coefficients (n-1 elements).
    d (ndarray): Right-hand side vector (n elements).

    Returns:
    x (ndarray): Solution vector (n elements).
    """
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    if b[0] != 0:  # Prevent division by zero
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
    else:
        c_[0] = 0  # or some other suitable value
        d_[0] = 0  # or some other suitable value

    for i in range(1, n - 1):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])
        else:
            c_[i] = 0  # Handle division by zero if necessary

    for i in range(1, n):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])
        else:
            d_[i] = 0  # Handle division by zero if necessary

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def natural_cubic_spline(x_i, y_i):
    """
    Constructs a natural cubic spline interpolation for given data points.

    Parameters:
    x_i (ndarray): x-coordinates of the data points.
    y_i (ndarray): y-coordinates of the data points.

    Returns:
    spline_coeffs (ndarray): Coefficients of the cubic spline for each interval.
    """
    n = len(x_i)
    h = np.diff(x_i)

    # Construct the tridiagonal system
    a = np.zeros(n)
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        a[i] = 2 * (h[i - 1] + h[i])
        b[i - 1] = h[i]
        c[i] = h[i - 1]
        alpha[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])

    # Adjusting arrays for the tridiagonal solver
    A = a[1:n - 1]
    B = b[:n - 2]
    C = c[2:n]
    D = alpha[1:n - 1]

    # Solve the tridiagonal system
    c_sol = tridiagonal_matrix_algorithm(C, A, B, D)

    # Insert the boundary conditions for c
    c = np.zeros(n)
    c[1:n - 1] = c_sol

    # Calculate the b and d coefficients
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y_i[:-1]

    for i in range(n - 1):
        b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    spline_coeffs = np.array([a, b, c[:-1], d]).T
    return spline_coeffs


def evaluate_spline(x, spline_coeffs, xi):
    """
    Evaluates the cubic spline at a given point.

    Parameters:
    x (ndarray): x-coordinates of the data points.
    spline_coeffs (ndarray): Coefficients of the cubic spline for each interval.
    xi (float): x-coordinate at which to evaluate the spline.

    Returns:
    yi (float): Interpolated y-coordinate at xi.
    """
    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(spline_coeffs) - 1)

    dx = xi - x[i]
    a, b, c, d = spline_coeffs[i]

    return a + b * dx + c * dx ** 2 + d * dx ** 3


# Example usage
x = np.array([0, 2, 3, 4, 7, 8,10])
y = np.array([4, 2, 8, 10, 4, -2,3])
spline_coeffs = natural_cubic_spline(x_i, y_i)

# Evaluate the spline at some points
x_eval = np.linspace(0, 8, 100)
y_eval = [evaluate_spline(x, spline_coeffs, xi) for xi in x_eval]

plt.plot(x_eval, y_eval, label="Cubic Spline")
plt.scatter(x_i, y_i, color='red', label="Data Points")
plt.legend()
plt.show()


# def natural_cubic_spline(x_i, y_i):
#     """
#     This function creates the natural cubic spline interpolation function and gives the coefficients of the cubic spline
#     :param x_i: the x values of the interpolation points
#     :param y_i: the y values of the interpolation points
#     :return: the coefficients of the cubic spline, the x values of the interpolation function, the y values of the interpolation function
#     """
#     n = len(x_i)
#     h = np.zeros(n - 1)
#     for i in range(n - 1):
#         h[i] = x_i[i + 1] - x_i[i]
#     a = np.zeros(n)
#     b = np.zeros(n - 1)
#     c = np.zeros(n)
#     d = np.zeros(n - 1)
#     for i in range(1, n - 1):
#         a[i] = 2 * (h[i - 1] + h[i])
#         b[i] = h[i]
#         c[i] = h[i - 1]
#         d[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])
#     # solve the system of linear tridiagonal matrix equations
#     c = tridiagonal_matrix_algorithm(c, a, b, d)
#     for i in range(1, n):
#         a[i] = y_i[i - 1]
#         b[i] = (y_i[i] - y_i[i - 1]) / h[i - 1] - h[i - 1] * (c[i] + 2 * c[i - 1]) / 3
#         c[i] = c[i - 1]
#         d[i] = (c[i] - c[i - 1]) / (3 * h[i - 1])
#     x = np.array([])
#     y = np.array([])
#     for i in range(n - 1):
#         x_i = np.linspace(x_i[i], x_i[i + 1], 100)
#         y_i = a[i] + b[i] * (x_i - x_i[i]) + c[i] * (x_i - x_i[i]) ** 2 + d[i] * (x_i - x_i[i]) ** 3
#         x = np.concatenate((x, x_i))
#         y = np.concatenate((y, y))
#     return a, b, c, d, x, y
#
#

def parametric_cubic_spline(x_i, y_i):
    pass

def main():
    """
    # The main function of the script
    # :return: plots the interpolation functions
    # """
    # # create the interpolation functions
    # a, b, c, d, x, y = natural_cubic_spline(x_i, y_i)
    # print(f'the coefficients of the cubic spline are: a = {a}, b = {b}, c = {c}, d = {d}')
    # plt.plot(x, y, label="cubic spline interpolation function", color='b')
    # plt.scatter(x_i, y_i, label="interpolation points", color='r')
    # plt.legend()
    # plt.show()
    #
    # # create the interpolation functions
    # a, b, c, d, x, y = parametric_cubic_spline(x_i, y_i)
    # print(f'the coefficients of the cubic spline are: a = {a}, b = {b}, c = {c}, d = {d}')
    # plt.plot(x, y, label="cubic spline interpolation function", color='b')
    # plt.scatter(x_i, y_i, label="interpolation points", color='r')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()