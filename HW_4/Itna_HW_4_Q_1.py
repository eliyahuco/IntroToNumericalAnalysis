"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_4 for the course intro to numerical analysis
the objective of this script is to calaculate integrals using the following methods:
1) trapezoidal rule
2) extrapolated richardson's rule
3) simpson's rule
4) romberg's rule
5) gauss quadrature

accuracy required: 10^-7
the assignment has two sections:
a) to calculate the integral of the function f(x) = e^(-x^2) from 0 to 2 using the methods 1 with 20 intervals equal in size and method 2
with iterations until the accuracy is reached
we will compare the results of the accuracy of the integration methods and with the analytical solution

b) to calculate the integral of the function f(x) = x*e^(2x) from 0 to 4 using the methods 1, 3, 4, 5
we will find the number of intervals required for each method to reach the accuracy required
we will compare the results of the accuracy of the integration methods and with the analytical solution

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools

# Given parameters
def trapezoidal_rule_integration(f, a, b, n):
    """
    This function calculates the integral of a function using the trapezoidal rule
    :param f: the function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return integral
