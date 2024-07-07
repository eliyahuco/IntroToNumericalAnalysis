"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 2 in HW_4 for the course intro to numerical analysis
the objective of this script is to calaculate integrals using the following methods:

1) trapezoidal rule
2) simpson's rule
3) romberg's rule
4) gauss quadrature

accuracy required: 10^-7

we wil calculate the integral of the function f(x) = x*e^(2x) from 0 to 4 using the methods 1, 3, 4, 5
we will find the number of intervals required for each method to reach the accuracy required
we will compare the results of the accuracy of the integration methods and with the analytical solution

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools
import math
import scipy.special as sp
from scipy import integrate

def simpson_third_rule_integration(f, a, b, n = 1):
    """
    This function calculates the integral of a function using the Simpson's rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    x = np.linspace(a, b, n+1)
    integral = 0
    h = (b - a) / 2
    c = (a + b) / 2
    if n == 1:
        return h / 3 * (f(a) + 4 * f(c) + f(b))
    else:
        for i in range(n ):
            a = x[i]
            b = x[i + 1]
            c = (a + b) / 2
            h = (b - a) / 2
            integral += (h / 3) * (f(a) + 4 * f(c) + f(b))
    return integral

def romberg_integration(f, a, b, n = 1):
    """
    This function calculates the integral of a function using the Romberg's rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    r = np.zeros((n, n))
    h = (b - a)
    r[0, 0] = h / 2 * (f(a) + f(b))
    for i in range(1, n):
        h /= 2
        r[i, 0] = r[i - 1, 0] / 2 + h * sum([f(a + (2 * k - 1) * h) for k in range(1, 2 ** i + 1)])
        for j in range(1, i + 1):
            r[i, j] = r[i, j - 1] + (r[i, j - 1] - r[i - 1, j - 1]) / (4 ** j - 1)
    return r[n - 1, n - 1]


f = lambda x: x * np.exp(2 * x)

integrate = lambda x: x*np.exp(2*x)/2 - np.exp(2*x)/4
integrate = integrate(4) - integrate(0)
print(integrate)
a = 0
b = 4
n = 0
accuracy = 10**-4
while True:
    n += 1
    integral = simpson_third_rule_integration(f, a, b, n)
    if abs(integral - integrate) < accuracy:
        break
print(f"simpson_third_rule_integration: {integral}, n: {n}")
n = 0
while True:
    n += 1
    integral = na_tools.trapezoidal_rule_integration(f, a, b, n)
    if abs(integral - integrate) < accuracy:
        break
print(f"trapezoidal_rule_integration: {integral}, n: {n}")
