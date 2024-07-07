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
import math

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools

# Given parameters


# function that changes the limits of the integral to (0,1)
def change_function_limits_to_0_1(f, a, b):
    """
    This function changes the limits of the integral to (0,1)
    :param f: the function to integrate given as a numpy array
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(a, b, abs(b - a) * 1000)
    h = b - a
    etha = (x - a) / h
    f = f(x)
    f_a = f[0]
    f_b = f[-1]
    f_new = f_a*(1-etha) + f_b*etha
    return f_new, h, 0, 1


# function that changes the limits of the integral to (-1,1)

def change_function_limits_to_minus_1_1(f, a, b):
    """
    This function changes the limits of the integral to (-1,1)
    :param f: the function to integrate given as a numpy array
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(a, b, abs(b - a) * 1000)
    h = b - a
    etha = (x - a) / h
    f = f(x)
    f_a = f[0]
    f_b = f[-1]
    f_new = 0.5 * (f_a*(1-etha) + f_b*etha)
    return f_new, h, -1, 1

#  function that changes the limits of function to (a,b) from (0,1)
def change_function_limits_to_a_b_from_0_1(f, a, b):
    """
    This function changes the limits of the integral to (a,b)
    :param f: the function to integrate given as a numpy array
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(0, 1, abs(b - a) * 1000)
    h = b - a
    etha = (x - a) / h
    f_a = f(a)
    f_b = f(b)
    f_new = f_a + (f_b - f_a) * etha
    return f_new, h, a, b

#  function that changes the limits of function to (a,b) from (-1,1)

def change_function_limits_to_a_b_from_minus_1_1(f, a, b):
    """
    This function changes the limits of the integral to (a,b)
    :param f: the function to integrate given as a numpy array
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(-1, 1, abs(b - a) * 1000)
    h = b - a
    etha = (x - a) / h
    f_a = f(a)
    f_b = f(b)
    f_new = 0.5 * (f_a + (f_b - f_a) * etha)
    return f_new, h, a, b

def trapezoidal_rule_integration(f, a, b, n):
    """
    This function calculates the integral of a function using the trapezoidal rule
    :param f: the function to integrate given as a numpy array
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    f = change_function_limits_to_0_1(f, a, b)[0]
    h = change_function_limits_to_0_1(f, a, b)[1]
    x = np.linspace(a, b, n + 1)
    f_x = f(x)
    integral = h * (f_x[0] + 2 * np.sum(f_x[1:n]) + f_x[n]) / 2
    return integral


x = np.linspace(0, 2, 20)
f = lambda x: np.exp(-x ** 2)
print(f)

print(trapezoidal_rule_integration(f, 0, 2, 20))
