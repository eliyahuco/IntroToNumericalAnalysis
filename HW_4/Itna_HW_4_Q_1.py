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
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(a, b, abs(b - a) * 1000)
    h = b - a
    etha = (x - a) / h
    f_a = f(a)
    f_b = f(b)
    f_new = (1-etha)*f_a + etha*f_b
    a = 0
    b = 1
    return h*f_new, h, a, b

# function that changes the limits of the integral to (-1,1)

def change_function_limits_to_minus_1_1(f, a, b):
    """
    This function changes the limits of the integral to (-1,1)
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :return: the new function and the new limits
    """
    x = np.linspace(a, b, abs(b - a) * 1000)
    a_0 = (a + b) / 2
    a_1 = (b - a) / 2
    t = a_0 + a_1 * x
    f = f(x)
    f_t = f(t)
    f_new = f_t
    a = -1
    b = 1
    return a_*f_new, a_1, a, b


#  function that changes the limits of function to (a,b) from (0,1)
def change_function_limits_to_a_b_from_0_1(f, a, b):
    """
    This function changes the limits of the integral to (a,b)
    :param f: the function to integrate given as a lambda function
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

def trapezoidal_rule_integration(f, a, b, n = 1):
    """
    This function calculates the integral of a function using the trapezoidal rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    x = np.linspace(a, b, n)
    integral = 0
    h = (b - a)
    if n == 1:
        return 0.5 * h * (f(a) + f(b))
    else:

        for i in range(n -1):
            a = x[i]
            b = x[i + 1]
            h = (b - a)
            integral += 0.5 * h * (f(a) + f(b))
    return integral


def simpson_third_rule_integration(f, a, b, n = 1):
    """
    This function calculates the integral of a function using the Simpson's rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    x = np.linspace(a, b, n)
    integral = 0
    h = (b - a) / 2
    c = (a + b) / 2
    if n == 1:
        return h / 3 * (f(a) + 4 * f(c) + f(b))
    else:
        for i in range(n - 1):
            a = x[i]
            b = x[i + 1]
            c = (a + b) / 2
            h = (b - a) / 2
            integral += (h / 3) * (f(a) + 4 * f(c) + f(b))
    return integral




f = lambda x: math.exp(-x**2)


print(trapezoidal_rule_integration(f, 0, 2, 10000))
print(simpson_third_rule_integration(f, 0, 2, 10000))
