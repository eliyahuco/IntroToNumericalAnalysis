"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_4 for the course intro to numerical analysis
the objective of this script is to calaculate integrals using the following methods:

1) trapezoidal rule
2) extrapolated richardson's rule

accuracy required: 10^-7

to calculate the integral of the function f(x) = e^(-x^2) from 0 to 2 using the methods 1 with 20 intervals equal in size.
also calculate the integral using method 2 with iterations until the accuracy is reached
we will compare the results of the accuracy of the integration methods and with the analytical solution

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""
import math

# Libraries in use
import numpy as np
import scipy.special as sp
from scipy import integrate
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
    x = np.linspace(a, b, n+1)
    integral = 0
    h = (b - a)
    if n == 1:
        return 0.5 * h * (f(a) + f(b))
    else:
        for i in range(n):
            a = x[i]
            b = x[i + 1]
            h = (b - a)
            integral += 0.5 * h * (f(a) + f(b))
    return integral

def extrapolated_richardson_rule_integration(f,a,b,accuracy = 10**-7):
    """
    This function calculates the integral of a function using the extrapolated richardson rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param accuracy: the required accuracy
    :return: the value of the integral
    """
    n = 1
    k = 1
    integral = trapezoidal_rule_integration(f, a, b, n)
    integral_ = trapezoidal_rule_integration(f, a, b, 2*n)

    b_h =((4**k)*integral_ - integral)/((4**k)-1)

    while abs(integral - integral_) > accuracy:
        n = 2*n
        integral = integral_
        integral_ = trapezoidal_rule_integration(f, a, b, 2*n)
        b_h =((4**k)*integral_ - integral)/((4**k)-1)
        k = k + 1
    print(f'\nacutacy requires after {k} iterations')

    return b_h

def main():
    """
    The main function of the script
    :return: plots the interpolation functions
    """
    # given parameters
    f = lambda x: np.exp(-x**2)
    a = 0
    b = 2
    accuracy = 10**-7

    # change the limits of the integral to (0,1)


    # calculate the integral using the trapezoidal rule

    print('\n' + '#' * 100)
    print('Trapesoidal rule:')
    integral_trapezoidal = trapezoidal_rule_integration(f, a, b, 20)
    print(f'\nThe integral of the function f(x) = e^(-x^2) from 0 to 2 using the trapezoidal rule with 20 intervals is: {np.round(integral_trapezoidal,7)}')

    # calculate the integral using the extrapolated richardson rule

    print('\n' + '#' * 100)
    print('Extrapolated richardson rule:')
    integral_richardson = extrapolated_richardson_rule_integration(f, a, b, accuracy)
    print(f'\nThe integral of the function f(x) = e^(-x^2) from 0 to 2 using the extrapolated richardson rule with accuracy 10^-7 is: {np.round(integral_richardson,7)}')

    # calculate the integral using the the real value of the integral
    print('\n' + '#' * 100)
    print('the analytical solution of the integral:')
    integral_real = (math.sqrt(math.pi) / 2) * (sp.erf(2) - sp.erf(0))
    print(f'\nThe real value of the integral of the function f(x) = e^(-x^2) from 0 to 2 is: {integral_real}')
    print('and it was calculated using the error function from the scipy library and the square root of pi')

    print('\n' + '#' * 100)
    print('Errors analysis:')
    print(f'we wil round the errors to 7 decimal points acording to the accuracy required  {accuracy}\n')
    print(f'The absolute error of the trapezoidal rule is: {np.round(abs(integral_real - integral_trapezoidal),7)}')
    print(f'The relative error of the trapezoidal rule is: {np.round(abs((integral_real - integral_trapezoidal) / integral_real),7)}')
    print(f'The relative error in percentage of the trapezoidal rule is: {np.round(abs((integral_real - integral_trapezoidal) / integral_real) * 100,7)} %')
    print('' + '-' * 100)
    print(f'The absolute error of the extrapolated richardson rule is: {np.round(abs(integral_real - integral_richardson),7)}')
    print(f'The relative error of the extrapolated richardson rule is: {np.round(abs((integral_real - integral_richardson) / integral_real),7)}')
    print(f'The relative error in percentage of the extrapolated richardson rule is: {np.round(abs((integral_real - integral_richardson) / integral_real) * 100,7)} %')
    print('\n' + '#' * 100)

    # plot the function
    x = np.linspace(a, b, 1000)
    f_x = f(x)
    plt.plot(x, f_x, label="f(x) = e^(-x^2)", color='g')
    plt.fill_between(x, f_x, color='g', alpha=0.2)

    plt.legend(fontsize=10, loc='upper right')
    plt.xlabel("x", fontweight='bold', fontsize=14)
    plt.ylabel("f(x)", fontweight='bold', fontsize=14)
    plt.title("The function f(x) = e^(-x^2)", fontweight='bold', fontsize=14)
    plt.grid()
    plt.show()

    print("\n")
    print("the script has finished running")
    print("thank you for using the script")

    return
if __name__ == '__main__':
    main()

