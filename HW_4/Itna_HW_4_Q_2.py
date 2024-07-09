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

def eight_tirds_simpson_rule_integration(f, a, b, n = 1):
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
    h = (b - a) / 3

    if n == 1:
        return ((3/8)*h) * (f(a) + 3 * f(a +h) + 3 * f(a + 2*h) + f(b))
    else:
        for i in range(n ):
            a = x[i]
            b = x[i + 1]
            c = (a + b) / 2
            h = (b - a) / 3
            integral += ((3/8)*h) * (f(a) + 3 * f(a +h) + 3 * f(a + 2*h) + f(b))
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
    h = b - a
    r[0, 0] = 0.5 * h * (f(a) + f(b))
    for i in range(1, n):
        h = h / 2
        sum = 0
        for k in range(1, 2 ** i, 2):
            sum += f(a + k * h)
        r[i, 0] = 0.5 * r[i - 1, 0] + sum * h
        for j in range(1, i + 1):
            r[i, j] = r[i, j - 1] + (r[i, j - 1] - r[i - 1, j - 1]) / (4 ** j - 1)
    return r[n - 1, n - 1]

dict_of_weights_for_quad = {2 :([-0.57735026, 0.57735026], [1, 1]),
                            3 :([-0.77459667, 0, 0.77459667], [0.55555556, 0.88888889, 0.55555556]),
                            4 :([-0.86113631, -0.33998104, 0.33998104, 0.86113631], [0.34785485, 0.65214515, 0.65214515, 0.34785485]),
                            5 :([-0.90617985, -0.53846931, 0, 0.53846931, 0.90617985], [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]),
                            6 :([-0.93246951, -0.66120939, -0.23861918, 0.23861918, 0.66120939, 0.93246951], [0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449]),
                            7 :([-0.94910791, -0.74153119, -0.40584515, 0, 0.40584515, 0.74153119, 0.94910791], [0.12948497, 0.27970540, 0.38183005, 0.41795918, 0.38183005, 0.27970540, 0.12948497]),
                            8 :([-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648, 0.96028986], [0.10122854, 0.22238103, 0.31370665, 0.36268378, 0.36268378, 0.31370665, 0.22238103, 0.10122854]),
                            9 :([-0.96816024, -0.83603111, -0.61337143, -0.32425342, 0, 0.32425342, 0.61337143, 0.83603111, 0.96816024], [0.08127439, 0.18064816, 0.26061070, 0.31234708, 0.33023936, 0.31234708, 0.26061070, 0.18064816, 0.08127439]),
                            10 :([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434, 0.14887434, 0.43339539, 0.67940957, 0.86506337, 0.97390653], [0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422, 0.26926672, 0.21908636, 0.14945135, 0.06667134])}


def gauss_quadrature(f, a, b, n = 2):
    """
    This function calculates the integral of a function using the Gauss quadrature
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    dict_of_weights_for_quad = {2: ([-0.57735026, 0.57735026], [1, 1]),
                                3: ([-0.77459667, 0, 0.77459667], [0.55555556, 0.88888889, 0.55555556]),
                                4: ([-0.86113631, -0.33998104, 0.33998104, 0.86113631],[0.34785485, 0.65214515, 0.65214515, 0.34785485]),
                                5: ([-0.90617985, -0.53846931, 0, 0.53846931, 0.90617985],
                                    [0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]),
                                6: ([-0.93246951, -0.66120939, -0.23861918, 0.23861918, 0.66120939, 0.93246951],
                                    [0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449]),
                                7: ([-0.94910791, -0.74153119, -0.40584515, 0, 0.40584515, 0.74153119, 0.94910791],
                                    [0.12948497, 0.27970540, 0.38183005, 0.41795918, 0.38183005, 0.27970540,
                                     0.12948497]),
                                8: (
                                [-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648,
                                 0.96028986],
                                [0.10122854, 0.22238103, 0.31370665, 0.36268378, 0.36268378, 0.31370665, 0.22238103,
                                 0.10122854]),
                                9: ([-0.96816024, -0.83603111, -0.61337143, -0.32425342, 0, 0.32425342, 0.61337143,
                                     0.83603111, 0.96816024],
                                    [0.08127439, 0.18064816, 0.26061070, 0.31234708, 0.33023936, 0.31234708, 0.26061070,
                                     0.18064816, 0.08127439]),
                                10: ([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434, 0.14887434,
                                      0.43339539, 0.67940957, 0.86506337, 0.97390653],
                                     [0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422,
                                      0.26926672, 0.21908636, 0.14945135, 0.06667134])}
    if n >10:
        x,w = sp.roots_legendre(n)
    else:
        x, w = dict_of_weights_for_quad[n]
    integral = 0
    for i in range(n):
        integral += w[i] * f(0.5 * (b - a) * x[i] + 0.5 * (a + b))
    return 0.5 * (b - a) * integral


# compare the results of the accuracy of the integration methods and with the analytical solution
def main():
    """
    The main function of the script
    :return: prints the results of the integration methods
    """
    # Given parameters
    f = lambda x: x * np.exp(2 * x)
    a = 0
    b = 4
    n = 50
    accuracy = 10 ** -7
    integrate = lambda x: x * np.exp(2 * x) / 2 - np.exp(2 * x) / 4
    integrate = integrate(4) - integrate(0)
    intervals_dict = {}
    print('\n' + '#' * 100)
    print(f'the analytical solution of the integral: {integrate}')
    print('\n' + '#' * 100)
    print('Trapezoidal rule:')
    n = 100000
    while True:
        n += 100000
        integral = na_tools.trapezoidal_rule_integration(f, a, b, n)
        if abs(integral - integrate) < accuracy:
            break
    print(f"trapezoidal rule integration: {integral}, for accuracy required we need {n} intervals")
    n_trapz = n
    intervals_dict['trapezoidal rule'] = n_trapz
    print('\n' + '#' * 100)
    print('Simpson third rule:')
    n = 1
    while True:
        n += 1
        integral = simpson_third_rule_integration(f, a, b, n)
        if abs(integral - integrate) < accuracy:
            break
    print(f"simpson third rule integration: {integral}, for accuracy required we need {n} intervals")
    n_simpson_third = n
    intervals_dict['simpson third rule'] = n_simpson_third
    print('\n' + '#' * 100)
    print('Eight thirds simpson rule:')
    n = 1
    while True:
        n += 1
        integral = eight_tirds_simpson_rule_integration(f, a, b, n)
        if abs(integral - integrate) < accuracy:
            break
    print(f"eight thirds simpson rule integration: {integral}, for accuracy required we need {n} intervals")
    n_eight_thirds = n
    intervals_dict['eight thirds simpson rule'] = n_eight_thirds
    print('\n' + '#' * 100)
    print('Romberg rule:')
    n = 1
    while True:
        n += 1
        integral = romberg_integration(f, a, b, n)
        if abs(integral - integrate) < accuracy:
            break
    print(f"romberg rule integration: {integral}, for accuracy required we need {n} intervals")
    n_romberg = n
    intervals_dict['romberg rule'] = n_romberg
    print('\n' + '#' * 100)
    print('Gauss quadrature:')
    n = 2
    while True:
        n += 1
        integral = gauss_quadrature(f, a, b, n)
        if abs(integral - integrate) < accuracy:
            break
    print(f"gauss quadrature integration: {integral}, for accuracy required we need {n} intervals")
    n_gauss = n
    intervals_dict['gauss quadrature'] = n_gauss
    print('\n' + '#' * 100)

    intervals_dict = {k: v for k, v in sorted(intervals_dict.items(), key=lambda item: item[1])}
    print(f'in conclusion, the analytical solution of the integral is: {integrate}')
    print(f'the number of intervals required for each method to reach the accuracy required is:\n')
    for key, value in intervals_dict.items():
        print(f'{key}: {value} intervals')
    print(f'\nthe best method is: {list(intervals_dict.keys())[0]}')
    print(f'the worst method is: {list(intervals_dict.keys())[-1]}')
    print('\n' + '#' * 100)


    # plot the results
    intervals_dict.pop('trapezoidal rule')
    print(f'we will plot the number of intervals required for each method to reach the accuracy required')
    print(f'not including the trapezoidal rule method')
    print(f'because the number of intervals required for the trapezoidal rule method is very high, and not in the same order of magnitude as the other methods')
    print('\n' + '#' * 100)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8), sharey=True, sharex=True)
    axs.bar(intervals_dict.keys(), intervals_dict.values(), color='b')
    axs.set_title('Number of intervals required for each method to reach the accuracy required', fontweight='bold', fontsize=14)
    axs.set_xlabel('Integration methods', fontweight='bold', fontsize=12)
    axs.set_ylabel('Number of intervals', fontweight='bold', fontsize=12)
    axs.grid()
    plt.show()

    print("\n")
    print("the script has finished running")
    print("thank you for using the script")







if __name__ == '__main__':
    main()