"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 in the course intro to numerical analysis
The objective of the script is find roots of a given polynom with several methods
---------------------------------------------------------------------------------

Given polynom:  f(x) = X⁴ + 2x³ -7x² + 3
segment in X axis [-5,3]

"""
#Labraries in use
import numpy as np
import math
import matplotlib as plt

# constant parameters
precision_requierd = 10**(-4) # ε
a = -5 #begining of segment
b = 3 #end of segment
x_line = np.linspace(a,b,abs(b-a)*10**4)
polynom_coefficients = [1,2,-7,0,3]
given_polynom = np.poly1d(polynom_coefficients)


# function to calculate the value of a polynom in a given x
def derivative_polynom_in_x(polynom,x_0,epsilon):
    """
    Returns the derivative of a given polynomial.

    Args:
        polynom (callable): The polynomial function to differentiate.

    Returns:
        callable: The derivative of the polynomial function.

    """
    a = x_0
    b = x_0 + epsilon
    u = polynom(a)
    v = polynom(b)
    return (v-u)/epsilon




# bisection method for finding the first root of a polynom
def bisection_search_first_guess(x_start,x_end,polynom, x_segment,epsilon):
    """
    Finds a root of the polynomial within a specified segment using the bisection method.

    Args:
        x_start (float): The starting value of the interval.
        x_end (float): The ending value of the interval.
        polynom (callable): The polynomial function to find the root of.
        x_segment (float): The segment of the x-axis to consider.
        epsilon (float): The acceptable error margin for the root.

    Returns:
        float: The approximated root of the polynomial within the given interval.

    Raises:
        ValueError: If the interval does not contain a root or if inputs are invalid.


     """
    a = x_start
    b = x_end
    u = polynom(a)
    v = polynom(b)
    if u == 0:
        return a
    elif v == 0:
        return b
    if u*v >= 0:
        for i in x_segment:
            multiplicaion_value = polynom(i)*v
            if multiplicaion_value < 0:
                a = i
                break
        u = polynom(a)
    if  u*v > 0:
        print("Bisection method is not applicabale here")
        return None
    while abs(b-a) > epsilon:
        c = 0.5 * (a + b)
        w = polynom(c)
        if w == 0:
            return c
        if u*w < 0:
            b = c
            v = w
        else:
            a = c
            u = w


    print(abs(b-a) < 10**(-4) )
    print(abs(polynom(c)))
    print(a)
    print(b)
    print(c)
    print(abs(c-a) < 10**(-4) )
    print(abs(b - c) < 10 ** (-4))
    print(polynom(c))
    print(abs(c)  < 10 ** (-4))
    return c

#newton-raphson method for finding a root of a polynom
def newton_raphson_method(polynom, x_0=0, epsilon=10**(-4)):
    """
    Finds a root of the polynomial using the Newton-Raphson method.

    Args:
        polynom (callable): The polynomial function to find the root of.
        x_0 (float): The initial guess for the root.
        epsilon (float): The acceptable error margin for the root.

    Returns:
        float: The approximated root of the polynomial.

    Raises:
        ValueError: If the method does not converge or if inputs are invalid.

    """
    x = x_0
    x_1 = x - polynom(x) / derivative_polynom_in_x(polynom,x,epsilon)
    while abs(x_1 - x) > epsilon:
        x = x_1
        x_1 = x - polynom(x) / derivative_polynom_in_x(polynom,x,epsilon)
    return x_1


def synthetic_division_step(polynom, root):
    """
    Performs a single step of the synthetic division method.

    Args:
        polynom (callable): The polynomial function to divide.
        root (float): The root to divide the polynomial by.

    Returns:
        tuple: The coefficients of the quotient and the remainder.

    """
    n = len(polynom.coefficients) - 1
    quotient = np.zeros(n)
    remainder = polynom.coefficients[-1]
    for i in range(n - 1, -1, -1):
        quotient[i] = remainder
        remainder = polynom.coefficients[i] + root * remainder
    return quotient, remainder
#synthetic division method for finding all roots of a polynom
def synthetic_division(polynom, first_root, epsilon):
    """
    Finds all roots of the polynomial using the synthetic division method.

    Args:
        polynom (callable): The polynomial function to find the roots of.
        first_root (float): The first root of the polynomial.
        epsilon (float): The acceptable error margin for the roots.

    Returns:
        list: The approximated roots of the polynomial.

    Raises:
        ValueError: If the method does not converge or if inputs are invalid.

    """
    roots = [first_root]
    quotient, remainder = synthetic_division_step(polynom, first_root)
    while len(quotient) > 1:
            root = newton_raphson_method(np.poly1d(quotient), 0, epsilon)
            if (root+10*epsilon or root-epsilon) in roots:
                break
            roots.append(root)
            quotient, remainder = synthetic_division_step(np.poly1d(quotient), root)
    return roots















if __name__ == "__main__":
    print(bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd))
    print(newton_raphson_method(given_polynom,0,precision_requierd))
    print(synthetic_division_step(given_polynom,-0.6180773365624102 ))
    # print(synthetic_division(given_polynom,bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd),precision_requierd))





