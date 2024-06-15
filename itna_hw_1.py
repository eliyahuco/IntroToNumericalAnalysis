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

# derivative of the given polynom without using numpy
def derivative_polynom(polynom):
    """
    Calculate the derivative of a given polynomial function.

    Args:
        polynom (callable): The polynomial function to calculate the derivative of.

    Returns:
        callable: The derivative of the given polynomial function.

    Raises:
        ValueError: If the inputs are invalid.
    """
    degree = len(polynom.coefficients) - 1
    derivative_coefficients = []
    for i in range(1,degree+1):
        derivative_coefficients.append(polynom.coefficients[i]*i)
    return np.poly1d(derivative_coefficients)


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

# newton - raphson method for finding the rest roots of a given polynom after given the first root
def newton_raphson_method(polynom,first_root,epsilon):
    """
    Finds the rest roots of the polynomial using the Newton-Raphson method.

    Args:
        polynom (callable): The polynomial function to find the root of.
        first_root (float): The first root of the polynomial.
        epsilon (float): The acceptable error margin for the root.

    Returns:
        list: The approximated roots of the polynomial.

    Raises:
        ValueError: If the inputs are invalid.
    """
    roots = []
    polynom = np.polyder(polynom)
    x = first_root
    while polynom(x) > epsilon:
        x = x - polynom(x) /np.polyder(polynom,x)
        roots.append(x)
    return roots






if __name__ == "__main__":
    newton_raphson_method(given_polynom, bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd ), precision_requierd)
    print("The roots of the given polynom are: ", newton_raphson_method(polynom, bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd ), precision_requierd))




