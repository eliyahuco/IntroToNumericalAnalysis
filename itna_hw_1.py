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



def bisection_search_first_guess(x_start,x_end,polynom, x_segment,epsilon):
     """
    Brief description of the function.

    Longer description of the function that can span multiple lines.
    Explain what the function does and provide any additional details that might
    be necessary to understand its behavior.

    Args:
        x_start (float): begining of the segment.
        x_end (float): end of the segment.
        polynom : Additional positional arguments.
        x_segment : Additional keyword arguments.
        epsilon (float): Additional keyword arguments.

    Keyword Args:
        kwarg1 (bool): Description of keyword argument 1 (if using kwargs).
        kwarg2 (float): Description of keyword argument 2 (if using kwargs).

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: Description of why the exception might be raised.
        TypeError: Description of why the exception might be raised.

    Examples:
        Example usage of the function.

        >>> result = function_name(10, 'example', kwarg1=True)
        >>> print(result)
        True

    Notes:
        Any additional notes or comments about the function.
    """
    a = x_start
    b = x_end
    u = polynom(a)
    v = polynom(b)
    if u*v >= 0:
        for i in x_segment:
            multiplicaion_value = polynom(i)*v
            if multiplicaion_value < 0:
                a = i
                break
        u = polynom(a)
        if  u*v > 0:
            print("Bisection method is not applicabale here")
    print(polynom(a))
    c = 0.5*(a+b)
    w = polynom(c)
    while abs(b-a) > epsilon:
        if w != 0:
            if u*w < 0:
                b = c
            else:
                a = c
            c = 0.5*(a+b)
    print(abs(b-a) < 10**(-4) )
    print(abs(polynom(c)) < 10**(-4))
    print(a)
    print(b)
    print(c)
    print(abs(c-a) < 10**(-4) )
    print(abs(c - b) < 10 ** (-4))
    print(polynom(-3.79123))








if __name__ == "__main__":
    bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd )







