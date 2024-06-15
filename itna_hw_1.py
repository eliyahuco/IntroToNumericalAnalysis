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






if __name__ == "__main__":
    print(bisection_search_first_guess(a,b,given_polynom,x_line,precision_requierd))




