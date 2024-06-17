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
from matplotlib import pyplot as plt

# constant parameters
precision_requierd = 10 ** (-4)  # ε
a = -5  # begining of segment
b = 3  # end of segment
x_line = np.linspace(a, b, abs(b - a) * 10 ** 4)
polynom_coefficients = [1, 2, -7, 0, 3]
given_polynom = np.poly1d(polynom_coefficients)
analytical_solution_wolfram_alpha = [0.5*(1-math.sqrt(5)), 0.5*(1+math.sqrt(5)), 0.5*(-3-math.sqrt(21)), 0.5*(math.sqrt(21)-3)]#analytical solution from wolfram alpha


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


#dvide a polynom by (x-x_0) where x_0 is a root of the polynom
def polynom_devision(polynom, x_0):
    """"
Returns the result of dividing a polynomial by (x-x_0).

    Args:
        polynom (callable): The polynomial function to divide.
        x_0 (float): The root to divide the polynomial by.

    Returns:
        tuple: The quotient and the remainder of the division.

    Raises:
        ValueError: If the root is not a root of the polynomial or if inputs are invalid.
    """

    polynom_coeff = polynom.coefficients
    n = len(polynom_coeff)
    quotient = np.zeros(n-1)
    remainder = polynom_coeff[0]
    for i in range(1,n):
        quotient[i-1] = remainder
        remainder = polynom_coeff[i] + remainder*x_0
    return np.poly1d(quotient), remainder






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
    if u*v >= 0: # if the segment does not contain a root
        for i in x_segment:#
            multiplicaion_value = polynom(i)*v
            if multiplicaion_value < 0:
                a = i
                break
        u = polynom(a)
    if  u*v > 0:# if the segment does not contain a root
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



#synthetic division for polynoms
def synthetic_devision_method(polynom, x_0):
    """
        Finds a root of a polynomial using the synthetic division method.

        Parameters:
        polynom (list of float): Coefficients of the polynomial in descending order.
        x_0 (float): Initial guess for the root.

        Returns:
        float: The root if the method converges, None otherwise.
        """

    x = x_0
    devided_polynom = polynom_devision(polynom, x)
    r_0 = devided_polynom[1]
    r_1 = polynom_devision(devided_polynom[0], x)[1]
    if r_1 == 0:
            print("The method does not converge")
            return None
    x = x - r_0 / r_1

    while abs(x-x_0) > 10**(-4):
        x_0 = x
        devided_polynom = polynom_devision(polynom, x)
        r_0 = devided_polynom[1]
        r_1 = polynom_devision(devided_polynom[0], x)[1]

        if r_1 == 0:
            print("The method does not converge")
            break

        x = x - r_0 / r_1

    return x
#plot the polynom in the segment
def plot_polynom(polynom, x_segment):
    """
    Plots the polynomial function within the specified segment.

    Args:
        polynom (callable): The polynomial function to plot.
        x_segment (tuple): The segment of the x-axis to consider.

    """
    y = polynom(x_segment)
    plt.plot(x_segment, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = X⁴ + 2x³ -7x² + 3')
    plt.grid()
    plt.show()

    #check if the roots are in the required precision and compare between the roots found by the different methods
    def check_roots_error_for_diffrenent_method(analytical_roors , polynom ,methods, epsilon = 10**(-4)):
        """
        Check if the roots are in the required precision and compare between the roots found by the different methods.

        Args:
            analytical_roots (list of float): The analytical roots of the polynomial.
            polynom (callable): The polynomial function to find the roots of.
            epsilon (float): The acceptable error margin for the roots.

        Returns:
            bool: True if the roots are in the required precision and the roots are the same, False otherwise.

        """
        if methods == "bisection":
            one_root = bisection_search_first_guess(a, b, polynom, x_line, epsilon)
            for i in analytical_roots:
                if abs(one_root - i) <
                    print()


        roots_newton_raphson = []
        for i in np.linspace(a, b, abs(b - a) * 10):
            new_root = newton_raphson_method(polynom, i)
            roots_newton_raphson.append(new_root)
        roots_newton_raphson = sorted(list(set(roots_newton_raphson)))
        if sorted(roots_newton_raphson) != sorted(analytical_roots):
            return False
        roots_synthetic_division = []
        for i in np.linspace(a, b, abs(b - a) * 10):
            new_root = synthetic_devision_method(polynom, i)
            roots_synthetic_division.append(new_root)
        roots_synthetic_division = sorted(list(set(roots_synthetic_division)))
        if sorted(roots_synthetic_division) != sorted(analytical_roots):
            return False
        return True

def main():
    """"
    The main function of the script.
    making use of the functions above to find the roots of the given polynom
    find the first root using bisection method and the rest using newton-raphson method and synthetic division method
    comper to the analytical solution and check if the error is less than the required precision
    compare between the roots found by the different methods
    and plot the polynom and the roots on the plot
    """


    first_root_using_bisection  = round(bisection_search_first_guess(a, b, given_polynom, x_line, precision_requierd),5)#find the first root using bisection method
    print(f'The first root using bisection method is: {first_root_using_bisection}')
    roots_newton_raphson = []#find all the roots using newton-raphson method
    for i in np.linspace(a, b, abs(b - a) *10): #seting the initial guess for the newton-raphson method using the segment in the x axis

        new_root = round(newton_raphson_method(given_polynom, i),5)
        roots_newton_raphson.append(new_root)
    roots_newton_raphson = sorted(list(set(roots_newton_raphson)))
    print(f'The roots using Newton-Raphson method are: {roots_newton_raphson}')

    roots_synthetic_division = []#find all the roots using synthetic division method
    for i in np.linspace(a, b, abs(b - a) * 10):#seting the initial guess for the synthetic division method using the segment in the x axis
        new_root = round(synthetic_devision_method(given_polynom, i), 5)
        roots_synthetic_division.append(new_root)
    roots_synthetic_division = sorted(list(set(roots_synthetic_division)))
    print(f'The roots using synthetic division method are: {roots_synthetic_division}')

    print(sorted(given_polynom.roots.round(5)))


    print(f'The analytical solution from wolfram alpa is: {analytical_solution_wolfram_alpha}')
    print(sorted(given_polynom.roots))
    print(sorted(analytical_solution_wolfram_alpha) == sorted(given_polynom.roots))#check if the roots are the same as the analytical solution


    for i in roots_newton_raphson:
        plt.scatter(i, given_polynom(i), color='red')#mark the roots on the plot
    plt.legend([f'Roots: {given_polynom.roots}'])#add legend to the plot
    plot_polynom(given_polynom, x_line)

    plt.show()






if __name__ == '__main__':
    main()

