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
a = -5  # beginning  of segment
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

#check if the roots are in the required precision and compare between the roots found by the different methods
def print_roots_errors_for_diffrenent_method(analytical_roots , polynom ,methods, epsilon = 10**(-4)):
    """
    Finds the roots of a polynomial using different methods and compares them to the analytical solution.


    """
    analytical_roots = sorted(analytical_roots)

    if methods == "bisection":
        print("bisection:")
        one_root = bisection_search_first_guess(a, b, polynom, x_line, epsilon)
        count = 0
        absulute_error = []
        relative_error = []
        for root in analytical_roots:
            if abs(one_root - root) > epsilon:
                count += 1
            absulute_error.append(abs(one_root - root))
            relative_error.append(abs(one_root - root) / root)

        index_analytical_one_root = absulute_error.index(min(absulute_error))
        analytical_one_root = analytical_roots[index_analytical_one_root]
        print(f'The root found by the bisection method is: {one_root} and the analytical root is: {analytical_one_root}')
        if count == len(analytical_roots):
            print("The root are not in the required precision")
            print(f'The absolute error is: {absulute_error[index_analytical_one_root]}')
            print(f'The relative error is: {relative_error[index_analytical_one_root]}')
        else:
            print("The root are in the required precision")
            print(f'The absolute error is: {abs(absulute_error[index_analytical_one_root])}')
            print(f'The relative error is: {abs(relative_error[index_analytical_one_root])}')
        return one_root,analytical_one_root

    if methods == "newton_raphson":
        print("newton_raphson:")
        roots_newton_raphson = []  # find all the roots using newton-raphson method
        for i in np.linspace(a, b,abs(b - a) * 10):  # seting the initial guess for the newton-raphson method using the segment in the x axis
            new_root = round(newton_raphson_method(given_polynom, i), 5)
            roots_newton_raphson.append(new_root)
        roots_newton_raphson = sorted(list(set(roots_newton_raphson)))
        print(f'The roots using Newton-Raphson method are: {roots_newton_raphson}')
        count = 0
        absulute_error = []
        relative_error = []
        if len(roots_newton_raphson) != len(analytical_roots):
            print("not all the roots are found")
            return False
        else:
            for root in range(len(analytical_roots)):
                if abs(roots_newton_raphson[root] - analytical_roots[root]) > epsilon:
                    count += 1
                absulute_error.append(abs(roots_newton_raphson[root] - analytical_roots[root]))
                relative_error.append(abs((roots_newton_raphson[root] - analytical_roots[root]) / analytical_roots[root]))
        if count == len(analytical_roots):
            print("The root are not in the required precision")
            print(f'The absolute error is: {absulute_error}')
            print(f'The relative error is: {relative_error}')
        else:
            print("The root are in the required precision")
            print(f'The absolute error is: {absulute_error}')
            print(f'The relative error is: {relative_error}')
        return roots_newton_raphson,analytical_roots

    if methods == "synthetic_division":
        print("synthetic_division:")
        roots_synthetic_division = []  # find all the roots using synthetic division method
        for i in np.linspace(a, b,
                             abs(b - a) * 10):  # seting the initial guess for the synthetic division method using the segment in the x axis
            new_root = round(synthetic_devision_method(given_polynom, i), 5)
            roots_synthetic_division.append(new_root)
        roots_synthetic_division = sorted(list(set(roots_synthetic_division)))
        print(f'The roots using synthetic division method are: {roots_synthetic_division}')
        count = 0
        absulute_error = []
        relative_error = []
        if len(roots_synthetic_division) != len(analytical_roots):
            print("not all the roots are found")
            return False
        else:
            for root in range(len(analytical_roots)):
                if abs(roots_synthetic_division[root] - analytical_roots[root]) > epsilon:
                    count += 1
                absulute_error.append(abs(roots_synthetic_division[root] - analytical_roots[root]))
                relative_error.append(abs((roots_synthetic_division[root] - analytical_roots[root]) / analytical_roots[root]))
        if count == len(analytical_roots):
            print("The root are not in the required precision")
            print(f'The absolute error is: {absulute_error}')
            print(f'The relative error is: {relative_error}')
        else:
            print("The root are in the required precision")
            print(f'The absolute error is: {absulute_error}')
            print(f'The relative error is: {relative_error}')
        return roots_synthetic_division,analytical_roots

def main():
    """"
    The main function of the script.
    making use of the functions above to find the roots of the given polynom f(x) = X⁴ + 2x³ -7x² + 3
    find the first root using bisection method and the rest using newton-raphson method and synthetic division method
    compare to the analytical solution and check if the error is less than the required precision
    compare between the roots found by the different methods
    and plot the polynom and the roots on the plot
    """

    methods = ["bisection", "newton_raphson", "synthetic_division"]
    print(f'The given polynom is: f(x) = X⁴ + 2x³ -7x² + 3')
    print(f'The segment in the x axis is: [{a}:{b}]')
    print(f'The analytical roots are: {analytical_solution_wolfram_alpha}')
    print(f'The required precision is: {precision_requierd}')
    print(f'the methods used are: {methods}')
    print("")
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print("")
    for method in methods:
        print_roots_errors_for_diffrenent_method(analytical_solution_wolfram_alpha, given_polynom, method, precision_requierd)
        print("")
        print('------------------------------------------------------------------------------------------------------------------------------------')
        print("")

    plt.figure(figsize=(8, 8))
    for i in analytical_solution_wolfram_alpha:
        plt.scatter(i, given_polynom(i), color='red')#mark the roots on the plot
    plt.legend([f'Roots: {given_polynom.roots}'])#add legend to the plot
    y = given_polynom(x_line)
    plt.plot(x_line, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = X⁴ + 2x³ -7x² + 3')
    plt.grid()
    plt.show()
    print("thank you for using the script")

if __name__ == '__main__':
    main()

