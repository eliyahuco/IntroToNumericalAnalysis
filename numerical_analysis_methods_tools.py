"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

this script we contain the functions that are used in the course intro to numerical analysis
the script we use as a library of methods and tools that are used in the assignments of the course
the functions will be general for use in many cases

--------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import math as math
from tabulate import tabulate
from scipy.linalg import solve
import tkinter as tk
from tkinter import simpledialog

def derivative_polynom_in_x(polynom,x_0,epsilon):
    """
    This function calculates the derivative of a polynomial at a given point using the definition of the derivative.
    The function receives the polynomial, the point x_0, and the step size epsilon.
    The function returns the derivative of the polynomial at the point x_0.
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
# Define the partial derivative of the functions f(x, y) and g(x, y)
def partial_derivative_of_function_2d(function, x, y, h = 10**-9):
    """"
    This function calculates the partial derivative of a 2D function.
    The function receives the function, the point (x, y) and the step size h.
    The function returns the partial derivative of the function at the point (x, y).
    """
    df_dx = (function(x + h, y) - function(x, y)) / h
    df_dy = (function(x, y + h) - function(x, y)) / h
    return df_dx, df_dy

# Define the Newton-Raphson method for simultaneous solution
def newton_raphson_method_for_simultaneous_solution( x0, y0, epsilon = 10**-4):
    """"
    This function solves the system of equations f(x, y) = 0 and g(x, y) = 0 using the Newton-Raphson method.
    The function receives the initial guess (x0, y0) and the precision requirement epsilon.
    The function returns the solution (x, y) of the system of equations.
    """
    x = x0
    y = y0
    if abs(f(x, y)) < epsilon and abs(g(x, y)) < epsilon:
        return x, y
    while abs(f(x, y)) > epsilon and abs(g(x, y)) > epsilon:
        df_dx, df_dy = partial_derivative_of_function_2d(f, x, y)
        dg_dx, dg_dy = partial_derivative_of_function_2d(g, x, y)
        jacobian = df_dx * dg_dy - df_dy * dg_dx
        x = x - (f(x, y) * dg_dy - g(x, y) * df_dy) / jacobian
        y = y - (g(x, y) * df_dx - f(x, y) * dg_dx) / jacobian
    return x,y

def pivot_row(A, b, i):
    """
    This function pivots the rows of the matrix of coefficients and the vector of constants
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :param i: the row index
    :return: the pivoted matrix of coefficients and the pivoted vector of constants
    """
    # find the pivot
    pivot = A[i, i]
    for j in range(i + 1, n):# loop over the rows
        if abs(A[j, i]) > abs(pivot):# find the row with the maximum value
            pivot = A[j, i]# update the pivot
            A[[i, j]] = A[[j, i]]# pivot the rows
            b[[i, j]] = b[[j, i]]
    return A, b
def rankin_matrix(A, b):
    """
    This function ranks the matrix of coefficients
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :return: the ranked matrix of coefficients and the ranked vector of constants
    """
    for i in range(len(b)):
        A, b = pivot_row(A, b, i)
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    return A, b

#gaeuss elimination method
#the algorithm is in order of O(n³) operations
def gauss_elimination(A, b):
    """
    This function solves a system of linear equations using the Gauss elimination method
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :return: the solution vector, the augmented matrix, the number of operations
    """
    A = np.c_[A, b]# create the augmented matrix
    num_operations = 0 # number of operations
    for i in range(n):# loop over the rows
        pivot = A[i, i]# find the pivot
        for j in range(i + 1, n):# loop over the rows
            factor = A[j, i] / pivot# find the factor
            num_operations += 1
            for k in range(n + 1):# loop over the columns
                A[j, k] = A[j, k] - factor * A[i, k]# update the augmented matrix
                num_operations += 1
    x = np.zeros(n)# create the solution vector
    for i in range(n - 1, -1, -1):# loop over the rows
        sum = 0# initialize the sum
        for j in range(i + 1, n):# loop over the columns
            sum += A[i, j] * x[j]# update the sum
            num_operations += 1
        x[i] = (A[i, n] - sum) / A[i, i] # update the solution vector
        num_operations += 1
    return x,A, num_operations


#LU decomposition method  with pivoting and without scipy library
#the algorithm is in order of O(²⁄₃n³) operations
def lu_decomposition_steps(A, b):
    """
    This function solves a system of linear equations using the LU decomposition method
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :return: the solution vector, the number of operations and the augmented matrix, the L matrix and the U matrix
    """
    n = len(b)
    num_operations = 0 # number of operations
    A = np.c_[A, b]# create the augmented matrix
    L = np.eye(n)# create the L matrix
    U = np.zeros((n, n))# create the U matrix
    for i in range(n):# loop over the rows
        for j in range(i, n):# loop over the columns
            sum = 0# initialize the sum
            for k in range(i):# loop over the columns
                sum += L[i, k] * U[k, j]# update the sum
                num_operations += 1
            U[i, j] = A[i, j] - sum# update the U matrix
            num_operations += 1
        for j in range(i + 1, n):# loop over the columns
            sum = 0# initialize the sum
            for k in range(i):# loop over the columns
                sum += L[j, k] * U[k, i]# update the sum
                num_operations += 1
            L[j, i] = (A[j, i] - sum) / U[i, i]# update the L matrix
            num_operations += 1
    y = np.zeros(n)# create the solution vector
    for i in range(n):# loop over the rows
        sum = 0# initialize the sum
        for j in range(i):# loop over the columns
            sum += L[i, j] * y[j]# update the sum
            num_operations += 1
        y[i] = (A[i, n] - sum) / L[i, i]# update the solution vector
    x = np.zeros(n)# create the solution vector
    for i in range(n - 1, -1, -1):# loop over the rows
        sum = 0# initialize the sum
        for j in range(i + 1, n):# loop over the columns
            sum += U[i, j] * x[j]# update the sum
            num_operations += 1
        x[i] = (y[i] - sum) / U[i, i]# update the solution vector
    return x,A,L,U, num_operations


#Gauss-Seidel method
#the algorithm is in order of O(n²) operations
def gauss_seidel(A, b, tol=1e-6):
    """
    This function solves a system of linear equations using the Gauss-Seidel method
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :param tol: the tolerance
    :return: the solution vector
    """
    num_iterations = 0 # number of iterations
    num_of_operations = 0 # number of operations
    n = len(b)
    for i in range(len(b)):
        A, b = pivot_row(A, b, i)
        num_of_operations += 1
    x = np.ones(n)
    for i in range(n):# loop over the rows
        sum = 0# initialize the sum
        for j in range(n):# loop over the columns
            if j != i:# update the sum
                sum += A[i, j] * x[j]
                num_of_operations += 1
        x[i] = (b[i] - sum) / A[i, i]# update the solution vector
        num_of_operations += 1
    x_prev = np.zeros(n)# create the previous solution vector
    while np.linalg.norm(x - x_prev) > tol:# loop over the rows
        x_prev = x.copy()# update the previous solution vector
        for i in range(n):# loop over the rows
            sum = 0# initialize the sum
            for j in range(n):# loop over the columns
                if j != i:# update the sum
                    sum += A[i, j] * x[j]
                    num_of_operations += 1
            x[i] = (b[i] - sum) / A[i, i]# update the solution vector
            num_of_operations += 1
        num_iterations += 1
    return x, num_iterations, num_of_operations

def inverse_matrix_with_lu_decomposition(A):
    """
    This function calculates the inverse of a matrix using the LU decomposition method
    :param A: the matrix of coefficients
    :return: the inverse of the matrix and the L and U matrices
    the function uses the LU decomposition from HW_2 Question 1
    the name of the file is Itna_HW_2_Q_1.py
    """

    A_inverse = np.zeros((n, n))# initialize the inverse matrix
    I = np.eye(n)# initialize the identity matrix
    for i in range(n):# loop over the columns of the identity matrix
        A_inverse[:, i] = hw1.lu_decomposition_steps(A, I[:, i])[0]# solve the system of linear equations

    return A_inverse, hw1.lu_decomposition_steps(A, I[:, i])[2], hw1.lu_decomposition_steps(A, I[:, i])[3]
def craete_lagrange_polynomial(x_i, x, i):
    """
    This function creates the Lagrange polynomial
    :param x_i: the x values of the interpolation points
    :param x: the x value to interpolate
    :param i: the index of the interpolation point
    :param n: the number of interpolation points
    :return: the value of the Lagrange polynomial at x
    """
    n = len(x_i)
    l_x = 1 # initialize the Lagrange polynomial
    for j in range(n):
        # calculate the Lagrange polynomial
        if i != j:
            l_x = l_x * (x - x_i[j]) / (x_i[i] - x_i[j])
    return l_x

def lagrange_interpolation_with_lagrange_polynomial(x_i, f, x):
    """
    This function creates the Lagrange interpolation function
    :param x_i: the x values of the interpolation points
    :param f: the f values of the interpolation points
    :param x: the x value to interpolate
    :param n: the number of interpolation points
    :return: the value of the interpolation function at x
    """
    n = len(x_i)
    L_x = 0 # initialize the interpolation function
    for i in range(n):
        l_x = craete_lagrange_polynomial(x_i, x, i) # initialize the Lagrange polynomial
        # calculate the interpolation function
        L_x = L_x + f[i] * l_x
    return L_x

def lagrange_interpolation(x_i, f, x):
    """
    This function creates the Lagrange interpolation function
    :param x_i: the x values of the interpolation points
    :param f: the f values of the interpolation points
    :param x: the x value to interpolate
    :param n: the number of interpolation points
    :return: the value of the interpolation function at x
    """
    n = len(x_i)
    L_x = 0 # initialize the interpolation function
    for i in range(n):
        l_x = 1 # initialize the Lagrange polynomial
        for j in range(n):
            # calculate the Lagrange polynomial
            if i != j:
                l_x = l_x * (x - x_i[j]) / (x_i[i] - x_i[j])
        # calculate the interpolation function
        L_x = L_x + f[i] * l_x
    return L_x
def get_interpolation_error(x, f_x, interpolation_function):
    """
    This function calculates the errors of the interpolation function
    :param x: the x value to interpolate
    :param f_x: the value of the function f(x) at x
    :param interpolation_function: the value of the interpolation function at x
    :return: the absolute error, the relative error, the relative error in percentage
    """
    abs_error = abs(f_x - interpolation_function)
    rel_error = abs((f_x - interpolation_function) / f_x)
    rel_error_percentage = abs((f_x - interpolation_function) / f_x) * 100
    return abs_error, rel_error, rel_error_percentage
def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    if b[0] != 0:
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
    else:
        c_[0] = 0
        d_[0] = 0

    for i in range(1, n - 1):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])
        else:
            c_[i] = 0

    for i in range(1, n):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])
        else:
            d_[i] = 0

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x
def natural_cubic_spline(x_i, y_i):
    n = len(x_i)
    h = np.diff(x_i)

    a = np.zeros(n)
    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)
    alpha = np.zeros(n)

    for i in range(1, n - 1):
        a[i] = 2 * (h[i - 1] + h[i])
        b[i - 1] = h[i]
        c[i] = h[i - 1]
        alpha[i] = 3 * ((y_i[i + 1] - y_i[i]) / h[i] - (y_i[i] - y_i[i - 1]) / h[i - 1])

    A = a[1:n - 1]
    B = b[:n - 2]
    C = c[2:n]
    D = alpha[1:n - 1]

    c_sol = tridiagonal_matrix_algorithm(C, A, B, D)

    c = np.zeros(n)
    c[1:n - 1] = c_sol

    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y_i[:-1]

    for i in range(n - 1):
        b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    spline_coeffs = np.array([a, b, c[:-1], d]).T
    return spline_coeffs

def evaluate_spline(x, spline_coeffs, xi):
    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(spline_coeffs) - 1)

    dx = xi - x[i]
    a, b, c, d = spline_coeffs[i]

    fi = a + b * dx + c * dx ** 2 + d * dx ** 3
    f_prime = b + 2 * c * dx + 3 * d * dx ** 2
    f_double_prime = 2 * c + 6 * d * dx

    return fi, f_prime, f_double_prime

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
        h = (b - a) / n
        for i in range(n):
            a = x[i]
            b = x[i + 1]

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
        k = k + 1
        integral = integral_
        integral_ = trapezoidal_rule_integration(f, a, b, 2*n)
        b_h =((4**k)*integral_ - integral)/((4**k)-1)

    return b_h

def legendre_polynomial(n):
        """
        This function calculates the nth-order Legendre polynomial using the recursive formula
        :param n: the order of the Legendre polynomial
        :return: the nth-order Legendre polynomial
        """
        if n == 0:
            return np.poly1d([1])
        elif n == 1:
            return np.poly1d([1, 0])

        P0 = np.poly1d([1])
        P1 = np.poly1d([1, 0])

        for k in range(2, n + 1):
            Pk = ((2 * k - 1) * np.poly1d([1, 0]) * P1 - (k - 1) * P0) / k
            P0, P1 = P1, Pk

        return Pk


def legendre_roots(n, tol=1e-12):
    """
    This function calculates the roots of the nth-order Legendre polynomial using Newton's method
    :param n: the order of the Legendre polynomial
    :param tol: the tolerance
    :return: the roots of the nth-order Legendre polynomial
    """
    Pn = legendre_polynomial(n)
    Pn_deriv = np.polyder(Pn)

    x = np.cos(np.pi * (np.arange(n) + 0.5) / n)

    for _ in range(100):
        x_new = x - Pn(x) / Pn_deriv(x)
        if np.all(np.abs(x - x_new) < tol):
            break
        x = x_new

    return x


def legendre_weights(n):
    '''
    This function calculates the weights of the Gauss-Legendre quadrature
    :param n: the number of points
    :return: the weights
    '''
    roots = legendre_roots(n)
    Pn = legendre_polynomial(n)
    Pn_deriv = np.polyder(Pn)

    weights = 2 / ((1 - roots ** 2) * (Pn_deriv(roots) ** 2))
    return weights

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

def gauss_quadrature(f, a, b, n=2):
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
                                8: ([-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648,
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

    if n < 2:
        raise ValueError("n must be at least 2")
    if n >= 2 and n <= 10:
        integral = 0
        x, w = dict_of_weights_for_quad[n]
        for i in range(n):
            transform_roots = 0.5 * (x[i] + 1) * (b - a) + a
            integral += w[i] * f(transform_roots)
        return 0.5 * (b - a) * integral
    else:
        x, w = na_tools.legendre_roots(n), na_tools.legendre_weights(n)

        # Transform roots to the interval [a, b]
        transformed_roots = 0.5 * (x + 1) * (b - a) + a

        # Apply the Gauss quadrature formula
        integral = 0.5 * (b - a) * np.sum(w * f(transformed_roots))
        return integral


def get_user_input(prompt):
    root = tk.Tk()# Create the root window
    root.withdraw()# Hide the root window
    user_input = simpledialog.askstring(title="Input", prompt=prompt)# Show the input dialog and get the user's input
    root.destroy()# Destroy the root window after getting the input
    return user_input

def get_user_input_float(prompt):
    while True:
        try:
            user_input = float(get_user_input(prompt))# Get the user's input
            break
        except ValueError:# Handle the exception
            print("Invalid input. Please enter a number.")# Print an error message
    return user_input
def get_user_input_int(prompt):
    while True:
        try:
            user_input = int(get_user_input(prompt))# Get the user's input
            break
        except ValueError:# Handle the exception
            print("Invalid input. Please enter a number.")# Print an error message
    return user_input

def get_user_input_list(prompt):
    while True:
        try:
            user_input = get_user_input(prompt)# Get the user's input
            user_input_list = list(map(float, user_input.split()))# Convert the input string to a list of floats
            break
        except ValueError:# Handle the exception
            print("Invalid input. Please enter a list of numbers separated by spaces.")# Print an error message
    return user_input_list

def get_user_input_matrix(prompt):
    while True:
        try:
            user_input = get_user_input(prompt)# Get the user's input
            user_input_matrix = np.array([list(map(float, row.split())) for row in user_input.split(";")])# Convert the input string to a matrix
            break
        except ValueError:# Handle the exception
            print("Invalid input. Please enter a matrix of numbers separated by semicolons.")# Print an error message
    return user_input_matrix

def get_user_input_vector(prompt):
    while True:
        try:
            user_input = get_user_input(prompt)# Get the user's input
            user_input_vector = np.array(list(map(float, user_input.split())))# Convert the input string to a vector
            break
        except ValueError:# Handle the exception
            print("Invalid input. Please enter a vector of numbers separated by spaces.")# Print an error message
    return user_input_vector


def euler_method_ode(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    for x in x_values[:-1]:
        y += h * ode_func(x, y)
        y_values.append(y)
    return x_values, np.array(y_values)

# Runge-Kutta Second Order Method (RK2)
def rk2_method(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    for x in x_values[:-1]:
        f1 = ode_func(x, y)
        f2 = ode_func(x + h, y + h * f1)
        y += h * (f1 + f2) / 2
        y_values.append(y)
    return x_values, np.array(y_values)


def eight_tirds_simpson_rule_integration(f, a, b, n=1):
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
        return ((3/8)*h) * (f(a) + 3 * f(a + h) + 3 * f(a + 2*h) + f(b))
    else:
        for i in range(n):
            a = x[i]
            b = x[i + 1]
            c = (a + b) / 2
            h = (b - a) / 3
            integral += ((3/8)*h) * (f(a) + 3 * f(a + h) + 3 * f(a + 2*h) + f(b))
    return integral


def trapezoidal_rule_integration(f, a, b, n=1):
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


def composite_trapezoidal_rule(f, a, b, n):
    """
    This function calculates the integral of a function using the composite trapezoidal rule
    :param f: the function to integrate given as a lambda function
    :param a: the lower limit of the integral
    :param b: the upper limit of the integral
    :param n: the number of intervals
    :return: the value of the integral
    """
    x = np.linspace(a, b, n + 1)
    integral = 0
    h = (b - a) / n
    for i in range(n+1):
        if i == 0 or i == n:
            integral += 0.5 * h * f(x[i])
        else:
            integral += h * f(x[i])
    return integral




def main():
    pass



if __name__ == '__main__':
    main()