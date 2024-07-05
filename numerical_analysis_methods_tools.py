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




def get_user_input(prompt):
    root = tk.Tk()# Create the root window
    root.withdraw()# Hide the root window
    user_input = simpledialog.askstring(title="Input", prompt=prompt)# Show the input dialog and get the user's input
    root.destroy()# Destroy the root window after getting the input
    return user_input







if __name__ == '__main__':
    pass