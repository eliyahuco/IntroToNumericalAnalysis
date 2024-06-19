"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_2 Question 1 in the course intro to numerical analysis
the objective of this script is to solve a system of linear equations using the fllowing methods: gauss elimination, LU decomposition, gauss-seidel
the script will give the user the option to choose the method he wants to use and will print the solution of the system of linear equations
the order of numbers of operations will be printed as well
the linear equations in matrix form are:
⎡ 3  -3  2 -4 ⎤ ⎡x₁⎤   ⎡ 7.9 ⎤
⎢-2  -1  3 -1 ⎢ ⎢x₂⎢   ⎢-12.5⎥
⎢ 5  -2 -3  2 ⎢ ⎢x₃⎢ = ⎢  18 ⎥
⎣-2   4  1  2 ⎦ ⎣x₄⎦   ⎣ -8.1⎦
"""

from tabulate import tabulate
import numpy as np
from scipy.linalg import lu
from scipy.linalg import solve


# Given parameters
A = np.array([[3, -3, 2, -4], [-2, -1, 3, -1], [5, -2, -3, 2], [-2, 4, 1, 2]])
b = np.array([7.9, -12.5, 18, -8.1])
n = len(b)

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
    # loop over the rows
    for j in range(i + 1, n):
        # find the row with the maximum value
        if abs(A[j, i]) > abs(pivot):
            # update the pivot
            pivot = A[j, i]
            # pivot the rows
            A[[i, j]] = A[[j, i]]
            b[[i, j]] = b[[j, i]]
    return A, b
def rankin_matrix(A, b):
    for i in range(len(b)):
        A, b = pivot_row(A, b, i)
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

    return A, b






#gaeuss elimination method
def gauss_elimination(A, b):
    """
    This function solves a system of linear equations using the Gauss elimination method
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :return: the solution vector, the augmented matrix, the number of operations
    """
    A = np.c_[A, b]# create the augmented matrix
    num_operations = 0 # number of operations
    # loop over the rows
    for i in range(n):
        # find the pivot
        pivot = A[i, i]
        # loop over the rows
        for j in range(i + 1, n):
            # find the factor
            factor = A[j, i] / pivot
            # loop over the columns
            for k in range(n + 1):
                # update the augmented matrix
                A[j, k] = A[j, k] - factor * A[i, k]
                num_operations += 1
    print(tabulate(A, headers=[f'x{i + 1}' for i in range(n)] + ['b']), '\n')
    # create the solution vector
    x = np.zeros(n)
    # loop over the rows
    for i in range(n - 1, -1, -1):
        # initialize the sum
        sum = 0
        # loop over the columns
        for j in range(i + 1, n):
            # update the sum
            sum += A[i, j] * x[j]
            num_operations += 1
        # update the solution vector
        x[i] = (A[i, n] - sum) / A[i, i]
    return x,A, num_operations

print(gauss_elimination(A, b)[0])

print(tabulate(gauss_elimination(A, b)[1], headers=[f'x{i + 1}' for i in range(n)] + ['b']), '\n')

#LU decomposition method  with pivoting and without scipy library
def lu_decomposition_steps(A, b):
    """
    This function solves a system of linear equations using the LU decomposition method
    :param A: the matrix of coefficients
    :param b: the vector of constants
    :return: the solution vector, the number of operations and the augmented matrix, the L matrix and the U matrix
    """
    num_operations = 0

    A = np.c_[A, b]# create the augmented matrix

    L = np.eye(n)# create the L matrix
    # create the U matrix
    U = np.zeros((n, n))
    # loop over the rows
    for i in range(n):
        # loop over the columns
        for j in range(i, n):
            # initialize the sum
            sum = 0
            # loop over the columns
            for k in range(i):
                # update the sum
                sum += L[i, k] * U[k, j]
                num_operations += 1
            # update the U matrix
            U[i, j] = A[i, j] - sum
            num_operations += 1
        # loop over the columns
        for j in range(i + 1, n):
            # initialize the sum
            sum = 0
            # loop over the columns
            for k in range(i):
                # update the sum
                sum += L[j, k] * U[k, i]
                num_operations += 1
            # update the L matrix
            L[j, i] = (A[j, i] - sum) / U[i, i]
            num_operations += 1
    # create the solution vector
    y = np.zeros(n)
    # loop over the rows
    for i in range(n):
        # initialize the sum
        sum = 0
        # loop over the columns
        for j in range(i):
            # update the sum
            sum += L[i, j] * y[j]
            num_operations += 1
        # update the solution vector
        y[i] = (A[i, n] - sum) / L[i, i]
    # create the solution vector
    x = np.zeros(n)
    # loop over the rows
    for i in range(n - 1, -1, -1):
        # initialize the sum
        sum = 0
        # loop over the columns
        for j in range(i + 1, n):
            # update the sum
            sum += U[i, j] * x[j]
            num_operations += 1
        # update the solution vector
        x[i] = (y[i] - sum) / U[i, i]
    return x,A,L,U, num_operations


print(lu_decomposition_steps(A,b)[0])
print(tabulate(lu_decomposition_steps(A,b)[3], headers=[f'x{i + 1}' for i in range(n)] + ['b']), '\n')

print(np.dot(lu_decomposition_steps(A,b)[2],lu_decomposition_steps(A,b)[3]))
#Gauss-Seidel method

