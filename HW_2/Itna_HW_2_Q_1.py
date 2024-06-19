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

from scipy.linalg import solve
import tkinter as tk
from tkinter import simpledialog


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
    # loop over the rows
    for i in range(n):
        # initialize the sum
        sum = 0
        # loop over the columns
        for j in range(n):
            # update the sum
            if j != i:
                sum += A[i, j] * x[j]
                num_of_operations += 1

        # update the solution vector
        x[i] = (b[i] - sum) / A[i, i]
        num_of_operations += 1
    # create the previous solution vector
    x_prev = np.zeros(n)
    # loop over the rows
    while np.linalg.norm(x - x_prev) > tol:
        # update the previous solution vector
        x_prev = x.copy()
        # loop over the rows
        for i in range(n):
            # initialize the sum
            sum = 0
            # loop over the columns
            for j in range(n):
                # update the sum
                if j != i:
                    sum += A[i, j] * x[j]
                    num_of_operations += 1
            # update the solution vector
            x[i] = (b[i] - sum) / A[i, i]
            num_of_operations += 1
        num_iterations += 1
    return x, num_iterations, num_of_operations

def get_user_input(prompt):
    # Create the root window
    root = tk.Tk()
    # Hide the root window
    root.withdraw()
    # Show the input dialog and get the user's input
    user_input = simpledialog.askstring(title="Input", prompt=prompt)
    # Destroy the root window after getting the input
    root.destroy()
    return user_input

def main():
    """
    This function runs the main program
    """
    # print the results
    user_input = get_user_input("please choose the number of themethod you want to use: \n 1 for gauss elimination \n 2 for LU decomposition \n 3 for gauss-seidel \n")
    print("The linear equations in matrix form are:")
    print("⎡ 3  -3  2 -4 ⎤ ⎡x₁⎤   ⎡ 7.9 ⎤ \n⎢-2  -1  3 -1 ⎢ ⎢x₂⎢   ⎢-12.5⎥ \n⎢ 5  -2 -3  2 ⎢ ⎢x₃⎢ = ⎢  18 ⎥ \n⎣-2   4  1  2 ⎦ ⎣x₄⎦   ⎣ -8.1⎦")
    print("\n")
    if user_input == '1':
        print('-Gauss Elimination Method-\n')
        print("The solution vector is:")
        print(gauss_elimination(A, b)[0])
        print("\n")
        print(f'Number of operations: {gauss_elimination(A, b)[2]}')
        print("the algorithm is in order of O(n³) operations, where n is the number of rows in the matrix of coefficients")
    elif user_input == '2':
        print('-LU Decomposition Method-\n')
        print("The solution vector is:")
        print(lu_decomposition_steps(A, b)[0])
        print("\n")
        print(f'Number of operations: {lu_decomposition_steps(A, b)[4]}')
        print("the algorithm is in order of O(²⁄₃n³) operations, where n is the number of rows in the matrix of coefficients")
        print("\nThe L matrix is:\n")
        print(lu_decomposition_steps(A, b)[2])
        print("\nThe U matrix is:\n")
        print(lu_decomposition_steps(A, b)[3])
    elif user_input == '3':
        print('-Gauss-Seidel Method-\n')
        print("The solution vector is:")
        print(gauss_seidel(A, b)[0])
        print("\n")
        print(f'Number of iterations: {gauss_seidel(A, b)[1]}')
        print(f'Number of operations: {gauss_seidel(A, b)[2]}')
        print("the algorithm is in order of O(n²) operations, where n is the number of rows in the matrix of coefficients")
        print("\n")
        print("the script has finished running")
        print("thank you for using the script")





if __name__ == '__main__':
    main()
