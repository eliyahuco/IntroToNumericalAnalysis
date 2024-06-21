"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_2 Question 1 in the course intro to numerical analysis
the objective of this script is to find the inverse of a matrix using the LU decomposition method
the script will use the LU decomposition from the HW_2 Question 1
the script will calculate the inverse of the matrix
and show the result to the user
the matrix is:
⎡ 4  8  4  0 ⎤
⎢ 1  4  7  2 ⎢
⎢ 1  5  4 -3 ⎢
⎣ 1  3  0 -2 ⎦
"""
# Libraries in use
from tabulate import tabulate
import numpy as np
import Itna_HW_2_Q_1 as hw1


# Given parameters
A = np.array([[4, 8, 4, 0], [1, 4, 7, 2], [1, 5, 4, -3], [1, 3, 0, -2]])
n = len(A)


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


def main():
    """
    The main function of the script
    :return: prints the inverse of the matrix and checks the result
    """
    print("\n the matrix is:\n")
    print(tabulate(A, tablefmt='plain'))
    # calculate the inverse of the matrix
    A_inv = inverse_matrix_with_lu_decomposition(A)
    print("\nthe inverse of the matrix is:\n")
    print(tabulate(A_inv[0], tablefmt='plain'))
    print("\nthe L matrix is from the LU decomposition is:\n")
    print(tabulate(A_inv[1], tablefmt='plain'))
    print("\nthe U matrix is from the LU decomposition is:\n")
    print(tabulate(A_inv[2], tablefmt='plain'))
    print("\nlet's check the result:\n")

    print("\nThe inverse of the matrix using numpy is:\n")
    print(tabulate(np.linalg.inv(A), tablefmt='plain'))
    # check if the result is correct
    print("\nThe difference between the two results is:\n")
    print(tabulate(abs(A_inv[0] - np.linalg.inv(A)), tablefmt='plain'))
    # check if the result is correct
    print("\nthe multiplication of the matrix and its inverse is:\n")
    print(tabulate(np.dot(A, A_inv[0]), tablefmt='plain'))
    print("\n we see that we get very small numbers that are not exactly 0")
    print("\n after rounding the result is:\n")
    print(tabulate(np.round(np.dot(A, A_inv[0])), tablefmt='plain'))
    print("\n we can see that the result is the identity matrix")
    print("\n")
    print("the script has finished running")
    print("thank you for using the script")

if __name__ == "__main__":
    main()
