"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 2 in HW_3 for the course intro to numerical analysis
the objective of this script is to calculate the value of the function f(x) = cos(4πx) using the lagrange interpolation.
the program will get from the user a value of x and will calculate the value of the function the lagrange interpolation that was created for Question 1
the segment of the x axis is [0,0.5]
the program will print the value of the function f(x) and the value of the interpolation function at x

we will use at the beginning four interpolation points in a equal distance on the x axis.
the program will plot the function f(x) = cos(4πx) and the interpolation function
the interpolation function will be shown too in the plot
then we will use eight interpolation points in a equal distance on the x axis.
and we will compare the results of the accuracy of the interpolation function
accuracy required: 10^-4

---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import math as math
import Itna_HW_3_Q_1 as hw3_1
import Inta_HW_2_Q_1 as hw2_1

# Given parameters
x_i = np.array([0, 0.125, 0.25, 0.375, 0.5])
f = np.cos(4 * np.pi * x_i)
n = len(x_i)

def main():
    """
    The main function of the script
    :return: prints the value of the function f(x) and the value of the interpolation function at x
    """
    # get the value of x from the user
    x = float(input("Please enter the value of x: "))
    # calculate the value of the function f(x) = cos(4πx)
    f_x = np.cos(4 * np.pi * x)
    # calculate the value of the interpolation function at x
    L_x = hw3_1.lagrange_interpolation_with_lagrange_polynomial(x_i, f, x)
    # print the value of the function f(x) and the value of the interpolation function at x
    print(f"The value of the function f(x) = cos(4πx) at x = {x} is: {f_x}")
    print(f"The value of the interpolation function at x = {x} is: {L_x}")

    # plot the function f(x) = cos(4πx) and the interpolation function
    x_plot = np.linspace(0, 0.5, 1000)
    f_plot = np.cos(4 * np.pi * x_plot)
    L_plot = np.zeros(1000)
    for i in range(1000):
        L_plot[i] = hw3_1.lagrange_interpolation_with_lagrange_polynomial(x_i, f, x_plot[i])

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f_plot, label="f(x) = cos(4πx)", color='b')
    plt.plot(x_plot, L_plot, label="Interpolation function", color='r')
    plt.scatter(x, f_x, label="f(x)", color='b', marker='o')
    plt.scatter(x, L_x, label="Interpolation function", color='r', marker='o')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = cos(4πx) and the interpolation function')
    plt.legend()
    plt.grid()
    plt.show()

    # calculate the value of the function f(x) = cos(4πx) and the value of the interpolation function at x
    # using eight interpolation points
    x_i_8 = np.linspace(0, 0.5, 8)

if __name__ == '__main__':
    main()