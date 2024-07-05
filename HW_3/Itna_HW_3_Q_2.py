"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 2 in HW_3 for the course intro to numerical analysis
the objective of this script is to calculate the value of the function f(x) = cos(4πx) using the lagrange interpolation.
the program will get from the user a value of x and will calculate the value of the function the lagrange interpolation that was created for Question 1
from the file numerical_analysis_methods_tools.py
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
import numerical_analysis_methods_tools as na_tools


# Given parameters
x_i = np.array([0, 0.125, 0.25, 0.375, 0.5])
f = np.cos(4 * np.pi * x_i)
n = len(x_i)

def main():
    # Define the x range from 0 to 0.5
    x = np.linspace(0, 0.5, 1000)
    f_x = np.cos(4 * np.pi * x)
    f_interpolation = np.zeros(len(x))
    for i in range(len(x)):
        f_interpolation[i] = na_tools.lagrange_interpolation_with_lagrange_polynomial(x_i, f, x[i])

    # Plot the function f(x) = cos(4πx) and the interpolation function
    plt.figure(figsize=(10, 6))
    plt.plot(x, f_x, label="f(x) = cos(4πx)", color='b')
    plt.plot(x, f_interpolation, label="Interpolation function", color='r')
    plt.scatter(x_i, f, color='g')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('f(x) = cos(4πx) and the interpolation function')
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()