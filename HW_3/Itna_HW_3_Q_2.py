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
x_i_1 = np.array([0, 0.125, 0.25, 0.375, 0.5])
x_i_2 = np.array([0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5])
f = np.round(np.cos(4 * np.pi * x_i_1), 4)
n = len(x_i_1)
print(f)
def main():
    # Define the x range from 0 to 0.5
    x = np.linspace(0, 0.5, 1000)
    f_x = np.cos(4 * np.pi * x)
    f_interpolation = np.zeros(len(x))




if __name__ == '__main__':
    main()