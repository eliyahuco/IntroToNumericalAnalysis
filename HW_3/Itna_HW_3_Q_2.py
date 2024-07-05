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
def get_interpolation_points_equal_distance(x_min, x_max, n):
    """
    This function creates the interpolation points in equal distance
    :param x_min: the minimum value of x
    :param x_max: the maximum value of x
    :param n: the number of interpolation points
    :return: the interpolation points
    """
    x_i = np.linspace(x_min, x_max, n)
    return x_i
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
def main():
    #given parameters
    x_segement = np.linspace(0, 0.5, 1000)
    x_min = x_segement[0]
    x_max = x_segement[-1]
    n = [4, 8]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True, sharex=True)

    x = na_tools.get_user_input_float(f'Please enter a value of x a number between {x_min} and {x_max}: ')
    while x < x_min or x > x_max:
        x = na_tools.get_user_input_float(
            f'the value of x is not in the range,\n Please enter the value in the range [{x_min} , {x_max}]: ')
    for i in range(len(n)):
        print('#' * 100)
        print(f'for {n[i]} interpolation points in equal distance:')
        x_i = get_interpolation_points_equal_distance(x_min, x_max, n[i])
        f = np.cos(4 * np.pi * x_i)
        print(f'the points (xᵢ,yᵢ) of the interpolation are: {list((zip(np.round(x_i,5), np.round(f,5))) )}')


        f_x = np.cos(4*np.pi*x)
        interpolation_function = na_tools.lagrange_interpolation_with_lagrange_polynomial(x_i, f, x)
        print('\n' + '#'*100)
        print(f'the value of f(x) = cos(4πx) at x = {x} is: {f_x}')
        print(f'the value of the interpolation function at x = {x} is: {interpolation_function}')

        print(f'\nErorrs analysis:')
        abs_error, rel_error, rel_error_percentage = get_interpolation_error(x, f_x, interpolation_function)
        print(f'The absolute error is: {abs_error}')
        print(f'The relative error is: {rel_error}')
        print(f'The relative error in percentage is: {rel_error_percentage}%')
        if abs(f_x - interpolation_function) > 10**(-4):
            print(f'The accuracy of the interpolation function with {n[i]} points is good at x = {x} ')
        else:
            print(f'The accuracy of the interpolation function with {n[i]} points is not good at x = {x} ')
        print('#'*100)
        print('\n' )
        if i == 0:
            axs[i].plot(x_segement, np.cos(4*np.pi*x_segement), label="f(x) = cos(4πx)", color='g')
            axs[i].plot(x_segement, na_tools.lagrange_interpolation(x_i, f, x_segement), label="interpolation function", color='b')
            axs[i].scatter(x, f_x, label="(x,f(x))", color='r', s=75, marker='o')
            axs[i].scatter(x_i, f, label="interpolation points", color='black', s=75)
            axs[i].set_xlabel("x", fontweight='bold', fontsize=16)
            axs[i].set_ylabel("f(x)", fontweight='bold', fontsize=16)
            axs[i].legend()
            axs[i].grid()
            axs[i].set_title(f'f(x) = cos(4πx) and the interpolation function (for {n[i]}  points)', fontsize=12, color='b', fontweight='bold')
        elif i == 1:
            axs[i].plot(x_segement, np.cos(4*np.pi*x_segement), label="f(x) = cos(4πx)", color='g')
            axs[i].plot(x_segement, na_tools.lagrange_interpolation(x_i, f, x_segement), label="interpolation function", color='b')
            axs[i].scatter(x, f_x, label="(x,f(x))", color='r', s=75, marker='o')
            axs[i].scatter(x_i, f, label="interpolation points", color='black', s=75)
            axs[i].set_xlabel("x", fontweight='bold', fontsize=16)
            axs[i].set_ylabel("f(x)", fontweight='bold', fontsize=16)
            axs[i].legend()
            axs[i].grid()
            axs[i].set_title(f'f(x) = cos(4πx) and the interpolation function (for {n[i]}  points)', fontsize=12, color='b', fontweight='bold')
    print(f'the precision of the interpolation with {n[-1]} points gets better than the precision of the interpolation with {n[0]} points')
    x_i4 = get_interpolation_points_equal_distance(x_min, x_max, n[0])
    f4 = np.cos(4 * np.pi * x_i4)
    x_i8 = get_interpolation_points_equal_distance(x_min, x_max, n[1])
    f8 = np.cos(4 * np.pi * x_i8)
    intrp4 = na_tools.lagrange_interpolation(x_i4, f4, x)
    intrp8 = na_tools.lagrange_interpolation(x_i8, f8, x)
    error_intrp4 = get_interpolation_error(x, f_x, intrp4)
    error_intrp8 = get_interpolation_error(x, f_x, intrp8)
    print(f'the precision of the interpolation with {n[1]} points is better than the precision of the interpolation with {n[0]} points by (absolute error): {error_intrp4[0] - error_intrp8[0]} ')
    print(f'and by (relative error): {error_intrp4[1] - error_intrp8[1]} \n'f'and by (relative error in percentage): {error_intrp4[2] - error_intrp8[2]} %')
    plt.show()
    #save the plot
    fig.savefig('plot of Interpolation vs f(x)_HW_3_Q_2.png')
    print("\n")
    print('#'*100)
    print('The plot of the interpolation function and f(x) = cos(4πx) is saved in the file: \"plot of Interpolation vs f(x)_HW_3_Q_2.png\"')
    print("the script has finished running")
    print("thank you for using the script")
    print('#'*100)
    return







if __name__ == '__main__':
    main()