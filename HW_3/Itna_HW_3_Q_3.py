"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 3 in HW_3 for the course intro to numerical analysis
the objective of this script is to create interpolation functions using the cubic spline method
the cubic spline method is based on solution of a system of linear tridiagonal matrix equations
the question has two sections:
a) to show a plot of the cubic spline interpolation function for the points:
    (xᵢ,yᵢ) = [(0,4), (2,2), (3,8),(4,10),(7,4),(8,-2)]
    the program will calculate the coefficients of the cubic spline and will plot the interpolation function
    we will show also the two first derivatives of the interpolation function
    we will print the coefficients of the cubic spline
b) to show a plot of the cubic spline interpolation function for the points:
    (xᵢ,yᵢ) = [(3,4), (2,3), (2.5,1), (4,2), (5,3.5), (4,4.5)]
    for this section we will use parametric cubic spline interpolation
    the program will calculate the coefficients of the cubic spline and will plot the interpolation function
    we will show also the two first derivatives of the interpolation function
    we will print the coefficients of the cubic spline
    we will compare the results of the accuracy of the interpolation function
    accuracy required: 10^-4

will use the techniques of natural cubic spline and parametric cubic spline:
natural cubic spline means that the second derivative at the edges is zero
parametric cubic spline means that x and y are themselves functions of a parameter t

we will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools

# Given parameters
x_i = np.array([0, 2, 3, 4, 7, 8])
y_i = np.array([4, 2, 8, 10, 4, -2])
n = len(x_i)


import numpy as np
import matplotlib.pyplot as plt

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

def parametric_cubic_spline(x_i, y_i):
    n = len(x_i)
    t = np.linspace(0, 1, n)

    spline_coeffs_x = natural_cubic_spline(t, x_i)
    spline_coeffs_y = natural_cubic_spline(t, y_i)

    return spline_coeffs_x, spline_coeffs_y, t

def main():
    """
    The main function of the script
    :return: plots the interpolation functions
    """

    x_i = np.array([0, 2, 3, 4, 7, 8])
    y_i = np.array([4, 2, 8, 10, 4, -2])
    n = len(x_i)

    spline_coeffs = natural_cubic_spline(x_i, y_i)
    print('\nThe coefficients of the cubic spline are:')
    print(spline_coeffs)

    x_segment = np.linspace(0, 8, 1000)
    y_segment = [evaluate_spline(x_i, spline_coeffs, xi)[0] for xi in x_segment]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(x_segment, y_segment, label='Cubic Spline Interpolation', color='b')
    ax[0].scatter(x_i, y_i, color='r', label='Interpolation Points')
    ax[0].set_title('Cubic Spline Interpolation', fontweight='bold', fontsize=14)
    ax[0].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[0].set_ylabel('y', fontweight='bold', fontsize=14)


    y_segment_first_derivative = [evaluate_spline(x_i, spline_coeffs, xi)[1] for xi in x_segment]
    ax[0].plot(x_segment, y_segment_first_derivative, label='First Derivative', color='g')


    y_segment_second_derivative = [evaluate_spline(x_i, spline_coeffs, xi)[2] for xi in x_segment]
    ax[0].plot(x_segment, y_segment_second_derivative, label='Second Derivative', color='y')
    ax[0].legend(fontsize = 8, loc='upper left')

    x_i = np.array([3, 2, 2.5, 4, 5, 4])
    y_i = np.array([4, 3, 1, 2, 3.5, 4.5])
    t_seg = np.linspace(0, 1, 1000)
    spline_coeffs_x, spline_coeffs_y, t = parametric_cubic_spline(x_i, y_i)
    x_segment = [evaluate_spline(t, spline_coeffs_x, ti)[0] for ti in t_seg]
    y_segment = [evaluate_spline(t, spline_coeffs_y, ti)[0] for ti in t_seg]

    ax[1].plot(x_segment, y_segment, label='Parametric Cubic Spline Interpolation', color='b')
    ax[1].scatter(x_i, y_i, color='r', label='Interpolation Points')
    ax[1].set_title('Parametric Cubic Spline Interpolation without Periodicity', fontweight='bold', fontsize=10)
    ax[1].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[1].set_ylabel('y', fontweight='bold', fontsize=14)
    ax[1].legend(fontsize = 8, loc='upper left')

    x_i = np.array([3, 2, 2.5, 4, 5, 4, 3])
    y_i = np.array([4, 3, 1, 2, 3.5, 4.5, 4])
    t_seg = np.linspace(0, 1, 1000)
    spline_coeffs_x, spline_coeffs_y, t = parametric_cubic_spline(x_i, y_i)
    x_segment_parametric = [evaluate_spline(t, spline_coeffs_x, ti)[0] for ti in t_seg]
    y_segment_parametric = [evaluate_spline(t, spline_coeffs_y, ti)[0] for ti in t_seg]

    ax[2].plot(x_segment_parametric, y_segment_parametric, label='Parametric Cubic Spline Interpolation with Periodicity', color='b')
    ax[2].scatter(x_i, y_i, color='r', label='Interpolation Points')
    ax[2].set_title('Parametric Cubic Spline Interpolation with Periodicity', fontweight='bold', fontsize=10)
    ax[2].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[2].set_ylabel('y', fontweight='bold', fontsize=14)
    ax[2].legend(fontsize = 8, loc='upper left')
    plt.tight_layout()
    plt.show()

    # save the plot
    fig.savefig('plot of Cubic_Spline_Interpolation_HW_3_Q_3.png')
    print("\n")
    print("The script has been completed successfully")
    print("the plot has been saved as \"plot of Cubic_Spline_Interpolation_HW_3_Q_3.png\"")
    print("thank you for using the script")

if __name__ == "__main__":
    main()


