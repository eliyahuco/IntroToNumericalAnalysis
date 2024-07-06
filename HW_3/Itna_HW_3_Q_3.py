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


def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    Solves a tridiagonal matrix equation using the Thomas algorithm
    :param a: lower diagonal of the matrix
    :param b: main diagonal of the matrix
    :param c: upper diagonal of the matrix
    :param d: right-hand side of the equation
    :return: the solution of the equation
    """

    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)
    if b[0] != 0:  # Prevent division by zero
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
    else:
        c_[0] = 0  # or some other suitable value
        d_[0] = 0  # or some other suitable value
    for i in range(1, n - 1):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])
        else:
            c_[i] = 0  # Handle division by zero if necessary
    for i in range(1, n):
        if (b[i] - a[i - 1] * c_[i - 1]) != 0:
            d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])
        else:
            d_[i] = 0  # Handle division by zero if necessarya
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def natural_cubic_spline(x_i, y_i):
    """
    Constructs a natural cubic spline interpolation for given data points.
    :param x_i: x-coordinates of the data points
    :param y_i: y-coordinates of the data points
    :return: Coefficients of the cubic spline for each interval
    """
    n = len(x_i)
    h = np.diff(x_i)

    # Construct the tridiagonal system
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

    # Adjusting arrays for the tridiagonal solver
    A = a[1:n - 1]
    B = b[:n - 2]
    C = c[2:n]
    D = alpha[1:n - 1]

    # Solve the tridiagonal system
    c_sol = tridiagonal_matrix_algorithm(C, A, B, D)

    # Insert the boundary conditions for c
    c = np.zeros(n)
    c[1:n - 1] = c_sol

    # Calculate the b and d coefficients
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y_i[:-1]

    for i in range(n - 1):
        b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    spline_coeffs = np.array([a, b, c[:-1], d]).T
    return spline_coeffs


def evaluate_spline(x, spline_coeffs, xi):
    """
    Evaluates the cubic spline interpolation at a point xi.
    :param x: x-coordinates of the data points
    :param spline_coeffs: Coefficients of the cubic spline for each interval
    :param xi: Point to evaluate the spline at
    :return: Value of the spline at xi
    """

    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(spline_coeffs) - 1)

    dx = xi - x[i]
    a, b, c, d = spline_coeffs[i]
    evalv = a + b * dx + c * dx ** 2 + d * dx ** 3
    return evalv

def parametric_cubic_spline(x_i, y_i):
    """
    Constructs a parametric cubic spline interpolation for given data points. using t as the parameter for x and y.
    :param x_i: x-coordinates of the data points
    :param y_i: y-coordinates of the data points
    :return: Coefficients of the cubic spline for each interval
    """

    n = len(x_i)
    t = np.linspace(0, 1, n)

    # Natural cubic splines for x(t) and y(t)
    spline_coeffs_x = natural_cubic_spline(t, x_i)
    spline_coeffs_y = natural_cubic_spline(t, y_i)

    return spline_coeffs_x, spline_coeffs_y, t

def derviative_cubic_spline(x_i, y_i):
    """
    Constructs the first derivative of a cubic spline interpolation for given data points.
    :param x_i: x-coordinates of the data points
    :param y_i: y-coordinates of the data points
    :return: Coefficients of the cubic spline for each interval
    """

    n = len(x_i)
    h = np.diff(x_i)

    # Construct the tridiagonal system
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

    # Adjusting arrays for the tridiagonal solver
    A = a[1:n - 1]
    B = b[:n - 2]
    C = c[2:n]
    D = alpha[1:n - 1]

    # Solve the tridiagonal system
    c_sol = tridiagonal_matrix_algorithm(C, A, B, D)

    # Insert the boundary conditions for c
    c = np.zeros(n)
    c[1:n - 1] = c_sol

    # Calculate the b and d coefficients
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y_i[:-1]

    for i in range(n - 1):
        b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    spline_coeffs = np.array([a, b, c[:-1], d]).T
    return spline_coeffs

def second_derivative_cubic_spline(x_i, y_i):
    """
    Constructs the second derivative of a cubic spline interpolation for given data points.
    :param x_i: x-coordinates of the data points
    :param y_i: y-coordinates of the data points
    :return: Coefficients of the cubic spline for each interval
    """

    n = len(x_i)
    h = np.diff(x_i)

    # Construct the tridiagonal system
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

    # Adjusting arrays for the tridiagonal solver
    A = a[1:n - 1]
    B = b[:n - 2]
    C = c[2:n]
    D = alpha[1:n - 1]

    # Solve the tridiagonal system
    c_sol = tridiagonal_matrix_algorithm(C, A, B, D)

    # Insert the boundary conditions for c
    c = np.zeros(n)
    c[1:n - 1] = c_sol

    # Calculate the b and d coefficients
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    a = y_i[:-1]

    for i in range(n - 1):
        b[i] = (y_i[i + 1] - y_i[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    spline_coeffs = np.array([a, b, c[:-1], d]).T
    return spline_coeffs



def main():

    #given parameters
    x_i = np.array([0, 2, 3, 4, 7, 8])
    y_i = np.array([4, 2, 8, 10, 4, -2])
    n = len(x_i)
    #a) natural cubic spline interpolation
    spline_coeffs = natural_cubic_spline(x_i, y_i)
    print('The coefficients of the cubic spline are:')
    print(spline_coeffs)
    x_segement = np.linspace(0, 8, 1000)
    y_segement = [evaluate_spline(x_i, spline_coeffs, xi) for xi in x_segement]
    fig, ax = plt.subplots(1,3, figsize=(20, 8))
    ax[0].plot(x_segement, y_segement, label='cubic spline interpolation', color='b')
    ax[0].scatter(x_i, y_i, color='r', label='interpolation points')
    ax[0].set_title('Cubic Spline Interpolation', fontweight='bold', fontsize=14)
    ax[0].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[0].set_ylabel('y', fontweight='bold', fontsize=14)
    ax[0].legend()
    #first derivative of the cubic spline interpolation
    spline_coeffs_first_derivative = derviative_cubic_spline(x_i, y_i)
    y_segement_first_derivative = [evaluate_spline(x_i, spline_coeffs_first_derivative, xi) for xi in x_segement]
    ax[0].plot(x_segement, y_segement_first_derivative, label='first derivative of cubic spline interpolation', color='g')
    ax[0].legend()
    #second derivative of the cubic spline interpolation
    spline_coeffs_second_derivative = second_derivative_cubic_spline(x_i, y_i)
    y_segement_second_derivative = [evaluate_spline(x_i, spline_coeffs_second_derivative, xi) for xi in x_segement]
    ax[0].plot(x_segement, y_segement_second_derivative, label='second derivative of cubic spline interpolation', color='y')
    ax[0].legend()
    #b) parametric cubic spline interpolation
    x_i = np.array([3, 2, 2.5, 4, 5, 4])
    y_i = np.array([4, 3, 1, 2, 3.5, 4.5])
    t_seg = np.linspace(0, 1, 1000)
    spline_coeffs_x, spline_coeffs_y, t = parametric_cubic_spline(x_i, y_i)
    x_segement = [evaluate_spline(t, spline_coeffs_x, ti) for ti in t_seg]
    y_segement = [evaluate_spline(t, spline_coeffs_y, ti) for ti in t_seg]
    ax[1].plot(x_segement, y_segement, label='parametric cubic spline interpolation', color='b')
    ax[1].scatter(x_i, y_i, color='r', label='interpolation points')
    ax[1].set_title('Parametric Cubic Spline Interpolation without periodicity', fontweight='bold', fontsize=14)
    ax[1].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[1].set_ylabel('y', fontweight='bold', fontsize=14)
    ax[1].legend()
    # addind the first derivative of the parametric cubic spline interpolation
    x_i = np.array([3, 2, 2.5, 4, 5, 4, 3])
    y_i = np.array([4, 3, 1, 2, 3.5, 4.5, 4])
    t_seg = np.linspace(0, 1, 1000)
    spline_coeffs_x, spline_coeffs_y, t = parametric_cubic_spline(x_i, y_i)
    x_segement_parametric = [evaluate_spline(t, spline_coeffs_x, ti) for ti in t_seg]
    y_segement_parametric = [evaluate_spline(t, spline_coeffs_y, ti) for ti in t_seg]
    ax[2].plot(x_segement_parametric, y_segement_parametric, label='parametric cubic spline interpolation with periodicity', color='b')
    ax[2].scatter(x_i, y_i, color='r', label='interpolation points')
    ax[2].set_title('Parametric Cubic Spline Interpolation wit periodicity', fontweight='bold', fontsize=14)
    ax[2].set_xlabel('x', fontweight='bold', fontsize=14)
    ax[2].set_ylabel('y', fontweight='bold', fontsize=14)
    ax[2].legend()


    plt.show()

if __name__ == '__main__':
    main()