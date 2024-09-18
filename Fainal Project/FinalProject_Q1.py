"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 1 in the final project for the course intro to numerical analysis

the objective of this script is to solve a problem of the wave equation in 2D space:
    ∂^2u/∂t^2 = c^2(∂^2u/∂x^2 + ∂^2u/∂z^2) + F(t)

        where:
        u(x,z,t) is the wave function
        c is the wave speed
        F(x,z,t) is the source function
        x,z are the spatial coordinates
        t is the time coordinate

            the source function is given as:
            F(t) = t*exp(2*pi*t)*sin(2*pi*t) for  0<=t<=0.05
            F(t) = 0 for t>0.05
            the source location is at x=3000 meters, z=2800 meters

there is a layer that defines the speed of the wave, the layer given by series of points: (x,z) = (0,2600), (1000,4000), (2600,3200), (4600,3600), (6000,2400)
above the layer, the speed of the wave is c1 = 2000 m/s, and below the layer, the speed of the wave is c2 = 3000 m/s
we will find the layer using cubic spline interpolation
then we will solve the wave equation using the finite difference 4th order in the spatial domain and 2nd order in the time domain
the spatial step dx = dz = 100 meters
the time step dt = 0.01 seconds, 0.03 seconds
we will show the wave field (snapshot):
for dt = 0.01 seconds at t = 0.15,0.4,0.7,1.0 seconds
for dt = 0.03 seconds at t = 0.15,0.3,0.6,0.9 seconds

at the end we will plot animation of the wave field with dt = 0.01 seconds

we will solve the wave equation in the explicit method

---------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numerical_analysis_methods_tools as na_tools

x_values = np.array([0, 1000, 2600, 4600, 6000])
z_values = np.array([-2600, -4000, -3200, -3600, -2400])
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

#plot the spline interpolation and add scatter ot the piont and the source location
spline_coefficients = natural_cubic_spline(x_values, z_values)
x_segment = np.linspace(0, 6000, 1000)

layer_values = np.array([evaluate_spline(x_values, spline_coefficients, x) for x in x_segment])
plt.plot(x_segment, layer_values[:, 0], label='Layer')
plt.scatter(x_values, z_values, color='red', label='Layer Points')
plt.scatter(3000, -2800, color='green', label='Source Location')
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.title('Layer of the wave speed')
plt.legend()
plt.show()