"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 1 in the final project for the course intro to numerical analysis

given an acoustic model with a uniform density and a source (that produces a waves) in at x = 3000 meters, z = 2800 meters

the domain is defined as:
0<=x<=6000 meters
0<=z<=6000 meters

there is a layer that defines the speed of the wave:
the layer given by series of points:
(x,z) = (0,2600), (1000,4000), (2600,3200), (4600,3600), (6000,2400)
the speed of the wave depends on the location of the wave with respect to the layer:
above the layer, the speed of the wave is c1 = 2000 m/s, and below the layer, the speed of the wave is c2 = 3000 m/s

the objective of this script is to solve a problem of the wave equation in 2D space:
    ∂^2u/∂t^2 = c^2(∂^2u/∂x^2 + ∂^2u/∂z^2) + F(x,z,t)

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

we will solve the wave equation in the explicit method


missions:

1) To represent the separation layer using cubic spline interpolation.
   *** Ensure that the velocity model is correct before solving the wave equation ***
2) Calculate the progression of the wave field in the medium using 4th-order finite difference for spatial steps and 2nd-order for time.
    *** Space step: Δx = Δz = 100 m, Time steps: Δt = 0.01 s and Δt = 0.03 s ***
    how the solution behaves with the time step Δt?
3) Show snapshots of the wave field at times
for Δt = 0.01 s at t = 0.15 s, 0.4 s, 0.7 s, 1 s
and
for Δt = 0.03 s at t = 0.15 s, 0.3 s, 0.6 s, 0.9 s .
4) Create a complete animation of the wave field for Δt = 0.01 s.

---------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#craete the cubic spline interpolation function
import numpy as np


def cubic_spline_interpolation(x_i, y_i):
    """
    Cubic spline interpolation for a given set of data points (x_i, y_i).

    Parameters:
    x_i : array_like
        x-coordinates of the data points.
    y_i : array_like
        y-coordinates of the data points.

    Returns:
    a, b, c, d : tuple
        Coefficients of the cubic spline.
    """
    n = len(x_i)
    h = np.diff(x_i)  # Step sizes between x_i
    b = np.diff(y_i) / h  # Slope between y_i values

    # Set up the tridiagonal system
    u = np.zeros(n)
    v = np.zeros(n)
    u[1:-1] = 2 * (h[:-1] + h[1:])  # Diagonal elements of the system
    v[1:-1] = 6 * (b[1:] - b[:-1])  # Right-hand side

    # Solve the tridiagonal system using forward elimination
    for i in range(1, n - 1):
        factor = h[i - 1] / u[i - 1]
        u[i] -= factor * h[i - 1]
        v[i] -= factor * v[i - 1]

    # Back-substitution to compute z (second derivatives)
    z = np.zeros(n)
    for i in range(n - 2, 0, -1):
        z[i] = (v[i] - h[i] * z[i + 1]) / u[i]

    # Compute the coefficients for the cubic spline
    a = y_i[:-1]
    c = z[:-1]
    d = (z[1:] - z[:-1]) / (3 * h)
    b = b - h * (2 * z[:-1] + z[1:]) / 3

    return a, b, c, d



# Define the data points for x and z axes
x_i = np.array([0, 1000, 2600, 4600, 6000])
z_i = np.array([2600, 4000, 3200, 3600, 2400])
