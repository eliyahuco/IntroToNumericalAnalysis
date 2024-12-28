"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 1 in the final project for the course intro to numerical analysis

given an acoustic model with a uniform density and a source (that produces a waves) in at x = 3000 meters, z = 2800 meters

the domain is defined as:
origin at (0,0) is at the left upper corner, while the positive x-axis is to the right and the positive z-axis is down
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
we will use the following discretization:
    ∂^2u/∂x^2 ≈ (-u(x+2Δx,z) + 16u(x+Δx,z) - 30u(x,z) + 16u(x-Δx,z) - u(x-2Δx,z))/(12*Δx^2)
    ∂^2u/∂z^2 ≈ (-u(x,z+2Δz) + 16u(x,z+Δz) - 30u(x,z) + 16u(x,z-Δz) - u(x,z-2Δz))/(12*Δz^2)
    ∂^2u/∂t^2 ≈ (u(x,z,t+Δt) - 2u(x,z,t) + u(x,z,t-Δt))/Δt^2
    where Δx = Δz = h is the spatial step and Δt is the time step

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
import numerical_analysis_methods_tools as na_tools


# Given parameters
c1 = 2000  # speed of the wave above the layer
c2 = 3000  # speed of the wave below the layer
h = 100  # spatial step in meters (Δx = Δz)
dt1 = 0.01  # time step in seconds
dt2 = 0.03  # time step in seconds
t_max = 1  # maximum time in seconds
t_source_max = 0.05  # maximum time for the source function in seconds
x_source = 3000  # source location in meters
z_source = 2800  # source location in meters
x_max = 6000  # maximum x in meters
z_max = 6000  # maximum z in meters

point_list = [(0, 2600), (1000, 4000), (2600, 3200), (4600, 3600), (6000, 2400)]  # points of the layer

# Source function
def source_function(t):
    if t <= t_source_max:
        return t * np.exp(-2 * np.pi * t) * np.sin(2 * np.pi * t)
    else:
        return 0

def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    This function solves a tridiagonal matrix using the Thomas algorithm
    :param a: the lower diagonal of the matrix
    :param b: the main diagonal of the matrix
    :param c: the upper diagonal of the matrix
    :param d: the right-hand side of the equation
    :return: the solution of the matrix
    """

    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom

    d_[-1] = (d[-1] - a[-2] * d_[-2]) / (b[-1] - a[-2] * c_[-2])

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

def cubic_spline_interpolation(x_i, y_i):
    """
    This function calculates the cubic spline interpolation of a given set of points
    :param x_i: the x values of the points
    :param y_i: the y values of the points
    :return: the coefficients of the cubic spline
    """
    n = len(x_i)
    h = np.diff(x_i)

    a = y_i[:-1]
    alpha = np.zeros(n - 1)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i] * (y_i[i + 1] - y_i[i]) - 3 / h[i - 1] * (y_i[i] - y_i[i - 1]))  # coeffs for cubic spline

    l = np.ones(n)
    mu = np.zeros(n - 1)
    z = np.zeros(n)

    l[0] = 1
    for i in range(1, n - 1):
        l[i] = 2 * (x_i[i + 1] - x_i[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    l[-1] = 1
    z[-1] = 0

    b = np.zeros(n - 1)
    c = np.zeros(n)
    d = np.zeros(n - 1)

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_i[j + 1] - y_i[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return a, b, c, d

# Create the cubic spline interpolation of the layer
x_i = [point[0] for point in point_list]
y_i = [ point[1] for point in point_list]  # Flip z-values to adjust for the layer (coordinates change)
a, b, c, d = cubic_spline_interpolation(x_i, y_i)

print('Cubic Spline Interpolation Coefficients:')
for i in range(len(a)):
    print(f'a_{i} = {a[i]}, b_{i} = {b[i]}, c_{i} = {c[i]}, d_{i} = {d[i]}')


# Plot the cubic spline interpolation of the layer
x = np.linspace(0, x_max, 1000)
y = np.linspace(0, z_max, 1000)
for i in range(len(x_i) - 1):
    mask = (x >= x_i[i]) & (x <= x_i[i + 1])
    y[mask] = a[i] + b[i] * (x[mask] - x_i[i]) + c[i] * (x[mask] - x_i[i]) ** 2 + d[i] * (x[mask] - x_i[i   ]) ** 3

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Cubic Spline Interpolation', color='brown', linewidth=2)
plt.scatter(x_i, y_i, color='black', label='Layer Points')
plt.scatter(x_source,  z_source, color='red', label='Source Point (3000, 2800)', marker='*', s=100)

# Flip the z-axis to make the values increase downward
plt.gca().invert_yaxis()

plt.xlabel('x (m)', fontsize=12)
plt.ylabel('z (m)', fontsize=12)
plt.title('Cubic Spline Interpolation of the Layer', fontsize=14)
plt.legend()
plt.grid()
plt.show()


# Solve the wave equation using the explicit method
def second_order_derivative_for_time(u, dt):
    """
    This function calculates the second-order derivative for time
    :param u: the wave function
    :param dt: the time step
    :return: the second-order derivative for time
    """
    return (u[2:] - 2 * u[1:-1] + u[:-2]) / dt ** 2

def forward_second_order_difference_for_time(u, dt):
    """
    This function calculates the forward second-order difference for time
    :param u: the wave function
    :param dt: the time step
    :return: the forward second-order difference for time
    """
    return (-3 * u[0:-2] + 4 * u[1:-1] - u[2:]) / (2 * dt)



def fourth_order_finite_difference_laplace(u, dx = 100, dz = 100):
    """
    This function calculates the fourth-order finite difference for the Laplace operator
    :param u: the wave function
    :param dx: the spatial step in x
    :param dz: the spatial step in z
    :return: the fourth-order finite difference for the Laplace operator
    """
    return (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]) / (12 * dx ** 2) + \
           (-u[2 * len(u):] + 16 * u[len(u):] - 30 * u + 16 * u[:-len(u)] - u[:-2 * len(u)]) / (12 * dz ** 2)

# check if the points are above or below the layer
def wave_speed(x, z):
    """
    This function calculates the wave speed based on the location of the wave
    :param x: the x-coordinate
    :param z: the z-coordinate
    :return: the wave speed
    """
    if z >= a[0] + b[0] * x + c[0] * x ** 2 + d[0] * x ** 3:
        return c1
    else:
        return c2

# Create the grid


def wave_equation_solver_explisit_next_step(u, u_new, u_old, F, dt, dx, dz):
    """
    This function calculates the next step of the wave equation using the explicit method
    :param u: the wave function
    :param u_new: the new wave function
    :param u_old: the old wave function
    :param F: the source function
    :param dt: the time step
    :param dx: the spatial step in x
    :param dz: the spatial step in z
    :return: the new wave function
    """
    c = wave_speed(X, Z)
    u_new[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - u_old[1:-1, 1:-1] + c ** 2 * dt ** 2 * (
            fourth_order_finite_difference_laplace(u, dx, dz) + F)
    return u_new

# Create the grid
X, Z = np.meshgrid(np.arange(0, x_max + h, h), np.arange(0, z_max + h, h))
U = np.zeros_like(X)
U_new = np.zeros_like(X)
U_old = np.zeros_like(X)

t = np.linspace(0, t_max, int(t_max / dt1) + 1)
def initialize_source_function(t):
    F = np.zeros_like(X)
    if t <= t_source_max:
        F = t * np.exp(-2 * np.pi * t) * np.sin(2 * np.pi * t)
    return F
for time in t:
    F = source_function(time)
    print(F)


# solve the wave equation
for time in t:
    F = initialize_source_function(time)
    U_new = wave_equation_solver_explisit_next_step(U, U_new, U_old, F, dt1, h, h)
    U_old = U.copy()
    U = U_new.copy()

    # Plot the wave field
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, Z, U, cmap='coolwarm')
    plt.colorbar()
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('z (m)', fontsize=12)
    plt.title(f'Wave Field at time t={time:.2f} s', fontsize=14)
    plt.grid()
    plt.show()
