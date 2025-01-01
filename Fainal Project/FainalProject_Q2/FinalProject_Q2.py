"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Description:

This script is question 2 in the final project for the course intro to numerical analysis
in Tel Aviv University.

The script solves an non-homogeneous Heat Equation with implicit finite difference method.

de domain is a square with the following boundaries:
    x = 0, x = 1.5, y = 0, y = 1.5
    origin at (0,0) is at the left upper corner, while the positive x-axis is to the right and the positive y-axis is down

the equation is:
    dT/dt = κ *(d^2u/dx^2 + d^2u/dy^2) + f(x,y,t)

    where f(x,y,t) = -10^4*exp(-((x-1)^2/2*σ_x^2)*exp(-((y-0.5)^2/2*σ_y^2)*exp(-0.1*t)

    while σ_x = σa_y = 0.00625 meters

    initial condition:
             κ = 1.786*10^-3 m^2/s
             T(x,y,0) = 10 [C]

    boundary conditions (for the first 60 seconds):
            T(X,0,t) = 100 [C]
            T(X,1.5,t) = 10 [C]
            T(1.5,Y,t) = 100 -60y [C]

                        { 100 -112.5y [C] if 0<= y <= 0.8
            T(0,Y,t) = <
                        { 10 [C] if 0.8 < y <= 1.5


    we will solve the wave equation in the implicit method, using the ADI method.
    we will use Finite Difference 2nd order for the spatial derivatives, and 1st order for the time derivative.
    we will use the following discretization:
        ∂^2T/∂x^2 ≈ (T(i+1,j) - 2T(i,j) + T(i-1,j))/Δx^2
        ∂^2T/∂y^2 ≈ (T(i,j+1) - 2T(i,j) + T(i,j-1))/Δy^2
        ∂T/∂t ≈ (T(i,j,n+1) - T(i,j,n))/dt
        where Δx = Δy = 0.05 [m], Δt = 0.1 [s]

missions:

1. solve the equation for 0<= t <= 60 [s]
2. Show snapshots of the temperature distribution at t = 15, 30, 60 [s]
3. 4) Create a complete animation of the temperature distribution for 0<= t <= 60 [s] while we show for every 10 time-steps.

---------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Constants
k = 1.786 * 10 ** -3  # m^2/s
sigma_x = 0.00625  # meters
sigma_y = 0.00625  # meters
h = 0.05  # meters (Δx = Δy)
dt = 0.1  # seconds
Lx = 1.5  # meters
Ly = 1.5  # meters
T0 = 10  # [C]
alpha = (k * dt) / (h ** 2)

heat_sink =  lambda x, y, t: -(10 ** 4) * np.exp(-((x - 1) ** 2 / (2 * sigma_x ** 2))) * np.exp(-((y - 0.5) ** 2 / (2 * sigma_y ** 2))) * np.exp(-0.1 * t)



# Grid
x = np.arange(0, Lx + h, h)
y = np.arange(0, Ly + h, h)
t = np.arange(0, 60 + dt, dt)

# Initial condition
T = np.zeros((len(x), len(y), len(t)))  # T(x,y,t) = T(i,j,n) - temperature at position (x,y) at time t
T[:, :, 0] = T0
T[:, 0, :] = 100
T[:, -1, :] = 10
T[-1, :, :] = (100 - 60 * y[:, None])

T[-1, y <= 0.8, :] = (100 - 112.5 * y[y <= 0.8][:, None])



T[-1, y > 0.8, :] = 10

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

def ADI_method(T, alpha, heat_sink, dt, x, y, n, t):
    """
    Solves the heat equation using the Alternating Direction Implicit (ADI) method.
    :param T: 3D temperature matrix (x, y, t)
    :param alpha: Diffusion coefficient
    :param heat_sink: Heat sink function (x, y, t)
    :param dt: Time step size
    :param x: Grid points in the x direction
    :param y: Grid points in the y direction
    :param n: Current time index
    :param t: Time array
    :return: Updated temperature matrix T after one time step
    """

    # Implicit in x direction
    for j in range(1, len(y) - 1):
        a = -alpha * np.ones(len(x) - 2)
        b = (1 + 2 * alpha) * np.ones(len(x) - 2)
        c = -alpha * np.ones(len(x) - 2)
        d = np.zeros(len(x) - 2)

        for i in range(1, len(x) - 1):
            d[i - 1] = (T[i, j, n] +
                        alpha * (T[i, j + 1, n] - 2 * T[i, j, n] + T[i, j - 1, n]) +
                        dt * heat_sink(x[i], y[j], t[n]))

        T[1:-1, j, n + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    # Implicit in y direction
    for i in range(1, len(x) - 1):
        a = -alpha * np.ones(len(y) - 2)
        b = (1 + 2 * alpha) * np.ones(len(y) - 2)
        c = -alpha * np.ones(len(y) - 2)
        d = np.zeros(len(y) - 2)

        for j in range(1, len(y) - 1):
            d[j - 1] = (T[i, j, n + 1] +
                        alpha * (T[i + 1, j, n + 1] - 2 * T[i, j, n + 1] + T[i - 1, j, n + 1]))

        T[i, 1:-1, n + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return T


# Solve the heat equation
for n in range(1, len(t)):
    T = ADI_method(T, alpha, heat_sink, dt, x, y, n - 1, t)

# Plot the results
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, T[:, :, 0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature [C]')
ax.set_title('Temperature distribution at t = 0 [s]')

def update_plot(n, T, ax):
    ax.clear()
    surf = ax.plot_surface(X, Y, T[:, :, n], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature [C]')
    ax.set_title(f'Temperature distribution at t = {t[n]:.1f} [s]')
    return surf

# Show snapshots of the temperature distribution at t = 15, 30, 60 [s]
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
update_plot(15, T, ax)
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
update_plot(30, T, ax)
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
update_plot(-1, T, ax)
plt.show()

# Create a complete animation of the temperature distribution for 0<= t <= 60 [s] while we show for every 10 time-steps.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ani = animation.FuncAnimation(fig, update_plot, frames=range(0, len(t), 10), fargs=(T, ax), repeat=False)
ani.save('heat_equation.gif', writer='imagemagick', fps=5)

plt.show()

