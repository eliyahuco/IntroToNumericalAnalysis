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

# Define the points for cubic spline interpolation
x_values = np.array([0, 1000, 2600, 4600, 6000])
z_values = np.array([2600, 4000, 3200, 3600, 2400])

# Tridiagonal matrix algorithm for solving the system in cubic spline interpolation
def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])
    for i in range(1, n):
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

# Natural cubic spline function
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

    return np.array([a, b, c[:-1], d]).T

# Evaluate spline at a point xi
def evaluate_spline(x, spline_coeffs, xi):
    i = np.searchsorted(x, xi) - 1
    i = np.clip(i, 0, len(spline_coeffs) - 1)
    dx = xi - x[i]
    a, b, c, d = spline_coeffs[i]
    return a + b * dx + c * dx ** 2 + d * dx ** 3

# Define the source function
def source_function(t):
    if t <= 0.05:
        return t * np.exp(2 * np.pi * t) * np.sin(2 * np.pi * t)
    else:
        return 0

# Get wave speed based on location and spline interpolation
def wave_speed_function(x, z, spline_coefficients):
    layer = evaluate_spline(x_values, spline_coefficients, x)
    return 2000 if z <= layer else 3000

# Define 4th-order finite difference laplacian
def laplacian_4th_order(u, dx):
    laplacian_u = np.zeros_like(u)
    for i in range(2, u.shape[0] - 2):
        for j in range(2, u.shape[1] - 2):
            laplacian_u[i, j] = (
                - (1 / 12) * (u[i - 2, j] + u[i + 2, j] + u[i, j - 2] + u[i, j + 2])
                + (4 / 3) * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1])
                - (5 / 2) * u[i, j]
            ) / dx ** 2
    return laplacian_u

# Main wave propagation loop
def update_wave(u, dx, dz, dt, t_max, x_grid, z_grid, spline_coefficients, x_max, z_max):
    nt = int(t_max / dt) + 1
    for n in range(1, nt):
        lap_u = laplacian_4th_order(u[:, :, 1], dx) + laplacian_4th_order(u[:, :, 1], dz)

        for i in range(2, u.shape[0] - 2):
            for j in range(2, u.shape[1] - 2):
                c = wave_speed_function(x_grid[i], z_grid[j], spline_coefficients)
                u[i, j, 2] = (2 * u[i, j, 1] - u[i, j, 0]
                              + dt ** 2 * c ** 2 * lap_u[i, j]
                              + dt ** 2 * source_function(n * dt) if i == 30 and j == 28 else 0)

        # Apply boundary conditions (simple reflecting)
        u[:, 0, 2] = u[:, 1, 2]
        u[:, -1, 2] = u[:, -2, 2]
        u[0, :, 2] = u[1, :, 2]
        u[-1, :, 2] = u[-2, :, 2]

        # Shift time steps
        u[:, :, 0] = u[:, :, 1]
        u[:, :, 1] = u[:, :, 2]

        # Save snapshots at specific times
        if np.isclose(n * dt, 0.15, atol=dt) or np.isclose(n * dt, 0.4, atol=dt) or np.isclose(n * dt, 0.7, atol=dt) or np.isclose(n * dt, 1.0, atol=dt):
            plt.imshow(u[:, :, 1], cmap='seismic', extent=[0, x_max, 0, z_max])
            plt.colorbar(label='Wave Amplitude')
            plt.title(f"Wave Field at t = {n * dt:.2f} seconds")
            plt.show()

# Function to animate the wave
def animate_wave(u, dt, x_max, z_max, nt):
    fig, ax = plt.subplots()
    cax = ax.imshow(u[:, :, 1], cmap='seismic', extent=[0, x_max, 0, z_max], animated=True)
    fig.colorbar(cax, label='Wave Amplitude')

    def update(frame):
        # Ensure correct shape for the frame data
        cax.set_array(u[:, :, 1].flatten())
        return cax,

    ani = animation.FuncAnimation(fig, update, frames=range(nt), blit=True)
    plt.show()

# Main function
def main():
    # Grid setup
    dx = dz = 100  # meters
    dt1 = 0.01  # seconds
    x_max, z_max = 6000, 6000
    t_max1 = 1.0
    x_grid = np.arange(0, x_max + dx, dx)
    z_grid = np.arange(0, z_max + dz, dz)
    nx, nz = len(x_grid), len(z_grid)
    nt1 = int(t_max1 / dt1) + 1

    # Initialize wave field u(x, z, t)
    u = np.zeros((nx, nz, 3))  # 3 time levels: n-1, n, n+1

    # Get spline coefficients for the layer
    spline_coefficients = natural_cubic_spline(x_values, z_values)

    # Update wave field
    update_wave(u, dx, dz, dt1, t_max1, x_grid, z_grid, spline_coefficients, x_max, z_max)

    # Animate wave field
    animate_wave(u, dt1, x_max, z_max, nt1)

if __name__ == '__main__':
    main()
    print("\nThe script has finished running.")