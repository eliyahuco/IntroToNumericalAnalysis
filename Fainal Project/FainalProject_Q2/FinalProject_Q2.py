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
import matplotlib.animation as animation


# Constants
k = 1.786 * 10 ** -3  # m^2/s
sigma_x = 0.00625  # meters
sigma_y = 0.00625  # meters
h = 0.05  # meters (\u0394x = \u0394y)
dt = 0.1  # seconds
Lx = 1.5  # meters
Ly = 1.5  # meters
T0 = 10  # [C]
alpha = (k * dt) / (h ** 2)

heat_sink = lambda x, y, t: -(10 ** (-4)) * np.exp(-((x - 1) ** 2 / (2 * sigma_x ** 2))) * np.exp(-((y - 0.5) ** 2 / (2 * sigma_y ** 2))) * np.exp(-0.1 * t)

# Grid
x = np.arange(0, Lx + h, h)
y = np.arange(0, Ly + h, h)
t = np.arange(0, 60 + dt, dt)

# Initial condition
T = np.zeros((len(x), len(y), len(t)))  # T(x,y,t) = T(i,j,n) - temperature at position (x,y) at time t

T[:, :, 0] = T0
T[y <= 0.8, 0, :] = (100 - 112.5 * y[y <= 0.8][:, None])
T[y > 0.8, 0, :] = 10
T[0, :, :] = 100

T[-1, : :] = 10

T[:, -1, :] = (100 - 60 * y[:, None])

def tridiagonal_matrix_algorithm(a, b, c, d):
    """
    Solve a tridiagonal matrix algorithm
    :param a: sub-diagonal in size n-1
    :param b: diagonal in size n
    :param c: super-diagonal in size n-1
    :param d: right-hand side in size n
    :return: solution of the tridiagonal matrix
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
    Solve the heat equation using the Alternating Direction Implicit (ADI) method
    :param T: temperature matrix
    :param alpha: alpha parameter
    :param heat_sink: heat sink function
    :param dt: time step
    :param x: x vector
    :param y: y vector
    :param n: time index
    :param t: time vector
    :return: updated temperature matrix
    """
    dt_2 = dt/2

    for j in range(1, len(y)-1):
        a = -(alpha/2) * np.ones(len(x) - 3)
        b = (1 + alpha) * np.ones(len(x) -2)
        c = -(alpha/2) * np.ones(len(x) - 3)
        d = np.zeros(len(x) - 2)
        for i in range(1, len(x) - 2):
            d[i-1]  = (1-alpha) * T[i, j, n] + (alpha/2) * (T[i, j + 1, n] + T[i, j - 1, n])  + dt_2 * heat_sink(x[i], y[j], t[n])

        T[1:-1, j, n + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    for i in range(1, len(x)-1):
        a = -(alpha/2) * np.ones(len(y) -3)
        b = (1 + alpha) * np.ones(len(y) - 2)
        c = -(alpha/2) * np.ones(len(y) - 3)
        d = np.zeros(len(y) - 2)
        for j in range(1, len(y) - 1):
            d[j-1] = (1-alpha) * T[i, j, n + 1] + (alpha/2) * (T[i + 1, j, n + 1] + T[i - 1, j, n + 1]) + dt_2 * heat_sink(x[i], y[j], t[n + 1])

        T[i, 1:-1, n + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return T

# Solve the heat equation

def main():
    # Constants
    k = 1.786 * 10 ** -3  # m^2/s
    sigma_x = 0.00625  # meters
    sigma_y = 0.00625  # meters
    h = 0.05  # meters (\u0394x = \u0394y)
    dt = 0.1  # seconds
    Lx = 1.5  # meters
    Ly = 1.5  # meters
    T0 = 10  # [C]
    alpha = (k * dt) / (h ** 2)

    heat_sink = lambda x, y, t: -(10 **(- 4)) * np.exp(-((x - 1) ** 2 / (2 * sigma_x ** 2))) * np.exp(
        -((y - 0.5) ** 2 / (2 * sigma_y ** 2))) * np.exp(-0.1 * t)

    # Grid
    x = np.arange(0, Lx + h, h)
    y = np.arange(0, Ly + h, h)
    t = np.arange(0, 60 + dt, dt)

    # Initial condition
    T = np.zeros((len(x), len(y), len(t)))  # T(x,y,t) = T(i,j,n) - temperature at position (x,y) at time t
    T[y <= 0.8, 0, :] = (100 - 112.5 * y[y <= 0.8][:, None])
    T[y > 0.8, 0, :] = 10
    T[0, :, :] = 100
    T[:, :, 0] = T0
    T[-1, ::] = 10

    T[:, -1, :] = (100 - 60 * y[:, None])

    print('Solving the heat equation using the Alternating Direction Implicit (ADI) method...\n')
    print(f'the grid size is {len(x)}x{len(y)}x{len(t)}\n')
    print(f'alpha = k*dt/h^2 = {alpha}\n')
    print(f'Time step = {dt} seconds\n')
    print(f'Total time = {t[-1]} seconds\n')
    print(f'Initial temperature = {T0} [C]\n')
    print(f'Heat sink function: f(x,y,t) = -(10^4)*exp(-((x-1)^2/(2*σ_x^2)))*exp(-((y-0.5)^2/(2*σ_y^2)))*exp(-0.1*t)\n')
    print('Boundary conditions:\n')
    print(f'T(X,0,t) = 100 [C]\n')
    print(f'T(X,1.5,t) = 10 [C]\n')
    print(f'T(1.5,Y,t) = 100 - 60y [C]\n')
    print(f'T(0,Y,t) = 100 - 112.5y [C] if 0<= y <= 0.8\n')
    print(f'T(0,Y,t) = 10 [C] if 0.8 < y <= 1.5\n')
    print('Solving the heat equation... it may take a while...\n')

    for n in range(T.shape[2] - 1):
        T = ADI_method(T, alpha, heat_sink, dt, x, y, n, t)

    # Plot the results in 2D

    for time in [15, 30, 60]:
        idx = int(time / dt)
        plt.figure(figsize=(6, 5))
        plt.imshow(T[:, :, idx], cmap='hot', origin='upper', extent=[0, Lx, 0, Ly])

        plt.title(f'Temperature distribution at t = {time} s')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        cbar = plt.colorbar()
        cbar.set_label('Temperature [C]')
        plt.savefig(f'temperature_distribution_t{time}s.png')
        plt.show()

    print('snapshots of the temperature distribution at t = 15, 30, 60 [s] saved as temperature_distribution_t15s.png, temperature_distribution_t30s.png, temperature_distribution_t60s.png\n')
    print('Creating animation... it may take a while...\n')
    # Function to update the frame

    fig, ax = plt.subplots()
    im = ax.imshow(T[:, :, 0], cmap='hot', origin='lower', extent=[0, Lx, 0, Ly])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature [C]')
    plt.gca().invert_yaxis()

    def animate(i):
        im.set_array(T[:, :, i])
        if i % 10 == 0:
            ax.set_title(f' Temperature distribution at t = {i * dt} s')

        return im,

    ani = animation.FuncAnimation(fig, animate, frames=T.shape[2], interval=100, blit=False)

    print('Creating animation...\n')
    ani.save('temperature_distribution.gif', writer='pillow', fps=10)
    print('Animation saved as temperature_distribution.gif')
    plt.show()

    print(f'\nsammry:\n')
    print(f'1. the grid size is {len(x)}x{len(y)}x{len(t)}\n')
    print(f'2. alpha = k*dt/h^2 = {alpha}\n')
    print(f'3. Time step = {dt} seconds\n')
    print(f'4. Total time = {t[-1]} seconds\n')
    print(f'5. Initial temperature = {T0} [C]\n')
    print(f'6. Heat sink function: f(x,y,t) = -(10^4)*exp(-((x-1)^2/(2*σ_x^2)))*exp(-((y-0.5)^2/(2*σ_y^2))*exp(-0.1*t)\n')
    print('7. Boundary conditions:\n')
    print(f'   T(X,0,t) = 100 [C]\n')
    print(f'   T(X,1.5,t) = 10 [C]\n')
    print(f'   T(1.5,Y,t) = 100 - 60y [C]\n')
    print(f'   T(0,Y,t) = 100 - 112.5y [C] if 0<= y <= 0.8\n')
    print(f'   T(0,Y,t) = 10 [C] if 0.8 < y <= 1.5\n')
    print(f'initial condition:\n')
    print(f'   T(x,y,0) = 10 [C]\n')
    print(f'snapshots of the temperature distribution at t = 15, 30, 60 [s] saved as temperature_distribution_t15s.png, temperature_distribution_t30s.png, temperature_distribution_t60s.png\n')
    print(f'Animation saved as temperature_distribution.gif\n')
    print(f'end of the program\n')
    print("thank you for using the program")

if __name__ == '__main__':
    main()