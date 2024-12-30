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
import math as m


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
x_axis = np.arange(0, x_max + h, h)  # x-axis
z_axis = np.arange(0, z_max + h, h)  # z-axis
x_line = np.linspace(0, 6000, 10000000)  # x values for the cubic spline interpolation of the layer
z_line = np.zeros_like(x_line)  # z values for the cubic spline interpolation of the layer

point_list = [(0, 2600), (1000, 4000), (2600, 3200), (4600, 3600), (6000, 2400)]  # points of the layer



# Create the grid


# Source function


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

def source_function(t, t_source_max=0.05):
    if t <= t_source_max:
        return t * np.exp(2 * np.pi * t) * np.sin(2 * np.pi * t)
    else:
        return 0

def laplace_operator_forth_order(u, dx=100, dz=100):
    """
    This function calculates the forth-order Laplace operator
    :param u: the wave function
    :param dx: the spatial step in x
    :param dz: the spatial step in z
    :return: the forth-order Laplace operator
    """
    for i in range(2, u.shape[0] - 2):
        for j in range(2, u.shape[1] - 2):
            u_xx = (-u[i + 2, j] + 16 * u[i + 1, j] - 30 * u[i, j] + 16 * u[i - 1, j] - u[i - 2, j]) / (12 * dx ** 2)
            u_zz = (-u[i, j + 2] + 16 * u[i, j + 1] - 30 * u[i, j] + 16 * u[i, j - 1] - u[i, j - 2]) / (12 * dz ** 2)
            u[i, j] = u_xx + u_zz
    return u

def wave_speed(x, z, c1, c2, a, b, c, d, x_i):
    """
    Calculate the wave speed at a given point (x, z) based on the layer's position.

    Parameters:
        x (float): x-coordinate of the point.
        z (float): z-coordinate of the point.
        c1 (float): Wave speed above the layer.
        c2 (float): Wave speed below the layer.
        a, b, c, d (arrays): Coefficients of the cubic spline interpolation.
        x_i (array): x-coordinates of the layer points.

    Returns:
        float: Wave speed at the point (x, z).
    """
    # Calculate the layer height (z-coordinate) at the given x using the cubic spline
    for i in range(len(x_i) - 1):
        if x_i[i] <= x <= x_i[i + 1]:
            z_layer = (
                a[i]
                + b[i] * (x - x_i[i])
                + c[i] * (x - x_i[i]) ** 2
                + d[i] * (x - x_i[i]) ** 3
            )
            break
    else:
        raise ValueError("x is out of the bounds of the layer points.")

    # Determine if the point is above or below the layer
    if z < z_layer:
        return c1
    else:
        return c2

def initialize_speed_field(X,Z,c1=2000,c2=3000,a=0,b=0,c=0,d=0,x_i=[0,1000,2600,4600,6000]):
    """
    This function initializes the speed field for the grid.
    Returns:
        2D numpy array: Speed field on the grid
    """
    speed_field = np.zeros_like(X)
    wave_speed_vectorized = np.vectorize(
        lambda x, z: wave_speed(x, z, c1, c2, a, b, c, d, x_i)
    )
    speed_field = wave_speed_vectorized(X, Z)
    return speed_field

def u_plus_1_next_step(u_n_minus_1, u_n, speed_field, dt, dx, dz,source_function):

    u_plus_1 = np.zeros_like(u_n)
    for i in range(2, u_n.shape[0] - 2):
        for j in range(2, u_n.shape[1] - 2):
            u_plus_1 = (2 * u_n[i, j] - u_n_minus_1[i, j] + (speed_field[i, j] ** 2 * dt ** 2) * laplace_operator_forth_order(u_n, dx, dz)
                        + source_function(dt,t_source_max)*dt**2)
    return u_plus_1



def animate_wave(U_snapshots, dt, dx, save_path="wave_animation.mp4"):
    """
    Create an animation of the wave field.

    Parameters:
        U_snapshots (list of np.ndarray): List of 2D wave field snapshots at different time steps.
        dt (float): Time step size.
        dx (float): Grid spacing.
        save_path (str): File path to save the animation.
    """
    # adding the spline interpolation of the layer
    x_i = [point[0] for point in point_list]
    y_i = [point[1] for point in point_list]
    a, b, c, d = cubic_spline_interpolation(x_i, y_i)
    x = np.linspace(0, x_max, 1000)
    y = np.linspace(0, z_max, 1000)

    for i in range(len(x_i) - 1):
        mask = (x >= x_i[i]) & (x <= x_i[i + 1])
        y[mask] = a[i] + b[i] * (x[mask] - x_i[i]) + c[i] * (x[mask] - x_i[i]) ** 2 + d[i] * (x[mask] - x_i[i
        ]) ** 3


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, label='Cubic Spline Interpolation', color='black', linewidth=2)



    # Create a mesh grid for plotting
    n, m = U_snapshots[0].shape
    x = np.linspace(0, (m - 1) * dx, m)
    y = np.linspace(0, (n - 1) * dx, n)
    X, Y = np.meshgrid(x, y)



    # Initialize the plot with the first snapshot
    wave_plot = ax.pcolormesh(X, Y, U_snapshots[0], shading='auto', cmap='coolwarm')
    ax.set_title("Wave Field Animation for {} s".format(len(U_snapshots) * dt))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    cbar = plt.colorbar(wave_plot, ax=ax)
    cbar.set_label("displacement (m)")
    plt.scatter(x_i, y_i, color='black', label='Layer Points')
    plt.scatter(x_source, z_source, color='red', label='Source Point (3000, 2800)', marker='*', s=150)
    plt.gca().invert_yaxis()
    ax.legend()

    def update(frame):
        """Update function for animation."""
        wave_plot.set_array(U_snapshots[frame].ravel())
        ax.set_title(f"Wave Field at t = {frame * dt:.3f} s")
        return wave_plot,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(U_snapshots), interval=100, blit=True)

    # Save the animation
    print(f'\nSaving animation at "{save_path}" in progress...')
    ani.save(save_path, writer="ffmpeg")
    print(f'Animation saved at "{save_path}"\n')


def main():
    # initialize the grid
    X, Z = np.meshgrid(x_axis, z_axis)

    # Initialize the speed field with the cubic spline interpolation
    x_i = [point[0] for point in point_list]
    y_i = [point[1] for point in point_list]
    a, b, c, d = cubic_spline_interpolation(x_i, y_i)
    x = np.linspace(0, x_max, 1000)
    y = np.linspace(0, z_max, 1000)

    for i in range(len(x_i) - 1):
        mask = (x >= x_i[i]) & (x <= x_i[i + 1])
        y[mask] = a[i] + b[i] * (x[mask] - x_i[i]) + c[i] * (x[mask] - x_i[i]) ** 2 + d[i] * (x[mask] - x_i[i]) ** 3
    speed_field = initialize_speed_field(X,Z,c1,c2,a,b,c,d,x_i)

    # initialize the wave field
    u_n_plus_1 = np.zeros_like(X, dtype=float)
    u_n = np.zeros_like(X, dtype=float)
    u_n_minus_1 = np.zeros_like(X, dtype=float)
    dt = [dt1, dt2]



    plt.figure(figsize=(10, 6), dpi=100)
    plt.pcolormesh(X, Z, speed_field, cmap="coolwarm", shading='auto')
    plt.colorbar(label="Wave velocity (m/s)")
    plt.plot(x, y, label='Cubic Spline Interpolation', color='black', linewidth=2)
    plt.scatter(x_i, y_i, color='black', label='Layer Points')
    plt.scatter(x_source, z_source, color='red', label='Source Point (3000, 2800)', marker='*', s=100)
    plt.xlabel("x (m)", fontsize=12)
    plt.ylabel("z (m)", fontsize=12)
    plt.title("Wave velocity field", fontsize=14)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.savefig("wave_velocity_field_and_layer.png")
    plt.show()

    plt.close()

    u_n_list_dt_1 = []
    u_n_list_dt_2 = []
    plt.figure(figsize=(10, 8), dpi=100)
    for d in range( len(dt)):

        u_n[28, 30] = source_function(dt[d], t_source_max)
        for t in np.arange(2*dt[d], t_max + dt[d] , dt[d]):
            t = round(t, 2)
            # Update the wave field
            for i in range(2, u_n.shape[0] - 2):
                for j in range(2, u_n.shape[1] - 2):

                    u_n_plus_1[i,j] = 2*u_n[i,j] - u_n_minus_1[i,j] + (speed_field[i,j]**2 * dt[d]**2)/(12*h**2) * (-u_n[i+2,j] + 16*u_n[i+1,j] - 60*u_n[i,j] + 16*u_n[i-1,j] - u_n[i-2,j] -u_n[i,j+2] + 16*u_n[i,j+1]  + 16*u_n[i,j-1] - u_n[i,j-2])+ source_function(t,t_source_max)*dt[d]**2
            u_n_minus_1 = u_n.copy()
            u_n = u_n_plus_1.copy()
            u_n_plus_1 = np.zeros_like(X, dtype=float)

            if dt[d] == dt1:
                u_n_list_dt_1.append(u_n)
            else:
                u_n_list_dt_2.append(u_n)


            # Plot the wave field
            # if t in [0.15, 0.4, 0.7, 1] and dt[d] == dt1:
            plt.clf()

            plt.pcolormesh(X, Z, u_n, cmap='coolwarm', shading='auto')
            plt.colorbar(label='Displacement (m)', orientation='vertical', pad=0.01, aspect=20, shrink=0.8)
            plt.plot(x, y, label='Cubic Spline Interpolation', color='black', linewidth=2)
            plt.scatter(x_i, y_i, color='purple', label='Layer Points')
            plt.scatter(x_source, z_source, color='red', label='Source Point (3000, 2800)', marker='*', s=150)
            plt.xlabel('x (m)', fontsize=12)
            plt.ylabel('z (m)', fontsize=12)
            plt.title(f'Wave Field at time t={t:.2f} s', fontsize=14)
            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid()
            if t in [0.15, 0.4, 0.7, 1] and dt[d] == dt1:
                # save the plot
                plt.savefig(f'Wave_Field_for_dt_{dt[d]}_at t = {t}_.png')
            elif t in [0.15, 0.3, 0.6, 0.9] and dt[d] == dt2:
                # save the plot
                plt.savefig(f'Wave_Field_for_dt_{dt[d]}_at t = {t}.png')

            plt.pause(dt1)

            # save the plot as animation

        plt.show()
        plt.close()



        # Create the animation

        if dt[d] == dt1:
            animate_wave(u_n_list_dt_1, dt[d], h, save_path="wave_animation_dt_0.01.gif")
        else:
            animate_wave(u_n_list_dt_2, dt[d], h, save_path="wave_animation_dt_0.03.gif")




if __name__ == '__main__':
    main()






# t = np.linspace(0, t_max, int(t_max / dt1) + 1)
# F = np.array([source_function(time) for time in t])
# print(F)
# print(F.shape)
#
# # Initialize the wave field
# U_old = np.zeros_like(X)
# U = np.zeros_like(X)
# U_new = np.zeros_like(X)
# print(len(U))
# print(U[:,:].shape)
# #solve the wave equation
# for i in range(1, len(t)):
#     U_new = wave_equation_solver_explisit_next_step(U, U_new, U_old, F[i], dt1, h, h)
#     U_old = U.copy()
#     U = U_new.copy()
#
#
#
#     # Plot the wave field
#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(X, Z, U, cmap='coolwarm')
#     plt.colorbar()
#     plt.xlabel('x (m)', fontsize=12)
#     plt.ylabel('z (m)', fontsize=12)
#     plt.title(f'Wave Field at time t={time:.2f} s', fontsize=14)
#     plt.grid()
#     plt.show()


#
# import numpy as np
# import math
# import matplotlib.pyplot as plt
#
# # Constants
# n = 5  # Number of known points
# x_points = 61  # Size of the matrix (number of x points)
# z_points = 61  # Size of the matrix (number of z points)
# dx = 100  # Δx = 10m
# dz = 100  # Δz = 10m
# v1 = 2000  # Velocity v1 = 2000 m/s
# v2 = 3000  # Velocity v2 = 3000 m/s
# dt = 0.01
# pi = math.pi
# eps = 1e-5
#
# # Functions
# def cubic_spline(x, y):
#     """Compute cubic spline coefficients for given x and y values."""
#     h = np.diff(x)
#     n = len(x)
#
#     # Build the tridiagonal system
#     b = np.zeros((n - 2, n - 2))
#     m = np.zeros(n)
#     for i in range(1, n - 1):
#         m[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
#
#     b[0, 0] = 2 * (h[0] + h[1])
#     b[0, 1] = h[1]
#     for i in range(1, n - 3):
#         b[i, i - 1] = h[i]
#         b[i, i] = 2 * (h[i] + h[i + 1])
#         b[i, i + 1] = h[i + 1]
#     b[-1, -2] = h[-2]
#     b[-1, -1] = 2 * (h[-2] + h[-1])
#
#     # Solve for second derivatives
#     m[1:-1] = np.linalg.solve(b, m[1:-1])
#
#     # Compute spline coefficients
#     s = np.zeros((n - 1, 4))
#     for i in range(n - 1):
#         s[i, 0] = (m[i + 1] - m[i]) / (6 * h[i])
#         s[i, 1] = m[i] / 2
#         s[i, 2] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * m[i] + m[i + 1]) / 6
#         s[i, 3] = y[i]
#
#     return s
#
# def interpolate_spline(x, s, x_vals):
#     """Interpolate using cubic spline coefficients."""
#     z_vals = np.zeros_like(x_vals)
#     for k, x_val in enumerate(x_vals):
#         for i in range(len(s)):
#             if x[i] <= x_val <= x[i + 1]:
#                 dx = x_val - x[i]
#                 z_vals[k] = (s[i, 0] * dx**3 + s[i, 1] * dx**2 + s[i, 2] * dx + s[i, 3])
#                 break
#     return z_vals
#
#
#
# # Initialize grids
# next_phase = np.zeros((x_points, z_points))
# cur_phase = np.zeros((x_points, z_points))
# prev_phase = np.zeros((x_points, z_points))
# x_vec = np.array([0, 1000, 2600, 4600, 6000])
# z_vec = np.array([2600, 4000, 3200, 3600, 2400])
# x_vals = np.arange(0, x_points * dx, dx)
#
# # Compute spline and fault line
# spline_coeffs = cubic_spline(x_vec, z_vec)
# z_fault = interpolate_spline(x_vec, spline_coeffs, x_vals)
#
# # Source location
# source_x_index = int(3000 / dx)
# source_z_index = int(2800 / dz)
#
# # Temporal propagation of the sound wave
# for t in np.arange(0, 1 + dt, dt):
#     for i in range(2, x_points - 2):
#         for j in range(2, z_points - 2):
#             # Determine velocity based on layer
#             velocity = v1 if i * dx <= z_fault[j] else v2
#
#             # 4th-order spatial derivative and 2nd-order time update
#             next_phase[i, j] = (
#                 (velocity**2 * dt**2) / (12 * dx**2) * (
#                     16 * (cur_phase[i + 1, j] + cur_phase[i - 1, j])
#                     - cur_phase[i + 2, j] - cur_phase[i - 2, j]
#                     + 16 * (cur_phase[i, j + 1] + cur_phase[i, j - 1])
#                     - cur_phase[i, j + 2] - cur_phase[i, j - 2] - 60 * cur_phase[i, j]
#                 )
#                 + 2 * cur_phase[i, j] - prev_phase[i, j]
#             )
#
#     print(next_phase)
#     # Add source term
#     if t <= 0.05:
#         next_phase[source_x_index, source_z_index] += t * math.exp(2 * pi * t) * math.sin(2 * pi * t)
#
#     # Update phases
#     prev_phase, cur_phase, next_phase = cur_phase, next_phase, prev_phase
#
#     # plot the wave field
#
#     if abs(t - 0.15) < eps or abs(t - 0.4) < eps or abs(t - 0.7) < eps or abs(t - 1) < eps:
#
#         plt.figure(figsize=(10, 6))
#         plt.imshow(cur_phase.T, cmap='coolwarm', extent=[0, 6000, 0, 6000])
#         # add units fo the colorbar
#         plt.colorbar(label='Amplitude')
#         # flliping the y axis
#         plt.gca().invert_yaxis()
#
#         plt.title(f'Wave Field at time t={t:.2f} s', fontsize=14)
#         plt.xlabel('x (m)', fontsize=12)
#         plt.ylabel('z (m)', fontsize=12)
#         plt.grid()
#         plt.show()
#


# u = U
# u = np.random.random((61,61))
# print(u.shape)
#
# for i,j in np.ndindex(u.shape):
#     u[i,j] = np.random.random()
#
# print(u)
# print(laplace_operator_forth_order(u, 100, 100)[2:-2,2:-2])