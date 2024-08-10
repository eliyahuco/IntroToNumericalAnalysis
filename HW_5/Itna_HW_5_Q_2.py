"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 2 in HW_5 for the course intro to numerical analysis

the objective of this script is to solve the following ODE:

m * d^2y/dt^2 + ky = 0

    where m is the mass, k is the spring constant and y is the displacement
    m = 2 kg
    k = 40 N/m
    y(0) = 0.7 meters
    t_max = 2.5 seconds

    middle parameter:
    v = dy/dt
    v0 = dy/dt(0) = 0
    dv/dt = -(k*y)/m

we will solve the ODE using the following methods:
1) Euler's method with step size of: 0.05 seconds, 0.1 seconds
2) Runge-Kutta fourth order method (RK4) with step size of: 0.1 seconds
3) Leapfrog method with step size of: 0.1 seconds

analytical solution:
y(t) = y(0) * cos(ω*t) + v0/ω * sin(ω*t)
v(t) = -y(0) * ω * sin(ω*t) + v0 * cos(ω*t)

    where ω = sqrt(k/m)

we will compare the results of the methods and with the analytical solution
also we will plot the results

---------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# Euler method for second-order ODE
def euler_method_2nd_order(t0, y0, v0, h, tmax, m, k):
    t_values = np.arange(t0, tmax + h, h)
    y_values = [y0]
    v_values = [v0]

    y = y0
    v = v0

    for t in t_values[:-1]:
        v_new = v + h * (-k / m * y)
        y_new = y + h * v
        y = y_new
        v = v_new

        y_values.append(y)
        v_values.append(v)

    return t_values, np.array(y_values), np.array(v_values)

# RK4 method for second-order ODE
def rk4_method_2nd_order(t0, y0, v0, h, tmax, m, k):
    t_values = np.arange(t0, tmax + h, h)
    y_values = [y0]
    v_values = [v0]

    y = y0
    v = v0

    for t in t_values[:-1]:
        f1_y = v
        f1_v = -k / m * y

        f2_y = v + h * f1_v / 2
        f2_v = -k / m * (y + h * f1_y / 2)

        f3_y = v + h * f2_v / 2
        f3_v = -k / m * (y + h * f2_y / 2)

        f4_y = v + h * f3_v
        f4_v = -k / m * (y + h * f3_y)

        y += h * (f1_y + 2 * f2_y + 2 * f3_y + f4_y) / 6
        v += h * (f1_v + 2 * f2_v + 2 * f3_v + f4_v) / 6

        y_values.append(y)
        v_values.append(v)

    return t_values, np.array(y_values), np.array(v_values)


# Leapfrog method for second-order ODE
def leapfrog_method(t0, y0, v0, h, tmax, m, k):
    t_values = np.arange(t0, tmax + h, h)
    y_values = [y0]
    v_values = [v0]

    y = y0
    v = v0

    for t in t_values[:-1]:
        v_half = v + h / 2 * (-k / m * y)
        y += h * v_half
        v = v_half + h / 2 * (-k / m * y)

        y_values.append(y)
        v_values.append(v)

    return t_values, np.array(y_values), np.array(v_values)

def main():
    m = 2
    k = 40
    y0 = 0.7
    v0 = 0
    t0 = 0
    tmax = 2.5

    h_values = [0.05, 0.1]
    h_rk4 = 0.1
    h_leapfrog = 0.1

    omega = np.sqrt(k / m)

    def analytical_solution(t):
        return y0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)

    def analytical_solution_v(t):
        return -y0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)

    print('\nAnalytical solution of the ODE:')
    print('Analytical solution of the displacement:')
    print("y(t) = y(0) * cos(ω*t) + v0/ω * sin(ω*t)")
    print(f"y(t) = {y0} * cos({omega}*t) + {v0}/{omega} * sin({omega}*t)\n")
    print('Analytical solution of the velocity:')
    print("v(t) = -y(0) * ω * sin(ω*t) + v0 * cos(ω*t)")
    print(f"v(t) = -{y0} * {omega} * sin({omega}*t) + {v0} * cos({omega}*t)\n")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot displacement
    for h in h_values:
        t_euler, y_euler, v_euler = euler_method_2nd_order(t0, y0, v0, h, tmax, m, k)
        axs[0].plot(t_euler, y_euler, label=f"Euler's Method h={h}")

    t_rk4, y_rk4, v_rk4 = rk4_method_2nd_order(t0, y0, v0, h_rk4, tmax, m, k)
    axs[0].plot(t_rk4, y_rk4, label=f"RK4 Method h={h_rk4}")

    t_leapfrog, y_leapfrog, v_leapfrog = leapfrog_method(t0, y0, v0, h_leapfrog, tmax, m, k)
    axs[0].plot(t_leapfrog, y_leapfrog, label=f"Leapfrog Method h={h_leapfrog}")

    t_analytical = np.linspace(t0, tmax, 1000)
    y_analytical = analytical_solution(t_analytical)
    axs[0].plot(t_analytical, y_analytical, label="Analytical Solution")

    axs[0].set_xlabel("Time (s)", fontweight='bold', fontsize=10)
    axs[0].set_ylabel("Displacement (m)", fontweight='bold', fontsize=10)
    axs[0].set_title("Displacement vs. Time", fontsize=14, fontweight='bold')
    axs[0].legend()
    axs[0].grid()

    # Plot velocity
    for h in h_values:
        t_euler, y_euler, v_euler = euler_method_2nd_order(t0, y0, v0, h, tmax, m, k)
        axs[1].plot(t_euler, v_euler, label=f"Euler's Method h={h}")

    axs[1].plot(t_rk4, v_rk4, label=f"RK4 Method h={h_rk4}")
    axs[1].plot(t_leapfrog, v_leapfrog, label=f"Leapfrog Method h={h_leapfrog}")

    v_analytical = analytical_solution_v(t_analytical)
    axs[1].plot(t_analytical, v_analytical, label="Analytical Solution")

    axs[1].set_xlabel("Time (s)", fontweight='bold', fontsize=10)
    axs[1].set_ylabel("Velocity (m/s)", fontweight='bold', fontsize=10)
    axs[1].set_title("Velocity vs. Time", fontsize=14, fontweight='bold')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    fig.savefig('plot_of_displacement_and_velocity_vs_time_Q_2.png')
    plt.show()


    print("\n")
    print("the script has finished running")

if __name__ == '__main__':
    main()
