"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 2 in HW_5 for the course intro to numerical analysis

the objective of this script is to solve the following ODE:

m * d^2y/dt^2 +ky = 0

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
3) leapfrog method with step size of: 0.1 seconds

analytical solution:
y(t) = y(0) * cos(ω*t) + v0/ω * sin(ω*t)
v(t) = -y(0) * ω * sin(ω*t) + v0 * cos(ω*t)

    where ω = sqrt(k/m)

"""

import numpy as np
import matplotlib.pyplot as plt
import numerical_analysis_methods_tools as na_tools


def euler_method_2nd_order(t0, y0, v0, h, tmax, m, k):
    # Time array
    t_values = np.arange(t0, tmax + h, h)

    # Initialize arrays for y and v
    y_values = [y0]
    v_values = [v0]

    # Initial conditions
    y = y0
    v = v0

    # Euler method loop
    for t in t_values[:-1]:
        y_new = y + h * v
        v_new = v + h * (-k / m * y)

        y_values.append(y_new)
        v_values.append(v_new)

        # Update y and v for next iteration
        y = y_new
        v = v_new

    return t_values, np.array(y_values), np.array(v_values)

def rk4_method(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    for x in x_values[:-1]:
        f1 = ode_func(x, y)
        f2 = ode_func(x + h / 2, y + h * f1 / 2)
        f3 = ode_func(x + h / 2, y + h * f2 / 2)
        f4 = ode_func(x + h, y + h * f3)
        y += h * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        y_values.append(y)
    return x_values, np.array(y_values)

def leapfrog_method(ode_func, x0, y0, v0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    v = v0
    for x in x_values[:-1]:
        v += h / 2 * ode_func(x, y)
        y += h * v
        v += h / 2 * ode_func(x + h, y)
        y_values.append(y)
    return x_values, np.array(y_values)

def main():
    # Given parameters
    m = 2
    k = 40
    y0 = 0.7
    v0 = 0
    t0 = 0
    tmax = 2.5

    # Step sizes for Euler's method
    h_values = [0.05, 0.1]

    # Step size for RK4 method
    h_rk4 = 0.1

    # Step size for leapfrog method
    h_leapfrog = 0.1

    # Analytical solution
    omega = np.sqrt(k / m)
    def analytical_solution(t):
        return y0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)

    def analytical_solution_v(t):
        return -y0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)

    # Plot results
    plt.figure(figsize=(10, 6))

    # Euler's method results
    for h in h_values:
        t_euler, y_euler, v_euler = euler_method_2nd_order(t0, y0, v0, h, tmax, m, k)
        plt.plot(t_euler, y_euler, label=f"Euler's Method h={h}")

    # RK4 method results
    t_rk4, y_rk4 = rk4_method(lambda t, y: v, t0, y0, h_rk4, tmax)
    plt.plot(t_rk4, y_rk4, label=f"RK4 Method h={h_rk4}")

    # Leapfrog method results
    t_leapfrog, y_leapfrog = leapfrog_method(lambda t, y: -k / m * y, t0, y0, v0, h_leapfrog, tmax)
    plt.plot(t_leapfrog, y_leapfrog, label=f"Leapfrog Method h={h_leapfrog}")

    # Analytical solution
    t_analytical = np.linspace(t0, tmax, 1000)
    y_analytical = analytical_solution(t_analytical)
    plt.plot(t_analytical, y_analytical, label="Analytical Solution")

    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Displacement vs. Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot results
    plt.figure(figsize=(10, 6))
