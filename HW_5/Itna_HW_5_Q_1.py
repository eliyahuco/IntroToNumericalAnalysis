"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_5 for the course intro to numerical analysis

the objective of this script is to solve the following ODE:

dm/dx = s

    where m is the momentum, s is the shear force and x is the position
    s = 10 - 2x
    m(x=0) = 0
    s(x=0) = 10
    x_max = 10 meters

we will solve the ODE using the following methods:
1) Euler's method with step size of: 0.05 meters, 0.25 meters
2) Runge-Kutta second order method (RK2)

we will compare the results of the methods and with the analytical solution
also we will plot the results

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""

# Define the shear force function s(x)
import numpy as np
import matplotlib.pyplot as plt
# Import your custom methods if needed, e.g., from numerical_analysis_methods_tools import euler_method

# Define the shear force function s(x)
def shear_force(x):
    return 10 - 2*x

# Define the ODE function
def ode(x, m):
    return shear_force(x)

# Euler's Method
def euler_method(x0, m0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    m_values = [m0]
    m = m0
    for x in x_values[:-1]:
        m += h * ode(x, m)
        m_values.append(m)
    return x_values, np.array(m_values)

# Runge-Kutta Second Order Method (RK2)
def rk2_method(x0, m0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    m_values = [m0]
    m = m0
    for x in x_values[:-1]:
        k1 = h * ode(x, m)
        k2 = h * ode(x + h/2, m + k1/2)
        m += k2
        m_values.append(m)
    return x_values, np.array(m_values)

# Analytical Solution (you'll need to derive this based on the given ODE)
def analytical_solution(x):
    return 10*x - x**2

# Initial conditions
x0 = 0
m0 = 0
xmax = 10

# Step sizes for Euler's method
h_values = [0.05, 0.25]

# Plot results
plt.figure(figsize=(10, 6))

# Euler's method results
for h in h_values:
    x_euler, m_euler = euler_method(x0, m0, h, xmax)
    plt.plot(x_euler, m_euler, label=f"Euler's Method h={h}")

# RK2 method results
x_rk2, m_rk2 = rk2_method(x0, m0, 0.05, xmax)
plt.plot(x_rk2, m_rk2, label="RK2 Method h=0.05", linestyle='--')

# Analytical solution
x_analytical = np.linspace(x0, xmax, 1000)
m_analytical = analytical_solution(x_analytical)
plt.plot(x_analytical, m_analytical, label="Analytical Solution", linestyle='dotted')

# Customize plot
plt.xlabel('Position x (meters)')
plt.ylabel('Momentum m')
plt.title('Comparison of Numerical Methods and Analytical Solution')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
