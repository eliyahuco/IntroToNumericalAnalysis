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


import numpy as np
import matplotlib.pyplot as plt

# Euler's Method
def euler_method_ode(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    for x in x_values[:-1]:
        y += h * ode_func(x, y)
        y_values.append(y)
    return x_values, np.array(y_values)

# Runge-Kutta Second Order Method (RK2)
def rk2_method(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0
    for x in x_values[:-1]:
        f1 = ode_func(x, y)
        f2 = ode_func(x + h, y + h * f1)
        y += h * (f1 + f2) / 2
        y_values.append(y)
    return x_values, np.array(y_values)


# Main function to execute the code
def main():
    def analytical_solution(x):
        return 10 * x - x ** 2
    print('\nanalytical solution for the ODE:')
    print('m(x) = 10*x - x^2')

    def ode_func(x, m):
        return 10 - 2*x
    # Initial conditions
    x0 = 0
    mo = 0
    xmax = 10

    # Step sizes for Euler's method
    h_values = [0.05, 0.25]

    # Plot results
    plt.figure(figsize=(10, 6))

    # Euler's method results
    for h in h_values:
        x_euler, y_euler = euler_method_ode(ode_func, x0, mo, h, xmax)
        plt.plot(x_euler, y_euler, label=f"Euler's Method h={h}")

    # RK2 method results
    x_rk2, y_rk2 = rk2_method(ode_func, x0, mo, 0.05, xmax)
    plt.plot(x_rk2, y_rk2, label="RK2 Method h=0.05", linestyle='--')

    # Analytical solution (if available)
    x_analytical = np.linspace(x0, xmax, 1000)
    y_analytical = analytical_solution(x_analytical)
    plt.plot(x_analytical, y_analytical, label="Analytical Solution", linestyle='dotted')

    # Customize plot
    plt.xlabel('Position x (meters)', fontsize=12, fontweight='bold')
    plt.ylabel('Dependent Variable M(x)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Numerical Methods and Analytical Solution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True)
    plt.show()

    # save the plot
    plt.savefig('comparison_of_numerical_methods_and_analytical_solution_Q1.png')


    print("\n")
    print("the script has finished running")

# Execute the main function
if __name__ == "__main__":
    main()
