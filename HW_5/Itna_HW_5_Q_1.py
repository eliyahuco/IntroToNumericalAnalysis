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
    print('momentum:')
    print('m(x) = 10*x - x^2')
    print('shear force:')
    print('s(x) = 10 - 2*x')


    def ode_func(x, m):
        return 10 - 2 * x

    # Initial conditions
    x0 = 0
    mo = 0
    xmax = 10

    # Step sizes for Euler's method
    h_values = [0.05, 0.25]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for ODE comparison
    for h in h_values:
        x_euler, y_euler = euler_method_ode(ode_func, x0, mo, h, xmax)
        axs[0].plot(x_euler, y_euler, label=f"Euler's Method h={h}")

    x_rk2, y_rk2 = rk2_method(ode_func, x0, mo, 0.05, xmax)
    axs[0].plot(x_rk2, y_rk2, label="RK2 Method h=0.05", linestyle='--')

    x_analytical = np.linspace(x0, xmax, 1000)
    y_analytical = analytical_solution(x_analytical)
    axs[0].plot(x_analytical, y_analytical, label="Analytical Solution", linestyle='dotted')

    axs[0].set_xlabel('Position x [meters]', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Moment M(x) [N*m]', fontsize=12, fontweight='bold')
    axs[0].set_title('Comparison of Numerical Methods and Analytical Solution', fontsize=14, fontweight='bold')
    axs[0].legend(fontsize=10, loc='upper right')
    axs[0].grid(True)

    # Plot for shear stress
    x_shear = np.linspace(x0, xmax, 1000)
    y_shear = ode_func(x_shear, 0)
    axs[1].plot(x_shear, y_shear, label='Shear Stress s(x) = 10 - 2x', color='r')

    axs[1].set_xlabel('Position x [meters]', fontsize=12, fontweight='bold')
    axs[1].set_ylabel('Shear Stress s(x) [N/m]', fontsize=12, fontweight='bold')
    axs[1].set_title('Shear Stress Distribution', fontsize=14, fontweight='bold')
    axs[1].legend(fontsize=10, loc='upper right')
    axs[1].grid(True)

    # Save and show plot
    plt.tight_layout()
    plt.savefig('comparison_and_shear_stress.png')
    plt.show()

    print("\n")
    print("the script has finished running")


# Execute the main function
if __name__ == "__main__":
    main()
