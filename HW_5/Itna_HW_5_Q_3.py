"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 3 in HW_5 for the course intro to numerical analysis

The objective of this script is to solve the following ODE:

dy/dx = (-2y)/(1+x)

We will solve the ODE using the fourth-order Adams-Bashforth as a predictor and the fourth-order Adams-Moulton as a corrector simultaneously.
For the first 3 steps, we will use the Runge-Kutta fourth-order method (RK4) to predict the next steps with h = 0.5.

Initial conditions:
y(x=0) = 2

Analytical solution:
y(x) = 2 / (1 + x)^2

We will compare the results of the methods and with the analytical solution.
Also, we will plot the results.
---------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


# Runge-Kutta Fourth Order Method (RK4)
def rk4_method_1st_order(ode_func, x0, y0, h, steps):
    x_values = [x0]
    y_values = [y0]
    y = y0

    for _ in range(steps):
        x = x_values[-1]
        f1 = ode_func(x, y)
        f2 = ode_func(x + h / 2, y + h / 2 * f1)
        f3 = ode_func(x + h / 2, y + h / 2 * f2)
        f4 = ode_func(x + h, y + h * f3)
        y += h * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        x_values.append(x + h)
        y_values.append(y)

    return np.array(x_values), np.array(y_values)


# Adams-Bashforth Fourth Order Predictor
def adams_bashforth_4th_order_predictor(ode_func, x_values, y_values, h):
    return y_values[-1] + h / 24 * (55 * ode_func(x_values[-1], y_values[-1])- 59 * ode_func(x_values[-2], y_values[-2])
            + 37 * ode_func(x_values[-3], y_values[-3])- 9 * ode_func(x_values[-4], y_values[-4]))


# Adams-Moulton Fourth Order Corrector
def adams_moulton_4th_order_corrector(ode_func, x_values, y_values, y_pred, h):
    return y_values[-1] + h / 24 * (9 * ode_func(x_values[-1] + h, y_pred) + 19 * ode_func(x_values[-1], y_values[-1])
            - 5 * ode_func(x_values[-2], y_values[-2]) + ode_func(x_values[-3], y_values[-3]))

def adam_bashforth_moulton_predictor_corrector(ode_func, x0, y0, h, xmax):
    x_values, y_values = rk4_method_1st_order(ode_func, x0, y0, h, 3)

    for _ in range(3, int((xmax - x0) / h)):
        y_pred = adams_bashforth_4th_order_predictor(ode_func, x_values, y_values, h)
        y_corr = adams_moulton_4th_order_corrector(ode_func, x_values, y_values, y_pred, h)
        x_values = np.append(x_values, x_values[-1] + h)
        y_values = np.append(y_values, y_corr)

    return x_values, y_values

def main():
    def ode_func(x, y):
        return (-2 * y) / (1 + x)

    x0 = 0
    y0 = 2
    h = 0.5
    xmax = 20

    def analytical_solution(x):
        return 2 / (1 + x) ** 2

    print('\nAnalytical solution for the ODE:')
    print('y(x) = 2 / (1 + x)^2')

    # Initial steps using RK4
    steps = int(3)  # First three steps
    x_values, y_values = rk4_method_1st_order(ode_func, x0, y0, h, steps)

    # Adams-Bashforth-Moulton Predictor-Corrector
    x_values, y_values = adam_bashforth_moulton_predictor_corrector(ode_func, x0, y0, h, xmax)

    # Analytical solution
    x_analytical = np.linspace(x0, xmax, 400)
    y_analytical = analytical_solution(x_analytical)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='Adams-Bashforth 4th Order Predictor & Adams-Moulton 4th Order Corrector',marker='o', color='g', linestyle='-', markersize=4, markerfacecolor='black')
    plt.plot(x_analytical, y_analytical, '--', label='Analytical Solution', color='purple')
    plt.xlabel('x', fontweight='bold', fontsize=14)
    plt.ylabel('y', fontweight='bold', fontsize=14)
    plt.title(r'Solution of $\frac{dy}{dx} = \frac{-2y}{1+x}$', fontsize=14, fontweight='bold')
    plt.text(4.5, 1.5, r'$y(0) = 2$', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(4.5, 1.7, "Analytical solution: " + r'$y(x) = \frac{2}{(1+x)^2}$', fontsize=12,bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.grid()
    plt.savefig('Adams_Bashforth_Moulton_Predictor_Corrector_and_Analytical_Solution_Plot_Q_3.png')
    plt.show()

    print("\nThe script has finished running.")
    print("Thank you for using the script.")
    print("\n")

if __name__ == '__main__':
    main()

