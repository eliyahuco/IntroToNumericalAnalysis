"""
Author: Eliyahu Cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 3 in HW_5 for the course intro to numerical analysis

the objective of this script is to solve the following ODE:

dy/dx = (-2y)/1+x

we will solve the ODE using fourth order Adams-Bashforth as a predictor and fourth order Adams-Moulton as a corrector
for the first 3 steps we will use the Runge-Kutta fourth order method (RK4) to predict the next steps
h = 0.5

initial conditions:
y(x=0) = 2

"""


def rk4_method_1st_order(ode_func, x0, y0, h, xmax):
    x_values = np.arange(x0, xmax + h, h)
    y_values = [y0]
    y = y0

    for x in x_values[:-1]:
        f1 = ode_func(x, y)
        f2 = ode_func(x + h / 2, y + h / 2 * f1)
        f3 = ode_func(x + h / 2, y + h / 2 * f2)
        f4 = ode_func(x + h, y + h * f3)
        y += h * (f1 + 2 * f2 + 2 * f3 + f4) / 6
        y_values.append(y)

    return x_values, np.array(y_values)