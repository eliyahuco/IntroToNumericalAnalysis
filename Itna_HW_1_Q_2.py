"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 Question number 2 in the course intro to numerical analysis.
The objective of the script is to solve the following Equation System using the Newton-Raphson method:
f(x,y) = 4y² + 4y - 52x - 1 = 0
g(x,y) = 169x² + 3y² - 111x - 10y = 0
starting from the initial guess (x0, y0) = (-0.01, -0.01).

In addition, the script will plot a 3D graph of the function f(x, y) = sin(4y)cos(0.5x) in the range of -10 < x < 10 and -5 < y < 5
with a precision requirement of 10⁻⁴.
---------------------------------------------------------------------------------
"""



# Libraries in use
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given parameters
x0 = -0.01
y0 = -0.01
epsilon = 10**(-4)

# Define the functions f(x, y) and g(x, y)
def f(x, y):
    return 4*y**2 + 4*y - 52*x - 1

def g(x, y):
    return 169*x**2 + 3*y**2 - 111*x - 10*y

def partial_derivative_of_function_2d(function, x, y, h = 10**-9):
    df_dx = (function(x + h, y) - function(x, y)) / h
    df_dy = (function(x, y + h) - function(x, y)) / h
    return df_dx, df_dy

def newton_raphson_method_for_simultaneous_solution(f, g, x0, y0, epsilon = 10**-4):
    x = x0
    y = y0
    while abs(f(x, y)) > epsilon and abs(g(x, y)) > epsilon:
        df_dx, df_dy = partial_derivative_of_function_2d(f, x, y)
        dg_dx, dg_dy = partial_derivative_of_function_2d(g, x, y)
        jacobian = df_dx * dg_dy - df_dy * dg_dx
        x = x - (f(x, y) * dg_dy - g(x, y) * df_dy) / jacobian
        y = y - (g(x, y) * df_dx - f(x, y) * dg_dx) / jacobian
    return x, y


print(partial_derivative_of_function_2d(f, 1, 1))
print(partial_derivative_of_function_2d(g, 1, 1))
print(newton_raphson_method_for_simultaneous_solution(f, g, x0, y0, epsilon))
print(f(-0.009629025334218142, 0.1122276298229395))
print(g(-0.009629025334218142, 0.1122276298229395))

# Define the range of x and y
x = np.linspace(-10, 10, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
function_to_plot = np.sin(4*Y)*np.cos(0.5*X)

# Plot the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, function_to_plot, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = sin(4y)cos(0.5x)')
plt.show()