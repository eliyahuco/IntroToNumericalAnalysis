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
    """
    This function calculates the value of f(x, y) = 4y² + 4y - 52x - 1 in the given point (x, y).
    """
    return 4*y**2 + 4*y - 52*x - 19

def g(x, y):
    """
    This function calculates the value of g(x, y) = 169x² + 3y² - 111x - 10y in the given point (x, y).
    """
    return 169*x**2 + 3*y**2 - 111*x - 10*y

# Define the partial derivative of the functions f(x, y) and g(x, y)
def partial_derivative_of_function_2d(function, x, y, h = 10**-9):
    """"
    This function calculates the partial derivative of a 2D function.
    The function receives the function, the point (x, y) and the step size h.
    The function returns the partial derivative of the function at the point (x, y).
    """
    df_dx = (function(x + h, y) - function(x, y)) / h
    df_dy = (function(x, y + h) - function(x, y)) / h
    return df_dx, df_dy

# Define the Newton-Raphson method for simultaneous solution
def newton_raphson_method_for_simultaneous_solution( x0, y0, epsilon = 10**-4):
    """"
    This function solves the system of equations f(x, y) = 0 and g(x, y) = 0 using the Newton-Raphson method.
    The function receives the initial guess (x0, y0) and the precision requirement epsilon.
    The function returns the solution (x, y) of the system of equations.
    """
    x = x0
    y = y0
    if abs(f(x, y)) < epsilon and abs(g(x, y)) < epsilon:
        return x, y
    while abs(f(x, y)) > epsilon and abs(g(x, y)) > epsilon:
        df_dx, df_dy = partial_derivative_of_function_2d(f, x, y)
        dg_dx, dg_dy = partial_derivative_of_function_2d(g, x, y)
        jacobian = df_dx * dg_dy - df_dy * dg_dx
        x = x - (f(x, y) * dg_dy - g(x, y) * df_dy) / jacobian
        y = y - (g(x, y) * df_dx - f(x, y) * dg_dx) / jacobian
    return x,y


# Define the range of x and y
x = np.linspace(-10, 10, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
function_to_plot = np.sin(4*Y)*np.cos(0.5*X)

# Main function
def main():
    print("\n-------------------\n---Question 2.1:---\n-------------------")
    print("The system of equations is:")
    print("f(x, y) = 4y² + 4y - 52x - 1 = 0")
    print("g(x, y) = 169x² + 3y² - 111x - 10y = 0")
    print("The initial guess is: x0 = -0.01, y0 = -0.01")
    print("The precision requirement is: epsilon = 10⁻⁴")
    print("\n")
    print("the value of f(x, y) and g(x, y) at the initial guess is:")
    print("f(x0, y0) = ", f(x0, y0))
    print("g(x0, y0) = ", g(x0, y0), "\n")
    print("The solution of the system of equations is:")
    solution_for_the_system = newton_raphson_method_for_simultaneous_solution( x0, y0, epsilon)
    print("x = ", solution_for_the_system[0], ", y = ", solution_for_the_system[1])
    print("\n")
    print("The value of f(x, y) at the solution is:")
    print("f(x, y) = ", f(solution_for_the_system[0], solution_for_the_system[1]))
    print("The value of g(x, y) at the solution is:")
    print("g(x, y) = ", g(solution_for_the_system[0], solution_for_the_system[1]))
    print("the solution mathces the precision requirement of 10⁻⁴.")

    print("\n-------------------\n---Question 2.2:---\n-------------------")
    print("The function to plot is: f(x, y) = sin(4y)cos(0.5x)")
    print("The range of x is: -10 < x < 10")
    print("The range of y is: -5 < y < 5")
    print("the graph will be displayed automatically.")
    print("the graph will be saved as '3D_graph_f(x,y)=sin(4y)cos(0.5x).png' in the current directory, and will be added to the submission.")
# Plot the 3D graph
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, function_to_plot, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('f(x, y) = sin(4y)cos(0.5x)')

    plt.show()
    #save the plot
    fig.savefig("3D_graph_f(x,y)=sin(4y)cos(0.5x).png")
    print("\n")
    print("the script has finished running.")
    print("thank you for using the script. have a nice day!")

if __name__ == "__main__":
    main()