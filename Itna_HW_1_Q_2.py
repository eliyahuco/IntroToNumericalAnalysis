"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 in the course intro to numerical analysis
The objective of the script is to solve the following Equation System using the Newton-Raphson method:
f(x,y) = 4y² +4y -52x-1 = 0
g(x,y) = 169x² + 3y² - 111x -10y = 0
starting from the initial guess (x0,y0) = (-0.01,-0.01)

in addition, the script will plot a 3D graph of the function f(x,y) = sin(4y)cos(0.5x) in the range of --10<x<10 and -5<y<5

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
