"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the Question 1 in HW_4 for the course intro to numerical analysis
the objective of this script is to calaculate integrals using the following methods:
1) trapezoidal rule
2) extrapolated richardson's rule
3) simpson's rule
4) romberg's rule
5) gauss quadrature

accuracy required: 10^-7
the assignment has two sections:
a) to calculate the integral of the function f(x) = e^(-x^2) from 0 to 2 using the methods 1 with 20 intervals equal in size and method 2
with iterations until the accuracy is reached
we will compare the results of the accuracy of the integration methods and with the analytical solution

b) to calculate the integral of the function f(x) = x*e^(2x) from 0 to 4 using the methods 1, 3, 4, 5
we will find the number of intervals required for each method to reach the accuracy required
we will compare the results of the accuracy of the integration methods and with the analytical solution

will use the file numerical_analysis_methods_tools.py for use functions from the previous assignments
---------------------------------------------------------------------------------
"""
