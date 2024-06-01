"""
Author: Eliyahu cohen
Email: cohen11@mail.tau.ac.il
---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 in the course intro to numerical analysis
The objective of the script is find roots of a given polynom with several methods
---------------------------------------------------------------------------------

Given polynom:  f(x) = X⁴ + 2x³ -7x² + 3
segment in X axis [-5,3]

"""
#Labraries in use
import numpy as np
import math
import matplotlib as plt

# constant parameters
precision_requierd = 10**(-4) # ε
a = -5 #begining of segment
b = 3 #end of segment
x_line = np.linspace(a,b,abs(b-a)*10**4)
polynom_coefficients = [1,2,-7,0,3]
given_polynom = np.poly1d(polynom_coefficients)




def bisection_search_first_guess(x_start,x_end,polynom, x_segment):
    a = x_start
    b = x_end
    if polynom(a)*polynom(b) >= 0:
        for i in x_segment:
            multiplicaion_value = polynom(i)*polynom(b)
            if multiplicaion_value < 0:
                a = i
                break
        if  polynom(a)*polynom(b) > 0:
            print("Bisection method is not applicabale here")

    c = 0.5*(a+b)
    print(c)





if __name__ == "__main__":
    bisection_search_first_guess(a,b,given_polynom,x_line)







