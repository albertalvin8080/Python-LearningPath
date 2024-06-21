import numpy as np
import os

os.system("cls")

# https://numpy.org/doc/stable/reference/routines.math.html

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32)
b = a.copy()
print(a)
print(np.add(a, b))
print(np.sqrt(a))
print(np.exp(a))
print(np.log(a))
# -1 -> negative
#  0 -> zero
#  1 -> positive
print(np.sign(a)) 
# efetuates `1/x` where `x` is the current item. It needs to be a float, otherwise the result will be truncated.
print(np.reciprocal(a)) 

print("----------------------------------------------------------------------")

# c = np.array([[-1,-2, -3, -4, -5], [-6, -7, -8, -9, -10]], dtype=np.float32)
c = a * -1 # Creates a separated array
print(a)
print(c)
print(np.log(c)) # Error: log of negative numbers
print(np.absolute(c))
print(np.sign(c))
print(np.reciprocal(c))

print("---------------------------------------------------------------------")

