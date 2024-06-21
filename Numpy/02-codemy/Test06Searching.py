import numpy as np
import os

os.system("cls")

a = np.arange(1, 11)
print(a)

condition1 = np.where(a % 2 == 0)
condition2 = np.where(a > 5)
print(condition1)
print(condition2)

print(a[condition1[0]])
print(a[condition2[0]])

print("---------------------------------------------------")

b = np.arange(1, 21).reshape(5, 4)
print(b)

condition3 = np.where(b % 2 == 0)
condition4 = np.where(b > 5)
print(condition3)
print(condition4)

# print(b[condition3[0]]) # Wrong 
print(b[condition3[0], condition3[1]]) # Correct For 2D arrays
print(b[condition4[0], condition4[1]])