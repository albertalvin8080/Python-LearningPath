import numpy as np
import os

os.system("cls")

a = np.arange(1, 21)
print(a)

filter1 = a % 2 == 0
filter2 = a < 10
print(filter1)
print(filter2)

print(a[filter1])
print(a[filter2])

print("---------------------------------------------------")

b = np.arange(1, 21).reshape(5, 4)
print(b)

filter3 = b % 2 == 0
filter4 = b < 10
print(filter3)
print(filter4)

print(b[filter3])
print(b[filter4])