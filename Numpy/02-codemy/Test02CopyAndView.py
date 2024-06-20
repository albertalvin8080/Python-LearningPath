import numpy as np
import os

os.system("cls")

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
print(a)

b = a.view()
b[0, 0] = 999

print(a)
print(b)

print("-------------------------------------------------------------------")

c = np.arange(1, 11).reshape(2, 5)
print(c)

d = c.copy()
d[0, 0] = -111

print(c)
print(d)
