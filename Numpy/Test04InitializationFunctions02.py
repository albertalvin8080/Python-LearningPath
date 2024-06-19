import numpy as np
import os

os.system("cls")

a = np.identity(n=4)
b = a.copy()
b[0, 1] = 99
# print(a)
# print(b)

# -------------------------------------
a = np.ones((5, 5), dtype=np.int32)
c = np.zeros((3, 3))
c[1, 1] = 9
a[1:4, 1:4] = c
# print(a)

# -------------------------------------
x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
z = np.repeat(x, repeats=3, axis=0)
# z = np.repeat(x, repeats=3, axis=1)
print(z)
