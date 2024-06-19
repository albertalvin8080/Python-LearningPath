import numpy as np
import os

os.system("cls")

a = np.ones((2, 3), dtype=np.int32)
b = np.full((3, 2), 2, dtype=np.int32)
# print(a)
# print(b)
# print(np.matmul(a, b))

print("-----------------------------------")
x = np.array([[9, 8, -3], [-3, -6, 8]])  # 2x3
z = np.array([[3, 4], [9, 8], [14, 17]]) # 3x2
# print(np.matmul(x, z))

print("-----------------------------------")
c = np.array([[2, 7], [9, 8]])
# print(np.linalg.det(c))

d = np.array([[2, 3, 7], [9, 8, 4], [3, -5, -8]])
# print(d)
# print(np.linalg.det(d))
