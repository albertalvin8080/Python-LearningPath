import numpy as np
import os

os.system("cls")

separator = "-----------------------------------------------"

a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
b = np.array([
    [[1, 2, 3], [4, 5, 6]], 
    [[7, 8, 9], [10, 11, 12]]
])
# print(a)
# print(b)

print(separator + "0")
print(a[:, 2])  # [3 8]
print(a[1, :])  # [ 6  7  8  9 10]

print(separator + "1")
print(b[0, 1, 1])  # 5
print(b[0, 1, 0])  # 4

print(separator + "2")
print(b[:, 1, :])  # [[ 4  5  6] [10 11 12]]
print(b[:, 1, 2])  # [ 6 12]
print(b[1, 1, :])  # [10 11 12]

print(separator + "3")
b[1, 1, :] = [99, 99, 99]
print(b)
b[1, :] = [
    [88, 88, 88],
    [88, 88, 88],
]  # each array inside this array represents a column
print(b)

print(separator + "4")
b[:, 1, :] = [[-1, -1, -1], [-1, -1, -1]]
print(b)
