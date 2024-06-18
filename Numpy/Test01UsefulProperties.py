import numpy as np
import os

os.system("cls")

# a = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]], dtype=np.int32)
# a = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32)
a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.int32)

print(f"{a}")
print()
print(f"dtype (items type)                  : {a.dtype}")
print(f"shape (row, column, ...)            : {a.shape}")
print(f"ndim (number of dimensions)         : {a.ndim}")
print(f"size (total number of items)        : {a.size}")
print(f"itemsize (number of bytes per item) : {a.itemsize}")
print(f"nbytes (total number of bytes)      : {a.nbytes}")
print(f"size * itemsize = nbytes            : {a.size * a.itemsize}")