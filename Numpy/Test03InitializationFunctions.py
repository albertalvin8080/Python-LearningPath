import numpy as np
import os

os.system("cls")

a = np.array([[[1, 2, 4], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

b = np.zeros((2,2,2,2), dtype=np.int32)
# print(b)

b = np.random.rand(3, 4)            # same result
b = np.random.random_sample((3, 4)) # same result
# print(b)

# b = np.random.randint(-2, -1, size=a.size)  # slightly different
# b = np.random.randint(-2, -1, size=a.shape) # slightly different
# print(b)

b = np.full(a.shape, 88, dtype=np.int32)
b = np.full_like(a, 88, dtype=np.int32)
# print(b)

b = np.full((3, 1, 2), fill_value=77, dtype=np.int32)
# print(b)