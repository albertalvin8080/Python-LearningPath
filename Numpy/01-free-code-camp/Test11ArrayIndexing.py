import numpy as np, os

os.system("cls")

# temp = [i for i in range(1, 30+1)]
temp = np.arange(1, 30 + 1)  # Prefer this sintax

a = np.reshape(temp, newshape=(6, 5))
print(a)

print("-------------------------------------------------------")

# NOTE: you may NOT use array indexing and slicing at the same time in the same position.

print(a[2:4, 0:2])  # Usual Indexing
print(a[[0, 1, 2, 3], [1, 2, 3, 4]])  # Array indexing

print(a[[0, 4, 5]])  # Array indexing
print(a[[0, 4, 5], 3:])  # Array indexing
