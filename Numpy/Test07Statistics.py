import numpy as np, os

os.system("cls")

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2x4
print(a)

# NOTE: you could also do `np.sum(a, axis=0)`.

print(a.sum())       # adds up all items    -> 36
print(a.sum(axis=0)) # adds up column items -> [ 6  8 10 12]
print(a.sum(axis=1)) # adds up row items    -> [10 26]

print("--------------------------------------------------------------")

b = np.array([[12, 14, 5], [-2, 13, 99]])

# NOTE: you could also do `np.min(b, axis=0)`.

print(b.min())       # minimum of all items   -> 1
print(b.min(axis=0)) # minimum of each column -> [-2 13  5]
print(b.min(axis=1)) # minimum of each row    -> [ 5 -2]

print(b.max())       # maximum of all items   -> 1
print(b.max(axis=0)) # maximum of each column -> [-2 13  5]
print(b.max(axis=1)) # maximum of each row    -> [ 5 -2]


