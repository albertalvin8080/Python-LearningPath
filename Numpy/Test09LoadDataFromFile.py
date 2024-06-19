import numpy as np, os

os.system("cls")

a = np.genfromtxt("data.txt", delimiter=",")
print(a)
a = a.astype(np.int32)
print(a)