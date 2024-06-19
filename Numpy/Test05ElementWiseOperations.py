import numpy as np
import os

os.system("cls")

a = np.array([1, 2, 3, 4])
b = np.array([11, 12, 13, 14])
print(a * 3)
print(a ** 3)
print(a / 3)
print(a // 3) # floor division operator (rounds down)
print(a - 4)
print(a + 4)
print(np.cos(a))
print(np.sin(a))

print(a * b)
print(a / b)

a += 30
print(a)

