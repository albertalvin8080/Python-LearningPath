import numpy as np
import os

os.system("cls")

a = np.random.randint(
    low=0,
    high=11,
    size=10,
)
print(a)
print(np.sort(a))

print("--------------------------------------")

b = np.random.randint(20, size=20).reshape(4,5)
print(b)
print(np.sort(b))         # Sorts each row individually
print(np.sort(b, axis=1)) # Also sorts each row individually
print(np.sort(b, axis=0)) # Sorts each column individually

# Sorting everything
c = b.reshape(-1)      # Flatten
c = np.sort(c)         # Sort unique row
c = c.reshape(b.shape) # Return to initial shape
print(c)