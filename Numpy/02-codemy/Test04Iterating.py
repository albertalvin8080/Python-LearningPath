import numpy as np
import os

os.system("cls")

a = np.arange(1, 36+1).reshape(3,4,3)
print(a)

# Avoid doing this.
for x in a:
    for y in x:
        for z in y:
            print(z, end=", ")

print()
print("---------------------------------------------------------")

# Do this instead. Also works for other dimensions other than 3D.
for x in np.nditer(a):
    print(x, end=", ")
