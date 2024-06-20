import numpy as np, os

os.system("cls")

a = np.array([[13, 18, 29, 40],[26, 0, -2, 12]], dtype=np.int32)
print(a)

print("-------------------------------------------------------")

print(a[a < 10])
print(a[a < 15])
# print(a[(a < 15) and (a > 12)]) # ERROR
print(a[(a < 15) & (a > 12)]) # OK
print((a < 15) & (a > 12))

print("-------------------------------------------------------")

# axis=0 -> column
# axis=1 -> row

print(np.any(a > 15))
print(np.any(a > 15, axis=0))
print(np.any(a > 100)) 
print(np.any(a > 100, axis=0))
print(np.any(a >= 40)) 
print(np.any(a >= 40, axis=0)) 
print(np.any(a >= 40, axis=1)) 

print("-------------------------------------------------------")

print(np.all(a > 0))
print(np.all(a > 0, axis=0))
print(np.all(a > 0, axis=1))