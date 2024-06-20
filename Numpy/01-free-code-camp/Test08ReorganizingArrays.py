import numpy as np, os

os.system("cls")

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2x4
print(a)

b = a.reshape((4, 2))
print(b)

print("--------------------------------------------------------------")

# NOTE: What matters is the ROW size.
# c = np.zeros(shape=(2,4), dtype=np.int32) # OK 
c = np.zeros(shape=(2,7), dtype=np.int32) # OK 
# c = np.zeros(shape=(3,4), dtype=np.int32) # ERROR
print(np.hstack([a, c]))

print("--------------------------------------------------------------")

# NOTE: What matters is the COLUMN size.
d = np.full(shape=(4, 4),fill_value=-1, dtype=np.int32) # OK 
# d = np.full(shape=(1, 4),fill_value=-1, dtype=np.int32) # OK
# d = np.full(shape=(4, 3),fill_value=-1, dtype=np.int32) # ERROR
print(np.vstack([a, d]))