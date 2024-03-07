import os
os.system('cls' if os.name == 'nt' else 'clear')

# list1 = [(v,v2,v3) for v in range(3) for v2 in range(3) for v3 in range(10)]

# i = 0
# for t in list1:
#     print(t, end=' ')
#     i += 1
# print(i)

# list2 = [v if v % 2 != 0 else 0 for v in range(100)]
# print(*list2)

list3 = [v for v in range(100) if v % 2 == 0 and v % 4 == 0]
print(*list3)

import pandas as pd

dc = {f'Coluna {x}':[f'valor-{y}' for y in range(10)] for x in range(6)}
data = pd.DataFrame(dc)
print(data)