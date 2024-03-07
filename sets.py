import os
os.system('cls')

# set1 = {1, 'carlos lima', 4, 5, 6, 'F', 'to pay respect'}
# print(set1)

set1 = {1,2,3,4,5}
set2 = {3,4,5,6,7,8,9}
print(f'Uniao: {(set1 | set2)}')
print(f'Intersecao: {(set1 & set2)}')
print(f'Diferenca: {(set1 - set2)}')
print(f'Diferenca simetrica: {(set1 ^ set2)}')

