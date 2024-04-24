import os
os.system('cls' if os.name == 'nt' else 'clear')

numeros = [1,2,3,4,5,6,7]
n1, n2, *n = numeros
# print(n1, n2, n)
print(*numeros, sep='-') #desempacotamento de lista
