import os
os.system('cls' if os.name == 'nt' else 'clear')

def gerar_num(x=1000):
    n = 0
    while n < x:
        yield n
        n += 1

g = gerar_num(10)
print(g.__next__())
for i in g:
    print(i, end=' ')
print()

gerador = (f'valor n{x}' for x in range(10,30,3)) # é um gerador, e NAO uma tupla
# print(gerador)
for i in gerador:
    print(i)