import os
os.system('cls' if os.name == 'nt' else 'clear')

def is_positive(n):
    if n < 0:
        raise ValueError('negative number')

def super_fat(n, flag=1):
    if n in (0,1):
        return 1
    elif flag == 0:
        return n * super_fat(n-1, flag=0)
    else:
        return super_fat(n, flag=0) * super_fat(n-1)
        
try:
    n = int(input('insira um numero: '))
    is_positive(n)
    s_fat = super_fat(n)
    print(f'Super fatorial: {s_fat}')

except ValueError as error:
    print(f'Error: {error}')

finally:
    print(f'Bloco finally.')
