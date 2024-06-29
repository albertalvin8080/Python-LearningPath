import os
os.system('cls' if os.name == 'nt' else 'clear')

nome = 'Otavio Mirando'
nome2 = 'Primeval Current'
i = 30
print(f'{nome:@>{i}}')
print(f'{nome2:@>{i}}')
print(f'{nome2:4}') # coloca 4 espacos em branco

nome3 = 'Testando e atirando'
nome3_formatado = '{n:$^50}'.format(n=nome3)
print(nome3_formatado)