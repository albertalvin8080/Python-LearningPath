with open('Python/Arquivos/texto.txt', '+a') as f:
    f.write('\nTentado appendar')
    f.seek(0, 0)
    string = f.readlines()

print(string)