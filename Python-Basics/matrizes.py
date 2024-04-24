matriz = [[3,4,5],[6,7,8],[9,10,11]]
matriz.append([12,13,14])

# matriz.append(['primeiro','segundo'])
# for a,b,c in matriz:
#     print(a,b,c)

matriz.append(["lucas"])
# matriz.append(12)
for linha in matriz:
    for elemento in linha:
        print(elemento,end=' ') # end é um parametro que recebe por padrao '\n'
    print()

