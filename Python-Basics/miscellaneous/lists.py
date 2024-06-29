lista = ["Mohg","Segundo","Terceiro"]
# lista.pop()
lista.append("Quarto Hello")
lista[0] = "Mohg, Lord of Blood"

del lista[1]

print(len(lista), lista)

lista2 = list(lista)
lista.clear()
lista3 = lista2 + lista2

print(len(lista), lista)
print(len(lista2), lista2)
print(len(lista3), lista3)

notas = [
    ['pedro','albuquerque',10],
    ['lucas','albuquerque',9.8],
    ['fedora','supremo',8],
    ['paulo','carioca',1],
    ['jao','paulo',4.3]
]
notas.sort(key=(lambda x: x[2]))
print(notas)

numeros = [1,2,3,4,5,6,7]
n1, n2, *n = numeros
print(n1, n2, n)
