# https://gabrielschade.github.io/2018/06/25/basics-python-9-lambda.html
# https://www.codingame.com/playgrounds/52499/programacao-python-intermediario---prof--marco-vaz/exercitando
import os
os.system('cls' if os.name == 'nt' else 'clear')

import random
# numeros = [random.randrange(0, 99) for _ in range(10)]
# div_3 = filter(lambda n : n % 3 == 0, numeros)

# for n in div_3:
#     print(n, end=' ')
# print()

# #1
# f1 = lambda string:f'Seu nome é {string}'
# print(f1('Putz'))

# #2
# nome = input('Insira seu nome: ')
# idade = int(input('Insira sua idade: '))
# f2 = lambda x, y :  print(f'Seu nome eh {x}, voce tem {y} anos')
# f2(nome, idade)

# #3
# n1 = float(input('Insira um numero:'))
# n2 = float(input('Insira outro numero:'))
# print( (lambda x, y : x * y)(n1, n2) )

# 4.88 * x - 4.86 * x = 1 (combustivel)

#4
# nums = [random.randrange(0,21) for _ in range(5)]
# above_10 = filter(lambda x : x > 10, nums)
# print(list(above_10))

# #5
# lista = [random.randrange(1,100) for _ in range(10)]
# print( list( filter(lambda x : x % 2 == 0, lista)) )
# print( list( filter(lambda x : not x % 2 == 0, lista)) )

# #6
# funcao = lambda y: y ** 2
# h = funcao(4)
# print(h)

#7
def conceito(media):
    if(media <= 4.9):
        print(f'{media:0>4} | D')
    elif(media <= 6.9):
        print(f'{media:0>4} | C')
    elif(media <= 8.9):
        print(f'{media:0>4} | B')
    else:
        print(f'{media:0>4} | A')

def conceituar(fn_media_p, matriz):
    for i in range(len(matriz)):
        media = fn_media_p(matriz[i][0], matriz[i][1])
        conceito(media)

matriz = [[random.randrange(0,11) for _ in range(2)] for _ in range(10)]
fn_media_p = lambda x, y : (x*4 + y*6)/(4+6)

ls_medias = [0] * 10 # muito cuidado com essa sintaxe, pode fazer a lista inteira apontar para o mesmo objeto
conceituar(fn_media_p, matriz)

# # 9
# def padaria(broas, paes):
#     total = 2.50 * broas + 0.80 * paes
#     custo_fabricacao = total * 0.43
#     poupanca = total * 0.15
#     euros = (total * 0.15) / 4.60
#     print(f'Total arrecadado = {total}')
#     print(f'Custo de fabricacao = {custo_fabricacao}')
#     print(f'Poupanca = {poupanca}')
#     print(f'Quantidade de Euros = {euros}')

# padaria(50,100)

# # 8
# def percentual(prazo):
#     if prazo == 6:
#         return 0.07/12
#     elif prazo == 12:
#         return 0.10/12
#     elif prazo == 18:
#         return 0.12/12
#     elif prazo == 24:
#         return 0.15/12
#     elif prazo == 36:
#         return 0.18/12
#     else:
#         print(f'PRAZO INVALIDO')

# prestacao = lambda financiamento, prazo, taxa: financiamento * ((1 + taxa)**prazo * taxa) / ((1 + taxa)**prazo - 1)

# financiamento = random.randrange(100_000,1_000_000)
# prazo = 36
# taxa = percentual(prazo)
# print(f'financiamento = {financiamento}')
# print(f'prazo = {prazo}')
# print(f'taxa = {taxa:.4f}')
# print(f'prestacao = {prestacao(financiamento, prazo, taxa):.2f}')