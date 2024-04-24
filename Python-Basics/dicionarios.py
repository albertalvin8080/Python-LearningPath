import os
os.system('cls' if os.name == 'nt' else 'clear')

dc1 = {
    'first' : "Great Tree",
    'second' : 'The Crucible',
    'third' : 'Erdtree'
}
dc1['dung'] = 'eater'
dc1['fourth'] = 'Haligtree'
del dc1['dung']
# dc1.pop('dung')

for d in dc1.values():
    print(d)

carro = {
    'modelo' : 'MWB',
    'cor' : 'Azul'
}
pc = {
    'modelo' : 'AMD Ryzen',
    'idade' : 0
}
celular = {
    'Marca':'Samsung',
    'cor':'Branco'
}

itens = {
    'i1':carro,
    'i2':pc
}
itens['i3'] = celular
# print(itens)

# for iKey in itens:
#     for key in itens[iKey]:
#         print(key+':', itens[iKey][key], end=' ')
#     print()

# for dc in itens.values():
#     for dcKey in dc:
#         print(dcKey+':', dc[dcKey], end=' ')
#     print()

# dTest = {
#     1 : 'Mamao',
#     2 : 'Maca',
#     3 : 'Laranja'
# }
# dTest.update({'Gostavo':'Lima','Pedro':'Bial'})

# print('Gostavo' in dTest) # d1.keys()
# print('Maca' in dTest.values())

## .popitem() elimina o ultimo par do docionario
## .pop(|key|) elimina o par da chave enviada do dicionario

# PERGUNTAS E RESPOSTAS
questionario = {
    'Pergunta n1':{
        'Enunciado':'Se eu tenho 3 laranjas e multiplico-as por 3.5, quantas laranjas eu tenho?',
        'alternativas':{'a':'12.5', 'b':'13.5', 'c':'10.5'},
        'resposta':'c'
    },
    'Pergunta n2':{
        'Enunciado':'Qual a chance de uma pessoa ser furry?',
        'alternativas':{'a':'100%', 'b':'50%', 'c':'10%'},
        'resposta':'b'
    },
    'Pergunta n3':{
        'Enunciado':'Se eu dou dinheiro pra alguém na rua eu sou necessariamente uma boa pessoa?',
        'alternativas':{'a':'Talvez', 'b':'Sim', 'c':'Nao'},
        'resposta':'a'
    }
}

cont = 0
acertos = 0
for k_questionario, v_questionario in questionario.items():
    print(f'{k_questionario}: {v_questionario["Enunciado"]}')

    for k_alternativa, v_alternativa in v_questionario['alternativas'].items():
        print(f'{k_alternativa}) {v_alternativa}')
    resposta = input('Insira a resposta: ')

    if resposta == v_questionario['resposta']:
        print(f'Resposta correta.')
        acertos += 1
    else:
        print(f'Resposta ERRADA.')
    cont += 1
    print()
    
print(f'Voce acertou {acertos} de {cont}.')
print(f'Sua porcentagem de acerto: {acertos/cont*100:.2f}%')