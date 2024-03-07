import json

dc = {
    'Funcionario 1':{
        'nome':'Lucas',
        'CEP':'Algum por aí',
        'Comida predileta':'Saiko'
    },
    'item1':'espada',
    'item2':'carro'
}

with open('Python/json/json.json', 'w', encoding='utf-8') as j:
    json.dump(dc, j, indent=2, ensure_ascii=False)

with open('Python/json/json.json', 'r', encoding='utf-8') as j:
    read = json.load(j)

# print(read)

for k in read:
    if type(read[k]) is dict:
        for k2 in read[k]:
            print(k2)
    print(k)