import json

texto = 'God é bom'
dc = {
    'nome':'Carlos moura',
    'CPF':'000.000.000-11',
    'endereco':{
        'Estado':'Acre',
        'Cidade':'Rondonia',
        'Bairro':'bairro generico',
        'Rua':'Rua nao sei das quantas'
    },
    'casado':False,
    'lema':None
}

with open(r'C:\Users\Albert\Documents\AAA-Programacao\Python\json\testejson.json', 'w', encoding='utf-8') as f:
    json.dump(dc, f, indent=2, ensure_ascii=False) # se eu usar ensure_ascii=, preciso usar encoding= tambem
    # json.dump(dc, f, indent=2)

with open(r'C:\Users\Albert\Documents\AAA-Programacao\Python\json\testejson.json', 'r', encoding='utf-8') as f:
    read = json.load(f)

print(*read)
