import random
import os
os.system('cls' if os.name == 'nt' else 'clear')

def operation(n1, n2, operator=''):
    match operator:
        case '+':
            return n1 + n2
        case '-': 
            return n1 - n2
        case '*': 
            return n1 * n2
        case '/': 
            return n1 / n2
        case _: 
            return None        

print(operation(4,5))

def div_total(divisor, *args):
    args = list(args)
    
    for i in range(len(args)):
        args[i] = args[i] / divisor

    args = tuple(args)
    return args

print(div_total(2.5,2,2.15,3,4,5,6,7))

def verif_palindromo(string):
    string = string.lower()
    string = string.replace(' ', '')

    # print(string[len(string)::-1]) fatiamento reverso de string
    if string == string[::-1]:
        return True
    else:
        return False
        
print(verif_palindromo('sossos'))

def argumentos(*args, **kwargs):
    print(*args, sep='-')
    for chave, valor in kwargs.items():
        print(chave, valor)

numeros = [random.randrange(0,99) for _ in range(5)]
argumentos(*numeros, pikaseca=34.4,grower=35.5)