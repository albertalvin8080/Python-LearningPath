import formatacao_teste as formatacao
# from . import formatacao_teste as formatacao -> so funciona se o __main__ estiver fora dessa pasta (pacote)
# print(dir(formatacao))
### depende de onde eu estou rodando o script __main__
### se eu importar pelo nome do pacote, esse modulo nao funciona, mas modulos_teste funciona, porque estah fora
### desse pacote

def power(n, e):
    result = 1
    while(e >= 1):
        result *= n
        e -= 1
    
    if isinstance(result, float):
        return formatacao.BR(result)
    else:
        return result

if __name__ == '__main__':
    print(power(3.14, 3))
