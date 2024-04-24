class Retangulo:
    def __init__(self, largura: float, altura: float) -> None:
        self._largura = largura
        self._altura = altura
        self._vertice_inferior_esquerdo = Ponto(0, 0)
        self._vertice_superior_direito = Ponto(largura, altura)
    
    def encontrar_centro(self):
        x = (self._vertice_inferior_esquerdo.x + self._vertice_superior_direito.x) / 2
        y = (self._vertice_inferior_esquerdo.y + self._vertice_superior_direito.y) / 2
        return Ponto(x, y)

class Ponto:
    def __init__(self, x: float, y: float) -> None:
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    def mostrar_ponto(self):
        print(f'({self._x}, {self._y})')

if __name__ == '__main__':
    lista = []
    flag = 1

    while(flag != 0):
        print('O que deseja fazer?')
        print('1 - Criar um Retangulo')
        print('2 - Mostrar centros dos retangulos')
        print('0 - Sair')
        op = float(input('? '))
        print()

        match op:
            case 1:
                print('Insira largura e altura: ')
                largura = float(input())
                altura = float(input())
                lista.append(Retangulo(largura, altura))
                print()

            case 2:
                if lista:
                    for retangulo in lista:
                        centro = retangulo.encontrar_centro()
                        centro.mostrar_ponto()
                    print()

            case 0:
                print('Encerrando programa...')
                flag = 0

            case _:
                print('*Opcao invalida*')
                print()