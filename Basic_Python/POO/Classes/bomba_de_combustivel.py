class BombaDeCombustivel:

    __limite_combustivel = 500 # litros

    def __init__(self, tipo_combustivel: str, valor_litro: float) -> None:
        self._tipo_combustivel = tipo_combustivel
        self._valor_litro = valor_litro
        self._quantidade_combustivel = 0

    @property
    def quantidade_combustivel(self):
        return self._quantidade_combustivel

    def abastecer_por_valor(self, valor: float):
        litros = valor / self._valor_litro
        
        if self._quantidade_combustivel - litros < 0:
            print('*Combustivel insuficiente*')
        else:
            self._quantidade_combustivel -= litros
            print(f'Foram colocados {litros:.2f} litros de {self._tipo_combustivel}')
            return litros
        
    def abastecer_por_litro(self, litros: float):
        if self._quantidade_combustivel - litros < 0:
            print('*Combustivel insuficiente*')
        else:
            valor = litros * self._valor_litro
            self._quantidade_combustivel -= litros
            print(f'O valor a ser pago eh de R${valor:.3f}')

    def alterar_valor_litro(self, novo_valor: float):
        self._valor_litro = novo_valor
    
    def alterar_tipo_combustivel(self, novo_tipo: str):
        self._tipo_combustivel = novo_tipo

    def abastecer_bomba(self, litros: float):
        if self._quantidade_combustivel + litros > self.__limite_combustivel:
            self._quantidade_combustivel = self.__limite_combustivel
        else:
            self._quantidade_combustivel += litros
        print('Bomba abastecida')

#########################################

if __name__ == '__main__':
    print()
    bomba = BombaDeCombustivel('Etanol Comum', 5.67)

    print(f'Quantidade de combustivel: {bomba.quantidade_combustivel:.2f}')
    bomba.abastecer_por_litro(22)

    bomba.abastecer_bomba(500)
    print(f'Quantidade de combustivel: {bomba.quantidade_combustivel:.2f}')

    bomba.abastecer_por_valor(500)
    print(f'Quantidade de combustivel: {bomba.quantidade_combustivel:.2f}')

    bomba.abastecer_por_litro(22)
    print(f'Quantidade de combustivel: {bomba.quantidade_combustivel:.2f}')

    bomba.abastecer_bomba(500)
    print(f'Quantidade de combustivel: {bomba.quantidade_combustivel:.2f}')
    print()

