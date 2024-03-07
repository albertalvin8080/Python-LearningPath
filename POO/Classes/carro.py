class Carro:

    __limite_combustivel = 50.0

    def __init__(self, consumo) -> None:
        self._nivel_combustivel = 0
        self._consumo = consumo # km / litro

    def mostrar_tanque(self) -> None:
        print(f'Tanque: {self._nivel_combustivel:.2f} litros...')
    
    def abastecer(self, litros) -> None:
        if self._nivel_combustivel + litros > self.__limite_combustivel:
            self._nivel_combustivel = self.__limite_combustivel
            print('Carro abastecido (demais...)')
        else:
            self._nivel_combustivel += litros
            print(f'Carro abastecido com {litros:.2f} litros...')

    def dirigir(self, kms: float) -> None:
        if self._nivel_combustivel <= 0:
            # print('Carro sem combustivel...')
            raise NoFuelError('Carro com tanque vazio')

        self._nivel_combustivel -= kms / self._consumo

        if self._nivel_combustivel < 0:
            print('Seu carro morreu na estrada...')
            self._nivel_combustivel = 0

        else:
            print(f'Dirigindo carro por {kms:.2f} Kms...')

    def __str__(self) -> str:
        return f'Carro (limite do tanque: {self.__limite_combustivel} litros)'

class NoFuelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

###################################

if __name__ == '__main__':
    meuFusca = Carro(15)

    print(meuFusca)
    meuFusca.mostrar_tanque()
    # meuFusca.dirigir(12)
    meuFusca.abastecer(40)
    meuFusca.dirigir(400)
    meuFusca.mostrar_tanque()
    meuFusca.abastecer(100)
    meuFusca.dirigir(57.89)
    meuFusca.mostrar_tanque()
    