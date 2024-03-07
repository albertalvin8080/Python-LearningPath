class Retangulo:
    def __init__(self, comprimento, largura) -> None:
        self.comprimento = comprimento
        self.largura = largura
    
    @property
    def comprimento(self):
        return self._comprimento
    @comprimento.setter
    def comprimento(self, value):
        if value > 0 and isinstance(value, float):
            self._comprimento = value
        else:
            raise ValueError

    @property
    def largura(self):
        return self._largura
    @largura.setter
    def largura(self, value):
        if value > 0 and isinstance(value, float):
            self._largura = value
        else:
            raise ValueError

    def calcular_area(self):
        return self._comprimento * self._largura
    
    def calcular_perimetro(self):
        return (self._comprimento + self._largura) * 2

if __name__ == '__main__':
    comprimento = float(input('Insira o comprimento: '))
    largura = float(input('Insira a largura: '))
    ret1 = Retangulo(comprimento, largura)

    # metros
    area_piso = 0.2
    comprimento_rodape = 0.5

    qnt_piso = ret1.calcular_area() / area_piso
    qnt_rodape = ret1.calcular_perimetro() / comprimento_rodape

    print(f'Pisos necessarios: {qnt_piso:.2f}')
    print(f'Rodapes necessarios: {qnt_rodape:.2f}')
