class Bola:
    def __init__(self, cor, circunferencia, material) -> None:
        self.cor = cor
        self.circunferencia = circunferencia
        self.material = material
    
    @property
    def cor(self):
        # print('Property Cor foi acessado')
        return self._cor
    @cor.setter
    def cor(self, value):
        if not isinstance(value, str):
            raise ValueError('A cor da bola precisa fazer sentido')
        else:
            self._cor = value

    @property
    def circunferencia(self):
        return self._circunferencia
    @circunferencia.setter
    def circunferencia(self, value):
        if not isinstance(value, float) and not isinstance(value,int) or value <= 0:
            raise ValueError('Circunferencia invalida')
        else:
            self._circunferencia = value
    
    @property
    def material(self):
        return self._material
    @material.setter
    def material(self, value):
        self._material = value

if __name__ == '__main__':
    bola1 = Bola('Azul', 45, 'plastico')
    bola2 = Bola('rosiu', 1, 'borracha')
    print(bola1.cor)
    print(bola2.cor)
    
