class ContaInvestimento:
    def __init__(self, saldo_inicial, taxa):
        self._saldo = saldo_inicial
        self._taxa_juros = taxa
    
    @property
    def saldo(self):
        return self._saldo

    def aplicar_juros(self):
        if self._saldo <= 0:
            print('Confira seu saldo, ele pode estar zerado ou mesmo negativo...')
        else:
            self._saldo += self._saldo * self._taxa_juros / 100
            return self._saldo

if __name__ == '__main__':
    invest = ContaInvestimento(1000, 1.3)
    print()

    for _ in range(6):
        print(f'Saldo: R${invest.saldo:.2f}')
        print(f'Saldo apos juros: R${invest.aplicar_juros():.2f}')
        print()
