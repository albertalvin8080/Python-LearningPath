class ContaCorrente:
    def __init__(self, conta: int, correntista: str, saldo=0.0) -> None:
        self._conta = conta
        self._correntista = correntista
        self._saldo = saldo

    @property
    def conta(self):
        return self._conta
    
    @property
    def correntista(self):
        return self._correntista

    def alterarCorrentista(self, novo_correntista):
        self._correntista = novo_correntista

    @property
    def saldo(self):
        return self._saldo
    
    def depositar(self, valor):
        if type(valor) not in (int, float) or valor <= 0:
            raise ValueError('Saldo invalido inserido')
        else:
            self._saldo += valor
    
    def sacar(self, valor):
        if self._saldo <= 0:
            raise ValueError('Voce nao possui saldo suficiente')
        
        elif type(valor) not in (int, float) or valor <= 0:
            raise ValueError('Saldo invalido inserido')
        
        elif self._saldo < valor:
            retorno = self._saldo
            self._saldo = 0

        else:
            retorno = valor
            self._saldo -= valor

        return retorno

if __name__ == '__main__':
    conta1 = ContaCorrente(1337, 'Alfredo', 1900)

    print(f'Saldo: R${conta1.saldo:.4f}')
    conta1.sacar(300)
    print(f'Saldo: R${conta1.saldo:.4f}')
    conta1.sacar(760.55)
    print(f'Saldo: R${conta1.saldo:.4f}')
    conta1.depositar(1324)
    print(f'Saldo: R${conta1.saldo:.4f}')
    conta1.sacar(1324)
    print(f'Saldo: R${conta1.saldo:.4f}')
    print(type((int, float)))
            
