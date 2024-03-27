from functools import singledispatch, singledispatchmethod

class Funcionario:
    # def __init__(self, nome) -> None:
    #     self._nome = nome
    #     self._salario = 1300

    def __init__(self, nome, salario) -> None:
        self._nome = nome
        self._salario = salario
    
    @property
    def nome(self):
        return self._nome
    @property
    def salario(self):
        return self._salario
    
    def aumentar_salario_em_percentual(self, taxa: int | float):
        self._salario += self._salario * taxa / 100
        print(f'Funcionario: {self._nome}\nNovo salario: {self._salario}')
    
    def __str__(self) -> str:
        return f'Funcionario: {self._nome}\nSalario: {self._salario}'
    
    @singledispatchmethod
    def func(self, value):
        raise NotImplementedError
    
    @func.register(int)
    def _(self, value):
        print('Voce enviou um inteiro:', value)
    
    @func.register
    def _(self, value: tuple):
        print('Voce enviou uma tupla:', value)

if __name__ == '__main__':
    print('#################################################')

    f1 = Funcionario('Pedro Bial', 4000)
    print(f1)
    f1.aumentar_salario_em_percentual(10)

    print()
    f1.func(18)
    f1.func(('oi', 9))
    # f1.func([])
    print()
