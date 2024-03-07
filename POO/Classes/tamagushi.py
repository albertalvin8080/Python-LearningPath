import random

class Tamagushi:
    __saude_limite = 10
    __fome_limite = 10
    __frases = [
        'Eu gosto de voce?',
        'Eu nao gosto de voce!',
        'Me deixe em paz!',
        'Just end me!!!',
        'Voce eh o melhor dono que eu ja tive.'
    ]

    def __init__(self, nome) -> None:
        self._nome = nome
        self._fome = random.randint(1, 10)
        self._saude = random.randint(1, 10)
        self._idade = 0
    
    @property
    def nome(self):
        return self._nome
    @nome.setter
    def nome(self, value):
        self._nome = value

    @property
    def fome(self):
        return self._fome
    # @fome.setter
    # def fome(self, value):
    #     self._fome = value

    @property
    def idade(self):
        return self._idade
    
    @property
    def saude(self):
        return self._saude
    
    def envelhecer(self):
        self._idade += 1
        self._fome -= 2
        self._saude -= 1

        if self._fome < 1:
            self._fome = 1

        if self._saude < 1:
            self._saude = 1

        print('Tamagushi envelheceu...')
        
    def retornar_humor(self):
        return (self._saude * 1 + self._fome * 2) / (1 + 2)
    
    def alimentar(self, comida):
        if self._fome >= self.__fome_limite:
            print('*Tamagushi estah cheio*')
        
        elif self._fome + comida > self.__fome_limite:
            self._fome = self.__fome_limite
            self._aumentar_saude(1)
            print('Tamagushi estah comendo (demais)...')
        else:
            self._fome += comida
            self._aumentar_saude(3)
            print('Tamagushi estah comendo...')
    
    def _aumentar_saude(self, value: int):
        if self._saude + value > self.__saude_limite:
            self._saude = self.__saude_limite
        else:
            self._saude += value

    def brincar(self):
        if self._saude <= 1:
            # raise SaudeTamagushiError('O tamagushi estah doente')
            print('*O tamagushi estah doente*')

        elif self._fome > 5:
            print('Brincando com Tamagushi...')
            self._fome -= 1

        elif self._fome > 1:
            print('Brincando com Tamagushi (com fome)...')
            self._fome -= 1
            self._saude -= 1
        
        else:
            print('*Nao pode brincar, o tamagushi estah faminto*')
    
    def ouvir_tamagushi(self):
        i = random.randint(0, 4)
        return self.__frases[i]

    def __str__(self) -> str:
        # return self._nome + self._idade + self._fome + self._saude
        return f'idade: {self._idade}, fome: {self._fome}, saude: {self._saude}, nome: {self._nome}, '
            
class SaudeTamagushiError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

############################

if __name__ == '__main__':
    # pet1 = Tamagushi('Miquella')
    
    tamagushis = [
        Tamagushi('Miquela'),
        Tamagushi('Mohg, Lord of Blood'),
        Tamagushi('Radahn'),
        Tamagushi('Asteel, Natural Born of the Void')
    ]

    flag = 0
    while flag != 1:
        print()
        print('- Escolha uma opcao -')
        print('1 - Ouvir tamagushis')
        print('2 - Brincar com tamagushis')
        print('3 - Alimentar tamagushis')
        print('4 - Envelhecer tamagushis')
        print('0 - Encerrar programa')
        op = int(input('? '))
        print()

        match op:
            case 1:
                for i in range(len(tamagushis)):
                    print(f'Tamagushi {i+1}: {tamagushis[i].ouvir_tamagushi()}')
            case 2:
                for tamagushi in tamagushis:
                    tamagushi.brincar()
            case 3:
                for tamagushi in tamagushis:
                    tamagushi.alimentar(3)
            case 4:
                for tamagushi in tamagushis:
                    tamagushi.envelhecer()
            case 0:
                print('- Programa Encerrado -')
                flag = 1
            case -1:
                for tamagushi in tamagushis:
                    print(tamagushi)
            case _:
                print('*Opcao invalida*')


    # for _ in range(3):
    #     print(f'Nome: {pet1.nome}, Idade: {pet1.idade}, Saude: {pet1.saude}, Fome: {pet1.fome}, Humor: {pet1.retornar_humor()}')
    #     pet1.brincar()
    #     pet1.envelhecer()
    #     print()

    # print(f'Nome: {pet1.nome}, Idade: {pet1.idade}, Saude: {pet1.saude}, Fome: {pet1.fome}, Humor: {pet1.retornar_humor()}')
    # pet1.alimentar(15)
    # print(f'Nome: {pet1.nome}, Idade: {pet1.idade}, Saude: {pet1.saude}, Fome: {pet1.fome}, Humor: {pet1.retornar_humor()}')
