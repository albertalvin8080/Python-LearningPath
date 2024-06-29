class Pessoa:
    def __init__(self, nome, idade, peso, altura) -> None:
        self.nome = nome
        self.__idade = idade
        self.__peso = peso
        self.__altura = altura
    
    @property
    def idade(self):
        return self.__nome
    @idade.setter
    def idade(self, value):
        if isinstance(value, str):
            self.__nome = value
        else:
            raise ValueError
        
    @property
    def altura(self):
        return self.__altura
    
    @property
    def idade(self):
        return self.__idade
    
    def envelhecer(self, anos):
        self.__idade += anos

        if self.__idade <= 21:
            crescimento = anos * 0.005
            # print(self.__altura) O arredondamento no print pode causar confusao
        else:
            crescimento = anos * 0.005 - (self.__idade - 21) * 0.005

        self.__crescer(crescimento)
    
    def __crescer(self, cm):
        self.__altura += cm

if __name__ == '__main__':
    pessoa1 = Pessoa('Juscelino', 5, 80, 1.50)
    print(f'Idade: {pessoa1.idade}; Altura: {pessoa1.altura:.3f}')
    pessoa1.envelhecer(3)
    print(f'Idade: {pessoa1.idade}; Altura: {pessoa1.altura:.3f}')
    pessoa1.envelhecer(20)
    print(f'Idade: {pessoa1.idade}; Altura: {pessoa1.altura:.3f}')
        