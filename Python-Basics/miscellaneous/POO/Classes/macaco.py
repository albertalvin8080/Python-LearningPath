class Macaco:
    def __init__(self, nome):
        self._nome = nome
        self._bucho = None

    def ver_bucho(self):
        # return self._bucho
        print(f"Bucho: {self._bucho}, Macaco '{self._nome}'")

    def comer(self, comida: object):
        print(f"Macaco '{self._nome}' comendo {comida}...")
        self._bucho = comida
    
    def digerir(self):
        print(f"Macaco '{self._nome}' digeriu {self._bucho}")
        self._bucho = None

    def __str__(self) -> str:
        return f'(Macaco: {self._nome})' # parecido com o toString() em Java

#################################

if __name__ == '__main__':
    m1 = Macaco('Pedro')
    m2 = Macaco('Lucas')
    print()

    m1.ver_bucho()
    m1.comer('banana podre')
    m1.ver_bucho()
    m1.digerir()
    print()

    m2.ver_bucho()
    m2.comer(m1)
    m2.ver_bucho()
    m2.digerir()