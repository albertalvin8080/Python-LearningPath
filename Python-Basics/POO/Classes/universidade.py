class Universidade:
    def __init__(self, nome) -> None:
        self._nome = nome
        # self._alunos = []
        self._professores = []
        self._endereco = None
    
    def inserir_endereco(self, bairro, cidade):
        self._endereco = Endereco(bairro, cidade)
    
    def mostrar_endereco(self):
        print('Endereco:', self._endereco._bairro, self._endereco._cidade)

    def inserir_professor(self, nome, materia):
        self._professores.append(Professor(nome, materia))
    
    def mostrar_professores(self):
        for professor in self._professores:
            print('Professor:',professor._nome, professor._materia)

    def __del__(self):
        print('Deletando universidade: ' + self._nome)

class Professor:
    def __init__(self, nome, materia) -> None:
        self._nome = nome
        self._materia = materia

    def __del__(self):
        print(f'Deletando professor: {self._nome}')

class Endereco:
    def __init__(self, bairro, cidade) -> None:
        self._bairro = bairro
        self._cidade = cidade
    
    def __del__(self):
        print('Deletando endereco')

#######################################

if __name__ == '__main__':
    u1 = Universidade('Volcano Manor')

    u1.inserir_endereco('Pinaculos Celestiais', "Erdtree's Town")
    u1.mostrar_endereco()

    u1.inserir_professor('Carlos', 'Portugues')
    u1.inserir_professor('Margit', 'Ingles')
    u1.inserir_professor('Caria Manor', 'Religiao')
    u1.mostrar_professores()
    print('##########################################')
