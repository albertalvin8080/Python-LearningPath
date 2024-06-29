from jogo_da_velha_erros import JogadorInvalidoError, PosicaoOcupadaError

class JogoDaVelha:
    __linhas = 3
    __colunas = 3
    __valor_padrao = '\u25A1'

    def __init__(self) -> None:
        self._X = 'X'
        self._tabuleiro = [[self.__valor_padrao for _ in range(self.__linhas)] for _ in range(self.__colunas)]
        self._O = 'O'
        self._jogador = 1 # 1 | 2
        self._vencedor = None

    @property
    def jogador(self) -> int:
        return self._jogador

    @property
    def vencedor(self) -> None | int:
        return self._vencedor

    def jogar(self, linha: int, coluna: int) -> None:
        jogada = self._X if self._jogador == 1 else self._O

        if self._tabuleiro[linha-1][coluna-1] == self.__valor_padrao:
            self._tabuleiro[linha-1][coluna-1] = jogada
            self._checar_vencedor()
            self._trocar_jogador()
        else:
            raise PosicaoOcupadaError('A posição escolhida já está ocupada')
            # return False

    def _trocar_jogador(self) -> None:
        if self._jogador == 1:
            self._jogador = 2

        elif self._jogador == 2:
            self._jogador = 1

        else:
            raise JogadorInvalidoError('O jogador foi alterado indevidamente')
    
    def mostrar_tabuleiro(self) -> None:
        for linha in self._tabuleiro:
            for elemento in linha:
                print(f"{elemento:3}", end=' ')
            print()
    
    def checar_conclusao_tabuleiro(self) -> bool:
        
        for linha in self._tabuleiro:
            for elemento in linha:
                if elemento == self.__valor_padrao:
                    return False
        
        return True

    
    def _checar_vencedor(self) -> None:
        # tentar fazer 1 por 1

        # for linha in range(self.__linhas):
        #     primeira_coluna = self._tabuleiro[linha][0]
        #     if primeira_coluna == r'{}':
        #         break
        #     flag = 0

        #     for coluna in range(1, self.__colunas):
        #         if self._tabuleiro[linha][coluna] == primeira_coluna:
        #             flag += 1
            
        #     if flag >= 2:
        #         return self._jogador

        # checagem horizontal
        for a, b, c in self._tabuleiro:
            if a != self.__valor_padrao and a == b and b == c:
                self._vencedor = self._jogador

        # checagem vertical
        for coluna in range(self.__colunas):
            primeira_linha = self._tabuleiro[0][coluna]
            if primeira_linha == self.__valor_padrao:
                break
            flag = 0

            for linha in range(1, self.__linhas):
                if self._tabuleiro[linha][coluna] == primeira_linha:
                    flag += 1

            if flag >= 2:
                self._vencedor = self._jogador


