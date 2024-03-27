from jogo_da_velha import * # procurar evitar fazer isso

tabuleiro = JogoDaVelha()

flag = 0
while flag != 1:
    try:
        print(f'Jogador: {tabuleiro.jogador}')
        tabuleiro.mostrar_tabuleiro()
        print('Insira a linha e a coluna:')

        l = int(input()[0])
        c = int(input()[0])

        tabuleiro.jogar(l, c)

        print()
    
    except ValueError as e:
        print('\n*Valor invalido inserido. Tente Novamente.*\n')
    
    except IndexError as e:
        print('\n*O valor inserido excede o limite de linhas e colunas*\n')

    except PosicaoOcupadaError as e:
        print('\n*Posição já ocupada. Tente novamente.*\n')
    
    else: # só executa se não houver exceção
        if tabuleiro.checar_conclusao_tabuleiro() == True:
            print('O jogo empatou.')
            tabuleiro.mostrar_tabuleiro()
            break

        elif tabuleiro.vencedor != None:
            print(f'Jogador {tabuleiro.vencedor} venceu!!!')
            tabuleiro.mostrar_tabuleiro()
            break

    # except JogadorInvalidoError as e:
    #     print('\n*Jogador alterado indevidamente*\n')

