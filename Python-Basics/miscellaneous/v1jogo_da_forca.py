word = list("Superioridade")

length = len(word)
placeholder = ["_" for _ in range(length)]

tries = 0
win = 0

while tries < length:
    print(''.join(placeholder))
    ip = input("Insira uma letra:")

    try:
        index = word.index(ip)
    except ValueError: 
        index = -1

    if index == -1:
        tries+=1
        print("WRONG")
    else:
        win += 1
        placeholder[index] = ip
        word[index] = '*'

    if win >= length:
        print("YOU WIN")
        break