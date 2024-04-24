import os
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

word = 'Super Android Aaabbbccc'
wordMainteiner = word
word = list(word)

# placeholder = ['_' if word[i] != ' ' else ' ' for i in range(len(word))]
placeholder = ['_' for _ in range(len(word))]
# print(''.join(placeholder))
# print(placeholder)

win = 0
misses = 0
maxMisses = 6

# printing white spaces ' '
for i in range(len(word)):
    if word[i] == ' ':
        word[i] = '*'
        placeholder[i] = ' '
        win += 1

while(misses < maxMisses and win < len(word)):
    count = 0
    clear_terminal()
    # print(win)
    print('Misses =',misses)

    print(''.join(placeholder))
    guess = input('Try a letter:')

    for indexLetter in range(len(word)):
        if word[indexLetter] == guess[0]:
            placeholder[indexLetter] = guess[0]
            word[indexLetter] = '*'
            count += 1
    
    if count == 0:
        misses+=1
    else:
        win+=count

print()
print('Word:',wordMainteiner)
print('Total Misses:',misses)
if win >= len(word):
    print('YOU WON')
else:
    print('YOU HAVE BEEN HANGED')
        

    