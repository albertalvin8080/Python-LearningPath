
print('/////////////////////////////////////')

# with open('./Python/Arquivos/abc.txt', '+r', encoding='utf-8') as fl:
#     text = input('Say something: ')
#     i = fl.write(text) # i vai ter o numero de bytes escritos no arquivo
#     fl.seek(0,0)
#     print(fl.read())

with open('./binario.bin','+ba') as fl: # modo binario nao aceita encoding='utf-8'
    # byte = '\nappending another line because of a test\nnothing so see here'
    # fl.write(byte.encode('utf-8'))
    fl.seek(0,0)
    print(fl.read().decode('utf-8'))