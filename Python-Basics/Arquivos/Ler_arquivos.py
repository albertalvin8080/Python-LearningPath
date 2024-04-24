import os
os.system('cls' if os.name == 'nt' else 'clear')

def conversion(size):
    Kb = 1024
    Mb = Kb ** 2
    Gb = Kb ** 3
    Tb = Kb ** 4
    Pb = Kb ** 5

    if size < Kb:
        text = 'B'

    elif size < Mb:
        size /= Kb
        text = 'Kb'

    elif size < Gb:
        size /= Mb
        text = 'Mb'

    elif size < Tb:
        size /= Gb
        text = 'Gb'

    else:
        size /= Tb
        text = 'Tb'
    
    return f'{size:.2f}{text}'

path_input = input('Insira um caminho: ')
path = r'' + path_input
find = input('Insira um caractere: ')
print()

for root, dirs, files in os.walk(path):
    # print(dirs)
    for fl in files:
        if find in fl:
            full_file_path = os.path.join(root, fl)
            name, extension = os.path.splitext(fl)
            size = os.path.getsize(full_file_path)

            print(f'Name: {name}')
            print(f'Extension: {extension}')
            print(f'Size: {conversion(size)}')
            print(f'Full Path: {full_file_path}')
            print()

# input()
# os.system('cls' if os.name == 'nt' else 'clear')
