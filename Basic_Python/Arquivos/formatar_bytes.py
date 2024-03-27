def formate(size):
    Kb = 1024
    Mb = Kb ** 2
    Gb = Kb ** 3
    Tb = Kb ** 4
    Pb = Kb ** 5

    if size < Kb:
        u = 'B'
    elif size < Mb:
        size /= Kb
        u = 'Kb'
    elif size < Gb:
        size /= Mb
        u = 'Mb'
    elif size < Tb:
        size /= Gb
        u = 'Gb'
    elif size < Pb:
        size /= Tb
        u = 'Tb'
    else:
        size /= Pb
        u = 'Pb'
    size = round(size, 2)
    return f'{size}{u}'

if __name__ == '__main__':
    print('main')