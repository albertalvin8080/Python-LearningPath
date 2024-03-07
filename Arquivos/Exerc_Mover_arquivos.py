import os
import shutil
print('////////////////////////////////////')

old_path = r'C:\Users\Albert\Downloads'
new_path = r'C:\Users\Albert\Downloads\Exerc_Mover'
find = '.jpg'

try:
    os.mkdir(r'C:\Users\Albert\Downloads\Exerc_Mover')
except FileExistsError as e:
    # print(f'{e}')
    print(f'Path \'{new_path}\' already exists\n')

for root, dirs, files in os.walk(old_path):
    for fl in files:
        if find in fl:
            old_file_path = os.path.join(root, fl)
            new_file_path = os.path.join(new_path, fl)
            shutil.move(old_file_path, new_file_path)
            print(f'{fl} moved')
