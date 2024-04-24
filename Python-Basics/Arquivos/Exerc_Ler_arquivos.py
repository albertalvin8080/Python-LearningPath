import os
import formatar_bytes
# print(dir(formatar_bytes))

print('///////////////////////////////')

old_path = r'C:\Users\Albert\Downloads'
new_path = r'C:\Users\Albert\Downloads\Nova pasta'
find = '00'

for root, dirs, files in os.walk(old_path):
    for fl in files:
        if find in fl:
            full_file_path = os.path.join(root, fl)
            name, extension = os.path.splitext(fl)
            size = os.path.getsize(full_file_path)
            print(f'Name: {name}')
            print(f'Extension: {extension}')
            print(f'Size: {size}')
            print(f'Formated Size: {formatar_bytes.formate(size)}')
            print(f'Full Path: {full_file_path}')
            print()
