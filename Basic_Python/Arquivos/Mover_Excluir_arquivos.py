import os
import shutil

os.system('cls' if os.name == 'nt' else 'clear')

original_path = r'C:\Users\Albert\Downloads'
new_path = r'C:\Users\Albert\Downloads\Nova Pasta'

# print(original_path)
for root, dirs, files in os.walk(original_path):
    for fl in files:
        original_file_path = os.path.join(root, fl)
        new_file_path = os.path.join(new_path, fl)

        # shutil.move(original_file_path, new_file_path)
        # shutil.copy(original_file_path, new_file_path)
        # os.remove(new_file_path)