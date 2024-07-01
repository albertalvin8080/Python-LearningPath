import os
from contextlib import contextmanager

os.system("cls")

# This Context Manager is used to enter the provided directory, and at the end of the with statement,
# the manager will change the current directory back to where it first was.
@contextmanager
def directory_manager(directory):
    try:
        print("try block")
        cwd = os.getcwd()
        os.chdir(directory)
        yield
    finally:
        print("finally block")
        os.chdir(cwd)

if __name__ == "__main__":
    with directory_manager("some-dir"):
        items = os.listdir()
        # raise ValueError("Testing finally")
        print(items)