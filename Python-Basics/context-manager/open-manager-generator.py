import os
from contextlib import contextmanager

os.system("cls")


# NOTE: This is just an example. open() by itself is a Context Manager.
@contextmanager
def open_manager(path, mode):
    try:
        print("Inside try")
        f = open(path, mode)
        yield f
    finally:
        print("Inside finally")
        f.close()


if __name__ == "__main__":
    with open_manager("./some-dir/hello.txt", "r") as f:
        text = f.read()
        # raise ValueError("Testing raise")
        print(text)
