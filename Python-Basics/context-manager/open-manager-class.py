import os

os.system("cls")


# NOTE: This is just an example. open() by itself is already a Context Manager.
class OpenManager:
    def __init__(self, path, mode) -> None:
        self.path = path
        self.mode = mode

    def __enter__(self):
        print("Inside __enter__")
        self.file = open(self.path, self.mode)
        return self.file

    def __exit__(self, exec_type, exec_val, traceback):
        print("Inside __exit__", exec_type, exec_val, traceback)
        self.file.close()


if __name__ == "__main__":
    with OpenManager("./some-dir/hello.txt", "r") as f:
        text = f.read()
        # raise ValueError("Testing raise")
        print(text)
