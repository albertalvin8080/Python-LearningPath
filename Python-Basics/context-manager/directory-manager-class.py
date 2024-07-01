import os

os.system("cls")

# This Context Manager is used to enter the provided directory, and at the end of the with statement,
# the manager will change the current directory back to where it first was.
class directoryManager:
    def __init__(self, directory) -> None:
        self.directory = directory

    def __enter__(self):
        print("Inside __enter__")
        self.cwd = os.getcwd()
        os.chdir(self.directory)
        # return # Doesn't need to return anything in this case
    
    def __exit__(self, exec_type, exec_val, traceback):
        print("Inside __exit__", exec_type, exec_val, traceback)
        os.chdir(self.cwd)


if __name__ == "__main__":

    with directoryManager("some-dir"):
        items = os.listdir()
        # raise ValueError("Testing exception raise") # __exit__ is called even when an exception is raised.
        print(items)