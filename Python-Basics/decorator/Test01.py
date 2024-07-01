import os

os.system("cls")


def dec1(f):
    def wrapper(*args):
        print("dec1 wrapper start")
        rv = f(*args)
        print("dec1 wrapper end")
        return rv

    return wrapper # Comment this to get a NoneType error.


@dec1
def fun(x):
    print("Inside fun()")
    return x

if __name__ == "__main__":
    # NOTE: Basically, what happens is:
    # 1. fun = dec1(fun)
    # 2. test = fun(3)

    test = fun(3) # OK
    # test = fun(3, 4) # ERROR: incorrect signature.
    print(test)
