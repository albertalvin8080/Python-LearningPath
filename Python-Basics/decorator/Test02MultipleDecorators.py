import os

os.system("cls")


def dec1(f):
    def wrapper(*args):
        print("dec1 wrapper start")
        rv = f(*args)
        print("dec1 wrapper end")
        return rv

    return wrapper


def dec2(f):
    def wrapper(*args):
        print("dec2 wrapper start")
        rv = f(*args)
        print("dec2 wrapper end")
        return rv

    return wrapper

# NOTE: dec1() wrapps fun() and dec2() wrapps dec1().
@dec2
@dec1
def fun(x):
    print("Inside fun()")
    return x

if __name__ == "__main__":
    test = fun(3)
    print(test)
