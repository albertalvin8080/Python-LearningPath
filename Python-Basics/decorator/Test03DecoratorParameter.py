import os

os.system("cls")


# 1. This decorator receives the message and returns the actual_decorator.
# 2. The actual decorator receives the decorated function func.
# 3. The actual decorator then returns a wrapper containing the function func.
def decorator(message):
    print(message)

    def actual_decorator(f):
        def wrapper(*args, **kwargs):
            print("Start wrapper")
            rv = f(*args, **kwargs)
            print("End wrapper")
            return rv

        return wrapper

    return actual_decorator


# You're actually calling the function `decorator(...)` here, that's why it needs to
# return a function (`actual_decorator`) which will receive the decorated function `func``.
@decorator("Hello, message")
def func(a, b):
    return a + b


result = func(2, 4)
print(result)
