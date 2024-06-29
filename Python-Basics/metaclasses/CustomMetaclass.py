import os

os.system("cls")


# NOTE: namespace == attributes and methods
class CustomMeta(type):
    def __new__(cls, class_name, bases, namespace):
        print(namespace)

        # Changing all attributes and methods to uppercase (just for test)
        temp = {}
        for key, value in namespace.items():
            if key.startswith("__"):
                temp[key] = value
            else:
                temp[key.upper()] = value

        # return type(class_name, bases, temp)  # don't forget to return the class_object
        return super().__new__(cls, class_name, bases, temp)


class MyClass(metaclass=CustomMeta):
    x = 1
    y = 35

    def __init__(self):
        print("Inside MyClass.__init__()")

    def fun(self):
        print("Inside MyClass.fun()")


my_obj = MyClass()

# my_obj.fun() # ERROR
my_obj.FUN()  # OK

# print(my_obj.y) # ERROR
print(my_obj.Y)  # OK
