import os

os.system("cls")

# def fun(self, x):
#     print(x)

MyClass = type("MyClass", (), {"fun": lambda self, x: print(x)})

# print(dir(MyClass))
print(MyClass)

my_obj = MyClass()
my_obj.fun("Inside Lambda func")

print(dir(my_obj)) # fun() is present.
