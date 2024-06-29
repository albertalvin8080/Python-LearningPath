import os

os.system("cls")


class MyClass:
    pass

myobj = MyClass()
print(myobj)
print(MyClass)

print(type(myobj))
print(type(MyClass))

print("-"*30)

# Params:
# 1. class_name
# 2. bases (ancestors)
# 3. namespace dict (attributes and methods) 
OtherClass = type("OtherClass", (), {})

other_obj = OtherClass()
print(other_obj)
print(OtherClass)

print(type(other_obj))
print(type(OtherClass))