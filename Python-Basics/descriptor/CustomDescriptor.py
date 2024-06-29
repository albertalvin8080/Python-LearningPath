import os

os.system("cls")


class Descriptor:

    # owner -> Class where the Descriptor is placed.
    # name -> Name of the attribute the Descriptor was assigned to.
    def __set_name__(self, owner, name):
        print(owner, name)
        self.public_name = name
        self.private_name = "_" + name

    # instance -> Instance of the class (object).
    # owner -> Class where the Descriptor is placed.
    def __get__(self, instance, owner):
        print(self, "get", instance, owner)
        return getattr(instance, self.private_name)

    # instance -> Instance of the class (object).
    # value -> Value to assigned to the private variable.
    def __set__(self, instance, value):
        print(self, "set", instance, value)
        setattr(instance, self.private_name, value)


class MyClass:
    age = Descriptor()


my_obj = MyClass()
my_obj.age = 34
print(my_obj.age)
