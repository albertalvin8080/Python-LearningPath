import os

os.system("cls")


# f -> function
def abstractmethod(f):
    # print(f)
    f.__abstract__ = True
    return f


# NOTE: When whe override methods, the attrubute __abstract__ is removed.
# That's how we know if we're dealing with a concrete or an abstract method.
def abstrac_tmethods(cls):
    print("Inside isabstract()", cls)
    # This will contain the names of the non-implemented methods, if any.
    abstract_methods_list = []

    # vars() -> Without arguments, equivalent to locals(). With an argument, equivalent to object.__dict__.
    for key, val in vars(cls).items():
        if getattr(val, "__abstract__", False):
            abstract_methods_list.append(key)

    return abstract_methods_list


class AbstractMeta(type):
    # __call__() is called when we do MetaWrapper() or any of its subclasses:
    # - MyAbstractClass()
    # - MyConcreteClass()
    def __call__(cls, *args, **kwargs):
        print("Inside AbstractMeta.__call__()")

        abs_methods_list = abstrac_tmethods(cls)
        # False if the there's no abstract method inside the returned list.
        if abs_methods_list:
            raise TypeError("no implementation for: " + ", ".join(abs_methods_list))
        return super().__call__(*args, **kwargs)


# The purpose of this blank class is to prevent users from knowing that there's a meta class called AbstractMeta.
# Basically, they don't need to know the existence of the Metaclass, only of this wrapper class.
class MetaWrapper(metaclass=AbstractMeta):
    pass


class MyAbstractClass(MetaWrapper):
    def __init__(self) -> None:
        print("Inside MyAbstractClass.__init__()")
        super().__init__()

    @abstractmethod
    def fun(self):
        pass

    @abstractmethod
    def hello(self):
        pass


# NOTE: When whe override methods, the attrubute __abstract__ is removed.
# That's how we know if we're dealing with a concrete or an abstract class.
class MyConcreteClass(MyAbstractClass):
    def __init__(self) -> None:
        print("Inside MyConcreteClass.__init__()")
        super().__init__()

    def fun(self):
        print("Inside MyConcreteClass.fun()")

    def hello(self):
        print("Saying hello")


if __name__ == "__main__":
    # MetaWrapper() # OK. Why? Because there're no abstract methods inside this wrapper.
    # MyAbstractClass() # ERROR: uninplemented abstract methods.

    my_obj = MyConcreteClass()  # OK: all abstract methods were overriden.
    print(my_obj)
    my_obj.fun()
    my_obj.hello()

    # NOTE: execution order:
    # 1. __prepare__(): Prepare the namespace for the class.
    # 2. __new__(): Create the class object.
    # 3. __init__(): Initialize the class object.
    # 4. __call__(): Create and initialize instances of the class.
