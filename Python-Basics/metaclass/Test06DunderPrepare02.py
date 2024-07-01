import os

os.system("cls")


class MyMeta(type):
    @classmethod
    def __prepare__(metacls, name: str, bases: tuple[type, ...]):
        print("Inside MyMeta.__prepare__()", metacls)
        return {"age": 35}  # Testing

    def __call__(self, *args, **kwds):
        # pass
        print("Inside MyMeta.__call__()", self)
        print(self.__mro__)
        print(self.mro())

    def __new__(cls, class_name, bases, namespace):
        print("Inside MyMeta.__new__()", cls)
        print(namespace)  # Should contain "age":35
        return super().__new__(cls, class_name, bases, namespace)


class MyClass(metaclass=MyMeta):
    pass


if __name__ == "__main__":
    MyClass()
