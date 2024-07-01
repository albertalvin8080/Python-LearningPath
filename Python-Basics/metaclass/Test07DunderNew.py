import os, time

os.system("cls")


class Meta(type):
    start = time.perf_counter()

    def __new__(mcs, class_name, bases, namespace):
        end = time.perf_counter()
        namespace["__class_instantiation_time__"] = end - mcs.start
        print(mcs.start, end)

        return super().__new__(mcs, class_name, bases, namespace)


class MyClass(metaclass=Meta):
    pass


if __name__ == "__main__":
    my_obj = MyClass()
    print(my_obj.__class_instantiation_time__)
    # print(dir(my_obj))
