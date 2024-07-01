import os, inspect
from typing import Any

os.system("cls")

# NOTE:
"""
  1 - Create a OverloadDictionary which stores methods decorated with @overload inside a OverloadList.
      It's necessary in order not to replace previously declared methods inside the class body.
  2 - Create a OverloadDescriptor which is a descriptor used to decide which method to call based on
      its signatures inside the OverloadList (the OverloadList stores functions, not just signatures. 
      You need to extract the signatures from them).
  3 - Create a Metaclass which uses the OverloadDictionary and replaces the OverloadList inside it
      with the OverloadDescriptor which will manage the list.
"""

# f -> function
def overload(f):
    f.__overload__ = True
    return f


# Used solely to differentiate a OverloadList from a normal list inside the OverloadDict and OverloadDescriptor.
class OverloadList(list):
    pass


class OverloadDict(dict):
    # At some point, `value` will be the function we are trying to override.
    def __setitem__(self, key: Any, value: Any) -> None:
        print("Inside OverloadDict.__setitem__()", value)
        assert isinstance(key, str), "Key should be a str."

        # Gets the previous value if already stored.
        previous_value = self.get(key, False)
        # Checks for the presence of the __overload__ attribute.
        isoverload = getattr(value, "__overload__", False)

        if not previous_value:
            new_value = OverloadList([value]) if isoverload else value
            return super().__setitem__(key, new_value)

        # At this point, we add the new overload to the OverloadList.
        elif isinstance(previous_value, OverloadList):
            if not isoverload:
                raise ValueError(
                    "Trying to overload a function which is not decorated."
                )
            previous_value.append(value)

        # If the value is not an overload, just replace it normally inside the dictionary.
        else:
            if isoverload:
                raise ValueError("Trying to overload an invalid type.")
            return super().__setitem(key, value)


# This guy here intercepts calls to a method and decides which overloaded method should be called.
class OverloadDescriptor:
    def __set_name__(self, owner, name):
        self.owner = owner  # Class where the Descriptor is instantiated.
        self.name = name  # Name of the variable which stores the Descriptor.

    def __init__(self, overload_list):
        if not isinstance(overload_list, OverloadList):
            raise TypeError("overload_list must be of type OverloadList.")
        if not overload_list:
            raise ValueError("overload_list must not be empty.")

        self.overload_list = overload_list
        self.signatures = [inspect.signature(method) for method in overload_list]

    def __get__(self, instance, _owner=None):
        print("Inside OverloadDescriptor.__get__()", instance)

        # TESTING
        # Test results:
        # 1 - The lambda doesn't receive the `self` parameter.
        # 2 - The lambda only receives the parameters passed in the call to fun(...).
        #     If no parameter was passed, none shall be received by the lambda.
        # 3 - Using *args to receive any parameters (or none) prevents the raising of an error.
        # return lambda *args: print(args) # NOTE: Uncomment this for testing.

        # You could also create a separated class which implements the dunder __call__(self, *args, **kwargs)
        #       and return it instead of creating this nested function here.
        # NOTE: Notice that we must NOT pass `self` for this bound function because it is not a callable class.
        #       But if you used a __call__(...), then of course you would pass it, but it would be the self
        #       for the callable class (MyCallable.__call__(...)), not for the class which will call it 
        #       (MyClass, in this case).
        def bound(*args, **kwargs):
            for method, signature in zip(self.overload_list, self.signatures):
                try:
                    # Get a BoundArguments object, that maps the passed args and kwargs to the function's signature.
                    binding = signature.bind(instance, *args, **kwargs)
                except TypeError:
                    # If this error is raised, it means that the signature or received 
                    # extra args/kwargs, or didn't receive needed args/kwargs.
                    pass # pass the except because we want to continue the search.
                else:
                    # Sets default values for missing arguments.
                    # - For variable-positional arguments (*args) the default is an empty tuple.
                    # - For variable-keyword arguments (**kwargs) the default is an empty dict.
                    binding.apply_defaults()
                    return method(instance, *args, **kwargs)

            raise ValueError("No overload matches the provided signature.")
        
        return bound


class Meta(type):
    @classmethod
    def __prepare__(metacls, name: str, bases: tuple[type, ...], /, **kwds: Any):
        print("Inside Meta.__prepare__()")
        return OverloadDict()

    def __new__(cls, name, bases, namespace):
        print("Inside Meta.__new__()", namespace)
        new_namespace = {
            key: OverloadDescriptor(val) if isinstance(val, OverloadList) else val
            for key, val in namespace.items()
        }
        return super().__new__(cls, name, bases, new_namespace)


class MyClass(metaclass=Meta):
    @overload
    def fun(self):
        print("Inside fun()")

    @overload 
    def fun(self, x):
        print(f"Inside fun(x): {x}")

    @overload # Comment this decorator to get an overload error.
    def fun(self, a, b):
        print(f"a + b = {a+b}")


if __name__ == "__main__":
    print("-"*50)
    my_obj = MyClass()
    my_obj.fun()
    my_obj.fun("Hello")
    my_obj.fun(2 + 12)
    # my_obj.fun(1, 2, 3) # ERROR: No matching signature.
