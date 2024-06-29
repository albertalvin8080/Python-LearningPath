import os

os.system("cls")


def create_MyClass():

    class_name = "MyClass"
    bases = ()
    # Beware of the indentation.
    body = """
x = 34
y = -34
lamb = lambda x: print(x)
def fun(self, x):
    print(x)
"""

    # This line prepares an empty namespace (a dictionary) where the class body will be executed.
    namespace = type.__prepare__(class_name, bases)
    print(namespace)

    # globals() -> Return the dictionary containing the current scope's global variables.
    # exec() -> Executes the given source in the context of globals and locals.
    #           It is being used to execute the class body within a specific namespace.
    #           Passing globals() ensures that the executed code has access to the same 
    #           global scope as the rest of your program, which can be important for maintaining consistency.
    # exec(body, globals(), namespace)
    exec(body, {}, namespace) # You may omit globals() in this case.

    print(globals())
    print(namespace)

    return type(class_name, bases, namespace)


my_obj = create_MyClass()
my_obj.lamb("Message for lamb()")
my_obj.lamb("Message for fun()")
