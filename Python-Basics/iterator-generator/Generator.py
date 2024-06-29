import os

os.system("cls")


# NOTE: A Generator is just an Iterator which receives
# implementations of __iter__() and __next__() automatically.
def custom_generator():
    start = 0
    end = 5

    while start < end:
        yield start
        start += 1


my_cgen = custom_generator()
print(dir(my_cgen))
print(custom_generator)

# Same output
print(my_cgen)
print(iter(my_cgen))

print(next(my_cgen))
print(next(my_cgen))
print(next(my_cgen))
print(next(my_cgen))
print(next(my_cgen))
# print(next(my_cgen))  # StopIteration exception.

print("-"*30)
for n in custom_generator():
    print(n, end=", ")
