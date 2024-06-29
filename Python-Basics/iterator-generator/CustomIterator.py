import os

os.system("cls")

class MyIterator:
    def __init__(self) -> None:
        self.start = 0
        self.end = 5

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start >= self.end:
            raise StopIteration("Bruh")
        value = self.start
        self.start += 1
        return value
    
myIter = MyIterator()
print(dir(myIter))
print(MyIterator)

# Same output
print(iter(myIter))
print(myIter.__iter__())

print(next(myIter))
print(next(myIter))
print(next(myIter))
print(next(myIter))
print(next(myIter))
# print(next(myIter)) # StopIteration exception

print("-"*30)
for n in MyIterator():
    print(n, end=", ")

