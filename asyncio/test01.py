import asyncio

async def say_hello():
    return "hello"

hello = asyncio.run(say_hello())
print(hello)