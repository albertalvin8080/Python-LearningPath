import asyncio

async def main(arg: int):
    print(f"inside main({arg})")
    task = asyncio.create_task(main(2))
    await asyncio.create_task(main(3)) # deadlock

asyncio.run(main(1))

# asyncio.create_task(main()) # doesn't work outside a coroutine