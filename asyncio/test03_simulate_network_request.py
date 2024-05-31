import asyncio

async def simulate_processing():
    for i in range(10):
        print(i)
        await asyncio.sleep(0.15)

async def simulate_network_request():
    await asyncio.sleep(1.1)
    return {"data" : "Not Found 404"}

async def main():
    print("Inside main()")

    task = asyncio.create_task(simulate_processing())
    task2 = asyncio.create_task(simulate_network_request())
    
    print("End of main()")

    data = await task2
    print(data)
    await task # If not present, will not print all numbers of the range.

asyncio.run(main())