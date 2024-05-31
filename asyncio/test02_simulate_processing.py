import asyncio

async def simulate_processing():
    for i in range(10):
        print(i)
        await asyncio.sleep(0.15)

async def main():
    print("Inside main()")
    await simulate_processing()
    print("End of main()")

asyncio.run(main())