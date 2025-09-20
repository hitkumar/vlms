# Good resource to learn this: https://www.youtube.com/watch?v=Qb9s3UiMSTA
import asyncio
import functools
import time


# Timing decorator for synchronous functions
def timer(func):
    """Decorator that prints the execution time of a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


# Timing decorator for asynchronous functions
def async_timer(func):
    """Decorator that prints the execution time of an async function"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Async function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


# synchronization primitives in asyncio
lock = asyncio.Lock()
shared_resource = 0
semaphore = asyncio.Semaphore(2)
event = asyncio.Event()


async def modify_shared_resource():
    global shared_resource
    async with lock:
        # critical section
        print(f"Resource before modification: {shared_resource}")
        shared_resource += 1
        await asyncio.sleep(1)
        print(f"Resource after modification: {shared_resource}")


async def main_lock():
    await asyncio.gather(*[modify_shared_resource() for _ in range(5)])


async def fetch_data(delay, id):
    print(f"start fetching data for {id}")
    await asyncio.sleep(delay)
    print(f"done fetching data for {id}")
    return {"data": "Some data", "id": id}


@async_timer
async def main_tasks():
    results = await asyncio.gather(
        fetch_data(1, 1),
        fetch_data(2, 2),
        fetch_data(3, 3),
    )
    for result in results:
        print(f"received result: {result}")


# separate coroutine function
@async_timer
async def main():
    print("start of main coroutine")
    task1 = fetch_data(1, 1)
    task2 = fetch_data(2, 2)
    task3 = fetch_data(3, 3)
    result1 = await task1
    result2 = await task2
    result3 = await task3
    print(f"result1: {result1}")
    print(f"result2: {result2}")
    print(f"result3: {result3}")


# asyncio.run(main())
asyncio.run(main_tasks())
# asyncio.run(main_lock())
# print(f"shared resource: {shared_resource}")
