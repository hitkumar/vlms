import heapq
from collections import deque


def isValid(s: str) -> bool:
    stack = []
    closeToOpen = {")": "(", "]": "[", "}": "{"}
    for c in s:
        if c in closeToOpen:
            if stack and stack[-1] == closeToOpen[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    return True if not stack else False


# nested functions
def outer(a, b):
    c = "c"

    def inner():
        print(c)


if __name__ == "__main__":
    # print(isValid("()[]{}"))
    # print(isValid("([)]"))
    # list as queue
    queue = []
    queue.append(1)
    queue.append(2)
    assert queue.pop(0) == 1

    stack = []
    stack.append(1)
    stack.append(2)
    assert stack.pop() == 2

    q1 = deque()

    # Python things taken from https://www.youtube.com/watch?v=0K_eZGS5NsU
    n = 0
    while n < 10:
        q1.append(n)
        n += 1

    for i in range(5, 1, -1):
        print(i)

    # Arrays
    arr = [1, 2, 3, 4, 5]
    arr[1:3]
    arr.sort(key=lambda x: x)

    arr = [[0] * 4 for _ in range(4)]
    print(ord("a"))

    mySet = {i for i in range(10)}
    myMap = {i: i * 2 for i in range(10)}

    # tuple can be keys for maps ans sets, Lists cannot be hashed. Tuples are immutable though
    minHeap = []
    heapq.heappush(minHeap, 3)
    heapq.heappush(minHeap, 1)
    heapq.heappush(minHeap, 2)
    print(minHeap[0])

    heapq.heapify([1, 2, 3, 4, 5])
