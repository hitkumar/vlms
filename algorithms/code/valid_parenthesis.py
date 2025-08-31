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
