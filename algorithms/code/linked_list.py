from typing import List, Optional


class ListNode:
    def __init__(self, x):
        self.item = x
        self.next = None

    # def later_node(self, i):
    #     if i == 0:
    #         return self
    #     assert self.next
    #     return self.next.later_node(i - 1)

    def later_node(self, i):
        if i == 0:
            return self
        node = self
        while i > 0:
            node = node.next
            i -= 1
        return node


class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        node = self.head
        while node:
            yield node.item
            node = node.next

    def build(self, X):
        for a in reversed(X):
            self.insert_first(a)

    def get_at(self, i):
        node = self.head.later_node(i)
        return node.item

    def set_at(self, i, x):
        node = self.head.later_node(i)
        node.item = x

    def insert_first(self, x):
        new_node = ListNode(x)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def delete_first(self):
        if self.size == 0:
            return
        x = self.head.item
        self.head = self.head.next
        self.size -= 1
        return x

    def insert_at(self, i, x):
        if i == 0:
            self.insert_first(x)
            return
        node = self.head.later_node(i - 1)
        new_node = ListNode(x)
        new_node.next = node.next
        node.next = new_node
        self.size += 1

    def delete_at(self, i):
        if i == 0:
            return self.delete_first()
        node = self.head.later_node(i - 1)
        x = node.next.item
        node.next = node.next.next
        self.size -= 1
        return x

    def insert_last(self, x):
        return self.insert_at(self.size - 1, x)

    def delete_last(self):
        return self.delete_at(self.size - 1)

    def reverseListUtil(self, head: Optional[ListNode]):
        if head.next == None:
            return (head, head)
        curr = head
        nextListHead, nextListTail = self.reverseListUtil(head.next)
        nextListTail.next = curr
        nextListTail = nextListTail.next
        return (nextListHead, nextListTail)

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None:
            return None
        reverseListHead, reverseListTail = self.reverseListUtil(head)
        reverseListTail.next = None
        return reverseListHead

    def reverseListIter(self, head: Optional[ListNode]):
        if head == None:
            return None
        prev, curr = None, head
        while curr != None:
            curr_next = curr.next
            curr.next = prev
            prev = curr
            curr = curr_next
            # if prev.next:
            #     print(curr.val, curr.next.val)
        return prev

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        def find_len(head: Optional[ListNode]):
            i = 0
            while head != None:
                i += 1
                head = head.next
            return i

        len = find_len(head)
        if len == 1:
            return None
        if n == len:
            return head.next

        headNode = head
        pos_from_start = len - n - 1
        i = 0
        while head != None and i < pos_from_start:
            head = head.next
            i += 1
        head.next = head.next.next
        return headNode

    def reorderListBrute(self, head: Optional[ListNode]) -> None:
        if head == None:
            return None

        listHead = head
        # could just use a python list as it is dynamic
        nodes = {}
        i = 0
        while head != None:
            nodes[i] = head
            i += 1
            head = head.next

        start, end = 0, i - 1
        dummyNode = ListNode(0)
        head = dummyNode
        while start < end:
            dummyNode.next = nodes[start]
            dummyNode = dummyNode.next
            dummyNode.next = nodes[end]
            dummyNode = dummyNode.next
            start += 1
            end -= 1

        if start == end:
            dummyNode.next = nodes[start]
            dummyNode = dummyNode.next
            start += 1
        dummyNode.next = None
        head = head.next

    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        # find middle and last of the list
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        second = slow.next
        slow.next = None

        # reverse the second part of the list
        prev, curr = None, second
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp

        # prev is the head of the reversed list, head is the head of first list
        while prev:
            head_next, prev_next = head.next, prev.next
            head.next = prev
            prev.next = head_next
            prev, head = prev_next, head_next


def main():
    l1 = LinkedList()
    l1.build([1, 2, 3, 4, 5])
    print(l1.head.item)
    assert l1.get_at(0) == 1
    assert l1.get_at(1) == 2
    assert l1.get_at(2) == 3
    assert l1.get_at(3) == 4
    # print(len(l1))


if __name__ == "__main__":
    main()
