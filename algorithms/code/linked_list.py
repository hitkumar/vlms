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
