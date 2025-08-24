from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeTwoLists(
    self, list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    if list1 is None:
        return list2
    if list2 is None:
        return list1
    if list1.val < list2.val:
        list1.next = self.mergeTwoLists(list1.next, list2)
        return list1
    else:
        list2.next = self.mergeTwoLists(list1, list2.next)
        return list2


def mergeTwoLists2Pointer(
    self, list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    if list1 is None:
        return list2
    if list2 is None:
        return list1
    dummyNode = ListNode()
    while list1 and list2:
        if list1.val < list2.val:
            dummyNode.next = list1
            list1 = list1.next
        else:
            dummyNode.next = list2
            list2 = list2.next
        dummyNode = dummyNode.next

    if list1:
        dummyNode.next = list1
    if list2:
        dummyNode.next = list2
    return dummyNode.next
