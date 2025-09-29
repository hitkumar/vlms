import heapq
from typing import List


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums
        heapq.heapify(self.nums)
        while len(self.nums) > k:
            heapq.heappop(self.nums)

    def add(self, val: int) -> int:
        heapq.heappush(self.nums, val)
        if len(self.nums) > self.k:
            heapq.heappop(self.nums)
        return self.nums[0]

    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
            if len(heap) > k:
                heapq.heappop(heap)

        return heap[0]

    def findKthLargestQuickSelect(self, nums: List[int], k: int) -> int:
        k = len(nums) - k

        def quick_select(l, r):
            pivot, p = nums[r], l
            for i in range(l, r):
                if nums[i] <= pivot:
                    nums[p], nums[i] = nums[i], nums[p]
                    p += 1
            nums[p], nums[r] = pivot, nums[p]
            if p == k:
                return nums[p]
            elif p < k:
                return quick_select(p + 1, r)
            else:
                return quick_select(l, p - 1)

        return quick_select(0, len(nums) - 1)

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # simulating a max heap using negative distance
        maxHeap = []
        for x, y in points:
            dist = -(x**2 + y**2)
            heapq.heappush(maxHeap, (dist, x, y))
            if len(maxHeap) > k:
                heapq.heappop(maxHeap)

        res = []
        assert len(maxHeap) == k
        while maxHeap:
            dist, x, y = heapq.heappop(maxHeap)
            res.append([x, y])
        return res

    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-x for x in stones]
        heapq.heapify(stones)
        while len(stones) > 1:
            largest = heapq.heappop(stones)
            second_largest = heapq.heappop(stones)
            if largest != second_largest:
                heapq.heappush(stones, -1 * abs(largest - second_largest))

        return stones[0] * -1 if stones else 0


if __name__ == "__main__":
    k = 3
    nums = [4, 5, 8, 2]
    obj = KthLargest(k, nums)
    print(obj.nums[0])
