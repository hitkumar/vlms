from collections import defaultdict, deque, heapq
from typing import List


class StaticArray:
    def __init__(self, n):
        self.data = [None] * n

    def get_at(self, i):
        if not (0 <= i < len(self.data)):
            raise IndexError
        return self.data[i]

    def set_at(self, i, x):
        if not (0 <= i < len(self.data)):
            raise IndexError
        self.data[i] = x


class Array_Seq:
    def __init__(self):
        self.A = []
        self.size = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        yield from self.A

    def build(self, X):
        self.A = [a for a in X]
        self.size = len(self.A)

    def _copy_forward(self, i, n, A, j):
        for k in range(n):
            A[j + k] = self.A[i + k]

    def insert_at(self, i, x):
        n = len(self)
        A = [None] * (n + 1)
        self._copy_forward(0, i, A, 0)
        A[i] = x
        self._copy_forward(i, n - i, A, i + 1)
        self.build(A)


class ArrayProblems:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numbers = {}
        for i, num in enumerate(nums):
            if target - num in numbers:
                return [numbers[target - num], i]
            numbers[num] = i
        return []

    def sorted_str(self, s: str):
        return "".join(sorted(s))

    def sort_efficient(self, s: str):
        count = [0] * 26
        for c in s:
            count[ord(c) - ord("a")] += 1
        return tuple(count)

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams = {}
        for s in strs:
            str_hash = self.sorted_str(s)
            if str_hash in anagrams:
                anagrams[str_hash].append(s)
            else:
                anagrams[str_hash] = [s]

        return [v for k, v in anagrams.items()]

    def groupAnagrams2(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        for s in strs:
            sortedS = self.sorted_str(s)
            res[sortedS].append(s)
        return list(res.values())

    def topKFrequent_Heap(self, nums: List[int], k: int) -> List[int]:
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        heap = []
        for num, freq in count.items():
            heapq.heappush(heap, (freq, num))
            if len(heap) > k:
                heapq.heappop(heap)

        res = []
        for i in range(k):
            res.append(heapq.heappop(heap)[1])
        return res

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Bucket sort approach in O(n) time
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        freqs = [[] for i in range(len(nums) + 1)]
        for num, cnt in count.items():
            freqs[cnt].append(num)
        res = []
        for i in range(len(freqs) - 1, 0, -1):
            for num in freqs[i]:
                res.append(num)
                if len(res) == k:
                    return res

        return []
