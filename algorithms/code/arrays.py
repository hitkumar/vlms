from typing import List


class ArrayProblems:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numbers = {}
        for i, num in enumerate(nums):
            if target - num in numbers:
                return [numbers[target - num], i]
            numbers[num] = i
        return []
