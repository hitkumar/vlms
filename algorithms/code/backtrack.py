class Backtracking:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []

        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            dfs(i + 1)

        dfs(0)
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for num in nums:
            res.extend([subset + [num] for subset in res])
        return res

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        subset = []

        def dfs(i):
            if i >= len(nums):
                res.append(subset.copy())
                return
            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            dfs(i + 1)

        nums.sort()
        dfs(0)
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        visited = set()
        rows, cols = len(board), len(board[0])
        # res = False
        directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        def backtrack(word, w_index, i, j):
            if w_index == len(word):
                return True
            if i < 0 or i >= rows or j < 0 or j >= cols or (i, j) in visited:
                return False
            if board[i][j] == word[w_index]:
                visited.add((i, j))
                for d in directions:
                    res = backtrack(word, w_index + 1, i + d[0], j + d[1])
                    if res == True:
                        return res
                visited.remove((i, j))

            return False

        for i in range(rows):
            for j in range(cols):
                if backtrack(word, 0, i, j):
                    return True

        return False

    def letterCombinations(self, digits: str) -> List[str]:
        lettersMap = {
            "2": "ABC",
            "3": "DEF",
            "4": "GHI",
            "5": "JKL",
            "6": "MNO",
            "7": "PQRS",
            "8": "TUV",
            "9": "WXYZ",
        }
        res = []
        curr_str = []

        def backtrack(digits, index):
            if digits == "":
                return
            if index == len(digits):
                res.append("".join(curr_str))
                return
            for c in lettersMap[digits[index]]:
                curr_str.append(c.lower())
                backtrack(digits, index + 1)
                curr_str.pop()

        backtrack(digits, 0)
        return res
