from typing import List, Optional, Tuple


class DynamicProgramming:

    def robBottomUp(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return nums[0]
        res = [0] * len(nums)
        res[0], res[1] = nums[0], max(nums[0], nums[1])
        # max_money = res[1]
        for i in range(2, len(nums)):
            res[i] = max(res[i - 1], res[i - 2] + nums[i])
        return res[-1]

    def robTopDown(self, nums: List[int]) -> int:
        memo = [-1] * len(nums)

        def dfs(i):
            if i >= len(nums):
                return 0
            if memo[i] != -1:
                return memo[i]
            memo[i] = max(dfs(i + 1), nums[i] + dfs(i + 2))
            return memo[i]

        return dfs(0)

    def lcs(self, A, B):

        a, b = len(A), len(B)
        x = [[0] * (b + 1) for _ in range(a + 1)]
        parent: list[list[Optional[Tuple[int, int]]]] = [
            [None] * (b + 1) for _ in range(a + 1)
        ]

        def build_lcs_string(parent):
            res = []
            i, j = 0, 0
            while i < a and j < b:
                if parent[i][j] == (i + 1, j + 1):
                    res.append(A[i])
                    i += 1
                    j += 1
                elif parent[i][j] == (i + 1, j):
                    i += 1
                else:
                    j += 1
            return "".join(res)

        for i in range(a - 1, -1, -1):
            for j in range(b - 1, -1, -1):
                if A[i] == B[j]:
                    x[i][j] = x[i + 1][j + 1] + 1
                    parent[i][j] = (i + 1, j + 1)
                else:
                    if x[i + 1][j] >= x[i][j + 1]:
                        x[i][j] = x[i + 1][j]
                        parent[i][j] = (i + 1, j)
                    else:
                        x[i][j] = x[i][j + 1]
                        parent[i][j] = (i, j + 1)

        return x[0][0], build_lcs_string(parent)

    from typing import Optional  # Added import to fix lint issues

    def lis(self, A):
        a = len(A)
        x = [1] * a
        parent: list[Optional[int]] = [None] * a

        def build_lis_string():
            lis, index = 1, 0
            for i in range(a):
                if x[i] > lis:
                    lis = x[i]
                    index = i

            res = []
            while index is not None and index < a:
                res.append(A[index])
                index = parent[index]
            return lis, "".join(res)

        for i in range(a - 1, -1, -1):
            for j in range(i + 1, a):
                if A[i] < A[j]:
                    if x[i] < x[j] + 1:
                        x[i] = x[j] + 1
                        parent[i] = j
        return build_lis_string()

    def coinChangeBottomUp(self, coins: List[int], amount: int) -> int:
        coin_change = {0: 0}

        def coin_change_util(coins, amount, coin_change):
            if amount in coin_change:
                return coin_change[amount]
            res = float("inf")
            for coin in coins:
                if amount - coin >= 0:
                    coin_change_small = coin_change_util(
                        coins, amount - coin, coin_change
                    )
                    res = (
                        min(res, 1 + coin_change_small)
                        if coin_change_small >= 0
                        else res
                    )

            res = -1 if res == float("inf") else res

            coin_change[amount] = res
            return res

        return coin_change_util(coins, amount, coin_change)

    def coinChangeTopDown(self, coins: List[int], amount: int) -> int:
        dp = [-1] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for c in coins:
                if i - c >= 0 and dp[i - c] >= 0:
                    dp[i] = min(dp[i], dp[i - c] + 1) if dp[i] != -1 else dp[i - c] + 1

        return dp[amount]


def main():
    dp = DynamicProgramming()
    dp_len, dp_str = dp.lcs("hieroglyphology", "michaelangelo")
    print(f"lcs len is {dp_len} and lcs string is {dp_str}")

    list_len, list_str = dp.lis("carbohydrate")
    print(f"lis len is {list_len} and lis string is {list_str}")


if __name__ == "__main__":
    main()
