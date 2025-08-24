from typing import List


def maxProfit(prices: List[int]) -> int:
    if prices is None or len(prices) == 0:
        return 0
    max_profit = 0
    min_value = prices[0]
    for i in range(1, len(prices)):
        max_profit = max(max_profit, prices[i] - min_value)
        min_value = min(min_value, prices[i])
    return max_profit


if __name__ == "__main__":
    print(
        maxProfit(
            [
                10,
                1,
                5,
                6,
                7,
            ]
        )
    )
    print(maxProfit([10, 8, 7, 6]))
