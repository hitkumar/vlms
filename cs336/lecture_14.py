import mmh3
import numpy as np
import torch
import torch.nn as nn

h1 = mmh3.hash("hello")
h2 = mmh3.hash("hello")
assert h1 == h2

items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
hash_dict = {}
for item in items:
    h = mmh3.hash(item)
    if h in hash_dict:
        hash_dict[h].append(item)
    else:
        hash_dict[h] = [item]

for key, value in hash_dict.items():
    print(key, value[0])


def compute_jaccard(A, B):
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union


A = {"1", "2", "3", "4"}
B = {"1", "2", "3", "5"}
jaccard = compute_jaccard(A, B)


def minhash(S: set[str], seed: int):
    return min(mmh3.hash(s, seed) for s in S)


def count(list, x):
    return sum(1 for y in list if y == x)


print(minhash(A, 0))
print(minhash(B, 0))

n = 100  # Generate this many random hash functions
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
estimated_jaccard = count(matches, True) / len(matches)  # @inspect estimated_jaccard
assert abs(estimated_jaccard - jaccard) < 0.01
