def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    s_dict = {}
    t_dict = {}
    for c in s:
        s_dict[c] = s_dict.get(c, 0) + 1
    for c in t:
        t_dict[c] = t_dict.get(c, 0) + 1

    # print(s_dict)
    # for k, v in s_dict.items():
    #     if k not in t_dict or v != t_dict[k]:
    #         return False
    #     del t_dict[k]

    # return len(t_dict) == 0
    return s_dict == t_dict


if __name__ == "__main__":
    print(isAnagram("anagram", "nagaram"))
    print(isAnagram("anagram", "nagarama"))
    print(isAnagram("nahan", "hanan"))
