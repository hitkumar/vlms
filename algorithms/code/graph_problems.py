from collections import defaultdict, deque, heapq
from typing import List, Optional


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class GraphProblems:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        def visit(n):
            if n not in adj_list:
                return
            visited.add(n)
            for e in adj_list[n]:
                if e not in visited:
                    visited.add(e)
                    visit(e)

        def bfs(n):
            q = deque([n])
            visited.add(n)
            while q:
                cur = q.popleft()
                for e in adj_list[cur]:
                    if e not in visited:
                        visited.add(e)
                        q.append(e)

        adj_list = defaultdict(list)
        for edge in edges:
            adj_list[edge[0]].append(edge[1])
            adj_list[edge[1]].append(edge[0])

        res = 0
        visited = set()
        for i in range(n):
            if i not in visited:
                res += 1
                visit(i)

        return res

    def numIslands(self, grid: List[List[str]]) -> int:
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        rows, cols = len(grid), len(grid[0])
        islands = 0

        def dfs(r, c):
            if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == "0":
                return
            grid[r][c] = "0"
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        def bfs(r, c):
            q = deque([(r, c)])
            grid[r][c] = "0"
            while q:
                row, col = q.popleft()
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < rows) and (0 <= nc < cols) and grid[nr][nc] == "1":
                        q.append((nr, nc))
                        grid[nr][nc] = "0"

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    bfs(r, c)
                    islands += 1
        return islands

    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        old_to_new = {}

        def dfs(node):
            if node in old_to_new:
                return old_to_new[node]
            new_node = Node(node.val)
            old_to_new[node] = new_node
            for nei in node.neighbors:
                new_node.neighbors.append(dfs(nei))
            return new_node

        def bfs(node):
            root = node
            q = deque([node])
            new_node = Node(node.val)
            old_to_new[node] = new_node
            while q:
                node = q.popleft()
                new_node = old_to_new[node]
                for nei in node.neighbors:
                    if nei not in old_to_new:
                        new_nei = Node(nei.val)
                        old_to_new[nei] = new_nei
                        q.append(nei)
                    new_node.neighbors.append(old_to_new[nei])

            return old_to_new[root]

        return bfs(node) if node else None
