from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeProblems:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        root.left = self.invertTree(root.left)
        root.right = self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root == None:
            return []
        queue = [root]
        new_queue = []
        res = []
        curr_level = []
        while len(queue) > 0:
            e = queue.pop(0)
            curr_level.append(e.val)
            if e.left:
                new_queue.append(e.left)
            if e.right:
                new_queue.append(e.right)

            if len(queue) == 0:
                queue = new_queue
                new_queue = []
                res.append(curr_level)
                curr_level = []

        return res

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None and q == None:
            return True
        if p == None or q == None:
            return False
        if p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if subRoot == None:
            return True
        if root == None:
            return False

        if self.isSameTree(root, subRoot):
            return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isValidBSTRec(self, root: Optional[TreeNode]) -> bool:
        def isValidBSTUtil(root: Optional[TreeNode], min, max):
            if not root:
                return True
            if not (min < root.val < max):
                return False
            isLeftValid, isRightValid = True, True
            isLeftValid = isValidBSTUtil(root.left, min, root.val)
            isRightValid = isValidBSTUtil(root.right, root.val, max)
            return isLeftValid and isRightValid

        return isValidBSTUtil(root, float("-inf"), float("inf"))

    def isValidBST(self, root: Optional[TreeNode]):
        if not root:
            return True

        q = deque([(root, float("-inf"), float("inf"))])
        while q:
            node, left, right = q.popleft()
            if not (left < node.val < right):
                return False
            if node.left:
                q.append((node.left, left, node.val))
            if node.right:
                q.append((node.right, node.val, right))

        return True

    def inorderTraversal(self, root: Optional[TreeNode], k: int, traversal: List[int]):
        if not root:
            return
        if k == 0:
            return
        self.inorderTraversal(root.left, k, traversal)
        if len(traversal) == k:
            return
        traversal.append(root.val)
        if len(traversal) == k:
            return
        self.inorderTraversal(root.right, k, traversal)

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        traversal = []
        self.inorderTraversal(root, k, traversal)
        print(traversal)
        if len(traversal) >= k:
            return traversal[k - 1]
        return -1

    def kthSmallestInorder(self, root: Optional[TreeNode], k: int) -> int:
        arr = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)
            arr.append(node.val)
            dfs(node.right)

        dfs(root)
        return arr[k - 1]

    def kthSmallestRec(self, root: Optional[TreeNode], k: int) -> int:
        cnt = k
        res = root.val

        def dfs(node):
            nonlocal cnt, res
            if not node:
                return
            dfs(node.left)
            cnt -= 1
            if cnt == 0:
                res = node.val
                return
            dfs(node.right)

        dfs(root)
        return res

    def kthSmallestIter(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        curr = root
        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right
        return -1

    def outer(self):
        x = 10

        def inner():
            nonlocal x
            x = 5
            print(f"inner x: {x}")

        inner()
        print(f"outer x: {x}")

    def lowestCommonAncestorRec(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        if not root or root.val == p.val or root.val == q.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        res = root
        while res:
            if p.val > res.val and q.val > res.val:
                res = res.right
            if p.val < res.val and q.val < res.val:
                res = res.left
            else:
                return res


def main():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    # print(TreeProblems().levelOrder(root))
    TreeProblems().outer()


if __name__ == "__main__":
    main()
