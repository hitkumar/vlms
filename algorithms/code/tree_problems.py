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


def main():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    print(TreeProblems().levelOrder(root))


if __name__ == "__main__":
    main()
