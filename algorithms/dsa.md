# Data structures and algorithms

This has some good implementations of data structures and algorithms in Python.
https://github.com/donsheehy/datastructures/tree/master

Complement this with solving problems

Intro to Algorithms MIT for a refresher: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/

Lectures notes

Lecture 1
- Introduces the concept of an algorithm and complexity analysis of Algorithms
- Pigeon hole principle.

Lecture 2
- API vs data structure
- API is an interface, data structure deals with how to store data
- Word size and address space are related.
- Static sequences can be thought of as an array, although Python doesn't have this.
- Static sequences vs LinkedList vs Dynamic array discussion.
- Python List is an example of dynamic array.

Lecture 3
- Sets and sorting
- Set interface
  - build(A)
  - len()
  - find(k)
  - insert(x)
  - delete(k)
  - findMax()
  - findMin()
- Sorting algorithms
  - Permutation sort (O(n.n!))
  - Selection sort (O(n^2))
  - Insertion sort (O(n^2))
  - Merge sort (O(n.log(n)))

Lecture 4
- Hashing
- Direct Access Array, store the item at index k so that find(k) is O(1)
- hash function h(k): {0, ... u-1} -> {0, ... m-1} where m << u.
- Two approaches to collision resolution
  - Open addressing which stores the item at a different index
  - Separate chaining which stores all the items at the same index in a sequence.
- Good hash function is important to reduce collisios and ensure that runtime is O(1)
- Common choice is division, h(k) = k mod m
- Universal hash function allows us to choose a dynamic hash function.

Lecture 5
- Linear Sorting
- For comparison based sorting, we can't better than O(n.log(n))
- For sorting, number of leaves in the tree is O(n!), so the height of the tree is O(log(n!)) which is O(n.log(n))
- Direct Access Sort Array allows us to sort in O(n) time in case range of keys is small.
- Tuple and counting sort is O(n + u) which is O(n) if u is O(n)
- Radix sort is O(n + n.log_n(u)). We sory by the least significant digit first and then the next one, need to sort log_n(u) times

Lecture 6
- Binary Trees
- Goal is to achieve worst case O(logN) performance for dynamic set and sequence operations.
- Depth of node is the number of edges from the root to the node.
- Height of node is the number of edges from the node to the deepest leaf.
- Operations in a binary tree should be O(h) where h is the height of the tree.
- Binary tree imposes an order from its traversal. Inorder traversal is one such traversal where nodes on the left are visited first, then the root and then the nodes on the right.
- Set: traversal order is sorted by increasing order of the key, also called BST property.
- Sequence: traversal order is the insertion order, need to maintain size of each node to ensure that sequence operations like get(index), set(index) can be done in O(h) time
- Next our goal is to keep the tree balanced so that O(h) ~ O(logn)

Lecture 7
- Binary trees: AVL trees
- Height balanced trees to get O(logn) time for all operations
- One way to achieve balanced tree is rotations without changing the traversal order which is what AVL tree uses.
- height balanced tree is one for which the difference in height between left and right subtree <= 1.
- Two types of rotations to balance at each node
  - Single rotation
  - Double rotation
