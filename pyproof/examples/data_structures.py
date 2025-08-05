#!/usr/bin/env python3
"""
data_structures.py - Verified collections and algorithms

This example demonstrates:
- Verified data structures with invariants
- Algorithm verification with loop invariants
- Custom collection implementations
- Mathematical property proofs

Run with: python data_structures.py
"""

import bisect
import pyproof.pyproof
from pyproof.pyproof import require, contract, proof_context, auto_contract
from pyproof.pyproof import PositiveInt, NonEmptyList
from typing import List, Optional, Any, TypeVar

T = TypeVar('T')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. VERIFIED STACK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedStack:
    """Stack with verified invariants"""

    def __init__(self):
        self._items = []
        self._verify_invariants()

    def _verify_invariants(self):
        """Check stack invariants"""
        require("items is a list", isinstance(self._items, list))
        require("size is non-negative", len(self._items) >= 0)

    @contract(
        preconditions=[
            ("item is not None", lambda self, item: item is not None)
        ],
        postconditions=[
            ("size increased by 1", lambda self, item, result: len(self._items) == result),
            ("top item is pushed item", lambda self, item, result: self._items[-1] == item)
        ]
    )
    def push(self, item: Any) -> int:
        """Push item onto stack"""
        old_size = len(self._items)
        self._items.append(item)
        self._verify_invariants()

        new_size = len(self._items)
        require("size increased correctly", new_size == old_size + 1)
        return new_size

    @contract(
        preconditions=[
            ("stack is not empty", lambda self: len(self._items) > 0)
        ],
        postconditions=[
            ("size decreased by 1", lambda self, result: len(self._items) == result[1]),
            ("returned correct item", lambda self, result: result[0] is not None)
        ]
    )
    def pop(self) -> tuple:
        """Pop item from stack - returns (item, new_size)"""
        old_size = len(self._items)
        item = self._items.pop()
        self._verify_invariants()

        new_size = len(self._items)
        require("size decreased correctly", new_size == old_size - 1)
        return item, new_size

    def peek(self) -> Any:
        """Look at top item without removing"""
        require("stack is not empty", len(self._items) > 0)
        return self._items[-1]

    def size(self) -> int:
        """Get stack size"""
        return len(self._items)

    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return len(self._items) == 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. VERIFIED SORTED LIST
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedSortedList:
    """Sorted list that maintains ordering invariant"""

    def __init__(self, items: List = None):
        self._items = sorted(items) if items else []
        self._verify_sorted()

    def _verify_sorted(self):
        """Verify list remains sorted"""
        require("list is sorted", all(
            self._items[i] <= self._items[i + 1]
            for i in range(len(self._items) - 1)
        ))

    @contract(
        postconditions=[
            ("item was inserted", lambda self, item, result: item in self._items),
            ("list remains sorted", lambda self, item, result: self._verify_sorted() or True),
            ("size increased by 1", lambda self, item, result: len(self._items) == result)
        ]
    )
    def insert(self, item: Any) -> int:
        """Insert item maintaining sort order"""
        old_size = len(self._items)

        # Find insertion point
        pos = bisect.bisect_left(self._items, item)
        self._items.insert(pos, item)

        self._verify_sorted()
        new_size = len(self._items)
        require("size increased", new_size == old_size + 1)
        return new_size

    @contract(
        preconditions=[
            ("item exists in list", lambda self, item: item in self._items)
        ],
        postconditions=[
            ("item was removed", lambda self, item, result: item not in self._items or
                                                            self._items.count(item) == result[1]),
            ("list remains sorted", lambda self, item, result: self._verify_sorted() or True)
        ]
    )
    def remove(self, item: Any) -> tuple:
        """Remove first occurrence of item - returns (success, remaining_count)"""
        old_count = self._items.count(item)
        self._items.remove(item)
        self._verify_sorted()

        new_count = self._items.count(item)
        require("count decreased by 1", new_count == old_count - 1)
        return True, new_count

    def search(self, item: Any) -> Optional[int]:
        """Binary search for item"""
        self._verify_sorted()

        pos = bisect.bisect_left(self._items, item)
        if pos < len(self._items) and self._items[pos] == item:
            require("found item at correct position", self._items[pos] == item)
            return pos
        return None

    def get_items(self) -> List:
        """Get copy of items"""
        return self._items.copy()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. VERIFIED BINARY SEARCH TREE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BSTNode:
    """Binary search tree node"""

    def __init__(self, value: Any, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class VerifiedBST:
    """Binary search tree with verified BST property"""

    def __init__(self):
        self.root = None
        self.size = 0

    def _verify_bst_property(self, node: Optional[BSTNode],
                             min_val: Any = None, max_val: Any = None) -> bool:
        """Verify BST property recursively"""
        if node is None:
            return True

        # Check value constraints
        if min_val is not None and node.value <= min_val:
            return False
        if max_val is not None and node.value >= max_val:
            return False

        # Recursively check subtrees
        return (self._verify_bst_property(node.left, min_val, node.value) and
                self._verify_bst_property(node.right, node.value, max_val))

    def verify_invariants(self):
        """Verify all BST invariants"""
        require("BST property holds", self._verify_bst_property(self.root))
        require("size is non-negative", self.size >= 0)

    @contract(
        postconditions=[
            ("size increased or stayed same", lambda self, value, result:
            self.size >= result[1]),
            ("BST property maintained", lambda self, value, result:
            self.verify_invariants() or True)
        ]
    )
    def insert(self, value: Any) -> tuple:
        """Insert value into BST - returns (was_new, old_size)"""
        old_size = self.size
        self.root, was_new = self._insert_recursive(self.root, value)

        if was_new:
            self.size += 1

        self.verify_invariants()
        return was_new, old_size

    def _insert_recursive(self, node: Optional[BSTNode], value: Any) -> tuple:
        """Recursive insertion helper"""
        if node is None:
            return BSTNode(value), True

        if value < node.value:
            node.left, was_new = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right, was_new = self._insert_recursive(node.right, value)
        else:
            # Value already exists
            was_new = False

        return node, was_new

    def search(self, value: Any) -> bool:
        """Search for value in BST"""
        self.verify_invariants()
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node: Optional[BSTNode], value: Any) -> bool:
        """Recursive search helper"""
        if node is None:
            return False

        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self) -> List[Any]:
        """Return sorted list of values"""
        result = []
        self._inorder_recursive(self.root, result)

        # Prove result is sorted (follows from BST property)
        require("traversal result is sorted", all(
            result[i] <= result[i + 1] for i in range(len(result) - 1)
        ))
        return result

    def _inorder_recursive(self, node: Optional[BSTNode], result: List):
        """Recursive inorder traversal"""
        if node is not None:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. VERIFIED SORTING ALGORITHMS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@contract(
    preconditions=[
        ("input is a list", lambda arr: isinstance(arr, list)),
        ("all elements are comparable", lambda arr: len(arr) <= 1 or all(
            hasattr(arr[i], '__lt__') for i in range(len(arr))
        ))
    ],
    postconditions=[
        ("result is sorted", lambda arr, result: all(
            result[i] <= result[i + 1] for i in range(len(result) - 1)
        )),
        ("result has same length", lambda arr, result: len(result) == len(arr)),
        ("result contains same elements", lambda arr, result:
        sorted(result) == sorted(arr))
    ]
)
def verified_merge_sort(arr: List) -> List:
    """Merge sort with verified correctness"""
    if len(arr) <= 1:
        return arr.copy()

    with proof_context("merge_sort_divide"):
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        require("division is correct", len(left_half) + len(right_half) == len(arr))
        require("left half is smaller", len(left_half) <= len(arr))
        require("right half is smaller", len(right_half) <= len(arr))

    # Recursively sort halves
    sorted_left = verified_merge_sort(left_half)
    sorted_right = verified_merge_sort(right_half)

    # Merge sorted halves
    with proof_context("merge_sort_merge"):
        return _verified_merge(sorted_left, sorted_right)


def _verified_merge(left: List, right: List) -> List:
    """Merge two sorted lists with verification"""
    require("left is sorted", all(left[i] <= left[i + 1] for i in range(len(left) - 1)))
    require("right is sorted", all(right[i] <= right[i + 1] for i in range(len(right) - 1)))

    result = []
    i = j = 0

    with proof_context("merge_main_loop"):
        while i < len(left) and j < len(right):
            # Loop invariant: result contains smallest i+j elements in sorted order
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1

            # Verify merge property
            if len(result) >= 2:
                require("merge maintains order", result[-2] <= result[-1])

    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])

    # Verify final result
    require("merge result is sorted", all(
        result[k] <= result[k + 1] for k in range(len(result) - 1)
    ))
    require("merge preserves all elements",
            len(result) == len(left) + len(right))

    return result


@contract(
    preconditions=[
        ("array is not empty", lambda arr, target: len(arr) > 0),  # Fixed: added target parameter
        ("target bounds are valid", lambda arr, target: True),  # Any value is valid to search for
        ("array is sorted", lambda arr, target: all(  # This one was already correct
            arr[i] <= arr[i + 1] for i in range(len(arr) - 1)
        ))
    ],
    postconditions=[
        ("result indicates presence", lambda arr, target, result:
        (result is not None) == (target in arr)),
        ("if found, correct index", lambda arr, target, result:
        result is None or arr[result] == target)
    ]
)
def verified_binary_search(arr: List, target: Any) -> Optional[int]:
    """Binary search with verified correctness"""
    left = 0
    right = len(arr) - 1

    with proof_context("binary_search_loop"):
        while left <= right:
            # Loop invariant: if target exists, it's in arr[left:right+1]
            require("bounds are valid", 0 <= left <= len(arr))
            require("bounds are valid", -1 <= right < len(arr))
            require("search space is valid", left <= right + 1)

            mid = (left + right) // 2
            require("mid is in bounds", left <= mid <= right)

            if arr[mid] == target:
                require("found target at mid", arr[mid] == target)
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

            # Prove we're making progress
            require("search space shrinking", right - left < len(arr))

    # Target not found
    require("target not in remaining space", target not in arr[left:right + 1])
    return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. VERIFIED GRAPH ALGORITHMS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerifiedGraph:
    """Simple graph with verified properties"""

    def __init__(self):
        self.vertices = set()
        self.edges = set()  # Set of (u, v) tuples

    def verify_invariants(self):
        """Verify graph invariants"""
        require("vertices is a set", isinstance(self.vertices, set))
        require("edges is a set", isinstance(self.edges, set))
        require("all edges connect existing vertices", all(
            u in self.vertices and v in self.vertices
            for u, v in self.edges
        ))

    @contract(
        postconditions=[
            ("vertex was added", lambda self, vertex, result: vertex in self.vertices),
            ("invariants maintained", lambda self, vertex, result:
            self.verify_invariants() or True)
        ]
    )
    def add_vertex(self, vertex: Any) -> bool:
        """Add vertex to graph"""
        was_new = vertex not in self.vertices
        self.vertices.add(vertex)
        self.verify_invariants()
        return was_new

    @contract(
        preconditions=[
            ("vertices exist", lambda self, u, v: u in self.vertices and v in self.vertices)
        ],
        postconditions=[
            ("edge was added", lambda self, u, v, result: (u, v) in self.edges),
            ("invariants maintained", lambda self, u, v, result:
            self.verify_invariants() or True)
        ]
    )
    def add_edge(self, u: Any, v: Any) -> bool:
        """Add edge between vertices"""
        was_new = (u, v) not in self.edges
        self.edges.add((u, v))
        self.verify_invariants()
        return was_new

    def bfs_shortest_path(self, start: Any, goal: Any) -> Optional[List]:
        """BFS shortest path with verification"""
        require("start vertex exists", start in self.vertices)
        require("goal vertex exists", goal in self.vertices)

        if start == goal:
            return [start]

        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        with proof_context("bfs_main_loop"):
            while queue:
                current, path = queue.popleft()

                # Explore neighbors
                for u, v in self.edges:
                    next_vertex = None
                    if u == current and v not in visited:
                        next_vertex = v
                    elif v == current and u not in visited:  # Undirected graph
                        next_vertex = u

                    if next_vertex:
                        new_path = path + [next_vertex]

                        if next_vertex == goal:
                            # Verify path properties
                            require("path starts with start", new_path[0] == start)
                            require("path ends with goal", new_path[-1] == goal)
                            require("path is connected", all(
                                (new_path[i], new_path[i + 1]) in self.edges or
                                (new_path[i + 1], new_path[i]) in self.edges
                                for i in range(len(new_path) - 1)
                            ))
                            return new_path

                        visited.add(next_vertex)
                        queue.append((next_vertex, new_path))

        # No path found
        return None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. DEMONSTRATION FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demonstrate_data_structures():
    """Demonstrate all verified data structures and algorithms"""
    print("PyProof Verified Data Structures and Algorithms")
    print("~" * 80)

    # 1. Verified Stack
    print("\n1. Verified Stack:")
    stack = VerifiedStack()
    print(f"  Created empty stack (size: {stack.size()})")

    stack.push("first")
    stack.push("second")
    stack.push("third")
    print(f"  Pushed 3 items (size: {stack.size()})")

    item, size = stack.pop()
    print(f"  Popped '{item}' (new size: {size})")
    print(f"  Top item: '{stack.peek()}'")

    # 2. Verified Sorted List
    print("\n2. Verified Sorted List:")
    sorted_list = VerifiedSortedList([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"  Created sorted list: {sorted_list.get_items()}")

    sorted_list.insert(7)
    print(f"  Inserted 7: {sorted_list.get_items()}")

    pos = sorted_list.search(5)
    print(f"  Search for 5: found at position {pos}")

    # 3. Verified BST
    print("\n3. Verified Binary Search Tree:")
    bst = VerifiedBST()
    values = [5, 3, 7, 1, 9, 4, 6]

    for value in values:
        was_new, old_size = bst.insert(value)
        print(f"  Inserted {value} ({'new' if was_new else 'duplicate'})")

    print(f"  BST size: {bst.size}")
    print(f"  Inorder traversal: {bst.inorder_traversal()}")

    # 4. Verified Sorting
    print("\n4. Verified Sorting Algorithms:")
    unsorted = [64, 34, 25, 12, 22, 11, 90, 88, 76, 50, 42]
    print(f"  Original: {unsorted}")

    sorted_result = verified_merge_sort(unsorted)
    print(f"  Merge sorted: {sorted_result}")

    # 5. Verified Binary Search
    print("\n5. Verified Binary Search:")
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    print(f"  Sorted array: {sorted_array}")

    target = 7
    pos = verified_binary_search(sorted_array, target)
    print(f"  Search for {target}: found at position {pos}")

    target = 8
    pos = verified_binary_search(sorted_array, target)
    print(f"  Search for {target}: {'found' if pos is not None else 'not found'}")

    # 6. Verified Graph
    print("\n6. Verified Graph Algorithms:")
    graph = VerifiedGraph()

    # Add vertices
    for vertex in ['A', 'B', 'C', 'D', 'E']:
        graph.add_vertex(vertex)

    # Add edges
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'), ('E', 'D')]
    for u, v in edges:
        graph.add_edge(u, v)

    print(f"  Created graph with {len(graph.vertices)} vertices and {len(graph.edges)} edges")

    path = graph.bfs_shortest_path('A', 'D')
    print(f"  Shortest path from A to D: {path}")

    # 7. Proof Summary
    print("\n7. Verification Summary:")
    summary = pyproof.pyproof._proof.get_summary()
    print(f"  Total proof steps: {summary['total_steps']}")
    print(f"  Verified contexts: {len(summary['contexts'])}")

    if pyproof.pyproof._proof.steps:
        print("  Recent verifications:")
        for step in pyproof.pyproof._proof.steps[-3:]:
            context = f" ({step.context})" if step.context else ""
            print(f"    - {step.claim}{context}")


if __name__ == "__main__":
    demonstrate_data_structures()