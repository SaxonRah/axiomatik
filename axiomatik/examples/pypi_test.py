#!/usr/bin/env python3
"""
basic_verification.py - Getting started with Axiomatik

This example demonstrates the fundamental Axiomatik concepts:
- Basic require() statements
- Function contracts
- Automatic contract generation
- Refinement types
- Proof contexts

Run with: python basic_verification.py
"""

import math
import axiomatik
from axiomatik import require, contract, auto_contract, proof_context
from axiomatik import PositiveInt, Percentage, NonEmptyList, RefinementType, ProofFailure


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. BASIC REQUIRE() STATEMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def safe_divide(a: float, b: float) -> float:
    """Basic division with runtime proof of safety"""
    require("denominator is not zero", b != 0)
    require("inputs are finite", math.isfinite(a) and math.isfinite(b))

    result = a / b
    require("result is finite", math.isfinite(result))
    return result


def safe_sqrt(x: float) -> float:
    """Square root with domain validation"""
    require("input is non-negative", x >= 0)
    require("input is finite", math.isfinite(x))

    result = math.sqrt(x)
    require("result squared equals input", abs(result * result - x) < 1e-10)
    require("result is non-negative", result >= 0)
    return result


def list_average(numbers: list) -> float:
    """Calculate average with comprehensive validation"""
    require("list is not empty", len(numbers) > 0)
    require("all elements are numbers",
            all(isinstance(x, (int, float)) for x in numbers))
    require("all elements are finite",
            all(math.isfinite(x) for x in numbers))

    total = sum(numbers)
    count = len(numbers)
    average = total / count

    require("average is finite", math.isfinite(average))
    require("average is within bounds", min(numbers) <= average <= max(numbers))
    return average


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. FUNCTION CONTRACTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@contract(
    preconditions=[
        ("list is not empty", lambda items: len(items) > 0),
        ("all items are comparable", lambda items: all(
            hasattr(x, '__lt__') for x in items
        ))
    ],
    postconditions=[
        ("result is in original list", lambda items, result: result in items),
        ("result is maximum element", lambda items, result:
        all(result >= item for item in items))
    ]
)
def find_maximum(items):
    """Find maximum element with proven correctness"""
    max_item = items[0]
    for item in items[1:]:
        if item > max_item:
            max_item = item
    return max_item


@contract(
    preconditions=[
        ("start <= end", lambda start, end: start <= end),
        ("both are integers", lambda start, end:
        isinstance(start, int) and isinstance(end, int))
    ],
    postconditions=[
        ("result has correct length", lambda start, end, result:
        len(result) == end - start),
        ("all elements in range", lambda start, end, result:
        all(start <= x < end for x in result)),
        ("elements are consecutive", lambda start, end, result:
        all(result[i] == start + i for i in range(len(result))))
    ]
)
def create_range(start: int, end: int) -> list:
    """Create range with proven properties"""
    return list(range(start, end))


@contract(
    preconditions=[
        ("matrix is rectangular", lambda matrix:
        len(set(len(row) for row in matrix)) <= 1),
        ("matrix is not empty", lambda matrix: len(matrix) > 0),
        ("all rows have elements", lambda matrix: all(len(row) > 0 for row in matrix))
    ],
    postconditions=[
        ("result has correct dimensions", lambda matrix, result:
        len(result[0]) == len(matrix) and len(result) == len(matrix[0]))
    ]
)
def matrix_transpose(matrix):
    """Matrix transpose with proven correctness"""
    rows = len(matrix)
    cols = len(matrix[0])

    # Create result matrix
    result = [[matrix[i][j] for i in range(rows)] for j in range(cols)]
    return result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. AUTOMATIC CONTRACT GENERATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@auto_contract
def calculate_grade(score: PositiveInt, max_score: PositiveInt) -> Percentage:
    """Calculate percentage grade - contracts auto-generated from types"""
    percentage = (score * 100) // max_score
    return min(100, percentage)  # Cap at 100%


@auto_contract
def process_names(names: NonEmptyList) -> list:
    """Process list of names - contracts auto-generated from types"""
    processed = []
    for name in names:
        # Capitalize and strip whitespace
        clean_name = str(name).strip().title()
        processed.append(clean_name)
    return processed


@auto_contract
def calculate_statistics(values: NonEmptyList) -> dict:
    """Calculate basic statistics with automatic verification"""
    n = len(values)
    total = sum(values)
    mean = total / n

    # Calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    std_dev = math.sqrt(variance)

    return {
        'count': n,
        'sum': total,
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': min(values),
        'max': max(values)
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. REFINEMENT TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Custom refinement type
EvenInt = RefinementType(
    int,
    lambda x: x % 2 == 0,
    "even integer"
)

NonNegativeFloat = RefinementType(
    float,
    lambda x: x >= 0.0,
    "non-negative float"
)


def demonstrate_refinement_types():
    """Show refinement type usage"""
    print("~~~~~~~ Refinement Types Demo ~~~~~~~")

    # Built-in refinement types
    try:
        age = PositiveInt(25)
        score = Percentage(85)
        items = NonEmptyList([1, 2, 3, 4, 5])

        print(f"Valid age: {age}")
        print(f"Valid score: {score}%")
        print(f"Valid items: {items}")

    except ProofFailure as e:
        print(f"Validation failed: {e}")

    # Custom refinement types
    try:
        even_num = EvenInt(42)
        print(f"Valid even number: {even_num}")

        # This will fail
        odd_num = EvenInt(43)
        print(f"This won't be reached: {odd_num}")

    except ProofFailure as e:
        print(f"Expected failure for odd number: {e}")

    try:
        distance = NonNegativeFloat(5.5)
        print(f"Valid distance: {distance}")

        # This will fail
        negative_distance = NonNegativeFloat(-2.5)
        print(f"This won't be reached: {negative_distance}")

    except ProofFailure as e:
        print(f"Expected failure for negative distance: {e}")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. PROOF CONTEXTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bubble_sort_with_proofs(arr: list) -> list:
    """Bubble sort with proof contexts for each phase"""
    require("input is a list", isinstance(arr, list))
    require("all elements are comparable", all(
        hasattr(x, '__lt__') for x in arr
    ))

    # Make a copy to avoid modifying original
    result = arr.copy()
    n = len(result)

    with proof_context("bubble_sort_main_loop"):
        for i in range(n):
            with proof_context(f"bubble_sort_pass_{i}"):
                swapped = False

                for j in range(0, n - i - 1):
                    if result[j] > result[j + 1]:
                        # Swap elements
                        result[j], result[j + 1] = result[j + 1], result[j]
                        swapped = True

                # Invariant: largest element in current pass is in correct position
                if i > 0:
                    require("pass invariant holds",
                            all(result[k] <= result[n - i] for k in range(n - i)))

                # If no swaps, array is sorted
                if not swapped:
                    break

    # Prove final result is sorted
    require("result is sorted", all(
        result[i] <= result[i + 1] for i in range(len(result) - 1)
    ))
    require("result has same length", len(result) == len(arr))
    require("result contains same elements", sorted(result) == sorted(arr))

    return result


def fibonacci_with_proofs(n: int) -> int:
    """Calculate Fibonacci number with proof contexts"""
    require("n is non-negative", n >= 0)
    require("n is reasonable size", n <= 100)  # Prevent overflow

    if n <= 1:
        return n

    with proof_context("fibonacci_calculation"):
        # Use iterative approach for efficiency
        a, b = 0, 1

        with proof_context("fibonacci_iteration"):
            for i in range(2, n + 1):
                # Invariant: a = F(i-2), b = F(i-1)
                next_fib = a + b
                require("fibonacci property", next_fib == a + b)
                require("growth is reasonable", next_fib >= b)  # Fibonacci is non-decreasing

                a, b = b, next_fib

    require("result is positive for positive n", n == 0 or b > 0)
    return b


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. DEMONSTRATION FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def demonstrate_basic_verification():
    """Demonstrate all basic Axiomatik features"""
    print("Axiomatik Basic Verification Examples")
    print("~" * 80)

    # 1. Basic require() statements
    print("\n1. Basic require() statements:")
    try:
        result = safe_divide(10.0, 2.0)
        print(f"  safe_divide(10.0, 2.0) = {result}")

        sqrt_result = safe_sqrt(16.0)
        print(f"  safe_sqrt(16.0) = {sqrt_result}")

        avg_result = list_average([1, 2, 3, 4, 5])
        print(f"  list_average([1,2,3,4,5]) = {avg_result}")

    except ProofFailure as e:
        print(f"  Proof failed: {e}")

    # 2. Function contracts
    print("\n2. Function contracts:")
    try:
        max_result = find_maximum([3, 1, 4, 1, 5, 9, 2, 6])
        print(f"  find_maximum([3,1,4,1,5,9,2,6]) = {max_result}")

        range_result = create_range(5, 10)
        print(f"  create_range(5, 10) = {range_result}")

        matrix = [[1, 2, 3], [4, 5, 6]]
        transpose_result = matrix_transpose(matrix)
        print(f"  matrix_transpose([[1,2,3],[4,5,6]]) = {transpose_result}")

    except ProofFailure as e:
        print(f"  Contract failed: {e}")

    # 3. Auto-contracts
    print("\n3. Auto-generated contracts:")
    try:
        grade = calculate_grade(PositiveInt(85), PositiveInt(100))
        print(f"  calculate_grade(85, 100) = {grade}%")

        # names = process_names(NonEmptyList(["alice", "bob", "charlie"]))
        names = process_names(["alice", "bob", "charlie"])
        print(f"  process_names(['alice','bob','charlie']) = {names}")

        # stats = calculate_statistics(NonEmptyList([1, 2, 3, 4, 5]))
        stats = calculate_statistics([1, 2, 3, 4, 5])
        print(f"  calculate_statistics([1,2,3,4,5]) = {stats}")

        # Test that auto-contracts properly enforce NonEmptyList constraints
        print("\n  Testing auto-contract refinement type enforcement:")

        # Test 1: Empty list should be rejected
        try:
            empty_result = process_names([])  # Should fail - empty list
            print(f"  XXX - Empty list accepted (this shouldn't happen): {empty_result}")
        except ProofFailure as e:
            print(f"  +++ - Empty list correctly rejected: {str(e)[:50]}...")

        # Test 2: Empty list for statistics should be rejected
        try:
            empty_stats = calculate_statistics([])  # Should fail - empty list
            print(f"  XXX - Empty stats list accepted (this shouldn't happen): {empty_stats}")
        except ProofFailure as e:
            print(f"  +++ - Empty stats list correctly rejected: {str(e)[:50]}...")

        # Test 3: Valid non-empty lists should be accepted (we already saw this works)
        print(f"  +++ - Non-empty lists correctly accepted")

    except ProofFailure as e:
        print(f"  Auto-contract failed: {e}")

    # 4. Refinement types
    print("\n4. Refinement types:")
    demonstrate_refinement_types()

    # 5. Proof contexts
    print("\n5. Proof contexts:")
    try:
        unsorted = [64, 34, 25, 12, 22, 11, 90]
        sorted_result = bubble_sort_with_proofs(unsorted)
        print(f"  bubble_sort({unsorted}) = {sorted_result}")

        fib_result = fibonacci_with_proofs(10)
        print(f"  fibonacci(10) = {fib_result}")

    except ProofFailure as e:
        print(f"  Proof context failed: {e}")

    # 6. Show proof summary
    print("\n6. Proof summary:")
    summary = axiomatik.axiomatik._proof.get_summary()
    print(f"  Total proof steps: {summary['total_steps']}")
    print(f"  Contexts verified: {list(summary['contexts'].keys())}")
    print(f"  Cache enabled: {summary['cache_enabled']}")

    # Show recent proof steps
    if axiomatik.axiomatik._proof.steps:
        print("\n  Recent proof steps:")
        for i, step in enumerate(axiomatik.axiomatik._proof.steps[-5:]):
            context = f" ({step.context})" if step.context else ""
            print(f"    {i + 1}. {step.claim}{context}")


if __name__ == "__main__":
    demonstrate_basic_verification()