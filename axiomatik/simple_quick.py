#!/usr/bin/env python3
"""
Simple Axiomatik Quick Start Guide - Complete Version

This file provides a complete, gentle introduction to Simple Axiomatik
for developers who want to add verification to their Python code.

Start here if you're new to verification or want to see what's possible.
"""

import simple_axiomatik as ax
from dataclasses import dataclass
from typing import List, Dict

# Step 1: Set your verification mode
ax.set_mode("dev")  # Full verification during development

print("Simple Axiomatik Complete Quick Start Guide")
print("=" * 50)
print(f"Current mode: {ax.get_mode()}")
print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 1: START SIMPLE WITH @verify
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 1: Basic Verification with @verify")
print("-" * 45)
print("Add simple checks to your existing functions")
print()


@ax.verify
def safe_division(a: float, b: float) -> float:
    """Your first verified function - prevents division by zero"""
    # Add a simple safety check
    ax.require(b != 0, "Cannot divide by zero")

    result = a / b

    # Verify the result makes mathematical sense
    ax.ensure(ax.approx_equal(result * b, a), "Division result doesn't check out")

    return result


@ax.verify
def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate distance between two points with verification"""
    ax.require(all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]),
               "All coordinates must be numbers")

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    ax.ensure(distance >= 0, "Distance cannot be negative")
    return distance


# Try the functions
print("Testing basic verification:")
try:
    result = safe_division(10.0, 2.0)
    print(f"  +++ - safe_division(10.0, 2.0) = {result}")

    distance = calculate_distance(0, 0, 3, 4)
    print(f"  +++ - distance from (0,0) to (3,4) = {distance}")

    # This will fail gracefully
    safe_division(10.0, 0.0)

except ax.VerificationError as e:
    print(f"  +++ - Caught division by zero error successfully")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 2: TYPE-BASED VERIFICATION WITH @checked
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 2: Automatic Type Verification with @checked")
print("-" * 52)
print("Let type hints do the verification work for you")
print()


@ax.checked
def process_items(items: ax.NonEmpty[list], multiplier: ax.Positive[int]) -> ax.Positive[int]:
    """Function that automatically validates input types"""
    # No manual checks needed - types handle validation!
    return len(items) * multiplier


@ax.checked
def calculate_percentage(part: ax.Positive[float], whole: ax.Positive[float]) -> ax.Percentage:
    """Calculate percentage with automatic range checking"""
    percentage = int((part / whole) * 100)
    return min(100, percentage)  # Cap at 100%


@ax.checked
def format_name(first: ax.NonEmpty[str], last: ax.NonEmpty[str]) -> ax.NonEmpty[str]:
    """Format full name with non-empty string validation"""
    return f"{first.strip()} {last.strip()}"


print("Testing type-based verification:")
try:
    # These work
    result = process_items(["a", "b", "c"], 3)
    print(f"  +++ - process_items(['a', 'b', 'c'], 3) = {result}")

    percentage = calculate_percentage(75.0, 100.0)
    print(f"  +++ - calculate_percentage(75, 100) = {percentage}%")

    name = format_name("Alice", "Smith")
    print(f"  +++ - format_name('Alice', 'Smith') = '{name}'")

    # This will fail - empty list
    process_items([], 2)

except ax.VerificationError as e:
    print(f"  +++ - Caught type constraint violation: Empty list rejected")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 3: COMBINE BOTH APPROACHES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 3: Combined Type + Logic Verification")
print("-" * 45)
print("Use both @verify and @checked for comprehensive validation")
print()


@ax.verify
@ax.checked
def calculate_compound_interest(
        principal: ax.Positive[float],
        rate: ax.Range[float, 0.0, 1.0],  # 0-100% as decimal
        years: ax.Positive[int]
) -> ax.Positive[float]:
    """Calculate compound interest with full verification"""

    # Type system ensures positive principal, valid rate range, positive years
    # Add business logic verification
    ax.require(principal <= 1_000_000, "Principal seems unreasonably high")
    ax.require(years <= 100, "Time period seems unreasonably long")

    result = principal * ((1 + rate) ** years)

    # Verify result is reasonable
    ax.ensure(result >= principal, "Result should be at least the principal")
    ax.ensure(result <= principal * 1000, "Result seems unreasonably high")

    return result


@ax.verify
@ax.checked
def analyze_test_scores(scores: ax.NonEmpty[List[int]]) -> Dict[str, float]:
    """Analyze test scores with comprehensive validation"""

    # Type system ensures non-empty list
    # Add domain-specific checks
    ax.require(all(0 <= score <= 100 for score in scores),
               "All scores must be between 0 and 100")
    ax.require(len(scores) >= 3, "Need at least 3 scores for meaningful analysis")

    mean = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    median = sorted_scores[len(scores) // 2]

    result = {
        'mean': mean,
        'median': median,
        'min': min(scores),
        'max': max(scores),
        'count': len(scores)
    }

    # Verify statistical properties
    ax.ensure(min(scores) <= mean <= max(scores), "Mean should be between min and max")
    ax.ensure(0 <= median <= 100, "Median should be valid score")

    return result


print("Testing combined verification:")
try:
    # Compound interest calculation
    investment = calculate_compound_interest(1000.0, 0.05, 10)
    print(f"  +++ - $1000 at 5% for 10 years = ${investment:,.2f}")

    # Test score analysis
    scores = [85, 92, 78, 95, 88, 91]
    analysis = analyze_test_scores(scores)
    print(f"  +++ - Test scores analysis: mean={analysis['mean']:.1f}, median={analysis['median']}")

except ax.VerificationError as e:
    print(f"  XXX - Verification failed: {e}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 4: AVAILABLE TYPE ALIASES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 4: Available Type Aliases")
print("-" * 35)
print("Rich type system for common constraints")
print()


# Show all available type aliases with examples
@ax.checked
def demo_type_aliases(
        positive_int: ax.PositiveInt,  # > 0
        positive_float: ax.PositiveFloat,  # > 0.0
        non_empty_list: ax.NonEmpty[list],  # len() > 0
        non_empty_string: ax.NonEmpty[str],  # len() > 0
        percentage: ax.Percentage,  # 0-100 integer
        custom_range: ax.Range[int, 1, 10]  # 1-10 inclusive
) -> Dict[str, any]:
    """Demonstrate all available type aliases"""

    return {
        'positive_int': positive_int,
        'positive_float': positive_float,
        'list_length': len(non_empty_list),
        'string_length': len(non_empty_string),
        'percentage': percentage,
        'custom_range': custom_range
    }


print("Type aliases available:")
print("  ax.Positive[int/float]      - Numbers > 0")
print("  ax.NonEmpty[list/str/etc]   - Collections with length > 0")
print("  ax.Range[type, min, max]    - Values within specific range")
print("  ax.Percentage               - Integers from 0-100")
print("  ax.PositiveInt              - Shorthand for ax.Positive[int]")
print("  ax.PositiveFloat            - Shorthand for ax.Positive[float]")
print("  ax.NonEmptyList             - Shorthand for ax.NonEmpty[list]")
print("  ax.NonEmptyStr              - Shorthand for ax.NonEmpty[str]")
print()

try:
    result = demo_type_aliases(
        positive_int=42,
        positive_float=3.14,
        non_empty_list=[1, 2, 3],
        non_empty_string="hello",
        percentage=85,
        custom_range=7
    )
    print(f"  +++ - Type aliases demo: {result}")
except ax.VerificationError as e:
    print(f"  XXX - Type validation failed: {e}")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 5: PROTOCOL VERIFICATION (CORRECTED SYNTAX)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 5: Protocol Verification")
print("-" * 35)
print("Ensure objects follow correct usage patterns")
print()


@ax.stateful(initial="closed")
class BankAccount:
    """Bank account with verified state transitions"""

    def __init__(self, account_number: str, initial_balance: float = 0.0):
        self.account_number = account_number
        self.balance = initial_balance
        self.is_frozen = False

    @ax.state("closed", "open")
    def open_account(self):
        """Open the account for transactions"""
        print(f"  Opening account {self.account_number}")
        ax.require(not self.is_frozen, "Cannot open frozen account")

    @ax.verify
    def deposit(self, amount: ax.Positive[float]) -> float:
        """Deposit money (available in open state)"""
        ax.require(amount > 0, "Deposit amount must be positive")
        ax.require(not self.is_frozen, "Account is frozen")

        old_balance = self.balance
        self.balance += amount

        ax.ensure(self.balance == old_balance + amount, "Balance update failed")
        print(f"  Deposited ${amount}, new balance: ${self.balance}")
        return self.balance

    @ax.verify
    def withdraw(self, amount: ax.Positive[float]) -> float:
        """Withdraw money with verification"""
        ax.require(amount > 0, "Withdrawal amount must be positive")
        ax.require(amount <= self.balance, "Insufficient funds")
        ax.require(not self.is_frozen, "Account is frozen")

        old_balance = self.balance
        self.balance -= amount

        ax.ensure(self.balance == old_balance - amount, "Balance update failed")
        print(f"  Withdrew ${amount}, new balance: ${self.balance}")
        return self.balance

    @ax.state(["open"], "frozen")
    def freeze_account(self):
        """Freeze account for security"""
        self.is_frozen = True
        print(f"  Account {self.account_number} frozen")

    @ax.state(["frozen"], "open")
    def unfreeze_account(self):
        """Unfreeze account"""
        self.is_frozen = False
        print(f"  Account {self.account_number} unfrozen")

    @ax.state(["open", "frozen"], "closed")
    def close_account(self):
        """Close the account"""
        ax.require(self.balance == 0, "Must withdraw all funds before closing")
        print(f"  Closed account {self.account_number}")


print("Testing protocol verification:")
try:
    account = BankAccount("ACC-12345", 100.0)
    account.open_account()
    account.deposit(50.0)
    account.withdraw(25.0)
    account.freeze_account()
    account.unfreeze_account()
    account.withdraw(125.0)  # Withdraw remaining balance
    account.close_account()
    print("  +++ - Account protocol followed correctly")

except ax.VerificationError as e:
    print(f"  +++ - Protocol violation caught: Account operations verified")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 6: DATACLASS INTEGRATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 6: Dataclass Integration")
print("-" * 35)
print("Automatic validation for dataclass fields")
print()


@ax.enable_for_dataclass
@dataclass
class Employee:
    """Employee record with validated fields"""
    employee_id: ax.PositiveInt
    name: ax.NonEmpty[str]
    email: ax.NonEmpty[str]
    age: ax.Range[int, 18, 70]
    salary: ax.Positive[float]
    department: ax.NonEmpty[str]

    def __post_init__(self):
        # Additional business logic validation
        ax.require("@" in self.email, "Email must contain @ symbol")
        ax.require(self.salary >= 30000, "Salary must be at least $30,000")


@ax.enable_for_dataclass
@dataclass
class Product:
    """Product with automatic validation"""
    id: ax.PositiveInt
    name: ax.NonEmpty[str]
    price: ax.Positive[float]
    stock: ax.Range[int, 0, 10000]
    category: ax.NonEmpty[str]


print("Testing dataclass integration:")
try:
    # Valid employee
    employee = Employee(
        employee_id=12345,
        name="Alice Johnson",
        email="alice@company.com",
        age=30,
        salary=75000.0,
        department="Engineering"
    )
    print(f"  +++ - Created employee: {employee.name}, {employee.department}")

    # Valid product
    product = Product(
        id=1001,
        name="Wireless Headphones",
        price=199.99,
        stock=50,
        category="Electronics"
    )
    print(f"  +++ - Created product: {product.name}, ${product.price}")

    # This would fail - invalid age
    # Employee(12346, "Bob", "bob@company.com", 17, 45000.0, "Sales")

except ax.VerificationError as e:
    print(f"  +++ - Dataclass validation caught constraint violation")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 7: PERFORMANCE TRACKING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 7: Performance Tracking")
print("-" * 35)
print("Monitor verification overhead")
print()


@ax.verify(track_performance=True)
def fibonacci_verified(n: ax.Range[int, 0, 40]) -> ax.PositiveInt:
    """Calculate Fibonacci number with performance tracking"""
    ax.require(n >= 0, "Fibonacci input must be non-negative")

    if n <= 1:
        return n

    # Iterative approach for efficiency
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b

    ax.ensure(b > 0, "Fibonacci result should be positive")
    return b


@ax.verify(track_performance=True)
def prime_check(n: ax.PositiveInt) -> bool:
    """Check if number is prime with performance tracking"""
    ax.require(n > 0, "Input must be positive")

    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    # Check odd divisors up to sqrt(n)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False

    return True


print("Running performance-tracked functions:")
# Run some operations to generate performance data
fib_results = []
for i in range(10, 21):
    result = fibonacci_verified(i)
    fib_results.append(result)

prime_count = 0
for i in range(100, 201):
    if prime_check(i):
        prime_count += 1

print(f"  +++ - Calculated Fibonacci numbers 10-20: {fib_results[-3:]}")
print(f"  +++ - Found {prime_count} primes between 100-200")
print("  +++ - Performance data collected")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 8: CONTEXT MANAGERS AND MODE SWITCHING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 8: Context Managers and Mode Switching")
print("-" * 48)
print("Control verification behavior dynamically")
print()


@ax.verify
def risky_operation(data: list) -> float:
    """Operation that might need different verification levels"""
    ax.require(len(data) > 0, "Data cannot be empty")
    ax.require(all(isinstance(x, (int, float)) for x in data), "All elements must be numbers")

    result = sum(x ** 2 for x in data) / len(data)

    ax.ensure(result >= 0, "Result should be non-negative")
    return result


print("Testing context managers:")

# Normal verification
try:
    result = risky_operation([1, 2, 3, 4, 5])
    print(f"  +++ - Normal mode result: {result}")
except ax.VerificationError as e:
    print(f"  XXX - Normal mode failed: {e}")

# Temporarily disable verification for performance
with ax.no_verification():
    result = risky_operation([1, 2, 3, 4, 5])
    print(f"  +++ - No verification mode result: {result}")

# Temporarily switch to production mode
with ax.production_mode():
    result = risky_operation([1, 2, 3, 4, 5])
    print(f"  +++ - Production mode result: {result}")

# Named verification context
with ax.verification_context("special_calculation"):
    ax.require(True, "This works in named context")
    print("  +++ - Named verification context works")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LEVEL 9: MODE CONFIGURATION FOR DIFFERENT ENVIRONMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("LEVEL 9: Environment-Based Configuration")
print("-" * 43)
print("Optimize verification for different deployment scenarios")
print()

# Show current configuration
print(f"Current mode: {ax.get_mode()}")

# Demonstrate different modes
modes_info = {
    "dev": "Full verification - catch all issues during development",
    "test": "Comprehensive verification - thorough testing with debug info",
    "prod": "Essential checks only - minimal overhead for production",
    "off": "No verification - maximum performance when needed"
}

print("\nAvailable modes:")
for mode, description in modes_info.items():
    print(f"  {mode:4} - {description}")

# Show how to configure for different environments
print(f"\nConfiguration examples:")
print(f"  # Development")
print(f"  ax.set_mode('dev')")
print(f"  ")
print(f"  # Production deployment")
print(f"  ax.set_mode('prod')")
print(f"  ")
print(f"  # Testing environment")
print(f"  ax.set_mode('test')")
print(f"  ")
print(f"  # Performance-critical sections")
print(f"  ax.set_mode('off')")

print()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FINAL SUMMARY AND NEXT STEPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("SUMMARY: What You've Learned")
print("-" * 35)

summary = """
+++ - Level 1: @ax.verify for manual require/ensure statements
+++ - Level 2: @ax.checked for automatic type-based verification  
+++ - Level 3: Combined approaches for comprehensive validation
+++ - Level 4: Rich type aliases (Positive, NonEmpty, Range, etc.)
+++ - Level 5: Protocol verification for stateful objects
+++ - Level 6: Dataclass integration with automatic field validation
+++ - Level 7: Performance tracking to monitor verification overhead
+++ - Level 8: Context managers for dynamic control
+++ - Level 9: Environment-based configuration

Key Benefits:
- Catch bugs early with clear, helpful error messages
- Gradually adopt verification in existing codebases  
- Type hints do most of the work automatically
- Production-ready with configurable overhead
- Works with existing Python tools and libraries
"""

print(summary)

# Generate final reports
print("\nFinal Status Report:")
print(ax.report())

print(f"\nPerformance Report:")
print(ax.performance_report())

print(f"\nNext Steps:")
print(f"1. Start with @ax.verify on critical functions")
print(f"2. Add @ax.checked for type-safe functions")
print(f"3. Use ax.set_mode('prod') in production")
print(f"4. Explore usage_examples.py for real-world patterns")
print(f"5. Check documentation for advanced features")

print(f"\nYou're ready to add verification to your Python projects!")
print(f"Current mode: {ax.get_mode()} - Remember to set appropriate mode for your environment")