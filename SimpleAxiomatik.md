# Simple Axiomatik

Simple Axiomatik is a Pythonic wrapper for Axiomatik, the runtime verification tool. It should feel natural and catches bugs early. Add verification to your existing Python code with minimal changes.

## What is Simple Axiomatik?

Simple Axiomatik makes runtime verification as easy as adding type hints. It helps you:

- Catch bugs before they cause problems
- Validate inputs automatically from type hints
- Ensure functions behave correctly
- Add verification incrementally to existing code
- Get clear, helpful error messages when things go wrong

## Quick Start

### Installation

```python
# Add simple_axiomatik.py to your project
import simple_axiomatik as ax
```

### Your First Verified Function

```python
@ax.verify
def safe_divide(a: float, b: float) -> float:
    ax.require(b != 0, "Cannot divide by zero")
    result = a / b
    ax.ensure(result * b == a, "Division check failed")  # approximately
    return result

# This works
result = safe_divide(10.0, 2.0)  # Returns 5.0

# This fails gracefully with helpful error
result = safe_divide(10.0, 0.0)  # VerificationError: Cannot divide by zero
```

### Automatic Type Validation

```python
@ax.checked
def process_items(items: ax.NonEmpty[list], multiplier: ax.PositiveInt) -> ax.PositiveInt:
    """Types do the validation automatically"""
    return len(items) * multiplier

# This works
result = process_items(["a", "b", "c"], 3)  # Returns 9

# This fails - empty list not allowed
result = process_items([], 3)  # VerificationError: list must be non-empty
```

## Core Features

### 1. Simple Decorators

**@ax.verify** - Add manual verification to any function
```python
@ax.verify
def calculate_grade(score: int, total: int) -> float:
    ax.require(0 <= score <= total, "Score must be between 0 and total")
    percentage = (score / total) * 100
    ax.ensure(0 <= percentage <= 100, "Percentage must be valid")
    return percentage
```

**@ax.checked** - Automatic verification from type hints
```python
@ax.checked
def format_name(first: ax.NonEmpty[str], last: ax.NonEmpty[str]) -> ax.NonEmpty[str]:
    return f"{first.strip()} {last.strip()}"  # Types handle validation
```

### 2. Rich Type System

```python
# Built-in type aliases
ax.PositiveInt        # Integers > 0
ax.PositiveFloat      # Floats > 0.0
ax.NonEmpty[list]     # Lists with len() > 0
ax.NonEmpty[str]      # Non-empty strings
ax.Percentage         # Integers 0-100
ax.Range[int, 1, 10]  # Custom ranges

# Use in function signatures
@ax.checked
def calculate_interest(
    principal: ax.Positive[float],
    rate: ax.Range[float, 0.0, 0.5],  # 0-50% annual rate
    years: ax.Range[int, 1, 50]
) -> ax.Positive[float]:
    return principal * (1 + rate) ** years
```

### 3. Protocol Verification

Ensure objects follow correct usage patterns:

```python
@ax.stateful(initial="closed")
class File:
    @ax.state("closed", "open")
    def open(self): pass
    
    @ax.state("open", "reading") 
    def read(self): pass
    
    @ax.state(["reading", "open"], "closed")
    def close(self): pass

# Usage is verified automatically
f = File()
f.open()    # OK: closed -> open
f.read()    # OK: open -> reading  
f.close()   # OK: reading -> closed
f.read()    # ERROR: closed -> reading not allowed
```

### 4. Dataclass Integration

```python
@ax.enable_for_dataclass
@dataclass
class User:
    name: ax.NonEmpty[str]
    age: ax.Range[int, 0, 150]
    email: ax.NonEmpty[str]
    
    def __post_init__(self):
        ax.require("@" in self.email, "Email must contain @")

# Validation happens automatically
user = User("Alice", 30, "alice@company.com")  # OK
user = User("", 30, "alice@company.com")       # ERROR: name cannot be empty
```

## Configuration

### Verification Modes

```python
# Development - full verification (default)
ax.set_mode("dev")

# Production - essential checks only  
ax.set_mode("prod")

# Testing - comprehensive verification with debug info
ax.set_mode("test")

# Disabled - no verification overhead
ax.set_mode("off")
```

### Context Managers

```python
# Temporarily disable verification
with ax.no_verification():
    result = expensive_function()

# Temporarily switch to production mode
with ax.production_mode():
    result = performance_critical_function()

# Named verification context
with ax.verification_context("data_processing"):
    ax.require(data_is_valid(), "Data validation failed")
```

## Error Handling

Simple Axiomatik provides clear, helpful error messages:

```python
try:
    result = process_items([], 5)
except ax.VerificationError as e:
    print(e)
    # Output:
    # process_items() verification failed
    # Condition: items satisfies non-empty list
    # Message: Type constraint violation for items
    # Values: items=[]
    # Suggestion: Ensure items meets the required constraints
```

## Performance Tracking

```python
@ax.verify(track_performance=True)
def monitored_function(n: int) -> int:
    time.sleep(0.001)  # Simulate work
    return n * 2

# Run some operations
for i in range(100):
    monitored_function(i)

# Get performance report
print(ax.performance_report())
# Output:
# Performance Report
# monitored_function                  1.2ms avg, called 100x, 120.0ms total
```

## Common Patterns

### Web API Validation

```python
@ax.enable_for_dataclass
@dataclass  
class CreateUserRequest:
    username: ax.NonEmpty[str]
    email: ax.NonEmpty[str] 
    age: ax.Range[int, 13, 120]
    
    def __post_init__(self):
        ax.require("@" in self.email, "Invalid email format")
        ax.require(len(self.username) >= 3, "Username too short")

@ax.checked
def create_user(request: CreateUserRequest) -> dict:
    # Validation happens automatically from types
    return {"user_id": 12345, "status": "created"}
```

### Financial Calculations

```python
@ax.checked  
def calculate_loan_payment(
    principal: ax.Positive[float],
    annual_rate: ax.Range[float, 0.0, 0.5],
    years: ax.Range[int, 1, 50]
) -> ax.Positive[float]:
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    if annual_rate == 0:
        return principal / num_payments
    
    return (principal * monthly_rate * (1 + monthly_rate) ** num_payments) / \
           ((1 + monthly_rate) ** num_payments - 1)
```

### Data Processing Pipeline

```python
@ax.stateful(initial="ready")
class DataProcessor:
    @ax.state("ready", "processing")
    def start_processing(self, data: ax.NonEmpty[list]):
        self.data = data
    
    @ax.verify
    def process_batch(self) -> dict:
        ax.require(len(self.data) > 0, "No data to process")
        # Process data...
        return {"processed": 10, "remaining": len(self.data)}
    
    @ax.state("processing", "complete") 
    def finish_processing(self):
        ax.require(len(self.data) == 0, "All data must be processed")
```

## Best Practices

### Start Simple
```python
# Begin with @ax.verify for critical functions
@ax.verify
def critical_calculation(x: float) -> float:
    ax.require(x > 0, "Input must be positive")
    result = complex_math(x)
    ax.ensure(result > 0, "Result should be positive")
    return result
```

### Add Types Gradually
```python
# Then add @ax.checked for automatic validation
@ax.checked 
def improved_function(x: ax.Positive[float]) -> ax.Positive[float]:
    # Type system handles the validation
    return complex_math(x)
```

### Combine Both Approaches
```python
# Use both for comprehensive validation
@ax.verify
@ax.checked
def comprehensive_function(
    data: ax.NonEmpty[list],
    threshold: ax.Positive[float]
) -> ax.PositiveInt:
    # Types validate structure, verify validates business logic
    ax.require(all(x > threshold for x in data), "All values must exceed threshold")
    return len([x for x in data if x > threshold * 2])
```

### Production Configuration
```python
# In production environments
if os.getenv("ENVIRONMENT") == "production":
    ax.set_mode("prod")  # Minimal verification overhead
else:
    ax.set_mode("dev")   # Full verification during development
```

## API Reference

### Decorators
- `@ax.verify` - Enable manual require/ensure statements
- `@ax.checked` - Automatic verification from type hints
- `@ax.verify(track_performance=True)` - Enable performance tracking
- `@ax.stateful(initial="state")` - Protocol verification for classes
- `@ax.state("from", "to")` - Mark state transition methods
- `@ax.enable_for_dataclass` - Automatic dataclass field validation

### Verification Functions
- `ax.require(condition, message)` - Check preconditions
- `ax.ensure(condition, message)` - Check postconditions  
- `ax.expect_approximately(a, b, tolerance)` - Floating point comparison

### Type Aliases
- `ax.Positive[int|float]` - Numbers > 0
- `ax.NonEmpty[list|str]` - Collections with length > 0
- `ax.Range[type, min, max]` - Values within bounds
- `ax.PositiveInt` - Shorthand for ax.Positive[int]
- `ax.PositiveFloat` - Shorthand for ax.Positive[float]
- `ax.Percentage` - Integers 0-100
- `ax.NonEmptyList` - Shorthand for ax.NonEmpty[list]
- `ax.NonEmptyStr` - Shorthand for ax.NonEmpty[str]

### Configuration
- `ax.set_mode(mode)` - Set verification mode ("dev"|"prod"|"test"|"off")
- `ax.get_mode()` - Get current mode
- `ax.report()` - Generate status report
- `ax.performance_report()` - Generate performance report

### Context Managers
- `ax.verification_context(name)` - Named verification context
- `ax.production_mode()` - Temporarily use production mode
- `ax.no_verification()` - Temporarily disable verification

## Integration

### Works With Existing Tools
- Type checkers (mypy, pyright)
- Testing frameworks (pytest, unittest)
- Web frameworks (FastAPI, Flask, Django)
- Data processing (pandas, numpy)
- Any Python library

### Gradual Adoption
```python
# Start with one function
@ax.verify
def important_function():
    pass

# Add more over time  
@ax.checked
def another_function():
    pass

# Eventually cover critical paths
@ax.verify
@ax.checked  
def fully_verified_function():
    pass
```

## FAQ

**Q: What's the performance overhead?**
A: Minimal in production mode. Use `ax.set_mode("prod")` for essential checks only.

**Q: Can I use this with existing code?**  
A: Yes! Add `@ax.verify` to functions incrementally. No existing code changes required.

**Q: How do I handle verification errors?**
A: Catch `ax.VerificationError` exceptions. They include helpful messages and suggestions.

**Q: Does this work with type checkers?**
A: Yes! Simple Axiomatik uses standard Python type hints that work with mypy, pyright, etc.

**Q: Can I disable verification in production?**
A: Yes! Use `ax.set_mode("off")` to disable all verification for maximum performance.

## Examples

See `simple_quick.py` for a complete tutorial and `simple_usage.py` for real-world examples including:

- Web API input validation
- Financial calculations with precision
- Data processing pipelines  
- Game state management
- Scientific data analysis
